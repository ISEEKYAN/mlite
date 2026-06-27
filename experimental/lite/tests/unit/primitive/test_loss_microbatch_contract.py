from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from megatron.lite.primitive import parallel as parallel_primitives
from megatron.lite.primitive.parallel import pipeline
from megatron.lite.primitive.train_step import run_microbatch_loop
from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime
from megatron.lite.runtime.contracts.handle import ModelHandle


pytestmark = pytest.mark.mlite


class _Metric:
    """Small stand-in for VERL's Metric without importing the connector dependency."""

    def __init__(self, aggregation: str, value: float | None = None):
        self.aggregation = aggregation
        self.values: list[float] = []
        if value is not None:
            self.values.append(value)

    def init_list(self):
        return _Metric(self.aggregation)

    def append(self, value):
        if isinstance(value, _Metric):
            assert value.aggregation == self.aggregation
            self.values.extend(value.values)
        else:
            self.values.append(value)

    def aggregate(self):
        if self.aggregation == "sum":
            return sum(self.values)
        return sum(self.values) / len(self.values)


class _ScalarModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))


@pytest.mark.parametrize("num_microbatches", [1, 4])
@pytest.mark.parametrize("loss_kind", ["internal", "external", "external_normalized"])
def test_run_microbatch_loop_loss_contract_is_microbatch_invariant(
    num_microbatches,
    loss_kind,
):
    model = _ScalarModel()
    batches = iter([{"value": 3.0, "micro": index} for index in range(num_microbatches)])
    normalized = loss_kind == "external_normalized"

    def forward_fn(module, batch):
        value = module.weight * batch["value"]
        return {"value": value, "loss": value}

    def loss_fn(output, batch):
        loss = output["value"] / num_microbatches if normalized else output["value"]
        return loss, {"metric": _Metric("sum", float(batch["micro"] + 1))}

    output = run_microbatch_loop(
        model,
        batches,
        num_microbatches,
        forward_fn,
        loss_fn=None if loss_kind == "internal" else loss_fn,
        loss_fn_already_normalized=normalized,
    )

    torch.testing.assert_close(model.weight.grad, torch.tensor(3.0))
    if loss_kind == "internal":
        assert "_loss_fn_metrics" not in output
    else:
        assert len(output["_loss_fn_metrics"]) == num_microbatches


@pytest.mark.parametrize("num_microbatches", [1, 4])
def test_runtime_aggregates_every_external_loss_metric(num_microbatches):
    model = _ScalarModel()
    handle = ModelHandle(
        model=model,
        parallel_state=SimpleNamespace(pp_size=1),
        _extras={
            "forward_step": lambda module, batch: {"value": module.weight * batch["value"]},
        },
    )
    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)
    batches = iter([{"value": 0.0, "micro": index} for index in range(num_microbatches)])

    def loss_fn(output, batch):
        return output["value"] + batch["micro"] + 1, {
            "sum_metric": _Metric("sum", float(batch["micro"] + 1)),
            "plain_metric": batch["micro"] + 1,
        }

    result = runtime.forward_backward(
        handle,
        batches,
        loss_fn,
        num_microbatches=num_microbatches,
        loss_fn_already_normalized=True,
    )

    assert result.metrics["sum_metric"].values == list(
        map(float, range(1, num_microbatches + 1))
    )
    assert result.metrics["sum_metric"].aggregate() == sum(range(1, num_microbatches + 1))
    assert result.metrics["plain_metric"] == list(range(1, num_microbatches + 1))
    assert result.metrics["loss"] == list(map(float, range(1, num_microbatches + 1)))


class _PipelineLastStageModel(_ScalarModel):
    def __init__(self):
        super().__init__()
        self.input_tensor = None

    def set_input_tensor(self, input_tensor):
        self.input_tensor = input_tensor

    def forward(self, *, hidden_states=None, **_kwargs):
        hidden_states = self.input_tensor if hidden_states is None else hidden_states
        return {"value": self.weight * hidden_states.sum()}


@pytest.mark.parametrize("num_microbatches", [1, 4])
@pytest.mark.parametrize("normalized", [False, True])
def test_pipeline_schedule_honors_external_loss_contract(
    monkeypatch,
    num_microbatches,
    normalized,
):
    model = _PipelineLastStageModel()
    sent_input_grads = []
    original_empty = torch.empty

    def cpu_empty(*args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return original_empty(*args, **kwargs)

    def fake_send_recv(
        send_fwd,
        send_bwd,
        recv_fwd,
        recv_bwd,
        _ps,
        tensor_shape,
        **_kwargs,
    ):
        assert not recv_bwd
        if send_bwd is not None:
            sent_input_grads.append(send_bwd.detach().clone())
        activation = (
            torch.full(tensor_shape, 3.0, requires_grad=True) if recv_fwd else None
        )
        return activation, None

    monkeypatch.setattr(pipeline.torch, "empty", cpu_empty)
    monkeypatch.setattr(pipeline, "_send_recv_pipeline", fake_send_recv)
    ps = SimpleNamespace(
        pp_size=2,
        pp_rank=1,
        pp_is_first=False,
        pp_is_last=True,
        dp_size=1,
    )

    def loss_fn(output, batch):
        loss = output["value"] / num_microbatches if normalized else output["value"]
        return loss, {"metric": _Metric("sum", float(batch["micro"] + 1))}

    outputs = parallel_primitives.forward_backward_pipelining(
        lambda module, _batch: module(),
        [model],
        iter([{"micro": index} for index in range(num_microbatches)]),
        SimpleNamespace(num_microbatches=num_microbatches),
        ps,
        tensor_shape=(1,),
        loss_fn=loss_fn,
        loss_fn_already_normalized=normalized,
    )

    torch.testing.assert_close(model.weight.grad, torch.tensor(3.0))
    torch.testing.assert_close(torch.stack(sent_input_grads).sum(), torch.tensor(1.0))
    assert len(outputs) == num_microbatches
    assert [item["metrics"]["metric"].values for item in outputs] == [
        [float(index)] for index in range(1, num_microbatches + 1)
    ]


def test_normalized_flag_requires_external_loss():
    model = _ScalarModel()
    with pytest.raises(ValueError, match="requires an external loss_fn"):
        run_microbatch_loop(
            model,
            iter([{"value": 1.0}]),
            1,
            lambda module, batch: {"loss": module.weight * batch["value"]},
            loss_fn_already_normalized=True,
        )
