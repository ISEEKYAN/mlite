# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch
from verl_mlite.engine.config import MegatronLiteEngineConfig
from verl_mlite.engine.mlite_engine import MegatronLiteEngine
from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime
from megatron.lite.runtime.contracts import ForwardResult, LossContext, ModelOutputs
from megatron.lite.runtime.contracts.handle import ModelHandle


class _MicroBatch:
    def __init__(self, index: int):
        self.index = index

    def to(self, _device):
        return self


class _ForwardOnlyRuntime:
    def __init__(self, raw_outputs: list[dict[str, torch.Tensor]]):
        self.raw_outputs = raw_outputs

    def forward_backward(self, _handle, data, loss_fn, **kwargs):
        assert kwargs["forward_only"] is True
        assert kwargs["num_microbatches"] == len(self.raw_outputs)
        assert loss_fn is not None
        for raw_output, (runtime_batch, loss_context) in zip(
            self.raw_outputs, data, strict=True
        ):
            loss_fn(raw_output, runtime_batch, loss_context)
        return ForwardResult(
            model_output=ModelOutputs(log_probs=self.raw_outputs[-1]["log_probs"]),
            metrics={"loss": 0.0},
        )


def _engine(raw_outputs: list[dict[str, torch.Tensor]]) -> MegatronLiteEngine:
    engine = MegatronLiteEngine(
        model_config=SimpleNamespace(
            local_path="/tmp/qwen35", hf_config={"model_type": "qwen3_5_moe"}, mtp=None
        ),
        engine_config=MegatronLiteEngineConfig(
            custom_backend_module=None,
            pp=1,
            impl_cfg={"use_thd": True},
        ),
        optimizer_config=SimpleNamespace(),
        checkpoint_config={},
    )
    engine.handle = object()
    engine.runtime = _ForwardOnlyRuntime(raw_outputs)
    engine._make_runtime_batch = MethodType(lambda _self, batch: batch, engine)
    engine._make_runtime_loss_context = MethodType(
        lambda _self, batch, *, loss_scale: LossContext(
            loss_scale=loss_scale, source_batch=batch
        ),
        engine,
    )
    engine._build_verl_model_output = MethodType(
        lambda _self, *, raw_output, runtime_batch: {
            key: value
            for key, value in raw_output.items()
            if key in {"log_probs", "entropy"} and value is not None
        },
        engine,
    )
    engine.get_data_parallel_size = MethodType(lambda _self: 1, engine)
    engine.get_data_parallel_group = MethodType(lambda _self: None, engine)
    engine.is_mp_src_rank_with_outputs = MethodType(lambda _self: True, engine)
    return engine


@pytest.mark.parametrize("num_microbatches", [1, 4])
def test_verl_loss_hook_preserves_logical_loss_and_ppo_gradient_scale(num_microbatches):
    engine = _engine([])
    weight = torch.nn.Parameter(torch.tensor(1.0))
    reduced_outputs = []

    def loss_function(*, model_output, data, dp_group):
        assert dp_group is None
        logical_loss = model_output["log_probs"].sum() / num_microbatches
        return logical_loss, {"pg_loss": logical_loss.detach(), "micro": data.index}

    runtime_loss_fn = engine._make_runtime_loss_fn(
        loss_function,
        forward_only=False,
        num_microbatches=num_microbatches,
        output_lst=reduced_outputs,
    )
    for index in range(num_microbatches):
        micro_batch = _MicroBatch(index)
        backward_loss, _metrics = runtime_loss_fn(
            {"log_probs": weight * 3.0},
            micro_batch,
            LossContext(source_batch=micro_batch),
        )
        # MLite and Megatron schedules both apply this fixed averaging step.
        (backward_loss / num_microbatches).backward()

    torch.testing.assert_close(weight.grad, torch.tensor(3.0))
    assert [item["loss"] for item in reduced_outputs] == [
        3.0 / num_microbatches
    ] * num_microbatches
    assert [item["metrics"]["micro"] for item in reduced_outputs] == list(
        range(num_microbatches)
    )


@pytest.mark.parametrize("num_microbatches", [1, 4])
def test_pp1_training_reports_unscaled_micro_losses_end_to_end(
    monkeypatch, num_microbatches
):
    engine = _engine([])
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    engine.handle = ModelHandle(
        model=model,
        parallel_state=SimpleNamespace(pp_size=1),
        _extras={
            "forward_step": lambda module, _batch: {
                "log_probs": module.weight.squeeze() * 3.0
            }
        },
    )
    engine.module = model
    engine.runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)
    captured = {}
    logical_batch = object()

    def postprocess_batch_func(*, output_lst, indices, data):
        assert indices is None
        assert data is logical_batch
        captured["outputs"] = output_lst
        return {
            "loss": [item["loss"] for item in output_lst],
            "metrics": {"micro": [item["metrics"]["micro"] for item in output_lst]},
        }

    def loss_function(*, model_output, data, dp_group):
        assert dp_group is None
        logical_loss = model_output["log_probs"] / num_microbatches
        return logical_loss, {"micro": data.index}

    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.postprocess_batch_func", postprocess_batch_func
    )
    monkeypatch.setattr("verl_mlite.engine.mlite_engine.get_device_id", lambda: "cpu")
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.tu.get_non_tensor_data",
        lambda **_kwargs: num_microbatches,
    )
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.tu.assign_non_tensor", lambda *_args, **_kwargs: None
    )

    result = engine._forward_backward_batch_with_runtime(
        data=logical_batch,
        micro_batches=[_MicroBatch(index) for index in range(num_microbatches)],
        indices=None,
        loss_function=loss_function,
        forward_only=False,
    )

    torch.testing.assert_close(model.weight.grad, torch.tensor([[3.0]]))
    assert result["loss"] == [3.0 / num_microbatches] * num_microbatches
    assert result["metrics"]["micro"] == list(range(num_microbatches))
    assert len(captured["outputs"]) == num_microbatches


@pytest.mark.parametrize("with_entropy", [False, True])
def test_pp1_forward_only_preserves_all_log_probs_and_optional_entropy(
    monkeypatch, with_entropy
):
    raw_outputs = []
    for index in range(2):
        output = {"log_probs": torch.tensor([float(index + 1)])}
        if with_entropy:
            output["entropy"] = torch.tensor([float(index + 11)])
        raw_outputs.append(output)
    engine = _engine(raw_outputs)
    captured = {}

    def postprocess_batch_func(*, output_lst, indices, data):
        captured["outputs"] = output_lst
        assert indices == [1, 0]
        assert data is logical_batch
        model_output = {
            key: torch.cat([item["model_output"][key] for item in output_lst])
            for key in output_lst[0]["model_output"]
        }
        return {"model_output": model_output}

    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.postprocess_batch_func", postprocess_batch_func
    )
    monkeypatch.setattr("verl_mlite.engine.mlite_engine.get_device_id", lambda: "cpu")
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.tu.get_non_tensor_data",
        lambda **_kwargs: 2,
    )
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.tu.assign_non_tensor", lambda *_args, **_kwargs: None
    )
    logical_batch = object()

    result = engine._forward_backward_batch_with_runtime(
        data=logical_batch,
        micro_batches=[_MicroBatch(0), _MicroBatch(1)],
        indices=[1, 0],
        loss_function=None,
        forward_only=True,
    )

    torch.testing.assert_close(result["model_output"]["log_probs"], torch.tensor([1.0, 2.0]))
    assert len(captured["outputs"]) == 2
    if with_entropy:
        torch.testing.assert_close(
            result["model_output"]["entropy"], torch.tensor([11.0, 12.0])
        )
    else:
        assert "entropy" not in result["model_output"]


def test_pp1_forward_only_requires_result_log_probs(monkeypatch):
    raw_outputs = [{"log_probs": torch.tensor([1.0])}]
    engine = _engine(raw_outputs)

    class _MissingLogProbRuntime(_ForwardOnlyRuntime):
        def forward_backward(self, *args, **kwargs):
            result = super().forward_backward(*args, **kwargs)
            result.model_output.log_probs = None
            return result

    engine.runtime = _MissingLogProbRuntime(raw_outputs)
    monkeypatch.setattr("verl_mlite.engine.mlite_engine.get_device_id", lambda: "cpu")
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.tu.get_non_tensor_data",
        lambda **_kwargs: 1,
    )
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.tu.assign_non_tensor", lambda *_args, **_kwargs: None
    )

    with pytest.raises(ValueError, match="must contain token log_probs"):
        engine._forward_backward_batch_with_runtime(
            data=object(),
            micro_batches=[_MicroBatch(0)],
            indices=None,
            loss_function=None,
            forward_only=True,
        )
