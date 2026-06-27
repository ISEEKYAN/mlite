# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch
from verl_mlite.engine.config import MegatronLiteEngineConfig
from verl_mlite.engine.mlite_engine import MegatronLiteEngine
from megatron.lite.runtime.contracts import ForwardResult, LossContext, ModelOutputs


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
    engine.is_mp_src_rank_with_outputs = MethodType(lambda _self: True, engine)
    return engine


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
