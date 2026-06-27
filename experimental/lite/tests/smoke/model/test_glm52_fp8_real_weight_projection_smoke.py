# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Pinned real-weight GLM-5.2-FP8 dequantized-BF16 q_a projection parity.

This gate range-fetches only three tensors from the immutable official HF
release: layer-0 ``q_a_proj`` FP8 weight, its FP32 block scale, and the BF16
``q_a_layernorm`` weight. It exercises MLite's production dequantizer plus the
production ``torch.nn.Linear`` + Transformer Engine RMSNorm path against
Transformers 5.12's public ``torch.nn.Linear`` + RMSNorm formula and an
independent FP32 reference. It is deliberately dequantized-BF16,
projection-level evidence, not HF quantized-runtime, full-model, or
long-context parity.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

pytestmark = [pytest.mark.mlite, pytest.mark.smoke, pytest.mark.gpu]

_WEIGHT = "model.layers.0.self_attn.q_a_proj.weight"
_SCALE = f"{_WEIGHT}_scale_inv"
_NORM = "model.layers.0.self_attn.q_a_layernorm.weight"
_SHARD = "model-00001-of-00141.safetensors"
_PAYLOAD_FILES = {
    _WEIGHT: "q_a_proj.bin",
    _SCALE: "q_a_proj_scale_inv.bin",
    _NORM: "q_a_layernorm.bin",
}


def _load_raw_tensor(path: Path, dtype: torch.dtype, shape: list[int]) -> torch.Tensor:
    payload = bytearray(path.read_bytes())
    return torch.frombuffer(payload, dtype=dtype).clone().reshape(shape)


def _load_and_validate_projection_authority(
    authority_dir: Path,
) -> tuple[dict, dict, dict[str, Path], int]:
    """Bind downloaded payloads back to the immutable in-repo authority."""
    fixture_path = (
        Path(__file__).resolve().parents[2]
        / "unit"
        / "model"
        / "glm52_fp8_header_authority.json"
    )
    authority = json.loads(fixture_path.read_text())
    manifest = json.loads((authority_dir / "manifest.json").read_text())
    source = authority["source"]
    assert set(manifest) == {"repo", "revision", "shard", "payloads"}
    assert manifest["repo"] == source["repo"]
    assert manifest["revision"] == source["revision"]
    assert manifest["shard"] == _SHARD

    tensors = authority["tensors"]
    expected_names = {_WEIGHT, _SCALE, _NORM}
    assert set(tensors) == expected_names
    assert set(manifest["payloads"]) == expected_names
    shard_contract = source["safetensors"][_SHARD]
    paths: dict[str, Path] = {}
    payload_bytes_total = 0
    for tensor_name in sorted(expected_names):
        entry = manifest["payloads"][tensor_name]
        range_contract = shard_contract["payload_ranges"][tensor_name]
        range_start, range_end = range_contract["file_range"]
        expected_bytes = range_end - range_start + 1
        assert set(entry) == {"file", "bytes", "sha256"}
        assert entry["file"] == _PAYLOAD_FILES[tensor_name]
        assert entry["bytes"] == expected_bytes
        assert entry["sha256"] == range_contract["sha256"]
        path = authority_dir / entry["file"]
        payload = path.read_bytes()
        assert len(payload) == expected_bytes
        assert hashlib.sha256(payload).hexdigest() == range_contract["sha256"]
        paths[tensor_name] = path
        payload_bytes_total += len(payload)
    assert paths.keys() == expected_names
    assert payload_bytes_total == 12_590_080
    return authority, tensors, paths, payload_bytes_total


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual_f = actual.detach().float().reshape(-1)
    expected_f = expected.detach().float().reshape(-1)
    delta = actual_f - expected_f
    expected_rms = expected_f.square().mean().sqrt().clamp_min(1e-12)
    actual_norm = actual_f.norm()
    expected_norm = expected_f.norm().clamp_min(1e-12)
    return {
        "cosine": float(F.cosine_similarity(actual_f, expected_f, dim=0).item()),
        "rms_relative": float(delta.square().mean().sqrt().div(expected_rms).item()),
        "norm_ratio": float(actual_norm.div(expected_norm).item()),
        "max_abs": float(delta.abs().max().item()),
    }


def _assert_projection_parity(
    label: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    cosine_min: float,
    rms_relative_max: float,
    max_abs_max: float,
) -> dict[str, float]:
    metrics = _metrics(actual, expected)
    assert metrics["cosine"] >= cosine_min, f"{label}: {metrics}"
    assert metrics["rms_relative"] <= rms_relative_max, f"{label}: {metrics}"
    assert 0.99 <= metrics["norm_ratio"] <= 1.01, f"{label}: {metrics}"
    assert metrics["max_abs"] <= max_abs_max, f"{label}: {metrics}"
    return metrics


def test_glm52_fp8_pinned_real_q_a_projection_matches_transformers() -> None:
    authority_dir_value = os.getenv("GLM52_FP8_PROJECTION_AUTHORITY_DIR")
    if not authority_dir_value:
        pytest.skip(
            "set GLM52_FP8_PROJECTION_AUTHORITY_DIR after running "
            "experimental/lite/tests/fetch_glm52_fp8_projection_authority.py"
        )
    authority_dir = Path(authority_dir_value)
    authority, tensors, paths, payload_bytes_total = (
        _load_and_validate_projection_authority(authority_dir)
    )
    source = authority["source"]
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the GLM-5.2 real-weight projection gate")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch float8_e4m3fn is required for the released FP8 weight")

    transformers = pytest.importorskip("transformers")
    assert transformers.__version__ == "5.12.0"
    pytest.importorskip("transformer_engine.pytorch")
    from megatron.lite.model.glm5.lite.checkpoint import _dequant_fp8_weight
    from megatron.lite.primitive.modules.attention.dsa import RMSNorm
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaRMSNorm

    raw_weight = _load_raw_tensor(
        paths[_WEIGHT], torch.float8_e4m3fn, tensors[_WEIGHT]["shape"]
    )
    scale = _load_raw_tensor(paths[_SCALE], torch.float32, tensors[_SCALE]["shape"])
    norm_weight = _load_raw_tensor(
        paths[_NORM], torch.bfloat16, tensors[_NORM]["shape"]
    )

    class Reader:
        index = {_SCALE: "range-payload"}

        @staticmethod
        def get_tensor(name: str) -> torch.Tensor:
            assert name == _SCALE
            return scale

    dequantized = _dequant_fp8_weight(Reader(), _WEIGHT, raw_weight)
    reference_scale = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
    reference_weight = raw_weight.float() * reference_scale
    torch.testing.assert_close(dequantized, reference_weight, atol=0, rtol=0)

    config_values = source["config_values"]
    hidden_size = config_values["hidden_size"]
    q_lora_rank = config_values["q_lora_rank"]
    eps = config_values["q_a_layernorm_effective_eps"]
    device = torch.device("cuda", torch.cuda.current_device())
    bf16_weight = dequantized.to(device=device, dtype=torch.bfloat16)
    bf16_norm_weight = norm_weight.to(device=device)

    mlite_linear = nn.Linear(
        hidden_size, q_lora_rank, bias=False, device=device, dtype=torch.bfloat16
    )
    mlite_norm = RMSNorm(q_lora_rank, eps=eps).to(device=device, dtype=torch.bfloat16)
    hf_linear = nn.Linear(
        hidden_size, q_lora_rank, bias=False, device=device, dtype=torch.bfloat16
    )
    hf_norm = GlmMoeDsaRMSNorm(q_lora_rank, eps=eps).to(
        device=device, dtype=torch.bfloat16
    )
    with torch.no_grad():
        mlite_linear.weight.copy_(bf16_weight)
        hf_linear.weight.copy_(bf16_weight)
        mlite_norm.weight.copy_(bf16_norm_weight)
        hf_norm.weight.copy_(bf16_norm_weight)
    for module in (mlite_linear, mlite_norm, hf_linear, hf_norm):
        module.requires_grad_(False)

    generator = torch.Generator(device=device).manual_seed(20260628)
    source_input = torch.randn(
        (1, 8, hidden_size), generator=generator, device=device, dtype=torch.bfloat16
    )
    mlite_input = source_input.detach().clone().requires_grad_(True)
    hf_input = source_input.detach().clone().requires_grad_(True)
    mlite_output = mlite_norm(mlite_linear(mlite_input))
    hf_output = hf_norm(hf_linear(hf_input))

    reference_linear = F.linear(source_input.float(), dequantized.to(device=device))
    reference_variance = reference_linear.square().mean(dim=-1, keepdim=True)
    reference_output = (
        reference_linear * torch.rsqrt(reference_variance + eps)
    ) * bf16_norm_weight.float()

    hf_metrics = _assert_projection_parity(
        "MLite-vs-Transformers dequantized-BF16 real q_a projection",
        mlite_output,
        hf_output,
        cosine_min=0.999,
        rms_relative_max=0.02,
        max_abs_max=0.05,
    )
    fp32_metrics = _assert_projection_parity(
        "MLite-vs-FP32 dequantized-BF16 real q_a projection",
        mlite_output,
        reference_output,
        cosine_min=0.999,
        rms_relative_max=0.02,
        max_abs_max=0.05,
    )

    gradient_probe = torch.randn(
        mlite_output.shape, generator=generator, device=device, dtype=torch.float32
    )
    (mlite_output.float() * gradient_probe).mean().backward()
    (hf_output.float() * gradient_probe).mean().backward()
    assert mlite_input.grad is not None and hf_input.grad is not None
    grad_metrics = _assert_projection_parity(
        "MLite-vs-Transformers dequantized-BF16 real q_a input gradient",
        mlite_input.grad,
        hf_input.grad,
        cosine_min=0.999,
        rms_relative_max=0.03,
        max_abs_max=0.005,
    )

    print(
        "NON_SKIP_GLM52_FP8_REAL_QA_PROJECTION_PARITY_PASSED "
        f"revision={source['revision']} real_payload_bytes={payload_bytes_total} "
        "evidence=dequantized_bf16_projection_level "
        f"hf_cosine={hf_metrics['cosine']:.9f} "
        f"hf_rms_relative={hf_metrics['rms_relative']:.9f} "
        f"fp32_cosine={fp32_metrics['cosine']:.9f} "
        f"fp32_rms_relative={fp32_metrics['rms_relative']:.9f} "
        f"grad_cosine={grad_metrics['cosine']:.9f} "
        f"grad_rms_relative={grad_metrics['rms_relative']:.9f}"
    )
