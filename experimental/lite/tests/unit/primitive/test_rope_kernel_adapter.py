from unittest import mock
from pathlib import Path

import pytest
import torch

from megatron.lite.primitive.kernels import rope


LITE_ROOT = Path(__file__).resolve().parents[3]


def test_generic_sbhd_dispatches_to_mcore_te_kernel():
    tensor = torch.randn(4, 1, 2, 8)
    freqs = torch.randn(4, 1, 1, 8)
    sbhd = mock.Mock(return_value=tensor + 1)
    with mock.patch.object(
        rope, "_load_generic_kernels", return_value=(sbhd, mock.Mock())
    ):
        out = rope.apply_fused_rotary(tensor, freqs)
    assert torch.equal(out, tensor + 1)
    sbhd.assert_called_once_with(tensor, freqs, interleaved=False)


def test_generic_thd_forwards_cp_coordinates():
    tensor = torch.randn(4, 2, 8)
    freqs = torch.randn(8, 1, 1, 8)
    cu = torch.tensor([0, 8], dtype=torch.int32)
    thd = mock.Mock(return_value=tensor)
    with mock.patch.object(
        rope, "_load_generic_kernels", return_value=(mock.Mock(), thd)
    ):
        rope.apply_fused_rotary(tensor, freqs, cu_seqlens=cu, cp_rank=1, cp_size=2)
    thd.assert_called_once_with(
        tensor, cu, freqs, cp_size=2, cp_rank=1, interleaved=False
    )


def test_mla_q_adapter_does_not_hide_inplace_semantics():
    tensor = torch.randn(4, 1, 2, 12)
    cos = sin = torch.randn(4, 1, 1, 4)
    fused = mock.Mock(return_value=tensor)
    with mock.patch.object(
        rope, "_load_mla_kernels", return_value=(fused, mock.Mock())
    ):
        out = rope.apply_fused_mla_rotary_for_q(
            tensor, cos, sin, nope_dim=8, rope_dim=4
        )
    assert out is tensor
    assert fused.call_args.args[0] is tensor


def test_invalid_layout_fails_closed_before_loading_kernel():
    tensor = torch.randn(4, 1, 2, 8).transpose(-1, -2)
    with pytest.raises(ValueError, match="contiguous last dimension"):
        rope.apply_fused_rotary(tensor, torch.randn(4, 1, 1, 8))


def test_all_model_impl_configs_default_rope_fusion_on():
    for family in ("qwen3_moe", "qwen3_5", "kimi_k2", "glm5", "deepseek_v4"):
        source = (
            LITE_ROOT / "megatron/lite/model" / family / "lite/protocol.py"
        ).read_text()
        assert "apply_rope_fusion: bool = True" in source


def test_rope_provider_references_are_confined_to_kernel_boundary():
    provider_terms = (
        "megatron.core.extensions.transformer_engine",
        "megatron.core.fusions.fused_mla_yarn_rope_apply",
    )
    violations = []
    boundary = LITE_ROOT / "megatron/lite/primitive/kernels/rope.py"
    for path in (LITE_ROOT / "megatron/lite").rglob("*.py"):
        if path == boundary:
            continue
        text = path.read_text(encoding="utf-8")
        if any(term in text for term in provider_terms):
            violations.append(str(path.relative_to(LITE_ROOT)))
    assert violations == []
