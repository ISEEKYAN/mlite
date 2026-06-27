# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU precision tests for the two GLM-5 DSA RoPE layouts."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


pytestmark = pytest.mark.mlite


def _hf_mla_interleaved_reference(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *, unsqueeze_dim: int = 2
) -> torch.Tensor:
    """Transformers v5.12.0 ``apply_rotary_pos_emb_interleave`` for one tensor.

    Vendored from transformers commit e0e7504bca2bfd1b85bb0eedb148f7b250226f06
    so this CPU gate does not require transformers as a test dependency.
    """
    cos = cos[..., : cos.shape[-1] // 2].unsqueeze(unsqueeze_dim)
    sin = sin[..., : sin.shape[-1] // 2].unsqueeze(unsqueeze_dim)
    x_even, x_odd = x[..., 0::2], x[..., 1::2]
    return torch.cat((x_even * cos - x_odd * sin, x_odd * cos + x_even * sin), dim=-1)


def _half_split_reference(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *, unsqueeze_dim: int = 2
) -> torch.Tensor:
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos.unsqueeze(unsqueeze_dim) + rotated * sin.unsqueeze(unsqueeze_dim)


def _rotary_inputs(dsa, *, dtype: torch.dtype = torch.float64):
    position_ids = torch.tensor([[0, 3, 11, 29]], dtype=torch.long)
    cos, sin = dsa.build_rotary_embeddings(
        position_ids=position_ids,
        dim=8,
        rope_theta=8_000_000.0,
        dtype=dtype,
    )
    values = torch.linspace(-2.5, 3.5, 2 * 4 * 3 * 8, dtype=dtype).reshape(2, 4, 3, 8)
    return values, cos.expand(2, -1, -1), sin.expand(2, -1, -1), position_ids


def test_mla_interleaved_rope_matches_transformers_v5_12_at_distinct_positions(
    transformer_engine_import_stub,
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.attention import dsa

    x, cos, sin, _ = _rotary_inputs(dsa)

    actual = dsa.apply_rotary_pos_emb(
        x, cos, sin, unsqueeze_dim=2, mla_interleaved=True
    )
    expected = _hf_mla_interleaved_reference(x, cos, sin)
    legacy = _half_split_reference(x, cos, sin)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    # Position zero is intentionally present, but non-zero positions must make
    # the incorrect half-split interpretation observable.
    assert not torch.allclose(actual[:, 1:], legacy[:, 1:])


def test_non_interleaved_rope_keeps_the_glm51_formula_bitwise(
    transformer_engine_import_stub,
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.attention import dsa

    x, cos, sin, _ = _rotary_inputs(dsa)

    actual = dsa.apply_rotary_pos_emb(
        x, cos, sin, unsqueeze_dim=2, mla_interleaved=False
    )
    expected = _half_split_reference(x, cos, sin)

    assert torch.equal(actual, expected)


def test_public_dsa_primitive_defaults_preserve_legacy_half_split(
    transformer_engine_import_stub, monkeypatch
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.attention import dsa

    monkeypatch.setattr(dsa, "RMSNorm", nn.RMSNorm)
    attention = dsa.DynamicSparseAttention(
        hidden_size=16,
        num_attention_heads=2,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=8,
        v_head_dim=4,
        index_n_heads=2,
        index_head_dim=12,
        index_topk=2,
        rms_norm_eps=1e-6,
    )

    assert attention.rope_interleaved is False
    assert attention.indexer is not None
    assert attention.indexer.rope_interleaved is False


@pytest.mark.parametrize("mla_interleaved", [False, True])
def test_dynamic_sparse_attention_main_rope_matches_independent_reference(
    transformer_engine_import_stub, monkeypatch, mla_interleaved
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.attention import dsa

    # The production class uses TE RMSNorm; nn.RMSNorm lets this precision
    # contract run on CPU without changing the projections under test.
    monkeypatch.setattr(dsa, "RMSNorm", nn.RMSNorm)
    captured = {}

    def _capture_fused(query, kv_full, *args, value_dim=None, **kwargs):
        del args, kwargs
        captured["query"] = query.detach().clone()
        captured["kv_full"] = kv_full.detach().clone()
        output = query.new_zeros(
            query.shape[0], query.shape[1], query.shape[2] * value_dim
        )
        return output, query.new_zeros((), dtype=torch.float32)

    monkeypatch.setattr(dsa, "_fused_indexer_sparse_attn", _capture_fused)
    attention = dsa.DynamicSparseAttention(
        hidden_size=16,
        num_attention_heads=2,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=8,
        v_head_dim=4,
        index_n_heads=2,
        index_head_dim=12,
        index_topk=2,
        rms_norm_eps=1e-6,
        rope_interleaved=mla_interleaved,
        indexer_rope_interleaved=False,
        indexer_rope_first=True,
        indexer_use_hadamard=False,
    )
    position_ids = torch.tensor([[0, 3, 11, 29]], dtype=torch.long)
    cos, sin = dsa.build_rotary_embeddings(
        position_ids=position_ids,
        dim=8,
        rope_theta=8_000_000.0,
        dtype=torch.float32,
    )
    x = torch.linspace(-1.5, 2.0, 4 * 16, dtype=torch.float32).reshape(1, 4, 16)

    attention(x, cos=cos, sin=sin, position_ids=position_ids)

    with torch.no_grad():
        q_resid = attention.q_a_layernorm(attention.q_a_proj(x))
        q = attention.q_b_proj(q_resid).view(1, 4, 2, 12)
        q_pe = q[..., -8:]
        k_pe = attention.kv_a_proj_with_mqa(x)[..., -8:].unsqueeze(2)
        reference = (
            _hf_mla_interleaved_reference if mla_interleaved else _half_split_reference
        )
        expected_q_pe = reference(q_pe, cos, sin)
        expected_k_pe = reference(k_pe, cos, sin).squeeze(2)

    torch.testing.assert_close(
        captured["query"].transpose(0, 1)[..., -8:], expected_q_pe
    )
    torch.testing.assert_close(
        captured["kv_full"].transpose(0, 1)[..., -8:], expected_k_pe
    )


@pytest.mark.parametrize("mla_interleaved", [False, True])
def test_dsa_indexer_honors_its_independent_rope_layout_flag(
    transformer_engine_import_stub, mla_interleaved
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.attention import dsa

    # HF v5.12.0 currently hard-codes half-split RoPE in its indexer despite
    # the checkpoint flag. Megatron and vLLM honor indexer_rope_interleave, so
    # this test covers the explicit MLite configuration contract independently
    # from the main-attention HF oracle above.
    indexer = dsa.DSAIndexer(
        hidden_size=16,
        q_lora_rank=8,
        qk_rope_head_dim=8,
        index_n_heads=2,
        index_head_dim=12,
        index_topk=2,
        rope_interleaved=mla_interleaved,
        rope_first=True,
        use_hadamard=False,
    )
    position_ids = torch.tensor([[0, 3, 11, 29]], dtype=torch.long)
    cos, sin = dsa.build_rotary_embeddings(
        position_ids=position_ids,
        dim=8,
        rope_theta=8_000_000.0,
        dtype=torch.float32,
    )
    x = torch.linspace(-1.0, 1.0, 4 * 16, dtype=torch.float32).reshape(1, 4, 16)
    q_resid = torch.linspace(-0.5, 0.75, 4 * 8, dtype=torch.float32).reshape(1, 4, 8)

    q_out, k_out, _ = indexer.forward_before_topk(x, q_resid, cos, sin, position_ids)

    with torch.no_grad():
        q = indexer.wq_b(q_resid).view(1, 4, 2, 12)
        k = indexer.k_norm(indexer.wk(x))
        reference = (
            _hf_mla_interleaved_reference if mla_interleaved else _half_split_reference
        )
        expected_q = torch.cat([reference(q[..., :8], cos, sin), q[..., 8:]], dim=-1)
        expected_k = torch.cat(
            [reference(k[..., :8].unsqueeze(2), cos, sin).squeeze(2), k[..., 8:]],
            dim=-1,
        )

    torch.testing.assert_close(q_out.transpose(0, 1), expected_q)
    torch.testing.assert_close(k_out.transpose(0, 1), expected_k)
