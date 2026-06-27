# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Independent GLM-5.2 attention parity against Transformers v5.12.0.

The production sparse kernel is replaced by a first-principles Torch oracle so
this test can run on CPU.  The HF module supplies the independent projection,
MLA-style RoPE, sparse-mask, softmax, and value-projection reference.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


pytestmark = pytest.mark.mlite


def _torch_sparse_attention(
    query: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
    topk_length=None,
    indexer_topk: int = 0,
    value_dim: int | None = None,
) -> torch.Tensor:
    """Small exact oracle for MLite's flattened sparse-attention contract."""
    del topk_length, indexer_topk
    seq, batch, heads, _ = query.shape
    assert value_dim is not None
    out = torch.empty(seq, batch, heads * value_dim, dtype=query.dtype)
    for query_idx in range(seq):
        for batch_idx in range(batch):
            flat_row = topk_idxs[query_idx * batch + batch_idx]
            valid = flat_row >= 0
            key_indices = flat_row[valid].long() % seq
            q = query[query_idx, batch_idx].float()
            selected_kv = kv[:, batch_idx].float()[key_indices]
            scores = torch.einsum("hd,kd->hk", q, selected_kv) * softmax_scale
            sink = attn_sink.float().view(heads, 1)
            probs = torch.softmax(torch.cat((scores, sink), dim=-1), dim=-1)[..., :-1]
            values = selected_kv[:, :value_dim]
            result = torch.einsum("hk,kv->hv", probs, values)
            out[query_idx, batch_idx] = result.to(query.dtype).reshape(-1)
    return out


def _torch_indexer_topk(
    q_indexer: torch.Tensor,
    k_indexer: torch.Tensor,
    weights: torch.Tensor,
    topk: int,
    ratio: int = 1,
    indexer_softmax_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent CPU oracle for MLite's inference indexer contract."""

    q = q_indexer.permute(1, 0, 2, 3).float()
    k = k_indexer.permute(1, 0, 2).float()
    w = weights.permute(1, 0, 2).float()
    scores = torch.relu(torch.einsum("bqhd,bkd->bqhk", q, k))
    scores = (scores * w.unsqueeze(-1)).sum(dim=2) * indexer_softmax_scale
    query_positions = torch.arange(scores.shape[1], device=scores.device)
    key_positions = torch.arange(scores.shape[2], device=scores.device)
    valid_counts = ((query_positions + 1) // ratio).clamp(max=scores.shape[2])
    valid = key_positions.unsqueeze(0) < valid_counts.unsqueeze(1)
    scores = scores.masked_fill(~valid.unsqueeze(0), -torch.inf)

    effective_topk = min(topk, scores.shape[-1])
    values, indices = torch.topk(scores, effective_topk, dim=-1)
    indices = torch.where(torch.isfinite(values), indices, -torch.ones_like(indices))
    if effective_topk < topk:
        indices = torch.nn.functional.pad(indices, (0, topk - effective_topk), value=-1)
    topk_length = (indices >= 0).sum(dim=-1, dtype=torch.int32)
    return indices.to(torch.int32), topk_length


def test_glm52_shared_attention_matches_transformers_v5_12(
    transformer_engine_import_stub, monkeypatch
):
    """Compare a real HF shared layer with identical MLite weights and top-k.

    A shared layer isolates the main-attention math from the known upstream
    ambiguity around ``indexer_rope_interleave``.  Sequence length is greater
    than one and row 1 attends row 0, so an incorrect half-split main RoPE is
    observable rather than hidden by same-position dot-product invariance.
    """
    transformers = pytest.importorskip("transformers")
    try:
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
            GlmMoeDsaAttention,
            GlmMoeDsaRotaryEmbedding,
        )
    except ImportError:
        pytest.skip("Transformers build does not provide GLM-MoE-DSA.")

    version = tuple(int(part) for part in transformers.__version__.split(".")[:2])
    if version < (5, 12):
        pytest.skip("GLM-5.2 parity requires Transformers >= 5.12.")

    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.attention import dsa

    monkeypatch.setattr(dsa, "RMSNorm", nn.RMSNorm)
    monkeypatch.setattr(dsa._dsa_kernels, "dsa_sparse_attn", _torch_sparse_attention)

    config = transformers.GlmMoeDsaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        index_head_dim=8,
        index_n_heads=2,
        index_topk=2,
        indexer_types=["full", "full", "full", "shared"],
        rope_parameters={"rope_type": "default", "rope_theta": 8_000_000.0},
        rms_norm_eps=1.0e-5,
    )
    config._attn_implementation = "eager"

    torch.manual_seed(20260627)
    hf_attention = GlmMoeDsaAttention(config, layer_idx=3).double().eval()
    mlite_attention = dsa.DynamicSparseAttention(
        hidden_size=16,
        num_attention_heads=2,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=2,
        rms_norm_eps=1.0e-5,
        rope_interleaved=True,
        indexer_rope_interleaved=True,
        indexer_rope_first=True,
        indexer_use_hadamard=False,
        layer_number=4,
        index_topk_freq=4,
        index_skip_topk_offset=3,
        indexer_type="shared",
    ).double().eval()
    mlite_attention.load_state_dict(hf_attention.state_dict(), strict=True)

    hidden_states = torch.randn(1, 4, 16, dtype=torch.float64)
    position_ids = torch.arange(4, dtype=torch.long).unsqueeze(0)
    rotary = GlmMoeDsaRotaryEmbedding(config)
    cos, sin = rotary(hidden_states, position_ids)
    topk = torch.tensor([[[0, 0], [0, 1], [1, 2], [2, 3]]], dtype=torch.int32)

    with torch.no_grad():
        hf_out, _, hf_topk = hf_attention(
            hidden_states,
            (cos, sin),
            None,
            position_ids=position_ids,
            prev_topk_indices=topk,
        )
        state = dsa.DSAIndexShareState({3: 1})
        state.save_topk(3, topk)
        mlite_out = mlite_attention(
            hidden_states,
            cos=cos,
            sin=sin,
            position_ids=position_ids,
            index_share_state=state,
        )

    assert torch.equal(hf_topk, topk)
    torch.testing.assert_close(mlite_out, hf_out, rtol=2.0e-5, atol=2.0e-5)
    print(
        "GLM52_HF_SHARED_ATTN_PARITY "
        f"max_abs={float((mlite_out - hf_out).abs().max().item()):.8e}"
    )


def test_glm52_transformers_v5_12_indexer_layout_divergence_is_explicit(
    transformer_engine_import_stub, monkeypatch
):
    """Prove full HF parity under HF's layout and expose the config conflict.

    Transformers v5.12 hard-codes half-split indexer RoPE while the published
    GLM-5.2 checkpoint sets ``indexer_rope_interleave=true``. Megatron, vLLM,
    and SGLang honor that flag. This test prevents the shared-layer parity gate
    above from being misreported as end-to-end HF parity.
    """

    transformers = pytest.importorskip("transformers")
    version = tuple(int(part) for part in transformers.__version__.split(".")[:2])
    if version != (5, 12):
        pytest.skip("This known-divergence contract is pinned to Transformers 5.12.")
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        GlmMoeDsaAttention,
        GlmMoeDsaRotaryEmbedding,
    )

    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.attention import dsa

    monkeypatch.setattr(dsa, "RMSNorm", nn.RMSNorm)
    monkeypatch.setattr(dsa._dsa_kernels, "dsa_sparse_attn", _torch_sparse_attention)
    captured_topk: list[torch.Tensor] = []

    def recording_indexer_topk(*args, **kwargs):
        topk_indices, topk_length = _torch_indexer_topk(*args, **kwargs)
        captured_topk.append(topk_indices.detach().clone())
        return topk_indices, topk_length

    monkeypatch.setattr(dsa._dsa_kernels, "indexer_topk", recording_indexer_topk)

    config = transformers.GlmMoeDsaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        index_head_dim=8,
        index_n_heads=2,
        index_topk=2,
        indexer_types=["full"] * 4,
        indexer_rope_interleave=True,
        rope_parameters={"rope_type": "default", "rope_theta": 8_000_000.0},
        rms_norm_eps=1.0e-5,
    )
    config._attn_implementation = "eager"

    torch.manual_seed(20260627)
    hf_attention = GlmMoeDsaAttention(config, layer_idx=2).double().eval()

    def make_mlite(*, indexer_rope_interleaved: bool):
        module = dsa.DynamicSparseAttention(
            hidden_size=16,
            num_attention_heads=2,
            q_lora_rank=8,
            kv_lora_rank=4,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=4,
            index_n_heads=2,
            index_head_dim=8,
            index_topk=2,
            rms_norm_eps=1.0e-5,
            rope_interleaved=True,
            indexer_rope_interleaved=indexer_rope_interleaved,
            indexer_rope_first=True,
            indexer_use_hadamard=False,
            layer_number=3,
            indexer_type="full",
            index_share_enabled=False,
        ).double().eval()
        module.load_state_dict(hf_attention.state_dict(), strict=True)
        return module

    mlite_hf_layout = make_mlite(indexer_rope_interleaved=False)
    mlite_checkpoint_layout = make_mlite(indexer_rope_interleaved=True)
    hidden_states = torch.randn(1, 6, 16, dtype=torch.float64)
    position_ids = torch.arange(6, dtype=torch.long).unsqueeze(0)
    rotary = GlmMoeDsaRotaryEmbedding(config)
    cos, sin = rotary(hidden_states, position_ids)

    with torch.no_grad():
        hf_out, _, hf_topk = hf_attention(
            hidden_states,
            (cos, sin),
            None,
            position_ids=position_ids,
        )
        hf_layout_out = mlite_hf_layout(
            hidden_states,
            cos=cos,
            sin=sin,
            position_ids=position_ids,
        )
        checkpoint_layout_out = mlite_checkpoint_layout(
            hidden_states,
            cos=cos,
            sin=sin,
            position_ids=position_ids,
        )

    assert hf_topk is not None
    mlite_topk = captured_topk[0]
    for query_idx in range(position_ids.shape[1]):
        hf_valid = hf_topk[0, query_idx]
        hf_valid = hf_valid[hf_valid <= query_idx].sort().values
        mlite_valid = mlite_topk[0, query_idx]
        mlite_valid = mlite_valid[mlite_valid >= 0].sort().values
        assert torch.equal(hf_valid, mlite_valid)
    hf_layout_max_abs = float((hf_layout_out - hf_out).abs().max().item())
    torch.testing.assert_close(hf_layout_out, hf_out, rtol=5.0e-5, atol=5.0e-5)

    # This is the unresolved source-of-truth conflict, not an accepted parity
    # threshold: the published/configured layout must be gated against the
    # official Megatron implementation before release.
    configured_max_abs = float((checkpoint_layout_out - hf_out).abs().max().item())
    assert configured_max_abs > 1.0e-3
    print(
        "GLM52_HF_INDEXER_LAYOUT_AUDIT "
        f"hf_layout_max_abs={hf_layout_max_abs:.8e} "
        f"checkpoint_layout_max_abs={configured_max_abs:.8e}"
    )
