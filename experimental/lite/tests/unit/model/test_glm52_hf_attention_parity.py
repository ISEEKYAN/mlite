# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Independent GLM-5.2 attention and indexer RoPE authorities.

The production sparse kernel is replaced by a first-principles Torch oracle so
this test can run on CPU. The indexer layout authority is pinned to the released
GLM-5.2 config and vLLM v0.23.0's GPT-J adjacent-pair implementation. The HF
module supplies its public interleaved helper, projection, sparse-mask, softmax,
and value-projection references; vanilla HF half-split indexer behavior remains
an explicit negative control rather than an authority.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.mlite

_ROPE_AUTHORITY_PATH = Path(__file__).with_name("glm52_rope_layout_authority.json")


def _load_rope_authority() -> dict:
    authority = json.loads(_ROPE_AUTHORITY_PATH.read_text())
    release = authority["release"]
    assert release == {
        "config": {
            "path": "config.json",
            "required_values": {
                "indexer_rope_interleave": True,
                "model_type": "glm_moe_dsa",
                "rope_interleave": True,
                "transformers_version": "5.12.0",
            },
            "sha256": (
                "817f5fb39ca5d4c4b5648de89ca00deaea7537d8c2f130172a459252a05c1073"
            ),
        },
        "repo": "zai-org/GLM-5.2",
        "revision": "4d67f66cc64d3219133b767c253b2ad1425c6c88",
    }
    vllm = authority["vllm"]
    assert vllm["release"] == "v0.23.0"
    assert vllm["revision"] == "0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665"
    assert vllm["files"]["vllm/model_executor/models/deepseek_v2.py"]["sha256"] == (
        "640b6e60fbacf859f60caff149015dbfc223dff0baf302c507ddd9009b7726a3"
    )
    assert vllm["files"]["vllm/model_executor/layers/rotary_embedding/common.py"][
        "sha256"
    ] == ("86c9f61904fd3066fd146f0f5b3c5932dd8252841947ffcfd699b18e6a8bde08")
    assert vllm["adjacent_pair_contract"] == {
        "config_field": "indexer_rope_interleave",
        "config_value": True,
        "is_neox_style": False,
        "layout": "gptj_adjacent_pair",
    }
    return authority


def _vllm_gptj_forward_static(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Pinned vLLM v0.23 ``forward_static(..., is_neox_style=False)`` formula."""

    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.stack((o1, o2), dim=-1).flatten(-2)


def _indexer_score_matrix(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    *,
    ratio: int,
    indexer_softmax_scale: float,
) -> torch.Tensor:
    """Return the complete causal score matrix from batch-first indexer inputs."""

    scores = torch.relu(torch.einsum("bqhd,bkd->bqhk", q.float(), k.float()))
    scores = (scores * weights.float().unsqueeze(-1)).sum(dim=2)
    scores = scores * indexer_softmax_scale
    query_positions = torch.arange(scores.shape[1], device=scores.device)
    key_positions = torch.arange(scores.shape[2], device=scores.device)
    valid_counts = ((query_positions + 1) // ratio).clamp(max=scores.shape[2])
    valid = key_positions.unsqueeze(0) < valid_counts.unsqueeze(1)
    return scores.masked_fill(~valid.unsqueeze(0), -torch.inf)


def _topk_from_score_matrix(
    scores: torch.Tensor, topk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    effective_topk = min(topk, scores.shape[-1])
    values, indices = torch.topk(scores, effective_topk, dim=-1)
    indices = torch.where(torch.isfinite(values), indices, -torch.ones_like(indices))
    if effective_topk < topk:
        indices = torch.nn.functional.pad(indices, (0, topk - effective_topk), value=-1)
    topk_length = (indices >= 0).sum(dim=-1, dtype=torch.int32)
    return indices.to(torch.int32), topk_length


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
    scores = _indexer_score_matrix(
        q,
        k,
        w,
        ratio=ratio,
        indexer_softmax_scale=indexer_softmax_scale,
    )
    return _topk_from_score_matrix(scores, topk)


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
    mlite_attention = (
        dsa.DynamicSparseAttention(
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
        )
        .double()
        .eval()
    )
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


def test_glm52_release_vllm_hf_helper_and_mlite_indexer_rope_authority(
    transformer_engine_import_stub, monkeypatch
):
    """Bind real MLite indexer scores/top-k to release and vLLM authorities.

    This does not claim that another Megatron implementation or Bridge already
    maps the release field. It proves the MLite ``DSAIndexer`` behavior directly.
    """

    transformers = pytest.importorskip("transformers")
    assert transformers.__version__ == "5.12.0"
    from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as hf_impl

    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.attention import dsa

    authority = _load_rope_authority()
    golden = authority["vllm"]["gptj_forward_static_golden"]
    assert golden["dtype"] == "float64"
    golden_x = torch.tensor(golden["x"], dtype=torch.float64)
    golden_cos = torch.tensor(golden["cos"], dtype=torch.float64)
    golden_sin = torch.tensor(golden["sin"], dtype=torch.float64)
    golden_output = torch.tensor(golden["output"], dtype=torch.float64)
    vllm_golden_output = _vllm_gptj_forward_static(golden_x, golden_cos, golden_sin)
    torch.testing.assert_close(
        vllm_golden_output, golden_output, rtol=0.0, atol=1.0e-15
    )
    hf_golden_output, _ = hf_impl.apply_rotary_pos_emb_interleave(
        golden_x,
        golden_x,
        torch.cat((golden_cos, golden_cos), dim=-1),
        torch.cat((golden_sin, golden_sin), dim=-1),
        unsqueeze_dim=2,
    )
    half = hf_golden_output.shape[-1] // 2
    hf_golden_as_gptj = torch.stack(
        (hf_golden_output[..., :half], hf_golden_output[..., half:]), dim=-1
    ).flatten(-2)
    torch.testing.assert_close(hf_golden_as_gptj, golden_output, rtol=0.0, atol=1.0e-15)

    indexer = (
        dsa.DSAIndexer(
            hidden_size=8,
            q_lora_rank=6,
            qk_rope_head_dim=4,
            index_n_heads=3,
            index_head_dim=6,
            index_topk=3,
            rope_interleaved=True,
            layer_norm_eps=1.0e-6,
            rope_first=True,
            use_hadamard=False,
        )
        .double()
        .eval()
    )
    with torch.no_grad():
        for parameter_index, parameter in enumerate(indexer.parameters()):
            coordinates = torch.arange(parameter.numel(), dtype=torch.float64)
            values = torch.sin(coordinates * 0.37 + parameter_index * 0.71)
            parameter.copy_(values.reshape_as(parameter) * 0.4)

    sequence_length = 7
    hidden_states = torch.sin(
        torch.arange(sequence_length * 8, dtype=torch.float64) * 0.19 + 0.3
    ).reshape(1, sequence_length, 8)
    q_resid = torch.cos(
        torch.arange(sequence_length * 6, dtype=torch.float64) * 0.23 - 0.2
    ).reshape(1, sequence_length, 6)
    position_ids = torch.arange(sequence_length, dtype=torch.long).unsqueeze(0)
    cos, sin = dsa.build_rotary_embeddings(
        position_ids=position_ids,
        dim=indexer.qk_rope_head_dim,
        rope_theta=97.0,
        dtype=torch.float64,
    )

    with torch.no_grad():
        mlite_q_sbh, mlite_k_sb, mlite_weights_sb = indexer.forward_before_topk(
            hidden_states, q_resid, cos, sin, position_ids
        )
        raw_q = indexer.wq_b(q_resid).view(
            1, sequence_length, indexer.num_heads, indexer.head_dim
        )
        raw_k = indexer.k_norm(indexer.wk(hidden_states)).unsqueeze(2)
        raw_q_rot, raw_q_pass = torch.split(
            raw_q,
            [indexer.qk_rope_head_dim, indexer.qk_nope_head_dim],
            dim=-1,
        )
        raw_k_rot, raw_k_pass = torch.split(
            raw_k,
            [indexer.qk_rope_head_dim, indexer.qk_nope_head_dim],
            dim=-1,
        )
        hf_q_rot, hf_k_rot = hf_impl.apply_rotary_pos_emb_interleave(
            raw_q_rot,
            raw_k_rot,
            cos,
            sin,
            unsqueeze_dim=2,
        )
        hf_q = torch.cat((hf_q_rot, raw_q_pass), dim=-1)
        hf_k = torch.cat((hf_k_rot, raw_k_pass), dim=-1).squeeze(2)
        vllm_q_rot = _vllm_gptj_forward_static(
            raw_q_rot,
            cos[..., : indexer.qk_rope_head_dim // 2],
            sin[..., : indexer.qk_rope_head_dim // 2],
        )
        vllm_k_rot = _vllm_gptj_forward_static(
            raw_k_rot,
            cos[..., : indexer.qk_rope_head_dim // 2],
            sin[..., : indexer.qk_rope_head_dim // 2],
        )
        vllm_q = torch.cat((vllm_q_rot, raw_q_pass), dim=-1)
        vllm_k = torch.cat((vllm_k_rot, raw_k_pass), dim=-1).squeeze(2)
        vanilla_q_rot, vanilla_k_rot = hf_impl.apply_rotary_pos_emb(
            raw_q_rot,
            raw_k_rot,
            cos,
            sin,
            unsqueeze_dim=2,
        )
        vanilla_q = torch.cat((vanilla_q_rot, raw_q_pass), dim=-1)
        vanilla_k = torch.cat((vanilla_k_rot, raw_k_pass), dim=-1).squeeze(2)

    mlite_q = mlite_q_sbh.transpose(0, 1)
    mlite_k = mlite_k_sb.transpose(0, 1)
    mlite_weights = mlite_weights_sb.transpose(0, 1)
    torch.testing.assert_close(mlite_q, hf_q, rtol=0.0, atol=1.0e-12)
    torch.testing.assert_close(mlite_k, hf_k, rtol=0.0, atol=1.0e-12)
    authority_scores = _indexer_score_matrix(
        hf_q,
        hf_k,
        mlite_weights,
        ratio=1,
        indexer_softmax_scale=indexer.softmax_scale,
    )
    vllm_scores = _indexer_score_matrix(
        vllm_q,
        vllm_k,
        mlite_weights,
        ratio=1,
        indexer_softmax_scale=indexer.softmax_scale,
    )
    mlite_scores = _indexer_score_matrix(
        mlite_q,
        mlite_k,
        mlite_weights,
        ratio=1,
        indexer_softmax_scale=indexer.softmax_scale,
    )
    vanilla_scores = _indexer_score_matrix(
        vanilla_q,
        vanilla_k,
        mlite_weights,
        ratio=1,
        indexer_softmax_scale=indexer.softmax_scale,
    )
    torch.testing.assert_close(mlite_scores, authority_scores, rtol=0.0, atol=1.0e-12)
    # vLLM keeps adjacent-pair output ordering while HF/MLite group the first
    # and second pair components. The dot product is permutation-equivalent;
    # FP32 accumulation order can differ by one ULP.
    torch.testing.assert_close(vllm_scores, authority_scores, rtol=1.0e-6, atol=1.0e-7)

    recorded_kernel_scores: list[torch.Tensor] = []

    def _recording_indexer_topk(
        q_indexer,
        k_indexer,
        weights,
        topk,
        ratio=1,
        indexer_softmax_scale=1.0,
    ):
        scores = _indexer_score_matrix(
            q_indexer.permute(1, 0, 2, 3),
            k_indexer.permute(1, 0, 2),
            weights.permute(1, 0, 2),
            ratio=ratio,
            indexer_softmax_scale=indexer_softmax_scale,
        )
        recorded_kernel_scores.append(scores.detach().clone())
        return _topk_from_score_matrix(scores, topk)

    monkeypatch.setattr(dsa._dsa_kernels, "indexer_topk", _recording_indexer_topk)
    with torch.no_grad():
        mlite_topk = indexer(hidden_states, q_resid, cos, sin, position_ids)
    assert len(recorded_kernel_scores) == 1
    torch.testing.assert_close(
        recorded_kernel_scores[0], authority_scores, rtol=0.0, atol=1.0e-12
    )
    authority_topk, _ = _topk_from_score_matrix(authority_scores, indexer.index_topk)
    vllm_topk, _ = _topk_from_score_matrix(vllm_scores, indexer.index_topk)
    assert torch.equal(mlite_topk, authority_topk)
    assert torch.equal(vllm_topk, authority_topk)

    vanilla_topk, _ = _topk_from_score_matrix(vanilla_scores, indexer.index_topk)
    finite = torch.isfinite(authority_scores)
    vanilla_score_max_abs = float(
        (vanilla_scores[finite] - authority_scores[finite]).abs().max().item()
    )
    differing_topk_rows = int(
        torch.any(vanilla_topk != authority_topk, dim=-1).sum().item()
    )
    assert vanilla_score_max_abs > 1.0e-3
    assert differing_topk_rows > 0
    print(
        "GLM52_RELEASE_VLLM_MLITE_INDEXER_ROPE_AUTHORITY_PASSED "
        f"release_revision={authority['release']['revision']} "
        f"vllm_revision={authority['vllm']['revision']} "
        f"score_shape={tuple(authority_scores.shape)} "
        f"vanilla_score_max_abs={vanilla_score_max_abs:.8e} "
        f"vanilla_differing_topk_rows={differing_topk_rows}"
    )
