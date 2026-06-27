# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import copy

import pytest
import torch

pytestmark = [pytest.mark.mlite, pytest.mark.smoke, pytest.mark.gpu]

_DIRECT_PUBLIC_OUTPUT_ATOL = 2.0e-3
_DIRECT_PUBLIC_LOSS_ATOL = 1.0e-6
_GLM_X_GRAD_MAX_ABS = 1.0e-6
_GLM_PARAM_GRAD_MAX_ABS = 4.0e-6
_INDEXER_LOSS_ATOL = 1.0e-6
_INDEXER_LOSS_RTOL = 5.0e-3
# Main-attention comparisons use a scale-independent envelope plus a
# path-specific outlier ceiling. Keep these separate from the stricter
# indexer-loss oracle above so their tolerances cannot weaken it.
_MAIN_LOSS_MAX_ABS_DIFF = 1.0e-5
_MAX_MAIN_LOSS_SYMMETRIC_REL = 1.0e-3
_GLM_FUSED_R2R_OUTPUT_MAX_ABS = 2.0e-2
_GLM_FUSED_VS_REFERENCE_OUTPUT_MAX_ABS = 4.0e-2
_DSV4_FUSED_VS_DECOMPOSED_OUTPUT_MAX_ABS = 2.0e-2
_DSV4_CSA_OUTPUT_MAX_ABS = 6.0e-3
_MIN_MAIN_OUTPUT_COSINE = 0.999
_MAX_MAIN_OUTPUT_RMS_REL = 0.02
_MIN_MAIN_OUTPUT_NORM_RATIO = 0.99
_MAX_MAIN_OUTPUT_NORM_RATIO = 1.01
_INDEXER_TOPK = 512
_SEQUENCE_LENGTH = 1024
_INDEXER_LOSS_COEFF = 1.0e-2
_MIN_INDEXER_GRAD_MAX_ABS = 1.0e-8
# Four-rank GB300 acceptance high-water marks were 0.999462 cosine, 0.032821
# RMS-relative, and 0.997976--1.001816 norm ratio for indexer gradients; DSv4
# main gradients reached 0.999956 cosine, 0.009290 RMS-relative, and
# 0.998507--1.000822 norm ratio. Keep bounded BF16/kernel headroom without
# retaining the former 0.99/0.20/0.90--1.10 compatibility envelope.
_MIN_INDEXER_GRAD_COSINE = 0.999
_MAX_INDEXER_GRAD_RMS_REL = 0.05
_MIN_INDEXER_GRAD_NORM_RATIO = 0.995
_MAX_INDEXER_GRAD_NORM_RATIO = 1.005
_MIN_MAIN_GRAD_COSINE = 0.9999
_MAX_MAIN_GRAD_RMS_REL = 0.02
_MIN_MAIN_GRAD_NORM_RATIO = 0.997
_MAX_MAIN_GRAD_NORM_RATIO = 1.003
_TOPK_SCORE_ATOL = 1.0e-5
_TOPK_SCORE_RTOL = 1.0e-4


def _make_dsa_pair(*, sparse_loss: bool):
    pytest.importorskip("cudnn", reason="GLM5 DSA accept-with-proof needs cudnn DSA.")
    from megatron.lite.primitive.modules.attention import DynamicSparseAttention

    common = dict(
        hidden_size=128,
        num_attention_heads=64,
        q_lora_rank=16,
        kv_lora_rank=512,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=_INDEXER_TOPK,
        rms_norm_eps=1e-5,
        rope_interleaved=True,
        indexer_rope_interleaved=True,
        index_topk_freq=2,
        index_skip_topk_offset=1,
        index_share_enabled=True,
        indexer_loss_coeff=_INDEXER_LOSS_COEFF,
        indexer_use_sparse_loss=sparse_loss,
    )
    source = DynamicSparseAttention(
        **common,
        layer_number=1,
        indexer_type="full",
        index_share_source_layer=1,
    )
    shared = DynamicSparseAttention(
        **common,
        layer_number=2,
        indexer_type="shared",
        index_share_source_layer=1,
    )
    assert source.indexer is not None
    assert shared.indexer is None
    return torch.nn.ModuleDict({"source": source, "shared": shared})


def _run_once(modules, x, cos, sin, position_ids, *, fused_training: bool):
    from megatron.lite.primitive.modules.attention import DSAIndexShareState

    class RecordingState(DSAIndexShareState):
        def __init__(self):
            super().__init__({1: 1})
            self.saved: list[torch.Tensor] = []
            self.consumed: list[torch.Tensor] = []

        def save_topk(self, layer_number, topk_indices, *, sequence_key=None):
            self.saved.append(topk_indices.detach().clone())
            return super().save_topk(
                layer_number, topk_indices, sequence_key=sequence_key
            )

        def get_topk(self, layer_number, source_layer, *, sequence_key=None):
            topk_indices = super().get_topk(
                layer_number, source_layer, sequence_key=sequence_key
            )
            self.consumed.append(topk_indices.detach().clone())
            return topk_indices

    modules.zero_grad(set_to_none=True)
    modules.train(fused_training)
    local_x = x.detach().clone().requires_grad_(True)
    index_share_state = RecordingState()
    source_out = modules["source"](
        local_x,
        cos=cos,
        sin=sin,
        position_ids=position_ids,
        index_share_state=index_share_state,
    )
    indexer_loss = _saved_indexer_loss(source_out)
    assert index_share_state.cached_tensor_count == 1
    hidden = local_x + source_out
    shared_out = modules["shared"](
        hidden,
        cos=cos,
        sin=sin,
        position_ids=position_ids,
        index_share_state=index_share_state,
    )
    out = hidden + shared_out
    assert len(index_share_state.saved) == 1
    assert len(index_share_state.consumed) == 1
    assert torch.equal(index_share_state.saved[0], index_share_state.consumed[0])
    assert index_share_state.cached_tensor_count == 0
    loss = out.float().square().mean()
    loss.backward()
    param_grads = {
        name: param.grad.detach().float().clone()
        for name, param in modules.named_parameters()
        if param.grad is not None
    }
    return {
        "loss": loss.detach().float().clone(),
        "indexer_loss": indexer_loss,
        "out": out.detach().float().clone(),
        "x_grad": local_x.grad.detach().float().clone(),
        "param_grads": param_grads,
        "topk_indices": index_share_state.consumed[0],
    }


def _saved_indexer_loss(output: torch.Tensor) -> torch.Tensor:
    """Extract the loss attached by DSAIndexerLossAutoScaler before backward."""

    stack = [output.grad_fn]
    # Keep strong references to visited nodes. Storing only ids permits an
    # autograd wrapper to be collected and its id reused during traversal.
    visited: dict[int, object] = {}
    matches: list[torch.Tensor] = []
    while stack:
        grad_fn = stack.pop()
        if grad_fn is None or id(grad_fn) in visited:
            continue
        visited[id(grad_fn)] = grad_fn
        if type(grad_fn).__name__ == "DSAIndexerLossAutoScalerBackward":
            (indexer_loss,) = grad_fn.saved_tensors
            matches.append(indexer_loss)
        stack.extend(parent for parent, _index in grad_fn.next_functions)
    assert len(matches) == 1, (
        f"expected exactly one DSAIndexerLossAutoScalerBackward, found {len(matches)}"
    )
    indexer_loss = matches[0]
    assert indexer_loss.ndim == 0
    assert torch.isfinite(indexer_loss)
    return indexer_loss.detach().float().clone()


def _causal_mask(
    seq_q: int, seq_k: int, *, ratio: int, device: torch.device
) -> torch.Tensor:
    q_idx = torch.arange(seq_q, device=device)
    k_idx = torch.arange(seq_k, device=device)
    q_global_start = seq_k * ratio - seq_q
    valid_per_q = torch.div(
        q_global_start + q_idx + 1,
        ratio,
        rounding_mode="floor",
    ).clamp(min=0, max=seq_k)
    return k_idx.unsqueeze(0) < valid_per_q.unsqueeze(1)


def _torch_indexer_scores(
    q_indexer: torch.Tensor,
    k_indexer: torch.Tensor,
    weights: torch.Tensor,
    *,
    ratio: int,
    indexer_softmax_scale: float,
) -> torch.Tensor:
    q_bshd = q_indexer.permute(1, 0, 2, 3).float()
    k_bsd = k_indexer.permute(1, 0, 2).float()
    # Production moves the positive softmax scale onto W, then quantizes the
    # scaled weights back to BF16 before the indexer GEMM. Reproduce that
    # boundary so exact top-k checks do not fail on an FP32/BF16 rank flip.
    w_bsh = (
        (weights.permute(1, 0, 2).float() * float(indexer_softmax_scale))
        .to(weights.dtype)
        .float()
    )
    scores = torch.einsum("bqhd,bkd->bqhk", q_bshd, k_bsd)
    scores = torch.relu(scores).mul(w_bsh.unsqueeze(-1)).sum(dim=2)
    causal = _causal_mask(
        scores.shape[1], scores.shape[2], ratio=ratio, device=scores.device
    )
    return torch.where(causal.unsqueeze(0), scores, torch.full_like(scores, -torch.inf))


def _torch_dense_indexer_scores(
    q_indexer: torch.Tensor,
    k_indexer: torch.Tensor,
    weights: torch.Tensor,
    *,
    ratio: int,
    indexer_softmax_scale: float,
) -> torch.Tensor:
    """Canonical full-KV score used by cuDNN dense score/backward."""

    q_bshd = q_indexer.permute(1, 0, 2, 3).float()
    k_bsd = k_indexer.permute(1, 0, 2).float()
    w_bsh = weights.permute(1, 0, 2).float()
    scores = torch.einsum("bqhd,bkd->bqhk", q_bshd, k_bsd)
    scores = torch.relu(scores).mul(w_bsh.unsqueeze(-1)).sum(dim=2)
    scores = scores * float(indexer_softmax_scale)
    causal = _causal_mask(
        scores.shape[1], scores.shape[2], ratio=ratio, device=scores.device
    )
    return torch.where(causal.unsqueeze(0), scores, torch.full_like(scores, -torch.inf))


def _torch_topk_from_scores(scores: torch.Tensor, topk: int) -> torch.Tensor:
    effective_topk = min(topk, scores.shape[-1])
    values, indices = torch.topk(scores, k=effective_topk, dim=-1)
    indices = torch.where(torch.isfinite(values), indices, torch.full_like(indices, -1))
    if effective_topk < topk:
        pad = torch.full(
            (*indices.shape[:-1], topk - effective_topk),
            -1,
            device=indices.device,
            dtype=indices.dtype,
        )
        indices = torch.cat([indices, pad], dim=-1)
    return indices.int()


def _gather_sequence(source: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    safe_indices = indices.clamp(min=0).long()
    expanded = source.unsqueeze(1).expand(-1, indices.shape[1], -1, -1)
    gathered = torch.gather(
        expanded,
        dim=2,
        index=safe_indices.unsqueeze(-1).expand(-1, -1, -1, source.shape[-1]),
    )
    return torch.where(indices.unsqueeze(-1) >= 0, gathered, torch.zeros_like(gathered))


def _torch_sparse_attention(
    query_states: torch.Tensor,
    kv_full: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_indices: torch.Tensor,
    *,
    softmax_scale: float,
    value_dim: int,
) -> torch.Tensor:
    query_bshd = query_states.permute(1, 0, 2, 3).float()
    kv_bsd = kv_full.permute(1, 0, 2).float()
    selected_keys = _gather_sequence(kv_bsd, topk_indices)
    selected_values = _gather_sequence(kv_bsd[..., :value_dim], topk_indices)
    scores = torch.einsum("bqhd,bqtd->bqht", query_bshd, selected_keys)
    scores = scores * float(softmax_scale)
    scores = torch.where(
        topk_indices.unsqueeze(2) >= 0, scores, torch.full_like(scores, -torch.inf)
    )
    sink = (
        attn_sink.float()
        .view(1, 1, -1, 1)
        .expand(scores.shape[0], scores.shape[1], -1, -1)
    )
    probs = torch.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :-1]
    out = torch.einsum("bqht,bqtr->bqhr", probs, selected_values)
    return out.permute(1, 0, 2, 3).reshape(
        query_states.shape[0], query_states.shape[1], query_states.shape[2] * value_dim
    )


def _torch_sparse_indexer_loss(
    indexer_scores: torch.Tensor,
    query_states: torch.Tensor,
    kv_full: torch.Tensor,
    topk_indices: torch.Tensor,
    *,
    softmax_scale: float,
    loss_coeff: float,
) -> torch.Tensor:
    """Independent sparse KL oracle for the fused indexer-loss backward."""

    valid = topk_indices >= 0
    query_bshd = query_states.detach().permute(1, 0, 2, 3).float()
    kv_bsd = kv_full.detach().permute(1, 0, 2).float()
    selected_keys = _gather_sequence(kv_bsd, topk_indices)
    attn_scores = torch.einsum("bqhd,bqtd->bqht", query_bshd, selected_keys)
    attn_scores = attn_scores * float(softmax_scale)
    attn_scores = torch.where(
        valid.unsqueeze(2), attn_scores, torch.full_like(attn_scores, -torch.inf)
    )
    row_valid = valid.any(dim=-1)
    safe_attn_scores = torch.where(
        row_valid.view(*row_valid.shape, 1, 1),
        attn_scores,
        torch.zeros_like(attn_scores),
    )
    attn_probs = torch.softmax(safe_attn_scores, dim=-1)
    attn_probs = torch.where(
        row_valid.view(*row_valid.shape, 1, 1),
        attn_probs,
        torch.zeros_like(attn_probs),
    )
    target_mass = attn_probs.sum(dim=2)
    target_mass = torch.where(valid, target_mass, torch.zeros_like(target_mass))

    target_denom = target_mass.sum(dim=-1, keepdim=True).clamp_min(1.0e-10)
    target = target_mass / target_denom

    safe_indices = topk_indices.clamp(min=0).long()
    selected_indexer_scores = torch.gather(indexer_scores, dim=-1, index=safe_indices)
    selected_indexer_scores = torch.where(
        valid,
        selected_indexer_scores,
        torch.full_like(selected_indexer_scores, torch.finfo(torch.float32).min),
    )
    predict = torch.softmax(selected_indexer_scores, dim=-1)

    eps = 1.0e-10
    kl_per_row = (target * (torch.log(target + eps) - torch.log(predict + eps))).sum(
        dim=-1
    )
    kl_per_row = torch.where(row_valid, kl_per_row, torch.zeros_like(kl_per_row))
    return float(loss_coeff) * kl_per_row.mean()


def _torch_dense_indexer_loss(
    indexer_scores: torch.Tensor,
    query_states: torch.Tensor,
    kv_full: torch.Tensor,
    *,
    softmax_scale: float,
    loss_coeff: float,
) -> torch.Tensor:
    """Independent full-KV KL oracle for GLM's default dense loss."""

    query_bshd = query_states.detach().permute(1, 0, 2, 3).float()
    kv_bsd = kv_full.detach().permute(1, 0, 2).float()
    attn_scores = torch.einsum("bqhd,bkd->bqhk", query_bshd, kv_bsd)
    attn_scores = attn_scores * float(softmax_scale)
    causal = _causal_mask(
        attn_scores.shape[1], attn_scores.shape[-1], ratio=1, device=attn_scores.device
    )
    attn_scores = torch.where(
        causal.view(1, causal.shape[0], 1, causal.shape[1]),
        attn_scores,
        torch.full_like(attn_scores, -torch.inf),
    )
    # Canonical dense DSA: every attention head independently softmaxes over
    # the complete causal KV axis, then the head probabilities are summed and
    # L1-normalized. No top-k tensor or attention sink participates.
    row_valid = causal.any(dim=-1).unsqueeze(0)
    safe_attn_scores = torch.where(
        row_valid.view(*row_valid.shape, 1, 1),
        attn_scores,
        torch.zeros_like(attn_scores),
    )
    attn_probs = torch.softmax(safe_attn_scores, dim=-1)
    attn_probs = torch.where(
        row_valid.view(*row_valid.shape, 1, 1),
        attn_probs,
        torch.zeros_like(attn_probs),
    )
    target_mass = attn_probs.sum(dim=2)
    target_mass = torch.where(
        causal.unsqueeze(0), target_mass, torch.zeros_like(target_mass)
    )
    target_denom = target_mass.sum(dim=-1, keepdim=True).clamp_min(1.0e-10)
    target = target_mass / target_denom

    safe_indexer_scores = torch.where(
        row_valid.unsqueeze(-1), indexer_scores, torch.zeros_like(indexer_scores)
    )
    predict = torch.softmax(safe_indexer_scores, dim=-1)
    eps = 1.0e-10
    terms = target * (torch.log(target + eps) - torch.log(predict + eps))
    terms = torch.where(causal.unsqueeze(0), terms, torch.zeros_like(terms))
    kl_per_row = terms.sum(dim=-1)
    kl_per_row = torch.where(row_valid, kl_per_row, torch.zeros_like(kl_per_row))
    return float(loss_coeff) * kl_per_row.mean()


def _torch_unfused_dsa_forward(
    module, x, cos, sin, position_ids, *, topk_indices: torch.Tensor | None = None
):
    from megatron.lite.primitive.modules.attention.dsa import (
        _rotary_embeddings_from_cache,
        apply_rotary_pos_emb,
    )

    batch, seq_len, _ = x.shape
    q_resid = module.q_a_layernorm(module.q_a_proj(x))
    q = module.q_b_proj(q_resid).view(
        batch, seq_len, module.num_heads, module.qk_head_dim
    )
    q_nope, q_pe = torch.split(
        q, [module.qk_nope_head_dim, module.qk_rope_head_dim], dim=-1
    )
    cos, sin = _rotary_embeddings_from_cache(
        cos,
        sin,
        position_ids,
        device=x.device,
        dtype=x.dtype,
        dim=module.qk_rope_head_dim,
    )
    q_pe = apply_rotary_pos_emb(
        q_pe,
        cos,
        sin,
        unsqueeze_dim=2,
        mla_interleaved=module.rope_interleaved,
    )

    k_up_weight, v_up_weight = module._split_kv_b_weights()
    q_nope = torch.einsum("bshd,hdr->bshr", q_nope, k_up_weight)
    query_states = torch.cat([q_nope, q_pe], dim=-1).transpose(0, 1).contiguous()

    kv_latent, k_pe = torch.split(
        module.kv_a_proj_with_mqa(x),
        [module.kv_lora_rank, module.qk_rope_head_dim],
        dim=-1,
    )
    kv_latent = module.kv_a_layernorm(kv_latent)
    k_pe = apply_rotary_pos_emb(
        k_pe.unsqueeze(2),
        cos,
        sin,
        unsqueeze_dim=2,
        mla_interleaved=module.rope_interleaved,
    ).squeeze(2)
    kv_full = torch.cat([kv_latent, k_pe], dim=-1).transpose(0, 1).contiguous()

    indexer_loss = x.new_zeros((), dtype=torch.float32)
    indexer_scores = None
    if module.indexer is not None:
        q_indexer, k_indexer, weights_indexer = module.indexer.forward_before_topk(
            x.detach(), q_resid.detach(), cos, sin, position_ids
        )
        indexer_scores = _torch_indexer_scores(
            q_indexer,
            k_indexer,
            weights_indexer,
            ratio=1,
            indexer_softmax_scale=module.indexer_softmax_scale,
        )
        if topk_indices is None:
            topk_indices = _torch_topk_from_scores(
                indexer_scores, min(module.index_topk, indexer_scores.shape[-1])
            )
        else:
            # Selection correctness is proved independently from the quantized
            # indexer scores.  Reusing the vendor-selected indices here makes
            # the attention/loss/gradient comparison well-defined when several
            # keys tie at the top-k cutoff but have different attention values.
            topk_indices = topk_indices.detach()
        if module.indexer_use_sparse_loss:
            indexer_loss = _torch_sparse_indexer_loss(
                indexer_scores,
                query_states,
                kv_full,
                topk_indices,
                softmax_scale=module.softmax_scale,
                loss_coeff=module.indexer_loss_coeff,
            )
        else:
            dense_indexer_scores = _torch_dense_indexer_scores(
                q_indexer,
                k_indexer,
                weights_indexer,
                ratio=1,
                indexer_softmax_scale=module.indexer_softmax_scale,
            )
            indexer_loss = _torch_dense_indexer_loss(
                dense_indexer_scores,
                query_states,
                kv_full,
                softmax_scale=module.softmax_scale,
                loss_coeff=module.indexer_loss_coeff,
            )
    else:
        assert topk_indices is not None
    out = _torch_sparse_attention(
        query_states,
        kv_full,
        module.attn_sink,
        topk_indices,
        softmax_scale=module.softmax_scale,
        value_dim=module.kv_lora_rank,
    )
    out = out.to(x.dtype).view(seq_len, batch, module.num_heads, module.kv_lora_rank)
    out = out.permute(1, 0, 2, 3).contiguous()
    out = torch.einsum("bshr,hvr->bshv", out, v_up_weight)
    out = out.reshape(batch, seq_len, module.num_heads * module.v_head_dim)
    out = module.o_proj(out)
    return out, topk_indices, indexer_loss, indexer_scores


def _run_once_torch_unfused(
    modules,
    x,
    cos,
    sin,
    position_ids,
    *,
    forced_source_topk: torch.Tensor | None = None,
):
    modules.zero_grad(set_to_none=True)
    modules.train(True)
    local_x = x.detach().clone().requires_grad_(True)
    source_out, topk_indices, indexer_loss, indexer_scores = _torch_unfused_dsa_forward(
        modules["source"],
        local_x,
        cos,
        sin,
        position_ids,
        topk_indices=forced_source_topk,
    )
    assert indexer_scores is not None
    hidden = local_x + source_out
    shared_out, reused_topk_indices, shared_indexer_loss, shared_indexer_scores = (
        _torch_unfused_dsa_forward(
            modules["shared"],
            hidden,
            cos,
            sin,
            position_ids,
            topk_indices=topk_indices,
        )
    )
    assert shared_indexer_scores is None
    assert torch.equal(reused_topk_indices, topk_indices)
    assert float(shared_indexer_loss.item()) == 0.0
    out = hidden + shared_out
    loss = out.float().square().mean()
    (loss + indexer_loss).backward()
    param_grads = {
        name: param.grad.detach().float().clone()
        for name, param in modules.named_parameters()
        if param.grad is not None
    }
    return {
        "loss": loss.detach().float().clone(),
        "indexer_loss": indexer_loss.detach().float().clone(),
        "out": out.detach().float().clone(),
        "x_grad": local_x.grad.detach().float().clone(),
        "param_grads": param_grads,
        "topk_indices": topk_indices.detach().clone(),
        "indexer_scores": indexer_scores.detach().float().clone(),
    }


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def _symmetric_relative_difference(actual: float, expected: float) -> float:
    denominator = abs(actual) + abs(expected)
    if denominator == 0.0:
        return 0.0
    return 2.0 * abs(actual - expected) / denominator


def _tensor_similarity_metrics(
    actual: torch.Tensor, expected: torch.Tensor
) -> dict[str, float]:
    actual_flat = actual.detach().float().reshape(-1)
    expected_flat = expected.detach().float().reshape(-1)
    assert torch.isfinite(actual_flat).all()
    assert torch.isfinite(expected_flat).all()
    actual_norm = torch.linalg.vector_norm(actual_flat)
    expected_norm = torch.linalg.vector_norm(expected_flat)
    actual_norm_value = float(actual_norm.item())
    expected_norm_value = float(expected_norm.item())
    if actual_norm_value == 0.0 and expected_norm_value == 0.0:
        return {
            "cosine": 1.0,
            "rms_relative": 0.0,
            "norm_ratio": 1.0,
            "max_abs": 0.0,
        }
    if expected_norm_value == 0.0:
        return {
            "cosine": 0.0,
            "rms_relative": float("inf"),
            "norm_ratio": float("inf"),
            "max_abs": _max_abs(actual_flat, expected_flat),
        }
    if actual_norm_value == 0.0:
        return {
            "cosine": 0.0,
            "rms_relative": 1.0,
            "norm_ratio": 0.0,
            "max_abs": _max_abs(actual_flat, expected_flat),
        }
    cosine = torch.dot(actual_flat, expected_flat) / (actual_norm * expected_norm)
    rms_diff = torch.sqrt(torch.mean((actual_flat - expected_flat).square()))
    rms_expected = torch.sqrt(torch.mean(expected_flat.square()))
    return {
        "cosine": float(cosine.item()),
        "rms_relative": float((rms_diff / rms_expected).item()),
        "norm_ratio": float((actual_norm / expected_norm).item()),
        "max_abs": _max_abs(actual_flat, expected_flat),
    }


def _similarity_extrema(
    comparisons: dict[str, dict[str, float]],
) -> dict[str, float]:
    assert comparisons
    return {
        "min_cosine": min(value["cosine"] for value in comparisons.values()),
        "max_rms_relative": max(
            value["rms_relative"] for value in comparisons.values()
        ),
        "min_norm_ratio": min(value["norm_ratio"] for value in comparisons.values()),
        "max_norm_ratio": max(value["norm_ratio"] for value in comparisons.values()),
        "max_abs": max(value["max_abs"] for value in comparisons.values()),
    }


def _assert_main_output_similarity(
    comparisons: dict[str, dict[str, float]], *, max_abs: float
) -> dict[str, float]:
    """Require both scale-independent agreement and a bounded worst outlier."""

    extrema = _similarity_extrema(comparisons)
    assert extrema["min_cosine"] >= _MIN_MAIN_OUTPUT_COSINE, comparisons
    assert extrema["max_rms_relative"] <= _MAX_MAIN_OUTPUT_RMS_REL, comparisons
    assert extrema["min_norm_ratio"] >= _MIN_MAIN_OUTPUT_NORM_RATIO, comparisons
    assert extrema["max_norm_ratio"] <= _MAX_MAIN_OUTPUT_NORM_RATIO, comparisons
    assert extrema["max_abs"] <= max_abs, comparisons
    return extrema


def _canonical_topk_set(topk_indices: torch.Tensor) -> torch.Tensor:
    """Canonicalize the vendor top-k's intentionally unspecified ordering."""

    return torch.sort(topk_indices, dim=-1).values


def _assert_topk_matches_quantized_scores(
    actual_indices: torch.Tensor,
    scores: torch.Tensor,
) -> dict[str, float | int]:
    """Validate top-k exactly away from ties and by cutoff at numeric ties."""

    assert actual_indices.shape[:2] == scores.shape[:2]
    topk = actual_indices.shape[-1]
    seq_k = scores.shape[-1]
    assert topk <= seq_k
    actual_valid = actual_indices >= 0
    valid_count = torch.isfinite(scores).sum(dim=-1)
    expected_count = valid_count.clamp(max=topk)
    torch.testing.assert_close(actual_valid.sum(dim=-1), expected_count, atol=0, rtol=0)
    assert torch.all((actual_indices < seq_k) | ~actual_valid)

    canonical_actual = _canonical_topk_set(actual_indices)
    duplicate = (canonical_actual[..., 1:] == canonical_actual[..., :-1]) & (
        canonical_actual[..., 1:] >= 0
    )
    assert not torch.any(duplicate), "vendor top-k returned duplicate valid indices"

    expected_indices = _torch_topk_from_scores(scores, topk)
    exact_rows = torch.all(
        canonical_actual == _canonical_topk_set(expected_indices), dim=-1
    )

    probe_k = min(topk + 1, seq_k)
    probe_values = torch.topk(scores, k=probe_k, dim=-1).values
    kth_slot = (expected_count - 1).clamp(min=0).unsqueeze(-1)
    cutoff = torch.gather(probe_values[..., :topk], -1, kth_slot).squeeze(-1)
    tolerance = _TOPK_SCORE_ATOL + _TOPK_SCORE_RTOL * cutoff.abs()

    safe_indices = actual_indices.clamp(min=0).long()
    selected_scores = torch.gather(scores, dim=-1, index=safe_indices)
    selected_scores = torch.where(
        actual_valid, selected_scores, torch.full_like(selected_scores, torch.inf)
    )
    minimum_selected = selected_scores.amin(dim=-1)
    nonempty = expected_count > 0
    selected_counts = torch.zeros_like(scores, dtype=torch.int32)
    selected_counts.scatter_add_(-1, safe_indices, actual_valid.to(torch.int32))
    selected_mask = selected_counts > 0
    unselected_scores = torch.where(
        torch.isfinite(scores) & ~selected_mask,
        scores,
        torch.full_like(scores, -torch.inf),
    )
    maximum_unselected = unselected_scores.amax(dim=-1)
    has_unselected = valid_count > expected_count
    pairwise_tolerance = _TOPK_SCORE_ATOL + _TOPK_SCORE_RTOL * torch.maximum(
        minimum_selected.abs(), maximum_unselected.abs()
    )
    shortfall = torch.where(
        nonempty & has_unselected,
        (maximum_unselected - minimum_selected).clamp_min(0),
        torch.zeros_like(cutoff),
    )
    assert torch.all(
        ~(nonempty & has_unselected)
        | (minimum_selected >= maximum_unselected - pairwise_tolerance)
    ), (
        "top-k omitted a score above the selected boundary: "
        f"max_shortfall={float(shortfall.max().item())}"
    )

    strict_rows = valid_count <= topk
    if probe_k > topk:
        next_score = probe_values[..., topk]
        margin = cutoff - next_score
        strict_rows = strict_rows | (margin > tolerance)
    assert torch.all(~strict_rows | exact_rows), (
        "top-k index multiset differs despite a numerically separated cutoff"
    )
    ambiguous_rows = (~strict_rows & nonempty).sum()
    return {
        "exact_rows": int(exact_rows.sum().item()),
        "ambiguous_rows": int(ambiguous_rows.item()),
        "max_score_shortfall": float(shortfall.max().item()),
    }


def test_topk_score_proof_rejects_missing_strict_winner_but_allows_boundary_ties():
    scores = torch.tensor([[[100.0, 1.0, 1.0]]])
    with pytest.raises(AssertionError, match="omitted a score"):
        _assert_topk_matches_quantized_scores(
            torch.tensor([[[1, 2]]], dtype=torch.int32), scores
        )

    proof = _assert_topk_matches_quantized_scores(
        torch.tensor([[[0, 2]]], dtype=torch.int32), scores
    )
    assert proof["ambiguous_rows"] == 1
    assert proof["max_score_shortfall"] == 0.0


def _max_param_grad_abs(a: dict, b: dict) -> float:
    a_keys = set(a["param_grads"])
    b_keys = set(b["param_grads"])
    assert a_keys == b_keys, (
        f"gradient key mismatch: only_a={sorted(a_keys - b_keys)}, "
        f"only_b={sorted(b_keys - a_keys)}"
    )
    if not a_keys:
        return 0.0
    return max(
        _max_abs(a["param_grads"][name], b["param_grads"][name]) for name in a_keys
    )


def _main_param_grad_similarity(
    actual: dict, expected: dict
) -> dict[str, dict[str, float]]:
    """Compare every non-indexer parameter separately so large tensors cannot hide drift."""

    actual_names = {
        name for name in actual["param_grads"] if not name.startswith("source.indexer.")
    }
    expected_names = {
        name
        for name in expected["param_grads"]
        if not name.startswith("source.indexer.")
    }
    assert actual_names == expected_names, (
        "main-gradient key mismatch: "
        f"only_actual={sorted(actual_names - expected_names)}, "
        f"only_expected={sorted(expected_names - actual_names)}"
    )
    assert actual_names
    return {
        name: _tensor_similarity_metrics(
            actual["param_grads"][name], expected["param_grads"][name]
        )
        for name in sorted(actual_names)
    }


def _assert_main_grad_similarity(
    comparisons: dict[str, dict[str, float]],
) -> dict[str, float]:
    extrema = _similarity_extrema(comparisons)
    assert extrema["min_cosine"] >= _MIN_MAIN_GRAD_COSINE, comparisons
    assert extrema["max_rms_relative"] <= _MAX_MAIN_GRAD_RMS_REL, comparisons
    assert extrema["min_norm_ratio"] >= _MIN_MAIN_GRAD_NORM_RATIO, comparisons
    assert extrema["max_norm_ratio"] <= _MAX_MAIN_GRAD_NORM_RATIO, comparisons
    return extrema


def _assert_meaningful_indexer_grads(result: dict) -> float:
    expected = {
        "source.indexer.wq_b.weight",
        "source.indexer.wk.weight",
        "source.indexer.k_norm.weight",
        "source.indexer.k_norm.bias",
        "source.indexer.weights_proj.weight",
    }
    grads = {
        name: grad
        for name, grad in result["param_grads"].items()
        if name.startswith("source.indexer.")
    }
    assert set(grads) == expected
    assert all(torch.isfinite(grad).all() for grad in grads.values())
    assert all(torch.count_nonzero(grad).item() > 0 for grad in grads.values())
    grad_max_abs = {
        name: float(grad.abs().max().item()) for name, grad in grads.items()
    }
    assert all(value > _MIN_INDEXER_GRAD_MAX_ABS for value in grad_max_abs.values()), (
        grad_max_abs
    )
    return max(grad_max_abs.values())


def _indexer_grad_similarity(
    actual: dict, expected: dict
) -> dict[str, tuple[float, float, float]]:
    names = sorted(
        name for name in expected["param_grads"] if name.startswith("source.indexer.")
    )
    assert names
    assert set(names) == {
        name for name in actual["param_grads"] if name.startswith("source.indexer.")
    }
    similarities: dict[str, tuple[float, float, float]] = {}
    for name in names:
        actual_grad = actual["param_grads"][name].float().reshape(-1)
        expected_grad = expected["param_grads"][name].float().reshape(-1)
        actual_norm = torch.linalg.vector_norm(actual_grad)
        expected_norm = torch.linalg.vector_norm(expected_grad)
        cosine = torch.dot(actual_grad, expected_grad) / (actual_norm * expected_norm)
        rms_diff = torch.sqrt(torch.mean((actual_grad - expected_grad).square()))
        rms_expected = torch.sqrt(torch.mean(expected_grad.square()))
        rms_relative = rms_diff / rms_expected
        norm_ratio = actual_norm / expected_norm
        similarities[name] = (
            float(cosine.item()),
            float(rms_relative.item()),
            float(norm_ratio.item()),
        )
    return similarities


@pytest.mark.parametrize(
    "sparse_loss", [True, False], ids=["sparse-loss", "dense-loss"]
)
def test_glm5_dsa_run_to_run_accept_with_proof(sparse_loss: bool, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for GLM5 DSA accept-with-proof smoke.")

    from megatron.lite.primitive.modules.attention import build_rope_cache
    from megatron.lite.primitive.modules.attention.dsa import (
        DSAIndexerLossAutoScaler,
    )

    device = torch.device("cuda", int(torch.cuda.current_device()))
    # The runtime loss scaler is process-global.  Pin this standalone oracle to
    # unit scale and let monkeypatch restore any suite-level prior value.
    monkeypatch.setattr(
        DSAIndexerLossAutoScaler,
        "main_loss_backward_scale",
        torch.ones((), device=device),
    )
    torch.manual_seed(20260626)
    fused = _make_dsa_pair(sparse_loss=sparse_loss).to(
        device=device, dtype=torch.bfloat16
    )
    unfused = copy.deepcopy(fused).to(device=device, dtype=torch.bfloat16)

    batch, seq, hidden = 1, _SEQUENCE_LENGTH, 128
    x = torch.randn(batch, seq, hidden, device=device, dtype=torch.bfloat16)
    cos, sin = build_rope_cache(
        dim=64,
        max_position_embeddings=seq,
        rope_theta=1_000_000.0,
        device=device,
    )
    position_ids = torch.arange(seq, device=device, dtype=torch.long).unsqueeze(0)

    fused_a = _run_once(fused, x, cos, sin, position_ids, fused_training=True)
    fused_b = _run_once(fused, x, cos, sin, position_ids, fused_training=True)
    unfused_a = _run_once_torch_unfused(unfused, x, cos, sin, position_ids)
    unfused_b = _run_once_torch_unfused(unfused, x, cos, sin, position_ids)
    matched_unfused_a = _run_once_torch_unfused(
        unfused,
        x,
        cos,
        sin,
        position_ids,
        forced_source_topk=fused_a["topk_indices"],
    )
    matched_unfused_b = _run_once_torch_unfused(
        unfused,
        x,
        cos,
        sin,
        position_ids,
        forced_source_topk=fused_b["topk_indices"],
    )

    fused_r2r_output_similarity = {
        "run_a_vs_run_b": _tensor_similarity_metrics(fused_a["out"], fused_b["out"])
    }
    fused_r2r_output_extrema = _similarity_extrema(fused_r2r_output_similarity)
    fused_r2r_out = fused_r2r_output_extrema["max_abs"]
    fused_a_loss = float(fused_a["loss"].item())
    fused_b_loss = float(fused_b["loss"].item())
    fused_r2r_loss = abs(fused_a_loss - fused_b_loss)
    fused_r2r_loss_symmetric_rel = _symmetric_relative_difference(
        fused_a_loss, fused_b_loss
    )
    fused_r2r_x_grad = _max_abs(fused_a["x_grad"], fused_b["x_grad"])
    fused_r2r_param_grad = _max_param_grad_abs(fused_a, fused_b)
    fused_r2r_x_grad_similarity = {
        "input": _tensor_similarity_metrics(fused_a["x_grad"], fused_b["x_grad"])
    }
    fused_r2r_param_grad_similarity = _main_param_grad_similarity(fused_a, fused_b)
    fused_r2r_x_grad_extrema = _similarity_extrema(fused_r2r_x_grad_similarity)
    fused_r2r_param_grad_extrema = _similarity_extrema(fused_r2r_param_grad_similarity)
    unfused_r2r_out = _max_abs(unfused_a["out"], unfused_b["out"])
    unfused_r2r_loss = abs(
        float(unfused_a["loss"].item()) - float(unfused_b["loss"].item())
    )
    unfused_r2r_x_grad = _max_abs(unfused_a["x_grad"], unfused_b["x_grad"])
    unfused_r2r_param_grad = _max_param_grad_abs(unfused_a, unfused_b)
    fused_vs_unfused_output_similarity = {
        "run_a": _tensor_similarity_metrics(fused_a["out"], matched_unfused_a["out"]),
        "run_b": _tensor_similarity_metrics(fused_b["out"], matched_unfused_b["out"]),
    }
    fused_vs_unfused_output_extrema = _similarity_extrema(
        fused_vs_unfused_output_similarity
    )
    fused_vs_unfused_out = fused_vs_unfused_output_extrema["max_abs"]
    fused_vs_unfused_x_grad = max(
        _max_abs(fused_a["x_grad"], matched_unfused_a["x_grad"]),
        _max_abs(fused_b["x_grad"], matched_unfused_b["x_grad"]),
    )
    fused_vs_unfused_param_grad = max(
        _max_param_grad_abs(fused_a, matched_unfused_a),
        _max_param_grad_abs(fused_b, matched_unfused_b),
    )
    fused_vs_unfused_x_grad_similarity = {
        "run_a": _tensor_similarity_metrics(
            fused_a["x_grad"], matched_unfused_a["x_grad"]
        ),
        "run_b": _tensor_similarity_metrics(
            fused_b["x_grad"], matched_unfused_b["x_grad"]
        ),
    }
    fused_vs_unfused_param_grad_similarity = {
        **{
            f"run_a:{name}": value
            for name, value in _main_param_grad_similarity(
                fused_a, matched_unfused_a
            ).items()
        },
        **{
            f"run_b:{name}": value
            for name, value in _main_param_grad_similarity(
                fused_b, matched_unfused_b
            ).items()
        },
    }
    fused_vs_unfused_x_grad_extrema = _similarity_extrema(
        fused_vs_unfused_x_grad_similarity
    )
    fused_vs_unfused_param_grad_extrema = _similarity_extrema(
        fused_vs_unfused_param_grad_similarity
    )
    loss_diff = max(
        abs(fused_a_loss - float(matched_unfused_a["loss"].item())),
        abs(fused_b_loss - float(matched_unfused_b["loss"].item())),
    )
    loss_symmetric_rel = max(
        _symmetric_relative_difference(
            fused_a_loss, float(matched_unfused_a["loss"].item())
        ),
        _symmetric_relative_difference(
            fused_b_loss, float(matched_unfused_b["loss"].item())
        ),
    )
    unfused_indexer_loss_diff = abs(
        float(unfused_a["indexer_loss"].item())
        - float(unfused_b["indexer_loss"].item())
    )
    fused_indexer_grad_max_abs = _assert_meaningful_indexer_grads(fused_a)
    _assert_meaningful_indexer_grads(fused_b)
    unfused_indexer_grad_max_abs = _assert_meaningful_indexer_grads(matched_unfused_a)
    _assert_meaningful_indexer_grads(matched_unfused_b)
    _assert_meaningful_indexer_grads(unfused_b)
    indexer_grad_similarity = {
        **{
            f"run_a:{name}": value
            for name, value in _indexer_grad_similarity(
                fused_a, matched_unfused_a
            ).items()
        },
        **{
            f"run_b:{name}": value
            for name, value in _indexer_grad_similarity(
                fused_b, matched_unfused_b
            ).items()
        },
    }
    min_indexer_grad_cosine = min(
        value[0] for value in indexer_grad_similarity.values()
    )
    max_indexer_grad_rms_relative = max(
        value[1] for value in indexer_grad_similarity.values()
    )
    min_indexer_grad_norm_ratio = min(
        value[2] for value in indexer_grad_similarity.values()
    )
    max_indexer_grad_norm_ratio = max(
        value[2] for value in indexer_grad_similarity.values()
    )
    fused_indexer_loss_r2r = abs(
        float(fused_a["indexer_loss"].item()) - float(fused_b["indexer_loss"].item())
    )
    fused_vs_unfused_indexer_loss = max(
        abs(
            float(fused_a["indexer_loss"].item())
            - float(matched_unfused_a["indexer_loss"].item())
        ),
        abs(
            float(fused_b["indexer_loss"].item())
            - float(matched_unfused_b["indexer_loss"].item())
        ),
    )

    assert torch.isfinite(fused_a["loss"])
    assert torch.isfinite(fused_a["indexer_loss"])
    assert float(fused_a["indexer_loss"].item()) > 0.0
    assert torch.isfinite(fused_b["loss"])
    assert torch.isfinite(fused_b["indexer_loss"])
    assert float(fused_b["indexer_loss"].item()) > 0.0
    assert torch.isfinite(unfused_a["loss"])
    assert torch.isfinite(unfused_a["indexer_loss"])
    assert float(unfused_a["indexer_loss"].item()) > 0.0
    fused_a_topk_set = _canonical_topk_set(fused_a["topk_indices"])
    fused_b_topk_set = _canonical_topk_set(fused_b["topk_indices"])
    unfused_a_topk_set = _canonical_topk_set(unfused_a["topk_indices"])
    unfused_b_topk_set = _canonical_topk_set(unfused_b["topk_indices"])
    fused_topk_set_r2r = torch.equal(fused_a_topk_set, fused_b_topk_set)
    unfused_topk_set_r2r = torch.equal(unfused_a_topk_set, unfused_b_topk_set)
    fused_vs_unfused_topk_set = torch.equal(fused_a_topk_set, unfused_a_topk_set)
    topk_score_proof_a = _assert_topk_matches_quantized_scores(
        fused_a["topk_indices"], unfused_a["indexer_scores"]
    )
    topk_score_proof_b = _assert_topk_matches_quantized_scores(
        fused_b["topk_indices"], unfused_a["indexer_scores"]
    )
    topk_ambiguous_rows = max(
        topk_score_proof_a["ambiguous_rows"],
        topk_score_proof_b["ambiguous_rows"],
    )
    topk_max_score_shortfall = max(
        topk_score_proof_a["max_score_shortfall"],
        topk_score_proof_b["max_score_shortfall"],
    )

    dense_topk_loss_diff = 0.0
    dense_topk_indexer_grad_diff = 0.0
    dense_topk_changed = False
    if not sparse_loss:
        from megatron.lite.primitive.kernels import dsa_kernels

        original_topk = dsa_kernels._indexer_topk_bshd

        def lowest_valid_topk(q_bshd, k_bsd, w_bsh, topk, ratio=4):
            _indices, _length, scores = original_topk(q_bshd, k_bsd, w_bsh, topk, ratio)
            del _indices, _length
            low_rank_scores = torch.where(
                torch.isfinite(scores), scores, torch.full_like(scores, torch.inf)
            )
            values, indices = torch.topk(low_rank_scores, k=topk, dim=-1, largest=False)
            indices = torch.where(
                torch.isfinite(values), indices, torch.full_like(indices, -1)
            ).int()
            lengths = (indices >= 0).sum(dim=-1).int()
            return indices, lengths, scores

        with monkeypatch.context() as dense_topk_patch:
            dense_topk_patch.setattr(
                dsa_kernels, "_indexer_topk_bshd", lowest_valid_topk
            )
            alternate_topk = _run_once(
                fused, x, cos, sin, position_ids, fused_training=True
            )

        dense_topk_changed = not torch.equal(
            _canonical_topk_set(alternate_topk["topk_indices"]), fused_a_topk_set
        )
        assert dense_topk_changed
        dense_topk_loss_diff = abs(
            float(alternate_topk["indexer_loss"].item())
            - float(fused_a["indexer_loss"].item())
        )
        torch.testing.assert_close(
            alternate_topk["indexer_loss"],
            fused_a["indexer_loss"],
            atol=_INDEXER_LOSS_ATOL,
            rtol=_INDEXER_LOSS_RTOL,
        )
        indexer_names = [
            name
            for name in fused_a["param_grads"]
            if name.startswith("source.indexer.")
        ]
        dense_topk_indexer_grad_diff = max(
            _max_abs(
                alternate_topk["param_grads"][name],
                fused_a["param_grads"][name],
            )
            for name in indexer_names
        )
        assert dense_topk_indexer_grad_diff <= 1.0e-6
    print(
        "GLM5_DSA_ACCEPTANCE_DIAGNOSTICS "
        f"sparse_loss={sparse_loss} "
        f"fused_topk_set_r2r={fused_topk_set_r2r} "
        f"unfused_topk_set_r2r={unfused_topk_set_r2r} "
        f"fused_vs_unfused_topk_set={fused_vs_unfused_topk_set} "
        f"loss_diff={loss_diff:.6e} "
        f"loss_symmetric_rel={loss_symmetric_rel:.6e} "
        f"fused_r2r_loss_diff={fused_r2r_loss:.6e} "
        f"fused_r2r_loss_symmetric_rel={fused_r2r_loss_symmetric_rel:.6e} "
        f"fused_r2r_out_max_abs={fused_r2r_out:.6e} "
        f"fused_r2r_out_min_cosine={fused_r2r_output_extrema['min_cosine']:.6e} "
        f"fused_r2r_out_max_rms_relative={fused_r2r_output_extrema['max_rms_relative']:.6e} "
        f"fused_r2r_out_min_norm_ratio={fused_r2r_output_extrema['min_norm_ratio']:.6e} "
        f"fused_r2r_out_max_norm_ratio={fused_r2r_output_extrema['max_norm_ratio']:.6e} "
        f"fused_r2r_x_grad_max_abs={fused_r2r_x_grad:.6e} "
        f"fused_r2r_param_grad_max_abs={fused_r2r_param_grad:.6e} "
        f"fused_r2r_x_grad_similarity={fused_r2r_x_grad_extrema} "
        f"fused_r2r_param_grad_similarity={fused_r2r_param_grad_extrema} "
        f"fused_vs_unfused_out_max_abs={fused_vs_unfused_out:.6e} "
        f"fused_vs_unfused_out_min_cosine={fused_vs_unfused_output_extrema['min_cosine']:.6e} "
        f"fused_vs_unfused_out_max_rms_relative={fused_vs_unfused_output_extrema['max_rms_relative']:.6e} "
        f"fused_vs_unfused_out_min_norm_ratio={fused_vs_unfused_output_extrema['min_norm_ratio']:.6e} "
        f"fused_vs_unfused_out_max_norm_ratio={fused_vs_unfused_output_extrema['max_norm_ratio']:.6e} "
        f"fused_vs_unfused_x_grad_max_abs={fused_vs_unfused_x_grad:.6e} "
        f"fused_vs_unfused_param_grad_max_abs={fused_vs_unfused_param_grad:.6e}"
        f" fused_vs_unfused_x_grad_similarity={fused_vs_unfused_x_grad_extrema}"
        f" fused_vs_unfused_param_grad_similarity={fused_vs_unfused_param_grad_extrema}"
        f" fused_indexer_loss_r2r={fused_indexer_loss_r2r:.6e}"
        f" fused_vs_unfused_indexer_loss={fused_vs_unfused_indexer_loss:.6e}"
        f" min_indexer_grad_cosine={min_indexer_grad_cosine:.6e}"
        f" max_indexer_grad_rms_relative={max_indexer_grad_rms_relative:.6e}"
        f" min_indexer_grad_norm_ratio={min_indexer_grad_norm_ratio:.6e}"
        f" max_indexer_grad_norm_ratio={max_indexer_grad_norm_ratio:.6e}"
        f" topk_exact_rows_a={topk_score_proof_a['exact_rows']}"
        f" topk_exact_rows_b={topk_score_proof_b['exact_rows']}"
        f" topk_ambiguous_rows={topk_ambiguous_rows}"
        f" topk_max_score_shortfall={topk_max_score_shortfall:.6e}"
        f" dense_topk_changed={dense_topk_changed}"
        f" dense_topk_loss_diff={dense_topk_loss_diff:.6e}"
        f" dense_topk_indexer_grad_diff={dense_topk_indexer_grad_diff:.6e}"
    )
    assert unfused_topk_set_r2r
    if fused_topk_set_r2r:
        assert fused_r2r_loss <= _MAIN_LOSS_MAX_ABS_DIFF
        assert fused_r2r_loss_symmetric_rel <= _MAX_MAIN_LOSS_SYMMETRIC_REL
        _assert_main_output_similarity(
            fused_r2r_output_similarity,
            max_abs=_GLM_FUSED_R2R_OUTPUT_MAX_ABS,
        )
        _assert_main_grad_similarity(fused_r2r_x_grad_similarity)
        _assert_main_grad_similarity(fused_r2r_param_grad_similarity)
        assert fused_r2r_x_grad <= _GLM_X_GRAD_MAX_ABS
        assert fused_r2r_param_grad <= _GLM_PARAM_GRAD_MAX_ABS
        assert fused_indexer_loss_r2r <= _INDEXER_LOSS_ATOL
    else:
        # Radix top-k is allowed to choose different members at a quantized
        # cutoff tie. Each run is checked against its own forced-top-k oracle.
        assert topk_ambiguous_rows > 0
        if not sparse_loss:
            # Canonical dense KL is independent of the selected sparse keys.
            assert fused_indexer_loss_r2r <= _INDEXER_LOSS_ATOL
    torch.testing.assert_close(
        fused_a["indexer_loss"],
        matched_unfused_a["indexer_loss"],
        atol=_INDEXER_LOSS_ATOL,
        rtol=_INDEXER_LOSS_RTOL,
    )
    torch.testing.assert_close(
        fused_b["indexer_loss"],
        matched_unfused_b["indexer_loss"],
        atol=_INDEXER_LOSS_ATOL,
        rtol=_INDEXER_LOSS_RTOL,
    )
    assert min_indexer_grad_cosine >= _MIN_INDEXER_GRAD_COSINE, indexer_grad_similarity
    assert max_indexer_grad_rms_relative <= _MAX_INDEXER_GRAD_RMS_REL, (
        indexer_grad_similarity
    )
    assert min_indexer_grad_norm_ratio >= _MIN_INDEXER_GRAD_NORM_RATIO, (
        indexer_grad_similarity
    )
    assert max_indexer_grad_norm_ratio <= _MAX_INDEXER_GRAD_NORM_RATIO, (
        indexer_grad_similarity
    )
    valid_topk = (fused_a["topk_indices"] >= 0).sum(dim=-1)
    assert _INDEXER_TOPK < seq
    assert fused_a["topk_indices"].shape[-1] == _INDEXER_TOPK
    assert int(valid_topk[0, -1].item()) == _INDEXER_TOPK
    assert int(valid_topk.max().item()) < seq
    assert unfused_r2r_loss == 0.0
    assert unfused_r2r_out == 0.0
    assert unfused_r2r_x_grad == 0.0
    assert unfused_r2r_param_grad == 0.0
    assert unfused_indexer_loss_diff == 0.0
    assert loss_diff <= _MAIN_LOSS_MAX_ABS_DIFF
    assert loss_symmetric_rel <= _MAX_MAIN_LOSS_SYMMETRIC_REL
    _assert_main_output_similarity(
        fused_vs_unfused_output_similarity,
        max_abs=_GLM_FUSED_VS_REFERENCE_OUTPUT_MAX_ABS,
    )
    _assert_main_grad_similarity(fused_vs_unfused_x_grad_similarity)
    _assert_main_grad_similarity(fused_vs_unfused_param_grad_similarity)
    assert fused_vs_unfused_x_grad <= _GLM_X_GRAD_MAX_ABS
    assert fused_vs_unfused_param_grad <= _GLM_PARAM_GRAD_MAX_ABS

    print(
        "NON_SKIP_GLM5_DSA_RUN_TO_RUN_ACCEPT_WITH_PROOF "
        f"fused_loss={float(fused_a['loss'].item()):.6e} "
        f"unfused_loss={float(unfused_a['loss'].item()):.6e} "
        f"unfused_indexer_loss={float(unfused_a['indexer_loss'].item()):.6e} "
        f"fused_indexer_loss={float(fused_a['indexer_loss'].item()):.6e} "
        f"index_heads=32 sparse_loss={sparse_loss} "
        f"topk_set_exact={fused_vs_unfused_topk_set} "
        "topk_score_proof_pass=True "
        f"topk_ambiguous_rows={topk_ambiguous_rows} "
        "matched_topk_reference=True "
        f"dense_topk_independent={not sparse_loss and dense_topk_changed} "
        f"indexer_topk={_INDEXER_TOPK} seq={seq} "
        f"loss_diff={loss_diff:.6e} "
        f"loss_symmetric_rel={loss_symmetric_rel:.6e} "
        f"fused_r2r_loss_diff={fused_r2r_loss:.6e} "
        f"fused_r2r_loss_symmetric_rel={fused_r2r_loss_symmetric_rel:.6e} "
        f"fused_r2r_out_max_abs={fused_r2r_out:.6e} "
        f"fused_r2r_out_min_cosine={fused_r2r_output_extrema['min_cosine']:.6e} "
        f"fused_r2r_out_max_rms_relative={fused_r2r_output_extrema['max_rms_relative']:.6e} "
        f"fused_r2r_out_min_norm_ratio={fused_r2r_output_extrema['min_norm_ratio']:.6e} "
        f"fused_r2r_out_max_norm_ratio={fused_r2r_output_extrema['max_norm_ratio']:.6e} "
        f"fused_r2r_x_grad_max_abs={fused_r2r_x_grad:.6e} "
        f"fused_r2r_param_grad_max_abs={fused_r2r_param_grad:.6e} "
        f"fused_r2r_x_grad_similarity={fused_r2r_x_grad_extrema} "
        f"fused_r2r_param_grad_similarity={fused_r2r_param_grad_extrema} "
        f"unfused_r2r_out_max_abs={unfused_r2r_out:.6e} "
        f"unfused_r2r_loss_diff={unfused_r2r_loss:.6e} "
        f"unfused_r2r_x_grad_max_abs={unfused_r2r_x_grad:.6e} "
        f"unfused_r2r_param_grad_max_abs={unfused_r2r_param_grad:.6e} "
        f"fused_vs_unfused_out_max_abs={fused_vs_unfused_out:.6e} "
        f"fused_vs_unfused_out_min_cosine={fused_vs_unfused_output_extrema['min_cosine']:.6e} "
        f"fused_vs_unfused_out_max_rms_relative={fused_vs_unfused_output_extrema['max_rms_relative']:.6e} "
        f"fused_vs_unfused_out_min_norm_ratio={fused_vs_unfused_output_extrema['min_norm_ratio']:.6e} "
        f"fused_vs_unfused_out_max_norm_ratio={fused_vs_unfused_output_extrema['max_norm_ratio']:.6e} "
        f"fused_vs_unfused_x_grad_max_abs={fused_vs_unfused_x_grad:.6e} "
        f"fused_vs_unfused_param_grad_max_abs={fused_vs_unfused_param_grad:.6e} "
        f"fused_vs_unfused_x_grad_similarity={fused_vs_unfused_x_grad_extrema} "
        f"fused_vs_unfused_param_grad_similarity={fused_vs_unfused_param_grad_extrema} "
        f"fused_indexer_grad_max_abs={fused_indexer_grad_max_abs:.6e} "
        f"unfused_indexer_grad_max_abs={unfused_indexer_grad_max_abs:.6e}"
        f" min_indexer_grad_cosine={min_indexer_grad_cosine:.6e}"
        f" max_indexer_grad_rms_relative={max_indexer_grad_rms_relative:.6e}"
        f" min_indexer_grad_norm_ratio={min_indexer_grad_norm_ratio:.6e}"
        f" max_indexer_grad_norm_ratio={max_indexer_grad_norm_ratio:.6e}"
        f" dense_topk_loss_diff={dense_topk_loss_diff:.6e}"
        f" dense_topk_indexer_grad_diff={dense_topk_indexer_grad_diff:.6e}"
    )


def test_dsv4_fused_dsa_legacy_two_output_api_real_gpu():
    """Prove the legacy DSv4 training API against its decomposed inference path."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the fused DSA API regression smoke.")
    pytest.importorskip(
        "cudnn", reason="Fused DSA API smoke needs the cuDNN DSA stack."
    )
    from megatron.lite.primitive.kernels import dsa_kernels

    device = torch.device("cuda", int(torch.cuda.current_device()))
    seq, batch, heads = 512, 1, 64
    ratio = 4
    n_comp = seq // ratio
    kv_offset = seq
    window_topk = 64
    query_dim, value_dim = 576, 512
    index_heads, index_dim = 32, 128
    requested_topk = n_comp + 64

    query_positions = torch.arange(seq, device=device).view(seq, 1)
    window_offsets = torch.arange(window_topk, device=device).view(1, window_topk)
    window = query_positions - (window_topk - 1 - window_offsets)
    window = torch.where(window >= 0, window, torch.full_like(window, -1))
    window = window.unsqueeze(0).expand(batch, -1, -1).to(torch.int32)

    def make_args():
        torch.manual_seed(20260627)

        def leaf(*shape, dtype=torch.bfloat16):
            return torch.randn(*shape, device=device, dtype=dtype).requires_grad_(True)

        return (
            leaf(seq, batch, heads, query_dim),
            leaf(seq + n_comp, batch, query_dim),
            leaf(heads, dtype=torch.float32),
            window.clone(),
            leaf(seq, batch, index_heads, index_dim),
            leaf(n_comp, batch, index_dim),
            leaf(seq, batch, index_heads),
            requested_topk,
            ratio,
            query_dim**-0.5,
            index_dim**-0.5,
            0.0,
            False,
            kv_offset,
            False,
            value_dim,
        )

    fused_results = []
    for name, call in (
        ("direct", dsa_kernels.FusedIndexerSparseAttnFunc.apply),
        ("public", dsa_kernels.fused_indexer_sparse_attn),
    ):
        args = make_args()
        result = call(*args)
        assert isinstance(result, tuple) and len(result) == 2
        output, indexer_loss = result
        assert output.shape == (seq, batch, heads * value_dim)
        assert indexer_loss.ndim == 0
        assert float(indexer_loss.item()) == 0.0
        objective = output.float().square().mean() + indexer_loss
        objective.backward()
        main_inputs = {"query": args[0], "kv_full": args[1], "attn_sink": args[2]}
        indexer_inputs = {
            "q_indexer": args[4],
            "k_indexer": args[5],
            "weights": args[6],
        }
        assert all(
            tensor.grad is not None
            for tensor in (*main_inputs.values(), *indexer_inputs.values())
        )
        assert all(
            torch.isfinite(tensor.grad).all()
            for tensor in (*main_inputs.values(), *indexer_inputs.values())
        )
        assert all(
            torch.count_nonzero(tensor.grad).item() == 0
            for tensor in indexer_inputs.values()
        )
        fused_results.append(
            {
                "output": output.detach().float(),
                "objective": objective.detach().float(),
                "main_grads": {
                    key: tensor.grad.detach().float()
                    for key, tensor in main_inputs.items()
                },
            }
        )
        print(
            "DSV4_FUSED_DSA_LEGACY_API "
            f"path={name} loss={float(objective.detach().item()):.8e}"
        )

    # Reproduce the exact DSv4/CSA inference decomposition: indexer_topk ->
    # global/compact indices -> dsa_sparse_attn.  This is independent of the
    # legacy fused autograd wrapper and covers the real compressed-KV layout.
    args = make_args()
    topk_indices, topk_length = dsa_kernels.indexer_topk(
        args[4],
        args[5],
        args[6],
        requested_topk,
        ratio,
        indexer_softmax_scale=args[10],
    )
    assert topk_indices.shape == (batch, seq, requested_topk)
    assert topk_length.shape == (batch, seq)
    assert torch.all(topk_indices[..., n_comp:] == -1)
    independent_indexer_scores = _torch_indexer_scores(
        args[4],
        args[5],
        args[6],
        ratio=ratio,
        indexer_softmax_scale=args[10],
    )
    padded_topk_score_proof = _assert_topk_matches_quantized_scores(
        topk_indices[..., :n_comp], independent_indexer_scores
    )
    selection_topk = n_comp // 2
    selection_indices, selection_length = dsa_kernels.indexer_topk(
        args[4],
        args[5],
        args[6],
        selection_topk,
        ratio,
        indexer_softmax_scale=args[10],
    )
    assert selection_indices.shape == (batch, seq, selection_topk)
    assert selection_length.shape == (batch, seq)
    assert int(selection_length[0, -1].item()) == selection_topk
    selection_topk_score_proof = _assert_topk_matches_quantized_scores(
        selection_indices, independent_indexer_scores
    )
    compressed_global = torch.where(
        topk_indices >= 0, topk_indices + kv_offset, topk_indices
    ).to(torch.int32)
    assert torch.all((compressed_global < 0) | (compressed_global >= kv_offset))
    flat_indices, flat_length = dsa_kernels.build_flat_topk_idxs(
        args[3],
        compressed_global,
        batch_size=batch,
        seqlen_kv=args[1].shape[0],
        compact=True,
    )
    assert flat_length is not None
    decomposed_output = dsa_kernels.dsa_sparse_attn(
        args[0],
        args[1],
        args[2],
        flat_indices,
        args[9],
        topk_length=flat_length,
        value_dim=value_dim,
    )
    decomposed_objective = decomposed_output.float().square().mean()
    decomposed_objective.backward()
    decomposed_main_grads = {
        "query": args[0].grad.detach().float(),
        "kv_full": args[1].grad.detach().float(),
        "attn_sink": args[2].grad.detach().float(),
    }
    assert all(torch.isfinite(grad).all() for grad in decomposed_main_grads.values())
    assert all(args[index].grad is None for index in (4, 5, 6))

    direct_public_output_similarity = {
        "direct_vs_public": _tensor_similarity_metrics(
            fused_results[0]["output"], fused_results[1]["output"]
        )
    }
    direct_public_output_extrema = _similarity_extrema(direct_public_output_similarity)
    fused_decomposed_output_similarity = {
        "public_vs_decomposed": _tensor_similarity_metrics(
            fused_results[1]["output"], decomposed_output.detach().float()
        )
    }
    fused_decomposed_output_extrema = _similarity_extrema(
        fused_decomposed_output_similarity
    )
    public_objective = float(fused_results[1]["objective"].item())
    decomposed_objective_value = float(decomposed_objective.detach().item())
    fused_decomposed_loss_diff = abs(public_objective - decomposed_objective_value)
    fused_decomposed_loss_symmetric_rel = _symmetric_relative_difference(
        public_objective, decomposed_objective_value
    )
    direct_public_grad_similarity = {}
    for name in decomposed_main_grads:
        direct_public_grad_similarity[name] = _tensor_similarity_metrics(
            fused_results[0]["main_grads"][name],
            fused_results[1]["main_grads"][name],
        )
    decomposed_grad_similarity = {
        name: _tensor_similarity_metrics(
            fused_results[1]["main_grads"][name],
            decomposed_grad,
        )
        for name, decomposed_grad in decomposed_main_grads.items()
    }
    direct_public_grad_extrema = _similarity_extrema(direct_public_grad_similarity)
    decomposed_grad_extrema = _similarity_extrema(decomposed_grad_similarity)
    print(
        "DSV4_FUSED_DSA_DECOMPOSED_DIAGNOSTICS "
        f"loss_diff={fused_decomposed_loss_diff:.6e} "
        f"loss_symmetric_rel={fused_decomposed_loss_symmetric_rel:.6e} "
        f"direct_public_output={direct_public_output_extrema} "
        f"fused_decomposed_output={fused_decomposed_output_extrema} "
        f"direct_public_grads={direct_public_grad_extrema} "
        f"fused_decomposed_grads={decomposed_grad_extrema}"
    )

    _assert_main_output_similarity(
        direct_public_output_similarity,
        max_abs=_DIRECT_PUBLIC_OUTPUT_ATOL,
    )
    torch.testing.assert_close(
        fused_results[0]["output"],
        fused_results[1]["output"],
        rtol=0.0,
        atol=_DIRECT_PUBLIC_OUTPUT_ATOL,
    )
    torch.testing.assert_close(
        fused_results[0]["objective"],
        fused_results[1]["objective"],
        rtol=0.0,
        atol=_DIRECT_PUBLIC_LOSS_ATOL,
    )
    _assert_main_output_similarity(
        fused_decomposed_output_similarity,
        max_abs=_DSV4_FUSED_VS_DECOMPOSED_OUTPUT_MAX_ABS,
    )
    assert fused_decomposed_loss_diff <= _MAIN_LOSS_MAX_ABS_DIFF
    assert fused_decomposed_loss_symmetric_rel <= _MAX_MAIN_LOSS_SYMMETRIC_REL
    for similarities in (direct_public_grad_similarity, decomposed_grad_similarity):
        assert min(value["cosine"] for value in similarities.values()) >= (
            _MIN_MAIN_GRAD_COSINE
        ), similarities
        assert max(value["rms_relative"] for value in similarities.values()) <= (
            _MAX_MAIN_GRAD_RMS_REL
        ), similarities
        assert min(value["norm_ratio"] for value in similarities.values()) >= (
            _MIN_MAIN_GRAD_NORM_RATIO
        ), similarities
        assert max(value["norm_ratio"] for value in similarities.values()) <= (
            _MAX_MAIN_GRAD_NORM_RATIO
        ), similarities
    print(
        "NON_SKIP_DSV4_FUSED_DSA_DECOMPOSED_PARITY_PASSED "
        f"ratio={ratio} n_comp={n_comp} requested_topk={requested_topk} "
        f"window_topk={window_topk} kv_offset={kv_offset} "
        "topk_padding=True topk_score_proof_pass=True "
        "nontrivial_topk_score_proof=True "
        "main_grad_parity=True zero_indexer_grads=True "
        f"selection_topk={selection_topk} "
        f"padded_topk_ambiguous_rows={padded_topk_score_proof['ambiguous_rows']} "
        f"selection_topk_ambiguous_rows={selection_topk_score_proof['ambiguous_rows']} "
        f"loss_diff={fused_decomposed_loss_diff:.6e} "
        f"loss_symmetric_rel={fused_decomposed_loss_symmetric_rel:.6e} "
        f"direct_public_out_max_abs={direct_public_output_extrema['max_abs']:.6e} "
        f"output_min_cosine={fused_decomposed_output_extrema['min_cosine']:.6e} "
        f"output_max_rms_relative={fused_decomposed_output_extrema['max_rms_relative']:.6e} "
        f"output_min_norm_ratio={fused_decomposed_output_extrema['min_norm_ratio']:.6e} "
        f"output_max_norm_ratio={fused_decomposed_output_extrema['max_norm_ratio']:.6e} "
        f"output_max_abs={fused_decomposed_output_extrema['max_abs']:.6e} "
        f"min_main_grad_cosine={decomposed_grad_extrema['min_cosine']:.6e} "
        f"max_main_grad_rms_relative={decomposed_grad_extrema['max_rms_relative']:.6e} "
        f"min_main_grad_norm_ratio={decomposed_grad_extrema['min_norm_ratio']:.6e} "
        f"max_main_grad_norm_ratio={decomposed_grad_extrema['max_norm_ratio']:.6e}"
    )


def test_dsv4_csa_torch_fused_module_parity_real_gpu():
    """Compare the real CSA module's Torch, decomposed, and legacy fused paths."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the DSv4 CSA module parity smoke.")
    pytest.importorskip("cudnn", reason="DSv4 CSA module parity needs cudnn DSA.")

    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
    from megatron.lite.primitive.modules.attention.csa import (
        CompressedSparseAttention,
    )
    from megatron.lite.primitive.parallel import ParallelState

    device = torch.device("cuda", int(torch.cuda.current_device()))
    # Keep the released top-k while making the module genuinely sparse:
    # 2560 / ratio 4 = 640 compressed entries, so topk=512 excludes 128.
    seq = 2560
    cfg = DeepseekV4Config(
        num_hidden_layers=1,
        hidden_size=128,
        num_attention_heads=64,
        num_key_value_heads=1,
        head_dim=512,
        qk_rope_head_dim=64,
        q_lora_rank=32,
        o_lora_rank=32,
        o_groups=8,
        max_position_embeddings=seq,
        compress_ratios=[4],
        sliding_window=128,
        index_head_dim=128,
        index_n_heads=64,
        index_topk=512,
        rms_norm_eps=1.0e-6,
    )
    ps = ParallelState()

    torch.manual_seed(20260627)
    reference_module = CompressedSparseAttention(cfg, layer_idx=0, ps=ps).to(
        device=device, dtype=torch.bfloat16
    )

    def clone_reference_module():
        module = CompressedSparseAttention(cfg, layer_idx=0, ps=ps).to(
            device=device, dtype=torch.bfloat16
        )
        module.load_state_dict(reference_module.state_dict(), strict=True)
        return module

    decomposed_module = clone_reference_module()
    legacy_fused_module = clone_reference_module()
    x = torch.randn(1, seq, cfg.hidden_size, device=device, dtype=torch.bfloat16)
    position_ids = torch.arange(seq, device=device, dtype=torch.long).unsqueeze(0)
    grad_output = torch.randn_like(x)

    def run(module, *, backend: str, training: bool):
        module.zero_grad(set_to_none=True)
        module.attention_backend = backend
        module.train(training)
        local_x = x.detach().clone().requires_grad_(True)
        output = module(local_x, position_ids=position_ids)
        assert output.shape == x.shape
        assert torch.isfinite(output.float()).all()
        output.backward(grad_output)
        assert local_x.grad is not None
        assert torch.isfinite(local_x.grad.float()).all()

        main_grads = {}
        indexer_grads = {}
        for name, parameter in module.named_parameters():
            if name.startswith("indexer."):
                indexer_grads[name] = parameter.grad
                continue
            assert parameter.grad is not None, f"missing main gradient for {name}"
            assert torch.isfinite(parameter.grad.float()).all(), name
            main_grads[name] = parameter.grad.detach().float().reshape(-1)
        assert main_grads
        assert indexer_grads
        assert all(grad is None for grad in indexer_grads.values())
        return {
            "output": output.detach().float(),
            "x_grad": local_x.grad.detach().float(),
            "main_grads": main_grads,
        }

    reference = run(reference_module, backend="torch", training=True)
    decomposed = run(decomposed_module, backend="fused", training=False)
    legacy_fused = run(legacy_fused_module, backend="fused", training=True)

    output_comparisons = {}
    gradient_comparisons = {}
    for name, actual in (
        ("decomposed", decomposed),
        ("legacy_fused", legacy_fused),
    ):
        assert set(actual["main_grads"]) == set(reference["main_grads"])
        output_comparisons[name] = _tensor_similarity_metrics(
            actual["output"], reference["output"]
        )
        gradient_comparisons[f"{name}:input"] = _tensor_similarity_metrics(
            actual["x_grad"], reference["x_grad"]
        )
        for parameter_name in sorted(reference["main_grads"]):
            gradient_comparisons[f"{name}:param:{parameter_name}"] = (
                _tensor_similarity_metrics(
                    actual["main_grads"][parameter_name],
                    reference["main_grads"][parameter_name],
                )
            )

    output_extrema = _similarity_extrema(output_comparisons)
    gradient_extrema = _similarity_extrema(gradient_comparisons)
    print(
        "DSV4_CSA_TORCH_FUSED_DIAGNOSTICS "
        f"output_by_path={output_comparisons} "
        f"gradient_extrema={gradient_extrema}"
    )

    _assert_main_output_similarity(
        output_comparisons,
        max_abs=_DSV4_CSA_OUTPUT_MAX_ABS,
    )
    assert min(value["cosine"] for value in gradient_comparisons.values()) >= (
        _MIN_MAIN_GRAD_COSINE
    ), gradient_comparisons
    assert max(value["rms_relative"] for value in gradient_comparisons.values()) <= (
        _MAX_MAIN_GRAD_RMS_REL
    ), gradient_comparisons
    assert min(value["norm_ratio"] for value in gradient_comparisons.values()) >= (
        _MIN_MAIN_GRAD_NORM_RATIO
    ), gradient_comparisons
    assert max(value["norm_ratio"] for value in gradient_comparisons.values()) <= (
        _MAX_MAIN_GRAD_NORM_RATIO
    ), gradient_comparisons

    print(
        "NON_SKIP_DSV4_CSA_TORCH_FUSED_PARITY_PASSED "
        f"seq={seq} ratio=4 n_comp={seq // 4} indexer_topk={cfg.index_topk} "
        "nontrivial_sparse_selection=True "
        "torch_vs_decomposed_output=True torch_vs_legacy_output=True "
        "main_grad_parity=True indexer_grads_none=True "
        f"output_min_cosine={output_extrema['min_cosine']:.6e} "
        f"output_max_rms_relative={output_extrema['max_rms_relative']:.6e} "
        f"output_min_norm_ratio={output_extrema['min_norm_ratio']:.6e} "
        f"output_max_norm_ratio={output_extrema['max_norm_ratio']:.6e} "
        f"output_max_abs={output_extrema['max_abs']:.6e} "
        f"min_main_grad_cosine={gradient_extrema['min_cosine']:.6e} "
        f"max_main_grad_rms_relative={gradient_extrema['max_rms_relative']:.6e} "
        f"min_main_grad_norm_ratio={gradient_extrema['min_norm_ratio']:.6e} "
        f"max_main_grad_norm_ratio={gradient_extrema['max_norm_ratio']:.6e}"
    )


def test_glm52_model_preserves_multisegment_positions_through_indexshare_and_mtp():
    """A real packed model must not replace reset positions with flat arange."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the GLM5 packed-position smoke.")
    pytest.importorskip("cudnn", reason="GLM5 packed-position smoke needs cudnn DSA.")

    from types import SimpleNamespace

    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.model.glm5.lite.model import Glm5Model
    from megatron.lite.primitive.parallel import ParallelState
    from megatron.lite.primitive.utils.packed_seq import PackedSeqParams

    device = torch.device("cuda", int(torch.cuda.current_device()))
    cfg = Glm5Config(
        num_hidden_layers=2,
        hidden_size=128,
        num_attention_heads=64,
        num_key_value_heads=64,
        head_dim=256,
        vocab_size=32,
        max_position_embeddings=1024,
        initializer_range=0.002,
        q_lora_rank=16,
        kv_lora_rank=512,
        qk_head_dim=256,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        index_head_dim=128,
        index_n_heads=32,
        index_topk=512,
        intermediate_size=20,
        moe_intermediate_size=6,
        first_k_dense_replace=3,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        num_nextn_predict_layers=1,
        index_topk_freq=2,
        index_skip_topk_offset=1,
        indexer_types=["full", "shared"],
        rope_interleave=True,
        indexer_rope_interleave=True,
    )
    ps = ParallelState()
    train_cfg = SimpleNamespace(
        tp=1,
        ep=1,
        etp=1,
        pp=1,
        cp=1,
        vpp=None,
        use_deepep=False,
        fp8=False,
        recompute_modules=[],
        deterministic=True,
    )

    torch.manual_seed(20260627)
    model = Glm5Model(
        cfg,
        train_cfg,
        ps,
        mtp_enable=True,
        mtp_enable_train=False,
    ).to(device=device, dtype=torch.bfloat16)
    model.eval()
    assert model.mtp is not None

    segment_length = 512
    total_tokens = segment_length * 2
    input_ids = torch.randint(
        0, cfg.vocab_size, (1, total_tokens), device=device, dtype=torch.long
    )
    segment_positions = torch.arange(segment_length, device=device, dtype=torch.long)
    reset_positions = torch.cat((segment_positions, segment_positions)).unsqueeze(0)
    cu_seqlens = torch.tensor(
        [0, segment_length, total_tokens], device=device, dtype=torch.int32
    )
    packed_seq_params = PackedSeqParams.from_cu_seqlens(
        cu_seqlens, max_seqlen=segment_length
    )

    captured: dict[int, torch.Tensor] = {}
    handles = []

    def capture_positions(module, _args, kwargs):
        captured[module.layer_number] = kwargs["position_ids"].detach().clone()

    attention_modules = [
        layer.self_attention.self_attention for layer in model.layers
    ] + [model.mtp.layers[0].transformer_layer.self_attention.self_attention]
    for attention in attention_modules:
        handles.append(
            attention.register_forward_pre_hook(capture_positions, with_kwargs=True)
        )

    try:
        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                position_ids=reset_positions,
                packed_seq_params=packed_seq_params,
            )
    finally:
        for handle in handles:
            handle.remove()

    assert output["logits"].shape == (1, total_tokens, cfg.vocab_size)
    assert torch.isfinite(output["logits"].float()).all()
    assert "mtp_logits" in output and len(output["mtp_logits"]) == 1
    assert torch.equal(captured[1], reset_positions)
    assert torch.equal(captured[2], reset_positions)
    assert torch.equal(captured[3], reset_positions)
    print(
        "NON_SKIP_GLM52_PACKED_POSITION_IDS_PASSED "
        f"segments=2 segment_length={segment_length} "
        f"trunk_reset={torch.equal(captured[2], reset_positions)} "
        f"mtp_rotary_reused={torch.equal(captured[3], reset_positions)}"
    )
