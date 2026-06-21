# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unit coverage for DSA kernel fallback paths."""

from __future__ import annotations

import os
from contextlib import contextmanager

import torch

from megatron.lite.primitive.kernels import dsa_kernels


@contextmanager
def _env(name: str, value: str):
    old_value = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old_value


def _reference_indexer_topk(q, k, weights, *, topk: int, ratio: int):
    q_bshd = q.permute(1, 0, 2, 3).contiguous()
    k_bsd = k.permute(1, 0, 2).contiguous()
    w_bsh = weights.permute(1, 0, 2).contiguous()
    b, sq, _heads, _dim = q_bshd.shape
    sk = k_bsd.shape[1]

    logits = torch.einsum("bqhd,bkd->bqhk", q_bshd.float(), k_bsd.float())
    scores = (torch.relu(logits) * w_bsh.float().unsqueeze(-1)).sum(dim=2)
    valid_per_q = ((torch.arange(sq, device=q.device) + 1) // ratio).clamp(max=sk)
    valid = torch.arange(sk, device=q.device).view(1, 1, sk) < valid_per_q.view(1, sq, 1)
    scores = scores.masked_fill(~valid, float("-inf"))

    topk_k = min(topk, sk)
    if topk_k > 0:
        topk_scores, indices = torch.topk(scores, k=topk_k, dim=-1)
        indices = indices.masked_fill(topk_scores == float("-inf"), -1)
    else:
        indices = torch.empty(b, sq, 0, device=q.device, dtype=torch.long)
    if topk_k < topk:
        pad = torch.full((b, sq, topk - topk_k), -1, dtype=torch.long, device=q.device)
        indices = torch.cat([indices, pad], dim=-1)
    indices = indices.to(torch.int32)
    lengths = (indices >= 0).sum(dim=-1).to(torch.int32)
    return indices, lengths


def _reference_flat_topk_idxs(local, *, batch_size: int, seqlen_kv: int, compact: bool):
    b, sq, topk = local.shape
    assert b == batch_size
    idxs = local.permute(1, 0, 2).reshape(sq * b, topk)
    batch_ids = (torch.arange(sq * b, device=local.device) % b).view(-1, 1)
    valid = idxs >= 0
    global_idxs = torch.where(valid, idxs * batch_size + batch_ids, idxs).to(torch.int32)
    if not compact:
        return global_idxs, None
    rows = []
    lengths = []
    for row in global_idxs:
        valid_row = row[row >= 0]
        lengths.append(valid_row.numel())
        rows.append(
            torch.cat(
                [
                    valid_row,
                    torch.full(
                        (row.numel() - valid_row.numel(),),
                        -1,
                        dtype=row.dtype,
                        device=row.device,
                    ),
                ]
            )
        )
    return torch.stack(rows), torch.tensor(lengths, dtype=torch.int32, device=local.device)


def _reference_sparse_attn(query, kv, topk, *, softmax_scale: float, topk_length):
    sq, b, num_heads, dim = query.shape
    skv = kv.shape[0]
    q_flat = query.reshape(sq * b, num_heads, dim)
    kv_flat = kv.reshape(skv * b, dim)
    out_rows = []
    for row_idx in range(sq * b):
        row = []
        for head_idx in range(num_heads):
            valid_count = int(topk_length[row_idx].item())
            row_indices = topk[row_idx, :valid_count].long()
            gathered = kv_flat.index_select(0, row_indices)
            logits = (gathered.float() @ q_flat[row_idx, head_idx].float()) * softmax_scale
            probs = torch.softmax(logits, dim=-1)
            row.append((probs[:, None] * gathered.float()).sum(dim=0).to(query.dtype))
        out_rows.append(torch.stack(row, dim=0))
    return torch.stack(out_rows, dim=0).reshape(sq, b, num_heads * dim)


def test_dsa_indexer_topk_torch_backend_returns_valid_indices():
    with _env("MLITE_DSA_INDEXER_TOPK_BACKEND", "torch"):
        q = torch.tensor(
            [[[[1.0, 0.0]]], [[[0.0, 1.0]]], [[[1.0, 1.0]]]],
            dtype=torch.float32,
        )
        k = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]], dtype=torch.float32)
        weights = torch.ones(3, 1, 1, dtype=torch.float32)
        expected_indices, expected_lengths = _reference_indexer_topk(
            q, k, weights, topk=2, ratio=1
        )

        for _ in range(3):
            topk_indices, topk_length = dsa_kernels.indexer_topk(q, k, weights, topk=2, ratio=1)
            assert torch.equal(topk_indices, expected_indices)
            assert torch.equal(topk_length, expected_lengths)

    assert topk_indices.shape == (1, 3, 2)
    assert topk_length.shape == (1, 3)
    assert topk_indices.dtype == torch.int32
    assert topk_length.dtype == torch.int32
    assert torch.all((topk_indices >= -1) & (topk_indices < 3))


def test_build_flat_topk_idxs_torch_compact_moves_invalid_slots_to_tail():
    with _env("MLITE_DSA_INDEXER_TOPK_BACKEND", "torch"):
        local = torch.tensor([[[2, -1, 0], [-1, 1, -1]]], dtype=torch.int32)
        expected_flat, expected_lengths = _reference_flat_topk_idxs(
            local, batch_size=1, seqlen_kv=3, compact=True
        )

        for _ in range(3):
            flat, lengths = dsa_kernels.build_flat_topk_idxs(
                local, batch_size=1, seqlen_kv=3, compact=True
            )
            assert torch.equal(flat, expected_flat)
            assert torch.equal(lengths, expected_lengths)

    assert flat.dtype == torch.int32
    assert lengths.dtype == torch.int32


def test_dsa_sparse_attn_torch_backend_is_forward_only_reference():
    with _env("MLITE_DSA_SPARSE_ATTN_BACKEND", "torch"):
        query = torch.tensor([[[[1.0, 0.0]]], [[[0.0, 1.0]]]], dtype=torch.float32)
        kv = torch.tensor([[[1.0, 0.0]], [[0.0, 2.0]]], dtype=torch.float32)
        topk = torch.tensor([[0, -1], [1, 0]], dtype=torch.int32)
        topk_length = torch.tensor([1, 2], dtype=torch.int32)
        expected = _reference_sparse_attn(
            query, kv, topk, softmax_scale=1.0, topk_length=topk_length
        )

        for _ in range(3):
            out = dsa_kernels.dsa_sparse_attn(
                query,
                kv,
                None,
                topk,
                softmax_scale=1.0,
                topk_length=topk_length,
                indexer_topk=0,
                value_dim=2,
            )
            assert torch.equal(out, expected)

    assert out.shape == (2, 1, 2)
    assert torch.isfinite(out).all()
    assert dsa_kernels.torch_sparse_attn_fallback_call_count() >= 1
