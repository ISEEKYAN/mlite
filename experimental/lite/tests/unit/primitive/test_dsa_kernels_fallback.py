# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unit coverage for DSA kernel fallback paths."""

from __future__ import annotations

import os

import torch

from megatron.lite.primitive.kernels import dsa_kernels


def test_dsa_indexer_topk_torch_backend_returns_valid_indices():
    old_backend = os.environ.get("MLITE_DSA_INDEXER_TOPK_BACKEND")
    os.environ["MLITE_DSA_INDEXER_TOPK_BACKEND"] = "torch"
    try:
        q = torch.tensor(
            [[[[1.0, 0.0]]], [[[0.0, 1.0]]], [[[1.0, 1.0]]]],
            dtype=torch.float32,
        )
        k = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]], dtype=torch.float32)
        weights = torch.ones(3, 1, 1, dtype=torch.float32)

        topk_indices, topk_length = dsa_kernels.indexer_topk(q, k, weights, topk=2, ratio=1)
    finally:
        if old_backend is None:
            os.environ.pop("MLITE_DSA_INDEXER_TOPK_BACKEND", None)
        else:
            os.environ["MLITE_DSA_INDEXER_TOPK_BACKEND"] = old_backend

    assert topk_indices.shape == (1, 3, 2)
    assert topk_length.shape == (1, 3)
    assert topk_indices.dtype == torch.int32
    assert topk_length.dtype == torch.int32
    assert torch.all((topk_indices >= -1) & (topk_indices < 3))
    assert torch.equal(topk_length, torch.tensor([[1, 2, 2]], dtype=torch.int32))


def test_build_flat_topk_idxs_torch_compact_moves_invalid_slots_to_tail():
    old_backend = os.environ.get("MLITE_DSA_INDEXER_TOPK_BACKEND")
    os.environ["MLITE_DSA_INDEXER_TOPK_BACKEND"] = "torch"
    try:
        local = torch.tensor([[[2, -1, 0], [-1, 1, -1]]], dtype=torch.int32)

        flat, lengths = dsa_kernels.build_flat_topk_idxs(
            local, batch_size=1, seqlen_kv=3, compact=True
        )
    finally:
        if old_backend is None:
            os.environ.pop("MLITE_DSA_INDEXER_TOPK_BACKEND", None)
        else:
            os.environ["MLITE_DSA_INDEXER_TOPK_BACKEND"] = old_backend

    assert torch.equal(flat, torch.tensor([[2, 0, -1], [1, -1, -1]], dtype=torch.int32))
    assert torch.equal(lengths, torch.tensor([2, 1], dtype=torch.int32))


def test_dsa_sparse_attn_torch_backend_is_forward_only_reference():
    old_backend = os.environ.get("MLITE_DSA_SPARSE_ATTN_BACKEND")
    os.environ["MLITE_DSA_SPARSE_ATTN_BACKEND"] = "torch"
    try:
        query = torch.tensor([[[[1.0, 0.0]]], [[[0.0, 1.0]]]], dtype=torch.float32)
        kv = torch.tensor([[[1.0, 0.0]], [[0.0, 2.0]]], dtype=torch.float32)
        topk = torch.tensor([[0, -1], [1, 0]], dtype=torch.int32)
        topk_length = torch.tensor([1, 2], dtype=torch.int32)

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
    finally:
        if old_backend is None:
            os.environ.pop("MLITE_DSA_SPARSE_ATTN_BACKEND", None)
        else:
            os.environ["MLITE_DSA_SPARSE_ATTN_BACKEND"] = old_backend

    assert out.shape == (2, 1, 2)
    assert torch.isfinite(out).all()
    assert dsa_kernels.torch_sparse_attn_fallback_call_count() >= 1
