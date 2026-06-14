# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Shared attention primitives for Megatron Lite native models."""

from __future__ import annotations

from megatron.lite.primitive.attention.dsa import (
    DSAIndexer,
    DSAIndexerLossAutoScaler,
    DynamicSparseAttention,
    RMSNorm,
    apply_rotary_emb,
    apply_rotary_pos_emb,
    build_rope_cache,
    build_rotary_embeddings,
    rotate_activation,
    rotate_half,
)


def __getattr__(name: str):
    if name == "MultiLatentAttention":
        from megatron.lite.primitive.attention.mla import MultiLatentAttention

        return MultiLatentAttention
    raise AttributeError(name)


__all__ = [
    "DSAIndexer",
    "DSAIndexerLossAutoScaler",
    "DynamicSparseAttention",
    "MultiLatentAttention",
    "RMSNorm",
    "apply_rotary_emb",
    "apply_rotary_pos_emb",
    "build_rope_cache",
    "build_rotary_embeddings",
    "rotate_activation",
    "rotate_half",
]
