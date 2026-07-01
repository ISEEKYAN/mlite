# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fused RoPE kernel boundary for MLite attention primitives."""

from __future__ import annotations

from importlib import import_module

import torch


def _require_contiguous_last_dim(tensor: torch.Tensor, name: str) -> None:
    if tensor.stride(-1) != 1:
        raise ValueError(f"{name} must have a contiguous last dimension")


def _load_generic_kernels():
    try:
        module = import_module("megatron.core.extensions.transformer_engine")
        return module.fused_apply_rotary_pos_emb, module.fused_apply_rotary_pos_emb_thd
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Fused generic RoPE requires Megatron-Core's Transformer Engine RoPE kernels"
        ) from exc


def _load_mla_kernels():
    try:
        module = import_module("megatron.core.fusions.fused_mla_yarn_rope_apply")
        return module.fused_mla_rope_inplace, module.fused_mla_rope_kv_split
    except (ImportError, AttributeError) as exc:
        raise RuntimeError("Fused MLA RoPE kernels are unavailable") from exc


def apply_fused_rotary(
    tensor: torch.Tensor,
    freqs: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor | None = None,
    cp_rank: int = 0,
    cp_size: int = 1,
    rotary_interleaved: bool = False,
) -> torch.Tensor:
    """Apply the upstream TE-backed fused SBHD or THD RoPE kernel."""
    _require_contiguous_last_dim(tensor, "tensor")
    if cu_seqlens is None:
        if tensor.ndim != 4:
            raise ValueError("Fused SBHD RoPE expects a 4-D [S,B,H,D] tensor")
        if cp_size != 1 or cp_rank != 0:
            raise ValueError("SBHD frequencies must already be CP-local")
        fused_sbhd, _ = _load_generic_kernels()
        return fused_sbhd(tensor, freqs, interleaved=rotary_interleaved)
    if tensor.ndim != 3:
        raise ValueError("Fused THD RoPE expects a 3-D [T,H,D] tensor")
    if cp_size < 1 or not 0 <= cp_rank < cp_size:
        raise ValueError(f"Invalid CP coordinates rank={cp_rank}, size={cp_size}")
    _, fused_thd = _load_generic_kernels()
    return fused_thd(
        tensor,
        cu_seqlens,
        freqs,
        cp_size=cp_size,
        cp_rank=cp_rank,
        interleaved=rotary_interleaved,
    )


def apply_fused_mla_rotary_for_q(
    tensor: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    nope_dim: int,
    rope_dim: int,
    cu_seqlens: torch.Tensor | None = None,
    cp_rank: int = 0,
    cp_size: int = 1,
    inverse: bool = False,
    remove_interleaving: bool = False,
) -> torch.Tensor:
    """Apply upstream in-place MLA Q RoPE; callers own any required clone."""
    _require_contiguous_last_dim(tensor, "tensor")
    if tensor.shape[-1] != nope_dim + rope_dim:
        raise ValueError("MLA Q last dimension does not match NoPE + RoPE dimensions")
    if rope_dim % 4:
        raise ValueError("MLA fused RoPE dimension must be divisible by 4")
    fused_q, _ = _load_mla_kernels()
    return fused_q(
        tensor,
        cos,
        sin,
        nope_dim,
        rope_dim,
        cu_seqlens,
        cp_rank,
        cp_size,
        False,
        inverse,
        remove_interleaving,
    )


def apply_fused_mla_rotary_for_kv(
    kv: torch.Tensor,
    k_pos_emb: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    rope_dim: int,
    key_nope_dim: int,
    value_dim: int,
    cu_seqlens_kv: torch.Tensor | None = None,
    cp_rank: int = 0,
    cp_size: int = 1,
    remove_interleaving: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split compressed KV and apply upstream fused MLA KV RoPE."""
    _require_contiguous_last_dim(kv, "kv")
    _require_contiguous_last_dim(k_pos_emb, "k_pos_emb")
    if kv.shape[-1] != key_nope_dim + value_dim:
        raise ValueError(
            "MLA KV last dimension does not match key NoPE + value dimensions"
        )
    if rope_dim % 4:
        raise ValueError("MLA fused RoPE dimension must be divisible by 4")
    _, fused_kv = _load_mla_kernels()
    return fused_kv(
        kv,
        k_pos_emb,
        cos,
        sin,
        rope_dim,
        key_nope_dim,
        value_dim,
        cu_seqlens_kv,
        cp_rank,
        cp_size,
        False,
        remove_interleaving,
    )


__all__ = [
    "apply_fused_mla_rotary_for_kv",
    "apply_fused_mla_rotary_for_q",
    "apply_fused_rotary",
]
