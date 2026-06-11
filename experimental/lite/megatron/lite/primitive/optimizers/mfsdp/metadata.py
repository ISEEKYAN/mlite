# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MCore-FSDP metadata adapters for Megatron Lite-owned module implementations."""

from __future__ import annotations

import torch.nn as nn  # pyright: ignore[reportMissingImports]

from megatron.lite.primitive.optimizers.mfsdp.patches import (
    clear_skip_tp_duplicate_sync,
    mark_skip_tp_duplicate_sync,
    set_mfsdp_param_name,
)
from megatron.lite.primitive.protocols import ExpertClassifierFn


_COLUMN_PARALLEL_WEIGHT_SUFFIXES: tuple[str, ...] = (
    "embed.embedding.weight",
    "mtp_embed.embedding.weight",
    "head.col.linear.weight",
    ".attn.qkv.linear.weight",
    ".attn.linear_qkv.weight",
    ".self_attention.linear_qkv.weight",
    ".self_attention.in_proj.weight",
    ".self_attention.gdn.in_proj.weight",
    ".eh_proj.linear.weight",
    ".experts.fc1.weight",
    ".experts.linear_fc1.weight",
    ".shared_experts.linear_fc1.weight",
    ".linear_fc1.weight",
)

_ROW_PARALLEL_WEIGHT_SUFFIXES: tuple[str, ...] = (
    ".attn.proj.linear.weight",
    ".attn.linear_proj.weight",
    ".self_attention.linear_proj.weight",
    ".self_attention.out_proj.weight",
    ".self_attention.gdn.out_proj.weight",
    ".linear_proj.weight",
    ".experts.fc2.weight",
    ".experts.linear_fc2.weight",
    ".shared_experts.linear_fc2.weight",
    ".linear_fc2.weight",
    ".down_proj.weight",
    ".o_proj.weight",
)

_MANUAL_MCORE_TP_MODE_WEIGHT_SUFFIXES: tuple[str, ...] = (
    "embed.embedding.weight",
    "mtp_embed.embedding.weight",
    "head.col.linear.weight",
)


def _matches_weight_suffix(name: str, suffix: str) -> bool:
    if suffix.startswith(".") and name == suffix[1:]:
        return True
    if name.endswith(suffix):
        return True
    idx = name.rfind(suffix)
    if idx < 0:
        return False
    return name[idx + len(suffix) :].isdigit()


def _infer_tp_partition_dim(name: str) -> int | None:
    """Infer Megatron Lite-owned TP weight partition dim for MCore-FSDP metadata."""
    if any(
        _matches_weight_suffix(name, suffix)
        for suffix in _COLUMN_PARALLEL_WEIGHT_SUFFIXES
    ):
        return 0
    if any(
        _matches_weight_suffix(name, suffix)
        for suffix in _ROW_PARALLEL_WEIGHT_SUFFIXES
    ):
        return 1
    return None


def _requires_manual_mcore_tp_mode(name: str) -> bool:
    return any(
        _matches_weight_suffix(name, suffix)
        for suffix in _MANUAL_MCORE_TP_MODE_WEIGHT_SUFFIXES
    )


def normalize_mfsdp_expert_tensor_parallel_attrs(
    model: nn.Module,
    is_expert_param: ExpertClassifierFn,
    *,
    etp_size: int,
) -> None:
    """Normalize expert TP metadata to expert-tensor-parallel, not dense TP.

    ``_mark_mc_parallel_attrs`` only knows dense TP size and conservatively marks
    every unmarked 2D param as tensor-parallel. Megatron Lite's routed experts are sharded by
    EP and optionally ETP; when ETP=1 they must not carry dense TP metadata or
    MCore-FSDP may scatter/gather optimizer main weights with the wrong mesh.
    """
    etp_size = max(int(etp_size), 1)
    for name, param in model.named_parameters():
        if not is_expert_param(name):
            continue
        if etp_size <= 1:
            param.tensor_model_parallel = False
            _clear_tp_attrs(param)
            continue
        if param.ndim <= 1:
            param.tensor_model_parallel = False
            _clear_tp_attrs(param)
            continue
        if getattr(param, "average_gradients_across_tp_domain", False):
            continue
        if getattr(param, "sequence_parallel", False):
            continue
        partition_dim = getattr(param, "partition_dim", None)
        if partition_dim not in (0, 1):
            partition_dim = _infer_tp_partition_dim(name)
        if partition_dim is None:
            param.tensor_model_parallel = False
            _clear_tp_attrs(param)
            continue
        _set_tp_attrs(param, partition_dim, set_mcore_mode=True)


def _clear_attr(obj, attr: str) -> None:
    if hasattr(obj, attr):
        delattr(obj, attr)


def _clear_tp_attrs(param) -> None:
    _clear_attr(param, "partition_dim")
    _clear_attr(param, "partition_stride")
    _clear_attr(param, "_tensor_parallel_mode")
    clear_skip_tp_duplicate_sync(param)


def _set_tp_attrs(param, partition_dim: int, *, set_mcore_mode: bool) -> None:
    param.tensor_model_parallel = True
    param.partition_dim = int(partition_dim)
    param.partition_stride = getattr(param, "partition_stride", 1)
    mark_skip_tp_duplicate_sync(param)
    if set_mcore_mode:
        param._tensor_parallel_mode = "column" if int(partition_dim) == 0 else "row"
    else:
        _clear_attr(param, "_tensor_parallel_mode")


def ensure_mfsdp_tp_partition_attrs(model: nn.Module) -> None:
    """Fill MCore-FSDP TP metadata for Megatron Lite-owned tensor-parallel params.

    MCore-native TP layers already set a precise partition dimension. Megatron Lite's
    custom column/vocab layers are output/vocab sharded (dim 0), while Megatron Lite's
    row-parallel projection weights are input sharded (dim 1). The partition
    attrs are enough for current MCore-FSDP to build TP-sharded DTensors, while
    the local patch below only guards duplicate-TP sync paths for marked Megatron Lite
    local shards.

    This pass is deliberately conservative with ``_tensor_parallel_mode``.
    Setting the mode on every Megatron Lite/TE dense TP weight changes Megatron-FSDP's
    model-buffer initialization path and has to be enabled only after a
    topology-specific precision gate proves it safe. Attention row-projection
    weights are intentionally kept on the local-shard path: assigning MCore
    ``_tensor_parallel_mode="row"`` currently hangs Qwen3MoE 4n32g before the
    first optimizer step. For the remaining local TP
    shards, the M-FSDP primitive marks them so its MCore patch skips only the
    incorrect duplicate-TP broadcast.
    """
    _ensure_mfsdp_tp_partition_attrs(model, set_mcore_mode_for_all=False)


def _ensure_mfsdp_tp_partition_attrs(
    model: nn.Module,
    *,
    set_mcore_mode_for_all: bool,
) -> None:
    for name, param in model.named_parameters():
        set_mfsdp_param_name(param, name)
        if param.ndim <= 1:
            if getattr(param, "tensor_model_parallel", False):
                param.tensor_model_parallel = False
                _clear_tp_attrs(param)
            continue
        if not getattr(param, "tensor_model_parallel", False):
            _clear_attr(param, "_tensor_parallel_mode")
            clear_skip_tp_duplicate_sync(param)
            continue
        partition_dim = getattr(param, "partition_dim", None)
        if partition_dim not in (0, 1):
            partition_dim = _infer_tp_partition_dim(name)
        if partition_dim is None:
            param.tensor_model_parallel = False
            _clear_tp_attrs(param)
            continue
        _set_tp_attrs(
            param,
            partition_dim,
            set_mcore_mode=(
                set_mcore_mode_for_all or _requires_manual_mcore_tp_mode(name)
            ),
        )


__all__ = [
    "ensure_mfsdp_tp_partition_attrs",
    "normalize_mfsdp_expert_tensor_parallel_attrs",
]
