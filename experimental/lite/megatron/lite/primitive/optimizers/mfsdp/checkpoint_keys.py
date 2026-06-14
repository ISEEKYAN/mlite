# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""M-FSDP checkpoint key canonicalization helpers."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any

MFSDP_CHECKPOINT_PS_ATTR = "_mlite_mfsdp_checkpoint_parallel_state"
MFSDP_CHECKPOINT_EXPERT_CLASSIFIER_ATTR = "_mlite_mfsdp_checkpoint_expert_classifier"


def attach_mfsdp_checkpoint_metadata(
    optimizer: Any,
    *,
    ps: Any,
    is_expert: Callable[[str], bool],
) -> None:
    """Attach Megatron Lite checkpoint metadata to an MCore optimizer and its leaves."""
    for target in _optimizer_and_leaves(optimizer):
        setattr(target, MFSDP_CHECKPOINT_PS_ATTR, ps)
        setattr(target, MFSDP_CHECKPOINT_EXPERT_CLASSIFIER_ATTR, is_expert)


def optimizer_checkpoint_parallel_state(optimizer: Any) -> Any | None:
    for target in _optimizer_and_leaves(optimizer):
        ps = getattr(target, MFSDP_CHECKPOINT_PS_ATTR, None)
        if ps is not None:
            return ps
        ps = getattr(target, "ps", None)
        if ps is not None:
            return ps
    return None


def optimizer_checkpoint_expert_classifier(
    optimizer: Any,
) -> Callable[[str], bool] | None:
    for target in _optimizer_and_leaves(optimizer):
        classifier = getattr(target, MFSDP_CHECKPOINT_EXPERT_CLASSIFIER_ATTR, None)
        if callable(classifier):
            return classifier
    return None


def expert_local_counts(
    names: Iterable[str],
    is_expert: Callable[[str], bool],
) -> dict[str, int]:
    local_indices: dict[str, set[int]] = {}
    for name in names:
        if not is_expert(name):
            continue
        match = re.search(r"weight(\d+)$", name)
        if match is None:
            continue
        prefix = expert_name_prefix(name)
        local_idx = int(match.group(1))
        local_indices.setdefault(prefix, set()).add(local_idx)
    return {
        prefix: len(indices)
        for prefix, indices in local_indices.items()
        if indices == set(range(len(indices)))
    }


def canonicalize_expert_checkpoint_key(
    key: str,
    name: str,
    *,
    ps: Any,
    is_expert: bool,
    local_counts: dict[str, int],
) -> tuple[str, str]:
    """Map M-FSDP local expert ids to global checkpoint ids.

    M-FSDP exposes expert weights with local EP ids, e.g. every EP rank has
    ``...fc1.weight0`` for a different global expert. DCP keys must identify
    the global expert, otherwise EP ranks write/read different experts under
    the same key.
    """
    if not is_expert or int(getattr(ps, "ep_size", 1) or 1) <= 1:
        return key, name
    local_match = re.search(r"weight(\d+)$", name)
    if local_match is None:
        return key, name
    local_idx = int(local_match.group(1))
    num_local = local_counts.get(expert_name_prefix(name))
    if num_local is None:
        return key, name
    if local_idx >= num_local:
        return key, name
    global_idx = int(getattr(ps, "ep_rank", 0)) * num_local + local_idx
    return (
        replace_trailing_expert_idx(key, global_idx),
        replace_trailing_expert_idx(name, global_idx),
    )


def expert_name_prefix(name: str) -> str:
    return re.sub(r"weight\d+$", "weight", name)


def replace_trailing_expert_idx(name: str, idx: int) -> str:
    return re.sub(r"weight\d+$", f"weight{idx}", name)


def normalize_mcore_fsdp_param_name(name: str) -> str:
    prefix = "module.module."
    return name[len(prefix) :] if name.startswith(prefix) else name


def _optimizer_and_leaves(optimizer: Any) -> tuple[Any, ...]:
    leaves = getattr(optimizer, "chained_optimizers", None)
    if leaves is None:
        return (optimizer,)
    return (optimizer, *tuple(leaves))


__all__ = [
    "attach_mfsdp_checkpoint_metadata",
    "canonicalize_expert_checkpoint_key",
    "expert_local_counts",
    "normalize_mcore_fsdp_param_name",
    "optimizer_checkpoint_expert_classifier",
    "optimizer_checkpoint_parallel_state",
]
