# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DTensor gradient helpers local to the Megatron-FSDP primitive."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

try:  # pragma: no cover - import availability is PyTorch-version dependent.
    from torch.distributed.tensor import DTensor, Replicate
except ImportError:  # pragma: no cover
    DTensor = Replicate = None  # type: ignore[assignment]


def sharded_grad_sq_sum(
    params: Iterable[nn.Parameter],
    *,
    accum_dtype: str | torch.dtype = torch.float32,
    default_device: torch.device | None = None,
    chunk_size_numel: int = 0,
    include_param: Callable[[nn.Parameter], bool] | None = None,
    scalar_all_reduce: Callable[[torch.Tensor, dist.ProcessGroup, dist.ReduceOp], None]
    | None = None,
) -> torch.Tensor:
    """Return global L2 grad squared-sum for M-FSDP Tensor/DTensor params.

    This intentionally mirrors FSDP2's helper semantics, but stays inside the
    M-FSDP package so the primitive does not depend on FSDP2 implementation
    modules. Reductions that are model-layout policy, such as PP or EP, remain
    in ``grad_norm.py``.
    """

    dtype = _resolve_torch_dtype(accum_dtype)
    groups = _group_grads(params)
    total: torch.Tensor | None = None
    for group in groups.values():
        local_sq = _group_local_sq_sum(
            group,
            dtype=dtype,
            chunk_size_numel=chunk_size_numel,
            include_param=include_param,
        )
        meta = group[0][2]
        if meta is not None and not _has_partial_placement(meta) and dist.is_initialized():
            _reduce_dtensor_scalar_(
                local_sq,
                meta,
                op=dist.ReduceOp.SUM,
                scalar_all_reduce=scalar_all_reduce,
            )
        total = local_sq if total is None else total.to(local_sq.device) + local_sq

    if total is None:
        return torch.zeros((), device=default_device or torch.device("cpu"), dtype=dtype)
    return total


def _resolve_torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        resolved = dtype
    else:
        name = dtype.removeprefix("torch.")
        resolved = getattr(torch, name, None)
    if not isinstance(resolved, torch.dtype):
        raise ValueError(f"Unsupported torch dtype for grad norm accumulation: {dtype!r}")
    if not torch.empty((), dtype=resolved).is_floating_point():
        raise ValueError(f"Grad norm accumulation dtype must be floating point: {dtype!r}")
    return resolved


def _group_grads(
    params: Iterable[nn.Parameter],
) -> dict[tuple[Any, ...], list[tuple[nn.Parameter, torch.Tensor, Any | None]]]:
    groups: dict[tuple[Any, ...], list[tuple[nn.Parameter, torch.Tensor, Any | None]]] = defaultdict(
        list
    )
    for param in params:
        grad = param.grad
        if grad is None:
            continue
        meta = _dtensor_meta(param, grad)
        if meta is None:
            key = ("tensor", _local_grad(grad, meta).device)
        else:
            key = (
                "dtensor",
                id(meta.device_mesh),
                tuple(
                    (type(placement).__name__, repr(placement))
                    for placement in meta.placements
                ),
            )
        groups[key].append((param, grad, meta))
    return groups


def _group_local_sq_sum(
    group: list[tuple[nn.Parameter, torch.Tensor, Any | None]],
    *,
    dtype: torch.dtype,
    chunk_size_numel: int = 0,
    include_param: Callable[[nn.Parameter], bool] | None = None,
) -> torch.Tensor:
    device = _local_grad(group[0][1], group[0][2]).device
    total = torch.zeros((), device=device, dtype=dtype)
    for param, grad, meta in group:
        if include_param is not None and not include_param(param):
            continue
        local_grad = _local_grad(grad, meta)
        total += _tensor_sq_sum(
            local_grad.detach(),
            dtype=dtype,
            chunk_size_numel=chunk_size_numel,
        )
    return total


def _tensor_sq_sum(
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype,
    chunk_size_numel: int = 0,
) -> torch.Tensor:
    if chunk_size_numel <= 0 or tensor.numel() <= chunk_size_numel:
        return tensor.to(dtype).pow(2).sum()
    try:
        flat = tensor.view(-1)
    except RuntimeError:
        flat = tensor.reshape(-1)
    total = torch.zeros((), device=tensor.device, dtype=dtype)
    for start in range(0, flat.numel(), chunk_size_numel):
        chunk = flat.narrow(0, start, min(chunk_size_numel, flat.numel() - start))
        total += chunk.to(dtype).pow(2).sum()
    return total


def _local_grad(grad: torch.Tensor, meta: Any | None) -> torch.Tensor:
    if meta is not None and _has_partial_placement(meta):
        full_tensor = getattr(grad, "full_tensor", None)
        if callable(full_tensor):
            return full_tensor()
    local_tensor = getattr(grad, "_local_tensor", None)
    if isinstance(local_tensor, torch.Tensor):
        return local_tensor
    to_local = getattr(grad, "to_local", None)
    if callable(to_local):
        return to_local()
    return grad


def _dtensor_meta(param: nn.Parameter, grad: torch.Tensor) -> Any | None:
    if _is_dtensor_like(grad):
        return grad
    if _is_dtensor_like(param):
        return param
    return None


def _is_dtensor_like(tensor: Any) -> bool:
    if DTensor is not None and isinstance(tensor, DTensor):
        return True
    return (
        callable(getattr(tensor, "to_local", None))
        and hasattr(tensor, "device_mesh")
        and hasattr(tensor, "placements")
    )


def _has_partial_placement(dtensor: Any) -> bool:
    return any(_placement_name(placement) == "Partial" for placement in dtensor.placements)


def _is_replicate_placement(placement: Any) -> bool:
    if Replicate is not None and isinstance(placement, Replicate):
        return True
    return _placement_name(placement) == "Replicate"


def _placement_name(placement: Any) -> str:
    return type(placement).__name__


def _reduce_dtensor_scalar_(
    value: torch.Tensor,
    dtensor: Any,
    *,
    op: dist.ReduceOp,
    scalar_all_reduce: Callable[[torch.Tensor, dist.ProcessGroup, dist.ReduceOp], None]
    | None = None,
) -> None:
    for mesh_dim, placement in enumerate(dtensor.placements):
        if _is_replicate_placement(placement):
            continue
        group = dtensor.device_mesh.get_group(mesh_dim)
        if dist.get_world_size(group) > 1:
            if scalar_all_reduce is None:
                dist.all_reduce(value, op=op, group=group)
            else:
                scalar_all_reduce(value, group, op)


__all__ = ["sharded_grad_sq_sum"]
