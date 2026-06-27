# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Pipeline-replicated input embedding support for MTP.

When MTP is placed on the loss stage, it still consumes the input embedding.
Under pipeline parallelism that stage owns a physical replica because the
canonical input embedding lives on the first stage.  The two parameters must
start equal and must receive the sum of both gradients before every optimizer
step; otherwise PP changes the model being trained and HF export silently
discards the MTP-stage value.

Every parameter/gradient data collective is preceded by a fixed-size metadata
collective.  This is deliberately more strict than relying on the data
collective to reject incompatible tensors: shape, dtype, or device mismatches
can otherwise leave one endpoint blocked indefinitely.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn

_MAX_TENSOR_NDIM = 8
_METADATA_VERSION = 1
_METADATA_SIZE = 8 + _MAX_TENSOR_NDIM

_DTYPE_CODES = {
    dtype: index
    for index, dtype in enumerate(
        (
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128,
            getattr(torch, "float8_e4m3fn", None),
            getattr(torch, "float8_e5m2", None),
            getattr(torch, "float8_e4m3fnuz", None),
            getattr(torch, "float8_e5m2fnuz", None),
        ),
        start=1,
    )
    if dtype is not None
}
_DEVICE_CODES = {
    "cpu": 1,
    "cuda": 2,
    "xpu": 3,
    "npu": 4,
    "hpu": 5,
    "mps": 6,
    "meta": 7,
    "privateuseone": 8,
}
_LAYOUT_CODES = {
    torch.strided: 1,
    torch.sparse_coo: 2,
    torch.sparse_csr: 3,
    torch.sparse_csc: 4,
    torch.sparse_bsr: 5,
    torch.sparse_bsc: 6,
}


@dataclass(frozen=True)
class _EmbeddingPreflightCache:
    """Static facts proven before the initialization data collective."""

    boundary: bool
    model_ids: tuple[int, ...]
    group: dist.ProcessGroup | None
    global_ranks: tuple[int, int] | None
    world_control_device: torch.device
    pair_control_device: torch.device | None
    pair_backend: str | None
    weight: nn.Parameter | None
    parameter_signature: tuple[object, ...] | None
    native_name: str | None
    base: nn.Module | None


@dataclass(frozen=True)
class _GroupContext:
    boundary: bool
    group: dist.ProcessGroup | None
    global_ranks: tuple[int, int] | None
    world_control_device: torch.device
    pair_control_device: torch.device | None
    pair_backend: str | None


def _unwrap_model(model: nn.Module) -> nn.Module:
    base = model
    seen: set[int] = set()
    while hasattr(base, "module"):
        ident = id(base)
        if ident in seen:
            break
        seen.add(ident)
        candidate = base.module
        if not isinstance(candidate, nn.Module) or candidate is base:
            break
        base = candidate
    return base


def _model_ids(model_chunks: Iterable[nn.Module]) -> tuple[int, ...]:
    return tuple(id(_unwrap_model(chunk)) for chunk in model_chunks)


def _embedding_weight(module: nn.Module | None) -> nn.Parameter | None:
    if module is None:
        return None
    embedding = getattr(module, "embedding", None)
    weight = getattr(embedding, "weight", None)
    return weight if isinstance(weight, nn.Parameter) else None


def _parameter_signature(parameter: nn.Parameter) -> tuple[object, ...]:
    return (
        tuple(parameter.shape),
        parameter.dtype,
        parameter.device,
        parameter.layout,
    )


def _local_shared_embedding_candidates(
    model_chunks: Iterable[nn.Module], ps
) -> list[tuple[nn.Parameter, str, nn.Module]]:
    candidates: list[tuple[nn.Parameter, str, nn.Module]] = []
    for chunk in model_chunks:
        base = _unwrap_model(chunk)
        if ps.pp_is_first:
            for attr in ("embed", "embed_tokens"):
                weight = _embedding_weight(getattr(base, attr, None))
                if weight is not None:
                    candidates.append((weight, f"{attr}.embedding.weight", base))
        if ps.pp_is_last:
            weight = _embedding_weight(getattr(base, "mtp_embed", None))
            if weight is not None:
                candidates.append((weight, "mtp_embed.embedding.weight", base))
    return candidates


def _group_backend_name(group: dist.ProcessGroup | None) -> str:
    # Unit tests exercise the protocol with emulated process groups.  Real
    # callers always initialize torch.distributed before reaching this module.
    if not dist.is_initialized():
        return "gloo"
    backend = str(dist.get_backend(group)).lower()
    return backend.rsplit(".", maxsplit=1)[-1]


def _control_device_for_group(
    group: dist.ProcessGroup | None,
    model_chunks: Iterable[nn.Module] = (),
) -> torch.device:
    """Choose a device supported by the process-group backend, not model data."""
    backend = _group_backend_name(group)
    if backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "NCCL metadata consensus requires an available CUDA device."
            )
        for chunk in model_chunks:
            parameter = next(_unwrap_model(chunk).parameters(), None)
            if parameter is not None and parameter.device.type == "cuda":
                return parameter.device
        return torch.device("cuda", torch.cuda.current_device())
    if backend in {"xccl", "xpu"}:
        xpu = getattr(torch, "xpu", None)
        if xpu is None or not xpu.is_available():
            raise RuntimeError(
                "XCCL metadata consensus requires an available XPU device."
            )
        return torch.device("xpu", xpu.current_device())
    if backend == "hccl":
        npu = getattr(torch, "npu", None)
        if npu is None or not npu.is_available():
            raise RuntimeError(
                "HCCL metadata consensus requires an available NPU device."
            )
        return torch.device("npu", npu.current_device())
    # Gloo and MPI require CPU control tensors. UCC supports CPU tensors and
    # CPU is deterministic across ranks even when model tensors differ.
    if backend in {"gloo", "mpi", "ucc"}:
        return torch.device("cpu")
    raise RuntimeError(f"Unsupported metadata collective backend: {backend!r}.")


def _data_device_supported(backend: str, device_type: str) -> bool:
    if backend == "nccl":
        return device_type == "cuda"
    if backend in {"xccl", "xpu"}:
        return device_type == "xpu"
    if backend in {"gloo", "mpi"}:
        return device_type == "cpu"
    if backend == "hccl":
        return device_type == "npu"
    if backend == "ucc":
        return device_type in {"cpu", "cuda"}
    # Unknown backends are rejected before a data collective. Supporting a new
    # backend must be an explicit compatibility decision.
    return False


def _world_error_consensus(local_error: bool, device: torch.device) -> bool:
    """Make every training rank take the same fail/continue branch."""
    world_error = torch.tensor([int(local_error)], dtype=torch.int32, device=device)
    dist.all_reduce(world_error, group=dist.group.WORLD)
    return bool(world_error.item())


def _process_group_global_ranks(
    group: dist.ProcessGroup, group_size: int
) -> tuple[int, ...]:
    """Read actual membership without issuing a collective."""
    get_ranks = getattr(dist, "get_process_group_ranks", None)
    if get_ranks is not None:
        return tuple(int(rank) for rank in get_ranks(group))
    get_global_rank = getattr(dist, "get_global_rank", None)
    if get_global_rank is None:
        raise RuntimeError("torch.distributed cannot inspect process-group membership.")
    return tuple(int(get_global_rank(group, rank)) for rank in range(group_size))


def _tensor_metadata(
    tensor: torch.Tensor | None,
    *,
    count: int,
    backend: str,
    control_device: torch.device,
    locally_valid: bool = True,
) -> torch.Tensor:
    """Encode presence and tensor facts into a fixed-size int64 record."""
    values = [0] * _METADATA_SIZE
    values[0] = _METADATA_VERSION
    values[1] = count
    if tensor is None:
        values[2] = int(locally_valid and count == 0)
        return torch.tensor(values, dtype=torch.int64, device=control_device)

    shape = tuple(tensor.shape)
    dtype_code = _DTYPE_CODES.get(tensor.dtype, 0)
    device_code = _DEVICE_CODES.get(tensor.device.type, 0)
    layout_code = _LAYOUT_CODES.get(tensor.layout, 0)
    data_compatible = _data_device_supported(backend, tensor.device.type)
    values[2] = int(
        locally_valid
        and count == 1
        and len(shape) <= _MAX_TENSOR_NDIM
        and dtype_code != 0
        and device_code != 0
        and layout_code == _LAYOUT_CODES[torch.strided]
        and data_compatible
    )
    values[3] = len(shape)
    values[4] = dtype_code
    values[5] = device_code
    values[6] = layout_code
    values[7] = int(data_compatible)
    for index, size in enumerate(shape[:_MAX_TENSOR_NDIM], start=8):
        values[index] = size
    return torch.tensor(values, dtype=torch.int64, device=control_device)


def _gather_pair_metadata(
    local_metadata: torch.Tensor, group: dist.ProcessGroup
) -> tuple[torch.Tensor, torch.Tensor]:
    gathered = [torch.empty_like(local_metadata) for _ in range(2)]
    dist.all_gather(gathered, local_metadata, group=group)
    return gathered[0], gathered[1]


def _metadata_pair_status(
    gathered: tuple[torch.Tensor, torch.Tensor], *, allow_absent_pair: bool
) -> tuple[bool, int]:
    """Return pair validity and total presence with one device synchronization."""
    first, last = gathered
    matching_records = torch.all(first == last)
    version_valid = (first[0] == _METADATA_VERSION) & (last[0] == _METADATA_VERSION)
    count_valid = (first[1] == 1) & (last[1] == 1)
    if allow_absent_pair:
        count_valid |= (first[1] == 0) & (last[1] == 0)
    records_valid = (first[2] == 1) & (last[2] == 1)
    success = matching_records & version_valid & count_valid & records_valid
    summary = torch.stack(
        (
            (~success).to(dtype=torch.int64),
            first[1] + last[1],
        )
    )
    pair_error, total_presence = summary.detach().cpu().tolist()
    return bool(pair_error), int(total_presence)


def _group_context(
    model_chunks: list[nn.Module], ps, *, operation: str
) -> _GroupContext:
    boundary = ps.pp_is_first or ps.pp_is_last
    try:
        world_control_device = _control_device_for_group(dist.group.WORLD, model_chunks)
    except (RuntimeError, ValueError, TypeError) as exc:
        raise RuntimeError(
            f"MTP PP embedding {operation} cannot create a WORLD control tensor: {exc}"
        ) from exc

    group = ps.embedding_group if boundary else None
    ranks_value = ps.embedding_global_ranks if boundary else None
    ranks: tuple[int, int] | None = None
    pair_backend: str | None = None
    pair_control_device: torch.device | None = None
    local_error = False
    if boundary:
        local_error = not (ps.pp_is_first ^ ps.pp_is_last)
        if (
            group is None
            or ranks_value is None
            or len(ranks_value) != 2
            or ranks_value[0] == ranks_value[1]
        ):
            local_error = True
        else:
            try:
                ranks = (int(ranks_value[0]), int(ranks_value[1]))
                if ps.pp_global_ranks is not None:
                    if len(ps.pp_global_ranks) < 2:
                        local_error = True
                    elif ps.pp_is_first:
                        local_error |= ranks[0] != ps.pp_global_ranks[0]
                    else:
                        local_error |= ranks[1] != ps.pp_global_ranks[-1]
                if dist.is_initialized():
                    rank = dist.get_rank()
                    expected_rank = ranks[0] if ps.pp_is_first else ranks[1]
                    local_error |= rank != expected_rank
                    group_size = dist.get_world_size(group)
                    local_error |= group_size != 2
                    if group_size == 2:
                        local_error |= (
                            _process_group_global_ranks(group, group_size) != ranks
                        )
            except (IndexError, RuntimeError, ValueError, TypeError):
                local_error = True
        if group is not None:
            try:
                pair_backend = _group_backend_name(group)
                pair_control_device = _control_device_for_group(group, model_chunks)
            except (RuntimeError, ValueError, TypeError):
                local_error = True

    if _world_error_consensus(local_error, world_control_device):
        raise RuntimeError(
            f"MTP PP embedding {operation} group preflight failed on at least one "
            f"rank; local_pp_rank={ps.pp_rank}, embedding_ranks={ranks_value}."
        )
    return _GroupContext(
        boundary=boundary,
        group=group,
        global_ranks=ranks,
        world_control_device=world_control_device,
        pair_control_device=pair_control_device,
        pair_backend=pair_backend,
    )


def _parameter_preflight(
    model_chunks: list[nn.Module], ps, *, operation: str
) -> _EmbeddingPreflightCache:
    """Strictly re-discover and compare both parameter endpoints."""
    context = _group_context(model_chunks, ps, operation=operation)
    candidates: list[tuple[nn.Parameter, str, nn.Module]] = []
    candidate_exception: Exception | None = None
    pair_error = False
    if context.boundary:
        assert context.group is not None
        assert context.pair_backend is not None
        assert context.pair_control_device is not None
        try:
            candidates = _local_shared_embedding_candidates(model_chunks, ps)
        except Exception as exc:  # converted into pair + WORLD consensus below
            candidate_exception = exc
        weight = candidates[0][0] if len(candidates) == 1 else None
        local_metadata = _tensor_metadata(
            weight,
            count=-1 if candidate_exception is not None else len(candidates),
            backend=context.pair_backend,
            control_device=context.pair_control_device,
            locally_valid=candidate_exception is None and len(candidates) == 1,
        )
        gathered = _gather_pair_metadata(local_metadata, context.group)
        pair_error, _ = _metadata_pair_status(gathered, allow_absent_pair=False)

    if _world_error_consensus(pair_error, context.world_control_device):
        names = [name for _, name, _ in candidates]
        local_tensor = candidates[0][0] if len(candidates) == 1 else None
        local_description = (
            "none"
            if local_tensor is None
            else f"shape={tuple(local_tensor.shape)}, dtype={local_tensor.dtype}, "
            f"device={local_tensor.device.type}, layout={local_tensor.layout}"
        )
        raise RuntimeError(
            f"MTP PP embedding {operation} parameter metadata preflight failed on "
            f"at least one pair; local_pp_rank={ps.pp_rank}, "
            f"local_candidates={names}, local_tensor={local_description}, "
            f"local_discovery_error={candidate_exception!r}."
        )

    weight: nn.Parameter | None = None
    native_name: str | None = None
    base: nn.Module | None = None
    if context.boundary:
        weight, native_name, base = candidates[0]
    return _EmbeddingPreflightCache(
        boundary=context.boundary,
        model_ids=_model_ids(model_chunks),
        group=context.group,
        global_ranks=context.global_ranks,
        world_control_device=context.world_control_device,
        pair_control_device=context.pair_control_device,
        pair_backend=context.pair_backend,
        weight=weight,
        parameter_signature=(
            _parameter_signature(weight) if weight is not None else None
        ),
        native_name=native_name,
        base=base,
    )


def _cached_parameter_is_current(
    cache: _EmbeddingPreflightCache, model_chunks: list[nn.Module], ps
) -> bool:
    if _model_ids(model_chunks) != cache.model_ids:
        return False
    if not cache.boundary:
        return not (ps.pp_is_first or ps.pp_is_last)
    if (
        cache.group is not ps.embedding_group
        or cache.global_ranks != tuple(ps.embedding_global_ranks or ())
        or cache.weight is None
        or cache.parameter_signature is None
        or cache.native_name is None
        or cache.base is None
    ):
        return False
    owner_name = cache.native_name.split(".", maxsplit=1)[0]
    return (
        _embedding_weight(getattr(cache.base, owner_name, None)) is cache.weight
        and _parameter_signature(cache.weight) == cache.parameter_signature
    )


def synchronize_mtp_embedding_parameters(
    model_chunks: Iterable[nn.Module], ps, *, enabled: bool
) -> None:
    """Initialize the MTP-stage replica from the first-stage input embedding."""
    if not enabled or ps.pp_size <= 1:
        return
    chunks = list(model_chunks)
    cache = _parameter_preflight(chunks, ps, operation="initialization")
    if not cache.boundary:
        ps.mtp_embedding_preflight_cache = cache
        return

    assert cache.weight is not None
    assert cache.native_name is not None
    assert cache.base is not None
    assert cache.group is not None
    weight, native_name, base = cache.weight, cache.native_name, cache.base
    canonical_name = native_name
    if ps.pp_is_last:
        canonical_name = (
            "embed_tokens.embedding.weight"
            if hasattr(base, "embed_tokens")
            else "embed.embedding.weight"
        )
        # SUM(first, zero-replica) broadcasts the canonical initialization while
        # preserving the same collective on both embedding-group ranks.
        weight.data.zero_()
        weight.shared = True
    weight.shared_embedding = True
    dist.all_reduce(weight.data, group=cache.group)

    # The dist-checkpoint adapter uses this metadata to store the last-stage
    # tensor as a replica of the first-stage logical embedding, not as a second
    # independently named training parameter.
    if ps.pp_is_last:
        base._mlite_tied_checkpoint_keys = {
            "mtp_embed.embedding.weight": canonical_name
        }
    # Cache only after the data collective succeeds. The optimizer hot path can
    # now reuse the proven group/candidate/parameter facts.
    ps.mtp_embedding_preflight_cache = cache


def allreduce_mtp_embedding_grads(
    model_chunks: Iterable[nn.Module], ps, *, enabled: bool
) -> None:
    """Sum first-stage and MTP-stage embedding gradients before optimizer step."""
    if not enabled or ps.pp_size <= 1:
        return
    chunks = list(model_chunks)
    # Validate the active pair on every rank before any endpoint can enter the
    # pair metadata collective.  In particular, a boundary rank may have lost
    # both its cached preflight and ``embedding_group``; raising locally there
    # would strand the other endpoint in all_gather and middle PP ranks in the
    # later WORLD consensus.
    context = _group_context(chunks, ps, operation="gradient")
    cache_value = getattr(ps, "mtp_embedding_preflight_cache", None)
    cache = cache_value if isinstance(cache_value, _EmbeddingPreflightCache) else None
    cache_missing = cache is None
    boundary = context.boundary
    world_control_device = context.world_control_device
    group = context.group
    pair_backend = context.pair_backend
    pair_control_device = context.pair_control_device
    if cache is None:
        # Join the same pair-metadata -> WORLD sequence as initialized peers so
        # an asymmetrically missing cache cannot strand them in a collective.
        weight = None
        native_name = None
        pair_error = True
    else:
        weight = cache.weight
        native_name = cache.native_name
        try:
            pair_error = not _cached_parameter_is_current(cache, chunks, ps)
        except Exception:  # converted into pair + WORLD consensus below
            pair_error = True

    grad: torch.Tensor | None = None
    present = 0
    if boundary:
        assert group is not None
        assert pair_backend is not None
        assert pair_control_device is not None
        if cache is not None:
            assert weight is not None
            grad_value = getattr(weight, "main_grad", None)
            if grad_value is None:
                grad_value = weight.grad
            if grad_value is not None:
                present = 1
                if isinstance(grad_value, torch.Tensor):
                    grad = grad_value
                else:
                    pair_error = True
            locally_valid = not pair_error
            if grad is not None:
                locally_valid &= tuple(grad.shape) == tuple(weight.shape)
            metadata_count = present
        else:
            # A negative count is a fixed-size invalid record. It can still be
            # exchanged safely with a peer that has a valid cache.
            locally_valid = False
            metadata_count = -1
        local_metadata = _tensor_metadata(
            grad,
            count=metadata_count,
            backend=pair_backend,
            control_device=pair_control_device,
            locally_valid=locally_valid,
        )
        gathered = _gather_pair_metadata(local_metadata, group)
        metadata_error, present = _metadata_pair_status(
            gathered, allow_absent_pair=True
        )
        pair_error |= metadata_error

    if _world_error_consensus(pair_error, world_control_device):
        local_name = native_name or f"pp_rank={ps.pp_rank}"
        local_description = (
            "absent"
            if grad is None
            else f"shape={tuple(grad.shape)}, dtype={grad.dtype}, "
            f"device={grad.device.type}, layout={grad.layout}"
        )
        raise RuntimeError(
            "MTP PP shared embedding gradient metadata preflight failed on at "
            f"least one pair; local={local_name}, local_pair_presence={present}/2, "
            f"local_grad={local_description}, local_cache_missing={cache_missing}."
        )
    if not boundary or present == 0:
        # Both copies can be frozen (for example LoRA-only training). With no
        # optimizer update on either side their already-synchronized values
        # remain equal, so no data collective is required.
        return
    assert grad is not None
    assert group is not None
    dist.all_reduce(grad, group=group)


def validate_mtp_embedding_parameter_replicas(
    model_chunks: Iterable[nn.Module], ps, *, enabled: bool
) -> None:
    """Prove exact first/MTP embedding equality before a lossy canonical save."""
    if not enabled or ps.pp_size <= 1:
        return
    chunks = list(model_chunks)
    # Saving deliberately bypasses the optimizer cache: parameters may have
    # been replaced or reshaped since initialization, and a stale cache must
    # never authorize a lossy canonical save.
    cache = _parameter_preflight(chunks, ps, operation="save validation")

    mismatch = False
    local_max_abs = 0.0
    if cache.boundary:
        assert cache.weight is not None
        assert cache.group is not None
        assert cache.global_ranks is not None
        weight = cache.weight
        canonical = weight.detach().clone()
        dist.broadcast(
            canonical,
            src=cache.global_ranks[0],
            group=cache.group,
        )
        mismatch = not torch.equal(canonical, weight.detach())
        if mismatch:
            local_max_abs = (
                (canonical.float() - weight.detach().float()).abs().max().item()
            )
    if _world_error_consensus(mismatch, cache.world_control_device):
        raise RuntimeError(
            "Refusing distributed checkpoint save because a PP MTP embedding "
            f"replica diverged from the canonical parameter; local_pp_rank={ps.pp_rank}, "
            f"local_max_abs={local_max_abs:.6e}."
        )


__all__ = [
    "allreduce_mtp_embedding_grads",
    "synchronize_mtp_embedding_parameters",
    "validate_mtp_embedding_parameter_replicas",
]
