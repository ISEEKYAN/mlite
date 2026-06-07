"""Small Megatron-FSDP patches owned by the M-FSDP primitive."""

from __future__ import annotations

import os
import re
from functools import wraps
from typing import Any, Callable


SKIP_TP_DUPLICATE_SYNC_ATTR = "_mlite_mfsdp_skip_tp_duplicate_sync"
PARAM_NAME_ATTR = "_mlite_mfsdp_param_name"
_PATCHED_ATTR = "_mlite_mfsdp_skip_tp_duplicate_sync_patch"
_ORIGINAL_ATTR = "_mlite_mfsdp_original_make_fsdp_dtensor"
_PARAM_SYNC_PATCHED_ATTR = "_mlite_mfsdp_param_sync_debug_patch"
_START_PARAM_SYNC_PATCHED_ATTR = "_mlite_mfsdp_start_param_sync_patch"


def mark_skip_tp_duplicate_sync(param: Any) -> None:
    setattr(param, SKIP_TP_DUPLICATE_SYNC_ATTR, True)


def clear_skip_tp_duplicate_sync(param: Any) -> None:
    if hasattr(param, SKIP_TP_DUPLICATE_SYNC_ATTR):
        delattr(param, SKIP_TP_DUPLICATE_SYNC_ATTR)


def set_mfsdp_param_name(param: Any, name: str) -> None:
    setattr(param, PARAM_NAME_ATTR, name)


def should_skip_tp_duplicate_sync(param: Any) -> bool:
    """Return whether MCore must not TP-broadcast this Megatron Lite-owned local shard."""
    if not bool(getattr(param, SKIP_TP_DUPLICATE_SYNC_ATTR, False)):
        return False
    return getattr(param, "_tensor_parallel_mode", None) not in ("column", "row")


def _wrap_make_fsdp_dtensor(make_fsdp_dtensor: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(make_fsdp_dtensor)
    def wrapped_make_fsdp_dtensor(*args, **kwargs):
        local_tensor = kwargs.get("local_tensor")
        if local_tensor is None and len(args) >= 1:
            local_tensor = args[0]
        param = kwargs.get("param")
        if param is None and len(args) >= 2:
            param = args[1]
        force_in = kwargs.get("force_sync_tp_duplicated_param")
        if param is not None and should_skip_tp_duplicate_sync(param):
            kwargs["force_sync_tp_duplicated_param"] = False
            kwargs["run_check"] = False
        if param is not None:
            _print_dtensor_debug(
                param,
                local_tensor,
                None,
                phase="begin",
                force_in=force_in,
                force_out=kwargs.get("force_sync_tp_duplicated_param"),
                is_sharded_param=kwargs.get("is_sharded_param"),
                is_expert_param=kwargs.get("is_expert_param"),
            )
        result = make_fsdp_dtensor(*args, **kwargs)
        if param is not None:
            _print_dtensor_debug(
                param,
                local_tensor,
                result,
                phase="end",
                force_in=force_in,
                force_out=kwargs.get("force_sync_tp_duplicated_param"),
                is_sharded_param=kwargs.get("is_sharded_param"),
                is_expert_param=kwargs.get("is_expert_param"),
            )
        return result

    setattr(wrapped_make_fsdp_dtensor, _PATCHED_ATTR, True)
    setattr(wrapped_make_fsdp_dtensor, _ORIGINAL_ATTR, make_fsdp_dtensor)
    return wrapped_make_fsdp_dtensor


def install_mfsdp_tp_duplicate_sync_patch() -> None:
    """Guard MCore duplicate-TP sync for marked Megatron Lite local TP shards.

    Megatron Lite modules keep their local TP shards as normal tensors. Current
    MCore-FSDP uses ``tensor_model_parallel`` plus ``partition_dim`` to choose
    TP-sharded DTensor placement, but some duplicate-TP sync paths can still be
    forced for params marked as duplicated. The patch is intentionally narrow:
    it only changes that sync decision for params marked by this primitive and
    leaves MCore DTensor placement untouched.
    """
    from megatron.core.distributed.fsdp.src.megatron_fsdp import (  # pyright: ignore[reportMissingImports]
        param_and_grad_buffer,
    )

    make_fsdp_dtensor = param_and_grad_buffer.make_fsdp_dtensor
    install_mfsdp_start_param_sync_patch()
    if bool(getattr(make_fsdp_dtensor, _PATCHED_ATTR, False)):
        install_mfsdp_param_sync_debug_patch(param_and_grad_buffer)
        return
    param_and_grad_buffer.make_fsdp_dtensor = _wrap_make_fsdp_dtensor(make_fsdp_dtensor)
    install_mfsdp_param_sync_debug_patch(param_and_grad_buffer)


def install_mfsdp_start_param_sync_patch() -> None:
    from megatron.core.distributed.fsdp.src.megatron_fsdp import (  # pyright: ignore[reportMissingImports]
        megatron_fsdp,
    )

    fsdp_cls = megatron_fsdp.MegatronFSDP
    start_param_sync = fsdp_cls.start_param_sync
    if bool(getattr(start_param_sync, _START_PARAM_SYNC_PATCHED_ATTR, False)):
        return

    @wraps(start_param_sync)
    def wrapped_start_param_sync(self, *args, **kwargs):
        if os.environ.get("MLITE_MFSDP_SKIP_START_PARAM_SYNC_ACTIVE", "0") == "1":
            if _debug_param_sync():
                try:
                    import torch.distributed as dist  # pyright: ignore[reportMissingImports]

                    rank = dist.get_rank() if dist.is_initialized() else 0
                except Exception:
                    rank = 0
                print(
                    f"[MFSDP_PARAM_SYNC] phase=start_param_sync_noop rank={rank}",
                    flush=True,
                )
            return None
        return start_param_sync(self, *args, **kwargs)

    setattr(wrapped_start_param_sync, _START_PARAM_SYNC_PATCHED_ATTR, True)
    fsdp_cls.start_param_sync = wrapped_start_param_sync


def install_mfsdp_param_sync_debug_patch(param_and_grad_buffer: Any) -> None:
    """Print MCore FSDP param all-gather bucket state when explicitly requested."""
    pipeline_cls = getattr(param_and_grad_buffer, "AllGatherPipeline", None)
    if pipeline_cls is None or bool(getattr(pipeline_cls, _PARAM_SYNC_PATCHED_ATTR, False)):
        return

    original_async = pipeline_cls.async_bucket_gather
    original_wait = pipeline_cls.wait_bucket_ready

    @wraps(original_async)
    def wrapped_async(self, bucket_id, bwd, *args, **kwargs):
        _print_param_sync_debug(self, "async_begin", bucket_id, bwd)
        result = original_async(self, bucket_id, bwd, *args, **kwargs)
        _print_param_sync_debug(self, "async_end", bucket_id, bwd)
        return result

    @wraps(original_wait)
    def wrapped_wait(self, bucket_id, bwd, *args, **kwargs):
        _print_param_sync_debug(self, "wait_begin", bucket_id, bwd)
        result = original_wait(self, bucket_id, bwd, *args, **kwargs)
        _print_param_sync_debug(self, "wait_end", bucket_id, bwd)
        return result

    pipeline_cls.async_bucket_gather = wrapped_async
    pipeline_cls.wait_bucket_ready = wrapped_wait
    setattr(pipeline_cls, _PARAM_SYNC_PATCHED_ATTR, True)


def _debug_dtensor() -> bool:
    return os.environ.get("MLITE_MFSDP_DEBUG_DTENSOR", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _debug_param_sync() -> bool:
    return os.environ.get("MLITE_MFSDP_DEBUG_PARAM_SYNC", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _debug_dtensor_ranks() -> set[int]:
    raw = os.environ.get("MLITE_MFSDP_DEBUG_DTENSOR_RANKS", "0,1")
    ranks: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            ranks.add(int(item))
        except ValueError:
            continue
    return ranks or {0}


def _debug_param_sync_ranks() -> set[int]:
    raw = os.environ.get("MLITE_MFSDP_DEBUG_PARAM_SYNC_RANKS", "0,1")
    ranks: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            ranks.add(int(item))
        except ValueError:
            continue
    return ranks or {0}


def _debug_dtensor_name_pattern() -> re.Pattern[str]:
    raw = os.environ.get(
        "MLITE_MFSDP_DEBUG_DTENSOR_NAMES",
        r"layers\.0\..*(qkv|linear_qkv|proj|linear_proj).*weight$",
    )
    return re.compile(raw)


def _print_dtensor_debug(
    param: Any,
    local_tensor: Any,
    result: Any,
    *,
    phase: str,
    force_in: Any,
    force_out: Any,
    is_sharded_param: Any,
    is_expert_param: Any,
) -> None:
    if not _debug_dtensor():
        return
    try:
        import torch.distributed as dist  # pyright: ignore[reportMissingImports]

        rank = dist.get_rank() if dist.is_initialized() else 0
    except Exception:
        rank = 0
    if rank not in _debug_dtensor_ranks():
        return
    name = str(getattr(param, PARAM_NAME_ATTR, ""))
    if not _debug_dtensor_name_pattern().search(name):
        return
    print(
        "[MFSDP_DTENSOR] "
        f"phase={phase} "
        f"rank={rank} "
        f"name={name or '<unknown>'} "
        f"tp={int(bool(getattr(param, 'tensor_model_parallel', False)))} "
        f"partition_dim={getattr(param, 'partition_dim', None)} "
        f"tp_mode={getattr(param, '_tensor_parallel_mode', None)} "
        f"skip_sync={int(should_skip_tp_duplicate_sync(param))} "
        f"force_in={force_in} "
        f"force_out={force_out} "
        f"is_sharded_param={is_sharded_param} "
        f"is_expert_param={is_expert_param} "
        f"param_shape={_shape_tuple(param)} "
        f"local_shape={_shape_tuple(local_tensor)} "
        f"result_shape={_shape_tuple(result)} "
        f"result_placements={getattr(result, 'placements', None)}",
        flush=True,
    )


def _print_param_sync_debug(pipeline: Any, phase: str, bucket_id: Any, bwd: Any) -> None:
    if not _debug_param_sync():
        return
    try:
        import torch.distributed as dist  # pyright: ignore[reportMissingImports]

        rank = dist.get_rank() if dist.is_initialized() else 0
    except Exception:
        dist = None
        rank = 0
    if rank not in _debug_param_sync_ranks():
        return

    wbuf = None
    group = None
    try:
        wbuf = pipeline.get_fsdp_buffer(bucket_id, bwd)
        group = getattr(wbuf, "data_parallel_group", None)
    except Exception:
        pass
    try:
        ranks = dist.get_process_group_ranks(group) if dist is not None and group is not None else None
    except Exception:
        ranks = None
    try:
        shard = wbuf.get_shard_from_local_buffer() if wbuf is not None else None
    except Exception:
        shard = None
    try:
        bucket_key = pipeline.get_bucket_key(bucket_id, bwd)
        status = pipeline.bucket_status.get(bucket_key)
    except Exception:
        bucket_key = None
        status = None

    print(
        "[MFSDP_PARAM_SYNC] "
        f"phase={phase} "
        f"rank={rank} "
        f"bucket_id={bucket_id} "
        f"bwd={bwd} "
        f"bucket_key={bucket_key} "
        f"status={status} "
        f"group_ranks={ranks} "
        f"buffer_numel={_numel(getattr(wbuf, 'data', None))} "
        f"shard_numel={_numel(shard)} "
        f"bucket_status={getattr(pipeline, 'bucket_status', None)}",
        flush=True,
    )


def _numel(value: Any) -> int | None:
    if value is None:
        return None
    numel = getattr(value, "numel", None)
    if numel is None:
        return None
    try:
        return int(numel())
    except TypeError:
        return None


def _shape_tuple(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(dim) for dim in shape)
    except TypeError:
        return None


__all__ = [
    "PARAM_NAME_ATTR",
    "SKIP_TP_DUPLICATE_SYNC_ATTR",
    "clear_skip_tp_duplicate_sync",
    "install_mfsdp_tp_duplicate_sync_patch",
    "mark_skip_tp_duplicate_sync",
    "set_mfsdp_param_name",
    "should_skip_tp_duplicate_sync",
]
