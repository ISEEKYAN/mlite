# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Megatron Core distributed checkpoint bridge for MLite dist_opt."""

from __future__ import annotations

import hashlib
import os
import struct
import threading
from collections.abc import Callable, Iterable, Mapping, MutableMapping
from dataclasses import replace
from types import MethodType
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.mapping import (
    LocalNonpersistentObject,
    ShardedBase,
    ShardedTensor,
    ShardedTensorFactory,
)
from megatron.core.dist_checkpointing.validation import StrictHandling
from megatron.lite.primitive.ckpt.hf_weights import (
    _distributed_raise_if_error,
    named_persistent_buffers,
)
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.protocols import (
    ExpertClassifierFn,
    PlacementFn,
    default_expert_classifier,
    default_placement_fn,
)

_DISTOPT_METADATA = {
    "distrib_optim_sharding_type": "fully_reshardable",
    "distrib_optim_fully_reshardable_mem_efficient": False,
    "chained_optim_avoid_prefix": True,
}
_DISTCKPT_COMMON_LOAD_LOCK = threading.RLock()


def attach_model_sharded_state_dict(
    model_chunks: Iterable[nn.Module],
    ps: ParallelState,
    *,
    get_placements: PlacementFn = default_placement_fn,
    is_expert: ExpertClassifierFn = default_expert_classifier,
) -> None:
    """Attach an MLite-local mcore sharded_state_dict method to dist_opt chunks."""

    for chunk in model_chunks:
        chunk.sharded_state_dict = MethodType(  # type: ignore[method-assign]
            _build_bound_sharded_state_dict(ps, get_placements, is_expert), chunk
        )
        chunk._mlite_dist_opt_sharded_state_dict = True  # type: ignore[attr-defined]
        chunk._mlite_dist_opt_parallel_state = ps  # type: ignore[attr-defined]


def supports_dist_opt_distckpt(
    model: nn.Module | Iterable[nn.Module], optimizer: Any
) -> bool:
    """Return whether this model/optimizer pair can use mcore dist_checkpointing."""

    if optimizer is not None and not callable(
        getattr(optimizer, "sharded_state_dict", None)
    ):
        return False
    return all(
        bool(getattr(chunk, "_mlite_dist_opt_sharded_state_dict", False))
        and callable(getattr(chunk, "sharded_state_dict", None))
        for chunk in _model_chunks(model)
    )


def save_dist_opt_checkpoint(
    model: nn.Module | Iterable[nn.Module],
    optimizer: Any,
    step: int,
    checkpoint_dir: str,
    *,
    save_model: bool = True,
    save_optimizer: bool = True,
) -> None:
    """Save model and DistributedOptimizer state through mcore dist_checkpointing."""

    if type(step) is not int or step < 0:
        raise TypeError("checkpoint step must be a non-negative integer")
    if save_optimizer and optimizer is None:
        raise ValueError("save_optimizer=True requires a non-None sharded optimizer")
    local_error = None
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _distributed_raise_if_error(
        local_error, context="distckpt checkpoint directory creation failed"
    )
    chunks = _model_chunks(model)
    ps = _chunk_parallel_state(chunks[0]) if chunks else None
    if ps is not None and ps.embedding_groups_initialized:
        from megatron.lite.primitive.parallel import (
            validate_mtp_embedding_parameter_replicas,
        )

        validate_mtp_embedding_parameter_replicas(chunks, ps, enabled=True)
    model_sd: dict[str, Any] = {}
    local_error = None
    if save_model or save_optimizer:
        try:
            model_sd = _model_sharded_state_dict(model)
            _validate_model_sharded_key_namespace(model_sd)
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
    _distributed_raise_if_error(
        local_error, context="distckpt model state construction failed"
    )
    state_dict: dict[str, Any] = {"step": int(step)}
    if save_model:
        state_dict.update(model_sd)
    local_error = None
    if save_optimizer and optimizer is not None:
        try:
            _synchronize_native_optimizer_steps(optimizer)
            patches = _patch_empty_native_optimizer_state_dicts(
                optimizer, fallback_step=step
            )
            try:
                state_dict["optimizer"] = optimizer.sharded_state_dict(
                    _single_or_all_model_state(model_sd), metadata=_DISTOPT_METADATA
                )
                _validate_optimizer_sharded_key_namespace(
                    state_dict["optimizer"], model_sd
                )
            finally:
                _restore_state_dict_patches(patches)
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
    _distributed_raise_if_error(
        local_error, context="distckpt optimizer state construction failed"
    )
    local_error = None
    try:
        dist_checkpointing.save(
            state_dict,
            checkpoint_dir,
            validate_access_integrity=False,
            content_metadata=_DISTOPT_METADATA,
        )
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _distributed_raise_if_error(local_error, context="distckpt checkpoint save failed")


def load_dist_opt_checkpoint(
    model: nn.Module | Iterable[nn.Module],
    optimizer: Any,
    checkpoint_dir: str,
    *,
    load_model: bool = True,
    load_optimizer: bool = True,
    expected_step: int | None = None,
    allow_legacy_checkpoint: bool = False,
) -> int:
    """Load a mcore dist_checkpointing checkpoint into model and DistributedOptimizer."""

    if load_optimizer and optimizer is None:
        raise ValueError("load_optimizer=True requires a non-None sharded optimizer")
    model_sd: dict[str, Any] = {}
    local_error = None
    if load_model or load_optimizer:
        try:
            model_sd = _model_sharded_state_dict(model)
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
    _distributed_raise_if_error(
        local_error, context="distckpt model state construction failed"
    )
    load_sd: dict[str, Any] = {"step": 0}
    if load_model:
        load_sd.update(model_sd)
    local_error = None
    if load_optimizer and optimizer is not None:
        try:
            patches = _patch_empty_native_optimizer_state_dicts(
                optimizer, fallback_step=0
            )
            try:
                load_sd["optimizer"] = optimizer.sharded_state_dict(
                    _single_or_all_model_state(model_sd),
                    is_loading=True,
                    metadata=_DISTOPT_METADATA,
                )
                _validate_optimizer_sharded_key_namespace(
                    load_sd["optimizer"], model_sd
                )
            finally:
                _restore_state_dict_patches(patches)
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
    _distributed_raise_if_error(
        local_error, context="distckpt optimizer state construction failed"
    )
    requested_keys, checkpoint_keys = _preflight_distckpt_checkpoint_metadata(
        load_sd,
        model_sd,
        checkpoint_dir,
        load_model=load_model,
        load_optimizer=load_optimizer,
    )
    preloaded_common_state, common_state_fingerprint = _preflight_distckpt_common_state(
        load_sd,
        checkpoint_dir,
        load_optimizer=load_optimizer,
        requested_sharded_keys=requested_keys,
        checkpoint_sharded_keys=checkpoint_keys,
        expected_step=expected_step,
        allow_legacy_checkpoint=allow_legacy_checkpoint,
    )
    _revalidate_distckpt_common_state_file(checkpoint_dir, common_state_fingerprint)
    state_dict: dict[str, Any] = {}
    local_error = None
    try:
        state_dict = _load_distckpt_with_preloaded_common(
            load_sd,
            checkpoint_dir,
            preloaded_common_state,
            validate_access_integrity=False,
            strict=StrictHandling.RAISE_UNEXPECTED,
        )
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _distributed_raise_if_error(local_error, context="distckpt checkpoint load failed")
    local_error = None
    loaded_step = int(state_dict.get("step", 0))
    if expected_step is not None and loaded_step != expected_step:
        local_error = (
            f"checkpoint payload step {loaded_step} does not match completion manifest "
            f"step {expected_step}"
        )
    _distributed_raise_if_error(
        local_error, context="distckpt checkpoint step validation failed"
    )
    local_error = None
    if load_optimizer and optimizer is not None and "optimizer" not in state_dict:
        local_error = (
            "checkpoint is missing optimizer state requested by load_optimizer=True"
        )
    _distributed_raise_if_error(
        local_error, context="distckpt checkpoint completeness validation failed"
    )
    local_error = None
    if load_model:
        try:
            _load_model_state_dict(model, state_dict, expected_state_dict=model_sd)
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
    _distributed_raise_if_error(
        local_error, context="distckpt model state application failed"
    )
    local_error = None
    try:
        if load_optimizer and optimizer is not None:
            if "optimizer" not in state_dict:
                raise RuntimeError(
                    "checkpoint is missing optimizer state requested by load_optimizer=True"
                )
            load_patches = _patch_native_optimizer_step_load(optimizer)
            try:
                optimizer.load_state_dict(state_dict["optimizer"])
            finally:
                _restore_set_state_patches(load_patches)
            _synchronize_native_optimizer_steps(optimizer)
        elif load_model and optimizer is not None:
            reload_model_params = getattr(optimizer, "reload_model_params", None)
            if callable(reload_model_params):
                reload_model_params()
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _distributed_raise_if_error(
        local_error, context="distckpt optimizer state application failed"
    )
    return loaded_step


def _iter_sharded_bases(value: Any) -> Iterable[ShardedBase]:
    """Yield effective sharded entries, expanding factories without mutation."""

    if isinstance(value, ShardedTensorFactory):
        yield from _iter_sharded_bases(value.build())
    elif isinstance(value, ShardedBase):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _iter_sharded_bases(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _iter_sharded_bases(child)


def _sharded_logical_keys(value: Any) -> set[str]:
    return set(_sharded_metadata_contracts(value))


def _sharded_metadata_contracts(
    value: Any,
) -> dict[str, tuple[str, tuple[int, ...], str | None, bool]]:
    contracts: dict[str, tuple[str, tuple[int, ...], str | None, bool]] = {}
    for sharded_base in _iter_sharded_bases(value):
        key = getattr(sharded_base, "key", None)
        if not isinstance(key, str) or not key:
            raise RuntimeError(
                f"distckpt sharded metadata contains an invalid logical key: {key!r}"
            )
        global_shape = getattr(sharded_base, "global_shape", None)
        if not isinstance(global_shape, tuple) or not all(
            type(dim) is int for dim in global_shape
        ):
            raise RuntimeError(
                f"distckpt sharded metadata for {key!r} has invalid global_shape "
                f"{global_shape!r}"
            )
        if isinstance(sharded_base, ShardedTensor):
            contract = (
                "tensor",
                global_shape,
                str(sharded_base.dtype),
                bool(sharded_base.allow_shape_mismatch),
            )
        else:
            contract = ("object", global_shape, None, False)
        previous = contracts.get(key)
        if previous is not None and previous != contract:
            raise RuntimeError(
                f"distckpt logical key {key!r} has inconsistent metadata contracts: "
                f"{previous!r} vs {contract!r}"
            )
        contracts[key] = contract
    return contracts


def _validate_model_sharded_key_namespace(model_sd: Mapping[str, Any]) -> None:
    reserved_model_keys = sorted(
        key for key in _sharded_logical_keys(model_sd) if key.startswith("optimizer.")
    )
    if reserved_model_keys:
        raise RuntimeError(
            "model checkpoint keys use the reserved 'optimizer.' prefix: "
            f"{reserved_model_keys}"
        )


def _validate_optimizer_sharded_key_namespace(
    optimizer_sd: Any, model_sd: Mapping[str, Any]
) -> None:
    optimizer_keys = _sharded_logical_keys(optimizer_sd)
    invalid_optimizer_keys = sorted(
        key for key in optimizer_keys if not key.startswith("optimizer.")
    )
    if invalid_optimizer_keys:
        raise RuntimeError(
            "optimizer checkpoint keys must use the reserved 'optimizer.' prefix: "
            f"{invalid_optimizer_keys}"
        )
    overlapping_keys = sorted(optimizer_keys & _sharded_logical_keys(model_sd))
    if overlapping_keys:
        raise RuntimeError(
            f"model and optimizer checkpoint keys overlap: {overlapping_keys}"
        )


def _load_checkpoint_sharded_metadata(checkpoint_dir: str) -> Mapping[str, Any]:
    # ``load_sharded_metadata`` includes both tensors and ShardedObjects but is
    # not re-exported from megatron.core.dist_checkpointing.
    from megatron.core.dist_checkpointing.serialization import load_sharded_metadata

    metadata = load_sharded_metadata(checkpoint_dir)
    if not isinstance(metadata, Mapping):
        raise TypeError(
            "MCore load_sharded_metadata returned "
            f"{type(metadata).__name__}, expected a mapping"
        )
    return metadata


def _load_checkpoint_common_state(checkpoint_dir: str) -> Mapping[str, Any]:
    common_state = dist_checkpointing.load_common_state_dict(checkpoint_dir)
    if not isinstance(common_state, Mapping):
        raise TypeError(
            "MCore load_common_state_dict returned "
            f"{type(common_state).__name__}, expected a mapping"
        )
    return common_state


def _common_state_contracts(
    value: Any, path: tuple[str, ...] = ()
) -> dict[tuple[str, ...], tuple[str, tuple[int, ...] | None, str | None]]:
    if isinstance(value, (ShardedBase, LocalNonpersistentObject)):
        return {}

    def type_name(item: Any) -> str:
        item_type = type(item)
        return f"{item_type.__module__}.{item_type.__qualname__}"

    if isinstance(value, Mapping):
        if not value:
            return {path + ("<empty>",): (type_name(value), None, None)}
        contracts = {}
        for key, child in value.items():
            segment = f"{type_name(key)}:{key!r}"
            contracts.update(_common_state_contracts(child, path + (segment,)))
        if contracts:
            contracts[path + ("<container>",)] = (type_name(value), None, None)
        return contracts
    if isinstance(value, (list, tuple)):
        if not value:
            return {path + ("<empty>",): (type_name(value), None, None)}
        contracts = {}
        for index, child in enumerate(value):
            contracts.update(_common_state_contracts(child, path + (f"index:{index}",)))
        if contracts:
            contracts[path + ("<container>",)] = (type_name(value), None, None)
        return contracts
    if isinstance(value, torch.Tensor):
        return {path: (type_name(value), tuple(value.shape), str(value.dtype))}
    return {path: (type_name(value), None, None)}


def _common_state_file_fingerprint(checkpoint_dir: str) -> tuple[int, str]:
    """Hash the exact MCore common-state payload without reserializing it."""

    from megatron.core.msc_utils import MultiStorageClientFeature

    path = os.path.join(checkpoint_dir, "common.pt")
    if MultiStorageClientFeature.is_enabled():
        msc = MultiStorageClientFeature.import_package()
        open_file = msc.open
    else:
        open_file = open
    digest = hashlib.sha256()
    size = 0
    with open_file(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def _common_state_semantic_fingerprint(value: Any) -> str:
    """Hash common-state values using a deterministic, pickle-free encoding."""

    digest = hashlib.sha256()
    active_containers: set[int] = set()

    def type_name(item: Any) -> str:
        item_type = type(item)
        return f"{item_type.__module__}.{item_type.__qualname__}"

    def frame_bytes(tag: bytes, payload: bytes) -> bytes:
        return (
            len(tag).to_bytes(4, "big")
            + tag
            + len(payload).to_bytes(8, "big")
            + payload
        )

    def stable_key_bytes(item: Any, path: str) -> bytes:
        item_type = type_name(item).encode("utf-8")
        if item is None:
            payload = b""
        elif isinstance(item, bool):
            payload = b"1" if item else b"0"
        elif isinstance(item, int):
            payload = str(item).encode("ascii")
        elif isinstance(item, float):
            payload = struct.pack(">d", item)
        elif isinstance(item, complex):
            payload = struct.pack(">dd", item.real, item.imag)
        elif isinstance(item, str):
            payload = item.encode("utf-8")
        elif isinstance(item, bytes):
            payload = item
        elif isinstance(item, tuple):
            container_id = id(item)
            if container_id in active_containers:
                raise ValueError(f"common-state cycle detected at {path}")
            active_containers.add(container_id)
            try:
                children = b"".join(
                    frame_bytes(b"item", stable_key_bytes(child, f"{path}[{index}]"))
                    for index, child in enumerate(item)
                )
                payload = frame_bytes(
                    b"length", str(len(item)).encode("ascii")
                ) + frame_bytes(b"children", children)
            finally:
                active_containers.remove(container_id)
        else:
            raise TypeError(
                f"unsupported common-state mapping key at {path}: {type_name(item)}"
            )
        return frame_bytes(b"type", item_type) + frame_bytes(b"value", payload)

    def update(tag: bytes, payload: bytes | memoryview = b"") -> None:
        digest.update(len(tag).to_bytes(4, "big"))
        digest.update(tag)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    def visit(item: Any, path: str) -> None:
        if isinstance(item, (ShardedBase, LocalNonpersistentObject)):
            raise TypeError(
                f"sharded/nonpersistent object leaked into common state at {path}: "
                f"{type_name(item)}"
            )
        update(b"type", type_name(item).encode("utf-8"))
        if item is None:
            update(b"none")
        elif isinstance(item, bool):
            update(b"bool", b"1" if item else b"0")
        elif isinstance(item, int):
            update(b"int", str(item).encode("ascii"))
        elif isinstance(item, float):
            update(b"float64", struct.pack(">d", item))
        elif isinstance(item, complex):
            update(b"complex128", struct.pack(">dd", item.real, item.imag))
        elif isinstance(item, str):
            update(b"str", item.encode("utf-8"))
        elif isinstance(item, bytes):
            update(b"bytes", item)
        elif isinstance(item, (bytearray, memoryview)):
            update(b"bytes-like", memoryview(item))
        elif isinstance(item, torch.Tensor):
            if item.layout != torch.strided or item.is_quantized:
                raise TypeError(
                    f"unsupported common-state tensor at {path}: "
                    f"layout={item.layout}, quantized={item.is_quantized}"
                )
            if item.device.type == "meta":
                raise TypeError(f"meta tensor is invalid in common state at {path}")
            cpu = item.detach().resolve_conj().resolve_neg().cpu().contiguous()
            update(b"tensor.dtype", str(cpu.dtype).encode("ascii"))
            update(
                b"tensor.shape",
                b",".join(str(dim).encode("ascii") for dim in cpu.shape),
            )
            raw = cpu.reshape(-1).view(torch.uint8).numpy()
            update(b"tensor.bytes", memoryview(raw))
        elif isinstance(item, Mapping):
            container_id = id(item)
            if container_id in active_containers:
                raise ValueError(f"common-state cycle detected at {path}")
            active_containers.add(container_id)
            try:
                encoded_items = sorted(
                    [
                        (stable_key_bytes(key, f"{path}.<key>"), key, child)
                        for key, child in item.items()
                    ],
                    key=lambda entry: entry[0],
                )
                for previous, current in zip(
                    encoded_items, encoded_items[1:], strict=False
                ):
                    if previous[0] == current[0]:
                        raise ValueError(
                            f"common-state mapping has duplicate canonical keys at {path}"
                        )
                update(b"mapping.length", str(len(encoded_items)).encode("ascii"))
                for key_bytes, key, child in encoded_items:
                    update(b"mapping.key", key_bytes)
                    visit(child, f"{path}[{key!r}]")
                update(b"mapping.end")
            finally:
                active_containers.remove(container_id)
        elif isinstance(item, (list, tuple)):
            container_id = id(item)
            if container_id in active_containers:
                raise ValueError(f"common-state cycle detected at {path}")
            active_containers.add(container_id)
            try:
                update(b"sequence.length", str(len(item)).encode("ascii"))
                for index, child in enumerate(item):
                    update(b"sequence.index", str(index).encode("ascii"))
                    visit(child, f"{path}[{index}]")
                update(b"sequence.end")
            finally:
                active_containers.remove(container_id)
        elif isinstance(item, (set, frozenset)):
            container_id = id(item)
            if container_id in active_containers:
                raise ValueError(f"common-state cycle detected at {path}")
            active_containers.add(container_id)
            try:
                encoded = sorted(
                    stable_key_bytes(child, f"{path}.<set-item>") for child in item
                )
                update(b"set.length", str(len(encoded)).encode("ascii"))
                for child in encoded:
                    update(b"set.item", child)
                update(b"set.end")
            finally:
                active_containers.remove(container_id)
        elif isinstance(item, torch.dtype):
            update(b"torch.dtype", str(item).encode("ascii"))
        elif isinstance(item, torch.device):
            update(b"torch.device", str(item).encode("ascii"))
        else:
            raise TypeError(
                f"unsupported common-state value at {path}: {type_name(item)}"
            )

    visit(value, "root")
    return digest.hexdigest()


def _validate_distckpt_content_metadata(
    common_state: Mapping[str, Any],
    *,
    load_optimizer: bool,
    allow_legacy_checkpoint: bool,
) -> None:
    """Require the payload format used to build the live load template."""

    if allow_legacy_checkpoint and not load_optimizer:
        return
    if "content_metadata" not in common_state:
        legacy_detail = (
            "optimizer restore cannot safely infer its sharding format"
            if load_optimizer
            else "set allow_legacy_checkpoint=True only for an explicit legacy load"
        )
        raise RuntimeError(
            "checkpoint common state is missing required content_metadata; "
            f"{legacy_detail}"
        )
    metadata = common_state["content_metadata"]
    if type(metadata) is not dict:
        raise RuntimeError(
            "checkpoint content_metadata must be a plain dict, got "
            f"{type(metadata).__module__}.{type(metadata).__qualname__}"
        )
    missing = sorted(set(_DISTOPT_METADATA) - set(metadata))
    unexpected = sorted(set(metadata) - set(_DISTOPT_METADATA))
    mismatched = sorted(
        key
        for key in set(metadata) & set(_DISTOPT_METADATA)
        if type(metadata[key]) is not type(_DISTOPT_METADATA[key])
        or metadata[key] != _DISTOPT_METADATA[key]
    )
    if missing or unexpected or mismatched:
        raise RuntimeError(
            "checkpoint content_metadata is incompatible with the MLite distckpt "
            f"format: missing={missing}, unexpected={unexpected}, "
            f"mismatched={mismatched}"
        )


def _common_state_format_discriminators(
    value: Any, path: tuple[str, ...] = ()
) -> dict[tuple[str, ...], tuple[str, Any]]:
    """Collect exact common-state fields that select optimizer load branches."""

    if isinstance(value, (ShardedBase, LocalNonpersistentObject)):
        return {}

    def type_name(item: Any) -> str:
        item_type = type(item)
        return f"{item_type.__module__}.{item_type.__qualname__}"

    discriminators: dict[tuple[str, ...], tuple[str, Any]] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            segment = f"{type_name(key)}:{key!r}"
            child_path = path + (segment,)
            if key == "param_state_sharding_type":
                discriminators[child_path] = (type_name(child), child)
            discriminators.update(
                _common_state_format_discriminators(child, child_path)
            )
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            discriminators.update(
                _common_state_format_discriminators(child, path + (f"index:{index}",))
            )
    return discriminators


def _assert_world_consensus(value: Any, *, context: str) -> None:
    """Require every WORLD rank to report the same picklable metadata value."""

    if not dist.is_available() or not dist.is_initialized():
        return
    gathered: list[Any] = [None] * dist.get_world_size()
    local_error = None
    try:
        dist.all_gather_object(gathered, value, group=dist.group.WORLD)
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _distributed_raise_if_error(local_error, context=f"{context} exchange failed")
    if any(item != gathered[0] for item in gathered[1:]):
        raise RuntimeError(f"{context}: rank values={gathered!r}")


def _preflight_distckpt_common_state(
    load_sd: Mapping[str, Any],
    checkpoint_dir: str,
    *,
    load_optimizer: bool,
    requested_sharded_keys: set[str],
    checkpoint_sharded_keys: set[str],
    expected_step: int | None,
    allow_legacy_checkpoint: bool,
) -> tuple[dict[str, Any], tuple[int, str]]:
    common_state: Mapping[str, Any] = {}
    common_contracts: dict = {}
    optimizer_contracts: dict = {}
    common_state_fingerprint: tuple[int, str] | None = None
    common_state_semantic_fingerprint: str | None = None
    loaded_step: int | None = None
    local_error = None
    try:
        before_fingerprint = _common_state_file_fingerprint(checkpoint_dir)
        common_state = _load_checkpoint_common_state(checkpoint_dir)
        common_state_fingerprint = _common_state_file_fingerprint(checkpoint_dir)
        if common_state_fingerprint != before_fingerprint:
            raise RuntimeError(
                "checkpoint common.pt changed while it was being preflighted: "
                f"before={before_fingerprint!r}, after={common_state_fingerprint!r}"
            )
        _validate_distckpt_content_metadata(
            common_state,
            load_optimizer=load_optimizer,
            allow_legacy_checkpoint=allow_legacy_checkpoint,
        )
        common_state_semantic_fingerprint = _common_state_semantic_fingerprint(
            common_state
        )
        common_contracts = _common_state_contracts(common_state)
        raw_step = common_state.get("step")
        if type(raw_step) is not int or raw_step < 0:
            raise RuntimeError(
                f"checkpoint common step must be a non-negative integer, got {raw_step!r}"
            )
        loaded_step = raw_step
        if expected_step is not None and loaded_step != expected_step:
            raise RuntimeError(
                f"checkpoint common step {loaded_step} does not match completion "
                f"manifest step {expected_step}"
            )
        if load_optimizer:
            expected_optimizer_contracts = _common_state_contracts(
                load_sd.get("optimizer", {})
            )
            optimizer_contracts = _common_state_contracts(
                common_state["optimizer"] if "optimizer" in common_state else {}
            )
            expected_discriminators = _common_state_format_discriminators(
                load_sd.get("optimizer", {})
            )
            saved_discriminators = _common_state_format_discriminators(
                common_state["optimizer"] if "optimizer" in common_state else {}
            )
            requested_optimizer_sharded = any(
                key.startswith("optimizer.") for key in requested_sharded_keys
            )
            checkpoint_optimizer_sharded = any(
                key.startswith("optimizer.") for key in checkpoint_sharded_keys
            )
            if (
                not expected_optimizer_contracts
                and requested_optimizer_sharded
                and ("optimizer" not in common_state or not common_state["optimizer"])
            ):
                optimizer_contracts = {}
            if not checkpoint_optimizer_sharded and "optimizer" not in common_state:
                raise RuntimeError(
                    "checkpoint is missing optimizer state requested by "
                    "load_optimizer=True"
                )
            if expected_optimizer_contracts != optimizer_contracts:
                expected_paths = set(expected_optimizer_contracts)
                saved_paths = set(optimizer_contracts)
                mismatched_paths = sorted(
                    path
                    for path in expected_paths & saved_paths
                    if expected_optimizer_contracts[path] != optimizer_contracts[path]
                )
                raise RuntimeError(
                    "checkpoint optimizer common-state schema mismatch: "
                    f"missing={sorted(expected_paths - saved_paths)}, "
                    f"unexpected={sorted(saved_paths - expected_paths)}, "
                    f"mismatched={mismatched_paths}"
                )
            if expected_discriminators != saved_discriminators:
                raise RuntimeError(
                    "checkpoint optimizer format discriminator mismatch: "
                    f"expected={expected_discriminators!r}, "
                    f"saved={saved_discriminators!r}"
                )
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _distributed_raise_if_error(
        local_error, context="distckpt common-state preflight failed"
    )
    _assert_world_consensus(
        {
            "step": loaded_step,
            "common_state_semantic_sha256": common_state_semantic_fingerprint,
            "common_contracts": common_contracts,
            "optimizer_contracts": optimizer_contracts,
        },
        context="distckpt common-state metadata differs across ranks",
    )
    assert common_state_fingerprint is not None
    assert common_state_semantic_fingerprint is not None
    return dict(common_state), common_state_fingerprint


def _revalidate_distckpt_common_state_file(
    checkpoint_dir: str, expected_fingerprint: tuple[int, str]
) -> None:
    current_fingerprint: tuple[int, str] | None = None
    local_error = None
    try:
        current_fingerprint = _common_state_file_fingerprint(checkpoint_dir)
        if current_fingerprint != expected_fingerprint:
            raise RuntimeError(
                "checkpoint common.pt changed after preflight: "
                f"expected={expected_fingerprint!r}, current={current_fingerprint!r}"
            )
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _distributed_raise_if_error(
        local_error, context="distckpt common-state revalidation failed"
    )


def _load_distckpt_with_preloaded_common(
    load_sd: Mapping[str, Any],
    checkpoint_dir: str,
    common_state: dict[str, Any],
    **kwargs,
) -> dict[str, Any]:
    """Bind MCore's actual load to the common state validated by preflight."""

    from megatron.core.dist_checkpointing import serialization

    expected_path = os.path.realpath(os.path.abspath(checkpoint_dir))
    owner_thread = threading.get_ident()
    with _DISTCKPT_COMMON_LOAD_LOCK:
        original_load_common = serialization.load_common

        def load_validated_common(actual_checkpoint_dir):
            actual_path = os.path.realpath(
                os.path.abspath(os.fspath(actual_checkpoint_dir))
            )
            if threading.get_ident() != owner_thread or actual_path != expected_path:
                return original_load_common(actual_checkpoint_dir)
            return common_state

        serialization.load_common = load_validated_common
        try:
            loaded = dist_checkpointing.load(load_sd, checkpoint_dir, **kwargs)
        finally:
            serialization.load_common = original_load_common
    if not isinstance(loaded, dict):
        raise TypeError(
            "MCore dist_checkpointing.load returned "
            f"{type(loaded).__name__}, expected dict"
        )
    return loaded


def _gather_world_metadata_contracts(
    local_contracts: dict[str, tuple[str, tuple[int, ...], str | None, bool]],
    *,
    context: str,
    require_identical: bool,
) -> dict[str, tuple[str, tuple[int, ...], str | None, bool]]:
    if not dist.is_available() or not dist.is_initialized():
        return dict(local_contracts)
    gathered: list[dict | None] = [None] * dist.get_world_size()
    local_error = None
    try:
        dist.all_gather_object(gathered, local_contracts)
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _distributed_raise_if_error(local_error, context=f"{context} exchange failed")
    normalized = [dict(contracts or {}) for contracts in gathered]
    if require_identical and any(
        contracts != normalized[0] for contracts in normalized[1:]
    ):
        local_error = f"rank metadata contracts differ: {normalized!r}"
        _distributed_raise_if_error(local_error, context=context)
    merged: dict[str, tuple[str, tuple[int, ...], str | None, bool]] = {}
    for rank_contracts in normalized:
        for key, contract in rank_contracts.items():
            previous = merged.get(key)
            if previous is not None and previous != contract:
                local_error = (
                    f"logical key {key!r} differs across requested ranks: "
                    f"{previous!r} vs {contract!r}"
                )
                _distributed_raise_if_error(local_error, context=context)
            merged[key] = contract
    return merged


def _validate_requested_checkpoint_contracts(
    requested: Mapping[str, tuple[str, tuple[int, ...], str | None, bool]],
    checkpoint: Mapping[str, tuple[str, tuple[int, ...], str | None, bool]],
) -> list[str]:
    mismatches: list[str] = []
    for key in sorted(set(requested) & set(checkpoint)):
        requested_kind, requested_shape, requested_dtype, allow_shape_mismatch = (
            requested[key]
        )
        checkpoint_kind, checkpoint_shape, checkpoint_dtype, _ = checkpoint[key]
        if requested_kind != checkpoint_kind:
            mismatches.append(f"{key}: type {checkpoint_kind!r} != {requested_kind!r}")
        if requested_dtype != checkpoint_dtype:
            mismatches.append(
                f"{key}: dtype {checkpoint_dtype!r} != {requested_dtype!r}"
            )
        if not allow_shape_mismatch and requested_shape != checkpoint_shape:
            mismatches.append(
                f"{key}: global_shape {checkpoint_shape!r} != {requested_shape!r}"
            )
    return mismatches


def _preflight_distckpt_checkpoint_metadata(
    load_sd: Mapping[str, Any],
    model_sd: Mapping[str, Any],
    checkpoint_dir: str,
    *,
    load_model: bool,
    load_optimizer: bool,
) -> tuple[set[str], set[str]]:
    """Reject component-aware key mismatches before MCore can mutate live tensors."""

    requested_local: dict = {}
    checkpoint_local: dict = {}
    local_error = None
    try:
        _validate_model_sharded_key_namespace(model_sd)
        requested_local = _sharded_metadata_contracts(load_sd)
        checkpoint_metadata = _load_checkpoint_sharded_metadata(checkpoint_dir)
        checkpoint_local = _sharded_metadata_contracts(checkpoint_metadata)
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _distributed_raise_if_error(
        local_error, context="distckpt checkpoint metadata read failed"
    )

    requested_contracts = _gather_world_metadata_contracts(
        requested_local, context="distckpt requested metadata", require_identical=False
    )
    checkpoint_contracts = _gather_world_metadata_contracts(
        checkpoint_local,
        context="distckpt on-disk metadata differs across ranks",
        require_identical=True,
    )
    requested_keys = set(requested_contracts)
    checkpoint_keys = set(checkpoint_contracts)

    requested_but_absent = sorted(requested_keys - checkpoint_keys)
    saved_but_unrequested = checkpoint_keys - requested_keys
    required_saved_but_unrequested = sorted(
        key
        for key in saved_but_unrequested
        if (load_optimizer and key.startswith("optimizer."))
        or (load_model and not key.startswith("optimizer."))
    )
    local_error = None
    contract_mismatches = _validate_requested_checkpoint_contracts(
        requested_contracts, checkpoint_contracts
    )
    if requested_but_absent or required_saved_but_unrequested or contract_mismatches:
        local_error = (
            "sharded key mismatch before load: "
            f"requested_but_absent={requested_but_absent}, "
            "saved_but_unrequested_for_requested_components="
            f"{required_saved_but_unrequested}, "
            f"contract_mismatches={contract_mismatches}"
        )
    _distributed_raise_if_error(
        local_error, context="distckpt checkpoint metadata preflight failed"
    )
    return requested_keys, checkpoint_keys


def _synchronize_native_optimizer_steps(optimizer: Any) -> None:
    """Align torch optimizer per-parameter steps before mcore fallback checkpointing."""

    seen: set[int] = set()

    def visit(obj: Any) -> None:
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)

        for child in _iter_optimizer_children(obj):
            visit(child)

        state = getattr(obj, "state", None)
        if isinstance(state, MutableMapping):
            _synchronize_step_mapping(state)

    visit(optimizer)


def _patch_empty_native_optimizer_state_dicts(
    optimizer: Any, *, fallback_step: int
) -> list[tuple[Any, Any]]:
    patches: list[tuple[Any, Any]] = []
    for dist_opt in _iter_distributed_optimizers(optimizer):
        inner = getattr(dist_opt, "optimizer", None)
        state = getattr(inner, "state", None)
        if not isinstance(state, MutableMapping) or state:
            continue
        original_state_dict = dist_opt.state_dict

        def patched_state_dict(
            original_state_dict=original_state_dict,
            dist_opt=dist_opt,
            fallback_step=fallback_step,
        ):
            try:
                return original_state_dict()
            except AssertionError:
                return _empty_native_optimizer_state_dict(dist_opt, fallback_step)

        dist_opt.state_dict = patched_state_dict  # type: ignore[method-assign]
        patches.append((dist_opt, original_state_dict))
    return patches


def _restore_state_dict_patches(patches: list[tuple[Any, Any]]) -> None:
    for dist_opt, original_state_dict in patches:
        dist_opt.state_dict = original_state_dict  # type: ignore[method-assign]


def _patch_native_optimizer_step_load(optimizer: Any) -> list[tuple[Any, Any]]:
    patches: list[tuple[Any, Any]] = []
    for dist_opt in _iter_distributed_optimizers(optimizer):
        original_set_state = dist_opt._set_main_param_and_optimizer_states

        def patched_set_state(
            model_param,
            tensors,
            dist_opt=dist_opt,
            original_set_state=original_set_state,
        ):
            removed_step = _pop_optimizer_step_for_model_param(
                dist_opt, model_param, tensors
            )
            try:
                return original_set_state(model_param, tensors)
            finally:
                if removed_step is not None:
                    state, step = removed_step
                    state["step"] = step

        dist_opt._set_main_param_and_optimizer_states = patched_set_state  # type: ignore[method-assign]
        patches.append((dist_opt, original_set_state))
    return patches


def _restore_set_state_patches(patches: list[tuple[Any, Any]]) -> None:
    for dist_opt, original_set_state in patches:
        dist_opt._set_main_param_and_optimizer_states = original_set_state  # type: ignore[method-assign]


def _pop_optimizer_step_for_model_param(
    dist_opt: Any, model_param, tensors: dict[str, Any]
) -> tuple[MutableMapping, Any] | None:
    if "step" in tensors:
        return None
    try:
        group_index, group_order = dist_opt.model_param_group_index_map[model_param]
        main_param = dist_opt.optimizer.param_groups[group_index]["params"][group_order]
        state = dist_opt.optimizer.state[main_param]
    except (KeyError, IndexError, TypeError):
        return None
    if not isinstance(state, MutableMapping) or "step" not in state:
        return None
    return state, state.pop("step")


def _iter_distributed_optimizers(optimizer: Any) -> Iterable[Any]:
    seen: set[int] = set()

    def visit(obj: Any):
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)

        inner = _safe_inner_optimizer(obj)
        if (
            callable(getattr(obj, "sharded_state_dict", None))
            and hasattr(obj, "gbuf_ranges")
            and hasattr(obj, "buffers")
            and inner is not None
        ):
            yield obj

        for child in _iter_optimizer_children(obj, known_inner=inner):
            yield from visit(child)

    yield from visit(optimizer)


def _iter_optimizer_children(
    obj: Any, *, known_inner: Any | None = None
) -> Iterable[Any]:
    chained = getattr(obj, "chained_optimizers", None)
    if isinstance(chained, Iterable):
        yield from chained

    sub_optimizers = getattr(obj, "sub_optimizers", None)
    if isinstance(sub_optimizers, Iterable):
        yield from sub_optimizers

    inner = _safe_inner_optimizer(obj) if known_inner is None else known_inner
    if inner is not None and inner is not obj:
        yield inner


def _safe_inner_optimizer(obj: Any) -> Any | None:
    if isinstance(getattr(obj, "chained_optimizers", None), Iterable):
        # Megatron-Core ChainedOptimizer exposes `.optimizer` only for the
        # single-optimizer compatibility case; multi-optimizer PP/EP chains
        # assert on access. The children above are the real traversal targets.
        return None
    return getattr(obj, "optimizer", None)


def _empty_native_optimizer_state_dict(
    dist_opt: Any, fallback_step: int
) -> dict[str, Any]:
    inner_state_dict = dist_opt.optimizer.state_dict()
    optimizer_state = {
        key: ([group.copy() for group in value] if key == "param_groups" else value)
        for key, value in inner_state_dict.items()
        if key != "state"
    }
    for param_group in optimizer_state["param_groups"]:
        param_group.pop("params", None)
        param_group["step"] = int(fallback_step)
    state_dict: dict[str, Any] = {"optimizer": optimizer_state}
    grad_scaler = getattr(dist_opt, "grad_scaler", None)
    if grad_scaler:
        state_dict["grad_scaler"] = grad_scaler.state_dict()
    return state_dict


def _synchronize_step_mapping(state: MutableMapping) -> None:
    steps: list[Any] = []
    for param_state in state.values():
        if isinstance(param_state, MutableMapping) and "step" in param_state:
            steps.append(param_state["step"])
    if not steps:
        return
    target = max(_step_as_int(step) for step in steps)
    for param_state in state.values():
        if isinstance(param_state, MutableMapping) and "step" in param_state:
            param_state["step"] = _step_like(param_state["step"], target)


def _step_as_int(step: Any) -> int:
    if isinstance(step, torch.Tensor):
        return int(step.detach().cpu().item())
    return int(step)


def _step_like(reference: Any, value: int) -> Any:
    if isinstance(reference, torch.Tensor):
        return torch.full_like(reference, value)
    return value


def _build_bound_sharded_state_dict(
    ps: ParallelState, get_placements: PlacementFn, is_expert: ExpertClassifierFn
) -> Callable:
    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple[tuple[int, int, int], ...] = (),
        metadata: dict | None = None,
    ) -> dict[str, ShardedTensor]:
        del metadata
        return _module_sharded_state_dict(
            _wrapped_module(self),
            ps,
            get_placements=get_placements,
            is_expert=is_expert,
            prefix=prefix,
            sharded_offsets=sharded_offsets,
        )

    return sharded_state_dict


def _module_sharded_state_dict(
    module: nn.Module,
    ps: ParallelState,
    *,
    get_placements: PlacementFn,
    is_expert: ExpertClassifierFn,
    prefix: str = "",
    sharded_offsets: tuple[tuple[int, int, int], ...] = (),
) -> dict[str, ShardedTensor]:
    state: dict[str, ShardedTensor] = {}
    for name, param in module.named_parameters():
        state[f"{prefix}{name}"] = _make_sharded_tensor(
            f"{prefix}{name}",
            param,
            ps,
            placements=get_placements(name),
            expert=is_expert(name),
            sharded_offsets=sharded_offsets,
        )
    # ``named_buffers`` also returns entries registered with
    # ``persistent=False``. Those are runtime caches and are deliberately absent
    # from ``state_dict``; putting them into a distributed checkpoint both
    # violates the PyTorch contract and can make rank-local cache shapes look
    # like checkpoint corruption on restore.
    for name, buffer in named_persistent_buffers(module):
        state[f"{prefix}{name}"] = _make_sharded_tensor(
            f"{prefix}{name}",
            buffer,
            ps,
            placements=get_placements(name),
            expert=is_expert(name),
            sharded_offsets=sharded_offsets,
        )
    return state


def _make_sharded_tensor(
    key: str,
    tensor: torch.Tensor,
    ps: ParallelState,
    *,
    placements: list,
    expert: bool,
    sharded_offsets: tuple[tuple[int, int, int], ...] = (),
) -> ShardedTensor:
    rank_offsets, replica_id = _rank_offsets_and_replica_id(
        placements, ps, expert=expert
    )
    return ShardedTensor.from_rank_offsets(
        key, tensor, *sharded_offsets, *rank_offsets, replica_id=replica_id
    )


def _rank_offsets_and_replica_id(
    placements: list, ps: ParallelState, *, expert: bool
) -> tuple[tuple[tuple[int, int, int], ...], tuple[int, ...]]:
    ranks, sizes = _mesh_ranks_and_sizes(ps, expert=expert)
    axis_fragments: dict[int, tuple[int, int]] = {}
    for placement, rank, size in zip(placements, ranks, sizes, strict=True):
        if _is_shard_placement(placement):
            dim = _shard_dim(placement)
            if dim is None:
                raise ValueError(
                    f"Unsupported Shard placement without dim: {placement!r}."
                )
            prev_rank, prev_size = axis_fragments.get(dim, (0, 1))
            axis_fragments[dim] = (prev_rank * size + rank, prev_size * size)
    rank_offsets = tuple(
        (dim, rank, size) for dim, (rank, size) in axis_fragments.items()
    )
    return rank_offsets, _replica_id(placements, ps, expert=expert)


def _replica_id(
    placements: list, ps: ParallelState, *, expert: bool
) -> tuple[int, int, int]:
    # PP stages own different parameters. They are not replicas of one
    # another, so PP rank must not make a shard non-main.
    if expert:
        return (
            0,
            _replica_axis_rank(placements, 2, ps.ep_rank),
            _replica_axis_rank(placements, 1, ps.expert_dp_rank),
        )
    dp_cp_rank = (
        0
        if _placement_is_sharded(placements, 1) or _placement_is_sharded(placements, 2)
        else ps.dp_cp_rank
    )
    return (0, _replica_axis_rank(placements, 3, ps.tp_rank), int(dp_cp_rank))


def _replica_axis_rank(placements: list, axis: int, rank: int) -> int:
    return 0 if _placement_is_sharded(placements, axis) else int(rank)


def _placement_is_sharded(placements: list, axis: int) -> bool:
    return axis < len(placements) and _is_shard_placement(placements[axis])


def _mesh_ranks_and_sizes(
    ps: ParallelState, *, expert: bool
) -> tuple[list[int], list[int]]:
    if expert:
        return (
            [ps.pp_rank, ps.expert_dp_rank, ps.ep_rank, ps.etp_rank],
            [ps.pp_size, ps.expert_dp_size, ps.ep_size, ps.etp_size],
        )
    return (
        [ps.pp_rank, ps.dp_rank, ps.cp_rank, ps.tp_rank],
        [ps.pp_size, ps.dp_size, ps.cp_size, ps.tp_size],
    )


def _is_shard_placement(placement: Any) -> bool:
    return type(placement).__name__ == "Shard"


def _shard_dim(placement: Any) -> int | None:
    dim = getattr(placement, "dim", None)
    if dim is None:
        dim = getattr(placement, "_dim", None)
    return None if dim is None else int(dim)


def _model_sharded_state_dict(model: nn.Module | Iterable[nn.Module]) -> dict[str, Any]:
    chunks = _model_chunks(model)
    ps = _chunk_parallel_state(chunks[0]) if chunks else None
    model_state: dict[str, Any] = {}
    logical_shard_owners: dict[tuple[str, Any, tuple[int, ...]], str] = {}
    for idx, chunk in enumerate(chunks):
        chunk_key = _model_chunk_key(ps, idx, len(chunks))
        chunk_state = _chunk_sharded_state_dict(
            chunk, _model_chunk_sharded_key_prefix(ps, idx, len(chunks))
        )
        for local_key, value in chunk_state.items():
            if not isinstance(value, ShardedTensor):
                continue
            replica_id = value.replica_id
            # The same logical tensor may legitimately have multiple local
            # fragments on one rank. Only an identical key/replica/offset is a
            # duplicate shard that would be overwritten during flattening.
            identity = (value.key, replica_id, value.global_offset)
            owner = f"{chunk_key}.{local_key}"
            previous_owner = logical_shard_owners.get(identity)
            if previous_owner is not None:
                raise RuntimeError(
                    "distckpt logical shard collision for "
                    f"key={value.key!r}, replica_id={replica_id!r}: "
                    f"{previous_owner} and {owner}"
                )
            logical_shard_owners[identity] = owner
        model_state[chunk_key] = chunk_state
    return model_state


def _model_chunk_key(ps: ParallelState | None, idx: int, num_chunks: int) -> str:
    if ps is not None and ps.pp_size > 1:
        key = f"model_pp{ps.pp_rank}"
        if num_chunks > 1:
            key = f"{key}_vpp{idx}"
        return key
    if num_chunks == 1:
        return "model"
    return f"model{idx}"


def _model_chunk_sharded_key_prefix(
    ps: ParallelState | None, idx: int, num_chunks: int
) -> str:
    if ps is None and num_chunks == 1:
        return ""
    if ps is not None and ps.pp_size <= 1 and num_chunks == 1:
        return ""
    return f"{_model_chunk_key(ps, idx, num_chunks)}."


def _chunk_sharded_state_dict(
    chunk: nn.Module, sharded_key_prefix: str
) -> dict[str, Any]:
    chunk_sd = chunk.sharded_state_dict()  # type: ignore[attr-defined]
    if not sharded_key_prefix:
        return chunk_sd
    tied_keys = getattr(_wrapped_module(chunk), "_mlite_tied_checkpoint_keys", {})
    out: dict[str, Any] = {}
    seen_tied: set[str] = set()
    for key, value in chunk_sd.items():
        if not isinstance(value, ShardedTensor):
            out[key] = value
            continue
        logical_key = f"{sharded_key_prefix}{value.key}"
        replica_id = value.replica_id
        # VPP normally includes the local chunk id in every logical checkpoint
        # key. The canonical input embedding is the exception: an MTP replica
        # on the last PP stage must point at one stable first-stage key without
        # knowing which VPP chunk owns the embedding. Normalize that one key to
        # the existing non-VPP spelling; duplicate ownership is still rejected
        # by ``_model_sharded_state_dict`` below.
        if sharded_key_prefix.startswith("model_pp0_vpp") and key in {
            "embed.embedding.weight",
            "embed_tokens.embedding.weight",
        }:
            logical_key = f"model_pp0.{value.key}"
        if key in tied_keys:
            if not sharded_key_prefix.startswith("model_pp"):
                raise RuntimeError(
                    "PP-tied MTP embedding checkpoint metadata requires a model_pp prefix; "
                    f"got {sharded_key_prefix!r}."
                )
            logical_key = f"model_pp0.{tied_keys[key]}"
            replica_id = (1, *tuple(replica_id)[1:])
            seen_tied.add(key)
        out[key] = replace(value, key=logical_key, replica_id=replica_id)
    missing_tied = set(tied_keys) - seen_tied
    if missing_tied:
        raise RuntimeError(
            f"PP-tied checkpoint parameters were not found: {sorted(missing_tied)}"
        )
    return out


def _single_or_all_model_state(model_sd: dict[str, Any]) -> dict[str, Any]:
    if "model" in model_sd:
        return model_sd["model"]
    return model_sd


def _load_model_state_dict(
    model: nn.Module | Iterable[nn.Module],
    state_dict: dict[str, Any],
    *,
    expected_state_dict: dict[str, Any] | None = None,
) -> None:
    chunks = _model_chunks(model)
    if not chunks:
        raise RuntimeError("distckpt model load requires at least one model chunk")
    ps = _chunk_parallel_state(chunks[0]) if chunks else None
    expected_state_dict = (
        _model_sharded_state_dict(model)
        if expected_state_dict is None
        else expected_state_dict
    )
    assignments: list[tuple[nn.Module, str, dict[str, Any], set[str]]] = []
    for idx, chunk in enumerate(chunks):
        key = _model_chunk_key(ps, idx, len(chunks))
        if key not in state_dict:
            raise RuntimeError(
                f"distckpt checkpoint is missing required model subtree {key!r}"
            )
        if key not in expected_state_dict:
            raise RuntimeError(
                f"distckpt load template is missing required model subtree {key!r}"
            )
        loaded_subtree = state_dict[key]
        expected_subtree = expected_state_dict[key]
        if not isinstance(loaded_subtree, dict) or not isinstance(
            expected_subtree, dict
        ):
            raise TypeError(f"distckpt model subtree {key!r} must be a dictionary")
        loaded_keys = set(loaded_subtree)
        expected_keys = set(expected_subtree)
        if loaded_keys != expected_keys:
            raise RuntimeError(
                f"distckpt model subtree {key!r} key mismatch: "
                f"missing={sorted(expected_keys - loaded_keys)}, "
                f"unexpected={sorted(loaded_keys - expected_keys)}"
            )

        current_state = _wrapped_module(chunk).state_dict()
        metadata_errors: list[str] = []
        for name, loaded_tensor in loaded_subtree.items():
            current_tensor = current_state.get(name)
            if not isinstance(current_tensor, torch.Tensor) or not isinstance(
                loaded_tensor, torch.Tensor
            ):
                metadata_errors.append(f"{name}: expected tensor checkpoint entry")
                continue
            if current_tensor.shape != loaded_tensor.shape:
                metadata_errors.append(
                    f"{name}: shape {tuple(loaded_tensor.shape)} != "
                    f"{tuple(current_tensor.shape)}"
                )
            if current_tensor.dtype != loaded_tensor.dtype:
                metadata_errors.append(
                    f"{name}: dtype {loaded_tensor.dtype} != {current_tensor.dtype}"
                )
        if metadata_errors:
            raise RuntimeError(
                f"distckpt model subtree {key!r} metadata mismatch: {metadata_errors}"
            )
        assignments.append((chunk, key, loaded_subtree, expected_keys))

    # Commit only after every chunk passes the read-only preflight. Shared/tied
    # module aliases are intentionally absent from ``named_parameters`` and thus
    # from the sharded template, so strict=False is required; missing checkpoint
    # keys and unexpected loaded keys were already rejected above.
    for chunk, key, loaded_subtree, expected_keys in assignments:
        incompatible = _wrapped_module(chunk).load_state_dict(
            loaded_subtree, strict=False
        )
        missing_required = sorted(set(incompatible.missing_keys) & expected_keys)
        if missing_required or incompatible.unexpected_keys:
            raise RuntimeError(
                f"distckpt model subtree {key!r} application mismatch: "
                f"missing_required={missing_required}, "
                f"unexpected={sorted(incompatible.unexpected_keys)}"
            )


def _chunk_parallel_state(chunk: nn.Module) -> ParallelState | None:
    ps = getattr(chunk, "_mlite_dist_opt_parallel_state", None)
    if ps is not None:
        return ps
    wrapped = _wrapped_module(chunk)
    return getattr(wrapped, "_mlite_dist_opt_parallel_state", None)


def _model_chunks(model: nn.Module | Iterable[nn.Module]) -> list[nn.Module]:
    if isinstance(model, nn.Module):
        return list(model) if isinstance(model, nn.ModuleList) else [model]
    chunks = list(model)
    if not all(isinstance(chunk, nn.Module) for chunk in chunks):
        raise TypeError("distckpt model chunks must be nn.Module instances.")
    return chunks


def _wrapped_module(model: nn.Module) -> nn.Module:
    module = getattr(model, "module", None)
    return module if isinstance(module, nn.Module) else model


__all__ = [
    "attach_model_sharded_state_dict",
    "load_dist_opt_checkpoint",
    "save_dist_opt_checkpoint",
    "supports_dist_opt_distckpt",
]
