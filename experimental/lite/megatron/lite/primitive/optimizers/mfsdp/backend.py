# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Runtime-facing Megatron-FSDP backend contract."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor  # pyright: ignore[reportMissingImports]

try:
    from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
        preprocess_state_dict_for_uneven_dtensor,
    )
except Exception:  # pragma: no cover - optional Megatron-Core integration.
    preprocess_state_dict_for_uneven_dtensor = None

from megatron.lite.primitive.optimizers.mfsdp.checkpoint_keys import (
    canonicalize_expert_checkpoint_key,
    expert_local_counts,
    normalize_mcore_fsdp_param_name,
    optimizer_checkpoint_expert_classifier,
    optimizer_checkpoint_parallel_state,
)
from megatron.lite.primitive.optimizers.mfsdp.patches import PARAM_NAME_ATTR

_MCORE_NON_PARAMETER_STATE_KEY = "mcore_non_parameter_state"
_INNER_OPTIMIZER_STATE_KEY = "inner_optimizer_state"
_PARAMETER_STATE_KEY = "parameter_state"
_STATE_FORMAT_VERSION_KEY = "mfsdp_state_format_version"
_PARAMETER_STATE_FORMAT_KEY = "parameter_state_format"
_LIVE_PARAMETER_STATE_FORMAT = "live_param_groups"
_FSDP_DTENSOR_PARAMETER_STATE_FORMAT = "fsdp_dtensor"
_LOCAL_RANK_PARAMETER_STATE_FORMAT = "fsdp_dtensor_local_rank"
_MAIN_PARAM_STATE_KEY = "param"
_LOCAL_RANK_STATE_KEY = "rank_state"
_STATE_FORMAT_VERSION = 2


@dataclass(frozen=True, slots=True)
class MegatronFSDPBackend:
    name: str = "megatron_fsdp"
    runtime_backend: str = "megatron_fsdp"

    def zero_grad(self, optimizer: Any) -> None:
        optimizer.zero_grad()

    def finish_grad_sync(self, optimizer: Any) -> None:
        if hasattr(optimizer, "finish_grad_sync"):
            optimizer.finish_grad_sync()

    def clip_grad_norm(self, optimizer: Any):
        if hasattr(optimizer, "clip_grad_norm"):
            return optimizer.clip_grad_norm()
        return None

    def step(self, optimizer: Any):
        return optimizer.step()

    def state_dict(self, optimizer: Any) -> dict:
        mcore_optimizer = _unwrap_mcore_optimizer(optimizer)
        if _is_mcore_mfsdp_optimizer(mcore_optimizer):
            non_parameter_state = mcore_optimizer.state_dict()
            parameter_state = _with_mcore_optimizer_group_meta(
                _get_fsdp_dtensor_parameter_state(
                    mcore_optimizer,
                    clone_main_params=True,
                ),
                non_parameter_state,
            )
            return {
                _STATE_FORMAT_VERSION_KEY: _STATE_FORMAT_VERSION,
                _MCORE_NON_PARAMETER_STATE_KEY: non_parameter_state,
                _PARAMETER_STATE_KEY: parameter_state,
                _PARAMETER_STATE_FORMAT_KEY: _FSDP_DTENSOR_PARAMETER_STATE_FORMAT,
            }
        return optimizer.state_dict()

    def dcp_state_dict(
        self,
        optimizer: Any,
        *,
        is_loading: bool,
        include_main_params: bool = False,
    ) -> dict:
        """Build an M-FSDP optimizer state for DCP without MCore load-time dummy step.

        MCore's fsdp_dtensor checkpoint loader initializes optimizer state by
        manufacturing zero gradients and doing an optimizer step. For 30B-class
        jobs that transient grad allocation is enough to OOM. DCP only needs a
        correctly shaped destination tree, so loading uses empty per-parameter
        state tensors keyed the same way as MCore's fsdp_dtensor save format.
        """
        mcore_optimizer = _unwrap_mcore_optimizer(optimizer)
        if not _is_mcore_mfsdp_optimizer(mcore_optimizer):
            return optimizer.state_dict()

        if is_loading:
            non_parameter_state = _mcore_non_parameter_state_template(mcore_optimizer)
            _debug_template_cuda_sync("after-non-parameter-template")
        else:
            non_parameter_state = mcore_optimizer.state_dict()
        if is_loading:
            _debug_template_cuda_sync("before-parameter-state-template")
            parameter_state = _get_fsdp_dtensor_parameter_state_template(
                mcore_optimizer,
                include_main_params=include_main_params,
            )
            _debug_template_cuda_sync("after-parameter-state-template")
        else:
            parameter_state = _get_fsdp_dtensor_parameter_state(
                mcore_optimizer,
                include_main_params=include_main_params,
                clone_main_params=False,
            )
        parameter_state = _with_mcore_optimizer_group_meta(
            parameter_state,
            non_parameter_state,
        )
        parameter_state_format = _FSDP_DTENSOR_PARAMETER_STATE_FORMAT
        if _use_local_rank_dcp_optimizer_state():
            parameter_state = _local_rank_fsdp_dtensor_parameter_state_for_dcp(
                parameter_state
            )
            parameter_state_format = _LOCAL_RANK_PARAMETER_STATE_FORMAT
        else:
            _preprocess_fsdp_dtensor_parameter_state_for_dcp(parameter_state)
        return {
            _STATE_FORMAT_VERSION_KEY: _STATE_FORMAT_VERSION,
            _MCORE_NON_PARAMETER_STATE_KEY: non_parameter_state,
            _PARAMETER_STATE_KEY: parameter_state,
            _PARAMETER_STATE_FORMAT_KEY: parameter_state_format,
        }

    def load_state_dict(self, optimizer: Any, state_dict: dict) -> None:
        mcore_optimizer = _unwrap_mcore_optimizer(optimizer)
        if _is_mcore_mfsdp_optimizer(mcore_optimizer):
            loaded_main_params = False
            parameter_state = state_dict.get(_PARAMETER_STATE_KEY)
            if parameter_state is not None:
                parameter_state_format = state_dict.get(_PARAMETER_STATE_FORMAT_KEY)
                if parameter_state_format in (None, _LIVE_PARAMETER_STATE_FORMAT):
                    _load_live_parameter_state(mcore_optimizer, parameter_state)
                    loaded_main_params = True
                elif parameter_state_format == _FSDP_DTENSOR_PARAMETER_STATE_FORMAT:
                    parameter_state = _with_mcore_optimizer_group_meta(
                        parameter_state,
                        state_dict.get(_MCORE_NON_PARAMETER_STATE_KEY),
                    )
                    loaded_main_params = _load_fsdp_dtensor_parameter_state(
                        mcore_optimizer,
                        parameter_state,
                    )
                elif parameter_state_format == _LOCAL_RANK_PARAMETER_STATE_FORMAT:
                    parameter_state = _current_local_rank_parameter_state(parameter_state)
                    parameter_state = _with_mcore_optimizer_group_meta(
                        parameter_state,
                        state_dict.get(_MCORE_NON_PARAMETER_STATE_KEY),
                    )
                    loaded_main_params = _load_fsdp_dtensor_parameter_state(
                        mcore_optimizer,
                        parameter_state,
                    )
                else:
                    raise ValueError(
                        f"Unsupported Megatron-FSDP parameter state format: "
                        f"{parameter_state_format!r}."
                    )
            elif _INNER_OPTIMIZER_STATE_KEY in state_dict:
                _load_inner_optimizer_state(
                    mcore_optimizer,
                    state_dict[_INNER_OPTIMIZER_STATE_KEY],
                )
                loaded_main_params = True
            _restore_mcore_non_parameter_state(
                mcore_optimizer,
                state_dict.get(_MCORE_NON_PARAMETER_STATE_KEY),
            )
            if loaded_main_params:
                _sync_main_weights_to_model_weights(mcore_optimizer)
            return
        optimizer.load_state_dict(state_dict)

    def state_dict_has_main_params(self, state_dict: Any) -> bool:
        return _state_dict_has_main_params(state_dict)

    def sync_model_weights_to_main_weights(self, optimizer: Any) -> bool:
        mcore_optimizer = _unwrap_mcore_optimizer(optimizer)
        if not _is_mcore_mfsdp_optimizer(mcore_optimizer):
            return False
        return _sync_model_weights_to_main_weights(mcore_optimizer)

    def finalize_grads(self, finalize_fn, model_chunks: list[Any], optimizer: Any) -> None:
        finalize_fn(model_chunks, optimizer)


BACKEND = MegatronFSDPBackend()


def is_megatron_fsdp_optimizer(optimizer: Any) -> bool:
    return _is_mcore_mfsdp_optimizer(_unwrap_mcore_optimizer(optimizer))


def _is_mcore_mfsdp_optimizer(optimizer: Any) -> bool:
    return any(
        bool(getattr(getattr(leaf, "ddp_config", None), "use_megatron_fsdp", False))
        for leaf in _iter_chained_optimizers(_unwrap_mcore_optimizer(optimizer))
    )


def _unwrap_mcore_optimizer(optimizer: Any) -> Any:
    return getattr(optimizer, "_inner_optimizer", optimizer)


def _inner_torch_optimizer(optimizer: Any) -> Any:
    return getattr(_unwrap_mcore_optimizer(optimizer), "optimizer", None)


def _restore_mcore_non_parameter_state(optimizer: Any, state_dict: Any) -> None:
    if not isinstance(state_dict, dict):
        if isinstance(state_dict, list):
            for leaf_optimizer, leaf_state in zip(
                _iter_chained_optimizers(optimizer),
                state_dict,
                strict=True,
            ):
                _restore_mcore_non_parameter_state(leaf_optimizer, leaf_state)
        return
    if "optimizer" not in state_dict and "grad_scaler" not in state_dict:
        return
    grad_scaler = getattr(optimizer, "grad_scaler", None)
    grad_scaler_state = state_dict.get("grad_scaler")
    if grad_scaler is not None and grad_scaler_state is not None:
        grad_scaler.load_state_dict(grad_scaler_state)


def _iter_chained_optimizers(optimizer: Any) -> list[Any]:
    chained = getattr(optimizer, "chained_optimizers", None)
    if chained is None:
        return [optimizer]
    return list(chained)


@torch.no_grad()
def _get_live_parameter_state(optimizer: Any) -> Any:
    states = []
    for chained_optimizer in _iter_chained_optimizers(optimizer):
        inner_optimizer = _inner_torch_optimizer(chained_optimizer)
        if inner_optimizer is None or not hasattr(inner_optimizer, "param_groups"):
            raise TypeError("Megatron-FSDP optimizer is missing live param_groups.")
        groups = []
        optimizer_state = getattr(inner_optimizer, "state", {})
        for group in inner_optimizer.param_groups:
            params = list(group.get("params", ()))
            groups.append(
                {
                    "group": {
                        key: _clone_state_value(value)
                        for key, value in group.items()
                        if key != "params"
                    },
                    "params": [_clone_state_value(param) for param in params],
                    "state": [
                        {
                            key: _clone_state_value(value)
                            for key, value in optimizer_state.get(param, {}).items()
                        }
                        for param in params
                    ],
                }
            )
        states.append({"param_groups": groups})
    return states[0] if len(states) == 1 else states


@torch.no_grad()
def _load_live_parameter_state(optimizer: Any, state: Any) -> None:
    chained_optimizers = _iter_chained_optimizers(optimizer)
    states = state if isinstance(state, list) else [state]
    if len(states) != len(chained_optimizers):
        raise ValueError(
            "Megatron-FSDP live parameter state count does not match chained optimizers: "
            f"{len(states)} != {len(chained_optimizers)}."
        )
    for chained_optimizer, chained_state in zip(chained_optimizers, states):
        inner_optimizer = _inner_torch_optimizer(chained_optimizer)
        if inner_optimizer is None or not hasattr(inner_optimizer, "param_groups"):
            raise TypeError("Megatron-FSDP optimizer is missing live param_groups.")
        saved_groups = chained_state.get("param_groups", [])
        if len(saved_groups) != len(inner_optimizer.param_groups):
            raise ValueError(
                "Megatron-FSDP live parameter group count mismatch: "
                f"{len(saved_groups)} != {len(inner_optimizer.param_groups)}."
            )
        optimizer_state = getattr(inner_optimizer, "state", None)
        for group, saved_group in zip(inner_optimizer.param_groups, saved_groups):
            for key, value in saved_group.get("group", {}).items():
                group[key] = _clone_state_value(value)
            params = list(group.get("params", ()))
            saved_params = saved_group.get("params", [])
            saved_states = saved_group.get("state", [])
            if len(saved_params) != len(params) or len(saved_states) != len(params):
                raise ValueError("Megatron-FSDP live parameter state has invalid group shape.")
            for param, saved_param, saved_param_state in zip(
                params,
                saved_params,
                saved_states,
            ):
                _copy_state_value_(param, saved_param)
                if optimizer_state is not None:
                    optimizer_state[param] = {
                        key: _clone_state_value(value)
                        for key, value in saved_param_state.items()
                    }


def _load_inner_optimizer_state(optimizer: Any, state: Any) -> None:
    chained_optimizers = _iter_chained_optimizers(optimizer)
    states = state if isinstance(state, list) else [state]
    if len(states) != len(chained_optimizers):
        raise ValueError(
            "Megatron-FSDP inner optimizer state count does not match chained optimizers: "
            f"{len(states)} != {len(chained_optimizers)}."
        )
    for chained_optimizer, chained_state in zip(chained_optimizers, states):
        inner_optimizer = _inner_torch_optimizer(chained_optimizer)
        if inner_optimizer is None or not hasattr(inner_optimizer, "load_state_dict"):
            raise TypeError(
                "Megatron-FSDP optimizer is missing an inner optimizer load_state_dict."
            )
        inner_optimizer.load_state_dict(chained_state)


def _sync_main_weights_to_model_weights(optimizer: Any) -> None:
    sync = getattr(optimizer, "_copy_main_params_to_model_params", None)
    if callable(sync):
        sync()
        return
    for chained_optimizer in _iter_chained_optimizers(optimizer):
        sync = getattr(chained_optimizer, "_copy_main_params_to_model_params", None)
        if callable(sync):
            sync()


@torch.no_grad()
def _sync_model_weights_to_main_weights(optimizer: Any) -> bool:
    synced_any = False
    for chained_optimizer in _iter_chained_optimizers(optimizer):
        if _sync_single_model_weights_to_main_weights(chained_optimizer):
            synced_any = True
    if synced_any:
        print(
            "[MFSDP_DCP_RESTORE] "
            f"rank={_debug_rank()} model_to_main_params_synced=True",
            flush=True,
        )
    return synced_any


def _sync_single_model_weights_to_main_weights(optimizer: Any) -> bool:
    ddp_config = getattr(optimizer, "ddp_config", None)
    if not bool(getattr(ddp_config, "use_megatron_fsdp", False)):
        sync = getattr(optimizer, "_copy_model_params_to_main_params", None)
        if callable(sync):
            sync()
            return True
        return False

    buffer_synced_any = False
    for model_chunk in getattr(optimizer, "model_chunks", ()):
        buffer = getattr(model_chunk, "param_and_grad_buffer", None)
        if buffer is not None:
            buffer_synced_any = (
                _sync_mfsdp_param_buffer_model_weights_to_main_weights(buffer)
                or buffer_synced_any
            )
    if buffer_synced_any:
        return True

    copied_any = False
    copied_any = (
        _copy_model_group_weights_to_main_group(
            optimizer,
            getattr(optimizer, "model_float16_groups", ()),
            getattr(optimizer, "shard_fp32_from_float16_groups", ()),
        )
        or copied_any
    )
    copied_any = (
        _copy_model_group_weights_to_main_group(
            optimizer,
            getattr(optimizer, "model_fp32_groups", ()),
            getattr(optimizer, "shard_fp32_groups", ()),
        )
        or copied_any
    )
    return copied_any


def _sync_mfsdp_param_buffer_model_weights_to_main_weights(buffer: Any) -> bool:
    copied_any = False
    optimizer_named_parameters = getattr(buffer, "optimizer_named_parameters", None)
    if optimizer_named_parameters:
        dist_param_by_orig = {
            getattr(dist_param, "orig_param", None): dist_param
            for _name, dist_param in optimizer_named_parameters
        }
    else:
        dist_param_by_orig = {}
    for group in getattr(buffer, "parameter_groups", ()):
        main_weight_buffer = getattr(group, "main_weight_buffer", None)
        if main_weight_buffer is None:
            continue
        model_weight_buffer = getattr(group, "model_weight_buffer", None)
        main_param_idx = getattr(main_weight_buffer, "param_idx", None)
        if not isinstance(main_param_idx, dict):
            raise TypeError("Megatron-FSDP main weight buffer is missing param_idx.")
        for orig_param in getattr(group, "params", ()):
            item_id = main_param_idx[orig_param]
            model_weight, main_weight = _mfsdp_param_buffer_model_and_main_weight(
                orig_param,
                item_id,
                model_weight_buffer,
                main_weight_buffer,
            )
            if model_weight.numel() == 0:
                continue
            if model_weight.numel() != main_weight.numel():
                name = getattr(buffer, "param_to_name", {}).get(orig_param)
                raise ValueError(
                    "Megatron-FSDP model-to-main buffer restore shape mismatch "
                    f"for {name or '<unnamed>'}: model={tuple(model_weight.shape)} "
                    f"main={tuple(main_weight.shape)}."
                )
            restored_main_weight = model_weight.view_as(main_weight).float()
            main_weight.copy_(restored_main_weight)
            dist_param = dist_param_by_orig.get(orig_param)
            if dist_param is not None:
                _copy_tensor_value_reshaped_(
                    _tensor_local_data(dist_param),
                    restored_main_weight,
                )
            copied_any = True
    if copied_any:
        copy_main_to_model = getattr(buffer, "copy_main_weights_to_model_weights", None)
        if callable(copy_main_to_model):
            copy_main_to_model()
    return copied_any


def _mfsdp_param_buffer_model_and_main_weight(
    orig_param: Any,
    item_id: int,
    model_weight_buffer: Any,
    main_weight_buffer: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    if model_weight_buffer is not None:
        if bool(getattr(model_weight_buffer, "is_data_distributed", False)) or bool(
            getattr(main_weight_buffer, "is_data_distributed", False)
        ):
            return (
                model_weight_buffer.get_item(item_id, only_shard=True),
                main_weight_buffer.get_item(item_id, only_shard=True),
            )
        return model_weight_buffer.get_item(item_id), main_weight_buffer.get_item(item_id)
    if bool(getattr(main_weight_buffer, "is_data_distributed", False)):
        raise TypeError(
            "Megatron-FSDP cannot rebuild a sharded main weight buffer without a model weight buffer."
        )
    return _tensor_local_data(orig_param).view(-1), main_weight_buffer.get_item(item_id)


def _copy_model_group_weights_to_main_group(
    optimizer: Any,
    model_groups: Any,
    shard_main_groups: Any,
) -> bool:
    copied_any = False
    for model_group, shard_main_group in zip(model_groups, shard_main_groups, strict=True):
        for model_param, shard_main_param in zip(model_group, shard_main_group, strict=True):
            if _is_distopt_quantized_param(optimizer, model_param):
                continue
            shard_model_param = _mfsdp_model_param_shard(optimizer, model_param)
            if shard_model_param.numel() != shard_main_param.numel():
                name = _optimizer_param_name(optimizer, model_param)
                raise ValueError(
                    "Megatron-FSDP model-to-main checkpoint restore shape mismatch "
                    f"for {name or '<unnamed>'}: model_shard={tuple(shard_model_param.shape)} "
                    f"main_shard={tuple(shard_main_param.shape)}."
                )
            shard_main_param.copy_(shard_model_param.view_as(shard_main_param))
            copied_any = True
    return copied_any


def _mfsdp_model_param_shard(optimizer: Any, model_param: Any) -> torch.Tensor:
    get_range_map = getattr(optimizer, "_get_model_param_range_map", None)
    if not callable(get_range_map):
        raise TypeError("Megatron-FSDP optimizer is missing _get_model_param_range_map.")
    range_map = get_range_map(model_param)
    if not isinstance(range_map, dict) or "gbuf_world_in_bucket" not in range_map:
        raise TypeError("Megatron-FSDP optimizer range map is missing gbuf_world_in_bucket.")
    world_range = range_map["gbuf_world_in_bucket"]
    gbuf_map = getattr(optimizer, "model_param_gbuf_map", None)
    if not isinstance(gbuf_map, dict) or model_param not in gbuf_map:
        raise TypeError("Megatron-FSDP optimizer is missing model_param_gbuf_map entry.")
    gbuf_index, _dtype, bucket_id = gbuf_map[model_param]
    buffers = getattr(optimizer, "buffers", None)
    try:
        bucket = buffers[gbuf_index].buckets[bucket_id]
        param_data = bucket.param_data
    except (TypeError, AttributeError, IndexError, KeyError) as exc:
        raise TypeError("Megatron-FSDP optimizer buffer metadata is invalid.") from exc
    return param_data.view(-1)[world_range.start:world_range.end]


def _is_distopt_quantized_param(optimizer: Any, model_param: Any) -> bool:
    is_quantized = getattr(optimizer, "_is_distopt_quantized_param", None)
    return bool(callable(is_quantized) and is_quantized(model_param))


def _tensor_local_data(value: Any) -> torch.Tensor:
    data = getattr(value, "data", value)
    if isinstance(data, DTensor):
        return _dtensor_local_tensor(data)
    if isinstance(data, torch.Tensor):
        return data
    raise TypeError(f"Expected Tensor or DTensor, got {type(value).__name__}.")


def _clone_state_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_state_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_state_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_state_value(item) for item in value)
    return value


def _copy_state_value_(target: Any, source: Any) -> None:
    if isinstance(target, DTensor) or isinstance(source, DTensor):
        _copy_tensor_value_reshaped_(
            _tensor_local_data(target),
            _tensor_local_data(source),
        )
        return
    if torch.is_tensor(target) and torch.is_tensor(source):
        with torch.no_grad():
            target.copy_(source)


def _copy_tensor_value_reshaped_(target: torch.Tensor, source: torch.Tensor) -> None:
    if target.numel() != source.numel():
        raise ValueError(
            "Cannot copy optimizer tensor state with different element counts: "
            f"target={tuple(target.shape)} source={tuple(source.shape)}."
        )
    with torch.no_grad():
        target.copy_(source.view_as(target))


def _get_fsdp_dtensor_parameter_state(
    optimizer: Any,
    *,
    include_main_params: bool = True,
    clone_main_params: bool = False,
) -> Any:
    states = []
    for chained_optimizer in _iter_chained_optimizers(optimizer):
        _ensure_fsdp_dtensor_optimizer_param_names(chained_optimizer)
        sharded_state_dict = getattr(chained_optimizer, "sharded_state_dict", None)
        if not callable(sharded_state_dict):
            raise TypeError(
                "Megatron-FSDP optimizer is missing sharded_state_dict for fsdp_dtensor state."
            )
        parameter_state = sharded_state_dict(
            is_loading=False,
            metadata={
                "distrib_optim_sharding_type": _FSDP_DTENSOR_PARAMETER_STATE_FORMAT
            },
        )
        parameter_state = _canonicalize_fsdp_dtensor_parameter_state(
            chained_optimizer,
            parameter_state,
        )
        if include_main_params:
            parameter_state = _with_fsdp_dtensor_main_params(
                parameter_state,
                chained_optimizer,
                clone_main_params=clone_main_params,
            )
            parameter_state = _canonicalize_fsdp_dtensor_parameter_state(
                chained_optimizer,
                parameter_state,
            )
        else:
            parameter_state = _without_fsdp_dtensor_main_params(parameter_state)
        states.append(parameter_state)
    return states[0] if len(states) == 1 else states


def _get_fsdp_dtensor_parameter_state_template(
    optimizer: Any,
    *,
    include_main_params: bool,
) -> Any:
    states = []
    for chained_optimizer in _iter_chained_optimizers(optimizer):
        _debug_template_cuda_sync("before-name-map", log_success=True)
        _ensure_fsdp_dtensor_optimizer_param_names(chained_optimizer)
        _debug_template_cuda_sync("after-name-map", log_success=True)
        state = {}
        for param in _iter_inner_optimizer_params(chained_optimizer):
            name = _optimizer_param_name(chained_optimizer, param)
            if name is None:
                continue
            _debug_template_cuda_sync(
                f"before-param-template:{name}",
                log_success=_debug_dcp_template_key_enabled(name),
            )
            state[name] = _optimizer_param_state_template(
                chained_optimizer,
                param,
                debug_name=name,
                include_main_params=include_main_params,
            )
        parameter_state = {
            "state": state,
            "param_to_group_meta": _param_groups_to_param2group_meta(
                chained_optimizer
            ),
        }
        states.append(
            _canonicalize_fsdp_dtensor_parameter_state(
                chained_optimizer,
                parameter_state,
            )
        )
    return states[0] if len(states) == 1 else states


def _optimizer_param_state_template(
    optimizer: Any,
    param: Any,
    *,
    debug_name: str | None = None,
    include_main_params: bool,
) -> dict[str, Any]:
    inner_optimizer = _inner_torch_optimizer(optimizer)
    live_state = {}
    if inner_optimizer is not None:
        optimizer_state = getattr(inner_optimizer, "state", {})
        live_state = (
            optimizer_state.get(param, {})
            if isinstance(optimizer_state, dict)
            else {}
        )

    if live_state:
        state = {
            key: _empty_state_value_like(
                value,
                debug_key=_debug_state_key(debug_name, str(key)),
            )
            for key, value in live_state.items()
        }
    else:
        state = {
            "exp_avg": _empty_state_value_like(
                param,
                debug_key=_debug_state_key(debug_name, "exp_avg"),
            ),
            "exp_avg_sq": _empty_state_value_like(
                param,
                debug_key=_debug_state_key(debug_name, "exp_avg_sq"),
            ),
        }
    if include_main_params:
        state[_MAIN_PARAM_STATE_KEY] = _checkpoint_tensor_view(param)
    return state


def _param_groups_to_param2group_meta(optimizer: Any) -> dict[str, Any]:
    inner_optimizer = _inner_torch_optimizer(optimizer)
    param_groups = getattr(inner_optimizer, "param_groups", None)
    if param_groups is None:
        return {}
    mcore_converter = getattr(optimizer, "_param_groups_to_param2group_meta", None)
    if callable(mcore_converter):
        return mcore_converter(param_groups)

    param_to_group_meta = {}
    for group in param_groups:
        group_meta = {key: value for key, value in group.items() if key != "params"}
        for param in group.get("params", ()):
            name = _optimizer_param_name(optimizer, param)
            if name is not None:
                param_to_group_meta[name] = dict(group_meta)
    return param_to_group_meta


def _with_optimizer_group_step_placeholders(state_dict: Any) -> Any:
    if isinstance(state_dict, list):
        return [_with_optimizer_group_step_placeholders(item) for item in state_dict]
    if not isinstance(state_dict, dict):
        return state_dict
    optimizer_state = state_dict.get("optimizer")
    if not isinstance(optimizer_state, dict):
        return state_dict
    param_groups = optimizer_state.get("param_groups")
    if not isinstance(param_groups, list):
        return state_dict

    updated_optimizer_state = dict(optimizer_state)
    updated_optimizer_state["param_groups"] = [
        _with_group_step_placeholder(group) for group in param_groups
    ]
    updated_state_dict = dict(state_dict)
    updated_state_dict["optimizer"] = updated_optimizer_state
    return updated_state_dict


def _with_group_step_placeholder(group: Any) -> Any:
    if not isinstance(group, dict) or "step" in group:
        return group
    updated_group = dict(group)
    updated_group["step"] = 0
    return updated_group


def _canonicalize_fsdp_dtensor_parameter_state(optimizer: Any, parameter_state: Any) -> Any:
    if not isinstance(parameter_state, dict):
        return parameter_state
    canonicalize = _optimizer_param_name_canonicalizer(optimizer)
    if canonicalize is None:
        return parameter_state

    updated = dict(parameter_state)
    changed = False

    state = parameter_state.get("state")
    if isinstance(state, dict):
        updated_state = {}
        state_sources = {}
        for name, value in state.items():
            canonical_name = canonicalize(name) if isinstance(name, str) else name
            _check_canonical_name_collision(state_sources, canonical_name, name)
            updated_state[canonical_name] = value
            state_sources[canonical_name] = name
            changed = changed or canonical_name != name
        updated["state"] = updated_state

    param_to_group_meta = parameter_state.get("param_to_group_meta")
    if isinstance(param_to_group_meta, dict):
        updated_meta = {}
        meta_sources = {}
        for name, value in param_to_group_meta.items():
            canonical_name = canonicalize(name) if isinstance(name, str) else name
            _check_canonical_name_collision(meta_sources, canonical_name, name)
            updated_meta[canonical_name] = value
            meta_sources[canonical_name] = name
            changed = changed or canonical_name != name
        updated["param_to_group_meta"] = updated_meta

    return updated if changed else parameter_state


def _check_canonical_name_collision(
    sources: dict[Any, Any],
    canonical_name: Any,
    source_name: Any,
) -> None:
    existing_source = sources.get(canonical_name)
    if existing_source is None or existing_source == source_name:
        return
    raise ValueError(
        "Megatron-FSDP optimizer checkpoint key canonicalization collision: "
        f"{source_name!r} and {existing_source!r} map to {canonical_name!r}."
    )


def _optimizer_param_name_canonicalizer(optimizer: Any):
    ps = optimizer_checkpoint_parallel_state(optimizer)
    is_expert = optimizer_checkpoint_expert_classifier(optimizer)
    if ps is None or is_expert is None:
        return None

    names = [
        name
        for name in _model_chunk_param_names(optimizer).values()
        if isinstance(name, str)
    ]
    if not names:
        mapping = getattr(optimizer, "param_to_name", None)
        if isinstance(mapping, dict):
            names.extend(name for name in mapping.values() if isinstance(name, str))
    local_counts = expert_local_counts(
        (normalize_mcore_fsdp_param_name(name) for name in names),
        is_expert,
    )

    def canonicalize(name: str) -> str:
        normalized = normalize_mcore_fsdp_param_name(name)
        _, canonical_normalized = canonicalize_expert_checkpoint_key(
            normalized,
            normalized,
            ps=ps,
            is_expert=is_expert(normalized),
            local_counts=local_counts,
        )
        if name.startswith("module.module."):
            return f"module.module.{canonical_normalized}"
        return canonical_normalized

    return canonicalize


def _mcore_non_parameter_state_template(optimizer: Any) -> Any:
    """Build load-time MCore optimizer metadata without calling MCore state_dict.

    MCore's Megatron-FSDP optimizer state_dict path is not a pure metadata
    query before optimizer moments exist. For DCP load we only need a
    correctly shaped tree for param-group metadata such as ``step``; parameter
    tensors are represented separately by the fsdp_dtensor parameter state.
    """
    states = [
        _single_mcore_non_parameter_state_template(chained_optimizer)
        for chained_optimizer in _iter_chained_optimizers(optimizer)
    ]
    return states[0] if len(states) == 1 else states


def _single_mcore_non_parameter_state_template(optimizer: Any) -> dict[str, Any]:
    inner_optimizer = _inner_torch_optimizer(optimizer)
    if inner_optimizer is None:
        return {}
    param_groups = getattr(inner_optimizer, "param_groups", None)
    if not isinstance(param_groups, list):
        return {}
    return {
        "optimizer": {
            "state": {},
            "param_groups": [
                _with_group_step_placeholder(
                    {key: value for key, value in group.items() if key != "params"}
                )
                for group in param_groups
                if isinstance(group, dict)
            ],
        },
    }


def _empty_state_value_like(value: Any, *, debug_key: str | None = None) -> Any:
    if isinstance(value, DTensor):
        return _empty_dtensor_like(value, debug_key=debug_key)
    data = getattr(value, "data", None)
    if isinstance(data, DTensor):
        return _empty_dtensor_like(data, debug_key=debug_key)
    if torch.is_tensor(value):
        return _empty_tensor_like(value)
    if isinstance(value, dict):
        return {
            key: _empty_state_value_like(
                item,
                debug_key=_debug_state_key(debug_key, str(key)),
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _empty_state_value_like(
                item,
                debug_key=_debug_state_key(debug_key, str(index)),
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        return tuple(
            _empty_state_value_like(
                item,
                debug_key=_debug_state_key(debug_key, str(index)),
            )
            for index, item in enumerate(value)
        )
    return value


def _empty_dtensor_like(
    value: DTensor,
    *,
    debug_key: str | None = None,
) -> DTensor:
    local = _dtensor_local_tensor(value)
    if _debug_dcp_template_key_enabled(debug_key):
        _log_dtensor_template("before-empty", value, local, debug_key)
        _cuda_synchronize_for_debug(local, debug_key)
    local_empty = torch.empty_strided(
        tuple(local.shape),
        tuple(local.stride()),
        dtype=local.dtype,
        device=local.device,
    )
    _initialize_optimizer_template_tensor_(local_empty)
    if _debug_dcp_template_key_enabled(debug_key):
        _log_tensor_template("after-empty", local_empty, debug_key)
        _cuda_synchronize_for_debug(local_empty, debug_key)
    return DTensor.from_local(
        local_empty,
        value.device_mesh,
        value.placements,
        shape=value.shape,
        stride=value.stride(),
    )


def _empty_tensor_like(value: torch.Tensor) -> torch.Tensor:
    tensor = torch.empty_like(value)
    _initialize_optimizer_template_tensor_(tensor)
    return tensor


def _initialize_optimizer_template_tensor_(tensor: torch.Tensor) -> None:
    fill = os.environ.get("MLITE_MFSDP_DCP_TEMPLATE_FILL", "").lower()
    if fill in {"nan", "sentinel"} and torch.is_floating_point(tensor):
        tensor.fill_(float("nan"))
    elif fill == "zero":
        tensor.zero_()


def _dtensor_local_tensor(value: DTensor) -> torch.Tensor:
    local = getattr(value, "_local_tensor", None)
    if isinstance(local, torch.Tensor):
        return local
    return value.to_local()


def _debug_state_key(base: str | None, suffix: str | None) -> str | None:
    if base is None:
        return suffix
    if not suffix:
        return base
    return f"{base}.{suffix}"


def _debug_dcp_template_enabled() -> bool:
    return os.environ.get("MLITE_MFSDP_DCP_DEBUG_TEMPLATE", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _debug_dcp_template_key_enabled(debug_key: str | None) -> bool:
    if not _debug_dcp_template_enabled():
        return False
    key_filter = os.environ.get("MLITE_MFSDP_DCP_DEBUG_TEMPLATE_FILTER", "")
    return not key_filter or (debug_key is not None and key_filter in debug_key)


def _debug_template_cuda_sync(phase: str, *, log_success: bool = False) -> None:
    if not _debug_dcp_template_enabled() or not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception as exc:
        print(
            "[MFSDP_DCP_TEMPLATE] "
            f"rank={_debug_rank()} phase={phase}-cuda-sync-error "
            f"error={type(exc).__name__}: {exc}",
            flush=True,
        )
        raise
    if log_success:
        print(
            "[MFSDP_DCP_TEMPLATE] "
            f"rank={_debug_rank()} phase={phase}-cuda-sync-ok",
            flush=True,
        )


def _debug_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return -1


def _log_dtensor_template(
    phase: str,
    value: DTensor,
    local: torch.Tensor,
    debug_key: str | None,
) -> None:
    print(
        "[MFSDP_DCP_TEMPLATE] "
        f"rank={_debug_rank()} phase={phase} key={debug_key} "
        f"global_shape={_safe_debug_value(lambda: tuple(value.shape))} "
        f"global_stride={_safe_debug_value(lambda: tuple(value.stride()))} "
        f"placements={_safe_debug_value(lambda: value.placements)} "
        f"mesh={_safe_debug_value(lambda: value.device_mesh)} "
        f"local_shape={_safe_debug_value(lambda: tuple(local.shape))} "
        f"local_stride={_safe_debug_value(lambda: tuple(local.stride()))} "
        f"dtype={_safe_debug_value(lambda: local.dtype)} "
        f"device={_safe_debug_value(lambda: local.device)} "
        f"storage_bytes={_safe_storage_bytes(local)}",
        flush=True,
    )


def _log_tensor_template(
    phase: str,
    tensor: torch.Tensor,
    debug_key: str | None,
) -> None:
    print(
        "[MFSDP_DCP_TEMPLATE] "
        f"rank={_debug_rank()} phase={phase} key={debug_key} "
        f"shape={_safe_debug_value(lambda: tuple(tensor.shape))} "
        f"stride={_safe_debug_value(lambda: tuple(tensor.stride()))} "
        f"dtype={_safe_debug_value(lambda: tensor.dtype)} "
        f"device={_safe_debug_value(lambda: tensor.device)} "
        f"storage_bytes={_safe_storage_bytes(tensor)}",
        flush=True,
    )


def _cuda_synchronize_for_debug(tensor: torch.Tensor, debug_key: str | None) -> None:
    device = getattr(tensor, "device", None)
    if device is None or device.type != "cuda":
        return
    try:
        torch.cuda.synchronize(device)
    except Exception as exc:
        print(
            "[MFSDP_DCP_TEMPLATE] "
            f"rank={_debug_rank()} phase=cuda-sync-error key={debug_key} "
            f"error={type(exc).__name__}: {exc}",
            flush=True,
        )
        raise


def _safe_storage_bytes(tensor: torch.Tensor) -> Any:
    return _safe_debug_value(lambda: tensor.untyped_storage().nbytes())


def _safe_debug_value(fn: Any) -> Any:
    try:
        return fn()
    except Exception as exc:
        return f"<{type(exc).__name__}: {exc}>"


def _load_fsdp_dtensor_parameter_state(optimizer: Any, state: Any) -> bool:
    chained_optimizers = _iter_chained_optimizers(optimizer)
    states = state if isinstance(state, list) else [state]
    if len(states) != len(chained_optimizers):
        raise ValueError(
            "Megatron-FSDP fsdp_dtensor parameter state count does not match chained optimizers: "
            f"{len(states)} != {len(chained_optimizers)}."
        )
    loaded_main_params = False
    for chained_optimizer, chained_state in zip(chained_optimizers, states):
        if chained_state is not None:
            _ensure_fsdp_dtensor_optimizer_param_names(chained_optimizer)
            chained_state = _canonicalize_fsdp_dtensor_parameter_state(
                chained_optimizer,
                chained_state,
            )
            _install_fsdp_dtensor_optimizer_state(chained_optimizer, chained_state)
            loaded_main_params = (
                _restore_fsdp_dtensor_main_params(chained_optimizer, chained_state)
                or loaded_main_params
            )
    return loaded_main_params


@torch.no_grad()
def _install_fsdp_dtensor_optimizer_state(optimizer: Any, parameter_state: Any) -> None:
    inner_optimizer = _inner_torch_optimizer(optimizer)
    if inner_optimizer is None:
        raise TypeError("Megatron-FSDP optimizer is missing an inner optimizer.")
    state_by_name = (
        parameter_state.get("state")
        if isinstance(parameter_state, dict)
        else None
    )
    if not isinstance(state_by_name, dict):
        return
    param_to_group_meta = parameter_state.get("param_to_group_meta")
    if not isinstance(param_to_group_meta, dict):
        raise TypeError("Megatron-FSDP optimizer state is missing param_to_group_meta.")

    installed_names: set[str] = set()
    optimizer_state = getattr(inner_optimizer, "state", None)
    if optimizer_state is None:
        optimizer_state = {}
        inner_optimizer.state = optimizer_state
    if not isinstance(optimizer_state, dict):
        raise TypeError("Megatron-FSDP inner optimizer state is not a dict.")
    precision_aware = _use_precision_aware_optimizer(optimizer)
    sanitize_stats = {"tensors": 0, "elements": 0}
    live_params = _iter_inner_optimizer_params(optimizer)
    for param in live_params:
        name = _optimizer_param_name(optimizer, param)
        if name is None:
            continue
        saved_state = state_by_name.get(name)
        if not isinstance(saved_state, dict):
            continue
        _install_fsdp_dtensor_param_state(
            inner_optimizer,
            optimizer_state,
            param,
            name,
            saved_state,
            precision_aware=precision_aware,
            sanitize_stats=sanitize_stats,
        )
        installed_names.add(name)
    _ensure_all_named_optimizer_states_installed(state_by_name, installed_names)
    _restore_mfsdp_inner_optimizer_param_groups(
        optimizer,
        inner_optimizer,
        param_to_group_meta,
    )
    _debug_optimizer_restore_summary(
        state_by_name=state_by_name,
        live_param_count=len(live_params),
        installed_count=len(installed_names),
        inner_state_count=len(optimizer_state),
        param_groups=getattr(inner_optimizer, "param_groups", None),
        precision_aware=precision_aware,
        sanitize_stats=sanitize_stats,
    )


def _install_fsdp_dtensor_param_state(
    inner_optimizer: Any,
    optimizer_state: dict[Any, Any],
    param: Any,
    name: str,
    saved_state: dict[str, Any],
    *,
    precision_aware: bool,
    sanitize_stats: dict[str, int],
) -> None:
    set_scaled_state = getattr(inner_optimizer, "set_scaled_state", None)
    param_state: dict[str, Any] = {}
    optimizer_state[param] = param_state
    for key, value in saved_state.items():
        if key == _MAIN_PARAM_STATE_KEY:
            continue
        _validate_optimizer_state_value(param, name, key, value)
        if torch.is_tensor(value):
            stored_value = _optimizer_state_tensor_for_inner_optimizer(
                value,
                param_name=name,
                state_key=str(key),
                sanitize_stats=sanitize_stats,
            )
            param_state[key] = stored_value
            if callable(set_scaled_state):
                set_scaled_state(param, key, _optimizer_state_value_for_te(stored_value))
        else:
            param_state[key] = _clone_state_value(value)
    _debug_optimizer_restore_param(name, param, param_state)


def _optimizer_state_tensor_for_inner_optimizer(
    value: torch.Tensor,
    *,
    param_name: str,
    state_key: str,
    sanitize_stats: dict[str, int],
) -> torch.Tensor:
    if isinstance(value, DTensor):
        local = _dtensor_local_tensor(value)
    else:
        local = value
    if state_key != "step" and torch.is_floating_point(local):
        _sanitize_optimizer_state_tensor_(
            local,
            sanitize_stats,
            param_name=param_name,
            state_key=state_key,
            source_value=value,
        )
    return local


def _sanitize_optimizer_state_tensor_(
    tensor: torch.Tensor,
    sanitize_stats: dict[str, int],
    *,
    param_name: str,
    state_key: str,
    source_value: Any,
) -> None:
    flat = tensor.detach().view(-1)
    chunk_size = _optimizer_state_sanitize_chunk_size()
    sanitized_tensor = False
    total_nonfinite = 0
    first_bad_index: int | None = None
    first_bad_count = 0
    for start in range(0, flat.numel(), chunk_size):
        chunk = flat[start : start + chunk_size]
        # Avoid materializing a full-param finite mask; Qwen3MoE restore runs near
        # the memory ceiling before optimizer state install.
        if bool(torch.isfinite(chunk).all().item()):
            continue
        if _debug_optimizer_sanitize_enabled():
            bad_mask = ~torch.isfinite(chunk)
            bad_count = int(bad_mask.sum().item())
            if first_bad_index is None:
                local_bad = torch.nonzero(bad_mask, as_tuple=False)
                first_bad_index = start + int(local_bad[0].item()) if local_bad.numel() else start
                first_bad_count = bad_count
        else:
            bad_count = chunk.numel()
        chunk.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        total_nonfinite += int(bad_count)
        sanitized_tensor = True
    if sanitized_tensor:
        sanitize_stats["tensors"] = int(sanitize_stats.get("tensors", 0)) + 1
        sanitize_stats["elements"] = int(sanitize_stats.get("elements", 0)) + total_nonfinite
        _debug_optimizer_sanitize_report(
            tensor,
            source_value,
            sanitize_stats,
            param_name=param_name,
            state_key=state_key,
            bad_elements=total_nonfinite,
            first_bad_index=first_bad_index,
            first_bad_chunk_count=first_bad_count,
        )


def _optimizer_state_sanitize_chunk_size() -> int:
    raw = os.environ.get("MLITE_MFSDP_DCP_SANITIZE_CHUNK_SIZE", "1048576")
    try:
        return max(int(raw), 1)
    except ValueError:
        return 1048576


def _debug_optimizer_sanitize_enabled() -> bool:
    return os.environ.get("MLITE_MFSDP_DCP_DEBUG_SANITIZE", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _debug_optimizer_sanitize_report(
    tensor: torch.Tensor,
    source_value: Any,
    sanitize_stats: dict[str, int],
    *,
    param_name: str,
    state_key: str,
    bad_elements: int,
    first_bad_index: int | None,
    first_bad_chunk_count: int,
) -> None:
    if not _debug_optimizer_sanitize_enabled():
        return
    max_reports = _debug_optimizer_sanitize_max_reports()
    reports = int(sanitize_stats.get("reports", 0))
    if reports >= max_reports:
        return
    sanitize_stats["reports"] = reports + 1
    source_shape = tuple(source_value.shape) if torch.is_tensor(source_value) else None
    source_local_shape = None
    source_placements = None
    if isinstance(source_value, DTensor):
        source_local_shape = tuple(_dtensor_local_tensor(source_value).shape)
        source_placements = tuple(str(placement) for placement in source_value.placements)
    print(
        "[MFSDP_DCP_RESTORE_SANITIZE] "
        f"rank={_debug_rank()} name={param_name} key={state_key} "
        f"shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device} "
        f"source_shape={source_shape} source_local_shape={source_local_shape} "
        f"source_placements={source_placements} bad_elements={bad_elements} "
        f"first_bad_index={first_bad_index} "
        f"first_bad_chunk_count={first_bad_chunk_count}",
        flush=True,
    )


def _debug_optimizer_sanitize_max_reports() -> int:
    raw = os.environ.get("MLITE_MFSDP_DCP_DEBUG_SANITIZE_MAX_REPORTS", "16")
    try:
        return max(int(raw), 0)
    except ValueError:
        return 16


def _use_precision_aware_optimizer(optimizer: Any) -> bool:
    config = getattr(optimizer, "config", None)
    return bool(getattr(config, "use_precision_aware_optimizer_no_fp8_or_ds_fp8", False))


def _optimizer_state_value_for_te(value: torch.Tensor) -> torch.Tensor:
    if value.dtype == torch.float32:
        return value
    return value.float()


def _restore_mfsdp_inner_optimizer_param_groups(
    optimizer: Any,
    inner_optimizer: Any,
    param_to_group_meta: dict[str, Any],
) -> None:
    param_groups = getattr(inner_optimizer, "param_groups", None)
    if not isinstance(param_groups, list):
        return
    for group in param_groups:
        if not isinstance(group, dict):
            continue
        group_meta = _group_meta_for_inner_optimizer_group(
            optimizer,
            group,
            param_to_group_meta,
        )
        if group_meta is None:
            continue
        for key, value in group_meta.items():
            if key != "params":
                group[key] = value


def _group_meta_for_inner_optimizer_group(
    optimizer: Any,
    group: dict[str, Any],
    param_to_group_meta: dict[str, Any],
) -> dict[str, Any] | None:
    matched_meta = None
    for param in group.get("params", ()):
        name = _optimizer_param_name(optimizer, param)
        if name is None:
            continue
        group_meta = param_to_group_meta.get(name)
        if not isinstance(group_meta, dict):
            continue
        if matched_meta is None:
            matched_meta = group_meta
            continue
        if matched_meta != group_meta:
            raise ValueError(
                "Megatron-FSDP optimizer checkpoint has inconsistent group metadata "
                f"within one live optimizer group: {name}."
            )
    return matched_meta


def _validate_optimizer_state_value(
    param: Any,
    name: str,
    state_key: str,
    value: Any,
) -> None:
    if not torch.is_tensor(value):
        return
    if state_key == "step":
        return
    expected_shape = _optimizer_state_expected_shape(param, value)
    if tuple(value.shape) != expected_shape:
        raise ValueError(
            "Megatron-FSDP optimizer checkpoint state shape mismatch for "
            f"{name}.{state_key}: checkpoint={tuple(value.shape)} expected={expected_shape}."
        )


def _optimizer_state_expected_shape(param: Any, value: Any) -> tuple[int, ...]:
    if isinstance(value, DTensor):
        return tuple(param.shape)
    if isinstance(param, DTensor):
        return tuple(_dtensor_local_tensor(param).shape)
    data = getattr(param, "data", None)
    if isinstance(data, DTensor):
        return tuple(_dtensor_local_tensor(data).shape)
    return tuple(param.shape)


def _ensure_all_named_optimizer_states_installed(
    state_by_name: dict[str, Any],
    installed_names: set[str],
) -> None:
    missing_names = sorted(
        name
        for name, saved_state in state_by_name.items()
        if isinstance(saved_state, dict)
        and any(key != _MAIN_PARAM_STATE_KEY for key in saved_state)
        and name not in installed_names
    )
    if missing_names:
        preview = ", ".join(missing_names[:5])
        if len(missing_names) > 5:
            preview = f"{preview}, ..."
        raise ValueError(
            "Megatron-FSDP optimizer checkpoint contains states that were not "
            f"matched to live optimizer parameters: {preview}"
        )


def _debug_optimizer_restore_enabled() -> bool:
    return os.environ.get("MLITE_MFSDP_DCP_DEBUG_RESTORE", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _debug_optimizer_restore_key_enabled(name: str | None) -> bool:
    if not _debug_optimizer_restore_enabled():
        return False
    key_filter = os.environ.get("MLITE_MFSDP_DCP_DEBUG_RESTORE_FILTER", "")
    return not key_filter or (name is not None and key_filter in name)


def _debug_optimizer_restore_param(
    name: str,
    param: Any,
    param_state: dict[str, Any],
) -> None:
    if not _debug_optimizer_restore_key_enabled(name):
        return
    state_parts = []
    for key, value in param_state.items():
        if torch.is_tensor(value):
            state_parts.append(
                f"{key}:shape={tuple(value.shape)},dtype={value.dtype},device={value.device}"
            )
        else:
            state_parts.append(f"{key}:type={type(value).__name__}")
    print(
        "[MFSDP_DCP_RESTORE] "
        f"rank={_debug_rank()} name={name} param_shape={_debug_shape(param)} "
        f"param_dtype={getattr(param, 'dtype', None)} states={';'.join(state_parts)}",
        flush=True,
    )


def _debug_shape(value: Any) -> Any:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return tuple(shape)


def _debug_optimizer_restore_summary(
    *,
    state_by_name: dict[str, Any],
    live_param_count: int,
    installed_count: int,
    inner_state_count: int,
    param_groups: Any,
    precision_aware: bool,
    sanitize_stats: dict[str, int],
) -> None:
    if _debug_rank() != 0:
        return
    saved_param_states = sum(
        1 for saved_state in state_by_name.values() if isinstance(saved_state, dict)
    )
    group_steps = _debug_param_group_steps(param_groups)
    print(
        "[MFSDP_DCP_RESTORE] "
        f"rank=0 summary saved_param_states={saved_param_states} "
        f"live_params={live_param_count} installed={installed_count} "
        f"inner_optimizer_state={inner_state_count} "
        f"precision_aware={precision_aware} group_steps={group_steps} "
        f"sanitized_tensors={int(sanitize_stats.get('tensors', 0))} "
        f"sanitized_elements={int(sanitize_stats.get('elements', 0))}",
        flush=True,
    )


def _debug_param_group_steps(param_groups: Any) -> Any:
    if not isinstance(param_groups, list):
        return None
    steps = []
    for group in param_groups:
        if not isinstance(group, dict):
            continue
        step = group.get("step")
        if torch.is_tensor(step):
            step = step.item()
        if step not in steps:
            steps.append(step)
    return steps


def _with_fsdp_dtensor_main_params(
    parameter_state: Any,
    optimizer: Any,
    *,
    clone_main_params: bool,
) -> Any:
    if not isinstance(parameter_state, dict):
        return parameter_state
    state = parameter_state.get("state")
    if not isinstance(state, dict):
        return parameter_state

    updated_state = dict(state)
    changed = False
    for param in _iter_inner_optimizer_params(optimizer):
        name = _optimizer_param_name(optimizer, param)
        if name is None:
            continue
        param_state = updated_state.get(name)
        if not isinstance(param_state, dict):
            continue
        param_state = dict(param_state)
        param_state[_MAIN_PARAM_STATE_KEY] = (
            _clone_state_value(param)
            if clone_main_params
            else _checkpoint_tensor_view(param)
        )
        updated_state[name] = param_state
        changed = True

    if not changed:
        return parameter_state
    updated_parameter_state = dict(parameter_state)
    updated_parameter_state["state"] = updated_state
    return updated_parameter_state


def _checkpoint_tensor_view(value: Any) -> Any:
    """Return a non-owning tensor view suitable for immediate DCP save.

    Qwen3MoE M-FSDP runs too close to the memory ceiling to clone fp32 master
    params while building the optimizer state tree. DCP only needs a stable
    read-only view during ``dcp.save``.
    """
    if isinstance(value, DTensor):
        local = _dtensor_local_tensor(value).detach()
        return DTensor.from_local(
            local,
            value.device_mesh,
            value.placements,
            shape=value.shape,
            stride=value.stride(),
        )
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, dict):
        return {key: _checkpoint_tensor_view(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_checkpoint_tensor_view(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_checkpoint_tensor_view(item) for item in value)
    return value


def _without_fsdp_dtensor_main_params(parameter_state: Any) -> Any:
    if not isinstance(parameter_state, dict):
        return parameter_state
    state = parameter_state.get("state")
    if not isinstance(state, dict):
        return parameter_state

    changed = False
    updated_state = {}
    for name, param_state in state.items():
        if not isinstance(param_state, dict) or _MAIN_PARAM_STATE_KEY not in param_state:
            updated_state[name] = param_state
            continue
        updated_param_state = dict(param_state)
        del updated_param_state[_MAIN_PARAM_STATE_KEY]
        updated_state[name] = updated_param_state
        changed = True

    if not changed:
        return parameter_state
    updated_parameter_state = dict(parameter_state)
    updated_parameter_state["state"] = updated_state
    return updated_parameter_state


def _use_local_rank_dcp_optimizer_state() -> bool:
    return os.environ.get(
        "MLITE_MFSDP_DCP_OPTIMIZER_LOCAL_SHARDS",
        "1",
    ).lower() not in {"0", "false", "no", "off", ""}


def _local_rank_fsdp_dtensor_parameter_state_for_dcp(parameter_state: Any) -> dict[str, Any]:
    """Store M-FSDP optimizer shards as rank-local tensors for DCP.

    Megatron-FSDP optimizer params are slices of flattened FSDP buffers. Their
    local tensor shapes do not necessarily form a legal Shard(dim) partition of
    the original parameter shape, so representing them as normal DTensors lets
    DCP restore only the regular shard prefix. Rank-local keys preserve the
    exact production resume semantics for the current topology, matching the
    model local-shard DCP path.
    """
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    return {
        "rank": rank,
        "world_size": world_size,
        _LOCAL_RANK_STATE_KEY: {
            f"rank{rank}": _local_rank_checkpoint_tensor_view(parameter_state)
        },
    }


def _current_local_rank_parameter_state(parameter_state: Any) -> Any:
    if not isinstance(parameter_state, dict):
        return parameter_state
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    rank_state = parameter_state.get(_LOCAL_RANK_STATE_KEY)
    if not isinstance(rank_state, dict):
        raise TypeError("Megatron-FSDP local-rank optimizer checkpoint is missing rank_state.")
    key = f"rank{rank}"
    if key not in rank_state:
        available = ", ".join(sorted(str(item) for item in rank_state)[:8])
        raise KeyError(
            "Megatron-FSDP local-rank optimizer checkpoint does not contain "
            f"{key}; available keys: {available}"
        )
    saved_world_size = parameter_state.get("world_size")
    if (
        saved_world_size is not None
        and dist.is_available()
        and dist.is_initialized()
        and int(saved_world_size) != dist.get_world_size()
    ):
        raise ValueError(
            "Megatron-FSDP local-rank optimizer checkpoint requires the same world size: "
            f"checkpoint={saved_world_size} current={dist.get_world_size()}."
        )
    return rank_state[key]


def _local_rank_checkpoint_tensor_view(value: Any) -> Any:
    if isinstance(value, DTensor):
        return _dtensor_local_tensor(value).detach()
    data = getattr(value, "data", None)
    if isinstance(data, DTensor):
        return _dtensor_local_tensor(data).detach()
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, dict):
        return {key: _local_rank_checkpoint_tensor_view(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_local_rank_checkpoint_tensor_view(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_local_rank_checkpoint_tensor_view(item) for item in value)
    return value


def _preprocess_fsdp_dtensor_parameter_state_for_dcp(parameter_state: Any) -> None:
    if preprocess_state_dict_for_uneven_dtensor is None:
        _debug_uneven_dtensor_preprocess("missing", parameter_state)
        return
    if isinstance(parameter_state, list):
        for item in parameter_state:
            _preprocess_fsdp_dtensor_parameter_state_for_dcp(item)
        return
    if not isinstance(parameter_state, dict):
        return
    try:
        _debug_uneven_dtensor_preprocess("before", parameter_state)
        preprocess_state_dict_for_uneven_dtensor(parameter_state)
        _debug_uneven_dtensor_preprocess("after", parameter_state)
    except Exception as exc:
        print(
            "[MFSDP_DCP_RESTORE] "
            f"rank={_debug_rank()} uneven_dtensor_preprocess_error="
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        raise


def _debug_uneven_dtensor_preprocess(stage: str, parameter_state: Any) -> None:
    if not _debug_uneven_dtensor_preprocess_enabled() or _debug_rank() != 0:
        return
    print(
        "[MFSDP_DCP_PREPROCESS] "
        f"rank=0 stage={stage} dtensors={_count_dtensors(parameter_state)}",
        flush=True,
    )


def _debug_uneven_dtensor_preprocess_enabled() -> bool:
    return os.environ.get("MLITE_MFSDP_DCP_DEBUG_PREPROCESS", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _count_dtensors(value: Any) -> int:
    if isinstance(value, DTensor):
        return 1
    if isinstance(value, dict):
        return sum(_count_dtensors(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_count_dtensors(item) for item in value)
    return 0


def _state_dict_has_main_params(state_dict: Any) -> bool:
    if not isinstance(state_dict, dict):
        return False
    if _INNER_OPTIMIZER_STATE_KEY in state_dict:
        return True
    parameter_state = state_dict.get(_PARAMETER_STATE_KEY)
    if parameter_state is None:
        return False
    parameter_states = parameter_state if isinstance(parameter_state, list) else [parameter_state]
    for item in parameter_states:
        if _parameter_state_has_main_params(item):
            return True
    return False


def _parameter_state_has_main_params(parameter_state: Any) -> bool:
    if isinstance(parameter_state, list):
        return any(_parameter_state_has_main_params(item) for item in parameter_state)
    if not isinstance(parameter_state, dict):
        return False
    rank_state = parameter_state.get(_LOCAL_RANK_STATE_KEY)
    if isinstance(rank_state, dict):
        return any(_parameter_state_has_main_params(item) for item in rank_state.values())
    state = parameter_state.get("state")
    if not isinstance(state, dict):
        return False
    return any(
        isinstance(param_state, dict) and _MAIN_PARAM_STATE_KEY in param_state
        for param_state in state.values()
    )


def _restore_fsdp_dtensor_main_params(optimizer: Any, parameter_state: Any) -> bool:
    if not isinstance(parameter_state, dict):
        return False
    state = parameter_state.get("state")
    if not isinstance(state, dict):
        return False

    restored_any = False
    restored_params = 0
    restored_buffers = 0
    for param in _iter_inner_optimizer_params(optimizer):
        name = _optimizer_param_name(optimizer, param)
        if name is None:
            continue
        param_state = state.get(name)
        if not isinstance(param_state, dict):
            continue
        saved_param = param_state.get(_MAIN_PARAM_STATE_KEY)
        if saved_param is None:
            continue
        _copy_state_value_(param, saved_param)
        restored_params += 1
        if _restore_mfsdp_buffer_main_weight_from_param(optimizer, param, saved_param):
            restored_buffers += 1
        restored_any = True
    _debug_main_param_restore_summary(restored_params, restored_buffers)
    return restored_any


def _restore_mfsdp_buffer_main_weight_from_param(
    optimizer: Any,
    param: Any,
    saved_param: Any,
) -> bool:
    orig_param = getattr(param, "orig_param", None)
    if orig_param is None:
        return False
    source = _tensor_local_data(saved_param)
    for buffer in _iter_mfsdp_param_and_grad_buffers(optimizer):
        if _restore_mfsdp_buffer_main_weight(buffer, orig_param, source):
            return True
    return False


def _restore_mfsdp_buffer_main_weight(
    buffer: Any,
    orig_param: Any,
    source: torch.Tensor,
) -> bool:
    param_to_group = getattr(buffer, "param_to_param_group", None)
    if not isinstance(param_to_group, dict) or orig_param not in param_to_group:
        return False
    parameter_groups = getattr(buffer, "parameter_groups", None)
    try:
        group = parameter_groups[param_to_group[orig_param]]
        main_weight_buffer = group.main_weight_buffer
    except (TypeError, AttributeError, IndexError, KeyError):
        return False
    if main_weight_buffer is None:
        return False
    main_param_idx = getattr(main_weight_buffer, "param_idx", None)
    if not isinstance(main_param_idx, dict) or orig_param not in main_param_idx:
        return False
    item_id = main_param_idx[orig_param]
    model_weight_buffer = getattr(group, "model_weight_buffer", None)
    use_shard = bool(
        (model_weight_buffer is not None and getattr(model_weight_buffer, "is_data_distributed", False))
        or getattr(main_weight_buffer, "is_data_distributed", False)
    )
    main_weight = main_weight_buffer.get_item(item_id, only_shard=use_shard)
    _copy_tensor_value_reshaped_(main_weight, source)
    return True


def _iter_mfsdp_param_and_grad_buffers(optimizer: Any) -> list[Any]:
    buffers = []
    seen: set[int] = set()
    for model_chunk in getattr(optimizer, "model_chunks", ()) or ():
        for owner in (model_chunk, getattr(model_chunk, "module", None)):
            if owner is None:
                continue
            buffer = getattr(owner, "param_and_grad_buffer", None)
            if buffer is None or id(buffer) in seen:
                continue
            seen.add(id(buffer))
            buffers.append(buffer)
    return buffers


def _debug_main_param_restore_summary(restored_params: int, restored_buffers: int) -> None:
    if _debug_rank() != 0:
        return
    if restored_params == 0:
        return
    print(
        "[MFSDP_DCP_RESTORE] "
        f"rank=0 main_params_restored={restored_params} "
        f"main_weight_buffers_restored={restored_buffers}",
        flush=True,
    )


def _with_mcore_optimizer_group_meta(parameter_state: Any, non_parameter_state: Any) -> Any:
    if isinstance(parameter_state, list):
        non_parameter_states = non_parameter_state if isinstance(non_parameter_state, list) else []
        if len(non_parameter_states) != len(parameter_state):
            return parameter_state
        return [
            _with_single_mcore_optimizer_group_meta(param_state, non_param_state)
            for param_state, non_param_state in zip(
                parameter_state,
                non_parameter_states,
                strict=True,
            )
        ]
    return _with_single_mcore_optimizer_group_meta(parameter_state, non_parameter_state)


def _with_single_mcore_optimizer_group_meta(parameter_state: Any, non_parameter_state: Any) -> Any:
    if not isinstance(parameter_state, dict):
        return parameter_state
    param_to_group_meta = parameter_state.get("param_to_group_meta")
    if not isinstance(param_to_group_meta, dict):
        return parameter_state

    optimizer_group_metas = _mcore_optimizer_group_metas(non_parameter_state)
    if not optimizer_group_metas:
        return parameter_state

    updated_param_to_group_meta = {}
    changed = False
    for param_name, group_meta in param_to_group_meta.items():
        if not isinstance(group_meta, dict):
            updated_param_to_group_meta[param_name] = group_meta
            continue
        optimizer_group_meta = _match_optimizer_group_meta(group_meta, optimizer_group_metas)
        if optimizer_group_meta is None:
            updated_param_to_group_meta[param_name] = group_meta
            continue
        merged_group_meta = dict(group_meta)
        for key, value in optimizer_group_meta.items():
            if key == "params":
                continue
            if key not in merged_group_meta or not _metadata_values_equal(
                merged_group_meta[key],
                value,
            ):
                merged_group_meta[key] = value
                changed = True
        updated_param_to_group_meta[param_name] = merged_group_meta

    if not changed:
        return parameter_state
    updated_parameter_state = dict(parameter_state)
    updated_parameter_state["param_to_group_meta"] = updated_param_to_group_meta
    return updated_parameter_state


def _mcore_optimizer_group_metas(non_parameter_state: Any) -> list[dict[str, Any]]:
    if not isinstance(non_parameter_state, dict):
        return []
    optimizer_state = non_parameter_state.get("optimizer")
    if not isinstance(optimizer_state, dict):
        return []
    param_groups = optimizer_state.get("param_groups")
    if not isinstance(param_groups, list):
        return []
    return [group for group in param_groups if isinstance(group, dict)]


def _match_optimizer_group_meta(
    parameter_group_meta: dict[str, Any],
    optimizer_group_metas: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if len(optimizer_group_metas) == 1:
        return optimizer_group_metas[0]
    for optimizer_group_meta in optimizer_group_metas:
        common_keys = (
            set(parameter_group_meta)
            & set(optimizer_group_meta)
            - {"params"}
        )
        if common_keys and all(
            _metadata_values_equal(parameter_group_meta[key], optimizer_group_meta[key])
            for key in common_keys
        ):
            return optimizer_group_meta
    return None


def _metadata_values_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _metadata_values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _metadata_values_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return left == right


def _ensure_fsdp_dtensor_optimizer_param_names(optimizer: Any) -> None:
    """Teach MCore's fsdp_dtensor state path about optimizer DTensor params.

    Megatron-FSDP builds torch optimizer param groups from DTensor-wrapped
    optimizer parameters. MCore's ``_param_name`` lazily maps original model
    parameters to names, so its fsdp_dtensor state path can miss the optimizer
    parameters unless the wrapper mirrors each ``orig_param`` name onto the
    DTensor parameter.
    """
    inner_optimizer = _inner_torch_optimizer(optimizer)
    if inner_optimizer is None or not hasattr(inner_optimizer, "param_groups"):
        return
    mapping = getattr(optimizer, "param_to_name", None)
    if not isinstance(mapping, dict):
        mapping = {}
        setattr(optimizer, "param_to_name", mapping)
    model_chunk_names = _model_chunk_param_names(optimizer)
    canonicalize = _optimizer_param_name_canonicalizer(optimizer)
    for group in inner_optimizer.param_groups:
        for param in group.get("params", ()):
            orig_param = getattr(param, "orig_param", None)
            name = _optimizer_param_name_candidate(
                optimizer,
                param,
                orig_param=orig_param,
                mapping=mapping,
                model_chunk_names=model_chunk_names,
            )
            if name is None:
                continue
            if canonicalize is not None:
                name = canonicalize(name)
            mapping[param] = name
            if orig_param is not None:
                mapping[orig_param] = name


def _model_chunk_param_names(optimizer: Any) -> dict[int, str]:
    param_names: dict[int, str] = {}
    model_chunks = getattr(optimizer, "model_chunks", ()) or ()
    for chunk in model_chunks:
        for owner in (chunk, getattr(chunk, "module", None)):
            if owner is None:
                continue
            buffer = getattr(owner, "param_and_grad_buffer", None)
            named_parameters = getattr(buffer, "optimizer_named_parameters", None)
            if named_parameters is None:
                continue
            for name, param in named_parameters:
                state_name = _mcore_fsdp_state_name(name)
                param_names.setdefault(id(param), state_name)
                orig_param = getattr(param, "orig_param", None)
                if orig_param is not None:
                    param_names.setdefault(id(orig_param), state_name)
                data = getattr(param, "data", None)
                if data is not None:
                    param_names.setdefault(id(data), state_name)
    if param_names:
        return param_names

    for chunk in model_chunks:
        named_parameters = getattr(chunk, "named_parameters", None)
        if callable(named_parameters):
            for name, param in named_parameters():
                param_names.setdefault(id(param), name)
                data = getattr(param, "data", None)
                if data is not None:
                    param_names.setdefault(id(data), name)
    return param_names


def _mcore_fsdp_state_name(name: str) -> str:
    if name.startswith("module.module."):
        return name
    return f"module.module.{name}"


def _optimizer_param_name_candidate(
    optimizer: Any,
    param: Any,
    *,
    orig_param: Any,
    mapping: dict[Any, str],
    model_chunk_names: dict[int, str],
) -> str | None:
    for candidate in (
        model_chunk_names.get(id(param)),
        model_chunk_names.get(id(orig_param)) if orig_param is not None else None,
        mapping.get(param),
        mapping.get(orig_param) if orig_param is not None else None,
        _attr_param_name(param),
        _attr_param_name(orig_param),
    ):
        if candidate:
            return candidate
    if _mcore_param_name_fallback_enabled():
        for candidate in (
            _safe_mcore_param_name(optimizer, param),
            _safe_mcore_param_name(optimizer, orig_param),
        ):
            if candidate:
                return candidate
    return None


def _mcore_param_name_fallback_enabled() -> bool:
    return os.environ.get(
        "MLITE_MFSDP_ALLOW_MCORE_PARAM_NAME_FALLBACK", ""
    ).lower() in {"1", "true", "yes", "on"}


def _attr_param_name(param: Any) -> str | None:
    name = getattr(param, PARAM_NAME_ATTR, None)
    return name if isinstance(name, str) and name else None


def _safe_mcore_param_name(optimizer: Any, param: Any) -> str | None:
    if param is None:
        return None
    param_name = getattr(optimizer, "_param_name", None)
    if not callable(param_name):
        return None
    try:
        return param_name(param)
    except (AssertionError, RuntimeError):
        return None


def _iter_inner_optimizer_params(optimizer: Any) -> list[Any]:
    inner_optimizer = _inner_torch_optimizer(optimizer)
    if inner_optimizer is None or not hasattr(inner_optimizer, "param_groups"):
        return []
    return [
        param
        for group in inner_optimizer.param_groups
        for param in group.get("params", ())
    ]


def _optimizer_param_name(optimizer: Any, param: Any) -> str | None:
    mapping = getattr(optimizer, "param_to_name", None)
    name = mapping.get(param) if isinstance(mapping, dict) else None
    if name is not None:
        return name
    return _safe_mcore_param_name(optimizer, param)


__all__ = ["BACKEND", "MegatronFSDPBackend", "is_megatron_fsdp_optimizer"]
