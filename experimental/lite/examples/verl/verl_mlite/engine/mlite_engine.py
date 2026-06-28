# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""External VERL engine backed by Megatron Lite runtime primitives."""

from __future__ import annotations

import copy
import math
import os
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from megatron.lite.model import resolve_model_type_from_hf
from megatron.lite.primitive.ckpt import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from megatron.lite.primitive.protocols import (
    default_expert_classifier,
    default_placement_fn,
)
from megatron.lite.runtime import create_runtime
from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
from megatron.lite.runtime.contracts import LossContext, PackedBatch
from megatron.lite.runtime.contracts.config import (
    OptimizerConfig as MegatronLiteOptimizerConfig,
)
from megatron.lite.runtime.contracts.config import ParallelConfig, RuntimeConfig
from tensordict import TensorDict
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_id, get_device_name
from verl.workers.config import HFModelConfig, OptimizerConfig
from verl_mlite.compat import load_verl_engine_api

try:
    # Recent VERL wraps per-step metric values in a Metric aggregator that
    # reduce_metrics() knows how to fold; older VERL expects list-of-scalars.
    from verl.utils.metric import Metric as _VerlMetric
except Exception:  # pragma: no cover - older VERL without Metric
    _VerlMetric = None

from .config import MegatronLiteEngineConfig

(
    BaseEngine,
    BaseEngineCtx,
    EngineRegistry,
    postprocess_batch_func,
    prepare_micro_batches,
) = load_verl_engine_api()

try:
    from verl.utils.dataset.dataset_utils import DatasetPadMode
except ImportError:

    class DatasetPadMode(Enum):
        NO_PADDING = "no_padding"


_LR_SCHEDULER_STATE = "lr_scheduler.pt"
_LR_SCHEDULER_FORMAT = "megatron_lite.lr_scheduler.v1"
_LR_SCHEDULER_PAYLOAD_FORMAT = "megatron_lite.lr_scheduler_sidecar.v1"
_LR_SCHEDULER_CONFIG_FIELDS = (
    "init_lr",
    "max_lr",
    "min_lr",
    "lr_warmup_steps",
    "lr_decay_steps",
    "lr_decay_style",
    "start_wd",
    "end_wd",
    "wd_incr_steps",
    "wd_incr_style",
    "wsd_decay_steps",
    "lr_wsd_decay_style",
)
_CHECKPOINT_CONTENT_KEYS = frozenset({"model", "optimizer", "extra"})


def _content_set(contents: Any, *, key: str = "checkpoint contents") -> set[str]:
    """Normalize VERL/Hydra checkpoint content values without substring matches."""

    def normalize_entry(item: str) -> str:
        return item.strip().strip("'\"").strip()

    if contents is None:
        return set()
    if isinstance(contents, str):
        contents = contents.strip()
        if contents.startswith("[") and contents.endswith("]"):
            contents = contents[1:-1].strip()
        if "," in contents:
            result = {
                entry
                for item in contents.split(",")
                if (entry := normalize_entry(item))
            }
        else:
            entry = normalize_entry(contents)
            result = {entry} if entry else set()
    else:
        if isinstance(contents, Mapping):
            raise TypeError(f"{key} must be None, a string, or a sequence of strings.")
        try:
            iterator = iter(contents)
        except TypeError as exc:
            raise TypeError(
                f"{key} must be None, a string, or a sequence of strings."
            ) from exc
        result = set()
        for item in iterator:
            if not isinstance(item, str):
                raise TypeError(
                    f"{key} entries must be strings, got {type(item).__name__}."
                )
            entry = normalize_entry(item)
            if entry:
                result.add(entry)

    unknown = sorted(result - _CHECKPOINT_CONTENT_KEYS)
    if unknown:
        raise ValueError(
            f"{key} contains unsupported entries {unknown}; "
            f"allowed={sorted(_CHECKPOINT_CONTENT_KEYS)}."
        )
    return result


def _content_set_with_consensus(contents: Any, *, key: str) -> set[str]:
    result: set[str] = set()
    local_error: str | None = None
    try:
        result = _content_set(contents, key=key)
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    if dist.is_initialized():
        errors: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(errors, local_error)
        first_error = next((error for error in errors if error is not None), None)
        if first_error is not None:
            raise RuntimeError(f"Megatron Lite {key} validation failed: {first_error}")
        per_rank: list[tuple[str, ...] | None] = [None] * dist.get_world_size()
        normalized = tuple(sorted(result))
        dist.all_gather_object(per_rank, normalized)
        if any(item != per_rank[0] for item in per_rank[1:]):
            raise RuntimeError(f"Megatron Lite {key} differs across ranks: {per_rank}.")
    elif local_error is not None:
        raise RuntimeError(f"Megatron Lite {key} validation failed: {local_error}")
    return result


def _validate_lr_scheduler_state(state: Any) -> None:
    """Reject incomplete or ambiguous scheduler state before core restore."""

    if not isinstance(state, dict):
        raise TypeError(
            "Megatron Lite LR scheduler checkpoint state must be a dictionary, "
            f"got {type(state).__name__}."
        )
    expected_keys = {"format", "num_steps", "config"}
    if set(state) != expected_keys:
        raise ValueError(
            "Megatron Lite LR scheduler checkpoint state has an invalid schema: "
            f"missing={sorted(expected_keys - set(state))}, "
            f"unexpected={sorted(set(state) - expected_keys)}."
        )
    if state["format"] != _LR_SCHEDULER_FORMAT:
        raise ValueError(
            "Unsupported Megatron Lite LR scheduler checkpoint format: "
            f"{state['format']!r}."
        )
    num_steps = state["num_steps"]
    if isinstance(num_steps, bool) or not isinstance(num_steps, int):
        raise TypeError(
            "Megatron Lite LR scheduler checkpoint step must be a non-negative integer."
        )
    if num_steps < 0:
        raise ValueError(
            "Megatron Lite LR scheduler checkpoint step must be a non-negative integer."
        )
    config = state["config"]
    if not isinstance(config, dict) or set(config) != set(_LR_SCHEDULER_CONFIG_FIELDS):
        raise ValueError(
            "Megatron Lite LR scheduler checkpoint config has an invalid schema."
        )
    int_fields = ("lr_warmup_steps", "lr_decay_steps", "wd_incr_steps")
    for name in int_fields:
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TypeError(
                f"Megatron Lite LR scheduler config {name} must be a non-negative integer."
            )
    if config["lr_decay_steps"] < config["lr_warmup_steps"] + 1:
        raise ValueError(
            "Megatron Lite LR scheduler config lr_decay_steps must be greater "
            "than lr_warmup_steps."
        )
    if config["wd_incr_steps"] < 1:
        raise ValueError(
            "Megatron Lite LR scheduler config wd_incr_steps must be at least one."
        )
    wsd_decay_steps = config["wsd_decay_steps"]
    if wsd_decay_steps is not None and (
        isinstance(wsd_decay_steps, bool)
        or not isinstance(wsd_decay_steps, int)
        or wsd_decay_steps < 0
    ):
        raise TypeError(
            "Megatron Lite LR scheduler config wsd_decay_steps must be None or "
            "a non-negative integer."
        )
    float_fields = ("init_lr", "max_lr", "min_lr", "start_wd", "end_wd")
    for name in float_fields:
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"Megatron Lite LR scheduler config {name} must be a finite number."
            )
        if not math.isfinite(float(value)):
            raise ValueError(
                f"Megatron Lite LR scheduler config {name} must be finite."
            )
    if config["min_lr"] < 0.0:
        raise ValueError("Megatron Lite LR scheduler min_lr must be non-negative.")
    if config["max_lr"] < config["min_lr"]:
        raise ValueError(
            "Megatron Lite LR scheduler max_lr must be greater than or equal to min_lr."
        )
    if config["init_lr"] < 0.0 or config["init_lr"] > config["max_lr"]:
        raise ValueError(
            "Megatron Lite LR scheduler init_lr must be non-negative and no greater "
            "than max_lr."
        )
    if config["start_wd"] < 0.0:
        raise ValueError("Megatron Lite LR scheduler start_wd must be non-negative.")
    if config["end_wd"] < config["start_wd"]:
        raise ValueError(
            "Megatron Lite LR scheduler end_wd must be greater than or equal to start_wd."
        )
    lr_styles = {"constant", "linear", "cosine", "inverse-square-root", "wsd"}
    wd_styles = {"constant", "linear", "cosine"}
    wsd_styles = {"linear", "cosine", "exponential", "minus_sqrt"}
    if config["lr_decay_style"] not in lr_styles:
        raise ValueError(
            "Megatron Lite LR scheduler checkpoint has unsupported lr_decay_style "
            f"{config['lr_decay_style']!r}."
        )
    if config["lr_decay_style"] == "wsd":
        max_wsd_steps = config["lr_decay_steps"] - config["lr_warmup_steps"]
        if wsd_decay_steps is None or not 1 <= wsd_decay_steps <= max_wsd_steps:
            raise ValueError(
                "Megatron Lite WSD scheduler requires 1 <= wsd_decay_steps <= "
                "lr_decay_steps - lr_warmup_steps."
            )
    if config["wd_incr_style"] not in wd_styles:
        raise ValueError(
            "Megatron Lite LR scheduler checkpoint has unsupported wd_incr_style "
            f"{config['wd_incr_style']!r}."
        )
    if config["lr_wsd_decay_style"] not in wsd_styles:
        raise ValueError(
            "Megatron Lite LR scheduler checkpoint has unsupported lr_wsd_decay_style "
            f"{config['lr_wsd_decay_style']!r}."
        )


def _scheduler_state_with_consensus(scheduler) -> dict[str, Any]:
    state: dict[str, Any] = {}
    local_error: str | None = None
    try:
        state = scheduler.state_dict()
        _validate_lr_scheduler_state(state)
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    if dist.is_initialized():
        errors: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(errors, local_error)
        first_error = next((error for error in errors if error is not None), None)
        if first_error is not None:
            raise RuntimeError(
                "Megatron Lite LR scheduler state serialization failed on at least "
                f"one rank: {first_error}"
            )
        per_rank_states: list[dict[str, Any] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(per_rank_states, state)
        if any(item != per_rank_states[0] for item in per_rank_states[1:]):
            raise RuntimeError(
                "Megatron Lite LR scheduler state differs across ranks: "
                f"states={per_rank_states}."
            )
    elif local_error is not None:
        raise RuntimeError(
            f"Megatron Lite LR scheduler state serialization failed: {local_error}"
        )
    return state


def _scheduler_payload(scheduler, *, checkpoint_step: int) -> dict[str, Any]:
    scheduler_state = _scheduler_state_with_consensus(scheduler)
    payload: dict[str, Any] = {}
    local_error: str | None = None
    try:
        payload = {
            "format": _LR_SCHEDULER_PAYLOAD_FORMAT,
            "checkpoint_step": int(checkpoint_step),
            "scheduler_state": scheduler_state,
        }
        _validate_lr_scheduler_payload(payload, expected_step=checkpoint_step)
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _distributed_raise_if_error(
        local_error, context="LR scheduler checkpoint payload validation failed"
    )
    if dist.is_initialized():
        per_rank_payloads: list[dict[str, Any] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(per_rank_payloads, payload)
        if any(item != per_rank_payloads[0] for item in per_rank_payloads[1:]):
            raise RuntimeError(
                "Megatron Lite LR scheduler checkpoint payload differs across ranks: "
                f"{per_rank_payloads}."
            )
    return payload


def _validate_lr_scheduler_payload(
    payload: Any, *, expected_step: int | None = None
) -> None:
    if not isinstance(payload, dict):
        raise TypeError(
            "Megatron Lite LR scheduler sidecar must be a dictionary, "
            f"got {type(payload).__name__}."
        )
    expected_keys = {"format", "checkpoint_step", "scheduler_state"}
    if set(payload) != expected_keys:
        raise ValueError(
            "Megatron Lite LR scheduler sidecar has an invalid schema: "
            f"missing={sorted(expected_keys - set(payload))}, "
            f"unexpected={sorted(set(payload) - expected_keys)}."
        )
    if payload["format"] != _LR_SCHEDULER_PAYLOAD_FORMAT:
        raise ValueError(
            "Unsupported Megatron Lite LR scheduler sidecar format: "
            f"{payload['format']!r}."
        )
    checkpoint_step = payload["checkpoint_step"]
    if (
        isinstance(checkpoint_step, bool)
        or not isinstance(checkpoint_step, int)
        or checkpoint_step < 0
    ):
        raise TypeError(
            "Megatron Lite LR scheduler sidecar checkpoint_step must be a "
            "non-negative integer."
        )
    if expected_step is not None and checkpoint_step != expected_step:
        raise RuntimeError(
            "Megatron Lite LR scheduler sidecar/core step mismatch: "
            f"scheduler={checkpoint_step}, core={expected_step}."
        )
    _validate_lr_scheduler_state(payload["scheduler_state"])
    scheduler_step = payload["scheduler_state"]["num_steps"]
    if scheduler_step != checkpoint_step:
        raise RuntimeError(
            "Megatron Lite LR scheduler progress/core step mismatch: "
            f"scheduler={scheduler_step}, core={checkpoint_step}."
        )


def _checkpoint_components_with_consensus(
    *, model: bool, optimizer: bool, extra: bool, scheduler_present: bool, context: str
) -> None:
    if not dist.is_initialized():
        return
    local = (bool(model), bool(optimizer), bool(extra), bool(scheduler_present))
    per_rank: list[tuple[bool, bool, bool, bool] | None] = [
        None
    ] * dist.get_world_size()
    dist.all_gather_object(per_rank, local)
    if any(item != per_rank[0] for item in per_rank[1:]):
        raise RuntimeError(
            f"Megatron Lite {context} component policy differs across ranks: "
            f"{per_rank}."
        )


def _distributed_raise_if_error(local_error: str | None, *, context: str) -> None:
    if dist.is_initialized():
        errors: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(errors, local_error)
        first_error = next((error for error in errors if error is not None), None)
        if first_error is not None:
            raise RuntimeError(f"Megatron Lite {context}: {first_error}")
    elif local_error is not None:
        raise RuntimeError(f"Megatron Lite {context}: {local_error}")


def _isolate_compile_cache_per_rank() -> None:
    """Avoid torchinductor/triton cache races between local torchrun ranks."""
    rank = os.environ.get("LOCAL_RANK") or os.environ.get("RANK")
    if rank is None:
        return
    for var in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        base = os.environ.get(var)
        if not base:
            continue
        base_var = f"VERL_MLITE_BASE_{var}"
        root = os.environ.setdefault(base_var, base)
        rank_dir = os.path.join(root, f"rank_{rank}")
        os.makedirs(rank_dir, exist_ok=True)
        os.environ[var] = rank_dir


def _is_no_padding_pad_mode(pad_mode: Any) -> bool:
    return (
        pad_mode == DatasetPadMode.NO_PADDING
        or getattr(pad_mode, "name", None) == "NO_PADDING"
        or getattr(pad_mode, "value", None) == "no_padding"
        or str(pad_mode) in {"no_padding", "DatasetPadMode.NO_PADDING"}
    )


class _MegatronLiteLRScheduler:
    def __init__(
        self,
        optimizer,
        *,
        init_lr: float,
        max_lr: float,
        min_lr: float,
        lr_warmup_steps: int,
        lr_decay_steps: int,
        lr_decay_style: str,
        start_wd: float,
        end_wd: float,
        wd_incr_steps: int,
        wd_incr_style: str,
        wsd_decay_steps: int | None,
        lr_wsd_decay_style: str,
        use_checkpoint_config: bool,
    ):
        self.optimizer = optimizer
        if not isinstance(use_checkpoint_config, bool):
            raise TypeError("use_checkpoint_config must be bool.")
        self.use_checkpoint_config = use_checkpoint_config
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_decay_steps = lr_decay_steps
        self.lr_decay_style = lr_decay_style.lower()
        if self.lr_decay_style == "inverse_square_root":
            self.lr_decay_style = "inverse-square-root"
        self.start_wd = start_wd
        self.end_wd = end_wd
        self.wd_incr_steps = wd_incr_steps
        self.wd_incr_style = wd_incr_style.lower()
        self.wsd_decay_steps = wsd_decay_steps
        self.lr_wsd_decay_style = lr_wsd_decay_style.lower()
        self.num_steps = 0
        _validate_lr_scheduler_state(self.state_dict())
        self._apply()

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": _LR_SCHEDULER_FORMAT,
            "num_steps": self.num_steps,
            "config": self._config_state(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        _validate_lr_scheduler_state(state)
        checkpoint_config = state["config"]
        if self.use_checkpoint_config:
            self._set_config_state(checkpoint_config)
        # Match VERL's MCore wrapper semantics: VERL passes
        # override_opt_param_scheduler=not use_checkpoint_opt_param_scheduler.
        # Thus false deliberately keeps runtime config (for example when
        # extending total_training_steps), while true adopts checkpoint config.
        self.num_steps = state["num_steps"]
        self._apply()

    def _config_state(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in _LR_SCHEDULER_CONFIG_FIELDS}

    def _set_config_state(self, config: dict[str, Any]) -> None:
        for name in _LR_SCHEDULER_CONFIG_FIELDS:
            setattr(self, name, config[name])

    def step(self, increment: int = 1) -> None:
        if (
            isinstance(increment, bool)
            or not isinstance(increment, int)
            or increment < 0
        ):
            raise ValueError("LR scheduler increment must be a non-negative integer.")
        self.num_steps += increment
        self._apply()

    def get_last_lr(self) -> list[float]:
        return [group["lr"] for group in self.optimizer.param_groups]

    def _apply(self) -> None:
        updates: list[tuple[dict[str, Any], float, float]] = []
        for param_group in self.optimizer.param_groups:
            lr = self._get_lr(param_group)
            wd_mult = self._finite_group_number(
                param_group, "wd_mult", default=1.0, non_negative=True
            )
            updates.append((param_group, lr, self._get_wd(param_group) * wd_mult))
        for param_group, lr, weight_decay in updates:
            if isinstance(param_group.get("lr"), torch.Tensor):
                param_group["lr"].fill_(lr)
            else:
                param_group["lr"] = lr
            param_group["weight_decay"] = weight_decay

    @staticmethod
    def _finite_group_number(
        param_group: dict[str, Any], name: str, *, default: float, non_negative: bool
    ) -> float:
        value = param_group.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Optimizer param-group {name} must be a finite number.")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"Optimizer param-group {name} must be finite.")
        if non_negative and value < 0.0:
            raise ValueError(f"Optimizer param-group {name} must be non-negative.")
        return value

    def _group_lr_bounds(
        self, param_group: dict[str, Any]
    ) -> tuple[float, float, float]:
        lr_mult = self._finite_group_number(
            param_group, "lr_mult", default=1.0, non_negative=True
        )
        has_explicit_bounds = "max_lr" in param_group or "min_lr" in param_group
        if has_explicit_bounds and lr_mult != 1.0:
            raise ValueError(
                "Optimizer param-group mixes explicit max_lr/min_lr with a non-unit "
                "legacy lr_mult; migrate to explicit final bounds."
            )
        if has_explicit_bounds:
            max_lr = self._finite_group_number(
                param_group, "max_lr", default=self.max_lr, non_negative=True
            )
            min_lr = self._finite_group_number(
                param_group, "min_lr", default=self.min_lr, non_negative=True
            )
            init_lr = self.init_lr
        else:
            max_lr = self.max_lr * lr_mult
            min_lr = self.min_lr * lr_mult
            init_lr = self.init_lr * lr_mult
        if max_lr < min_lr:
            raise ValueError(
                "Optimizer param-group max_lr must be greater than or equal to min_lr."
            )
        if init_lr > max_lr:
            raise ValueError(
                "Optimizer param-group warmup init_lr must be no greater than max_lr."
            )
        return init_lr, max_lr, min_lr

    def _get_lr(self, param_group: dict[str, Any]) -> float:
        init_lr, max_lr, min_lr = self._group_lr_bounds(param_group)
        if self.lr_warmup_steps > 0 and self.num_steps <= self.lr_warmup_steps:
            ratio = self.num_steps / self.lr_warmup_steps
            return init_lr + (max_lr - init_lr) * ratio

        if self.lr_decay_style == "constant":
            return max_lr

        if self.num_steps > self.lr_decay_steps:
            return min_lr

        if self.lr_decay_style == "inverse-square-root":
            warmup = max(self.lr_warmup_steps, 1)
            step = max(self.num_steps, 1)
            return max(min_lr, max_lr * math.sqrt(warmup) / math.sqrt(step))

        if self.lr_decay_style == "wsd":
            return self._get_wsd_lr(max_lr=max_lr, min_lr=min_lr)

        decay_span = max(self.lr_decay_steps - self.lr_warmup_steps, 1)
        ratio = min(max((self.num_steps - self.lr_warmup_steps) / decay_span, 0.0), 1.0)
        return self._decay(max_lr, min_lr, ratio, self.lr_decay_style)

    def _get_wsd_lr(self, *, max_lr: float, min_lr: float) -> float:
        decay_steps = self.wsd_decay_steps or 0
        decay_start = max(self.lr_decay_steps - decay_steps, self.lr_warmup_steps)
        if decay_steps <= 0 or self.num_steps <= decay_start:
            return max_lr
        ratio = min((self.num_steps - decay_start) / max(decay_steps, 1), 1.0)
        if self.lr_wsd_decay_style == "linear":
            coeff = 1.0 - ratio
        elif self.lr_wsd_decay_style == "cosine":
            coeff = 0.5 * (math.cos(math.pi * ratio) + 1.0)
        elif self.lr_wsd_decay_style == "exponential":
            coeff = 2.0 * (0.5**ratio) - 1.0
        elif self.lr_wsd_decay_style == "minus_sqrt":
            coeff = 1.0 - math.sqrt(ratio)
        else:  # Guard direct construction before state validation can regress.
            raise ValueError(
                f"Unsupported WSD LR decay style: {self.lr_wsd_decay_style!r}."
            )
        return min_lr + coeff * (max_lr - min_lr)

    def _get_wd(self, param_group: dict[str, Any]) -> float:
        start_wd = self._finite_group_number(
            param_group, "start_wd", default=self.start_wd, non_negative=True
        )
        end_wd = self._finite_group_number(
            param_group, "end_wd", default=self.end_wd, non_negative=True
        )
        if end_wd < start_wd:
            raise ValueError(
                "Optimizer param-group end_wd must be greater than or equal to start_wd."
            )
        if self.wd_incr_style == "constant":
            if start_wd != end_wd:
                raise ValueError(
                    "Constant weight-decay schedule requires start_wd == end_wd."
                )
            return end_wd
        ratio = min(max(self.num_steps / self.wd_incr_steps, 0.0), 1.0)
        return self._decay(start_wd, end_wd, ratio, self.wd_incr_style)

    @staticmethod
    def _decay(start: float, end: float, ratio: float, style: str) -> float:
        if style == "linear":
            return start + (end - start) * ratio
        if style == "cosine":
            coeff = 0.5 * (math.cos(math.pi * ratio) + 1.0)
            return end + (start - end) * coeff
        if style == "constant":
            return start
        raise ValueError(f"Unsupported scheduler decay style: {style!r}")


class _LRSchedulerCheckpointTarget:
    """Small rollback/fingerprint adapter for the generic checkpoint transaction."""

    _GROUP_FIELDS = (
        "lr",
        "weight_decay",
        "min_lr",
        "max_lr",
        "lr_mult",
        "wd_mult",
        "start_wd",
        "end_wd",
    )

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self._last_checkpoint_step: int | None = None

    @staticmethod
    def _plain_value(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, dict):
            return {
                key: _LRSchedulerCheckpointTarget._plain_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return tuple(
                _LRSchedulerCheckpointTarget._plain_value(item) for item in value
            )
        return value

    def _group_state(self) -> tuple[tuple[tuple[str, bool, Any], ...], ...]:
        return tuple(
            tuple(
                (field, field in group, self._plain_value(group.get(field)))
                for field in self._GROUP_FIELDS
            )
            for group in self.scheduler.optimizer.param_groups
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "scheduler_state": copy.deepcopy(self.scheduler.state_dict()),
            "param_groups": tuple(
                tuple(
                    (field, field in group, copy.deepcopy(group.get(field)))
                    for field in self._GROUP_FIELDS
                )
                for group in self.scheduler.optimizer.param_groups
            ),
            "checkpoint_step": self._last_checkpoint_step,
        }

    def validate_step(self, candidate: Any, expected_step: int | None) -> None:
        _validate_lr_scheduler_payload(candidate, expected_step=expected_step)

    def apply(self, candidate: Any) -> None:
        _validate_lr_scheduler_payload(candidate)
        self.scheduler.load_state_dict(copy.deepcopy(candidate["scheduler_state"]))
        self._last_checkpoint_step = candidate["checkpoint_step"]

    def restore(self, snapshot: dict[str, Any]) -> None:
        self.scheduler.load_state_dict(copy.deepcopy(snapshot["scheduler_state"]))
        group_states = snapshot["param_groups"]
        if len(group_states) != len(self.scheduler.optimizer.param_groups):
            raise RuntimeError(
                "LR scheduler optimizer param-group count changed during checkpoint load."
            )
        for group, saved_fields in zip(
            self.scheduler.optimizer.param_groups, group_states, strict=True
        ):
            for field, present, value in saved_fields:
                if present:
                    current = group.get(field)
                    if isinstance(current, torch.Tensor) and isinstance(
                        value, torch.Tensor
                    ):
                        current.copy_(value)
                    else:
                        group[field] = copy.deepcopy(value)
                else:
                    group.pop(field, None)
        self._last_checkpoint_step = snapshot["checkpoint_step"]

    def fingerprint(self) -> dict[str, Any]:
        return {
            "scheduler_state": self._plain_value(self.scheduler.state_dict()),
            "param_groups": self._group_state(),
            "checkpoint_step": self._last_checkpoint_step,
        }


def _legacy_scheduler_payload_with_consensus(
    local_path: str, scheduler
) -> dict[str, Any] | None:
    """Load the one explicit pre-manifest VERL scheduler migration layout."""

    from megatron.lite.primitive.ckpt import dcp as dcp_impl

    payload: dict[str, Any] | None = None
    local_error: str | None = None
    try:
        resolved = Path(
            dcp_impl._resolve_step_checkpoint_path(  # noqa: SLF001 - migration boundary
                local_path, allow_legacy_checkpoint=True
            )
        )
        if dcp_impl._checkpoint_manifest_path(resolved).exists():  # noqa: SLF001
            payload = None
        else:
            try:
                checkpoint_step = int(resolved.name.removeprefix("step_"))
            except ValueError as exc:
                raise RuntimeError(
                    f"Legacy checkpoint path does not encode a step: {resolved}"
                ) from exc

            requested = Path(local_path)
            candidates = [
                requested / _LR_SCHEDULER_STATE,
                resolved / _LR_SCHEDULER_STATE,
            ]
            if requested.name.startswith("step_"):
                candidates.append(requested.parent / _LR_SCHEDULER_STATE)
            existing = list(
                dict.fromkeys(path for path in candidates if path.is_file())
            )
            if len(existing) != 1:
                raise FileNotFoundError(
                    "explicit legacy scheduler migration requires exactly one recognized "
                    f"{_LR_SCHEDULER_STATE}; found {existing}"
                )
            legacy_state = torch.load(
                existing[0], map_location="cpu", weights_only=False
            )
            if not isinstance(legacy_state, dict) or set(legacy_state) != {"num_steps"}:
                raise RuntimeError(
                    "legacy LR scheduler state must contain exactly num_steps; "
                    "got keys="
                    f"{sorted(legacy_state) if isinstance(legacy_state, dict) else None}"
                )
            num_steps = legacy_state["num_steps"]
            current_state = copy.deepcopy(scheduler.state_dict())
            current_state["num_steps"] = num_steps
            payload = {
                "format": _LR_SCHEDULER_PAYLOAD_FORMAT,
                "checkpoint_step": checkpoint_step,
                "scheduler_state": current_state,
            }
            _validate_lr_scheduler_payload(payload, expected_step=checkpoint_step)
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"

    if dist.is_initialized():
        errors: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(errors, local_error)
        first_error = next((error for error in errors if error is not None), None)
        if first_error is not None:
            raise RuntimeError(f"Legacy LR scheduler migration failed: {first_error}")
        per_rank_payloads: list[dict[str, Any] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(per_rank_payloads, payload)
        if any(item != per_rank_payloads[0] for item in per_rank_payloads[1:]):
            raise RuntimeError(
                "Legacy LR scheduler migration payload differs across ranks: "
                f"{per_rank_payloads}."
            )
        payload = per_rank_payloads[0]
    elif local_error is not None:
        raise RuntimeError(f"Legacy LR scheduler migration failed: {local_error}")

    if payload is not None and (not dist.is_initialized() or dist.get_rank() == 0):
        print(
            "Migrating a pre-manifest VERL LR scheduler with the current runtime "
            "schedule config; re-save the checkpoint immediately.",
            flush=True,
        )
    return payload


def _build_lr_scheduler(optimizer, opt: MegatronLiteOptimizerConfig):
    """Build a Megatron-style LR scheduler for Megatron Lite's optimizer."""
    total_steps = opt.total_training_steps
    if total_steps <= 0:
        return None

    warmup_steps = opt.lr_warmup_steps if opt.lr_warmup_steps is not None else -1
    if warmup_steps <= 0 and opt.lr_warmup_steps_ratio > 0:
        warmup_steps = int(opt.lr_warmup_steps_ratio * total_steps)
    warmup_steps = max(warmup_steps, 0)

    decay_steps = opt.lr_decay_steps if opt.lr_decay_steps is not None else total_steps
    min_lr = opt.min_lr if opt.min_lr is not None else 0.0
    for param_group in optimizer.param_groups:
        if param_group.get("min_lr") is None:
            param_group["min_lr"] = min_lr

    return _MegatronLiteLRScheduler(
        optimizer,
        init_lr=opt.lr_warmup_init,
        max_lr=opt.lr,
        min_lr=min_lr,
        lr_warmup_steps=warmup_steps,
        lr_decay_steps=decay_steps,
        lr_decay_style=opt.lr_decay_style,
        start_wd=opt.weight_decay,
        end_wd=opt.weight_decay,
        wd_incr_steps=total_steps,
        wd_incr_style=opt.weight_decay_incr_style,
        wsd_decay_steps=opt.lr_wsd_decay_steps,
        lr_wsd_decay_style=opt.lr_wsd_decay_style,
        use_checkpoint_config=opt.use_checkpoint_opt_param_scheduler,
    )


class _MegatronLiteModeCtx(BaseEngineCtx):
    """Wrap Megatron Lite runtime contexts with VERL's offload behavior."""

    def __init__(self, engine: MegatronLiteEngine, mode: str, **kwargs):
        super().__init__(engine=engine, mode=mode, **kwargs)
        self._runtime_ctx = None

    def __enter__(self):
        super().__enter__()
        assert self.engine.runtime is not None and self.engine.handle is not None
        if self.mode == "train":
            self._runtime_ctx = self.engine.runtime.train_mode(self.engine.handle)
        else:
            self._runtime_ctx = self.engine.runtime.eval_mode(self.engine.handle)
        self._runtime_ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self._runtime_ctx is not None
        self._runtime_ctx.__exit__(exc_type, exc_val, exc_tb)
        super().__exit__(exc_type, exc_val, exc_tb)
        return False


@EngineRegistry.register(model_type="language_model", backend="mlite", device="cuda")
class MegatronLiteEngine(BaseEngine):
    """VERL BaseEngine implementation that delegates model lifecycle to Megatron Lite."""

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: MegatronLiteEngineConfig,
        optimizer_config: OptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        super().__init__()
        _isolate_compile_cache_per_rank()
        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config

        self.mode = None
        self.device_name = get_device_name()
        self.runtime = None
        self.handle = None
        self.module = None
        self._mlite_config = None
        self._rank = dist.get_rank() if dist.is_initialized() else 0
        self._checkpoint_load_poisoned = False

    @property
    def is_param_offload_enabled(self) -> bool:
        return self.engine_config.param_offload

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        return self.engine_config.optimizer_offload

    def initialize(self):
        if self.engine_config.full_determinism:
            from verl.workers.engine.utils import enable_full_determinism

            enable_full_determinism(seed=self.engine_config.seed)

        self._mlite_config = self._build_mlite_config()
        self.runtime = create_runtime(
            RuntimeConfig(
                backend="mlite",
                hf_path=self.model_config.local_path,
                backend_cfg=self._mlite_config,
            )
        )
        self.handle = self.runtime.build_model()
        self.module = self._extract_primary_module()

        if self.handle._optimizer is not None and self.handle._lr_scheduler is None:
            self.handle._lr_scheduler = _build_lr_scheduler(
                self.handle._optimizer, self._mlite_config.optimizer
            )

        self.to(
            device="cpu",
            model=self.is_param_offload_enabled,
            optimizer=self.is_optimizer_offload_enabled,
            grad=self.is_param_offload_enabled,
        )

    def train_mode(self, **kwargs):
        self._require_initialized()
        return _MegatronLiteModeCtx(self, mode="train", **kwargs)

    def eval_mode(self, **kwargs):
        self._require_initialized()
        return _MegatronLiteModeCtx(self, mode="eval", **kwargs)

    def optimizer_zero_grad(self):
        self._require_initialized()
        self.runtime.zero_grad(self.handle)

    def optimizer_step(self):
        self._require_initialized()
        _, grad_norm, _ = self.runtime.optimizer_step(self.handle)
        return grad_norm

    def lr_scheduler_step(self):
        self._require_initialized()
        if self.handle._lr_scheduler is not None:
            self.handle._lr_scheduler.step(1)
            return self.handle._optimizer.param_groups[0]["lr"]
        return 0.0

    def forward_backward_batch(
        self, data: TensorDict, loss_function, forward_only: bool = False
    ) -> dict[str, Any]:
        self._require_initialized()
        pad_mode = tu.get_non_tensor_data(
            data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING
        )
        if not _is_no_padding_pad_mode(pad_mode):
            raise NotImplementedError(
                "MegatronLiteEngine only supports pad_mode=no_padding for now."
            )

        tu.assign_non_tensor(data, sp_size=self.engine_config.cp)

        token_mask = (
            data["loss_mask"] if "loss_mask" in data.keys() else data["response_mask"]
        )
        batch_num_tokens = token_mask.sum().to(get_device_id())
        torch.distributed.all_reduce(
            batch_num_tokens,
            op=torch.distributed.ReduceOp.SUM,
            group=self.get_data_parallel_group(),
        )
        tu.assign_non_tensor(data, batch_num_tokens=batch_num_tokens.item())
        tu.assign_non_tensor(data, dp_size=self.get_data_parallel_size())

        micro_batches, indices = prepare_micro_batches(
            data=data,
            dp_group=self.get_data_parallel_group(),
            same_micro_num_in_dp=True,
        )

        # Megatron drives every forward through the runtime's forward_backward
        # callback; the engine never calls the module directly.
        return self._forward_backward_batch_with_runtime(
            data=data,
            micro_batches=micro_batches,
            indices=indices,
            loss_function=loss_function,
            forward_only=forward_only,
        )

    def get_per_tensor_param(self, **kwargs):
        self._require_initialized()
        if self.is_param_offload_enabled:
            self.to("cuda", model=True, optimizer=False, grad=False)
        export_kwargs = {
            key: kwargs[key]
            for key in ("limit", "include_mtp_only", "include_local_prefixes")
            if key in kwargs
        }
        assert self._mlite_config is not None
        if self._mlite_config.model_name == "qwen3_5":
            export_kwargs["target"] = "vllm"
        if self.engine_config.export_dtype:
            export_kwargs["export_dtype"] = self.engine_config.export_dtype
        return self.runtime.export_weights(self.handle, **export_kwargs), None

    def get_data_parallel_size(self):
        if self.handle is None:
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            return world_size // (
                self.engine_config.tp * self.engine_config.cp * self.engine_config.pp
            )
        return self.handle.dp_size

    def get_data_parallel_rank(self):
        if self.handle is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
            dense_dp = self.get_data_parallel_size()
            return (rank // (self.engine_config.tp * self.engine_config.cp)) % dense_dp
        return self.handle.dp_rank

    def get_data_parallel_group(self):
        if self.handle is None:
            if (
                self.engine_config.tp == 1
                and self.engine_config.cp == 1
                and self.engine_config.pp == 1
                and dist.is_initialized()
            ):
                return dist.group.WORLD
            return None
        return self.handle.dp_group

    def to(
        self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True
    ):
        self._require_initialized()
        if model or not (optimizer or grad):
            super().to(device=device, model=model, optimizer=optimizer, grad=grad)
        self.runtime.to(
            self.handle, device, model=model, optimizer=optimizer, grad=grad
        )

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: str | None = None,
        global_step: int = 0,
        max_ckpt_to_keep: int | None = None,
        **kwargs,
    ) -> None:
        del hdfs_path, max_ckpt_to_keep, kwargs
        self._require_initialized()

        save_contents = self.checkpoint_config.get("save_contents", None)
        save_content_keys = _content_set_with_consensus(
            save_contents, key="checkpoint_config.save_contents"
        )
        save_model = save_contents is None or "model" in save_content_keys
        save_optimizer = save_contents is None or "optimizer" in save_content_keys
        save_extra = save_contents is None or "extra" in save_content_keys
        scheduler = self.handle._lr_scheduler
        _checkpoint_components_with_consensus(
            model=save_model,
            optimizer=save_optimizer,
            extra=save_extra,
            scheduler_present=scheduler is not None,
            context="checkpoint save",
        )
        if scheduler is not None and save_optimizer and not save_extra:
            raise ValueError(
                "Saving optimizer state without extra state would omit the active LR "
                "scheduler and cannot produce an exact-resume checkpoint. Include "
                "'extra' in checkpoint_config.save_contents."
            )
        if not save_model and not save_optimizer and not save_extra:
            if self._rank == 0:
                print(
                    f"Skipping Megatron Lite checkpoint save at step {global_step}: save_contents={save_contents}"
                )
            if dist.is_initialized():
                dist.barrier()
            return

        reload_params_for_save = self.is_param_offload_enabled
        reload_started = False
        placement_fn = None
        expert_classifier = None
        setup_error: str | None = None
        try:
            os.makedirs(local_path, exist_ok=True)
            placement_fn, expert_classifier = self._checkpoint_hooks()
            if reload_params_for_save:
                reload_started = True
                self.to(device="cuda", model=True, optimizer=False, grad=False)
                torch.cuda.synchronize()
        except Exception as exc:
            setup_error = f"{type(exc).__name__}: {exc}"
        try:
            _distributed_raise_if_error(
                setup_error, context="checkpoint save setup failed"
            )
            extra_states = (
                {
                    _LR_SCHEDULER_STATE: _scheduler_payload(
                        scheduler, checkpoint_step=global_step
                    )
                }
                if save_extra and scheduler is not None
                else None
            )
            save_training_checkpoint(
                self.module,
                self.handle._optimizer,
                global_step,
                local_path,
                self.handle._config.parallel,
                self.handle._parallel_state,
                get_placements=placement_fn,
                is_expert=expert_classifier,
                save_rng=save_extra,
                save_model=save_model,
                save_optimizer=save_optimizer,
                extra_states=extra_states,
            )
            if dist.is_initialized():
                dist.barrier()
        finally:
            cleanup_error: str | None = None
            if reload_started:
                try:
                    self.to(device="cpu", model=True, optimizer=False, grad=False)
                except Exception as exc:
                    cleanup_error = f"{type(exc).__name__}: {exc}"
            _distributed_raise_if_error(
                cleanup_error, context="checkpoint save offload cleanup failed"
            )

    def load_checkpoint(
        self,
        local_path: str,
        hdfs_path: str | None = None,
        del_local_after_load: bool = True,
        **kwargs,
    ) -> None:
        allow_legacy_checkpoint = bool(kwargs.pop("allow_legacy_checkpoint", False))
        del hdfs_path, del_local_after_load, kwargs
        self._require_initialized()

        try:
            load_contents = self.checkpoint_config.get(
                "load_contents", self.checkpoint_config.get("save_contents", None)
            )
            load_content_keys = _content_set_with_consensus(
                load_contents, key="checkpoint_config.load_contents"
            )
            load_model = load_contents is None or "model" in load_content_keys
            load_optimizer = load_contents is None or "optimizer" in load_content_keys
            load_extra = load_contents is None or "extra" in load_content_keys
            scheduler = self.handle._lr_scheduler
            load_scheduler = load_extra and scheduler is not None
            _checkpoint_components_with_consensus(
                model=load_model,
                optimizer=load_optimizer,
                extra=load_extra,
                scheduler_present=scheduler is not None,
                context="checkpoint load",
            )
            if scheduler is not None and load_optimizer and not load_extra:
                raise ValueError(
                    "Loading optimizer state without extra state would restore optimizer "
                    "LR/WD without the matching LR scheduler progress. Include 'extra' in "
                    "checkpoint_config.load_contents."
                )
            if not load_model and not load_optimizer and not load_extra:
                if dist.is_initialized():
                    dist.barrier()
                return

            reload_params_for_load = self.is_param_offload_enabled and load_model
            reload_started = False
            placement_fn = None
            expert_classifier = None
            setup_error: str | None = None
            try:
                placement_fn, expert_classifier = self._checkpoint_hooks()
                if reload_params_for_load:
                    reload_started = True
                    self.to(device="cuda", model=True, optimizer=False, grad=False)
                    torch.cuda.synchronize()
            except Exception as exc:
                setup_error = f"{type(exc).__name__}: {exc}"
            try:
                _distributed_raise_if_error(
                    setup_error, context="checkpoint load setup failed"
                )
                loaded_extra_states: dict[str, Any] | None = (
                    {} if load_scheduler else None
                )
                scheduler_target = (
                    _LRSchedulerCheckpointTarget(scheduler) if load_scheduler else None
                )
                legacy_scheduler_payload = (
                    _legacy_scheduler_payload_with_consensus(local_path, scheduler)
                    if load_scheduler and allow_legacy_checkpoint
                    else None
                )
                strict_scheduler_extra = (
                    load_scheduler and legacy_scheduler_payload is None
                )
                if legacy_scheduler_payload is not None:
                    from megatron.lite.primitive.ckpt import dcp as dcp_impl

                    assert scheduler_target is not None
                    scheduler_target.validate_step(
                        legacy_scheduler_payload,
                        legacy_scheduler_payload["checkpoint_step"],
                    )
                    dcp_impl._preflight_extra_state_targets(  # noqa: SLF001
                        {_LR_SCHEDULER_STATE: scheduler_target},
                        {_LR_SCHEDULER_STATE: legacy_scheduler_payload},
                    )
                restored_step = load_training_checkpoint(
                    self.module,
                    self.handle._optimizer,
                    local_path,
                    self.handle._config.parallel,
                    self.handle._parallel_state,
                    get_placements=placement_fn,
                    is_expert=expert_classifier,
                    load_rng=load_extra,
                    load_model=load_model,
                    load_optimizer=load_optimizer,
                    allow_legacy_checkpoint=allow_legacy_checkpoint,
                    load_extra_state_files=(
                        (_LR_SCHEDULER_STATE,) if strict_scheduler_extra else None
                    ),
                    loaded_extra_states=loaded_extra_states,
                    extra_state_validators=(
                        {_LR_SCHEDULER_STATE: _validate_lr_scheduler_payload}
                        if strict_scheduler_extra
                        else None
                    ),
                    extra_state_targets=(
                        {_LR_SCHEDULER_STATE: scheduler_target}
                        if strict_scheduler_extra and scheduler_target is not None
                        else None
                    ),
                )
                if legacy_scheduler_payload is not None:
                    assert scheduler_target is not None
                    legacy_error: str | None = None
                    try:
                        _validate_lr_scheduler_payload(
                            legacy_scheduler_payload, expected_step=restored_step
                        )
                    except Exception as exc:
                        legacy_error = f"{type(exc).__name__}: {exc}"
                    _distributed_raise_if_error(
                        legacy_error,
                        context="legacy LR scheduler/core step validation failed",
                    )
                    dcp_impl._commit_extra_state_targets(  # noqa: SLF001
                        {_LR_SCHEDULER_STATE: scheduler_target},
                        {_LR_SCHEDULER_STATE: legacy_scheduler_payload},
                    )
                    assert loaded_extra_states is not None
                    loaded_extra_states[_LR_SCHEDULER_STATE] = legacy_scheduler_payload
                post_load_error: str | None = None
                try:
                    if load_scheduler:
                        assert loaded_extra_states is not None
                        if _LR_SCHEDULER_STATE not in loaded_extra_states:
                            raise RuntimeError(
                                "Megatron Lite checkpoint load completed without the "
                                f"required {_LR_SCHEDULER_STATE} extra state."
                            )
                        payload = loaded_extra_states[_LR_SCHEDULER_STATE]
                        _validate_lr_scheduler_payload(
                            payload, expected_step=restored_step
                        )
                except Exception as exc:
                    post_load_error = f"{type(exc).__name__}: {exc}"
                _distributed_raise_if_error(
                    post_load_error,
                    context="checkpoint load post-commit validation failed",
                )
                if dist.is_initialized():
                    dist.barrier()
            finally:
                cleanup_error: str | None = None
                if reload_started:
                    try:
                        self.to(device="cpu", model=True, optimizer=False, grad=False)
                    except Exception as exc:
                        cleanup_error = f"{type(exc).__name__}: {exc}"
                _distributed_raise_if_error(
                    cleanup_error, context="checkpoint load offload cleanup failed"
                )
        except Exception:
            self._checkpoint_load_poisoned = True
            raise

    def is_mp_src_rank_with_outputs(self):
        if self.handle is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
            dense_dp = self.get_data_parallel_size()
            tp_rank = rank % self.engine_config.tp
            cp_rank = (rank // self.engine_config.tp) % self.engine_config.cp
            pp_rank = rank // (self.engine_config.tp * self.engine_config.cp * dense_dp)
            return (
                tp_rank == 0 and cp_rank == 0 and pp_rank == self.engine_config.pp - 1
            )
        return self.runtime.is_mp_src_rank_with_outputs(self.handle)

    def _require_initialized(self) -> None:
        if self.runtime is None or self.handle is None:
            raise RuntimeError("MegatronLiteEngine is not initialized yet.")
        if self._checkpoint_load_poisoned:
            raise RuntimeError(
                "MegatronLiteEngine is poisoned by a failed checkpoint load; "
                "discard and reinitialize this engine before further use."
            )

    def _build_mlite_config(self) -> MegatronLiteConfig:
        return MegatronLiteConfig(
            model_name=self._resolve_model_name(),
            impl=self.engine_config.impl,
            hf_path=self.model_config.local_path,
            parallel=ParallelConfig(
                tp=self.engine_config.tp,
                etp=self.engine_config.etp or 1,
                ep=self.engine_config.ep,
                pp=self.engine_config.pp,
                vpp=self.engine_config.vpp,
                cp=self.engine_config.cp,
            ),
            optimizer=self._build_mlite_optimizer_config(),
            attention_backend_override=self.engine_config.attention_backend_override,
            router_aux_loss_coef=self.engine_config.router_aux_loss_coef,
            load_hf_weights=self.engine_config.load_hf_weights,
            impl_cfg=self._build_impl_cfg(),
        )

    def _resolve_model_name(self) -> str:
        if self.engine_config.model_name != "auto":
            return self.engine_config.model_name
        return resolve_model_type_from_hf(self.model_config.hf_config)

    def _build_impl_cfg(self) -> dict[str, Any]:
        impl_cfg = dict(self.engine_config.impl_cfg)
        if impl_cfg.get("use_thd", True) is not True:
            raise ValueError(
                "MegatronLiteEngine supports only THD/no-padding SFT; set engine.impl_cfg.use_thd=True."
            )
        impl_cfg["use_thd"] = True
        cross_entropy_fusion = getattr(self.engine_config, "cross_entropy_fusion", None)
        if cross_entropy_fusion is None:
            cross_entropy_fusion = getattr(
                self.engine_config, "use_fused_kernels", False
            )
        impl_cfg.setdefault("cross_entropy_fusion", bool(cross_entropy_fusion))
        mtp_cfg = getattr(self.model_config, "mtp", None)
        if mtp_cfg is not None:
            mtp_enable = bool(getattr(mtp_cfg, "enable", False))
            mtp_enable_train = mtp_enable and bool(
                getattr(mtp_cfg, "enable_train", False)
            )
            impl_cfg["mtp_enable"] = mtp_enable
            impl_cfg["mtp_enable_train"] = mtp_enable_train
            impl_cfg["mtp_detach_encoder"] = bool(
                getattr(mtp_cfg, "detach_encoder", False)
            )
            impl_cfg["mtp_loss_scaling_factor"] = float(
                getattr(mtp_cfg, "mtp_loss_scaling_factor", 0.1)
            )
        if self.engine_config.full_determinism:
            impl_cfg.setdefault("deterministic", True)
        if self.engine_config.forward_only:
            impl_cfg["optimizer"] = None
        return impl_cfg

    def _build_mlite_optimizer_config(self) -> MegatronLiteOptimizerConfig:
        optimizer_name = self._normalize_optimizer_name(self.optimizer_config)
        betas = tuple(getattr(self.optimizer_config, "betas", (0.9, 0.999)))
        override = getattr(self.optimizer_config, "override_optimizer_config", {}) or {}
        offload_fraction = override.get(
            "offload_fraction", override.get("optimizer_offload_fraction")
        )
        if offload_fraction is None and override.get("optimizer_cpu_offload"):
            offload_fraction = 1.0
        if offload_fraction is None and self.is_optimizer_offload_enabled:
            offload_fraction = 1.0

        min_lr = getattr(self.optimizer_config, "min_lr", None)
        min_lr_ratio = getattr(self.optimizer_config, "min_lr_ratio", None)
        if min_lr is None:
            min_lr = (
                0.0 if min_lr_ratio is None else self.optimizer_config.lr * min_lr_ratio
            )

        lr_decay_style = getattr(self.optimizer_config, "lr_decay_style", None)
        if lr_decay_style is None:
            lr_decay_style = getattr(
                self.optimizer_config, "lr_scheduler_type", "constant"
            )

        return MegatronLiteOptimizerConfig(
            optimizer=optimizer_name,
            lr=self.optimizer_config.lr,
            min_lr=min_lr,
            clip_grad=self.optimizer_config.clip_grad,
            weight_decay=self.optimizer_config.weight_decay,
            lr_warmup_steps_ratio=self.optimizer_config.lr_warmup_steps_ratio,
            total_training_steps=self.optimizer_config.total_training_steps,
            lr_warmup_steps=self.optimizer_config.lr_warmup_steps,
            lr_warmup_init=getattr(self.optimizer_config, "lr_warmup_init", 0.0),
            lr_decay_steps=getattr(self.optimizer_config, "lr_decay_steps", None),
            lr_decay_style=lr_decay_style,
            weight_decay_incr_style=getattr(
                self.optimizer_config, "weight_decay_incr_style", "constant"
            ),
            lr_wsd_decay_style=getattr(
                self.optimizer_config, "lr_wsd_decay_style", "exponential"
            ),
            lr_wsd_decay_steps=getattr(
                self.optimizer_config, "lr_wsd_decay_steps", None
            ),
            use_checkpoint_opt_param_scheduler=getattr(
                self.optimizer_config, "use_checkpoint_opt_param_scheduler", False
            ),
            adam_beta1=betas[0],
            adam_beta2=betas[1],
            adam_eps=override.get("adam_eps", override.get("eps")),
            offload_fraction=offload_fraction,
            use_precision_aware_optimizer=override.get("use_precision_aware_optimizer"),
            decoupled_weight_decay=override.get("decoupled_weight_decay"),
        )

    @staticmethod
    def _normalize_optimizer_name(config: OptimizerConfig) -> str:
        optimizer_name = getattr(config, "optimizer", "adam")
        lower = str(optimizer_name).lower()
        if "adam" in lower:
            return "adam"
        raise ValueError(
            f"MegatronLiteEngine only supports Adam-style optimizers today, got {optimizer_name!r}"
        )

    def _extract_primary_module(self):
        model = self.handle._model
        if isinstance(model, list | tuple):
            if not model:
                raise RuntimeError(
                    "Megatron Lite runtime returned an empty model chunk list."
                )
            if len(model) > 1:
                return torch.nn.ModuleList(model)
            return model[0]
        return model

    def _forward_backward_batch_with_runtime(
        self,
        *,
        data: TensorDict,
        micro_batches: list[TensorDict],
        indices,
        loss_function,
        forward_only: bool,
    ) -> dict[str, Any]:
        runtime_batches = []
        num_micro_batches = len(micro_batches)
        batch_num_tokens = tu.get_non_tensor_data(
            data=data, key="batch_num_tokens", default=None
        )
        if batch_num_tokens is None:
            raise ValueError(
                "MegatronLiteEngine PP/CP SFT requires batch_num_tokens for VERL-compatible loss scaling."
            )
        if batch_num_tokens <= 0:
            raise ValueError(
                f"batch_num_tokens must be positive, got {batch_num_tokens}."
            )
        loss_scale = (
            self.get_data_parallel_size() * num_micro_batches / float(batch_num_tokens)
        )
        for micro_idx, micro_batch in enumerate(micro_batches):
            tu.assign_non_tensor(micro_batch, micro_batch_idx=micro_idx)
            micro_batch = micro_batch.to(get_device_id())
            runtime_batches.append(
                (
                    self._make_runtime_batch(micro_batch),
                    self._make_runtime_loss_context(micro_batch, loss_scale=loss_scale),
                )
            )

        runtime_loss_fn = None
        if loss_function is not None or forward_only:
            runtime_loss_fn = self._make_runtime_loss_fn(
                loss_function, forward_only=forward_only
            )

        result = self.runtime.forward_backward(
            self.handle,
            iter(runtime_batches),
            loss_fn=runtime_loss_fn,
            num_microbatches=num_micro_batches,
            forward_only=forward_only,
        )
        metrics = dict(result.metrics)
        micro_outputs = metrics.pop("_micro_outputs", None)
        if micro_outputs is not None and self.is_mp_src_rank_with_outputs():
            return postprocess_batch_func(
                output_lst=micro_outputs, indices=indices, data=data
            )
        loss = float(metrics.get("loss", 0.0))
        return {
            "model_output": {},
            "loss": [loss],
            # Pass Metric aggregators through unchanged (reduce_metrics folds them);
            # list-wrap plain scalars as the legacy contract expects.
            "metrics": {
                key: (
                    value
                    if (_VerlMetric is not None and isinstance(value, _VerlMetric))
                    else [value]
                )
                for key, value in metrics.items()
            },
        }

    def _make_runtime_batch(self, micro_batch: TensorDict) -> PackedBatch:
        """Flatten a jagged no-padding batch to a model-agnostic ``PackedBatch``.

        No CP split, no padding, no ``PackedSeqParams`` here: each model's
        protocol owns its pack/unpack pair (zigzag vs contiguous). ``labels`` are
        the unrolled tokens; the protocol rolls them while packing.
        """
        input_ids = micro_batch["input_ids"]
        if not getattr(input_ids, "is_nested", False):
            raise NotImplementedError(
                "MegatronLiteEngine supports only nested no-padding THD batches."
            )
        loss_mask = self._loss_mask_for_packing(micro_batch, input_ids)
        return PackedBatch(
            input_ids=input_ids.values().contiguous(),
            labels=input_ids.values().contiguous(),
            loss_mask=(
                None if loss_mask is None else loss_mask.values().contiguous().float()
            ),
            seq_lens=input_ids.offsets().diff().to(dtype=torch.int64),
        )

    def _make_runtime_loss_context(
        self, micro_batch: TensorDict, *, loss_scale: float
    ) -> LossContext:
        return LossContext(
            temperature=float(self._scalar_temperature(micro_batch)),
            calculate_entropy=bool(
                tu.get_non_tensor_data(
                    data=micro_batch, key="calculate_entropy", default=False
                )
            ),
            return_log_probs=True,
            loss_scale=loss_scale,
            source_batch=micro_batch,
        )

    @staticmethod
    def _loss_mask_for_packing(
        micro_batch: TensorDict, input_ids: torch.Tensor
    ) -> torch.Tensor | None:
        if "loss_mask" not in micro_batch.keys():
            return None

        loss_mask = micro_batch["loss_mask"]
        if getattr(loss_mask, "is_nested", False):
            return loss_mask

        rows = []
        for seq_ids, row_mask in zip(input_ids.unbind(0), loss_mask, strict=True):
            seq_len = seq_ids.numel()
            response_tokens = int(row_mask.sum().item())
            if response_tokens > seq_len:
                raise ValueError(
                    f"response loss mask has {response_tokens} tokens but packed input sequence has {seq_len} tokens"
                )
            full_mask = torch.zeros(
                seq_len, dtype=row_mask.dtype, device=row_mask.device
            )
            if response_tokens:
                full_mask[-response_tokens:] = row_mask[:response_tokens]
            rows.append(full_mask)
        return torch.nested.as_nested_tensor(rows, layout=torch.jagged)

    def _build_verl_model_output(
        self, *, raw_output: dict[str, torch.Tensor], runtime_batch: PackedBatch
    ) -> dict[str, torch.Tensor]:
        log_probs = raw_output.get("log_probs")
        if log_probs is None:
            raise ValueError(
                "Megatron Lite THD model output must contain token log_probs."
            )
        proto = self.handle._extras.get("protocol")
        unpack = getattr(proto, "unpack_forward_output", None)
        if unpack is None:
            raise ValueError(
                "Model protocol must expose unpack_forward_output to reverse THD outputs."
            )
        output = {"log_probs": unpack(self.module, runtime_batch, log_probs)}
        entropy = raw_output.get("entropy")
        if entropy is not None:
            output["entropy"] = unpack(self.module, runtime_batch, entropy)
        return output

    def _make_runtime_loss_fn(self, loss_function, *, forward_only: bool):
        def _loss_fn(
            raw_output: dict[str, torch.Tensor],
            runtime_batch: PackedBatch,
            loss_context: LossContext,
        ):
            micro_batch = loss_context.source_batch
            model_output = self._build_verl_model_output(
                raw_output=raw_output, runtime_batch=runtime_batch
            )
            raw_output["_verl_model_output"] = model_output
            if loss_function is not None:
                loss, metrics = loss_function(
                    model_output=model_output,
                    data=micro_batch,
                    dp_group=self.get_data_parallel_group(),
                )
            else:
                loss = torch.zeros((), device=get_device_id(), dtype=torch.float32)
                metrics = {}

            if raw_output.get("mtp_loss") is not None:
                metrics = dict(metrics)
                mtp_loss = self._reduce_mtp_metric(raw_output["mtp_loss"])
                metrics["mtp_losses/mtp_1_loss"] = (
                    float(mtp_loss.item())
                    if mtp_loss.numel() == 1
                    else mtp_loss.cpu().tolist()
                )

            raw_output["_verl_metrics"] = metrics
            return loss, metrics

        return _loss_fn

    def _mtp_enable_train(self) -> bool:
        mtp_cfg = getattr(self.model_config, "mtp", None)
        return bool(
            mtp_cfg is not None
            and getattr(mtp_cfg, "enable", False)
            and getattr(mtp_cfg, "enable_train", False)
        )

    def _reduce_mtp_metric(self, mtp_loss: torch.Tensor) -> torch.Tensor:
        mtp_loss = mtp_loss.detach().float().clone()
        dp_group = self.get_data_parallel_group()
        if dist.is_initialized() and dp_group is not None:
            dist.all_reduce(mtp_loss, op=dist.ReduceOp.AVG, group=dp_group)
        return mtp_loss

    @staticmethod
    def _scalar_temperature(micro_batch: TensorDict) -> float:
        if "temperature" not in micro_batch.keys():
            return 1.0
        temperature = micro_batch["temperature"]
        if not isinstance(temperature, torch.Tensor):
            return float(temperature)
        values = (
            temperature.values()
            if getattr(temperature, "is_nested", False)
            else temperature.reshape(-1)
        )
        if values.numel() == 0:
            return 1.0
        first = values[0].detach()
        if not torch.all(values.detach() == first).item():
            raise NotImplementedError(
                "MegatronLiteEngine currently supports scalar temperature only."
            )
        return float(first.float().item())

    def _checkpoint_hooks(self):
        proto = self.handle._extras.get("protocol")
        placement_fn = getattr(proto, "PLACEMENT_FN", default_placement_fn)
        expert_classifier = getattr(
            proto, "EXPERT_CLASSIFIER", default_expert_classifier
        )
        return placement_fn, expert_classifier
