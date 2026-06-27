# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""AdamW helpers for the FSDP2 optimizer primitive."""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn


def local_grad_sq_sum(
    params: Iterable[nn.Parameter],
    *,
    dtype: torch.dtype,
    default_device: torch.device | None = None,
) -> torch.Tensor:
    total: torch.Tensor | None = None
    for param in params:
        grad = param.grad
        if grad is None:
            continue
        grad = to_local_tensor(grad)
        if total is None:
            total = torch.zeros((), device=grad.device, dtype=dtype)
        total += grad.detach().to(dtype).pow(2).sum()
    if total is None:
        return torch.zeros(
            (), device=default_device or torch.device("cpu"), dtype=dtype
        )
    return total


def to_local_tensor(tensor):
    local_tensor = getattr(tensor, "_local_tensor", None)
    if isinstance(local_tensor, torch.Tensor):
        return local_tensor
    to_local = getattr(tensor, "to_local", None)
    if callable(to_local):
        return to_local()
    return tensor


def fsdp2_model_param_dtype(param: nn.Parameter) -> torch.dtype | None:
    dtype = getattr(param, "_fsdp2_model_param_dtype", None)
    return dtype if isinstance(dtype, torch.dtype) else None


def has_dtensor_grad_or_param(param: nn.Parameter) -> bool:
    grad = param.grad
    return is_dtensor_like(param) or (grad is not None and is_dtensor_like(grad))


def is_dtensor_like(tensor: Any) -> bool:
    return (
        callable(getattr(tensor, "to_local", None))
        and hasattr(tensor, "device_mesh")
        and hasattr(tensor, "placements")
    )


def copy_local_tensor_to_param_(
    param: nn.Parameter, local_tensor: torch.Tensor
) -> None:
    if not is_dtensor_like(param):
        param.detach().copy_(local_tensor.to(device=param.device, dtype=param.dtype))
        return

    # Copy straight into the param's local shard. Reconstructing via
    # DTensor.from_local mis-sizes an unevenly-sharded param (it infers global =
    # local * mesh, e.g. a (3,) param over 8 ranks -> 0 or 8), so copy local->local
    # (master is init'd from this same local shard, so shapes match).
    local_param = to_local_tensor(param)
    local_param.copy_(
        local_tensor.to(device=local_param.device, dtype=local_param.dtype)
    )


def all_reduce_grad_(grad: torch.Tensor, *, group: dist.ProcessGroup) -> None:
    # ``to_local_tensor`` returns the DTensor's local shard storage, so the
    # in-place all-reduce updates the grad directly -- no DTensor.from_local
    # round-trip (which mis-sizes unevenly-sharded grads, e.g. (3,) over 8 ranks).
    local_grad = to_local_tensor(grad)
    dist.all_reduce(local_grad, op=dist.ReduceOp.SUM, group=group)


class ChainedOptimizer:
    def __init__(self, optimizers: Iterable[torch.optim.Optimizer]):
        self.optimizers = list(optimizers)

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []
        for optimizer in self.optimizers:
            groups.extend(optimizer.param_groups)
        return groups

    def zero_grad(self, *args, **kwargs) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(*args, **kwargs)

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": "chained_torch_optimizer",
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        optimizer_states = state_dict.get("optimizers")
        if not isinstance(optimizer_states, list) or len(optimizer_states) != len(
            self.optimizers
        ):
            raise ValueError("Invalid chained torch optimizer state_dict.")
        for optimizer, optimizer_state in zip(
            self.optimizers, optimizer_states, strict=True
        ):
            optimizer.load_state_dict(optimizer_state)


class FP32AdamW:
    """AdamW with FP32 master params for BF16/DTensor model weights."""

    def __init__(
        self,
        params: Iterable[nn.Parameter] | Iterable[dict[str, Any]],
        *,
        lr: float,
        weight_decay: float,
        betas: tuple[float, float],
        eps: float,
        cpu_update: bool = False,
        model_param_dtypes: dict[int, torch.dtype] | None = None,
    ):
        self.param_groups = normalize_param_groups(
            params, default_weight_decay=weight_decay
        )
        self.params: list[nn.Parameter] = []
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.eps = eps
        self.cpu_update = bool(cpu_update)
        self.step_count = 0
        self.state: dict[nn.Parameter, dict[str, torch.Tensor]] = {}
        self._master_for_param: dict[nn.Parameter, torch.Tensor] = {}
        self._model_param_dtypes_by_id = dict(model_param_dtypes or {})
        self._model_dtype_for_param: dict[nn.Parameter, torch.dtype] = {}

        for group in self.param_groups:
            group.setdefault("lr", lr)
            group.setdefault("wd_mult", 1.0)
            group_weight_decay = float(group.get("weight_decay", weight_decay))
            group["weight_decay"] = group_weight_decay
            for param in group["params"]:
                self.params.append(param)
                model_dtype = self._model_param_dtypes_by_id.get(id(param))
                if model_dtype is not None:
                    self._model_dtype_for_param[param] = model_dtype
                master = self._init_master_param(param)
                self.state[param] = {
                    "master_param": master,
                    "exp_avg": torch.zeros_like(master, dtype=torch.float32),
                    "exp_avg_sq": torch.zeros_like(master, dtype=torch.float32),
                    "step": 0,
                }
                self._master_for_param[param] = master

    def _init_master_param(self, param: nn.Parameter) -> torch.Tensor:
        if self.cpu_update:
            local_param = to_local_tensor(param.detach())
            return local_param.detach().to(device="cpu", dtype=torch.float32).clone()
        if self._model_param_dtype(param) is not None:
            return param.detach().to(dtype=torch.float32).clone()
        return (
            param.detach()
            if param.dtype is torch.float32
            else param.detach().to(dtype=torch.float32).clone()
        )

    def _model_param_dtype(self, param: nn.Parameter) -> torch.dtype | None:
        return self._model_dtype_for_param.get(param) or fsdp2_model_param_dtype(param)

    def zero_grad(self, *args, **kwargs) -> None:
        set_to_none = kwargs.get("set_to_none", False)
        if args:
            set_to_none = bool(args[0])
        for param in self.params:
            if set_to_none:
                param.grad = None
            elif param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

    def step(self) -> None:
        self._step_param_groups()

    def _step_param_groups(self) -> None:
        self.step_count += 1
        beta1, beta2 = self.betas

        for group in self.param_groups:
            group_lr = float(group.get("lr", self.lr))
            group_weight_decay = float(group.get("weight_decay", self.weight_decay))
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                state = self.state[param]
                state["step"] = int(state["step"]) + 1
                param_step = int(state["step"])
                bias_correction1 = 1.0 - beta1**param_step
                bias_correction2_sqrt = (1.0 - beta2**param_step) ** 0.5
                group_step_size = group_lr / bias_correction1
                master = state["master_param"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                if group_weight_decay != 0.0:
                    master.mul_(1.0 - group_lr * group_weight_decay)
                grad = self._prepare_grad(grad, master)
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().div_(bias_correction2_sqrt).add_(self.eps)
                master.addcdiv_(
                    exp_avg.to(dtype=torch.float32), denom, value=-group_step_size
                )
                self._copy_master_to_param(param, master)

    def _prepare_grad(self, grad: torch.Tensor, master: torch.Tensor) -> torch.Tensor:
        if self.cpu_update:
            grad = to_local_tensor(grad)
            return grad.detach().to(device=master.device, dtype=torch.float32)
        return grad.detach().to(dtype=torch.float32)

    def _copy_master_to_param(self, param: nn.Parameter, master: torch.Tensor) -> None:
        model_dtype = self._model_param_dtype(param)
        if model_dtype is not None:
            master = master.to(dtype=model_dtype).to(dtype=param.dtype)
        if not self.cpu_update:
            param.detach().copy_(master.to(dtype=param.dtype))
            return
        copy_local_tensor_to_param_(param, master)

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": "fp32_adamw",
            "version": 2,
            "step_count": self.step_count,
            "config": {
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "betas": self.betas,
                "eps": self.eps,
                "cpu_update": self.cpu_update,
            },
            "master_params": [
                self.state[param]["master_param"] for param in self.params
            ],
            "exp_avgs": [self.state[param]["exp_avg"] for param in self.params],
            "exp_avg_sqs": [self.state[param]["exp_avg_sq"] for param in self.params],
            "steps": [int(self.state[param]["step"]) for param in self.params],
            "param_groups": [
                {
                    "param_count": len(group["params"]),
                    "options": {
                        key: value for key, value in group.items() if key != "params"
                    },
                }
                for group in self.param_groups
            ],
        }

    @staticmethod
    def _validate_option_value(name: str, value: Any, reference: Any) -> None:
        if isinstance(reference, torch.Tensor):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"FP32 AdamW option {name} must be a tensor.")
            if value.shape != reference.shape or value.dtype != reference.dtype:
                raise ValueError(
                    f"FP32 AdamW option {name} shape/dtype mismatch: "
                    f"checkpoint={tuple(value.shape)}/{value.dtype}, "
                    f"runtime={tuple(reference.shape)}/{reference.dtype}."
                )
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f"FP32 AdamW option {name} must be finite.")
            return
        if isinstance(reference, bool):
            if not isinstance(value, bool):
                raise TypeError(f"FP32 AdamW option {name} must be bool.")
            return
        if isinstance(reference, (int, float)) and not isinstance(reference, bool):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"FP32 AdamW option {name} must be numeric.")
            if not math.isfinite(float(value)):
                raise ValueError(f"FP32 AdamW option {name} must be finite.")
            if (
                name
                in {
                    "lr",
                    "initial_lr",
                    "weight_decay",
                    "eps",
                    "wd_mult",
                    "lr_mult",
                    "max_lr",
                    "min_lr",
                    "start_wd",
                    "end_wd",
                }
                and value < 0
            ):
                raise ValueError(f"FP32 AdamW option {name} must be non-negative.")
            return
        if isinstance(reference, tuple):
            if not isinstance(value, tuple) or len(value) != len(reference):
                raise TypeError(f"FP32 AdamW option {name} must be a matching tuple.")
            for index, (item, expected) in enumerate(
                zip(value, reference, strict=True)
            ):
                FP32AdamW._validate_option_value(f"{name}[{index}]", item, expected)
            return
        if value is not None and reference is None:
            raise TypeError(f"FP32 AdamW option {name} must remain None.")
        if reference is not None and not isinstance(value, type(reference)):
            raise TypeError(
                f"FP32 AdamW option {name} must have type {type(reference).__name__}."
            )

    def validate_state_dict(self, state_dict: dict[str, Any]) -> None:
        expected_keys = {
            "type",
            "version",
            "step_count",
            "config",
            "master_params",
            "exp_avgs",
            "exp_avg_sqs",
            "steps",
            "param_groups",
        }
        if not isinstance(state_dict, dict) or set(state_dict) != expected_keys:
            raise ValueError("Invalid FP32 AdamW v2 state_dict schema.")
        if state_dict["type"] != "fp32_adamw" or state_dict["version"] != 2:
            raise ValueError("Unsupported FP32 AdamW state_dict format.")
        step_count = state_dict["step_count"]
        if (
            isinstance(step_count, bool)
            or not isinstance(step_count, int)
            or step_count < 0
        ):
            raise ValueError("Invalid FP32 AdamW step_count.")
        config = state_dict["config"]
        runtime_config = {
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "betas": self.betas,
            "eps": self.eps,
            "cpu_update": self.cpu_update,
        }
        if not isinstance(config, dict) or set(config) != set(runtime_config):
            raise ValueError("Invalid FP32 AdamW config schema.")
        for name, reference in runtime_config.items():
            self._validate_option_value(name, config[name], reference)
        beta1, beta2 = config["betas"]
        if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
            raise ValueError("FP32 AdamW betas must lie in [0, 1).")
        if config["cpu_update"] != self.cpu_update:
            raise ValueError(
                "FP32 AdamW cpu_update cannot change across checkpoint resume."
            )

        for target_name, key in (
            ("master_params", "master_param"),
            ("exp_avgs", "exp_avg"),
            ("exp_avg_sqs", "exp_avg_sq"),
        ):
            loaded = state_dict[target_name]
            if not isinstance(loaded, list) or len(loaded) != len(self.params):
                raise ValueError(f"Invalid FP32 AdamW {target_name} state.")
            for index, (param, src) in enumerate(zip(self.params, loaded, strict=True)):
                target = self.state[param][key]
                if not isinstance(src, torch.Tensor):
                    raise TypeError(
                        f"FP32 AdamW {target_name}[{index}] must be a tensor."
                    )
                if src.shape != target.shape or src.dtype != target.dtype:
                    raise ValueError(
                        f"FP32 AdamW {target_name}[{index}] shape/dtype mismatch."
                    )
        loaded_steps = state_dict["steps"]
        if (
            not isinstance(loaded_steps, list)
            or len(loaded_steps) != len(self.params)
            or any(
                isinstance(step, bool) or not isinstance(step, int) or step < 0
                for step in loaded_steps
            )
        ):
            raise ValueError("Invalid FP32 AdamW steps state.")
        if any(step > step_count for step in loaded_steps):
            raise ValueError("FP32 AdamW parameter step exceeds global step_count.")

        saved_groups = state_dict["param_groups"]
        if not isinstance(saved_groups, list) or len(saved_groups) != len(
            self.param_groups
        ):
            raise ValueError("Invalid FP32 AdamW param-group count.")
        for index, (saved_group, runtime_group) in enumerate(
            zip(saved_groups, self.param_groups, strict=True)
        ):
            if not isinstance(saved_group, dict) or set(saved_group) != {
                "param_count",
                "options",
            }:
                raise ValueError(f"Invalid FP32 AdamW param_groups[{index}] schema.")
            if saved_group["param_count"] != len(runtime_group["params"]):
                raise ValueError(
                    f"FP32 AdamW param_groups[{index}] cardinality mismatch."
                )
            options = saved_group["options"]
            runtime_options = {
                key: value for key, value in runtime_group.items() if key != "params"
            }
            if not isinstance(options, dict) or set(options) != set(runtime_options):
                raise ValueError(
                    f"FP32 AdamW param_groups[{index}] option schema mismatch."
                )
            for name, reference in runtime_options.items():
                self._validate_option_value(name, options[name], reference)
            if options.get("max_lr", self.lr) < options.get("min_lr", 0.0):
                raise ValueError(
                    f"FP32 AdamW param_groups[{index}] max_lr must be >= min_lr."
                )
            if options.get("end_wd", self.weight_decay) < options.get(
                "start_wd", self.weight_decay
            ):
                raise ValueError(
                    f"FP32 AdamW param_groups[{index}] end_wd must be >= start_wd."
                )

    def migrate_legacy_state_dict(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Upgrade the one pre-v2 format using explicitly selected runtime config.

        The legacy payload did not record betas/eps or complete group options, so
        this is intentionally not part of normal exact resume. The DCP loader may
        call it only behind its explicit legacy-checkpoint opt-in.
        """

        legacy_keys = {
            "type",
            "step_count",
            "master_params",
            "exp_avgs",
            "exp_avg_sqs",
            "steps",
            "weight_decays",
        }
        if not isinstance(state_dict, dict) or set(state_dict) != legacy_keys:
            raise ValueError("Invalid legacy FP32 AdamW state_dict schema.")
        if state_dict["type"] != "fp32_adamw":
            raise ValueError("Invalid legacy FP32 AdamW state_dict type.")
        weight_decays = state_dict["weight_decays"]
        if not isinstance(weight_decays, list) or len(weight_decays) != len(
            self.params
        ):
            raise ValueError("Invalid legacy FP32 AdamW weight_decays state.")

        migrated = self.state_dict()
        for name in ("step_count", "master_params", "exp_avgs", "exp_avg_sqs", "steps"):
            migrated[name] = state_dict[name]
        offset = 0
        for group_index, (group, migrated_group) in enumerate(
            zip(self.param_groups, migrated["param_groups"], strict=True)
        ):
            count = len(group["params"])
            group_weight_decays = weight_decays[offset : offset + count]
            offset += count
            if not group_weight_decays:
                continue
            first = group_weight_decays[0]
            if (
                isinstance(first, bool)
                or not isinstance(first, (int, float))
                or not math.isfinite(float(first))
                or first < 0
                or any(value != first for value in group_weight_decays[1:])
            ):
                raise ValueError(
                    "Legacy FP32 AdamW weight decay is invalid or inconsistent "
                    f"within param group {group_index}."
                )
            migrated_group["options"]["weight_decay"] = float(first)
        self.validate_state_dict(migrated)
        return migrated

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.validate_state_dict(state_dict)
        self.step_count = state_dict["step_count"]
        config = state_dict["config"]
        self.lr = float(config["lr"])
        self.weight_decay = float(config["weight_decay"])
        self.betas = tuple(config["betas"])
        self.eps = float(config["eps"])
        for target_name, key in (
            ("master_params", "master_param"),
            ("exp_avgs", "exp_avg"),
            ("exp_avg_sqs", "exp_avg_sq"),
        ):
            for param, src in zip(self.params, state_dict[target_name], strict=True):
                self.state[param][key].copy_(src)
        for param, step in zip(self.params, state_dict["steps"], strict=True):
            self.state[param]["step"] = step
        for saved_group, runtime_group in zip(
            state_dict["param_groups"], self.param_groups, strict=True
        ):
            for name, value in saved_group["options"].items():
                current = runtime_group.get(name)
                if isinstance(current, torch.Tensor) and isinstance(
                    value, torch.Tensor
                ):
                    current.copy_(value)
                else:
                    runtime_group[name] = value


def build_adamw_optimizer(
    params: Iterable[nn.Parameter] | Iterable[dict[str, Any]],
    *,
    all_params: Iterable[nn.Parameter],
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
    foreach: bool | str,
    use_fp32_master: bool,
    cpu_update: bool,
    model_param_dtypes: dict[int, torch.dtype] | None,
    opt,
) -> Any:
    param_groups = normalize_param_groups(params, default_weight_decay=weight_decay)
    fused_adam = maybe_build_te_fused_adam_optimizer(
        param_groups,
        all_params=all_params,
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
        opt=opt,
        use_fp32_master=use_fp32_master,
    )
    if fused_adam is not None:
        return fused_adam
    if use_fp32_master:
        return FP32AdamW(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            cpu_update=cpu_update,
            model_param_dtypes=model_param_dtypes,
        )
    if foreach not in {True, False, "auto"}:
        raise ValueError(
            f"adamw_foreach must be True, False, or 'auto', got {foreach!r}."
        )
    if foreach is False:
        return torch.optim.AdamW(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            foreach=False,
        )

    dtensor_param_groups, tensor_param_groups = split_dtensor_and_tensor_param_groups(
        param_groups, default_weight_decay=weight_decay
    )
    split_param_groups = [
        group for group in (dtensor_param_groups, tensor_param_groups) if group
    ]
    if foreach == "auto" and not dtensor_param_groups:
        return torch.optim.AdamW(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            foreach=False,
        )
    if len(split_param_groups) <= 1:
        return torch.optim.AdamW(
            split_param_groups[0] if split_param_groups else param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            foreach=True,
        )
    return ChainedOptimizer(
        torch.optim.AdamW(
            group, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps, foreach=True
        )
        for group in split_param_groups
    )


def maybe_build_te_fused_adam_optimizer(
    param_groups: list[dict[str, Any]],
    *,
    all_params: Iterable[nn.Parameter],
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
    opt,
    use_fp32_master: bool,
) -> Any | None:
    if not get_bool_opt(opt, "fsdp2_use_te_fused_adam", default=False):
        return None
    from transformer_engine.pytorch.optimizers.fused_adam import FusedAdam

    all_param_list = list(all_params)
    master_weights = get_bool_opt(
        opt,
        "master_weights",
        default=use_fp32_master and should_use_master_weights(all_param_list),
    )
    kwargs = dict(
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
        adam_w_mode=True,
        master_weights=master_weights,
        master_weight_dtype=get_dtype_opt(
            opt, "master_weight_dtype", default=torch.float32
        ),
        store_param_remainders=get_bool_opt(
            opt, "store_param_remainders", default=master_weights
        ),
        exp_avg_dtype=get_dtype_opt(opt, "exp_avg_dtype", default=torch.float32),
        exp_avg_sq_dtype=get_dtype_opt(opt, "exp_avg_sq_dtype", default=torch.float32),
    )
    return FusedAdam(
        param_groups, **filter_supported_kwargs(FusedAdam.__init__, kwargs)
    )


def filter_supported_kwargs(
    fn: Callable[..., Any], kwargs: dict[str, Any]
) -> dict[str, Any]:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in params}


def should_use_master_weights(params: Iterable[nn.Parameter]) -> bool:
    return any(
        param.is_floating_point() and param.dtype is not torch.float32
        for param in params
    )


def get_bool_opt(opt, attr: str, *, default: bool) -> bool:
    value = get_opt_value(opt, attr)
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def get_dtype_opt(opt, attr: str, *, default: torch.dtype) -> torch.dtype:
    value = get_opt_value(opt, attr)
    if value is None:
        return default
    if isinstance(value, torch.dtype):
        return value
    name = str(value).removeprefix("torch.")
    resolved = getattr(torch, name, None)
    if not isinstance(resolved, torch.dtype):
        raise ValueError(f"Unsupported dtype for FSDP2 TE FusedAdam: {value!r}.")
    return resolved


def get_opt_value(opt, attr: str):
    if opt is None:
        return None
    if isinstance(opt, dict):
        value = opt.get(attr)
        override = opt.get("override_optimizer_config")
    else:
        value = getattr(opt, attr, None)
        override = getattr(opt, "override_optimizer_config", None)
    if value is not None:
        return value
    if isinstance(override, dict):
        return override.get(attr)
    return None


def normalize_param_groups(
    params: Iterable[nn.Parameter] | Iterable[dict[str, Any]],
    *,
    default_weight_decay: float,
) -> list[dict[str, Any]]:
    items = list(params)
    if not items:
        return []
    if all(isinstance(item, dict) for item in items):
        groups: list[dict[str, Any]] = []
        for item in items:
            group = dict(item)
            group_params = list(group.get("params", ()))
            if not group_params:
                continue
            group["params"] = group_params
            group.setdefault("weight_decay", default_weight_decay)
            groups.append(group)
        return groups
    return [{"params": items, "weight_decay": default_weight_decay}]


def split_dtensor_and_tensor_param_groups(
    param_groups: Iterable[dict[str, Any]], *, default_weight_decay: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dtensor_groups: list[dict[str, Any]] = []
    tensor_groups: list[dict[str, Any]] = []
    for group in param_groups:
        dtensor_params, tensor_params = split_dtensor_and_tensor_params(group["params"])
        metadata = {key: value for key, value in group.items() if key != "params"}
        metadata.setdefault("weight_decay", default_weight_decay)
        if dtensor_params:
            dtensor_groups.append({**metadata, "params": dtensor_params})
        if tensor_params:
            tensor_groups.append({**metadata, "params": tensor_params})
    return dtensor_groups, tensor_groups


def split_dtensor_and_tensor_params(
    params: Iterable[nn.Parameter],
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    dtensor_params: list[nn.Parameter] = []
    tensor_params: list[nn.Parameter] = []
    for param in params:
        if is_dtensor_like(param):
            dtensor_params.append(param)
        else:
            tensor_params.append(param)
    return dtensor_params, tensor_params


def iter_torch_optimizers(optimizer: Any) -> Iterable[torch.optim.Optimizer]:
    if isinstance(optimizer, ChainedOptimizer):
        yield from optimizer.optimizers
    else:
        yield optimizer


def dtensor_from_local(
    local_tensor: torch.Tensor,
    device_mesh: Any,
    placements: Any,
    *,
    shape: Any = None,
    stride: Any = None,
) -> torch.Tensor:
    from torch.distributed.tensor import DTensor

    # ``DTensor.from_local`` infers the global shape as local_shard * mesh, which
    # is WRONG for unevenly-sharded params (FSDP2 pads the last shard). Pass the
    # original global shape/stride so the round-trip is exact for any dim not
    # divisible by the mesh size.
    if shape is not None:
        return DTensor.from_local(
            local_tensor, device_mesh, placements, shape=shape, stride=stride
        )
    return DTensor.from_local(local_tensor, device_mesh, placements)


__all__ = [
    "all_reduce_grad_",
    "build_adamw_optimizer",
    "copy_local_tensor_to_param_",
    "dtensor_from_local",
    "filter_supported_kwargs",
    "fsdp2_model_param_dtype",
    "get_bool_opt",
    "get_dtype_opt",
    "get_opt_value",
    "has_dtensor_grad_or_param",
    "is_dtensor_like",
    "iter_torch_optimizers",
    "local_grad_sq_sum",
    "normalize_param_groups",
    "to_local_tensor",
]
