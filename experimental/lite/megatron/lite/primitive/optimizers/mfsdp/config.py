"""Megatron-FSDP config lowering and validation.

This module is the isolation boundary for MCore-specific optimizer/DDP config
shape. The runtime-facing primitive should stay close to FSDP2, while this
module owns Megatron-FSDP's supported surface and fail-fast checks.
"""

from __future__ import annotations

import os
from dataclasses import fields, is_dataclass
from typing import Any

import torch  # pyright: ignore[reportMissingImports]

_SUPPORTED_MODELS = {
    "qwen3_moe",
    "qwen3_moe_mcore_hybrid",
    "qwen3_5",
    "qwen3_5_mcore_hybrid",
}
_SUPPORTED_OPTIMIZERS = {"adam", "sgd"}
_SUPPORTED_SHARDING_STRATEGIES = {"no_shard", "optim", "optim_grads", "optim_grads_params"}
_REQUIRED_DDP_KNOB_VALUES = {
    "use_distributed_optimizer": True,
    "use_megatron_fsdp": True,
}
_UNSUPPORTED_OPTIMIZATION_KNOBS = {
    "use_hsdp",
    "hsdp",
}
_DDP_ALIAS_KEYS = {
    "mfsdp_sharding_strategy": "data_parallel_sharding_strategy",
    "megatron_fsdp_sharding_strategy": "data_parallel_sharding_strategy",
}
_DDP_DEFAULT_KEYS = {
    "use_distributed_optimizer",
    "use_megatron_fsdp",
    "data_parallel_sharding_strategy",
    "overlap_grad_reduce",
    "overlap_param_gather",
    "grad_reduce_in_fp32",
    "nccl_ub",
    "fsdp_double_buffer",
    "num_distributed_optimizer_instances",
}
_DTYPE_DDP_KEYS = {
    "megatron_fsdp_main_params_dtype",
    "megatron_fsdp_main_grads_dtype",
    "megatron_fsdp_grad_comm_dtype",
}
_OPTIMIZER_BASE_KEYS = {
    "optimizer",
    "lr",
    "min_lr",
    "clip_grad",
    "weight_decay",
    "lr_warmup_steps_ratio",
    "total_training_steps",
    "lr_warmup_steps",
    "lr_warmup_init",
    "lr_decay_steps",
    "lr_decay_style",
    "weight_decay_incr_style",
    "lr_wsd_decay_style",
    "lr_wsd_decay_steps",
    "use_checkpoint_opt_param_scheduler",
    "adam_beta1",
    "adam_beta2",
    "adam_eps",
    "offload_fraction",
    "use_precision_aware_optimizer",
    "decoupled_weight_decay",
    "override_optimizer_config",
}


def validate_mfsdp_config(engine_cfg) -> None:
    """Validate the supported surface for optimizer='megatron_fsdp'."""
    if engine_cfg.model_name not in _SUPPORTED_MODELS:
        raise ValueError(
            "optimizer_impl='megatron_fsdp' currently supports only qwen3_moe "
            "and qwen3_5 variants."
        )
    validate_mfsdp_topology(engine_cfg.parallel, model_name=engine_cfg.model_name)
    opt = engine_cfg.optimizer
    validate_optimizer_name(getattr(opt, "optimizer", "adam"))
    validate_precision_aware_disabled(opt)
    validate_optimization_knobs(opt)
    validate_mfsdp_topology_optimizer_combo(engine_cfg.parallel, opt)
    if os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS") == "1":
        raise ValueError("Megatron-FSDP requires CUDA_DEVICE_MAX_CONNECTIONS > 1 or unset.")


def validate_mfsdp_topology(parallel_cfg, *, model_name: str | None = None) -> None:
    """Validate supported topology surfaces for Megatron-FSDP."""
    pp = int(getattr(parallel_cfg, "pp", 1) or 1)
    vpp = int(getattr(parallel_cfg, "vpp", 1) or 1)

    if vpp > 1 and pp <= 1:
        raise ValueError(
            "optimizer_impl='megatron_fsdp' requires pp>1 when vpp>1."
        )


def validate_mfsdp_topology_optimizer_combo(parallel_cfg, opt) -> None:
    pp = int(getattr(parallel_cfg, "pp", 1) or 1)
    vpp = int(getattr(parallel_cfg, "vpp", 1) or 1)
    values = dict(getattr(opt, "override_optimizer_config", None) or {})
    instances = values.get(
        "num_distributed_optimizer_instances",
        getattr(opt, "num_distributed_optimizer_instances", None),
    )
    if _invalid_distributed_optimizer_instances(instances):
        return
    normalized_instances = _distributed_optimizer_instance_override(instances)
    if normalized_instances is not None and normalized_instances > 1 and (pp > 1 or vpp > 1):
        raise ValueError(
            "optimizer_impl='megatron_fsdp' does not support "
            "num_distributed_optimizer_instances>1 with pp/vpp yet."
        )


def split_mfsdp_overrides(
    opt,
    ddp_config_cls,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split generic optimizer overrides into OptimizerConfig vs DDPConfig keys."""
    ddp_fields = {field.name for field in fields(ddp_config_cls)}
    raw_overrides = _collect_mfsdp_overrides(opt, ddp_fields)
    ddp_overrides: dict[str, Any] = {}
    opt_overrides: dict[str, Any] = {}

    for key, value in raw_overrides.items():
        ddp_key = _DDP_ALIAS_KEYS.get(key, key)
        if ddp_key == "num_distributed_optimizer_instances":
            if _invalid_distributed_optimizer_instances(value):
                raise ValueError(
                    "optimizer_impl='megatron_fsdp' requires "
                    "num_distributed_optimizer_instances to be a positive integer."
                )
            normalized = _distributed_optimizer_instance_override(value)
            if normalized is not None:
                if ddp_key not in ddp_fields:
                    raise ValueError(
                        "Megatron-Core DistributedDataParallelConfig does not support "
                        f"{ddp_key!r} in this environment."
                )
                ddp_overrides[ddp_key] = normalized
        elif ddp_key in ddp_fields:
            _validate_ddp_knob_value(ddp_key, value)
            ddp_overrides[ddp_key] = value
        elif ddp_key in _UNSUPPORTED_OPTIMIZATION_KNOBS:
            if _truthy_feature_value(value):
                raise ValueError(
                    "optimizer_impl='megatron_fsdp' does not accept hsdp/use_hsdp aliases; "
                    "set num_distributed_optimizer_instances explicitly."
                )
        elif ddp_key in _DDP_DEFAULT_KEYS or ddp_key in _DTYPE_DDP_KEYS:
            raise ValueError(
                f"Megatron-Core DistributedDataParallelConfig does not support "
                f"{ddp_key!r} in this environment."
            )
        else:
            opt_overrides[key] = value
    validate_optimization_knobs(opt, opt_overrides)
    return opt_overrides, ddp_overrides


def validate_optimizer_name(optimizer_name: str) -> None:
    if optimizer_name not in _SUPPORTED_OPTIMIZERS:
        raise ValueError(
            "optimizer_impl='megatron_fsdp' supports only adam/sgd through "
            f"Megatron-Core DistributedOptimizer, got {optimizer_name!r}."
        )


def _collect_mfsdp_overrides(opt, ddp_fields: set[str]) -> dict[str, Any]:
    """Collect explicit overrides from dict-style and benchmark attribute-style inputs."""
    raw_overrides = dict(getattr(opt, "override_optimizer_config", None) or {})
    optimizer_base_keys = set(_OPTIMIZER_BASE_KEYS)
    if is_dataclass(opt):
        optimizer_base_keys.update(field.name for field in fields(opt))

    for key, value in vars(opt).items():
        if key.startswith("_") or key in optimizer_base_keys:
            continue
        raw_overrides.setdefault(key, value)
    return raw_overrides


def validate_precision_aware_disabled(opt, opt_overrides: dict[str, Any] | None = None) -> None:
    precision_aware = getattr(opt, "use_precision_aware_optimizer", None)
    raw_overrides = dict(getattr(opt, "override_optimizer_config", None) or {})
    if "use_precision_aware_optimizer" in raw_overrides:
        precision_aware = raw_overrides["use_precision_aware_optimizer"]
    if opt_overrides and "use_precision_aware_optimizer" in opt_overrides:
        precision_aware = opt_overrides["use_precision_aware_optimizer"]
    if precision_aware not in {None, True, False}:
        raise ValueError("use_precision_aware_optimizer must be a boolean when set.")
    if precision_aware is True:
        raise ValueError(
            "optimizer_impl='megatron_fsdp' does not support "
            "use_precision_aware_optimizer=True in this image; Slurm smoke hit "
            "transformer_engine::multi_tensor_scale_cuda segfault."
        )


def validate_optimization_knobs(opt, opt_overrides: dict[str, Any] | None = None) -> None:
    """Validate Phase 3.4 optimization knobs before lowering into MCore config."""
    values = dict(getattr(opt, "override_optimizer_config", None) or {})
    if opt_overrides:
        values.update(opt_overrides)
    for key in _UNSUPPORTED_OPTIMIZATION_KNOBS:
        value = values.get(key, getattr(opt, key, None))
        if _truthy_feature_value(value):
            raise ValueError(
                "optimizer_impl='megatron_fsdp' does not accept hsdp/use_hsdp aliases; "
                "set num_distributed_optimizer_instances explicitly."
            )
    instances = values.get(
        "num_distributed_optimizer_instances",
        getattr(opt, "num_distributed_optimizer_instances", None),
    )
    if _invalid_distributed_optimizer_instances(instances):
        raise ValueError(
            "optimizer_impl='megatron_fsdp' requires "
            "num_distributed_optimizer_instances to be a positive integer."
        )
    nccl_ub = values.get("nccl_ub", getattr(opt, "nccl_ub", None))
    if _truthy_feature_value(nccl_ub) and _cuda_alloc_conf_expands_segments():
        raise ValueError(
            "optimizer_impl='megatron_fsdp' requires PYTORCH_CUDA_ALLOC_CONF without "
            "expandable_segments:True when nccl_ub=True; unset it before enabling UBR."
        )
    for key, required_value in _REQUIRED_DDP_KNOB_VALUES.items():
        value = values.get(key, getattr(opt, key, required_value))
        if value != required_value:
            raise ValueError(
                f"optimizer_impl='megatron_fsdp' requires {key}={required_value!r}."
            )
    validate_precision_aware_disabled(opt, opt_overrides)


def build_mfsdp_ddp_config(ddp_config_cls, overrides: dict[str, Any]):
    ddp_fields = {field.name for field in fields(ddp_config_cls)}
    required_fields = {
        "use_distributed_optimizer",
        "use_megatron_fsdp",
        "data_parallel_sharding_strategy",
    }
    missing = sorted(required_fields - ddp_fields)
    if missing:
        raise ValueError(
            "Installed Megatron-Core DistributedDataParallelConfig is missing "
            f"required Megatron-FSDP fields: {missing}."
        )
    kwargs: dict[str, Any] = {
        "use_distributed_optimizer": True,
        "use_megatron_fsdp": True,
        "data_parallel_sharding_strategy": "optim_grads_params",
        "overlap_grad_reduce": False,
        "overlap_param_gather": False,
        "grad_reduce_in_fp32": True,
        "megatron_fsdp_main_params_dtype": torch.float32,
        "megatron_fsdp_main_grads_dtype": None,
        "megatron_fsdp_grad_comm_dtype": None,
    }
    kwargs.update(overrides)
    strategy = kwargs["data_parallel_sharding_strategy"]
    if strategy not in _SUPPORTED_SHARDING_STRATEGIES:
        raise ValueError(
            "Megatron-FSDP data_parallel_sharding_strategy must be one of "
            f"{sorted(_SUPPORTED_SHARDING_STRATEGIES)}, got {strategy!r}."
        )
    for key, required_value in _REQUIRED_DDP_KNOB_VALUES.items():
        if kwargs.get(key) != required_value:
            raise ValueError(
                f"optimizer_impl='megatron_fsdp' requires {key}={required_value!r}."
            )
    for key in _DTYPE_DDP_KEYS:
        if key in kwargs:
            kwargs[key] = coerce_dtype(kwargs.get(key), key=key)
    supported_kwargs = {key: value for key, value in kwargs.items() if key in ddp_fields}
    return ddp_config_cls(**supported_kwargs)


def coerce_dtype(value: Any, *, key: str) -> torch.dtype | None:
    if value is None or isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        dtype_name = value.lower().removeprefix("torch.")
        if dtype_name in {"none", "null"}:
            return None
        mapping = {
            "fp32": torch.float32,
            "float32": torch.float32,
            "float": torch.float32,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "half": torch.float16,
        }
        if dtype_name in mapping:
            return mapping[dtype_name]
    raise ValueError(f"{key} must be a torch.dtype, None, or dtype string, got {value!r}.")


def _validate_ddp_knob_value(key: str, value: Any) -> None:
    if key in _REQUIRED_DDP_KNOB_VALUES and value != _REQUIRED_DDP_KNOB_VALUES[key]:
        raise ValueError(
            f"optimizer_impl='megatron_fsdp' requires "
            f"{key}={_REQUIRED_DDP_KNOB_VALUES[key]!r}."
        )


def _truthy_feature_value(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, int | float) and value == 0:
        return False
    if isinstance(value, str) and value.lower() in {"", "0", "false", "none", "null", "off"}:
        return False
    return True


def _cuda_alloc_conf_expands_segments() -> bool:
    value = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    for item in value.split(","):
        key, _, raw = item.strip().partition(":")
        if key.lower() == "expandable_segments" and raw.lower() in {"1", "true", "yes", "on"}:
            return True
    return False


def _invalid_distributed_optimizer_instances(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, int | float):
        return int(value) != value or value < 1
    if isinstance(value, str):
        normalized = value.lower()
        if normalized in {"", "none", "null"}:
            return False
        try:
            return int(normalized) < 1
        except ValueError:
            return True
    return True


def _distributed_optimizer_instance_override(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return int(value)
    if isinstance(value, str):
        normalized = value.lower()
        if normalized in {"", "none", "null"}:
            return None
        return int(normalized)
    return None


__all__ = [
    "build_mfsdp_ddp_config",
    "coerce_dtype",
    "split_mfsdp_overrides",
    "validate_mfsdp_config",
    "validate_mfsdp_topology",
    "validate_mfsdp_topology_optimizer_combo",
    "validate_optimization_knobs",
    "validate_optimizer_name",
    "validate_precision_aware_disabled",
]
