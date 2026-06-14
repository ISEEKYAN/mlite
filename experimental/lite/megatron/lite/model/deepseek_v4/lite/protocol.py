# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
from megatron.lite.model.deepseek_v4.lite.checkpoint import (
    EXPERT_CLASSIFIER,
    export_hf_weights as _export_hf_weights_impl,
    load_hf_weights as _load_hf_weights_impl,
    save_hf_weights as _save_hf_weights_impl,
)
from megatron.lite.primitive.parallel.cp import (
    contiguous_position_ids_for_cp,
    local_contiguous_sequence_tensor_for_cp,
    split_packed_contiguous_for_cp,
)
from megatron.lite.primitive.bundle import ModelBundle
from megatron.lite.primitive.parallel import ParallelState, init_parallel
from megatron.lite.primitive.recompute import apply_recompute, parse_recompute_spec
from megatron.lite.runtime.contracts import Batch, OptimizerConfig, PackedBatch, ParallelConfig


def is_expert_param(name: str) -> bool:
    return EXPERT_CLASSIFIER(name)


@dataclass(frozen=True)
class ImplConfig:
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    optimizer: str | None = "dist_opt"
    optimizer_config: OptimizerConfig | None = None
    hf_path: str = ""
    recompute: list[str] = field(default_factory=list)
    offload: list[str] = field(default_factory=list)
    use_thd: bool = False
    use_deepep: bool = False
    attention_backend_override: str | None = None
    deterministic: bool = True
    mtp_enable: bool = True
    mtp_enable_train: bool = False
    mtp_detach_encoder: bool = False
    mtp_num_layers: int | None = None
    num_nextn_predict_layers: int | None = None
    mtp_loss_scaling_factor: float = 0.1


MODULE_MAP = {
    "attn": lambda layer: layer.self_attn,
    "core_attn": lambda layer: layer.self_attn,
    "moe": lambda layer: layer.mlp,
    "experts": lambda layer: layer.mlp.experts,
    "router": lambda layer: layer.mlp.gate,
    "attn_norm": lambda layer: layer.input_layernorm,
    "ffn_norm": lambda layer: layer.post_attention_layernorm,
}

_OPTIONAL_MODEL_KWARGS = ("attention_mask", "loss_mask", "temperature", "calculate_entropy")
_PACKED_MODEL_KWARGS = {
    "input_ids",
    "position_ids",
    "attention_mask",
    "labels",
    "loss_mask",
    "temperature",
    "calculate_entropy",
    "enable_mtp",
}
_MISSING = object()


def build_model_config(source: str | Path | dict, **overrides) -> DeepseekV4Config:
    if isinstance(source, dict):
        cfg = DeepseekV4Config._from_hf_dict(source)
    else:
        cfg = DeepseekV4Config.from_hf(str(source))
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def _normalize_ds4_position_ids(position_ids: torch.Tensor | None) -> torch.Tensor | None:
    if position_ids is None:
        return None
    if position_ids.dim() == 3:
        if position_ids.size(0) == 3:
            position_ids = position_ids[0]
        elif position_ids.size(1) == 1:
            position_ids = position_ids.squeeze(1)
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)
    return position_ids


def _as_batch_row(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is not None and tensor.dim() == 1:
        return tensor.unsqueeze(0)
    return tensor


def _infer_cp_local_seq_len(
    *,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor | None,
    cp_size: int,
) -> int:
    seq_len = input_ids.size(1)
    if cp_size <= 1:
        return seq_len
    if position_ids is not None:
        pos_len = position_ids.size(-1)
        if pos_len == seq_len * cp_size:
            return seq_len
        if pos_len == seq_len:
            return seq_len
    if seq_len % cp_size == 0:
        return seq_len // cp_size
    return seq_len


def _batch_value(batch: Batch, key: str, default: Any = None) -> Any:
    if hasattr(batch, key):
        value = getattr(batch, key)
        return default if value is None and default is _MISSING else value
    extras = getattr(batch, "extras", None)
    return extras.get(key, default) if isinstance(extras, dict) else default


def _prepare_packed_contiguous_cp_kwargs(
    model: nn.Module,
    kwargs: dict[str, Any],
    batch: PackedBatch,
) -> dict[str, Any]:
    ps = getattr(model, "ps", ParallelState())
    cu_seqlens = batch.cu_seqlens
    if cu_seqlens is None:
        raise ValueError("Packed DS4 CP requires PackedBatch.cu_seqlens.")

    if kwargs.get("position_ids") is None:
        total_tokens = int(cu_seqlens[-1].item())
        kwargs["position_ids"] = torch.arange(
            total_tokens,
            device=kwargs["input_ids"].device,
        ).unsqueeze(0)

    for key in ("input_ids", "labels", "loss_mask", "position_ids"):
        if kwargs.get(key) is not None:
            kwargs[key] = split_packed_contiguous_for_cp(
                kwargs[key],
                cu_seqlens,
                cp_rank=ps.cp_rank,
                cp_size=ps.cp_size,
                name=key,
            )
    kwargs["enable_mtp"] = False
    return {key: value for key, value in kwargs.items() if key in _PACKED_MODEL_KWARGS}


def _base_model_forward_kwargs(batch: Batch) -> dict[str, Any]:
    input_ids = _batch_value(batch, "input_ids", _MISSING)
    if input_ids is _MISSING:
        raise KeyError("DeepSeek V4 forward batch requires input_ids.")
    input_ids = _as_batch_row(input_ids)
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "labels": _as_batch_row(_batch_value(batch, "labels")),
    }
    for key in _OPTIONAL_MODEL_KWARGS:
        value = _batch_value(batch, key, _MISSING)
        if value is not _MISSING:
            if key == "loss_mask":
                value = _as_batch_row(value)
            kwargs[key] = value
    position_ids = _batch_value(batch, "position_ids")
    if isinstance(batch, PackedBatch) and position_ids is None:
        position_ids = batch.make_position_ids()
    position_ids = _normalize_ds4_position_ids(position_ids)
    if position_ids is not None:
        kwargs["position_ids"] = position_ids
    return kwargs


def _prepare_contiguous_cp_kwargs(model: nn.Module, kwargs: dict[str, Any]) -> dict[str, Any]:
    ps = getattr(model, "ps", ParallelState())
    local_seq_len = _infer_cp_local_seq_len(
        input_ids=kwargs["input_ids"],
        position_ids=kwargs.get("position_ids"),
        cp_size=ps.cp_size,
    )
    kwargs["input_ids"] = local_contiguous_sequence_tensor_for_cp(
        kwargs["input_ids"],
        local_seq_len=local_seq_len,
        cp_rank=ps.cp_rank,
        cp_size=ps.cp_size,
        name="input_ids",
    )
    if kwargs.get("position_ids") is None:
        full_seq_len = local_seq_len * ps.cp_size
        position_ids = contiguous_position_ids_for_cp(
            full_seq_len,
            cp_rank=ps.cp_rank,
            cp_size=ps.cp_size,
            device=kwargs["input_ids"].device,
        ).expand(kwargs["input_ids"].size(0), -1)
    else:
        position_ids = local_contiguous_sequence_tensor_for_cp(
            kwargs["position_ids"],
            local_seq_len=kwargs["input_ids"].size(1),
            cp_rank=ps.cp_rank,
            cp_size=ps.cp_size,
            name="position_ids",
        )
    kwargs["position_ids"] = position_ids
    for key in ("labels", "loss_mask"):
        if kwargs.get(key) is not None:
            kwargs[key] = local_contiguous_sequence_tensor_for_cp(
                kwargs[key],
                local_seq_len=kwargs["input_ids"].size(1),
                cp_rank=ps.cp_rank,
                cp_size=ps.cp_size,
                name=key,
            )
    return kwargs


def _prepare_model_forward_kwargs(model: nn.Module, batch: Batch) -> dict[str, Any]:
    kwargs = _base_model_forward_kwargs(batch)
    if isinstance(batch, PackedBatch):
        kwargs["enable_mtp"] = False
        return _prepare_packed_contiguous_cp_kwargs(model, kwargs, batch)
    return _prepare_contiguous_cp_kwargs(model, kwargs)


def _forward_step(model: nn.Module, batch: Batch) -> dict:
    return model(**_prepare_model_forward_kwargs(model, batch))


def _apply_mtp_config(model_cfg: DeepseekV4Config, impl_cfg: ImplConfig) -> None:
    override = impl_cfg.num_nextn_predict_layers
    if override is None:
        override = impl_cfg.mtp_num_layers
    if override is not None:
        if override < 0:
            raise ValueError(f"DeepSeek V4 MTP layer count must be >=0, got {override}.")
        model_cfg.num_nextn_predict_layers = int(override)
    if impl_cfg.mtp_enable:
        if model_cfg.num_nextn_predict_layers <= 0:
            raise ValueError("mtp_enable=True but DeepSeek V4 config has no MTP layers.")
        model_cfg.mtp_loss_scaling_factor = impl_cfg.mtp_loss_scaling_factor
    else:
        model_cfg.num_nextn_predict_layers = 0


def _optimizer_backend_name(optimizer: Any) -> str | None:
    if isinstance(optimizer, dict) or isinstance(optimizer, OptimizerConfig):
        return "dist_opt"
    return optimizer


def _configure_attention_backend(chunks: list[nn.Module], *, backend: str | None) -> None:
    backend_name = backend or "torch"
    for chunk in chunks:
        for module in chunk.modules():
            if hasattr(module, "attention_backend"):
                module.attention_backend = backend_name


def _iter_transformer_units(chunk: nn.Module) -> list[nn.Module]:
    model = getattr(chunk, "model", None)
    if model is None:
        return []
    layers = list(getattr(model, "layers", {}).values())
    mtp_layers = list(getattr(model, "mtp", []))
    return [*layers, *mtp_layers]


def build_model(model_cfg: DeepseekV4Config, *, impl_cfg: ImplConfig) -> ModelBundle:
    from megatron.lite.model.deepseek_v4.lite.model import DeepseekV4ForCausalLM

    _ = impl_cfg.use_thd
    _apply_mtp_config(model_cfg, impl_cfg)
    ps = init_parallel(impl_cfg.parallel)
    p = impl_cfg.parallel
    vpp = None if p.vpp == 1 else p.vpp
    train_cfg = SimpleNamespace(
        tp=ps.tp_size,
        ep=ps.ep_size,
        etp=ps.etp_size,
        pp=ps.pp_size,
        cp=ps.cp_size,
        vpp=vpp,
        fp8=False,
        use_deepep=impl_cfg.use_deepep,
    )

    def _chunk(i: int | None = None):
        return (
            DeepseekV4ForCausalLM(
                model_cfg,
                train_cfg=train_cfg,
                ps=ps,
                vpp=vpp,
                vpp_chunk_id=i,
                use_deepep=impl_cfg.use_deepep,
            )
            .to(torch.bfloat16)
            .cuda()
        )

    chunks = [_chunk(i) for i in range(vpp)] if vpp is not None else [_chunk()]
    _configure_attention_backend(chunks, backend=impl_cfg.attention_backend_override)

    recompute_spec = parse_recompute_spec(impl_cfg.recompute)
    if recompute_spec:
        for chunk in chunks:
            apply_recompute(_iter_transformer_units(chunk), recompute_spec, MODULE_MAP)

    if impl_cfg.offload:
        from megatron.lite.primitive.recompute import apply_offload

        for chunk in chunks:
            apply_offload(_iter_transformer_units(chunk), impl_cfg.offload, MODULE_MAP)

    optimizer = None
    finalize_grads = None
    post_model_load_hook = None
    optimizer_backend = "none"
    optimizer_name = _optimizer_backend_name(impl_cfg.optimizer)
    if optimizer_name == "dist_opt":
        from megatron.lite.primitive.ckpt import attach_model_sharded_state_dict
        from megatron.lite.primitive.optimizers.megatron_wrap import (
            build_dist_opt_training_optimizer,
        )
        from megatron.lite.runtime.megatron_utils import register_training_hooks

        optimizer, finalize_grads = build_dist_opt_training_optimizer(
            chunks,
            model_cfg=model_cfg,
            impl_cfg=impl_cfg,
            ps=ps,
            model_name="deepseek_v4",
            is_expert=is_expert_param,
            deterministic=impl_cfg.deterministic,
        )
        attach_model_sharded_state_dict(chunks, ps, is_expert=is_expert_param)
        register_training_hooks(chunks, optimizer)
        optimizer_backend = "dist_opt"
    elif optimizer_name == "fsdp2":
        optimizer_backend = "fsdp2"

        def _post_model_load_hook():
            from megatron.lite.model.deepseek_v4.lite.model import DeepseekV4Layer
            from megatron.lite.primitive.optimizers.fsdp2 import build_fsdp2_training_optimizer

            return {
                "optimizer": build_fsdp2_training_optimizer(
                    chunks,
                    impl_cfg.optimizer_config,
                    ps,
                    unit_modules=(DeepseekV4Layer,),
                    expert_classifier=is_expert_param,
                    deterministic=impl_cfg.deterministic,
                    vpp=impl_cfg.parallel.vpp,
                    leaf_module_names=(),
                    use_fp32_shards=False,
                )
            }

        post_model_load_hook = _post_model_load_hook
    elif optimizer_name is None:
        optimizer_backend = "none"
    else:
        raise ValueError(f"Unknown DeepSeek V4 lite optimizer: {impl_cfg.optimizer!r}.")

    return ModelBundle(
        chunks=chunks,
        parallel_state=ps,
        optimizer=optimizer,
        finalize_grads=finalize_grads,
        forward_step=_forward_step,
        extras={
            "model_cfg": model_cfg,
            "optimizer_backend": optimizer_backend,
            "post_model_load_hook": post_model_load_hook,
        },
    )


def load_hf_weights(
    chunk: nn.Module, hf_path: str, model_cfg: DeepseekV4Config, ps: ParallelState
) -> None:
    if not hf_path:
        return
    _load_hf_weights_impl(chunk, hf_path, model_cfg, ps)


def export_hf_weights(
    chunks: list[nn.Module], model_cfg: DeepseekV4Config, ps: ParallelState, **kwargs
):
    yield from _export_hf_weights_impl(chunks, model_cfg, ps, **kwargs)


def save_hf_weights(
    chunks: list[nn.Module], path: str, model_cfg: DeepseekV4Config, ps: ParallelState, **kwargs
) -> None:
    _save_hf_weights_impl(chunks, path, model_cfg, ps, **kwargs)


def vocab_size(model_cfg: DeepseekV4Config) -> int | None:
    return model_cfg.vocab_size
