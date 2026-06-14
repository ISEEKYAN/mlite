"""GLM-5 native lite protocol for Megatron Lite runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from megatron.lite.model.glm5.config import Glm5Config
from megatron.lite.model.glm5.lite.checkpoint import (
    EXPERT_CLASSIFIER,
    export_hf_weights as _export_hf_weights_impl,
    save_hf_weights as _save_hf_weights_impl,
    save_weights as _save_weights_impl,
)
from megatron.lite.model.glm5.lite.checkpoint import load_hf_weights as _load_hf_weights_impl
from megatron.lite.primitive.bundle import ModelBundle
from megatron.lite.primitive.parallel import ParallelState, init_parallel
from megatron.lite.primitive.recompute import apply_recompute, parse_recompute_spec, wrap_checkpoint
from megatron.lite.runtime.contracts.config import OptimizerConfig, ParallelConfig

MODULE_MAP = {
    "self_attn": lambda layer: layer.self_attn,
    "mlp": lambda layer: layer.mlp,
    "moe": lambda layer: layer.mlp if hasattr(layer.mlp, "dispatcher") else None,
    "experts": lambda layer: getattr(getattr(layer, "mlp", None), "experts", None),
    "shared_experts": lambda layer: getattr(getattr(layer, "mlp", None), "shared_experts", None),
}


def is_expert_param(name: str) -> bool:
    return EXPERT_CLASSIFIER(name)


@dataclass(frozen=True)
class ImplConfig:
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    optimizer: str | None = "distopt"
    hf_path: str = ""
    deterministic: bool = True
    recompute: str | list[str] | None = None
    offload: list[str] = field(default_factory=list)
    use_deepep: bool = False
    optimizer_config: OptimizerConfig | None = None
    mtp_enable: bool = False
    mtp_enable_train: bool = False
    mtp_detach_encoder: bool = False
    mtp_loss_scaling_factor: float = 0.1
    mtp_use_repeated_layer: bool | None = None


def build_model_config(source: str | Path | dict, **overrides) -> Glm5Config:
    if isinstance(source, dict):
        cfg = Glm5Config._from_hf_dict(source)
    else:
        cfg = Glm5Config.from_hf(str(source))
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def _forward_step(model: nn.Module, batch: dict) -> dict:
    kwargs: dict[str, Any] = {"input_ids": batch["input_ids"], "labels": batch.get("labels")}
    if kwargs["input_ids"].dim() == 1:
        kwargs["input_ids"] = kwargs["input_ids"].unsqueeze(0)
    for key in ("position_ids", "attention_mask"):
        if key in batch:
            kwargs[key] = batch[key]
    for key in ("loss_mask", "temperature", "calculate_entropy"):
        if key in batch:
            kwargs[key] = batch[key]
    return model(**kwargs)


def _apply_glm5_recompute(
    layers: nn.ModuleList, recompute_spec: list[str], ps: ParallelState
) -> None:
    if not recompute_spec:
        return
    if "full" not in recompute_spec or ps.ep_size <= 1:
        apply_recompute(layers, recompute_spec, MODULE_MAP)
        return

    for layer in layers:
        wrap_checkpoint(layer, preserve_rng_state=True)

    remaining = [name for name in recompute_spec if name != "full"]
    if remaining:
        apply_recompute(layers, remaining, MODULE_MAP)


def _validate_parallel_scope(p: ParallelConfig) -> None:
    etp = 1 if p.etp is None else p.etp
    if (p.tp, etp, p.vpp) != (1, 1, 1):
        raise NotImplementedError(
            "GLM5 native lite currently supports TP=ETP=VPP=1. "
            "EP/PP/CP are wired through existing Megatron Lite primitives."
        )
    if p.ep < 1 or p.pp < 1 or p.cp < 1:
        raise ValueError(
            "ParallelConfig ep/pp/cp must be >= 1, " f"got ep={p.ep}, pp={p.pp}, cp={p.cp}."
        )


def build_model(model_cfg: Glm5Config, *, impl_cfg: ImplConfig) -> ModelBundle:
    from megatron.lite.model.glm5.lite.model import Glm5ForCausalLM

    _validate_parallel_scope(impl_cfg.parallel)
    mtp_enable = bool(impl_cfg.mtp_enable)
    mtp_enable_train = mtp_enable and bool(impl_cfg.mtp_enable_train)
    if mtp_enable:
        if model_cfg.num_nextn_predict_layers <= 0:
            raise ValueError("mtp_enable=True but HF config has no num_nextn_predict_layers.")
        model_cfg.mtp_loss_scaling_factor = impl_cfg.mtp_loss_scaling_factor
        if impl_cfg.mtp_use_repeated_layer is not None:
            model_cfg.mtp_use_repeated_layer = impl_cfg.mtp_use_repeated_layer
    else:
        model_cfg.num_nextn_predict_layers = 0

    ps = init_parallel(impl_cfg.parallel)
    recompute_spec = parse_recompute_spec(impl_cfg.recompute)
    train_cfg = SimpleNamespace(use_deepep=impl_cfg.use_deepep)
    chunk = (
        Glm5ForCausalLM(
            model_cfg,
            train_cfg=train_cfg,
            ps=ps,
            mtp_enable=mtp_enable,
            mtp_enable_train=mtp_enable_train,
            mtp_detach_encoder=impl_cfg.mtp_detach_encoder,
        )
        .to(torch.bfloat16)
        .cuda()
    )
    chunks = [chunk]

    _apply_glm5_recompute(chunk.model.layers, recompute_spec, ps)

    if impl_cfg.offload:
        from megatron.lite.primitive.recompute import apply_offload

        apply_offload(chunk.model.layers, impl_cfg.offload, MODULE_MAP)

    optimizer = None
    optimizer_backend = "none"
    finalize_grads = None
    post_model_load_hook = None
    if impl_cfg.optimizer == "distopt":
        optimizer, finalize_grads = _build_distopt_optimizer(chunks, model_cfg, impl_cfg, ps)
        from megatron.lite.runtime.megatron_utils import register_training_hooks

        register_training_hooks(chunks, optimizer)
        optimizer_backend = "distopt"
    elif impl_cfg.optimizer == "fsdp2":
        optimizer_backend = "fsdp2"

        def _post_model_load_hook():
            from megatron.lite.model.glm5.lite.model import Glm5Layer
            from megatron.lite.primitive.optimizers.fsdp2 import build_fsdp2_training_optimizer

            return {
                "optimizer": build_fsdp2_training_optimizer(
                    chunks,
                    impl_cfg.optimizer_config,
                    ps,
                    unit_modules=(Glm5Layer,),
                    expert_classifier=is_expert_param,
                    deterministic=impl_cfg.deterministic,
                    vpp=1,
                    leaf_module_names=(),
                )
            }

        post_model_load_hook = _post_model_load_hook
    elif impl_cfg.optimizer is not None:
        raise ValueError(f"Unsupported GLM5 optimizer: {impl_cfg.optimizer!r}")

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


def _build_distopt_optimizer(chunks, model_cfg, impl_cfg, ps):
    from megatron.lite.primitive.optimizers.megatron_wrap import build_distopt_training_optimizer

    return build_distopt_training_optimizer(
        chunks,
        model_cfg=model_cfg,
        impl_cfg=impl_cfg,
        model_name="glm5",
        ps=ps,
        is_expert=is_expert_param,
        deterministic=impl_cfg.deterministic,
    )


def load_hf_weights(
    chunk: nn.Module, hf_path: str, model_cfg: Glm5Config, ps: ParallelState
) -> None:
    if not hf_path:
        return
    _load_hf_weights_impl(chunk, hf_path, model_cfg, ps)


def export_hf_weights(chunks: list[nn.Module], model_cfg: Glm5Config, ps: ParallelState, **kwargs):
    yield from _export_hf_weights_impl(chunks, model_cfg, ps, **kwargs)


def save_hf_weights(
    chunks: list[nn.Module],
    path: str,
    model_cfg: Glm5Config,
    ps: ParallelState,
    *,
    export_dtype: torch.dtype | None = None,
) -> None:
    _save_hf_weights_impl(chunks, path, model_cfg, ps, export_dtype=export_dtype)


def save_weights(
    chunks: list[nn.Module], path: str, model_cfg: Glm5Config, ps: ParallelState
) -> None:
    _save_weights_impl(chunks, path, model_cfg, ps)


def vocab_size(model_cfg: Glm5Config) -> int | None:
    return model_cfg.vocab_size
