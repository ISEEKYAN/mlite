"""Kimi K2 lite impl — native model protocol for Megatron Lite runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from megatron.lite.model.kimi_k2.config import KimiK2Config
from megatron.lite.model.kimi_k2.lite.checkpoint import (
    load_hf_weights as _load_hf_weights_impl,
)
from megatron.lite.primitive.bundle import ModelBundle
from megatron.lite.primitive.parallel import ParallelState, init_parallel
from megatron.lite.primitive.recompute import apply_recompute, parse_recompute_spec
from megatron.lite.runtime.contracts import OptimizerConfig, ParallelConfig


def is_expert_param(name: str) -> bool:
    return "experts" in name and "router" not in name and "shared" not in name


def _maybe(module_name: str):
    def getter(layer):
        module = getattr(layer, module_name, None)
        return module

    return getter


def _moe_module(name: str):
    def getter(layer):
        moe = getattr(layer, "moe", None)
        return getattr(moe, name, None) if moe is not None else None

    return getter


MODULE_MAP = {
    "core_attn": lambda layer: layer.self_attention.core_attn,
    "experts": _moe_module("experts"),
    "moe": _maybe("moe"),
    "router": _moe_module("router"),
    "mlp": _maybe("mlp"),
    "mlp_norm": lambda layer: layer.mlp_norm,
    "attn_proj": lambda layer: layer.self_attention.linear_proj,
    "mla": lambda layer: layer.self_attention,
}


@dataclass(frozen=True)
class ImplConfig:
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    optimizer: str | None = "mc_full"
    recompute: list[str] = field(default_factory=list)
    offload: list[str] = field(default_factory=list)
    use_deepep: bool = False
    use_thd: bool = False
    hf_path: str = ""
    attention_backend_override: str | None = None
    router_aux_loss_coef: float | None = None
    router_bias_rate: float = 0.0
    deterministic: bool = True
    optimizer_config: OptimizerConfig | None = None


def build_model_config(source: str | Path | dict, **overrides) -> KimiK2Config:
    if isinstance(source, dict):
        cfg = KimiK2Config._from_hf_dict(source)
    else:
        cfg = KimiK2Config.from_hf(str(source))
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def _forward_step(model: nn.Module, batch: dict) -> dict:
    kwargs: dict[str, Any] = {
        "input_ids": batch["input_ids"],
        "labels": batch["labels"],
    }
    if "packed_seq_params" in batch:
        kwargs["packed_seq_params"] = batch["packed_seq_params"]
    for key in ("loss_mask", "temperature", "use_fused_kernels", "calculate_entropy"):
        if key in batch:
            kwargs[key] = batch[key]
    if kwargs["input_ids"].dim() == 1:
        kwargs["input_ids"] = kwargs["input_ids"].unsqueeze(0)
    return model(**kwargs)


def _make_aux_loss_hook():
    from megatron.lite.primitive.modules.moe import MoEAuxLossAutoScaler

    def hook(scale: torch.Tensor) -> None:
        MoEAuxLossAutoScaler.set_loss_scale(scale)

    return hook


def _build_mc_optimizer(chunks, model_cfg: KimiK2Config, impl_cfg: ImplConfig, ps: ParallelState):
    from megatron.lite.primitive.optimizers.megatron_wrap import (
        build_mc_full_stack,
        finalize_mc_full_grads,
    )

    opt_cfg = impl_cfg.optimizer_config
    if opt_cfg is None:
        opt_cfg = SimpleNamespace(
            optimizer="adam",
            lr=1e-4,
            weight_decay=0.01,
            clip_grad=1.0,
            offload_fraction=None,
            adam_beta1=None,
            adam_beta2=None,
            adam_eps=None,
        )

    engine_cfg = SimpleNamespace(
        model_name="kimi_k2",
        parallel=impl_cfg.parallel,
        optimizer=opt_cfg,
    )
    chunks[:], optimizer = build_mc_full_stack(
        chunks,
        model_cfg=model_cfg,
        engine_cfg=engine_cfg,
        ps=ps,
        is_expert=is_expert_param,
    )

    def finalize() -> None:
        finalize_mc_full_grads(chunks, optimizer)

    return optimizer, finalize


def build_model(model_cfg: KimiK2Config, *, impl_cfg: ImplConfig) -> ModelBundle:
    from megatron.lite.model.kimi_k2.lite.model import KimiK2Model

    p = impl_cfg.parallel
    if impl_cfg.use_deepep and (p.etp is not None and p.etp > 1):
        raise ValueError("use_deepep and etp>1 are mutually exclusive")
    if impl_cfg.router_aux_loss_coef is not None:
        model_cfg.aux_loss_alpha = impl_cfg.router_aux_loss_coef

    ps = init_parallel(p)
    recompute_spec = parse_recompute_spec(impl_cfg.recompute)
    vpp = None if p.vpp == 1 else p.vpp
    train_cfg = SimpleNamespace(
        tp=ps.tp_size,
        ep=ps.ep_size,
        etp=ps.etp_size,
        pp=ps.pp_size,
        cp=ps.cp_size,
        vpp=vpp,
        use_deepep=impl_cfg.use_deepep,
        fp8=False,
        recompute_modules=recompute_spec,
        deterministic=impl_cfg.deterministic,
    )
    model_kwargs: dict[str, Any] = dict(
        router_bias_rate=impl_cfg.router_bias_rate,
        use_thd=impl_cfg.use_thd,
        hf_path=impl_cfg.hf_path,
        attention_backend_override=impl_cfg.attention_backend_override,
    )

    if vpp is None:
        chunks = [
            KimiK2Model(model_cfg, train_cfg, ps, **model_kwargs)
            .to(torch.bfloat16)
            .cuda()
        ]
    else:
        chunks = [
            KimiK2Model(
                model_cfg,
                train_cfg,
                ps,
                vpp_chunk_id=i,
                **model_kwargs,
            )
            .to(torch.bfloat16)
            .cuda()
            for i in range(vpp)
        ]

    if recompute_spec:
        for chunk in chunks:
            apply_recompute(chunk.layers, recompute_spec, MODULE_MAP)

    if impl_cfg.offload:
        from megatron.lite.primitive.recompute import apply_offload

        for chunk in chunks:
            apply_offload(chunk.layers, impl_cfg.offload, MODULE_MAP)

    optimizer = None
    finalize_grads = None
    optimizer_backend = "none"
    if impl_cfg.optimizer == "mc_full":
        optimizer, finalize_grads = _build_mc_optimizer(chunks, model_cfg, impl_cfg, ps)
        from megatron.lite.runtime.megatron_utils import register_training_hooks

        register_training_hooks(chunks, optimizer)
        optimizer_backend = "mc_full"
    elif impl_cfg.optimizer is not None:
        raise ValueError(f"Unknown kimi_k2 lite optimizer: {impl_cfg.optimizer!r}.")

    return ModelBundle(
        chunks=chunks,
        parallel_state=ps,
        optimizer=optimizer,
        finalize_grads=finalize_grads,
        forward_step=_forward_step,
        extras={
            "model_cfg": model_cfg,
            "optimizer_backend": optimizer_backend,
            "pre_forward_hook": _make_aux_loss_hook(),
        },
    )


def load_hf_weights(
    chunk: nn.Module, hf_path: str, model_cfg: KimiK2Config, ps: ParallelState
) -> None:
    if not hf_path:
        return
    _load_hf_weights_impl(chunk, hf_path, model_cfg, ps)


def vocab_size(model_cfg) -> int | None:
    cfg = getattr(model_cfg, "text_config", model_cfg)
    return getattr(cfg, "vocab_size", None)


__all__ = [
    "ImplConfig",
    "build_model",
    "build_model_config",
    "is_expert_param",
    "load_hf_weights",
    "vocab_size",
]
