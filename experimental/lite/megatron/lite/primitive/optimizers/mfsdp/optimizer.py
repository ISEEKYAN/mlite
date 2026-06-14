# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Megatron-FSDP wrap backend for Megatron Lite.

This primitive intentionally stays thin: Megatron Lite owns model construction,
HF loading, and the runtime loop; Megatron-Core owns the FSDP wrapper,
ParamAndGradBuffer, and DistributedOptimizer integration.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch.nn as nn  # pyright: ignore[reportMissingImports]

from megatron.lite.primitive.optimizers.megatron_wrap import (
    _build_transformer_config,
    _ensure_mc_mpu_parallel_state,
    _mark_mc_parallel_attrs,
    build_mc_optimizer_config,
)
from megatron.lite.primitive.optimizers.mfsdp.config import (
    build_mfsdp_ddp_config,
    split_mfsdp_overrides,
    validate_mfsdp_config,
    validate_optimizer_name,
)
from megatron.lite.primitive.optimizers.mfsdp.checkpoint_keys import (
    attach_mfsdp_checkpoint_metadata,
)
from megatron.lite.primitive.optimizers.mfsdp.grad_norm import (
    CanonicalGradNormMegatronFSDPOptimizer,
)
from megatron.lite.primitive.optimizers.mfsdp.metadata import (
    ensure_mfsdp_tp_partition_attrs,
    normalize_mfsdp_expert_tensor_parallel_attrs,
)
from megatron.lite.primitive.optimizers.mfsdp.patches import (
    install_mfsdp_tp_duplicate_sync_patch,
)
from megatron.lite.primitive.optimizers.mfsdp.process_groups import (
    build_mfsdp_pg_collection,
    install_mfsdp_mesh_patch,
)
from megatron.lite.primitive.protocols import ExpertClassifierFn, default_expert_classifier


def build_mfsdp_stack(
    model_chunks: list[nn.Module],
    *,
    model_cfg,
    engine_cfg,
    ps,
    is_expert: ExpertClassifierFn | None = None,
    proto=None,
    fsdp_unit_modules: tuple[type[nn.Module] | str, ...] | None = None,
    skip_fsdp_wrap: bool = False,
):
    """Wrap model chunks with Megatron-FSDP and build the matching optimizer."""
    from megatron.core.distributed import (  # pyright: ignore[reportMissingImports]
        DistributedDataParallelConfig,
        FullyShardedDataParallel,
    )
    from megatron.core.distributed.finalize_model_grads import (  # pyright: ignore[reportMissingImports]
        finalize_model_grads,
    )
    from megatron.core.optimizer import (
        get_megatron_optimizer,  # pyright: ignore[reportMissingImports]
    )
    from megatron.core.transformer.enums import ModelType  # pyright: ignore[reportMissingImports]

    validate_mfsdp_config(engine_cfg)
    install_mfsdp_mesh_patch()
    install_mfsdp_tp_duplicate_sync_patch()

    p = engine_cfg.parallel
    opt = engine_cfg.optimizer
    opt_overrides, ddp_overrides = split_mfsdp_overrides(
        opt,
        DistributedDataParallelConfig,
    )

    mc_transformer_cfg = _build_transformer_config(model_cfg, engine_cfg)
    mc_transformer_cfg.finalize_model_grads_func = finalize_model_grads
    if is_expert is not None:
        is_expert_param = is_expert
    elif proto is not None and hasattr(proto, "EXPERT_CLASSIFIER"):
        is_expert_param = proto.EXPERT_CLASSIFIER
    else:
        is_expert_param = default_expert_classifier

    _ensure_mc_mpu_parallel_state(engine_cfg)
    use_mpu_groups = bool(getattr(engine_cfg, "deterministic", False))
    pg_collection = None if use_mpu_groups else build_mfsdp_pg_collection(ps, engine_cfg)

    if skip_fsdp_wrap:
        wrapped_chunks = list(model_chunks)
    else:
        ddp_config = build_mfsdp_ddp_config(DistributedDataParallelConfig, ddp_overrides)
        wrapped_chunks = []
        unit_modules = list(fsdp_unit_modules) if fsdp_unit_modules is not None else None
        for chunk_idx, chunk in enumerate(model_chunks):
            chunk.model_type = ModelType.encoder_or_decoder
            _mark_mc_parallel_attrs(chunk, is_expert_param, tp_size=p.tp)
            normalize_mfsdp_expert_tensor_parallel_attrs(
                chunk,
                is_expert_param,
                etp_size=int(p.etp or 1),
            )
            ensure_mfsdp_tp_partition_attrs(chunk)
            wrapped_chunks.append(
                FullyShardedDataParallel(
                    mc_transformer_cfg,
                    ddp_config,
                    chunk,
                    fsdp_unit_modules=unit_modules,
                    disable_bucketing=(chunk_idx > 0),
                    pg_collection=pg_collection,
                )
            )

    opt_config = build_mc_optimizer_config(
        opt,
        override_optimizer_config=opt_overrides or None,
    )
    validate_optimizer_name(opt_config.optimizer)
    opt_config.use_distributed_optimizer = True
    if skip_fsdp_wrap or use_mpu_groups:
        optimizer = get_megatron_optimizer(config=opt_config, model_chunks=wrapped_chunks)
        optimizer._mc_pg_collection = None  # pyright: ignore[reportAttributeAccessIssue]
    else:
        optimizer = get_megatron_optimizer(
            config=opt_config,
            model_chunks=wrapped_chunks,
            use_gloo_process_groups=False,
            pg_collection=pg_collection,
        )
        optimizer._mc_pg_collection = pg_collection  # pyright: ignore[reportAttributeAccessIssue]
    optimizer_leaves = getattr(optimizer, "chained_optimizers", None)
    for optimizer_owner in (optimizer, *tuple(optimizer_leaves or ())):
        if not hasattr(optimizer_owner, "model_chunks"):
            optimizer_owner.model_chunks = wrapped_chunks  # pyright: ignore[reportAttributeAccessIssue]
    attach_mfsdp_checkpoint_metadata(optimizer, ps=ps, is_expert=is_expert_param)
    param_is_expert, param_names = _build_wrapped_param_metadata(wrapped_chunks, is_expert_param)
    optimizer = CanonicalGradNormMegatronFSDPOptimizer(
        optimizer,
        ps,
        model_chunks=wrapped_chunks,
        param_is_expert=param_is_expert,
        param_names=param_names,
    )
    attach_mfsdp_checkpoint_metadata(optimizer, ps=ps, is_expert=is_expert_param)
    return wrapped_chunks, optimizer


def build_mfsdp_training_optimizer(
    model_chunks: list[nn.Module],
    *,
    model_cfg,
    impl_cfg,
    ps,
    model_name: str,
    is_expert: ExpertClassifierFn | None = None,
    fsdp_unit_modules: tuple[type[nn.Module] | str, ...] | None = None,
    deterministic: bool | None = None,
):
    """Build the Megatron-FSDP stack from a Megatron Lite ImplConfig."""
    opt = impl_cfg.optimizer_config
    if opt is None:
        opt = SimpleNamespace(
            optimizer="adam",
            lr=1e-4,
            min_lr=0.0,
            weight_decay=0.01,
            clip_grad=1.0,
            offload_fraction=None,
            adam_beta1=None,
            adam_beta2=None,
            adam_eps=None,
        )
    if deterministic is None:
        from megatron.lite.primitive.deterministic import deterministic_requested

        deterministic = deterministic_requested()

    engine_cfg = SimpleNamespace(
        model_name=model_name,
        parallel=impl_cfg.parallel,
        optimizer=opt,
        deterministic=bool(deterministic),
    )
    model_chunks[:], optimizer = build_mfsdp_stack(
        model_chunks,
        model_cfg=model_cfg,
        engine_cfg=engine_cfg,
        ps=ps,
        is_expert=is_expert,
        fsdp_unit_modules=fsdp_unit_modules,
    )

    def finalize_grads() -> None:
        finalize_mfsdp_grads(model_chunks, optimizer)

    return optimizer, finalize_grads


def finalize_mfsdp_grads(model_chunks: list[nn.Module], optimizer) -> None:
    """Run Megatron-Core gradient finalization for Megatron-FSDP chunks."""
    from megatron.core.distributed.finalize_model_grads import (  # pyright: ignore[reportMissingImports]
        finalize_model_grads,
    )

    finalize_model_grads(model_chunks, pg_collection=optimizer._mc_pg_collection)


def _build_wrapped_param_metadata(
    model_chunks: list[nn.Module],
    is_expert_param: ExpertClassifierFn,
) -> tuple[dict[int, bool], dict[int, str]]:
    param_is_expert: dict[int, bool] = {}
    param_names: dict[int, str] = {}
    for chunk in model_chunks:
        for name, param in chunk.named_parameters():
            param_is_expert[id(param)] = bool(is_expert_param(name)) or not getattr(
                param,
                "allreduce",
                True,
            )
            param_names[id(param)] = name
    return param_is_expert, param_names


__all__ = [
    "build_mfsdp_stack",
    "build_mfsdp_training_optimizer",
    "finalize_mfsdp_grads",
]
