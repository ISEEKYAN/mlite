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
        expert_deferred_scale = None
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
        expert_deferred_scale = _defer_expert_grad_microbatch_scaling(
            wrapped_chunks, ddp_config
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
    if expert_deferred_scale is not None:
        # The per-microbatch scaling was neutralized on the unsharded expert grad
        # buffers (see _defer_expert_grad_microbatch_scaling); re-apply it exactly
        # once per step in grad_norm._scale_mfsdp_expert_grads via _expert_grad_scale.
        optimizer._mlite_mfsdp_expert_deferred_scale = expert_deferred_scale
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


def _defer_expert_grad_microbatch_scaling(model_chunks, ddp_config) -> float | None:
    """Work around a Megatron-FSDP multi-microbatch bug for unsharded expert grads.

    For expert params whose grad buffer is *unsharded* (expert-DP size == 1), the
    Megatron-FSDP grad-reduce pipeline pre-scales the *entire* accumulator by
    ``gradient_scaling_factor`` on every microbatch's reduce
    (``gradient_reduce_preprocessing`` -> ``grad_data.mul_(scaling_factor)``).
    Because an unsharded buffer accumulates across microbatches in place
    (``main_grad.add_``), each earlier microbatch is re-scaled once per subsequent
    microbatch, yielding ``0.5*g_mb0 + g_mb1`` instead of ``0.5*(g_mb0 + g_mb1)``
    for NUM_MB=2. Sharded (dense) buffers are unaffected because each microbatch's
    contribution is reduce-scattered into a separate persistent shard.

    Fix (contained in the M-FSDP primitive): null the per-microbatch scaling on
    these buffers so they accumulate an unscaled SUM (the size-1 collective is a
    no-op anyway), and re-apply the factor exactly once per step in
    ``grad_norm._scale_mfsdp_expert_grads`` via ``_expert_grad_scale``. Returns the
    (single, shared) deferred factor, or ``None`` if nothing needed neutralizing.
    """
    if getattr(ddp_config, "average_in_collective", False):
        # ReduceOp.AVG path does not use the in-place pre-scale; not affected.
        return None
    deferred: float | None = None
    for chunk in model_chunks:
        buffer = getattr(chunk, "param_and_grad_buffer", None)
        if buffer is None:
            continue
        for group in getattr(buffer, "parameter_groups", ()):
            if not getattr(group, "is_expert_param", False):
                continue
            gbuf = getattr(group, "main_grad_buffer", None)
            if gbuf is None or getattr(gbuf, "is_data_distributed", False):
                # Sharded expert buffers reduce-scatter correctly; leave them alone.
                continue
            stashed = getattr(gbuf, "_mlite_deferred_grad_scale", None)
            if stashed is not None:
                factor = stashed  # Idempotent: already neutralized on a prior build.
            else:
                factor = getattr(gbuf, "gradient_scaling_factor", None)
                if factor is None or factor == 1.0:
                    # No scaling applied per microbatch -> nothing to defer.
                    continue
                gbuf._mlite_deferred_grad_scale = float(factor)
                gbuf.gradient_scaling_factor = None
            if deferred is None:
                deferred = float(factor)
            elif abs(deferred - float(factor)) > 1e-12:
                raise ValueError(
                    "Megatron-FSDP expert grad buffers have inconsistent "
                    f"gradient_scaling_factor ({deferred} vs {factor}); "
                    "deferred expert grad scaling assumes a single shared factor."
                )
    return deferred


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
