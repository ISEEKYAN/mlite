# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Canonical grad-norm handling for the Megatron-FSDP primitive.

Megatron-Core's FSDP optimizer currently flattens FSDP DTensor gradients to
local tensors before it computes the norm. That makes the final answer depend
on the optimizer's single grad-stats group. Megatron Lite needs the same semantics
as the fsdp2 primitive: compute dense and expert shards in their own parallel
domains, then combine through pipeline parallelism.
"""

from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.distributed as dist

from megatron.lite.primitive.optimizers.mfsdp.dtensor_grad import sharded_grad_sq_sum
from megatron.lite.primitive.optimizers.mfsdp.patches import (
    PARAM_NAME_ATTR,
    should_skip_tp_duplicate_sync,
)
from megatron.lite.primitive.parallel.state import ParallelState


def _phase_debug(phase: str) -> None:
    if os.environ.get("MLITE_BENCH_PHASE_DEBUG", "0") != "1":
        return
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(f"[MFSDP_STEP_PHASE] rank={rank} phase={phase} t={time.time():.6f}", flush=True)


@dataclass(frozen=True, slots=True)
class GradNormBreakdown:
    dense_sq: float
    dense_tp_sharded_sq: float
    dense_tp_replicated_sq: float
    expert_sq: float
    global_dense_sq: float
    global_dense_tp_sharded_sq: float
    global_dense_tp_replicated_sq: float
    global_expert_sq: float
    total_sq: float
    grad_norm: float
    dense_params: int
    dense_tp_replicated_params: int
    expert_params: int
    dense_names: tuple[str, ...] = ()
    dense_tp_replicated_names: tuple[str, ...] = ()
    expert_names: tuple[str, ...] = ()


class CanonicalGradNormMegatronFSDPOptimizer:
    """Wrap a Megatron-Core optimizer with Megatron Lite's M-FSDP grad norm.

    The wrapper delegates everything except ``step``. ``step`` follows
    Megatron-Core's optimizer sequence, but replaces the norm/clip section with
    ``compute_mfsdp_grad_norm``.
    """

    name = "megatron_fsdp"

    def __init__(
        self,
        optimizer: Any,
        ps: ParallelState,
        *,
        model_chunks: list[Any] | None = None,
        param_is_expert: dict[int, bool] | None = None,
        param_names: dict[int, str] | None = None,
    ):
        self._inner_optimizer = optimizer
        self.ps = ps
        self._model_chunks = list(model_chunks or ())
        self._param_is_expert = dict(param_is_expert or {})
        self._param_names = dict(param_names or {})
        self._debug_update_step = 0
        if _debug_param_names():
            _print_param_name_debug(self._param_is_expert, self._param_names)
        if _debug_updates():
            _install_update_debug_hooks(self._inner_optimizer)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner_optimizer, name)

    @property
    def optimizer(self) -> Any:
        return getattr(self._inner_optimizer, "optimizer", None)

    @property
    def param_groups(self) -> Any:
        return getattr(self._inner_optimizer, "param_groups", [])

    def zero_grad(self, *args, **kwargs):
        return self._inner_optimizer.zero_grad(*args, **kwargs)

    def state_dict(self):
        return self._inner_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        return self._inner_optimizer.load_state_dict(state_dict)

    def reload_model_params(self) -> None:
        """M-FSDP is built after model load, so main params are already current."""
        return None

    @torch.no_grad()
    def step(self):
        if not _use_canonical_grad_norm():
            _phase_debug("inner_step:start")
            result = self._inner_optimizer.step()
            _phase_debug("inner_step:end")
            return result

        _phase_debug("prepare_grads:start")
        found_inf_flag = self._inner_optimizer.prepare_grads()
        _phase_debug("prepare_grads:end")
        if found_inf_flag:
            return False, None, None

        _phase_debug("sync_main_grads:start")
        _sync_mfsdp_model_chunk_main_grads(
            self._inner_optimizer,
            model_chunks=self._model_chunks,
        )
        _phase_debug("sync_main_grads:end")

        mcore_norm_before_scale = None
        if _debug_grad_norm():
            _phase_debug("mcore_get_grad_norm:start")
            mcore_norm_before_scale = self._inner_optimizer.get_grad_norm()
            _phase_debug("mcore_get_grad_norm:end")
        expert_scale = _expert_grad_scale(self.ps, self._inner_optimizer)
        _phase_debug("expert_grad_scale:start")
        _scale_mfsdp_expert_grads(
            self._inner_optimizer,
            param_is_expert=self._param_is_expert,
            scale=expert_scale,
        )
        _phase_debug("expert_grad_scale:end")
        _phase_debug("canonical_grad_norm:start")
        breakdown = compute_mfsdp_grad_norm(
            self._inner_optimizer,
            self.ps,
            param_is_expert=self._param_is_expert,
            param_names=self._param_names,
        )
        _phase_debug("canonical_grad_norm:end")
        grad_norm = breakdown.grad_norm

        if _debug_grad_norm():
            assert mcore_norm_before_scale is not None
            _print_grad_norm_debug(
                mcore_norm_before_scale,
                breakdown,
                expert_scale=expert_scale,
            )

        if not math.isfinite(grad_norm):
            return False, grad_norm, None

        _phase_debug("clip_grads:start")
        _clip_mfsdp_grads_by_total_norm(
            self._inner_optimizer,
            grad_norm,
            model_chunks=self._model_chunks,
            ps=self.ps,
            param_is_expert=self._param_is_expert,
            param_names=self._param_names,
        )
        _phase_debug("clip_grads:end")
        if _debug_grad_norm():
            _phase_debug("postclip_grad_norm:start")
            postclip_mcore_norm = self._inner_optimizer.get_grad_norm()
            postclip_breakdown = compute_mfsdp_grad_norm(
                self._inner_optimizer,
                self.ps,
                param_is_expert=self._param_is_expert,
                param_names=self._param_names,
            )
            _phase_debug("postclip_grad_norm:end")
            _print_postclip_grad_norm_debug(postclip_mcore_norm, postclip_breakdown)

        config = getattr(self._inner_optimizer, "config", None)
        log_num_zeros = bool(getattr(config, "log_num_zeros_in_grad", False))
        _phase_debug("count_zeros:start")
        num_zeros_in_grad = self._inner_optimizer.count_zeros() if log_num_zeros else None
        _phase_debug("count_zeros:end")
        if _debug_updates():
            _set_update_debug_step(self._inner_optimizer, self._debug_update_step)
            _print_update_debug(self._inner_optimizer, self._debug_update_step, "pre_step")
        _phase_debug("step_with_ready_grads:start")
        if _skip_post_step_param_sync():
            if os.environ.get("MLITE_MFSDP_DEBUG_PARAM_SYNC", "0") == "1":
                rank = dist.get_rank() if dist.is_initialized() else 0
                print(
                    f"[MFSDP_PARAM_SYNC] phase=skip_post_step_sync_active rank={rank}",
                    flush=True,
                )
            old_active = os.environ.get("MLITE_MFSDP_SKIP_START_PARAM_SYNC_ACTIVE")
            os.environ["MLITE_MFSDP_SKIP_START_PARAM_SYNC_ACTIVE"] = "1"
            originals = _disable_chunk_param_sync(self._model_chunks)
            try:
                update_successful = self._inner_optimizer.step_with_ready_grads()
            finally:
                _restore_chunk_param_sync(originals)
                if old_active is None:
                    os.environ.pop("MLITE_MFSDP_SKIP_START_PARAM_SYNC_ACTIVE", None)
                else:
                    os.environ["MLITE_MFSDP_SKIP_START_PARAM_SYNC_ACTIVE"] = old_active
        else:
            update_successful = self._inner_optimizer.step_with_ready_grads()
        _phase_debug("step_with_ready_grads:end")
        if _debug_updates():
            _print_update_debug(self._inner_optimizer, self._debug_update_step, "post_step")
            self._debug_update_step += 1
        if _debug_finite_scan():
            _print_finite_scan(
                self._inner_optimizer,
                "post_step",
                param_names=self._param_names,
            )

        return update_successful, grad_norm, num_zeros_in_grad


def _skip_post_step_param_sync() -> bool:
    return os.environ.get(
        "MLITE_MFSDP_SKIP_POST_STEP_PARAM_SYNC", "0"
    ).lower() in ("1", "true", "yes", "on")


def _disable_chunk_param_sync(model_chunks: Iterable[Any]) -> list[tuple[Any, Any]]:
    originals: list[tuple[Any, Any]] = []

    def _noop_start_param_sync(*_args: Any, **_kwargs: Any) -> None:
        if os.environ.get("MLITE_MFSDP_DEBUG_PARAM_SYNC", "0") == "1":
            rank = dist.get_rank() if dist.is_initialized() else 0
            print(
                f"[MFSDP_PARAM_SYNC] phase=post_step_noop rank={rank}",
                flush=True,
            )

    for chunk in model_chunks:
        if not hasattr(chunk, "start_param_sync"):
            continue
        originals.append((chunk, chunk.start_param_sync))
        chunk.start_param_sync = _noop_start_param_sync
    return originals


def _restore_chunk_param_sync(originals: Iterable[tuple[Any, Any]]) -> None:
    for chunk, original in originals:
        chunk.start_param_sync = original


@torch.no_grad()
def compute_mfsdp_grad_norm(
    optimizer: Any,
    ps: ParallelState,
    *,
    param_is_expert: dict[int, bool] | None = None,
    param_names: dict[int, str] | None = None,
) -> GradNormBreakdown:
    device = _first_grad_device(optimizer)
    dense_dtensor_sharded_params: list[torch.nn.Parameter] = []
    dense_dtensor_sharded_sq = _new_accumulator(device)
    dense_dtensor_tp_sharded_sq = _new_accumulator(device)
    dense_tp_replicated_sq = _new_accumulator(device)
    dense_plain_sharded_sq = _new_accumulator(device)
    dense_replicated_sq = _new_accumulator(device)
    expert_dtensor_sharded_sq = _new_accumulator(device)
    expert_plain_sharded_sq = _new_accumulator(device)
    dense_params = 0
    dense_tp_replicated_param_count = 0
    expert_params = 0
    dense_names: list[str] = []
    dense_tp_replicated_names: list[str] = []
    expert_names: list[str] = []

    for leaf_optimizer in _iter_leaf_optimizers(optimizer):
        for param_group in _iter_param_groups(leaf_optimizer):
            group_is_expert = bool(param_group.get("is_expert_parallel", False))
            for param in param_group.get("params", ()):
                is_expert = _param_is_expert(param, group_is_expert, param_is_expert)
                tp_group = ps.etp_group if is_expert else ps.tp_group
                grad = _grad_for_norm(param, leaf_optimizer)
                if grad is None:
                    continue
                include_in_norm = _include_param_in_norm(param, tp_group)
                has_dtensor = _has_dtensor_grad_or_param(param, grad)
                is_tp_replicated = _is_tp_replicated_grad_param(param, is_expert)
                if not include_in_norm and not has_dtensor and not is_tp_replicated:
                    continue
                if is_expert:
                    if has_dtensor:
                        if include_in_norm:
                            expert_dtensor_sharded_sq = (
                                expert_dtensor_sharded_sq
                                + _grad_sq_sum(grad).to(expert_dtensor_sharded_sq.device)
                            )
                    elif include_in_norm:
                        expert_plain_sharded_sq = expert_plain_sharded_sq + _grad_sq_sum(
                            grad
                        ).to(expert_plain_sharded_sq.device)
                    if include_in_norm:
                        expert_params += 1
                    if include_in_norm and _debug_param_names():
                        expert_names.append(_param_name(param, param_names))
                else:
                    if is_tp_replicated:
                        if _include_shared_param_in_norm(param):
                            dense_tp_replicated_sq = dense_tp_replicated_sq + _grad_sq_sum(
                                grad
                            ).to(dense_tp_replicated_sq.device)
                        dense_tp_replicated_param_count += 1
                        if _debug_param_names():
                            dense_tp_replicated_names.append(_param_name(param, param_names))
                    elif has_dtensor:
                        dense_dtensor_sharded_params.append(param)
                        if include_in_norm:
                            if _is_tp_sharded_grad_param(param, False):
                                dense_dtensor_tp_sharded_sq = (
                                    dense_dtensor_tp_sharded_sq
                                    + _grad_sq_sum(grad).to(
                                        dense_dtensor_tp_sharded_sq.device
                                    )
                                )
                            else:
                                dense_dtensor_sharded_sq = (
                                    dense_dtensor_sharded_sq
                                    + _grad_sq_sum(grad).to(dense_dtensor_sharded_sq.device)
                                )
                    elif include_in_norm and _is_sharded_param_grad(param, grad):
                        dense_plain_sharded_sq = dense_plain_sharded_sq + _grad_sq_sum(
                            grad
                        ).to(dense_plain_sharded_sq.device)
                    elif include_in_norm:
                        grad_sq = _grad_sq_sum(grad)
                        dense_replicated_sq = dense_replicated_sq + grad_sq.to(
                            dense_replicated_sq.device
                        )
                    if include_in_norm:
                        dense_params += 1
                    if include_in_norm and _debug_param_names():
                        dense_names.append(_param_name(param, param_names))

    _phase_debug("canonical_grad_norm:dense_dtensor_non_tp_sq:start")
    _sum_in_group(
        dense_dtensor_sharded_sq,
        ps.dp_cp_group or ps.dp_group,
        tag="dense_dtensor_non_tp_dp",
    )
    _phase_debug("canonical_grad_norm:dense_dtensor_non_tp_sq:end")
    _phase_debug("canonical_grad_norm:dense_dtensor_tp_sharded_sq:start")
    _sum_in_group(
        dense_dtensor_tp_sharded_sq,
        ps.dp_cp_group or ps.dp_group,
        tag="dense_dtensor_tp_sharded_dp",
    )
    _sum_in_group(
        dense_dtensor_tp_sharded_sq,
        ps.tp_group,
        tag="dense_dtensor_tp_group",
    )
    _phase_debug("canonical_grad_norm:dense_dtensor_tp_sharded_sq:end")
    _phase_debug("canonical_grad_norm:dense_tp_replicated_sq:start")
    _sum_in_group(
        dense_tp_replicated_sq,
        ps.dp_cp_group or ps.dp_group,
        tag="dense_tp_replicated_dp",
    )
    _phase_debug("canonical_grad_norm:dense_tp_replicated_sq:end")
    _phase_debug("canonical_grad_norm:dense_plain_reduce:start")
    _sum_in_group(
        dense_plain_sharded_sq,
        ps.dp_cp_group or ps.dp_group,
        tag="dense_plain_dp",
    )
    _sum_in_group(dense_plain_sharded_sq, ps.tp_group, tag="dense_plain_tp")
    _sum_in_group(dense_replicated_sq, ps.tp_group, tag="dense_replicated_tp")
    _phase_debug("canonical_grad_norm:dense_plain_reduce:end")
    dense_sq = (
        dense_dtensor_sharded_sq
        + dense_dtensor_tp_sharded_sq.to(dense_dtensor_sharded_sq.device)
        + dense_tp_replicated_sq.to(dense_dtensor_sharded_sq.device)
        + dense_plain_sharded_sq.to(dense_dtensor_sharded_sq.device)
        + dense_replicated_sq.to(dense_dtensor_sharded_sq.device)
    )
    if _debug_grad_norm_candidates():
        _print_dense_dtensor_candidate_debug(
            dense_dtensor_sharded_params,
            ps,
            default_device=device,
            param_names=param_names,
        )

    _phase_debug("canonical_grad_norm:expert_dtensor_sq:start")
    _sum_in_group(
        expert_dtensor_sharded_sq,
        ps.ep_dp_group,
        tag="expert_dtensor_ep_dp",
    )
    _phase_debug("canonical_grad_norm:expert_dtensor_sq:end")
    _phase_debug("canonical_grad_norm:expert_reduce:start")
    _sum_in_group(expert_plain_sharded_sq, ps.ep_dp_group, tag="expert_plain_ep_dp")
    _sum_in_group(expert_plain_sharded_sq, ps.etp_group, tag="expert_plain_etp")
    expert_sq = expert_dtensor_sharded_sq + expert_plain_sharded_sq.to(
        expert_dtensor_sharded_sq.device
    )
    _sum_in_group(expert_sq, ps.ep_group, tag="expert_ep")
    _phase_debug("canonical_grad_norm:expert_reduce:end")

    if _debug_grad_norm():
        global_dense_sq = dense_sq.clone()
        global_dense_tp_sharded_sq = dense_dtensor_tp_sharded_sq.clone()
        global_dense_tp_replicated_sq = dense_tp_replicated_sq.clone()
        global_expert_sq = expert_sq.clone()
        _phase_debug("canonical_grad_norm:debug_global_dense_pp_reduce:start")
        _sum_in_group(global_dense_sq, ps.pp_group, tag="debug_global_dense_pp")
        _phase_debug("canonical_grad_norm:debug_global_dense_pp_reduce:end")
        _phase_debug("canonical_grad_norm:debug_global_dense_tp_pp_reduce:start")
        _sum_in_group(
            global_dense_tp_sharded_sq,
            ps.pp_group,
            tag="debug_global_dense_tp_pp",
        )
        _phase_debug("canonical_grad_norm:debug_global_dense_tp_pp_reduce:end")
        _phase_debug("canonical_grad_norm:debug_global_dense_tp_repl_pp_reduce:start")
        _sum_in_group(
            global_dense_tp_replicated_sq,
            ps.pp_group,
            tag="debug_global_dense_tp_repl_pp",
        )
        _phase_debug("canonical_grad_norm:debug_global_dense_tp_repl_pp_reduce:end")
        _phase_debug("canonical_grad_norm:debug_global_expert_pp_reduce:start")
        _sum_in_group(global_expert_sq, ps.pp_group, tag="debug_global_expert_pp")
        _phase_debug("canonical_grad_norm:debug_global_expert_pp_reduce:end")
        total_sq = global_dense_sq + global_expert_sq.to(global_dense_sq.device)
    else:
        global_dense_sq = dense_sq
        global_dense_tp_sharded_sq = dense_dtensor_tp_sharded_sq
        global_dense_tp_replicated_sq = dense_tp_replicated_sq
        global_expert_sq = expert_sq
        total_sq = dense_sq + expert_sq.to(dense_sq.device)
        _phase_debug("canonical_grad_norm:total_pp_reduce:start")
        _sum_in_group(total_sq, ps.pp_group, tag="total_pp")
        _phase_debug("canonical_grad_norm:total_pp_reduce:end")
    grad_norm = total_sq.sqrt()
    _phase_debug("canonical_grad_norm:grad_norm_item:start")
    grad_norm_value = float(grad_norm.float().item())
    _phase_debug("canonical_grad_norm:grad_norm_item:end")
    collect_component_stats = _debug_grad_norm()
    if collect_component_stats:
        _phase_debug("canonical_grad_norm:component_items:start")
        dense_sq_value = float(dense_sq.float().item())
        dense_tp_sharded_sq_value = float(dense_dtensor_tp_sharded_sq.float().item())
        dense_tp_replicated_sq_value = float(dense_tp_replicated_sq.float().item())
        expert_sq_value = float(expert_sq.float().item())
        global_dense_sq_value = float(global_dense_sq.float().item())
        global_dense_tp_sharded_sq_value = float(
            global_dense_tp_sharded_sq.float().item()
        )
        global_dense_tp_replicated_sq_value = float(
            global_dense_tp_replicated_sq.float().item()
        )
        global_expert_sq_value = float(global_expert_sq.float().item())
        total_sq_value = float(total_sq.float().item())
        _phase_debug("canonical_grad_norm:component_items:end")
    else:
        dense_sq_value = float("nan")
        dense_tp_sharded_sq_value = float("nan")
        dense_tp_replicated_sq_value = float("nan")
        expert_sq_value = float("nan")
        global_dense_sq_value = float("nan")
        global_dense_tp_sharded_sq_value = float("nan")
        global_dense_tp_replicated_sq_value = float("nan")
        global_expert_sq_value = float("nan")
        total_sq_value = grad_norm_value * grad_norm_value
    return GradNormBreakdown(
        dense_sq=dense_sq_value,
        dense_tp_sharded_sq=dense_tp_sharded_sq_value,
        dense_tp_replicated_sq=dense_tp_replicated_sq_value,
        expert_sq=expert_sq_value,
        global_dense_sq=global_dense_sq_value,
        global_dense_tp_sharded_sq=global_dense_tp_sharded_sq_value,
        global_dense_tp_replicated_sq=global_dense_tp_replicated_sq_value,
        global_expert_sq=global_expert_sq_value,
        total_sq=total_sq_value,
        grad_norm=grad_norm_value,
        dense_params=dense_params,
        dense_tp_replicated_params=dense_tp_replicated_param_count,
        expert_params=expert_params,
        dense_names=tuple(dense_names[:20]),
        dense_tp_replicated_names=tuple(dense_tp_replicated_names[:20]),
        expert_names=tuple(expert_names[:20]),
    )


def _clip_mfsdp_grads_by_total_norm(
    optimizer: Any,
    grad_norm: float,
    *,
    model_chunks: list[Any] | None = None,
    ps: ParallelState | None = None,
    param_is_expert: dict[int, bool] | None = None,
    param_names: dict[int, str] | None = None,
) -> None:
    if not math.isfinite(float(grad_norm)):
        return
    clip_coeff = _mfsdp_clip_coeff(optimizer, grad_norm)
    if clip_coeff is None or clip_coeff >= 1.0:
        return
    if _debug_clip_rank_stats():
        _print_clip_rank_stats(
            grad_norm=float(grad_norm),
            clip_coeff=clip_coeff,
            scaled_params=0,
            local_sq_before=0.0,
            local_sq_after=0.0,
            stage="pre",
        )
    debug_clip = _debug_clip()
    seen_param_ids: set[int] = set()
    scaled_params = 0
    debug_local_sq_before = 0.0
    debug_local_sq_after = 0.0
    debug_dtensor_local = 0
    debug_dtensor_to_local = 0
    for leaf_optimizer in _iter_leaf_optimizers(optimizer):
        if bool(getattr(leaf_optimizer, "is_stub_optimizer", False)):
            continue
        config = getattr(leaf_optimizer, "config", None)
        clip_grad = float(getattr(config, "clip_grad", 0.0))
        if clip_grad <= 0.0:
            continue
        use_decoupled_grad = bool(
            getattr(config, "use_precision_aware_optimizer_no_fp8_or_ds_fp8", False)
        )
        for param_group in _iter_param_groups(leaf_optimizer):
            for param in param_group.get("params", ()):
                param_id = id(param)
                if param_id in seen_param_ids:
                    continue
                seen_param_ids.add(param_id)
                grad = (
                    getattr(param, "decoupled_grad", None)
                    if use_decoupled_grad
                    else getattr(param, "grad", None)
                )
                if grad is not None:
                    if debug_clip:
                        local_grad = _to_local_grad(grad)
                        if local_grad is not None:
                            debug_local_sq_before += float(
                                _grad_sq_sum(local_grad).float().item()
                            )
                        if getattr(grad, "_local_tensor", None) is not None:
                            debug_dtensor_local += 1
                        elif callable(getattr(grad, "to_local", None)):
                            debug_dtensor_to_local += 1
                    _scale_grad_in_place(grad, clip_coeff)
                    if debug_clip:
                        local_grad = _to_local_grad(grad)
                        if local_grad is not None:
                            debug_local_sq_after += float(
                                _grad_sq_sum(local_grad).float().item()
                            )
                    scaled_params += 1
    if debug_clip:
        _print_clip_debug(
            float(grad_norm),
            clip_coeff,
            scaled_params,
            mode="param_grads",
            local_sq_before=debug_local_sq_before,
            local_sq_after=debug_local_sq_after,
            dtensor_local=debug_dtensor_local,
            dtensor_to_local=debug_dtensor_to_local,
        )
    if _debug_clip_rank_stats():
        _print_clip_rank_stats(
            grad_norm=float(grad_norm),
            clip_coeff=clip_coeff,
            scaled_params=scaled_params,
            local_sq_before=debug_local_sq_before,
            local_sq_after=debug_local_sq_after,
            stage="post_scale",
        )


def _mfsdp_clip_coeff(optimizer: Any, grad_norm: float) -> float | None:
    clip_grad = _mfsdp_clip_grad(optimizer)
    if clip_grad is None:
        return None
    return clip_grad / (float(grad_norm) + 1.0e-6)


def _mfsdp_clip_grad(optimizer: Any) -> float | None:
    for leaf_optimizer in _iter_leaf_optimizers(optimizer):
        if bool(getattr(leaf_optimizer, "is_stub_optimizer", False)):
            continue
        config = getattr(leaf_optimizer, "config", None)
        clip_grad = float(getattr(config, "clip_grad", 0.0))
        if clip_grad > 0.0:
            return clip_grad
    return None


def _mfsdp_model_chunks_for_clip(
    optimizer: Any,
    model_chunks: list[Any] | None,
) -> list[Any]:
    chunks: list[Any] = []
    seen_ids: set[int] = set()

    def add_chunk(chunk: Any) -> None:
        chunk_id = id(chunk)
        if chunk_id in seen_ids:
            return
        seen_ids.add(chunk_id)
        chunks.append(chunk)

    for chunk in model_chunks or ():
        add_chunk(chunk)
    for chunk in getattr(optimizer, "model_chunks", ()) or ():
        add_chunk(chunk)
    for leaf_optimizer in _iter_leaf_optimizers(optimizer):
        for chunk in getattr(leaf_optimizer, "model_chunks", ()) or ():
            add_chunk(chunk)
    return chunks


def _sync_mfsdp_model_chunk_main_grads(
    optimizer: Any,
    *,
    model_chunks: list[Any] | None = None,
) -> bool:
    synced_any = False
    for chunk in _mfsdp_model_chunks_for_clip(optimizer, model_chunks):
        buffer = getattr(chunk, "param_and_grad_buffer", None)
        update_main_grads = getattr(buffer, "update_main_grads", None)
        if callable(update_main_grads):
            update_main_grads()
            synced_any = True
            continue
        scale_gradients = getattr(chunk, "scale_gradients", None)
        if callable(scale_gradients):
            scale_gradients(1.0)
            synced_any = True
    return synced_any


def _scale_mfsdp_expert_grads(
    optimizer: Any,
    *,
    param_is_expert: dict[int, bool],
    scale: float,
) -> None:
    if scale == 1.0:
        return
    seen_param_ids: set[int] = set()
    for leaf_optimizer in _iter_leaf_optimizers(optimizer):
        for param_group in _iter_param_groups(leaf_optimizer):
            group_is_expert = bool(param_group.get("is_expert_parallel", False))
            for param in param_group.get("params", ()):
                param_id = id(param)
                if param_id in seen_param_ids:
                    continue
                seen_param_ids.add(param_id)
                if not _param_is_expert(param, group_is_expert, param_is_expert):
                    continue
                grad = getattr(param, "grad", None)
                if grad is not None:
                    _scale_grad_in_place(grad, scale)


def _expert_grad_scale(ps: ParallelState, optimizer: Any = None) -> float:
    override = os.environ.get("MLITE_MFSDP_EXPERT_GRAD_SCALE")
    if override is not None:
        return float(override)
    if optimizer is not None:
        # When the unsharded expert grad buffers had their per-microbatch
        # gradient_scaling_factor neutralized at build time (see
        # optimizer._defer_expert_grad_microbatch_scaling), re-apply that factor
        # exactly once here so the expert grad becomes scale*(g_mb0 + g_mb1).
        deferred = getattr(optimizer, "_mlite_mfsdp_expert_deferred_scale", None)
        if deferred is not None:
            return float(deferred)
    return 1.0


def _iter_leaf_optimizers(optimizer: Any) -> list[Any]:
    chained = getattr(optimizer, "chained_optimizers", None)
    if chained is None:
        return [optimizer]
    return list(chained)


def _iter_param_groups(optimizer: Any) -> Iterable[dict[str, Any]]:
    param_groups = getattr(optimizer, "param_groups", None)
    if param_groups is None:
        inner_optimizer = getattr(optimizer, "optimizer", None)
        param_groups = getattr(inner_optimizer, "param_groups", None)
    return param_groups or ()


def _param_is_expert(
    param: torch.Tensor,
    group_is_expert: bool,
    param_is_expert: dict[int, bool] | None,
) -> bool:
    if param_is_expert is not None and id(param) in param_is_expert:
        return bool(param_is_expert[id(param)])
    if not getattr(param, "allreduce", True):
        return True
    return group_is_expert


def _param_name(param: torch.Tensor, param_names: dict[int, str] | None) -> str:
    if param_names is None:
        return "<unknown>"
    return param_names.get(id(param), "<unknown>")


def _is_sharded_param_grad(param: torch.Tensor, grad: torch.Tensor) -> bool:
    if getattr(param, "__fsdp_param__", False):
        return True
    if getattr(getattr(param, "grad", None), "_local_tensor", None) is not None:
        return True
    return getattr(grad, "_spec", None) is not None


def _has_dtensor_grad_or_param(param: torch.Tensor, grad: torch.Tensor) -> bool:
    raw_grad = getattr(param, "grad", None)
    return _is_dtensor_like(grad) or _is_dtensor_like(raw_grad) or _is_dtensor_like(param)


def _is_dtensor_like(tensor: Any) -> bool:
    return (
        callable(getattr(tensor, "to_local", None))
        and hasattr(tensor, "device_mesh")
        and hasattr(tensor, "placements")
    )


def _scale_grad_in_place(grad: Any, scale: float) -> None:
    local_tensor = getattr(grad, "_local_tensor", None)
    if isinstance(local_tensor, torch.Tensor):
        local_tensor.mul_(scale)
        return
    to_local = getattr(grad, "to_local", None)
    if callable(to_local):
        local_grad = to_local()
        local_grad.mul_(scale)
        return
    grad.mul_(scale)


def _grad_for_norm(param: torch.Tensor, optimizer: Any) -> torch.Tensor | None:
    if getattr(param, "__fsdp_param__", False):
        return _to_local_grad(getattr(param, "grad", None))

    config = getattr(optimizer, "config", None)
    use_decoupled_grad = bool(
        getattr(config, "use_precision_aware_optimizer_no_fp8_or_ds_fp8", False)
    )
    if use_decoupled_grad:
        grad = getattr(param, "decoupled_grad", None)
    else:
        grad = getattr(param, "grad", None)
    return _to_local_grad(grad)


def _include_param_in_norm(param: torch.Tensor, tp_group: Any) -> bool:
    from megatron.core.tensor_parallel import (  # pyright: ignore[reportMissingImports]
        param_is_not_tensor_parallel_duplicate,
    )
    from megatron.core.transformer.module import (  # pyright: ignore[reportMissingImports]
        param_is_not_shared,
    )

    return bool(param_is_not_shared(param)) and bool(
        param_is_not_tensor_parallel_duplicate(param, tp_group=tp_group)
    )


def _include_shared_param_in_norm(param: torch.Tensor) -> bool:
    from megatron.core.transformer.module import (  # pyright: ignore[reportMissingImports]
        param_is_not_shared,
    )

    return bool(param_is_not_shared(param))


def _is_tp_replicated_grad_param(param: torch.Tensor, is_expert: bool) -> bool:
    if is_expert:
        return False
    return bool(getattr(param, "sequence_parallel", False))


def _is_tp_sharded_grad_param(param: torch.Tensor, is_expert: bool) -> bool:
    if is_expert:
        return False
    if _is_tp_replicated_grad_param(param, is_expert):
        return False
    if not bool(getattr(param, "tensor_model_parallel", False)):
        return False
    partition_dim = getattr(param, "partition_dim", None)
    if partition_dim == 1:
        return True
    return partition_dim == 0 and _is_vocab_tp_sharded_param(param)


def _is_vocab_tp_sharded_param(param: torch.Tensor) -> bool:
    name = _param_policy_name(param)
    return (
        name.endswith("embed.embedding.weight")
        or name.endswith("mtp_embed.embedding.weight")
        or name.endswith("head.col.linear.weight")
    )


def _param_policy_name(param: torch.Tensor) -> str:
    name = str(getattr(param, PARAM_NAME_ATTR, ""))
    if name:
        return name
    orig_param = getattr(param, "orig_param", None)
    if orig_param is not None:
        return str(getattr(orig_param, PARAM_NAME_ATTR, ""))
    return ""


def _dense_dtensor_candidate_sq(
    params: list[torch.nn.Parameter],
    ps: ParallelState,
    *,
    default_device: torch.device,
    include_param,
) -> tuple[torch.Tensor, torch.Tensor]:
    local_sq = sharded_grad_sq_sum(
        params,
        accum_dtype=torch.float32,
        default_device=default_device,
        include_param=include_param,
    )
    tp_sq = local_sq.clone()
    _sum_in_group(tp_sq, ps.tp_group)
    return local_sq, tp_sq


def _print_dense_dtensor_candidate_debug(
    params: list[torch.nn.Parameter],
    ps: ParallelState,
    *,
    default_device: torch.device,
    param_names: dict[int, str] | None,
) -> None:
    include_base = lambda param: (
        not _is_tp_replicated_grad_param(param, False)
        and _include_param_in_norm(param, ps.tp_group)
    )
    row_sq, row_tp_sq = _dense_dtensor_candidate_sq(
        params,
        ps,
        default_device=default_device,
        include_param=lambda param: (
            include_base(param) and getattr(param, "partition_dim", None) == 1
        ),
    )
    column_sq, column_tp_sq = _dense_dtensor_candidate_sq(
        params,
        ps,
        default_device=default_device,
        include_param=lambda param: (
            include_base(param) and getattr(param, "partition_dim", None) == 0
        ),
    )
    local_tp_column_sq, local_tp_column_tp_sq = _dense_dtensor_candidate_sq(
        params,
        ps,
        default_device=default_device,
        include_param=lambda param: (
            include_base(param)
            and getattr(param, "partition_dim", None) == 0
            and should_skip_tp_duplicate_sync(param)
        ),
    )
    mcore_tp_column_sq, mcore_tp_column_tp_sq = _dense_dtensor_candidate_sq(
        params,
        ps,
        default_device=default_device,
        include_param=lambda param: (
            include_base(param)
            and getattr(param, "partition_dim", None) == 0
            and not should_skip_tp_duplicate_sync(param)
        ),
    )
    no_partition_sq, no_partition_tp_sq = _dense_dtensor_candidate_sq(
        params,
        ps,
        default_device=default_device,
        include_param=lambda param: (
            include_base(param) and getattr(param, "partition_dim", None) not in (0, 1)
        ),
    )
    shared_excluded_sq, shared_excluded_tp_sq = _dense_dtensor_candidate_sq(
        params,
        ps,
        default_device=default_device,
        include_param=lambda param: (
            not _is_tp_replicated_grad_param(param, False)
            and not _include_shared_param_in_norm(param)
        ),
    )
    tp_duplicate_excluded_sq, tp_duplicate_excluded_tp_sq = _dense_dtensor_candidate_sq(
        params,
        ps,
        default_device=default_device,
        include_param=lambda param: (
            not _is_tp_replicated_grad_param(param, False)
            and _include_shared_param_in_norm(param)
            and not _include_param_in_norm(param, ps.tp_group)
        ),
    )
    vocab_column_sq, vocab_column_tp_sq = _dense_dtensor_candidate_sq(
        params,
        ps,
        default_device=default_device,
        include_param=lambda param: include_base(param)
        and _is_column_category(param, "vocab", param_names),
    )
    attn_qkv_column_sq, attn_qkv_column_tp_sq = _dense_dtensor_candidate_sq(
        params,
        ps,
        default_device=default_device,
        include_param=lambda param: include_base(param)
        and _is_column_category(param, "attn_qkv", param_names),
    )
    attn_aux_column_sq, attn_aux_column_tp_sq = _dense_dtensor_candidate_sq(
        params,
        ps,
        default_device=default_device,
        include_param=lambda param: include_base(param)
        and _is_column_category(param, "attn_aux", param_names),
    )
    fc1_column_sq, fc1_column_tp_sq = _dense_dtensor_candidate_sq(
        params,
        ps,
        default_device=default_device,
        include_param=lambda param: include_base(param)
        and _is_column_category(param, "fc1", param_names),
    )
    other_column_sq, other_column_tp_sq = _dense_dtensor_candidate_sq(
        params,
        ps,
        default_device=default_device,
        include_param=lambda param: include_base(param)
        and _is_column_category(param, "other", param_names),
    )
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    shared_names = _candidate_names(
        params,
        include_param=lambda param: (
            not _is_tp_replicated_grad_param(param, False)
            and not _include_shared_param_in_norm(param)
        ),
        param_names=param_names,
    )
    tp_duplicate_names = _candidate_names(
        params,
        include_param=lambda param: (
            not _is_tp_replicated_grad_param(param, False)
            and _include_shared_param_in_norm(param)
            and not _include_param_in_norm(param, ps.tp_group)
        ),
        param_names=param_names,
    )
    print(
        "[MFSDP_GN_CANDIDATES] "
        f"row_sq={float(row_sq.float().item()):.9e} "
        f"row_tp_sq={float(row_tp_sq.float().item()):.9e} "
        f"column_sq={float(column_sq.float().item()):.9e} "
        f"column_tp_sq={float(column_tp_sq.float().item()):.9e} "
        f"local_tp_column_sq={float(local_tp_column_sq.float().item()):.9e} "
        f"local_tp_column_tp_sq={float(local_tp_column_tp_sq.float().item()):.9e} "
        f"mcore_tp_column_sq={float(mcore_tp_column_sq.float().item()):.9e} "
        f"mcore_tp_column_tp_sq={float(mcore_tp_column_tp_sq.float().item()):.9e} "
        f"no_partition_sq={float(no_partition_sq.float().item()):.9e} "
        f"no_partition_tp_sq={float(no_partition_tp_sq.float().item()):.9e} "
        f"shared_excluded_sq={float(shared_excluded_sq.float().item()):.9e} "
        f"shared_excluded_tp_sq={float(shared_excluded_tp_sq.float().item()):.9e} "
        f"tp_duplicate_excluded_sq={float(tp_duplicate_excluded_sq.float().item()):.9e} "
        f"tp_duplicate_excluded_tp_sq={float(tp_duplicate_excluded_tp_sq.float().item()):.9e} "
        f"vocab_column_sq={float(vocab_column_sq.float().item()):.9e} "
        f"vocab_column_tp_sq={float(vocab_column_tp_sq.float().item()):.9e} "
        f"attn_qkv_column_sq={float(attn_qkv_column_sq.float().item()):.9e} "
        f"attn_qkv_column_tp_sq={float(attn_qkv_column_tp_sq.float().item()):.9e} "
        f"attn_aux_column_sq={float(attn_aux_column_sq.float().item()):.9e} "
        f"attn_aux_column_tp_sq={float(attn_aux_column_tp_sq.float().item()):.9e} "
        f"fc1_column_sq={float(fc1_column_sq.float().item()):.9e} "
        f"fc1_column_tp_sq={float(fc1_column_tp_sq.float().item()):.9e} "
        f"other_column_sq={float(other_column_sq.float().item()):.9e} "
        f"other_column_tp_sq={float(other_column_tp_sq.float().item()):.9e} "
        f"shared_names={shared_names} "
        f"tp_duplicate_names={tp_duplicate_names}",
        flush=True,
    )


def _candidate_names(
    params: list[torch.nn.Parameter],
    *,
    include_param,
    param_names: dict[int, str] | None,
    limit: int = 8,
) -> list[str]:
    names: list[str] = []
    for param in params:
        if not include_param(param):
            continue
        names.append(_param_debug_name(param, param_names))
        if len(names) >= limit:
            break
    return names


def _param_debug_name(
    param: torch.nn.Parameter,
    param_names: dict[int, str] | None,
) -> str:
    if param_names is not None and id(param) in param_names:
        return param_names[id(param)]
    return str(getattr(param, PARAM_NAME_ATTR, ""))


def _is_column_category(
    param: torch.nn.Parameter,
    category: str,
    param_names: dict[int, str] | None,
) -> bool:
    if getattr(param, "partition_dim", None) != 0:
        return False
    name = _param_debug_name(param, param_names)
    is_vocab = (
        name.endswith("embed.embedding.weight")
        or name.endswith("mtp_embed.embedding.weight")
        or name.endswith("head.col.linear.weight")
    )
    is_attn_qkv = (
        ".attn.qkv." in name
        or ".attn.linear_qkv." in name
        or ".self_attention.linear_qkv." in name
        or ".self_attention.in_proj." in name
        or ".self_attention.gdn.in_proj." in name
    )
    is_attn_aux = ".eh_proj." in name
    is_fc1 = (
        ".shared_experts." in name
        or ".linear_fc1." in name
        or ".experts.fc1." in name
        or ".experts.linear_fc1." in name
    )
    if category == "vocab":
        return is_vocab
    if category == "attn_qkv":
        return is_attn_qkv
    if category == "attn_aux":
        return is_attn_aux
    if category == "fc1":
        return is_fc1
    if category == "other":
        return not (is_vocab or is_attn_qkv or is_attn_aux or is_fc1)
    return False


def _to_local_grad(grad: Any) -> torch.Tensor | None:
    if grad is None:
        return None
    local_tensor = getattr(grad, "_local_tensor", None)
    if local_tensor is not None:
        return local_tensor
    try:
        from megatron.core.utils import to_local_if_dtensor  # pyright: ignore[reportMissingImports]

        return to_local_if_dtensor(grad)
    except Exception:
        return grad


def _grad_sq_sum(grad: torch.Tensor) -> torch.Tensor:
    local = grad.detach()
    if local.dtype != torch.float32:
        local = local.float()
    return local.pow(2).sum(dtype=torch.float32)


def _first_grad_device(optimizer: Any) -> torch.device:
    for leaf_optimizer in _iter_leaf_optimizers(optimizer):
        for param_group in _iter_param_groups(leaf_optimizer):
            for param in param_group.get("params", ()):
                grad = _to_local_grad(getattr(param, "grad", None))
                if grad is not None:
                    return grad.device
    return torch.device("cuda", torch.cuda.current_device())


def _new_accumulator(device: torch.device) -> torch.Tensor:
    return torch.zeros((), dtype=torch.float32, device=device)


def _sum_in_group(value: torch.Tensor, group: Any, *, tag: str = "sum") -> None:
    if group is None or not dist.is_initialized():
        return
    if dist.get_world_size(group) <= 1:
        return
    dist.all_reduce(value, op=dist.ReduceOp.SUM, group=group)
    _sync_scalar_reduce_if_needed(value, tag)


def _scalar_all_reduce(
    value: torch.Tensor,
    group: dist.ProcessGroup,
    op: dist.ReduceOp,
    tag: str,
) -> None:
    dist.all_reduce(value, op=op, group=group)
    _sync_scalar_reduce_if_needed(value, tag)


def _sync_scalar_reduce_if_needed(value: torch.Tensor, tag: str) -> None:
    if not _sync_scalar_reductions():
        return
    if value.device.type != "cuda":
        return
    _phase_debug(f"canonical_grad_norm:{tag}:cuda_sync:start")
    torch.cuda.synchronize(value.device)
    _phase_debug(f"canonical_grad_norm:{tag}:cuda_sync:end")


def _use_canonical_grad_norm() -> bool:
    raw = os.environ.get("MLITE_MFSDP_USE_CANONICAL_GRAD_NORM", "1")
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _sync_scalar_reductions() -> bool:
    raw = os.environ.get("MLITE_MFSDP_SYNC_SCALAR_REDUCTIONS", "0")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _debug_grad_norm() -> bool:
    raw = os.environ.get("MLITE_MFSDP_DEBUG_GRAD_NORM", "0")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _debug_grad_norm_candidates() -> bool:
    raw = os.environ.get("MLITE_MFSDP_DEBUG_GRAD_NORM_CANDIDATES", "0")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _debug_param_names() -> bool:
    raw = os.environ.get("MLITE_MFSDP_DEBUG_PARAM_NAMES", "0")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _debug_clip() -> bool:
    raw = os.environ.get("MLITE_MFSDP_DEBUG_CLIP", "0")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _debug_clip_rank_stats() -> bool:
    raw = os.environ.get("MLITE_MFSDP_DEBUG_CLIP_RANKS", "0")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _debug_updates() -> bool:
    raw = os.environ.get("MLITE_MFSDP_DEBUG_UPDATES", "0")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _debug_finite_scan() -> bool:
    raw = os.environ.get("MLITE_MFSDP_DEBUG_FINITE_SCAN", "0")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _debug_finite_scan_max_reports() -> int:
    raw = os.environ.get("MLITE_MFSDP_DEBUG_FINITE_MAX_REPORTS", "3")
    try:
        return max(int(raw), 1)
    except ValueError:
        return 3


def _debug_finite_scan_ranks() -> set[int] | None:
    raw = os.environ.get("MLITE_MFSDP_DEBUG_FINITE_RANKS", "all").strip()
    if raw.lower() in {"", "all", "*"}:
        return None
    ranks: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            ranks.add(int(item))
        except ValueError:
            continue
    return ranks or {0}


def _debug_update_max_steps() -> int:
    raw = os.environ.get("MLITE_MFSDP_DEBUG_UPDATE_MAX_STEPS", "1")
    try:
        return max(int(raw), 0)
    except ValueError:
        return 1


def _debug_update_ranks() -> set[int]:
    raw = os.environ.get("MLITE_MFSDP_DEBUG_UPDATE_RANKS", "0,1")
    ranks: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            ranks.add(int(item))
        except ValueError:
            continue
    return ranks or {0}


def _debug_update_name_pattern() -> re.Pattern[str]:
    raw = os.environ.get(
        "MLITE_MFSDP_DEBUG_UPDATE_NAMES",
        "|".join(
            (
                r"embed\.embedding\.weight$",
                r"head\.col\.linear\.weight$",
                r"layers\.0\..*(qkv|linear_qkv|proj|linear_proj).*weight$",
                r"layers\.0\..*(experts|linear_fc|linear_fc1|linear_fc2).*weight$",
            )
        ),
    )
    return re.compile(raw)


def _install_update_debug_hooks(optimizer: Any) -> None:
    for buffer in _iter_mfsdp_param_and_grad_buffers(optimizer):
        if getattr(buffer, "_mlite_mfsdp_update_debug_wrapped", False):
            continue
        original = buffer.copy_main_weights_to_model_weights

        def wrapped_copy_main_weights_to_model_weights(
            original=original,
            buffer=buffer,
        ):
            step = int(getattr(buffer, "_mlite_mfsdp_update_debug_step", 0))
            _print_update_debug_for_buffer(buffer, step, "pre_copy")
            result = original()
            _print_update_debug_for_buffer(buffer, step, "post_copy")
            return result

        buffer._mlite_mfsdp_original_copy_main_weights_to_model_weights = original
        buffer.copy_main_weights_to_model_weights = wrapped_copy_main_weights_to_model_weights
        buffer._mlite_mfsdp_update_debug_wrapped = True


def _set_update_debug_step(optimizer: Any, step: int) -> None:
    for buffer in _iter_mfsdp_param_and_grad_buffers(optimizer):
        buffer._mlite_mfsdp_update_debug_step = int(step)


def _print_update_debug(optimizer: Any, step: int, stage: str) -> None:
    if step >= _debug_update_max_steps():
        return
    for buffer in _iter_mfsdp_param_and_grad_buffers(optimizer):
        _print_update_debug_for_buffer(buffer, step, stage)


def _print_update_debug_for_buffer(buffer: Any, step: int, stage: str) -> None:
    if step >= _debug_update_max_steps():
        return
    if dist.is_initialized():
        rank = dist.get_rank()
        if rank not in _debug_update_ranks():
            return
    else:
        rank = 0
    pattern = _debug_update_name_pattern()
    printed = 0
    for name, param in getattr(buffer, "optimizer_named_parameters", ()):
        if not pattern.search(name):
            continue
        orig_param = getattr(param, "orig_param", None)
        skip_param = orig_param if orig_param is not None else param
        model_tensor, main_tensor = _mfsdp_buffer_tensors_for_debug(buffer, orig_param)
        model_sig = _tensor_debug_signature(model_tensor)
        main_sig = _tensor_debug_signature(main_tensor)
        param_sig = _tensor_debug_signature(param)
        grad_sig = _tensor_debug_signature(getattr(param, "grad", None))
        diff_norm = _tensor_debug_diff_norm(model_tensor, main_tensor)
        param_main_diff_norm = _tensor_debug_diff_norm(param, main_tensor)
        buffer_meta = _mfsdp_buffer_metadata_for_debug(buffer, orig_param)
        print(
            "[MFSDP_UPDATE] "
            f"rank={rank} "
            f"step={step} "
            f"stage={stage} "
            f"name={name} "
            f"tp={int(bool(getattr(param, 'tensor_model_parallel', False)))} "
            f"partition_dim={getattr(param, 'partition_dim', None)} "
            f"tp_mode={getattr(param, '_tensor_parallel_mode', None)} "
            f"skip_sync={int(should_skip_tp_duplicate_sync(skip_param))} "
            f"{buffer_meta} "
            f"model={model_sig} "
            f"main={main_sig} "
            f"param={param_sig} "
            f"grad={grad_sig} "
            f"model_main_diff_norm={diff_norm} "
            f"param_main_diff_norm={param_main_diff_norm}",
            flush=True,
        )
        printed += 1
        if printed >= 12:
            break


def _print_finite_scan(
    optimizer: Any,
    stage: str,
    *,
    param_names: dict[int, str] | None,
) -> None:
    rank = dist.get_rank() if dist.is_initialized() else 0
    ranks = _debug_finite_scan_ranks()
    should_print_details = ranks is None or rank in ranks
    max_reports = _debug_finite_scan_max_reports()
    total_tensors = 0
    bad_tensors = 0
    bad_elements = 0
    reports: list[str] = []
    category_counts: dict[str, list[int]] = {}
    category_reports: dict[str, list[str]] = {}

    def check_tensor(label: str, tensor: Any) -> None:
        nonlocal total_tensors, bad_tensors, bad_elements
        local = _debug_local_tensor(tensor)
        if local is None or not isinstance(local, torch.Tensor):
            return
        category = _finite_scan_category(label)
        counts = category_counts.setdefault(category, [0, 0, 0])
        counts[0] += 1
        total_tensors += 1
        detached = local.detach()
        if detached.numel() == 0:
            return
        finite = torch.isfinite(detached)
        if bool(finite.all().item()):
            return
        bad_tensors += 1
        nonfinite = int((~finite).sum().item())
        bad_elements += nonfinite
        counts[1] += 1
        counts[2] += nonfinite
        if len(reports) >= max_reports:
            global_reports_full = True
        else:
            global_reports_full = False
        stats = detached.float()
        finite_stats = stats[finite]
        if finite_stats.numel() > 0:
            finite_min = float(finite_stats.min().item())
            finite_max = float(finite_stats.max().item())
        else:
            finite_min = float("nan")
            finite_max = float("nan")
        first_bad = int((~finite).reshape(-1).nonzero()[0].item())
        bad_value = float(stats.reshape(-1)[first_bad].item())
        report = (
            f"{label}:shape={tuple(detached.shape)} "
            f"dtype={detached.dtype} nonfinite={nonfinite} "
            f"first_bad={first_bad} first_bad_value={bad_value:.9e} "
            f"finite_min={finite_min:.9e} finite_max={finite_max:.9e}"
        )
        if not global_reports_full:
            reports.append(report)
        category_list = category_reports.setdefault(category, [])
        if len(category_list) < max_reports:
            category_list.append(report)

    for leaf_optimizer in _iter_leaf_optimizers(optimizer):
        leaf_name = type(leaf_optimizer).__name__
        for param_group_index, param_group in enumerate(_iter_param_groups(leaf_optimizer)):
            for param_index, param in enumerate(param_group.get("params", ())):
                name = _param_debug_name(param, param_names)
                if not name:
                    name = f"group{param_group_index}.param{param_index}"
                check_tensor(f"{leaf_name}.param.{name}", param)
                check_tensor(f"{leaf_name}.grad.{name}", getattr(param, "grad", None))
                check_tensor(
                    f"{leaf_name}.decoupled_grad.{name}",
                    getattr(param, "decoupled_grad", None),
                )
        inner_optimizer = getattr(leaf_optimizer, "optimizer", None)
        state = getattr(inner_optimizer, "state", None)
        if isinstance(state, dict):
            for state_param, state_dict in state.items():
                state_name = _param_debug_name(state_param, param_names)
                if not state_name:
                    state_name = f"state_param_{id(state_param)}"
                if not isinstance(state_dict, dict):
                    continue
                for key, value in state_dict.items():
                    if torch.is_tensor(value) or _is_dtensor_like(value):
                        check_tensor(f"{leaf_name}.optimizer_state.{state_name}.{key}", value)

    for buffer_index, buffer in enumerate(_iter_mfsdp_param_and_grad_buffers(optimizer)):
        prefix = f"buffer{buffer_index}"
        for name, param in getattr(buffer, "optimizer_named_parameters", ()):
            orig_param = getattr(param, "orig_param", None)
            model_tensor, main_tensor = _mfsdp_buffer_tensors_for_debug(buffer, orig_param)
            check_tensor(f"{prefix}.model.{name}", model_tensor)
            check_tensor(f"{prefix}.main.{name}", main_tensor)

    if dist.is_initialized() and torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
        counts = torch.tensor(
            [total_tensors, bad_tensors, bad_elements],
            dtype=torch.long,
            device=device,
        )
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        world_total_tensors = int(counts[0].item())
        world_bad_tensors = int(counts[1].item())
        world_bad_elements = int(counts[2].item())
    else:
        world_total_tensors = total_tensors
        world_bad_tensors = bad_tensors
        world_bad_elements = bad_elements

    if should_print_details:
        print(
            "[MFSDP_FINITE_SCAN] "
            f"rank={rank} "
            f"stage={stage} "
            f"local_tensors={total_tensors} "
            f"local_bad_tensors={bad_tensors} "
            f"local_bad_elements={bad_elements} "
            f"world_tensors={world_total_tensors} "
            f"world_bad_tensors={world_bad_tensors} "
            f"world_bad_elements={world_bad_elements} "
            f"categories={_finite_scan_category_summary(category_counts)} "
            f"category_reports={category_reports} "
            f"reports={reports}",
            flush=True,
        )


def _finite_scan_category(label: str) -> str:
    if ".optimizer_state." in label:
        return "optimizer_state"
    if ".decoupled_grad." in label:
        return "decoupled_grad"
    if ".grad." in label:
        return "grad"
    if ".param." in label:
        return "optimizer_param"
    if ".model." in label:
        return "buffer_model"
    if ".main." in label:
        return "buffer_main"
    return "other"


def _finite_scan_category_summary(category_counts: dict[str, list[int]]) -> dict[str, str]:
    return {
        category: f"total={counts[0]},bad={counts[1]},bad_elements={counts[2]}"
        for category, counts in sorted(category_counts.items())
    }


def _mfsdp_buffer_tensors_for_debug(buffer: Any, orig_param: Any) -> tuple[Any, Any]:
    if orig_param is None:
        return None, None
    try:
        group = buffer.parameter_groups[buffer.param_to_param_group[orig_param]]
        mbuf = group.main_weight_buffer
        wbuf = group.model_weight_buffer
        if mbuf is None:
            return None, None
        item_id = mbuf.param_idx[orig_param]
        use_shard = bool(
            (wbuf is not None and getattr(wbuf, "is_data_distributed", False))
            or getattr(mbuf, "is_data_distributed", False)
        )
        main_tensor = mbuf.get_item(item_id, only_shard=use_shard)
        if wbuf is None:
            model_tensor = None
        else:
            model_tensor = wbuf.get_item(item_id, only_shard=use_shard)
        return model_tensor, main_tensor
    except Exception as exc:
        return f"error:{type(exc).__name__}:{exc}", None


def _mfsdp_buffer_metadata_for_debug(buffer: Any, orig_param: Any) -> str:
    if orig_param is None:
        return "buffer=none"
    try:
        group = buffer.parameter_groups[buffer.param_to_param_group[orig_param]]
        mbuf = group.main_weight_buffer
        wbuf = group.model_weight_buffer
        if mbuf is None:
            return "buffer=main_none"
        item_id = mbuf.param_idx[orig_param]
        use_shard = bool(
            (wbuf is not None and getattr(wbuf, "is_data_distributed", False))
            or getattr(mbuf, "is_data_distributed", False)
        )
        return (
            f"item_id={item_id} "
            f"use_shard={int(use_shard)} "
            f"wbuf_dist={int(bool(wbuf is not None and getattr(wbuf, 'is_data_distributed', False)))} "
            f"mbuf_dist={int(bool(getattr(mbuf, 'is_data_distributed', False)))} "
            f"wbuf_dp={_buffer_dp_debug(wbuf)} "
            f"mbuf_dp={_buffer_dp_debug(mbuf)} "
            f"wbuf_item={_buffer_item_debug(wbuf, item_id)} "
            f"mbuf_item={_buffer_item_debug(mbuf, item_id)}"
        )
    except Exception as exc:
        return f"buffer_error={type(exc).__name__}:{exc}"


def _buffer_dp_debug(buf: Any) -> str:
    if buf is None:
        return "none"
    return f"{getattr(buf, 'dp_rank', None)}/{getattr(buf, 'dp_world_size', None)}"


def _buffer_item_debug(buf: Any, item_id: int) -> str:
    if buf is None:
        return "none"
    try:
        local = buf._get_item_local_index(item_id)
        shard = buf._get_item_local_shard_index(item_id)
        global_item = buf.locate_item_in_global_item(item_id)
        return f"local={local},shard={shard},global={global_item}"
    except Exception as exc:
        return f"error:{type(exc).__name__}:{exc}"


def _iter_mfsdp_param_and_grad_buffers(optimizer: Any) -> Iterable[Any]:
    for leaf_optimizer in _iter_leaf_optimizers(optimizer):
        for model_chunk in getattr(leaf_optimizer, "model_chunks", ()):
            buffer = getattr(model_chunk, "param_and_grad_buffer", None)
            if buffer is not None:
                yield buffer


def _tensor_debug_signature(tensor: Any) -> str:
    if isinstance(tensor, str):
        return tensor
    local = _debug_local_tensor(tensor)
    if local is None:
        return "none"
    detached = local.detach()
    if detached.numel() == 0:
        return f"shape={tuple(detached.shape)} numel=0"
    stats = detached.float()
    flat = stats.reshape(-1)
    sample = flat.narrow(0, 0, min(flat.numel(), 4096))
    return (
        f"shape={tuple(detached.shape)} "
        f"numel={detached.numel()} "
        f"sample_sum={float(sample.sum().item()):.9e} "
        f"sample_norm={float(sample.norm().item()):.9e} "
        f"first={float(flat[0].item()):.9e}"
    )


def _tensor_debug_diff_norm(left: Any, right: Any) -> str:
    left_local = _debug_local_tensor(left)
    right_local = _debug_local_tensor(right)
    if left_local is None or right_local is None:
        return "nan"
    if tuple(left_local.shape) != tuple(right_local.shape):
        return "shape_mismatch"
    left_flat = left_local.detach().float().reshape(-1)
    right_flat = right_local.detach().float().reshape(-1)
    sample_numel = min(left_flat.numel(), 4096)
    diff = left_flat.narrow(0, 0, sample_numel) - right_flat.narrow(0, 0, sample_numel)
    return f"{float(diff.norm().item()):.9e}"


def _debug_local_tensor(tensor: Any) -> torch.Tensor | None:
    if tensor is None:
        return None
    data = getattr(tensor, "data", tensor)
    local_tensor = getattr(data, "_local_tensor", None)
    if isinstance(local_tensor, torch.Tensor):
        return local_tensor
    to_local = getattr(data, "to_local", None)
    if callable(to_local):
        local = to_local()
        if isinstance(local, torch.Tensor):
            return local
    if isinstance(data, torch.Tensor):
        return data
    return None


def _print_clip_debug(
    grad_norm: float,
    clip_coeff: float | None,
    scaled_params: int,
    *,
    mode: str,
    local_sq_before: float = 0.0,
    local_sq_after: float = 0.0,
    dtensor_local: int = 0,
    dtensor_to_local: int = 0,
) -> None:
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    expected_after = (
        local_sq_before * float(clip_coeff) * float(clip_coeff)
        if clip_coeff is not None and clip_coeff < 1.0
        else local_sq_before
    )
    local_ratio = local_sq_after / expected_after if expected_after else float("nan")
    print(
        "[MFSDP_CLIP] "
        f"mode={mode} "
        f"grad_norm={grad_norm:.9f} "
        f"clip_coeff={float(clip_coeff) if clip_coeff is not None else float('nan'):.9f} "
        f"scaled_params={scaled_params} "
        f"local_sq_before={local_sq_before:.9e} "
        f"local_sq_after={local_sq_after:.9e} "
        f"expected_after={expected_after:.9e} "
        f"local_ratio={local_ratio:.9f} "
        f"dtensor_local={dtensor_local} "
        f"dtensor_to_local={dtensor_to_local}",
        flush=True,
    )


def _print_clip_rank_stats(
    *,
    grad_norm: float,
    clip_coeff: float,
    scaled_params: int,
    local_sq_before: float,
    local_sq_after: float,
    stage: str,
) -> None:
    if not dist.is_initialized():
        return
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else None
    if device is None:
        return
    values = torch.tensor(
        [
            float(grad_norm),
            float(clip_coeff),
            float(scaled_params),
            float(local_sq_before),
            float(local_sq_after),
        ],
        dtype=torch.float64,
        device=device,
    )
    mins = values.clone()
    maxs = values.clone()
    dist.all_reduce(mins, op=dist.ReduceOp.MIN)
    dist.all_reduce(maxs, op=dist.ReduceOp.MAX)
    if dist.get_rank() != 0:
        return
    print(
        "[MFSDP_CLIP_RANKS] "
        f"stage={stage} "
        f"world={dist.get_world_size()} "
        f"grad_norm_min={float(mins[0].item()):.9f} "
        f"grad_norm_max={float(maxs[0].item()):.9f} "
        f"clip_coeff_min={float(mins[1].item()):.9f} "
        f"clip_coeff_max={float(maxs[1].item()):.9f} "
        f"scaled_params_min={int(mins[2].item())} "
        f"scaled_params_max={int(maxs[2].item())} "
        f"local_sq_before_min={float(mins[3].item()):.9e} "
        f"local_sq_before_max={float(maxs[3].item()):.9e} "
        f"local_sq_after_min={float(mins[4].item()):.9e} "
        f"local_sq_after_max={float(maxs[4].item()):.9e}",
        flush=True,
    )


def _print_param_name_debug(
    param_is_expert: dict[int, bool],
    param_names: dict[int, str],
) -> None:
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    dense_names = [
        name for param_id, name in param_names.items() if not param_is_expert.get(param_id, False)
    ]
    expert_names = [
        name for param_id, name in param_names.items() if param_is_expert.get(param_id, False)
    ]
    print(
        "[MFSDP_PARAM_NAMES] "
        f"dense_count={len(dense_names)} "
        f"expert_count={len(expert_names)} "
        f"dense_sample={dense_names[:20]} "
        f"expert_sample={expert_names[:20]}",
        flush=True,
    )


def _print_grad_norm_debug(
    mcore_norm: float,
    breakdown: GradNormBreakdown,
    *,
    expert_scale: float,
) -> None:
    if _debug_clip_rank_stats():
        _print_norm_rank_stats("MFSDP_GN_RANKS", breakdown.grad_norm)
        _print_breakdown_rank_stats("MFSDP_GN_COMPONENT_RANKS", breakdown)
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    ratio = float(mcore_norm) / breakdown.grad_norm if breakdown.grad_norm else float("nan")
    print(
        "[MFSDP_GN] "
        f"mcore={float(mcore_norm):.9f} "
        f"canonical={breakdown.grad_norm:.9f} "
        f"ratio={ratio:.9f} "
        f"dense_sq={breakdown.dense_sq:.9e} "
        f"dense_tp_sharded_sq={breakdown.dense_tp_sharded_sq:.9e} "
        f"dense_tp_replicated_sq={breakdown.dense_tp_replicated_sq:.9e} "
        f"expert_sq={breakdown.expert_sq:.9e} "
        f"global_dense_sq={breakdown.global_dense_sq:.9e} "
        f"global_dense_tp_sharded_sq={breakdown.global_dense_tp_sharded_sq:.9e} "
        f"global_dense_tp_replicated_sq={breakdown.global_dense_tp_replicated_sq:.9e} "
        f"global_expert_sq={breakdown.global_expert_sq:.9e} "
        f"dense_params={breakdown.dense_params} "
        f"dense_tp_replicated_params={breakdown.dense_tp_replicated_params} "
        f"expert_params={breakdown.expert_params} "
        f"expert_scale={expert_scale:.9f} "
        f"dense_names={list(breakdown.dense_names)} "
        f"dense_tp_replicated_names={list(breakdown.dense_tp_replicated_names)} "
        f"expert_names={list(breakdown.expert_names)}",
        flush=True,
    )


def _print_postclip_grad_norm_debug(
    mcore_norm: float,
    breakdown: GradNormBreakdown,
) -> None:
    if _debug_clip_rank_stats():
        _print_norm_rank_stats("MFSDP_GN_POSTCLIP_RANKS", breakdown.grad_norm)
        _print_breakdown_rank_stats("MFSDP_GN_POSTCLIP_COMPONENT_RANKS", breakdown)
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    print(
        "[MFSDP_GN_POSTCLIP] "
        f"mcore={float(mcore_norm):.9f} "
        f"canonical={breakdown.grad_norm:.9f} "
        f"dense_sq={breakdown.dense_sq:.9e} "
        f"dense_tp_sharded_sq={breakdown.dense_tp_sharded_sq:.9e} "
        f"dense_tp_replicated_sq={breakdown.dense_tp_replicated_sq:.9e} "
        f"expert_sq={breakdown.expert_sq:.9e}",
        flush=True,
    )


def _print_norm_rank_stats(tag: str, value: float) -> None:
    if not dist.is_initialized():
        return
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda", torch.cuda.current_device())
    min_value = torch.tensor(float(value), dtype=torch.float64, device=device)
    max_value = min_value.clone()
    dist.all_reduce(min_value, op=dist.ReduceOp.MIN)
    dist.all_reduce(max_value, op=dist.ReduceOp.MAX)
    if dist.get_rank() != 0:
        return
    print(
        f"[{tag}] "
        f"world={dist.get_world_size()} "
        f"min={float(min_value.item()):.9f} "
        f"max={float(max_value.item()):.9f}",
        flush=True,
    )


def _print_breakdown_rank_stats(tag: str, breakdown: GradNormBreakdown) -> None:
    if not dist.is_initialized():
        return
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda", torch.cuda.current_device())
    values = torch.tensor(
        [
            breakdown.dense_sq,
            breakdown.dense_tp_sharded_sq,
            breakdown.dense_tp_replicated_sq,
            breakdown.expert_sq,
            breakdown.global_dense_sq,
            breakdown.global_dense_tp_sharded_sq,
            breakdown.global_dense_tp_replicated_sq,
            breakdown.global_expert_sq,
            breakdown.total_sq,
        ],
        dtype=torch.float64,
        device=device,
    )
    mins = values.clone()
    maxs = values.clone()
    dist.all_reduce(mins, op=dist.ReduceOp.MIN)
    dist.all_reduce(maxs, op=dist.ReduceOp.MAX)
    if dist.get_rank() != 0:
        return
    print(
        f"[{tag}] "
        f"world={dist.get_world_size()} "
        f"dense_sq_min={float(mins[0].item()):.9e} "
        f"dense_sq_max={float(maxs[0].item()):.9e} "
        f"dense_tp_sharded_sq_min={float(mins[1].item()):.9e} "
        f"dense_tp_sharded_sq_max={float(maxs[1].item()):.9e} "
        f"dense_tp_replicated_sq_min={float(mins[2].item()):.9e} "
        f"dense_tp_replicated_sq_max={float(maxs[2].item()):.9e} "
        f"expert_sq_min={float(mins[3].item()):.9e} "
        f"expert_sq_max={float(maxs[3].item()):.9e} "
        f"global_dense_sq_min={float(mins[4].item()):.9e} "
        f"global_dense_sq_max={float(maxs[4].item()):.9e} "
        f"global_dense_tp_sharded_sq_min={float(mins[5].item()):.9e} "
        f"global_dense_tp_sharded_sq_max={float(maxs[5].item()):.9e} "
        f"global_dense_tp_replicated_sq_min={float(mins[6].item()):.9e} "
        f"global_dense_tp_replicated_sq_max={float(maxs[6].item()):.9e} "
        f"global_expert_sq_min={float(mins[7].item()):.9e} "
        f"global_expert_sq_max={float(maxs[7].item()):.9e} "
        f"total_sq_min={float(mins[8].item()):.9e} "
        f"total_sq_max={float(maxs[8].item()):.9e}",
        flush=True,
    )


__all__ = [
    "CanonicalGradNormMegatronFSDPOptimizer",
    "GradNormBreakdown",
    "compute_mfsdp_grad_norm",
]
