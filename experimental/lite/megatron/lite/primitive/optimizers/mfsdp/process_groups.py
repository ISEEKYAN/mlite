# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Megatron-FSDP process-group adapter.

This module is intentionally M-FSDP-local.  It translates Megatron Lite's
``ParallelState`` rank layout into the process groups and mesh shape expected
by the installed Megatron-Core FSDP wrapper.
"""

from __future__ import annotations

from typing import Any

import torch  # pyright: ignore[reportMissingImports]
import torch.distributed as dist  # pyright: ignore[reportMissingImports]


def install_mfsdp_mesh_patch() -> None:
    """Make Megatron-Core FSDP mesh helpers pipeline-stage aware.

    The Megatron-Core build in the runbook image creates FSDP meshes from the
    global world size, which assumes ``world = dp_cp * ep * tp``. Megatron Lite's
    PP layout is ``world = pp * dp_cp * tp`` for dense params and
    ``world = pp * expert_dp * ep * etp`` for expert params.  Build the mesh
    from the rank block containing the current pipeline stage instead.
    """

    from megatron.core.distributed.fsdp import (  # pyright: ignore[reportMissingImports]
        mcore_fsdp_adapter,
    )

    if getattr(mcore_fsdp_adapter, "_mlite_mfsdp_pp_mesh_patch", False):
        return

    mcore_fsdp_adapter._mlite_original_get_dp_tp_mesh = mcore_fsdp_adapter._get_dp_tp_mesh
    mcore_fsdp_adapter._mlite_original_get_hsdp_tp_mesh = (
        mcore_fsdp_adapter._get_hsdp_tp_mesh
    )
    mcore_fsdp_adapter._get_dp_tp_mesh = _get_pp_local_dp_tp_mesh
    mcore_fsdp_adapter._get_hsdp_tp_mesh = _get_pp_local_hsdp_tp_mesh
    mcore_fsdp_adapter._mlite_mfsdp_pp_mesh_patch = True


def build_mfsdp_pg_collection(ps, engine_cfg):
    """Build MCore ``ProcessGroupCollection`` for the Megatron-FSDP primitive."""

    from megatron.core.process_groups_config import (  # pyright: ignore[reportMissingImports]
        ProcessGroupCollection,
    )

    if ps.pp_group is None:
        raise ValueError("optimizer_impl='megatron_fsdp' requires a local pp_group.")

    rank = dist.get_rank()
    world = dist.get_world_size()
    num_dist_opt_instances = _distributed_optimizer_instances(engine_cfg)
    if num_dist_opt_instances < 1:
        raise ValueError("num_distributed_optimizer_instances must be a positive integer.")
    if num_dist_opt_instances > 1:
        if ps.pp_size > 1:
            raise ValueError(
                "optimizer_impl='megatron_fsdp' has not closed HSDP with PP/VPP yet."
            )
        if ps.dp_cp_size % num_dist_opt_instances != 0:
            raise ValueError(
                "optimizer_impl='megatron_fsdp' requires dp_cp_size to be divisible by "
                "num_distributed_optimizer_instances."
            )
        if ps.expert_dp_size % num_dist_opt_instances != 0:
            raise ValueError(
                "optimizer_impl='megatron_fsdp' requires expert_dp_size to be divisible by "
                "num_distributed_optimizer_instances."
            )

    singleton_group = None
    for singleton_rank in range(world):
        group = dist.new_group([singleton_rank])
        if rank == singleton_rank:
            singleton_group = group
    if singleton_group is None:
        raise RuntimeError("Failed to construct singleton process group for optional reductions.")

    if ps.pp_size == 1:
        mp_group = ps.tp_group
        tp_ep_pp_group = ps.tp_ep_group
    else:
        mp_group = None
        for dp_idx in range(ps.dp_size):
            for cp_idx in range(ps.cp_size):
                ranks = [
                    _dense_rank(ps, tp_idx, cp_idx, dp_idx, pp_idx)
                    for pp_idx in range(ps.pp_size)
                    for tp_idx in range(ps.tp_size)
                ]
                group = dist.new_group(ranks)
                if rank in ranks:
                    mp_group = group

        tp_ep_pp_group = None
        for expert_dp_idx in range(ps.expert_dp_size):
            ranks = [
                _expert_rank(ps, etp_idx, ep_idx, expert_dp_idx, pp_idx)
                for pp_idx in range(ps.pp_size)
                for ep_idx in range(ps.ep_size)
                for etp_idx in range(ps.etp_size)
            ]
            group = dist.new_group(ranks)
            if rank in ranks:
                tp_ep_pp_group = group

        if mp_group is None or tp_ep_pp_group is None:
            raise RuntimeError("Failed to construct M-FSDP pipeline-aware process groups.")

    intra_dp_cp_group = ps.dp_cp_group
    intra_expt_dp_group = ps.ep_dp_group
    inter_dist_opt_group = None
    intra_dist_opt_group = dist.group.WORLD
    if num_dist_opt_instances > 1:
        (
            intra_dp_cp_group,
            intra_expt_dp_group,
            inter_dist_opt_group,
            intra_dist_opt_group,
        ) = _build_hsdp_groups(ps, num_dist_opt_instances, rank)

    return ProcessGroupCollection(
        tp=ps.tp_group,
        cp=ps.cp_group,
        pp=ps.pp_group,
        ep=ps.ep_group,
        mp=mp_group,
        dp=ps.dp_group,
        dp_cp=ps.dp_cp_group,
        expt_dp=ps.ep_dp_group,
        intra_dp_cp=intra_dp_cp_group,
        intra_expt_dp=intra_expt_dp_group,
        inter_dist_opt=inter_dist_opt_group,
        expt_tp=ps.etp_group,
        tp_ep=ps.tp_ep_group,
        tp_ep_pp=tp_ep_pp_group,
        intra_dist_opt=intra_dist_opt_group,
        embd=singleton_group,
        pos_embd=singleton_group,
    )


def _build_hsdp_groups(ps, num_dist_opt_instances: int, rank: int):
    inner_dp_cp = ps.dp_cp_size // num_dist_opt_instances
    inner_expert_dp = ps.expert_dp_size // num_dist_opt_instances
    intra_dp_cp_group = None
    intra_expt_dp_group = None
    inter_dist_opt_group = None
    intra_dist_opt_group = None

    for pp_idx in range(ps.pp_size):
        for tp_idx in range(ps.tp_size):
            for outer_idx in range(num_dist_opt_instances):
                start = outer_idx * inner_dp_cp
                ranks = []
                for local_dp_cp_idx in range(inner_dp_cp):
                    dp_cp_idx = start + local_dp_cp_idx
                    dp_idx = dp_cp_idx // ps.cp_size
                    cp_idx = dp_cp_idx % ps.cp_size
                    ranks.append(_dense_rank(ps, tp_idx, cp_idx, dp_idx, pp_idx))
                group = dist.new_group(ranks)
                if rank in ranks:
                    intra_dp_cp_group = group

    for pp_idx in range(ps.pp_size):
        for etp_idx in range(ps.etp_size):
            for ep_idx in range(ps.ep_size):
                for outer_idx in range(num_dist_opt_instances):
                    start = outer_idx * inner_expert_dp
                    ranks = [
                        _expert_rank(ps, etp_idx, ep_idx, expert_dp_idx, pp_idx)
                        for expert_dp_idx in range(start, start + inner_expert_dp)
                    ]
                    group = dist.new_group(ranks)
                    if rank in ranks:
                        intra_expt_dp_group = group

    for pp_idx in range(ps.pp_size):
        for tp_idx in range(ps.tp_size):
            for local_dp_cp_idx in range(inner_dp_cp):
                ranks = []
                for outer_idx in range(num_dist_opt_instances):
                    dp_cp_idx = outer_idx * inner_dp_cp + local_dp_cp_idx
                    dp_idx = dp_cp_idx // ps.cp_size
                    cp_idx = dp_cp_idx % ps.cp_size
                    ranks.append(_dense_rank(ps, tp_idx, cp_idx, dp_idx, pp_idx))
                group = dist.new_group(ranks)
                if rank in ranks:
                    inter_dist_opt_group = group

    for pp_idx in range(ps.pp_size):
        for outer_idx in range(num_dist_opt_instances):
            ranks = []
            start = outer_idx * inner_dp_cp
            for local_dp_cp_idx in range(inner_dp_cp):
                dp_cp_idx = start + local_dp_cp_idx
                dp_idx = dp_cp_idx // ps.cp_size
                cp_idx = dp_cp_idx % ps.cp_size
                for tp_idx in range(ps.tp_size):
                    ranks.append(_dense_rank(ps, tp_idx, cp_idx, dp_idx, pp_idx))
            group = dist.new_group(ranks)
            if rank in ranks:
                intra_dist_opt_group = group

    if (
        intra_dp_cp_group is None
        or intra_expt_dp_group is None
        or inter_dist_opt_group is None
        or intra_dist_opt_group is None
    ):
        raise RuntimeError("Failed to construct Megatron-FSDP HSDP process groups.")

    return (
        intra_dp_cp_group,
        intra_expt_dp_group,
        inter_dist_opt_group,
        intra_dist_opt_group,
    )


def _get_pp_local_dp_tp_mesh(dp_cp_group, tp_group, ep_size=1):
    rank = dist.get_rank()
    dp_cp_size = dp_cp_group.size()
    tp_size = dist.get_world_size(tp_group) if tp_group is not None else 1
    tp_idx = _group_index(tp_group, rank) if tp_group is not None else 0
    dp_cp_idx = _group_index(dp_cp_group, rank)
    ep_idx = ((rank - tp_idx) // tp_size) % int(ep_size)
    base_rank = rank - ((dp_cp_idx * int(ep_size) + ep_idx) * tp_size + tp_idx)
    mesh = torch.arange(
        base_rank,
        base_rank + dp_cp_size * int(ep_size) * tp_size,
    ).reshape(dp_cp_size, int(ep_size), tp_size)
    return mesh.permute(1, 0, 2)[ep_idx].contiguous()


def _get_pp_local_hsdp_tp_mesh(outer_fsdp_dp_group, dp_cp_group, tp_group):
    rank = dist.get_rank()
    outer_size = outer_fsdp_dp_group.size()
    fsdp_size = dp_cp_group.size()
    tp_size = dist.get_world_size(tp_group) if tp_group is not None else 1
    outer_idx = _group_index(outer_fsdp_dp_group, rank)
    fsdp_idx = _group_index(dp_cp_group, rank)
    tp_idx = _group_index(tp_group, rank) if tp_group is not None else 0
    base_rank = rank - ((outer_idx * fsdp_size + fsdp_idx) * tp_size + tp_idx)
    return torch.arange(
        base_rank,
        base_rank + outer_size * fsdp_size * tp_size,
    ).reshape(outer_size, fsdp_size, tp_size)


def _group_index(group, rank: int) -> int:
    ranks = dist.get_process_group_ranks(group)
    try:
        return ranks.index(rank)
    except ValueError as exc:
        raise RuntimeError(f"Rank {rank} is not in process group ranks {ranks}.") from exc


def _dense_rank(ps, tp_i: int, cp_i: int, dp_i: int, pp_i: int) -> int:
    return ((pp_i * ps.dp_size + dp_i) * ps.cp_size + cp_i) * ps.tp_size + tp_i


def _expert_rank(ps, etp_i: int, ep_i: int, edp_i: int, pp_i: int) -> int:
    return ((pp_i * ps.expert_dp_size + edp_i) * ps.ep_size + ep_i) * ps.etp_size + etp_i


def _distributed_optimizer_instances(engine_cfg: Any) -> int:
    opt = getattr(engine_cfg, "optimizer", None)
    if opt is None:
        return 1
    values = dict(getattr(opt, "override_optimizer_config", None) or {})
    raw = values.get(
        "num_distributed_optimizer_instances",
        getattr(opt, "num_distributed_optimizer_instances", None),
    )
    if raw is None or raw is False or raw is True:
        return 1
    if isinstance(raw, int | float):
        return int(raw)
    if isinstance(raw, str):
        normalized = raw.lower()
        if normalized in {"", "none", "null"}:
            return 1
        return int(normalized)
    return 1


__all__ = [
    "build_mfsdp_pg_collection",
    "install_mfsdp_mesh_patch",
]
