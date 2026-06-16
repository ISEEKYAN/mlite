# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MoE router implementations: TopKRouter (softmax) and SigmoidTopKRouter."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import torch  # pyright: ignore[reportMissingImports]
import torch.distributed as dist  # pyright: ignore[reportMissingImports]
import torch.nn as nn  # pyright: ignore[reportMissingImports]

from megatron.lite.primitive.modules.moe import MoEAuxLossAutoScaler
from megatron.lite.primitive.utils.moe import (
    compute_routing_scores_for_aux_loss,
    router_gating_linear,
    switch_load_balancing_loss_func,
    topk_routing_with_score_function,
)

if TYPE_CHECKING:
    from megatron.lite.primitive.parallel import ParallelState


class RouterReplayAction(Enum):
    """Action for a router-replay step (mirrors the verl/mcore contract)."""

    RECORD = "record"  # capture the topk expert indices for later replay
    REPLAY_FORWARD = "replay_forward"  # force the recorded indices in the forward pass
    REPLAY_BACKWARD = "replay_backward"  # force them again during backward recompute


class RouterReplay:
    """Native (no-mcore) record/replay of MoE routing decisions.

    One instance per MoE router; instances register themselves in
    ``global_router_replay_instances`` in construction order so the runtime can
    address per-layer routing tensors positionally. API mirrors the verl
    ``router_replay_patch.RouterReplay`` so the mlite verl engine and the
    model-agnostic runtime drive every model through this single code path.

    Replay only overrides the expert *indices*; the gating *scores* are gathered
    fresh from the current ``probs_dense`` so gradients still flow through the
    gate while the routing selection is held fixed.
    """

    global_router_replay_instances: list["RouterReplay"] = []

    # ── global controls (one call fans out to every layer) ──
    @staticmethod
    def set_replay_data(all_layers_topk_indices: list[torch.Tensor]) -> None:
        instances = RouterReplay.global_router_replay_instances
        if len(all_layers_topk_indices) != len(instances):
            raise ValueError(
                f"router replay expects {len(instances)} per-layer tensors, "
                f"got {len(all_layers_topk_indices)}."
            )
        for inst, idx in zip(instances, all_layers_topk_indices, strict=True):
            inst.set_target_indices(idx)

    @staticmethod
    def get_recorded_data() -> list[torch.Tensor | None]:
        return [inst.get_recorded_indices() for inst in RouterReplay.global_router_replay_instances]

    @staticmethod
    def clear_global_indices() -> None:
        for inst in RouterReplay.global_router_replay_instances:
            inst.clear_indices()

    @staticmethod
    def set_global_router_replay_action(action: RouterReplayAction) -> None:
        for inst in RouterReplay.global_router_replay_instances:
            inst.set_router_replay_action(action)

    @staticmethod
    def clear_global_router_replay_action() -> None:
        for inst in RouterReplay.global_router_replay_instances:
            inst.clear_router_replay_action()

    @staticmethod
    def clear_global_router_replay_instances() -> None:
        RouterReplay.global_router_replay_instances.clear()

    # ── per-instance state ──
    def __init__(self) -> None:
        self.target_topk_idx: torch.Tensor | None = None
        self.recorded_topk_idx: torch.Tensor | None = None
        self.router_replay_action: RouterReplayAction | None = None
        self.replay_backward_list: list[torch.Tensor] = []
        RouterReplay.global_router_replay_instances.append(self)

    def set_target_indices(self, topk_indices: torch.Tensor) -> None:
        self.target_topk_idx = topk_indices
        # Each forward (incl. the backward recompute) pops one entry, in order.
        self.replay_backward_list.append(topk_indices)

    def get_recorded_indices(self) -> torch.Tensor | None:
        return self.recorded_topk_idx

    def record_indices(self, topk_indices: torch.Tensor) -> None:
        self.recorded_topk_idx = topk_indices

    def clear_indices(self) -> None:
        self.recorded_topk_idx = None
        self.target_topk_idx = None
        self.replay_backward_list = []

    def set_router_replay_action(self, action: RouterReplayAction) -> None:
        self.router_replay_action = action

    def clear_router_replay_action(self) -> None:
        self.router_replay_action = None

    def apply(
        self,
        probs_dense: torch.Tensor,
        topk_scores: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Shared replay hook every router calls after computing its own topk.

        ``probs_dense`` is the dense gating score tensor ``[tokens, num_experts]``;
        ``topk_scores``/``topk_indices`` are this router's freshly-computed
        ``[tokens, topk]`` selection. Returns the (possibly overridden) pair.
        """
        action = self.router_replay_action
        if action == RouterReplayAction.RECORD:
            self.record_indices(topk_indices)
            return topk_scores, topk_indices
        if action == RouterReplayAction.REPLAY_FORWARD:
            target = self._take_target(self.target_topk_idx).to(probs_dense.device)
            return probs_dense.gather(-1, target).to(topk_scores.dtype), target
        if action == RouterReplayAction.REPLAY_BACKWARD:
            if not self.replay_backward_list:
                raise RuntimeError("router replay backward list exhausted; recompute/forward mismatch.")
            target = self.replay_backward_list.pop(0).to(probs_dense.device)
            return probs_dense.gather(-1, target).to(topk_scores.dtype), target
        return topk_scores, topk_indices

    @staticmethod
    def _take_target(target: torch.Tensor | None) -> torch.Tensor:
        if target is None:
            raise RuntimeError("router replay is in replay mode but no target indices were set.")
        return target


def attach_router_replay(model: nn.Module) -> int:
    """Enable router replay on every MoE router in ``model`` (one shared path).

    Walks the module tree and gives each replay-capable router (any module that
    exposes a ``router_replay`` slot — all mlite routers do) a fresh
    :class:`RouterReplay`, registered in module-traversal (== layer) order. This
    lets the model-agnostic runtime turn replay on for all five models without
    threading a constructor flag through any model. Returns the router count.

    Each rank attaches only its *local* routers (its pipeline stage's layers),
    so the per-layer registry is naturally local; the runtime gathers/scatters
    routing tensors across pipeline ranks.
    """
    RouterReplay.clear_global_router_replay_instances()
    count = 0
    for module in model.modules():
        if hasattr(module, "router_replay"):
            module.router_replay = RouterReplay()
            count += 1
    return count


def detach_router_replay(model: nn.Module) -> None:
    """Disable router replay and clear the global registry (inverse of attach)."""
    for module in model.modules():
        if hasattr(module, "router_replay"):
            module.router_replay = None
    RouterReplay.clear_global_router_replay_instances()


def _ordered_topk_from_routing_map(
    probs_dense: torch.Tensor, routing_map: torch.Tensor, topk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    expert_ids = torch.arange(
        probs_dense.size(-1), device=probs_dense.device, dtype=torch.long
    ).expand_as(routing_map)
    masked_ids = torch.where(
        routing_map, expert_ids, torch.full_like(expert_ids, probs_dense.size(-1))
    )
    topk_indices = torch.sort(masked_ids, dim=-1).values[:, :topk]
    topk_scores = torch.gather(probs_dense, dim=-1, index=topk_indices)
    return topk_scores, topk_indices


class TopKRouter(nn.Module):
    """TopK gating with optional high-precision router logits/probabilities."""

    def __init__(
        self,
        config,
        ps: ParallelState,
        *,
        router_bias_rate: float = 0.0,
        compute_aux_loss: bool = True,
        use_pre_softmax: bool = False,
        moe_router_fusion: bool = False,
        router_dtype: torch.dtype | None = None,
        enable_routing_replay: bool = False,
    ):
        super().__init__()
        if router_bias_rate > 0:
            raise NotImplementedError(
                "expert-bias EMA is not implemented in the primitive router; "
                "use load_balancing_type='none' or extend ParallelState."
            )
        self.topk = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.aux_loss_coeff = config.router_aux_loss_coef
        self.router_bias_rate = router_bias_rate
        self.compute_aux_loss = compute_aux_loss
        self.use_pre_softmax = use_pre_softmax
        self.moe_router_fusion = moe_router_fusion
        self.router_dtype = router_dtype
        self.router_replay = RouterReplay() if enable_routing_replay else None

        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.register_buffer(
            "expert_bias", torch.zeros(config.num_experts, dtype=torch.float32), persistent=False
        )

        self._aux_loss_group = ps.tp_group if ps.tp_size > 1 else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        router_dtype = self.router_dtype or x.dtype
        logits = router_gating_linear(x, self.gate.weight, None, router_dtype)
        logits = logits.view(-1, self.num_experts)
        num_tokens = logits.size(0)
        if self.moe_router_fusion:
            probs_dense, _ = topk_routing_with_score_function(
                logits,
                self.topk,
                use_pre_softmax=self.use_pre_softmax,
                score_function="softmax",
                fused=True,
            )
            topk_scores, topk_indices = torch.topk(probs_dense, k=self.topk, dim=-1)
        else:
            probs_dense, routing_map = topk_routing_with_score_function(
                logits,
                self.topk,
                use_pre_softmax=self.use_pre_softmax,
                score_function="softmax",
                fused=False,
            )
            topk_scores, topk_indices = _ordered_topk_from_routing_map(
                probs_dense, routing_map, self.topk
            )
        if self.router_replay is not None:
            topk_scores, topk_indices = self.router_replay.apply(
                probs_dense, topk_scores, topk_indices
            )
        if self.router_dtype is None:
            topk_scores = topk_scores.to(x.dtype)

        if self.compute_aux_loss and self.training and torch.is_grad_enabled():
            routing_map, aux_scores = compute_routing_scores_for_aux_loss(
                logits, self.topk, score_function="softmax", fused=self.moe_router_fusion
            )
            tokens_per_expert = routing_map.sum(dim=0).to(torch.int64)
            total_num_tokens = num_tokens
            if self._aux_loss_group is not None:
                dist.all_reduce(tokens_per_expert, group=self._aux_loss_group)
                total_num_tokens = num_tokens * dist.get_world_size(group=self._aux_loss_group)
            aux_loss = switch_load_balancing_loss_func(
                aux_scores,
                tokens_per_expert,
                total_num_tokens,
                self.topk,
                self.num_experts,
                self.aux_loss_coeff,
                fused=False,
            )
            topk_scores = MoEAuxLossAutoScaler.apply(topk_scores, aux_loss)

        return topk_scores, topk_indices


class SigmoidTopKRouter(nn.Module):
    """Sigmoid-family TopK router for DeepSeek-style MoE."""

    def __init__(
        self,
        config,
        ps: ParallelState,
        *,
        router_bias_rate: float = 0.0,
        compute_aux_loss: bool = True,
        use_pre_softmax: bool = False,
        moe_router_fusion: bool = False,
        enable_routing_replay: bool = False,
    ):
        super().__init__()
        if router_bias_rate > 0:
            raise NotImplementedError(
                "expert-bias EMA is not implemented in the primitive router; "
                "use load_balancing_type='none' or extend ParallelState."
            )
        self.topk = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts
        self.aux_loss_coeff = getattr(config, "aux_loss_alpha", 0.0)
        self.scaling_factor = config.routed_scaling_factor
        self.score_function = getattr(config, "scoring_func", "sigmoid")
        self.router_bias_rate = router_bias_rate
        self.compute_aux_loss = compute_aux_loss
        self.use_pre_softmax = use_pre_softmax
        self.moe_router_fusion = moe_router_fusion
        self.router_replay = RouterReplay() if enable_routing_replay else None

        self.gate = nn.Linear(config.hidden_size, config.n_routed_experts, bias=False)
        self.register_buffer(
            "expert_bias",
            torch.zeros(config.n_routed_experts, dtype=torch.float32),
            persistent=False,
        )

        self._aux_loss_group = ps.tp_group if ps.tp_size > 1 else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate(x)
        logits = logits.view(-1, self.num_experts)
        num_tokens = logits.size(0)
        probs_dense, routing_map = topk_routing_with_score_function(
            logits,
            self.topk,
            score_function=self.score_function,
            expert_bias=self.expert_bias.to(logits.dtype),
            scaling_factor=(self.scaling_factor or None),
            fused=self.moe_router_fusion,
        )
        topk_scores, topk_indices = _ordered_topk_from_routing_map(
            probs_dense, routing_map, self.topk
        )
        if self.router_replay is not None:
            topk_scores, topk_indices = self.router_replay.apply(
                probs_dense, topk_scores, topk_indices
            )
        topk_scores = topk_scores.to(logits.dtype)

        if self.compute_aux_loss and self.training and torch.is_grad_enabled():
            _, aux_scores = compute_routing_scores_for_aux_loss(
                logits, self.topk, score_function=self.score_function, fused=self.moe_router_fusion
            )
            tokens_per_expert = routing_map.sum(dim=0).to(torch.int64)
            total_num_tokens = num_tokens
            if self._aux_loss_group is not None:
                dist.all_reduce(tokens_per_expert, group=self._aux_loss_group)
                total_num_tokens = num_tokens * dist.get_world_size(group=self._aux_loss_group)
            aux_loss = switch_load_balancing_loss_func(
                aux_scores,
                tokens_per_expert,
                total_num_tokens,
                self.topk,
                self.num_experts,
                self.aux_loss_coeff,
                fused=False,
            )
            topk_scores = MoEAuxLossAutoScaler.apply(topk_scores, aux_loss)

        return topk_scores, topk_indices


__all__ = [
    "RouterReplay",
    "RouterReplayAction",
    "SigmoidTopKRouter",
    "TopKRouter",
    "attach_router_replay",
    "detach_router_replay",
]
