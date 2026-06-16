# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Model-agnostic router-replay driver for the mlite runtime.

Drives RECORD / REPLAY of MoE routing across the runtime's microbatch loop using
the shared :class:`RouterReplay` registry in ``primitive/modules/router.py`` — a
single code path for every model. The model contributes only its THD CP layout
through the protocol's ``pack_routed_experts`` / ``unpack_routed_experts`` pair
(shared zigzag default; DeepSeek-V4 provides a contiguous variant).

Modes:
- ``record``: capture each layer's topk expert indices during the (forward-only)
  log-prob pass and return them as ``out["routed_experts"]`` for verl to store.
- ``replay``: force the recorded indices in the policy-update forward (and the
  backward recompute), reading them from ``batch.routed_experts``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist
from megatron.lite.model import protocol_utils
from megatron.lite.primitive.modules.router import (
    RouterReplay,
    RouterReplayAction,
    attach_router_replay,
    detach_router_replay,
)
from megatron.lite.primitive.parallel.thd import parallel_state_from_model

_RECORD = "record"
_REPLAY = "replay"


def _pack_routed_experts(protocol, model, batch, routed):
    """Use the model protocol's pack if defined, else the shared zigzag THD one."""
    fn = getattr(protocol, "pack_routed_experts", None) if protocol is not None else None
    return (fn or protocol_utils.pack_routed_experts)(model, batch, routed)


def _unpack_routed_experts(protocol, model, batch, recorded):
    fn = getattr(protocol, "unpack_routed_experts", None) if protocol is not None else None
    return (fn or protocol_utils.unpack_routed_experts)(model, batch, recorded)


class RouterReplayDriver:
    """Owns the replay lifecycle for one ``forward_backward`` call."""

    def __init__(self, handle, action: str):
        if action not in (_RECORD, _REPLAY):
            raise ValueError(f"router replay action must be 'record' or 'replay', got {action!r}.")
        self.handle = handle
        self.action = action
        self._chunks = handle._extras.get("model_chunks", [handle._model])
        self._protocol = handle._extras.get("protocol")
        self._num_routers = 0
        self._ps = None
        self._pp_offset = 0  # this rank's first global MoE-layer index
        self._pp_total = 0  # total MoE layers across all pipeline stages
        self._last_batch = None  # stashed for post-schedule record collection
        self._last_model = None

    @classmethod
    def maybe_create(cls, handle, router_replay: Any) -> RouterReplayDriver | None:
        """Build a driver from a ``{'action': ...}`` spec, or ``None`` if disabled."""
        if not router_replay:
            return None
        action = router_replay.get("action") if isinstance(router_replay, dict) else router_replay
        if action in (None, "disabled"):
            return None
        return cls(handle, action)

    # ── lifecycle ──
    def begin(self) -> None:
        # Attach across all chunks in one registry so the global per-layer order
        # matches pipeline-local layer order (clear once, then append per chunk).
        RouterReplay.clear_global_router_replay_instances()
        self._num_routers = sum(
            attach_router_replay(chunk, reset=False) for chunk in self._chunks
        )
        if self._num_routers == 0:
            raise RuntimeError("router replay requested but the model has no MoE routers.")
        self._ps = parallel_state_from_model(self._chunks[-1])
        self._compute_pp_layout()
        if self.action == _RECORD:
            RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)

    def _compute_pp_layout(self) -> None:
        """All-gather per-rank MoE-router counts so record/replay can map this
        stage's local routers to their global MoE-layer indices."""
        ps = self._ps
        if ps is None or ps.pp_size <= 1:
            self._pp_offset, self._pp_total = 0, self._num_routers
            return
        counts = [torch.zeros(1, dtype=torch.long, device="cuda") for _ in range(ps.pp_size)]
        dist.all_gather(
            counts, torch.tensor([self._num_routers], dtype=torch.long, device="cuda"), group=ps.pp_group
        )
        counts = [int(c.item()) for c in counts]
        self._pp_offset = sum(counts[: ps.pp_rank])
        self._pp_total = sum(counts)

    def wrap(self, forward_step: Callable) -> Callable:
        if self.action == _RECORD:
            return self._wrap_record(forward_step)
        return self._wrap_replay(forward_step)

    def end(self) -> None:
        RouterReplay.clear_global_indices()
        RouterReplay.clear_global_router_replay_action()
        for chunk in self._chunks:
            detach_router_replay(chunk)

    # ── per-microbatch wrappers ──
    def _wrap_record(self, forward_step: Callable) -> Callable:
        def _stepped(model, batch):
            RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)
            out = forward_step(model, batch)
            # Collection is deferred to collect_recorded() after the schedule so a
            # pipeline stage's local layers can be PP-gathered into global order.
            self._last_batch, self._last_model = batch, model
            return out

        return _stepped

    def _wrap_replay(self, forward_step: Callable) -> Callable:
        def _stepped(model, batch):
            routed = getattr(batch, "routed_experts", None)
            if routed is None:
                raise ValueError("router replay 'replay' mode requires batch.routed_experts.")
            routed = self._select_local_layers(routed)
            targets = _pack_routed_experts(self._protocol, model, batch, routed)
            RouterReplay.set_replay_data(targets)
            RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
            return forward_step(model, batch)

        return _stepped

    def collect_recorded(self):
        """Assemble the recorded routing as jagged ``[bs, seq, total_layers, topk]``
        (PP-gathered), or ``None`` if nothing was recorded."""
        if self.action != _RECORD or self._last_batch is None:
            return None
        recorded = RouterReplay.get_recorded_data()
        if not recorded or any(item is None for item in recorded):
            raise RuntimeError("router replay record finished with missing per-layer indices.")
        local = torch.stack([item.to(torch.long) for item in recorded], dim=1)  # [tok, n_local, topk]
        full = self._pp_gather_layers(local)
        nested = _unpack_routed_experts(self._protocol, self._last_model, self._last_batch, full)
        return _to_uint8_if_small(nested)

    # ── pipeline helpers ──
    def _select_local_layers(self, routed):
        """Slice the full routing down to this pipeline stage's MoE layers."""
        ps = self._ps
        if ps is None or ps.pp_size <= 1:
            return routed
        lo, hi = self._pp_offset, self._pp_offset + self._num_routers
        rows = [row[:, lo:hi, :] for row in routed.unbind(0)]
        return torch.nested.as_nested_tensor(rows, layout=torch.jagged)

    def _pp_gather_layers(self, local: torch.Tensor) -> torch.Tensor:
        """All-gather per-stage routing ``[tok, n_local, topk]`` and concat along
        the layer axis in pipeline-rank order. Assumes a uniform per-stage layer
        count (even split); uneven layouts are not yet supported."""
        ps = self._ps
        if ps is None or ps.pp_size <= 1:
            return local
        if self._pp_total != self._num_routers * ps.pp_size:
            raise NotImplementedError(
                "router replay PP gather requires an even MoE-layer split across stages "
                f"(got total={self._pp_total}, local={self._num_routers}, pp={ps.pp_size})."
            )
        parts = [torch.empty_like(local) for _ in range(ps.pp_size)]
        dist.all_gather(parts, local.contiguous(), group=ps.pp_group)
        return torch.cat(parts, dim=1)


def _to_uint8_if_small(nested):
    """Match verl's uint8 routed_experts when expert ids fit in a byte."""
    try:
        values = nested.values() if getattr(nested, "is_nested", False) else nested
        if values.numel() and int(values.max().item()) <= 255:
            return nested.to(torch.uint8)
    except Exception:
        pass
    return nested


__all__ = ["RouterReplayDriver"]
