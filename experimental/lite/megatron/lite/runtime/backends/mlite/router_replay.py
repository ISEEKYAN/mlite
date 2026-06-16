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
from megatron.lite.model import protocol_utils
from megatron.lite.primitive.modules.router import (
    RouterReplay,
    RouterReplayAction,
    attach_router_replay,
    detach_router_replay,
)

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
        # Single pass across all chunks so the global per-layer registry is in
        # pipeline-local layer order (attach_router_replay clears on each call,
        # so we inline the equivalent walk here for the multi-chunk case).
        RouterReplay.clear_global_router_replay_instances()
        if len(self._chunks) == 1:
            self._num_routers = attach_router_replay(self._chunks[0])
        else:
            self._num_routers = 0
            for chunk in self._chunks:
                for module in chunk.modules():
                    if hasattr(module, "router_replay"):
                        module.router_replay = RouterReplay()
                        self._num_routers += 1
        if self._num_routers == 0:
            raise RuntimeError("router replay requested but the model has no MoE routers.")
        if self.action == _RECORD:
            RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)

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
            recorded = RouterReplay.get_recorded_data()
            if any(item is None for item in recorded):
                raise RuntimeError("router replay record completed with missing per-layer indices.")
            stacked = torch.stack([item.to(torch.long) for item in recorded], dim=1)
            nested = _unpack_routed_experts(self._protocol, model, batch, stacked)
            out = dict(out)
            out["routed_experts"] = _to_uint8_if_small(nested)
            RouterReplay.clear_global_indices()
            return out

        return _stepped

    def _wrap_replay(self, forward_step: Callable) -> Callable:
        def _stepped(model, batch):
            routed = getattr(batch, "routed_experts", None)
            if routed is None:
                raise ValueError("router replay 'replay' mode requires batch.routed_experts.")
            targets = _pack_routed_experts(self._protocol, model, batch, routed)
            RouterReplay.set_replay_data(targets)
            RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
            return forward_step(model, batch)

        return _stepped


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
