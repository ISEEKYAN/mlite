# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Native (no-mcore) router-replay semantics — the single shared code path that
all five MoE models (qwen3.5 / qwen3_moe / kimi / glm5 / ds4) route through.

These exercise the primitive layer directly (CPU), so they cover the unified
``RouterReplay`` logic and both shared routers without needing GPU. Per-model
clones (kimi/glm5) call the identical ``RouterReplay.apply`` hook and are covered
by the verl-faithful GPU smoke.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.mlite


@pytest.fixture(autouse=True)
def _te_import_stub(transformer_engine_import_stub):
    transformer_engine_import_stub()


@pytest.fixture(autouse=True)
def _clean_replay_registry():
    from megatron.lite.primitive.modules.router import RouterReplay

    RouterReplay.clear_global_router_replay_instances()
    yield
    RouterReplay.clear_global_router_replay_instances()


def _topk_router(enable_routing_replay: bool):
    from megatron.lite.primitive.modules.router import TopKRouter
    from megatron.lite.primitive.parallel import ParallelState

    config = SimpleNamespace(
        hidden_size=8, num_experts=8, num_experts_per_tok=2, router_aux_loss_coef=0.1
    )
    return TopKRouter(
        config,
        ParallelState(),
        compute_aux_loss=False,
        enable_routing_replay=enable_routing_replay,
    )


def _sigmoid_router(enable_routing_replay: bool):
    from megatron.lite.primitive.modules.router import SigmoidTopKRouter
    from megatron.lite.primitive.parallel import ParallelState

    config = SimpleNamespace(
        hidden_size=8,
        n_routed_experts=8,
        num_experts_per_tok=2,
        routed_scaling_factor=1.0,
        scoring_func="sigmoid",
        aux_loss_alpha=0.0,
    )
    return SigmoidTopKRouter(
        config,
        ParallelState(),
        compute_aux_loss=False,
        enable_routing_replay=enable_routing_replay,
    )


ROUTER_FACTORIES = {"topk_softmax": _topk_router, "sigmoid_topk": _sigmoid_router}


@pytest.mark.parametrize("factory", ROUTER_FACTORIES.values(), ids=ROUTER_FACTORIES.keys())
def test_record_captures_topk_indices(factory):
    from megatron.lite.primitive.modules.router import RouterReplay, RouterReplayAction

    router = factory(True)
    router.eval()
    hidden = torch.randn(6, router.gate.in_features)

    RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)
    scores, indices = router(hidden)

    recorded = RouterReplay.get_recorded_data()
    assert len(recorded) == 1
    assert torch.equal(recorded[0], indices)
    # RECORD leaves the freshly-computed routing untouched.
    assert scores.shape == (6, router.topk)


@pytest.mark.parametrize("factory", ROUTER_FACTORIES.values(), ids=ROUTER_FACTORIES.keys())
def test_replay_forward_reproduces_recorded_indices(factory):
    from megatron.lite.primitive.modules.router import RouterReplay, RouterReplayAction

    router = factory(True)
    router.eval()
    hidden = torch.randn(6, router.gate.in_features)

    RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)
    _, recorded_indices = router(hidden)
    target = recorded_indices.clone()

    RouterReplay.set_replay_data([target])
    RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    replay_scores, replay_indices = router(hidden)

    assert torch.equal(replay_indices, target)
    # Scores are gathered fresh from the current dense probs at the replayed
    # indices (verl semantics): same input → identical to the recorded forward.
    assert replay_scores.shape == (6, router.topk)
    assert torch.isfinite(replay_scores).all()


@pytest.mark.parametrize("factory", ROUTER_FACTORIES.values(), ids=ROUTER_FACTORIES.keys())
def test_replay_forward_overrides_recomputed_routing(factory):
    """The whole point: after the gate weights move, the forward must still
    select the *recorded* experts, not whatever topk now recomputes."""
    from megatron.lite.primitive.modules.router import RouterReplay, RouterReplayAction

    router = factory(True)
    router.eval()
    hidden = torch.randn(6, router.gate.in_features)

    RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)
    _, recorded_indices = router(hidden)
    target = recorded_indices.clone()

    # Perturb the gate so a fresh topk would (very likely) choose other experts.
    with torch.no_grad():
        router.gate.weight.add_(torch.randn_like(router.gate.weight) * 5.0)

    RouterReplay.set_global_router_replay_action(None)
    _, recomputed_indices = router(hidden)

    RouterReplay.set_replay_data([target])
    RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    _, replay_indices = router(hidden)

    assert torch.equal(replay_indices, target)
    # Sanity: the perturbation actually changed the natural routing, so the
    # override above is meaningful (not vacuously true).
    assert not torch.equal(recomputed_indices, target)


@pytest.mark.parametrize("factory", ROUTER_FACTORIES.values(), ids=ROUTER_FACTORIES.keys())
def test_replay_forward_keeps_gate_gradient(factory):
    from megatron.lite.primitive.modules.router import RouterReplay, RouterReplayAction

    router = factory(True)
    router.train()
    hidden = torch.randn(6, router.gate.in_features)

    RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)
    _, recorded_indices = router(hidden)

    RouterReplay.set_replay_data([recorded_indices.clone()])
    RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    replay_scores, _ = router(hidden)
    replay_scores.sum().backward()

    assert router.gate.weight.grad is not None
    assert torch.isfinite(router.gate.weight.grad).all()
    assert router.gate.weight.grad.abs().sum().item() > 0.0


def test_replay_backward_pops_target_in_order():
    from megatron.lite.primitive.modules.router import RouterReplay, RouterReplayAction

    router = _topk_router(True)
    router.eval()
    hidden = torch.randn(6, router.gate.in_features)

    RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)
    _, recorded_indices = router(hidden)
    target = recorded_indices.clone()

    # Forward (replay) then backward-recompute (replay again) must both work.
    RouterReplay.set_replay_data([target])
    RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    _, fwd_indices = router(hidden)
    RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)
    _, bwd_indices = router(hidden)

    assert torch.equal(fwd_indices, target)
    assert torch.equal(bwd_indices, target)
    # The single queued entry is consumed; a second recompute would now error.
    with pytest.raises(RuntimeError):
        router(hidden)


def test_disabled_replay_is_noop():
    from megatron.lite.primitive.modules.router import RouterReplay, RouterReplayAction

    router = _topk_router(False)
    router.eval()
    hidden = torch.randn(6, router.gate.in_features)

    # No instance is registered when replay is disabled.
    assert RouterReplay.global_router_replay_instances == []
    RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)
    scores, indices = router(hidden)
    assert scores.shape == (6, router.topk)
    assert RouterReplay.get_recorded_data() == []


def test_set_replay_data_length_mismatch_raises():
    from megatron.lite.primitive.modules.router import RouterReplay

    _topk_router(True)  # registers exactly one instance
    with pytest.raises(ValueError):
        RouterReplay.set_replay_data([torch.zeros(2, 2, dtype=torch.long)] * 2)
