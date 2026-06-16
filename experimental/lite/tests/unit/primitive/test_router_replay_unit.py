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


def _device():
    # TopKRouter.gate uses TE's router_gating_linear (CUDA-only gemm); run on GPU
    # when one is present (e.g. inside the container), CPU otherwise (login node
    # with the TE import stub handling the fallback).
    return "cuda" if torch.cuda.is_available() else "cpu"


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
    ).to(_device())


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
    ).to(_device())


ROUTER_FACTORIES = {"topk_softmax": _topk_router, "sigmoid_topk": _sigmoid_router}


@pytest.mark.parametrize("factory", ROUTER_FACTORIES.values(), ids=ROUTER_FACTORIES.keys())
def test_record_captures_topk_indices(factory):
    from megatron.lite.primitive.modules.router import RouterReplay, RouterReplayAction

    router = factory(True)
    router.eval()
    hidden = torch.randn(6, router.gate.in_features, device=_device())

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
    hidden = torch.randn(6, router.gate.in_features, device=_device())

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
    hidden = torch.randn(6, router.gate.in_features, device=_device())

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
    hidden = torch.randn(6, router.gate.in_features, device=_device())

    RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)
    _, recorded_indices = router(hidden)

    RouterReplay.set_replay_data([recorded_indices.clone()])
    RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    replay_scores, _ = router(hidden)
    # Sum-of-squares, not sum: a post-softmax router's topk scores sum to 1 per
    # token (constant), so a plain .sum() would have zero gradient by design.
    (replay_scores.float() ** 2).sum().backward()

    assert router.gate.weight.grad is not None
    assert torch.isfinite(router.gate.weight.grad).all()
    assert router.gate.weight.grad.abs().sum().item() > 0.0


def test_replay_backward_pops_target_in_order():
    from megatron.lite.primitive.modules.router import RouterReplay, RouterReplayAction

    router = _topk_router(True)
    router.eval()
    hidden = torch.randn(6, router.gate.in_features, device=_device())

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
    hidden = torch.randn(6, router.gate.in_features, device=_device())

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


def test_attach_skips_router_replay_excluded_modules():
    """DeepSeek-V4 hash-routed layers set ``_router_replay_exclude`` so they stay
    out of the replay registry (their routing is weight-independent)."""
    import torch.nn as nn

    from megatron.lite.primitive.modules.router import RouterReplay, attach_router_replay

    class _FakeRouter(nn.Module):
        def __init__(self, exclude=False):
            super().__init__()
            self.router_replay = None
            if exclude:
                self._router_replay_exclude = True

    model = nn.ModuleList([_FakeRouter(), _FakeRouter(exclude=True), _FakeRouter()])
    count = attach_router_replay(model)
    assert count == 2
    assert len(RouterReplay.global_router_replay_instances) == 2


@pytest.mark.parametrize("contiguous", [False, True], ids=["zigzag", "contiguous"])
def test_routed_experts_pack_unpack_round_trip(contiguous):
    """pack -> stack -> unpack recovers the routing (CP=1; pure tensor path,
    covers both the shared zigzag layout and DS4's contiguous layout)."""
    from megatron.lite.model.protocol_utils import pack_routed_experts, unpack_routed_experts
    from megatron.lite.primitive.parallel import ParallelState
    from megatron.lite.runtime.contracts import PackedBatch

    seq_lens = torch.tensor([3, 5], dtype=torch.int64)
    total = int(seq_lens.sum())
    model = SimpleNamespace(ps=ParallelState())  # cp1 tp1
    batch = PackedBatch(
        input_ids=torch.zeros(total, dtype=torch.long),
        labels=torch.zeros(total, dtype=torch.long),
        seq_lens=seq_lens,
    )
    num_layers, topk = 2, 2
    rows = [
        torch.randint(0, 8, (int(n), num_layers, topk), dtype=torch.long) for n in seq_lens
    ]
    routed = torch.nested.as_nested_tensor(rows, layout=torch.jagged)

    targets = pack_routed_experts(model, batch, routed, contiguous=contiguous)
    assert len(targets) == num_layers
    stacked = torch.stack(targets, dim=1)  # [tokens_padded, num_layers, topk]
    recovered = unpack_routed_experts(model, batch, stacked, contiguous=contiguous)

    for original, got in zip(rows, recovered.unbind(0), strict=True):
        assert torch.equal(original, got)
