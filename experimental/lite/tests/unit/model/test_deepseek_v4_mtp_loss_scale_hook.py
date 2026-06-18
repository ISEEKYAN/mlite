# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DeepSeek-V4 must register a ``pre_forward_hook`` that scales the multi-token
prediction (MTP) auxiliary loss by ``1/num_microbatches``.

The runtime divides the main loss by ``num_microbatches`` before backward, but
the MTP loss is threaded into the graph through ``MTPLossAutoScaler`` whose
backward uses a class-level ``main_loss_backward_scale`` (default ``1.0``). The
hook is the only thing that sets that scale to ``1/num_microbatches`` per
microbatch (matching Megatron-core ``pipeline_parallel/schedules.forward_step``).

Without the hook the scale stays ``1.0``: with ``num_microbatches == 1`` that
happens to be correct, but under gradient accumulation (``num_microbatches > 1``)
the MTP gradient is amplified ``num_microbatches``-fold relative to the main
loss. DeepSeek-V4's only active auxiliary loss is MTP (the MoE router is
aux-loss-free and the DSA indexer KL loss is disabled), so the hook only scales
the MTP scaler.

This is a CPU unit test: it stubs Transformer Engine and exercises the scaler
arithmetic directly, with no GPU / ``build_model``.
"""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.mlite


def _accumulated_mtp_grad(num_microbatches: int, *, hook) -> torch.Tensor:
    """Mimic ``run_microbatch_loop`` for the MTP path only.

    A shared parameter ``w`` feeds an MTP loss that is injected through
    ``MTPLossAutoScaler``; the main loss is set to zero so the returned gradient
    is purely the MTP-injected contribution. Per microbatch the loop divides the
    forwarded output by ``num_microbatches`` before backward, exactly like the
    real runtime. When ``hook`` is provided it is called with
    ``1/num_microbatches`` before each microbatch (as the runtime does).
    """
    from megatron.lite.primitive.modules.mtp import MTPLossAutoScaler

    MTPLossAutoScaler.main_loss_backward_scale = 1.0  # default, as at process start
    w = torch.ones(3, requires_grad=True)
    x_mtp = torch.tensor([1.0, 2.0, 3.0])
    for _ in range(num_microbatches):
        if hook is not None:
            hook(torch.tensor(1.0 / num_microbatches))
        main_loss = (w * 0.0).sum()  # zero main loss -> isolates the MTP grad
        mtp_loss = (w * x_mtp).sum()
        out = MTPLossAutoScaler.apply(main_loss, mtp_loss)
        (out / num_microbatches).backward()
    grad = w.grad.detach().clone()
    MTPLossAutoScaler.main_loss_backward_scale = 1.0  # leave global pristine
    return grad


def test_ds4_hook_sets_mtp_scale(transformer_engine_import_stub):
    """The DeepSeek-V4 hook sets the MTP backward scale to the value it is given."""
    transformer_engine_import_stub()
    from megatron.lite.model.deepseek_v4.lite.protocol import _make_aux_loss_hook
    from megatron.lite.primitive.modules.mtp import MTPLossAutoScaler

    hook = _make_aux_loss_hook()
    try:
        hook(torch.tensor(0.25))
        assert MTPLossAutoScaler.main_loss_backward_scale == pytest.approx(0.25)
    finally:
        MTPLossAutoScaler.main_loss_backward_scale = 1.0


def test_ds4_hook_keeps_mtp_grad_invariant_to_num_microbatches(transformer_engine_import_stub):
    """With the hook, the accumulated MTP gradient is independent of nmb;
    without it, nmb>1 amplifies the MTP gradient nmb-fold (the regression)."""
    transformer_engine_import_stub()
    from megatron.lite.model.deepseek_v4.lite.protocol import _make_aux_loss_hook

    hook = _make_aux_loss_hook()

    grad_nmb1 = _accumulated_mtp_grad(1, hook=hook)
    grad_nmb4 = _accumulated_mtp_grad(4, hook=hook)
    # Correct behaviour: gradient is invariant to the microbatch count.
    torch.testing.assert_close(grad_nmb4, grad_nmb1)

    # Document the bug the hook fixes: no hook -> scale stuck at 1.0 -> the
    # nmb=4 MTP gradient is exactly 4x the nmb=1 one.
    buggy_nmb1 = _accumulated_mtp_grad(1, hook=None)
    buggy_nmb4 = _accumulated_mtp_grad(4, hook=None)
    torch.testing.assert_close(buggy_nmb1, grad_nmb1)  # nmb=1 is fine either way
    torch.testing.assert_close(buggy_nmb4, 4.0 * grad_nmb1)


def test_ds4_build_model_wires_pre_forward_hook(transformer_engine_import_stub):
    """The protocol source registers the hook in the ModelBundle extras so the
    runtime (runtime.py / train_step / pipeline) can pick it up."""
    transformer_engine_import_stub()
    import inspect

    from megatron.lite.model.deepseek_v4.lite import protocol

    src = inspect.getsource(protocol.build_model)
    assert '"pre_forward_hook": _make_aux_loss_hook()' in src
