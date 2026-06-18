# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DeepSeek-V4 must scale the MTP auxiliary-loss gradient per step.

DS4 injects the MTP auxiliary loss via ``MTPLossAutoScaler.apply`` (model.py),
but ``MTPLossAutoScaler.main_loss_backward_scale`` only tracks the main-loss
scale (DP size / gradient accumulation) if some per-step hook calls
``set_loss_scale``. Every sibling protocol (kimi_k2 / glm5 / qwen3_5 /
qwen3_moe) registers ``_make_aux_loss_hook`` as ``extras["pre_forward_hook"]``;
DS4 used to be the only one missing it, so its MTP gradient stayed at the
class-default scale of 1.0 and was mis-weighted vs the main loss. The runtime
calls ``extras["pre_forward_hook"]`` each step
(runtime/backends/mlite/runtime.py).

This regression is train-time only (invisible to forward/eval), so it must be
guarded by an explicit wiring test, not a forward-parity check.

- ``test_ds4_aux_loss_hook_sets_mtp_scale`` is a pure-CPU unit test of the hook
  itself (always runs).
- ``test_ds4_build_wires_pre_forward_hook`` builds the model and asserts the
  bundle actually carries a working ``pre_forward_hook`` (GPU + a DS4 HF dir;
  skips otherwise).

Pre-fix, the first test fails at import (``_make_aux_loss_hook`` did not exist)
and the second fails the ``callable(hook)`` assert (extras had no
``pre_forward_hook``).
"""
from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.mlite


def test_ds4_aux_loss_hook_sets_mtp_scale():
    """The DS4 aux-loss hook must push the per-step scale into MTPLossAutoScaler."""
    from megatron.lite.model.deepseek_v4.lite.protocol import _make_aux_loss_hook
    from megatron.lite.primitive.modules.mtp import MTPLossAutoScaler

    saved = MTPLossAutoScaler.main_loss_backward_scale
    try:
        # Start at the (buggy) default so a no-op hook would leave it unchanged.
        MTPLossAutoScaler.main_loss_backward_scale = 1.0
        hook = _make_aux_loss_hook()
        assert callable(hook)

        hook(torch.tensor(0.25))
        assert MTPLossAutoScaler.main_loss_backward_scale == pytest.approx(0.25)

        # also accepts a plain float (set_loss_scale does float(scale))
        hook(0.5)
        assert MTPLossAutoScaler.main_loss_backward_scale == pytest.approx(0.5)
    finally:
        MTPLossAutoScaler.main_loss_backward_scale = saved


def test_ds4_build_wires_pre_forward_hook():
    """build_model must expose a working pre_forward_hook in the bundle extras.

    Reuses the bench's create_runtime build path. Needs CUDA + a DS4 HF dir
    (set ``DS4_TOY`` or have the default toy mounted); skips otherwise. Weights
    are not loaded -- only the structure/wiring is under test.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required to build the DS4 model.")
    hf_path = os.environ.get("DS4_TOY", "/home/scratch.lmei_other/dsv4-flash-toy")
    if not os.path.isdir(hf_path):
        pytest.skip(f"No DS4 HF dir at {hf_path} (set DS4_TOY).")

    from megatron.lite.primitive.modules.mtp import MTPLossAutoScaler
    from megatron.lite.runtime import RuntimeConfig, create_runtime
    from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
    from megatron.lite.runtime.contracts.config import OptimizerConfig, ParallelConfig

    bcfg = MegatronLiteConfig(
        model_name="deepseek_v4",
        impl="lite",
        hf_path=hf_path,
        parallel=ParallelConfig(tp=1, etp=1, ep=1, pp=1, cp=1),
        optimizer=OptimizerConfig(lr=0.0, min_lr=0.0, weight_decay=0.0),
        load_hf_weights=False,
        impl_cfg={"optimizer": "fsdp2", "deterministic": True},
    )
    rt = create_runtime(RuntimeConfig(backend="mlite", hf_path=hf_path, backend_cfg=bcfg))
    handle = rt.build_model()

    hook = handle._extras.get("pre_forward_hook")
    assert callable(hook), "DS4 bundle is missing a callable pre_forward_hook"

    saved = MTPLossAutoScaler.main_loss_backward_scale
    try:
        MTPLossAutoScaler.main_loss_backward_scale = 1.0
        hook(torch.tensor(0.3))
        assert MTPLossAutoScaler.main_loss_backward_scale == pytest.approx(0.3)
    finally:
        MTPLossAutoScaler.main_loss_backward_scale = saved
