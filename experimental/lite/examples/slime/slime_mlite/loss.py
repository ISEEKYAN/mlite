# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Loss functions for the Megatron Lite slime backend.

Megatron Lite's ``forward_backward`` calls ``loss_fn(model_output, batch)`` per
micro-batch, where ``model_output["log_probs"]`` is the per-token log-probability
of each label (i.e. ``-cross_entropy``), packed as ``[1, total_tokens]``. The
runtime divides each micro-batch loss by ``num_microbatches`` before backward,
so a loss_fn returns the per-micro-batch mean directly.
"""

from __future__ import annotations

from typing import Any, Callable

import torch


def make_sft_loss_fn() -> Callable[[dict[str, Any], dict[str, Any]], tuple[torch.Tensor, dict]]:
    """Supervised fine-tuning loss: mean negative log-likelihood over masked tokens."""

    def _sft_loss(model_output: dict[str, Any], batch: dict[str, Any]) -> tuple[torch.Tensor, dict]:
        log_probs = model_output["log_probs"]
        if log_probs is None:
            raise ValueError("Megatron Lite model output must contain per-token log_probs for SFT.")
        nll = -log_probs.reshape(-1)

        loss_mask = batch.get("loss_mask")
        if loss_mask is not None:
            mask = loss_mask.reshape(-1).to(dtype=nll.dtype)
            denom = mask.sum().clamp_min(1.0)
            loss = (nll * mask).sum() / denom
        else:
            loss = nll.mean()

        return loss, {"loss": loss.detach()}

    return _sft_loss
