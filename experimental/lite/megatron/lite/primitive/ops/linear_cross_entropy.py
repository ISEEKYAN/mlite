# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Linear + vocab-parallel cross entropy helpers.

This primitive is intentionally self-contained: application connectors may use
it, but it must not import connector packages or kernels from those packages.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from megatron.lite.primitive.ops.cross_entropy import vocab_parallel_cross_entropy


def _all_reduce_if_needed(tensor: torch.Tensor, group, op=dist.ReduceOp.SUM) -> torch.Tensor:
    if group is not None and dist.is_initialized() and dist.get_world_size(group) > 1:
        dist.all_reduce(tensor, op=op, group=group)
    return tensor


def _vocab_parallel_entropy(logits: torch.Tensor, tp_group=None) -> torch.Tensor:
    logits = logits.float()
    logits_max = logits.max(dim=-1).values
    _all_reduce_if_needed(logits_max, tp_group, op=dist.ReduceOp.MAX)

    shifted = logits - logits_max.unsqueeze(-1)
    exp_logits = torch.exp(shifted)
    sum_exp = exp_logits.sum(dim=-1)
    _all_reduce_if_needed(sum_exp, tp_group)

    weighted_logits = (exp_logits * logits).sum(dim=-1)
    _all_reduce_if_needed(weighted_logits, tp_group)
    expected_logits = weighted_logits / sum_exp
    return torch.log(sum_exp) + logits_max - expected_logits


def _reshape_like_labels(values: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if values.shape != labels.shape and values.numel() == labels.numel():
        return values.reshape(labels.shape)
    return values


def linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    tp_group=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return token log-probs and entropy from local primitive ops."""
    logits = torch.matmul(hidden, weight.t())
    if temperature != 1.0:
        logits = logits / float(temperature)
    loss = vocab_parallel_cross_entropy(logits.clone(), labels, tp_group)
    entropy = _vocab_parallel_entropy(logits, tp_group)
    return _reshape_like_labels(-loss, labels), _reshape_like_labels(entropy, labels)


__all__ = ["linear_cross_entropy"]
