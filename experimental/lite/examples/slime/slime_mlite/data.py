# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Bridge slime rollout data into Megatron Lite THD micro-batches.

slime hands the train actor a per-DP-rank shard of samples (full prompt+response
token streams plus a response-only loss mask). Megatron Lite consumes packed
THD micro-batches (1-D concatenated tokens with ``PackedSeqParams``), so this
module groups the shard into micro-batches and packs each one with Megatron
Lite's own ``pack_nested_thd`` primitive (no reinvented packing/CP logic).

Loss-mask convention matches slime's megatron backend (see
``slime/backends/megatron_utils/data.py:get_batch``): the response-only mask is
left-padded by ``prompt_length - 1`` and right-padded by 1 so that mask position
``i`` selects the token predicted at position ``i`` (i.e. ``token[i+1]``). Labels
are produced by Megatron Lite's ``roll_labels=True`` (shift input_ids left), so
the already-aligned mask is passed with ``roll_loss_mask=False``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from megatron.lite.primitive.parallel import pack_nested_thd


def _group_microbatches(total_lengths: list[int], *, micro_batch_size: int,
                        use_dynamic_batch_size: bool, max_tokens_per_gpu: int) -> list[list[int]]:
    """Return sample-index groups, one per micro-batch."""
    num_samples = len(total_lengths)
    if num_samples == 0:
        return []

    if not use_dynamic_batch_size:
        mbs = max(int(micro_batch_size), 1)
        return [list(range(i, min(i + mbs, num_samples))) for i in range(0, num_samples, mbs)]

    budget = max(int(max_tokens_per_gpu), 1)
    groups: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0
    for idx, length in enumerate(total_lengths):
        length = int(length)
        if current and current_tokens + length > budget:
            groups.append(current)
            current, current_tokens = [], 0
        current.append(idx)
        current_tokens += length
    if current:
        groups.append(current)
    return groups


def _aligned_loss_mask(loss_mask: torch.Tensor, total_length: int, response_length: int) -> torch.Tensor:
    """Left/right pad a response-only mask to the full token stream (next-token aligned)."""
    prompt_length = total_length - response_length
    left_pad = max(prompt_length - 1, 0)
    right_pad = total_length - response_length - left_pad
    return F.pad(loss_mask.float(), (left_pad, right_pad), value=0.0)


def build_sft_microbatches(
    rollout_data: dict[str, Any],
    *,
    micro_batch_size: int,
    use_dynamic_batch_size: bool,
    max_tokens_per_gpu: int,
    tp_size: int,
    cp_size: int,
    cp_rank: int,
    cp_group: Any | None,
    temperature: float = 1.0,
    use_fused_kernels: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    """Build packed THD micro-batch dicts for ``runtime.forward_backward``.

    Returns ``(microbatches, num_microbatches)``.
    """
    tokens = rollout_data["tokens"]
    loss_masks = rollout_data["loss_masks"]
    total_lengths = [int(x) for x in rollout_data["total_lengths"]]
    response_lengths = [int(x) for x in rollout_data["response_lengths"]]

    groups = _group_microbatches(
        total_lengths,
        micro_batch_size=micro_batch_size,
        use_dynamic_batch_size=use_dynamic_batch_size,
        max_tokens_per_gpu=max_tokens_per_gpu,
    )

    microbatches: list[dict[str, Any]] = []
    for group in groups:
        seq_tokens = [tokens[i].reshape(-1).long() for i in group]
        seq_masks = [
            _aligned_loss_mask(loss_masks[i].reshape(-1), total_lengths[i], response_lengths[i])
            for i in group
        ]
        nested_tokens = torch.nested.as_nested_tensor(seq_tokens, layout=torch.jagged)
        nested_masks = torch.nested.as_nested_tensor(seq_masks, layout=torch.jagged)

        packed = pack_nested_thd(
            nested_tokens,
            tp_size=tp_size,
            cp_size=cp_size,
            cp_rank=cp_rank,
            cp_group=cp_group,
            labels=nested_tokens,
            roll_labels=True,
            loss_mask=nested_masks,
            roll_loss_mask=False,
        )
        microbatches.append(
            {
                "input_ids": packed.input_ids,
                "labels": packed.labels,
                "position_ids": packed.position_ids,
                "packed_seq_params": packed.packed_seq_params,
                "loss_mask": packed.loss_mask,
                "temperature": temperature,
                "use_fused_kernels": use_fused_kernels,
                "calculate_entropy": False,
            }
        )

    return microbatches, len(microbatches)
