# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Helpers shared by model protocol forward steps.

The verl/runtime layers hand each protocol a raw, model-agnostic ``PackedBatch``
(true per-sequence lengths, no padding, no ``PackedSeqParams``). Each model owns
its pack/unpack pair: ``pack_thd_forward_kwargs`` pads + CP-splits the batch into
model forward kwargs, and ``unpack_thd_forward_output`` reverses a model output
back to jagged true-length form. THD models share the zigzag-CP pair below;
models with a different CP layout (e.g. DeepSeek-V4 contiguous DSA) provide their
own pair.
"""

from __future__ import annotations

from typing import Any

import torch
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.parallel.thd import (
    pack_nested_thd,
    parallel_state_from_model,
    prepare_packed_thd_kwargs_for_context_parallel,
    split_packed_to_cp_local,
    thd_pack_meta,
    unpack_thd_to_nested,
)
from megatron.lite.primitive.utils.packed_seq import PackedSeqParams
from megatron.lite.runtime.contracts.data import PackedBatch
from megatron.lite.runtime.contracts.loss import get_loss_context


def _parallel_state(model) -> ParallelState:
    return parallel_state_from_model(model) or ParallelState()


def nested_from_packed(tensor: torch.Tensor | None, seq_lens: torch.Tensor):
    """Split a 1-D packed (true, unpadded) tensor back into a jagged nested tensor."""
    if tensor is None:
        return None
    if tensor.dim() == 2 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)
    if tensor.dim() != 1:
        raise ValueError(f"PackedBatch tensor must be 1-D, got {tuple(tensor.shape)}.")
    pieces = []
    offset = 0
    for length_t in seq_lens:
        length = int(length_t.item())
        pieces.append(tensor.narrow(0, offset, length))
        offset += length
    if offset != tensor.numel():
        raise ValueError(f"PackedBatch sizes sum to {offset}, tensor has {tensor.numel()} tokens.")
    return torch.nested.as_nested_tensor(pieces, layout=torch.jagged)


def pack_thd_forward_kwargs(model, batch: PackedBatch) -> dict[str, Any]:
    """Pad + zigzag-CP-split a raw THD batch into model forward kwargs.

    Pads each sequence to the TE/zigzag alignment, then CP-splits tokens,
    labels, masks and position ids through the shared THD primitive — the same
    layout the model was validated against, now produced inside the protocol
    rather than the connector.
    """
    ps = _parallel_state(model)
    seq_lens = batch.seq_lens
    packed = pack_nested_thd(
        nested_from_packed(batch.input_ids, seq_lens),
        tp_size=ps.tp_size,
        cp_size=ps.cp_size,
        cp_rank=ps.cp_rank,
        cp_group=ps.cp_group if ps.cp_size > 1 else None,
        split_cp=False,
        labels=nested_from_packed(batch.labels, seq_lens),
        roll_labels=batch.labels is not None,
        loss_mask=nested_from_packed(batch.loss_mask, seq_lens),
        roll_loss_mask=batch.loss_mask is not None,
    )
    max_seqlen = int(packed.padded_lengths.max().item()) if packed.padded_lengths.numel() else 0
    # pack_nested_thd already returns [1, T] token rows; do not unsqueeze again.
    kwargs: dict[str, Any] = {
        "input_ids": packed.input_ids,
        "labels": packed.labels,
        "loss_mask": packed.loss_mask,
        "position_ids": packed.position_ids,
        "packed_seq_params": PackedSeqParams.from_cu_seqlens(
            packed.cu_seqlens_padded, max_seqlen=max_seqlen
        ),
    }
    prepare_packed_thd_kwargs_for_context_parallel(model, kwargs)
    return kwargs


def unpack_thd_forward_output(model, batch: PackedBatch, output: torch.Tensor) -> torch.Tensor:
    """Reverse a zigzag-CP THD model output back to jagged true-length form."""
    ps = _parallel_state(model)
    meta = thd_pack_meta(
        batch.seq_lens,
        tp_size=ps.tp_size,
        cp_size=ps.cp_size,
        cp_group=ps.cp_group if ps.cp_size > 1 else None,
    )
    return unpack_thd_to_nested(output, meta, contiguous=False)


def pack_routed_experts(
    model, batch: PackedBatch, routed_experts, *, contiguous: bool = False
) -> list[torch.Tensor]:
    """Split a full routing tensor into one CP rank's per-layer replay targets.

    ``routed_experts`` is jagged ``[bs, seq, num_layers, topk]`` (true lengths,
    matching ``batch`` tokens). Pads each sequence to the same THD layout the
    model uses for tokens, CP-splits (zigzag by default), and returns a list of
    ``num_layers`` integer tensors shaped ``[local_padded_tokens, topk]`` — the
    per-layer targets the runtime feeds to ``RouterReplay.set_replay_data``.
    """
    ps = _parallel_state(model)
    meta = thd_pack_meta(
        batch.seq_lens,
        tp_size=ps.tp_size,
        cp_size=ps.cp_size,
        cp_group=ps.cp_group if ps.cp_size > 1 else None,
        contiguous=contiguous,
    )
    rows = (
        list(routed_experts.unbind(0))
        if getattr(routed_experts, "is_nested", False)
        else [routed_experts[i] for i in range(routed_experts.size(0))]
    )
    if len(rows) != int(meta.lengths.numel()):
        raise ValueError(
            f"routed_experts has {len(rows)} sequences, batch has {int(meta.lengths.numel())}."
        )
    first = rows[0]
    if first.dim() != 3:
        raise ValueError(
            f"routed_experts rows must be [seq, num_layers, topk], got {tuple(first.shape)}."
        )
    num_layers, topk = int(first.size(1)), int(first.size(2))
    total_padded = int(meta.cu_seqlens_padded[-1].item())
    device = batch.input_ids.device
    full = torch.zeros(total_padded, num_layers, topk, dtype=torch.long, device=device)
    for idx, row in enumerate(rows):
        length = int(meta.lengths[idx].item())
        if int(row.size(0)) != length:
            raise ValueError(
                f"routed_experts seq {idx} has {row.size(0)} tokens, expected {length}."
            )
        start = int(meta.cu_seqlens_padded[idx].item())
        full[start : start + length] = row.to(device=device, dtype=torch.long)
    local = split_packed_to_cp_local(
        full,
        cu_seqlens_padded=meta.cu_seqlens_padded,
        cp_size=ps.cp_size,
        cp_rank=ps.cp_rank,
        dim=0,
    )
    return [local[:, layer, :].contiguous() for layer in range(num_layers)]


def unpack_routed_experts(model, batch: PackedBatch, recorded, *, contiguous: bool = False):
    """Reverse recorded per-layer routing back to jagged ``[bs, seq, layers, topk]``.

    ``recorded`` is this rank's stacked routing ``[local_padded_tokens, num_layers,
    topk]``; CP-gathers and strips padding via the shared THD unpack (mirrors
    ``unpack_thd_forward_output``).
    """
    ps = _parallel_state(model)
    num_layers, topk = int(recorded.size(1)), int(recorded.size(2))
    meta = thd_pack_meta(
        batch.seq_lens,
        tp_size=ps.tp_size,
        cp_size=ps.cp_size,
        cp_group=ps.cp_group if ps.cp_size > 1 else None,
        contiguous=contiguous,
    )
    # Flatten (layers, topk) into one feature dim so unpack_thd_to_nested's
    # singleton-squeeze heuristic can't drop a 1-layer model's layer axis, then
    # restore [seq, layers, topk] per sequence.
    flat = recorded.reshape(recorded.size(0), num_layers * topk)
    nested = unpack_thd_to_nested(flat, meta, contiguous=contiguous)
    rows = [row.reshape(row.size(0), num_layers, topk) for row in nested.unbind(0)]
    return torch.nested.as_nested_tensor(rows, layout=torch.jagged)


def add_loss_context_kwargs(kwargs: dict[str, Any], *, include_return_log_probs: bool = False) -> None:
    loss_context = get_loss_context()
    if loss_context is None:
        return
    kwargs["temperature"] = loss_context.temperature
    kwargs["calculate_entropy"] = loss_context.calculate_entropy
    if include_return_log_probs:
        kwargs["return_log_probs"] = loss_context.return_log_probs


def add_cross_entropy_fusion(kwargs: dict[str, Any], model) -> None:
    kwargs["use_fused_kernels"] = bool(getattr(model, "cross_entropy_fusion", False))


def set_cross_entropy_fusion(chunks: list, enabled: bool) -> None:
    for chunk in chunks:
        chunk.cross_entropy_fusion = bool(enabled)


__all__ = [
    "add_cross_entropy_fusion",
    "add_loss_context_kwargs",
    "nested_from_packed",
    "pack_routed_experts",
    "pack_thd_forward_kwargs",
    "set_cross_entropy_fusion",
    "unpack_routed_experts",
    "unpack_thd_forward_output",
]
