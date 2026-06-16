# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Context/pipeline parallel sequence splitting utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from megatron.lite.primitive.utils import ensure_divisible

if TYPE_CHECKING:
    from megatron.lite.primitive.parallel.state import ParallelState


@dataclass
class PipelineChunkLayout:
    layer_indices: list[int] = field(default_factory=list)
    has_embed: bool = False
    has_head: bool = False


def auto_pipeline_layer_counts(
    num_hidden_layers: int,
    pp_size: int,
    *,
    extra_first: int = 0,
    extra_last: int = 0,
) -> list[int]:
    """Megatron-style auto pipeline layout: split ``num_hidden_layers`` transformer
    layers across ``pp_size`` stages so the per-stage cost is balanced, even when the
    layer count is *not* divisible by ``pp_size``.

    ``extra_first`` / ``extra_last`` are pseudo-layers that model the extra cost the
    first/last stage carries on top of the transformer layers (embedding on the first
    stage; final norm + LM head + MTP layers on the last stage). They only steer the
    balancing — they are not real transformer layers and are never emitted in the
    returned counts. Accounting for them keeps the first/last stages from being
    overloaded once embedding/head/MTP work is added back in.

    Layer order is preserved: stage ``s`` always owns a contiguous range, so the
    global layer index of every transformer layer is stable and HF/distckpt weight
    mapping is unaffected by the split.
    """
    assert pp_size >= 1
    if pp_size == 1:
        return [num_hidden_layers]

    # Lay out a virtual sequence of [embed pseudo | real layers | head/MTP pseudo]
    # units and split it into ``pp_size`` contiguous chunks as evenly as possible
    # (remainder to the earlier stages). A real layer lands on the stage whose chunk
    # covers its virtual position; the pseudo units only shift those boundaries.
    total = num_hidden_layers + extra_first + extra_last
    base, rem = divmod(total, pp_size)
    chunk_ends: list[int] = []
    acc = 0
    for i in range(pp_size):
        acc += base + (1 if i < rem else 0)
        chunk_ends.append(acc)

    counts = [0] * pp_size
    stage = 0
    for layer in range(num_hidden_layers):
        vpos = extra_first + layer
        while vpos >= chunk_ends[stage]:
            stage += 1
        counts[stage] += 1
    return counts


def build_pipeline_chunk_layout(
    num_hidden_layers: int,
    ps: ParallelState,
    vpp: int | None = None,
    vpp_chunk_id: int | None = None,
    *,
    num_mtp_layers: int = 0,
) -> PipelineChunkLayout:
    """Compute layer_indices, has_embed, has_head for this PP rank / VPP chunk.

    When ``num_hidden_layers`` is divisible by the pipeline width the layout is the
    plain even split (unchanged behaviour). Otherwise — and only for the non-interleaved
    path (``vpp`` unset or 1) — the layers are auto-balanced via
    :func:`auto_pipeline_layer_counts` so a non-divisible count no longer raises and
    does not require hand-tuning TP/PP. ``num_mtp_layers`` (the multi-token-prediction
    layers that always live on the last stage) is folded into the balancing so the last
    stage is not overloaded.

    Interleaved VPP (``vpp > 1``) still requires divisibility by ``pp_size * vpp``;
    balanced interleaving is out of scope.
    """
    if vpp is not None and vpp > 1:
        if vpp_chunk_id is not None:
            layers_per_chunk = ensure_divisible(num_hidden_layers, ps.pp_size * vpp)
            start = ps.pp_rank * layers_per_chunk + vpp_chunk_id * (ps.pp_size * layers_per_chunk)
            layer_indices = list(range(start, start + layers_per_chunk))
            has_embed = ps.pp_is_first and vpp_chunk_id == 0
            has_head = ps.pp_is_last and vpp_chunk_id == vpp - 1
        else:
            layers_per_chunk = ensure_divisible(num_hidden_layers, ps.pp_size * vpp)
            layer_indices = []
            for chunk in range(vpp):
                start = ps.pp_rank * layers_per_chunk + chunk * (ps.pp_size * layers_per_chunk)
                layer_indices.extend(range(start, start + layers_per_chunk))
            has_embed = ps.pp_is_first
            has_head = ps.pp_is_last
        return PipelineChunkLayout(
            layer_indices=layer_indices, has_embed=has_embed, has_head=has_head
        )

    # Non-interleaved path (vpp is None or 1).
    if num_hidden_layers % ps.pp_size == 0:
        layers_per_stage = num_hidden_layers // ps.pp_size
        start = ps.pp_rank * layers_per_stage
        layer_indices = list(range(start, start + layers_per_stage))
    else:
        # Auto layout for non-divisible counts: account for embedding on the first
        # stage and final norm + head + MTP layers on the last stage.
        counts = auto_pipeline_layer_counts(
            num_hidden_layers,
            ps.pp_size,
            extra_first=1,
            extra_last=1 + max(0, num_mtp_layers),
        )
        start = sum(counts[: ps.pp_rank])
        layer_indices = list(range(start, start + counts[ps.pp_rank]))
    has_embed = ps.pp_is_first
    has_head = ps.pp_is_last
    return PipelineChunkLayout(layer_indices=layer_indices, has_embed=has_embed, has_head=has_head)


__all__ = ["PipelineChunkLayout", "auto_pipeline_layer_counts", "build_pipeline_chunk_layout"]
