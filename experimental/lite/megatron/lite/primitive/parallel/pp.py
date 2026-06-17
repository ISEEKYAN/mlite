# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Pipeline parallel layer layout.

The per-stage split is delegated to Megatron-core's canonical
``PipelineParallelLayerLayout`` — the same machinery Megatron / Megatron-Bridge use
for DeepSeek-style models with an arbitrary (non-divisible) decoder count plus MTP.

We build the canonical flatten unit sequence ``[E, t, t, …, (m), L]`` — embedding
first, loss last (both forced by ``validate_layer_layout``), MTP after the decoders
and before loss — and balance *all* of those units evenly across the ``pp``
stages (pp-only; VPP/interleaving not yet supported). This is the
``account_for_embedding_in_pipeline_split`` +
``account_for_loss_in_pipeline_split`` semantics: the stage holding the embedding,
and the stage holding loss (+ MTP), each take *one fewer decoder* per occupied slot
so every stage carries a balanced amount of compute (the first/last stages are not
overloaded with a full decoder share *plus* embedding/loss).

``PipelineParallelLayerLayout`` then owns the per-stage offset/id maths and
validates the result (embedding/loss exactly once, decoders == num_layers, MTP ==
mtp_num_layers and after all decoders). We do not use the
``account_for_*`` switches on ``transformer_block.get_num_layers_to_build`` directly
because that path still asserts ``(num_layers + accounted) % pp == 0`` and so cannot
place e.g. DeepSeek-V4's 43 decoders over pp8/pp2; the explicit layout has no such
divisibility requirement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from megatron.lite.primitive.parallel.state import ParallelState


@dataclass
class PipelineChunkLayout:
    layer_indices: list[int] = field(default_factory=list)
    has_embed: bool = False
    has_head: bool = False


def _build_layout(num_hidden_layers: int, pp_size: int, num_mtp_layers: int):
    """Auto-inferred balanced layout as a Megatron-core ``PipelineParallelLayerLayout``.

    From ``(num_layers, pp, mtp_num_layers)`` alone, lay the canonical unit sequence
    ``[embedding, decoder*num_layers, mtp*K, loss]`` (embedding first, loss last, MTP
    just before loss) into contiguous, as-even-as-possible chunks across the ``pp``
    stages, and hand it to ``PipelineParallelLayerLayout``. Because embedding / MTP /
    loss occupy slots, the stages holding them carry fewer decoders — Megatron's
    ``account_for_embedding/loss`` head/tail balancing — and the mcore class itself
    owns the offset / per-stage decoder counts and validates the result (it handles
    arbitrary, non-divisible 43/61/78 layer counts without a divisibility assert).
    """
    from megatron.core.transformer.pipeline_parallel_layer_layout import (
        PipelineParallelLayerLayout,
    )

    units = (
        ["embedding"]
        + ["decoder"] * num_hidden_layers
        + ["mtp"] * max(num_mtp_layers, 0)
        + ["loss"]
    )
    base, remainder = divmod(len(units), pp_size)
    sizes = [base + (1 if stage < remainder else 0) for stage in range(pp_size)]
    layout_list: list[list[str]] = []
    pos = 0
    for size in sizes:
        layout_list.append(units[pos : pos + size])
        pos += size

    layout = PipelineParallelLayerLayout(layout_list, pipeline_model_parallel_size=pp_size)
    # mcore validates: embedding/loss exactly once, decoders == num_layers,
    # mtp == mtp_num_layers and after all decoders.
    layout.validate_layer_layout(num_hidden_layers, num_mtp_layers or None)
    return layout


def build_pipeline_chunk_layout(
    num_hidden_layers: int,
    ps: ParallelState,
    vpp: int | None = None,
    vpp_chunk_id: int | None = None,
    *,
    num_mtp_layers: int = 0,
) -> PipelineChunkLayout:
    """Compute layer_indices, has_embed, has_head for this PP rank.

    The user sets only ``pp``; the layout is auto-inferred from
    ``(num_hidden_layers, pp, num_mtp_layers)`` via Megatron-core's
    ``PipelineParallelLayerLayout`` (see ``_build_layout``), so any decoder count —
    incl. DeepSeek-V4's 43, Kimi's 61, GLM-5's 78 — plus MTP is balanced across PP
    with the embedding/loss occupying the head/tail slots. ``get_layer_id_list``
    returns this stage's decoder ids straight from the mcore layout.

    VPP / interleaving is not supported yet (pp-only).
    """
    if (vpp is not None and vpp > 1) or vpp_chunk_id is not None:
        raise NotImplementedError(
            "VPP / interleaved pipeline layout is not supported yet; use vpp=1 (pp-only)."
        )

    # No pipeline: this stage owns every layer; embedding and head live here.
    if ps.pp_size <= 1:
        return PipelineChunkLayout(
            layer_indices=list(range(num_hidden_layers)), has_embed=True, has_head=True
        )

    from megatron.core.transformer.enums import LayerType

    layout = _build_layout(num_hidden_layers, ps.pp_size, num_mtp_layers)
    layer_indices = layout.get_layer_id_list(
        layer_type=LayerType.decoder, vp_stage=0, pp_rank=ps.pp_rank
    )
    return PipelineChunkLayout(
        layer_indices=layer_indices, has_embed=ps.pp_is_first, has_head=ps.pp_is_last
    )


__all__ = ["PipelineChunkLayout", "build_pipeline_chunk_layout"]
