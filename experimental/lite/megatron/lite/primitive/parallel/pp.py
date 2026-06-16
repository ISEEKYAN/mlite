# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Pipeline parallel layer layout.

The per-stage layer split is delegated to Megatron-core's own pipeline layout
machinery — we do not implement a custom distribution algorithm. A non-divisible
layer count is auto-balanced by turning on Megatron's
``account_for_embedding_in_pipeline_split`` / ``account_for_loss_in_pipeline_split``
switches, which fold the embedding (first stage) and the final norm + loss/head
(last stage) into the balance exactly as Megatron does for uneven pipelines.
``megatron.core.transformer.{transformer_block.get_num_layers_to_build,
transformer_layer.get_transformer_layer_offset}`` then return this stage's layer
count and offset; we only translate that into mlite's ``PipelineChunkLayout``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from megatron.lite.primitive.parallel.state import ParallelState


@dataclass
class PipelineChunkLayout:
    layer_indices: list[int] = field(default_factory=list)
    has_embed: bool = False
    has_head: bool = False


def _mcore_layout_config(num_hidden_layers: int, pp_size: int, vpp: int | None) -> SimpleNamespace:
    """Minimal TransformerConfig-shaped object carrying only the fields Megatron's
    ``get_num_layers_to_build`` / ``get_transformer_layer_offset`` read.

    When the layer count is not divisible by the pipeline width we enable
    Megatron's embedding/loss accounting so it auto-balances the split; a divisible
    count keeps both switches off and yields Megatron's plain even split (unchanged).
    """
    non_divisible = num_hidden_layers % pp_size != 0
    return SimpleNamespace(
        num_layers=num_hidden_layers,
        pipeline_model_parallel_size=pp_size,
        virtual_pipeline_model_parallel_size=(vpp if vpp and vpp > 1 else None),
        num_layers_in_first_pipeline_stage=None,
        num_layers_in_last_pipeline_stage=None,
        account_for_embedding_in_pipeline_split=non_divisible,
        account_for_loss_in_pipeline_split=non_divisible,
        pipeline_model_parallel_layout=None,
    )


def _mcore_decoder_indices(cfg: SimpleNamespace, pp_rank: int, vp_stage: int | None) -> list[int]:
    """Decoder layer ids owned by (pp_rank, vp_stage), straight from Megatron-core."""
    from megatron.core.transformer.transformer_block import get_num_layers_to_build
    from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

    offset = get_transformer_layer_offset(cfg, vp_stage=vp_stage, pp_rank=pp_rank)
    count = get_num_layers_to_build(cfg, vp_stage=vp_stage, pp_rank=pp_rank)
    return list(range(offset, offset + count))


def build_pipeline_chunk_layout(
    num_hidden_layers: int,
    ps: ParallelState,
    vpp: int | None = None,
    vpp_chunk_id: int | None = None,
    *,
    num_mtp_layers: int = 0,
) -> PipelineChunkLayout:
    """Compute layer_indices, has_embed, has_head for this PP rank / VPP chunk.

    Wires to Megatron-core's layer layout (see module docstring); a non-divisible
    ``num_hidden_layers`` is auto-balanced via Megatron's embedding/loss accounting
    rather than raising. ``num_mtp_layers`` is accepted for API symmetry with the
    model builders: MTP heads are appended on the last stage by the model itself
    (Megatron likewise restricts MTP to the last stage), so they are not part of the
    decoder split laid out here.
    """
    del num_mtp_layers

    # No pipeline: this stage owns every layer; embedding and head live here.
    if ps.pp_size <= 1:
        return PipelineChunkLayout(
            layer_indices=list(range(num_hidden_layers)), has_embed=True, has_head=True
        )

    cfg = _mcore_layout_config(num_hidden_layers, ps.pp_size, vpp)
    interleaved = vpp is not None and vpp > 1

    if interleaved and vpp_chunk_id is None:
        # Build every virtual chunk owned by this PP rank, in global layer order.
        layer_indices: list[int] = []
        for chunk in range(vpp):
            layer_indices.extend(_mcore_decoder_indices(cfg, ps.pp_rank, chunk))
        has_embed = ps.pp_is_first
        has_head = ps.pp_is_last
    else:
        vp_stage = vpp_chunk_id if interleaved else None
        layer_indices = _mcore_decoder_indices(cfg, ps.pp_rank, vp_stage)
        if interleaved and vpp_chunk_id is not None:
            has_embed = ps.pp_is_first and vpp_chunk_id == 0
            has_head = ps.pp_is_last and vpp_chunk_id == vpp - 1
        else:
            has_embed = ps.pp_is_first
            has_head = ps.pp_is_last

    return PipelineChunkLayout(layer_indices=layer_indices, has_embed=has_embed, has_head=has_head)


__all__ = ["PipelineChunkLayout", "build_pipeline_chunk_layout"]
