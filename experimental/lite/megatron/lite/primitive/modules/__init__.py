"""Shared model modules owned by Megatron Lite.

Exports stay lazy so importing one lightweight primitive does not eagerly import
optional heavyweight dependencies used by another primitive.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from megatron.lite.primitive.modules.dispatcher import TokenDispatcher
    from megatron.lite.primitive.modules.experts import Experts
    from megatron.lite.primitive.modules.gated_delta_net import GatedDeltaNet
    from megatron.lite.primitive.modules.gqa import GQAttention, split_grouped_qkvg
    from megatron.lite.primitive.modules.mla_dsa import (
        DSAIndexer,
        MLADSA,
        RMSNorm,
        build_rotary_embeddings,
    )
    from megatron.lite.primitive.modules.moe import MoEAuxLossAutoScaler, _AllToAll
    from megatron.lite.primitive.modules.mrope import MultimodalRotaryEmbedding
    from megatron.lite.primitive.modules.mtp import MTPBlock, MTPDecoderLayer, MTPLossAutoScaler
    from megatron.lite.primitive.modules.router import SigmoidTopKRouter, TopKRouter

_LAZY_EXPORTS = {
    "DSAIndexer": "megatron.lite.primitive.modules.mla_dsa",
    "Experts": "megatron.lite.primitive.modules.experts",
    "GatedDeltaNet": "megatron.lite.primitive.modules.gated_delta_net",
    "GQAttention": "megatron.lite.primitive.modules.gqa",
    "MLADSA": "megatron.lite.primitive.modules.mla_dsa",
    "MTPBlock": "megatron.lite.primitive.modules.mtp",
    "MTPDecoderLayer": "megatron.lite.primitive.modules.mtp",
    "MTPLossAutoScaler": "megatron.lite.primitive.modules.mtp",
    "MoEAuxLossAutoScaler": "megatron.lite.primitive.modules.moe",
    "MultimodalRotaryEmbedding": "megatron.lite.primitive.modules.mrope",
    "RMSNorm": "megatron.lite.primitive.modules.mla_dsa",
    "SigmoidTopKRouter": "megatron.lite.primitive.modules.router",
    "TokenDispatcher": "megatron.lite.primitive.modules.dispatcher",
    "TopKRouter": "megatron.lite.primitive.modules.router",
    "_AllToAll": "megatron.lite.primitive.modules.moe",
    "build_rotary_embeddings": "megatron.lite.primitive.modules.mla_dsa",
    "split_grouped_qkvg": "megatron.lite.primitive.modules.gqa",
}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        mod = importlib.import_module(_LAZY_EXPORTS[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DSAIndexer",
    "Experts",
    "GatedDeltaNet",
    "GQAttention",
    "MLADSA",
    "MTPBlock",
    "MTPDecoderLayer",
    "MTPLossAutoScaler",
    "MoEAuxLossAutoScaler",
    "MultimodalRotaryEmbedding",
    "RMSNorm",
    "SigmoidTopKRouter",
    "TokenDispatcher",
    "TopKRouter",
    "_AllToAll",
    "build_rotary_embeddings",
    "split_grouped_qkvg",
]
