# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Shared model modules owned by Megatron Lite."""

from __future__ import annotations

_EXPORTS = {
    "Experts": ("megatron.lite.primitive.modules.experts", "Experts"),
    "GQAttention": ("megatron.lite.primitive.modules.gqa", "GQAttention"),
    "MoEAuxLossAutoScaler": ("megatron.lite.primitive.modules.moe", "MoEAuxLossAutoScaler"),
    "MultimodalRotaryEmbedding": (
        "megatron.lite.primitive.modules.mrope",
        "MultimodalRotaryEmbedding",
    ),
    "SigmoidTopKRouter": ("megatron.lite.primitive.modules.router", "SigmoidTopKRouter"),
    "TokenDispatcher": ("megatron.lite.primitive.modules.dispatcher", "TokenDispatcher"),
    "TopKRouter": ("megatron.lite.primitive.modules.router", "TopKRouter"),
    "_AllToAll": ("megatron.lite.primitive.modules.moe", "_AllToAll"),
    "split_grouped_qkvg": ("megatron.lite.primitive.modules.gqa_utils", "split_grouped_qkvg"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import importlib

    module_name, attr_name = _EXPORTS[name]
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = [
    "Experts",
    "GQAttention",
    "MoEAuxLossAutoScaler",
    "MultimodalRotaryEmbedding",
    "SigmoidTopKRouter",
    "split_grouped_qkvg",
    "TokenDispatcher",
    "TopKRouter",
    "_AllToAll",
]
