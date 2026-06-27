# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Native DeepSeek V4 (ds4flash) lite implementation."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from megatron.lite.model.deepseek_v4.lite.model import DeepseekV4Model

__all__ = ["DeepseekV4Model"]


def __getattr__(name: str):
    """Load the TE-dependent model only when callers request the model class.

    Checkpoint conversion and protocol inspection are intentionally usable in
    CPU-only environments.  Importing either submodule must therefore not pull
    in ``model.py`` (and Transformer Engine) as a package-import side effect.
    """
    if name == "DeepseekV4Model":
        from megatron.lite.model.deepseek_v4.lite.model import DeepseekV4Model

        return DeepseekV4Model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
