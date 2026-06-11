# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Train-side lightweight protocols and defaults."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from torch.distributed.tensor import Replicate  # pyright: ignore[reportMissingImports]

from megatron.lite.primitive.bundle import ModelBundle


class ModelBuildProtocol(Protocol):
    """Protocol implemented by registered model implementation modules."""

    def build_model_config(self, hf_path: str = "") -> Any:
        """Build or load a model config object."""

    def build_model(self, model_cfg: Any, impl_cfg: Any | None = None) -> ModelBundle:
        """Build model chunks and return a runtime-consumable bundle."""


ExpertClassifierFn = Callable[[str], bool]
PlacementFn = Callable[[str], list]


def default_expert_classifier(name: str) -> bool:
    """Default: params with 'experts' (but not 'router' or 'shared') are expert params."""
    return "experts" in name and "router" not in name and "shared" not in name


def default_placement_fn(name: str) -> list:
    """Default: all Replicate (safe but no resharding benefit)."""
    return [Replicate(), Replicate(), Replicate(), Replicate()]


__all__ = [
    "ExpertClassifierFn",
    "ModelBuildProtocol",
    "PlacementFn",
    "default_expert_classifier",
    "default_placement_fn",
]
