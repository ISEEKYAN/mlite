"""Megatron-FSDP optimizer primitive package."""

from __future__ import annotations

from megatron.lite.primitive.optimizers.mfsdp.backend import (
    BACKEND,
    MegatronFSDPBackend,
)
from megatron.lite.primitive.optimizers.mfsdp.config import validate_mfsdp_config
from megatron.lite.primitive.optimizers.mfsdp.optimizer import (
    build_mfsdp_stack,
    build_mfsdp_training_optimizer,
    finalize_mfsdp_grads,
)

__all__ = [
    "BACKEND",
    "MegatronFSDPBackend",
    "build_mfsdp_stack",
    "build_mfsdp_training_optimizer",
    "finalize_mfsdp_grads",
    "validate_mfsdp_config",
]
