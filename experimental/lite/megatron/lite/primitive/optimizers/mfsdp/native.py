# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Boundary scaffold for a future Megatron Lite-native Megatron-FSDP adapter.

This module is intentionally not registered as an optimizer backend. It records
the Phase 3 boundary: keep the wrap primitive as the oracle, and only move
small, proven interface pieces into Megatron Lite-native code when the wrapper exposes a
specific blocker.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NativeMFSDPBoundary:
    """Ownership boundary for any future native work."""

    bb_owned: tuple[str, ...] = (
        "config_lowering",
        "parallel_state_to_process_groups",
        "hf_load_and_checkpoint_adapter",
        "runtime_hook_scheduling",
        "debug_and_measurement",
    )
    mcore_owned: tuple[str, ...] = (
        "param_and_grad_buffer",
        "fsdp_flatten_shard_unshard",
        "communication_scheduling",
        "optimizer_state_sharding",
        "dtensor_checkpoint_semantics",
        "te_fp8_hooks",
    )


NATIVE_BOUNDARY = NativeMFSDPBoundary()


def build_native_mfsdp_stack(*_args, **_kwargs):
    raise NotImplementedError(
        "Megatron Lite-native Megatron-FSDP is a Phase 3 boundary only. Use "
        "optimizer='megatron_fsdp' as the registered wrapper primitive."
    )


__all__ = [
    "NATIVE_BOUNDARY",
    "NativeMFSDPBoundary",
    "build_native_mfsdp_stack",
]
