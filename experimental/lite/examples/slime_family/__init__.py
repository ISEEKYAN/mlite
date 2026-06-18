# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Runtime patch for slime-family training backends."""

from .mlite_backend_patch import MLiteTrainRayActor, patch_slime_family_backends

__all__ = ["MLiteTrainRayActor", "patch_slime_family_backends"]
