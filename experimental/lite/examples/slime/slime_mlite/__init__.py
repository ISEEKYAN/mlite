# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Megatron Lite training backend for the slime RL framework.

Importing this package registers ``mlite`` with slime's train-backend registry,
so a slime launch only needs ``--train-backend mlite --train-backend-module
slime_mlite``. All Megatron Lite specific code lives here; slime itself carries
only a small backend-dispatch seam (see ``slime.ray.train_backend``).
"""

from __future__ import annotations


def _register() -> None:
    from slime.ray.train_backend import register_train_backend

    from .arguments import mlite_parse_args, validate_args

    def _actor_loader() -> type:
        from .actor import MLiteTrainRayActor

        return MLiteTrainRayActor

    register_train_backend(
        "mlite",
        actor_loader=_actor_loader,
        parse_args=mlite_parse_args,
        validate_args=validate_args,
    )


_register()
