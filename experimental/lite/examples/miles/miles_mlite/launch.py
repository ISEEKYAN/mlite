# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Launch miles with the slime-family Megatron Lite backend patch."""

from __future__ import annotations

import asyncio

from miles.utils.arguments import parse_args
from miles.utils.tracking_utils import finish_tracking
from slime_family.arguments import add_mlite_arguments, validate_mlite_args


def _select_train_loop(args):
    if getattr(args, "colocate", False):
        from train import train
    else:
        from train_async import train
    return train


def main() -> None:
    args = parse_args(add_custom_arguments=add_mlite_arguments)
    if getattr(args, "mlite_backend_patch", False):
        from slime_family.mlite_backend_patch import patch_slime_family_backends

        patch_slime_family_backends()
        validate_mlite_args(args)
    train = _select_train_loop(args)
    try:
        asyncio.run(train(args))
    finally:
        finish_tracking()


if __name__ == "__main__":
    main()
