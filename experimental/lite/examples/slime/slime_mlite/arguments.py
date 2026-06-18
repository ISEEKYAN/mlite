# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Argument parsing for the Megatron Lite slime backend.

Megatron Lite is Megatron-Core based, so the slime launch reuses Megatron's
argument parser for model/parallelism/optimizer flags exactly like the built-in
``megatron`` backend. ``mlite_parse_args`` is a thin wrapper that layers a few
``--mlite-*`` flags (model implementation knobs that have no Megatron
equivalent) on top of that parser. Megatron Lite reads the model architecture
straight from the HF checkpoint, so we skip Megatron's HF<->arg validation.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

__all__ = ["add_mlite_arguments", "mlite_parse_args", "validate_args"]

# User-facing optimizer-backend names. ``dist_opt`` selects Megatron-Core's
# distributed optimizer (DDP wrapper + distributed optimizer); ``fsdp2`` selects Megatron Lite's
# FSDP2 wrapper. Megatron Lite's protocol layer still keys its internal
# ``impl_cfg.optimizer`` enum on the legacy value ``"mc"`` for the distributed
# optimizer; that rename is a separate cross-model cleanup, so we expose the
# current name here and bridge it in exactly one place below.
_OPTIMIZER_BACKEND_TO_IMPL = {
    "dist_opt": "mc",
    "fsdp2": "fsdp2",
}


def optimizer_backend_to_impl(backend: str) -> str:
    """Map a user-facing optimizer-backend name to the Megatron Lite impl key."""
    if backend not in _OPTIMIZER_BACKEND_TO_IMPL:
        raise ValueError(
            f"Unsupported --mlite-optimizer-backend {backend!r}; "
            f"expected one of {sorted(_OPTIMIZER_BACKEND_TO_IMPL)}."
        )
    return _OPTIMIZER_BACKEND_TO_IMPL[backend]


def add_mlite_arguments(parser):
    """Register Megatron Lite specific flags that have no Megatron equivalent."""
    group = parser.add_argument_group(title="megatron-lite")
    group.add_argument(
        "--mlite-model-name",
        type=str,
        default="auto",
        help="Megatron Lite model registry name; 'auto' infers it from the HF config.",
    )
    group.add_argument(
        "--mlite-impl",
        type=str,
        default="lite",
        help="Megatron Lite model implementation variant.",
    )
    group.add_argument(
        "--mlite-optimizer-backend",
        type=str,
        default="dist_opt",
        choices=sorted(_OPTIMIZER_BACKEND_TO_IMPL),
        help="Optimizer backend: dist_opt (Megatron-Core distributed optimizer) or fsdp2.",
    )
    group.add_argument(
        "--mlite-attention-backend",
        type=str,
        default=None,
        help="Override attention backend (defaults to --attention-backend, else 'flash').",
    )
    group.add_argument(
        "--mlite-optimizer-offload",
        action="store_true",
        default=False,
        help="Offload optimizer state to CPU (offload_fraction=1.0 + precision-aware "
        "optimizer + decoupled weight decay), matching the verl example. Needed to fit "
        "large MoE models on limited GPU memory.",
    )
    return parser


def mlite_parse_args(extra_args_provider, skip_hf_validate: bool = False):
    """Parse Megatron + slime + Megatron-Lite args.

    Mirrors ``slime.backends.megatron_utils.arguments.megatron_parse_args`` but
    skips the Megatron<->HF config cross-check: Megatron Lite builds the model
    from the HF checkpoint directly, so the Megatron model args only need to be
    self-consistent, not mirror the HF config.
    """
    from megatron.training.arguments import parse_args as _megatron_parse_args
    from slime.backends.megatron_utils.arguments import set_default_megatron_args

    def _provider(parser):
        if extra_args_provider is not None:
            parser = extra_args_provider(parser)
        return add_mlite_arguments(parser)

    args = _megatron_parse_args(extra_args_provider=_provider, ignore_unknown_args=True)

    args.rank = 0
    args.world_size = args.actor_num_nodes * args.actor_num_gpus_per_node
    args = set_default_megatron_args(args)
    return args


def validate_args(args):
    """Light validation for the Megatron Lite backend.

    Unlike the megatron backend we do not run Megatron's ``validate_args``
    (which assumes a Megatron-Core training loop and an HF-matching config).
    """
    # slime always uses variable sequence lengths (packed THD).
    args.variable_seq_lengths = True
    if not getattr(args, "hf_checkpoint", None):
        raise ValueError("--hf-checkpoint is required for the Megatron Lite backend.")
