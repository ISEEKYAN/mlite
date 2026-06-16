# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Data contracts — forward_backward input/output types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


class Batch:
    """Protocol for data passed to Runtime.forward_backward.

    Sequences are packed without padding.  ``sizes()`` gives per-sequence
    lengths so the model protocol can derive its own internal forward metadata.

    Subclass this to keep runtime inputs typed while avoiding dict-shaped model
    internals at the public boundary.
    """

    def __len__(self) -> int:
        """Number of sequences in this batch."""
        raise NotImplementedError

    def sizes(self) -> torch.Tensor:
        """Per-sequence token counts.  Shape ``[num_seqs]``."""
        raise NotImplementedError


@dataclass(slots=True)
class PackedBatch(Batch):
    """Variable-length packed batch — no padding.

    All token-level tensors are 1-D with length ``sum(seq_lens)``.
    It deliberately carries only model-agnostic data; THD metadata such as
    position ids or packed sequence parameters is derived inside the backend or
    model protocol that immediately consumes it.
    """

    input_ids: torch.Tensor  # [total_tokens]
    labels: torch.Tensor  # [total_tokens]
    seq_lens: torch.Tensor  # [num_seqs]
    loss_mask: torch.Tensor | None = None  # [total_tokens]

    def __len__(self) -> int:
        return len(self.seq_lens)

    def sizes(self) -> torch.Tensor:
        return self.seq_lens

    @property
    def total_tokens(self) -> int:
        return int(self.seq_lens.sum())


@dataclass(slots=True)
class TrainBatch:
    """Legacy fixed-shape batch (padded).  Use PackedBatch for new code."""

    input_ids: torch.Tensor
    labels: torch.Tensor
    loss_mask: torch.Tensor | None = None
    position_ids: torch.Tensor | None = None
    routed_experts: torch.Tensor | None = None
    cp_size: int | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ModelOutputs:
    """Model forward output."""

    loss: torch.Tensor | None = None
    vocab_parallel_logits: torch.Tensor | None = None
    log_probs: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None
    values: torch.Tensor | None = None
    # MTP
    mtp_logits: torch.Tensor | None = None
    mtp_loss: torch.Tensor | None = None
    # Router Replay: recorded routing decisions
    routed_experts: torch.Tensor | None = None


@dataclass(slots=True)
class ForwardResult:
    """Output of forward_backward."""

    model_output: ModelOutputs = field(default_factory=ModelOutputs)
    metrics: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "Batch",
    "ForwardResult",
    "ModelOutputs",
    "PackedBatch",
    "TrainBatch",
]
