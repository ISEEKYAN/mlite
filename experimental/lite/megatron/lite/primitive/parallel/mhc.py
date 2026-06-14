# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from typing import Any

import torch


def expand_mhc_hidden_for_pipeline(hidden: torch.Tensor, *, hc_mult: int) -> torch.Tensor:
    if hidden.dim() == 3:
        return hidden.unsqueeze(2).expand(-1, -1, hc_mult, -1).contiguous()
    return hidden


def contract_mhc_hidden_for_pipeline(
    hidden: torch.Tensor,
    *,
    norm: Any,
    head: Any,
    return_source: bool = False,
):
    if head is None or norm is None:
        if return_source:
            return hidden, None
        return hidden
    source = hidden
    contracted = norm(head(hidden))
    if return_source:
        return contracted, source
    return contracted
