# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unit tests for ``roll_contiguous_left_for_cp`` (the CP-boundary-correct MTP
roll used by DeepSeek-V4 under context parallel).

Kept in a standalone module (not ``test_parallel_unit.py``) so it does not
inherit that file's module-level ``megatron.core`` importorskip -- these tests
only need ``megatron.lite.primitive.parallel``.
"""
from __future__ import annotations

import pytest
import torch
from megatron.lite.primitive.parallel import roll_contiguous_left_for_cp
from megatron.lite.primitive.parallel.cp import contiguous_slice_for_cp

pytestmark = pytest.mark.mlite


def test_roll_contiguous_left_for_cp_cp1_matches_plain_roll():
    tensor = torch.arange(1, 9).unsqueeze(0)  # [1, 8]
    rolled, token_sum = roll_contiguous_left_for_cp(
        tensor, cp_rank=0, cp_size=1, cp_group=None, seq_dim=-1
    )
    assert torch.equal(rolled, torch.tensor([[2, 3, 4, 5, 6, 7, 8, 0]]))
    assert token_sum.item() == 35  # 2+3+...+8


def test_roll_contiguous_left_for_cp_requires_group_for_cp_gt_1():
    tensor = torch.arange(1, 9).unsqueeze(0)
    with pytest.raises(ValueError):
        roll_contiguous_left_for_cp(tensor, cp_rank=0, cp_size=2, cp_group=None, seq_dim=-1)


def test_roll_contiguous_left_for_cp_crosses_cp_boundary():
    """Under contiguous CP slicing the last token of rank r's slice must roll to
    the *first token of rank r+1's slice* (not wrap onto rank r's own first
    token). Verified against the global plain roll via gloo; needs torchrun."""
    import os

    import torch.distributed as dist

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("Run with torchrun so CP ranks are available.")
    if not dist.is_initialized():
        dist.init_process_group("gloo")
    world = dist.get_world_size()
    rank = dist.get_rank()
    if world < 2:
        pytest.skip("Contiguous CP roll boundary test requires at least 2 ranks.")

    full_len = 4 * world
    full = torch.arange(1, full_len + 1).unsqueeze(0)  # [1, full_len]
    local = contiguous_slice_for_cp(full, rank, world, seq_dim=1)

    rolled_local, _ = roll_contiguous_left_for_cp(
        local, cp_rank=rank, cp_size=world, cp_group=dist.group.WORLD, seq_dim=-1
    )

    parts = [torch.empty_like(rolled_local) for _ in range(world)]
    dist.all_gather(parts, rolled_local.contiguous(), group=dist.group.WORLD)
    gathered = torch.cat(parts, dim=1)

    expected = torch.roll(full, shifts=-1, dims=1)
    expected[:, -1] = 0
    assert torch.equal(gathered, expected)

    if rank == 0:
        print(
            f"NON_SKIP_DS4_CP_CONTIGUOUS_ROLL_PASSED world_size={world} "
            f"gathered={gathered.tolist()} expected={expected.tolist()}"
        )
