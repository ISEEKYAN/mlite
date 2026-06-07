from __future__ import annotations

import torch

from megatron.lite.primitive.parallel.cp import (
    zigzag_reconstruct_from_cp_parts,
    zigzag_slice_for_cp,
    zigzag_split_for_cp,
)


def test_dense_zigzag_split_reconstruct_roundtrip_with_grad():
    cp_size = 4
    full = torch.arange(2 * 32 * 3, dtype=torch.float32).reshape(2, 32, 3)
    full.requires_grad_()

    parts = [zigzag_slice_for_cp(full, rank, cp_size, seq_dim=1) for rank in range(cp_size)]
    split_parts = [zigzag_split_for_cp(full, rank, cp_size, seq_dim=1) for rank in range(cp_size)]

    for got, expected in zip(parts, split_parts, strict=True):
        assert torch.equal(got, expected)

    reconstructed = zigzag_reconstruct_from_cp_parts(parts, seq_dim=1)
    assert torch.equal(reconstructed, full)

    reconstructed.square().sum().backward()
    assert torch.equal(full.grad, 2 * full.detach())
