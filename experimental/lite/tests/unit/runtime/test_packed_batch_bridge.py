# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unit coverage for the PackedBatch -> bridge forward metadata boundary.

CPU-only. These tests keep the public data contract model-agnostic while allowing
BridgeRuntime to render Megatron-Core THD kwargs transiently at the forward call.
"""

from __future__ import annotations

from dataclasses import fields

import torch

from megatron.lite.primitive.data import infinite_packed_batches
from megatron.lite.runtime.backends.bridge.runtime import _bridge_forward_kwargs_from_packed_batch
from megatron.lite.runtime.contracts.data import Batch, PackedBatch


def _packed_batch() -> PackedBatch:
    return PackedBatch(
        input_ids=torch.arange(8, dtype=torch.long),
        labels=torch.arange(100, 108, dtype=torch.long),
        seq_lens=torch.tensor([3, 5], dtype=torch.int64),
    )


def test_packed_batch_contract_is_model_agnostic() -> None:
    batch = _packed_batch()

    assert [field.name for field in fields(PackedBatch)] == [
        "input_ids",
        "labels",
        "seq_lens",
        "loss_mask",
    ]
    assert not hasattr(batch, "position_ids")
    assert not hasattr(batch, "packed_seq_params")
    assert not hasattr(batch, "to_bridge_dict")


def test_bridge_forward_kwargs_are_transient_bridge_metadata() -> None:
    batch = _packed_batch()
    out = _bridge_forward_kwargs_from_packed_batch(batch)

    assert set(out) == {"input_ids", "labels", "position_ids", "packed_seq_params"}
    assert out["input_ids"].shape == (1, 8)
    assert out["labels"].shape == (1, 8)
    assert out["position_ids"].shape == (1, 8)
    assert torch.equal(out["input_ids"].reshape(-1), batch.input_ids)
    assert torch.equal(out["labels"].reshape(-1), batch.labels)
    assert torch.equal(
        out["position_ids"].reshape(-1),
        torch.tensor([0, 1, 2, 0, 1, 2, 3, 4]),
    )

    psp = out["packed_seq_params"]
    assert psp.qkv_format == "thd"
    assert torch.equal(psp.cu_seqlens_q, torch.tensor([0, 3, 8], dtype=torch.int32))
    assert psp.max_seqlen_q == 5


def test_bridge_forward_kwargs_carry_loss_mask_only_inside_bridge() -> None:
    batch = _packed_batch()
    batch.loss_mask = torch.tensor([1, 1, 0, 1, 1, 1, 0, 1], dtype=torch.long)
    out = _bridge_forward_kwargs_from_packed_batch(batch)
    assert "loss_mask" in out
    assert torch.equal(out["loss_mask"].reshape(-1), batch.loss_mask)


def test_packed_batch_is_batch_subclass() -> None:
    assert issubclass(PackedBatch, Batch)


def test_infinite_packed_batches_shape_and_determinism() -> None:
    gen_a = infinite_packed_batches(vocab_size=32, seq_len=6, device="cpu", seed=7)
    gen_b = infinite_packed_batches(vocab_size=32, seq_len=6, device="cpu", seed=7)

    a = next(gen_a)
    b = next(gen_b)
    assert isinstance(a, PackedBatch)
    assert a.input_ids.shape == (6,)
    assert a.labels.shape == (6,)
    assert torch.equal(a.seq_lens, torch.tensor([6], dtype=torch.int64))
    assert torch.equal(a.input_ids, b.input_ids)
    assert torch.equal(a.labels, b.labels)
