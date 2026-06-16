# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import pytest
import torch

from megatron.lite.primitive.parallel import (
    ParallelState,
    build_pipeline_chunk_layout,
    pack_nested_thd,
    parallel_state_from_model,
    prepare_packed_thd_for_context_parallel,
    reconstruct_packed_from_cp_parts,
    roll_packed_thd_left,
    split_packed_to_cp_local,
    zigzag_position_ids_for_cp,
    zigzag_reconstruct_from_cp_parts,
    zigzag_slice_for_cp,
    zigzag_split_for_cp,
)

pytestmark = pytest.mark.mlite


def test_cp_zigzag_split_slice_and_reconstruct_match():
    tensor = torch.arange(16).reshape(1, 8, 2)
    parts = [zigzag_split_for_cp(tensor, rank, cp_size=2, seq_dim=1) for rank in range(2)]

    assert torch.equal(parts[0], tensor[:, [0, 1, 6, 7], :])
    assert torch.equal(parts[1], tensor[:, [2, 3, 4, 5], :])
    assert torch.equal(zigzag_slice_for_cp(tensor, 0, cp_size=2, seq_dim=1), parts[0])
    assert torch.equal(zigzag_slice_for_cp(tensor, 1, cp_size=2, seq_dim=1), parts[1])
    assert torch.equal(zigzag_reconstruct_from_cp_parts(parts, seq_dim=1), tensor)


def test_cp_position_ids_follow_zigzag_order():
    assert torch.equal(
        zigzag_position_ids_for_cp(8, cp_rank=0, cp_size=2, device=torch.device("cpu")),
        torch.tensor([[0, 1, 6, 7]]),
    )
    assert torch.equal(
        zigzag_position_ids_for_cp(8, cp_rank=1, cp_size=2, device=torch.device("cpu")),
        torch.tensor([[2, 3, 4, 5]]),
    )


# The pipeline layout wires to Megatron-core's layer-layout machinery, so these
# tests need megatron.core importable (present in the mlite GPU/smoke containers).
_mcore_layout = pytest.importorskip("megatron.core.transformer.transformer_block")


def _ranks(pp_size: int) -> list[ParallelState]:
    return [
        ParallelState(
            pp_size=pp_size,
            pp_rank=r,
            pp_is_first=(r == 0),
            pp_is_last=(r == pp_size - 1),
        )
        for r in range(pp_size)
    ]


def _layout_indices(num_layers: int, pp_size: int, **kw) -> list[list[int]]:
    return [
        build_pipeline_chunk_layout(num_layers, ps, **kw).layer_indices for ps in _ranks(pp_size)
    ]


def test_pp_layout_marks_stage_boundaries_and_vpp_chunks():
    rank0 = ParallelState(pp_size=2, pp_rank=0, pp_is_first=True, pp_is_last=False)
    rank1 = ParallelState(pp_size=2, pp_rank=1, pp_is_first=False, pp_is_last=True)

    assert build_pipeline_chunk_layout(8, rank0).layer_indices == [0, 1, 2, 3]
    assert build_pipeline_chunk_layout(8, rank0).has_embed is True
    assert build_pipeline_chunk_layout(8, rank0).has_head is False
    assert build_pipeline_chunk_layout(8, rank1).layer_indices == [4, 5, 6, 7]
    assert build_pipeline_chunk_layout(8, rank1).has_embed is False
    assert build_pipeline_chunk_layout(8, rank1).has_head is True

    vpp_rank0_chunk1 = build_pipeline_chunk_layout(8, rank0, vpp=2, vpp_chunk_id=1)
    vpp_rank1_chunk1 = build_pipeline_chunk_layout(8, rank1, vpp=2, vpp_chunk_id=1)
    assert vpp_rank0_chunk1.layer_indices == [4, 5]
    assert vpp_rank0_chunk1.has_head is False
    assert vpp_rank1_chunk1.layer_indices == [6, 7]
    assert vpp_rank1_chunk1.has_head is True


def test_pp_layout_rejects_non_divisible_vpp_layer_counts():
    # 10 layers over pp2 x vpp3 is not representable; Megatron asserts (vp_size
    # must divide the per-rank layers) rather than silently mis-splitting.
    ps = ParallelState(pp_size=2, pp_rank=0, pp_is_first=True, pp_is_last=False)
    with pytest.raises(AssertionError):
        build_pipeline_chunk_layout(10, ps, vpp=3)


def test_pp_layout_auto_balances_non_divisible_counts_without_error():
    # 6 layers, pp=4 — not divisible; Megatron's embedding/loss accounting balances
    # it to [1, 2, 2, 1] (embedding pulls one off the first stage, loss off the last)
    # instead of raising.
    indices = _layout_indices(6, 4)
    assert indices == [[0], [1, 2], [3, 4], [5]]
    flat = [i for stage in indices for i in stage]
    assert flat == list(range(6))  # contiguous, ordered, complete
    sizes = [len(s) for s in indices]
    assert max(sizes) - min(sizes) <= 1  # balanced
    # vpp=1 (the model default) takes the same non-interleaved path.
    assert _layout_indices(6, 4, vpp=1) == indices
    # num_mtp_layers does not change the decoder split (MTP is appended separately).
    assert _layout_indices(6, 4, num_mtp_layers=1) == indices


def test_pp_layout_divisible_split_is_even_and_unchanged():
    # Divisible counts keep Megatron's plain even split (no checkpoint/layout drift).
    assert _layout_indices(8, 4) == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert _layout_indices(8, 4, vpp=1) == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert _layout_indices(8, 4, num_mtp_layers=3) == [[0, 1], [2, 3], [4, 5], [6, 7]]


def test_pp_layout_single_stage_owns_all_layers():
    ps = ParallelState(pp_size=1, pp_rank=0, pp_is_first=True, pp_is_last=True)
    layout = build_pipeline_chunk_layout(5, ps)
    assert layout.layer_indices == [0, 1, 2, 3, 4]
    assert layout.has_embed and layout.has_head


def test_virtual_pipeline_rank_is_tracked_on_lite_parallel_state():
    from megatron.lite.primitive.parallel.pipeline import _set_virtual_pipeline_rank

    ps = ParallelState(pp_size=2, pp_rank=1, pp_is_first=False, pp_is_last=True)

    _set_virtual_pipeline_rank(ps, chunk_id=1, num_chunks=2)

    assert ps.virtual_pipeline_size == 2
    assert ps.virtual_pipeline_rank == 1

    _set_virtual_pipeline_rank(ps, chunk_id=None, num_chunks=2)

    assert ps.virtual_pipeline_size is None
    assert ps.virtual_pipeline_rank is None


def test_thd_roll_keeps_sequence_boundaries():
    cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)
    rolled, token_sum = roll_packed_thd_left(torch.arange(8), cu_seqlens_padded=cu_seqlens, dims=0)

    assert torch.equal(rolled, torch.tensor([1, 2, 3, 0, 5, 6, 7, 0]))
    assert token_sum.item() == 24


def test_thd_cp_split_and_reconstruct_roundtrip():
    cu_seqlens = torch.tensor([0, 8], dtype=torch.int32)
    tensor = torch.arange(8)
    parts = [
        split_packed_to_cp_local(
            tensor, cu_seqlens_padded=cu_seqlens, cp_size=2, cp_rank=rank, dim=0
        )
        for rank in range(2)
    ]

    assert torch.equal(parts[0], torch.tensor([0, 1, 6, 7]))
    assert torch.equal(parts[1], torch.tensor([2, 3, 4, 5]))
    assert torch.equal(
        reconstruct_packed_from_cp_parts(parts, cu_seqlens_padded=cu_seqlens, cp_size=2, dim=0),
        tensor,
    )


def test_plain_thd_batch_is_split_by_protocol_context_parallel_helper():
    ids = torch.nested.as_nested_tensor(
        [torch.arange(1, 6), torch.arange(11, 18)],
        layout=torch.jagged,
    )
    labels = torch.nested.as_nested_tensor(
        [torch.arange(101, 106), torch.arange(111, 118)],
        layout=torch.jagged,
    )
    loss_mask = torch.nested.as_nested_tensor(
        [torch.ones(5), torch.ones(7)],
        layout=torch.jagged,
    )
    packed = pack_nested_thd(
        ids,
        cp_size=2,
        split_cp=False,
        labels=labels,
        loss_mask=loss_mask,
    )

    assert packed.input_ids.shape == (1, 16)
    assert packed.cp_size == 2
    assert packed.packed_seq_params.local_cp_size is None

    local_params, local_tensors = prepare_packed_thd_for_context_parallel(
        packed.packed_seq_params,
        (packed.input_ids, packed.labels, packed.loss_mask, packed.position_ids),
        cp_size=2,
        cp_rank=0,
    )

    expected_ids = split_packed_to_cp_local(
        packed.input_ids,
        cu_seqlens_padded=packed.cu_seqlens_padded,
        cp_size=2,
        cp_rank=0,
        dim=1,
    )
    expected_pos = split_packed_to_cp_local(
        packed.position_ids,
        cu_seqlens_padded=packed.cu_seqlens_padded,
        cp_size=2,
        cp_rank=0,
        dim=1,
    )
    local_ids, local_labels, local_loss_mask, local_pos = local_tensors
    assert torch.equal(local_ids, expected_ids)
    assert torch.equal(local_pos, expected_pos)
    assert local_labels is not None
    assert local_loss_mask is not None
    assert local_params.local_cp_size == 2
    assert local_params.cp_rank == 0


def test_parallel_state_from_model_unwraps_ddp_style_module():
    class Model:
        ps = ParallelState(cp_size=2, cp_rank=1)

    class Wrapper:
        module = Model()

    assert parallel_state_from_model(Wrapper()).cp_rank == 1


def test_protocol_context_parallel_helper_keeps_pre_split_thd_batch_idempotent():
    ids = torch.nested.as_nested_tensor([torch.arange(8)], layout=torch.jagged)
    packed = pack_nested_thd(ids, cp_size=2, cp_rank=1)

    local_params, local_tensors = prepare_packed_thd_for_context_parallel(
        packed.packed_seq_params,
        (packed.input_ids, packed.position_ids),
        cp_size=2,
        cp_rank=1,
    )

    assert local_params is packed.packed_seq_params
    assert torch.equal(local_tensors[0], packed.input_ids)
    assert local_params.local_cp_size == 2


def test_protocol_context_parallel_helper_is_noop_without_packed_thd_params():
    tensor = torch.arange(8)

    local_params, local_tensors = prepare_packed_thd_for_context_parallel(
        None,
        (tensor,),
        cp_size=2,
        cp_rank=0,
    )

    assert local_params is None
    assert local_tensors[0] is tensor
