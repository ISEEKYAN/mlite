# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather

from megatron.lite.primitive.parallel.cp import contiguous_slice_for_cp


def _all_gather_cp(tensor: torch.Tensor, group: dist.ProcessGroup) -> list[torch.Tensor]:
    return list(all_gather(tensor.contiguous(), group=group))


def local_position_ids_for_cp(position_ids, *, batch, local_seq_len, cp_rank, cp_size):
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)
    if position_ids.dim() != 2:
        raise ValueError("position_ids must have shape (S,) or (B, S).")
    if position_ids.size(0) == 1 and batch > 1:
        position_ids = position_ids.expand(batch, -1)
    if position_ids.size(0) != batch:
        raise ValueError(
            f"position_ids batch={position_ids.size(0)} does not match input batch={batch}."
        )
    if cp_size <= 1 or position_ids.size(1) == local_seq_len:
        return position_ids

    full_seq_len = local_seq_len * cp_size
    if position_ids.size(1) != full_seq_len:
        raise ValueError(
            "CP expects position_ids to be either CP-local or full-length; "
            f"got {position_ids.size(1)} for local_seq_len={local_seq_len}, cp={cp_size}."
        )
    return contiguous_slice_for_cp(position_ids, cp_rank, cp_size, seq_dim=1)


def local_sequence_tensor_for_cp(
    tensor,
    *,
    local_seq_len,
    cp_rank,
    cp_size,
    seq_dim=1,
    name: str = "tensor",
    unsqueeze_1d: bool = True,
):
    if tensor is None or cp_size <= 1:
        return tensor
    if unsqueeze_1d and tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    full_seq_len = local_seq_len * cp_size
    seq_len = tensor.size(seq_dim)
    if seq_len == local_seq_len:
        return tensor
    if seq_len != full_seq_len:
        raise ValueError(
            f"CP expects {name} to be either CP-local or full-length; "
            f"got {seq_len} for local_seq_len={local_seq_len}, cp={cp_size}."
        )
    return contiguous_slice_for_cp(tensor, cp_rank, cp_size, seq_dim=seq_dim)


def iter_cp_sources(tensor, position_ids, *, cp_rank, cp_size, cp_group):
    if cp_size <= 1:
        yield cp_rank, tensor, position_ids
        return
    if cp_group is None:
        raise RuntimeError("CP source iteration requires a context-parallel process group.")
    tensor_parts = _all_gather_cp(tensor, cp_group)
    position_parts = _all_gather_cp(position_ids.to(dtype=torch.long), cp_group)
    for rank, (source_tensor, source_positions) in enumerate(zip(tensor_parts, position_parts)):
        yield rank, source_tensor, source_positions


def _gather_contiguous_tail(tensor, *, tail_len, cp_size, cp_group, seq_dim):
    if cp_size <= 1 or tail_len <= 0:
        return None
    if cp_group is None:
        raise RuntimeError("CP chunk-tail gather requires a context-parallel process group.")
    if tensor.size(seq_dim) < tail_len:
        raise ValueError(f"CP chunk tail needs len >= {tail_len}, got {tensor.size(seq_dim)}.")
    tail = tensor.narrow(seq_dim, tensor.size(seq_dim) - tail_len, tail_len)
    return _all_gather_cp(tail.contiguous(), cp_group)


def compress_contiguous_chunks_for_cp(
    compressor,
    tensor,
    *,
    position_ids,
    cp_rank,
    cp_size,
    cp_group,
    compress_kwargs: dict[str, Any] | None = None,
    seq_dim=1,
    compressed_seq_dim=2,
):
    kwargs = compress_kwargs or {}
    compress_ratio = int(compressor.compress_ratio)
    if cp_size <= 1:
        compressed = compressor(tensor, position_ids=position_ids, **kwargs)
        if compressed is None:
            return None
        cutoff = (tensor.size(seq_dim) // compress_ratio) * compress_ratio
        comp_pos = position_ids[:, :cutoff:compress_ratio]
        return compressed, comp_pos

    drop_prefix = 0
    tail_parts = None
    if compressor.overlap:
        tail_parts = _gather_contiguous_tail(
            tensor,
            tail_len=compress_ratio,
            cp_size=cp_size,
            cp_group=cp_group,
            seq_dim=seq_dim,
        )
        zero_tail = tensor.new_zeros(())
        for tail in tail_parts:
            zero_tail = zero_tail + tail.to(dtype=tensor.dtype).sum() * 0.0
        tensor = tensor + zero_tail
    if tail_parts is not None and cp_rank > 0:
        prefix = tail_parts[cp_rank - 1].to(device=tensor.device, dtype=tensor.dtype)
        prefix_pos = position_ids[:, :compress_ratio] - compress_ratio
        tensor = torch.cat([prefix, tensor], dim=seq_dim)
        position_ids = torch.cat([prefix_pos, position_ids], dim=1)
        drop_prefix = 1

    compressed = compressor(tensor, position_ids=position_ids, **kwargs)
    if compressed is None:
        return None
    cutoff = (tensor.size(seq_dim) // compress_ratio) * compress_ratio
    comp_pos = position_ids[:, :cutoff:compress_ratio]
    if drop_prefix:
        compressed = compressed.narrow(
            compressed_seq_dim,
            drop_prefix,
            compressed.size(compressed_seq_dim) - drop_prefix,
        )
        comp_pos = comp_pos[:, drop_prefix:]
    if compressed.size(compressed_seq_dim) == 0:
        return None
    return compressed, comp_pos
