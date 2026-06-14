# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Context parallel zigzag sequence splitting helpers."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
import torch.distributed as dist


def zigzag_split_for_cp(
    tensor: torch.Tensor,
    cp_rank: int,
    cp_size: int,
    seq_dim: int = 1,
) -> torch.Tensor:
    if cp_size <= 1:
        return tensor
    seq_len = tensor.shape[seq_dim]
    assert (
        seq_len % (2 * cp_size) == 0
    ), f"seq_len={seq_len} must be divisible by 2*cp_size={2 * cp_size}"
    shape = list(tensor.shape)
    shape[seq_dim : seq_dim + 1] = [2 * cp_size, seq_len // (2 * cp_size)]
    tensor = tensor.view(*shape)
    idx = torch.tensor(
        [cp_rank, 2 * cp_size - cp_rank - 1],
        dtype=torch.long,
        device=tensor.device,
    )
    tensor = tensor.index_select(seq_dim, idx)
    shape[seq_dim : seq_dim + 2] = [seq_len // cp_size]
    return tensor.reshape(*shape)


def zigzag_reconstruct_from_cp_parts(
    parts: list[torch.Tensor] | tuple[torch.Tensor, ...], seq_dim: int = 1
) -> torch.Tensor:
    """Reconstruct a full sequence from per-rank zigzag CP shards."""
    cp_size = len(parts)
    if cp_size <= 1:
        return parts[0]
    local_len = parts[0].shape[seq_dim]
    assert (
        local_len % 2 == 0
    ), f"local seq_len={local_len} must be divisible by 2 for zigzag CP reconstruction"
    for idx, part in enumerate(parts):
        assert (
            part.shape == parts[0].shape
        ), f"CP part {idx} shape {tuple(part.shape)} != {tuple(parts[0].shape)}"

    chunk = local_len // 2
    full_len = local_len * cp_size
    out_shape = list(parts[0].shape)
    out_shape[seq_dim] = full_len
    full = torch.zeros(out_shape, dtype=parts[0].dtype, device=parts[0].device)
    for rank, part in enumerate(parts):
        first = part.narrow(seq_dim, 0, chunk)
        second = part.narrow(seq_dim, chunk, chunk)
        full.narrow(seq_dim, rank * chunk, chunk).copy_(first)
        full.narrow(seq_dim, full_len - (rank + 1) * chunk, chunk).copy_(second)
    return full


def zigzag_slice_for_cp(
    tensor: torch.Tensor, cp_rank: int, cp_size: int, seq_dim: int = 1
) -> torch.Tensor:
    """Return one rank's zigzag CP shard from a full sequence tensor."""
    if cp_size <= 1:
        return tensor
    seq_len = tensor.shape[seq_dim]
    assert (
        seq_len % (2 * cp_size) == 0
    ), f"seq_len={seq_len} must be divisible by 2*cp_size={2 * cp_size}"
    chunk = seq_len // (2 * cp_size)
    first = tensor.narrow(seq_dim, cp_rank * chunk, chunk)
    second_start = seq_len - (cp_rank + 1) * chunk
    second = tensor.narrow(seq_dim, second_start, chunk)
    return torch.cat((first, second), dim=seq_dim).contiguous()


def contiguous_slice_for_cp(
    tensor: torch.Tensor, cp_rank: int, cp_size: int, seq_dim: int = 1
) -> torch.Tensor:
    if cp_size <= 1:
        return tensor
    seq_len = tensor.shape[seq_dim]
    assert seq_len % cp_size == 0, f"seq_len={seq_len} must be divisible by cp_size={cp_size}"
    chunk = seq_len // cp_size
    return tensor.narrow(seq_dim, cp_rank * chunk, chunk).contiguous()


def split_packed_contiguous_for_cp(
    tensor: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    *,
    cp_rank: int,
    cp_size: int,
    name: str,
) -> torch.Tensor | None:
    if tensor is None or cp_size <= 1:
        return tensor
    total_tokens = int(cu_seqlens[-1].item())
    matches = [idx for idx, size in enumerate(tensor.shape) if size == total_tokens]
    if not matches:
        raise ValueError(f"Packed {name} has no sequence dimension of length {total_tokens}.")
    seq_dim = matches[-1]
    parts = []
    for idx in range(cu_seqlens.size(0) - 1):
        start, end = int(cu_seqlens[idx].item()), int(cu_seqlens[idx + 1].item())
        seq_len = end - start
        if seq_len % cp_size:
            raise ValueError(f"Packed sample {idx} length {seq_len} must divide cp={cp_size}.")
        local_len = seq_len // cp_size
        parts.append(tensor.narrow(seq_dim, start + cp_rank * local_len, local_len))
    return torch.cat(parts, dim=seq_dim).contiguous()


def zigzag_to_contiguous_chunks(
    tensor: torch.Tensor,
    cp_group: dist.ProcessGroup | None,
    seq_dim: int = 1,
) -> torch.Tensor:
    return _zigzag_contiguous_chunk_swap(tensor, cp_group, seq_dim, to_contiguous=True)


def contiguous_to_zigzag_chunks(
    tensor: torch.Tensor,
    cp_group: dist.ProcessGroup | None,
    seq_dim: int = 1,
) -> torch.Tensor:
    return _zigzag_contiguous_chunk_swap(tensor, cp_group, seq_dim, to_contiguous=False)


def _zigzag_contiguous_chunk_swap(
    tensor: torch.Tensor,
    cp_group: Optional[dist.ProcessGroup],
    seq_dim: int,
    *,
    to_contiguous: bool,
) -> torch.Tensor:
    cp_size = dist.get_world_size(cp_group) if cp_group is not None else 1
    if cp_size <= 1:
        return tensor
    cp_rank = dist.get_rank(cp_group)

    if seq_dim != 0:
        tensor = tensor.movedim(seq_dim, 0)
    tensor = tensor.contiguous()

    local_len = tensor.size(0)
    if local_len % 2 != 0:
        raise ValueError(
            f"zigzag/contiguous CP chunk swap requires even local sequence length, got {local_len}."
        )
    chunk_len = local_len // 2

    def rank_to_chunks(rank: int, in_zigzag: bool) -> tuple[int, int]:
        if in_zigzag:
            return rank, 2 * cp_size - rank - 1
        return 2 * rank, 2 * rank + 1

    def chunk_to_dest(chunk_idx: int, target_zigzag: bool) -> tuple[int, int]:
        if target_zigzag:
            if chunk_idx < cp_size:
                return chunk_idx, 0
            return 2 * cp_size - chunk_idx - 1, 1
        return chunk_idx // 2, chunk_idx % 2

    source_in_zigzag = to_contiguous
    target_in_zigzag = not to_contiguous
    local_chunks = [tensor[:chunk_len], tensor[chunk_len:]]
    local_chunk_indices = rank_to_chunks(cp_rank, source_in_zigzag)
    local_dests = [chunk_to_dest(chunk_idx, target_in_zigzag) for chunk_idx in local_chunk_indices]
    local_slot_order = sorted(range(2), key=lambda slot: local_dests[slot])
    send_buf = torch.cat([local_chunks[slot] for slot in local_slot_order], dim=0).contiguous()

    input_split_chunks = [0] * cp_size
    for dst_rank, _dst_slot in local_dests:
        input_split_chunks[dst_rank] += 1

    output_split_chunks = [0] * cp_size
    recv_dst_slots_per_source: list[list[int]] = [[] for _ in range(cp_size)]
    for src_rank in range(cp_size):
        src_chunks = rank_to_chunks(src_rank, source_in_zigzag)
        src_dests = [chunk_to_dest(chunk_idx, target_in_zigzag) for chunk_idx in src_chunks]
        src_slot_order = sorted(range(2), key=lambda slot: src_dests[slot])
        for slot in src_slot_order:
            dst_rank, dst_slot = src_dests[slot]
            if dst_rank == cp_rank:
                output_split_chunks[src_rank] += 1
                recv_dst_slots_per_source[src_rank].append(dst_slot)

    input_split_sizes = [count * chunk_len for count in input_split_chunks]
    output_split_sizes = [count * chunk_len for count in output_split_chunks]
    recv_shape = (sum(output_split_sizes), *send_buf.shape[1:])
    recv_buf = torch.empty(recv_shape, dtype=send_buf.dtype, device=send_buf.device)
    from torch.distributed.nn.functional import all_to_all_single

    recv_buf = all_to_all_single(
        recv_buf,
        send_buf,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=cp_group,
    )

    target_slots: list[torch.Tensor | None] = [None, None]
    offset = 0
    for src_rank in range(cp_size):
        for dst_slot in recv_dst_slots_per_source[src_rank]:
            target_slots[dst_slot] = recv_buf[offset : offset + chunk_len]
            offset += chunk_len
    if any(slot is None for slot in target_slots):
        raise RuntimeError("Incomplete CP chunk reassembly.")

    out = torch.cat([slot for slot in target_slots if slot is not None], dim=0)
    if seq_dim != 0:
        out = out.movedim(0, seq_dim)
    return out.contiguous()


def zigzag_position_ids_for_cp(
    seq_len: int,
    cp_rank: int,
    cp_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Return global position IDs for this CP rank under zigzag splitting.

    Returns shape [1, seq_len // cp_size] matching batch dim convention.
    """
    if cp_size <= 1:
        return torch.arange(seq_len, device=device).unsqueeze(0)
    chunk = seq_len // (2 * cp_size)
    first = torch.arange(cp_rank * chunk, (cp_rank + 1) * chunk, device=device)
    second_start = (2 * cp_size - cp_rank - 1) * chunk
    second = torch.arange(second_start, second_start + chunk, device=device)
    return torch.cat([first, second]).unsqueeze(0)


def contiguous_position_ids_for_cp(
    seq_len: int,
    *,
    cp_rank: int,
    cp_size: int,
    device: torch.device,
) -> torch.Tensor:
    if cp_size <= 1:
        return torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    assert seq_len % cp_size == 0, f"seq_len={seq_len} must be divisible by cp_size={cp_size}"
    local_len = seq_len // cp_size
    start = cp_rank * local_len
    return torch.arange(start, start + local_len, device=device, dtype=torch.long).unsqueeze(0)


def _ring_shift(
    tensor: torch.Tensor,
    group: dist.ProcessGroup,
    *,
    send_delta: int,
    recv_delta: int,
) -> torch.Tensor:
    rank = group.rank()
    size = group.size()
    send = tensor.new_zeros((size, *tensor.shape))
    recv = torch.empty_like(send)
    send[(rank + send_delta) % size].copy_(tensor.contiguous())
    dist.all_to_all_single(recv, send, group=group)
    return recv[(rank + recv_delta) % size].contiguous()


class _RingExchangeForCP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group: dist.ProcessGroup):
        cp_size = group.size()
        ctx.group = group
        ctx.cp_size = cp_size
        if cp_size <= 1:
            return tensor
        return _ring_shift(tensor, group, send_delta=1, recv_delta=-1)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.cp_size <= 1:
            return grad_output, None
        return _ring_shift(grad_output, ctx.group, send_delta=-1, recv_delta=1), None


def _ring_exchange_for_cp(tensor: torch.Tensor, group: dist.ProcessGroup | None) -> torch.Tensor:
    if group is None or group.size() <= 1:
        return tensor
    return _RingExchangeForCP.apply(tensor, group)


def local_contiguous_sequence_tensor_for_cp(
    tensor: torch.Tensor | None,
    *,
    local_seq_len: int,
    cp_rank: int,
    cp_size: int,
    seq_dim: int = 1,
    name: str = "tensor",
    unsqueeze_1d: bool = True,
) -> torch.Tensor | None:
    if tensor is None or cp_size <= 1:
        return tensor
    if unsqueeze_1d and tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    full_seq_len = local_seq_len * cp_size
    if tensor.size(seq_dim) == local_seq_len:
        return tensor
    if tensor.size(seq_dim) == full_seq_len:
        return contiguous_slice_for_cp(tensor, cp_rank, cp_size, seq_dim=seq_dim)
    raise ValueError(
        f"Contiguous CP expects {name} to be either CP-local or full-length; "
        f"got {tensor.size(seq_dim)} for local_seq_len={local_seq_len}, cp={cp_size}."
    )


def iter_cp_sources(
    tensor: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    cp_rank: int,
    cp_size: int,
    cp_group: dist.ProcessGroup | None,
) -> Iterator[tuple[int, torch.Tensor, torch.Tensor]]:
    if cp_size <= 1:
        yield cp_rank, tensor, position_ids
        return
    if cp_group is None:
        raise RuntimeError("CP source iteration requires a context-parallel process group.")
    source_rank = cp_rank
    source_tensor = tensor
    source_positions = position_ids
    for step in range(cp_size):
        yield source_rank, source_tensor, source_positions
        if step + 1 < cp_size:
            source_tensor = _ring_exchange_for_cp(source_tensor, cp_group)
            source_positions = _ring_exchange_for_cp(source_positions.contiguous(), cp_group)
            source_rank = (source_rank - 1) % cp_size


def _drop_prefix_along_dim(tensor: torch.Tensor, dim: int, prefix_len: int) -> torch.Tensor:
    if prefix_len <= 0:
        return tensor
    return tensor.narrow(dim, prefix_len, tensor.size(dim) - prefix_len)


def _previous_contiguous_chunk_tail(
    tensor: torch.Tensor,
    *,
    tail_len: int,
    cp_rank: int,
    cp_size: int,
    cp_group: dist.ProcessGroup | None,
    seq_dim: int = 1,
) -> torch.Tensor | None:
    if cp_size <= 1 or tail_len <= 0:
        return None
    if cp_group is None:
        raise RuntimeError("CP chunk-tail gather requires a context-parallel process group.")
    if tensor.size(seq_dim) < tail_len:
        raise ValueError(
            f"CP chunk-tail gather requires local_len >= {tail_len}, got {tensor.size(seq_dim)}."
        )
    tail = tensor.narrow(seq_dim, tensor.size(seq_dim) - tail_len, tail_len).contiguous()
    return _ring_exchange_for_cp(tail, cp_group)


def compress_contiguous_chunks_for_cp(
    compressor,
    tensor: torch.Tensor,
    *,
    position_ids: torch.Tensor,
    cp_rank: int,
    cp_size: int,
    cp_group: dist.ProcessGroup | None,
    compress_kwargs: dict[str, Any] | None = None,
    seq_dim: int = 1,
    compressed_seq_dim: int = 2,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    kwargs = compress_kwargs or {}
    compress_ratio = int(compressor.compress_ratio)
    if cp_size <= 1:
        compressed = compressor(tensor, position_ids=position_ids, **kwargs)
        if compressed is None:
            return None
        cutoff = (tensor.size(seq_dim) // compress_ratio) * compress_ratio
        comp_pos = position_ids[:, :cutoff:compress_ratio]
        return compressed, comp_pos

    previous_tail = (
        _previous_contiguous_chunk_tail(
            tensor,
            tail_len=compress_ratio,
            cp_rank=cp_rank,
            cp_size=cp_size,
            cp_group=cp_group,
            seq_dim=seq_dim,
        )
        if compressor.overlap
        else None
    )
    chunk, chunk_pos, drop_prefix = tensor, position_ids, 0
    if compressor.overlap and cp_rank > 0:
        if previous_tail is None:
            raise RuntimeError("CP compressor boundary tails are missing.")
        prefix = previous_tail.to(device=tensor.device, dtype=tensor.dtype)
        prefix_pos = chunk_pos[:, :compress_ratio] - compress_ratio
        chunk = torch.cat([prefix, chunk], dim=seq_dim)
        chunk_pos = torch.cat([prefix_pos, chunk_pos], dim=1)
        drop_prefix = 1
    compressed = compressor(chunk, position_ids=chunk_pos, **kwargs)
    if compressed is None:
        return None
    comp_pos = chunk_pos[
        :, : (chunk.size(seq_dim) // compress_ratio) * compress_ratio : compress_ratio
    ]
    if drop_prefix:
        compressed = _drop_prefix_along_dim(compressed, compressed_seq_dim, drop_prefix)
        comp_pos = comp_pos[:, drop_prefix:]
    return None if compressed.size(compressed_seq_dim) == 0 else (compressed, comp_pos)


def split_packed_for_cp(
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    cp_rank: int,
    cp_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if cp_size <= 1:
        return input_ids, position_ids, cu_seqlens, max_seqlen

    num_seqs = cu_seqlens.size(0) - 1
    ids_parts: list[torch.Tensor] = []
    pos_parts: list[torch.Tensor] = []
    new_lengths: list[int] = []

    for i in range(num_seqs):
        start = int(cu_seqlens[i].item())
        end = int(cu_seqlens[i + 1].item())
        seq_len = end - start
        assert (
            seq_len % (2 * cp_size) == 0
        ), f"Sample {i} length {seq_len} not divisible by 2*cp_size={2 * cp_size}"
        chunk = seq_len // (2 * cp_size)
        c1 = start + cp_rank * chunk
        c2 = start + (2 * cp_size - cp_rank - 1) * chunk
        ids_parts.append(input_ids[c1 : c1 + chunk])
        ids_parts.append(input_ids[c2 : c2 + chunk])
        pos_parts.append(position_ids[c1 : c1 + chunk])
        pos_parts.append(position_ids[c2 : c2 + chunk])
        new_lengths.append(2 * chunk)

    new_ids = torch.cat(ids_parts)
    new_pos = torch.cat(pos_parts)
    lens = torch.tensor(new_lengths, dtype=torch.int32, device=cu_seqlens.device)
    new_cu = torch.zeros(num_seqs + 1, dtype=torch.int32, device=cu_seqlens.device)
    torch.cumsum(lens, dim=0, out=new_cu[1:])
    return new_ids, new_pos, new_cu, max(new_lengths)
