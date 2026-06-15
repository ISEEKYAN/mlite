# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from typing import Any

import torch
import torch.distributed as dist

from megatron.lite.primitive.parallel.cp import zigzag_slice_for_cp


def _group_global_rank(group, rank):
    if hasattr(dist, "get_global_rank"):
        return dist.get_global_rank(group, rank)
    return rank


def _send_recv_ring(tensor, *, send_rank, recv_rank, group):
    recv = torch.empty_like(tensor)
    ops = [
        dist.P2POp(dist.isend, tensor.contiguous(), _group_global_rank(group, send_rank), group),
        dist.P2POp(dist.irecv, recv, _group_global_rank(group, recv_rank), group),
    ]
    for req in dist.batch_isend_irecv(ops):
        req.wait()
    return recv


class _RingExchangeForCP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group: dist.ProcessGroup):
        cp_size = group.size()
        cp_rank = group.rank()
        ctx.group = group
        ctx.cp_rank = cp_rank
        return _send_recv_ring(
            tensor,
            send_rank=(cp_rank + 1) % cp_size,
            recv_rank=(cp_rank - 1) % cp_size,
            group=group,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        cp_rank = ctx.cp_rank
        cp_size = ctx.group.size()
        return (
            _send_recv_ring(
                grad_output,
                send_rank=(cp_rank - 1) % cp_size,
                recv_rank=(cp_rank + 1) % cp_size,
                group=ctx.group,
            ),
            None,
        )


def _ring_exchange_for_cp(tensor, group):
    return tensor if group is None or group.size() <= 1 else _RingExchangeForCP.apply(tensor, group)


def split_zigzag_local_chunks(
    tensor, cp_rank: int, cp_size: int, seq_dim: int = 1
) -> tuple[tuple[int, torch.Tensor], ...]:
    if cp_size <= 1:
        return ((0, tensor),)
    local_len = tensor.shape[seq_dim]
    assert local_len % 2 == 0, f"local seq_len={local_len} must be even for zigzag CP"
    chunk_len = local_len // 2
    chunk_ids = (cp_rank, 2 * cp_size - cp_rank - 1)
    return tuple(
        (chunk_id, tensor.narrow(seq_dim, offset * chunk_len, chunk_len))
        for offset, chunk_id in enumerate(chunk_ids)
    )


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
    if cp_size <= 1:
        return position_ids

    full_seq_len = local_seq_len * cp_size
    if position_ids.size(1) == local_seq_len:
        return position_ids
    if position_ids.size(1) == full_seq_len:
        return zigzag_slice_for_cp(position_ids, cp_rank, cp_size, seq_dim=1)
    raise ValueError(
        "CP expects position_ids to be either CP-local or full-length; "
        f"got {position_ids.size(1)} for local_seq_len={local_seq_len}, cp={cp_size}."
    )


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
    if seq_len == full_seq_len:
        return zigzag_slice_for_cp(tensor, cp_rank, cp_size, seq_dim=seq_dim)
    raise ValueError(
        f"CP expects {name} to be either CP-local or full-length; "
        f"got {seq_len} for local_seq_len={local_seq_len}, cp={cp_size}."
    )


def iter_cp_sources(tensor, position_ids, *, cp_rank, cp_size, cp_group):
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


def _gather_zigzag_chunk_tails(
    tensor,
    *,
    tail_len,
    cp_rank,
    cp_size,
    cp_group,
    seq_dim=1,
):
    if cp_size <= 1 or tail_len <= 0:
        return None
    if cp_group is None:
        raise RuntimeError("CP chunk-tail gather requires a context-parallel process group.")
    chunks = split_zigzag_local_chunks(tensor, cp_rank, cp_size, seq_dim=seq_dim)
    tails = []
    for _chunk_id, chunk in chunks:
        if chunk.size(seq_dim) < tail_len:
            raise ValueError(f"CP chunk tail needs len >= {tail_len}, got {chunk.size(seq_dim)}.")
        tails.append(chunk.narrow(seq_dim, chunk.size(seq_dim) - tail_len, tail_len).contiguous())
    packed = torch.stack(tails, dim=1)
    from torch.distributed.nn.functional import all_gather

    return list(all_gather(packed, group=cp_group))


def _zigzag_chunk_owner(chunk_id, cp_size):
    if cp_size <= 1:
        return 0, 0
    owner = min(chunk_id, 2 * cp_size - 1 - chunk_id)
    owner_chunks = (owner, 2 * cp_size - 1 - owner)
    return owner, owner_chunks.index(chunk_id)


def compress_zigzag_chunks_for_cp(
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

    boundary_tails = (
        _gather_zigzag_chunk_tails(
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
    comp_parts: list[torch.Tensor] = []
    pos_parts: list[torch.Tensor] = []
    tensor_chunks = split_zigzag_local_chunks(tensor, cp_rank, cp_size, seq_dim=seq_dim)
    pos_chunks = dict(split_zigzag_local_chunks(position_ids, cp_rank, cp_size, seq_dim=1))
    for chunk_id, chunk in tensor_chunks:
        chunk_pos = pos_chunks[chunk_id]
        drop_prefix = 0
        if compressor.overlap and chunk_id > 0:
            if boundary_tails is None:
                raise RuntimeError("CP compressor boundary tails are missing.")
            owner, slot = _zigzag_chunk_owner(chunk_id - 1, cp_size)
            prefix = boundary_tails[owner][:, slot].to(device=tensor.device, dtype=tensor.dtype)
            prefix_pos = chunk_pos[:, :compress_ratio] - compress_ratio
            chunk = torch.cat([prefix, chunk], dim=seq_dim)
            chunk_pos = torch.cat([prefix_pos, chunk_pos], dim=1)
            drop_prefix = 1
        compressed = compressor(chunk, position_ids=chunk_pos, **kwargs)
        if compressed is None:
            continue
        cutoff = (chunk.size(seq_dim) // compress_ratio) * compress_ratio
        comp_pos = chunk_pos[:, :cutoff:compress_ratio]
        if drop_prefix:
            compressed = compressed.narrow(
                compressed_seq_dim,
                drop_prefix,
                compressed.size(compressed_seq_dim) - drop_prefix,
            )
            comp_pos = comp_pos[:, drop_prefix:]
        if compressed.size(compressed_seq_dim) == 0:
            continue
        comp_parts.append(compressed)
        pos_parts.append(comp_pos)
    if not comp_parts:
        return None
    return torch.cat(comp_parts, dim=compressed_seq_dim), torch.cat(pos_parts, dim=1)
