"""Native DeepSeek V4 lite model with CPU-smoke-friendly torch modules."""

from __future__ import annotations

from contextlib import nullcontext
import math
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
from megatron.lite.primitive.parallel.cp import (
    zigzag_reconstruct_from_cp_parts,
    zigzag_slice_for_cp,
)
from megatron.lite.primitive.parallel.state import ParallelState


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * norm.to(dtype=x.dtype) * self.weight.to(device=x.device, dtype=x.dtype)


class DeepseekV4GroupedLinear(nn.Module):
    def __init__(self, in_features_per_group: int, out_features: int, n_groups: int):
        super().__init__()
        self.in_features_per_group = in_features_per_group
        self.out_features = out_features
        self.n_groups = n_groups
        self.weight = nn.Parameter(torch.empty(out_features, in_features_per_group))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_per_group = self.out_features // self.n_groups
        weight = self.weight.view(self.n_groups, out_per_group, self.in_features_per_group)
        return torch.einsum("...gd,god->...go", x, weight)


def _hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    iters: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    split_sizes = [hc_mult, hc_mult, hc_mult * hc_mult]
    pre_mix, post_mix, comb_mix = mixes.split(split_sizes, dim=-1)
    base_pre, base_post, base_comb = hc_base.float().split(split_sizes, dim=-1)
    scale = hc_scale.float()
    pre = torch.sigmoid(pre_mix * scale[0] + base_pre) + eps
    post = torch.sigmoid(post_mix * scale[1] + base_post) + eps
    log_comb = (comb_mix * scale[2] + base_comb).view(*comb_mix.shape[:-1], hc_mult, hc_mult)
    for _ in range(iters):
        log_comb = log_comb - log_comb.logsumexp(-1, keepdim=True)
        log_comb = log_comb - log_comb.logsumexp(-2, keepdim=True)
    return pre, post, log_comb.exp() + eps


class DeepseekV4HyperConnection(nn.Module):
    def __init__(self, hidden_size: int, hc_mult: int, sinkhorn_iters: int, eps: float):
        super().__init__()
        mix = (2 + hc_mult) * hc_mult
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps
        self.fn = nn.Parameter(torch.empty(mix, hc_mult * hidden_size, dtype=torch.float32))
        self.base = nn.Parameter(torch.zeros(mix, dtype=torch.float32))
        self.scale = nn.Parameter(torch.ones(3, dtype=torch.float32))
        nn.init.xavier_uniform_(self.fn)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() == 3:
            x = x.unsqueeze(2).expand(*x.shape[:2], self.hc_mult, x.size(-1))
        shape, dtype = x.shape, x.dtype
        xf = x.flatten(2).float()
        rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        mixes = F.linear(xf, self.fn.float()) * rsqrt
        pre, post, comb = _hc_split_sinkhorn(
            mixes, self.scale, self.base, self.hc_mult, self.sinkhorn_iters, self.eps
        )
        y = torch.sum(pre.unsqueeze(-1) * xf.view(shape), dim=2)
        return y.to(dtype), post, comb

    @staticmethod
    def post(
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        y = post.unsqueeze(-1) * x.unsqueeze(-2).float()
        y = y + torch.sum(comb.unsqueeze(-1) * residual.float().unsqueeze(-2), dim=2)
        return y.to(dtype=x.dtype)


class DeepseekV4HyperHead(nn.Module):
    def __init__(self, hidden_size: int, hc_mult: int, eps: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.eps = eps
        self.hc_fn = nn.Parameter(torch.empty(hc_mult, hc_mult * hidden_size, dtype=torch.float32))
        self.hc_base = nn.Parameter(torch.zeros(hc_mult, dtype=torch.float32))
        self.hc_scale = nn.Parameter(torch.ones(1, dtype=torch.float32))
        nn.init.xavier_uniform_(self.hc_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            return x
        shape, dtype = x.shape, x.dtype
        xf = x.flatten(2).float()
        rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        mixes = F.linear(xf, self.hc_fn.float()) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float()) + self.eps
        y = torch.sum(pre.unsqueeze(-1) * xf.view(shape), dim=2)
        return y.to(dtype)


def _build_cos_sin(
    position_ids: torch.Tensor,
    rope_head_dim: int,
    rope_theta: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, rope_head_dim, 2, device=device, dtype=torch.float32) / rope_head_dim)
    )
    freqs = torch.einsum("bs,d->bsd", position_ids.to(torch.float32), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


def _apply_partial_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_head_dim: int,
) -> torch.Tensor:
    if rope_head_dim == 0:
        return x
    rope = x[..., -rope_head_dim:]
    tail = x[..., :-rope_head_dim]
    rope_pairs = rope.unflatten(-1, (-1, 2))
    a, b = rope_pairs[..., 0], rope_pairs[..., 1]
    c = cos[..., : rope_head_dim // 2]
    s = sin[..., : rope_head_dim // 2]
    while c.ndim < a.ndim:
        c = c.unsqueeze(1)
        s = s.unsqueeze(1)
    out_a = a * c - b * s
    out_b = a * s + b * c
    rope_out = torch.stack([out_a, out_b], dim=-1).flatten(-2)
    return torch.cat([tail, rope_out], dim=-1)


def _clamped_swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    gate = gate.float()
    up = up.float()
    if limit > 0:
        up = torch.clamp(up, min=-limit, max=limit)
        gate = torch.clamp(gate, max=limit)
    return F.silu(gate) * up


def _all_gather_cp_parts(tensor: torch.Tensor, ps: ParallelState) -> list[torch.Tensor]:
    if ps.cp_size <= 1:
        return [tensor]
    if ps.cp_group is None:
        raise RuntimeError("DeepSeek V4 CP requires ParallelState.cp_group.")
    try:
        from torch.distributed.nn.functional import all_gather

        return list(all_gather(tensor.contiguous(), group=ps.cp_group))
    except Exception:
        parts = [torch.empty_like(tensor) for _ in range(ps.cp_size)]
        dist.all_gather(parts, tensor.contiguous(), group=ps.cp_group)
        return parts


def _expand_batch_position_ids(position_ids: torch.Tensor, batch: int) -> torch.Tensor:
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)
    if position_ids.dim() != 2:
        raise ValueError("DeepSeek V4 expects position_ids shape (S,) or (B,S).")
    if position_ids.size(0) == 1 and batch > 1:
        position_ids = position_ids.expand(batch, -1)
    if position_ids.size(0) != batch:
        raise ValueError(
            f"position_ids batch={position_ids.size(0)} does not match input batch={batch}."
        )
    return position_ids


def _full_position_ids_for_cp(
    position_ids: torch.Tensor,
    *,
    batch: int,
    local_seq_len: int,
    ps: ParallelState,
) -> torch.Tensor:
    position_ids = _expand_batch_position_ids(position_ids, batch)
    if ps.cp_size <= 1:
        return position_ids

    full_seq_len = local_seq_len * ps.cp_size
    if position_ids.size(1) == full_seq_len:
        return position_ids
    if position_ids.size(1) != local_seq_len:
        raise ValueError(
            "DeepSeek V4 CP expects position_ids to be either CP-local or full-length; "
            f"got {position_ids.size(1)} for local_seq_len={local_seq_len}, cp={ps.cp_size}."
        )
    parts = _all_gather_cp_parts(position_ids.contiguous(), ps)
    return zigzag_reconstruct_from_cp_parts(parts, seq_dim=1)


def _labels_for_local_logits(
    labels: torch.Tensor,
    *,
    local_seq_len: int,
    ps: ParallelState,
) -> torch.Tensor:
    if labels.dim() == 1:
        labels = labels.unsqueeze(0)
    if ps.cp_size <= 1:
        return labels
    full_seq_len = local_seq_len * ps.cp_size
    if labels.size(1) == full_seq_len:
        return zigzag_slice_for_cp(labels, ps.cp_rank, ps.cp_size, seq_dim=1)
    if labels.size(1) != local_seq_len:
        raise ValueError(
            "DeepSeek V4 CP expects labels to be either CP-local or full-length; "
            f"got {labels.size(1)} for local_seq_len={local_seq_len}, cp={ps.cp_size}."
        )
    return labels


def _full_input_ids_for_cp(
    input_ids: torch.Tensor | None,
    *,
    local_seq_len: int,
    full_seq_len: int,
    ps: ParallelState,
) -> torch.Tensor | None:
    if input_ids is None or ps.cp_size <= 1:
        return input_ids
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.size(1) == full_seq_len:
        return input_ids
    if input_ids.size(1) != local_seq_len:
        raise ValueError(
            "DeepSeek V4 CP expects input_ids to be either CP-local or full-length; "
            f"got {input_ids.size(1)} for local_seq_len={local_seq_len}, cp={ps.cp_size}."
        )
    return zigzag_reconstruct_from_cp_parts(
        _all_gather_cp_parts(input_ids.contiguous(), ps),
        seq_dim=1,
    )


def _full_tensor_for_cp(
    tensor: torch.Tensor,
    *,
    full_seq_len: int,
    ps: ParallelState,
) -> tuple[torch.Tensor, int | None]:
    if ps.cp_size <= 1 or tensor.size(1) == full_seq_len:
        return tensor, None
    local_seq_len = tensor.size(1)
    if local_seq_len * ps.cp_size != full_seq_len:
        raise ValueError(
            "DeepSeek V4 CP tensor length does not match the full sequence length; "
            f"got local_seq_len={local_seq_len}, full_seq_len={full_seq_len}, cp={ps.cp_size}."
        )
    return (
        zigzag_reconstruct_from_cp_parts(
            _all_gather_cp_parts(tensor.contiguous(), ps),
            seq_dim=1,
        ),
        local_seq_len,
    )


class DeepseekV4Compressor(nn.Module):
    def __init__(self, config: DeepseekV4Config, compress_ratio: int, head_dim: int):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.rope_head_dim = min(config.qk_rope_head_dim, head_dim)
        self.overlap = compress_ratio == 4
        self.coff = 2 if self.overlap else 1
        self.wkv = nn.Linear(config.hidden_size, self.coff * head_dim, bias=False)
        self.wgate = nn.Linear(config.hidden_size, self.coff * head_dim, bias=False)
        self.ape = nn.Parameter(torch.empty(compress_ratio, self.coff * head_dim, dtype=torch.float32))
        self.norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        nn.init.normal_(self.ape, mean=0.0, std=config.initializer_range)

    def _overlap_transform(self, tensor: torch.Tensor, fill_value: float) -> torch.Tensor:
        bsz, n_blocks, ratio, _, head_dim = tensor.shape
        out = tensor.new_full((bsz, n_blocks, 2 * ratio, head_dim), fill_value)
        out[:, :, ratio:] = tensor[:, :, :, 1]
        out[:, 1:, :ratio] = tensor[:, :-1, :, 0]
        return out

    def forward(self, x: torch.Tensor, *, position_ids: torch.Tensor, rope_theta: float) -> torch.Tensor | None:
        bsz, seq_len, _ = x.shape
        ratio = self.compress_ratio
        n_blocks = seq_len // ratio
        if n_blocks == 0:
            return None
        cutoff = n_blocks * ratio
        content = self.wkv(x[:, :cutoff])
        gate = self.wgate(x[:, :cutoff])
        content = content.view(bsz, n_blocks, ratio, self.coff, self.head_dim)
        gate = gate.view_as(content)
        gate = gate + self.ape.view(1, 1, ratio, self.coff, self.head_dim).to(gate.device)
        if self.overlap:
            content = self._overlap_transform(content, 0.0)
            gate = self._overlap_transform(gate, float("-inf"))
        else:
            content = content.squeeze(3)
            gate = gate.squeeze(3)
        weights = torch.softmax(gate.float(), dim=2).to(dtype=content.dtype)
        compressed = self.norm((content * weights).sum(dim=2))
        compressed = compressed.unsqueeze(1)
        compressed_positions = position_ids[:, :cutoff:ratio]
        cos, sin = _build_cos_sin(
            compressed_positions,
            self.rope_head_dim,
            rope_theta,
            device=x.device,
            dtype=compressed.dtype,
        )
        return _apply_partial_rope(compressed, cos, sin, self.rope_head_dim)


class DeepseekV4DSAIndexer(nn.Module):
    def __init__(self, config: DeepseekV4Config, compress_ratio: int):
        super().__init__()
        self.index_n_heads = config.index_n_heads
        self.index_head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.rope_head_dim = min(config.qk_rope_head_dim, config.index_head_dim)
        self.softmax_scale = self.index_head_dim ** -0.5
        self.wq_b = nn.Linear(config.q_lora_rank, config.index_n_heads * config.index_head_dim, bias=False)
        self.weights_proj = nn.Linear(config.hidden_size, config.index_n_heads, bias=False)
        self.compressor = DeepseekV4Compressor(config, compress_ratio, config.index_head_dim)

    def forward(
        self,
        x: torch.Tensor,
        q_low: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        rope_theta: float,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        compressed = self.compressor(x, position_ids=position_ids, rope_theta=rope_theta)
        if compressed is None:
            return None
        bsz, seq_len, _ = x.shape
        n_compressed = compressed.size(2)
        cos, sin = _build_cos_sin(
            position_ids,
            self.rope_head_dim,
            rope_theta,
            device=x.device,
            dtype=x.dtype,
        )
        q = self.wq_b(q_low).view(bsz, seq_len, self.index_n_heads, self.index_head_dim).transpose(1, 2)
        q = _apply_partial_rope(q, cos, sin, self.rope_head_dim)
        k = compressed.squeeze(1)
        weights = self.weights_proj(x).float() * (self.index_n_heads ** -0.5) * self.softmax_scale
        scores = torch.einsum("bhsd,btd->bsht", q.float(), k.float()).relu()
        scores = (scores * weights.unsqueeze(-1)).sum(dim=2)
        visible = (torch.arange(seq_len, device=x.device) + 1) // self.compressor.compress_ratio
        c_pos = torch.arange(n_compressed, device=x.device).view(1, 1, n_compressed)
        valid = c_pos < visible.view(1, seq_len, 1)
        scores = scores.masked_fill(~valid, -float("inf"))
        topk = min(self.index_topk, n_compressed)
        indices = scores.topk(topk, dim=-1).indices
        return scores, indices


class DeepseekV4Attention(nn.Module):
    def __init__(self, config: DeepseekV4Config, *, layer_idx: int, ps: ParallelState):
        super().__init__()
        self.config = config
        self.ps = ps
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.num_heads_per_group = config.num_attention_heads // config.o_groups
        self.compress_ratio = config.compress_ratios[layer_idx] if config.compress_ratios else 0
        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.wo_a = DeepseekV4GroupedLinear(
            self.num_heads_per_group * self.head_dim,
            config.o_groups * config.o_lora_rank,
            config.o_groups,
        )
        self.wo_b = nn.Linear(config.o_groups * config.o_lora_rank, config.hidden_size, bias=False)
        self.sinks = nn.Parameter(torch.zeros(self.num_heads))
        self.compressor = (
            DeepseekV4Compressor(config, self.compress_ratio, self.head_dim)
            if self.compress_ratio > 1
            else None
        )
        self.indexer = (
            DeepseekV4DSAIndexer(config, self.compress_ratio)
            if self.compress_ratio == 4
            else None
        )

    def _local_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        q_pos = torch.arange(seq_len, device=device).unsqueeze(1)
        k_pos = torch.arange(seq_len, device=device).unsqueeze(0)
        return (k_pos <= q_pos) & (k_pos >= q_pos - self.config.sliding_window + 1)

    def _cp_full_inputs(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        already_full: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int | None]:
        if self.ps.cp_size <= 1:
            return x, _expand_batch_position_ids(position_ids, x.size(0)), attention_mask, None

        local_seq_len = x.size(1)
        position_ids = _expand_batch_position_ids(position_ids, x.size(0))
        if already_full:
            return x, position_ids, attention_mask, None

        full_x = zigzag_reconstruct_from_cp_parts(
            _all_gather_cp_parts(x, self.ps),
            seq_dim=1,
        )
        full_position_ids = _full_position_ids_for_cp(
            position_ids,
            batch=x.size(0),
            local_seq_len=local_seq_len,
            ps=self.ps,
        )
        if attention_mask is not None:
            full_seq_len = local_seq_len * self.ps.cp_size
            if attention_mask.size(-1) != full_seq_len or attention_mask.size(-2) != full_seq_len:
                raise ValueError("DeepSeek V4 CP requires a full-sequence attention_mask.")
        return full_x, full_position_ids, attention_mask, local_seq_len

    def forward(
        self,
        x: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cp_already_full: bool = False,
    ) -> torch.Tensor:
        x, position_ids, attention_mask, cp_local_seq_len = self._cp_full_inputs(
            x,
            position_ids,
            attention_mask,
            already_full=cp_already_full,
        )
        batch, seq_len, _ = x.shape
        cos, sin = _build_cos_sin(
            position_ids,
            self.rope_head_dim,
            self.config.rope_theta,
            device=x.device,
            dtype=x.dtype,
        )
        q_low = self.q_norm(self.wq_a(x))
        q = self.wq_b(q_low).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = q * torch.rsqrt(q.float().pow(2).mean(dim=-1, keepdim=True) + self.config.rms_norm_eps).to(dtype=q.dtype)
        kv = self.kv_norm(self.wkv(x)).view(batch, seq_len, 1, self.head_dim).transpose(1, 2)
        q = _apply_partial_rope(q, cos, sin, self.rope_head_dim)
        kv = _apply_partial_rope(kv, cos, sin, self.rope_head_dim)
        k = kv.expand(-1, self.num_heads, -1, -1)
        v = kv.expand(-1, self.num_heads, -1, -1)

        local_scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        local_mask = self._local_mask(seq_len, x.device).view(1, 1, seq_len, seq_len)
        local_scores = local_scores.masked_fill(~local_mask, -float("inf"))
        if attention_mask is not None:
            local_scores = local_scores + attention_mask.to(dtype=local_scores.dtype)

        score_parts = [local_scores]
        value_parts = [v]
        compressed = None
        if self.compressor is not None:
            compressed = self.compressor(
                x, position_ids=position_ids, rope_theta=self.config.compress_rope_theta
            )
        if compressed is not None:
            compressed_heads = compressed.expand(-1, self.num_heads, -1, -1)
            compressed_scores = torch.matmul(q, compressed_heads.transpose(-1, -2)) / (self.head_dim ** 0.5)
            n_compressed = compressed.size(2)
            visible = (torch.arange(seq_len, device=x.device) + 1) // self.compress_ratio
            c_pos = torch.arange(n_compressed, device=x.device).view(1, 1, 1, n_compressed)
            compressed_valid = c_pos < visible.view(1, 1, seq_len, 1)
            compressed_scores = compressed_scores.masked_fill(~compressed_valid, -float("inf"))
            if self.indexer is not None:
                indexer_out = self.indexer(
                    x, q_low, position_ids=position_ids, rope_theta=self.config.compress_rope_theta
                )
                if indexer_out is not None:
                    index_scores, topk_indices = indexer_out
                    topk_mask = torch.zeros_like(index_scores, dtype=torch.bool)
                    topk_mask.scatter_(-1, topk_indices, True)
                    compressed_scores = compressed_scores + index_scores.unsqueeze(1)
                    compressed_scores = compressed_scores.masked_fill(~topk_mask.unsqueeze(1), -float("inf"))
            score_parts.append(compressed_scores)
            value_parts.append(compressed_heads)

        scores = torch.cat(score_parts, dim=-1)
        sink = self.sinks.view(1, self.num_heads, 1, 1).expand(batch, -1, seq_len, -1).to(dtype=scores.dtype)
        probs = torch.softmax(torch.cat([scores, sink], dim=-1), dim=-1).to(dtype=q.dtype)

        context = q.new_zeros(batch, self.num_heads, seq_len, self.head_dim)
        offset = 0
        for part_idx, values in enumerate(value_parts):
            next_offset = offset + values.size(2)
            partial = torch.matmul(probs[..., offset:next_offset], values)
            if part_idx > 0:
                partial = _apply_partial_rope(partial, cos, -sin, self.rope_head_dim)
            context = context + partial
            offset = next_offset

        context = context.transpose(1, 2)
        grouped = context.reshape(batch, seq_len, self.config.o_groups, self.num_heads_per_group * self.head_dim)
        out = self.wo_b(self.wo_a(grouped).flatten(2))
        if cp_local_seq_len is not None:
            out = zigzag_slice_for_cp(out, self.ps.cp_rank, self.ps.cp_size, seq_dim=1)
        return out


class DeepseekV4Router(nn.Module):
    def __init__(self, config: DeepseekV4Config, *, is_hash_layer: bool):
        super().__init__()
        self.is_hash_layer = is_hash_layer
        self.topk = config.num_experts_per_tok
        self.route_scale = config.routed_scaling_factor
        self.norm_topk_prob = config.norm_topk_prob
        self.scoring_func = config.scoring_func
        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.hidden_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.is_hash_layer:
            self.register_buffer(
                "tid2eid",
                torch.zeros(config.vocab_size, self.topk, dtype=torch.int64),
                persistent=True,
            )
        else:
            self.register_buffer(
                "e_score_correction_bias",
                torch.zeros(config.n_routed_experts, dtype=torch.float32),
            )

    def forward(
        self,
        x: torch.Tensor,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = F.linear(x.float(), self.weight.float())
        if self.scoring_func == "sqrtsoftplus":
            scores = F.softplus(logits).sqrt()
        elif self.scoring_func == "sigmoid":
            scores = logits.sigmoid()
        else:
            scores = logits.softmax(dim=-1)

        if self.is_hash_layer and input_ids is not None:
            indices = self.tid2eid[input_ids.reshape(-1).to(torch.int64)]
        else:
            bias = getattr(self, "e_score_correction_bias", None)
            scores_for_choice = scores if bias is None else scores + bias.to(dtype=scores.dtype)
            indices = scores_for_choice.topk(self.topk, dim=-1, sorted=False).indices

        weights = scores.gather(1, indices)
        if self.norm_topk_prob and self.topk > 1:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return (weights * self.route_scale).to(dtype=x.dtype), indices


class DeepseekV4MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, swiglu_limit: float):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = _clamped_swiglu(self.gate_proj(x), self.up_proj(x), self.swiglu_limit)
        return self.down_proj(y.to(dtype=x.dtype))


class DeepseekV4MoE(nn.Module):
    def __init__(self, config: DeepseekV4Config, *, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.gate = DeepseekV4Router(config, is_hash_layer=layer_idx < config.num_hash_layers)
        self.experts = nn.ModuleList(
            [
                DeepseekV4MLP(config.hidden_size, config.moe_intermediate_size, config.swiglu_limit)
                for _ in range(config.n_routed_experts)
            ]
        )
        shared_intermediate = config.n_shared_experts * config.moe_intermediate_size
        self.shared_experts = (
            DeepseekV4MLP(config.hidden_size, shared_intermediate, config.swiglu_limit)
            if config.n_shared_experts > 0
            else None
        )

    def forward(self, x: torch.Tensor, *, input_ids: torch.Tensor | None = None) -> torch.Tensor:
        shape = x.shape
        x_flat = x.reshape(-1, self.hidden_size)
        weights, indices = self.gate(x_flat, input_ids=input_ids)
        out = torch.zeros_like(x_flat)
        for expert_id, expert in enumerate(self.experts):
            token_pos, slot = torch.where(indices == expert_id)
            if token_pos.numel() == 0:
                continue
            expert_out = expert(x_flat.index_select(0, token_pos))
            out.index_add_(0, token_pos, expert_out * weights[token_pos, slot].unsqueeze(-1))
        if self.shared_experts is not None:
            out = out + self.shared_experts(x_flat)
        return out.view(shape)


class DeepseekV4Layer(nn.Module):
    def __init__(self, config: DeepseekV4Config, layer_idx: int, ps: ParallelState):
        super().__init__()
        self.ps = ps
        self.self_attn = DeepseekV4Attention(config, layer_idx=layer_idx, ps=ps)
        self.mlp = DeepseekV4MoE(config, layer_idx=layer_idx)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_hc = DeepseekV4HyperConnection(
            config.hidden_size, config.hc_mult, config.hc_sinkhorn_iters, config.hc_eps
        )
        self.ffn_hc = DeepseekV4HyperConnection(
            config.hidden_size, config.hc_mult, config.hc_sinkhorn_iters, config.hc_eps
        )

    def _cp_full_layer_inputs(
        self,
        x: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        input_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, int | None]:
        position_ids = _expand_batch_position_ids(position_ids, x.size(0))
        if self.ps.cp_size <= 1:
            return x, position_ids, attention_mask, input_ids, None

        full_seq_len = position_ids.size(1)
        x, cp_local_seq_len = _full_tensor_for_cp(
            x,
            full_seq_len=full_seq_len,
            ps=self.ps,
        )
        if cp_local_seq_len is None:
            return x, position_ids, attention_mask, input_ids, None

        input_ids = _full_input_ids_for_cp(
            input_ids,
            local_seq_len=cp_local_seq_len,
            full_seq_len=full_seq_len,
            ps=self.ps,
        )
        if attention_mask is not None:
            if attention_mask.size(-1) != full_seq_len or attention_mask.size(-2) != full_seq_len:
                raise ValueError("DeepSeek V4 CP requires a full-sequence attention_mask.")
        return x, position_ids, attention_mask, input_ids, cp_local_seq_len

    def forward(
        self,
        x: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x, position_ids, attention_mask, input_ids, cp_local_seq_len = self._cp_full_layer_inputs(
            x,
            position_ids=position_ids,
            attention_mask=attention_mask,
            input_ids=input_ids,
        )
        residual = x
        attn_in, post, comb = self.attn_hc(x)
        attn_out = self.self_attn(
            self.input_layernorm(attn_in),
            position_ids=position_ids,
            attention_mask=attention_mask,
            cp_already_full=self.ps.cp_size > 1,
        )
        x = DeepseekV4HyperConnection.post(attn_out, residual, post, comb)

        residual = x
        ffn_in, post, comb = self.ffn_hc(x)
        ffn_out = self.mlp(self.post_attention_layernorm(ffn_in), input_ids=input_ids)
        out = DeepseekV4HyperConnection.post(ffn_out, residual, post, comb)
        if cp_local_seq_len is not None:
            out = zigzag_slice_for_cp(out, self.ps.cp_rank, self.ps.cp_size, seq_dim=1)
        return out


class DeepseekV4MTPBlock(DeepseekV4Layer):
    def __init__(self, config: DeepseekV4Config, layer_idx: int, ps: ParallelState):
        super().__init__(config, layer_idx=layer_idx, ps=ps)
        self.e_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.h_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hc_head = DeepseekV4HyperHead(config.hidden_size, config.hc_mult, config.hc_eps)

    def forward(
        self,
        x: torch.Tensor,
        *,
        input_ids: torch.Tensor,
        embed_tokens: nn.Embedding,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        position_ids = _expand_batch_position_ids(position_ids, x.size(0))
        cp_local_seq_len = None
        if self.ps.cp_size > 1:
            full_seq_len = position_ids.size(1)
            x, cp_local_seq_len = _full_tensor_for_cp(
                x,
                full_seq_len=full_seq_len,
                ps=self.ps,
            )
            if cp_local_seq_len is not None:
                input_ids = _full_input_ids_for_cp(
                    input_ids,
                    local_seq_len=cp_local_seq_len,
                    full_seq_len=full_seq_len,
                    ps=self.ps,
                )
        embedded = self.enorm(embed_tokens(input_ids))
        projected = self.e_proj(embedded).unsqueeze(2) + self.h_proj(self.hnorm(x))
        out = super().forward(
            projected,
            position_ids=position_ids,
            attention_mask=attention_mask,
            input_ids=input_ids,
        )
        if cp_local_seq_len is not None:
            out = zigzag_slice_for_cp(out, self.ps.cp_rank, self.ps.cp_size, seq_dim=1)
        return out

    def contract(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.hc_head(x))


class DeepseekV4Model(nn.Module):
    def __init__(self, config: DeepseekV4Config, ps: ParallelState):
        super().__init__()
        self.config = config
        self.ps = ps
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DeepseekV4Layer(config, layer_idx=i, ps=ps) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hc_head = DeepseekV4HyperHead(config.hidden_size, config.hc_mult, config.hc_eps)
        self.mtp = nn.ModuleList(
            [
                DeepseekV4MTPBlock(config, layer_idx=config.num_hidden_layers + i, ps=ps)
                for i in range(config.num_nextn_predict_layers)
            ]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        hidden = self.embed_tokens(input_ids)
        hidden = hidden.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
        batch, seq_len = input_ids.shape
        if position_ids is None:
            full_seq_len = seq_len * self.ps.cp_size
            position_ids = torch.arange(full_seq_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        else:
            position_ids = _expand_batch_position_ids(position_ids, batch)
        for layer in self.layers:
            hidden = layer(
                hidden,
                position_ids=position_ids,
                attention_mask=attention_mask,
                input_ids=input_ids,
            )
        full_seq_len = position_ids.size(1)
        hidden, cp_local_seq_len = _full_tensor_for_cp(
            hidden,
            full_seq_len=full_seq_len,
            ps=self.ps,
        )
        hidden = self.norm(self.hc_head(hidden))
        if cp_local_seq_len is not None:
            hidden = zigzag_slice_for_cp(hidden, self.ps.cp_rank, self.ps.cp_size, seq_dim=1)
        return hidden


class DeepseekV4ForCausalLM(nn.Module):
    def __init__(
        self,
        config: DeepseekV4Config,
        train_cfg: SimpleNamespace | None = None,
        ps: ParallelState | None = None,
    ):
        super().__init__()
        self.config = config
        self.train_cfg = train_cfg or SimpleNamespace(fp8=False)
        self.ps = ps or ParallelState()
        self.model = DeepseekV4Model(config, self.ps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._input_tensor: torch.Tensor | None = None

    def set_input_tensor(self, input_tensor):
        if isinstance(input_tensor, list):
            input_tensor = input_tensor[0] if input_tensor else None
        self._input_tensor = input_tensor

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        hidden_states: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if hidden_states is not None:
            raise NotImplementedError("DeepSeek V4 lite does not support pipeline hidden-state injection yet.")
        if input_ids is None:
            raise ValueError("input_ids is required")
        fp8_ctx = nullcontext()
        with fp8_ctx:
            hidden = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
            full_seq_len = hidden.size(1)
            if self.ps.cp_size > 1:
                if position_ids is not None:
                    pos = _expand_batch_position_ids(position_ids, hidden.size(0))
                    full_seq_len = pos.size(1)
                else:
                    full_seq_len = input_ids.size(1) * self.ps.cp_size
            hidden_for_head, cp_local_seq_len = _full_tensor_for_cp(
                hidden,
                full_seq_len=full_seq_len,
                ps=self.ps,
            )
            logits = self.lm_head(hidden_for_head)
            if cp_local_seq_len is not None:
                logits = zigzag_slice_for_cp(logits, self.ps.cp_rank, self.ps.cp_size, seq_dim=1)
        output = {"hidden_states": hidden, "logits": logits}
        if labels is not None:
            labels = _labels_for_local_logits(
                labels,
                local_seq_len=logits.size(1),
                ps=self.ps,
            )
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                labels.reshape(-1),
                ignore_index=-100,
            )
            output["loss"] = loss
        return output
