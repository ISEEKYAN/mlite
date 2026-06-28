# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
DSA kernel wrappers for Megatron's DSv4 sparse attention.

Mirrors the three integration paths of the old standalone ``dsa_kernels``
package, but built on top of

* :mod:`cudnn.deepseek_sparse_attention` (a.k.a. ``DSA``) — CuTe-DSL backward
  + indexer score kernels + TRT-LLM radix top-K, shipped as part of
  cuDNN Frontend.
* :mod:`flash_mla` — production sparse-attention forward kernel, expected to
  be available as a separate PyPI package.

Public API (same shape as the old ``dsa_kernels`` package):

* ``build_flat_topk_idxs`` / ``local_to_global_flat`` — index helpers.
* ``dsa_sparse_attn`` — Path A / Path C step 2, differentiable sparse attention.
* ``indexer_topk`` — Path C inference indexer scoring + top-K.
* ``fused_indexer_sparse_attn`` — Path B training, fused indexer loss +
  sparse attention with shared backward.
"""

from __future__ import annotations

from importlib import import_module
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Lazy kernel imports
# ---------------------------------------------------------------------------


_flash_mla_sparse_fwd = None
_DSA = None
_indexer_fwd_sm90: Optional[Callable] = None
_indexer_fwd_sm100: Optional[Callable] = None

_SCORE_MEMORY_CHECK_THRESHOLD_BYTES = 1024**3
_SCORE_MEMORY_MAX_FREE_FRACTION = 0.70
_DENSE_LSE_MAX_SCORE_BYTES = 256 * 1024 * 1024
_DENSE_KL_MAX_TEMP_BYTES = 256 * 1024 * 1024


def _bottom_right_valid_kv_counts(
    seq_q: int, seq_k: int, ratio: int, device: torch.device
) -> Tensor:
    """Return the cuDNN DSA bottom-right causal KV count for every query."""

    if ratio < 1 or seq_q > seq_k * ratio:
        raise ValueError(
            "DSA bottom-right causal mask requires ratio >= 1 and "
            f"seq_q <= seq_k * ratio, got seq_q={seq_q}, seq_k={seq_k}, ratio={ratio}"
        )
    q_positions = torch.arange(seq_q, device=device, dtype=torch.int64)
    q_global_start = seq_k * ratio - seq_q
    return torch.div(
        q_global_start + q_positions + 1, ratio, rounding_mode="floor"
    ).clamp(min=0, max=seq_k)


def _estimate_dsa_score_peak_bytes(
    batch: int, seq_q: int, seq_k: int, *, dense_loss: bool
) -> int:
    """Estimate unavoidable FP32 full-score storage for the current kernels."""

    score_bytes = batch * seq_q * seq_k * torch.float32.itemsize
    score_matrices = 2 if dense_loss else 1
    dense_scratch = max(_DENSE_LSE_MAX_SCORE_BYTES, _DENSE_KL_MAX_TEMP_BYTES)
    return score_matrices * score_bytes + (dense_scratch if dense_loss else 0)


def _guard_dsa_score_memory(
    tensor: Tensor, batch: int, seq_q: int, seq_k: int, *, dense_loss: bool
) -> None:
    """Fail before a predictable full-score OOM with an actionable message."""

    estimated = _estimate_dsa_score_peak_bytes(
        batch, seq_q, seq_k, dense_loss=dense_loss
    )
    if not tensor.is_cuda or estimated < _SCORE_MEMORY_CHECK_THRESHOLD_BYTES:
        return
    free_bytes, _total_bytes = torch.cuda.mem_get_info(tensor.device)
    allowed = int(free_bytes * _SCORE_MEMORY_MAX_FREE_FRACTION)
    if estimated > allowed:
        gib = 1024**3
        mode = "dense auxiliary loss" if dense_loss else "indexer top-k"
        raise RuntimeError(
            f"DSA {mode} would materialize full FP32 score tensors with an "
            f"estimated peak of {estimated / gib:.1f} GiB for "
            f"(batch={batch}, seq_q={seq_q}, seq_k={seq_k}), but only "
            f"{free_bytes / gib:.1f} GiB is currently free. Current MLite DSA "
            "still materializes the full top-k score matrix and, for dense "
            "KL, two full score matrices plus bounded Q/K-chunk scratch; "
            "reduce the training sequence length instead of relying on a "
            "CUDA OOM."
        )


def _ensure_flash_mla():
    """Lazily import the FlashMLA sparse-forward kernel.

    FlashMLA ships ``flash_mla_sparse_fwd`` with a multi-head-KV signature;
    :func:`_dsa_fwd_flash_mla` below is a thin adapter that unbatches the
    DSA-shape inputs and pads ``TopK`` to the alignment expected by
    FlashMLA's SM90 / SM100 kernels.
    """
    global _flash_mla_sparse_fwd
    if _flash_mla_sparse_fwd is not None:
        return

    try:
        from flash_mla import flash_mla_sparse_fwd as _fwd
    except ImportError as e:
        raise ImportError(
            "FlashMLA is required for DSA sparse attention forward. "
            "Install from https://github.com/deepseek-ai/FlashMLA/tree/nv_dev "
            "so that `from flash_mla import flash_mla_sparse_fwd` succeeds."
        ) from e
    _flash_mla_sparse_fwd = _fwd


def _get_topk_alignment() -> int:
    """Minimum ``TopK`` alignment required by the current GPU architecture.

    * SM90 : dual-warpgroup loop steps by 2 blocks → ``2 * B_TOPK = 128``
    * SM100: single-pipeline loop steps by 1 block → ``B_TOPK`` (64 for
      head64, 128 for head128). DSA uses ``D = 512`` which maps to the
      head64 kernel path → 64.
    """
    sm = torch.cuda.get_device_capability()
    if sm[0] >= 10:
        return 64
    return 128


def _dsa_fwd_flash_mla(
    q: Tensor,
    kv: Tensor,
    topk_idxs: Tensor,
    softmax_scale: float,
    d_v: int = 512,
    attn_sink: Optional[Tensor] = None,
    topk_length: Optional[Tensor] = None,
    indexer_topk: int = 0,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """DSA-shaped adapter around :func:`flash_mla.flash_mla_sparse_fwd`.

    Accepts flat (unbatched) tensors with global indices; pads ``TopK`` to
    the GPU-specific alignment; returns ``(out, lse, lse_indexer)``.
    """
    assert not (
        indexer_topk > 0 and topk_length is not None
    ), "indexer_topk > 0 requires non-compact mode (topk_length must be None)"
    _ensure_flash_mla()

    _total_S_q, _H, _D = q.shape
    TopK = topk_idxs.shape[-1]
    topk_align = _get_topk_alignment()
    TopK_padded = (TopK + topk_align - 1) // topk_align * topk_align
    if TopK_padded != TopK:
        pad_width = TopK_padded - TopK
        topk_idxs = torch.nn.functional.pad(topk_idxs, (0, pad_width), value=-1)

    kv_3d = kv.unsqueeze(1)  # (total_S_kv, 1, D)  h_kv=1
    indices = topk_idxs.unsqueeze(1)  # (total_S_q, 1, TopK_padded) h_kv=1

    with torch.cuda.nvtx.range("flash_mla_sparse_fwd"):
        res = _flash_mla_sparse_fwd(
            q,
            kv_3d,
            indices,
            softmax_scale,
            d_v=d_v,
            attn_sink=attn_sink,
            topk_length=topk_length,
            indexer_topk=indexer_topk,
        )
        if indexer_topk > 0:
            out, _max_logits, lse, lse_indexer = res
        else:
            out, _max_logits, lse = res
            lse_indexer = None

    if indexer_topk > 0:
        # When indexer_topk == total TopK, lse_indexer should equal lse but
        # the kernel may not snapshot correctly; fall back to lse.
        if indexer_topk >= TopK:
            return out, lse, lse.clone()
        return out, lse, lse_indexer
    return out, lse, None


def _ensure_dsa_namespace():
    """Lazily import the cudnn-frontend DSA namespace."""
    global _DSA
    if _DSA is not None:
        return
    try:
        from cudnn import DSA as _ns
    except ImportError as e:
        try:
            from cudnn.deepseek_sparse_attention import DSA as _ns
        except ImportError:
            raise ImportError(
                "cudnn-frontend DSA namespace not available. Install with "
                "`pip install nvidia-cudnn-frontend[cutedsl]`; newer "
                "versions expose it as `cudnn.deepseek_sparse_attention.DSA`."
            ) from e
    _DSA = _ns


def _load_indexer_fwd_sm90():
    """Load the H100 SM90 indexer forward entry only when it is selected."""
    global _indexer_fwd_sm90
    if _indexer_fwd_sm90 is None:
        try:
            module = import_module(
                "cudnn.deepseek_sparse_attention.indexer_forward._interface_sm90"
            )
            _indexer_fwd_sm90 = module.indexer_fwd
        except (AttributeError, ImportError) as exc:
            raise ImportError(
                "H100 DSA indexer forward requires the SM90 cudnn route "
                "`cudnn.deepseek_sparse_attention.indexer_forward._interface_sm90.indexer_fwd`."
            ) from exc
    return _indexer_fwd_sm90


def _load_indexer_fwd_sm100():
    """Load the Blackwell SM100 indexer forward entry only when it is selected."""
    global _indexer_fwd_sm100
    if _indexer_fwd_sm100 is None:
        try:
            module = import_module(
                "cudnn.deepseek_sparse_attention.indexer_forward._interface"
            )
            _indexer_fwd_sm100 = module.indexer_fwd
        except (AttributeError, ImportError) as exc:
            raise ImportError(
                "Blackwell DSA indexer forward requires the SM100 cudnn route "
                "`cudnn.deepseek_sparse_attention.indexer_forward._interface.indexer_fwd` "
                "(exported by cudnn-frontend as indexer_fwd_sm100)."
            ) from exc
    return _indexer_fwd_sm100


def _select_indexer_forward(device):
    major, _minor = torch.cuda.get_device_capability(device)
    if major == 9:
        return _load_indexer_fwd_sm90()
    if major >= 10:
        return _load_indexer_fwd_sm100()
    return None


def _dsa_indexer_forward_wrapper(
    q: Tensor,
    k: Tensor,
    w: Tensor,
    *,
    ratio: int = 4,
    qhead_per_kv_head: Optional[int] = None,
    sm_scale: float = 1.0,
    cu_seqlens_q: Optional[Tensor] = None,
    cu_seqlens_k: Optional[Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
):
    """Route indexer forward to architecture-specific CuTe-DSL backends."""
    if q.is_cuda:
        indexer_fwd = _select_indexer_forward(q.device)
        if indexer_fwd is not None:
            return {
                "scores": indexer_fwd(
                    q,
                    k,
                    w,
                    ratio=ratio,
                    qhead_per_kv_head=qhead_per_kv_head,
                    sm_scale=sm_scale,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                )
            }
    _ensure_dsa_namespace()
    return _DSA.indexer_forward_wrapper(
        q,
        k,
        w,
        ratio=ratio,
        qhead_per_kv_head=qhead_per_kv_head,
        sm_scale=sm_scale,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    )


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------


def local_to_global_flat(local_idxs: Tensor, batch_size: int, seqlen_kv: int) -> Tensor:
    """Convert local per-batch indices to global flat indices.

    Follows the convention used by FlashMLA / SparseAttentionBackward:
    flat row order is SBHD ``row[s * B + b]``; global index is
    ``local * B + b`` for valid entries and ``-1`` otherwise.

    Args:
        local_idxs: ``(b, sq, topk)`` int, values in ``[0, seqlen_kv)`` or -1.
        batch_size: ``B``.
        seqlen_kv: KV sequence length per batch (used for shape assertions
            only; callers compute the values).

    Returns:
        ``(sq*b, topk)`` int32.
    """
    b, sq, topk = local_idxs.shape
    assert b == batch_size

    idxs_sb = local_idxs.permute(1, 0, 2).reshape(sq * b, topk)
    valid = idxs_sb >= 0
    batch_ids = torch.arange(sq * b, device=local_idxs.device) % b
    batch_ids_exp = batch_ids.unsqueeze(1).expand_as(idxs_sb)
    idxs_sb = torch.where(valid, idxs_sb * b + batch_ids_exp, idxs_sb)
    return idxs_sb.int()


def build_flat_topk_idxs(
    *idx_groups: Tensor, batch_size: int, seqlen_kv: int, compact: bool = False
) -> Tuple[Tensor, Optional[Tensor]]:
    """Combine local per-batch index groups and convert to flat global form.

    Each *idx_group* is ``(b, sq, topk_i)`` with local per-batch KV indices
    (already in ``kv_full`` index space, i.e. with any compressed-position
    offset applied). ``-1`` marks invalid positions.

    Args:
        *idx_groups: one or more ``(b, sq, topk_i)`` int tensors.
        batch_size: ``B``.
        seqlen_kv: total KV sequence length per batch.
        compact: if True, pack valid entries to the front of each row and
            additionally return ``topk_length``; if False, leave as-is and
            return ``None``.

    Returns:
        ``(topk_idxs, topk_length)`` where
        ``topk_idxs`` is ``(sq*b, total_topk)`` int32 (flat global) and
        ``topk_length`` is ``(sq*b,)`` int32 when ``compact``, else ``None``.
    """
    combined = torch.cat(idx_groups, dim=-1)  # (b, sq, total_topk)
    b, sq, total_topk = combined.shape

    # Globalize first, compact second. Both ops are element-wise + (-1)-preserving,
    # so swapping the order is a no-op for correctness; the win is that the
    # global indices come out already in (sq*b, total_topk) flat layout, which is
    # exactly the row order the cuDNN compactify kernel returns its per-row
    # ``length`` in — no extra permute on the length tensor.
    global_idxs = local_to_global_flat(combined, b, seqlen_kv)

    topk_length_flat = None
    if compact:
        if global_idxs.is_cuda:
            # Fast path: single warp-per-row CuTe DSL kernel from cuDNN's DSA
            # namespace. Replaces a stable argsort + gather + sum + permute
            # chain with one global-load + global-store per element.
            _ensure_dsa_namespace()
            res = _DSA.compactify_wrapper(global_idxs)
            global_idxs, topk_length_flat = res["indices"], res["topk_length"]
        else:
            # CPU fallback so the unit tests that exercise this helper without
            # CUDA still work. Production callers always go through the CUDA
            # path above.
            valid_mask = global_idxs >= 0
            sorted_indices = valid_mask.int().argsort(
                dim=-1, descending=True, stable=True
            )
            global_idxs = global_idxs.gather(-1, sorted_indices)
            topk_length_flat = valid_mask.sum(dim=-1).int()

    return global_idxs, topk_length_flat


# ---------------------------------------------------------------------------
# Path A + Path C step 2: differentiable sparse attention
# ---------------------------------------------------------------------------


class SparseAttnFunc(torch.autograd.Function):
    """SM100 sparse attention fwd + bwd on flat tensors.

    Forward uses :mod:`flash_mla`; backward uses cuDNN Frontend's
    :attr:`cudnn.DSA.sparse_attention_backward_wrapper`.
    """

    @staticmethod
    def forward(
        ctx,
        q: Tensor,  # (total_sq, H, D) bf16
        kv: Tensor,  # (total_skv, D) bf16
        attn_sink: Tensor,  # (H,) f32
        topk_idxs: Tensor,  # (total_sq, TopK) int32 global
        topk_length: Optional[Tensor],  # (total_sq,) int32 or None
        softmax_scale: float,
        indexer_topk: int,
        value_dim: Optional[int],
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """Run FlashMLA sparse-attention forward and save tensors for backward."""
        out, lse, lse_indexer = _dsa_fwd_flash_mla(
            q,
            kv,
            topk_idxs,
            softmax_scale,
            attn_sink=attn_sink,
            topk_length=topk_length,
            indexer_topk=indexer_topk,
            d_v=512 if value_dim is None else value_dim,
        )

        ctx.save_for_backward(q, kv, attn_sink, topk_idxs, out, lse)
        ctx.softmax_scale = softmax_scale
        ctx.topk_length = topk_length
        return out, lse, lse_indexer

    @staticmethod
    def backward(ctx, dO, d_lse, d_lse_indexer):
        """Compute sparse-attention backward via cuDNN DSA wrapper."""
        _ensure_dsa_namespace()

        q, kv, attn_sink, topk_idxs, out, lse = ctx.saved_tensors

        result = _DSA.sparse_attention_backward_wrapper(
            q,
            kv,
            out,
            dO,
            lse,
            attn_sink,
            topk_idxs,
            softmax_scale=ctx.softmax_scale,
            topk_length=ctx.topk_length,
        )
        dq, dkv, d_sink = result["dq"], result["dkv"], result["d_sink"]
        return dq, dkv, d_sink, None, None, None, None, None


def dsa_sparse_attn(
    query: Tensor,
    kv: Tensor,
    attn_sink: Tensor,
    topk_idxs: Tensor,
    softmax_scale: float,
    topk_length: Optional[Tensor] = None,
    indexer_topk: int = 0,
    value_dim: Optional[int] = None,
) -> Tensor:
    """Sparse attention (Path A / Path C step 2).

    Args:
        query: ``(sq, b, np, d)`` bf16 SBHD.
        kv:    ``(skv, b, d)`` bf16 SBD (K=V).
        attn_sink: ``(np,)`` f32.
        topk_idxs: ``(sq*b, topk)`` int32 — **flat global** indices produced
            by :func:`build_flat_topk_idxs`.
        softmax_scale: scalar float.
        topk_length: ``(sq*b,)`` int32 — optional compact fast-path. Must be
            ``None`` when ``indexer_topk > 0`` (FlashMLA constraint).
        indexer_topk: int; ``0`` for Paths A/C, positive for Path B to enable
            FlashMLA's ``lse_indexer`` output.
        value_dim: FlashMLA value dimension. Defaults to ``512`` to preserve
            the existing DSA wrapper behavior.

    Returns:
        ``(sq, b, np * d_v)`` bf16 output.
    """
    sq, b, np_, d = query.shape
    skv = kv.shape[0]

    q_flat = query.reshape(sq * b, np_, d)
    kv_flat = kv.reshape(skv * b, d)

    out_flat, _lse, _lse_indexer = SparseAttnFunc.apply(
        q_flat,
        kv_flat,
        attn_sink,
        topk_idxs,
        topk_length,
        softmax_scale,
        indexer_topk,
        value_dim,
    )

    d_v = out_flat.shape[-1]
    return out_flat.reshape(sq, b, np_, d_v).reshape(sq, b, np_ * d_v)


# ---------------------------------------------------------------------------
# Path C inference: indexer scoring + top-K
# ---------------------------------------------------------------------------


def _indexer_topk_bshd(
    q_bshd: Tensor, k_bsd: Tensor, w_bsh: Tensor, topk: int, ratio: int = 4
) -> Tuple[Tensor, Tensor, Tensor]:
    """BSHD-layout core for :func:`indexer_topk`.

    Internal entry point used by both the public SBHD wrapper and Path B's
    ``FusedIndexerSparseAttnFunc.forward`` so the SBHD→BSHD permute can be
    performed once at the call site and reused across both the indexer
    forward and the score-backward kernels (predict / target).

    Args:
        q_bshd: ``(b, sq, idx_nh, idx_hd)`` bf16, C-contiguous.
        k_bsd:  ``(b, sk, idx_hd)`` bf16, C-contiguous.
        w_bsh:  ``(b, sq, idx_nh)`` bf16, C-contiguous, **already
            ``indexer_softmax_scale``-scaled** by the caller.
        topk:   number of top-K indices to return per query.
        ratio:  compression ratio for the kernel's causal mask.

    Returns:
        ``(topk_indices, topk_length, scores)`` where:

        * ``topk_indices``: ``(b, sq, topk)`` int32, invalid slots ``-1``.
        * ``topk_length``:  ``(b, sq)`` int32, per-row valid count.
        * ``scores``: ``(b, sq, sk)`` fp32, raw scores from
          :attr:`cudnn.DSA.indexer_forward_wrapper` with ``-inf`` on
          causally-masked positions.
    """
    _ensure_dsa_namespace()

    b, sq, _idx_nh, _idx_hd = q_bshd.shape
    sk = k_bsd.shape[1]
    device = q_bshd.device
    valid_per_q = _bottom_right_valid_kv_counts(sq, sk, ratio, device).to(torch.int32)
    _guard_dsa_score_memory(q_bshd, b, sq, sk, dense_loss=False)

    k_bshd = k_bsd.unsqueeze(2)  # (b, sk, 1, idx_hd)

    scores = _dsa_indexer_forward_wrapper(q_bshd, k_bshd, w_bsh, ratio=ratio)[
        "scores"
    ]  # (b, sq, sk) fp32, -inf on masked positions

    # Top-K selection via the TRT-LLM CuTe-DSL radix kernel.
    n_rows = b * sq
    scores_flat = scores.reshape(n_rows, sk).contiguous()
    seq_lens = valid_per_q.repeat(b)  # (b*sq,), row-major over (b, sq)

    topk_k = min(topk, sk)
    tk_result = _DSA.indexer_top_k_wrapper(
        scores_flat, seq_lens, top_k=topk_k, next_n=1, return_val=False
    )
    topk_indices = tk_result["indices"].view(b, sq, topk_k)

    if topk_k < topk:
        pad = torch.full((b, sq, topk - topk_k), -1, dtype=torch.int32, device=device)
        topk_indices = torch.cat([topk_indices, pad], dim=-1)

    topk_length = (topk_indices >= 0).sum(dim=-1).int()  # (b, sq)
    return topk_indices.int(), topk_length, scores


def _sbhd_to_bshd_indexer_inputs(
    q_indexer: Tensor, k_indexer: Tensor, weights: Tensor, indexer_softmax_scale: float
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Permute the indexer inputs SBHD→BSHD once, returning both the raw
    BSHD weights and (when needed) a separate scaled copy.

    The ``relu(c·x) = c·relu(x)`` trick lets us push the indexer softmax
    scale onto ``W`` (``(B, S_q, H)``, small) instead of the score tensor
    (``(B, S_q, S_k)``, big). The raw ``w_bsh`` is preserved for the
    backward GEMM path, which takes ``sm_scale`` directly. When
    ``indexer_softmax_scale == 1.0`` the two views alias each other.

    Returns ``(q_bshd, k_bsd, w_bsh, w_bsh_scaled)``.
    """
    q_bshd = q_indexer.permute(1, 0, 2, 3).contiguous()
    k_bsd = k_indexer.permute(1, 0, 2).contiguous()
    w_bsh = weights.permute(1, 0, 2).contiguous()

    if indexer_softmax_scale != 1.0:
        w_bsh_scaled = (w_bsh.float() * indexer_softmax_scale).to(w_bsh.dtype)
    else:
        w_bsh_scaled = w_bsh

    return q_bshd, k_bsd, w_bsh, w_bsh_scaled


def indexer_topk(
    q_indexer: Tensor,
    k_indexer: Tensor,
    weights: Tensor,
    topk: int,
    ratio: int = 4,
    indexer_softmax_scale: float = 1.0,
) -> Tuple[Tensor, Tensor]:
    """Score + top-K selection for inference (no KL loss, no backward).

    Built on cuDNN Frontend's CuTe-DSL indexer forward kernel followed by
    TRT-LLM's radix top-K kernel.

    Args:
        q_indexer: ``(sq, b, idx_nh, idx_hd)`` bf16 SBHD.
        k_indexer: ``(sk, b, idx_hd)`` bf16 SBD.
        weights:   ``(sq, b, idx_nh)`` bf16 SBH — raw (unscaled) weights.
        topk: number of top-K indices to select.
        ratio: compression ratio for the causal mask.
        indexer_softmax_scale: scale applied to the indexer ``Q @ K^T``
            scores (typically ``idx_hd ** -0.5``). Applied internally via
            the weights-scaling trick (``relu(c·x) = c·relu(x)`` for
            ``c > 0``) so the caller passes raw weights. Default ``1.0``
            means weights are treated as already-scaled.

    Returns:
        topk_indices: ``(b, sq, topk)`` int32 — local per-batch indices into
            ``k_indexer``; invalid positions are ``-1``.
        topk_length:  ``(b, sq)`` int32 — per-query valid count.
    """
    q_bshd, k_bsd, _w_bsh_raw, w_bsh_scaled = _sbhd_to_bshd_indexer_inputs(
        q_indexer, k_indexer, weights, indexer_softmax_scale
    )
    topk_indices, topk_length, _ = _indexer_topk_bshd(
        q_bshd, k_bsd, w_bsh_scaled, topk, ratio
    )
    return topk_indices, topk_length


# ---------------------------------------------------------------------------
# Path B: fused indexer + sparse attention (training)
# ---------------------------------------------------------------------------


_CANONICAL_KL_EPS = 1.0e-10


def _compute_indexer_predict(
    q_indexer_bshd: Tensor,
    k_indexer_bsd: Tensor,
    weights_bsh: Tensor,
    topk_indices: Tensor,
    qhead_per_kv_head: int,
) -> Tensor:
    """Compute ``predict`` distribution (softmax over top-K of indexer scores).

    Wraps :attr:`cudnn.DSA.sparse_indexer_score_recompute_wrapper`.

    Args:
        q_indexer_bshd: ``(B, S_q, H_q, D)`` bf16.
        k_indexer_bsd:  ``(B, S_k, D)`` bf16.
        weights_bsh:    ``(B, S_q, H_q)`` bf16.
        topk_indices:   ``(B, S_q, topk)`` int32.
        qhead_per_kv_head: ``H_q`` (MQA).

    Returns:
        predict: ``(B, S_q, topk)`` fp32, softmax over the top-K axis.
    """
    _ensure_dsa_namespace()
    result = _DSA.sparse_indexer_score_recompute_wrapper(
        q_indexer_bshd,
        k_indexer_bsd,
        weights_bsh,
        topk_indices,
        qhead_per_kv_head=qhead_per_kv_head,
    )
    return result["predict"]


def _compute_attn_target(
    q_attn_bshd: Tensor,
    k_attn_bsd: Tensor,
    lse: Tensor,
    topk_indices: Tensor,
    softmax_scale: float,
    qhead_per_kv_head: int,
) -> Tensor:
    """Compute ``target`` distribution (L1-normalised head-sum softmax).

    Wraps :attr:`cudnn.DSA.sparse_attn_score_recompute_wrapper`.

    Shapes match :func:`_compute_indexer_predict`; ``lse`` is
    ``(B, S_q, H_q)`` FP32 (comes from the attention forward pass).
    """
    _ensure_dsa_namespace()
    result = _DSA.sparse_attn_score_recompute_wrapper(
        q_attn_bshd,
        k_attn_bsd,
        lse,
        topk_indices,
        softmax_scale,
        qhead_per_kv_head=qhead_per_kv_head,
    )
    return result["target"]


def _kl_loss_from_target_predict(
    target: Tensor,
    predict: Tensor,
    topk_indices: Tensor,
    loss_coeff: float,
    calculate_per_token_loss: bool = False,
) -> Tensor:
    """KL(target || predict) reduced over ``(B, S_q)`` and scaled by loss_coeff.

    Rows with no valid top-K positions (early query rows with ratio causal
    masking) contribute 0 to the loss — the sparse score kernels produce
    garbage for those rows, mirroring ``compute_dsa_indexer_loss``'s
    ``row_valid`` handling. The default mean is taken over all ``(B, S_q)``
    positions. Per-token-loss mode returns a raw local sum so finalize can
    apply the global token divisor.
    """
    eps = _CANONICAL_KL_EPS
    kl_per_row = (target * (torch.log(target + eps) - torch.log(predict + eps))).sum(
        dim=-1
    )  # (B, S_q)

    row_valid = (topk_indices >= 0).any(dim=-1)  # (B, S_q)
    kl_per_row = torch.where(row_valid, kl_per_row, torch.zeros_like(kl_per_row))
    loss = kl_per_row.sum() if calculate_per_token_loss else kl_per_row.mean()
    return loss_coeff * loss


def _scale_target_for_canonical_log_eps_backward_(
    target: Tensor, predict: Tensor
) -> Tensor:
    """Adapt a vendor ``-target`` score-grad to ``-target*P/(P+eps)``."""

    return target.mul_(predict / (predict + _CANONICAL_KL_EPS))


def _scale_dense_target_for_canonical_log_eps_backward_(
    attn_score: Tensor, index_score: Tensor, index_lse: Tensor, ratio: int
) -> Tensor:
    """Apply the canonical log-epsilon score-grad factor in bounded blocks."""

    batch, seq_q, seq_k = index_score.shape
    valid_kv = _bottom_right_valid_kv_counts(seq_q, seq_k, ratio, index_score.device)
    queries_per_block, keys_per_block = _dense_kl_block_shape(batch, seq_q, seq_k)
    for q_start in range(0, seq_q, queries_per_block):
        q_end = min(q_start + queries_per_block, seq_q)
        block_valid_kv = valid_kv[q_start:q_end]
        for k_start in range(0, seq_k, keys_per_block):
            k_end = min(k_start + keys_per_block, seq_k)
            k_positions = torch.arange(
                k_start, k_end, device=index_score.device, dtype=torch.int64
            )
            position_valid = k_positions.view(1, -1) < block_valid_kv.view(-1, 1)
            predict = torch.exp(
                index_score[:, q_start:q_end, k_start:k_end]
                - index_lse[:, q_start:q_end].unsqueeze(-1)
            )
            factor = predict / (predict + _CANONICAL_KL_EPS)
            target_block = attn_score[:, q_start:q_end, k_start:k_end]
            target_block.mul_(
                torch.where(
                    position_valid.unsqueeze(0), factor, torch.zeros_like(factor)
                )
            )
    return attn_score


# ---------------------------------------------------------------------------
# Dense path (``sparse_loss=False``) — full-KV indexer loss
# ---------------------------------------------------------------------------


def _compute_dense_indexer_score(
    q_indexer_bshd: Tensor,
    k_indexer_bshd: Tensor,
    weights_bsh: Tensor,
    qhead_per_kv_head: int,
    indexer_softmax_scale: float,
    ratio: int,
) -> Tuple[Tensor, Tensor]:
    """Dense indexer score forward over the full ``S_k`` axis.

    Wraps :attr:`cudnn.DSA.dense_indexer_score_recompute_wrapper`. Returns
    ``(out, denom)`` where

    * ``out``    : ``(B, S_q, S_k)`` fp32, the raw head-reduced score
      ``S[b,q,k] = indexer_softmax_scale * sum_h ReLU(Q_h · K_k^T) · W_{b,q,h}``
      with the kernel's ``ratio``-causal mask applied to invalid columns.
    * ``denom``  : ``(B, S_q)`` fp32, the LSE denom of ``out`` along
      ``S_k`` — i.e. ``predict = exp(out - denom[..., None])`` is the
      indexer softmax distribution over the full KV.

    Both outputs are forwarded into :func:`_kl_loss_from_dense_scores`
    *and* saved for the dense-path backward, where the dense indexer-grad
    kernel consumes them directly.
    """
    _ensure_dsa_namespace()
    result = _DSA.dense_indexer_score_recompute_wrapper(
        q_indexer_bshd,
        k_indexer_bshd,
        weights_bsh,
        qhead_per_kv_head=qhead_per_kv_head,
        sm_scale=indexer_softmax_scale,
        ratio=ratio,
    )
    return result["out"], result["denom"]


def _compute_dense_attn_score(
    q_attn_bshd: Tensor,
    k_attn_bshd: Tensor,
    lse: Tensor,
    qhead_per_kv_head: int,
    softmax_scale: float,
    ratio: int,
) -> Tuple[Tensor, Tensor]:
    """Dense attention score forward over the full ``S_k`` axis.

    Wraps :attr:`cudnn.DSA.dense_attn_score_recompute_wrapper`. Returns
    ``(out, denom)`` where

    * ``out``   : ``(B, S_q, S_k)`` fp32, the head-summed unnormalized
      attention probability ``S[b,q,k] = sum_h exp(Q_h · K_k^T · scale - LSE[b,q,h])``
      with ``ratio`` causal mask applied.
    * ``denom`` : ``(B, S_q)`` fp32, the L1-norm denom ``sum_k S[b,q,:]``.
      ``target = out / denom[..., None]`` is the L1-normalized
      head-summed attention distribution.
    """
    _ensure_dsa_namespace()
    result = _DSA.dense_attn_score_recompute_wrapper(
        q_attn_bshd,
        k_attn_bshd,
        lse,
        softmax_scale,
        qhead_per_kv_head=qhead_per_kv_head,
        ratio=ratio,
    )
    return result["out"], result["denom"]


def _compute_full_causal_attn_lse(
    q_attn_bshd: Tensor,
    k_attn_bshd: Tensor,
    softmax_scale: float,
    ratio: int,
    *,
    max_score_bytes: int = _DENSE_LSE_MAX_SCORE_BYTES,
) -> Tensor:
    """Return the full-causal, per-head attention LSE used by dense DSA KL.

    FlashMLA's ``lse_indexer`` only covers the selected sparse indices.  It
    therefore cannot normalize the canonical dense target, where every
    attention head first softmaxes over *all* causally valid KV positions.
    Materializing ``(B, S_q, H_q, S_k)`` is prohibitive, so this helper
    recomputes the exact FP32 LSE in bounded query blocks.

    The bottom-right ratio mask matches cuDNN DSA score-recompute. Fully
    masked rows receive FlashMLA's ``+inf`` lonely-query sentinel; the
    downstream dense score kernel masks every KV entry in those rows and
    reports a zero L1 norm.
    """

    if q_attn_bshd.ndim != 4 or k_attn_bshd.ndim != 4:
        raise ValueError(
            "dense DSA LSE expects BSHD q/k tensors, got "
            f"q={tuple(q_attn_bshd.shape)} k={tuple(k_attn_bshd.shape)}"
        )
    if max_score_bytes < torch.empty((), dtype=torch.float32).element_size():
        raise ValueError(f"max_score_bytes must be positive, got {max_score_bytes}")

    batch, seq_q, q_heads, head_dim = q_attn_bshd.shape
    k_batch, seq_k, kv_heads, k_head_dim = k_attn_bshd.shape
    if batch != k_batch or head_dim != k_head_dim:
        raise ValueError(
            "dense DSA LSE q/k shape mismatch: "
            f"q={tuple(q_attn_bshd.shape)} k={tuple(k_attn_bshd.shape)}"
        )
    if kv_heads <= 0 or q_heads % kv_heads != 0:
        raise ValueError(
            f"dense DSA LSE requires q_heads ({q_heads}) divisible by kv_heads ({kv_heads})"
        )
    minimum_score_bytes = batch * q_heads * torch.float32.itemsize
    if max_score_bytes < minimum_score_bytes:
        raise ValueError(
            "max_score_bytes cannot hold one KV score for every batch/head, "
            f"need at least {minimum_score_bytes}, got {max_score_bytes}"
        )
    if seq_q == 0:
        return torch.empty(
            (batch, 0, q_heads), device=q_attn_bshd.device, dtype=torch.float32
        )
    valid_kv_per_query = _bottom_right_valid_kv_counts(
        seq_q, seq_k, ratio, q_attn_bshd.device
    )

    fp32_bytes = torch.empty((), dtype=torch.float32).element_size()
    max_score_elements = max_score_bytes // fp32_bytes
    score_elements_per_qk = batch * q_heads
    keys_per_block = max(1, min(seq_k, max_score_elements // score_elements_per_qk))
    queries_per_block = max(
        1, min(seq_q, max_score_elements // (score_elements_per_qk * keys_per_block))
    )
    qheads_per_kv_head = q_heads // kv_heads
    k_f32 = k_attn_bshd.detach().float()
    full_lse = torch.empty(
        (batch, seq_q, q_heads), device=q_attn_bshd.device, dtype=torch.float32
    )

    for q_start in range(0, seq_q, queries_per_block):
        q_end = min(q_start + queries_per_block, seq_q)
        block_q = (
            q_attn_bshd[:, q_start:q_end]
            .detach()
            .float()
            .reshape(batch, q_end - q_start, kv_heads, qheads_per_kv_head, head_dim)
        )
        valid_kv = valid_kv_per_query[q_start:q_end]
        block_lse = torch.full(
            (batch, q_end - q_start, kv_heads, qheads_per_kv_head),
            -torch.inf,
            device=q_attn_bshd.device,
            dtype=torch.float32,
        )
        for k_start in range(0, seq_k, keys_per_block):
            k_end = min(k_start + keys_per_block, seq_k)
            # (B, Q, H_kv, H_q/H_kv, D) x (B, K, H_kv, D)
            #   -> (B, Q, H_kv, H_q/H_kv, K)
            scores = torch.einsum("bqgnd,bkgd->bqgnk", block_q, k_f32[:, k_start:k_end])
            scores.mul_(float(softmax_scale))
            key_positions = torch.arange(
                k_start, k_end, device=q_attn_bshd.device, dtype=torch.int64
            )
            causal = key_positions.view(1, -1) < valid_kv.view(-1, 1)
            scores.masked_fill_(
                ~causal.view(1, q_end - q_start, 1, 1, k_end - k_start), -torch.inf
            )
            block_lse = torch.logaddexp(block_lse, torch.logsumexp(scores, dim=-1))
        block_lse = block_lse.reshape(batch, q_end - q_start, q_heads)
        # Match FlashMLA's lonely-query convention. +inf also prevents an
        # implementation from exponentiating large QK values before applying
        # the downstream all-masked-row gate.
        block_lse = torch.where(
            valid_kv.view(1, -1, 1) > 0,
            block_lse,
            torch.full_like(block_lse, torch.inf),
        )
        full_lse[:, q_start:q_end].copy_(block_lse)

    return full_lse


def _dense_kl_block_shape(batch: int, seq_q: int, seq_k: int) -> tuple[int, int]:
    """Choose Q/K blocks that bound canonical dense-KL temporaries."""

    # target, predict, two log operands, KL terms, and masks coexist in the
    # scalar reduction. Six FP32-equivalent matrices is a conservative bound.
    max_elements = _DENSE_KL_MAX_TEMP_BYTES // (6 * torch.float32.itemsize)
    elements_per_qk = max(batch, 1)
    keys_per_block = max(1, min(seq_k, max_elements // elements_per_qk))
    queries_per_block = max(
        1, min(seq_q, max_elements // (elements_per_qk * keys_per_block))
    )
    return queries_per_block, keys_per_block


def _kl_loss_from_dense_scores(
    attn_score: Tensor,
    attn_l1norm: Tensor,
    index_score: Tensor,
    index_lse: Tensor,
    loss_coeff: float,
    calculate_per_token_loss: bool = False,
    ratio: int = 1,
) -> Tensor:
    """KL(target || predict) over the **full** KV axis, averaged over ``(B, S_q)``.

    Derives ``target = attn_score / attn_l1norm`` (L1-normalised, matches
    ``compute_dsa_indexer_loss``'s ``attention_scores / sum`` step) and
    ``log_predict = index_score - index_lse`` (LSE-normalised log-softmax),
    then computes ``KL = sum_k target * (log target - log predict)`` and
    scales by ``loss_coeff``.

    Rows where the kernel's ``ratio`` causal mask leaves no valid KV
    position have ``attn_l1norm <= 0`` (L1) or ``index_lse == -inf``
    (LSE); those rows contribute 0 to the loss — the same ``row_valid``
    semantics as the reference ``compute_dsa_indexer_loss``.
    """
    if attn_score.shape != index_score.shape or attn_score.ndim != 3:
        raise ValueError(
            "dense DSA KL expects matching (B, S_q, S_k) score tensors, got "
            f"attn={tuple(attn_score.shape)} index={tuple(index_score.shape)}"
        )
    batch, seq_q, seq_k = index_score.shape
    if attn_l1norm.shape != (batch, seq_q) or index_lse.shape != (batch, seq_q):
        raise ValueError(
            "dense DSA KL denominator shapes must be (B, S_q), got "
            f"attn={tuple(attn_l1norm.shape)} index={tuple(index_lse.shape)}"
        )
    if batch == 0 or seq_q == 0:
        return index_score.new_zeros(())

    eps = _CANONICAL_KL_EPS
    valid_kv = _bottom_right_valid_kv_counts(seq_q, seq_k, ratio, index_score.device)
    queries_per_block, keys_per_block = _dense_kl_block_shape(batch, seq_q, seq_k)
    loss_sum = index_score.new_zeros(())
    for q_start in range(0, seq_q, queries_per_block):
        q_end = min(q_start + queries_per_block, seq_q)
        block_valid_kv = valid_kv[q_start:q_end]
        block_row_valid = (
            (block_valid_kv.view(1, -1) > 0)
            & (attn_l1norm[:, q_start:q_end] > eps)
            & torch.isfinite(index_lse[:, q_start:q_end])
        )
        safe_l1 = attn_l1norm[:, q_start:q_end].clamp(min=eps)
        safe_lse = torch.where(
            block_row_valid,
            index_lse[:, q_start:q_end],
            torch.zeros_like(index_lse[:, q_start:q_end]),
        )
        block_kl = index_score.new_zeros((batch, q_end - q_start))
        for k_start in range(0, seq_k, keys_per_block):
            k_end = min(k_start + keys_per_block, seq_k)
            k_positions = torch.arange(
                k_start, k_end, device=index_score.device, dtype=torch.int64
            )
            position_valid = k_positions.view(1, -1) < block_valid_kv.view(-1, 1)
            position_valid = position_valid.unsqueeze(0)
            target = attn_score[:, q_start:q_end, k_start:k_end] / safe_l1.unsqueeze(-1)
            predict = torch.exp(
                index_score[:, q_start:q_end, k_start:k_end] - safe_lse.unsqueeze(-1)
            )
            kl_terms = target * (torch.log(target + eps) - torch.log(predict + eps))
            block_kl.add_(
                torch.where(position_valid, kl_terms, torch.zeros_like(kl_terms)).sum(
                    dim=-1
                )
            )
        loss_sum.add_(
            torch.where(block_row_valid, block_kl, torch.zeros_like(block_kl)).sum()
        )

    loss = loss_sum if calculate_per_token_loss else loss_sum / (batch * seq_q)
    return loss_coeff * loss


_MIN_INDEXER_BACKWARD_HEADS = 64


def _pad_indexer_heads_for_backward(
    q_indexer_bshd: Tensor, weights_bsh: Tensor
) -> Tuple[Tensor, Tensor, int]:
    """Pad released GLM's 32 heads to the vendor backward minimum of 64.

    Zero-valued query and weight heads contribute exactly zero to the indexer
    score and its key gradient.  This preserves the released checkpoint math
    while allowing the current SM90/SM100 cuDNN kernels, which assert
    ``heads >= 64``, to execute.  Returned query/weight gradients are sliced
    back to ``original_heads`` immediately after the kernel call.
    """

    original_heads = q_indexer_bshd.shape[2]
    if weights_bsh.shape[2] != original_heads:
        raise ValueError(
            "indexer query/weight head mismatch: "
            f"q={original_heads}, weights={weights_bsh.shape[2]}"
        )
    if original_heads >= _MIN_INDEXER_BACKWARD_HEADS:
        return q_indexer_bshd, weights_bsh, original_heads

    pad_heads = _MIN_INDEXER_BACKWARD_HEADS - original_heads
    q_padded = torch.nn.functional.pad(q_indexer_bshd, (0, 0, 0, pad_heads))
    weights_padded = torch.nn.functional.pad(weights_bsh, (0, pad_heads))
    return q_padded.contiguous(), weights_padded.contiguous(), original_heads


class _FusedIndexerSparseAttnWithTopKFunc(torch.autograd.Function):
    """Internal Path B autograd that additionally returns top-k indices.

    Differentiable w.r.t. ``query``, ``kv_full``, ``attn_sink``,
    ``q_indexer``, ``k_indexer``, ``weights``.

    Two indexer-loss variants, selected by the ``sparse_loss`` argument
    (matches ``compute_dsa_indexer_loss`` in the reference ``dsa.py``):

    * **Sparse loss** (``sparse_loss=True``) — KL is computed only over
      the top-K KV positions the indexer has selected.
    * **Dense loss** (``sparse_loss=False``, the default) — KL is
      computed over *all* causally valid KV positions.

    Both variants share the FlashMLA sparse-attention forward + the
    cuDNN sparse-attn backward; only the indexer-loss path branches.
    """

    @staticmethod
    def forward(
        ctx,
        # Sparse attn inputs (differentiable)
        query: Tensor,  # (sq, b, np, d) bf16
        kv_full: Tensor,  # (skv, b, d) bf16
        attn_sink: Tensor,  # (np,) f32
        # Window indices (not differentiable)
        window_idxs: Tensor,  # (b, sq, win_topk) int32
        # Indexer inputs (differentiable)
        q_indexer: Tensor,  # (sq, b, idx_nh, idx_hd) bf16
        k_indexer: Tensor,  # (n_comp, b, idx_hd) bf16
        weights: Tensor,  # (sq, b, idx_nh) bf16 — raw (unscaled)
        # Scalars
        indexer_topk: int,
        ratio: int,
        softmax_scale: float,
        indexer_softmax_scale: float,
        loss_coeff: float,
        sparse_loss: bool,
        kv_offset: int,
        calculate_per_token_loss: bool,
        value_dim: Optional[int],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Fused forward: indexer scoring, sparse attention, KL loss, and indexer backward."""
        _ensure_dsa_namespace()

        sq, b, np_, d = query.shape
        skv = kv_full.shape[0]
        n_comp = k_indexer.shape[0]

        _guard_dsa_score_memory(
            query, b, sq, n_comp, dense_loss=loss_coeff > 0 and not sparse_loss
        )

        requested_topk = indexer_topk
        effective_topk = min(requested_topk, n_comp)

        # ---- 1. Permute indexer inputs SBHD->BSHD ONCE. -------------------
        q_idx_bshd, k_idx_bsd, w_bsh, w_bsh_scaled = _sbhd_to_bshd_indexer_inputs(
            q_indexer, k_indexer, weights, indexer_softmax_scale
        )

        # ---- 2. Indexer scoring + top-K (with scores retained). -------------
        topk_indices_cmp, _, indexer_scores = _indexer_topk_bshd(
            q_idx_bshd, k_idx_bsd, w_bsh_scaled, effective_topk, ratio
        )  # topk_indices_cmp: (b, sq, effective_topk) int32; indexer_scores: (b, sq, n_comp) fp32

        # ---- 3. Combine indices (indexer first, then window). --------------
        compress_topk_idxs = torch.where(
            topk_indices_cmp >= 0, topk_indices_cmp + kv_offset, -1
        )
        if requested_topk > effective_topk:
            pad = torch.full(
                (b, sq, requested_topk - effective_topk),
                -1,
                device=compress_topk_idxs.device,
                dtype=compress_topk_idxs.dtype,
            )
            compress_topk_idxs = torch.cat([compress_topk_idxs, pad], dim=-1)
        combined_local = torch.cat([compress_topk_idxs, window_idxs], dim=-1)
        global_idxs = local_to_global_flat(combined_local, b, skv)

        # ---- 4. FlashMLA forward (non-compact, indexer_topk > 0). ---------
        q_flat = query.reshape(sq * b, np_, d)
        kv_flat = kv_full.reshape(skv * b, d)
        # Sparse KL needs the selected-index LSE. Dense KL must not depend on
        # top-k at all, so avoid asking FlashMLA for that auxiliary output.
        flash_indexer_topk = requested_topk if sparse_loss and loss_coeff > 0 else 0
        out_flat, lse, lse_indexer = _dsa_fwd_flash_mla(
            q_flat,
            kv_flat,
            global_idxs,
            softmax_scale,
            attn_sink=attn_sink,
            topk_length=None,
            indexer_topk=flash_indexer_topk,
            d_v=512 if value_dim is None else value_dim,
        )

        # ---- 5. Derive predict from indexer_scores, compute target. --------
        # Attention-path tensors (detached — loss is not differentiable through them).
        q_attn_bshd = query.detach().permute(1, 0, 2, 3).contiguous()
        k_attn_compressed_bsd = (
            kv_full[kv_offset:].detach().permute(1, 0, 2).contiguous()
        )

        if loss_coeff <= 0:
            indexer_loss = torch.zeros((), device=query.device, dtype=torch.float32)
        elif sparse_loss:
            if lse_indexer is None:
                raise RuntimeError("FlashMLA did not return the sparse indexer LSE")
            lse_indexer_bsqh = lse_indexer.reshape(sq, b, np_).permute(1, 0, 2)
            # Derive predict: gather topk scores from indexer_scores → softmax.
            safe_indices = topk_indices_cmp.clamp(min=0).long()
            gathered_scores = torch.gather(indexer_scores, dim=2, index=safe_indices)
            gathered_scores = torch.where(
                topk_indices_cmp >= 0, gathered_scores, torch.finfo(torch.float32).min
            )
            predict = torch.softmax(gathered_scores, dim=-1)  # (b, sq, topk) fp32

            target = _compute_attn_target(
                q_attn_bshd,
                k_attn_compressed_bsd,
                lse_indexer_bsqh,
                topk_indices_cmp,
                softmax_scale,
                qhead_per_kv_head=np_,
            )

            indexer_loss = _kl_loss_from_target_predict(
                target, predict, topk_indices_cmp, loss_coeff, calculate_per_token_loss
            )
        else:
            # Dense backward consumes the raw score/denominator pair emitted
            # by cuDNN's dense recompute kernel. Reusing the top-k path's
            # BF16-pre-scaled scores would violate that kernel contract.
            del indexer_scores
            index_score, index_lse = _compute_dense_indexer_score(
                q_idx_bshd,
                k_idx_bsd.unsqueeze(2),
                w_bsh,
                qhead_per_kv_head=q_idx_bshd.shape[2],
                indexer_softmax_scale=indexer_softmax_scale,
                ratio=ratio,
            )

            # Canonical dense DSA first normalizes every attention head over
            # the complete causal KV axis. FlashMLA's sparse/indexer LSE is
            # selected-top-k-only and would make this target depend on top-k.
            full_attn_lse = _compute_full_causal_attn_lse(
                q_attn_bshd, k_attn_compressed_bsd.unsqueeze(2), softmax_scale, ratio
            )

            attn_score, attn_l1norm = _compute_dense_attn_score(
                q_attn_bshd,
                k_attn_compressed_bsd.unsqueeze(2),
                full_attn_lse,
                qhead_per_kv_head=np_,
                softmax_scale=softmax_scale,
                ratio=ratio,
            )

            indexer_loss = _kl_loss_from_dense_scores(
                attn_score,
                attn_l1norm,
                index_score,
                index_lse,
                loss_coeff,
                calculate_per_token_loss,
                ratio,
            )

        # ---- 6. Eagerly compute indexer backward (grad_loss=1). ------------
        # The actual grad_loss scaling is deferred to backward (when
        # DSAIndexerLossAutoScaler provides the correct scale).
        indexer_loss_coeff = loss_coeff
        if calculate_per_token_loss:
            indexer_loss_coeff = loss_coeff * (b * sq)

        unit_grad_loss = torch.ones((), device=query.device, dtype=torch.float32)

        if loss_coeff > 0:
            q_idx_bwd, w_bwd, original_index_heads = _pad_indexer_heads_for_backward(
                q_idx_bshd, w_bsh
            )
            if sparse_loss:
                # Vendor score-grad implements -target followed by the
                # softmax Jacobian. For canonical log(p + eps), the signal is
                # -target * p/(p+eps). Apply that factor in-place after the
                # scalar loss has been materialized.
                _scale_target_for_canonical_log_eps_backward_(target, predict)
                ig = _DSA.indexer_backward_wrapper(
                    q_idx_bwd,
                    w_bwd,
                    k_idx_bsd,
                    target,
                    predict,
                    topk_indices_cmp,
                    sm_scale=indexer_softmax_scale,
                    loss_coeff=indexer_loss_coeff,
                    grad_loss=unit_grad_loss,
                    block_I=128,
                )
            else:
                _scale_dense_target_for_canonical_log_eps_backward_(
                    attn_score, index_score, index_lse, ratio
                )
                ig = _DSA.dense_indexer_backward_wrapper(
                    q_idx_bwd,
                    w_bwd,
                    k_idx_bsd,
                    attn_score,
                    attn_l1norm,
                    index_score,
                    index_lse,
                    sm_scale=indexer_softmax_scale,
                    loss_coeff=indexer_loss_coeff,
                    grad_loss=unit_grad_loss,
                    ratio=ratio,
                    block_I=128,
                )
            # Remove any compatibility padding, then BSHD -> SBHD to match
            # the original autograd inputs.  dK already sums across heads.
            precomputed_grad_q_indexer = (
                ig["d_index_q"][:, :, :original_index_heads, :]
                .permute(1, 0, 2, 3)
                .contiguous()
            )
            precomputed_grad_k_indexer = ig["d_index_k"].permute(1, 0, 2).contiguous()
            precomputed_grad_weights = (
                ig["d_weights"][:, :, :original_index_heads]
                .permute(1, 0, 2)
                .contiguous()
            )
        else:
            precomputed_grad_q_indexer = torch.zeros_like(q_indexer)
            precomputed_grad_k_indexer = torch.zeros_like(k_indexer)
            precomputed_grad_weights = torch.zeros_like(weights)

        # ---- 7. Save context (only sparse-attn bwd tensors + indexer grads).
        ctx.save_for_backward(
            q_flat,
            kv_flat,
            attn_sink,
            global_idxs,
            out_flat,
            lse,
            precomputed_grad_q_indexer,
            precomputed_grad_k_indexer,
            precomputed_grad_weights,
        )
        ctx.softmax_scale = softmax_scale
        ctx.sq = sq
        ctx.b = b
        ctx.np_ = np_
        ctx.d = d
        ctx.skv = skv

        # ---- 8. Return. ---------------------------------------------------
        d_v = out_flat.shape[-1]
        output = out_flat.reshape(sq, b, np_, d_v).reshape(sq, b, np_ * d_v)
        return output, indexer_loss, compress_topk_idxs

    @staticmethod
    def backward(ctx, grad_output, grad_loss, grad_topk_indices=None):
        """Backward: sparse attention bwd + scale pre-computed indexer grads."""
        del grad_topk_indices
        (
            q_flat,
            kv_flat,
            attn_sink,
            global_idxs,
            out_flat,
            lse,
            precomputed_grad_q_indexer,
            precomputed_grad_k_indexer,
            precomputed_grad_weights,
        ) = ctx.saved_tensors

        sq, b, np_, d = ctx.sq, ctx.b, ctx.np_, ctx.d
        skv = ctx.skv

        # ---- 1. Sparse attn backward. -------------------------------------
        d_v = out_flat.shape[-1]
        dO_flat = grad_output.reshape(sq * b, np_, d_v)

        attn_bwd = _DSA.sparse_attention_backward_wrapper(
            q_flat,
            kv_flat,
            out_flat,
            dO_flat,
            lse,
            attn_sink,
            global_idxs,
            softmax_scale=ctx.softmax_scale,
            topk_length=None,
        )
        grad_query = attn_bwd["dq"].reshape(sq, b, np_, d)
        grad_kv_full = attn_bwd["dkv"].reshape(skv, b, d)
        d_sink = attn_bwd["d_sink"]

        # ---- 2. Scale pre-computed indexer grads by grad_loss. -------------
        grad_q_indexer = precomputed_grad_q_indexer * grad_loss
        grad_k_indexer = precomputed_grad_k_indexer * grad_loss
        grad_weights = precomputed_grad_weights * grad_loss

        # Grads: query, kv_full, attn_sink, window_idxs, q_indexer, k_indexer,
        #   weights, indexer_topk, ratio, softmax_scale, indexer_softmax_scale,
        #   loss_coeff, sparse_loss, kv_offset, calculate_per_token_loss, value_dim
        return (
            grad_query,
            grad_kv_full,
            d_sink,
            None,
            grad_q_indexer,
            grad_k_indexer,
            grad_weights,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class FusedIndexerSparseAttnFunc(_FusedIndexerSparseAttnWithTopKFunc):
    """Legacy two-output fused DSA autograd function.

    Direct callers historically unpacked ``FusedIndexerSparseAttnFunc.apply``
    as ``(output, indexer_loss)``. Keep that contract stable; IndexShare uses
    the explicitly named private three-output function instead.
    """

    @staticmethod
    def forward(
        ctx,
        query: Tensor,
        kv_full: Tensor,
        attn_sink: Tensor,
        window_idxs: Tensor,
        q_indexer: Tensor,
        k_indexer: Tensor,
        weights: Tensor,
        indexer_topk: int,
        ratio: int,
        softmax_scale: float,
        indexer_softmax_scale: float,
        loss_coeff: float,
        sparse_loss: bool,
        kv_offset: int,
        calculate_per_token_loss: bool,
        value_dim: Optional[int],
    ) -> Tuple[Tensor, Tensor]:
        output, indexer_loss, _topk_indices = (
            _FusedIndexerSparseAttnWithTopKFunc.forward(
                ctx,
                query,
                kv_full,
                attn_sink,
                window_idxs,
                q_indexer,
                k_indexer,
                weights,
                indexer_topk,
                ratio,
                softmax_scale,
                indexer_softmax_scale,
                loss_coeff,
                sparse_loss,
                kv_offset,
                calculate_per_token_loss,
                value_dim,
            )
        )
        return output, indexer_loss

    @staticmethod
    def backward(ctx, grad_output, grad_loss):
        return _FusedIndexerSparseAttnWithTopKFunc.backward(
            ctx, grad_output, grad_loss, None
        )


def _fused_indexer_sparse_attn_with_topk_apply(
    query: Tensor,
    kv_full: Tensor,
    attn_sink: Tensor,
    window_idxs: Tensor,
    q_indexer: Tensor,
    k_indexer: Tensor,
    weights: Tensor,
    indexer_topk: int,
    ratio: int,
    softmax_scale: float,
    indexer_softmax_scale: float = 1.0,
    loss_coeff: float = 0.0,
    sparse_loss: bool = False,
    kv_offset: int = 0,
    calculate_per_token_loss: bool = False,
    value_dim: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    return _FusedIndexerSparseAttnWithTopKFunc.apply(
        query,
        kv_full,
        attn_sink,
        window_idxs,
        q_indexer,
        k_indexer,
        weights,
        indexer_topk,
        ratio,
        softmax_scale,
        indexer_softmax_scale,
        loss_coeff,
        sparse_loss,
        kv_offset,
        calculate_per_token_loss,
        value_dim,
    )


def fused_indexer_sparse_attn(
    query: Tensor,
    kv_full: Tensor,
    attn_sink: Tensor,
    window_idxs: Tensor,
    q_indexer: Tensor,
    k_indexer: Tensor,
    weights: Tensor,
    indexer_topk: int,
    ratio: int,
    softmax_scale: float,
    indexer_softmax_scale: float = 1.0,
    loss_coeff: float = 0.0,
    sparse_loss: bool = False,
    kv_offset: int = 0,
    calculate_per_token_loss: bool = False,
    value_dim: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """Path B (training): fused indexer (+KL loss) + sparse attention.

    See :class:`FusedIndexerSparseAttnFunc` for the detailed data flow.

    Args:
        query:        ``(sq, b, np, d)`` bf16 SBHD — attention query.
        kv_full:      ``(skv, b, d)`` bf16 SBD — original + compressed KV.
        attn_sink:    ``(np,)`` f32 — learnable sink per head.
        window_idxs:  ``(b, sq, win_topk)`` int32 — local window indices.
        q_indexer:    ``(sq, b, idx_nh, idx_hd)`` bf16 — indexer query.
        k_indexer:    ``(n_comp, b, idx_hd)`` bf16 — indexer key (compressed).
        weights:      ``(sq, b, idx_nh)`` bf16 — raw indexer weights.
        indexer_topk: number of top-K compressed positions to select.
        ratio:        compression ratio used for the causal mask.
        softmax_scale: attention ``Q @ K^T`` scale, typically
            ``1/sqrt(v_head_dim)``.
        indexer_softmax_scale: indexer ``Q @ K^T`` scale, typically
            ``1/sqrt(idx_hd)``. Applied internally — caller passes raw
            (unscaled) ``weights``.
        loss_coeff:   coefficient scaling the KL divergence loss.
        sparse_loss:  if ``True``, KL is computed only over the top-K
            positions (cheap, less informative); if ``False`` (the
            default, matches ``transformer_config.dsa_indexer_use_sparse_loss``),
            KL is computed over the full causally-valid KV (more
            informative, matches the DeepSeek-V3.2 paper, larger
            intermediate-tensor footprint). See
            :class:`FusedIndexerSparseAttnFunc` for the full data flow
            of each variant.
        kv_offset:    start of compressed region within ``kv_full``.
        calculate_per_token_loss: if True, report raw local KL sum and
            compensate the cuDNN backward wrappers' local averaging.
        value_dim: FlashMLA value dimension. Defaults to ``512`` to preserve
            the existing DSA wrapper behavior.

    Returns:
        ``(output, indexer_loss)`` where ``output`` is ``(sq, b, np * d_v)``
        bf16 and ``indexer_loss`` is a scalar f32.
    """
    output, indexer_loss = FusedIndexerSparseAttnFunc.apply(
        query,
        kv_full,
        attn_sink,
        window_idxs,
        q_indexer,
        k_indexer,
        weights,
        indexer_topk,
        ratio,
        softmax_scale,
        indexer_softmax_scale,
        loss_coeff,
        sparse_loss,
        kv_offset,
        calculate_per_token_loss,
        value_dim,
    )
    return output, indexer_loss


def fused_indexer_sparse_attn_with_topk(
    query: Tensor,
    kv_full: Tensor,
    attn_sink: Tensor,
    window_idxs: Tensor,
    q_indexer: Tensor,
    k_indexer: Tensor,
    weights: Tensor,
    indexer_topk: int,
    ratio: int,
    softmax_scale: float,
    indexer_softmax_scale: float = 1.0,
    loss_coeff: float = 0.0,
    sparse_loss: bool = False,
    kv_offset: int = 0,
    calculate_per_token_loss: bool = False,
    value_dim: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Path B plus the fused top-k indices for DSA IndexShare source layers."""
    return _fused_indexer_sparse_attn_with_topk_apply(
        query,
        kv_full,
        attn_sink,
        window_idxs,
        q_indexer,
        k_indexer,
        weights,
        indexer_topk,
        ratio,
        softmax_scale,
        indexer_softmax_scale,
        loss_coeff,
        sparse_loss,
        kv_offset,
        calculate_per_token_loss,
        value_dim,
    )


__all__ = [
    "build_flat_topk_idxs",
    "local_to_global_flat",
    "dsa_sparse_attn",
    "indexer_topk",
    "fused_indexer_sparse_attn",
    "fused_indexer_sparse_attn_with_topk",
]
