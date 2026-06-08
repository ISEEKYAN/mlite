"""Chunk sizing helpers for EP chunked MoE paths."""

from __future__ import annotations

import math
import os
from typing import Literal

ChunkSpec = int | Literal["auto"]
ChunkDirection = Literal["forward", "fused_backward"]


def _env_get(name: str, *legacy_names: str) -> str | None:
    for key in (name, *legacy_names):
        value = os.environ.get(key)
        if value:
            return value
    return None


def parse_ep_chunk_spec(
    value: ChunkSpec | str | None,
    *,
    default: ChunkSpec = "auto",
) -> ChunkSpec:
    """Normalize a chunk count, accepting positive integers or ``"auto"``."""
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "auto":
            return "auto"
        try:
            value = int(normalized)
        except ValueError as exc:
            raise ValueError("chunk spec must be an integer or 'auto'") from exc
    value = int(value)
    if value < 1:
        raise ValueError("chunk count must be >= 1")
    return value


def resolve_ep_chunk_count(
    num_tokens: int,
    *,
    ep_size: int,
    hidden_size: int,
    spec: ChunkSpec | str = "auto",
    direction: ChunkDirection = "forward",
) -> int:
    """Resolve an EP chunk count using the validated Bumblebee defaults."""
    del hidden_size
    spec = parse_ep_chunk_spec(spec)
    if spec != "auto":
        return int(spec)
    if ep_size <= 1 or num_tokens < 16_384:
        return 1
    if direction == "forward":
        return 2 if num_tokens < 32_768 else 3
    if direction == "fused_backward":
        return 2
    raise ValueError("direction must be 'forward' or 'fused_backward'")


def ep_chunk_ranges(
    num_tokens: int,
    num_chunks: int,
    *,
    weights_env: str | tuple[str, ...] = (
        "MEGATRON_LITE_EP_CHUNK_WEIGHTS",
        "BUMBLEBEE_EP_CHUNK_WEIGHTS",
    ),
) -> list[tuple[int, int]]:
    """Split token rows into non-empty contiguous chunks.

    Weighted splits apportion all rows by positive finite weights.  A final
    repair pass keeps chunks non-empty when ``num_tokens >= num_chunks``.
    """
    num_chunks = min(max(int(num_chunks), 1), max(num_tokens, 1))
    if num_tokens == 0:
        return [(0, 0)]

    env_names = (weights_env,) if isinstance(weights_env, str) else weights_env
    raw = _env_get(*env_names)
    if raw:
        try:
            weights = [float(item.strip()) for item in raw.split(",") if item.strip()]
        except ValueError as exc:
            raise ValueError("EP chunk weights must be comma-separated positive numbers") from exc
        if len(weights) != num_chunks:
            raise ValueError(f"EP chunk weights must provide {num_chunks} values")
        if any(not math.isfinite(weight) or weight <= 0.0 for weight in weights):
            raise ValueError("EP chunk weights must contain finite positive values")

        exact = [num_tokens * weight / sum(weights) for weight in weights]
        sizes = [int(value) for value in exact]
        remainder = num_tokens - sum(sizes)
        for idx in sorted(
            range(num_chunks),
            key=lambda idx: (exact[idx] - sizes[idx], -idx),
            reverse=True,
        )[:remainder]:
            sizes[idx] += 1

        for empty_idx in [idx for idx, size in enumerate(sizes) if size == 0]:
            donor_idx = max(
                (idx for idx, size in enumerate(sizes) if size > 1),
                key=lambda idx: (sizes[idx], -idx),
            )
            sizes[donor_idx] -= 1
            sizes[empty_idx] = 1

        out: list[tuple[int, int]] = []
        start = 0
        for size in sizes:
            end = start + size
            out.append((start, end))
            start = end
        return out

    base = num_tokens // num_chunks
    remainder = num_tokens % num_chunks
    out = []
    start = 0
    for idx in range(num_chunks):
        end = start + base + (1 if idx < remainder else 0)
        if start < end:
            out.append((start, end))
        start = end
    return out or [(0, num_tokens)]


__all__ = [
    "ChunkDirection",
    "ChunkSpec",
    "ep_chunk_ranges",
    "parse_ep_chunk_spec",
    "resolve_ep_chunk_count",
]
