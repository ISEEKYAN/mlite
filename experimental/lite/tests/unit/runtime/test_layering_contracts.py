# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Static layering guards for MLite public data boundaries."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


LITE_ROOT = Path(__file__).resolve().parents[3]
BENCH_ROOT = LITE_ROOT / "examples" / "bench"
VERL_MLITE_ROOT = LITE_ROOT / "examples" / "verl" / "verl_mlite"
RUNTIME_ROOT = LITE_ROOT / "megatron" / "lite" / "runtime"
BRIDGE_RUNTIME = RUNTIME_ROOT / "backends" / "bridge" / "runtime.py"

ALLOW_BEGIN = "MLITE_LAYERING_ALLOW_BRIDGE_FORWARD_METADATA_BEGIN"
ALLOW_END = "MLITE_LAYERING_ALLOW_BRIDGE_FORWARD_METADATA_END"


def _python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if path.is_file())


def _allow_ranges(path: Path) -> list[range]:
    if path != BRIDGE_RUNTIME:
        return []

    ranges: list[range] = []
    start: int | None = None
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if ALLOW_BEGIN in line:
            if start is not None:
                raise AssertionError(f"nested allow range in {path}")
            start = lineno
        elif ALLOW_END in line:
            if start is None:
                raise AssertionError(f"unmatched allow range end in {path}:{lineno}")
            ranges.append(range(start, lineno + 1))
            start = None
    if start is not None:
        raise AssertionError(f"unclosed allow range in {path}")
    return ranges


def _violations(paths: Iterable[Path], denied_terms: set[str]) -> list[str]:
    found: list[str] = []
    for path in paths:
        ranges = _allow_ranges(path)
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if any(lineno in allowed for allowed in ranges):
                continue
            for term in sorted(denied_terms):
                if term in line:
                    rel = path.relative_to(LITE_ROOT)
                    found.append(f"{rel}:{lineno}: {term}")
    return found


def test_bench_layer_does_not_see_model_internal_batch_fields() -> None:
    violations = _violations(
        _python_files(BENCH_ROOT),
        {"packed_seq_params", "position_ids", "to_bridge_dict"},
    )
    assert violations == []


def test_verl_mlite_layer_does_not_see_packed_seq_params() -> None:
    violations = _violations(
        _python_files(VERL_MLITE_ROOT),
        {"packed_seq_params", "to_bridge_dict"},
    )
    assert violations == []


def test_runtime_packed_seq_params_is_bridge_forward_transient_only() -> None:
    violations = _violations(_python_files(RUNTIME_ROOT), {"packed_seq_params"})
    assert violations == []
