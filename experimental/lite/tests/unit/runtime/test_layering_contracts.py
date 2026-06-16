# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Static layering guards for MLite public data boundaries and imports."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path


LITE_ROOT = Path(__file__).resolve().parents[3]
BENCH_ROOT = LITE_ROOT / "examples" / "bench"
VERL_MLITE_ROOT = LITE_ROOT / "examples" / "verl" / "verl_mlite"
RUNTIME_ROOT = LITE_ROOT / "megatron" / "lite" / "runtime"
MODEL_ROOT = LITE_ROOT / "megatron" / "lite" / "model"
PRIMITIVE_ROOT = LITE_ROOT / "megatron" / "lite" / "primitive"
BRIDGE_RUNTIME = RUNTIME_ROOT / "backends" / "bridge" / "runtime.py"

ALLOW_BEGIN = "MLITE_LAYERING_ALLOW_BRIDGE_FORWARD_METADATA_BEGIN"
ALLOW_END = "MLITE_LAYERING_ALLOW_BRIDGE_FORWARD_METADATA_END"
LAYER_ROOTS = {
    "bench": BENCH_ROOT,
    "verl_mlite": VERL_MLITE_ROOT,
    "runtime": RUNTIME_ROOT,
    "model": MODEL_ROOT,
    "primitive": PRIMITIVE_ROOT,
}
MODEL_PACKAGE_PREFIXES = (
    "megatron.lite.model.deepseek_v4",
    "megatron.lite.model.glm5",
    "megatron.lite.model.kimi_k2",
    "megatron.lite.model.qwen3_5",
    "megatron.lite.model.qwen3_moe",
)
MODEL_NAME_TERMS = {"deepseek_v4", "glm5", "kimi_k2", "qwen3", "qwen3_5", "qwen3_moe"}
DENIED_IMPORT_PREFIXES = {
    "bench": ("examples.verl", "verl", "verl_mlite", "megatron.lite.model"),
    "verl_mlite": ("examples.bench", *MODEL_PACKAGE_PREFIXES),
    "runtime": ("examples", "verl", "verl_mlite", *MODEL_PACKAGE_PREFIXES),
    "model": ("examples", "verl", "verl_mlite", "megatron.lite.runtime.backends"),
    "primitive": (
        "examples",
        "verl",
        "verl_mlite",
        "megatron.lite.model",
        "megatron.lite.runtime.backends",
        "megatron.lite.runtime.megatron_utils",
    ),
}


def _python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if path.is_file())


def _matches_prefix(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(prefix + ".")


def _imported_modules(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imports.append((node.lineno, node.module))
    return imports


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


def _import_violations(layer: str) -> list[str]:
    denied = DENIED_IMPORT_PREFIXES[layer]
    found: list[str] = []
    for path in _python_files(LAYER_ROOTS[layer]):
        for lineno, module in _imported_modules(path):
            for prefix in denied:
                if _matches_prefix(module, prefix):
                    rel = path.relative_to(LITE_ROOT)
                    found.append(f"{rel}:{lineno}: {module} matches denied {prefix}")
    return found


def test_layer_import_boundaries() -> None:
    violations = []
    for layer in LAYER_ROOTS:
        violations.extend(_import_violations(layer))
    assert violations == []


def test_bench_layer_does_not_see_model_internal_batch_fields() -> None:
    violations = _violations(
        _python_files(BENCH_ROOT),
        {"packed_seq_params", "position_ids", "to_bridge_dict"},
    )
    assert violations == []


def test_verl_mlite_layer_does_not_see_model_internal_batch_fields() -> None:
    violations = _violations(
        _python_files(VERL_MLITE_ROOT),
        {"packed_seq_params", "position_ids", "to_bridge_dict"},
    )
    assert violations == []


def test_runtime_packed_seq_params_is_bridge_forward_transient_only() -> None:
    violations = _violations(_python_files(RUNTIME_ROOT), {"packed_seq_params"})
    assert violations == []


def test_primitive_layer_is_model_name_agnostic() -> None:
    violations = _violations(_python_files(PRIMITIVE_ROOT), MODEL_NAME_TERMS)
    assert violations == []
