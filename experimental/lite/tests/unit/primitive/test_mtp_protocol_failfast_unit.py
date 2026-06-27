# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Static contract tests for PP-replicated MTP embedding protocol gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


pytestmark = [pytest.mark.mlite]

_PROTOCOLS = (
    "glm5",
    "deepseek_v4",
    "kimi_k2",
    "qwen3_5",
    "qwen3_moe",
)


def _protocol_path(model_name: str) -> Path:
    lite_root = Path(__file__).resolve().parents[3] / "megatron" / "lite"
    return lite_root / "model" / model_name / "lite" / "protocol.py"


def _model_path(model_name: str) -> Path:
    lite_root = Path(__file__).resolve().parents[3] / "megatron" / "lite"
    return lite_root / "model" / model_name / "lite" / "model.py"


def _call_lines(function: ast.FunctionDef, name: str) -> list[int]:
    return sorted(
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == name)
            or (isinstance(node.func, ast.Attribute) and node.func.attr == name)
        )
    )


@pytest.mark.parametrize("model_name", _PROTOCOLS)
def test_fsdp2_pp_mtp_gate_precedes_parallel_and_cuda_initialization(
    model_name: str,
) -> None:
    """Reject every FSDP2+PP+MTP model before allocating parallel/CUDA state.

    Even when MTP loss is disabled, the first-stage input embedding still
    receives the trunk gradient. FSDP2 has no first/last-stage replica sync, so
    allowing that configuration would silently diverge ``mtp_embed`` after the
    first optimizer step. Pure load/export remains available with
    ``optimizer=None`` because the gate is explicitly scoped to FSDP2.
    """

    path = _protocol_path(model_name)
    tree = ast.parse(path.read_text(), filename=str(path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_model"
    )

    gates: list[tuple[ast.If, ast.Raise]] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.If):
            continue
        for child in node.body:
            if not isinstance(child, ast.Raise) or child.exc is None:
                continue
            if not isinstance(child.exc, ast.Call):
                continue
            if not isinstance(child.exc.func, ast.Name):
                continue
            if child.exc.func.id != "NotImplementedError":
                continue
            message = ast.unparse(child.exc)
            if "PP-replicated MTP input embedding" in message:
                gates.append((node, child))

    assert len(gates) == 1, f"{model_name}: expected one FSDP2 PP+MTP gate"
    gate, _raise = gates[0]
    condition = ast.unparse(gate.test)
    condition_names = {
        node.id for node in ast.walk(gate.test) if isinstance(node, ast.Name)
    }
    assert "fsdp2" in condition
    assert "mtp_enable" in condition_names
    assert "mtp_enable_train" not in condition_names
    assert ".pp" in condition and "> 1" in condition

    init_parallel_lines = _call_lines(function, "init_parallel")
    cuda_lines = _call_lines(function, "cuda")
    assert init_parallel_lines, (
        f"{model_name}: build_model does not initialize parallel state"
    )
    assert cuda_lines, (
        f"{model_name}: build_model does not materialize CUDA model chunks"
    )
    assert gate.lineno < min(init_parallel_lines), (
        f"{model_name}: FSDP2 PP+MTP gate runs after init_parallel"
    )
    assert gate.lineno < min(cuda_lines), (
        f"{model_name}: FSDP2 PP+MTP gate runs after CUDA construction"
    )


def test_deepseek_v4_packed_mtp_requires_single_sequence_without_cp() -> None:
    path = _protocol_path("deepseek_v4")
    tree = ast.parse(path.read_text(), filename=str(path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_prepare_packed_batch_kwargs"
    )

    enable_mtp_values = [
        value
        for node in ast.walk(function)
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values)
        if isinstance(key, ast.Constant) and key.value == "enable_mtp"
    ]
    assert len(enable_mtp_values) == 1

    gate = enable_mtp_values[0]
    assert isinstance(gate, ast.BoolOp) and isinstance(gate.op, ast.And)
    assert {ast.unparse(term) for term in gate.values} == {
        "ps.cp_size == 1",
        "seq_lens.numel() == 1",
    }


@pytest.mark.parametrize(
    ("model_name", "class_name"),
    (("qwen3_5", "Qwen35Model"), ("qwen3_moe", "Qwen3MoEModel")),
)
def test_qwen_models_pass_mtp_slots_to_pipeline_layout_and_honor_layout_gate(
    model_name: str, class_name: str
) -> None:
    path = _model_path(model_name)
    tree = ast.parse(path.read_text(), filename=str(path))
    model_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    constructor = next(
        node
        for node in model_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )

    layout_calls = [
        node
        for node in ast.walk(constructor)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_pipeline_chunk_layout"
    ]
    assert len(layout_calls) == 1
    keyword = next(
        (item for item in layout_calls[0].keywords if item.arg == "num_mtp_layers"),
        None,
    )
    assert keyword is not None
    expression = ast.unparse(keyword.value)
    assert "config.num_nextn_predict_layers" in expression
    assert "mtp_enable" in expression

    mtp_gates = []
    for node in ast.walk(constructor):
        if not isinstance(node, ast.If):
            continue
        assigns_mtp = any(
            isinstance(child, ast.Assign)
            and any(
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr == "mtp"
                for target in child.targets
            )
            for child in ast.walk(node)
        )
        if assigns_mtp:
            mtp_gates.append(ast.unparse(node.test))

    assert any("layout.has_mtp" in condition for condition in mtp_gates)
