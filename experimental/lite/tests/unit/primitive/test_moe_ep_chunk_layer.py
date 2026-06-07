from __future__ import annotations

import ast
from pathlib import Path


_SOURCE = Path(
    "experimental/lite/megatron/lite/primitive/modules/moe_ep_chunk.py"
).read_text()
_TREE = ast.parse(_SOURCE)


def _method_source(class_name: str, method_name: str) -> str:
    for node in ast.walk(_TREE):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return ast.get_source_segment(_SOURCE, item) or ""
    raise AssertionError(f"{class_name}.{method_name} not found")


def test_forward_keeps_chunked_submit_path_outside_autograd():
    forward_source = _method_source("EPChunkedMoELayer", "forward")

    assert "torch.is_grad_enabled()" in forward_source
    assert "self._forward_full" in forward_source
    assert "self._forward_chunked_no_grad" in forward_source


def test_chunked_forward_uses_isolated_dispatchers():
    chunked_source = _method_source("EPChunkedMoELayer", "_forward_chunked_no_grad")

    assert "self._chunk_dispatcher(idx)" in chunked_source
    assert "submit_deepep_dispatch" in chunked_source
    assert "finish_deepep_dispatch" in chunked_source
    assert "submit_deepep_combine" in chunked_source
    assert "finish_deepep_combine" in chunked_source
