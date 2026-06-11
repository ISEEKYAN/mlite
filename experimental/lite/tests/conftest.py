# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import os
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

LITE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
VERL_EXAMPLE_ROOT = LITE_ROOT / "examples" / "verl"
for root in (REPO_ROOT, LITE_ROOT, VERL_EXAMPLE_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def pytest_configure(config):
    config.addinivalue_line("markers", "mlite: mark a test as Megatron Lite validation coverage")
    config.addinivalue_line(
        "markers",
        "smoke: mark a Megatron Lite smoke test; skipped unless --mlite-smoke or MLITE_RUN_SMOKE=1 is set",
    )
    config.addinivalue_line("markers", "gpu: mark a test as requiring CUDA")
    config.addinivalue_line("markers", "distributed: mark a test as requiring torch.distributed")


def pytest_addoption(parser):
    parser.addoption(
        "--mlite-smoke", action="store_true", default=False, help="run Megatron Lite smoke tests"
    )


def pytest_collection_modifyitems(config, items):
    run_smoke = config.getoption("--mlite-smoke") or os.getenv("MLITE_RUN_SMOKE") == "1"
    if run_smoke:
        return
    skip_smoke = pytest.mark.skip(reason="set --mlite-smoke or MLITE_RUN_SMOKE=1 to run")
    for item in items:
        if "smoke" in item.keywords:
            item.add_marker(skip_smoke)


def _install_transformer_engine_import_stub(monkeypatch) -> None:
    try:
        import transformer_engine.pytorch  # noqa: F401

        return
    except ModuleNotFoundError as exc:
        if exc.name not in {"transformer_engine", "transformer_engine.pytorch"}:
            raise
    except OSError:
        pass

    class _UnavailableTE:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Transformer Engine is not installed in this test environment.")

    root = types.ModuleType("transformer_engine")
    root.__version__ = "0.0.0"
    pytorch = types.ModuleType("transformer_engine.pytorch")
    pytorch.__path__ = []
    pytorch.DotProductAttention = _UnavailableTE
    pytorch.LayerNormLinear = _UnavailableTE
    pytorch.Linear = _UnavailableTE
    pytorch.RMSNorm = _UnavailableTE
    root.pytorch = pytorch
    common = types.ModuleType("transformer_engine.common")
    recipe = types.ModuleType("transformer_engine.common.recipe")
    common.recipe = recipe
    root.common = common

    class _Format:
        E4M3 = "e4m3"
        HYBRID = "hybrid"

    class _DelayedScaling:
        def __init__(self, *args, **kwargs):
            pass

    recipe.Format = _Format
    recipe.DelayedScaling = _DelayedScaling
    recipe.Float8CurrentScaling = _DelayedScaling
    recipe.Float8BlockScaling = _DelayedScaling
    recipe.MXFP8BlockScaling = _DelayedScaling
    recipe.NVFP4BlockScaling = _DelayedScaling
    recipe.CustomRecipe = _DelayedScaling

    class _UnavailableTETensor:
        pass

    float8_tensor = types.ModuleType("transformer_engine.pytorch.float8_tensor")
    float8_tensor.Float8Tensor = _UnavailableTETensor
    tensor = types.ModuleType("transformer_engine.pytorch.tensor")
    tensor.__path__ = []
    tensor.QuantizedTensor = _UnavailableTETensor
    tensor_float8 = types.ModuleType("transformer_engine.pytorch.tensor.float8_tensor")
    tensor_float8.Float8Tensor = _UnavailableTETensor
    tensor_mxfp8 = types.ModuleType("transformer_engine.pytorch.tensor.mxfp8_tensor")
    tensor_mxfp8.MXFP8Tensor = _UnavailableTETensor
    tensor_utils = types.ModuleType("transformer_engine.pytorch.tensor.utils")
    tensor_utils.replace_raw_data = lambda tensor, *args, **kwargs: tensor
    tensor_utils.cast_master_weights_to_fp8 = lambda *args, **kwargs: None
    tensor_utils.quantize_master_weights = lambda *args, **kwargs: None
    tensor_utils.post_all_gather_processing = lambda *args, **kwargs: None
    fp8 = types.ModuleType("transformer_engine.pytorch.fp8")

    class _FP8GlobalStateManager:
        @staticmethod
        def is_fp8_enabled() -> bool:
            return False

    fp8.FP8GlobalStateManager = _FP8GlobalStateManager
    fp8.fp8_autocast = lambda *args, **kwargs: nullcontext()
    fp8.fp8_model_init = lambda *args, **kwargs: nullcontext()
    module_base = types.ModuleType("transformer_engine.pytorch.module.base")
    module_base.get_dummy_wgrad = lambda *args, **kwargs: None
    module_base.TransformerEngineBaseModule = _UnavailableTE

    monkeypatch.setitem(sys.modules, "transformer_engine", root)
    monkeypatch.setitem(sys.modules, "transformer_engine.common", common)
    monkeypatch.setitem(sys.modules, "transformer_engine.common.recipe", recipe)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", pytorch)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.float8_tensor", float8_tensor)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.tensor", tensor)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.tensor.float8_tensor", tensor_float8)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.tensor.mxfp8_tensor", tensor_mxfp8)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.tensor.utils", tensor_utils)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.fp8", fp8)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.module.base", module_base)


@pytest.fixture
def transformer_engine_import_stub(monkeypatch):
    def install() -> None:
        _install_transformer_engine_import_stub(monkeypatch)

    return install


def _install_megatron_core_unit_stubs(monkeypatch) -> None:
    import megatron

    core = types.ModuleType("megatron.core")
    core.__path__ = []
    transformer = types.ModuleType("megatron.core.transformer")
    transformer.__path__ = []
    moe = types.ModuleType("megatron.core.transformer.moe")
    moe.__path__ = []
    moe_utils = types.ModuleType("megatron.core.transformer.moe.moe_utils")

    def router_gating_linear(x, weight, bias, router_dtype):
        return F.linear(x.to(router_dtype), weight.to(router_dtype), bias)

    def topk_routing_with_score_function(
        logits,
        topk,
        *,
        score_function,
        expert_bias=None,
        scaling_factor=None,
        fused=False,
        **_kwargs,
    ):
        scores = torch.sigmoid(logits) if score_function == "sigmoid" else torch.softmax(logits, -1)
        if expert_bias is not None:
            scores = scores + expert_bias
        values, indices = torch.topk(scores, k=topk, dim=-1)
        values = values / values.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(values.dtype).eps)
        if scaling_factor is not None:
            values = values * scaling_factor
        probs_dense = torch.zeros_like(scores).scatter(1, indices, values)
        routing_map = torch.zeros_like(scores, dtype=torch.bool).scatter(1, indices, True)
        return probs_dense, None if fused else routing_map

    def compute_routing_scores_for_aux_loss(logits, topk, *, score_function, fused=False):
        scores = torch.sigmoid(logits) if score_function == "sigmoid" else torch.softmax(logits, -1)
        _probs_dense, routing_map = topk_routing_with_score_function(
            logits, topk, score_function=score_function, fused=False
        )
        return routing_map, scores

    def switch_load_balancing_loss_func(
        probs, tokens_per_expert, total_num_tokens, topk, num_experts, moe_aux_loss_coeff, **_kwargs
    ):
        expert_fraction = tokens_per_expert.to(probs.dtype) / max(total_num_tokens * topk, 1)
        expert_probability = probs.mean(dim=0)
        return num_experts * torch.sum(expert_fraction * expert_probability) * moe_aux_loss_coeff

    def permute(hidden_states, routing_map, *, probs=None, num_out_tokens=None, **_kwargs):
        expert_ids, row_ids = routing_map.T.nonzero(as_tuple=True)
        if num_out_tokens is not None:
            expert_ids = expert_ids[:num_out_tokens]
            row_ids = row_ids[:num_out_tokens]
        permuted = hidden_states[row_ids]
        permuted_probs = None if probs is None else probs[row_ids, expert_ids]
        return permuted, permuted_probs, row_ids

    def unpermute(permuted, sorted_indices, *, restore_shape, **_kwargs):
        restored = permuted.new_zeros(restore_shape)
        restored.index_add_(0, sorted_indices, permuted)
        return restored

    moe_utils.router_gating_linear = router_gating_linear
    moe_utils.topk_routing_with_score_function = topk_routing_with_score_function
    moe_utils.compute_routing_scores_for_aux_loss = compute_routing_scores_for_aux_loss
    moe_utils.switch_load_balancing_loss_func = switch_load_balancing_loss_func
    moe_utils.permute = permute
    moe_utils.unpermute = unpermute
    moe_utils.te_general_gemm = None
    moe.moe_utils = moe_utils
    transformer.moe = moe

    models = types.ModuleType("megatron.core.models")
    models.__path__ = []
    models_common = types.ModuleType("megatron.core.models.common")
    models_common.__path__ = []
    embeddings = types.ModuleType("megatron.core.models.common.embeddings")
    embeddings.__path__ = []
    rope_utils = types.ModuleType("megatron.core.models.common.embeddings.rope_utils")
    rotary = types.ModuleType("megatron.core.models.common.embeddings.rotary_pos_embedding")

    def _identity_rotary(tensor, *_args, **_kwargs):
        return tensor

    class RotaryEmbedding:
        def __init__(self, *args, **kwargs):
            pass

        def forward(self, *args, **kwargs):
            return torch.empty(0)

        __call__ = forward

    rope_utils._apply_rotary_pos_emb_bshd = _identity_rotary
    rope_utils._apply_rotary_pos_emb_thd = _identity_rotary
    rope_utils.get_pos_emb_on_this_cp_rank = lambda tensor, *args, **kwargs: tensor
    rotary.RotaryEmbedding = RotaryEmbedding
    embeddings.rope_utils = rope_utils
    embeddings.rotary_pos_embedding = rotary
    models_common.embeddings = embeddings
    models.common = models_common
    core.transformer = transformer
    core.models = models

    monkeypatch.setattr(megatron, "core", core, raising=False)
    for name, module in {
        "megatron.core": core,
        "megatron.core.transformer": transformer,
        "megatron.core.transformer.moe": moe,
        "megatron.core.transformer.moe.moe_utils": moe_utils,
        "megatron.core.models": models,
        "megatron.core.models.common": models_common,
        "megatron.core.models.common.embeddings": embeddings,
        "megatron.core.models.common.embeddings.rope_utils": rope_utils,
        "megatron.core.models.common.embeddings.rotary_pos_embedding": rotary,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


@pytest.fixture
def megatron_core_unit_stubs(monkeypatch):
    _install_transformer_engine_import_stub(monkeypatch)
    _install_megatron_core_unit_stubs(monkeypatch)
