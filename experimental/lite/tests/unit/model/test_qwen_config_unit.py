from __future__ import annotations

import pytest
import torch

from megatron.lite.model.qwen3_moe.config import Qwen3MoEConfig
from megatron.lite.model.qwen3_moe.lite.checkpoint import Qwen3MoEWeightSpec
from megatron.lite.model.registry import (
    resolve_model_type_from_hf,
    resolve_runtime_model_name,
)


pytestmark = pytest.mark.mlite


def _require_transformer_engine() -> None:
    try:
        __import__("transformer_engine.pytorch")
    except (ImportError, OSError) as exc:
        pytest.skip(f"transformer_engine.pytorch is unavailable: {exc}")


def _tiny_qwen3_hf_dict() -> dict:
    return {
        "model_type": "qwen3_moe",
        "hidden_size": 16,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_hidden_layers": 1,
        "vocab_size": 64,
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 8,
        "rope_parameters": {"rope_theta": 12345.0},
    }

def test_registry_resolves_qwen_lite_model_names():
    assert resolve_model_type_from_hf({"model_type": "qwen3_moe"}) == "qwen3"
    assert resolve_runtime_model_name("qwen3", "lite") == "qwen3"
    assert resolve_runtime_model_name("qwen3_moe", "lite") == "qwen3_moe"


def test_qwen3_config_from_hf_dict_derives_head_dim_and_rope_theta():
    cfg = Qwen3MoEConfig._from_hf_dict(_tiny_qwen3_hf_dict())

    assert cfg.hidden_size == 16
    assert cfg.head_dim == 4
    assert cfg.layer_types == ["full_attention"]
    assert cfg.rope_theta == 12345.0


def test_qwen3_config_rejects_invalid_expert_topk():
    hf = _tiny_qwen3_hf_dict()
    hf["num_experts_per_tok"] = 3

    with pytest.raises(ValueError, match="num_experts_per_tok"):
        Qwen3MoEConfig._from_hf_dict(hf)

def test_qwen_lite_protocols_build_configs_from_hf_dicts():
    _require_transformer_engine()

    from megatron.lite.model.qwen3_moe.lite import protocol as qwen3_protocol

    qwen3_cfg = qwen3_protocol.build_model_config(_tiny_qwen3_hf_dict(), vocab_size=128)

    assert qwen3_cfg.vocab_size == 128


def test_qwen3_weight_spec_round_trips_qkv_and_expert_tensors():
    cfg = Qwen3MoEConfig._from_hf_dict(_tiny_qwen3_hf_dict())
    spec = Qwen3MoEWeightSpec(cfg)
    q = torch.arange(64, dtype=torch.float32).reshape(16, 4)
    k = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    v = torch.arange(32, dtype=torch.float32).reshape(8, 4)

    packed = spec.hf_to_native("layers.0.attn.qkv.linear.weight", [q, k, v])
    exported = dict(spec.native_to_hf("layers.0.attn.qkv.linear.weight", packed))

    torch.testing.assert_close(exported["model.layers.0.self_attn.q_proj.weight"], q)
    torch.testing.assert_close(exported["model.layers.0.self_attn.k_proj.weight"], k)
    torch.testing.assert_close(exported["model.layers.0.self_attn.v_proj.weight"], v)

    gate = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    up = gate + 100
    fc1 = spec.hf_to_native("layers.0.moe.experts._fc1_weight_0", [gate, up])
    exported_fc1 = dict(spec.native_to_hf("layers.0.moe.experts._fc1_weight_0", fc1))

    torch.testing.assert_close(exported_fc1["model.layers.0.mlp.experts.0.gate_proj.weight"], gate)
    torch.testing.assert_close(exported_fc1["model.layers.0.mlp.experts.0.up_proj.weight"], up)
