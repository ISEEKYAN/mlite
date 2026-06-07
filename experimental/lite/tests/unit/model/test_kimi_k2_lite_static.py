"""Static smoke coverage for Kimi K2 lite native implementation."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_kimi_k2_lite_registry_resolves():
    from megatron.lite.model.registry import (
        get_train_runtime_module,
        resolve_model_type_from_hf,
        resolve_runtime_model_name,
    )

    runtime_name = resolve_runtime_model_name("kimi_k2", "lite")
    assert runtime_name == "kimi_k2"
    assert resolve_model_type_from_hf({"model_type": "kimi_k2"}) == "kimi_k2"
    assert resolve_model_type_from_hf({"model_type": "deepseek_v3"}) == "kimi_k2"
    module = get_train_runtime_module(runtime_name)
    assert module.__name__ == "megatron.lite.model.kimi_k2.lite.protocol"


def test_kimi_k2_config_reads_hf_fields():
    from megatron.lite.model.kimi_k2.config import KimiK2Config

    cfg = KimiK2Config._from_hf_dict(
        {
            "model_type": "kimi_k2",
            "num_hidden_layers": 3,
            "hidden_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "vocab_size": 128,
            "intermediate_size": 64,
            "moe_intermediate_size": 16,
            "n_routed_experts": 8,
            "n_shared_experts": 2,
            "num_experts_per_tok": 2,
            "n_group": 4,
            "topk_group": 2,
            "topk_method": "noaux_tc",
            "norm_topk_prob": True,
            "scoring_func": "sigmoid",
            "seq_aux": True,
            "first_k_dense_replace": 1,
            "q_lora_rank": 12,
            "kv_lora_rank": 10,
            "qk_nope_head_dim": 6,
            "qk_rope_head_dim": 2,
            "v_head_dim": 8,
            "rope_scaling": {"type": "yarn", "factor": 32.0},
        }
    )

    assert cfg.num_experts == 8
    assert cfg.n_group == 4
    assert cfg.topk_group == 2
    assert cfg.shared_expert_intermediate_size == 32
    assert cfg.q_head_dim == 8
    assert not cfg.is_moe_layer(0)
    assert cfg.is_moe_layer(1)


def test_kimi_k2_lite_does_not_import_wrappers_or_sibling_models():
    root = (
        Path(__file__).resolve().parents[3]
        / "megatron"
        / "lite"
        / "model"
        / "kimi_k2"
        / "lite"
    )
    forbidden = (
        "megatron.lite.model.qwen3_5",
        "megatron.lite.model.qwen3_moe",
        "bridge_model",
        "hybrid",
        "build_mcore_context",
        "from mbridge",
        "import mbridge",
    )
    for path in root.glob("*.py"):
        text = path.read_text()
        for token in forbidden:
            assert token not in text


def test_kimi_k2_lite_implementation_files_stay_small():
    root = (
        Path(__file__).resolve().parents[3]
        / "megatron"
        / "lite"
        / "model"
        / "kimi_k2"
        / "lite"
    )
    for name in ("model.py", "protocol.py", "checkpoint.py"):
        line_count = len((root / name).read_text().splitlines())
        assert line_count < 1000, f"{name} has {line_count} lines"


def test_kimi_k2_lite_uses_mla_primitive():
    root = (
        Path(__file__).resolve().parents[3]
        / "megatron"
        / "lite"
        / "model"
        / "kimi_k2"
        / "lite"
    )
    model_text = (root / "model.py").read_text()

    assert "from megatron.lite.primitive.modules.mla import MultiLatentAttention" in model_text
    assert "class MultiLatentAttention" not in model_text
    assert "SigmoidTopKRouter" in model_text


def test_kimi_k2_fp8_checkpoint_dequant_cpu_path():
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch float8_e4m3fn is required for this smoke.")

    from megatron.lite.model.kimi_k2.lite.checkpoint import _dequant_fp8_weight

    class Reader:
        index = {"w_scale_inv": "fake.safetensors"}

        @staticmethod
        def get_tensor(name):
            assert name == "w_scale_inv"
            return torch.full((1, 1), 2.0, dtype=torch.float32)

    weight = torch.tensor([[1.0, -2.0], [3.0, -4.0]], dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    out = _dequant_fp8_weight(Reader(), "w", weight)

    torch.testing.assert_close(out, weight.float() * 2.0)


def test_kimi_k2_int4_checkpoint_dequant_cpu_path():
    torch = pytest.importorskip("torch")

    from megatron.lite.model.kimi_k2.lite.checkpoint import _get

    values = torch.tensor([[-8, -7, -1, 0, 1, 2, 6, 7, -8, 7]], dtype=torch.int8)
    unsigned = (values + 8).to(torch.int32)
    packed = torch.zeros((1, 2), dtype=torch.int32)
    for offset in range(8):
        packed[:, 0] |= unsigned[:, offset] << (4 * offset)
    for offset in range(2):
        packed[:, 1] |= unsigned[:, 8 + offset] << (4 * offset)

    class Reader:
        index = {
            "w_packed": "fake.safetensors",
            "w_scale": "fake.safetensors",
            "w_shape": "fake.safetensors",
        }

        @staticmethod
        def get_tensor(name):
            return {
                "w_packed": packed,
                "w_scale": torch.tensor([[0.5, 2.0]], dtype=torch.float32),
                "w_shape": torch.tensor([1, 10], dtype=torch.int64),
            }[name]

    out = _get(Reader(), "w")
    expected = torch.cat([values[:, :5].float() * 0.5, values[:, 5:].float() * 2.0], dim=1)

    torch.testing.assert_close(out, expected)


def test_kimi_k2_real_checkpoint_prefix_helpers():
    from megatron.lite.model.kimi_k2.lite.checkpoint import _lm_head_name, _text_prefix

    class Reader:
        index = {
            "language_model.model.embed_tokens.weight": "fake.safetensors",
            "language_model.lm_head.weight": "fake.safetensors",
        }

    prefix = _text_prefix(Reader())

    assert prefix == "language_model.model"
    assert _lm_head_name(Reader(), prefix) == "language_model.lm_head.weight"
