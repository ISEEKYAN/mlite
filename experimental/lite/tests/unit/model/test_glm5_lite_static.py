"""Static and CPU smoke tests for native GLM-5 lite."""

from __future__ import annotations

from pathlib import Path


def _tiny_config_kwargs():
    return dict(
        num_hidden_layers=2,
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=32,
        max_position_embeddings=16,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_head_dim=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        index_head_dim=8,
        index_n_heads=2,
        index_topk=2,
        intermediate_size=20,
        moe_intermediate_size=6,
        first_k_dense_replace=1,
        n_routed_experts=3,
        n_shared_experts=1,
        num_experts_per_tok=2,
    )


def test_glm5_registry_resolves_lite():
    from megatron.lite.model.registry import (
        get_train_runtime_module,
        resolve_model_type_from_hf,
        resolve_runtime_model_name,
    )

    runtime_name = resolve_runtime_model_name("glm5", "lite")
    assert runtime_name == "glm5"
    module = get_train_runtime_module(runtime_name)
    assert module.__name__ == "megatron.lite.model.glm5.lite.protocol"
    assert resolve_model_type_from_hf({"model_type": "glm_moe_dsa"}) == "glm5"


def test_glm5_config_reads_hf_architecture_fields():
    from megatron.lite.model.glm5.config import Glm5Config

    cfg = Glm5Config._from_hf_dict(
        {
            "model_type": "glm_moe_dsa",
            "hidden_size": 6144,
            "num_hidden_layers": 78,
            "num_attention_heads": 64,
            "num_key_value_heads": 64,
            "q_lora_rank": 2048,
            "kv_lora_rank": 512,
            "qk_head_dim": 256,
            "qk_nope_head_dim": 192,
            "qk_rope_head_dim": 64,
            "v_head_dim": 256,
            "index_head_dim": 128,
            "index_n_heads": 32,
            "index_topk": 2048,
            "first_k_dense_replace": 3,
            "n_routed_experts": 256,
            "n_shared_experts": 1,
            "num_experts_per_tok": 8,
            "vocab_size": 154880,
            "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
        }
    )

    assert cfg.q_lora_rank == 2048
    assert cfg.kv_lora_rank == 512
    assert cfg.index_topk == 2048
    assert cfg.rope_theta == 1_000_000.0
    assert cfg.is_moe_layer(2) is False
    assert cfg.is_moe_layer(3) is True


def test_glm5_config_ignores_null_hf_optional_fields():
    from megatron.lite.model.glm5.config import Glm5Config

    cfg = Glm5Config._from_hf_dict(
        {
            "model_type": "glm_moe_dsa",
            "indexer_rope_first": None,
            "indexer_use_hadamard": None,
            "mlp_layer_types": None,
        }
    )

    assert cfg.indexer_rope_first is True
    assert cfg.indexer_use_hadamard is False
    assert cfg.mlp_layer_types is None


def test_glm5_lite_does_not_import_wrappers_or_sibling_models():
    root = Path(__file__).resolve().parents[3] / "megatron" / "lite" / "model" / "glm5" / "lite"
    for path in root.glob("*.py"):
        text = path.read_text()
        assert "megatron.lite.model.qwen" not in text
        assert "mbridge" not in text
        assert "MCore" not in text
        assert "megatron.core" not in text


def test_glm5_lite_uses_primitive_mla_dsa():
    root = Path(__file__).resolve().parents[3] / "megatron" / "lite"
    model_text = (root / "model" / "glm5" / "lite" / "model.py").read_text()
    primitive_text = (root / "primitive" / "modules" / "mla_dsa.py").read_text()

    assert "from megatron.lite.primitive.modules.mla_dsa import MLADSA" in model_text
    assert "class MLADSA" in primitive_text
    assert "class DSAIndexer" in primitive_text


def test_glm5_lite_model_exports_hf_style_state_names():
    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.model.glm5.lite.model import Glm5ForCausalLM

    model = Glm5ForCausalLM(Glm5Config(**_tiny_config_kwargs()))
    keys = set(model.state_dict())

    assert "model.embed_tokens.weight" in keys
    assert "model.layers.0.self_attn.q_a_proj.weight" in keys
    assert "model.layers.0.mlp.gate_proj.weight" in keys
    assert "model.layers.1.mlp.gate.weight" in keys
    assert "model.layers.1.mlp.experts.0.gate_proj.weight" in keys
    assert "model.layers.1.mlp.shared_experts.gate_proj.weight" in keys
    assert "lm_head.weight" in keys


def test_glm5_hf_loader_resolves_grouped_expert_tensors():
    import torch

    from megatron.lite.model.glm5.lite.checkpoint import _resolve_hf_tensor

    class FakeReader:
        def __init__(self, tensors):
            self.tensors = tensors
            self.index = {name: "model.safetensors" for name in tensors}

        def get_tensor(self, name):
            return self.tensors[name]

    hf_gate_up = torch.arange(3 * 12 * 16, dtype=torch.float32).reshape(3, 12, 16)
    hf_down = torch.arange(3 * 16 * 6, dtype=torch.float32).reshape(3, 16, 6)
    hf_reader = FakeReader(
        {
            "model.layers.1.mlp.experts.gate_up_proj": hf_gate_up,
            "model.layers.1.mlp.experts.down_proj": hf_down,
        }
    )

    gate_target = torch.empty(6, 16)
    down_target = torch.empty(16, 6)
    assert torch.equal(
        _resolve_hf_tensor(
            hf_reader,
            "model.layers.1.mlp.experts.2.gate_proj.weight",
            gate_target,
        ),
        hf_gate_up[2, :6, :],
    )
    assert torch.equal(
        _resolve_hf_tensor(
            hf_reader,
            "model.layers.1.mlp.experts.2.up_proj.weight",
            gate_target,
        ),
        hf_gate_up[2, 6:, :],
    )
    assert torch.equal(
        _resolve_hf_tensor(
            hf_reader,
            "model.layers.1.mlp.experts.2.down_proj.weight",
            down_target,
        ),
        hf_down[2],
    )

    automodel_gate_up = torch.arange(3 * 16 * 12, dtype=torch.float32).reshape(3, 16, 12)
    automodel_down = torch.arange(3 * 6 * 16, dtype=torch.float32).reshape(3, 6, 16)
    automodel_reader = FakeReader(
        {
            "model.layers.1.mlp.experts.gate_and_up_projs": automodel_gate_up,
            "model.layers.1.mlp.experts.down_projs": automodel_down,
        }
    )
    assert torch.equal(
        _resolve_hf_tensor(
            automodel_reader,
            "model.layers.1.mlp.experts.1.gate_proj.weight",
            gate_target,
        ),
        automodel_gate_up[1, :, :6].T,
    )
    assert torch.equal(
        _resolve_hf_tensor(
            automodel_reader,
            "model.layers.1.mlp.experts.1.up_proj.weight",
            gate_target,
        ),
        automodel_gate_up[1, :, 6:].T,
    )
    assert torch.equal(
        _resolve_hf_tensor(
            automodel_reader,
            "model.layers.1.mlp.experts.1.down_proj.weight",
            down_target,
        ),
        automodel_down[1].T,
    )


def test_glm5_hf_loader_slices_full_expert_tensors_for_proxy_targets():
    import torch

    from megatron.lite.model.glm5.lite.checkpoint import _slice_to_target_shape

    source = torch.arange(256 * 8, dtype=torch.float32).reshape(256, 8)
    target = torch.empty(4, 8)
    assert torch.equal(_slice_to_target_shape(source, target), source[:4])

    source3d = torch.arange(256 * 6 * 8, dtype=torch.float32).reshape(256, 6, 8)
    target3d = torch.empty(4, 6, 8)
    assert torch.equal(_slice_to_target_shape(source3d, target3d), source3d[:4])


def test_glm5_protocol_allows_cp_only_parallel_scope():
    import pytest

    from megatron.lite.model.glm5.lite.protocol import _validate_parallel_scope
    from megatron.lite.runtime.contracts import ParallelConfig

    _validate_parallel_scope(ParallelConfig(tp=1, ep=1, etp=1, cp=2, pp=1, vpp=1))
    with pytest.raises(NotImplementedError):
        _validate_parallel_scope(ParallelConfig(tp=2, ep=1, etp=1, cp=1, pp=1, vpp=1))


def test_glm5_protocol_uses_mlite_optimizer_api():
    from megatron.lite.model.glm5.lite.protocol import ImplConfig

    protocol_path = (
        Path(__file__).resolve().parents[3]
        / "megatron"
        / "lite"
        / "model"
        / "glm5"
        / "lite"
        / "protocol.py"
    )
    protocol_text = protocol_path.read_text()

    assert ImplConfig().optimizer == "mc"
    assert "build_mc_training_optimizer" in protocol_text
    assert "build_mc_full_stack" not in protocol_text


def test_glm5_lite_tiny_cpu_forward_backward():
    import torch

    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.model.glm5.lite.model import Glm5ForCausalLM

    torch.manual_seed(1234)
    model = Glm5ForCausalLM(Glm5Config(**_tiny_config_kwargs()))
    input_ids = torch.randint(0, model.config.vocab_size, (2, 5))
    labels = torch.randint(0, model.config.vocab_size, (2, 5))

    output = model(input_ids=input_ids, labels=labels)

    assert output["hidden_states"].shape == (2, 5, model.config.hidden_size)
    assert output["loss"].ndim == 0
    output["loss"].backward()
    grad_norm = sum(
        param.grad.detach().float().norm()
        for param in model.parameters()
        if param.grad is not None
    )
    assert torch.isfinite(grad_norm)
