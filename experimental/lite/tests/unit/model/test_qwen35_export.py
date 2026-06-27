# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from megatron.lite.model.qwen3_5.config import Qwen35Config
from megatron.lite.model.qwen3_5.lite.checkpoint import (
    PLACEMENT_FN,
    Qwen35WeightSpec,
    _merge_full_attn_qkvg,
    _merge_gate_up_tp_shards,
    _merge_linear_attn_conv1d_tp_shards,
    _merge_linear_attn_in_proj_tp_shards,
    export_hf_weights,
)
from megatron.lite.model.registry import (
    TRAIN_RUNTIME_MODULES,
    resolve_runtime_model_name,
)

_QWEN35_VISION_EXACT_REQUIRED_ENV = "MLITE_REQUIRE_QWEN35_VISION_EXACT"
_QWEN35_VISION_EXACT_PASS_MARKER = "QWEN35_VISION_EXACT_PASS"
_QWEN35_MTP_AUTHORITY_REVISION = "59d61f3ce65a6d9863b86d2e96597125219dc754"
_QWEN35_MTP_SCHEMA_SHA256 = (
    "54a42a904540f171d00a8296f2cd457dcdf4b167290f8a4455ee58c3dcb36871"
)


def _transformers_5_12_for_vision_exact():
    required = os.getenv(_QWEN35_VISION_EXACT_REQUIRED_ENV) == "1"
    try:
        transformers = __import__("transformers")
    except (ImportError, OSError) as exc:
        message = (
            "Qwen3.5 vision exact parity requires Transformers 5.12.0; "
            f"import failed: {type(exc).__name__}: {exc}"
        )
        if required:
            pytest.fail(message, pytrace=False)
        pytest.skip(message)

    version = getattr(transformers, "__version__", "unknown")
    if version != "5.12.0":
        message = (
            "Qwen3.5 vision exact parity is pinned to Transformers 5.12.0; "
            f"found {version}. Set {_QWEN35_VISION_EXACT_REQUIRED_ENV}=1 in "
            "Phase A to make this prerequisite fail instead of skip."
        )
        if required:
            pytest.fail(message, pytrace=False)
        pytest.skip(message)
    return transformers


def _tiny_config() -> Qwen35Config:
    return Qwen35Config(
        num_hidden_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        vocab_size=16,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=4,
        shared_expert_intermediate_size=4,
        linear_num_key_heads=2,
        linear_key_head_dim=2,
        linear_num_value_heads=2,
        linear_value_head_dim=2,
        linear_conv_kernel_dim=2,
        layer_types=["full_attention"],
        partial_rotary_factor=1.0,
    )


def _single_rank_parallel_state() -> SimpleNamespace:
    return SimpleNamespace(
        pp_size=1,
        pp_rank=0,
        tp_size=1,
        tp_rank=0,
        tp_group=None,
        ep_size=1,
        ep_rank=0,
        ep_group=None,
        etp_size=1,
        etp_rank=0,
        etp_group=None,
    )


class _TinyQwen35MTPModule(nn.Module):
    """CPU-only native MTP head with the same parameter names as Qwen35Model."""

    def __init__(self, config: Qwen35Config) -> None:
        super().__init__()
        self.layer_indices: list[int] = []
        next_value = 1

        def parameter(module: nn.Module, name: str, shape: tuple[int, ...]) -> None:
            nonlocal next_value
            numel = math.prod(shape)
            tensor = torch.arange(
                next_value, next_value + numel, dtype=torch.float32
            ).reshape(shape)
            next_value += numel
            module.register_parameter(name, nn.Parameter(tensor))

        self.mtp = nn.Module()
        layer = nn.Module()
        self.mtp.layers = nn.ModuleList([layer])
        for name in ("enorm", "hnorm", "final_layernorm"):
            module = nn.Module()
            parameter(module, "weight", (config.hidden_size,))
            setattr(layer, name, module)

        layer.eh_proj = nn.Module()
        layer.eh_proj.linear = nn.Module()
        parameter(
            layer.eh_proj.linear, "weight", (config.hidden_size, 2 * config.hidden_size)
        )

        transformer = nn.Module()
        layer.transformer_layer = transformer
        transformer.full_attn = nn.Module()
        transformer.full_attn.qkv = nn.Module()
        transformer.full_attn.qkv.linear = nn.Module()
        parameter(
            transformer.full_attn.qkv.linear, "layer_norm_weight", (config.hidden_size,)
        )
        q_heads_per_group = config.num_attention_heads // config.num_key_value_heads
        qkv_rows = (
            (2 * q_heads_per_group + 2) * config.head_dim * config.num_key_value_heads
        )
        parameter(
            transformer.full_attn.qkv.linear, "weight", (qkv_rows, config.hidden_size)
        )
        transformer.full_attn.q_norm = nn.Module()
        parameter(transformer.full_attn.q_norm, "weight", (config.head_dim,))
        transformer.full_attn.k_norm = nn.Module()
        parameter(transformer.full_attn.k_norm, "weight", (config.head_dim,))
        transformer.full_attn.proj = nn.Module()
        transformer.full_attn.proj.linear = nn.Module()
        parameter(
            transformer.full_attn.proj.linear,
            "weight",
            (config.hidden_size, config.num_attention_heads * config.head_dim),
        )

        transformer.mlp_norm = nn.Module()
        parameter(transformer.mlp_norm, "weight", (config.hidden_size,))
        transformer.moe = nn.Module()
        transformer.moe.router = nn.Module()
        transformer.moe.router.gate = nn.Module()
        parameter(
            transformer.moe.router.gate,
            "weight",
            (config.num_experts, config.hidden_size),
        )
        transformer.moe.shared_expert = nn.Module()
        transformer.moe.shared_expert.gate_up = nn.Module()
        transformer.moe.shared_expert.gate_up.linear = nn.Module()
        parameter(
            transformer.moe.shared_expert.gate_up.linear,
            "weight",
            (2 * config.shared_expert_intermediate_size, config.hidden_size),
        )
        transformer.moe.shared_expert.down = nn.Module()
        transformer.moe.shared_expert.down.linear = nn.Module()
        parameter(
            transformer.moe.shared_expert.down.linear,
            "weight",
            (config.hidden_size, config.shared_expert_intermediate_size),
        )
        transformer.moe.shared_expert.shared_gate = nn.Module()
        parameter(
            transformer.moe.shared_expert.shared_gate, "weight", (1, config.hidden_size)
        )

        transformer.moe.experts = nn.Module()
        transformer.moe.experts.fc1 = nn.Module()
        transformer.moe.experts.fc2 = nn.Module()
        for expert_idx in range(config.num_experts):
            parameter(
                transformer.moe.experts.fc1,
                f"weight{expert_idx}",
                (2 * config.moe_intermediate_size, config.hidden_size),
            )
            parameter(
                transformer.moe.experts.fc2,
                f"weight{expert_idx}",
                (config.hidden_size, config.moe_intermediate_size),
            )


def test_qwen35_protocol_registers_vllm_export_entrypoint() -> None:
    key = resolve_runtime_model_name("qwen3_5", "lite")
    module = __import__(TRAIN_RUNTIME_MODULES[key], fromlist=["export_hf_weights"])

    assert key == "qwen3_5"
    assert callable(module.export_hf_weights)


def test_qwen35_official_vision_checkpoint_load_export_exact(tmp_path) -> None:
    transformers = _transformers_5_12_for_vision_exact()
    from megatron.lite.model.qwen3_5.lite import checkpoint
    from safetensors.torch import save_file
    from transformers import Qwen3_5VisionConfig, Qwen3_5VisionModel

    vision_cfg = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=4,
        patch_size=2,
        temporal_patch_size=1,
        spatial_merge_size=1,
        out_hidden_size=8,
        num_position_embeddings=16,
    )
    torch.manual_seed(20260628)
    authority = Qwen3_5VisionModel(vision_cfg)
    source = {
        f"model.visual.{name}": tensor.detach().clone()
        for name, tensor in authority.state_dict().items()
    }
    save_file(source, tmp_path / "model.safetensors")

    class VisionOnlyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer_indices: list[int] = []
            self.vision_model = Qwen3_5VisionModel(vision_cfg).to(torch.bfloat16)
            self.mtp = None

    target = VisionOnlyModel()
    # Keep this deliberately smaller than the vision patch-embedding output.
    # Generic vocab trimming must never truncate ``vision_model.patch_embed``.
    cfg = SimpleNamespace(vocab_size=1)
    ps = _single_rank_parallel_state()
    checkpoint.load_hf_weights(target, str(tmp_path), cfg, ps)

    target_state = target.vision_model.state_dict()
    assert target_state.keys() == authority.state_dict().keys()
    for name, expected in authority.state_dict().items():
        torch.testing.assert_close(
            target_state[name], expected.to(torch.bfloat16), atol=0.0, rtol=0.0
        )

    exported = dict(checkpoint.export_hf_weights(target, cfg, ps))
    assert exported.keys() == source.keys()
    for hf_name, expected in source.items():
        torch.testing.assert_close(
            exported[hf_name], expected.to(torch.bfloat16), atol=0.0, rtol=0.0
        )

    placements = PLACEMENT_FN("vision_model.patch_embed.proj.weight")
    assert all(type(placement).__name__ == "Replicate" for placement in placements)
    spec = Qwen35WeightSpec(cfg, target="vllm")
    with pytest.raises(NotImplementedError, match="vision export"):
        spec.native_to_hf("vision_model.patch_embed.proj.weight", torch.ones(1))
    print(f"{_QWEN35_VISION_EXACT_PASS_MARKER} transformers={transformers.__version__}")


def test_qwen35_public_vllm_export_rejects_vision_before_first_yield() -> None:
    class TextThenVisionModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer_indices: list[int] = []
            self.embed = nn.Module()
            self.embed.embedding = nn.Embedding(4, 2)
            self.vision_model = nn.Linear(2, 2, bias=False)

    generator = export_hf_weights(
        TextThenVisionModel(),
        SimpleNamespace(vocab_size=4),
        _single_rank_parallel_state(),
        target="vllm",
    )
    with pytest.raises(NotImplementedError, match="vision export"):
        next(generator)


def test_qwen35_export_uses_hf_checkpoint_names_without_module_prefix() -> None:
    class TinyQwen35Module(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Module()
            self.embed.embedding = nn.Embedding(16, 8)
            self.norm = nn.LayerNorm(8)
            self.head = nn.Module()
            self.head.col = nn.Module()
            self.head.col.linear = nn.Linear(8, 16, bias=False)

    cfg = _tiny_config()
    model = TinyQwen35Module()

    exported = dict(export_hf_weights(model, cfg, _single_rank_parallel_state()))

    assert set(exported) == {
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
    }
    assert all(not name.startswith("module.") for name in exported)
    assert all(not name.startswith(("embed.", "norm.", "head.")) for name in exported)


def test_qwen35_mtp_hf_schema_exact_and_load_export_roundtrip(monkeypatch) -> None:
    from megatron.lite.model.qwen3_5.lite import checkpoint

    cfg = _tiny_config()
    cfg.num_nextn_predict_layers = 1
    cfg.mtp_layer_types = ["full_attention"]
    source = _TinyQwen35MTPModule(cfg)
    ps = _single_rank_parallel_state()

    exported = dict(export_hf_weights(source, cfg, ps))
    expected_shapes = {
        "mtp.pre_fc_norm_embedding.weight": (cfg.hidden_size,),
        "mtp.pre_fc_norm_hidden.weight": (cfg.hidden_size,),
        "mtp.fc.weight": (cfg.hidden_size, 2 * cfg.hidden_size),
        "mtp.norm.weight": (cfg.hidden_size,),
        "mtp.layers.0.input_layernorm.weight": (cfg.hidden_size,),
        "mtp.layers.0.self_attn.q_proj.weight": (
            2 * cfg.num_attention_heads * cfg.head_dim,
            cfg.hidden_size,
        ),
        "mtp.layers.0.self_attn.k_proj.weight": (
            cfg.num_key_value_heads * cfg.head_dim,
            cfg.hidden_size,
        ),
        "mtp.layers.0.self_attn.v_proj.weight": (
            cfg.num_key_value_heads * cfg.head_dim,
            cfg.hidden_size,
        ),
        "mtp.layers.0.self_attn.q_norm.weight": (cfg.head_dim,),
        "mtp.layers.0.self_attn.k_norm.weight": (cfg.head_dim,),
        "mtp.layers.0.self_attn.o_proj.weight": (
            cfg.hidden_size,
            cfg.num_attention_heads * cfg.head_dim,
        ),
        "mtp.layers.0.post_attention_layernorm.weight": (cfg.hidden_size,),
        "mtp.layers.0.mlp.gate.weight": (cfg.num_experts, cfg.hidden_size),
        "mtp.layers.0.mlp.shared_expert.gate_proj.weight": (
            cfg.shared_expert_intermediate_size,
            cfg.hidden_size,
        ),
        "mtp.layers.0.mlp.shared_expert.up_proj.weight": (
            cfg.shared_expert_intermediate_size,
            cfg.hidden_size,
        ),
        "mtp.layers.0.mlp.shared_expert.down_proj.weight": (
            cfg.hidden_size,
            cfg.shared_expert_intermediate_size,
        ),
        # Qwen/Qwen3.5-35B-A3B's published safetensors manifest carries the
        # shared-expert gate as one row over the hidden dimension.
        "mtp.layers.0.mlp.shared_expert_gate.weight": (1, cfg.hidden_size),
    }
    for expert_idx in range(cfg.num_experts):
        prefix = f"mtp.layers.0.mlp.experts.{expert_idx}"
        expected_shapes.update(
            {
                f"{prefix}.gate_proj.weight": (
                    cfg.moe_intermediate_size,
                    cfg.hidden_size,
                ),
                f"{prefix}.up_proj.weight": (
                    cfg.moe_intermediate_size,
                    cfg.hidden_size,
                ),
                f"{prefix}.down_proj.weight": (
                    cfg.hidden_size,
                    cfg.moe_intermediate_size,
                ),
            }
        )

    assert {name: tuple(tensor.shape) for name, tensor in exported.items()} == (
        expected_shapes
    )
    assert all(tensor.dtype == torch.float32 for tensor in exported.values())

    target = copy.deepcopy(source)
    for parameter in target.parameters():
        parameter.data.zero_()

    class MemoryReader:
        index = {name: "memory.safetensors" for name in exported}

        def __init__(self, _path: str) -> None:
            pass

        @staticmethod
        def get_tensor(name: str) -> torch.Tensor:
            return exported[name].clone()

    monkeypatch.setattr(checkpoint, "SafeTensorReader", MemoryReader)
    checkpoint.load_hf_weights(target, "memory", cfg, ps)

    source_state = source.state_dict()
    target_state = target.state_dict()
    assert source_state.keys() == target_state.keys()
    for name in source_state:
        torch.testing.assert_close(
            target_state[name], source_state[name], atol=0.0, rtol=0.0
        )


def test_qwen35_mtp_schema_matches_pinned_released_header_authority() -> None:
    """Match production mapping against immutable Hub safetensors headers.

    The compact fixture is an exact expansion of the MTP entries in the two
    released shards that contain them.  Its canonical digest is pinned here so
    the authority cannot drift together with test-generated expectations.
    Meta tensors exercise the real weight mapper at 35B dimensions without
    allocating the roughly 1.6 GiB predictor payload.
    """
    authority_path = Path(__file__).with_name("qwen35_mtp_header_authority.json")
    authority = json.loads(authority_path.read_text())
    assert authority["source"]["repo"] == "Qwen/Qwen3.5-35B-A3B"
    assert authority["source"]["revision"] == _QWEN35_MTP_AUTHORITY_REVISION
    assert authority["canonical_mtp_schema_sha256"] == _QWEN35_MTP_SCHEMA_SHA256

    dtype = authority["dtype"]
    expected = {
        name: {"dtype": dtype, "shape": shape}
        for name, shape in authority["non_expert_shapes"].items()
    }
    for template, shape in authority["expert_templates"].items():
        for expert_idx in range(authority["num_experts"]):
            expected[template.format(expert_idx=expert_idx)] = {
                "dtype": dtype,
                "shape": shape,
            }
    canonical = json.dumps(
        expected, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    assert len(expected) == 785
    assert hashlib.sha256(canonical).hexdigest() == _QWEN35_MTP_SCHEMA_SHA256

    cfg = _tiny_config()
    cfg.hidden_size = 2048
    cfg.num_attention_heads = 16
    cfg.num_key_value_heads = 2
    cfg.head_dim = 256
    cfg.num_experts = authority["num_experts"]
    cfg.moe_intermediate_size = 512
    cfg.shared_expert_intermediate_size = 512
    cfg.num_nextn_predict_layers = 1
    cfg.mtp_layer_types = ["full_attention"]
    with torch.device("meta"):
        source = _TinyQwen35MTPModule(cfg).to(dtype=torch.bfloat16)

    spec = Qwen35WeightSpec(cfg)
    actual: dict[str, dict[str, str | list[int]]] = {}
    for native_name, tensor in source.named_parameters():
        for hf_name, hf_tensor in spec.native_to_hf(native_name, tensor):
            assert hf_name not in actual, f"duplicate mapped MTP key: {hf_name}"
            assert hf_tensor.dtype == torch.bfloat16
            actual[hf_name] = {"dtype": "BF16", "shape": list(hf_tensor.shape)}

    assert len(actual) == 785
    assert actual == expected
    assert not spec._expert_export_buffers
    print(
        "NON_SKIP_QWEN35_PINNED_MTP_HEADER_AUTHORITY_PASSED "
        f"revision={_QWEN35_MTP_AUTHORITY_REVISION} mtp_keys={len(actual)} "
        f"dtype={dtype} schema_sha256={_QWEN35_MTP_SCHEMA_SHA256}",
        flush=True,
    )


def test_qwen35_mtp_fc_tp2_load_export_shape_roundtrip(monkeypatch) -> None:
    from megatron.lite.primitive.ckpt import hf_weights
    from torch.distributed.tensor import Replicate, Shard

    cfg = _tiny_config()
    cfg.num_nextn_predict_layers = 1
    cfg.mtp_layer_types = ["full_attention"]
    spec = Qwen35WeightSpec(cfg)
    native_name = "mtp.layers.0.eh_proj.linear.weight"
    full = torch.arange(
        cfg.hidden_size * 2 * cfg.hidden_size, dtype=torch.float32
    ).reshape(cfg.hidden_size, 2 * cfg.hidden_size)
    shards = full.chunk(2, dim=0)
    tp_group = object()
    ps = SimpleNamespace(tp_size=2, tp_group=tp_group)

    def fake_allgather_concat(tensor, world_size, group, dim):
        assert world_size == 2
        assert group is tp_group
        assert dim == 0
        torch.testing.assert_close(tensor, shards[1])
        return torch.cat(shards, dim=0)

    monkeypatch.setattr(hf_weights, "allgather_concat", fake_allgather_concat)

    assert spec.tp_spec(native_name) == (0, 0)
    placements = PLACEMENT_FN(native_name)
    assert all(isinstance(placement, Replicate) for placement in placements[:3])
    assert isinstance(placements[3], Shard)
    assert placements[3].dim == 0
    gathered = hf_weights._gather_dense(native_name, shards[1], spec, ps)
    exported = dict(spec.native_to_hf(native_name, gathered))

    assert exported.keys() == {"mtp.fc.weight"}
    assert exported["mtp.fc.weight"].shape == full.shape
    torch.testing.assert_close(exported["mtp.fc.weight"], full)


def test_qwen35_export_dtype_cast_is_opt_in() -> None:
    class TinyQwen35Module(nn.Module):
        def __init__(self, config: Qwen35Config) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(8)
            self.layers = nn.ModuleList([nn.Module()])
            self.layers[0].moe = nn.Module()
            self.layers[0].moe.experts = nn.Module()
            self.layers[0].moe.experts.fc1 = nn.Module()

            rows = config.moe_intermediate_size * 2
            for expert_idx in range(config.num_experts):
                tensor = torch.arange(
                    rows * config.hidden_size, dtype=torch.float32
                ).reshape(rows, config.hidden_size)
                tensor = tensor + expert_idx * 1000
                self.layers[0].moe.experts.fc1.register_parameter(
                    f"weight{expert_idx}", nn.Parameter(tensor)
                )

    cfg = _tiny_config()
    model = TinyQwen35Module(cfg)

    default_export = dict(export_hf_weights(model, cfg, _single_rank_parallel_state()))
    bf16_export = dict(
        export_hf_weights(
            model, cfg, _single_rank_parallel_state(), export_dtype="bfloat16"
        )
    )

    assert default_export["model.language_model.norm.weight"].dtype == torch.float32
    assert bf16_export["model.language_model.norm.weight"].dtype == torch.bfloat16
    assert (
        default_export["model.language_model.layers.0.mlp.experts.gate_up_proj"].dtype
        == torch.float32
    )
    assert (
        bf16_export["model.language_model.layers.0.mlp.experts.gate_up_proj"].dtype
        == torch.bfloat16
    )


def test_qwen35_export_preserves_runtime_parameter_dtype_by_default() -> None:
    class TinyQwen35Module(nn.Module):
        def __init__(self, config: Qwen35Config) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(8).to(torch.bfloat16)
            self.layers = nn.ModuleList([nn.Module()])
            self.layers[0].moe = nn.Module()
            self.layers[0].moe.experts = nn.Module()
            self.layers[0].moe.experts.fc1 = nn.Module()

            rows = config.moe_intermediate_size * 2
            for expert_idx in range(config.num_experts):
                tensor = torch.arange(
                    rows * config.hidden_size, dtype=torch.bfloat16
                ).reshape(rows, config.hidden_size)
                tensor = tensor + expert_idx * 1000
                self.layers[0].moe.experts.fc1.register_parameter(
                    f"weight{expert_idx}", nn.Parameter(tensor)
                )

    cfg = _tiny_config()
    model = TinyQwen35Module(cfg)

    exported = dict(export_hf_weights(model, cfg, _single_rank_parallel_state()))

    assert exported["model.language_model.norm.weight"].dtype == torch.bfloat16
    assert (
        exported["model.language_model.layers.0.mlp.experts.gate_up_proj"].dtype
        == torch.bfloat16
    )


def test_qwen35_export_batches_ep_expert_gather(monkeypatch) -> None:
    class TinyQwen35Module(nn.Module):
        def __init__(self, config: Qwen35Config) -> None:
            super().__init__()
            self.layers = nn.ModuleList([nn.Module()])
            self.layers[0].moe = nn.Module()
            self.layers[0].moe.experts = nn.Module()
            self.layers[0].moe.experts.fc1 = nn.Module()

            rows = config.moe_intermediate_size * 2
            for local_idx in range(config.num_experts // 2):
                tensor = torch.arange(
                    rows * config.hidden_size, dtype=torch.bfloat16
                ).reshape(rows, config.hidden_size)
                tensor = tensor + local_idx * 1000
                self.layers[0].moe.experts.fc1.register_parameter(
                    f"weight{local_idx}", nn.Parameter(tensor)
                )

    cfg = _tiny_config()
    model = TinyQwen35Module(cfg)
    ps = SimpleNamespace(
        pp_size=1,
        tp_size=1,
        tp_group=None,
        ep_size=2,
        ep_group=object(),
        etp_size=1,
        etp_group=None,
    )
    gather_calls = []

    def fake_all_gather(outputs, tensor, group=None):
        del group
        gather_calls.append(tensor.clone())
        outputs[0].copy_(tensor)
        outputs[1].copy_(tensor + 2000)

    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.hf_weights.dist.all_gather", fake_all_gather
    )

    exported = dict(export_hf_weights(model, cfg, ps))

    assert len(gather_calls) == 1
    assert gather_calls[0].shape[0] == cfg.num_experts // ps.ep_size
    local_tensors = [
        model.layers[0].moe.experts.fc1.weight0.detach(),
        model.layers[0].moe.experts.fc1.weight1.detach(),
    ]
    expected = torch.stack(
        [
            local_tensors[0],
            local_tensors[1],
            local_tensors[0] + 2000,
            local_tensors[1] + 2000,
        ],
        dim=0,
    )
    assert torch.equal(
        exported["model.language_model.layers.0.mlp.experts.gate_up_proj"], expected
    )


def test_qwen35_export_uses_packed_expert_group_names(monkeypatch) -> None:
    class TinyQwen35Module(nn.Module):
        def __init__(self, config: Qwen35Config) -> None:
            super().__init__()
            self.layers = nn.ModuleList([nn.Module()])
            self.layers[0].moe = nn.Module()
            self.layers[0].moe.experts = nn.Module()
            self.layers[0].moe.experts.fc1 = nn.Module()

            rows = config.moe_intermediate_size * 2
            for expert_idx in range(config.num_experts):
                tensor = torch.arange(
                    rows * config.hidden_size, dtype=torch.bfloat16
                ).reshape(rows, config.hidden_size)
                tensor = tensor + expert_idx * 1000
                self.layers[0].moe.experts.fc1.register_parameter(
                    f"weight{expert_idx}", nn.Parameter(tensor)
                )

    cfg = _tiny_config()
    seen_native_names = []
    original = Qwen35WeightSpec.native_to_hf

    def spy_native_to_hf(self, native_name, tensor):
        seen_native_names.append(native_name)
        return original(self, native_name, tensor)

    monkeypatch.setattr(Qwen35WeightSpec, "native_to_hf", spy_native_to_hf)

    exported = dict(
        export_hf_weights(TinyQwen35Module(cfg), cfg, _single_rank_parallel_state())
    )

    assert seen_native_names == ["layers.0.moe.experts.fc1.packed"]
    assert set(exported) == {"model.language_model.layers.0.mlp.experts.gate_up_proj"}


def test_qwen35_export_rank0_only_still_participates_in_ep_gather(monkeypatch) -> None:
    class TinyQwen35Module(nn.Module):
        def __init__(self, config: Qwen35Config) -> None:
            super().__init__()
            self.layers = nn.ModuleList([nn.Module()])
            self.layers[0].moe = nn.Module()
            self.layers[0].moe.experts = nn.Module()
            self.layers[0].moe.experts.fc1 = nn.Module()

            rows = config.moe_intermediate_size * 2
            for local_idx in range(config.num_experts // 2):
                tensor = (
                    torch.zeros(rows, config.hidden_size, dtype=torch.bfloat16)
                    + local_idx
                )
                self.layers[0].moe.experts.fc1.register_parameter(
                    f"weight{local_idx}", nn.Parameter(tensor)
                )

    cfg = _tiny_config()
    ps = SimpleNamespace(
        pp_size=1,
        tp_size=1,
        tp_group=None,
        ep_size=2,
        ep_group=object(),
        etp_size=1,
        etp_group=None,
    )
    gather_calls = []

    def fake_all_gather(outputs, tensor, group=None):
        del group
        gather_calls.append(tensor.clone())
        outputs[0].copy_(tensor)
        outputs[1].copy_(tensor + 2)

    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.hf_weights.dist.is_initialized", lambda: True
    )
    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.hf_weights.dist.get_rank", lambda: 1
    )
    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.hf_weights.dist.all_gather", fake_all_gather
    )

    exported = list(export_hf_weights(TinyQwen35Module(cfg), cfg, ps, rank0_only=True))

    assert exported == []
    assert len(gather_calls) == 1


def test_qwen35_export_maps_top_level_and_layer_norm_names() -> None:
    cfg = _tiny_config()
    spec = Qwen35WeightSpec(cfg)
    tensor = torch.arange(cfg.hidden_size)

    cases = {
        "embed.embedding.weight": "model.language_model.embed_tokens.weight",
        "norm.weight": "model.language_model.norm.weight",
        "head.col.linear.weight": "lm_head.weight",
        "layers.0.full_attn.qkv.linear.layer_norm_weight": (
            "model.language_model.layers.0.input_layernorm.weight"
        ),
        "layers.0.mlp_norm.weight": "model.language_model.layers.0.post_attention_layernorm.weight",
    }

    for native_name, hf_name in cases.items():
        exported = dict(spec.native_to_hf(native_name, tensor))
        assert set(exported) == {hf_name}
        assert torch.equal(exported[hf_name], tensor)


def test_qwen35_export_unpacks_full_attention_q_gate() -> None:
    cfg = _tiny_config()
    spec = Qwen35WeightSpec(cfg)
    hidden = cfg.hidden_size
    q_gate = torch.arange(cfg.num_attention_heads * 2 * cfg.head_dim * hidden).reshape(
        -1, hidden
    )
    key = torch.arange(
        q_gate.numel(), q_gate.numel() + cfg.num_key_value_heads * cfg.head_dim * hidden
    ).reshape(-1, hidden)
    value = torch.arange(
        key[-1, -1] + 1,
        key[-1, -1] + 1 + cfg.num_key_value_heads * cfg.head_dim * hidden,
    ).reshape(-1, hidden)

    packed = _merge_full_attn_qkvg(q_gate, key, value, cfg=cfg)
    exported = dict(spec.native_to_hf("layers.0.full_attn.qkv.linear.weight", packed))

    assert set(exported) == {
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.0.self_attn.k_proj.weight",
        "model.language_model.layers.0.self_attn.v_proj.weight",
    }
    assert torch.equal(
        exported["model.language_model.layers.0.self_attn.q_proj.weight"], q_gate
    )
    assert torch.equal(
        exported["model.language_model.layers.0.self_attn.k_proj.weight"], key
    )
    assert torch.equal(
        exported["model.language_model.layers.0.self_attn.v_proj.weight"], value
    )


def test_qwen35_export_maps_linear_attention_to_hf_checkpoint_names() -> None:
    cfg = _tiny_config()
    spec = Qwen35WeightSpec(cfg)
    qk_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim
    v_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim
    rows = qk_dim * 2 + v_dim * 2 + cfg.linear_num_value_heads * 2
    tensor = torch.arange(rows * cfg.hidden_size).reshape(rows, cfg.hidden_size)

    exported = dict(
        spec.native_to_hf("layers.0.linear_attn.in_proj.linear.weight", tensor)
    )

    assert set(exported) == {
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.language_model.layers.0.linear_attn.in_proj_z.weight",
        "model.language_model.layers.0.linear_attn.in_proj_b.weight",
        "model.language_model.layers.0.linear_attn.in_proj_a.weight",
    }
    assert (
        exported["model.language_model.layers.0.linear_attn.in_proj_qkv.weight"].shape[
            0
        ]
        == qk_dim * 2 + v_dim
    )
    assert (
        exported["model.language_model.layers.0.linear_attn.in_proj_z.weight"].shape[0]
        == v_dim
    )
    assert (
        exported["model.language_model.layers.0.linear_attn.in_proj_b.weight"].shape[0]
        == cfg.linear_num_value_heads
    )
    assert (
        exported["model.language_model.layers.0.linear_attn.in_proj_a.weight"].shape[0]
        == cfg.linear_num_value_heads
    )


def test_qwen35_export_reorders_linear_attention_tp_shards_before_hf_split() -> None:
    cfg = _tiny_config()
    qk_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim
    v_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim
    hidden = cfg.hidden_size
    parts = [
        torch.arange(0, qk_dim * hidden).reshape(qk_dim, hidden),
        torch.arange(100, 100 + qk_dim * hidden).reshape(qk_dim, hidden),
        torch.arange(200, 200 + v_dim * hidden).reshape(v_dim, hidden),
        torch.arange(300, 300 + v_dim * hidden).reshape(v_dim, hidden),
        torch.arange(400, 400 + cfg.linear_num_value_heads * hidden).reshape(
            cfg.linear_num_value_heads, hidden
        ),
        torch.arange(500, 500 + cfg.linear_num_value_heads * hidden).reshape(
            cfg.linear_num_value_heads, hidden
        ),
    ]
    full = torch.cat(parts, dim=0)
    shards = [
        torch.cat([part.chunk(2, dim=0)[rank] for part in parts], dim=0)
        for rank in range(2)
    ]

    merged = _merge_linear_attn_in_proj_tp_shards(shards, cfg=cfg)

    assert torch.equal(merged, full)


def test_qwen35_export_reorders_linear_attention_conv1d_tp_shards() -> None:
    cfg = _tiny_config()
    qk_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim
    v_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim
    trailing = (1, cfg.linear_conv_kernel_dim)
    parts = [
        torch.arange(
            0, qk_dim * trailing[0] * trailing[1], dtype=torch.float32
        ).reshape(qk_dim, *trailing),
        torch.arange(
            100, 100 + qk_dim * trailing[0] * trailing[1], dtype=torch.float32
        ).reshape(qk_dim, *trailing),
        torch.arange(
            200, 200 + v_dim * trailing[0] * trailing[1], dtype=torch.float32
        ).reshape(v_dim, *trailing),
    ]
    full = torch.cat(parts, dim=0)
    shards = [
        torch.cat([part.chunk(2, dim=0)[rank] for part in parts], dim=0)
        for rank in range(2)
    ]

    merged = _merge_linear_attn_conv1d_tp_shards(shards, cfg=cfg)

    assert torch.equal(merged, full)


def test_qwen35_export_uses_mbridge_conv1d_tp_gather(monkeypatch) -> None:
    class TinyQwen35Module(nn.Module):
        def __init__(self, local_shard: torch.Tensor) -> None:
            super().__init__()
            self.layers = nn.ModuleList([nn.Module()])
            self.layers[0].linear_attn = nn.Module()
            self.layers[0].linear_attn.conv1d = nn.Module()
            self.layers[0].linear_attn.conv1d.register_parameter(
                "weight", nn.Parameter(local_shard.clone())
            )

    cfg = _tiny_config()
    qk_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim
    v_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim
    trailing = (1, cfg.linear_conv_kernel_dim)
    parts = [
        torch.arange(
            0, qk_dim * trailing[0] * trailing[1], dtype=torch.float32
        ).reshape(qk_dim, *trailing),
        torch.arange(
            100, 100 + qk_dim * trailing[0] * trailing[1], dtype=torch.float32
        ).reshape(qk_dim, *trailing),
        torch.arange(
            200, 200 + v_dim * trailing[0] * trailing[1], dtype=torch.float32
        ).reshape(v_dim, *trailing),
    ]
    full = torch.cat(parts, dim=0)
    shards = [
        torch.cat([part.chunk(2, dim=0)[rank] for part in parts], dim=0)
        for rank in range(2)
    ]
    ps = SimpleNamespace(
        pp_size=1,
        tp_size=2,
        tp_group=object(),
        ep_size=1,
        ep_group=None,
        etp_size=1,
        etp_group=None,
    )
    gather_calls = []

    def fake_all_gather(outputs, tensor, group=None):
        assert group is ps.tp_group
        gather_calls.append(tensor.clone())
        outputs[0].copy_(shards[0])
        outputs[1].copy_(shards[1])

    monkeypatch.setattr(
        "megatron.lite.model.qwen3_5.lite.checkpoint.dist.all_gather", fake_all_gather
    )

    exported = dict(export_hf_weights(TinyQwen35Module(shards[0]), cfg, ps))

    assert len(gather_calls) == 1
    assert torch.equal(gather_calls[0], shards[0])
    assert torch.equal(
        exported["model.language_model.layers.0.linear_attn.conv1d.weight"], full
    )


def test_qwen35_export_reorders_shared_expert_gate_up_tp_shards() -> None:
    gate = torch.arange(0, 32).reshape(4, 8)
    up = torch.arange(100, 132).reshape(4, 8)
    full = torch.cat([gate, up], dim=0)
    shards = [
        torch.cat([gate.chunk(2, dim=0)[rank], up.chunk(2, dim=0)[rank]], dim=0)
        for rank in range(2)
    ]

    merged = _merge_gate_up_tp_shards(shards)

    assert torch.equal(merged, full)


def test_qwen35_export_restores_zero_centered_linear_attention_norm() -> None:
    cfg = _tiny_config()
    spec = Qwen35WeightSpec(cfg)
    tensor = torch.tensor([-0.5, 0.0, 0.5])

    exported = dict(spec.native_to_hf("layers.0.linear_attn.norm.weight", tensor))

    assert set(exported) == {"model.language_model.layers.0.linear_attn.norm.weight"}
    assert torch.equal(
        exported["model.language_model.layers.0.linear_attn.norm.weight"], tensor + 1
    )


def test_qwen35_export_preserves_official_gdn_mixed_dtypes() -> None:
    spec = Qwen35WeightSpec(_tiny_config())
    prefix = "layers.0.linear_attn"

    a_log = dict(
        spec.native_to_hf(f"{prefix}.A_log", torch.ones(2, dtype=torch.float32))
    )
    norm = dict(
        spec.native_to_hf(f"{prefix}.norm.weight", torch.ones(2, dtype=torch.float32))
    )
    dt_bias = dict(
        spec.native_to_hf(f"{prefix}.dt_bias", torch.ones(2, dtype=torch.bfloat16))
    )

    assert next(iter(a_log.values())).dtype == torch.float32
    assert next(iter(norm.values())).dtype == torch.float32
    assert next(iter(dt_bias.values())).dtype == torch.bfloat16


def test_qwen35_export_maps_shared_expert_to_hf_checkpoint_names() -> None:
    cfg = _tiny_config()
    spec = Qwen35WeightSpec(cfg)
    tensor = torch.arange(
        cfg.shared_expert_intermediate_size * 2 * cfg.hidden_size
    ).reshape(-1, cfg.hidden_size)

    exported = dict(
        spec.native_to_hf("layers.0.moe.shared_expert.gate_up.linear.weight", tensor)
    )

    assert set(exported) == {
        "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
        "model.language_model.layers.0.mlp.shared_expert.up_proj.weight",
    }
    gate, up = tensor.chunk(2, dim=0)
    assert torch.equal(
        exported["model.language_model.layers.0.mlp.shared_expert.gate_proj.weight"],
        gate,
    )
    assert torch.equal(
        exported["model.language_model.layers.0.mlp.shared_expert.up_proj.weight"], up
    )


def test_qwen35_export_packs_base_expert_fc1_to_hf_gate_up_proj() -> None:
    cfg = _tiny_config()
    spec = Qwen35WeightSpec(cfg)
    base = torch.arange(cfg.moe_intermediate_size * 2 * cfg.hidden_size).reshape(
        -1, cfg.hidden_size
    )

    exported = {}
    expert_tensors = []
    for expert_idx in range(cfg.num_experts):
        tensor = base + expert_idx * 1000
        expert_tensors.append(tensor)
        exported.update(
            dict(
                spec.native_to_hf(
                    f"layers.0.moe.experts.fc1.weight{expert_idx}", tensor
                )
            )
        )

    assert set(exported) == {"model.language_model.layers.0.mlp.experts.gate_up_proj"}
    assert torch.equal(
        exported["model.language_model.layers.0.mlp.experts.gate_up_proj"],
        torch.stack(expert_tensors, dim=0),
    )


def test_qwen35_export_matches_mbridge_qwen35_moe_packed_expert_contract() -> None:
    cfg = _tiny_config()
    spec = Qwen35WeightSpec(cfg)
    rows = cfg.moe_intermediate_size * 2
    fc1_tensors = [
        torch.arange(rows * cfg.hidden_size, dtype=torch.bfloat16).reshape(
            rows, cfg.hidden_size
        )
        + expert_idx * 1000
        for expert_idx in range(cfg.num_experts)
    ]
    fc2_tensors = [
        torch.arange(
            cfg.hidden_size * cfg.moe_intermediate_size, dtype=torch.bfloat16
        ).reshape(cfg.hidden_size, cfg.moe_intermediate_size)
        + expert_idx * 1000
        for expert_idx in range(cfg.num_experts)
    ]

    fc1_exported = {}
    fc2_exported = {}
    for expert_idx, (fc1, fc2) in enumerate(zip(fc1_tensors, fc2_tensors, strict=True)):
        fc1_exported.update(
            dict(spec.native_to_hf(f"layers.0.moe.experts.fc1.weight{expert_idx}", fc1))
        )
        fc2_exported.update(
            dict(spec.native_to_hf(f"layers.0.moe.experts.fc2.weight{expert_idx}", fc2))
        )

    assert set(fc1_exported) == {
        "model.language_model.layers.0.mlp.experts.gate_up_proj"
    }
    assert set(fc2_exported) == {"model.language_model.layers.0.mlp.experts.down_proj"}
    assert torch.equal(
        fc1_exported["model.language_model.layers.0.mlp.experts.gate_up_proj"],
        torch.stack(fc1_tensors, dim=0),
    )
    assert torch.equal(
        fc2_exported["model.language_model.layers.0.mlp.experts.down_proj"],
        torch.stack(fc2_tensors, dim=0),
    )


def test_qwen35_export_vllm_target_uses_runtime_prefix_and_packed_expert_names() -> (
    None
):
    cfg = _tiny_config()
    spec = Qwen35WeightSpec(cfg, target="vllm")
    dense = torch.arange(cfg.hidden_size)

    exported_embed = dict(spec.native_to_hf("embed.embedding.weight", dense))
    exported_norm = dict(spec.native_to_hf("norm.weight", dense))
    exported_head = dict(spec.native_to_hf("head.col.linear.weight", dense))
    exported_mlp_norm = dict(spec.native_to_hf("layers.0.mlp_norm.weight", dense))
    assert set(exported_embed) == {"language_model.model.embed_tokens.weight"}
    assert set(exported_norm) == {"language_model.model.norm.weight"}
    assert set(exported_head) == {"language_model.lm_head.weight"}
    assert set(exported_mlp_norm) == {
        "language_model.model.layers.0.post_attention_layernorm.weight"
    }
    assert torch.equal(
        exported_embed["language_model.model.embed_tokens.weight"], dense
    )
    assert torch.equal(exported_norm["language_model.model.norm.weight"], dense)
    assert torch.equal(exported_head["language_model.lm_head.weight"], dense)
    assert torch.equal(
        exported_mlp_norm[
            "language_model.model.layers.0.post_attention_layernorm.weight"
        ],
        dense,
    )

    fc1_tensors = [
        torch.arange(
            cfg.moe_intermediate_size * 2 * cfg.hidden_size, dtype=torch.bfloat16
        ).reshape(-1, cfg.hidden_size)
        + expert_idx * 1000
        for expert_idx in range(cfg.num_experts)
    ]
    fc2_tensors = [
        torch.arange(
            cfg.hidden_size * cfg.moe_intermediate_size, dtype=torch.bfloat16
        ).reshape(cfg.hidden_size, cfg.moe_intermediate_size)
        + expert_idx * 2000
        for expert_idx in range(cfg.num_experts)
    ]

    fc1_exported = {}
    fc2_exported = {}
    for expert_idx, (fc1, fc2) in enumerate(zip(fc1_tensors, fc2_tensors, strict=True)):
        fc1_exported.update(
            dict(spec.native_to_hf(f"layers.0.moe.experts.fc1.weight{expert_idx}", fc1))
        )
        fc2_exported.update(
            dict(spec.native_to_hf(f"layers.0.moe.experts.fc2.weight{expert_idx}", fc2))
        )

    assert set(fc1_exported) == {
        "language_model.model.layers.0.mlp.experts.gate_up_proj"
    }
    assert set(fc2_exported) == {"language_model.model.layers.0.mlp.experts.down_proj"}
    assert torch.equal(
        fc1_exported["language_model.model.layers.0.mlp.experts.gate_up_proj"],
        torch.stack(fc1_tensors, dim=0),
    )
    assert torch.equal(
        fc2_exported["language_model.model.layers.0.mlp.experts.down_proj"],
        torch.stack(fc2_tensors, dim=0),
    )


def test_qwen35_export_vllm_target_packs_experts_with_runtime_prefix() -> None:
    class TinyQwen35Module(nn.Module):
        def __init__(self, config: Qwen35Config) -> None:
            super().__init__()
            self.layers = nn.ModuleList([nn.Module()])
            self.layers[0].moe = nn.Module()
            self.layers[0].moe.experts = nn.Module()
            self.layers[0].moe.experts.fc1 = nn.Module()
            self.layers[0].moe.experts.fc2 = nn.Module()

            rows = config.moe_intermediate_size * 2
            for expert_idx in range(config.num_experts):
                fc1 = torch.arange(
                    rows * config.hidden_size, dtype=torch.bfloat16
                ).reshape(rows, config.hidden_size)
                fc1 = fc1 + expert_idx * 1000
                fc2 = torch.arange(
                    config.hidden_size * config.moe_intermediate_size,
                    dtype=torch.bfloat16,
                ).reshape(config.hidden_size, config.moe_intermediate_size)
                fc2 = fc2 + expert_idx * 2000
                self.layers[0].moe.experts.fc1.register_parameter(
                    f"weight{expert_idx}", nn.Parameter(fc1)
                )
                self.layers[0].moe.experts.fc2.register_parameter(
                    f"weight{expert_idx}", nn.Parameter(fc2)
                )

    cfg = _tiny_config()
    model = TinyQwen35Module(cfg)

    exported = dict(
        export_hf_weights(model, cfg, _single_rank_parallel_state(), target="vllm")
    )

    assert "model.language_model.layers.0.mlp.experts.gate_up_proj" not in exported
    assert "model.language_model.layers.0.mlp.experts.down_proj" not in exported
    assert (
        "language_model.model.layers.0.mlp.experts.0.gate_proj.weight" not in exported
    )
    assert (
        "language_model.model.layers.0.mlp.experts.0.down_proj.weight" not in exported
    )
    assert set(exported) == {
        "language_model.model.layers.0.mlp.experts.gate_up_proj",
        "language_model.model.layers.0.mlp.experts.down_proj",
    }

    expected_fc1 = []
    expected_fc2 = []
    for expert_idx in range(cfg.num_experts):
        fc1 = getattr(model.layers[0].moe.experts.fc1, f"weight{expert_idx}").detach()
        fc2 = getattr(model.layers[0].moe.experts.fc2, f"weight{expert_idx}").detach()
        expected_fc1.append(fc1)
        expected_fc2.append(fc2)

    assert torch.equal(
        exported["language_model.model.layers.0.mlp.experts.gate_up_proj"],
        torch.stack(expected_fc1, dim=0),
    )
    assert torch.equal(
        exported["language_model.model.layers.0.mlp.experts.down_proj"],
        torch.stack(expected_fc2, dim=0),
    )


def test_qwen35_export_packs_base_expert_fc2_and_expert_metadata() -> None:
    cfg = _tiny_config()
    spec = Qwen35WeightSpec(cfg)
    base = torch.arange(cfg.hidden_size * cfg.moe_intermediate_size).reshape(
        cfg.hidden_size, cfg.moe_intermediate_size
    )
    native_name = "layers.0.moe.experts.fc2.weight2"

    exported = {}
    expert_tensors = []
    for expert_idx in range(cfg.num_experts):
        tensor = base + expert_idx * 1000
        expert_tensors.append(tensor)
        exported.update(
            dict(
                spec.native_to_hf(
                    f"layers.0.moe.experts.fc2.weight{expert_idx}", tensor
                )
            )
        )

    assert set(exported) == {"model.language_model.layers.0.mlp.experts.down_proj"}
    assert torch.equal(
        exported["model.language_model.layers.0.mlp.experts.down_proj"],
        torch.stack(expert_tensors, dim=0),
    )
    assert spec.is_expert(native_name)
    assert spec.expert_global_id(native_name) == 2
    assert spec.expert_local_name(native_name, 0) == "layers.0.moe.experts.fc2.weight0"
    assert spec.tp_spec(native_name) == (1, 1)
