"""Static and CPU smoke tests for native DeepSeek V4 lite."""

from __future__ import annotations

import math
import re
from pathlib import Path


def _tiny_config_kwargs():
    return dict(
        vocab_size=64,
        hidden_size=24,
        moe_intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=6,
        qk_rope_head_dim=4,
        q_lora_rank=8,
        o_lora_rank=6,
        o_groups=2,
        n_routed_experts=3,
        n_shared_experts=1,
        num_experts_per_tok=2,
        num_hash_layers=1,
        hc_mult=2,
        compress_ratios=[4, 8, 4],
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        num_nextn_predict_layers=1,
    )


def _proxy_hf_tensors_from_native_state(state, config):
    import torch

    from megatron.lite.model.deepseek_v4.lite.checkpoint import _hf_name_for_state_key

    hf_tensors = {}
    native_state = {}
    for index, (name, target) in enumerate(state.items()):
        hf_name = _hf_name_for_state_key(name)
        assert hf_name is not None
        if target.dtype.is_floating_point:
            values = torch.arange(target.numel(), dtype=torch.float32).reshape(target.shape)
            values = ((values % 19) - 9) / 512.0 + ((index % 7) + 1) / 1024.0
            if name.endswith(".norm.weight"):
                values = values + 1.0
            tensor = values.to(dtype=target.dtype)
        else:
            tensor = (
                torch.arange(target.numel(), dtype=torch.int64).reshape(target.shape)
                % config.n_routed_experts
            ).to(dtype=target.dtype)
        native_state[name] = tensor.contiguous()
        hf_tensors[hf_name] = tensor.contiguous()
    return hf_tensors, native_state


def test_deepseek_v4_registry_resolves_lite():
    from megatron.lite.model.registry import (
        get_train_runtime_module,
        resolve_model_type_from_hf,
        resolve_runtime_model_name,
    )

    runtime_name = resolve_runtime_model_name("deepseek_v4", "lite")
    assert runtime_name == "deepseek_v4"
    module = get_train_runtime_module(runtime_name)
    assert module.__name__ == "megatron.lite.model.deepseek_v4.lite.protocol"
    assert resolve_model_type_from_hf({"model_type": "deepseek_v4"}) == "deepseek_v4"


def test_deepseek_v4_config_reads_hf_fields():
    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config

    cfg = DeepseekV4Config._from_hf_dict(
        {
            "model_type": "deepseek_v4",
            "hidden_size": 6144,
            "num_hidden_layers": 61,
            "num_attention_heads": 128,
            "num_key_value_heads": 1,
            "head_dim": 128,
            "qk_rope_head_dim": 64,
            "q_lora_rank": 1536,
            "o_lora_rank": 1024,
            "o_groups": 8,
            "n_routed_experts": 256,
            "n_shared_experts": 1,
            "num_experts_per_tok": 6,
            "compress_ratios": [0] * 62,
            "num_nextn_predict_layers": 1,
            "vocab_size": 129280,
            "rope_parameters": {"rope_theta": 10000.0},
        }
    )

    assert cfg.q_lora_rank == 1536
    assert cfg.o_lora_rank == 1024
    assert cfg.compress_ratios == [0] * 62
    assert cfg.rope_theta == 10000.0


def test_deepseek_v4_lite_does_not_import_wrappers_or_sibling_models():
    root = (
        Path(__file__).resolve().parents[3]
        / "megatron"
        / "lite"
        / "model"
        / "deepseek_v4"
        / "lite"
    )
    for path in root.glob("*.py"):
        text = path.read_text()
        assert "mbridge" not in text
        assert "megatron.core" not in text
        assert "megatron.lite.model.qwen" not in text
        assert "Automodel" not in text


def test_deepseek_v4_lite_implementation_files_stay_small():
    root = (
        Path(__file__).resolve().parents[3]
        / "megatron"
        / "lite"
        / "model"
        / "deepseek_v4"
        / "lite"
    )
    for name in ("model.py", "protocol.py", "checkpoint.py"):
        line_count = len((root / name).read_text().splitlines())
        assert line_count < 1000, f"{name} has {line_count} lines"


def test_deepseek_v4_model_exports_hf_style_state_names():
    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
    from megatron.lite.model.deepseek_v4.lite.model import DeepseekV4ForCausalLM

    model = DeepseekV4ForCausalLM(DeepseekV4Config(**_tiny_config_kwargs()))
    keys = set(model.state_dict())

    assert "model.embed_tokens.weight" in keys
    assert "model.layers.0.self_attn.wq_a.weight" in keys
    assert "model.layers.0.self_attn.q_norm.weight" in keys
    assert "model.layers.0.self_attn.wq_b.weight" in keys
    assert "model.layers.0.self_attn.wkv.weight" in keys
    assert "model.layers.0.self_attn.compressor.wkv.weight" in keys
    assert "model.layers.0.self_attn.indexer.wq_b.weight" in keys
    assert "model.layers.1.self_attn.compressor.wgate.weight" in keys
    assert "model.layers.0.self_attn.wo_a.weight" in keys
    assert "model.layers.0.self_attn.sinks" in keys
    assert "model.layers.0.mlp.gate.weight" in keys
    assert "model.layers.0.mlp.gate.tid2eid" in keys
    assert "model.layers.1.mlp.gate.e_score_correction_bias" in keys
    assert "model.layers.0.mlp.gate.e_score_correction_bias" not in keys
    assert "model.layers.1.mlp.gate.tid2eid" not in keys
    assert "model.layers.0.mlp.experts.0.gate_proj.weight" in keys
    assert "model.layers.0.mlp.shared_experts.gate_proj.weight" in keys
    assert "model.layers.0.attn_hc.fn" in keys
    assert "model.hc_head.hc_fn" in keys
    assert "model.mtp.0.e_proj.weight" in keys
    assert "model.mtp.0.self_attn.wq_a.weight" in keys
    assert "model.mtp.0.hc_head.hc_fn" in keys
    assert "lm_head.weight" in keys


def test_deepseek_v4_hf_name_mapping_covers_native_keys():
    from megatron.lite.model.deepseek_v4.lite.checkpoint import _hf_name_for_state_key

    assert _hf_name_for_state_key("model.embed_tokens.weight") == "embed.weight"
    assert _hf_name_for_state_key("model.layers.2.self_attn.sinks") == "layers.2.attn.attn_sink"
    assert _hf_name_for_state_key("model.layers.2.self_attn.compressor.wkv.weight") == "layers.2.attn.compressor.wkv.weight"
    assert _hf_name_for_state_key("model.layers.2.self_attn.indexer.wq_b.weight") == "layers.2.attn.indexer.wq_b.weight"
    assert _hf_name_for_state_key("model.layers.2.mlp.gate.e_score_correction_bias") == "layers.2.ffn.gate.bias"
    assert _hf_name_for_state_key("model.layers.2.mlp.gate.tid2eid") == "layers.2.ffn.gate.tid2eid"
    assert _hf_name_for_state_key("model.layers.2.mlp.shared_experts.up_proj.weight") == "layers.2.ffn.shared_experts.w3.weight"
    assert _hf_name_for_state_key("model.layers.2.mlp.experts.7.down_proj.weight") == "layers.2.ffn.experts.7.w2.weight"
    assert _hf_name_for_state_key("model.layers.2.attn_hc.scale") == "layers.2.hc_attn_scale"
    assert _hf_name_for_state_key("model.hc_head.hc_scale") == "hc_head_scale"
    assert _hf_name_for_state_key("model.mtp.0.self_attn.wq_a.weight") == "mtp.0.attn.wq_a.weight"
    assert _hf_name_for_state_key("model.mtp.0.e_proj.weight") == "mtp.0.e_proj.weight"
    assert _hf_name_for_state_key("model.mtp.0.hc_head.hc_scale") == "mtp.0.hc_head_scale"


def test_deepseek_v4_hf_name_mapping_covers_every_native_state_key():
    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
    from megatron.lite.model.deepseek_v4.lite.checkpoint import _hf_name_for_state_key
    from megatron.lite.model.deepseek_v4.lite.model import DeepseekV4ForCausalLM

    model = DeepseekV4ForCausalLM(DeepseekV4Config(**_tiny_config_kwargs()))
    mapped = {name: _hf_name_for_state_key(name) for name in model.state_dict()}

    assert [name for name, hf_name in mapped.items() if hf_name is None] == []
    hf_names = list(mapped.values())
    assert len(hf_names) == len(set(hf_names))


def test_deepseek_v4_load_hf_weights_from_synthetic_safetensors(tmp_path):
    import torch
    from safetensors.torch import save_file

    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
    from megatron.lite.model.deepseek_v4.lite.checkpoint import (
        _dequantize_scaled_tensor,
        _hf_name_for_state_key,
        _scale_name_for_hf_name,
        load_hf_weights,
    )
    from megatron.lite.model.deepseek_v4.lite.model import DeepseekV4ForCausalLM
    from megatron.lite.primitive.parallel import ParallelState

    cfg = DeepseekV4Config(**_tiny_config_kwargs())
    model = DeepseekV4ForCausalLM(cfg)
    state = model.state_dict()
    hf_tensors = {}
    expected = {}
    for index, (name, target) in enumerate(state.items()):
        hf_name = _hf_name_for_state_key(name)
        assert hf_name is not None
        if target.dtype.is_floating_point:
            if name.endswith(".weight") and target.ndim >= 2:
                raw = (
                    torch.arange(target.numel(), dtype=torch.int16).reshape(target.shape)
                    % 15
                ).to(torch.int8)
                scale_shape = tuple(max(1, (dim + 1) // 2) for dim in target.shape)
                scale = torch.full(scale_shape, 127, dtype=torch.uint8)
                if scale.numel():
                    scale.flatten()[::2] = 128
                tensor = raw
                hf_tensors[_scale_name_for_hf_name(hf_name)] = scale
                expected[name] = _dequantize_scaled_tensor(raw, scale, target.shape).to(dtype=target.dtype)
            else:
                tensor = (
                    torch.arange(target.numel(), dtype=torch.float32).reshape(target.shape)
                    + float(index)
                )
                if target.numel():
                    tensor = tensor / max(target.numel(), 1)
                expected[name] = tensor.to(dtype=target.dtype)
        else:
            tensor = (
                torch.arange(target.numel(), dtype=torch.int64).reshape(target.shape)
                % cfg.n_routed_experts
            )
            expected[name] = tensor.to(dtype=target.dtype)
        hf_tensors[hf_name] = tensor.contiguous()

    save_file(hf_tensors, tmp_path / "model.safetensors")
    load_hf_weights(model, str(tmp_path), cfg, ParallelState())

    loaded = model.state_dict()
    for name, tensor in expected.items():
        if tensor.dtype.is_floating_point:
            torch.testing.assert_close(loaded[name], tensor)
        else:
            assert torch.equal(loaded[name], tensor)


def test_deepseek_v4_proxy_hf_loaded_forward_matches_native_bitwise(tmp_path):
    import torch
    from safetensors.torch import save_file

    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
    from megatron.lite.model.deepseek_v4.lite.checkpoint import load_hf_weights
    from megatron.lite.model.deepseek_v4.lite.model import DeepseekV4ForCausalLM
    from megatron.lite.primitive.parallel import ParallelState

    cfg = DeepseekV4Config(**_tiny_config_kwargs())
    native_ref = DeepseekV4ForCausalLM(cfg)
    bb_model = DeepseekV4ForCausalLM(cfg)
    hf_tensors, native_state = _proxy_hf_tensors_from_native_state(native_ref.state_dict(), cfg)

    save_file(hf_tensors, tmp_path / "model.safetensors")
    native_ref.load_state_dict(native_state, strict=True)
    load_hf_weights(bb_model, str(tmp_path), cfg, ParallelState())
    native_ref.eval()
    bb_model.eval()

    input_ids = (torch.arange(8, dtype=torch.long).unsqueeze(0) * 7) % cfg.vocab_size
    position_ids = torch.arange(input_ids.size(1), dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        native_out = native_ref(input_ids=input_ids, position_ids=position_ids)
        bb_out = bb_model(input_ids=input_ids, position_ids=position_ids)

    assert torch.equal(bb_out["hidden_states"], native_out["hidden_states"])
    assert torch.equal(bb_out["logits"], native_out["logits"])

    mtp_hidden = torch.arange(
        input_ids.size(0) * input_ids.size(1) * cfg.hc_mult * cfg.hidden_size,
        dtype=torch.float32,
    ).reshape(input_ids.size(0), input_ids.size(1), cfg.hc_mult, cfg.hidden_size)
    mtp_hidden = (mtp_hidden % 23) / 128.0
    with torch.no_grad():
        native_mtp = native_ref.model.mtp[0](
            mtp_hidden,
            input_ids=input_ids,
            embed_tokens=native_ref.model.embed_tokens,
            position_ids=position_ids,
        )
        bb_mtp = bb_model.model.mtp[0](
            mtp_hidden,
            input_ids=input_ids,
            embed_tokens=bb_model.model.embed_tokens,
            position_ids=position_ids,
        )
        native_mtp_contract = native_ref.model.mtp[0].contract(native_mtp)
        bb_mtp_contract = bb_model.model.mtp[0].contract(bb_mtp)

    assert torch.equal(bb_mtp, native_mtp)
    assert torch.equal(bb_mtp_contract, native_mtp_contract)


def test_deepseek_v4_real_hf_index_coverage():
    import json
    import os

    import pytest

    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
    from megatron.lite.model.deepseek_v4.lite.checkpoint import expected_hf_names

    snapshot = Path(
        os.environ.get(
            "DSV4_HF_SNAPSHOT",
            "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_mcore/checkpoints/hf/DeepSeek-V4-Flash",
        )
    )
    index_path = snapshot / "model.safetensors.index.json"
    config_path = snapshot / "config.json"
    if not index_path.exists() or not config_path.exists():
        pytest.skip("DeepSeek-V4-Flash HF index/config not present in local cache")

    hf_weight_map = json.loads(index_path.read_text())["weight_map"]
    cfg = DeepseekV4Config._from_hf_dict(json.loads(config_path.read_text()))
    expected = expected_hf_names(cfg, available_hf_names=hf_weight_map)

    assert len(hf_weight_map) == 69187
    assert sorted(set(hf_weight_map) - expected) == []
    assert sorted(expected - set(hf_weight_map)) == []


def _deepseek_v4_hf_snapshot() -> Path:
    import os

    return Path(
        os.environ.get(
            "DSV4_HF_SNAPSHOT",
            "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_mcore/checkpoints/hf/DeepSeek-V4-Flash",
        )
    )


def _state_key_for_real_proxy_hf_name(name: str) -> str | None:
    if name in {
        "embed.weight",
        "head.weight",
        "norm.weight",
        "hc_head_fn",
        "hc_head_base",
        "hc_head_scale",
    }:
        return {
            "embed.weight": "model.embed_tokens.weight",
            "head.weight": "lm_head.weight",
            "norm.weight": "model.norm.weight",
            "hc_head_fn": "model.hc_head.hc_fn",
            "hc_head_base": "model.hc_head.hc_base",
            "hc_head_scale": "model.hc_head.hc_scale",
        }[name]
    if name.endswith(".scale"):
        return None

    matched = re.match(r"^(layers\.(\d+)|mtp\.(\d+))\.(.+)$", name)
    if matched is None:
        return None
    native = f"model.layers.{matched.group(2)}" if matched.group(2) is not None else f"model.mtp.{matched.group(3)}"
    suffix = matched.group(4)

    direct = {
        "attn_norm.weight": "input_layernorm.weight",
        "ffn_norm.weight": "post_attention_layernorm.weight",
        "attn.attn_sink": "self_attn.sinks",
        "attn.wq_a.weight": "self_attn.wq_a.weight",
        "attn.q_norm.weight": "self_attn.q_norm.weight",
        "attn.wq_b.weight": "self_attn.wq_b.weight",
        "attn.wkv.weight": "self_attn.wkv.weight",
        "attn.kv_norm.weight": "self_attn.kv_norm.weight",
        "attn.wo_a.weight": "self_attn.wo_a.weight",
        "attn.wo_b.weight": "self_attn.wo_b.weight",
        "attn.compressor.ape": "self_attn.compressor.ape",
        "attn.compressor.wkv.weight": "self_attn.compressor.wkv.weight",
        "attn.compressor.wgate.weight": "self_attn.compressor.wgate.weight",
        "attn.compressor.norm.weight": "self_attn.compressor.norm.weight",
        "attn.indexer.wq_b.weight": "self_attn.indexer.wq_b.weight",
        "attn.indexer.compressor.ape": "self_attn.indexer.compressor.ape",
        "attn.indexer.compressor.wkv.weight": "self_attn.indexer.compressor.wkv.weight",
        "attn.indexer.compressor.wgate.weight": "self_attn.indexer.compressor.wgate.weight",
        "attn.indexer.compressor.norm.weight": "self_attn.indexer.compressor.norm.weight",
        "attn.indexer.weights_proj.weight": "self_attn.indexer.weights_proj.weight",
        "ffn.gate.weight": "mlp.gate.weight",
        "ffn.gate.bias": "mlp.gate.e_score_correction_bias",
        "ffn.gate.tid2eid": "mlp.gate.tid2eid",
        "ffn.shared_experts.w1.weight": "mlp.shared_experts.gate_proj.weight",
        "ffn.shared_experts.w3.weight": "mlp.shared_experts.up_proj.weight",
        "ffn.shared_experts.w2.weight": "mlp.shared_experts.down_proj.weight",
        "hc_attn_fn": "attn_hc.fn",
        "hc_attn_base": "attn_hc.base",
        "hc_attn_scale": "attn_hc.scale",
        "hc_ffn_fn": "ffn_hc.fn",
        "hc_ffn_base": "ffn_hc.base",
        "hc_ffn_scale": "ffn_hc.scale",
        "e_proj.weight": "e_proj.weight",
        "h_proj.weight": "h_proj.weight",
        "enorm.weight": "enorm.weight",
        "hnorm.weight": "hnorm.weight",
        "norm.weight": "norm.weight",
        "hc_head_fn": "hc_head.hc_fn",
        "hc_head_base": "hc_head.hc_base",
        "hc_head_scale": "hc_head.hc_scale",
    }
    if suffix in direct:
        return f"{native}.{direct[suffix]}"

    expert = re.match(r"^ffn\.experts\.(\d+)\.(w1|w3|w2)\.weight$", suffix)
    if expert is not None:
        proj = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}[expert.group(2)]
        return f"{native}.mlp.experts.{expert.group(1)}.{proj}.weight"
    return None


def _mbridge_ref_dequant(tensor, scale, shape):
    """mbridge#147 DeepSeek weight_dequant reference: y = x.float() * block_scale."""
    scale_f = scale.float()
    target = tuple(int(dim) for dim in shape)
    while scale_f.ndim > len(target) and scale_f.shape[0] == 1:
        scale_f = scale_f.squeeze(0)
    while scale_f.ndim < len(target):
        scale_f = scale_f.unsqueeze(-1)
    expanded = scale_f
    for dim, size in enumerate(target):
        if expanded.shape[dim] == size:
            continue
        expanded = expanded.repeat_interleave(math.ceil(size / expanded.shape[dim]), dim=dim)
    return tensor.float() * expanded[tuple(slice(0, size) for size in target)]


def test_deepseek_v4_real_checkpoint_proxy_loader_bitwise():
    import json
    import os

    import pytest
    import torch
    from safetensors import safe_open

    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
    from megatron.lite.model.deepseek_v4.lite.checkpoint import (
        _copy_param,
        _hf_name_for_state_key,
        _scale_name_for_hf_name,
    )

    if os.environ.get("DSV4_REAL_PROXY_PARITY") != "1":
        pytest.skip("set DSV4_REAL_PROXY_PARITY=1 to run real DeepSeek-V4-Flash proxy parity")

    snapshot = _deepseek_v4_hf_snapshot()
    index_path = snapshot / "model.safetensors.index.json"
    config_path = snapshot / "config.json"
    if not index_path.exists() or not config_path.exists():
        pytest.skip(f"DeepSeek-V4-Flash config/index not present: {snapshot}")

    hf_index = json.loads(index_path.read_text())["weight_map"]
    first_shard = snapshot / hf_index["layers.0.attn.wq_a.weight"]
    if not first_shard.exists():
        pytest.skip(f"DeepSeek-V4-Flash full safetensor shards not present: {snapshot}")

    cfg = DeepseekV4Config._from_hf_dict(json.loads(config_path.read_text()))
    proxy_layers = (0, cfg.num_hidden_layers - 1)
    prefixes = {f"layers.{idx}." for idx in proxy_layers}
    include_mtp = os.environ.get("DSV4_REAL_PROXY_INCLUDE_MTP") == "1"
    if include_mtp and cfg.num_nextn_predict_layers:
        prefixes.update(f"mtp.{idx}." for idx in range(cfg.num_nextn_predict_layers))

    state_to_hf: dict[str, str] = {}
    for hf_name in hf_index:
        if not any(hf_name.startswith(prefix) for prefix in prefixes):
            continue
        state_key = _state_key_for_real_proxy_hf_name(hf_name)
        if state_key is None:
            if not hf_name.endswith(".scale"):
                raise AssertionError(f"real proxy HF name is not mapped to native state: {hf_name}")
            continue
        assert _hf_name_for_state_key(state_key) == hf_name
        state_to_hf[state_key] = hf_name
    for hf_name in ("norm.weight", "hc_head_fn", "hc_head_base", "hc_head_scale"):
        state_key = _state_key_for_real_proxy_hf_name(hf_name)
        assert state_key is not None
        assert _hf_name_for_state_key(state_key) == hf_name
        state_to_hf[state_key] = hf_name

    contexts = []
    handles = {}
    tensor_device = "cuda" if torch.cuda.is_available() else "cpu"

    def read_tensor(name: str) -> torch.Tensor:
        filename = hf_index[name]
        handle = handles.get(filename)
        if handle is None:
            context = safe_open(str(snapshot / filename), framework="pt", device=tensor_device)
            handle = context.__enter__()
            contexts.append(context)
            handles[filename] = handle
        return handle.get_tensor(name)

    checked = 0
    quantized = 0
    global_max = 0.0
    worst = ""
    try:
        for state_key, hf_name in sorted(state_to_hf.items()):
            raw = read_tensor(hf_name)
            scale_name = _scale_name_for_hf_name(hf_name)
            scale = read_tensor(scale_name) if scale_name in hf_index else None
            if scale is not None:
                quantized += 1
                expected = _mbridge_ref_dequant(raw, scale, raw.shape).to(torch.float32)
                target = torch.empty(raw.shape, dtype=torch.float32, device=raw.device)
            elif raw.dtype.is_floating_point:
                expected = raw.to(torch.float32)
                target = torch.empty(raw.shape, dtype=torch.float32, device=raw.device)
            else:
                expected = raw.to(torch.int64)
                target = torch.empty(raw.shape, dtype=torch.int64, device=raw.device)

            _copy_param(target, raw, scale=scale)
            if target.dtype.is_floating_point:
                diff = (target - expected).abs()
                max_abs = float(diff.max().item()) if diff.numel() else 0.0
            else:
                max_abs = 0.0 if torch.equal(target, expected) else float("inf")
            if max_abs > global_max:
                global_max = max_abs
                worst = state_key
            assert torch.equal(target, expected), f"{state_key} ({hf_name}) max_abs={max_abs}"
            checked += 1
    finally:
        while contexts:
            contexts.pop().__exit__(None, None, None)

    print(
        "[DSV4_REAL_PROXY] "
        f"snapshot={snapshot} device={tensor_device} proxy_layers={proxy_layers} include_mtp={include_mtp} "
        f"checked={checked} quantized={quantized} max_abs={global_max} worst={worst}"
    )
    assert checked > 0
    assert quantized > 0
    assert global_max == 0.0


def test_deepseek_v4_protocol_rejects_cp_and_other_parallel_scope():
    import pytest

    from megatron.lite.model.deepseek_v4.lite.protocol import _validate_parallel_scope
    from megatron.lite.runtime.contracts import ParallelConfig

    _validate_parallel_scope(ParallelConfig(tp=1, ep=1, etp=1, cp=1, pp=1, vpp=1))
    with pytest.raises(NotImplementedError):
        _validate_parallel_scope(ParallelConfig(tp=1, ep=1, etp=1, cp=2, pp=1, vpp=1))
    with pytest.raises(NotImplementedError):
        _validate_parallel_scope(ParallelConfig(tp=2, ep=1, etp=1, cp=1, pp=1, vpp=1))


def test_deepseek_v4_lite_tiny_cpu_forward_backward():
    import torch

    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
    from megatron.lite.model.deepseek_v4.lite.model import DeepseekV4ForCausalLM

    torch.manual_seed(1234)
    model = DeepseekV4ForCausalLM(DeepseekV4Config(**_tiny_config_kwargs()))
    input_ids = torch.randint(0, model.config.vocab_size, (2, 5))
    labels = torch.randint(0, model.config.vocab_size, (2, 5))

    output = model(input_ids=input_ids, labels=labels)

    assert output["hidden_states"].shape == (2, 5, model.config.hidden_size)
    assert output["logits"].shape == (2, 5, model.config.vocab_size)
    assert output["loss"].ndim == 0
    output["loss"].backward()
    grad_norm = sum(
        param.grad.detach().float().norm()
        for param in model.parameters()
        if param.grad is not None
    )
    assert torch.isfinite(grad_norm)
