"""Direct HF safetensor loading for native DeepSeek V4 lite."""

from __future__ import annotations

from collections.abc import Iterable
import math
import re

import torch
import torch.nn as nn

from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
from megatron.lite.primitive.ckpt.hf_weights import SafeTensorReader, unwrap_model
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.utils import log_rank0


def EXPERT_CLASSIFIER(name: str) -> bool:
    return ".experts." in name and ".shared_experts." not in name


_LAYER_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^model\.layers\.(\d+)\.input_layernorm\.weight$"), r"layers.\1.attn_norm.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.post_attention_layernorm\.weight$"), r"layers.\1.ffn_norm.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.wq_a\.weight$"), r"layers.\1.attn.wq_a.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.q_norm\.weight$"), r"layers.\1.attn.q_norm.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.wq_b\.weight$"), r"layers.\1.attn.wq_b.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.wkv\.weight$"), r"layers.\1.attn.wkv.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.kv_norm\.weight$"), r"layers.\1.attn.kv_norm.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.wo_a\.weight$"), r"layers.\1.attn.wo_a.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.wo_b\.weight$"), r"layers.\1.attn.wo_b.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.sinks$"), r"layers.\1.attn.attn_sink"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.compressor\.(.+)$"), r"layers.\1.attn.compressor.\2"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.indexer\.(.+)$"), r"layers.\1.attn.indexer.\2"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.gate\.weight$"), r"layers.\1.ffn.gate.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.gate\.e_score_correction_bias$"), r"layers.\1.ffn.gate.bias"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.gate\.tid2eid$"), r"layers.\1.ffn.gate.tid2eid"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.shared_experts\.gate_proj\.weight$"), r"layers.\1.ffn.shared_experts.w1.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.shared_experts\.up_proj\.weight$"), r"layers.\1.ffn.shared_experts.w3.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.shared_experts\.down_proj\.weight$"), r"layers.\1.ffn.shared_experts.w2.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.attn_hc\.fn$"), r"layers.\1.hc_attn_fn"),
    (re.compile(r"^model\.layers\.(\d+)\.attn_hc\.base$"), r"layers.\1.hc_attn_base"),
    (re.compile(r"^model\.layers\.(\d+)\.attn_hc\.scale$"), r"layers.\1.hc_attn_scale"),
    (re.compile(r"^model\.layers\.(\d+)\.ffn_hc\.fn$"), r"layers.\1.hc_ffn_fn"),
    (re.compile(r"^model\.layers\.(\d+)\.ffn_hc\.base$"), r"layers.\1.hc_ffn_base"),
    (re.compile(r"^model\.layers\.(\d+)\.ffn_hc\.scale$"), r"layers.\1.hc_ffn_scale"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.gate_proj\.weight$"), r"layers.\1.ffn.experts.\2.w1.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.up_proj\.weight$"), r"layers.\1.ffn.experts.\2.w3.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.down_proj\.weight$"), r"layers.\1.ffn.experts.\2.w2.weight"),
]

_MTP_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^model\.mtp\.(\d+)\.input_layernorm\.weight$"), r"mtp.\1.attn_norm.weight"),
    (re.compile(r"^model\.mtp\.(\d+)\.post_attention_layernorm\.weight$"), r"mtp.\1.ffn_norm.weight"),
    (re.compile(r"^model\.mtp\.(\d+)\.self_attn\.wq_a\.weight$"), r"mtp.\1.attn.wq_a.weight"),
    (re.compile(r"^model\.mtp\.(\d+)\.self_attn\.q_norm\.weight$"), r"mtp.\1.attn.q_norm.weight"),
    (re.compile(r"^model\.mtp\.(\d+)\.self_attn\.wq_b\.weight$"), r"mtp.\1.attn.wq_b.weight"),
    (re.compile(r"^model\.mtp\.(\d+)\.self_attn\.wkv\.weight$"), r"mtp.\1.attn.wkv.weight"),
    (re.compile(r"^model\.mtp\.(\d+)\.self_attn\.kv_norm\.weight$"), r"mtp.\1.attn.kv_norm.weight"),
    (re.compile(r"^model\.mtp\.(\d+)\.self_attn\.wo_a\.weight$"), r"mtp.\1.attn.wo_a.weight"),
    (re.compile(r"^model\.mtp\.(\d+)\.self_attn\.wo_b\.weight$"), r"mtp.\1.attn.wo_b.weight"),
    (re.compile(r"^model\.mtp\.(\d+)\.self_attn\.sinks$"), r"mtp.\1.attn.attn_sink"),
    (re.compile(r"^model\.mtp\.(\d+)\.self_attn\.compressor\.(.+)$"), r"mtp.\1.attn.compressor.\2"),
    (re.compile(r"^model\.mtp\.(\d+)\.self_attn\.indexer\.(.+)$"), r"mtp.\1.attn.indexer.\2"),
    (re.compile(r"^model\.mtp\.(\d+)\.mlp\.gate\.weight$"), r"mtp.\1.ffn.gate.weight"),
    (re.compile(r"^model\.mtp\.(\d+)\.mlp\.gate\.e_score_correction_bias$"), r"mtp.\1.ffn.gate.bias"),
    (re.compile(r"^model\.mtp\.(\d+)\.mlp\.gate\.tid2eid$"), r"mtp.\1.ffn.gate.tid2eid"),
    (re.compile(r"^model\.mtp\.(\d+)\.mlp\.shared_experts\.gate_proj\.weight$"), r"mtp.\1.ffn.shared_experts.w1.weight"),
    (re.compile(r"^model\.mtp\.(\d+)\.mlp\.shared_experts\.up_proj\.weight$"), r"mtp.\1.ffn.shared_experts.w3.weight"),
    (re.compile(r"^model\.mtp\.(\d+)\.mlp\.shared_experts\.down_proj\.weight$"), r"mtp.\1.ffn.shared_experts.w2.weight"),
    (re.compile(r"^model\.mtp\.(\d+)\.attn_hc\.fn$"), r"mtp.\1.hc_attn_fn"),
    (re.compile(r"^model\.mtp\.(\d+)\.attn_hc\.base$"), r"mtp.\1.hc_attn_base"),
    (re.compile(r"^model\.mtp\.(\d+)\.attn_hc\.scale$"), r"mtp.\1.hc_attn_scale"),
    (re.compile(r"^model\.mtp\.(\d+)\.ffn_hc\.fn$"), r"mtp.\1.hc_ffn_fn"),
    (re.compile(r"^model\.mtp\.(\d+)\.ffn_hc\.base$"), r"mtp.\1.hc_ffn_base"),
    (re.compile(r"^model\.mtp\.(\d+)\.ffn_hc\.scale$"), r"mtp.\1.hc_ffn_scale"),
    (re.compile(r"^model\.mtp\.(\d+)\.mlp\.experts\.(\d+)\.gate_proj\.weight$"), r"mtp.\1.ffn.experts.\2.w1.weight"),
    (re.compile(r"^model\.mtp\.(\d+)\.mlp\.experts\.(\d+)\.up_proj\.weight$"), r"mtp.\1.ffn.experts.\2.w3.weight"),
    (re.compile(r"^model\.mtp\.(\d+)\.mlp\.experts\.(\d+)\.down_proj\.weight$"), r"mtp.\1.ffn.experts.\2.w2.weight"),
    (re.compile(r"^model\.mtp\.(\d+)\.e_proj\.weight$"), r"mtp.\1.e_proj.weight"),
    (re.compile(r"^model\.mtp\.(\d+)\.h_proj\.weight$"), r"mtp.\1.h_proj.weight"),
    (re.compile(r"^model\.mtp\.(\d+)\.enorm\.weight$"), r"mtp.\1.enorm.weight"),
    (re.compile(r"^model\.mtp\.(\d+)\.hnorm\.weight$"), r"mtp.\1.hnorm.weight"),
    (re.compile(r"^model\.mtp\.(\d+)\.norm\.weight$"), r"mtp.\1.norm.weight"),
    (re.compile(r"^model\.mtp\.(\d+)\.hc_head\.hc_fn$"), r"mtp.\1.hc_head_fn"),
    (re.compile(r"^model\.mtp\.(\d+)\.hc_head\.hc_base$"), r"mtp.\1.hc_head_base"),
    (re.compile(r"^model\.mtp\.(\d+)\.hc_head\.hc_scale$"), r"mtp.\1.hc_head_scale"),
]

_TOP_LEVEL = {
    "model.embed_tokens.weight": "embed.weight",
    "model.norm.weight": "norm.weight",
    "model.hc_head.hc_fn": "hc_head_fn",
    "model.hc_head.hc_base": "hc_head_base",
    "model.hc_head.hc_scale": "hc_head_scale",
    "lm_head.weight": "head.weight",
}


def _has(reader: SafeTensorReader, name: str) -> bool:
    if reader.index:
        return name in reader.index
    try:
        reader.get_tensor(name)
    except Exception:
        return False
    return True


def _scale_name_for_hf_name(name: str) -> str:
    if name.endswith(".weight"):
        return f"{name[:-7]}.scale"
    return f"{name}.scale"


def _scale_to_float(scale: torch.Tensor) -> torch.Tensor:
    if scale.dtype.is_floating_point:
        return scale.float()
    if scale.dtype == torch.uint8:
        return torch.pow(torch.tensor(2.0, dtype=torch.float32), scale.float() - 127.0)
    return scale.float()


def _expand_block_scale(scale: torch.Tensor, target_shape: torch.Size | tuple[int, ...]) -> torch.Tensor:
    target = tuple(int(dim) for dim in target_shape)
    while scale.ndim > len(target) and scale.shape[0] == 1:
        scale = scale.squeeze(0)
    while scale.ndim < len(target):
        scale = scale.unsqueeze(-1)
    if tuple(scale.shape) == target:
        return scale
    out = scale
    for dim, size in enumerate(target):
        if out.shape[dim] == size:
            continue
        repeat = math.ceil(size / out.shape[dim])
        out = out.repeat_interleave(repeat, dim=dim)
    slices = tuple(slice(0, size) for size in target)
    return out[slices]


def _dequantize_scaled_tensor(tensor: torch.Tensor, scale: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    scale_f = _expand_block_scale(_scale_to_float(scale), shape)
    return tensor.float() * scale_f


def _copy_param(
    param: nn.Parameter | torch.Tensor,
    tensor: torch.Tensor,
    *,
    scale: torch.Tensor | None = None,
) -> None:
    if scale is not None:
        tensor = _dequantize_scaled_tensor(tensor, scale, param.shape)
    elif param.dtype.is_floating_point and not tensor.dtype.is_floating_point:
        raise RuntimeError(
            f"Refusing to copy quantized tensor with dtype {tensor.dtype} into {tuple(param.shape)} "
            "without a matching .scale tensor."
        )
    param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))


def _hf_name_for_state_key(name: str) -> str | None:
    mapped = _TOP_LEVEL.get(name)
    if mapped is not None:
        return mapped
    for pattern, replacement in (*_LAYER_PATTERNS, *_MTP_PATTERNS):
        new_key, count = pattern.subn(replacement, name)
        if count:
            return new_key
    return None


def _layer_base_hf_names(prefix: str, config: DeepseekV4Config, *, layer_idx: int) -> list[str]:
    names = [
        f"{prefix}.attn_norm.weight",
        f"{prefix}.ffn_norm.weight",
        f"{prefix}.hc_attn_fn",
        f"{prefix}.hc_attn_base",
        f"{prefix}.hc_attn_scale",
        f"{prefix}.hc_ffn_fn",
        f"{prefix}.hc_ffn_base",
        f"{prefix}.hc_ffn_scale",
        f"{prefix}.attn.attn_sink",
        f"{prefix}.attn.wq_a.weight",
        f"{prefix}.attn.q_norm.weight",
        f"{prefix}.attn.wq_b.weight",
        f"{prefix}.attn.wkv.weight",
        f"{prefix}.attn.kv_norm.weight",
        f"{prefix}.attn.wo_a.weight",
        f"{prefix}.attn.wo_b.weight",
        f"{prefix}.ffn.gate.weight",
    ]
    if layer_idx < config.num_hash_layers:
        names.append(f"{prefix}.ffn.gate.tid2eid")
    else:
        names.append(f"{prefix}.ffn.gate.bias")
    for proj in ("w1", "w3", "w2"):
        names.append(f"{prefix}.ffn.shared_experts.{proj}.weight")
    for expert_id in range(config.n_routed_experts):
        for proj in ("w1", "w3", "w2"):
            names.append(f"{prefix}.ffn.experts.{expert_id}.{proj}.weight")
    if config.compress_ratios and config.compress_ratios[layer_idx] > 1:
        names.extend(
            [
                f"{prefix}.attn.compressor.ape",
                f"{prefix}.attn.compressor.wkv.weight",
                f"{prefix}.attn.compressor.wgate.weight",
                f"{prefix}.attn.compressor.norm.weight",
            ]
        )
    if config.compress_ratios and config.compress_ratios[layer_idx] == 4:
        names.extend(
            [
                f"{prefix}.attn.indexer.wq_b.weight",
                f"{prefix}.attn.indexer.compressor.ape",
                f"{prefix}.attn.indexer.compressor.wkv.weight",
                f"{prefix}.attn.indexer.compressor.wgate.weight",
                f"{prefix}.attn.indexer.compressor.norm.weight",
                f"{prefix}.attn.indexer.weights_proj.weight",
            ]
        )
    return names


def expected_hf_names(
    config: DeepseekV4Config,
    *,
    available_hf_names: Iterable[str] | None = None,
) -> set[str]:
    names = {
        "embed.weight",
        "norm.weight",
        "head.weight",
        "hc_head_fn",
        "hc_head_base",
        "hc_head_scale",
    }
    for layer_idx in range(config.num_hidden_layers):
        names.update(_layer_base_hf_names(f"layers.{layer_idx}", config, layer_idx=layer_idx))
    for mtp_idx in range(config.num_nextn_predict_layers):
        layer_idx = config.num_hidden_layers + mtp_idx
        prefix = f"mtp.{mtp_idx}"
        names.update(_layer_base_hf_names(prefix, config, layer_idx=layer_idx))
        names.update(
            {
                f"{prefix}.e_proj.weight",
                f"{prefix}.h_proj.weight",
                f"{prefix}.enorm.weight",
                f"{prefix}.hnorm.weight",
                f"{prefix}.norm.weight",
                f"{prefix}.hc_head_fn",
                f"{prefix}.hc_head_base",
                f"{prefix}.hc_head_scale",
            }
        )
    if available_hf_names is not None:
        available = set(available_hf_names)
        for name in tuple(names):
            scale_name = _scale_name_for_hf_name(name)
            if scale_name in available:
                names.add(scale_name)
    return names


def load_hf_weights(model: nn.Module, path: str, config: DeepseekV4Config, ps: ParallelState) -> None:
    if (ps.tp_size, ps.ep_size, ps.etp_size, ps.cp_size, ps.pp_size) != (1, 1, 1, 1, 1):
        raise NotImplementedError("DeepSeek V4 direct HF load currently supports only TP=EP=ETP=CP=PP=1.")

    del config
    reader = SafeTensorReader(path)
    base_model = unwrap_model(model)
    state = base_model.state_dict()
    loaded = 0
    missing: list[str] = []
    for name, target in state.items():
        hf_name = _hf_name_for_state_key(name)
        if hf_name is None or not _has(reader, hf_name):
            missing.append(name)
            continue
        scale_name = _scale_name_for_hf_name(hf_name)
        scale = reader.get_tensor(scale_name) if _has(reader, scale_name) else None
        _copy_param(target, reader.get_tensor(hf_name), scale=scale)
        loaded += 1

    log_rank0(f"DeepSeek V4 native loaded {loaded} tensors from {path}")
    for name in missing:
        log_rank0(f"WARNING: DeepSeek V4 checkpoint tensor missing: {name}")


__all__ = [
    "EXPERT_CLASSIFIER",
    "_dequantize_scaled_tensor",
    "_hf_name_for_state_key",
    "_scale_name_for_hf_name",
    "expected_hf_names",
    "load_hf_weights",
]
