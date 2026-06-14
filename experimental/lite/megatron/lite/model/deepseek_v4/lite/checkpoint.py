# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from collections.abc import Iterable
import math
import re

import torch
import torch.distributed as dist
import torch.nn as nn

from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
from megatron.lite.primitive.ckpt.hf_weights import (
    SafeTensorReader,
    _cast_export_tensor,
    _resolve_export_dtype,
    save_safetensors,
    unwrap_model,
)
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.utils import ensure_divisible, log_rank0


def EXPERT_CLASSIFIER(name: str) -> bool:
    return ".experts." in name and ".shared_experts." not in name


_BLOCK_KEY_RE = re.compile(r"^model\.(layers|mtp)\.(\d+)\.(.+)$")
_GROUPED_EXPERT_RE = re.compile(r"^mlp\.experts\.fc([12])\.weight(\d+)$")
_PROJ_TO_HF = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}

_TOP_LEVEL = {
    "model.embed_tokens.weight": "embed.weight",
    "model.norm.weight": "norm.weight",
    "model.hc_head.hc_fn": "hc_head_fn",
    "model.hc_head.hc_base": "hc_head_base",
    "model.hc_head.hc_scale": "hc_head_scale",
    "lm_head.weight": "head.weight",
}


def _map_block_attr(attr: str, block: str) -> str | tuple[str, ...] | None:
    if attr == "input_layernorm.weight":
        return "attn_norm.weight"
    if attr == "post_attention_layernorm.weight":
        return "ffn_norm.weight"
    if attr.startswith("self_attn."):
        suffix = attr.removeprefix("self_attn.")
        return "attn.attn_sink" if suffix == "sinks" else f"attn.{suffix}"
    if attr.startswith("mlp.gate."):
        suffix = attr.removeprefix("mlp.gate.")
        return "ffn.gate." + {
            "gate.weight": "weight",
            "weight": "weight",
            "expert_bias": "bias",
            "e_score_correction_bias": "bias",
            "tid2eid": "tid2eid",
        }.get(suffix, suffix)
    if attr.startswith("mlp.shared_experts."):
        proj = attr.removeprefix("mlp.shared_experts.").removesuffix(".weight")
        if proj == "gate_up":
            return "ffn.shared_experts.w1.weight", "ffn.shared_experts.w3.weight"
        if proj == "down":
            return "ffn.shared_experts.w2.weight"
        return f"ffn.shared_experts.{_PROJ_TO_HF.get(proj, proj)}.weight"
    for prefix, target in (("attn_hc", "hc_attn"), ("ffn_hc", "hc_ffn"), ("hc_head", "hc_head")):
        if attr.startswith(f"{prefix}."):
            return f"{target}_{attr.rsplit('.', 1)[-1].removeprefix('hc_')}"
    if block == "mtp" and attr in {
        "e_proj.weight",
        "h_proj.weight",
        "enorm.weight",
        "hnorm.weight",
        "norm.weight",
    }:
        return attr
    return None


_FP4_E2M1_TABLE = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def _has(reader: SafeTensorReader, name: str) -> bool:
    if reader.index:
        return name in reader.index
    try:
        reader.get_tensor(name)
    except Exception:
        return False
    return True


def _scale_name_for_hf_name(name: str) -> str:
    return f"{name[:-7] if name.endswith('.weight') else name}.scale"


def _scale_to_float(scale: torch.Tensor) -> torch.Tensor:
    if scale.dtype.is_floating_point:
        return scale.float()
    if scale.dtype == torch.uint8:
        return torch.pow(torch.tensor(2.0, dtype=torch.float32), scale.float() - 127.0)
    return scale.float()


def _expand_block_scale(
    scale: torch.Tensor, target_shape: torch.Size | tuple[int, ...]
) -> torch.Tensor:
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


def _unpack_fp4_e2m1_if_needed(
    tensor: torch.Tensor, target_shape: torch.Size | tuple[int, ...]
) -> torch.Tensor:
    target = tuple(int(dim) for dim in target_shape)
    if (
        tensor.dtype != torch.int8
        or tensor.ndim != len(target)
        or tuple(tensor.shape[:-1]) != target[:-1]
        or tensor.shape[-1] * 2 != target[-1]
    ):
        return tensor.float()

    table = torch.tensor(_FP4_E2M1_TABLE, dtype=torch.float32, device=tensor.device)
    packed = tensor.view(torch.uint8)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    return torch.stack((table[low.long()], table[high.long()]), dim=-1).flatten(-2)


def _dequantize_scaled_tensor(
    tensor: torch.Tensor, scale: torch.Tensor, shape: torch.Size
) -> torch.Tensor:
    scale_f = _expand_block_scale(_scale_to_float(scale), shape)
    return _unpack_fp4_e2m1_if_needed(tensor, shape) * scale_f


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


def _global_expert_idx(local_idx: int, config: DeepseekV4Config, ps: ParallelState) -> int:
    num_local = ensure_divisible(config.n_routed_experts, ps.ep_size)
    return ps.ep_rank * num_local + local_idx


def _hf_names_for_state_key(name: str, config: DeepseekV4Config, ps: ParallelState) -> list[str]:
    mapped = _TOP_LEVEL.get(name)
    if mapped is not None:
        return [mapped]
    match = _BLOCK_KEY_RE.match(name)
    if match is None:
        return []
    block, index, attr = match.groups()
    prefix = f"layers.{index}" if block == "layers" else f"mtp.{index}"
    if attr.startswith("self_attn.compressor."):
        return [f"{prefix}.attn.compressor.{attr.removeprefix('self_attn.compressor.')}"]
    if attr.startswith("self_attn.indexer."):
        return [f"{prefix}.attn.indexer.{attr.removeprefix('self_attn.indexer.')}"]
    mapped = _map_block_attr(attr, block)
    if mapped is not None:
        if isinstance(mapped, tuple):
            return [f"{prefix}.{part}" for part in mapped]
        return [f"{prefix}.{mapped}"]
    expert = _GROUPED_EXPERT_RE.match(attr)
    if expert is None:
        return []
    fc, local_idx = expert.groups()
    expert_id = _global_expert_idx(int(local_idx), config, ps)
    expert_prefix = f"{prefix}.ffn.experts.{expert_id}"
    if fc == "1":
        return [f"{expert_prefix}.w1.weight", f"{expert_prefix}.w3.weight"]
    return [f"{expert_prefix}.w2.weight"]


def _read_hf_tensor(
    reader: SafeTensorReader, hf_name: str, target_shape: torch.Size | tuple[int, ...]
) -> torch.Tensor:
    scale_name = _scale_name_for_hf_name(hf_name)
    tensor = reader.get_tensor(hf_name)
    scale = reader.get_tensor(scale_name) if _has(reader, scale_name) else None
    if scale is not None:
        return _dequantize_scaled_tensor(tensor, scale, torch.Size(target_shape))
    return tensor


def load_hf_weights(
    model: nn.Module, path: str, config: DeepseekV4Config, ps: ParallelState
) -> None:
    if (ps.tp_size, ps.etp_size) != (1, 1):
        raise NotImplementedError("DeepSeek V4 direct HF load currently supports only TP=ETP=1.")

    reader = SafeTensorReader(path)
    base_model = unwrap_model(model)
    state = base_model.state_dict()
    loaded = 0
    missing: list[str] = []
    for name, target in state.items():
        hf_names = _hf_names_for_state_key(name, config, ps)
        if not hf_names or not all(_has(reader, hf_name) for hf_name in hf_names):
            missing.append(name)
            continue
        if len(hf_names) == 2:
            first = target.shape[0] // 2
            tensor = torch.cat(
                [
                    _read_hf_tensor(reader, hf_names[0], (first, *target.shape[1:])),
                    _read_hf_tensor(
                        reader, hf_names[1], (target.shape[0] - first, *target.shape[1:])
                    ),
                ],
                dim=0,
            )
            target.data.copy_(tensor.to(device=target.device, dtype=target.dtype))
        else:
            scale_name = _scale_name_for_hf_name(hf_names[0])
            scale = reader.get_tensor(scale_name) if _has(reader, scale_name) else None
            _copy_param(target, reader.get_tensor(hf_names[0]), scale=scale)
        loaded += 1

    log_rank0(f"DeepSeek V4 native loaded {loaded} tensors from {path}")
    for name in missing:
        log_rank0(f"WARNING: DeepSeek V4 checkpoint tensor missing: {name}")


def _iter_unwrapped_chunks(model: nn.Module | Iterable[nn.Module]) -> Iterable[nn.Module]:
    if isinstance(model, nn.Module):
        yield unwrap_model(model)
        return
    for chunk in model:
        if not isinstance(chunk, nn.Module):
            raise TypeError(
                f"DeepSeek V4 HF export expects nn.Module chunks, got {type(chunk).__name__}."
            )
        yield unwrap_model(chunk)


def _local_hf_state(
    model: nn.Module | Iterable[nn.Module], config: DeepseekV4Config, ps: ParallelState
) -> dict[str, torch.Tensor]:
    exported: dict[str, torch.Tensor] = {}
    for chunk in _iter_unwrapped_chunks(model):
        for native_name, tensor in chunk.state_dict().items():
            hf_names = _hf_names_for_state_key(native_name, config, ps)
            if not hf_names:
                raise KeyError(f"DeepSeek V4 native state key has no HF mapping: {native_name}")
            pieces = (
                tensor.detach().cpu().contiguous().chunk(2, dim=0)
                if len(hf_names) == 2
                else (tensor.detach().cpu().contiguous(),)
            )
            for hf_name, hf_tensor in zip(hf_names, pieces, strict=True):
                exported[hf_name] = hf_tensor
    return exported


def _gather_pp_hf_state(
    local_state: dict[str, torch.Tensor], ps: ParallelState
) -> dict[str, torch.Tensor]:
    if ps.pp_size <= 1:
        return local_state
    if not dist.is_initialized() or ps.pp_group is None:
        raise RuntimeError("DeepSeek V4 HF export with PP>1 requires an initialized PP group.")
    gathered: list[dict[str, torch.Tensor] | None] = [None] * ps.pp_size
    dist.all_gather_object(gathered, local_state, group=ps.pp_group)
    merged: dict[str, torch.Tensor] = {}
    for shard in gathered:
        if shard:
            merged.update(shard)
    return merged


def export_hf_weights(
    model: nn.Module | Iterable[nn.Module],
    config: DeepseekV4Config,
    ps: ParallelState,
    *,
    rank0_only: bool = False,
    export_dtype: str | torch.dtype | None = None,
    limit: int | None = None,
    **_kwargs,
):
    rank = dist.get_rank() if dist.is_initialized() else 0
    export_dtype_resolved = _resolve_export_dtype(export_dtype)
    exported = _gather_pp_hf_state(_local_hf_state(model, config, ps), ps)
    if rank0_only and rank != 0:
        return
    for index, hf_name in enumerate(sorted(exported)):
        if limit is not None and index >= limit:
            return
        yield hf_name, _cast_export_tensor(exported[hf_name], export_dtype_resolved)


def save_hf_weights(
    model: nn.Module | Iterable[nn.Module],
    path: str,
    config: DeepseekV4Config,
    ps: ParallelState,
    *,
    export_dtype: str | torch.dtype | None = None,
) -> None:
    rank = dist.get_rank() if dist.is_initialized() else 0
    tensors = dict(export_hf_weights(model, config, ps, rank0_only=True, export_dtype=export_dtype))
    if rank == 0:
        save_safetensors(tensors, path)
    if dist.is_initialized():
        dist.barrier()
