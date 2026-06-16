# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DeepSeek V4 (ds4flash) lite native <-> HF checkpoint mapping.

Like kimi_k2 / glm5: ``DeepseekV4WeightSpec`` encodes the per-param native -> HF
name (+ TP/EP shard spec); export/save route through the shared
``primitive/ckpt/hf_weights.py`` exporter (its PP ``all_gather_object`` is
reached by all ranks before any ``rank0_only`` filter, so PP>1 export doesn't
desync).  Native names are bare ``DeepseekV4Model`` keys; ``self.layers`` is a
ModuleDict keyed by GLOBAL layer index (kimi uses a local ModuleList), so the
exporter's local->global remap is an identity here.

HF targets are canonical HF DeepSeek (``model.embed_tokens.weight`` /
``model.norm.weight`` / ``lm_head.weight`` / ``model.layers.<i>.self_attn.*`` /
``...mlp.experts.<id>.{gate,up,down}_proj.weight``).  DS4 extras:
  * CSA: ``self_attn.*`` incl. ``compressor.*`` / ``indexer.*``; ``sinks`` ->
    ``self_attn.attn_sink``.
  * mHC: ``attn_hc`` / ``ffn_hc`` -> ``...self_attn.hc_*`` / ``...mlp.hc_*``;
    model-wide ``hc_head`` -> ``model.hc_head.*`` (no HF analogue, kept
    model.-rooted; fidelity vs Megatron's latest mHC is a TODO).
  * MTP: folded into the decoder namespace at ``model.layers.<num_hidden+i>``.

CSA is not TP-capable: DS4 runs TP=ETP=1 (only EP shards experts), like GLM-5.
"""

from __future__ import annotations

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
    parse_expert_idx,
    to_global_layer_name,
    unwrap_model,
)
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.utils import ensure_divisible, log_rank0


def EXPERT_CLASSIFIER(name: str) -> bool:
    return ".experts." in name and ".shared_experts." not in name


def PLACEMENT_FN(param_name: str) -> list:
    # distckpt sharded placement (TP=ETP=1 for ds4; shares kimi/glm5's
    # Experts/SwiGLUMLP/VocabParallel structure). EP-sharded experts must carry
    # an explicit placement or the dist-opt checkpoint won't restore them
    # bit-exactly. The CSA/mHC/MTP-norm params fall through to all-Replicate.
    from torch.distributed.tensor import Replicate, Shard

    if ".experts." in param_name and ".shared_experts." not in param_name:
        if "fc1" in param_name:
            return [Replicate(), Replicate(), Shard(0), Shard(0)]
        if "fc2" in param_name:
            return [Replicate(), Replicate(), Shard(0), Shard(1)]
        return [Replicate(), Replicate(), Replicate(), Replicate()]
    if "eh_proj.linear.weight" in param_name:
        return [Replicate(), Replicate(), Replicate(), Shard(0)]
    if "gate_up" in param_name:
        return [Replicate(), Replicate(), Replicate(), Shard(0)]
    if "down" in param_name:
        return [Replicate(), Replicate(), Replicate(), Shard(1)]
    if "embed" in param_name or "head" in param_name:
        return [Replicate(), Replicate(), Replicate(), Shard(0)]
    return [Replicate(), Replicate(), Replicate(), Replicate()]


# Native <-> HF name mapping (shared by export spec and load path).  Native
# names are bare DeepseekV4Model state_dict keys with GLOBAL layer indices.
_BLOCK_KEY_RE = re.compile(r"^(layers|mtp)\.(\d+)\.(.+)$")
_GROUPED_EXPERT_RE = re.compile(r"^mlp\.experts\.fc([12])\.weight(\d+)$")
_PROJ_TO_HF = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}

# Native top-level params -> HF target names.  mHC ``hc_head`` has no HF
# analogue, so it keeps a ``model.``-rooted name to stay self-consistent.
_TOP_LEVEL = {
    "embed_tokens.embedding.weight": "model.embed_tokens.weight",
    "norm.weight": "model.norm.weight",
    "hc_head.hc_fn": "model.hc_head.hc_fn",
    "hc_head.hc_base": "model.hc_head.hc_base",
    "hc_head.hc_scale": "model.hc_head.hc_scale",
    "lm_head.col.linear.weight": "lm_head.weight",
}


def _map_block_attr(attr: str, block: str) -> str | tuple[str, ...] | None:
    if attr == "input_layernorm.weight":
        return "input_layernorm.weight"
    if attr == "post_attention_layernorm.weight":
        return "post_attention_layernorm.weight"
    if attr.startswith("self_attn."):
        suffix = attr.removeprefix("self_attn.")
        return "self_attn.attn_sink" if suffix == "sinks" else f"self_attn.{suffix}"
    if attr.startswith("mlp.gate."):
        suffix = attr.removeprefix("mlp.gate.")
        return "mlp.gate." + {
            "gate.weight": "weight",
            "weight": "weight",
            "expert_bias": "e_score_correction_bias",
            "e_score_correction_bias": "e_score_correction_bias",
            "tid2eid": "tid2eid",
        }.get(suffix, suffix)
    if attr.startswith("mlp.shared_experts."):
        proj = attr.removeprefix("mlp.shared_experts.").removesuffix(".weight")
        if proj == "gate_up":
            return "mlp.shared_experts.gate_proj.weight", "mlp.shared_experts.up_proj.weight"
        if proj == "down":
            return "mlp.shared_experts.down_proj.weight"
        return f"mlp.shared_experts.{_PROJ_TO_HF.get(proj, proj)}.weight"
    # mHC params have no upstream HF analogue; keep them under self_attn / mlp /
    # hc_head sub-namespaces so every key stays ``model.layers.{i}.*``-rooted.
    for prefix, target in (("attn_hc", "self_attn.hc"), ("ffn_hc", "mlp.hc"), ("hc_head", "hc_head")):
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


def _global_expert_idx_from_local(local_idx: int, config: DeepseekV4Config, ps: ParallelState) -> int:
    num_local = ensure_divisible(config.n_routed_experts, ps.ep_size)
    return ps.ep_rank * num_local + local_idx


def _hf_names_for_state_key(name: str, config: DeepseekV4Config) -> list[str]:
    """Map a bare DS4 native key (global layer idx, global expert id) to HF name(s).

    Callers (load path + shared exporter) supply global expert ids first.
    """
    mapped = _TOP_LEVEL.get(name)
    if mapped is not None:
        return [mapped]
    match = _BLOCK_KEY_RE.match(name)
    if match is None:
        return []
    block, index, attr = match.groups()
    # HF places MTP layers right after the decoder stack, so they share the
    # ``model.layers.{i}`` namespace with a continued global index.
    if block == "layers":
        prefix = f"model.layers.{index}"
    else:  # mtp
        prefix = f"model.layers.{config.num_hidden_layers + int(index)}"
    # CSA lives under ``self_attn.self_attn.*`` (the SBHD wrapper adds one extra
    # ``self_attn`` level).  Collapse it so the HF attention names are unchanged.
    if attr.startswith("self_attn.self_attn."):
        attr = "self_attn." + attr.removeprefix("self_attn.self_attn.")
    if attr.startswith("self_attn.compressor."):
        return [f"{prefix}.self_attn.compressor.{attr.removeprefix('self_attn.compressor.')}"]
    if attr.startswith("self_attn.indexer."):
        return [f"{prefix}.self_attn.indexer.{attr.removeprefix('self_attn.indexer.')}"]
    mapped = _map_block_attr(attr, block)
    if mapped is not None:
        if isinstance(mapped, tuple):
            return [f"{prefix}.{part}" for part in mapped]
        return [f"{prefix}.{mapped}"]
    expert = _GROUPED_EXPERT_RE.match(attr)
    if expert is None:
        return []
    fc, expert_id = expert.groups()
    expert_prefix = f"{prefix}.mlp.experts.{int(expert_id)}"
    if fc == "1":
        return [f"{expert_prefix}.gate_proj.weight", f"{expert_prefix}.up_proj.weight"]
    return [f"{expert_prefix}.down_proj.weight"]


# ======================================================================
# FP4 / scaled-tensor dequant helpers (load path).
# ======================================================================

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


def _is_native_metadata_key(name: str) -> bool:
    return name.endswith("._extra_state")


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
    """Load HF safetensors into the DS4 model.

    Kept as DS4's native loader (the inverse of the spec's ``native_to_hf``):
    it walks the native ``state_dict`` and resolves each key's HF name(s) via
    ``_hf_names_for_state_key`` -- the SAME mapping the export spec uses, so the
    round-trip names stay consistent.  EP-local expert ids are converted to
    global before mapping.  CSA is TP=ETP=1, so there is no TP split here.
    """
    if (ps.tp_size, ps.etp_size) != (1, 1):
        raise NotImplementedError("DeepSeek V4 direct HF load currently supports only TP=ETP=1.")

    reader = SafeTensorReader(path)
    base_model = unwrap_model(model)
    state = base_model.state_dict()
    loaded = 0
    missing: list[str] = []
    for name, target in state.items():
        if _is_native_metadata_key(name):
            continue
        hf_names = _hf_names_for_state_key(_to_global_expert_name(name, config, ps), config)
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


def _to_global_expert_name(name: str, config: DeepseekV4Config, ps: ParallelState) -> str:
    """Rewrite an EP-local expert ``weight<local>`` suffix to its global id.

    The native ``state_dict`` carries the EP-local expert index; the HF target
    name uses the global expert id.  Non-expert names pass through unchanged.
    """
    match = _BLOCK_KEY_RE.match(name)
    if match is None:
        return name
    block, index, attr = match.groups()
    expert = _GROUPED_EXPERT_RE.match(attr)
    if expert is None:
        return name
    fc, local_idx = expert.groups()
    global_idx = _global_expert_idx_from_local(int(local_idx), config, ps)
    return f"{block}.{index}.mlp.experts.fc{fc}.weight{global_idx}"


# ======================================================================
# Export: shared TP/ETP/EP/PP gather via DeepseekV4WeightSpec.
# ======================================================================


class DeepseekV4WeightSpec:
    """Export DS4 lite weights to HF DeepSeek-V4 names (CSA / mHC / MTP / MoE).

    Mirrors ``KimiK2WeightSpec`` / ``Glm5WeightSpec`` on DS4's bare native names
    with global layer indices.  The shared exporter rewrites EP-local expert
    ``weight<local>`` ids to global before calling ``native_to_hf``.
    """

    def __init__(self, config: DeepseekV4Config):
        self.config = config

    @property
    def num_experts(self) -> int:
        return self.config.n_routed_experts

    def weight_map(self) -> dict[str, list[str]]:
        return {}

    def hf_to_native(self, native_name: str, hf_tensors: list[torch.Tensor]) -> torch.Tensor:
        del native_name
        return hf_tensors[0]

    def native_to_hf(
        self, native_name: str, tensor: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]:
        # ``native_name`` is the global native name; experts already carry the
        # global expert id (shared exporter rewrote weight<local> -> weight<gid>).
        hf_names = _hf_names_for_state_key(native_name, self.config)
        if not hf_names:
            return []
        if len(hf_names) == 1:
            return [(hf_names[0], tensor)]
        if len(hf_names) == 2:
            # 2 targets == fused gate/up split into (w1, w3) for shared/routed
            # experts; split the leading dim exactly as the bespoke export did.
            first, second = tensor.chunk(2, dim=0)
            return [
                (hf_names[0], first.contiguous()),
                (hf_names[1], second.contiguous()),
            ]
        raise AssertionError(f"Unexpected HF name fan-out for {native_name}: {hf_names}")

    def qkv_spec(self, native_name: str) -> tuple[int, int, int] | None:
        del native_name
        return None

    def tp_spec(self, native_name: str) -> tuple[int, int] | None:
        # DS4 is TP=ETP=1 (CSA is not TP-capable); only EP shards experts.  The
        # expert (split_dim, ETP) entries are declared so the shared ETP path
        # would be correct if ETP were ever enabled; embed/head/eh_proj carry
        # the vocab split-dim spec for completeness (no-op at TP=1).
        if self.is_expert(native_name):
            if ".fc1." in native_name:
                return (0, 1)
            if ".fc2." in native_name:
                return (1, 1)
            return None
        if native_name.endswith(".eh_proj.linear.weight"):
            return (0, 0)
        if native_name in {
            "embed_tokens.embedding.weight",
            "lm_head.col.linear.weight",
        }:
            return (0, 0)
        return None

    def is_expert(self, native_name: str) -> bool:
        return ".mlp.experts." in native_name and ".shared_experts." not in native_name

    def expert_global_id(self, native_name: str) -> int | None:
        if self.is_expert(native_name):
            return parse_expert_idx(native_name)
        return None

    def expert_local_name(self, native_name: str, local_idx: int) -> str:
        prefix = native_name.rsplit(".weight", 1)[0]
        return f"{prefix}.weight{local_idx}"


def export_hf_weights(model, config: DeepseekV4Config, ps: ParallelState, **kwargs):
    """Export DS4 weights as HF (name, tensor) pairs via the SHARED exporter.

    Identical structure to kimi/glm5: delegate to the shared ``_export`` (which
    does the TP/ETP/EP/PP gather, including the PP ``all_gather_object`` reached
    by ALL ranks before any ``rank0_only`` filter), then append the persistent
    router buffers (``tid2eid`` for hash layers, ``expert_bias`` for non-hash
    layers) which the parameter-only ``_export`` does not visit.
    """
    from megatron.lite.primitive.ckpt.hf_weights import export_hf_weights as _export

    spec = DeepseekV4WeightSpec(config)
    rank0_only = bool(kwargs.get("rank0_only", False))
    export_dtype = _resolve_export_dtype(kwargs.get("export_dtype"))
    yield from _export(model, spec, ps, vocab_size=config.vocab_size, **kwargs)

    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank0_only and rank != 0:
        return
    chunks = list(model) if isinstance(model, list | nn.ModuleList) else [model]
    for chunk in chunks:
        base_chunk = unwrap_model(chunk)
        layer_map = (
            {i: base_chunk.layer_indices[i] for i in range(len(base_chunk.layer_indices))}
            if hasattr(base_chunk, "layer_indices")
            else {}
        )
        for name, buffer in base_chunk.named_buffers():
            # Persistent router buffers carried into HF: hash-layer ``tid2eid``
            # and the (made-persistent for non-hash layers) ``expert_bias``.
            if not (name.endswith(".mlp.gate.tid2eid") or name.endswith(".mlp.gate.expert_bias")):
                continue
            global_name = to_global_layer_name(name, layer_map)
            for hf_name, hf_tensor in spec.native_to_hf(global_name, buffer.detach().cpu()):
                yield hf_name, _cast_export_tensor(hf_tensor, export_dtype)


def save_hf_weights(model, path: str, config: DeepseekV4Config, ps: ParallelState, **kwargs) -> None:
    from megatron.lite.primitive.ckpt.hf_weights import save_safetensors

    rank = dist.get_rank() if dist.is_initialized() else 0
    out = dict(export_hf_weights(model, config, ps, rank0_only=True, **kwargs))
    if rank == 0 and out:
        save_safetensors(out, path)
    if dist.is_initialized():
        dist.barrier()


__all__ = [
    "EXPERT_CLASSIFIER",
    "DeepseekV4WeightSpec",
    "PLACEMENT_FN",
    "export_hf_weights",
    "load_hf_weights",
    "save_hf_weights",
]
