"""Direct HF safetensor loading for native GLM-5."""

from __future__ import annotations

import re
from collections.abc import Iterator

import torch
import torch.nn as nn

from megatron.lite.model.glm5.config import Glm5Config
from megatron.lite.primitive.ckpt.hf_weights import SafeTensorReader, save_safetensors, unwrap_model
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.utils import ensure_divisible
from megatron.lite.primitive.utils import log_rank0


def EXPERT_CLASSIFIER(name: str) -> bool:
    # Routed experts only: exclude the shared expert and the router gate.
    return ".experts." in name and ".shared_experts." not in name and ".gate." not in name


def _has(reader: SafeTensorReader, name: str) -> bool:
    if reader.index:
        return name in reader.index
    try:
        reader.get_tensor(name)
    except Exception:
        return False
    return True


def _slice_to_target_shape(
    tensor: torch.Tensor, target: nn.Parameter | torch.Tensor
) -> torch.Tensor:
    if tensor.shape == target.shape:
        return tensor
    if tensor.ndim != target.ndim:
        raise RuntimeError(
            f"Cannot load tensor with ndim {tensor.ndim} into target ndim {target.ndim}: "
            f"{tuple(tensor.shape)} -> {tuple(target.shape)}"
        )
    if any(dst > src for src, dst in zip(tensor.shape, target.shape, strict=True)):
        raise RuntimeError(
            f"Cannot shrink source tensor {tuple(tensor.shape)} to target {tuple(target.shape)}"
        )
    slices = tuple(slice(0, dim) for dim in target.shape)
    return tensor[slices].contiguous()


def _copy_param(param: nn.Parameter | torch.Tensor, tensor: torch.Tensor) -> None:
    local_tensor = getattr(param, "_local_tensor", None)
    fsdp_slice = getattr(param, "megatron_fsdp_slice", None)
    if local_tensor is not None and fsdp_slice is not None:
        reference = getattr(param, "orig_param", param)
        fitted = _slice_to_target_shape(tensor, reference)
        local_fitted = fitted.flatten()[fsdp_slice]
        if local_fitted.numel() != local_tensor.numel():
            raise RuntimeError(
                "Cannot load MegatronFSDP shard: "
                f"source slice has {local_fitted.numel()} values but target local tensor "
                f"has {local_tensor.numel()} values"
            )
        local_fitted = local_fitted.view(local_tensor.shape).contiguous()
        local_tensor.copy_(local_fitted.to(device=local_tensor.device, dtype=local_tensor.dtype))
        return

    if local_tensor is not None and hasattr(param, "_spec"):
        from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

        fitted = _slice_to_target_shape(tensor, param)
        spec = param._spec
        local_shape, global_offset = compute_local_shape_and_global_offset(
            tuple(fitted.shape), spec.mesh, spec.placements
        )
        shard_slices = tuple(
            slice(offset, offset + size)
            for offset, size in zip(global_offset, local_shape, strict=True)
        )
        local_fitted = fitted[shard_slices].contiguous()
        if local_fitted.shape != local_tensor.shape:
            local_fitted = _slice_to_target_shape(local_fitted, local_tensor)
        local_tensor.copy_(local_fitted.to(device=local_tensor.device, dtype=local_tensor.dtype))
        return

    fitted = _slice_to_target_shape(tensor, param)
    param.data.copy_(fitted.to(device=param.device, dtype=param.dtype))


_SPLIT_EXPERT_RE = re.compile(
    r"^(model\.layers\.\d+\.mlp\.experts)\.(\d+)\." r"(gate_proj|up_proj|down_proj)\.weight$"
)
_LAYER_RE = re.compile(r"^(model\.layers\.)(\d+)(\..*)$")
_MTP_LAYER_RE = re.compile(r"^(model\.mtp\.layers\.)(\d+)(\..*)$")
_LOCAL_EXPERT_RE = re.compile(r"^(model\.layers\.\d+\.mlp\.experts\.)(\d+)(\..*)$")
_PACKED_EXPERT_RE = re.compile(r"^(model\.layers\.\d+\.mlp\.experts)\.(gate_up_proj|down_proj)$")

# Native (Megatron-lite) param names now come from the SHARED primitives
# (SwiGLUMLP: gate_up/down; SigmoidTopKRouter: gate.gate.weight + expert_bias;
# Experts: fc1.weight{N}/fc2.weight{N}).  These regexes translate native names
# into the HF DeepSeek/GLM names so save_hf / load_hf keep identical HF keys.
_GROUPED_EXPERT_RE = re.compile(
    r"^(model\.(?:layers|mtp\.layers)\.\d+(?:\.transformer_layer)?\.mlp\.experts)\.fc([12])\.weight(\d+)$"
)
_FUSED_GATE_UP_RE = re.compile(
    r"^(model\.(?:layers|mtp\.layers)\.\d+(?:\.transformer_layer)?\.mlp"
    r"(?:\.shared_experts)?)\.gate_up\.weight$"
)
# SwiGLUMLP down projection (dense MLP + shared experts) -> HF `down_proj`.
_DOWN_RE = re.compile(
    r"^(model\.(?:layers|mtp\.layers)\.\d+(?:\.transformer_layer)?\.mlp"
    r"(?:\.shared_experts)?)\.down\.weight$"
)


def _native_suffix_to_hf(name: str) -> str:
    """Translate a native param suffix to the HF-equivalent suffix.

    Handles the renamed router and the SwiGLUMLP down projection that map 1:1
    to HF.  Fused gate_up and grouped experts (which split/merge tensors) are
    handled separately by the resolver / exporter.
    """
    # Router: shared SigmoidTopKRouter exposes `gate.gate.weight` + `expert_bias`.
    name = name.replace(".mlp.gate.gate.weight", ".mlp.gate.weight")
    name = name.replace(".mlp.gate.expert_bias", ".mlp.gate.e_score_correction_bias")
    # SwiGLUMLP `down` -> HF `down_proj` (dense MLP and shared experts).
    down = _DOWN_RE.match(name)
    if down is not None:
        return f"{down.group(1)}.down_proj.weight"
    return name


def _resolve_hf_tensor(
    reader: SafeTensorReader, name: str, target: nn.Parameter | torch.Tensor
) -> torch.Tensor | None:
    if _has(reader, name):
        return reader.get_tensor(name)
    if name.endswith(".final_layernorm.weight"):
        shared_head_norm = name[: -len(".final_layernorm.weight")] + ".shared_head.norm.weight"
        if _has(reader, shared_head_norm):
            return reader.get_tensor(shared_head_norm)

    match = _SPLIT_EXPERT_RE.match(name)
    if match is None:
        return None

    prefix, expert_idx_text, projection = match.groups()
    expert_idx = int(expert_idx_text)
    if projection in {"gate_proj", "up_proj"}:
        gate_up_name = f"{prefix}.gate_up_proj"
        if _has(reader, gate_up_name):
            gate_up = reader.get_tensor(gate_up_name)
            split = target.shape[0]
            offset = 0 if projection == "gate_proj" else split
            return gate_up[expert_idx, offset : offset + split, :]

        gate_and_up_name = f"{prefix}.gate_and_up_projs"
        if _has(reader, gate_and_up_name):
            gate_and_up = reader.get_tensor(gate_and_up_name)
            split = target.shape[0]
            offset = 0 if projection == "gate_proj" else split
            return gate_and_up[expert_idx, :, offset : offset + split].T.contiguous()

    if projection == "down_proj":
        down_name = f"{prefix}.down_proj"
        if _has(reader, down_name):
            return reader.get_tensor(down_name)[expert_idx]

        down_projs_name = f"{prefix}.down_projs"
        if _has(reader, down_projs_name):
            return reader.get_tensor(down_projs_name)[expert_idx].T.contiguous()

    return None


def _local_layer_indices(model: nn.Module) -> list[int]:
    if hasattr(model, "layer_indices"):
        return list(model.layer_indices)
    nested = getattr(model, "model", None)
    if nested is not None and hasattr(nested, "layer_indices"):
        return list(nested.layer_indices)
    return []


def _globalize_layer_name(
    name: str, *, config: Glm5Config, model: nn.Module
) -> str | None:
    """Rewrite local layer / MTP indices to the global HF layer index.

    The returned name still carries the *native* suffix (router/expert/fused
    names); suffix translation is applied by the caller.
    """
    if name == "model.mtp_embed.weight":
        return "model.embed_tokens.weight"

    mtp_match = _MTP_LAYER_RE.match(name)
    if mtp_match is not None:
        _prefix, mtp_idx_text, suffix = mtp_match.groups()
        mtp_idx = int(mtp_idx_text)
        if mtp_idx >= config.num_nextn_predict_layers:
            return None
        global_layer = config.num_hidden_layers + mtp_idx
        if suffix.startswith(".transformer_layer."):
            suffix = suffix[len(".transformer_layer") :]
        return f"model.layers.{global_layer}{suffix}"

    layer_indices = _local_layer_indices(model)
    match = _LAYER_RE.match(name)
    if match is not None and layer_indices:
        prefix, local_idx_text, suffix = match.groups()
        local_idx = int(local_idx_text)
        if local_idx >= len(layer_indices):
            return None
        return f"{prefix}{layer_indices[local_idx]}{suffix}"
    return name


def _global_expert_id(local_expert: int, *, config: Glm5Config, ps: ParallelState) -> int:
    experts_per_rank = ensure_divisible(config.n_routed_experts, ps.ep_size)
    return ps.ep_rank * experts_per_rank + local_expert


def _to_hf_state_name(
    name: str, *, config: Glm5Config, model: nn.Module, ps: ParallelState
) -> str | None:
    """Map a native (post-globalized) name to its single HF name (1:1 cases).

    Returns None for names that fan out to multiple HF tensors (fused gate_up,
    grouped experts) — those are handled by the dedicated load/export paths.
    """
    name = _globalize_layer_name(name, config=config, model=model)
    if name is None:
        return None

    if _FUSED_GATE_UP_RE.match(name) or _GROUPED_EXPERT_RE.match(name):
        return None

    name = _native_suffix_to_hf(name)

    # Pre-existing per-expert (non-grouped) layout: remap local→global expert id.
    match = _LOCAL_EXPERT_RE.match(name)
    if match is not None:
        prefix, local_expert_text, suffix = match.groups()
        global_expert = _global_expert_id(int(local_expert_text), config=config, ps=ps)
        return f"{prefix}{global_expert}{suffix}"
    return name


def _resolve_named_parameter_tensor(
    reader: SafeTensorReader,
    name: str,
    target: nn.Parameter | torch.Tensor,
    *,
    config: Glm5Config,
    ps: ParallelState,
) -> torch.Tensor | None:
    match = _PACKED_EXPERT_RE.match(name)
    if match is None:
        return _resolve_hf_tensor(reader, name, target)

    prefix, projection = match.groups()
    experts_per_rank = ensure_divisible(config.n_routed_experts, ps.ep_size)
    expert_start = ps.ep_rank * experts_per_rank
    local_experts = target.shape[0]
    tensors: list[torch.Tensor] = []
    for local_expert in range(local_experts):
        global_expert = expert_start + local_expert
        expert_prefix = f"{prefix}.{global_expert}"
        if projection == "gate_up_proj":
            split = target.shape[1] // 2
            gate = _resolve_hf_tensor(
                reader, f"{expert_prefix}.gate_proj.weight", target[local_expert, :split]
            )
            up = _resolve_hf_tensor(
                reader, f"{expert_prefix}.up_proj.weight", target[local_expert, split:]
            )
            if gate is None or up is None:
                return None
            tensors.append(torch.cat([gate, up], dim=0).contiguous())
        else:
            down = _resolve_hf_tensor(
                reader, f"{expert_prefix}.down_proj.weight", target[local_expert]
            )
            if down is None:
                return None
            tensors.append(down)
    return torch.stack(tensors, dim=0).contiguous()


def _resolve_fused_or_grouped(
    reader: SafeTensorReader,
    global_name: str,
    target: nn.Parameter | torch.Tensor,
    *,
    config: Glm5Config,
    ps: ParallelState,
) -> torch.Tensor | None:
    """Resolve shared-primitive params that map to multiple HF tensors.

    - SwiGLUMLP `mlp[.shared_experts].gate_up.weight` <- HF gate_proj + up_proj
    - shared Experts `mlp.experts.fc1.weight{N}` <- HF expert N gate_proj+up_proj
    - shared Experts `mlp.experts.fc2.weight{N}` <- HF expert N down_proj
    """
    fused = _FUSED_GATE_UP_RE.match(global_name)
    if fused is not None:
        mlp_prefix = fused.group(1)
        split = target.shape[0] // 2
        gate = _resolve_hf_tensor(
            reader, f"{mlp_prefix}.gate_proj.weight", target[:split]
        )
        up = _resolve_hf_tensor(reader, f"{mlp_prefix}.up_proj.weight", target[split:])
        if gate is None or up is None:
            return None
        return torch.cat([gate, up], dim=0).contiguous()

    grouped = _GROUPED_EXPERT_RE.match(global_name)
    if grouped is not None:
        experts_prefix, fc, local_expert_text = grouped.groups()
        global_expert = _global_expert_id(int(local_expert_text), config=config, ps=ps)
        ep = f"{experts_prefix}.{global_expert}"
        if fc == "1":
            split = target.shape[0] // 2
            gate = _resolve_hf_tensor(reader, f"{ep}.gate_proj.weight", target[:split])
            up = _resolve_hf_tensor(reader, f"{ep}.up_proj.weight", target[split:])
            if gate is None or up is None:
                return None
            return torch.cat([gate, up], dim=0).contiguous()
        return _resolve_hf_tensor(reader, f"{ep}.down_proj.weight", target)

    return None


def load_hf_weights(model: nn.Module, path: str, config: Glm5Config, ps: ParallelState) -> None:
    if ps.tp_size != 1 or ps.etp_size != 1:
        raise NotImplementedError("GLM5 direct HF load currently supports TP=ETP=1.")

    reader = SafeTensorReader(path)
    base_model = unwrap_model(model)
    loaded = 0
    missing: list[str] = []
    state_names = set(base_model.state_dict())
    targets = list(base_model.named_parameters())
    targets.extend(
        (name, target) for name, target in base_model.named_buffers() if name in state_names
    )
    for name, target in targets:
        global_name = _globalize_layer_name(name, config=config, model=base_model)
        if global_name is None:
            continue
        if _FUSED_GATE_UP_RE.match(global_name) or _GROUPED_EXPERT_RE.match(global_name):
            tensor = _resolve_fused_or_grouped(
                reader, global_name, target, config=config, ps=ps
            )
            ref_name = global_name
        else:
            hf_name = _to_hf_state_name(name, config=config, model=base_model, ps=ps)
            if hf_name is None:
                continue
            tensor = _resolve_named_parameter_tensor(reader, hf_name, target, config=config, ps=ps)
            ref_name = hf_name
        if tensor is None:
            missing.append(ref_name)
            continue
        _copy_param(target, tensor)
        loaded += 1

    log_rank0(f"GLM5 native loaded {loaded} tensors from {path}")
    for name in missing:
        log_rank0(f"WARNING: GLM5 checkpoint tensor missing: {name}")


def _rank0() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _validate_export_scope(ps: ParallelState) -> None:
    if ps.tp_size != 1 or ps.etp_size != 1:
        raise NotImplementedError("GLM5 direct HF export currently supports TP=ETP=1.")
    if ps.ep_size != 1:
        raise NotImplementedError("GLM5 direct HF export currently supports EP=1.")


def _native_to_hf_pairs(
    name: str,
    tensor: torch.Tensor,
    *,
    config: Glm5Config,
    model: nn.Module,
    ps: ParallelState,
) -> list[tuple[str, torch.Tensor]]:
    """Convert one native state-dict entry to its [(hf_name, hf_tensor)] list.

    Splits fused gate_up (SwiGLUMLP / shared Experts fc1) back into HF
    gate_proj/up_proj and renames the shared router / down projections.
    """
    global_name = _globalize_layer_name(name, config=config, model=model)
    if global_name is None:
        return []

    fused = _FUSED_GATE_UP_RE.match(global_name)
    if fused is not None:
        mlp_prefix = fused.group(1)
        gate, up = tensor.chunk(2, dim=0)
        return [
            (f"{mlp_prefix}.gate_proj.weight", gate.contiguous()),
            (f"{mlp_prefix}.up_proj.weight", up.contiguous()),
        ]

    grouped = _GROUPED_EXPERT_RE.match(global_name)
    if grouped is not None:
        experts_prefix, fc, local_expert_text = grouped.groups()
        global_expert = _global_expert_id(int(local_expert_text), config=config, ps=ps)
        ep = f"{experts_prefix}.{global_expert}"
        if fc == "1":
            gate, up = tensor.chunk(2, dim=0)
            return [
                (f"{ep}.gate_proj.weight", gate.contiguous()),
                (f"{ep}.up_proj.weight", up.contiguous()),
            ]
        return [(f"{ep}.down_proj.weight", tensor)]

    hf_name = _to_hf_state_name(name, config=config, model=model, ps=ps)
    if hf_name is None:
        return []
    return [(hf_name, tensor)]


def export_hf_weights(
    model: nn.Module | list[nn.Module],
    config: Glm5Config,
    ps: ParallelState,
    *,
    rank0_only: bool = False,
    export_dtype: torch.dtype | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    _validate_export_scope(ps)
    if rank0_only and _rank0() != 0:
        return

    def _cast(t: torch.Tensor) -> torch.Tensor:
        t = t.detach().cpu().contiguous()
        if export_dtype is not None and t.is_floating_point():
            t = t.to(dtype=export_dtype)
        return t

    chunks = model if isinstance(model, list) else [model]
    for chunk in chunks:
        base_model = unwrap_model(chunk)
        state = base_model.state_dict()
        for name, tensor in state.items():
            for hf_name, hf_tensor in _native_to_hf_pairs(
                name, tensor, config=config, model=base_model, ps=ps
            ):
                yield hf_name, _cast(hf_tensor)


def save_hf_weights(
    model: nn.Module | list[nn.Module],
    path: str,
    config: Glm5Config,
    ps: ParallelState,
    *,
    export_dtype: torch.dtype | None = None,
) -> None:
    tensors = dict(export_hf_weights(model, config, ps, rank0_only=True, export_dtype=export_dtype))
    if _rank0() == 0 and tensors:
        save_safetensors(tensors, path)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def save_weights(
    model: nn.Module | list[nn.Module], path: str, config: Glm5Config, ps: ParallelState
) -> None:
    rank = _rank0()
    if rank != 0:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        return
    tensors: dict[str, torch.Tensor] = {}
    chunks = model if isinstance(model, list) else [model]
    for idx, chunk in enumerate(chunks):
        base_model = unwrap_model(chunk)
        prefix = "" if len(chunks) == 1 else f"chunk{idx}."
        for name, tensor in base_model.state_dict().items():
            tensors[prefix + name] = tensor.detach().cpu().contiguous()
    save_safetensors(tensors, path)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


__all__ = [
    "EXPERT_CLASSIFIER",
    "export_hf_weights",
    "load_hf_weights",
    "save_hf_weights",
    "save_weights",
]
