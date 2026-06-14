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
    return ".experts." in name


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


def _to_hf_state_name(
    name: str, *, config: Glm5Config, model: nn.Module, ps: ParallelState
) -> str | None:
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
        name = f"model.layers.{global_layer}{suffix}"
    else:
        layer_indices = _local_layer_indices(model)
        match = _LAYER_RE.match(name)
        if match is not None and layer_indices:
            prefix, local_idx_text, suffix = match.groups()
            local_idx = int(local_idx_text)
            if local_idx >= len(layer_indices):
                return None
            name = f"{prefix}{layer_indices[local_idx]}{suffix}"

    match = _LOCAL_EXPERT_RE.match(name)
    if match is None:
        return name

    prefix, local_expert_text, suffix = match.groups()
    experts_per_rank = ensure_divisible(config.n_routed_experts, ps.ep_size)
    global_expert = ps.ep_rank * experts_per_rank + int(local_expert_text)
    return f"{prefix}{global_expert}{suffix}"


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
        hf_name = _to_hf_state_name(name, config=config, model=base_model, ps=ps)
        if hf_name is None:
            continue
        tensor = _resolve_named_parameter_tensor(reader, hf_name, target, config=config, ps=ps)
        if tensor is None:
            missing.append(hf_name)
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

    chunks = model if isinstance(model, list) else [model]
    for chunk in chunks:
        base_model = unwrap_model(chunk)
        state = base_model.state_dict()
        for name, tensor in state.items():
            hf_name = _to_hf_state_name(name, config=config, model=base_model, ps=ps)
            if hf_name is None:
                continue
            exported = tensor.detach().cpu().contiguous()
            if export_dtype is not None and exported.is_floating_point():
                exported = exported.to(dtype=export_dtype)
            yield hf_name, exported


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
