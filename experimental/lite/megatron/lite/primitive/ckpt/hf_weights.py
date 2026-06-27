# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""HF ↔ Megatron Lite weight conversion.

Everything needed for HuggingFace safetensors ↔ Megatron Lite model conversion:
- HFWeights protocol (model implements this)
- SafeTensorReader / save_safetensors (file I/O)
- Tensor utilities (split_dim, allgather_concat, remap_layer_index, ...)
- Generic load_hf_weights / export_hf_weights / save_hf_weights (orchestration)
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch
import torch.distributed as dist
import torch.nn as nn

try:
    from torch.distributed.tensor import DTensor
except Exception:  # pragma: no cover - older torch without DTensor
    DTensor = None  # type: ignore[assignment]


def _materialize_dtensor(tensor: torch.Tensor) -> torch.Tensor:
    """Reconstruct a plain local tensor from an FSDP2 ``DTensor`` parameter.

    FSDP2 (``fully_shard``) stores parameters as ``DTensor`` sharded over the
    data-parallel mesh, while manual TP/EP sharding leaves each rank holding its
    own local shard as a regular tensor. The HF export gather (``_gather_dense`` /
    ``_gather_expert``) and the downstream rollout weight sender both assume plain
    ``torch.Tensor`` inputs; handing them a ``DTensor`` raises
    ``aten.copy_.default: got mixed torch.Tensor and DTensor``. ``full_tensor()``
    gathers the FSDP shards back into the full (TP/EP-local) tensor; non-DTensor
    params (dist_opt backend) pass through untouched.
    """
    if DTensor is not None and isinstance(tensor, DTensor):
        return tensor.full_tensor()
    return tensor


@runtime_checkable
class HFWeights(Protocol):
    """Protocol for HF ↔ Megatron Lite weight conversion.

    Model-specific implementations only do tensor math, never distributed comm.
    """

    def weight_map(self) -> dict[str, list[str]]:
        """Megatron Lite param name → [HF param names]. Multiple = concat (QKV, gate+up)."""
        ...

    def hf_to_native(
        self, native_name: str, hf_tensors: list[torch.Tensor]
    ) -> torch.Tensor:
        """Convert HF tensors → single Megatron Lite tensor (e.g. merge QKV)."""
        ...

    def native_to_hf(
        self, native_name: str, tensor: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]:
        """Convert Megatron Lite tensor → [(hf_name, hf_tensor)] (e.g. split QKV back)."""
        ...

    def tp_spec(self, native_name: str) -> tuple[int, int] | None:
        """TP sharding: ``(split_dim, 0=TP|1=ETP)``, or None if replicated."""
        ...

    def qkv_spec(self, native_name: str) -> tuple[int, int, int] | None:
        """If native_name is a fused QKV weight, return (num_q_heads, num_kv_heads, head_dim).

        Needed for correct GQA TP sharding — Q/K/V must be split independently.
        Return None for non-QKV parameters.
        """
        return None

    @property
    def num_experts(self) -> int:
        """Total number of experts (needed for EP gather index math)."""
        ...

    def is_expert(self, native_name: str) -> bool:
        """Whether this param belongs to an expert (for EP sharding)."""
        ...

    def expert_global_id(self, native_name: str) -> int | None:
        """Global expert ID from synthetic name. None if not expert."""
        ...

    def expert_local_name(self, native_name: str, local_idx: int) -> str:
        """Synthetic expert name → actual model param name."""
        ...


# ======================================================================
# SafeTensors I/O
# ======================================================================


def _require_safetensors_io():
    """Import the optional file-I/O dependency only at an actual I/O boundary.

    Model/config inspection and pure name/shape conversion are useful in CPU
    build and lint environments that do not install checkpoint codecs.  Real HF
    reads and writes still fail closed with an actionable dependency error.
    """
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except (ImportError, OSError) as exc:
        raise ImportError(
            "Hugging Face checkpoint I/O requires the optional 'safetensors' package."
        ) from exc
    return safe_open, save_file


class SafeTensorReader:
    """Read individual tensors from an HF safetensors directory."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.index = self._load_index()

    def _load_index(self) -> dict[str, str]:
        idx_file = self.path / "model.safetensors.index.json"
        if idx_file.exists():
            with open(idx_file) as f:
                return json.load(f)["weight_map"]
        return {}

    def get_tensor(self, name: str) -> torch.Tensor:
        if self.index:
            filepath = self.path / self.index[name]
        else:
            filepath = self.path / "model.safetensors"
        safe_open, _save_file = _require_safetensors_io()
        with safe_open(str(filepath), framework="pt", device="cpu") as f:
            return f.get_tensor(name)


def unwrap_model(model: nn.Module) -> nn.Module:
    """Strip nested wrapper modules like DDP -> model."""
    base_model = model
    seen: set[int] = set()
    while hasattr(base_model, "module"):
        ident = id(base_model)
        if ident in seen:
            break
        seen.add(ident)
        next_model = base_model.module
        if not isinstance(next_model, nn.Module) or next_model is base_model:
            break
        base_model = next_model
    return base_model


def named_persistent_buffers(model: nn.Module):
    """Yield only buffers that PyTorch includes in ``state_dict``.

    ``named_buffers`` also exposes entries explicitly registered as
    non-persistent. Exporting those implementation caches creates keys that a
    checkpoint cannot load back and, for hash routing, can misrepresent unused
    correction bias as model state.
    """
    for full_name, buffer in model.named_buffers():
        module_name, _, local_name = full_name.rpartition(".")
        owner = model.get_submodule(module_name) if module_name else model
        if local_name in owner._non_persistent_buffers_set:
            continue
        yield full_name, buffer


def save_safetensors(
    tensors: dict[str, torch.Tensor], path: str, filename: str = "model.safetensors"
) -> None:
    os.makedirs(path, exist_ok=True)
    _safe_open, save_file = _require_safetensors_io()
    save_file(tensors, os.path.join(path, filename))


def _distributed_raise_if_error(
    local_error: str | None,
    *,
    context: str,
    error_type: type[Exception] = RuntimeError,
    participating_group: dist.ProcessGroup | None = None,
) -> None:
    """Propagate a local failure within the ranks participating in an operation.

    ``participating_group=None`` deliberately means WORLD and is used by the
    distributed save protocol.  Lower-level exporters must pass their own
    process group so exporting one model replica does not require unrelated DP
    replicas to enter a WORLD collective.
    """
    if not dist.is_initialized():
        if local_error is not None:
            raise error_type(f"{context}: {local_error}")
        return
    group = dist.group.WORLD if participating_group is None else participating_group
    backend = str(
        dist.get_backend() if participating_group is None else dist.get_backend(group)
    ).lower()
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if "nccl" in backend
        else torch.device("cpu")
    )
    failed = torch.tensor(
        [int(local_error is not None)], dtype=torch.int32, device=device
    )
    dist.all_reduce(failed, group=group)
    if not failed.item():
        return
    group_size = (
        dist.get_world_size()
        if participating_group is None
        else dist.get_world_size(group)
    )
    messages: list[str | None] = [None] * group_size
    dist.all_gather_object(messages, local_error, group=group)
    first_error = next((message for message in messages if message is not None), None)
    raise error_type(f"{context}: {first_error or 'unknown distributed failure'}")


def materialize_hf_load_state(
    builder: Callable[[], dict[str, torch.Tensor]],
    *,
    context: str,
    participating_group: dist.ProcessGroup | None = None,
) -> dict[str, torch.Tensor]:
    """Run a read-only HF load builder and propagate every rank-local failure.

    Model-specific loaders can own different PP layers and therefore read
    different checkpoint keys.  A missing/corrupt key on one stage must not let
    the other stages proceed to mutation (or to their next collective).  The
    builder is required to be side-effect free with respect to model state;
    this function catches its local error and makes every participating rank
    fail at the same consensus point.
    """
    loaded: dict[str, torch.Tensor] = {}
    local_error = None
    try:
        loaded = builder()
        if not isinstance(loaded, dict):
            raise TypeError(
                f"HF load builder returned {type(loaded).__name__}, expected dict"
            )
    except Exception as exc:  # every rank still enters the consensus below
        local_error = f"{type(exc).__name__}: {exc}"
    _distributed_raise_if_error(
        local_error,
        context=f"{context} materialization failed",
        participating_group=participating_group,
    )
    return loaded


def _resolve_load_target(name: str, targets: dict[str, torch.Tensor]) -> str | None:
    if name in targets:
        return name
    suffix = f".{name}"
    matches = sorted(key for key in targets if key.endswith(suffix))
    if not matches:
        return None
    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous wrapped checkpoint target for {name!r}: {matches}"
        )
    return matches[0]


def _is_adapter_parameter(name: str) -> bool:
    lowered = name.lower()
    return "lora" in lowered or "adapter" in lowered


def _validate_load_dtype(name: str, source: torch.Tensor, target: torch.Tensor) -> None:
    """Apply the explicit dtype policy used by native HF checkpoint loads.

    Floating checkpoints may be cast to the model's configured floating dtype.
    Quantized/integer data may not be copied into floating parameters unless a
    model-specific loader has already dequantized it.  Persistent integer/bool
    state is schema data and therefore requires an exact dtype match.
    """
    if target.is_floating_point():
        if not source.is_floating_point():
            raise RuntimeError(
                f"checkpoint dtype mismatch for {name}: source={source.dtype}, "
                f"target={target.dtype}; quantized tensors must be dequantized first"
            )
        return
    if target.is_complex():
        if not source.is_complex():
            raise RuntimeError(
                f"checkpoint dtype mismatch for {name}: source={source.dtype}, "
                f"target={target.dtype}"
            )
        return
    if source.dtype != target.dtype:
        raise RuntimeError(
            f"checkpoint dtype mismatch for {name}: source={source.dtype}, "
            f"target={target.dtype}; non-floating state requires an exact dtype"
        )


def _copy_tensor_for_hf_load(target: torch.Tensor, source: torch.Tensor) -> None:
    """Small indirection kept monkeypatchable for transactional-copy tests."""
    target.copy_(source)


def _distributed_any_error(
    local_error: str | None, *, participating_group: dist.ProcessGroup | None
) -> bool:
    if not dist.is_initialized():
        return local_error is not None
    group = dist.group.WORLD if participating_group is None else participating_group
    backend = str(dist.get_backend(group)).lower()
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if "nccl" in backend
        else torch.device("cpu")
    )
    failed = torch.tensor(
        [int(local_error is not None)], dtype=torch.int32, device=device
    )
    dist.all_reduce(failed, group=group)
    return bool(failed.item())


def _prepare_hf_state_copy(
    model: nn.Module,
    loaded: dict[str, torch.Tensor],
    *,
    allow_missing_parameter: Callable[[str], bool],
) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    parameters = dict(model.named_parameters())
    buffers = dict(named_persistent_buffers(model))
    overlap = sorted(set(parameters).intersection(buffers))
    if overlap:
        raise RuntimeError(f"parameter/buffer name collision: {overlap}")
    targets: dict[str, torch.Tensor] = {**parameters, **buffers}
    resolved: dict[str, tuple[str, torch.Tensor]] = {}
    for native_name, source in loaded.items():
        if not isinstance(native_name, str):
            raise TypeError(
                f"checkpoint target name must be str, got {type(native_name).__name__}"
            )
        if not isinstance(source, torch.Tensor):
            raise TypeError(
                f"checkpoint tensor for {native_name} must be torch.Tensor, "
                f"got {type(source).__name__}"
            )
        actual = _resolve_load_target(native_name, targets)
        if actual is None:
            raise RuntimeError(f"checkpoint tensor has no native target: {native_name}")
        if actual in resolved:
            previous = resolved[actual][0]
            raise RuntimeError(
                f"checkpoint tensors {previous!r} and {native_name!r} both "
                f"resolve to native target {actual!r}"
            )
        target = targets[actual]
        if source.shape != target.shape:
            raise RuntimeError(
                f"checkpoint shape mismatch for {actual}: "
                f"source={tuple(source.shape)} target={tuple(target.shape)}"
            )
        if target.is_meta:
            raise RuntimeError(
                f"checkpoint target {actual} is still on the meta device"
            )
        _validate_load_dtype(actual, source, target)
        resolved[actual] = (native_name, source)

    missing_parameters = sorted(
        name
        for name in parameters
        if name not in resolved and not allow_missing_parameter(name)
    )
    missing_buffers = sorted(name for name in buffers if name not in resolved)
    if missing_parameters or missing_buffers:
        details = []
        if missing_parameters:
            details.append(f"parameters={missing_parameters}")
        if missing_buffers:
            details.append(f"persistent_buffers={missing_buffers}")
        raise RuntimeError(
            "checkpoint does not cover all required native state: " + ", ".join(details)
        )
    return [(name, targets[name], resolved[name][1]) for name in sorted(resolved)]


def copy_hf_states_atomically(
    model_states: list[tuple[nn.Module, dict[str, torch.Tensor]]],
    *,
    context: str,
    participating_group: dist.ProcessGroup | None = None,
    allow_missing_parameter: Callable[[str], bool] = _is_adapter_parameter,
) -> int:
    """Preflight and commit every VPP chunk as one distributed transaction."""
    prepared: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    local_error = None
    try:
        if not model_states:
            raise ValueError("HF load transaction requires at least one model chunk")
        for chunk_idx, (model, loaded) in enumerate(model_states):
            try:
                chunk_prepared = _prepare_hf_state_copy(
                    model, loaded, allow_missing_parameter=allow_missing_parameter
                )
            except Exception as exc:
                raise RuntimeError(f"chunk{chunk_idx}: {exc}") from exc
            prepared.extend(
                (f"chunk{chunk_idx}.{name}", target, source)
                for name, target, source in chunk_prepared
            )
    except Exception as exc:  # every rank still enters the consensus below
        local_error = f"{type(exc).__name__}: {exc}"

    _distributed_raise_if_error(
        local_error,
        context=f"{context} preflight failed",
        participating_group=participating_group,
    )

    # The source dictionaries are private staging state. Releasing their keys
    # before commit lets rollback snapshots reuse that host-memory envelope.
    for _model, loaded in model_states:
        loaded.clear()
    backups: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    local_error = None
    with torch.no_grad():
        for index, (name, target, source) in enumerate(prepared):
            try:
                backup = target.detach().to(device="cpu", copy=True)
                backups.append((name, target, backup))
                converted = source.to(device=target.device, dtype=target.dtype)
                _copy_tensor_for_hf_load(target, converted)
                # Drop the no-longer-needed source reference.  The rollback
                # snapshot now occupies its host-memory slot conceptually.
                prepared[index] = (name, target, backup)
            except Exception as exc:
                local_error = f"{type(exc).__name__}: copying {name}: {exc}"
                break

    if _distributed_any_error(local_error, participating_group=participating_group):
        rollback_error = None
        with torch.no_grad():
            for name, target, backup in backups:
                try:
                    restored = backup.to(device=target.device, dtype=target.dtype)
                    _copy_tensor_for_hf_load(target, restored)
                except Exception as exc:
                    rollback_error = rollback_error or (
                        f"{type(exc).__name__}: restoring {name}: {exc}"
                    )
        combined_error = local_error
        if rollback_error is not None:
            combined_error = (
                f"{combined_error}; rollback failed: {rollback_error}"
                if combined_error is not None
                else f"rollback failed: {rollback_error}"
            )
        _distributed_raise_if_error(
            combined_error,
            context=f"{context} atomic copy failed",
            participating_group=participating_group,
        )
        raise AssertionError("unreachable after distributed HF load failure")

    return len(prepared)


def copy_hf_state_atomically(
    model: nn.Module,
    loaded: dict[str, torch.Tensor],
    *,
    context: str,
    participating_group: dist.ProcessGroup | None = None,
    allow_missing_parameter: Callable[[str], bool] = _is_adapter_parameter,
) -> int:
    """Strictly preflight and transactionally copy one native HF load state.

    The multi-chunk implementation retains CPU rollback snapshots until the
    complete transaction succeeds, so this wrapper preserves the original
    single-model API without weakening VPP atomicity.
    """
    return copy_hf_states_atomically(
        [(model, loaded)],
        context=context,
        participating_group=participating_group,
        allow_missing_parameter=allow_missing_parameter,
    )


def load_hf_model_chunks_atomically(
    models: nn.Module | list[nn.Module],
    builder: Callable[[nn.Module], dict[str, torch.Tensor]],
    *,
    context: str,
    participating_group: dist.ProcessGroup | None = None,
    allow_missing_parameter: Callable[[str], bool] = _is_adapter_parameter,
) -> int:
    """Materialize all VPP chunks before committing any native model state."""
    chunks = [models] if isinstance(models, nn.Module) else list(models)
    staged: list[tuple[nn.Module, dict[str, torch.Tensor]]] = []
    for chunk_idx, chunk in enumerate(chunks):
        loaded = materialize_hf_load_state(
            lambda chunk=chunk: builder(chunk),
            context=f"{context} chunk {chunk_idx}",
            participating_group=participating_group,
        )
        staged.append((unwrap_model(chunk), loaded))
    return copy_hf_states_atomically(
        staged,
        context=context,
        participating_group=participating_group,
        allow_missing_parameter=allow_missing_parameter,
    )


def materialize_hf_weights_distributed(weights) -> dict[str, torch.Tensor]:
    """Materialize an export generator with duplicate and WORLD-safe error handling."""
    items: list[tuple[str, torch.Tensor]] = []
    local_error = None
    try:
        items = list(weights)
        seen: set[str] = set()
        duplicates: set[str] = set()
        for name, _tensor in items:
            if name in seen:
                duplicates.add(name)
            seen.add(name)
        duplicates = sorted(duplicates)
        if duplicates:
            raise AssertionError(f"duplicate HF export keys: {duplicates}")
    except Exception as exc:  # every rank reaches the consensus below
        local_error = f"{type(exc).__name__}: {exc}"
    _distributed_raise_if_error(local_error, context="HF weight materialization failed")
    return dict(items)


def save_hf_weight_pairs_distributed(weights, hf_path: str) -> None:
    """Materialize, write on rank 0, and propagate write failures to every rank."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    out = materialize_hf_weights_distributed(weights)
    local_error = None
    if rank == 0:
        if not out:
            local_error = "ValueError: rank 0 materialized no HF weights"
        else:
            try:
                save_safetensors(out, hf_path)
            except Exception as exc:
                local_error = f"{type(exc).__name__}: {exc}"
    _distributed_raise_if_error(local_error, context="HF safetensors write failed")
    if dist.is_initialized():
        dist.barrier()


def gather_pipeline_state_dict(
    local_state: dict[str, torch.Tensor],
    ps,
    *,
    participating_group: dist.ProcessGroup | None = None,
) -> dict[str, torch.Tensor]:
    """Gather small non-parameter export state across a pipeline group.

    The main HF exporter already gathers parameters across PP. Model-specific
    persistent buffers (router correction bias, hash-routing tables, etc.) must
    follow the same rule before a ``rank0_only`` writer filters other ranks.
    Duplicate native names are always rejected. Legal PP ownership is disjoint;
    accepting identical (often all-zero) buffers would hide a broken global
    layer remap or duplicated stage.
    """
    if ps.pp_size <= 1:
        return local_state
    all_states: list[dict[str, torch.Tensor] | None] = [None] * ps.pp_size
    dist.all_gather_object(all_states, local_state, group=ps.pp_group)
    gathered: dict[str, torch.Tensor] = {}
    local_error = None
    for stage_idx, state in enumerate(all_states):
        if state is None:
            local_error = local_error or (
                f"pipeline export buffer state missing for stage {stage_idx}"
            )
            continue
        for name, tensor in state.items():
            if name in gathered:
                local_error = local_error or (
                    f"pipeline export buffer collision for {name}"
                )
                continue
            gathered[name] = tensor
    _distributed_raise_if_error(
        local_error,
        context="PP buffer export failed",
        error_type=AssertionError,
        participating_group=(
            ps.pp_group if participating_group is None else participating_group
        ),
    )
    return gathered


def _validate_mtp_embedding_replica(state: dict[str, torch.Tensor]) -> None:
    """Refuse to drop a divergent PP MTP embedding during HF export.

    HF formats carry one canonical input embedding.  A PP loss stage may own a
    physical ``mtp_embed`` replica, but it is legal to omit that duplicate only
    after proving exact equality with the first-stage tensor.
    """
    replica_name = "mtp_embed.embedding.weight"
    replica = state.get(replica_name)
    if replica is None:
        return
    canonical_names = [
        name
        for name in ("embed.embedding.weight", "embed_tokens.embedding.weight")
        if name in state
    ]
    if len(canonical_names) != 1:
        raise AssertionError(
            "HF export found an MTP embedding replica without exactly one canonical "
            f"input embedding: candidates={canonical_names}"
        )
    canonical_name = canonical_names[0]
    canonical = state[canonical_name]
    if canonical.shape != replica.shape or canonical.dtype != replica.dtype:
        raise AssertionError(
            "HF export MTP embedding replica metadata mismatch: "
            f"{canonical_name} shape={tuple(canonical.shape)} dtype={canonical.dtype}, "
            f"{replica_name} shape={tuple(replica.shape)} dtype={replica.dtype}"
        )
    if not torch.equal(canonical, replica):
        max_abs = (
            (canonical.detach().float() - replica.detach().float()).abs().max().item()
        )
        raise AssertionError(
            "HF export refuses to drop a divergent PP MTP embedding replica: "
            f"canonical={canonical_name}, replica={replica_name}, max_abs={max_abs:.6e}"
        )


def _resolve_export_dtype(export_dtype: str | torch.dtype | None) -> torch.dtype | None:
    if export_dtype is None:
        return None
    if isinstance(export_dtype, torch.dtype):
        return export_dtype
    normalized = str(export_dtype).lower()
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "float": torch.float32,
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported export_dtype={export_dtype!r}")
    return aliases[normalized]


def _cast_export_tensor(
    tensor: torch.Tensor, export_dtype: torch.dtype | None
) -> torch.Tensor:
    if export_dtype is None or not tensor.is_floating_point():
        return tensor
    return tensor.to(dtype=export_dtype)


def _is_vocab_parallel_state(name: str) -> bool:
    return name in {
        "embed.embedding.weight",
        "embed_tokens.embedding.weight",
        "mtp_embed.embedding.weight",
        "head.col.linear.weight",
        "lm_head.col.linear.weight",
    }


# ======================================================================
# Tensor utilities
# ======================================================================


def split_dim(
    tensor: torch.Tensor, rank: int, world: int, dim: int = 0
) -> torch.Tensor:
    if world <= 1:
        return tensor
    return tensor.chunk(world, dim=dim)[rank].contiguous()


def split_qkv(
    tensor: torch.Tensor,
    rank: int,
    world: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """TP-shard a fused [Q, K, V] weight, splitting Q/K/V heads independently.

    Naive ``split_dim`` would slice across the Q/K/V boundary incorrectly
    when num_q_heads != num_kv_heads (GQA).
    """
    if world <= 1:
        return tensor
    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim
    q = tensor[:q_size]
    k = tensor[q_size : q_size + kv_size]
    v = tensor[q_size + kv_size :]
    q_shard = q.chunk(world, dim=0)[rank]
    k_shard = k.chunk(world, dim=0)[rank]
    v_shard = v.chunk(world, dim=0)[rank]
    return torch.cat([q_shard, k_shard, v_shard], dim=0).contiguous()


def split_gate_up(tensor: torch.Tensor, rank: int, world: int) -> torch.Tensor:
    if world <= 1:
        return tensor
    ffn = tensor.shape[0] // 2
    gate = tensor[:ffn].chunk(world, dim=0)[rank]
    up = tensor[ffn:].chunk(world, dim=0)[rank]
    return torch.cat([gate, up], dim=0).contiguous()


def allgather_concat(
    tensor: torch.Tensor, world_size: int, group: dist.ProcessGroup | None, dim: int
) -> torch.Tensor:
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor.contiguous(), group=group)
    return torch.cat(gathered, dim=dim)


def remap_layer_index(name: str, global_to_local: dict[int, int]) -> str | None:
    if not global_to_local:
        return name
    m = re.match(r"(layers\.)(\d+)(\..*)", name)
    if not m:
        return name
    gidx = int(m.group(2))
    if gidx not in global_to_local:
        return None
    return f"{m.group(1)}{global_to_local[gidx]}{m.group(3)}"


def extract_layer_idx(name: str) -> int:
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else 0


def parse_expert_idx(name: str) -> int:
    m = re.search(r"weight(\d+)$", name)
    return int(m.group(1)) if m else 0


def set_expert_idx(name: str, idx: int) -> str:
    return re.sub(r"weight\d+$", f"weight{idx}", name)


def to_global_layer_name(name: str, layer_map: dict[int, int]) -> str:
    if not layer_map:
        return name

    def _replace(m: re.Match) -> str:
        return f"layers.{layer_map.get(int(m.group(1)), int(m.group(1)))}."

    # Only decoder-trunk names use PP-local layer indices. MTP namespaces such
    # as ``mtp.layers.0`` are already depth-local and must not be remapped to a
    # trunk stage's first global decoder index.
    return re.sub(r"^layers\.(\d+)\.", _replace, name)


def gather_gate_up(
    tensor: torch.Tensor, world_size: int, group: dist.ProcessGroup
) -> torch.Tensor:
    ffn_local = tensor.shape[0] // 2
    gate_full = allgather_concat(tensor[:ffn_local], world_size, group, dim=0)
    up_full = allgather_concat(tensor[ffn_local:], world_size, group, dim=0)
    return torch.cat([gate_full, up_full], dim=0)


# ======================================================================
# Generic load / export / save using HFWeights
# ======================================================================


def load_hf_weights(
    model: nn.Module | list[nn.Module],
    hf_path: str,
    spec: HFWeights,
    ps,
    *,
    vocab_size: int | None = None,
    participating_group: dist.ProcessGroup | None = None,
) -> None:
    """Load HF safetensors into a Megatron Lite model using HFWeights.

    Handles PP layer filtering, TP split, EP shard assignment.
    ``ps`` is a ParallelState (lazy import to avoid GPU dep at module level).
    Materialization and the final copy are one distributed transaction scoped
    to ``participating_group``; ``None`` retains the historical WORLD scope.
    """
    load_hf_model_chunks_atomically(
        model,
        lambda chunk: _materialize_generic_hf_weights(
            chunk, hf_path, spec, ps, vocab_size=vocab_size
        ),
        context=f"{type(spec).__name__} HF load",
        participating_group=participating_group,
    )


def _materialize_generic_hf_weights(
    model: nn.Module, hf_path: str, spec: HFWeights, ps, *, vocab_size: int | None
) -> dict[str, torch.Tensor]:
    """Build a complete native state without mutating the target model."""
    from megatron.lite.primitive.parallel import pad_vocab_for_tp

    base_model = unwrap_model(model)
    reader = SafeTensorReader(hf_path)
    wmap = spec.weight_map()

    global_to_local: dict[int, int] = (
        {gi: li for li, gi in enumerate(base_model.layer_indices)}
        if hasattr(base_model, "layer_indices")
        else {}
    )

    state = base_model.state_dict()
    loaded: dict[str, torch.Tensor] = {}
    num_experts_total = getattr(spec, "num_experts", None)
    expert_shard = None
    if num_experts_total is None:
        expert_ids = [spec.expert_global_id(name) for name in wmap]
        expert_ids = [expert_id for expert_id in expert_ids if expert_id is not None]
        if expert_ids:
            num_experts_total = max(expert_ids) + 1
    if num_experts_total:
        from megatron.lite.primitive.utils import ensure_divisible

        experts_per_rank = ensure_divisible(num_experts_total, ps.ep_size)
        local_start = ps.ep_rank * experts_per_rank
        expert_shard = (experts_per_rank, local_start)

    for native_name, hf_names in wmap.items():
        mapped = remap_layer_index(native_name, global_to_local)
        if mapped is None:
            continue

        expert_gid = spec.expert_global_id(mapped)
        if expert_gid is not None:
            _load_expert_weight(
                mapped, hf_names, reader, spec, ps, loaded, expert_gid, expert_shard
            )
            continue

        hf_tensors = [reader.get_tensor(n) for n in hf_names]
        tensor = spec.hf_to_native(mapped, hf_tensors)

        tp_info = spec.tp_spec(mapped)
        if tp_info is not None:
            split_d, tp_or_etp = tp_info
            if tp_or_etp == 0:
                if vocab_size is not None and _is_vocab_parallel_state(mapped):
                    padded = pad_vocab_for_tp(vocab_size, ps.tp_size)
                    if tensor.size(0) < padded:
                        pad = torch.zeros(
                            padded - tensor.size(0),
                            *tensor.shape[1:],
                            dtype=tensor.dtype,
                        )
                        tensor = torch.cat([tensor, pad], dim=0)
                qkv = spec.qkv_spec(mapped) if hasattr(spec, "qkv_spec") else None
                if qkv is not None:
                    tensor = split_qkv(tensor, ps.tp_rank, ps.tp_size, *qkv)
                else:
                    tensor = split_dim(tensor, ps.tp_rank, ps.tp_size, dim=split_d)
            else:
                tensor = split_dim(tensor, ps.etp_rank, ps.etp_size, dim=split_d)

        actual = _resolve_param_name(mapped, state)
        if actual:
            loaded[actual] = tensor.to(dtype=torch.bfloat16)
        elif re.search(r"(?:^|\.)layers\.\d+\.", mapped):
            raise KeyError(
                f"WeightSpec maps local layer tensor {mapped!r}, but the target "
                "model has no matching parameter"
            )
    return loaded


def _load_expert_weight(
    native_name, hf_names, reader, spec, ps, loaded, expert_gid, expert_shard
):
    if expert_shard is None:
        raise RuntimeError(
            "Expert weight encountered but expert shard metadata is unavailable."
        )
    experts_per_rank, local_start = expert_shard
    if expert_gid < local_start or expert_gid >= local_start + experts_per_rank:
        return

    hf_tensors = [reader.get_tensor(n) for n in hf_names]
    tensor = spec.hf_to_native(native_name, hf_tensors)

    if ps.etp_size > 1:
        tp_info = spec.tp_spec(native_name)
        if tp_info is not None:
            split_d, _ = tp_info
            if "fc1" in native_name:
                tensor = split_gate_up(tensor, ps.etp_rank, ps.etp_size)
            else:
                tensor = split_dim(tensor, ps.etp_rank, ps.etp_size, dim=split_d)

    loaded[spec.expert_local_name(native_name, expert_gid - local_start)] = tensor.to(
        dtype=torch.bfloat16
    )


def _resolve_param_name(name: str, state_dict: dict) -> str | None:
    if name in state_dict:
        return name
    suffix = f".{name}"
    matches = sorted(key for key in state_dict if key.endswith(suffix))
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"ambiguous wrapped parameter match for {name!r}: {matches}")
    return matches[0]


def export_hf_weights(
    model: nn.Module | list[nn.Module],
    spec: HFWeights,
    ps,
    *,
    vocab_size: int | None = None,
    limit: int | None = None,
    rank0_only: bool = False,
    export_dtype: str | torch.dtype | None = None,
    participating_group: dist.ProcessGroup | None = None,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Export model weights as HF-format (name, tensor) pairs.

    Gathers across TP/ETP/EP/PP so the output is the full unsharded HF state on
    every participating rank. RL weight sync needs every colocated rollout rank
    to receive weights; save paths can pass ``rank0_only=True`` to avoid
    materializing duplicate writers.
    """
    if isinstance(model, nn.ModuleList):
        chunks: list[nn.Module] = list(model)
    elif isinstance(model, list):
        chunks = model
    else:
        chunks = [model]

    rank = dist.get_rank() if dist.is_initialized() else 0
    resolved_export_dtype = _resolve_export_dtype(export_dtype)

    if ps.pp_size <= 1:
        exported_params = 0
        seen_native_names: set[str] = set()
        expert_groups: dict[str, list[tuple[int, str, torch.Tensor]]] = {}
        for chunk in chunks:
            base_chunk = unwrap_model(chunk)
            layer_map = (
                {
                    i: base_chunk.layer_indices[i]
                    for i in range(len(base_chunk.layer_indices))
                }
                if hasattr(base_chunk, "layer_indices")
                else {}
            )
            for name, param in base_chunk.named_parameters():
                gname = to_global_layer_name(name, layer_map)
                if gname in seen_native_names:
                    raise AssertionError(
                        f"local export parameter collision for {gname}"
                    )
                seen_native_names.add(gname)
                tensor = _materialize_dtensor(param.data.detach())

                gathered_one: dict[str, torch.Tensor] = {}
                if spec.is_expert(gname):
                    if limit is None:
                        expert_groups.setdefault(_expert_group_key(gname), []).append(
                            (parse_expert_idx(gname), gname, tensor)
                        )
                        exported_params += 1
                        continue
                    _gather_expert(gname, tensor, spec, ps, gathered_one)
                else:
                    gathered_one[gname] = _gather_dense(gname, tensor, spec, ps)

                exported_params += 1
                if not rank0_only or rank == 0:
                    for native_name, gathered_tensor in gathered_one.items():
                        if vocab_size is not None and _is_vocab_parallel_state(
                            native_name
                        ):
                            gathered_tensor = gathered_tensor[:vocab_size]
                        for hf_name, hf_tensor in spec.native_to_hf(
                            native_name, gathered_tensor
                        ):
                            yield (
                                hf_name,
                                _cast_export_tensor(hf_tensor, resolved_export_dtype),
                            )

                if limit is not None and exported_params >= limit:
                    return

        for group_key in sorted(expert_groups):
            gathered_group: dict[str, torch.Tensor] = {}
            _gather_expert_group(expert_groups[group_key], spec, ps, gathered_group)
            if not rank0_only or rank == 0:
                for native_name in sorted(gathered_group, key=parse_expert_idx):
                    gathered_tensor = gathered_group[native_name]
                    for hf_name, hf_tensor in spec.native_to_hf(
                        native_name, gathered_tensor
                    ):
                        yield (
                            hf_name,
                            _cast_export_tensor(hf_tensor, resolved_export_dtype),
                        )
        return

    gathered: dict[str, torch.Tensor] = {}
    local_error = None
    for chunk in chunks:
        base_chunk = unwrap_model(chunk)
        # Map local layer indices to global for PP
        layer_map = (
            {
                i: base_chunk.layer_indices[i]
                for i in range(len(base_chunk.layer_indices))
            }
            if hasattr(base_chunk, "layer_indices")
            else {}
        )
        for name, param in base_chunk.named_parameters():
            gname = to_global_layer_name(name, layer_map)
            t = _materialize_dtensor(param.data.detach())

            gathered_one: dict[str, torch.Tensor] = {}
            if spec.is_expert(gname):
                _gather_expert(gname, t, spec, ps, gathered_one)
            else:
                gathered_one[gname] = _gather_dense(gname, t, spec, ps)

            # Multiple VPP chunks may live on one rank.  Their global native
            # names must still be disjoint; assigning directly into ``gathered``
            # would silently keep the last chunk before the PP collision check.
            for gathered_name, gathered_tensor in gathered_one.items():
                if gathered_name in gathered:
                    local_error = local_error or (
                        f"pipeline export parameter collision for {gathered_name}"
                    )
                    continue
                gathered[gathered_name] = gathered_tensor

    # PP gather
    if ps.pp_size > 1:
        all_states: list[dict | None] = [None] * ps.pp_size
        dist.all_gather_object(all_states, gathered, group=ps.pp_group)
        gathered = {}
        for stage_idx, state in enumerate(all_states):
            if state is None:
                local_error = local_error or (
                    f"pipeline export parameter state missing for stage {stage_idx}"
                )
                continue
            for name, tensor in state.items():
                if name in gathered:
                    local_error = local_error or (
                        f"pipeline export parameter collision for {name}"
                    )
                    continue
                gathered[name] = tensor

    _distributed_raise_if_error(
        local_error,
        context="PP parameter export failed",
        error_type=AssertionError,
        participating_group=(
            ps.pp_group if participating_group is None else participating_group
        ),
    )
    local_error = None
    try:
        _validate_mtp_embedding_replica(gathered)
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _distributed_raise_if_error(
        local_error,
        context="PP MTP embedding export validation failed",
        error_type=AssertionError,
        participating_group=(
            ps.pp_group if participating_group is None else participating_group
        ),
    )

    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank0_only and rank != 0:
        return

    # Vocab trim
    if vocab_size is not None:
        for key in list(gathered.keys()):
            if _is_vocab_parallel_state(key):
                gathered[key] = gathered[key][:vocab_size]

    # Convert Megatron Lite names → HF names via spec
    for native_name, tensor in gathered.items():
        for hf_name, hf_tensor in spec.native_to_hf(native_name, tensor):
            yield hf_name, _cast_export_tensor(hf_tensor, resolved_export_dtype)


def _gather_dense(name: str, tensor: torch.Tensor, spec: HFWeights, ps) -> torch.Tensor:
    """Gather a dense (non-expert) param across TP."""
    custom_gather = getattr(spec, "gather_dense", None)
    if callable(custom_gather):
        gathered = custom_gather(name, tensor, ps)
        if gathered is not None:
            return gathered.cpu()

    tp_info = spec.tp_spec(name)
    if tp_info is not None and ps.tp_size > 1:
        split_d, tp_or_etp = tp_info
        if tp_or_etp == 0:
            tensor = allgather_concat(tensor, ps.tp_size, ps.tp_group, dim=split_d)
    return tensor.cpu()


def _gather_expert(
    name: str, tensor: torch.Tensor, spec: HFWeights, ps, out: dict[str, torch.Tensor]
) -> None:
    """Gather an expert param across ETP + EP."""
    tensor = _gather_expert_etp(name, tensor, spec, ps)

    # EP gather: global_id = ep_rank * n_local + local_id.
    local_idx = parse_expert_idx(name)
    if ps.ep_size > 1 and ps.ep_group is not None:
        n_local = spec.num_experts // ps.ep_size
        ep_gathered = [torch.empty_like(tensor) for _ in range(ps.ep_size)]
        dist.all_gather(ep_gathered, tensor.contiguous(), group=ps.ep_group)
        for ep_rank, ep_tensor in enumerate(ep_gathered):
            global_idx = ep_rank * n_local + local_idx
            out[set_expert_idx(name, global_idx)] = ep_tensor.cpu()
    else:
        out[name] = tensor.cpu()


def _gather_expert_etp(
    name: str, tensor: torch.Tensor, spec: HFWeights, ps
) -> torch.Tensor:
    # ETP gather
    if ps.etp_size > 1 and ps.etp_group is not None:
        tp_info = spec.tp_spec(name)
        if tp_info is not None:
            split_d, _ = tp_info
            if "fc1" in name:
                return gather_gate_up(tensor, ps.etp_size, ps.etp_group)
            return allgather_concat(tensor, ps.etp_size, ps.etp_group, dim=split_d)
    return tensor


def _expert_group_key(name: str) -> str:
    return re.sub(r"weight\d+$", "weight", name)


def _gather_expert_group(
    entries: list[tuple[int, str, torch.Tensor]],
    spec: HFWeights,
    ps,
    out: dict[str, torch.Tensor],
) -> None:
    """Gather local experts in one EP collective per layer/kind."""
    prepared = [
        (local_idx, name, _gather_expert_etp(name, tensor, spec, ps))
        for local_idx, name, tensor in sorted(entries)
    ]
    packed_group_name = getattr(spec, "packed_expert_group_name", None)
    if callable(packed_group_name):
        packed_name = packed_group_name(prepared[0][1])
        if packed_name is not None:
            if ps.ep_size <= 1 or ps.ep_group is None:
                out[packed_name] = torch.stack(
                    [tensor.contiguous() for _, _, tensor in prepared], dim=0
                ).cpu()
                return

            stacked = torch.stack(
                [tensor.contiguous() for _, _, tensor in prepared], dim=0
            )
            ep_gathered = [torch.empty_like(stacked) for _ in range(ps.ep_size)]
            dist.all_gather(ep_gathered, stacked, group=ps.ep_group)
            out[packed_name] = torch.cat(ep_gathered, dim=0).cpu()
            return

    if ps.ep_size <= 1 or ps.ep_group is None:
        for _, name, tensor in prepared:
            out[name] = tensor.cpu()
        return

    n_local = spec.num_experts // ps.ep_size
    stacked = torch.stack([tensor.contiguous() for _, _, tensor in prepared], dim=0)
    ep_gathered = [torch.empty_like(stacked) for _ in range(ps.ep_size)]
    dist.all_gather(ep_gathered, stacked, group=ps.ep_group)
    for ep_rank, ep_tensor in enumerate(ep_gathered):
        for slot, (local_idx, name, _) in enumerate(prepared):
            global_idx = ep_rank * n_local + local_idx
            out[set_expert_idx(name, global_idx)] = ep_tensor[slot].cpu()


def save_hf_weights(
    model: nn.Module | list[nn.Module],
    hf_path: str,
    spec: HFWeights,
    ps,
    *,
    vocab_size: int | None = None,
) -> None:
    """Export + write to safetensors."""
    save_hf_weight_pairs_distributed(
        export_hf_weights(model, spec, ps, vocab_size=vocab_size, rank0_only=True),
        hf_path,
    )
