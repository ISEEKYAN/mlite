# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Raw HF weight update bridge for miles/slime-family rollout engines."""

from __future__ import annotations

import logging
import importlib
from collections.abc import Sequence

import ray
import torch.distributed as dist

logger = logging.getLogger(__name__)


class RawHFWeightUpdater:
    """Send MLite-exported HF-format weights through slime-family update APIs."""

    def __init__(self, args, runtime, handle, *, family_pkg: str) -> None:
        self.args = args
        self.runtime = runtime
        self.handle = handle
        self.family_pkg = family_pkg
        self.weight_version = 0
        self._ipc_gather_group = None
        self._ipc_gather_src = None
        self._ipc_engine = None
        self._model_update_groups = None
        self.rollout_engines = []
        self.distributed_rollout_engines = []
        self.use_distribute = False

    @property
    def _ps(self):
        return self.handle._parallel_state

    @property
    def _is_distributed_src_rank(self) -> bool:
        ps = self._ps
        return (
            int(getattr(ps, "dp_rank", 0) or 0) == 0
            and int(getattr(ps, "cp_rank", 0) or 0) == 0
            and int(getattr(ps, "tp_rank", 0) or 0) == 0
            and int(getattr(ps, "pp_rank", 0) or 0) == 0
        )

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence,
        rollout_engine_lock,
        *,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        broadcast = importlib.import_module(
            f"{self.family_pkg}.backends.megatron_utils.update_weight.update_weight_from_distributed.broadcast"
        )
        connect_rollout_engines_from_distributed = broadcast.connect_rollout_engines_from_distributed
        disconnect_rollout_engines_from_distributed = broadcast.disconnect_rollout_engines_from_distributed

        self.rollout_engines = list(rollout_engines)
        self.rollout_engine_lock = rollout_engine_lock

        if engine_gpu_counts is None:
            engine_gpu_counts = [self.args.rollout_num_gpus_per_engine] * len(rollout_engines)
        if engine_gpu_offsets is None:
            engine_gpu_offsets = []
            offset = 0
            for count in engine_gpu_counts:
                engine_gpu_offsets.append(offset)
                offset += count

        total_actor_gpus = self.args.actor_num_nodes * self.args.actor_num_gpus_per_node
        colocate_engine_nums = 0
        for gpu_offset, gpu_count in zip(engine_gpu_offsets, engine_gpu_counts, strict=True):
            if gpu_offset + gpu_count > total_actor_gpus:
                break
            colocate_engine_nums += 1

        self.use_distribute = len(rollout_engines) > colocate_engine_nums
        self.distributed_rollout_engines = list(rollout_engines[colocate_engine_nums:])
        self.rollout_engines = list(rollout_engines[:colocate_engine_nums])

        if self.use_distribute and self._is_distributed_src_rank:
            if self._model_update_groups is not None:
                disconnect_rollout_engines_from_distributed(
                    self.args, "mlite", self._model_update_groups, self.distributed_rollout_engines
                )
            self._model_update_groups = connect_rollout_engines_from_distributed(
                self.args,
                "mlite",
                self.distributed_rollout_engines,
                engine_gpu_counts=engine_gpu_counts[colocate_engine_nums:],
            )

        self._connect_colocated_groups(engine_gpu_counts[:colocate_engine_nums], engine_gpu_offsets[:colocate_engine_nums])

    def _connect_colocated_groups(self, engine_gpu_counts, engine_gpu_offsets) -> None:
        rank = dist.get_rank()
        self._ipc_gather_group = None
        self._ipc_gather_src = None
        self._ipc_engine = None

        for engine_idx, (offset, count) in enumerate(zip(engine_gpu_offsets, engine_gpu_counts, strict=True)):
            group_ranks = list(range(offset, offset + count))
            new_group = dist.new_group(ranks=group_ranks, backend="gloo")
            if rank in group_ranks:
                self._ipc_gather_group = new_group
                self._ipc_gather_src = offset
                self._ipc_engine = self.rollout_engines[engine_idx]

    def update_weights(self) -> None:
        common = importlib.import_module(f"{self.family_pkg}.backends.megatron_utils.update_weight.common")
        tensor_update = importlib.import_module(
            f"{self.family_pkg}.backends.megatron_utils.update_weight.update_weight_from_tensor"
        )
        broadcast = importlib.import_module(
            f"{self.family_pkg}.backends.megatron_utils.update_weight.update_weight_from_distributed.broadcast"
        )
        distributed_utils = importlib.import_module(f"{self.family_pkg}.utils.distributed_utils")
        _check_weight_sync_results = common._check_weight_sync_results
        post_process_weights = common.post_process_weights
        _send_to_colocated_engine = tensor_update._send_to_colocated_engine
        update_weights_from_distributed = broadcast.update_weights_from_distributed
        get_gloo_group = distributed_utils.get_gloo_group

        self.weight_version += 1
        rank = dist.get_rank()
        if rank == 0:
            mode = self.args.pause_generation_mode
            ray.get([engine.pause_generation.remote(mode=mode) for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

        refs = []
        live_tensors = []
        for chunk in self._export_weight_chunks():
            colocated_refs, long_lived = _send_to_colocated_engine(
                hf_named_tensors=chunk,
                ipc_engine=self._ipc_engine,
                ipc_gather_src=self._ipc_gather_src,
                ipc_gather_group=self._ipc_gather_group,
                weight_version=self.weight_version,
            )
            refs.extend(colocated_refs)
            if long_lived is not None:
                live_tensors.append(long_lived)
            if self.use_distribute and self._is_distributed_src_rank:
                refs.extend(
                    update_weights_from_distributed(
                        "mlite",
                        self._model_update_groups,
                        self.weight_version,
                        self.distributed_rollout_engines,
                        chunk,
                    )
                )
        if refs:
            _check_weight_sync_results(ray.get(refs), is_lora=False)
        del live_tensors

        dist.barrier(group=get_gloo_group())
        if rank == 0:
            post_process_weights(
                rollout_engines=self.rollout_engines,
                restore_weights_before_load=False,
                post_process_quantization=True,
            )
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

    def _export_weight_chunks(self):
        chunk = []
        chunk_bytes = 0
        limit = int(getattr(self.args, "update_weight_buffer_size", 2**30))
        export_kwargs = {}
        if getattr(self.args, "mlite_export_dtype", None):
            export_kwargs["export_dtype"] = self.args.mlite_export_dtype
        for name, tensor in self.runtime.export_weights(self.handle, **export_kwargs):
            tensor = tensor.detach()
            item_bytes = tensor.numel() * tensor.element_size()
            if chunk and chunk_bytes + item_bytes > limit:
                yield chunk
                chunk = []
                chunk_bytes = 0
            chunk.append((name, tensor))
            chunk_bytes += item_bytes
        if chunk:
            yield chunk
