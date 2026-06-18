# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Megatron Lite training actor for slime.

``MLiteTrainRayActor`` implements slime's ``TrainRayActor`` contract on top of
the Megatron Lite runtime (``create_runtime`` / ``build_model`` /
``forward_backward`` / ``optimizer_step`` / ``save_checkpoint``). Because
Megatron Lite loads weights straight from the HF checkpoint, there is no
HF->torch_dist conversion stage like the megatron backend.

S1 scope: model build + supervised fine-tuning training step + checkpoint save.
Weight resync to the rollout engine (``update_weights``) and offload-driven
sleep/wake are wired as follow-ups (S2); ``update_weights`` is a no-op in the
SFT-only ``--debug-train-only`` path that S1 targets.
"""

from __future__ import annotations

import logging
from argparse import Namespace
from typing import Any

import torch

from slime.ray.train_actor import TrainRayActor
from slime.utils.data import process_rollout_data
from slime.utils.misc import Box

from .arguments import optimizer_backend_to_impl
from .data import build_sft_microbatches
from .loss import make_sft_loss_fn

logger = logging.getLogger(__name__)


class MLiteTrainRayActor(TrainRayActor):
    def _build_mlite_config(self, args: Namespace):
        from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
        from megatron.lite.runtime.contracts.config import OptimizerConfig, ParallelConfig

        parallel = ParallelConfig(
            tp=args.tensor_model_parallel_size,
            etp=getattr(args, "expert_tensor_parallel_size", None),
            ep=getattr(args, "expert_model_parallel_size", 1),
            pp=args.pipeline_model_parallel_size,
            vpp=getattr(args, "virtual_pipeline_model_parallel_size", None) or 1,
            cp=args.context_parallel_size,
        )
        optimizer = OptimizerConfig(
            optimizer=args.optimizer,
            lr=args.lr,
            min_lr=args.min_lr if args.min_lr is not None else 0.0,
            clip_grad=args.clip_grad,
            weight_decay=args.weight_decay,
            lr_decay_style=args.lr_decay_style,
            adam_beta1=getattr(args, "adam_beta1", None),
            adam_beta2=getattr(args, "adam_beta2", None),
            adam_eps=getattr(args, "adam_eps", None),
        )
        if getattr(args, "mlite_optimizer_offload", False):
            optimizer.offload_fraction = 1.0
            optimizer.use_precision_aware_optimizer = True
            optimizer.decoupled_weight_decay = True
        # Megatron parses --attention-backend into an AttnBackend enum; Megatron
        # Lite's attention_backend_override expects the string name (e.g. "flash").
        attention_backend = args.mlite_attention_backend
        if attention_backend is None:
            raw = getattr(args, "attention_backend", None)
            attention_backend = getattr(raw, "name", raw)
        attention_backend = attention_backend or "flash"
        return MegatronLiteConfig(
            model_name=args.mlite_model_name,
            impl=args.mlite_impl,
            hf_path=args.hf_checkpoint,
            parallel=parallel,
            optimizer=optimizer,
            attention_backend_override=attention_backend,
            load_hf_weights=True,
            impl_cfg={
                "use_thd": True,
                "optimizer": optimizer_backend_to_impl(args.mlite_optimizer_backend),
            },
        )

    def init(
        self,
        args: Namespace,
        role: str,
        with_ref: bool = False,
        with_opd_teacher: bool = False,
    ) -> int | None:
        if args.debug_rollout_only:
            self.args = args
            return 0

        super().init(args, role, with_ref, with_opd_teacher)

        if role != "actor":
            raise NotImplementedError(
                "Megatron Lite slime backend currently supports the actor role only."
            )

        from megatron.lite.runtime import RuntimeConfig, create_runtime

        self._cfg = self._build_mlite_config(args)
        self.runtime = create_runtime(
            RuntimeConfig(backend="mlite", hf_path=args.hf_checkpoint, backend_cfg=self._cfg)
        )
        self.handle = self.runtime.build_model()

        ps = self.handle._parallel_state
        self.train_parallel_config = {
            "dp_size": ps.dp_size,
            "cp_size": ps.cp_size,
            "vpp_size": self._cfg.parallel.vpp or 1,
            "microbatch_group_size_per_vp_stage": 1,
        }

        start_rollout_id = 0
        if getattr(args, "load", None):
            loaded = self.runtime.load_checkpoint(self.handle, args.load)
            start_rollout_id = int(loaded) + 1

        return start_rollout_id

    def _get_rollout_data(self, rollout_data_ref: Box) -> dict[str, Any]:
        ps = self.handle._parallel_state
        rollout_data = process_rollout_data(self.args, rollout_data_ref, ps.dp_rank, ps.dp_size)
        device = torch.cuda.current_device()
        rollout_data["tokens"] = [
            t.to(device=device, dtype=torch.long, non_blocking=True) for t in rollout_data["tokens"]
        ]
        rollout_data["loss_masks"] = [
            t.to(device=device, dtype=torch.float32, non_blocking=True)
            for t in rollout_data["loss_masks"]
        ]
        return rollout_data

    def train(self, rollout_id: int, rollout_data_ref: Box, external_data=None):
        if self.args.debug_rollout_only:
            return None

        if getattr(self.args, "loss_type", "sft_loss") != "sft_loss":
            raise NotImplementedError(
                f"Megatron Lite slime backend S1 supports --loss-type sft_loss only, "
                f"got {self.args.loss_type!r} (RL losses land in a follow-up)."
            )

        rollout_data = self._get_rollout_data(rollout_data_ref)
        ps = self.handle._parallel_state
        microbatches, num_microbatches = build_sft_microbatches(
            rollout_data,
            micro_batch_size=getattr(self.args, "micro_batch_size", 1) or 1,
            use_dynamic_batch_size=getattr(self.args, "use_dynamic_batch_size", False),
            max_tokens_per_gpu=getattr(self.args, "max_tokens_per_gpu", 0) or 0,
            tp_size=ps.tp_size,
            cp_size=ps.cp_size,
            cp_rank=ps.cp_rank,
            cp_group=ps.cp_group,
            use_fused_kernels=False,
        )
        if num_microbatches == 0:
            logger.warning("rollout %s: empty data shard on dp_rank %s", rollout_id, ps.dp_rank)
            return None

        loss_fn = make_sft_loss_fn()
        with self.runtime.train_mode(self.handle):
            self.runtime.zero_grad(self.handle)
            result = self.runtime.forward_backward(
                self.handle,
                iter(microbatches),
                loss_fn=loss_fn,
                num_microbatches=num_microbatches,
                forward_only=False,
            )
            _, grad_norm, _ = self.runtime.optimizer_step(self.handle)
            lr = self.runtime.lr_scheduler_step(self.handle)

        loss = result.metrics.get("loss")
        logger.info(
            "rollout %s | train/loss %s | grad_norm %.4f | lr %s | num_microbatches %s",
            rollout_id,
            loss,
            float(grad_norm),
            lr,
            num_microbatches,
        )
        return None

    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        if self.args.debug_rollout_only:
            return
        save_dir = getattr(self.args, "save", None)
        if not save_dir:
            return
        self.runtime.save_checkpoint(self.handle, save_dir, step=rollout_id)

    def update_weights(self) -> None:
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return
        raise NotImplementedError(
            "Megatron Lite weight resync to the rollout engine is implemented in a follow-up."
        )

    def sleep(self, *args, **kwargs) -> None:
        if not getattr(self.args, "offload_train", False):
            return
        self.runtime.to(self.handle, "cpu")

    def wake_up(self, *args, **kwargs) -> None:
        if not getattr(self.args, "offload_train", False):
            return
        self.runtime.to(self.handle, "cuda")

    def _get_parallel_config(self):
        return self.train_parallel_config
