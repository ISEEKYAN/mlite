# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Monkeypatch slime-family Megatron train actors to use Megatron Lite.

The supported family members (slime, miles, vime) choose their train actor by a
function-local import of ``<pkg>.backends.megatron_utils.actor.MegatronTrainRayActor``.
Replacing that source symbol before actor-group construction lets examples use
the existing ``--train-backend megatron`` slot without changing fork code or CLI
choices.
"""

from __future__ import annotations

import importlib
import logging
import sys
from argparse import Namespace
from types import ModuleType, SimpleNamespace
from typing import Any

import torch

from .arguments import optimizer_backend_to_impl, validate_mlite_args
from .data import build_runtime_microbatches
from .loss import make_runtime_loss_fn
from .weight_update import RawHFWeightUpdater

logger = logging.getLogger(__name__)

SUPPORTED_PACKAGES = ("slime", "miles", "vime")
MLiteTrainRayActor = None


def _group(rank: int, size: int, group):
    return SimpleNamespace(rank=rank, size=size, group=group, gloo_group=None)


def _install_family_parallel_state(pkg: str, ps) -> None:
    try:
        parallel_mod = importlib.import_module(f"{pkg}.backends.training_utils.parallel")
    except ImportError:
        return
    state = parallel_mod.ParallelState(
        intra_dp=_group(ps.dp_rank, ps.dp_size, ps.dp_group),
        intra_dp_cp=_group(getattr(ps, "dp_cp_rank", ps.dp_rank), getattr(ps, "dp_cp_size", ps.dp_size), getattr(ps, "dp_cp_group", ps.dp_group)),
        cp=_group(ps.cp_rank, ps.cp_size, ps.cp_group),
        tp=_group(ps.tp_rank, ps.tp_size, ps.tp_group),
        pp=_group(ps.pp_rank, ps.pp_size, ps.pp_group),
        ep=_group(getattr(ps, "ep_rank", 0), getattr(ps, "ep_size", 1), getattr(ps, "ep_group", None)),
        etp=_group(getattr(ps, "etp_rank", 0), getattr(ps, "etp_size", 1), getattr(ps, "etp_group", None)),
        cp_comm_type=getattr(ps, "cp_comm_type", None),
        is_pp_last_stage=ps.pp_rank == ps.pp_size - 1,
        vpp_size=1,
        microbatch_group_size_per_vp_stage=1,
    )
    parallel_mod.set_parallel_state(state)


class _MLiteTrainRayActorMixin:
    _slime_family_pkg: str

    def _build_mlite_config(self, args: Namespace):
        from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
        from megatron.lite.runtime.contracts.config import OptimizerConfig, ParallelConfig

        validate_mlite_args(args)
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
        super().init(args, role, with_ref, with_opd_teacher=with_opd_teacher)
        if role != "actor":
            raise NotImplementedError("Megatron Lite slime-family backend supports actor training only.")
        if with_ref or with_opd_teacher:
            raise NotImplementedError("Reference/teacher model swapping is not implemented for the MLite patch.")

        if args.debug_rollout_only:
            self.args = args
            return 0

        from megatron.lite.runtime import RuntimeConfig, create_runtime

        self._cfg = self._build_mlite_config(args)
        self.runtime = create_runtime(
            RuntimeConfig(backend="mlite", hf_path=args.hf_checkpoint, backend_cfg=self._cfg)
        )
        self.handle = self.runtime.build_model()

        ps = self.handle._parallel_state
        _install_family_parallel_state(self._slime_family_pkg, ps)
        self.train_parallel_config = {
            "dp_size": ps.dp_size,
            "cp_size": ps.cp_size,
            "vpp_size": self._cfg.parallel.vpp or 1,
            "microbatch_group_size_per_vp_stage": 1,
        }
        self.weight_updater = RawHFWeightUpdater(
            args, self.runtime, self.handle, family_pkg=self._slime_family_pkg
        )

        start_rollout_id = 0
        if getattr(args, "load", None):
            loaded = self.runtime.load_checkpoint(self.handle, args.load)
            start_rollout_id = int(loaded) + 1

        if getattr(args, "offload_train", False) or getattr(args, "mlite_param_offload", False):
            self.sleep()
        return start_rollout_id

    def _process_rollout_data(self, rollout_data_ref):
        data_mod = importlib.import_module(f"{self._slime_family_pkg}.utils.data")
        ps = self.handle._parallel_state
        rollout_data = data_mod.process_rollout_data(self.args, rollout_data_ref, ps.dp_rank, ps.dp_size)
        rollout_data["tokens"] = [torch.as_tensor(t, dtype=torch.long) for t in rollout_data["tokens"]]
        rollout_data["loss_masks"] = [torch.as_tensor(t, dtype=torch.float32) for t in rollout_data["loss_masks"]]
        device = torch.device("cuda", torch.cuda.current_device())
        for key in ("rollout_log_probs", "log_probs", "ref_log_probs", "advantages", "returns"):
            if key in rollout_data and rollout_data[key] is not None:
                rollout_data[key] = [
                    torch.as_tensor(t, dtype=torch.float32, device=device).reshape(-1)
                    for t in rollout_data[key]
                ]
        return rollout_data

    def _compute_advantages_and_returns(self, rollout_data) -> None:
        loss_mod = importlib.import_module(f"{self._slime_family_pkg}.backends.training_utils.loss")
        loss_mod.compute_advantages_and_returns(self.args, rollout_data)

    def _build_microbatches(self, rollout_data, *, calculate_entropy: bool = False):
        return build_runtime_microbatches(
            rollout_data,
            micro_batch_size=getattr(self.args, "micro_batch_size", 1) or 1,
            use_dynamic_batch_size=getattr(self.args, "use_dynamic_batch_size", False),
            max_tokens_per_gpu=getattr(self.args, "max_tokens_per_gpu", 0) or 0,
            calculate_entropy=calculate_entropy,
            temperature=float(getattr(self.args, "rollout_temperature", 1.0) or 1.0),
        )

    def _compute_log_probs(self, rollout_data) -> list[torch.Tensor]:
        microbatches = self._build_microbatches(
            rollout_data,
            calculate_entropy=bool(getattr(self.args, "use_rollout_entropy", False)),
        )
        if not microbatches:
            return []
        store: list[dict[str, list[torch.Tensor]]] = []
        with self.runtime.eval_mode(self.handle):
            self.runtime.forward_backward(
                self.handle,
                (mb.as_runtime_item() for mb in microbatches),
                loss_fn=make_runtime_loss_fn(self.args, self.handle, forward_store=store),
                num_microbatches=len(microbatches),
                forward_only=True,
            )
        return [item for micro in store for item in micro["log_probs"]]

    def train(self, rollout_id: int, rollout_data_ref) -> None:
        self._last_rollout_id = rollout_id
        if getattr(self.args, "offload_train", False) or getattr(self.args, "mlite_param_offload", False):
            self.wake_up()

        rollout_data = self._process_rollout_data(rollout_data_ref)
        if self.args.debug_rollout_only:
            return None

        loss_type = getattr(self.args, "loss_type", "sft_loss")
        if loss_type == "policy_loss" and getattr(self.args, "compute_advantages_and_returns", True):
            if not getattr(self.args, "use_rollout_logprobs", False) or getattr(self.args, "get_mismatch_metrics", False):
                rollout_data["log_probs"] = self._compute_log_probs(rollout_data)
            elif "rollout_log_probs" not in rollout_data:
                rollout_data["log_probs"] = self._compute_log_probs(rollout_data)
            self._compute_advantages_and_returns(rollout_data)

        microbatches = self._build_microbatches(rollout_data)
        if not microbatches:
            logger.warning("rollout %s: empty data shard", rollout_id)
            return None

        with self.runtime.train_mode(self.handle):
            self.runtime.zero_grad(self.handle)
            result = self.runtime.forward_backward(
                self.handle,
                (mb.as_runtime_item() for mb in microbatches),
                loss_fn=make_runtime_loss_fn(self.args, self.handle),
                num_microbatches=len(microbatches),
                forward_only=False,
            )
            _, grad_norm, _ = self.runtime.optimizer_step(self.handle)
            lr = self.runtime.lr_scheduler_step(self.handle)

        logger.info(
            "rollout %s | train/loss %s | grad_norm %.4f | lr %s | num_microbatches %s",
            rollout_id,
            result.metrics.get("loss"),
            float(grad_norm),
            lr,
            len(microbatches),
        )
        return None

    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        if self.args.debug_rollout_only:
            return
        save_dir = getattr(self.args, "save", None)
        if not save_dir:
            return
        self.runtime.save_checkpoint(self.handle, save_dir, step=rollout_id)

    def update_weights(self, info=None) -> None:
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return
        if info is None:
            raise ValueError("update_weights requires rollout engine info from the slime-family rollout manager.")
        if getattr(self.args, "offload_train", False) or getattr(self.args, "mlite_param_offload", False):
            self.wake_up()

        rollout_engines = info.rollout_engines
        rollout_engine_lock = info.rollout_engine_lock
        has_new_engines = info.has_new_engines
        engine_gpu_counts = getattr(info, "engine_gpu_counts", None)
        engine_gpu_offsets = getattr(info, "engine_gpu_offsets", None)
        del info

        if has_new_engines:
            self.weight_updater.connect_rollout_engines(
                rollout_engines,
                rollout_engine_lock,
                engine_gpu_counts=engine_gpu_counts,
                engine_gpu_offsets=engine_gpu_offsets,
            )
            import ray

            if torch.distributed.get_rank() == 0:
                ray.get(self.rollout_manager.clear_updatable_has_new_engines.remote())

        if getattr(self.args, "debug_skip_weight_update", False):
            logger.warning("Skipping MLite actor-to-rollout weight update because --debug-skip-weight-update is set.")
            return
        self.weight_updater.update_weights()

    def sleep(self, *args, **kwargs) -> None:
        if not (getattr(self.args, "offload_train", False) or getattr(self.args, "mlite_param_offload", False)):
            return
        self.runtime.to(self.handle, "cpu")

    def wake_up(self, *args, **kwargs) -> None:
        if not (getattr(self.args, "offload_train", False) or getattr(self.args, "mlite_param_offload", False)):
            return
        self.runtime.to(self.handle, "cuda")

    def connect_actor_critic(self, critic_group=None, **kwargs):
        raise NotImplementedError("Megatron Lite slime-family backend does not support critic training yet.")

    def _get_parallel_config(self):
        return self.train_parallel_config


def _make_actor_class(pkg: str):
    train_actor_mod = importlib.import_module(f"{pkg}.ray.train_actor")
    base_cls = train_actor_mod.TrainRayActor
    return type(
        "MLiteTrainRayActor",
        (_MLiteTrainRayActorMixin, base_cls),
        {
            "__module__": __name__,
            "__doc__": f"Megatron Lite TrainRayActor patched into {pkg}.",
            "_slime_family_pkg": pkg,
        },
    )


def _load_or_synthesize_actor_module(pkg: str, import_error: ImportError) -> ModuleType:
    module_name = f"{pkg}.backends.megatron_utils.actor"
    actor_mod = ModuleType(module_name)
    actor_mod.__package__ = f"{pkg}.backends.megatron_utils"
    actor_mod.__doc__ = f"Synthetic MLite actor patch module for {pkg}."
    sys.modules[module_name] = actor_mod

    parent_mod = importlib.import_module(actor_mod.__package__)
    setattr(parent_mod, "actor", actor_mod)
    logger.warning("Using synthetic %s because the original module failed to import: %s", module_name, import_error)
    return actor_mod


def patch_slime_family_backends(packages: tuple[str, ...] = SUPPORTED_PACKAGES) -> dict[str, type]:
    """Patch installed slime-family packages and return patched actor classes."""
    global MLiteTrainRayActor

    patched: dict[str, type] = {}
    for pkg in packages:
        try:
            importlib.import_module(pkg)
        except ImportError:
            continue

        actor_cls = _make_actor_class(pkg)
        actor_mod_name = f"{pkg}.backends.megatron_utils.actor"
        try:
            actor_mod = importlib.import_module(actor_mod_name)
        except ImportError as exc:
            actor_mod = _load_or_synthesize_actor_module(pkg, exc)
        actor_mod.MegatronTrainRayActor = actor_cls
        patched[pkg] = actor_cls
        if MLiteTrainRayActor is None:
            MLiteTrainRayActor = actor_cls
        logger.info("Patched %s MegatronTrainRayActor with Megatron Lite.", pkg)
    return patched
