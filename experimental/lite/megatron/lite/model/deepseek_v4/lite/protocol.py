"""DeepSeek V4 native lite protocol for Megatron Lite runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
from megatron.lite.model.deepseek_v4.lite.checkpoint import (
    EXPERT_CLASSIFIER,
)
from megatron.lite.model.deepseek_v4.lite.checkpoint import (
    load_hf_weights as _load_hf_weights_impl,
)
from megatron.lite.primitive.bundle import ModelBundle
from megatron.lite.primitive.parallel import ParallelState, init_parallel
from megatron.lite.runtime.contracts import ParallelConfig


def is_expert_param(name: str) -> bool:
    return EXPERT_CLASSIFIER(name)


@dataclass(frozen=True)
class ImplConfig:
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    optimizer: str | None = None
    hf_path: str = ""
    deterministic: bool = True


def build_model_config(source: str | Path | dict, **overrides) -> DeepseekV4Config:
    if isinstance(source, dict):
        cfg = DeepseekV4Config._from_hf_dict(source)
    else:
        cfg = DeepseekV4Config.from_hf(str(source))
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def _forward_step(model: nn.Module, batch: dict) -> dict:
    kwargs: dict[str, Any] = {
        "input_ids": batch["input_ids"],
        "labels": batch.get("labels"),
    }
    for key in ("position_ids", "attention_mask"):
        if key in batch:
            kwargs[key] = batch[key]
    if kwargs["input_ids"].dim() == 1:
        kwargs["input_ids"] = kwargs["input_ids"].unsqueeze(0)
    return model(**kwargs)


def _validate_parallel_scope(p: ParallelConfig) -> None:
    etp = 1 if p.etp is None else p.etp
    if p.cp < 1:
        raise ValueError(f"DeepSeek V4 native lite requires cp>=1, got {p.cp}.")
    if (p.tp, etp, p.ep, p.pp, p.vpp) != (1, 1, 1, 1, 1):
        raise NotImplementedError(
            "DeepSeek V4 native lite currently supports CP with TP=EP=ETP=PP=VPP=1."
        )


def build_model(model_cfg: DeepseekV4Config, *, impl_cfg: ImplConfig) -> ModelBundle:
    from megatron.lite.model.deepseek_v4.lite.model import DeepseekV4ForCausalLM

    _validate_parallel_scope(impl_cfg.parallel)
    ps = init_parallel(impl_cfg.parallel)
    train_cfg = SimpleNamespace(fp8=False)
    chunk = DeepseekV4ForCausalLM(model_cfg, train_cfg=train_cfg, ps=ps).to(torch.bfloat16).cuda()

    if impl_cfg.optimizer is not None:
        raise NotImplementedError("DeepSeek V4 native lite stage-1 does not build an optimizer yet.")

    return ModelBundle(
        chunks=[chunk],
        parallel_state=ps,
        optimizer=None,
        forward_step=_forward_step,
        extras={"model_cfg": model_cfg, "optimizer_backend": "none"},
    )


def load_hf_weights(chunk: nn.Module, hf_path: str, model_cfg: DeepseekV4Config, ps: ParallelState) -> None:
    if not hf_path:
        return
    _load_hf_weights_impl(chunk, hf_path, model_cfg, ps)


def vocab_size(model_cfg: DeepseekV4Config) -> int | None:
    return model_cfg.vocab_size
