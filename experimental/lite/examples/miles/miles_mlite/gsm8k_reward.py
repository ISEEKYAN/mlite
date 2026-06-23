# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""GSM8K rule-based reward for the miles MLite smoke."""

from __future__ import annotations

import re
import logging

from miles.rollout.rm_hub.math_utils import extract_answer, grade_answer_mathd, grade_answer_sympy
from miles.utils.metric_utils import compute_pass_rate
from miles.utils.types import Sample

logger = logging.getLogger(__name__)
_GSM8K_ANSWER_RE = re.compile(r"####\s*([^\n<]+)")


def _clean_answer(answer) -> str:
    text = str(answer).strip()
    text = text.removeprefix("$").strip()
    text = text.rstrip(".。").strip()
    text = text.replace(",", "")
    return text


def _extract_gsm8k_answer(response: str) -> str | None:
    matches = _GSM8K_ANSWER_RE.findall(response)
    if matches:
        return _clean_answer(matches[-1])
    boxed = extract_answer(response)
    if boxed is not None:
        return _clean_answer(boxed)
    return None


def _score(sample: Sample) -> float:
    answer = _extract_gsm8k_answer(sample.response)
    if answer is None or sample.label is None:
        return 0.0
    label = _clean_answer(sample.label)
    return 1.0 if grade_answer_mathd(answer, label) or grade_answer_sympy(answer, label) else 0.0


async def reward_func(args, samples: Sample | list[Sample], **kwargs) -> float | list[float]:
    if isinstance(samples, list):
        return [_score(sample) for sample in samples]
    return _score(samples)


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    rewards = [float(sample.get_reward_value(args)) for sample in samples]
    passrate = compute_pass_rate(rewards, group_size=args.n_samples_per_prompt, num_groups=args.rollout_batch_size)
    reward_mean = sum(rewards) / len(rewards) if rewards else 0.0
    nonzero = sum(1 for reward in rewards if reward != 0.0)
    metrics = {
        "reward/score/mean": reward_mean,
        "reward/score/nonzero": nonzero,
        "reward/score/count": len(rewards),
    }
    metrics.update({f"reward/passrate/{key}": value for key, value in passrate.items()})
    logger.info("reward/score/passrate rollout %s: %s", rollout_id, metrics)
    return False
