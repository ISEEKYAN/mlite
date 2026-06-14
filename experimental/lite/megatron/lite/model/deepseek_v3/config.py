# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DeepSeek-V3 model configuration."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from megatron.lite.primitive.config import load_hf_config_dict

DEFAULT_DEEPSEEK_V3_VARIANT = "kimi-k2"

_KIMI_K25_FAMILY = {
    "rms_norm_eps": 1e-5,
    "max_position_embeddings": 262144,
    "rope_scaling": {
        "type": "yarn",
        "factor": 64.0,
        "original_max_position_embeddings": 4096,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
    },
}


@dataclass(frozen=True)
class DeepSeekV3VariantConfig:
    name: str
    defaults: dict[str, Any] = field(default_factory=dict)
    hf_model_types: tuple[str, ...] = ()


DEEPSEEK_V3_VARIANTS = {
    "kimi-k2": DeepSeekV3VariantConfig(
        "kimi-k2",
        hf_model_types=("deepseek_v3", "kimi_k2"),
    ),
    "kimi-k2.5": DeepSeekV3VariantConfig(
        "kimi-k2.5",
        defaults=_KIMI_K25_FAMILY,
        hf_model_types=("kimi_k25",),
    ),
    "kimi-k2.6": DeepSeekV3VariantConfig(
        "kimi-k2.6",
        defaults=_KIMI_K25_FAMILY,
        hf_model_types=("kimi_k26",),
    ),
    "kimi-k2.7": DeepSeekV3VariantConfig(
        "kimi-k2.7",
        defaults=_KIMI_K25_FAMILY,
        hf_model_types=("kimi_k27",),
    ),
}

DEEPSEEK_V3_HF_MODEL_TYPES = tuple(
    hf_model_type
    for variant_config in DEEPSEEK_V3_VARIANTS.values()
    for hf_model_type in variant_config.hf_model_types
)

_HF_FIELDS = frozenset(
    {
        "num_hidden_layers",
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
        "rms_norm_eps",
        "max_position_embeddings",
        "intermediate_size",
        "moe_intermediate_size",
        "n_routed_experts",
        "n_shared_experts",
        "num_experts_per_tok",
        "n_group",
        "topk_group",
        "topk_method",
        "norm_topk_prob",
        "scoring_func",
        "seq_aux",
        "first_k_dense_replace",
        "moe_layer_freq",
        "aux_loss_alpha",
        "routed_scaling_factor",
        "q_lora_rank",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "rope_theta",
        "rope_scaling",
        "tie_word_embeddings",
        "num_nextn_predict_layers",
        "mtp_loss_scaling_factor",
        "mtp_use_repeated_layer",
    }
)


def _variant_entry(variant: str) -> DeepSeekV3VariantConfig:
    if variant not in DEEPSEEK_V3_VARIANTS:
        raise ValueError(
            f"Unknown DeepSeekV3 variant: {variant!r}. "
            f"Available variants: {tuple(DEEPSEEK_V3_VARIANTS)}"
        )
    return DEEPSEEK_V3_VARIANTS[variant]


def _variant_defaults(variant: str) -> dict[str, Any]:
    return deepcopy(_variant_entry(variant).defaults)


def _explicit_variant_from_hf(hf: dict) -> str | None:
    variant = hf.get("variant")
    if isinstance(variant, str):
        return variant
    text_config = hf.get("text_config")
    if isinstance(text_config, dict) and isinstance(text_config.get("variant"), str):
        return text_config["variant"]
    return None


@dataclass
class DeepSeekV3Config:
    """Pure architecture parameters for DeepSeekV3 Instruct."""

    variant: str = DEFAULT_DEEPSEEK_V3_VARIANT
    num_hidden_layers: int = 61
    hidden_size: int = 7168
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    vocab_size: int = 163840
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 131072
    intermediate_size: int = 18432
    moe_intermediate_size: int = 2048
    n_routed_experts: int = 384
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    n_group: int | None = 1
    topk_group: int | None = 1
    topk_method: str = "noaux_tc"
    norm_topk_prob: bool = True
    scoring_func: str = "sigmoid"
    seq_aux: bool = True
    first_k_dense_replace: int = 1
    moe_layer_freq: int = 1
    aux_loss_alpha: float = 0.001
    routed_scaling_factor: float = 2.827
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    rope_theta: float = 50_000.0
    rope_scaling: dict = field(
        default_factory=lambda: {
            "type": "yarn",
            "factor": 32.0,
            "original_max_position_embeddings": 4096,
            "beta_fast": 1.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        }
    )
    tie_word_embeddings: bool = False
    num_nextn_predict_layers: int = 0
    mtp_loss_scaling_factor: float = 0.1
    mtp_use_repeated_layer: bool = False

    @property
    def num_experts(self) -> int:
        return self.n_routed_experts

    @property
    def shared_expert_intermediate_size(self) -> int:
        return self.moe_intermediate_size * self.n_shared_experts

    @property
    def q_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    def __post_init__(self) -> None:
        _variant_entry(self.variant)
        errors: list[str] = []

        def _check(cond: bool, msg: str) -> None:
            if not cond:
                errors.append(msg)

        _check(self.num_hidden_layers >= 1, "num_hidden_layers must be >= 1")
        _check(self.hidden_size > 0, "hidden_size must be > 0")
        _check(self.num_attention_heads >= 1, "num_attention_heads must be >= 1")
        _check(
            self.num_attention_heads % self.num_key_value_heads == 0,
            "num_attention_heads must be divisible by num_key_value_heads",
        )
        _check(self.q_lora_rank > 0, "q_lora_rank must be > 0")
        _check(self.kv_lora_rank > 0, "kv_lora_rank must be > 0")
        _check(self.qk_nope_head_dim > 0, "qk_nope_head_dim must be > 0")
        _check(self.qk_rope_head_dim > 0, "qk_rope_head_dim must be > 0")
        _check(self.v_head_dim > 0, "v_head_dim must be > 0")
        _check(self.n_routed_experts >= 1, "n_routed_experts must be >= 1")
        _check(
            1 <= self.num_experts_per_tok <= self.n_routed_experts,
            "num_experts_per_tok must be in [1, n_routed_experts]",
        )
        if self.n_group is not None or self.topk_group is not None:
            _check(
                self.n_group is not None and self.topk_group is not None,
                "n_group and topk_group must be set together",
            )
            if self.n_group is not None and self.topk_group is not None:
                _check(self.n_group >= 1, "n_group must be >= 1")
                _check(1 <= self.topk_group <= self.n_group, "topk_group must be in [1, n_group]")
                _check(
                    self.n_routed_experts % self.n_group == 0,
                    "n_routed_experts must be divisible by n_group",
                )
        _check(0 <= self.first_k_dense_replace <= self.num_hidden_layers, "bad dense prefix")
        _check(self.num_nextn_predict_layers >= 0, "num_nextn_predict_layers must be >= 0")
        if errors:
            raise ValueError("Invalid DeepSeekV3Config:\n  " + "\n  ".join(errors))

    @classmethod
    def from_hf(cls, path: str, **overrides) -> DeepSeekV3Config:
        return cls._from_hf_dict(load_hf_config_dict(path), **overrides)

    @classmethod
    def from_hf_config(cls, hf_config, **overrides) -> DeepSeekV3Config:
        return cls._from_hf_dict(hf_config.to_dict(), **overrides)

    @classmethod
    def _from_hf_dict(cls, hf: dict, **overrides) -> DeepSeekV3Config:
        overrides = dict(overrides)
        if "variant" in overrides:
            variant = overrides.pop("variant")
        else:
            variant = _explicit_variant_from_hf(hf) or DEFAULT_DEEPSEEK_V3_VARIANT
        _variant_entry(variant)
        if "text_config" in hf and isinstance(hf["text_config"], dict):
            hf = hf["text_config"]
        kwargs = _variant_defaults(variant)
        kwargs.update({k: v for k, v in hf.items() if k in _HF_FIELDS})
        if "rope_scaling" in kwargs and kwargs["rope_scaling"] is None:
            kwargs.pop("rope_scaling")
        kwargs["variant"] = variant
        kwargs.update(overrides)
        return cls(**kwargs)

    @classmethod
    def from_variant(cls, variant: str, **overrides) -> DeepSeekV3Config:
        kwargs = _variant_defaults(variant)
        kwargs["variant"] = variant
        kwargs.update(overrides)
        return cls(**kwargs)

    def is_moe_layer(self, layer_idx: int) -> bool:
        if layer_idx < self.first_k_dense_replace:
            return False
        freq = int(self.moe_layer_freq or 1)
        return (layer_idx - self.first_k_dense_replace) % freq == 0


__all__ = [
    "DEFAULT_DEEPSEEK_V3_VARIANT",
    "DEEPSEEK_V3_HF_MODEL_TYPES",
    "DEEPSEEK_V3_VARIANTS",
    "DeepSeekV3Config",
    "DeepSeekV3VariantConfig",
]
