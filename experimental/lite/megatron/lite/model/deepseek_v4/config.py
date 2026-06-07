"""DeepSeek V4 model configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from megatron.lite.primitive.config import load_hf_config_dict

_HF_FIELDS = frozenset({
    "attention_dropout",
    "compress_ratios",
    "compress_rope_theta",
    "hc_eps",
    "hc_mult",
    "hc_sinkhorn_iters",
    "head_dim",
    "hidden_act",
    "hidden_size",
    "index_head_dim",
    "index_n_heads",
    "index_topk",
    "initializer_range",
    "max_position_embeddings",
    "moe_intermediate_size",
    "n_routed_experts",
    "n_shared_experts",
    "norm_topk_prob",
    "num_attention_heads",
    "num_experts_per_tok",
    "num_hash_layers",
    "num_hidden_layers",
    "num_key_value_heads",
    "num_nextn_predict_layers",
    "o_groups",
    "o_lora_rank",
    "q_lora_rank",
    "qk_rope_head_dim",
    "rms_norm_eps",
    "rope_theta",
    "routed_scaling_factor",
    "scoring_func",
    "sliding_window",
    "swiglu_limit",
    "topk_method",
    "vocab_size",
})


@dataclass
class DeepseekV4Config:
    """Pure DeepSeek V4 architecture parameters used by the native lite path."""

    vocab_size: int = 129280
    hidden_size: int = 4096
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 43
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    head_dim: int = 128
    qk_rope_head_dim: int = 64
    q_lora_rank: int = 1024
    o_lora_rank: int = 1024
    o_groups: int = 8
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 6
    routed_scaling_factor: float = 1.5
    norm_topk_prob: bool = True
    scoring_func: str = "sqrtsoftplus"
    topk_method: str = "noaux_tc"
    hidden_act: str = "silu"
    swiglu_limit: float = 10.0
    max_position_embeddings: int = 1_048_576
    rope_theta: float = 10_000.0
    compress_rope_theta: float = 160_000.0
    compress_ratios: list[int] = field(default_factory=list)
    sliding_window: int = 128
    num_hash_layers: int = 3
    hc_eps: float = 1e-6
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    index_head_dim: int = 128
    index_n_heads: int = 64
    index_topk: int = 512
    num_nextn_predict_layers: int = 1
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        errors: list[str] = []

        def check(cond: bool, message: str) -> None:
            if not cond:
                errors.append(message)

        check(self.num_hidden_layers >= 1, "num_hidden_layers must be >= 1")
        check(self.hidden_size > 0, "hidden_size must be > 0")
        check(self.num_attention_heads >= 1, "num_attention_heads must be >= 1")
        check(self.num_key_value_heads == 1, "stage-1 native DeepSeek V4 expects num_key_value_heads == 1")
        check(self.num_attention_heads % self.o_groups == 0, "num_attention_heads must be divisible by o_groups")
        check(self.head_dim > 0, "head_dim must be > 0")
        check(0 < self.qk_rope_head_dim <= self.head_dim, "qk_rope_head_dim must be in (0, head_dim]")
        check(self.q_lora_rank > 0, "q_lora_rank must be > 0")
        check(self.o_lora_rank > 0, "o_lora_rank must be > 0")
        check(self.n_routed_experts >= 1, "n_routed_experts must be >= 1")
        check(
            1 <= self.num_experts_per_tok <= self.n_routed_experts,
            "num_experts_per_tok must be in [1, n_routed_experts]",
        )
        check(self.moe_intermediate_size > 0, "moe_intermediate_size must be > 0")
        check(self.hc_mult >= 1, "hc_mult must be >= 1")
        check(self.vocab_size > 0, "vocab_size must be > 0")
        check(self.scoring_func in {"sqrtsoftplus", "sigmoid", "softmax"}, "unsupported scoring_func")
        check(self.topk_method in {"noaux_tc", "greedy"}, "unsupported topk_method")
        if self.compress_ratios:
            expected_ratio_lengths = {
                self.num_hidden_layers,
                self.num_hidden_layers + self.num_nextn_predict_layers,
            }
            check(
                len(self.compress_ratios) in expected_ratio_lengths,
                "len(compress_ratios) must equal num_hidden_layers or "
                "num_hidden_layers + num_nextn_predict_layers",
            )

        if errors:
            raise ValueError(
                f"Invalid DeepseekV4Config ({len(errors)} error"
                f"{'s' if len(errors) != 1 else ''}):\n  " + "\n  ".join(errors)
            )

    @classmethod
    def from_hf(cls, path: str, **overrides) -> DeepseekV4Config:
        return cls._from_hf_dict(load_hf_config_dict(path), **overrides)

    @classmethod
    def from_hf_config(cls, hf_config, **overrides) -> DeepseekV4Config:
        data = hf_config.to_dict() if hasattr(hf_config, "to_dict") else vars(hf_config)
        return cls._from_hf_dict(data, **overrides)

    @classmethod
    def _from_hf_dict(cls, hf: dict[str, Any], **overrides) -> DeepseekV4Config:
        kwargs = {key: value for key, value in hf.items() if key in _HF_FIELDS and value is not None}
        rope_parameters = hf.get("rope_parameters")
        if isinstance(rope_parameters, dict):
            if "rope_theta" not in kwargs:
                kwargs["rope_theta"] = float(rope_parameters.get("rope_theta", cls.rope_theta))
        kwargs.update(overrides)
        return cls(**kwargs)
