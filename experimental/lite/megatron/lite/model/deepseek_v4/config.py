# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from megatron.lite.primitive.config import load_hf_config_dict


_DSV4_LAYER_TYPE_TO_COMPRESS_RATIO = {
    "sliding_attention": 0,
    "compressed_sparse_attention": 4,
    "heavily_compressed_attention": 128,
}


def _dsv4_compress_ratios(hf: dict[str, Any], *, num_hidden_layers: int, num_mtp_layers: int) -> list[int]:
    expected_len = num_hidden_layers + num_mtp_layers
    if hf.get("compress_ratios") is not None:
        ratios = [int(ratio) for ratio in hf["compress_ratios"]]
    else:
        layer_types = hf.get("layer_types")
        compress_rates = hf.get("compress_rates")
        if layer_types is None or compress_rates is None:
            return []

        ratios = []
        for layer_type in layer_types:
            if layer_type == "sliding_attention":
                ratios.append(0)
            elif layer_type in compress_rates:
                ratios.append(int(compress_rates[layer_type]))
            elif layer_type in _DSV4_LAYER_TYPE_TO_COMPRESS_RATIO:
                ratios.append(_DSV4_LAYER_TYPE_TO_COMPRESS_RATIO[layer_type])
            else:
                raise ValueError(f"Unsupported DeepSeek-V4 attention layer type: {layer_type!r}")

    if len(ratios) == num_hidden_layers and num_mtp_layers:
        ratios.extend([0] * num_mtp_layers)

    if len(ratios) < expected_len:
        raise ValueError(
            f"DeepSeek-V4 compression ratios length ({len(ratios)}) is shorter than "
            f"num_hidden_layers + num_nextn_predict_layers ({expected_len})."
        )
    return ratios[:expected_len]


def _dsv4_num_hash_layers(hf: dict[str, Any]) -> int | None:
    if hf.get("num_hash_layers") is not None:
        return int(hf["num_hash_layers"])

    mlp_layer_types = hf.get("mlp_layer_types")
    if mlp_layer_types is None:
        return None

    n_hash = 0
    for layer_type in mlp_layer_types:
        if layer_type != "hash_moe":
            break
        n_hash += 1

    if any(layer_type == "hash_moe" for layer_type in mlp_layer_types[n_hash:]):
        raise ValueError("DeepSeek-V4 hash MoE layers must be a contiguous prefix.")
    return n_hash


@dataclass
class DeepseekV4Config:
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
    swiglu_limit: float = 10.0
    max_position_embeddings: int = 1_048_576
    rope_theta: float = 10_000.0
    compress_rope_theta: float = 160_000.0
    rotary_scaling_factor: float = 40.0
    original_max_position_embeddings: int = 4096
    beta_fast: float = 32.0
    beta_slow: float = 1.0
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
    mtp_loss_scaling_factor: float = 0.1
    tie_word_embeddings: bool = False
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02

    @property
    def num_experts(self) -> int:
        return self.n_routed_experts

    @classmethod
    def from_hf(cls, path: str, **overrides) -> DeepseekV4Config:
        return cls._from_hf_dict(load_hf_config_dict(path), **overrides)

    @classmethod
    def _from_hf_dict(cls, hf: dict[str, Any], **overrides) -> DeepseekV4Config:
        hf_fields = {item.name for item in fields(cls)}
        kwargs = {key: value for key, value in hf.items() if key in hf_fields and value is not None}
        if "num_nextn_predict_layers" not in kwargs and hf.get("num_nextn_predict") is not None:
            kwargs["num_nextn_predict_layers"] = int(hf["num_nextn_predict"])
        if "num_hash_layers" not in kwargs:
            num_hash_layers = _dsv4_num_hash_layers(hf)
            if num_hash_layers is not None:
                kwargs["num_hash_layers"] = num_hash_layers
        if "compress_ratios" not in kwargs:
            num_hidden_layers = int(kwargs.get("num_hidden_layers", cls.num_hidden_layers))
            num_mtp_layers = int(kwargs.get("num_nextn_predict_layers", cls.num_nextn_predict_layers) or 0)
            compress_ratios = _dsv4_compress_ratios(
                hf, num_hidden_layers=num_hidden_layers, num_mtp_layers=num_mtp_layers
            )
            if compress_ratios:
                kwargs["compress_ratios"] = compress_ratios
        if kwargs.get("tie_word_embeddings"):
            raise ValueError("DeepSeek-V4 MLite expects untied embeddings (tie_word_embeddings=False).")
        rope_scaling = hf.get("rope_scaling")
        if isinstance(rope_scaling, dict):
            if rope_scaling.get("factor") is not None:
                kwargs["rotary_scaling_factor"] = float(rope_scaling["factor"])
            if rope_scaling.get("original_max_position_embeddings") is not None:
                kwargs["original_max_position_embeddings"] = int(
                    rope_scaling["original_max_position_embeddings"]
                )
            if rope_scaling.get("beta_fast") is not None:
                kwargs["beta_fast"] = float(rope_scaling["beta_fast"])
            if rope_scaling.get("beta_slow") is not None:
                kwargs["beta_slow"] = float(rope_scaling["beta_slow"])
        rope_parameters = hf.get("rope_parameters")
        if isinstance(rope_parameters, dict):
            main_rope = rope_parameters.get("main")
            if isinstance(main_rope, dict):
                if main_rope.get("rope_theta") is not None:
                    kwargs["rope_theta"] = float(main_rope["rope_theta"])
            elif rope_parameters.get("rope_theta") is not None:
                kwargs["rope_theta"] = float(rope_parameters["rope_theta"])

            compress_rope = rope_parameters.get("compress")
            if isinstance(compress_rope, dict):
                if compress_rope.get("rope_theta") is not None:
                    kwargs["compress_rope_theta"] = float(compress_rope["rope_theta"])
                if compress_rope.get("factor") is not None:
                    kwargs["rotary_scaling_factor"] = float(compress_rope["factor"])
                if compress_rope.get("original_max_position_embeddings") is not None:
                    kwargs["original_max_position_embeddings"] = int(
                        compress_rope["original_max_position_embeddings"]
                    )
                if compress_rope.get("beta_fast") is not None:
                    kwargs["beta_fast"] = float(compress_rope["beta_fast"])
                if compress_rope.get("beta_slow") is not None:
                    kwargs["beta_slow"] = float(compress_rope["beta_slow"])
        kwargs.update(overrides)
        return cls(**kwargs)
