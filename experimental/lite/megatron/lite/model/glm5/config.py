"""GLM-5 architecture config."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import fields as dc_fields
from typing import Any

from megatron.lite.primitive.config import load_hf_config_dict

_HF_FIELDS = frozenset(
    {
        "attention_dropout",
        "calculate_per_token_loss",
        "dsa_indexer_loss_coeff",
        "dsa_indexer_use_sparse_loss",
        "first_k_dense_replace",
        "head_dim",
        "hidden_size",
        "index_head_dim",
        "index_n_heads",
        "index_skip_topk_offset",
        "index_topk",
        "index_topk_freq",
        "index_topk_pattern",
        "indexer_types",
        "indexer_layer_norm_eps",
        "indexer_rope_interleave",
        "indexer_rope_first",
        "indexer_use_hadamard",
        "initializer_range",
        "intermediate_size",
        "kv_lora_rank",
        "max_position_embeddings",
        "mlp_layer_types",
        "moe_intermediate_size",
        "n_group",
        "n_routed_experts",
        "n_shared_experts",
        "norm_topk_prob",
        "mtp_loss_scaling_factor",
        "mtp_use_repeated_layer",
        "num_attention_heads",
        "num_experts_per_tok",
        "num_hidden_layers",
        "num_key_value_heads",
        "num_nextn_predict_layers",
        "q_lora_rank",
        "qk_head_dim",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "latent_rms_norm_eps",
        "rms_norm_eps",
        "rope_interleave",
        "rope_theta",
        "routed_scaling_factor",
        "topk_group",
        "v_head_dim",
        "vocab_size",
    }
)
_SERIALIZED_INTERNAL_FIELDS = frozenset({"dsa_rope_layout_revision"})

# ``index_share_for_mtp_iteration`` is intentionally absent.  It is a serving
# proposer control (draft step 0 computes; later draft steps reuse), not a
# backbone architecture field.  MLite does not own that speculative iteration
# loop, so claiming support here would silently turn metadata into a no-op.


_INDEXER_TYPES = frozenset({"full", "shared"})
_INDEXER_PATTERN_TYPES = {
    "F": "full",
    "S": "shared",
    "full": "full",
    "shared": "shared",
}


def _infer_dsa_indexer_type(layer_number: int, *, topk_freq: int, skip_topk_offset: int) -> str:
    if topk_freq <= 1:
        return "full"
    skip_topk = (max(layer_number - skip_topk_offset, 0) % topk_freq) != 0
    return "shared" if skip_topk else "full"


def _normalize_dsa_indexer_pattern(pattern: str | list[str]) -> tuple[str, ...]:
    values = list(pattern)
    normalized: list[str] = []
    for idx, value in enumerate(values):
        try:
            normalized.append(_INDEXER_PATTERN_TYPES[value])
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "index_topk_pattern entries must be 'F', 'S', 'full', or 'shared'; "
                f"got index_topk_pattern[{idx}]={value!r}"
            ) from exc
    return tuple(normalized)


@dataclass
class Glm5Config:
    """Pure GLM-5 MoE + MLA + DSA architecture parameters."""

    num_hidden_layers: int = 78
    hidden_size: int = 6144
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    head_dim: int = 64
    vocab_size: int = 154880
    max_position_embeddings: int = 202752
    rms_norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    initializer_range: float = 0.02

    q_lora_rank: int = 2048
    kv_lora_rank: int = 512
    qk_head_dim: int = 256
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256

    index_head_dim: int = 128
    index_n_heads: int = 32
    index_topk: int = 2048
    indexer_layer_norm_eps: float = 1e-6
    indexer_rope_interleave: bool = False
    indexer_rope_first: bool = True
    indexer_use_hadamard: bool = False
    dsa_indexer_loss_coeff: float = 0.0
    dsa_indexer_use_sparse_loss: bool = False
    calculate_per_token_loss: bool = False
    rope_interleave: bool = False
    rope_theta: float = 1_000_000.0
    latent_rms_norm_eps: float = 1e-6

    intermediate_size: int = 12288
    moe_intermediate_size: int = 2048
    first_k_dense_replace: int = 3
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    n_group: int = 1
    topk_group: int = 1
    routed_scaling_factor: float = 2.5
    norm_topk_prob: bool = True
    num_nextn_predict_layers: int = 1
    mtp_loss_scaling_factor: float = 0.1
    mtp_use_repeated_layer: bool = False
    mlp_layer_types: list[str] | None = None

    # GLM-5.2 extension fields stay at the end so positional construction of
    # every pre-existing Glm5Config field keeps its historical slot.
    index_topk_freq: int = 1
    index_skip_topk_offset: int = 0
    index_topk_pattern: str | list[str] | None = None
    indexer_types: list[str] | None = None
    # Internal compatibility revision. ``None`` infers configured RoPE only
    # for configs carrying GLM-5.2 IndexShare schedule metadata. Persisting the
    # resolved value prevents a later all-full override from silently changing
    # the checkpoint's rotary convention.
    dsa_rope_layout_revision: str | None = None

    def __post_init__(self):
        if self.dsa_rope_layout_revision is None:
            self.dsa_rope_layout_revision = (
                "configured" if self.has_dsa_index_share_schedule else "legacy"
            )
        self._validate()

    @property
    def num_experts(self) -> int:
        return self.n_routed_experts

    def is_moe_layer(self, layer_idx: int) -> bool:
        if self.mlp_layer_types is not None and layer_idx < len(self.mlp_layer_types):
            return self.mlp_layer_types[layer_idx] == "sparse"
        return layer_idx >= self.first_k_dense_replace

    @property
    def uses_dsa_index_share(self) -> bool:
        return "shared" in self.resolved_dsa_indexer_types

    @property
    def has_dsa_index_share_schedule(self) -> bool:
        """Whether the current config carries GLM-5.2-style schedule metadata.

        GLM-5/5.1 checkpoints expose RoPE interleave metadata too, but MLite's
        pre-5.2 implementation historically used half-split RoPE. This metadata
        infers the initial layout revision, even for an explicit all-full
        schedule; the resolved revision itself remains stable after mutation.
        """

        return (
            self.indexer_types is not None
            or self.index_topk_pattern is not None
            or self.index_topk_freq > 1
            or self.index_skip_topk_offset > 0
        )

    @property
    def uses_configured_dsa_rope_layout(self) -> bool:
        return self.dsa_rope_layout_revision == "configured"

    @property
    def resolved_dsa_indexer_types(self) -> tuple[str, ...]:
        """Canonical per-backbone-layer IndexShare schedule.

        HF's explicit ``indexer_types`` is authoritative.  The older
        ``index_topk_pattern`` is the next-priority source, and freq/offset is
        only a fallback when neither explicit representation is present.
        Appended MTP layers are deliberately excluded from this tuple because
        they always build full indexers.
        """

        if self.indexer_types is not None:
            return tuple(self.indexer_types)
        if self.index_topk_pattern is not None:
            return _normalize_dsa_indexer_pattern(self.index_topk_pattern)
        return tuple(
            _infer_dsa_indexer_type(
                layer_idx + 1,
                topk_freq=self.index_topk_freq,
                skip_topk_offset=self.index_skip_topk_offset,
            )
            for layer_idx in range(self.num_hidden_layers)
        )

    def dsa_indexer_type(self, layer_idx: int) -> str:
        if layer_idx < 0:
            raise ValueError(f"layer_idx must be non-negative, got {layer_idx}")
        layer_count = self.num_hidden_layers + self.num_nextn_predict_layers
        if layer_idx >= layer_count:
            raise ValueError(
                f"layer_idx must be less than total backbone + MTP layers ({layer_count}), "
                f"got {layer_idx}"
            )
        if layer_idx >= self.num_hidden_layers:
            return "full"
        return self.resolved_dsa_indexer_types[layer_idx]

    def builds_dsa_indexer(self, layer_idx: int) -> bool:
        return self.dsa_indexer_type(layer_idx) == "full"

    def dsa_indexer_source_layer(self, layer_idx: int) -> int:
        if self.dsa_indexer_type(layer_idx) == "full":
            return layer_idx
        for source_idx in range(layer_idx - 1, -1, -1):
            if self.dsa_indexer_type(source_idx) == "full":
                return source_idx
        raise ValueError(
            "DSA IndexShare schedule makes layer "
            f"{layer_idx} shared before any full source layer."
        )

    def dsa_index_share_decoder_layer_groups(self) -> list[list[int]] | None:
        """Return indivisible PP groups from the canonical backbone schedule."""

        if not self.uses_dsa_index_share:
            return None
        groups: list[list[int]] = []
        current: list[int] = []
        current_source: int | None = None
        for layer_idx in range(self.num_hidden_layers):
            source_idx = self.dsa_indexer_source_layer(layer_idx)
            if current and source_idx != current_source:
                groups.append(current)
                current = []
            current.append(layer_idx)
            current_source = source_idx
        if current:
            groups.append(current)
        return groups

    def _validate(self) -> None:
        errors: list[str] = []

        def check(cond: bool, message: str) -> None:
            if not cond:
                errors.append(message)

        check(self.num_hidden_layers >= 1, "num_hidden_layers must be >= 1")
        check(self.hidden_size > 0, "hidden_size must be > 0")
        check(self.q_lora_rank > 0, "q_lora_rank must be > 0")
        check(self.kv_lora_rank > 0, "kv_lora_rank must be > 0")
        check(
            self.qk_head_dim == self.qk_nope_head_dim + self.qk_rope_head_dim,
            "qk_head_dim must equal qk_nope_head_dim + qk_rope_head_dim",
        )
        check(
            self.index_head_dim >= self.qk_rope_head_dim,
            "index_head_dim must be >= qk_rope_head_dim",
        )
        check(self.index_topk_freq >= 1, "index_topk_freq must be >= 1")
        check(
            self.index_skip_topk_offset >= 0,
            "index_skip_topk_offset must be >= 0",
        )
        check(
            self.dsa_rope_layout_revision in {"legacy", "configured"},
            "dsa_rope_layout_revision must be 'legacy' or 'configured'",
        )
        check(self.dsa_indexer_loss_coeff >= 0.0, "dsa_indexer_loss_coeff must be >= 0")
        check(
            self.num_key_value_heads == self.num_attention_heads,
            "initial GLM5 native path expects MLA heads to be ungrouped",
        )
        check(self.vocab_size > 0, "vocab_size must be > 0")
        check(self.num_nextn_predict_layers >= 0, "num_nextn_predict_layers must be >= 0")
        check(self.n_routed_experts >= 1, "n_routed_experts must be >= 1")
        check(
            1 <= self.num_experts_per_tok <= self.n_routed_experts,
            "num_experts_per_tok must be in [1, n_routed_experts]",
        )
        check(1 <= self.topk_group <= self.n_group, "topk_group must be in [1, n_group]")
        if self.mlp_layer_types is not None:
            expected_layer_type_lengths = {
                self.num_hidden_layers,
                self.num_hidden_layers + self.num_nextn_predict_layers,
            }
            check(
                len(self.mlp_layer_types) in expected_layer_type_lengths,
                "len(mlp_layer_types) must equal num_hidden_layers or "
                "num_hidden_layers + num_nextn_predict_layers",
            )
            for idx, layer_type in enumerate(self.mlp_layer_types):
                check(
                    layer_type in {"dense", "sparse"},
                    f"mlp_layer_types[{idx}] must be 'dense' or 'sparse'",
                )

        resolved_indexer_types: tuple[str, ...] | None = None
        if self.indexer_types is not None:
            check(
                len(self.indexer_types) == self.num_hidden_layers,
                "len(indexer_types) must equal num_hidden_layers; MTP layers always "
                "build full indexers and must not appear in indexer_types",
            )
            for idx, indexer_type in enumerate(self.indexer_types):
                check(
                    indexer_type in _INDEXER_TYPES,
                    f"indexer_types[{idx}] must be 'full' or 'shared'",
                )
            if len(self.indexer_types) == self.num_hidden_layers and all(
                value in _INDEXER_TYPES for value in self.indexer_types
            ):
                resolved_indexer_types = tuple(self.indexer_types)
        elif self.index_topk_pattern is not None:
            try:
                resolved_indexer_types = _normalize_dsa_indexer_pattern(
                    self.index_topk_pattern
                )
            except ValueError as exc:
                check(False, str(exc))
            if resolved_indexer_types is not None:
                check(
                    len(resolved_indexer_types) == self.num_hidden_layers,
                    "len(index_topk_pattern) must equal num_hidden_layers",
                )
        elif self.index_topk_freq >= 1 and self.index_skip_topk_offset >= 0:
            resolved_indexer_types = tuple(
                _infer_dsa_indexer_type(
                    layer_idx + 1,
                    topk_freq=self.index_topk_freq,
                    skip_topk_offset=self.index_skip_topk_offset,
                )
                for layer_idx in range(self.num_hidden_layers)
            )

        if (
            resolved_indexer_types is not None
            and len(resolved_indexer_types) == self.num_hidden_layers
        ):
            first_full_idx: int | None = None
            for idx, indexer_type in enumerate(resolved_indexer_types):
                if indexer_type == "full":
                    first_full_idx = idx
                elif first_full_idx is None:
                    check(
                        False,
                        "DSA IndexShare schedule makes layer "
                        f"{idx} shared before any full source layer",
                    )

        if errors:
            raise ValueError(
                f"Invalid Glm5Config ({len(errors)} error"
                f"{'s' if len(errors) != 1 else ''}):\n  " + "\n  ".join(errors)
            )

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in dc_fields(self)}

    @classmethod
    def from_hf(cls, path_or_name: str, **overrides) -> Glm5Config:
        return cls._from_hf_dict(load_hf_config_dict(path_or_name), **overrides)

    @classmethod
    def from_hf_config(cls, hf_config, **overrides) -> Glm5Config:
        hf = hf_config.to_dict() if hasattr(hf_config, "to_dict") else vars(hf_config)
        return cls._from_hf_dict(hf, **overrides)

    @classmethod
    def _from_hf_dict(cls, hf: dict[str, Any], **overrides) -> Glm5Config:
        kwargs = {
            key: value
            for key, value in hf.items()
            if key in (_HF_FIELDS | _SERIALIZED_INTERNAL_FIELDS) and value is not None
        }
        # Resolve the architecture revision from the source checkpoint before
        # applying local schedule overrides. Turning a real 5.2 config into an
        # all-full schedule must not also reinterpret its rotary layout, while
        # adding an all-full schedule to a 5.1 config must not silently opt in.
        if "dsa_rope_layout_revision" not in kwargs:
            source_has_index_share_schedule = (
                hf.get("indexer_types") is not None
                or hf.get("index_topk_pattern") is not None
                or int(hf.get("index_topk_freq") or 1) > 1
                or int(hf.get("index_skip_topk_offset") or 0) > 0
            )
            kwargs["dsa_rope_layout_revision"] = (
                "configured" if source_has_index_share_schedule else "legacy"
            )
        if "num_nextn_predict_layers" not in kwargs and hf.get("num_nextn_predict") is not None:
            kwargs["num_nextn_predict_layers"] = int(hf["num_nextn_predict"])
        rope_parameters = hf.get("rope_parameters")
        if isinstance(rope_parameters, dict) and "rope_theta" not in kwargs:
            kwargs["rope_theta"] = float(rope_parameters.get("rope_theta", cls.rope_theta))
        kwargs.update(overrides)
        return cls(**kwargs)
