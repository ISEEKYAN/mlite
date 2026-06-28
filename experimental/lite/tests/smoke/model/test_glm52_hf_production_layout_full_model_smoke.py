# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""ADAPTED Transformers 5.12 full-model check for GLM-5.2 RoPE layout.

Transformers 5.12 already uses its interleaved helper for the main MLA path,
but ``GlmMoeDsaIndexer.forward`` calls the half-split helper unconditionally
and ignores the published ``indexer_rope_interleave=true`` checkpoint field.
The reference below applies one deliberately narrow, instance-local adapter:
only the full layer's indexer is replaced by a test-local subclass whose
forward calls HF's public interleaved helper. The HF modeling module and its
globals are never modified; the main attention executes the unmodified path.

The test first proves that vanilla HF and the configured layout choose
different top-k rows and produce different logits.  It then compares the
test-locally adapted ``GlmMoeDsaForCausalLM`` with the actual MLite
``Glm5Model`` through a complete two-layer F/S IndexShare model: dense MLP,
MoE, per-layer hidden states, logits, fixed external-label shifted CE, and the
gradient at the embedding output. Optional production kernels are replaced
with explicit Torch implementations so this supporting comparison is
deterministic and CPU-only. Release semantics are gated independently by the
pinned release/vLLM score-and-top-k oracle.

This is not unmodified-Transformers parity, a production sparse-kernel test, or
coverage for MTP, CP, PP, recompute/offload, or the 202,752-token target.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

pytestmark = [pytest.mark.mlite, pytest.mark.smoke]

_TRANSFORMERS_AUTHORITY = "5.12.0"
_BATCH = 2
_SEQ = 10
_VOCAB = 71
_HIDDEN = 32
_INDEX_TOPK = 3


class _TorchRMSNorm(nn.Module):
    """Torch stand-in for TE RMSNorm with the HF GLM accumulation contract."""

    def __init__(self, hidden_size: int, eps: float = 1.0e-6, **_kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        normalized = hidden_states.float()
        variance = normalized.square().mean(dim=-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.eps)
        return self.weight * normalized.to(dtype=input_dtype)


class _TorchLinear(nn.Module):
    """Single-rank Torch replacement for the subset of ``te.Linear`` used here."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        params_dtype: torch.dtype = torch.bfloat16,
        **_kwargs,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=params_dtype)
        )
        self.bias = (
            nn.Parameter(torch.empty(out_features, dtype=params_dtype))
            if bias
            else None
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states, self.weight, self.bias)


class _TorchLayerNormLinear(_TorchLinear):
    """Torch RMSNorm+linear replacement for MLite's fused dense-MLP input."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        normalization: str,
        eps: float,
        zero_centered_gamma: bool = False,
        **kwargs,
    ):
        assert normalization == "RMSNorm"
        assert zero_centered_gamma is False
        super().__init__(in_features, out_features, **kwargs)
        self.layer_norm_weight = nn.Parameter(torch.ones(in_features))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        normalized = hidden_states.float()
        variance = normalized.square().mean(dim=-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.eps)
        normalized = self.layer_norm_weight * normalized.to(dtype=input_dtype)
        return F.linear(normalized, self.weight, self.bias)


class _TorchGroupedLinear(nn.Module):
    """Expert-major Torch replacement preserving TE's ``weightN`` state names."""

    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        *,
        bias: bool = False,
        params_dtype: torch.dtype = torch.bfloat16,
        **_kwargs,
    ):
        super().__init__()
        assert bias is False
        self.num_experts = num_experts
        self.out_features = out_features
        for expert_idx in range(num_experts):
            weight = nn.Parameter(
                torch.empty(out_features, in_features, dtype=params_dtype)
            )
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            self.register_parameter(f"weight{expert_idx}", weight)

    def forward(
        self, hidden_states: torch.Tensor, tokens_per_expert: list[int]
    ) -> torch.Tensor:
        assert len(tokens_per_expert) == self.num_experts
        pieces = hidden_states.split([int(count) for count in tokens_per_expert], dim=0)
        outputs = [
            F.linear(piece, getattr(self, f"weight{expert_idx}"))
            for expert_idx, piece in enumerate(pieces)
        ]
        if not outputs:
            return hidden_states.new_empty((0, self.out_features))
        return torch.cat(outputs, dim=0)


def _torch_indexer_topk(
    q_indexer: torch.Tensor,
    k_indexer: torch.Tensor,
    weights: torch.Tensor,
    topk: int,
    ratio: int = 1,
    indexer_softmax_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """First-principles oracle for MLite's inference indexer contract."""

    q = q_indexer.permute(1, 0, 2, 3).float()
    k = k_indexer.permute(1, 0, 2).float()
    w = weights.permute(1, 0, 2).float()
    scores = torch.relu(torch.einsum("bqhd,bkd->bqhk", q, k))
    scores = (scores * w.unsqueeze(-1)).sum(dim=2) * indexer_softmax_scale
    query_positions = torch.arange(scores.shape[1], device=scores.device)
    key_positions = torch.arange(scores.shape[2], device=scores.device)
    valid_counts = ((query_positions + 1) // ratio).clamp(max=scores.shape[2])
    valid = key_positions.unsqueeze(0) < valid_counts.unsqueeze(1)
    scores = scores.masked_fill(~valid.unsqueeze(0), -torch.inf)

    effective_topk = min(topk, scores.shape[-1])
    values, indices = torch.topk(scores, effective_topk, dim=-1)
    indices = torch.where(torch.isfinite(values), indices, -torch.ones_like(indices))
    if effective_topk < topk:
        indices = F.pad(indices, (0, topk - effective_topk), value=-1)
    topk_length = (indices >= 0).sum(dim=-1, dtype=torch.int32)
    return indices.to(torch.int32), topk_length


def _torch_sparse_attention(
    query: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
    topk_length=None,
    indexer_topk: int = 0,
    value_dim: int | None = None,
) -> torch.Tensor:
    """Differentiable exact sparse attention for MLite's flattened indices."""

    del topk_length, indexer_topk
    seq, batch, heads, _ = query.shape
    assert value_dim is not None
    rows = []
    for query_idx in range(seq):
        batches = []
        for batch_idx in range(batch):
            flat_row = topk_idxs[query_idx * batch + batch_idx]
            valid_flat = flat_row[flat_row >= 0].long()
            assert torch.all(valid_flat % batch == batch_idx)
            key_indices = valid_flat // batch
            selected_kv = kv[:, batch_idx][key_indices]
            scores = (
                torch.einsum("hd,kd->hk", query[query_idx, batch_idx], selected_kv)
                * softmax_scale
            )
            sink = attn_sink.to(dtype=scores.dtype).view(heads, 1)
            probabilities = torch.softmax(
                torch.cat((scores, sink), dim=-1), dim=-1, dtype=torch.float32
            )[..., :-1].to(dtype=query.dtype)
            values = selected_kv[:, :value_dim]
            result = torch.einsum("hk,kv->hv", probabilities, values)
            batches.append(result.reshape(-1))
        rows.append(torch.stack(batches, dim=0))
    return torch.stack(rows, dim=0)


def _hf_config():
    import transformers

    assert transformers.__version__ == _TRANSFORMERS_AUTHORITY, (
        "GLM-5.2 ADAPTED HF comparison is pinned to Transformers "
        f"{_TRANSFORMERS_AUTHORITY}; got {transformers.__version__}"
    )
    config = transformers.GlmMoeDsaConfig(
        vocab_size=_VOCAB,
        hidden_size=_HIDDEN,
        intermediate_size=48,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
        index_head_dim=8,
        index_n_heads=4,
        index_topk=_INDEX_TOPK,
        indexer_types=["full", "shared"],
        index_topk_freq=2,
        index_skip_topk_offset=1,
        indexer_rope_interleave=True,
        indexer_rope_first=True,
        indexer_use_hadamard=False,
        rope_interleave=True,
        rope_parameters={"rope_type": "default", "rope_theta": 8_000_000.0},
        latent_rms_norm_eps=1.0e-6,
        indexer_layer_norm_eps=1.0e-6,
        rms_norm_eps=1.0e-5,
        first_k_dense_replace=1,
        mlp_layer_types=["dense", "sparse"],
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.5,
        norm_topk_prob=True,
        num_nextn_predict_layers=0,
        max_position_embeddings=128,
        attention_bias=False,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    return config


def _checkpoint_state_for_lite(hf_model: nn.Module) -> dict[str, torch.Tensor]:
    """Expose packed HF experts under the per-expert names accepted by MLite."""

    state = {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in hf_model.state_dict().items()
    }
    for layer_idx, layer_type in enumerate(hf_model.config.mlp_layer_types):
        if layer_type != "sparse":
            continue
        prefix = f"model.layers.{layer_idx}.mlp.experts"
        gate_up = state[f"{prefix}.gate_up_proj"]
        down = state[f"{prefix}.down_proj"]
        assert gate_up.shape == (
            hf_model.config.n_routed_experts,
            2 * hf_model.config.moe_intermediate_size,
            hf_model.config.hidden_size,
        )
        assert down.shape == (
            hf_model.config.n_routed_experts,
            hf_model.config.hidden_size,
            hf_model.config.moe_intermediate_size,
        )
        for expert_idx in range(hf_model.config.n_routed_experts):
            gate, up = gate_up[expert_idx].chunk(2, dim=0)
            expert_prefix = f"{prefix}.{expert_idx}"
            state[f"{expert_prefix}.gate_proj.weight"] = gate.contiguous().clone()
            state[f"{expert_prefix}.up_proj.weight"] = up.contiguous().clone()
            state[f"{expert_prefix}.down_proj.weight"] = (
                down[expert_idx].contiguous().clone()
            )
    return state


def _layer_capture(layers, *, sequence_first: bool):
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for index, layer in enumerate(layers):

        def _hook(_module, _args, output, *, layer_index=index):
            value = output[0] if isinstance(output, tuple) else output
            assert isinstance(value, torch.Tensor)
            if sequence_first:
                value = value.transpose(0, 1)
            captured[layer_index] = value.detach().clone()

        handles.append(layer.register_forward_hook(_hook))
    return captured, handles


def _embedding_grad_capture(embedding):
    captured: list[torch.Tensor] = []

    def _hook(_module, _args, output):
        assert isinstance(output, torch.Tensor)
        output.retain_grad()
        captured.append(output)

    return captured, embedding.register_forward_hook(_hook)


def _metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    assert candidate.shape == reference.shape
    reference = reference.detach().double()
    candidate = candidate.detach().double()
    assert torch.isfinite(reference).all()
    assert torch.isfinite(candidate).all()
    reference_flat = reference.reshape(-1)
    candidate_flat = candidate.reshape(-1)
    reference_norm = torch.linalg.vector_norm(reference_flat)
    candidate_norm = torch.linalg.vector_norm(candidate_flat)
    assert reference_norm > 0 and candidate_norm > 0
    difference = candidate - reference
    rms_scale = torch.maximum(
        torch.sqrt(torch.mean(reference.square())),
        torch.sqrt(torch.mean(candidate.square())),
    ).clamp_min(torch.finfo(torch.float64).tiny)
    return {
        "cosine": F.cosine_similarity(reference_flat, candidate_flat, dim=0).item(),
        "rms_relative": (
            torch.sqrt(torch.mean(difference.square())) / rms_scale
        ).item(),
        "norm_ratio": (candidate_norm / reference_norm).item(),
        "max_abs": difference.abs().max().item(),
    }


def _assert_parity(
    name: str,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    cosine_min: float,
    rms_relative_max: float,
    norm_ratio_min: float,
    norm_ratio_max: float,
    max_abs_max: float,
) -> dict[str, float]:
    values = _metrics(reference, candidate)
    print(
        f"[glm52-hf-production-layout] {name}: shape={tuple(reference.shape)} "
        f"cosine={values['cosine']:.12f} "
        f"rms_relative={values['rms_relative']:.12e} "
        f"norm_ratio={values['norm_ratio']:.12f} "
        f"max_abs={values['max_abs']:.12e}"
    )
    assert values["cosine"] >= cosine_min, values
    assert values["rms_relative"] <= rms_relative_max, values
    assert norm_ratio_min <= values["norm_ratio"] <= norm_ratio_max, values
    assert values["max_abs"] <= max_abs_max, values
    return values


def _shifted_external_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    assert logits.shape == (_BATCH, _SEQ, _VOCAB)
    assert labels.shape == (_BATCH, _SEQ)
    return F.cross_entropy(
        logits[:, :-1].float().reshape(-1, _VOCAB),
        labels[:, 1:].reshape(-1),
    )


def _capture_first_attention_topk(model, input_ids, position_ids):
    captured: list[torch.Tensor] = []

    def _hook(_module, _args, output):
        assert isinstance(output, tuple) and len(output) == 3
        assert output[2] is not None
        captured.append(output[2].detach().clone())

    handle = model.model.layers[0].self_attn.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                position_ids=position_ids,
                use_cache=False,
                logits_to_keep=0,
                return_dict=True,
            ).logits.detach()
    finally:
        handle.remove()
    assert len(captured) == 1
    return logits, captured[0]


def _make_instance_local_interleaved_indexer(hf_impl, vanilla_indexer):
    """Replace one HF indexer without changing any HF module-global symbol."""

    class _AdaptedInterleavedGlmMoeDsaIndexer(hf_impl.GlmMoeDsaIndexer):
        @torch.no_grad()
        def forward(
            self,
            hidden_states,
            q_resid,
            position_embeddings,
            attention_mask,
            position_ids,
            past_key_values=None,
        ):
            batch_size, seq_len, _ = hidden_states.shape
            cos, sin = position_embeddings
            q = self.wq_b(q_resid).view(
                batch_size, seq_len, self.n_heads, self.head_dim
            )
            q_rot, q_pass = torch.split(
                q,
                [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim],
                dim=-1,
            )

            k = self.k_norm(self.wk(hidden_states)).unsqueeze(2)
            k_rot, k_pass = torch.split(
                k,
                [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim],
                dim=-1,
            )
            q_rot, k_rot = hf_impl.apply_rotary_pos_emb_interleave(
                q_rot, k_rot, cos, sin, unsqueeze_dim=2
            )
            q = torch.cat((q_rot, q_pass), dim=-1)
            k = torch.cat((k_rot, k_pass), dim=-1).squeeze(2)

            if past_key_values is not None:
                k = past_key_values.update_indexer(k, self.layer_idx)

            scores = torch.matmul(q.float(), k.transpose(-1, -2).float().unsqueeze(1))
            scores = F.relu(scores * self.softmax_scale)
            weights = self.weights_proj(
                hidden_states.to(self.weights_proj.weight.dtype)
            ).float() * (self.n_heads**-0.5)
            index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)
            if attention_mask is not None:
                index_scores = index_scores + attention_mask
            else:
                key_positions = torch.arange(
                    index_scores.shape[-1], device=index_scores.device
                )
                causal = key_positions[None, None, :] > position_ids[:, :, None]
                index_scores = index_scores.masked_fill(causal, float("-inf"))

            topk = min(self.index_topk, index_scores.shape[-1])
            return index_scores.topk(topk, dim=-1).indices.to(torch.int32)

    reference_parameter = next(vanilla_indexer.parameters())
    adapted = _AdaptedInterleavedGlmMoeDsaIndexer(
        vanilla_indexer.config, vanilla_indexer.layer_idx
    ).to(device=reference_parameter.device, dtype=reference_parameter.dtype)
    adapted.load_state_dict(vanilla_indexer.state_dict(), strict=True)
    adapted.train(vanilla_indexer.training)
    return adapted


def test_glm52_transformers_512_adapted_production_layout_full_model_parity(
    tmp_path, transformer_engine_import_stub, monkeypatch
):
    """Configured/interleaved F/S GLM-5.2 vs instance-locally ADAPTED HF."""

    transformers = pytest.importorskip("transformers")
    pytest.importorskip("safetensors")
    assert transformers.__version__ == _TRANSFORMERS_AUTHORITY
    from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as hf_impl

    transformer_engine_import_stub()
    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.model.glm5.lite import model as lite_impl
    from megatron.lite.model.glm5.lite.checkpoint import load_hf_weights
    from megatron.lite.primitive.ckpt.hf_weights import save_safetensors
    from megatron.lite.primitive.modules import experts as experts_impl
    from megatron.lite.primitive.modules.attention import dsa
    from megatron.lite.primitive.parallel import linear as linear_impl
    from megatron.lite.primitive.parallel.state import ParallelState

    # Keep this lane independent of TE and fused DSA availability.  These are
    # implementation adapters only; the actual MLite model/layer wiring,
    # checkpoint loader, IndexShare state, and autograd graph remain in use.
    monkeypatch.setattr(lite_impl.te, "RMSNorm", _TorchRMSNorm)
    monkeypatch.setattr(lite_impl.te, "Linear", _TorchLinear)
    monkeypatch.setattr(lite_impl.te, "LayerNormLinear", _TorchLayerNormLinear)
    monkeypatch.setattr(
        lite_impl.te, "GroupedLinear", _TorchGroupedLinear, raising=False
    )
    monkeypatch.setattr(linear_impl.te, "Linear", _TorchLinear)
    monkeypatch.setattr(linear_impl.te, "LayerNormLinear", _TorchLayerNormLinear)
    monkeypatch.setattr(
        experts_impl.te, "GroupedLinear", _TorchGroupedLinear, raising=False
    )
    monkeypatch.setattr(dsa, "RMSNorm", _TorchRMSNorm)
    recorded_lite_topk: list[torch.Tensor] = []

    def _recording_indexer_topk(*args, **kwargs):
        indices, lengths = _torch_indexer_topk(*args, **kwargs)
        recorded_lite_topk.append(indices.detach().clone())
        return indices, lengths

    monkeypatch.setattr(dsa._dsa_kernels, "indexer_topk", _recording_indexer_topk)
    monkeypatch.setattr(dsa._dsa_kernels, "dsa_sparse_attn", _torch_sparse_attention)

    torch.manual_seed(20260627)
    hf_config = _hf_config()
    hf_model = hf_impl.GlmMoeDsaForCausalLM(hf_config).double().eval()
    assert type(hf_model.model.layers[1].mlp.experts).__name__ == "GlmMoeDsaNaiveMoe"

    cfg = Glm5Config.from_hf_config(hf_config)
    assert cfg.resolved_dsa_indexer_types == ("full", "shared")
    assert cfg.uses_configured_dsa_rope_layout is True
    assert cfg.rope_interleave is True
    assert cfg.indexer_rope_interleave is True
    assert cfg.mlp_layer_types == ["dense", "sparse"]
    assert hf_config.indexer_rope_interleave is True

    checkpoint_dir = tmp_path / "hf_production_layout"
    save_safetensors(_checkpoint_state_for_lite(hf_model), str(checkpoint_dir))
    ps = ParallelState()
    train_config = SimpleNamespace(
        tp=1,
        ep=1,
        etp=1,
        pp=1,
        cp=1,
        vpp=None,
        use_deepep=False,
        fp8=False,
        recompute_modules=[],
        offload_modules=[],
        deterministic=True,
    )
    lite_model = (
        lite_impl.Glm5Model(
            cfg,
            train_config,
            ps,
            attention_backend_override="unfused",
            mtp_enable=False,
        )
        .double()
        .eval()
    )
    load_hf_weights(lite_model, str(checkpoint_dir), cfg, ps)
    for layer_idx, layer in enumerate(lite_model.layers):
        attention = layer.self_attention.self_attention
        assert attention.rope_interleaved is True
        if layer_idx == 0:
            assert attention.indexer is not None
            assert attention.indexer.rope_interleaved is True
        else:
            assert attention.skip_topk is True
            assert attention.indexer is None

    input_ids = torch.tensor(
        [
            [3, 5, 7, 11, 13, 17, 19, 23, 29, 31],
            [37, 41, 43, 47, 53, 59, 61, 67, 2, 4],
        ],
        dtype=torch.long,
    )
    labels = torch.tensor(
        [
            [6, 10, 14, 18, 22, 26, 30, 34, 38, 42],
            [46, 50, 54, 58, 62, 66, 70, 1, 9, 15],
        ],
        dtype=torch.long,
    )
    position_ids = torch.arange(_SEQ, dtype=torch.long).unsqueeze(0).expand(_BATCH, -1)

    # Unmodified Transformers 5.12 is the negative control: its indexer ignores
    # indexer_rope_interleave even though the released GLM-5.2 config enables it.
    vanilla_logits, vanilla_topk = _capture_first_attention_topk(
        hf_model, input_ids, position_ids
    )
    vanilla_indexer = hf_model.model.layers[0].self_attn.indexer
    assert type(vanilla_indexer) is hf_impl.GlmMoeDsaIndexer
    vanilla_half_split_helper = hf_impl.apply_rotary_pos_emb
    adapted_indexer = _make_instance_local_interleaved_indexer(hf_impl, vanilla_indexer)
    assert type(adapted_indexer).__name__ == "_AdaptedInterleavedGlmMoeDsaIndexer"
    hf_model.model.layers[0].self_attn.indexer = adapted_indexer
    assert hf_impl.apply_rotary_pos_emb is vanilla_half_split_helper

    adapted_topk: list[torch.Tensor] = []

    def _adapted_topk_hook(_module, _args, output):
        assert isinstance(output, tuple) and output[2] is not None
        adapted_topk.append(output[2].detach().clone())

    topk_handle = hf_model.model.layers[0].self_attn.register_forward_hook(
        _adapted_topk_hook
    )
    hf_layers, hf_layer_handles = _layer_capture(
        hf_model.model.layers, sequence_first=False
    )
    lite_layers, lite_layer_handles = _layer_capture(
        lite_model.layers, sequence_first=True
    )
    hf_embeddings, hf_embedding_handle = _embedding_grad_capture(
        hf_model.model.embed_tokens
    )
    assert lite_model.embed is not None
    lite_embeddings, lite_embedding_handle = _embedding_grad_capture(
        lite_model.embed.embedding
    )
    try:
        hf_output = hf_model(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=False,
            logits_to_keep=0,
            return_dict=True,
        )
        lite_output = lite_model(
            input_ids=input_ids,
            position_ids=position_ids,
            packed_seq_params=None,
        )
    finally:
        for handle in [
            topk_handle,
            *hf_layer_handles,
            *lite_layer_handles,
            hf_embedding_handle,
            lite_embedding_handle,
        ]:
            handle.remove()

    assert len(adapted_topk) == 1
    assert len(recorded_lite_topk) == 1
    adapted_topk_tensor = adapted_topk[0]
    lite_topk_tensor = recorded_lite_topk[0]
    assert adapted_topk_tensor.shape == (_BATCH, _SEQ, _INDEX_TOPK)
    assert lite_topk_tensor.shape == (_BATCH, _SEQ, _INDEX_TOPK)
    for batch_idx in range(_BATCH):
        for query_idx in range(_SEQ):
            hf_valid = adapted_topk_tensor[batch_idx, query_idx]
            hf_valid = hf_valid[hf_valid <= query_idx].sort().values
            lite_valid = lite_topk_tensor[batch_idx, query_idx]
            lite_valid = lite_valid[lite_valid >= 0].sort().values
            assert torch.equal(hf_valid, lite_valid), (
                f"top-k mismatch at batch={batch_idx}, query={query_idx}: "
                f"hf={hf_valid.tolist()} lite={lite_valid.tolist()}"
            )

    fully_valid = slice(_INDEX_TOPK - 1, None)
    vanilla_valid_rows = vanilla_topk[:, fully_valid].sort(dim=-1).values
    adapted_valid_rows = adapted_topk_tensor[:, fully_valid].sort(dim=-1).values
    differing_topk_rows = int(
        torch.any(vanilla_valid_rows != adapted_valid_rows, dim=-1).sum().item()
    )
    assert differing_topk_rows > 0, (
        "the deterministic batch did not expose the vanilla HF half-split "
        "indexer layout conflict"
    )

    hf_logits = hf_output.logits
    lite_logits = lite_output["logits"]
    assert hf_logits.shape == (_BATCH, _SEQ, _VOCAB)
    assert lite_logits.shape == (_BATCH, _SEQ, _VOCAB)
    vanilla_logit_max_abs = float((vanilla_logits - hf_logits).abs().max().item())
    assert vanilla_logit_max_abs > 1.0e-6

    assert set(hf_layers) == {0, 1}
    assert set(lite_layers) == {0, 1}
    layer_metrics = []
    for layer_idx, layer_type in enumerate(cfg.mlp_layer_types):
        assert hf_layers[layer_idx].shape == (_BATCH, _SEQ, _HIDDEN)
        assert lite_layers[layer_idx].shape == (_BATCH, _SEQ, _HIDDEN)
        layer_metrics.append(
            _assert_parity(
                f"layer-{layer_idx}-{layer_type}",
                hf_layers[layer_idx],
                lite_layers[layer_idx],
                cosine_min=0.999999,
                rms_relative_max=1.0e-4,
                norm_ratio_min=0.9999,
                norm_ratio_max=1.0001,
                max_abs_max=1.0e-4,
            )
        )
    logits_metrics = _assert_parity(
        "final-logits",
        hf_logits,
        lite_logits,
        cosine_min=0.999999,
        rms_relative_max=1.0e-4,
        norm_ratio_min=0.9999,
        norm_ratio_max=1.0001,
        max_abs_max=1.0e-4,
    )

    hf_loss = _shifted_external_ce(hf_logits, labels)
    lite_loss = _shifted_external_ce(lite_logits, labels)
    loss_abs = float((hf_loss - lite_loss).abs().item())
    loss_relative = loss_abs / max(
        abs(float(hf_loss.item())), abs(float(lite_loss.item())), 1.0e-12
    )
    print(
        "[glm52-hf-production-layout] shifted-external-CE: "
        f"hf={float(hf_loss.item()):.12f} lite={float(lite_loss.item()):.12f} "
        f"abs={loss_abs:.12e} relative={loss_relative:.12e}"
    )
    assert loss_abs <= 1.0e-5
    assert loss_relative <= 3.0e-6

    hf_loss.backward()
    lite_loss.backward()
    assert len(hf_embeddings) == 1
    assert len(lite_embeddings) == 1
    hf_embedding_grad = hf_embeddings[0].grad
    lite_embedding_grad = lite_embeddings[0].grad
    assert hf_embedding_grad is not None
    assert lite_embedding_grad is not None
    embedding_grad_metrics = _assert_parity(
        "embedding-output-gradient",
        hf_embedding_grad,
        lite_embedding_grad,
        cosine_min=0.99999,
        rms_relative_max=5.0e-4,
        norm_ratio_min=0.9995,
        norm_ratio_max=1.0005,
        max_abs_max=1.0e-5,
    )
    hf_grad_norm = float(torch.linalg.vector_norm(hf_embedding_grad.double()).item())
    lite_grad_norm = float(
        torch.linalg.vector_norm(lite_embedding_grad.double()).item()
    )
    assert hf_grad_norm > 0.0 and lite_grad_norm > 0.0

    print(
        "NON_SKIP_GLM52_TRANSFORMERS_512_ADAPTED_PRODUCTION_LAYOUT_"
        "FULL_MODEL_PARITY_PASSED "
        "adapter=instance_local_interleaved_indexer_subclass "
        "scope=tiny_full_model_dense_moe_indexshare_F_S "
        "configured_main_rope=true configured_indexer_rope=true "
        f"vanilla_differing_topk_rows={differing_topk_rows} "
        f"vanilla_logit_max_abs={vanilla_logit_max_abs:.9e} "
        f"hidden_min_cosine={min(item['cosine'] for item in layer_metrics):.9f} "
        f"hidden_max_rms_relative="
        f"{max(item['rms_relative'] for item in layer_metrics):.9e} "
        f"logits_cosine={logits_metrics['cosine']:.9f} "
        f"logits_rms_relative={logits_metrics['rms_relative']:.9e} "
        f"loss_abs={loss_abs:.9e} loss_relative={loss_relative:.9e} "
        f"embedding_grad_cosine={embedding_grad_metrics['cosine']:.9f} "
        f"embedding_grad_rms_relative="
        f"{embedding_grad_metrics['rms_relative']:.9e} "
        "excludes=unmodified_hf_vanilla,production_sparse_kernel,mtp,cp,pp,"
        "recompute_offload,long_context"
    )
