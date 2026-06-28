# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Independent Transformers 5.12 authority for DSv4 ratio-4 CSA attention.

This is deliberately scoped to one *complete attention block*, not advertised
as whole-model DeepSeek-V4 parity.  It covers the production ratio-4 path end
to end: query/KV projections, partial YaRN RoPE, sliding KV, overlapping
compressor, Lightning Indexer top-k, attention sink, inverse RoPE, and grouped
output projection.  The official HF eager implementation is compared with both
MLite's Torch oracle and its fused inference path.

Unlike the historical ``ds4_hf_ref`` script, this gate requires exact shapes
and computes CE against one fixed external label tensor.  It never truncates to
``min(numel)`` and never derives labels from either implementation's logits.

Excluded by construction: decoder mHC, Hash-MoE, checkpoint dequantization,
MTP, PP/CP, and optimizer behavior.  Those need separate gates and must not be
inferred from this test.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.mlite, pytest.mark.smoke, pytest.mark.gpu]

_TRANSFORMERS_AUTHORITY = "5.12.0"
_BATCH = 1
_SEQ = 512
_VOCAB = 64
_COMPRESS_RATIO = 4
_INDEX_TOPK = 64


def _hf_config():
    import transformers
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
        DeepseekV4Config,
    )

    assert transformers.__version__ == _TRANSFORMERS_AUTHORITY, (
        "DSv4 HF authority is pinned to Transformers "
        f"{_TRANSFORMERS_AUTHORITY}; got {transformers.__version__}"
    )
    config = DeepseekV4Config(
        vocab_size=_VOCAB,
        hidden_size=128,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=64,
        num_key_value_heads=1,
        head_dim=512,
        q_lora_rank=32,
        o_lora_rank=32,
        o_groups=8,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        routed_scaling_factor=1.5,
        layer_types=["compressed_sparse_attention"],
        compress_rates={
            "compressed_sparse_attention": _COMPRESS_RATIO,
            "heavily_compressed_attention": 128,
        },
        mlp_layer_types=["hash_moe"],
        max_position_embeddings=1_048_576,
        partial_rotary_factor=64 / 512,
        rope_theta=10_000.0,
        compress_rope_theta=160_000.0,
        rope_parameters={
            "rope_type": "yarn",
            "factor": 16.0,
            "original_max_position_embeddings": 65_536,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
        },
        sliding_window=128,
        index_head_dim=128,
        index_n_heads=64,
        index_topk=_INDEX_TOPK,
        hc_mult=2,
        hc_sinkhorn_iters=4,
        num_nextn_predict_layers=0,
        rms_norm_eps=1.0e-6,
        initializer_range=0.02,
        use_cache=False,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    assert config.layer_types == ["compressed_sparse_attention"]
    assert config.compress_rates["compressed_sparse_attention"] == _COMPRESS_RATIO
    assert config.index_topk == _INDEX_TOPK
    return config


def _lite_config():
    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config

    return DeepseekV4Config(
        vocab_size=_VOCAB,
        hidden_size=128,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=64,
        num_key_value_heads=1,
        head_dim=512,
        qk_rope_head_dim=64,
        q_lora_rank=32,
        o_lora_rank=32,
        o_groups=8,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        routed_scaling_factor=1.5,
        max_position_embeddings=1_048_576,
        rope_theta=10_000.0,
        compress_rope_theta=160_000.0,
        rotary_scaling_factor=16.0,
        original_max_position_embeddings=65_536,
        beta_fast=32.0,
        beta_slow=1.0,
        compress_ratios=[_COMPRESS_RATIO],
        sliding_window=128,
        num_hash_layers=1,
        hc_mult=2,
        hc_sinkhorn_iters=4,
        index_head_dim=128,
        index_n_heads=64,
        index_topk=_INDEX_TOPK,
        num_nextn_predict_layers=0,
        rms_norm_eps=1.0e-6,
        initializer_range=0.02,
    )


def _copy_hf_ratio4_attention_weights(hf_attention, lite_attention) -> None:
    """Copy every MLite attention parameter from its official HF counterpart."""
    hf_compressor = hf_attention.compressor
    lite_compressor = lite_attention.compressor
    assert hf_compressor is not None
    assert lite_compressor is not None
    hf_indexer = hf_compressor.indexer
    lite_indexer = lite_attention.indexer
    assert hf_indexer is not None
    assert lite_indexer is not None

    pairs = [
        ("sinks", lite_attention.sinks, hf_attention.sinks),
        ("q-a", lite_attention.wq_a.weight, hf_attention.q_a_proj.weight),
        ("q-a-norm", lite_attention.q_norm.weight, hf_attention.q_a_norm.weight),
        ("q-b", lite_attention.wq_b.weight, hf_attention.q_b_proj.weight),
        ("kv", lite_attention.wkv.weight, hf_attention.kv_proj.weight),
        ("kv-norm", lite_attention.kv_norm.weight, hf_attention.kv_norm.weight),
        ("o-a", lite_attention.wo_a.weight, hf_attention.o_a_proj.weight),
        ("o-b", lite_attention.wo_b.weight, hf_attention.o_b_proj.weight),
        ("compressor-kv", lite_compressor.wkv.weight, hf_compressor.kv_proj.weight),
        (
            "compressor-gate",
            lite_compressor.wgate.weight,
            hf_compressor.gate_proj.weight,
        ),
        ("compressor-position", lite_compressor.ape, hf_compressor.position_bias),
        ("compressor-norm", lite_compressor.norm.weight, hf_compressor.kv_norm.weight),
        ("indexer-query", lite_indexer.wq_b.weight, hf_indexer.q_b_proj.weight),
        (
            "indexer-weights",
            lite_indexer.weights_proj.weight,
            hf_indexer.scorer.weights_proj.weight,
        ),
        (
            "indexer-compressor-kv",
            lite_indexer.compressor.wkv.weight,
            hf_indexer.kv_proj.weight,
        ),
        (
            "indexer-compressor-gate",
            lite_indexer.compressor.wgate.weight,
            hf_indexer.gate_proj.weight,
        ),
        (
            "indexer-compressor-position",
            lite_indexer.compressor.ape,
            hf_indexer.position_bias,
        ),
        (
            "indexer-compressor-norm",
            lite_indexer.compressor.norm.weight,
            hf_indexer.kv_norm.weight,
        ),
    ]

    copied: set[int] = set()
    with torch.no_grad():
        for name, destination, source in pairs:
            assert destination.shape == source.shape, (
                f"{name}: MLite shape {tuple(destination.shape)} != "
                f"HF shape {tuple(source.shape)}"
            )
            destination.copy_(
                source.to(device=destination.device, dtype=destination.dtype)
            )
            copied.add(id(destination))

    named_parameters = {
        id(parameter): name for name, parameter in lite_attention.named_parameters()
    }
    missing = sorted(
        name for ident, name in named_parameters.items() if ident not in copied
    )
    unexpected = sorted(ident for ident in copied if ident not in named_parameters)
    assert not missing, f"unmapped MLite attention parameters: {missing}"
    assert not unexpected, f"copied objects are not MLite parameters: {unexpected}"


def _sliding_causal_mask(
    *, batch: int, seq: int, window: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    query = torch.arange(seq, device=device).view(seq, 1)
    key = torch.arange(seq, device=device).view(1, seq)
    allowed = (key <= query) & (key >= query - window + 1)
    mask = torch.zeros((seq, seq), device=device, dtype=dtype)
    mask.masked_fill_(~allowed, -float("inf"))
    return mask.view(1, 1, seq, seq).expand(batch, -1, -1, -1).contiguous()


def _metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    assert (
        candidate.shape == reference.shape
    ), f"candidate shape {tuple(candidate.shape)} != reference shape {tuple(reference.shape)}"
    reference = reference.detach().float()
    candidate = candidate.detach().float()
    assert torch.isfinite(reference).all()
    assert torch.isfinite(candidate).all()
    reference_flat = reference.reshape(-1)
    candidate_flat = candidate.reshape(-1)
    reference_norm = torch.linalg.vector_norm(reference_flat)
    candidate_norm = torch.linalg.vector_norm(candidate_flat)
    assert reference_norm > 0 and candidate_norm > 0
    diff_rms = torch.sqrt(torch.mean((reference - candidate).square()))
    scale = torch.maximum(
        torch.sqrt(torch.mean(reference.square())),
        torch.sqrt(torch.mean(candidate.square())),
    ).clamp_min(torch.finfo(torch.float32).tiny)
    return {
        "cosine": F.cosine_similarity(reference_flat, candidate_flat, dim=0).item(),
        "rms_relative": (diff_rms / scale).item(),
        "norm_ratio": (candidate_norm / reference_norm).item(),
        "max_abs": (reference - candidate).abs().max().item(),
    }


def _assert_hf_parity(
    name: str, reference: torch.Tensor, candidate: torch.Tensor
) -> dict[str, float]:
    values = _metrics(reference, candidate)
    print(f"[dsv4-hf-ratio4] {name}: shape={tuple(reference.shape)} metrics={values}")
    assert values["cosine"] >= 0.9985, values
    assert values["rms_relative"] <= 0.05, values
    assert 0.99 <= values["norm_ratio"] <= 1.01, values
    assert values["max_abs"] <= 0.06, values
    return values


def _assert_lite_backend_parity(
    name: str, reference: torch.Tensor, candidate: torch.Tensor
) -> dict[str, float]:
    values = _metrics(reference, candidate)
    print(f"[dsv4-hf-ratio4] {name}: shape={tuple(reference.shape)} metrics={values}")
    assert values["cosine"] >= 0.9995, values
    assert values["rms_relative"] <= 0.02, values
    assert 0.99 <= values["norm_ratio"] <= 1.01, values
    assert values["max_abs"] <= 0.03, values
    return values


def _fixed_label_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    assert logits.shape[:-1] == labels.shape
    assert logits.shape[-1] == _VOCAB
    return F.cross_entropy(logits.float().reshape(-1, _VOCAB), labels.reshape(-1))


def _assert_loss_parity(
    name: str,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    abs_max: float,
    relative_max: float,
) -> dict[str, float]:
    reference_value = float(reference.detach().float().item())
    candidate_value = float(candidate.detach().float().item())
    absolute = abs(reference_value - candidate_value)
    relative = absolute / max(abs(reference_value), abs(candidate_value), 1.0e-12)
    print(
        f"[dsv4-hf-ratio4] {name}: reference={reference_value:.9f} "
        f"candidate={candidate_value:.9f} abs={absolute:.9f} relative={relative:.9f}"
    )
    assert absolute <= abs_max
    assert relative <= relative_max
    return {"absolute": absolute, "relative": relative}


def test_dsv4_transformers_512_ratio4_csa_attention_numeric_parity():
    """HF eager vs MLite Torch/fused for the complete ratio-4 attention block."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for DSv4 ratio-4 HF numeric parity.")
    pytest.importorskip("transformers")
    pytest.importorskip("transformer_engine.pytorch")
    pytest.importorskip("cudnn", reason="MLite fused ratio-4 CSA requires cuDNN DSA.")

    from megatron.lite.primitive.modules.attention.csa import CompressedSparseAttention
    from megatron.lite.primitive.parallel import ParallelState
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4ForCausalLM,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(20260627)
    torch.cuda.manual_seed_all(20260627)

    hf_model = DeepseekV4ForCausalLM(_hf_config())
    hf_attention = hf_model.model.layers[0].self_attn.to(
        device=device, dtype=torch.bfloat16
    )
    assert type(hf_attention.compressor).__name__ == "DeepseekV4CSACompressor"
    assert type(hf_attention.compressor.indexer).__name__ == "DeepseekV4Indexer"
    # Keep inverse frequencies in fp32, matching HF's normal construction.
    hf_rotary = hf_model.model.rotary_emb.to(device=device)
    hf_attention.eval()

    lite_config = _lite_config()
    lite_torch = CompressedSparseAttention(
        lite_config, layer_idx=0, ps=ParallelState()
    ).to(device=device, dtype=torch.bfloat16)
    _copy_hf_ratio4_attention_weights(hf_attention, lite_torch)
    lite_fused = CompressedSparseAttention(
        lite_config, layer_idx=0, ps=ParallelState()
    ).to(device=device, dtype=torch.bfloat16)
    lite_fused.load_state_dict(lite_torch.state_dict(), strict=True)
    lite_torch.attention_backend = "torch"
    lite_fused.attention_backend = "fused"
    lite_torch.eval()
    lite_fused.eval()

    assert lite_torch.compress_ratio == _COMPRESS_RATIO
    assert lite_torch.compressor is not None
    assert lite_torch.indexer is not None
    compressed_entries = _SEQ // _COMPRESS_RATIO
    assert (
        _INDEX_TOPK < compressed_entries
    ), "the parity batch must make Lightning Indexer selection genuinely sparse"

    generator = torch.Generator(device="cpu").manual_seed(20260628)
    hidden_cpu = torch.randn(
        _BATCH, _SEQ, lite_config.hidden_size, generator=generator, dtype=torch.float32
    )
    hidden = hidden_cpu.to(device=device, dtype=torch.bfloat16)
    hf_hidden = hidden.detach().clone().requires_grad_(True)
    lite_torch_hidden = hidden.detach().clone().requires_grad_(True)
    lite_fused_hidden = hidden.detach().clone().requires_grad_(True)
    position_ids = torch.arange(_SEQ, device=device, dtype=torch.long).unsqueeze(0)
    attention_mask = _sliding_causal_mask(
        batch=_BATCH,
        seq=_SEQ,
        window=lite_config.sliding_window,
        device=device,
        dtype=torch.bfloat16,
    )

    hf_position_embeddings = {
        "compress": hf_rotary(
            hf_hidden, position_ids=position_ids, layer_type="compress"
        )
    }
    hf_output, _hf_attention_weights = hf_attention(
        hf_hidden,
        position_embeddings=hf_position_embeddings,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=None,
    )
    lite_torch_output = lite_torch(lite_torch_hidden, position_ids=position_ids)
    lite_fused_output = lite_fused(lite_fused_hidden, position_ids=position_ids)

    expected_output_shape = (_BATCH, _SEQ, lite_config.hidden_size)
    assert tuple(hf_output.shape) == expected_output_shape
    assert tuple(lite_torch_output.shape) == expected_output_shape
    assert tuple(lite_fused_output.shape) == expected_output_shape
    attention_hf_torch = _assert_hf_parity(
        "attention/hf-vs-lite-torch", hf_output, lite_torch_output
    )
    attention_hf_fused = _assert_hf_parity(
        "attention/hf-vs-lite-fused", hf_output, lite_fused_output
    )
    attention_torch_fused = _assert_lite_backend_parity(
        "attention/lite-torch-vs-fused", lite_torch_output, lite_fused_output
    )

    head_generator = torch.Generator(device="cpu").manual_seed(20260629)
    head_weight = torch.randn(
        _VOCAB, lite_config.hidden_size, generator=head_generator, dtype=torch.float32
    ).mul_(lite_config.hidden_size**-0.5)
    head_weight = head_weight.to(device=device, dtype=torch.bfloat16)
    hf_logits = F.linear(hf_output, head_weight)
    lite_torch_logits = F.linear(lite_torch_output, head_weight)
    lite_fused_logits = F.linear(lite_fused_output, head_weight)
    expected_logits_shape = (_BATCH, _SEQ, _VOCAB)
    assert tuple(hf_logits.shape) == expected_logits_shape
    assert tuple(lite_torch_logits.shape) == expected_logits_shape
    assert tuple(lite_fused_logits.shape) == expected_logits_shape
    logits_hf_torch = _assert_hf_parity(
        "logits/hf-vs-lite-torch", hf_logits, lite_torch_logits
    )
    logits_hf_fused = _assert_hf_parity(
        "logits/hf-vs-lite-fused", hf_logits, lite_fused_logits
    )
    logits_torch_fused = _assert_lite_backend_parity(
        "logits/lite-torch-vs-fused", lite_torch_logits, lite_fused_logits
    )

    # External deterministic targets: never use either output's argmax/self-CE.
    labels = ((torch.arange(_SEQ, device=device) * 17 + 3) % _VOCAB).unsqueeze(0)
    assert tuple(labels.shape) == (_BATCH, _SEQ)
    assert not torch.equal(labels, hf_logits.argmax(dim=-1))
    hf_loss = _fixed_label_ce(hf_logits, labels)
    lite_torch_loss = _fixed_label_ce(lite_torch_logits, labels)
    lite_fused_loss = _fixed_label_ce(lite_fused_logits, labels)
    loss_hf_torch = _assert_loss_parity(
        "fixed-label-CE/hf-vs-lite-torch",
        hf_loss,
        lite_torch_loss,
        abs_max=0.005,
        relative_max=0.001,
    )
    loss_hf_fused = _assert_loss_parity(
        "fixed-label-CE/hf-vs-lite-fused",
        hf_loss,
        lite_fused_loss,
        abs_max=0.005,
        relative_max=0.001,
    )
    loss_torch_fused = _assert_loss_parity(
        "fixed-label-CE/lite-torch-vs-fused",
        lite_torch_loss,
        lite_fused_loss,
        abs_max=0.002,
        relative_max=0.0005,
    )
    hf_loss.backward()
    lite_torch_loss.backward()
    lite_fused_loss.backward()
    assert hf_hidden.grad is not None
    assert lite_torch_hidden.grad is not None
    assert lite_fused_hidden.grad is not None
    input_grad_hf_torch = _assert_hf_parity(
        "input-gradient/hf-vs-lite-torch", hf_hidden.grad, lite_torch_hidden.grad
    )
    input_grad_hf_fused = _assert_hf_parity(
        "input-gradient/hf-vs-lite-fused", hf_hidden.grad, lite_fused_hidden.grad
    )
    input_grad_torch_fused = _assert_lite_backend_parity(
        "input-gradient/lite-torch-vs-fused",
        lite_torch_hidden.grad,
        lite_fused_hidden.grad,
    )
    hf_attention_metrics = (attention_hf_torch, attention_hf_fused)
    hf_logits_metrics = (logits_hf_torch, logits_hf_fused)
    hf_input_grad_metrics = (input_grad_hf_torch, input_grad_hf_fused)
    print(
        "NON_SKIP_DSV4_TRANSFORMERS_512_RATIO4_CSA_PARITY_PASSED "
        f"seq={_SEQ} ratio={_COMPRESS_RATIO} compressed={compressed_entries} "
        f"topk={_INDEX_TOPK} genuine_sparse=True mapped_params=18 "
        f"attention_min_cosine={min(item['cosine'] for item in hf_attention_metrics):.9f} "
        f"attention_max_rms_relative="
        f"{max(item['rms_relative'] for item in hf_attention_metrics):.9f} "
        f"attention_torch_fused_cosine={attention_torch_fused['cosine']:.9f} "
        f"logits_min_cosine={min(item['cosine'] for item in hf_logits_metrics):.9f} "
        f"logits_max_rms_relative="
        f"{max(item['rms_relative'] for item in hf_logits_metrics):.9f} "
        f"logits_torch_fused_cosine={logits_torch_fused['cosine']:.9f} "
        f"loss_hf_torch_abs={loss_hf_torch['absolute']:.9f} "
        f"loss_hf_fused_abs={loss_hf_fused['absolute']:.9f} "
        f"loss_torch_fused_abs={loss_torch_fused['absolute']:.9f} "
        f"input_grad_min_cosine="
        f"{min(item['cosine'] for item in hf_input_grad_metrics):.9f} "
        f"input_grad_max_rms_relative="
        f"{max(item['rms_relative'] for item in hf_input_grad_metrics):.9f} "
        f"input_grad_torch_fused_cosine={input_grad_torch_fused['cosine']:.9f}"
    )
