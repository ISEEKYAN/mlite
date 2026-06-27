# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Qwen3.5 HF<->lite checkpoint NUMERIC round-trip + ground-truth smoke.

The existing export unit tests (tests/unit/model/test_qwen35_export.py) and the
save/load/export coverage smoke only assert NAMES / DTYPE / FINITENESS, never
numeric fidelity — so a transform that is self-consistent on the round-trip but
wrong vs real HF would pass silently. This smoke closes that blind spot:

  test_qwen35_save_load_numeric_roundtrip
    build A(seed1) -> save_hf -> build B(seed2) -> load_hf(B) -> assert every
    param allclose(A, B). Catches ASYMMETRIC load/export bugs (load != inverse
    of export). Vocab embed/head rows >= vocab_size are unused TP-padding (random
    in A, zeroed on reload) and are excluded; the real model has vocab%128==0 so
    no padding exists.

  test_qwen35_transformers_512_checkpoint_ground_truth
    instantiate the official Transformers 5.12 Qwen3.5 MoE implementation with
    one full-attention and one linear-attention layer, save its native packed
    safetensors, load through lite, export, and compare every text tensor.  This
    is independent ground truth: neither the source checkpoint nor the expected
    tensors pass through lite's exporter.

  test_qwen35_transformers_512_full_model_numeric_parity
    reload that official checkpoint into both Transformers and lite, execute the
    same two-sequence batch through the full text model, and compare per-layer
    hidden states, final logits, shifted causal-LM CE, and embedding-output
    gradients under fixed acceptance thresholds.  This closes the semantic gap
    left by a tensor-only checkpoint round-trip.

  test_qwen35_transformers_512_mtp_decoder_equation_parity
    Transformers 5.12 intentionally ignores root ``mtp.*`` weights, so construct
    the released MTP equation explicitly around its independent public
    Qwen3.5 full-attention decoder layer.  The reference reads official-format
    tensors directly (never through lite export), then compares predictor hidden
    states, shared-head logits, shifted CE, and canonical embedding gradients.

  test_qwen35_real_hf_load_export_matches_original  [opt-in: QWEN35_HF_DIR]
    GROUND TRUTH against real Qwen3.5 safetensors: load real HF -> export ->
    compare to the ORIGINAL safetensors tensor-by-tensor. Catches HF-file-level
    mismatches that native round-trip cannot see. It does NOT prove the loaded
    native layout is semantically correct if load/export are mutually inverse but
    both wrong vs forward semantics. Needs an 80GB GPU; loads only the first 8
    decoder layers to bound memory/time.

Run single GPU:
  torchrun --nproc_per_node=1 -m pytest tests/smoke/primitive/test_qwen35_hf_numeric_roundtrip_smoke.py
"""

from __future__ import annotations

import os
import re
import json

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

pytestmark = [pytest.mark.mlite, pytest.mark.smoke, pytest.mark.gpu]

_FP32_CHECKPOINT_SUFFIXES = (
    ".linear_attn.A_log",
    ".linear_attn.norm.weight",
)


def _qwen35_tiny_cfg():
    pytest.importorskip("fla", reason="qwen3_5 needs the FLA / GatedDeltaNet stack.")
    pytest.importorskip(
        "transformer_engine.pytorch",
        reason="qwen3_5 smoke needs real Transformer Engine.",
    )
    from megatron.lite.model.qwen3_5.config import Qwen35Config

    return Qwen35Config(
        num_hidden_layers=2,
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=64,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        linear_num_key_heads=2,
        linear_key_head_dim=4,
        linear_num_value_heads=2,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        layer_types=["full_attention", "linear_attention"],
        partial_rotary_factor=1.0,
        max_position_embeddings=4096,
    )


def _build(cfg, seed, *, optimizer="dist_opt", mtp_enable=False):
    from megatron.lite.model.qwen3_5.lite import protocol
    from megatron.lite.runtime.contracts.config import ParallelConfig

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    parallel = ParallelConfig(tp=1, pp=1, cp=1, ep=1, etp=1, vpp=1)
    impl = protocol.ImplConfig(
        parallel=parallel,
        optimizer=optimizer,
        use_deepep=False,
        deterministic=True,
        mtp_enable=mtp_enable,
    )
    bundle = protocol.build_model(cfg, impl_cfg=impl)
    return bundle, protocol


def _named(chunks):
    out = {}
    for i, chunk in enumerate(chunks):
        for name, param in chunk.named_parameters():
            out[f"{i}.{name}"] = param.detach().float().cpu().clone()
    return out


def _read_safetensors_dir(path):
    from safetensors import safe_open

    tensors = {}
    for filename in sorted(os.listdir(path)):
        if not filename.endswith(".safetensors"):
            continue
        with safe_open(
            os.path.join(path, filename), framework="pt", device="cpu"
        ) as handle:
            for key in handle.keys():
                assert key not in tensors, f"duplicate safetensors key: {key}"
                tensors[key] = handle.get_tensor(key)
    assert tensors, f"no safetensors found under {path}"
    return tensors


def _save_transformers_512_tiny_qwen35(path):
    """Create an official tiny Qwen3.5 MoE checkpoint with production key layout."""
    import transformers
    from transformers import (
        Qwen3_5MoeConfig,
        Qwen3_5MoeForConditionalGeneration,
        Qwen3_5MoeTextConfig,
        Qwen3_5MoeVisionConfig,
    )

    assert transformers.__version__ == "5.12.0", (
        "the Qwen3.5 HF ground-truth contract is pinned to Transformers 5.12.0; "
        f"got {transformers.__version__}"
    )
    text_config = Qwen3_5MoeTextConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=4096,
        linear_num_key_heads=2,
        linear_key_head_dim=4,
        linear_num_value_heads=2,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts=4,
        num_experts_per_tok=2,
        layer_types=["full_attention", "linear_attention"],
        partial_rotary_factor=1.0,
        rope_parameters={"rope_type": "default", "rope_theta": 10_000_000.0},
        tie_word_embeddings=False,
    )
    # The public checkpoint is multimodal, hence its text tensors use the real
    # `model.language_model.*` namespace consumed by lite.  Keep the unused
    # vision tower tiny; the comparison below deliberately scopes to text.
    vision_config = Qwen3_5MoeVisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=4,
        in_channels=3,
        patch_size=2,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=16,
        num_position_embeddings=16,
    )
    config = Qwen3_5MoeConfig(
        text_config=text_config,
        vision_config=vision_config,
        tie_word_embeddings=False,
    )
    torch.manual_seed(20260627)
    model = Qwen3_5MoeForConditionalGeneration(config).to(torch.bfloat16)
    model.save_pretrained(
        path,
        safe_serialization=True,
        max_shard_size="1GB",
        # Transformers defaults to a reverse-converted legacy per-expert
        # representation.  False preserves both its native 5.12 packed layout
        # and the packed layout of Qwen/Qwen3.5-35B-A3B on the Hub.
        save_original_format=False,
    )

    # The strict Transformers 5.12 text-config constructor does not accept the
    # released checkpoint's auxiliary field, although from_pretrained tolerates
    # it in serialized Hub configs. Add it through the real config.json shape.
    config_path = os.path.join(path, "config.json")
    with open(config_path) as handle:
        serialized_config = json.load(handle)
    serialized_config["text_config"]["mtp_num_hidden_layers"] = 1
    with open(config_path, "w") as handle:
        json.dump(serialized_config, handle, indent=2, sort_keys=True)
        handle.write("\n")

    # Transformers 5.12 intentionally ignores the released checkpoint's
    # auxiliary `mtp.*` tensors at model load time. Add a deterministic tiny
    # predictor in the exact Qwen/Qwen3.5-35B-A3B on-disk schema so lite's HF
    # loader/exporter is still tested against that production contract.
    from safetensors.torch import save_file

    assert not os.path.exists(os.path.join(path, "model.safetensors.index.json"))
    tensors = _read_safetensors_dir(path)
    for key, tensor in list(tensors.items()):
        if key.endswith(_FP32_CHECKPOINT_SUFFIXES):
            tensors[key] = tensor.float()
    layer_prefix = "model.language_model.layers.0"
    mtp_layer_prefix = "mtp.layers.0"
    for suffix in (
        "input_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "self_attn.o_proj.weight",
        "post_attention_layernorm.weight",
        "mlp.gate.weight",
        "mlp.shared_expert.gate_proj.weight",
        "mlp.shared_expert.up_proj.weight",
        "mlp.shared_expert.down_proj.weight",
        "mlp.shared_expert_gate.weight",
    ):
        tensors[f"{mtp_layer_prefix}.{suffix}"] = tensors[
            f"{layer_prefix}.{suffix}"
        ].clone()

    packed_gate_up = tensors[f"{layer_prefix}.mlp.experts.gate_up_proj"]
    packed_down = tensors[f"{layer_prefix}.mlp.experts.down_proj"]
    for expert_idx in range(text_config.num_experts):
        gate, up = packed_gate_up[expert_idx].chunk(2, dim=0)
        expert_prefix = f"{mtp_layer_prefix}.mlp.experts.{expert_idx}"
        tensors[f"{expert_prefix}.gate_proj.weight"] = gate.contiguous().clone()
        tensors[f"{expert_prefix}.up_proj.weight"] = up.contiguous().clone()
        tensors[f"{expert_prefix}.down_proj.weight"] = (
            packed_down[expert_idx].contiguous().clone()
        )

    tensors["mtp.pre_fc_norm_embedding.weight"] = tensors[
        f"{layer_prefix}.input_layernorm.weight"
    ].clone()
    tensors["mtp.pre_fc_norm_hidden.weight"] = tensors[
        f"{layer_prefix}.post_attention_layernorm.weight"
    ].clone()
    tensors["mtp.norm.weight"] = tensors["model.language_model.norm.weight"].clone()
    tensors["mtp.fc.weight"] = (
        torch.arange(
            text_config.hidden_size * 2 * text_config.hidden_size,
            dtype=torch.float32,
        )
        .reshape(text_config.hidden_size, 2 * text_config.hidden_size)
        .to(torch.bfloat16)
    )
    save_file(tensors, os.path.join(path, "model.safetensors"))


def _force_hf_torch_reference_kernels(model) -> None:
    """Use deterministic HF reference kernels instead of optional FLA fast paths."""
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as hf_impl

    text_model = model.model.language_model
    text_model.config._attn_implementation = "eager"
    text_model.config._experts_implementation = "eager"
    linear_layers = 0
    for layer in text_model.layers:
        # The Transformers decorator otherwise dispatches to grouped-mm on GB300.
        # Keep the authority lane on its explicit per-expert Torch implementation.
        layer.mlp.experts.config._experts_implementation = "eager"
        linear_attn = getattr(layer, "linear_attn", None)
        if linear_attn is None:
            continue
        linear_layers += 1
        linear_attn.causal_conv1d_fn = None
        linear_attn.chunk_gated_delta_rule = hf_impl.torch_chunk_gated_delta_rule
        linear_attn.recurrent_gated_delta_rule = (
            hf_impl.torch_recurrent_gated_delta_rule
        )
        # FLA availability is process-global and would otherwise instantiate its
        # fused gated RMSNorm inside the HF reference model. Replace it with the
        # Transformers implementation while preserving the checkpoint weight.
        reference_norm = hf_impl.Qwen3_5MoeRMSNormGated(
            linear_attn.head_v_dim,
            eps=linear_attn.layer_norm_epsilon,
        ).to(
            device=linear_attn.norm.weight.device,
            dtype=linear_attn.norm.weight.dtype,
        )
        reference_norm.load_state_dict(linear_attn.norm.state_dict(), strict=True)
        linear_attn.norm = reference_norm
    assert linear_layers == 1, (
        f"expected one linear-attention layer, got {linear_layers}"
    )


def _build_transformers_512_mtp_decoder_equation_reference(source_dir: str):
    """Build the released Qwen3.5 MTP equation from independent HF modules.

    Transformers 5.12 deliberately has no root MTP wrapper and ignores ``mtp.*``
    while loading a causal LM.  Its Qwen3.5 decoder layer is nevertheless an
    independent authority for the predictor block.  This helper wires the four
    released root tensors around that layer and loads every tensor directly from
    the official on-disk namespace.  No lite checkpoint/export mapping is used.
    """
    from transformers import Qwen3_5MoeConfig
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as hf_impl

    source = _read_safetensors_dir(source_dir)
    text_config = Qwen3_5MoeConfig.from_pretrained(source_dir).text_config
    assert text_config.layer_types[0] == "full_attention"
    text_config._attn_implementation = "eager"
    text_config._experts_implementation = "eager"

    reference = torch.nn.ModuleDict(
        {
            "embedding": torch.nn.Embedding(
                text_config.vocab_size, text_config.hidden_size
            ),
            "enorm": hf_impl.Qwen3_5MoeRMSNorm(
                text_config.hidden_size, eps=text_config.rms_norm_eps
            ),
            "hnorm": hf_impl.Qwen3_5MoeRMSNorm(
                text_config.hidden_size, eps=text_config.rms_norm_eps
            ),
            "fc": torch.nn.Linear(
                2 * text_config.hidden_size,
                text_config.hidden_size,
                bias=False,
            ),
            "layer": hf_impl.Qwen3_5MoeDecoderLayer(text_config, layer_idx=0),
            "final_norm": hf_impl.Qwen3_5MoeRMSNorm(
                text_config.hidden_size, eps=text_config.rms_norm_eps
            ),
            "rotary": hf_impl.Qwen3_5MoeTextRotaryEmbedding(text_config),
            "head": torch.nn.Linear(
                text_config.hidden_size, text_config.vocab_size, bias=False
            ),
        }
    ).to(device="cuda", dtype=torch.bfloat16)
    reference["layer"].mlp.experts.config._experts_implementation = "eager"

    layer_prefix = "mtp.layers.0"
    direct_layer_suffixes = (
        "input_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "self_attn.o_proj.weight",
        "post_attention_layernorm.weight",
        "mlp.gate.weight",
        "mlp.shared_expert.gate_proj.weight",
        "mlp.shared_expert.up_proj.weight",
        "mlp.shared_expert.down_proj.weight",
        "mlp.shared_expert_gate.weight",
    )
    layer_state = {
        suffix: source[f"{layer_prefix}.{suffix}"] for suffix in direct_layer_suffixes
    }
    layer_state["mlp.experts.gate_up_proj"] = torch.stack(
        [
            torch.cat(
                [
                    source[f"{layer_prefix}.mlp.experts.{expert_idx}.gate_proj.weight"],
                    source[f"{layer_prefix}.mlp.experts.{expert_idx}.up_proj.weight"],
                ],
                dim=0,
            )
            for expert_idx in range(text_config.num_experts)
        ],
        dim=0,
    ).contiguous()
    layer_state["mlp.experts.down_proj"] = torch.stack(
        [
            source[f"{layer_prefix}.mlp.experts.{expert_idx}.down_proj.weight"]
            for expert_idx in range(text_config.num_experts)
        ],
        dim=0,
    ).contiguous()
    reference["layer"].load_state_dict(layer_state, strict=True)

    root_state = {
        "embedding": "model.language_model.embed_tokens.weight",
        "enorm": "mtp.pre_fc_norm_embedding.weight",
        "hnorm": "mtp.pre_fc_norm_hidden.weight",
        "fc": "mtp.fc.weight",
        "final_norm": "mtp.norm.weight",
        "head": "lm_head.weight",
    }
    for module_name, source_name in root_state.items():
        reference[module_name].load_state_dict(
            {"weight": source[source_name]}, strict=True
        )
    reference.eval()
    return reference


def _forward_transformers_512_mtp_decoder_equation(
    reference,
    *,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    encoder_hidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate one released MTP depth in batch-first layout."""
    shifted_ids = torch.roll(input_ids, shifts=-1, dims=-1)
    shifted_ids[:, -1] = 0

    decoder_input = reference["enorm"](reference["embedding"](shifted_ids))
    encoder_input = reference["hnorm"](encoder_hidden)
    projected = reference["fc"](torch.cat((decoder_input, encoder_input), dim=-1))

    position_embeddings = reference["rotary"](projected, position_ids)
    sequence_length = projected.shape[1]
    causal_mask = torch.full(
        (sequence_length, sequence_length),
        torch.finfo(projected.dtype).min,
        dtype=projected.dtype,
        device=projected.device,
    )
    causal_mask = torch.triu(causal_mask, diagonal=1)[None, None, :, :]
    predictor = reference["layer"](
        projected,
        position_embeddings=position_embeddings,
        attention_mask=causal_mask,
        position_ids=position_ids[0],
        use_cache=False,
    )
    hidden = reference["final_norm"](predictor)
    return hidden, reference["head"](hidden), shifted_ids


def _shifted_causal_ce(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    assert logits.ndim == 3
    assert input_ids.ndim == 2
    assert logits.shape[:2] == input_ids.shape
    assert logits.shape[1] >= 2
    return F.cross_entropy(
        logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
        input_ids[:, 1:].reshape(-1),
    )


def _roll_left_zero_reference(tensor: torch.Tensor) -> torch.Tensor:
    """Independent dense equivalent of one MTP left shift."""
    rolled = torch.roll(tensor, shifts=-1, dims=-1).clone()
    rolled[..., -1] = 0
    return rolled


def _mtp_depth1_ce_from_raw_tokens(
    logits: torch.Tensor, raw_input_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Depth-1 MTP CE targets raw token i+2, never the visible i+1 token.

    Main causal labels already target the next raw token.  Qwen MTP rolls those
    labels and their validity mask once more, matching ``_apply_mtp_loss`` while
    keeping this reference independent from lite's roll helper.
    """
    assert logits.ndim == 3
    assert raw_input_ids.ndim == 2
    assert logits.shape[:2] == raw_input_ids.shape
    assert logits.shape[1] >= 3

    main_labels = _roll_left_zero_reference(raw_input_ids)
    main_loss_mask = torch.ones_like(raw_input_ids, dtype=torch.float32)
    main_loss_mask[..., -1] = 0
    mtp_labels = _roll_left_zero_reference(main_labels)
    mtp_loss_mask = _roll_left_zero_reference(main_loss_mask)
    token_loss = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        mtp_labels.reshape(-1),
        reduction="none",
    ).reshape_as(mtp_loss_mask)
    loss = (token_loss * mtp_loss_mask).sum() / mtp_loss_mask.sum()
    return loss, mtp_labels, mtp_loss_mask


def _assert_tensor_parity(
    name: str,
    hf_value: torch.Tensor,
    lite_value: torch.Tensor,
    *,
    cosine_min: float,
    rms_relative_max: float,
    max_abs_max: float,
    norm_ratio_min: float = 0.99,
    norm_ratio_max: float = 1.01,
) -> dict[str, float]:
    assert lite_value.shape == hf_value.shape, (
        f"{name}: lite shape {tuple(lite_value.shape)} != HF shape {tuple(hf_value.shape)}"
    )
    hf = hf_value.detach().float()
    lite = lite_value.detach().float()
    assert torch.isfinite(hf).all(), f"{name}: HF produced non-finite values"
    assert torch.isfinite(lite).all(), f"{name}: lite produced non-finite values"

    hf_flat = hf.reshape(-1)
    lite_flat = lite.reshape(-1)
    hf_norm = torch.linalg.vector_norm(hf_flat)
    lite_norm = torch.linalg.vector_norm(lite_flat)
    assert hf_norm > 0 and lite_norm > 0, (
        f"{name}: zero-norm comparison is not meaningful"
    )
    cosine = F.cosine_similarity(hf_flat, lite_flat, dim=0).item()
    diff_rms = torch.sqrt(torch.mean((hf - lite).square()))
    symmetric_scale = torch.maximum(
        torch.sqrt(torch.mean(hf.square())), torch.sqrt(torch.mean(lite.square()))
    ).clamp_min(torch.finfo(torch.float32).tiny)
    rms_relative = (diff_rms / symmetric_scale).item()
    max_abs = (hf - lite).abs().max().item()
    norm_ratio = (lite_norm / hf_norm).item()
    print(
        f"[qwen35-hf-parity] {name}: shape={tuple(hf.shape)} "
        f"cosine={cosine:.9f} rms_relative={rms_relative:.9f} "
        f"norm_ratio={norm_ratio:.9f} max_abs={max_abs:.9f}"
    )

    assert cosine >= cosine_min, f"{name}: cosine {cosine} < {cosine_min}"
    assert rms_relative <= rms_relative_max, (
        f"{name}: RMS-relative {rms_relative} > {rms_relative_max}"
    )
    assert norm_ratio_min <= norm_ratio <= norm_ratio_max, (
        f"{name}: norm ratio {norm_ratio} outside [{norm_ratio_min}, {norm_ratio_max}]"
    )
    assert max_abs <= max_abs_max, f"{name}: max-abs {max_abs} > {max_abs_max}"
    return {
        "cosine": cosine,
        "rms_relative": rms_relative,
        "norm_ratio": norm_ratio,
        "max_abs": max_abs,
    }


def _layer_capture(layers, *, sequence_first: bool):
    captured: dict[int, torch.Tensor] = {}
    handles = []

    for index, layer in enumerate(layers):

        def _hook(_module, _args, output, *, layer_index=index):
            value = output[0] if isinstance(output, tuple) else output
            assert isinstance(value, torch.Tensor)
            if sequence_first:
                value = value.transpose(0, 1)
            captured[layer_index] = value.detach().float().clone()

        handles.append(layer.register_forward_hook(_hook))
    return captured, handles


def _embedding_grad_capture(embedding):
    captured: list[torch.Tensor] = []

    def _hook(_module, _args, output):
        assert isinstance(output, torch.Tensor)
        output.retain_grad()
        captured.append(output)

    return captured, embedding.register_forward_hook(_hook)


@pytest.fixture(scope="module", autouse=True)
def _single_rank_dist():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for qwen3.5 numeric round-trip smoke.")
    created = False
    if not dist.is_initialized():
        for k, v in {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29597",
            "RANK": "0",
            "WORLD_SIZE": "1",
            "LOCAL_RANK": "0",
        }.items():
            os.environ.setdefault(k, v)
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        created = True
    yield
    if created and dist.is_initialized():
        dist.destroy_process_group()


def test_qwen35_save_load_numeric_roundtrip(tmp_path):
    if dist.get_world_size() != 1:
        pytest.skip("numeric round-trip smoke runs single-rank (tp1).")
    cfg = _qwen35_tiny_cfg()
    a, proto = _build(cfg, seed=1)
    b, _ = _build(cfg, seed=2)

    before = _named(a.chunks)
    b_before = _named(b.chunks)
    assert sum(1 for k in before if not torch.equal(before[k], b_before[k])) > 0

    out_dir = str(tmp_path / "hf")
    os.makedirs(out_dir, exist_ok=True)
    proto.save_hf_weights(a.chunks, out_dir, cfg, a.parallel_state)
    proto.load_hf_weights(b.chunks[0], out_dir, cfg, b.parallel_state)

    pa, pb = _named(a.chunks), _named(b.chunks)
    vocab = cfg.vocab_size
    mism = []
    for k in pa:
        ta, tb = pa[k], pb[k]
        if k.endswith("embed.embedding.weight") or k.endswith("head.col.linear.weight"):
            assert ta.shape[0] >= vocab
            ta, tb = ta[:vocab], tb[:vocab]  # exclude unused TP-padding rows
        # atol/rtol=1e-3 are bf16-cast round-trip tolerances.
        if not torch.allclose(ta, tb, atol=1e-3, rtol=1e-3):
            mism.append(f"{k} (max_abs_diff={(ta - tb).abs().max().item()})")
    assert not mism, "HF save->load not numeric round-trip:\n" + "\n".join(mism)


def test_qwen35_transformers_512_checkpoint_ground_truth(tmp_path):
    """Compare lite load/export directly with official HF-created safetensors."""
    if dist.get_world_size() != 1:
        pytest.skip("Transformers ground-truth smoke runs single-rank (tp1).")
    pytest.importorskip("transformers")
    pytest.importorskip("safetensors")

    from megatron.lite.model.qwen3_5.config import Qwen35Config

    source_dir = str(tmp_path / "transformers_hf")
    exported_dir = str(tmp_path / "lite_export")
    os.makedirs(source_dir)
    os.makedirs(exported_dir)
    _save_transformers_512_tiny_qwen35(source_dir)

    source = _read_safetensors_dir(source_dir)
    expected = {
        key: tensor
        for key, tensor in source.items()
        if key == "lm_head.weight"
        or key.startswith("model.language_model.")
        or key.startswith("mtp.")
    }
    assert expected
    assert "model.language_model.layers.0.self_attn.q_proj.weight" in expected
    assert "model.language_model.layers.1.linear_attn.in_proj_qkv.weight" in expected
    for layer_idx in range(2):
        prefix = f"model.language_model.layers.{layer_idx}.mlp.experts"
        assert f"{prefix}.gate_up_proj" in expected
        assert f"{prefix}.down_proj" in expected
    assert not any(
        re.search(r"\.mlp\.experts\.\d+\.(gate|up|down)_proj\.weight$", key)
        for key in expected
        if key.startswith("model.language_model.")
    ), (
        "Transformers emitted legacy per-expert tensors instead of the production packed layout"
    )
    mtp_expected = {key for key in expected if key.startswith("mtp.")}
    assert {
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.fc.weight",
        "mtp.norm.weight",
    } <= mtp_expected
    expected_fp32 = {key for key in expected if key.endswith(_FP32_CHECKPOINT_SUFFIXES)}
    assert expected_fp32 == {
        "model.language_model.layers.1.linear_attn.A_log",
        "model.language_model.layers.1.linear_attn.norm.weight",
    }
    assert all(expected[key].dtype == torch.float32 for key in expected_fp32)
    assert all(
        tensor.dtype == torch.bfloat16
        for key, tensor in expected.items()
        if key not in expected_fp32
    )
    assert (
        expected["model.language_model.layers.1.linear_attn.dt_bias"].dtype
        == torch.bfloat16
    )

    cfg = Qwen35Config.from_hf(source_dir)
    assert cfg.layer_types == ["full_attention", "linear_attention"]
    assert cfg.num_nextn_predict_layers == 1
    assert cfg.mtp_layer_types == ["full_attention"]
    assert len(mtp_expected) == 17 + 3 * cfg.num_experts
    for expert_idx in range(cfg.num_experts):
        prefix = f"mtp.layers.0.mlp.experts.{expert_idx}"
        assert {
            f"{prefix}.gate_proj.weight",
            f"{prefix}.up_proj.weight",
            f"{prefix}.down_proj.weight",
        } <= mtp_expected
    assert "mtp.layers.0.mlp.experts.gate_up_proj" not in mtp_expected
    assert "mtp.layers.0.mlp.experts.down_proj" not in mtp_expected

    bundle, protocol = _build(cfg, seed=20260628, mtp_enable=True)
    protocol.load_hf_weights(bundle.chunks[0], source_dir, cfg, bundle.parallel_state)
    protocol.save_hf_weights(bundle.chunks, exported_dir, cfg, bundle.parallel_state)
    exported = _read_safetensors_dir(exported_dir)

    assert set(exported) == set(expected), (
        f"missing={sorted(set(expected) - set(exported))}; "
        f"unexpected={sorted(set(exported) - set(expected))}"
    )
    mismatches = []
    for key in sorted(expected):
        want, got = expected[key], exported[key]
        if want.shape != got.shape:
            mismatches.append(f"{key}: shape {tuple(got.shape)} != {tuple(want.shape)}")
        elif want.dtype != got.dtype:
            mismatches.append(f"{key}: dtype {got.dtype} != {want.dtype}")
        elif not torch.equal(want, got):
            max_abs = (want.float() - got.float()).abs().max().item()
            mismatches.append(f"{key}: values differ (max_abs={max_abs})")
    assert not mismatches, (
        "lite does not preserve Transformers 5.12 HF tensors:\n" + "\n".join(mismatches)
    )
    print(
        "QWEN35_HF_MTP_SCHEMA_DTYPE_PARITY "
        f"mtp_keys={len(mtp_expected)} "
        "A_log=torch.float32 norm=torch.float32 dt_bias=torch.bfloat16 exact=True"
    )


def test_qwen35_transformers_512_mtp_decoder_equation_parity(tmp_path):
    """Gate MTP semantics with HF's decoder layer plus the released equation.

    This is intentionally not described as a Transformers MTP-model comparison:
    Transformers 5.12 drops root ``mtp.*`` weights.  The independent side loads
    the official-format tensors directly into HF Qwen3.5 primitives and spells
    out shifted embedding -> pre norms -> fc -> predictor -> final norm -> head.
    """
    if dist.get_world_size() != 1:
        pytest.skip("Transformers MTP equation parity runs single-rank (tp1).")
    pytest.importorskip("transformers")
    pytest.importorskip("safetensors")

    import transformers

    from megatron.lite.model.qwen3_5.config import Qwen35Config

    assert transformers.__version__ == "5.12.0", (
        "the Qwen3.5 MTP decoder authority is pinned to Transformers 5.12.0; "
        f"got {transformers.__version__}"
    )
    torch.manual_seed(20260630)
    torch.cuda.manual_seed_all(20260630)

    source_dir = str(tmp_path / "transformers_hf_mtp_equation")
    os.makedirs(source_dir)
    _save_transformers_512_tiny_qwen35(source_dir)
    reference = _build_transformers_512_mtp_decoder_equation_reference(source_dir)

    cfg = Qwen35Config.from_hf(source_dir)
    assert cfg.num_nextn_predict_layers == 1
    assert cfg.mtp_layer_types == ["full_attention"]
    lite_bundle, protocol = _build(cfg, seed=20260701, optimizer=None, mtp_enable=True)
    lite_model = lite_bundle.chunks[0]
    protocol.load_hf_weights(lite_model, source_dir, cfg, lite_bundle.parallel_state)
    lite_model.eval()
    assert lite_model.embed is not None
    assert lite_model.head is not None
    assert lite_model.mtp is not None
    assert len(lite_model.mtp.layers) == 1
    lite_mtp_layer = lite_model.mtp.layers[0]
    assert lite_mtp_layer.embedding is lite_model.embed

    input_ids = torch.tensor(
        [
            [3, 5, 7, 11, 13, 17, 19, 23],
            [29, 31, 37, 41, 43, 47, 53, 59],
        ],
        dtype=torch.long,
        device="cuda",
    )
    batch_size, sequence_length = input_ids.shape
    position_ids = (
        torch.arange(sequence_length, device=input_ids.device)
        .view(1, 1, -1)
        .expand(3, batch_size, -1)
        .contiguous()
    )
    encoder_hidden = torch.sin(
        torch.linspace(
            -1.25,
            1.75,
            steps=batch_size * sequence_length * cfg.hidden_size,
            dtype=torch.float32,
            device="cuda",
        )
    ).reshape(batch_size, sequence_length, cfg.hidden_size)
    encoder_hidden = encoder_hidden.to(torch.bfloat16)

    reference.zero_grad(set_to_none=True)
    lite_model.zero_grad(set_to_none=True)
    reference_hidden, reference_logits, reference_shifted_ids = (
        _forward_transformers_512_mtp_decoder_equation(
            reference,
            input_ids=input_ids,
            position_ids=position_ids,
            encoder_hidden=encoder_hidden,
        )
    )
    lite_hidden_sbh, lite_shifted_ids, lite_shifted_positions = lite_mtp_layer(
        input_ids=input_ids,
        position_ids=position_ids,
        hidden_states=encoder_hidden.transpose(0, 1).contiguous(),
        rotary_position_ids=position_ids,
        packed_seq_params=None,
    )
    lite_hidden = lite_hidden_sbh.transpose(0, 1).contiguous()
    lite_logits = lite_model.head.gather(lite_model.head(lite_hidden_sbh))
    lite_logits = lite_logits.transpose(0, 1).contiguous()

    expected_shifted_positions = torch.roll(position_ids, shifts=-1, dims=-1)
    expected_shifted_positions[..., -1] = 0
    assert torch.equal(reference_shifted_ids, lite_shifted_ids)
    assert lite_shifted_positions is not None
    assert torch.equal(lite_shifted_positions, expected_shifted_positions)
    hidden_metrics = _assert_tensor_parity(
        "mtp-predictor-hidden",
        reference_hidden,
        lite_hidden,
        cosine_min=0.999,
        rms_relative_max=0.02,
        max_abs_max=0.05,
    )
    logits_metrics = _assert_tensor_parity(
        "mtp-shared-head-logits",
        reference_logits,
        lite_logits,
        cosine_min=0.999,
        rms_relative_max=0.02,
        max_abs_max=0.05,
    )

    reference_ce, reference_mtp_labels, reference_mtp_mask = (
        _mtp_depth1_ce_from_raw_tokens(reference_logits, input_ids)
    )
    lite_ce, lite_mtp_labels, lite_mtp_mask = _mtp_depth1_ce_from_raw_tokens(
        lite_logits, input_ids
    )
    assert torch.equal(reference_mtp_labels, lite_mtp_labels)
    assert torch.equal(reference_mtp_mask, lite_mtp_mask)
    assert torch.equal(reference_mtp_labels[:, :-2], input_ids[:, 2:])
    assert torch.count_nonzero(reference_mtp_mask[:, -2:]).item() == 0
    assert reference_mtp_mask.sum().item() == batch_size * (sequence_length - 2)
    ce_abs = (reference_ce - lite_ce).abs().item()
    ce_scale = max(abs(reference_ce.item()), abs(lite_ce.item()), 1.0e-12)
    ce_relative = ce_abs / ce_scale
    print(
        "[qwen35-mtp-equation-parity] depth-1 shifted-CE (raw i+2 target): "
        f"hf_equation={reference_ce.item():.9f} lite={lite_ce.item():.9f} "
        f"abs={ce_abs:.9f} relative={ce_relative:.9f}"
    )
    assert ce_abs <= 0.005, f"MTP shifted CE absolute error {ce_abs} > 0.005"
    assert ce_relative <= 0.001, f"MTP shifted CE relative error {ce_relative} > 0.001"

    reference_ce.backward()
    lite_ce.backward()
    reference_embedding_grad = reference["embedding"].weight.grad
    lite_embedding_grad = lite_model.embed.embedding.weight.grad
    assert reference_embedding_grad is not None
    assert lite_embedding_grad is not None
    lite_embedding_grad = lite_embedding_grad[: cfg.vocab_size]
    assert torch.count_nonzero(reference_embedding_grad).item() > 0
    assert torch.count_nonzero(lite_embedding_grad).item() > 0
    embedding_grad_metrics = _assert_tensor_parity(
        "mtp-canonical-embedding-gradient",
        reference_embedding_grad,
        lite_embedding_grad,
        cosine_min=0.999,
        rms_relative_max=0.02,
        max_abs_max=0.01,
    )
    reference_grad_norm = torch.linalg.vector_norm(
        reference_embedding_grad.float()
    ).item()
    lite_grad_norm = torch.linalg.vector_norm(lite_embedding_grad.float()).item()
    grad_norm_ratio = lite_grad_norm / reference_grad_norm
    assert 0.99 <= grad_norm_ratio <= 1.01, (
        "MTP canonical embedding gradient norm ratio "
        f"{grad_norm_ratio} is outside [0.99, 1.01]"
    )
    print(
        "NON_SKIP_QWEN35_TRANSFORMERS_512_MTP_DECODER_EQUATION_PARITY_PASSED "
        "authority=transformers_decoder_plus_released_root_equation "
        "mtp_layers=1 layer_type=full_attention depth2_target=True batch=2 seq=8 "
        f"hidden_cosine={hidden_metrics['cosine']:.9f} "
        f"hidden_rms_relative={hidden_metrics['rms_relative']:.9f} "
        f"logits_cosine={logits_metrics['cosine']:.9f} "
        f"logits_rms_relative={logits_metrics['rms_relative']:.9f} "
        f"ce_abs={ce_abs:.9f} ce_relative={ce_relative:.9f} "
        f"embedding_grad_cosine={embedding_grad_metrics['cosine']:.9f} "
        f"embedding_grad_rms_relative="
        f"{embedding_grad_metrics['rms_relative']:.9f} "
        f"embedding_grad_norm_ratio={grad_norm_ratio:.9f}"
    )


def test_qwen35_transformers_512_full_model_numeric_parity(tmp_path):
    """Gate real HF-vs-lite semantics, not merely checkpoint key inversion."""
    if dist.get_world_size() != 1:
        pytest.skip("Transformers full-model parity runs single-rank (tp1).")
    pytest.importorskip("transformers")
    pytest.importorskip("safetensors")

    import transformers
    from transformers import Qwen3_5MoeForConditionalGeneration

    from megatron.lite.model.qwen3_5.config import Qwen35Config

    assert transformers.__version__ == "5.12.0", (
        "the Qwen3.5 HF numeric authority is pinned to Transformers 5.12.0; "
        f"got {transformers.__version__}"
    )
    torch.manual_seed(20260629)
    torch.cuda.manual_seed_all(20260629)

    source_dir = str(tmp_path / "transformers_hf_numeric")
    os.makedirs(source_dir)
    _save_transformers_512_tiny_qwen35(source_dir)

    hf_model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        source_dir,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).cuda()
    _force_hf_torch_reference_kernels(hf_model)
    hf_model.eval()

    cfg = Qwen35Config.from_hf(source_dir)
    assert cfg.layer_types == ["full_attention", "linear_attention"]
    lite_bundle, protocol = _build(cfg, seed=20260630, optimizer=None)
    lite_model = lite_bundle.chunks[0]
    protocol.load_hf_weights(lite_model, source_dir, cfg, lite_bundle.parallel_state)
    lite_model.eval()

    # Fixed, non-padding token batch.  Tokens are deliberately unique so the
    # embedding-output gradient probe is not obscured by repeated-row reduction.
    input_ids = torch.tensor(
        [
            [3, 5, 7, 11, 13, 17, 19, 23],
            [29, 31, 37, 41, 43, 47, 53, 59],
        ],
        dtype=torch.long,
        device="cuda",
    )
    attention_mask = torch.ones_like(input_ids)
    position_ids = (
        torch.arange(input_ids.shape[1], device=input_ids.device)
        .view(1, 1, -1)
        .expand(3, input_ids.shape[0], -1)
        .contiguous()
    )
    expected_hidden_shape = (2, 8, 16)
    expected_logits_shape = (2, 8, 64)

    hf_layers, hf_layer_handles = _layer_capture(
        hf_model.model.language_model.layers, sequence_first=False
    )
    lite_layers, lite_layer_handles = _layer_capture(
        lite_model.layers, sequence_first=True
    )
    hf_embeddings, hf_embedding_handle = _embedding_grad_capture(
        hf_model.model.language_model.embed_tokens
    )
    assert lite_model.embed is not None
    lite_embeddings, lite_embedding_handle = _embedding_grad_capture(
        lite_model.embed.embedding
    )

    try:
        hf_output = hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=input_ids,
            use_cache=False,
            output_router_logits=False,
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
            *hf_layer_handles,
            *lite_layer_handles,
            hf_embedding_handle,
            lite_embedding_handle,
        ]:
            handle.remove()

    hf_logits = hf_output.logits
    lite_logits = lite_output["logits"]
    assert tuple(hf_logits.shape) == expected_logits_shape
    assert tuple(lite_logits.shape) == expected_logits_shape
    assert set(hf_layers) == {0, 1}
    assert set(lite_layers) == {0, 1}
    layer_metrics = []
    for layer_index, layer_type in enumerate(cfg.layer_types):
        assert tuple(hf_layers[layer_index].shape) == expected_hidden_shape
        assert tuple(lite_layers[layer_index].shape) == expected_hidden_shape
        layer_metrics.append(
            _assert_tensor_parity(
                f"layer-{layer_index}-{layer_type}",
                hf_layers[layer_index],
                lite_layers[layer_index],
                cosine_min=0.999,
                rms_relative_max=0.02,
                max_abs_max=0.05,
            )
        )

    logits_metrics = _assert_tensor_parity(
        "final-logits",
        hf_logits,
        lite_logits,
        cosine_min=0.999,
        rms_relative_max=0.02,
        max_abs_max=0.05,
    )

    hf_shifted_ce = _shifted_causal_ce(hf_logits, input_ids)
    lite_shifted_ce = _shifted_causal_ce(lite_logits, input_ids)
    assert hf_output.loss is not None
    torch.testing.assert_close(
        hf_output.loss.float(), hf_shifted_ce, atol=1e-6, rtol=1e-6
    )
    loss_abs = (hf_shifted_ce - lite_shifted_ce).abs().item()
    loss_scale = max(abs(hf_shifted_ce.item()), abs(lite_shifted_ce.item()), 1e-12)
    loss_relative = loss_abs / loss_scale
    print(
        "[qwen35-hf-parity] shifted-CE: "
        f"hf={hf_shifted_ce.item():.9f} lite={lite_shifted_ce.item():.9f} "
        f"abs={loss_abs:.9f} relative={loss_relative:.9f}"
    )
    assert loss_abs <= 0.005, f"shifted CE absolute error {loss_abs} > 0.005"
    assert loss_relative <= 0.001, f"shifted CE relative error {loss_relative} > 0.001"

    hf_shifted_ce.backward()
    lite_shifted_ce.backward()
    assert len(hf_embeddings) == 1
    assert len(lite_embeddings) == 1
    hf_embedding_grad = hf_embeddings[0].grad
    lite_embedding_grad = lite_embeddings[0].grad
    assert hf_embedding_grad is not None
    assert lite_embedding_grad is not None
    embedding_grad_metrics = _assert_tensor_parity(
        "embedding-output-gradient",
        hf_embedding_grad,
        lite_embedding_grad,
        cosine_min=0.999,
        rms_relative_max=0.02,
        max_abs_max=0.01,
    )
    hf_grad_norm = torch.linalg.vector_norm(hf_embedding_grad.float()).item()
    lite_grad_norm = torch.linalg.vector_norm(lite_embedding_grad.float()).item()
    grad_norm_ratio = lite_grad_norm / hf_grad_norm
    print(
        "[qwen35-hf-parity] embedding-output-gradient norms: "
        f"hf={hf_grad_norm:.9f} lite={lite_grad_norm:.9f} ratio={grad_norm_ratio:.9f}"
    )
    assert 0.99 <= grad_norm_ratio <= 1.01, (
        f"embedding-output gradient norm ratio {grad_norm_ratio} is outside [0.99, 1.01]"
    )
    print(
        "NON_SKIP_QWEN35_TRANSFORMERS_512_FULL_MODEL_PARITY_PASSED "
        "layers=2 layer_types=[full_attention,linear_attention] batch=2 seq=8 "
        f"hidden_min_cosine={min(item['cosine'] for item in layer_metrics):.9f} "
        f"hidden_max_rms_relative="
        f"{max(item['rms_relative'] for item in layer_metrics):.9f} "
        f"hidden_min_norm_ratio={min(item['norm_ratio'] for item in layer_metrics):.9f} "
        f"hidden_max_norm_ratio={max(item['norm_ratio'] for item in layer_metrics):.9f} "
        f"logits_cosine={logits_metrics['cosine']:.9f} "
        f"logits_rms_relative={logits_metrics['rms_relative']:.9f} "
        f"logits_norm_ratio={logits_metrics['norm_ratio']:.9f} "
        f"loss_abs={loss_abs:.9f} loss_relative={loss_relative:.9f} "
        f"embedding_grad_cosine={embedding_grad_metrics['cosine']:.9f} "
        f"embedding_grad_rms_relative={embedding_grad_metrics['rms_relative']:.9f} "
        f"embedding_grad_norm_ratio={grad_norm_ratio:.9f}"
    )


@pytest.mark.skipif(
    not os.environ.get("QWEN35_HF_DIR"),
    reason="set QWEN35_HF_DIR to a real Qwen3.5 HF checkpoint to run the ground-truth check.",
)
def test_qwen35_real_hf_load_export_matches_original(tmp_path):
    if dist.get_world_size() != 1:
        pytest.skip("ground-truth smoke runs single-rank (tp1).")
    import json

    from safetensors import safe_open

    from megatron.lite.model.qwen3_5.config import Qwen35Config
    from megatron.lite.model.qwen3_5.lite import protocol
    from megatron.lite.runtime.contracts.config import ParallelConfig

    model_dir = os.environ["QWEN35_HF_DIR"]
    cfg = Qwen35Config.from_hf(model_dir)
    n = min(8, cfg.num_hidden_layers)
    cfg.num_hidden_layers = n
    cfg.layer_types = cfg.layer_types[:n]
    cfg.num_nextn_predict_layers = 0
    # Ensure both attention branches are actually exercised in the truncated model.
    assert "full_attention" in cfg.layer_types
    assert "linear_attention" in cfg.layer_types

    parallel = ParallelConfig(tp=1, pp=1, cp=1, ep=1, etp=1, vpp=1)
    impl = protocol.ImplConfig(
        parallel=parallel, optimizer="dist_opt", use_deepep=False, deterministic=True
    )
    bundle = protocol.build_model(cfg, impl_cfg=impl)
    protocol.load_hf_weights(bundle.chunks[0], model_dir, cfg, bundle.parallel_state)

    out_dir = str(tmp_path / "hf_export")
    os.makedirs(out_dir, exist_ok=True)
    protocol.save_hf_weights(bundle.chunks, out_dir, cfg, bundle.parallel_state)

    orig_map = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))[
        "weight_map"
    ]

    def _orig(key):
        with safe_open(os.path.join(model_dir, orig_map[key]), framework="pt") as fh:
            return fh.get_tensor(key).float()

    exported = {}
    for fn in os.listdir(out_dir):
        if fn.endswith(".safetensors"):
            with safe_open(os.path.join(out_dir, fn), framework="pt") as fh:
                for key in fh.keys():
                    exported[key] = fh.get_tensor(key).float()

    # Derive the expected comparison set from the ORIGINAL weight map (not from
    # `exported`): otherwise a tensor the exporter silently drops would never be
    # compared and a missing-export bug would pass undetected.
    #
    # Scope to what this truncated text-only build actually produces: the first
    # `n` decoder layers (prefix `model.language_model.layers.{i}.`) plus the
    # global embed/norm/head tensors. The original checkpoint also carries the
    # MTP head (`mtp.*`, disabled here via num_nextn_predict_layers=0) and the
    # vision tower (`model.visual.*`, not built on the text path); both are
    # intentionally NOT built/exported, so they are excluded from expected_keys.
    global_keys = (
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
    )
    layer_prefixes = tuple(f"model.language_model.layers.{i}." for i in range(n))
    expected_keys = {
        k for k in orig_map if k.startswith(layer_prefixes) or k in global_keys
    }
    assert expected_keys, "no comparable tensors found in original weight map."

    missing = expected_keys - set(exported)
    assert not missing, "export missing tensors: " + ", ".join(sorted(missing))

    mism = []
    for k in sorted(expected_keys):
        o, e = _orig(k), exported[k]
        if o.shape != e.shape:
            mism.append(f"{k} shape {tuple(o.shape)} != {tuple(e.shape)}")
        elif not torch.allclose(
            o, e, atol=2e-2, rtol=2e-2
        ):  # bf16-cast export tolerance
            mism.append(f"{k} max_abs_diff={(o - e).abs().max().item()}")
    assert not mism, (
        "lite export does not match original HF safetensors:\n" + "\n".join(mism)
    )
