from __future__ import annotations

import pytest


def _make_train_config(ps):
    from types import SimpleNamespace

    return SimpleNamespace(
        tp=ps.tp_size,
        ep=ps.ep_size,
        etp=ps.etp_size,
        pp=ps.pp_size,
        cp=ps.cp_size,
        vpp=None,
        use_deepep=False,
        fp8=False,
        recompute_modules=[],
        deterministic=True,
    )


def _make_glm5_model(cfg, ps, **kwargs):
    from megatron.lite.model.glm5.lite.model import Glm5Model

    return Glm5Model(cfg, _make_train_config(ps), ps, **kwargs)


def _init_dist_or_skip():
    from datetime import timedelta
    import os

    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for GLM5 CP smoke.")
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("Run with torchrun so CP ranks are available.")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        timeout_s = int(os.environ.get("MLITE_DIST_TIMEOUT_S", "180"))
        if timeout_s <= 0:
            raise ValueError(f"MLITE_DIST_TIMEOUT_S must be positive, got {timeout_s}.")
        dist.init_process_group("nccl", timeout=timedelta(seconds=timeout_s))
    if dist.get_world_size() < 2:
        pytest.skip("GLM5 CP smoke requires at least 2 ranks.")
    return torch.device("cuda", local_rank)


def _tiny_config_kwargs():
    return dict(
        num_hidden_layers=2,
        hidden_size=128,
        num_attention_heads=64,
        num_key_value_heads=64,
        head_dim=256,
        vocab_size=32,
        max_position_embeddings=512,
        initializer_range=0.002,
        q_lora_rank=16,
        kv_lora_rank=512,
        qk_head_dim=256,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        index_head_dim=128,
        index_n_heads=32,
        index_topk=512,
        intermediate_size=20,
        moe_intermediate_size=6,
        first_k_dense_replace=1,
        n_routed_experts=3,
        n_shared_experts=1,
        num_experts_per_tok=3,
    )


def _tiny_hf_parity_config_kwargs():
    kwargs = _tiny_config_kwargs()
    kwargs.update(index_topk=512, max_position_embeddings=512)
    return kwargs


def _fused_dsa_seq_len(world: int) -> int:
    seq = 512
    if seq % (2 * world) != 0:
        pytest.skip(
            f"GLM5 fused DSA CP smoke requires seq={seq} divisible by 2*world={2 * world}."
        )
    return seq


def _sparse_fused_dsa_seq_len(world: int) -> int:
    seq = 1024
    if seq % (2 * world) != 0:
        pytest.skip(
            f"GLM5 sparse fused DSA CP smoke requires seq={seq} "
            f"divisible by 2*world={2 * world}."
        )
    return seq


def _to_hf_deepseek_v3_config(cfg):
    from transformers.models.deepseek_v3.configuration_deepseek_v3 import (
        DeepseekV3Config,
    )

    return DeepseekV3Config(
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        moe_intermediate_size=cfg.moe_intermediate_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        vocab_size=cfg.vocab_size,
        n_shared_experts=cfg.n_shared_experts,
        n_routed_experts=cfg.n_routed_experts,
        routed_scaling_factor=cfg.routed_scaling_factor,
        kv_lora_rank=cfg.kv_lora_rank,
        q_lora_rank=cfg.q_lora_rank,
        qk_rope_head_dim=cfg.qk_rope_head_dim,
        v_head_dim=cfg.v_head_dim,
        qk_nope_head_dim=cfg.qk_nope_head_dim,
        n_group=cfg.n_group,
        topk_group=cfg.topk_group,
        num_experts_per_tok=cfg.num_experts_per_tok,
        first_k_dense_replace=cfg.first_k_dense_replace,
        norm_topk_prob=cfg.norm_topk_prob,
        max_position_embeddings=cfg.max_position_embeddings,
        rms_norm_eps=cfg.rms_norm_eps,
        tie_word_embeddings=False,
        rope_theta=cfg.rope_theta,
        rope_scaling=None,
        rope_interleave=cfg.rope_interleave,
        attention_bias=False,
        attention_dropout=0.0,
        use_cache=False,
    )


def _capture_tensor_output(outputs):
    return outputs[0].detach() if isinstance(outputs, tuple) else outputs.detach()


def _distributed_diff_stats(actual, expected) -> tuple[float, float]:
    import torch
    import torch.distributed as dist

    diff = (actual.float() - expected.float()).abs()
    max_abs = diff.max()
    scale = torch.maximum(
        actual.float().abs().max(), expected.float().abs().max()
    ).clamp_min(1e-6)
    stats = torch.stack([max_abs, scale])
    if dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.MAX)
    return float(stats[0].item()), float((stats[0] / stats[1]).item())


def _hf_state_dict_for_glm5_loader(model):
    return {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in model.state_dict().items()
    }


def _make_dsa(
    *,
    cp_size: int = 1,
    cp_rank: int = 0,
    cp_group=None,
    rope_interleaved: bool = False,
    indexer_loss_coeff: float = 0.0,
):
    from megatron.lite.primitive.modules.attention import DynamicSparseAttention

    return DynamicSparseAttention(
        hidden_size=128,
        num_attention_heads=64,
        q_lora_rank=16,
        kv_lora_rank=512,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=512,
        rms_norm_eps=1e-5,
        rope_interleaved=rope_interleaved,
        indexer_rope_interleaved=rope_interleaved,
        indexer_loss_coeff=indexer_loss_coeff,
        cp_size=cp_size,
        cp_rank=cp_rank,
        cp_group=cp_group,
    )


def _wrap_dsa(dsa, ps, *, rope_theta: float = 1_000_000.0):
    import torch.nn as nn

    from megatron.lite.model.glm5.lite.model import Glm5DSAAttention

    attention = Glm5DSAAttention.__new__(Glm5DSAAttention)
    nn.Module.__init__(attention)
    attention.ps = ps
    attention.qk_rope_head_dim = dsa.qk_rope_head_dim
    attention.rope_theta = rope_theta
    attention.self_attention = dsa
    return attention


@pytest.mark.gpu
@pytest.mark.parametrize(
    "rope_interleaved", [False, True], ids=["half-split", "interleaved"]
)
def test_glm5_dsa_cp2_matches_full_sequence_reference_forward_and_grad(
    rope_interleaved: bool,
):
    import torch
    import torch.distributed as dist

    from megatron.lite.primitive.parallel.cp import zigzag_slice_for_cp
    from megatron.lite.primitive.parallel.state import ParallelState

    device = _init_dist_or_skip()
    world = dist.get_world_size()
    rank = dist.get_rank()
    ps = ParallelState(cp_group=dist.group.WORLD, cp_size=world, cp_rank=rank)

    torch.manual_seed(2026)
    cp_attn = _wrap_dsa(
        _make_dsa(
            cp_size=world,
            cp_rank=rank,
            cp_group=ps.cp_group,
            rope_interleaved=rope_interleaved,
            indexer_loss_coeff=1.0e-2,
        ),
        ps,
    ).to(device=device, dtype=torch.bfloat16)
    torch.manual_seed(2026)
    ref_ps = ParallelState()
    ref_attn = _wrap_dsa(
        _make_dsa(
            rope_interleaved=rope_interleaved,
            indexer_loss_coeff=1.0e-2,
        ),
        ref_ps,
    ).to(device=device, dtype=torch.bfloat16)

    batch, seq = 1, _sparse_fused_dsa_seq_len(world)
    assert cp_attn.self_attention.index_topk < seq
    torch.manual_seed(99)
    full_x = torch.randn(batch, seq, 128, device=device, dtype=torch.bfloat16)
    local_x = (
        zigzag_slice_for_cp(full_x, rank, world, seq_dim=1)
        .detach()
        .requires_grad_(True)
    )
    ref_x = full_x.detach().clone().requires_grad_(True)

    # Exercise the exported wrapper's position_ids=None fallback, including
    # global zigzag positions and rank-3 rotary reconstruction inside DSA.
    cp_out = cp_attn(local_x.transpose(0, 1).contiguous()).transpose(0, 1).contiguous()
    ref_out = ref_attn(ref_x.transpose(0, 1).contiguous()).transpose(0, 1).contiguous()
    expected = zigzag_slice_for_cp(ref_out, rank, world, seq_dim=1)
    torch.testing.assert_close(cp_out, expected, atol=3e-2, rtol=3e-2)

    cp_out.float().sum().backward()
    ref_out.float().sum().backward()
    expected_grad = zigzag_slice_for_cp(ref_x.grad, rank, world, seq_dim=1)
    assert local_x.grad is not None
    torch.testing.assert_close(local_x.grad, expected_grad, atol=8e-2, rtol=8e-2)

    ref_params = dict(ref_attn.named_parameters())
    for name, param in cp_attn.named_parameters():
        assert param.grad is not None, name
        assert ref_params[name].grad is not None, name
        cp_grad = param.grad.detach().float().clone()
        dist.all_reduce(cp_grad, op=dist.ReduceOp.SUM)
        is_indexer_param = ".indexer." in name
        if is_indexer_param:
            # Every CP rank reconstructs the same full sequence, so the
            # detached indexer auxiliary loss produces replicated gradients.
            # Runtime replicated-gradient synchronization averages these.
            cp_grad.div_(world)
            assert torch.count_nonzero(cp_grad).item() > 0, name
        torch.testing.assert_close(
            cp_grad,
            ref_params[name].grad.detach().float(),
            atol=8e-2,
            rtol=8e-2,
            msg=(
                f"CP-averaged indexer gradient mismatch for {name}"
                if is_indexer_param
                else f"CP-summed parameter gradient mismatch for {name}"
            ),
        )
    if rank == 0:
        print(
            "NON_SKIP_GLM5_NONPACKED_CP_RANK3_ROTARY_PASSED "
            f"cp={world} rope_interleaved={rope_interleaved} "
            "indexer_loss_coeff=1.0e-2 indexer_topk=512 seq=1024"
        )


@pytest.mark.gpu
def test_glm5_tiny_model_cp2_matches_full_sequence_reference_forward():
    import torch
    import torch.distributed as dist

    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.primitive.parallel.cp import zigzag_slice_for_cp
    from megatron.lite.primitive.parallel.state import ParallelState

    device = _init_dist_or_skip()
    world = dist.get_world_size()
    rank = dist.get_rank()
    cfg = Glm5Config(**_tiny_config_kwargs())
    cfg.mlp_layer_types = ["dense", "dense"]
    ps = ParallelState(cp_group=dist.group.WORLD, cp_size=world, cp_rank=rank)

    torch.manual_seed(777)
    cp_model = _make_glm5_model(cfg, ps=ps).to(device=device, dtype=torch.bfloat16)
    torch.manual_seed(777)
    ref_model = _make_glm5_model(cfg, ps=ParallelState()).to(
        device=device, dtype=torch.bfloat16
    )
    cp_model.eval()
    ref_model.eval()

    batch, seq = 1, _fused_dsa_seq_len(world)
    torch.manual_seed(100)
    full_hidden = torch.randn(
        batch, seq, cfg.hidden_size, device=device, dtype=torch.bfloat16
    )
    local_hidden = zigzag_slice_for_cp(full_hidden, rank, world, seq_dim=1).contiguous()

    with torch.no_grad():
        cp_hidden = cp_model(hidden_states=local_hidden)["hidden_states"]
        ref_hidden = ref_model(hidden_states=full_hidden)["hidden_states"]
    expected = zigzag_slice_for_cp(ref_hidden, rank, world, seq_dim=1)

    torch.testing.assert_close(cp_hidden, expected, atol=1e-1, rtol=1e-1)


@pytest.mark.gpu
def test_glm5_tiny_model_cp2_forward_backward_smoke():
    import torch
    import torch.distributed as dist

    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.primitive.parallel.cp import zigzag_slice_for_cp
    from megatron.lite.primitive.parallel.state import ParallelState

    device = _init_dist_or_skip()
    world = dist.get_world_size()
    rank = dist.get_rank()
    cfg = Glm5Config(**_tiny_config_kwargs())
    cfg.mlp_layer_types = ["dense", "dense"]
    ps = ParallelState(cp_group=dist.group.WORLD, cp_size=world, cp_rank=rank)

    torch.manual_seed(1234)
    model = _make_glm5_model(cfg, ps=ps).to(device=device, dtype=torch.bfloat16)

    batch, seq = 1, _fused_dsa_seq_len(world)
    torch.manual_seed(55)
    full_ids = torch.randint(0, cfg.vocab_size, (batch, seq), device=device)
    input_ids = zigzag_slice_for_cp(full_ids, rank, world, seq_dim=1).contiguous()

    output = model(input_ids=input_ids)
    assert output["hidden_states"].shape == (batch, seq // world, cfg.hidden_size)
    assert torch.isfinite(output["hidden_states"].float()).all()
    loss = output["hidden_states"].float().square().mean()
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()

    grad_norm = torch.zeros((), device=device)
    for param in model.parameters():
        if param.grad is not None:
            grad_norm = grad_norm + param.grad.detach().float().norm()
    assert torch.isfinite(grad_norm)


@pytest.mark.gpu
def test_glm5_packed_thd_variable_sequence_cp2_direct_forward_backward_smoke():
    import torch
    import torch.distributed as dist

    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.primitive.parallel.state import ParallelState
    from megatron.lite.primitive.parallel.thd import (
        pack_nested_thd,
        unpack_packed_thd_to_nested,
    )

    device = _init_dist_or_skip()
    world = dist.get_world_size()
    rank = dist.get_rank()
    cfg_kwargs = _tiny_config_kwargs()
    cfg_kwargs.update(max_position_embeddings=64, num_nextn_predict_layers=1)
    cfg = Glm5Config(**cfg_kwargs)
    cfg.mlp_layer_types = ["dense", "dense"]
    ps = ParallelState(cp_group=dist.group.WORLD, cp_size=world, cp_rank=rank)

    torch.manual_seed(20260614)
    model = _make_glm5_model(cfg, ps=ps, mtp_enable=True, mtp_enable_train=True).to(
        device=device, dtype=torch.bfloat16
    )
    model.train()

    lengths = [16, 20, 24]
    ids = torch.nested.as_nested_tensor(
        [
            torch.randint(0, cfg.vocab_size, (length,), device=device, dtype=torch.long)
            for length in lengths
        ],
        layout=torch.jagged,
    )
    labels = torch.nested.as_nested_tensor(
        [
            torch.randint(0, cfg.vocab_size, (length,), device=device, dtype=torch.long)
            for length in lengths
        ],
        layout=torch.jagged,
    )
    loss_mask = torch.nested.as_nested_tensor(
        [torch.ones(length, device=device, dtype=torch.float32) for length in lengths],
        layout=torch.jagged,
    )
    packed = pack_nested_thd(
        ids,
        cp_size=world,
        cp_rank=rank,
        cp_group=ps.cp_group,
        labels=labels,
        loss_mask=loss_mask,
    )

    out = model(
        input_ids=packed.input_ids,
        labels=packed.labels,
        loss_mask=packed.loss_mask,
        position_ids=packed.position_ids,
        packed_seq_params=packed.packed_seq_params,
    )
    assert torch.isfinite(out["loss"])
    assert "mtp_loss" in out
    out["loss"].backward()

    grad_norm = torch.zeros((), device=device)
    for param in model.parameters():
        if param.grad is not None:
            grad_norm = grad_norm + param.grad.detach().float().norm()
    assert torch.isfinite(grad_norm)

    nested_log_probs = unpack_packed_thd_to_nested(out["log_probs"], packed)
    assert nested_log_probs.offsets().numel() == len(lengths) + 1
    assert [int(x) for x in nested_log_probs.offsets().diff().cpu()] == lengths

    if rank == 0:
        print(
            "NON_SKIP_GLM5_THD_CP_DIRECT_SMOKE_PASSED "
            f"world_size={world} lengths={lengths} "
            f"loss={float(out['loss'].detach().item()):.6e} "
            f"grad_norm={float(grad_norm.detach().item()):.6e}"
        )


@pytest.mark.gpu
def test_glm5_packed_thd_protocol_indexshare_mtp_cp2_forward_backward_smoke():
    import torch
    import torch.distributed as dist

    device = _init_dist_or_skip()
    world = dist.get_world_size()
    rank = dist.get_rank()

    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.model.glm5.lite.protocol import (
        _forward_step,
        unpack_forward_output,
    )
    from megatron.lite.primitive.parallel.state import ParallelState
    from megatron.lite.runtime.contracts.data import PackedBatch

    lengths = [640, 64, 80]
    cp_alignment = 2 * world
    assert all(length % cp_alignment == 0 for length in lengths)
    assert sum(lengths) % world == 0

    cfg_kwargs = _tiny_config_kwargs()
    cfg_kwargs.update(
        max_position_embeddings=max(lengths),
        num_hidden_layers=6,
        num_nextn_predict_layers=1,
        dsa_indexer_loss_coeff=1.0e-2,
    )
    indexer_types = ["full", "full", "full", "shared", "shared", "shared"]
    cfg = Glm5Config(
        **cfg_kwargs,
        index_topk_freq=4,
        index_skip_topk_offset=3,
        indexer_types=indexer_types,
    )
    cfg.mlp_layer_types = ["dense"] * 7
    assert cfg.index_topk < max(lengths)
    ps = ParallelState(cp_group=dist.group.WORLD, cp_size=world, cp_rank=rank)

    torch.manual_seed(20260614)
    model = _make_glm5_model(cfg, ps=ps, mtp_enable=True, mtp_enable_train=True).to(
        device=device, dtype=torch.bfloat16
    )
    model.train()

    batch = PackedBatch(
        input_ids=torch.cat(
            [
                torch.randint(
                    0, cfg.vocab_size, (length,), device=device, dtype=torch.long
                )
                for length in lengths
            ]
        ),
        labels=torch.cat(
            [
                torch.randint(
                    0, cfg.vocab_size, (length,), device=device, dtype=torch.long
                )
                for length in lengths
            ]
        ),
        seq_lens=torch.tensor(lengths, device=device, dtype=torch.int32),
        loss_mask=torch.ones(sum(lengths), device=device, dtype=torch.float32),
    )

    out = _forward_step(model, batch)
    assert torch.isfinite(out["loss"])
    assert "mtp_loss" in out
    assert torch.isfinite(out["mtp_loss"])
    assert float(out["mtp_loss"].detach().item()) > 0.0

    trunk_indexers = []
    for layer_idx, layer in enumerate(model.layers):
        dsa = layer.self_attention.self_attention
        if indexer_types[layer_idx] == "full":
            assert dsa.skip_topk is False
            assert dsa.indexer is not None
            trunk_indexers.append((layer_idx, dsa.indexer))
        else:
            assert dsa.skip_topk is True
            assert dsa.indexer is None
    assert len(trunk_indexers) == indexer_types.count("full")

    assert model.mtp is not None
    mtp_dsa = model.mtp.layers[0].transformer_layer.self_attention.self_attention
    assert mtp_dsa.skip_topk is False
    assert mtp_dsa.indexer is not None
    out["loss"].backward()

    def assert_finite_nonzero_indexer_grads(indexer, *, label):
        parameters = list(indexer.named_parameters())
        assert parameters, f"{label} has no indexer parameters"
        for name, parameter in parameters:
            assert parameter.grad is not None, f"missing gradient for {label}.{name}"
            gradient = parameter.grad.detach().float()
            assert torch.isfinite(gradient).all(), (
                f"non-finite gradient for {label}.{name}"
            )
            assert torch.count_nonzero(gradient).item() > 0, (
                f"zero gradient for {label}.{name}"
            )

    for layer_idx, indexer in trunk_indexers:
        assert_finite_nonzero_indexer_grads(
            indexer, label=f"layers.{layer_idx}.indexer"
        )
    assert_finite_nonzero_indexer_grads(mtp_dsa.indexer, label="mtp.layers.0.indexer")

    grad_norm = torch.zeros((), device=device)
    for param in model.parameters():
        if param.grad is not None:
            grad_norm = grad_norm + param.grad.detach().float().norm()
    assert torch.isfinite(grad_norm)
    assert float(grad_norm.item()) > 0.0

    nested_log_probs = unpack_forward_output(model, batch, out["log_probs"])
    assert nested_log_probs.offsets().numel() == len(lengths) + 1
    assert [int(x) for x in nested_log_probs.offsets().diff().cpu()] == lengths

    if rank == 0:
        print(
            "NON_SKIP_GLM5_THD_CP_SMOKE_PASSED "
            f"world_size={world} lengths={lengths} index_share=true mtp_indexer=full "
            "indexer_loss_coeff=1.0e-2 indexer_topk=512 nontrivial_sparse_segment=true "
            f"loss={float(out['loss'].detach().item()):.6e} "
            f"mtp_loss={float(out['mtp_loss'].detach().item()):.6e} "
            f"grad_norm={float(grad_norm.detach().item()):.6e}"
        )


@pytest.mark.gpu
def test_glm5_tiny_model_cp2_matches_hf_reference_logits(tmp_path):
    import math

    import torch
    import torch.distributed as dist
    import torch.nn.functional as F

    device = _init_dist_or_skip()
    world = dist.get_world_size()
    rank = dist.get_rank()

    from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
        DeepseekV3ForCausalLM,
    )

    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.model.glm5.lite.checkpoint import load_hf_weights
    from megatron.lite.primitive.ckpt.hf_weights import save_safetensors
    from megatron.lite.primitive.parallel.cp import zigzag_slice_for_cp
    from megatron.lite.primitive.parallel.state import ParallelState

    cfg = Glm5Config(**_tiny_hf_parity_config_kwargs())

    min_cosine = 0.999
    max_rms_relative = 0.02
    min_norm_ratio = 0.99
    max_norm_ratio = 1.01
    max_abs_diff = 0.05
    max_loss_abs_diff = 5.0e-3
    max_loss_relative_diff = 2.0e-3

    def global_parity_metrics(actual, expected):
        assert actual.shape == expected.shape
        actual_fp64 = actual.detach().to(dtype=torch.float64)
        expected_fp64 = expected.detach().to(dtype=torch.float64)
        difference = actual_fp64 - expected_fp64
        sums = torch.stack(
            [
                torch.sum(actual_fp64 * expected_fp64),
                torch.sum(actual_fp64.square()),
                torch.sum(expected_fp64.square()),
                torch.sum(difference.square()),
                torch.tensor(actual_fp64.numel(), device=device, dtype=torch.float64),
            ]
        )
        max_abs = difference.abs().max()
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(max_abs, op=dist.ReduceOp.MAX)

        dot_product, actual_square, expected_square, diff_square, count = sums
        cosine = dot_product / torch.sqrt(actual_square * expected_square).clamp_min(
            1.0e-30
        )
        rms_relative = torch.sqrt(diff_square / count) / torch.sqrt(
            expected_square / count
        ).clamp_min(1.0e-30)
        norm_ratio = torch.sqrt(actual_square / expected_square.clamp_min(1.0e-30))
        return (
            float(cosine.item()),
            float(rms_relative.item()),
            float(norm_ratio.item()),
            float(max_abs.item()),
        )

    def assert_fixed_parity_gates(label, actual, expected):
        cosine, rms_relative, norm_ratio, max_abs = global_parity_metrics(
            actual, expected
        )
        if rank == 0:
            print(
                f"glm5_legacy_hf_full_model_parity tensor={label} "
                f"cosine={cosine:.9f} rms_relative={rms_relative:.9e} "
                f"norm_ratio={norm_ratio:.9f} max_abs_diff={max_abs:.9e} "
                f"gates=cosine>={min_cosine},rms_relative<={max_rms_relative},"
                f"norm_ratio=[{min_norm_ratio},{max_norm_ratio}],"
                f"max_abs_diff<={max_abs_diff}"
            )
        assert math.isfinite(cosine) and cosine >= min_cosine, (
            f"{label} cosine {cosine:.9f} is below {min_cosine}"
        )
        assert math.isfinite(rms_relative) and rms_relative <= max_rms_relative, (
            f"{label} RMS-relative {rms_relative:.9e} exceeds {max_rms_relative}"
        )
        assert (
            math.isfinite(norm_ratio) and min_norm_ratio <= norm_ratio <= max_norm_ratio
        ), (
            f"{label} norm ratio {norm_ratio:.9f} is outside "
            f"[{min_norm_ratio}, {max_norm_ratio}]"
        )
        assert math.isfinite(max_abs) and max_abs <= max_abs_diff, (
            f"{label} max-abs {max_abs:.9e} exceeds {max_abs_diff}"
        )

    torch.manual_seed(20260611)
    hf_ref = DeepseekV3ForCausalLM(_to_hf_deepseek_v3_config(cfg)).to(
        device=device, dtype=torch.bfloat16
    )
    hf_ref.eval()
    rank_tmp_path = tmp_path / f"rank{rank}"
    save_safetensors(_hf_state_dict_for_glm5_loader(hf_ref), str(rank_tmp_path))

    ps = ParallelState(cp_group=dist.group.WORLD, cp_size=world, cp_rank=rank)
    native = _make_glm5_model(cfg, ps=ps).to(device=device, dtype=torch.bfloat16)
    native.eval()
    load_hf_weights(native, str(rank_tmp_path), cfg, ps)

    batch, seq = 1, _fused_dsa_seq_len(world)
    torch.manual_seed(311)
    full_ids = torch.randint(0, cfg.vocab_size, (batch, seq), device=device)
    local_ids = zigzag_slice_for_cp(full_ids, rank, world, seq_dim=1).contiguous()
    label_generator = torch.Generator(device=device)
    label_generator.manual_seed(20260627)
    full_labels = torch.randint(
        0,
        cfg.vocab_size,
        (batch, seq),
        generator=label_generator,
        device=device,
    )
    local_labels = zigzag_slice_for_cp(full_labels, rank, world, seq_dim=1).contiguous()
    local_seq = seq // world
    assert full_ids.shape == (batch, seq)
    assert local_ids.shape == (batch, local_seq)
    assert full_labels.shape == (batch, seq)
    assert local_labels.shape == (batch, local_seq)

    hf_layer_outputs = []
    native_layer_outputs = []
    hooks = []
    for layer in hf_ref.model.layers:
        hooks.append(
            layer.register_forward_hook(
                lambda _module, _inputs, outputs: hf_layer_outputs.append(
                    _capture_tensor_output(outputs)
                )
            )
        )
    for layer in native.layers:
        hooks.append(
            layer.register_forward_hook(
                lambda _module, _inputs, outputs: native_layer_outputs.append(
                    _capture_tensor_output(outputs)
                )
            )
        )

    with torch.no_grad():
        if rank == 0:
            print(
                "glm5_legacy_hf_full_model_parity "
                "reference=transformers.DeepseekV3ForCausalLM "
                "scope=legacy_glm5_glm51_dense_full_topk_deepseek_v3_hf_reference "
                "excludes=glm52_production_layout"
            )
        hf_logits = hf_ref(full_ids).logits
        native_logits = native(input_ids=local_ids)["logits"]

    for hook in hooks:
        hook.remove()

    assert len(hf_layer_outputs) == cfg.num_hidden_layers
    assert len(native_layer_outputs) == cfg.num_hidden_layers
    assert hf_logits.shape == (batch, seq, cfg.vocab_size)
    assert native_logits.shape == (batch, local_seq, cfg.vocab_size)
    for layer_idx, (native_sbhd, full_expected) in enumerate(
        zip(native_layer_outputs, hf_layer_outputs, strict=True)
    ):
        assert full_expected.shape == (batch, seq, cfg.hidden_size)
        assert native_sbhd.shape == (local_seq, batch, cfg.hidden_size)
        actual = native_sbhd.transpose(0, 1).contiguous()
        assert actual.shape == (batch, local_seq, cfg.hidden_size)
        expected = zigzag_slice_for_cp(
            full_expected, rank, world, seq_dim=1
        ).contiguous()
        assert expected.shape == (batch, local_seq, cfg.hidden_size)
        assert_fixed_parity_gates(f"layer_{layer_idx}", actual, expected)

    expected = zigzag_slice_for_cp(hf_logits, rank, world, seq_dim=1).contiguous()
    assert expected.shape == (batch, local_seq, cfg.vocab_size)
    assert_fixed_parity_gates("logits", native_logits, expected)

    native_loss_sum = F.cross_entropy(
        native_logits.float().reshape(-1, cfg.vocab_size),
        local_labels.reshape(-1),
        reduction="sum",
    ).to(dtype=torch.float64)
    reference_loss_sum = F.cross_entropy(
        expected.float().reshape(-1, cfg.vocab_size),
        local_labels.reshape(-1),
        reduction="sum",
    ).to(dtype=torch.float64)
    loss_count = torch.tensor(local_labels.numel(), device=device, dtype=torch.float64)
    loss_stats = torch.stack([native_loss_sum, reference_loss_sum, loss_count])
    dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)
    native_loss = float((loss_stats[0] / loss_stats[2]).item())
    reference_loss = float((loss_stats[1] / loss_stats[2]).item())
    loss_abs_diff = abs(native_loss - reference_loss)
    loss_relative_diff = loss_abs_diff / max(abs(reference_loss), 1.0e-12)
    if rank == 0:
        print(
            "glm5_legacy_hf_full_model_parity external_label_ce "
            f"native_loss={native_loss:.9e} reference_loss={reference_loss:.9e} "
            f"abs_diff={loss_abs_diff:.9e} relative_diff={loss_relative_diff:.9e} "
            f"gates=abs_diff<={max_loss_abs_diff},"
            f"relative_diff<={max_loss_relative_diff}"
        )
    assert math.isfinite(native_loss)
    assert math.isfinite(reference_loss)
    assert loss_abs_diff <= max_loss_abs_diff, (
        f"external-label CE loss abs diff {loss_abs_diff:.9e} exceeds "
        f"{max_loss_abs_diff}"
    )
    assert loss_relative_diff <= max_loss_relative_diff, (
        f"external-label CE loss relative diff {loss_relative_diff:.9e} exceeds "
        f"{max_loss_relative_diff}"
    )

    if rank == 0:
        print(
            "NON_SKIP_GLM5_LEGACY_HF_FULL_MODEL_PARITY_PASSED "
            "scope=legacy_glm5_glm51_dense_full_topk_deepseek_v3_hf_reference "
            "excludes=glm52_production_layout"
        )
