# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DeepSeek-V4 (ds4) MTP under context-parallel (CP>1) + the aux-loss hook.

ds4 previously forced ``cp_size == 1`` for MTP: its next-token roll used a plain
``torch.roll`` that wraps each CP rank's last token onto its *own* first token
rather than the first token of the next rank's slice. The roll now goes through
the shared ``roll_contiguous_left_for_cp`` primitive (gather full -> roll ->
re-slice), so MTP runs under CP. These tests exercise the real ds4 model on GPU.

Attention note: ds4 CSA only has a fused sparse kernel at ``cp_size == 1``; under
CP>1 (and under the default "torch" backend) it runs the dense full-attention
reconstruction path (``iter_cp_sources``). The cp2-vs-cp1 check below runs *both*
sides on that same dense path, so it isolates the CP sequence-split + MTP-roll
correctness -- it is deliberately NOT a fused-vs-fused precision comparison.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.gpu


def _init_dist_or_skip():
    import os

    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for ds4 CP/MTP smoke.")
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("Run with torchrun so CP ranks are available.")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    return torch.device("cuda", local_rank)


def _tiny_ds4_config():
    pytest.importorskip("cudnn", reason="deepseek_v4 fused DSA stack must import.")
    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config

    return DeepseekV4Config(
        vocab_size=64,
        hidden_size=128,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=1,
        head_dim=64,
        qk_rope_head_dim=16,
        q_lora_rank=32,
        o_lora_rank=32,
        o_groups=2,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        routed_scaling_factor=1.5,
        max_position_embeddings=4096,
        compress_ratios=[4, 4],
        sliding_window=64,
        num_hash_layers=2,
        hc_mult=2,
        index_head_dim=64,
        index_n_heads=8,
        index_topk=512,
        num_nextn_predict_layers=1,
        rms_norm_eps=1e-6,
    )


def _build_ds4_chunk(cfg, *, cp_size, cp_rank, cp_group, device, seed=20260618):
    from types import SimpleNamespace

    import torch
    from megatron.lite.model.deepseek_v4.lite.model import DeepseekV4Model
    from megatron.lite.primitive.parallel.state import ParallelState

    ps = ParallelState(cp_group=cp_group, cp_size=cp_size, cp_rank=cp_rank)
    train_cfg = SimpleNamespace(
        tp=1, ep=1, etp=1, pp=1, cp=cp_size, vpp=None, fp8=False, use_deepep=False
    )
    # Same seed -> identical weights for the cp_size and cp_size==1 builds (TP=1,
    # so sharding is a no-op), which makes the cp2-vs-cp1 comparison meaningful.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = DeepseekV4Model(cfg, train_cfg, ps, mtp_enable=True, mtp_enable_train=True).to(
        device=device, dtype=torch.bfloat16
    )
    return model, ps


def _grad_norm(model):
    import torch

    total = torch.zeros((), device=next(model.parameters()).device)
    for param in model.parameters():
        if param.grad is not None:
            total = total + param.grad.detach().float().norm()
    return total


def test_ds4_pre_forward_hook_sets_mtp_and_indexer_loss_scale():
    """Task (1): the ds4 bundle's pre_forward_hook must sync *both* the MTP and
    the (defensive) DSA-indexer backward scales to 1/num_microbatches, mirroring
    GLM-5. Built from a real ModelBundle so the hook is the one actually wired."""
    import os

    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for ds4 hook test.")
    if "RANK" not in os.environ:
        pytest.skip("Run with torchrun (build_model initializes parallel state).")
    if not dist.is_initialized():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        dist.init_process_group("nccl")

    cfg = _tiny_ds4_config()
    from megatron.lite.model.deepseek_v4.lite import protocol
    from megatron.lite.primitive.modules.attention.dsa import DSAIndexerLossAutoScaler
    from megatron.lite.primitive.modules.mtp import MTPLossAutoScaler
    from megatron.lite.runtime.contracts.config import ParallelConfig

    impl_cfg = protocol.ImplConfig(
        parallel=ParallelConfig(tp=1, ep=1, etp=1, pp=1, cp=1),
        optimizer=None,
        deterministic=True,
    )
    bundle = protocol.build_model(cfg, impl_cfg=impl_cfg)
    hook = bundle.extras["pre_forward_hook"]
    assert hook is not None

    num_microbatches = 4
    expected = 1.0 / num_microbatches
    scale = torch.tensor(expected, device="cuda")
    hook(scale)

    # MTPLossAutoScaler stores the scale as a Python float; the DSA-indexer
    # scaler stores it as a tensor. Compare numerically in both cases.
    def _as_float(value):
        return float(value.item()) if hasattr(value, "item") else float(value)

    assert MTPLossAutoScaler.main_loss_backward_scale is not None
    assert DSAIndexerLossAutoScaler.main_loss_backward_scale is not None
    assert abs(_as_float(MTPLossAutoScaler.main_loss_backward_scale) - expected) < 1e-9
    assert abs(_as_float(DSAIndexerLossAutoScaler.main_loss_backward_scale) - expected) < 1e-9


def test_ds4_cp2_mtp_forward_backward_smoke():
    """Task (2): ds4 with MTP + CP>1 runs (the cp_size==1 gate is gone). The
    presence of ``mtp_loss`` proves MTP actually executed under CP."""
    import torch
    import torch.distributed as dist
    from megatron.lite.primitive.parallel.cp import (
        contiguous_position_ids_for_cp,
        contiguous_slice_for_cp,
    )

    device = _init_dist_or_skip()
    world = dist.get_world_size()
    rank = dist.get_rank()
    if world < 2:
        pytest.skip("ds4 CP MTP smoke requires at least 2 ranks.")

    cfg = _tiny_ds4_config()
    model, _ps = _build_ds4_chunk(
        cfg, cp_size=world, cp_rank=rank, cp_group=dist.group.WORLD, device=device
    )
    model.train()

    batch, seq = 2, 64
    assert seq % world == 0
    torch.manual_seed(100)
    full_ids = torch.randint(0, cfg.vocab_size, (batch, seq), device=device)
    full_labels = torch.randint(0, cfg.vocab_size, (batch, seq), device=device)
    full_loss_mask = torch.ones(batch, seq, device=device)

    local_ids = contiguous_slice_for_cp(full_ids, rank, world, seq_dim=1)
    local_labels = contiguous_slice_for_cp(full_labels, rank, world, seq_dim=1)
    local_mask = contiguous_slice_for_cp(full_loss_mask, rank, world, seq_dim=1)
    local_pos = contiguous_position_ids_for_cp(seq, rank, world, device).expand(batch, -1)

    out = model(
        input_ids=local_ids,
        position_ids=local_pos,
        labels=local_labels,
        loss_mask=local_mask,
        enable_mtp=True,
    )
    assert "mtp_loss" in out, "MTP did not run under CP>1 (the cp gate is still closed)."
    assert torch.isfinite(out["loss"])
    assert torch.isfinite(out["mtp_loss"])

    out["loss"].backward()
    grad_norm = _grad_norm(model)
    assert torch.isfinite(grad_norm)

    if rank == 0:
        print(
            "NON_SKIP_DS4_CP_MTP_SMOKE_PASSED "
            f"world_size={world} batch={batch} seq={seq} "
            f"loss={float(out['loss'].detach().item()):.6e} "
            f"mtp_loss={float(out['mtp_loss'].detach().item()):.6e} "
            f"grad_norm={float(grad_norm.detach().item()):.6e}"
        )


def test_ds4_cp2_mtp_matches_full_sequence_reference():
    """Task (2) numerical check: the per-token main-loss log-probs from the CP
    model (dense path, local slice) match the contiguous slice of a cp_size==1
    full-sequence reference built from identical weights. This isolates the CP
    sequence split + MTP roll on the shared dense attention path."""
    import torch
    import torch.distributed as dist
    from megatron.lite.primitive.parallel.cp import (
        contiguous_position_ids_for_cp,
        contiguous_slice_for_cp,
    )

    device = _init_dist_or_skip()
    world = dist.get_world_size()
    rank = dist.get_rank()
    if world < 2:
        pytest.skip("ds4 CP reference match requires at least 2 ranks.")

    cfg = _tiny_ds4_config()
    cp_model, _ = _build_ds4_chunk(
        cfg, cp_size=world, cp_rank=rank, cp_group=dist.group.WORLD, device=device
    )
    ref_model, _ = _build_ds4_chunk(
        cfg, cp_size=1, cp_rank=0, cp_group=None, device=device
    )
    cp_model.eval()
    ref_model.eval()

    batch, seq = 2, 64
    assert seq % world == 0
    torch.manual_seed(101)
    full_ids = torch.randint(0, cfg.vocab_size, (batch, seq), device=device)
    full_labels = torch.randint(0, cfg.vocab_size, (batch, seq), device=device)
    full_pos = torch.arange(seq, device=device).unsqueeze(0).expand(batch, -1)

    local_ids = contiguous_slice_for_cp(full_ids, rank, world, seq_dim=1)
    local_labels = contiguous_slice_for_cp(full_labels, rank, world, seq_dim=1)
    local_pos = contiguous_position_ids_for_cp(seq, rank, world, device).expand(batch, -1)

    with torch.no_grad():
        cp_out = cp_model(
            input_ids=local_ids, position_ids=local_pos, labels=local_labels, enable_mtp=True
        )
        ref_out = ref_model(
            input_ids=full_ids, position_ids=full_pos, labels=full_labels, enable_mtp=True
        )

    # log_probs are [B, S]; slice the reference to this rank's contiguous span.
    ref_local = contiguous_slice_for_cp(ref_out["log_probs"], rank, world, seq_dim=1)
    torch.testing.assert_close(cp_out["log_probs"], ref_local, atol=8e-2, rtol=8e-2)

    if rank == 0:
        diff = (cp_out["log_probs"].float() - ref_local.float()).abs().max()
        print(
            "NON_SKIP_DS4_CP_MTP_REFERENCE_MATCH_PASSED "
            f"world_size={world} max_abs_logprob_diff={float(diff.item()):.6e}"
        )
