from __future__ import annotations

from types import SimpleNamespace

import pytest


def _init_dist_or_skip():
    import os

    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Kimi K2 CP smoke.")
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("Run with torchrun so CP ranks are available.")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    if dist.get_world_size() < 2:
        pytest.skip("Kimi K2 CP smoke requires at least 2 ranks.")
    return torch.device("cuda", local_rank)


def _tiny_config():
    from megatron.lite.model.kimi_k2.config import KimiK2Config

    return KimiK2Config(
        num_hidden_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
        intermediate_size=96,
        moe_intermediate_size=16,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        first_k_dense_replace=1,
        q_lora_rank=16,
        kv_lora_rank=12,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=8,
        max_position_embeddings=128,
        rope_theta=10000.0,
        rope_scaling={
            "type": "yarn",
            "factor": 1.0,
            "original_max_position_embeddings": 128,
            "beta_fast": 1.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        },
    )


def _train_cfg(cp: int):
    return SimpleNamespace(
        tp=1,
        ep=1,
        etp=1,
        pp=1,
        cp=cp,
        vpp=None,
        use_deepep=False,
        fp8=False,
        recompute_modules={},
        deterministic=False,
    )


def test_kimi_k2_mla_cp2_matches_full_sequence_reference_forward_and_grad():
    import torch
    import torch.distributed as dist

    device = _init_dist_or_skip()
    from megatron.lite.primitive.modules.mla import MultiLatentAttention
    from megatron.lite.primitive.parallel import ParallelState
    from megatron.lite.primitive.parallel.cp import zigzag_slice_for_cp
    from megatron.lite.primitive.parallel.state import init_parallel
    from megatron.lite.runtime.contracts import ParallelConfig

    world = dist.get_world_size()
    rank = dist.get_rank()
    cfg = _tiny_config()
    ps = init_parallel(ParallelConfig(tp=1, ep=1, etp=1, cp=world, pp=1))

    kwargs = dict(
        hidden_size=cfg.hidden_size,
        num_attention_heads=cfg.num_attention_heads,
        q_lora_rank=cfg.q_lora_rank,
        kv_lora_rank=cfg.kv_lora_rank,
        qk_nope_head_dim=cfg.qk_nope_head_dim,
        qk_rope_head_dim=cfg.qk_rope_head_dim,
        v_head_dim=cfg.v_head_dim,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=cfg.rope_theta,
        rope_scaling=cfg.rope_scaling,
        use_thd=False,
    )
    torch.manual_seed(20260531)
    cp_layer = MultiLatentAttention(ps=ps, **kwargs).to(device=device, dtype=torch.bfloat16)
    torch.manual_seed(20260531)
    ref_layer = MultiLatentAttention(ps=ParallelState(), **kwargs).to(
        device=device,
        dtype=torch.bfloat16,
    )

    seq, batch = 8 * world, 1
    torch.manual_seed(123)
    full_x = torch.randn(seq, batch, cfg.hidden_size, device=device, dtype=torch.bfloat16)
    local_x = zigzag_slice_for_cp(full_x, rank, world, seq_dim=0).detach().requires_grad_(True)
    ref_x = full_x.detach().clone().requires_grad_(True)

    cp_out = cp_layer(local_x)
    ref_out = ref_layer(ref_x)
    expected = zigzag_slice_for_cp(ref_out, rank, world, seq_dim=0)
    torch.testing.assert_close(cp_out, expected, atol=5e-2, rtol=5e-2)

    cp_out.float().sum().backward()
    expected.float().sum().backward()
    expected_grad = zigzag_slice_for_cp(ref_x.grad, rank, world, seq_dim=0)
    assert local_x.grad is not None
    torch.testing.assert_close(local_x.grad, expected_grad, atol=1e-1, rtol=1e-1)


def test_kimi_k2_tiny_model_cp2_matches_full_sequence_reference_forward():
    import torch
    import torch.distributed as dist

    device = _init_dist_or_skip()
    from megatron.lite.model.kimi_k2.lite.model import KimiK2Model
    from megatron.lite.primitive.parallel import ParallelState
    from megatron.lite.primitive.parallel.cp import zigzag_slice_for_cp
    from megatron.lite.primitive.parallel.state import init_parallel
    from megatron.lite.runtime.contracts import ParallelConfig

    world = dist.get_world_size()
    rank = dist.get_rank()
    cfg = _tiny_config()
    ps = init_parallel(ParallelConfig(tp=1, ep=1, etp=1, cp=world, pp=1))

    torch.manual_seed(777)
    cp_model = KimiK2Model(cfg, _train_cfg(world), ps, use_thd=False).to(
        device=device,
        dtype=torch.bfloat16,
    )
    torch.manual_seed(777)
    ref_model = KimiK2Model(cfg, _train_cfg(1), ParallelState(), use_thd=False).to(
        device=device,
        dtype=torch.bfloat16,
    )
    cp_model.eval()
    ref_model.eval()

    batch, seq = 1, 8 * world
    torch.manual_seed(100)
    full_ids = torch.randint(0, cfg.vocab_size, (batch, seq), device=device)
    full_labels = torch.randint(0, cfg.vocab_size, (batch, seq), device=device)
    input_ids = zigzag_slice_for_cp(full_ids, rank, world, seq_dim=1).contiguous()
    labels = zigzag_slice_for_cp(full_labels, rank, world, seq_dim=1).contiguous()

    with torch.no_grad():
        cp_out = cp_model(input_ids=input_ids, labels=labels)
        ref_out = ref_model(input_ids=full_ids, labels=full_labels)

    expected_hidden = zigzag_slice_for_cp(ref_out["hidden_states"], rank, world, seq_dim=0)
    expected_log_probs = zigzag_slice_for_cp(ref_out["log_probs"], rank, world, seq_dim=1)
    torch.testing.assert_close(cp_out["hidden_states"], expected_hidden, atol=1e-1, rtol=1e-1)
    torch.testing.assert_close(cp_out["log_probs"], expected_log_probs, atol=1e-1, rtol=1e-1)
