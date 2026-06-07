from __future__ import annotations

import pytest


def _init_dist_or_skip():
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
        dist.init_process_group("nccl")
    if dist.get_world_size() < 2:
        pytest.skip("GLM5 CP smoke requires at least 2 ranks.")
    return torch.device("cuda", local_rank)


def _tiny_config_kwargs():
    return dict(
        num_hidden_layers=2,
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=32,
        max_position_embeddings=32,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_head_dim=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        index_head_dim=8,
        index_n_heads=2,
        index_topk=2,
        intermediate_size=20,
        moe_intermediate_size=6,
        first_k_dense_replace=1,
        n_routed_experts=3,
        n_shared_experts=1,
        num_experts_per_tok=3,
    )


def _make_mla_dsa(*, cp_size: int = 1, cp_rank: int = 0, cp_group=None):
    from megatron.lite.primitive.modules.mla_dsa import MLADSA

    return MLADSA(
        hidden_size=16,
        num_attention_heads=2,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=2,
        rms_norm_eps=1e-5,
        cp_size=cp_size,
        cp_rank=cp_rank,
        cp_group=cp_group,
    )


@pytest.mark.gpu
def test_glm5_mla_dsa_cp2_matches_full_sequence_reference_forward_and_grad():
    import torch
    import torch.distributed as dist

    from megatron.lite.primitive.modules.mla_dsa import build_rope_cache
    from megatron.lite.primitive.parallel.cp import (
        zigzag_position_ids_for_cp,
        zigzag_slice_for_cp,
    )
    from megatron.lite.primitive.parallel.state import ParallelState

    device = _init_dist_or_skip()
    world = dist.get_world_size()
    rank = dist.get_rank()
    ps = ParallelState(cp_group=dist.group.WORLD, cp_size=world, cp_rank=rank)

    torch.manual_seed(2026)
    cp_attn = _make_mla_dsa(cp_size=world, cp_rank=rank, cp_group=ps.cp_group).to(
        device=device,
        dtype=torch.bfloat16,
    )
    torch.manual_seed(2026)
    ref_attn = _make_mla_dsa().to(device=device, dtype=torch.bfloat16)

    batch, seq = 1, 8 * world
    torch.manual_seed(99)
    full_x = torch.randn(batch, seq, 16, device=device, dtype=torch.bfloat16)
    local_x = zigzag_slice_for_cp(full_x, rank, world, seq_dim=1).detach().requires_grad_(True)
    ref_x = full_x.detach().clone().requires_grad_(True)

    cos, sin = build_rope_cache(
        dim=4,
        max_position_embeddings=seq,
        rope_theta=1_000_000.0,
        device=device,
    )
    local_pos = zigzag_position_ids_for_cp(seq, rank, world, device).expand(batch, -1)
    full_pos = torch.arange(seq, device=device, dtype=torch.long).unsqueeze(0).expand(batch, -1)

    cp_out = cp_attn(local_x, cos=cos, sin=sin, position_ids=local_pos)
    ref_out = ref_attn(ref_x, cos=cos, sin=sin, position_ids=full_pos)
    expected = zigzag_slice_for_cp(ref_out, rank, world, seq_dim=1)
    torch.testing.assert_close(cp_out, expected, atol=3e-2, rtol=3e-2)

    cp_out.float().sum().backward()
    ref_out.float().sum().backward()
    expected_grad = zigzag_slice_for_cp(ref_x.grad, rank, world, seq_dim=1)
    assert local_x.grad is not None
    torch.testing.assert_close(local_x.grad, expected_grad, atol=8e-2, rtol=8e-2)


@pytest.mark.gpu
def test_glm5_tiny_model_cp2_matches_full_sequence_reference_forward():
    import torch
    import torch.distributed as dist

    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.model.glm5.lite.model import Glm5ForCausalLM
    from megatron.lite.primitive.parallel.cp import zigzag_slice_for_cp
    from megatron.lite.primitive.parallel.state import ParallelState

    device = _init_dist_or_skip()
    world = dist.get_world_size()
    rank = dist.get_rank()
    cfg = Glm5Config(**_tiny_config_kwargs())
    cfg.mlp_layer_types = ["dense", "dense"]
    ps = ParallelState(cp_group=dist.group.WORLD, cp_size=world, cp_rank=rank)

    torch.manual_seed(777)
    cp_model = Glm5ForCausalLM(cfg, ps=ps).to(device=device, dtype=torch.bfloat16)
    torch.manual_seed(777)
    ref_model = Glm5ForCausalLM(cfg, ps=ParallelState()).to(device=device, dtype=torch.bfloat16)
    cp_model.eval()
    ref_model.eval()

    batch, seq = 1, 8 * world
    torch.manual_seed(100)
    full_hidden = torch.randn(batch, seq, cfg.hidden_size, device=device, dtype=torch.bfloat16)
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
    from megatron.lite.model.glm5.lite.model import Glm5ForCausalLM
    from megatron.lite.primitive.parallel.cp import zigzag_slice_for_cp
    from megatron.lite.primitive.parallel.state import ParallelState

    device = _init_dist_or_skip()
    world = dist.get_world_size()
    rank = dist.get_rank()
    cfg = Glm5Config(**_tiny_config_kwargs())
    ps = ParallelState(cp_group=dist.group.WORLD, cp_size=world, cp_rank=rank)

    torch.manual_seed(1234)
    model = Glm5ForCausalLM(cfg, ps=ps).to(device=device, dtype=torch.bfloat16)

    batch, seq = 1, 8 * world
    torch.manual_seed(55)
    full_ids = torch.randint(0, cfg.vocab_size, (batch, seq), device=device)
    full_labels = torch.randint(0, cfg.vocab_size, (batch, seq), device=device)
    input_ids = zigzag_slice_for_cp(full_ids, rank, world, seq_dim=1).contiguous()
    labels = zigzag_slice_for_cp(full_labels, rank, world, seq_dim=1).contiguous()

    output = model(input_ids=input_ids, labels=labels)
    assert output["hidden_states"].shape == (batch, seq // world, cfg.hidden_size)
    assert output["loss"].ndim == 0
    output["loss"].backward()

    grad_norm = torch.zeros((), device=device)
    for param in model.parameters():
        if param.grad is not None:
            grad_norm = grad_norm + param.grad.detach().float().norm()
    assert torch.isfinite(grad_norm)
