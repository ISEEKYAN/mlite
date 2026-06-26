# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import copy

import pytest
import torch

pytestmark = [pytest.mark.mlite, pytest.mark.smoke, pytest.mark.gpu]


def _make_dsa():
    pytest.importorskip("cudnn", reason="GLM5 DSA accept-with-proof needs cudnn DSA.")
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
        layer_number=1,
    )


def _run_once(module, x, cos, sin, position_ids, *, fused_training: bool):
    module.zero_grad(set_to_none=True)
    module.train(fused_training)
    local_x = x.detach().clone().requires_grad_(True)
    out = module(local_x, cos=cos, sin=sin, position_ids=position_ids)
    loss = out.float().square().mean()
    loss.backward()
    param_grads = {
        name: param.grad.detach().float().clone()
        for name, param in module.named_parameters()
        if param.grad is not None
    }
    return {
        "loss": loss.detach().float().clone(),
        "out": out.detach().float().clone(),
        "x_grad": local_x.grad.detach().float().clone(),
        "param_grads": param_grads,
    }


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def _max_param_grad_abs(a: dict, b: dict) -> float:
    common = set(a["param_grads"]) & set(b["param_grads"])
    if not common:
        return 0.0
    return max(_max_abs(a["param_grads"][name], b["param_grads"][name]) for name in common)


def test_glm5_dsa_run_to_run_accept_with_proof():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for GLM5 DSA accept-with-proof smoke.")

    from megatron.lite.primitive.modules.attention import build_rope_cache

    device = torch.device("cuda", int(torch.cuda.current_device()))
    torch.manual_seed(20260626)
    fused = _make_dsa().to(device=device, dtype=torch.bfloat16)
    golden = copy.deepcopy(fused).to(device=device, dtype=torch.bfloat16)

    batch, seq, hidden = 1, 512, 128
    x = torch.randn(batch, seq, hidden, device=device, dtype=torch.bfloat16)
    cos, sin = build_rope_cache(
        dim=64,
        max_position_embeddings=seq,
        rope_theta=1_000_000.0,
        device=device,
    )
    position_ids = torch.arange(seq, device=device, dtype=torch.long).unsqueeze(0)

    fused_a = _run_once(fused, x, cos, sin, position_ids, fused_training=True)
    fused_b = _run_once(fused, x, cos, sin, position_ids, fused_training=True)
    golden_a = _run_once(golden, x, cos, sin, position_ids, fused_training=False)
    golden_b = _run_once(golden, x, cos, sin, position_ids, fused_training=False)

    fused_r2r_out = _max_abs(fused_a["out"], fused_b["out"])
    fused_r2r_x_grad = _max_abs(fused_a["x_grad"], fused_b["x_grad"])
    fused_r2r_param_grad = _max_param_grad_abs(fused_a, fused_b)
    golden_r2r_out = _max_abs(golden_a["out"], golden_b["out"])
    golden_r2r_x_grad = _max_abs(golden_a["x_grad"], golden_b["x_grad"])
    golden_r2r_param_grad = _max_param_grad_abs(golden_a, golden_b)
    fused_vs_golden_out = _max_abs(fused_a["out"], golden_a["out"])
    fused_vs_golden_x_grad = _max_abs(fused_a["x_grad"], golden_a["x_grad"])
    fused_vs_golden_param_grad = _max_param_grad_abs(fused_a, golden_a)
    loss_diff = abs(float(fused_a["loss"].item()) - float(golden_a["loss"].item()))

    noise_floor = max(
        fused_r2r_x_grad,
        fused_r2r_param_grad,
        golden_r2r_x_grad,
        golden_r2r_param_grad,
    )
    assert torch.isfinite(fused_a["loss"])
    assert torch.isfinite(golden_a["loss"])
    assert fused_vs_golden_out <= 5.0e-2
    assert fused_vs_golden_x_grad <= max(5.0e-1, 16.0 * noise_floor)
    assert fused_vs_golden_param_grad <= max(5.0e-1, 16.0 * noise_floor)

    print(
        "NON_SKIP_GLM5_DSA_RUN_TO_RUN_ACCEPT_WITH_PROOF "
        f"fused_loss={float(fused_a['loss'].item()):.6e} "
        f"golden_loss={float(golden_a['loss'].item()):.6e} "
        f"loss_diff={loss_diff:.6e} "
        f"fused_r2r_out_max_abs={fused_r2r_out:.6e} "
        f"fused_r2r_x_grad_max_abs={fused_r2r_x_grad:.6e} "
        f"fused_r2r_param_grad_max_abs={fused_r2r_param_grad:.6e} "
        f"golden_r2r_out_max_abs={golden_r2r_out:.6e} "
        f"golden_r2r_x_grad_max_abs={golden_r2r_x_grad:.6e} "
        f"golden_r2r_param_grad_max_abs={golden_r2r_param_grad:.6e} "
        f"fused_vs_golden_out_max_abs={fused_vs_golden_out:.6e} "
        f"fused_vs_golden_x_grad_max_abs={fused_vs_golden_x_grad:.6e} "
        f"fused_vs_golden_param_grad_max_abs={fused_vs_golden_param_grad:.6e} "
        f"noise_floor={noise_floor:.6e}"
    )
