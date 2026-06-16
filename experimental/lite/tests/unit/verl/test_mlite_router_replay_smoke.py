# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""verl-faithful router-replay alignment smoke (runs through the real runtime).

RECORD a MoE model's routing during a (forward-only) log-prob pass, then REPLAY
those decisions in a second pass and assert:
1. replay reproduces the recorded forward's log-probs bitwise (replay routes the
   model exactly the way it was recorded), and
2. after the gate weights move, replay still forces the *recorded* routing — its
   log-probs diverge from the fresh natural routing.

This is the same RECORD -> REPLAY contract verl uses for MoE RL; here it goes
through the shared primitive ``RouterReplay`` + runtime driver + protocol THD
pack/unpack, the single path all five models share.
"""

from __future__ import annotations

import dataclasses
import json
from types import MethodType, SimpleNamespace

import pytest

pytestmark = [
    pytest.mark.mlite,
    pytest.mark.smoke,
    pytest.mark.gpu,
    pytest.mark.distributed,
]


def _init_dist_or_skip():
    import os

    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MLite router-replay smoke.")
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("Run with torchrun so CP ranks are available.")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    return torch.device("cuda", local_rank)


def _write_kimi_config(path) -> None:
    config = {
        "model_type": "deepseek_v3",
        "num_hidden_layers": 3,
        "hidden_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "vocab_size": 128,
        "intermediate_size": 96,
        "moe_intermediate_size": 16,
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "n_group": 2,
        "topk_group": 1,
        "first_k_dense_replace": 1,
        "q_lora_rank": 16,
        "kv_lora_rank": 12,
        "qk_nope_head_dim": 8,
        "qk_rope_head_dim": 8,
        "v_head_dim": 8,
        "max_position_embeddings": 128,
        "rope_theta": 10000.0,
        "rope_scaling": {
            "type": "yarn",
            "factor": 1.0,
            "original_max_position_embeddings": 128,
            "beta_fast": 1.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        },
    }
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")


def _write_qwen3_moe_config(path) -> None:
    config = {
        "model_type": "qwen3_moe",
        "num_hidden_layers": 2,
        "hidden_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "vocab_size": 128,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 128,
        "router_aux_loss_coef": 0.001,
        "num_nextn_predict_layers": 0,
        "layer_types": ["full_attention", "full_attention"],
    }
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")


def _write_glm5_moe_config(path) -> None:
    config = {
        "model_type": "glm_moe_dsa",
        "num_hidden_layers": 2,
        "hidden_size": 128,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "head_dim": 256,
        "vocab_size": 32,
        "max_position_embeddings": 64,
        "initializer_range": 0.002,
        "q_lora_rank": 16,
        "kv_lora_rank": 512,
        "qk_head_dim": 256,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256,
        "index_head_dim": 128,
        "index_n_heads": 32,
        "index_topk": 512,
        "intermediate_size": 20,
        "moe_intermediate_size": 6,
        "first_k_dense_replace": 1,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 3,
        "num_nextn_predict_layers": 0,
        "mlp_layer_types": ["dense", "sparse"],
    }
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")


def _write_deepseek_v4_config(path) -> None:
    config = {
        "model_type": "deepseek_v4",
        "num_hidden_layers": 2,
        "hidden_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 32,
        "vocab_size": 128,
        "max_position_embeddings": 128,
        "initializer_range": 0.02,
        "q_lora_rank": 64,
        "qk_rope_head_dim": 16,
        "o_lora_rank": 64,
        "o_groups": 4,
        "index_head_dim": 16,
        "index_n_heads": 4,
        "index_topk": 4,
        "moe_intermediate_size": 32,
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "num_hash_layers": 1,
        "num_nextn_predict_layers": 0,
        "compress_ratios": [4, 4],
        "compress_rope_theta": 160000.0,
        "hc_mult": 2,
        "hc_eps": 1e-6,
        "hc_sinkhorn_iters": 4,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "sliding_window": 128,
        "swiglu_limit": 10.0,
        "scoring_func": "sqrtsoftplus",
        "topk_method": "noaux_tc",
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.0,
    }
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")


def _optimizer_config() -> SimpleNamespace:
    return SimpleNamespace(
        optimizer="adam",
        lr=1e-6,
        min_lr=None,
        min_lr_ratio=None,
        clip_grad=1.0,
        weight_decay=0.0,
        lr_warmup_steps_ratio=0.0,
        total_training_steps=1,
        lr_warmup_steps=0,
        lr_warmup_init=0.0,
        lr_decay_steps=None,
        lr_decay_style="constant",
        weight_decay_incr_style="constant",
        lr_wsd_decay_style="exponential",
        lr_wsd_decay_steps=None,
        use_checkpoint_opt_param_scheduler=False,
        betas=(0.9, 0.95),
        override_optimizer_config={},
    )


def _parallel_dims(world):
    """Parallel layout from $MLITE_RR_PARALLEL (e.g. 'pp2cp2tp2ep2'); default cp=world."""
    import os
    import re

    spec = os.environ.get("MLITE_RR_PARALLEL", "")
    dims = {"tp": 1, "ep": 1, "pp": 1, "cp": 1}
    if not spec:
        dims["cp"] = world
        return dims
    for key, val in re.findall(r"(tp|ep|pp|cp)(\d+)", spec):
        dims[key] = int(val)
    return dims


def _build_engine(tmp_path, model_name, model_type, write_config, world):
    from verl_mlite.compat import apply_runtime_patches

    apply_runtime_patches()
    from verl_mlite.engine.config import MegatronLiteEngineConfig
    from verl_mlite.engine.mlite_engine import MegatronLiteEngine

    hf_path = tmp_path / f"tiny-{model_name}"
    write_config(hf_path)
    dims = _parallel_dims(world)
    engine = MegatronLiteEngine(
        model_config=SimpleNamespace(
            local_path=str(hf_path),
            hf_config={"model_type": model_type},
            mtp=None,
        ),
        engine_config=MegatronLiteEngineConfig(
            model_name=model_name,
            tp=dims["tp"],
            ep=dims["ep"],
            pp=dims["pp"],
            cp=dims["cp"],
            impl_cfg={
                "use_thd": True,
                "optimizer": None,
                "deterministic": True,
                "mtp_enable": False,
            },
            use_fused_kernels=False,
        ),
        optimizer_config=_optimizer_config(),
        checkpoint_config={},
    )
    original_build_config = engine._build_mlite_config

    def _no_weights(self):
        config = original_build_config()
        config.load_hf_weights = False
        return config

    engine._build_mlite_config = MethodType(_no_weights, engine)
    engine.initialize()
    return engine


def _flat(log_probs):
    """Raw runtime log_probs are packed/strided; nested only after protocol unpack."""
    return log_probs.values() if getattr(log_probs, "is_nested", False) else log_probs


def _forward(engine, runtime_batch, loss_context, router_replay):
    result = engine.runtime.forward_backward(
        engine.handle,
        iter([(runtime_batch, loss_context)]),
        loss_fn=None,
        num_microbatches=1,
        forward_only=True,
        router_replay=router_replay,
    )
    return result


@pytest.mark.parametrize(
    ("model_name", "model_type", "write_config", "vocab_size", "lengths"),
    [
        ("kimi_k2", "deepseek_v3", _write_kimi_config, 128, [16, 24, 32]),
        ("qwen3_moe", "qwen3_moe", _write_qwen3_moe_config, 128, [16, 24, 32]),
        ("glm5", "glm_moe_dsa", _write_glm5_moe_config, 32, [16, 24, 32]),
        ("deepseek_v4", "deepseek_v4", _write_deepseek_v4_config, 128, [16, 20, 24]),
    ],
)
def test_router_replay_record_then_replay_aligns(
    tmp_path, model_name, model_type, write_config, vocab_size, lengths
):
    import torch
    import torch.distributed as dist

    TensorDict = pytest.importorskip("tensordict").TensorDict

    device = _init_dist_or_skip()
    world = dist.get_world_size()
    rank = dist.get_rank()
    engine = _build_engine(tmp_path, model_name, model_type, write_config, world)

    torch.manual_seed(1234 + rank)
    input_ids = torch.nested.as_nested_tensor(
        [torch.randint(0, vocab_size, (n,), device=device, dtype=torch.long) for n in lengths],
        layout=torch.jagged,
    )
    loss_mask = torch.nested.as_nested_tensor(
        [torch.ones(n, device=device, dtype=torch.float32) for n in lengths],
        layout=torch.jagged,
    )
    micro_batch = TensorDict(
        {"input_ids": input_ids, "loss_mask": loss_mask},
        batch_size=[len(lengths)],
        device=device,
    )
    runtime_batch = engine._make_runtime_batch(micro_batch)
    loss_context = engine._make_runtime_loss_context(micro_batch, loss_scale=1.0)

    # 0) Run-to-run noise floor: two identical no-replay forwards. Deterministic
    #    models give 0; non-deterministic fused kernels (e.g. DSA attention) give
    #    a small floor we accept replay reproduction against (red-line #4).
    lp_a = _flat(_forward(engine, runtime_batch, loss_context, None).model_output.log_probs)
    lp_b = _flat(_forward(engine, runtime_batch, loss_context, None).model_output.log_probs)
    noise = (lp_a - lp_b).abs().max().item()

    # 1) RECORD the routing during a log-prob pass.
    rec = _forward(engine, runtime_batch, loss_context, {"action": "record"})
    routed = rec.model_output.routed_experts
    assert routed is not None, "record mode must emit routed_experts"
    # [bs, seq, num_moe_layers, topk]
    assert [int(x) for x in routed.offsets().diff().cpu()] == lengths
    lp_record = _flat(rec.model_output.log_probs)

    # 2) REPLAY the recorded routing — must reproduce the recorded forward within
    #    the kernel-noise floor (bitwise when the model is deterministic).
    replay_batch = dataclasses.replace(runtime_batch, routed_experts=routed)
    rep = _forward(engine, replay_batch, loss_context, {"action": "replay"})
    lp_replay = _flat(rep.model_output.log_probs)
    replay_diff = (lp_replay - lp_record).abs().max().item()
    tol = max(noise * 4.0, 1e-6)
    assert replay_diff <= tol, (
        f"replay must reproduce the recorded forward (diff={replay_diff:.3e} > tol={tol:.3e}, "
        f"noise floor={noise:.3e})"
    )

    # 3) Move the gate so natural routing changes; replay must still force the
    #    recorded experts (log-probs diverge from fresh routing, well above noise).
    with torch.no_grad():
        for name, param in engine.module.named_parameters():
            if name.endswith("router.gate.weight") or name.endswith("gate.weight"):
                param.add_(torch.randn_like(param) * 3.0)
    lp_natural = _flat(_forward(engine, runtime_batch, loss_context, None).model_output.log_probs)
    lp_replay2 = _flat(
        _forward(engine, replay_batch, loss_context, {"action": "replay"}).model_output.log_probs
    )
    override_diff = (lp_replay2 - lp_natural).abs().max().item()
    assert torch.isfinite(lp_replay2).all()
    assert override_diff > max(noise * 10.0, 1e-4), (
        f"replay should override perturbed routing (override={override_diff:.3e}, noise={noise:.3e})"
    )

    if rank == 0:
        print(
            "NON_SKIP_VERL_MLITE_ROUTER_REPLAY_ALIGN_PASSED "
            f"model={model_name} world_size={world} lengths={lengths} "
            f"noise={noise:.3e} replay_vs_record={replay_diff:.3e} "
            f"perturbed_replay_vs_natural={override_diff:.3e}"
        )
