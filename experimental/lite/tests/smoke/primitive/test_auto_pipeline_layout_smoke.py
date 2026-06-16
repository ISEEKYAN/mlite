# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Auto pipeline-layout smoke: non-divisible layer counts run end-to-end on PP>1.

The claim under test: when ``num_hidden_layers`` is *not* divisible
by the pipeline width, ``build_pipeline_chunk_layout`` auto-balances the layers
across PP stages (Megatron-style uneven split, accounting for embedding / head /
MTP overhead) instead of raising "not divisible" — so no hand-tuning of TP/PP is
required.

Matrix: {qwen3_5, qwen3_moe, kimi_k2, glm5, deepseek_v4}, each built with an
odd ``num_hidden_layers`` (3) on a pp=2 topology so the split is necessarily
uneven ([2, 1]). DeepSeek-V4 additionally carries an MTP layer on the last
stage, exercising the MTP-aware balancing.

PP is the variable under test and is fixed at 2 for every model; the remaining
dims match the save/load/export smoke's proven dist_opt topology so an orthogonal
limitation cannot mask the layout result:
  * tp2/pp2/cp1/ep2 for the TP-capable models (qwen3_5, qwen3_moe, kimi_k2);
  * tp1/pp2/cp1/ep2 for glm5 / deepseek_v4 (native lite is TP=1 only).
CP is held at 1: cp>1 with these tiny proxy sequences makes Transformer Engine
report "no dot product attention backend available" (and risks the known
fused-DSA CP+tiny-seq hang) — both orthogonal to pipeline layout. CP fidelity
is covered by the dedicated CP smokes.

Run with torchrun --nproc_per_node=8 -m pytest --mlite-smoke, selecting per-env
subsets with -k like the save/load/export smoke: qwen3_5 on the qwen3.5 site; the
glm5 / deepseek_v4 / qwen3_moe / kimi_k2 models on the DSA overlay. Models also
gate themselves with importorskip.
"""
from __future__ import annotations

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from megatron.lite.primitive.ckpt.hf_weights import unwrap_model
from megatron.lite.primitive.deterministic import set_deterministic
from megatron.lite.primitive.parallel.pp import auto_pipeline_layer_counts
from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime
from megatron.lite.runtime.contracts.config import OptimizerConfig, ParallelConfig
from megatron.lite.runtime.contracts.data import PackedBatch

pytestmark = [pytest.mark.mlite, pytest.mark.smoke, pytest.mark.gpu, pytest.mark.distributed]

# pp=2 with an odd layer count forces an uneven split; small enough to stay fast.
_NUM_LAYERS = 3
_PP = 2

# GLM5 / DeepSeek-V4 native lite support TP=1 only (matches the save/load smoke).
_TP1_ONLY = {"glm5", "deepseek_v4"}


def _require_te() -> None:
    te = pytest.importorskip(
        "transformer_engine.pytorch",
        reason="auto pipeline-layout smoke requires real Transformer Engine.",
    )
    assert hasattr(te, "Linear"), "smoke requires real Transformer Engine Linear."


def _optimizer_config() -> OptimizerConfig:
    return OptimizerConfig(
        optimizer="adam",
        lr=1.0e-3,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1.0e-8,
        clip_grad=1.0,
        offload_fraction=0.0,
    )


def _random_packed_batch(vocab_size: int) -> PackedBatch:
    return PackedBatch(
        input_ids=torch.randint(0, vocab_size, (2048,), device="cuda"),
        labels=torch.randint(0, vocab_size, (2048,), device="cuda"),
        seq_lens=torch.full((1,), 2048, dtype=torch.int64, device="cuda"),
    )


# ──────────────────────────────────────────────────────────────────────────
# Non-divisible model configs (mirror the save/load smoke's tiny configs but
# with num_hidden_layers=3 and layer_types extended to match).
# ──────────────────────────────────────────────────────────────────────────
def _qwen3_5():
    pytest.importorskip("fla", reason="qwen3_5 needs the FLA / GatedDeltaNet stack.")
    _require_te()
    from megatron.lite.model.qwen3_5.config import Qwen35Config
    from megatron.lite.model.qwen3_5.lite import protocol

    cfg = Qwen35Config(
        num_hidden_layers=_NUM_LAYERS,
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
        layer_types=["full_attention", "linear_attention", "full_attention"],
        partial_rotary_factor=1.0,
        max_position_embeddings=4096,
    )
    return cfg, protocol


def _qwen3_moe():
    _require_te()
    from megatron.lite.model.qwen3_moe.config import Qwen3MoEConfig
    from megatron.lite.model.qwen3_moe.lite import protocol

    cfg = Qwen3MoEConfig(
        num_hidden_layers=_NUM_LAYERS,
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=64,
        num_experts=4,
        num_experts_per_tok=1,
        moe_intermediate_size=8,
        max_position_embeddings=4096,
        layer_types=["full_attention", "full_attention", "full_attention"],
    )
    return cfg, protocol


def _kimi_k2():
    _require_te()
    from megatron.lite.model.kimi_k2.config import KimiK2Config
    from megatron.lite.model.kimi_k2.lite import protocol

    cfg = KimiK2Config(
        num_hidden_layers=_NUM_LAYERS,
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
        max_position_embeddings=4096,
        rope_theta=10000.0,
        rope_scaling={
            "type": "yarn",
            "factor": 1.0,
            "original_max_position_embeddings": 4096,
            "beta_fast": 1.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        },
    )
    return cfg, protocol


def _glm5():
    pytest.importorskip("cudnn", reason="glm5 fused DSA needs the cudnn DSA stack.")
    _require_te()
    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.model.glm5.lite import protocol

    cfg = Glm5Config(
        num_hidden_layers=_NUM_LAYERS,
        hidden_size=128,
        num_attention_heads=64,
        num_key_value_heads=64,
        head_dim=256,
        vocab_size=32,
        max_position_embeddings=4096,
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
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
    )
    return cfg, protocol


def _deepseek_v4():
    pytest.importorskip("cudnn", reason="deepseek_v4 fused DSA needs the cudnn DSA stack.")
    _require_te()
    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
    from megatron.lite.model.deepseek_v4.lite import protocol

    cfg = DeepseekV4Config(
        vocab_size=64,
        hidden_size=128,
        moe_intermediate_size=16,
        num_hidden_layers=_NUM_LAYERS,
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
        sliding_window=128,
        num_hash_layers=2,
        hc_mult=2,
        index_head_dim=64,
        index_n_heads=8,
        index_topk=512,
        # Real MTP: an extra nextn layer on the last stage exercises the
        # MTP-aware branch of the auto layout balancing.
        num_nextn_predict_layers=1,
        rms_norm_eps=1e-6,
    )
    return cfg, protocol


MODELS = {
    "qwen3_5": _qwen3_5,
    "qwen3_moe": _qwen3_moe,
    "kimi_k2": _kimi_k2,
    "glm5": _glm5,
    "deepseek_v4": _deepseek_v4,
}


# ──────────────────────────────────────────────────────────────────────────
# Distributed harness (mirrors the save/load/export smoke).
# ──────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module", autouse=True)
def _single_node_cuda_dist():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for auto pipeline-layout smoke.")
    if int(os.environ.get("WORLD_SIZE", "1")) > 8:
        pytest.skip("Megatron Lite smoke tests are capped at single-node 8 GPUs.")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29577")

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    created_pg = False
    if not dist.is_initialized():
        timeout_s = int(os.environ.get("MLITE_DIST_TIMEOUT_S", "180"))
        dist.init_process_group(
            backend="nccl", init_method="env://", timeout=timedelta(seconds=timeout_s)
        )
        created_pg = True
    yield
    try:
        from megatron.core import parallel_state as mpu

        if mpu.is_initialized():
            mpu.destroy_model_parallel()
    finally:
        if created_pg and dist.is_initialized():
            dist.destroy_process_group()


@pytest.fixture(autouse=True)
def _reset_parallel_state_between_tests():
    yield
    from megatron.core import parallel_state as mpu

    if mpu.is_initialized():
        mpu.destroy_model_parallel()


def _proxy_topology(model_name: str) -> ParallelConfig:
    # PP is held at 2 (the layout axis under test); the rest match the save/load
    # smoke's proven dist_opt topology. CP=1 — see module docstring.
    if model_name in _TP1_ONLY:  # glm5 / deepseek_v4: native lite is TP=1 only.
        return ParallelConfig(tp=1, ep=2, etp=1, pp=_PP, cp=1)
    return ParallelConfig(tp=2, ep=2, etp=1, pp=_PP, cp=1)


def _build_handle(model_name: str, *, seed: int):
    from types import SimpleNamespace

    from megatron.lite.runtime.contracts.handle import ModelHandle

    cfg, protocol = MODELS[model_name]()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    parallel = _proxy_topology(model_name)
    impl_cfg = protocol.ImplConfig(
        parallel=parallel,
        optimizer="dist_opt",
        optimizer_config=_optimizer_config(),
        use_deepep=False,
        deterministic=True,
    )
    bundle = protocol.build_model(cfg, impl_cfg=impl_cfg)
    chunks = bundle.chunks
    extras = dict(bundle.extras)
    extras.update(
        {
            "model_chunks": chunks,
            "forward_step": bundle.forward_step,
            "finalize_grads": bundle.finalize_grads,
            "protocol": protocol,
        }
    )
    handle = ModelHandle(
        model=chunks,
        optimizer=bundle.optimizer,
        parallel_state=bundle.parallel_state,
        config=SimpleNamespace(parallel=parallel),
        _extras=extras,
    )
    return handle, cfg


def _local_layer_indices(handle) -> list[int]:
    chunk = unwrap_model(handle._extras["model_chunks"][0])
    return list(getattr(chunk, "layer_indices"))


@pytest.mark.parametrize("model_name", list(MODELS))
def test_non_divisible_layers_auto_balance_and_train(model_name):
    """A non-divisible layer count builds an uneven PP split and trains a step."""
    if dist.get_world_size() != 8:
        pytest.skip("auto pipeline-layout proxy smoke requires exactly 8 GPUs.")

    set_deterministic(2026)

    handle, cfg = _build_handle(model_name, seed=4242)
    ps = handle._parallel_state
    assert ps.pp_size == _PP

    # The non-divisible count must produce the balanced uneven split, not raise.
    expected_counts = auto_pipeline_layer_counts(
        cfg.num_hidden_layers,
        ps.pp_size,
        extra_first=1,
        extra_last=1 + (getattr(cfg, "num_nextn_predict_layers", 0) or 0),
    )
    local = _local_layer_indices(handle)
    start = sum(expected_counts[: ps.pp_rank])
    assert local == list(range(start, start + expected_counts[ps.pp_rank])), (
        f"{model_name} rank{ps.pp_rank}: layer_indices {local} "
        f"!= expected balanced split {expected_counts}"
    )
    assert sum(expected_counts) == cfg.num_hidden_layers
    assert max(expected_counts) - min(expected_counts) <= 1  # balanced

    # End-to-end: one real train step over the uneven pipeline must produce a
    # finite loss (proves PP P2P across stages of unequal depth actually runs).
    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)
    batch = _random_packed_batch(cfg.vocab_size)
    runtime.zero_grad(handle)
    result = runtime.forward_backward(handle, iter([batch]), None, num_microbatches=1)
    runtime.optimizer_step(handle)

    loss = result.model_output.loss
    assert loss is not None and torch.isfinite(loss).all(), (
        f"{model_name}: non-finite loss {loss} on uneven PP layout"
    )
