# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Save / load / export coverage smoke across the supported models and backends.

Matrix: {qwen3_5, qwen3_moe, kimi_k2, glm5, deepseek_v4} x {dist_opt, fsdp2}.

Each (model, backend) case does a faithful round-trip:
  build -> train one real step -> save checkpoint -> build fresh -> load ->
  assert parameters restored bit-exactly -> export HF weights (bf16) ->
  reload exported safetensors and assert dtype/finiteness.

dist_opt cases use the Megatron distributed-checkpoint path on a
tp2/ep2/pp2 topology (exactly 8 GPUs). fsdp2 cases use torch DCP on a
pure-DP mesh. Both checkpoint paths are exercised through the unified
MegatronLiteRuntime so the matrix guards the runtime entry points.

Run with torchrun --nproc_per_node=8 -m pytest, selecting per-env subsets
with -k (qwen3_5 needs the qwen3.5 canary site; the four deepseek/qwen3_moe
models need the DSA overlay). Models gate themselves with importorskip so a
wrong-env invocation skips rather than errors.
"""
from __future__ import annotations

import os
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.lite.primitive.deterministic import set_deterministic
from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime
from megatron.lite.runtime.contracts.config import OptimizerConfig, ParallelConfig
from megatron.lite.runtime.contracts.data import PackedBatch
from megatron.lite.runtime.contracts.handle import ModelHandle

pytestmark = [pytest.mark.mlite, pytest.mark.smoke, pytest.mark.gpu, pytest.mark.distributed]


# ──────────────────────────────────────────────────────────────────────────
# Model registry: name -> builder returning (tiny_config, protocol_module).
# Each builder importorskips its env-specific deps so a wrong-env run skips.
# ──────────────────────────────────────────────────────────────────────────
def _require_te() -> None:
    te = pytest.importorskip(
        "transformer_engine.pytorch",
        reason="save/load/export smoke requires real Transformer Engine.",
    )
    assert hasattr(te, "Linear"), "smoke requires real Transformer Engine Linear."


def _qwen3_5():
    pytest.importorskip("fla", reason="qwen3_5 needs the FLA / GatedDeltaNet stack.")
    _require_te()
    from megatron.lite.model.qwen3_5.config import Qwen35Config
    from megatron.lite.model.qwen3_5.lite import protocol

    cfg = Qwen35Config(
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
        # Mix full + linear attention so the distckpt linear_attn TP-shard
        # path (the in_proj/conv1d/layer_norm sharding fixes) is covered.
        layer_types=["full_attention", "linear_attention"],
        partial_rotary_factor=1.0,
        max_position_embeddings=4096,
    )
    return cfg, protocol


def _qwen3_moe():
    _require_te()
    from megatron.lite.model.qwen3_moe.config import Qwen3MoEConfig
    from megatron.lite.model.qwen3_moe.lite import protocol

    cfg = Qwen3MoEConfig(
        num_hidden_layers=2,
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=64,
        num_experts=4,
        num_experts_per_tok=1,
        moe_intermediate_size=8,
        max_position_embeddings=4096,
        layer_types=["full_attention", "full_attention"],
    )
    return cfg, protocol


def _kimi_k2():
    _require_te()
    from megatron.lite.model.kimi_k2.config import KimiK2Config
    from megatron.lite.model.kimi_k2.lite import protocol

    cfg = KimiK2Config(
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
        num_hidden_layers=2,
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
        n_routed_experts=3,
        n_shared_experts=1,
        num_experts_per_tok=3,
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
        compress_ratios=[4],
        sliding_window=128,
        num_hash_layers=2,
        hc_mult=2,
        index_head_dim=64,
        index_n_heads=8,
        index_topk=512,
        # DeepSeek-V4 really has MTP; its ImplConfig defaults mtp_enable=True and
        # requires >=1 nextn layer, so give it one (exercises MTP weight IO too).
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

BACKENDS = ("dist_opt", "fsdp2")


# ──────────────────────────────────────────────────────────────────────────
# Distributed harness
# ──────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module", autouse=True)
def _single_node_cuda_dist():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for save/load/export smoke.")
    if int(os.environ.get("WORLD_SIZE", "1")) > 8:
        pytest.skip("Megatron Lite smoke tests are capped at single-node 8 GPUs.")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29555")

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    created_pg = False
    if not dist.is_initialized():
        # Short timeout so a desynced collective fails fast instead of stalling
        # the whole job on the multi-minute NCCL watchdog default.
        dist.init_process_group(
            backend="nccl", init_method="env://", timeout=timedelta(seconds=180)
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
    """Tear down Megatron model-parallel groups after each case.

    The matrix builds models with different topologies (tp2/ep2/pp2 for
    dist_opt, pure-DP for fsdp2). Leaving a prior case's mpu groups initialized
    desyncs the next case's collectives, so reset between tests.
    """
    yield
    from megatron.core import parallel_state as mpu

    if mpu.is_initialized():
        mpu.destroy_model_parallel()


# GLM5 / DeepSeek-V4 native lite support TP=ETP=VPP=1 only (EP/PP/CP wired
# through primitives), so their dist_opt topology shards via ep/pp/cp instead.
_TP1_ONLY = {"glm5", "deepseek_v4"}


def _topology(model_name: str, backend: str) -> ParallelConfig:
    if backend == "fsdp2":
        # fsdp2 shards over the full data-parallel mesh (pure DP); etp must be a
        # concrete int because init_parallel computes expert_dp = world/(etp*ep*pp).
        return ParallelConfig(tp=1, ep=1, etp=1, pp=1, cp=1)
    if model_name in _TP1_ONLY:
        # tp1 x pp2 x cp1 x dp4 = 8 ranks; ep2 within the expert space.
        # CP is intentionally 1: save/load fidelity does not need the DSA CP path
        # (covered by the dedicated CP smokes), and CP+tiny-seq risks fused-DSA hangs.
        return ParallelConfig(tp=1, ep=2, etp=1, pp=2, cp=1)
    # tp2 x pp2 x cp1 x dp2 = 8 ranks; ep2 within the expert space.
    return ParallelConfig(tp=2, ep=2, etp=1, pp=2, cp=1)


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


def _build_handle(model_name: str, backend: str, *, seed: int):
    cfg, protocol = MODELS[model_name]()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    parallel = _topology(model_name, backend)
    impl_cfg = protocol.ImplConfig(
        parallel=parallel,
        optimizer=backend,
        optimizer_config=_optimizer_config(),
        use_deepep=False,
        deterministic=True,
    )
    bundle = protocol.build_model(cfg, impl_cfg=impl_cfg)
    chunks = bundle.chunks

    if bundle.extras.get("optimizer_backend") == "fsdp2":
        for chunk in chunks:
            if hasattr(chunk, "initialize_weights"):
                chunk.initialize_weights()
        optimizer = bundle.extras["post_model_load_hook"]()["optimizer"]
    else:
        optimizer = bundle.optimizer

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
        optimizer=optimizer,
        parallel_state=bundle.parallel_state,
        config=SimpleNamespace(parallel=parallel),
        _extras=extras,
    )
    return handle, cfg, protocol


def _shared_tmp_path(tmp_path, suffix: str) -> str:
    payload = [os.path.join(str(tmp_path), suffix) if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    path = payload[0]
    if dist.get_rank() == 0:
        os.makedirs(path, exist_ok=True)
    dist.barrier()
    return path


def _random_packed_batch(vocab_size: int) -> PackedBatch:
    return PackedBatch(
        input_ids=torch.randint(0, vocab_size, (2048,), device="cuda"),
        labels=torch.randint(0, vocab_size, (2048,), device="cuda"),
        seq_lens=torch.full((1,), 2048, dtype=torch.int64, device="cuda"),
    )


def _train_step(handle: ModelHandle, backend: str, cfg) -> None:
    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)
    if backend == "dist_opt":
        batch = _random_packed_batch(cfg.vocab_size)
        runtime.zero_grad(handle)
        runtime.forward_backward(handle, iter([batch]), None, num_microbatches=1)
        runtime.optimizer_step(handle)
        runtime.zero_grad(handle)
    else:
        # fsdp2: pure-DP path, drive the model + optimizer directly.
        model = handle._extras["model_chunks"][0]
        model.train()
        input_ids = torch.randint(0, cfg.vocab_size, (1, 2048), device="cuda")
        handle._optimizer.zero_grad()
        out = model(input_ids=input_ids)
        logits = out["logits"] if isinstance(out, dict) else out
        loss = logits.float().square().mean()
        loss.backward()
        success, grad_norm, _ = handle._optimizer.step()
        assert success
        assert torch.isfinite(torch.tensor(float(grad_norm)))


def _local_named_params(handle: ModelHandle) -> dict[str, torch.Tensor]:
    from megatron.lite.primitive.optimizers.fsdp2.adamw import to_local_tensor

    params: dict[str, torch.Tensor] = {}
    for chunk_idx, chunk in enumerate(handle._extras["model_chunks"]):
        for name, param in chunk.named_parameters():
            params[f"{chunk_idx}.{name}"] = to_local_tensor(param.detach()).cpu().float().clone()
    return params


def _assert_params_bitwise_equal(lhs: ModelHandle, rhs: ModelHandle) -> None:
    lhs_params = _local_named_params(lhs)
    rhs_params = _local_named_params(rhs)
    assert lhs_params.keys() == rhs_params.keys()
    assert lhs_params, "expected at least one local parameter to compare."
    for name in lhs_params:
        torch.testing.assert_close(lhs_params[name], rhs_params[name], atol=0.0, rtol=0.0)


def _export_and_reload(handle: ModelHandle, cfg, protocol, out_dir: str) -> None:
    """Export HF weights (bf16) and assert the reloaded shards are valid.

    Prefer the model's ``save_hf_weights`` wrapper; some models only expose the
    ``export_hf_weights`` generator, so fall back to gathering it and writing
    the safetensors ourselves (rank 0). Every supported model must offer one of
    the two — otherwise it has no export path at all, which is a hard failure.
    """
    chunks = handle._extras["model_chunks"]
    ps = handle._parallel_state

    if hasattr(protocol, "save_hf_weights"):
        protocol.save_hf_weights(chunks, out_dir, cfg, ps)
    elif hasattr(protocol, "export_hf_weights"):
        from megatron.lite.primitive.ckpt.hf_weights import save_safetensors

        weights = dict(protocol.export_hf_weights(chunks, cfg, ps, rank0_only=True))
        if dist.get_rank() == 0 and weights:
            save_safetensors(weights, out_dir)
    else:
        raise AssertionError(f"{protocol.__name__} exposes no HF export path.")
    dist.barrier()

    if dist.get_rank() != 0:
        dist.barrier()
        return

    from safetensors import safe_open

    shards = [f for f in os.listdir(out_dir) if f.endswith(".safetensors")]
    assert shards, f"no safetensors exported to {out_dir}"
    n_tensors = 0
    for shard in shards:
        with safe_open(os.path.join(out_dir, shard), framework="pt") as fh:
            for key in fh.keys():
                tensor = fh.get_tensor(key)
                assert tensor.dtype == torch.bfloat16, f"{key} exported as {tensor.dtype}, want bf16"
                assert torch.isfinite(tensor.float()).all(), f"{key} has non-finite values"
                assert key.startswith("model.") or key in ("lm_head.weight",), (
                    f"unexpected non-HF export key: {key}"
                )
                n_tensors += 1
    assert n_tensors > 0, "exported zero tensors"
    dist.barrier()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("model_name", list(MODELS))
def test_save_load_roundtrip(model_name, backend, tmp_path):
    """Checkpoint save -> fresh build -> load restores parameters bit-exactly.

    Covers all 5 models x {dist_opt + distckpt, fsdp2 + dcp} (10 combos) — the
    primary regression guard for the runtime checkpoint entry points.
    """
    if dist.get_world_size() != 8:
        pytest.skip("save/load proxy smoke requires exactly 8 GPUs.")

    set_deterministic(2026)

    saved, cfg, _protocol = _build_handle(model_name, backend, seed=4242)
    _train_step(saved, backend, cfg)

    ckpt_dir = _shared_tmp_path(tmp_path, "ckpt")
    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)
    runtime.save_checkpoint(saved, ckpt_dir, step=1)

    loaded, _cfg2, _proto2 = _build_handle(model_name, backend, seed=9999)
    assert runtime.load_checkpoint(loaded, ckpt_dir) == 1
    _assert_params_bitwise_equal(saved, loaded)


@pytest.mark.parametrize("model_name", list(MODELS))
def test_export_hf_bf16_reload(model_name, tmp_path):
    """Export HF weights (bf16) and reload the safetensors shards.

    Export output is backend-agnostic, so this exercises the canonical export
    path: a dist_opt model whose TP/EP/PP shards are gathered to full HF
    tensors. NOTE: exporting directly from a live fsdp2 (DTensor-sharded) model
    is NOT covered here — save_hf_weights' gather is not DTensor-aware and
    deadlocks; tracked as a known gap (see TASK-2.16.9 log / K-note).
    """
    if dist.get_world_size() != 8:
        pytest.skip("export proxy smoke requires exactly 8 GPUs.")

    set_deterministic(2026)

    handle, cfg, protocol = _build_handle(model_name, "dist_opt", seed=4242)
    _train_step(handle, "dist_opt", cfg)

    export_dir = _shared_tmp_path(tmp_path, "hf_export")
    _export_and_reload(handle, cfg, protocol, export_dir)
