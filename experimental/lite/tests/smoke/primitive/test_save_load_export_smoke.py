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

Run with torchrun --nproc_per_node=8 -m pytest --mlite-smoke, selecting per-env subsets
with -k (qwen3_5 needs the qwen3.5 canary site; the four deepseek/qwen3_moe
models need the DSA overlay). Models gate themselves with importorskip so a
wrong-env invocation skips rather than errors.
"""

from __future__ import annotations

import copy
import os
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
from megatron.lite.primitive.deterministic import set_deterministic
from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime
from megatron.lite.runtime.contracts.config import OptimizerConfig, ParallelConfig
from megatron.lite.runtime.contracts.data import PackedBatch
from megatron.lite.runtime.contracts.handle import ModelHandle

pytestmark = [
    pytest.mark.mlite,
    pytest.mark.smoke,
    pytest.mark.gpu,
    pytest.mark.distributed,
]


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
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        # Exercise GLM5.2 rather than the legacy all-full GLM5 path in both
        # exact-resume backends.  The explicit list is the canonical HF
        # schedule; freq/offset remain present so a future fallback cannot
        # silently change this fixture's model family.
        index_topk_freq=2,
        index_skip_topk_offset=1,
        indexer_types=["full", "shared"],
        rope_interleave=True,
        indexer_rope_interleave=True,
    )
    assert cfg.resolved_dsa_indexer_types == ("full", "shared")
    assert cfg.uses_dsa_index_share is True
    assert cfg.uses_configured_dsa_rope_layout is True
    return cfg, protocol


def _deepseek_v4():
    pytest.importorskip(
        "cudnn", reason="deepseek_v4 fused DSA needs the cudnn DSA stack."
    )
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
        compress_ratios=[4, 4],
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

    # Diagnostic: dump all thread stacks and force-exit after N seconds so a
    # hang self-reports fast (keeps each run well under the time budget) instead
    # of stalling on NCCL's slow watchdog/abort. Opt-in via MLITE_HANG_DUMP_S.
    hang_dump_s = os.environ.get("MLITE_HANG_DUMP_S")
    if hang_dump_s:
        import faulthandler

        faulthandler.dump_traceback_later(int(hang_dump_s), exit=True)

    created_pg = False
    if not dist.is_initialized():
        # Short timeout so a desynced collective fails fast instead of stalling
        # the whole job on the multi-minute NCCL watchdog default.
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


# Process groups created by MLite's ``init_parallel`` are independent of
# MCore's globals and must be destroyed explicitly after every matrix case.  A
# round-trip builds both a source and a fresh destination model, so retaining
# these groups across ten parameterized cases exhausts communicators quickly.
_BUILT_PARALLEL_STATES: list = []
_PS_GROUP_ATTRS = (
    "tp_group",
    "ep_group",
    "etp_group",
    "cp_group",
    "pp_group",
    "pp_cpu_group",
    "embedding_group",
    "dp_group",
    "dp_cp_group",
    "tp_ep_group",
    "ep_dp_group",
)


@pytest.fixture(autouse=True)
def _reset_parallel_state_between_tests():
    """Tear down MCore and MLite process groups after each case.

    The matrix builds models with different topologies (tp2/ep2/pp2 for
    dist_opt, pp2/dp4 for fsdp2). Leaving a prior case's groups initialized
    leaks roughly twenty MLite communicators per round-trip and can desync a
    later case's collectives, so reset both group owners between tests.
    """
    yield
    import gc

    from megatron.core import parallel_state as mpu

    if mpu.is_initialized():
        mpu.destroy_model_parallel()
    gc.collect()

    cleanup_errors: list[str] = []
    destroyed_groups: list[object] = []
    for ps in _BUILT_PARALLEL_STATES:
        for attr in _PS_GROUP_ATTRS:
            group = getattr(ps, attr, None)
            if group is None or group is dist.group.WORLD:
                continue
            setattr(ps, attr, None)
            if any(group is destroyed for destroyed in destroyed_groups):
                continue
            destroyed_groups.append(group)
            try:
                dist.destroy_process_group(group)
            except Exception as exc:  # report on every rank after best-effort cleanup
                cleanup_errors.append(f"{attr}: {type(exc).__name__}: {exc}")
    _BUILT_PARALLEL_STATES.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # A rank-local cleanup failure must fail the entire distributed case rather
    # than letting peers enter the next test with asymmetric group state.
    gathered_errors: list[list[str] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_errors, cleanup_errors)
    if any(gathered_errors):
        pytest.fail(f"MLite process-group cleanup failed by rank: {gathered_errors}")


# GLM5 / DeepSeek-V4 native lite support TP=ETP=VPP=1 only (EP/PP/CP wired
# through primitives), so their dist_opt topology shards via ep/pp/cp instead.
_TP1_ONLY = {"glm5", "deepseek_v4"}


def _topology(model_name: str, backend: str) -> ParallelConfig:
    if backend == "fsdp2":
        # fsdp2 + pp2: FSDP2 shards over dp(=4) within each of 2 pipeline stages,
        # so save/load is exercised with pipeline parallelism (not just pure DP).
        return ParallelConfig(tp=1, ep=1, etp=1, pp=2, cp=1)
    # Diagnostic hook: MLITE_FORCE_TOPO="tp,ep,etp,pp,cp" overrides any model's
    # topology to isolate which parallel dim triggers a hang.
    forced = os.environ.get("MLITE_FORCE_TOPO")
    if forced:
        tp, ep, etp, pp, cp = (int(x) for x in forced.split(","))
        return ParallelConfig(tp=tp, ep=ep, etp=etp, pp=pp, cp=cp)
    # Diagnostic hook: force the tp1/pp2 topology on any model to isolate
    # whether a tp1+pp2 pipeline-P2P bug is generic (not DSA-specific).
    if model_name in _TP1_ONLY or os.environ.get("MLITE_FORCE_TP1"):
        # tp1 x pp2 x cp1 x dp4 = 8 ranks; ep2 within the expert space.
        # CP is intentionally 1: save/load fidelity does not need the DSA CP path
        # (covered by the dedicated CP smokes), and CP+tiny-seq risks fused-DSA hangs.
        return ParallelConfig(tp=1, ep=2, etp=1, pp=2, cp=1)
    # tp2 x pp2 x cp1 x dp2 = 8 ranks; ep2 within the expert space.
    return ParallelConfig(tp=2, ep=2, etp=1, pp=2, cp=1)


def _optimizer_config(offload_fraction: float = 0.0) -> OptimizerConfig:
    return OptimizerConfig(
        optimizer="adam",
        lr=1.0e-3,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1.0e-8,
        clip_grad=1.0,
        offload_fraction=offload_fraction,
    )


def _build_handle(
    model_name: str,
    backend: str,
    *,
    seed: int,
    topology: ParallelConfig | None = None,
    offload_fraction: float = 0.0,
):
    cfg, protocol = MODELS[model_name]()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    parallel = topology or _topology(model_name, backend)
    # PP-replicated MTP embeddings need first/last-stage gradient summation.
    # dist_opt provides that protocol; FSDP2 currently does not, so its DS4
    # checkpoint lane intentionally validates the backbone without mounting
    # MTP. The dist_opt lane enables and trains DS4 MTP so its parameters,
    # optimizer state, shared embedding, and resumed next step are all covered.
    train_mtp = model_name == "deepseek_v4" and backend == "dist_opt"
    impl_cfg = protocol.ImplConfig(
        parallel=parallel,
        optimizer=backend,
        optimizer_config=_optimizer_config(offload_fraction),
        use_deepep=False,
        deterministic=True,
        mtp_enable=train_mtp,
        mtp_enable_train=train_mtp,
    )
    bundle = protocol.build_model(cfg, impl_cfg=impl_cfg)
    _BUILT_PARALLEL_STATES.append(bundle.parallel_state)
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


def _assert_glm52_index_share_model_contract(handle: ModelHandle, cfg) -> None:
    """Prove the distributed GLM resume fixture really built full/shared DSA."""

    assert cfg.resolved_dsa_indexer_types == ("full", "shared")
    assert cfg.uses_dsa_index_share is True
    assert cfg.uses_configured_dsa_rope_layout is True

    local_full = 0
    local_shared = 0
    for chunk in handle._extras["model_chunks"]:
        for layer_idx, layer in zip(chunk.layer_indices, chunk.layers, strict=True):
            attention = layer.self_attention.self_attention
            if cfg.dsa_indexer_type(layer_idx) == "shared":
                local_shared += 1
                assert attention.skip_topk is True
                assert attention.indexer is None
            else:
                local_full += 1
                assert attention.skip_topk is False
                assert attention.indexer is not None

    counts = torch.tensor([local_full, local_shared], device="cuda", dtype=torch.int64)
    dist.all_reduce(counts, group=dist.group.WORLD)
    assert int(counts[0]) > 0, "GLM5.2 resume fixture built no full indexer layer"
    assert int(counts[1]) > 0, "GLM5.2 resume fixture built no shared indexer layer"


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


def _train_step(
    handle: ModelHandle,
    backend: str,
    cfg,
    *,
    batch: PackedBatch | None = None,
    seed: int | None = None,
) -> None:
    # Unified path for both backends: the runtime routes pp>1 through the
    # pipeline schedule regardless of optimizer backend, so fsdp2 also exercises
    # pipeline parallelism here (not just pure DP).
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)
    if batch is None:
        batch = _random_packed_batch(cfg.vocab_size)
    runtime.zero_grad(handle)
    runtime.forward_backward(handle, iter([batch]), None, num_microbatches=1)
    runtime.optimizer_step(handle)
    runtime.zero_grad(handle)


def _local_named_params(handle: ModelHandle) -> dict[str, torch.Tensor]:
    from megatron.lite.primitive.optimizers.fsdp2.adamw import to_local_tensor

    params: dict[str, torch.Tensor] = {}
    for chunk_idx, chunk in enumerate(handle._extras["model_chunks"]):
        for name, param in chunk.named_parameters():
            params[f"{chunk_idx}.{name}"] = (
                to_local_tensor(param.detach()).cpu().clone()
            )
    return params


def _local_persistent_buffers(handle: ModelHandle) -> dict[str, torch.Tensor]:
    from megatron.lite.primitive.ckpt.hf_weights import named_persistent_buffers
    from megatron.lite.primitive.optimizers.fsdp2.adamw import to_local_tensor

    buffers: dict[str, torch.Tensor] = {}
    for chunk_idx, chunk in enumerate(handle._extras["model_chunks"]):
        for name, buffer in named_persistent_buffers(chunk):
            buffers[f"{chunk_idx}.{name}"] = (
                to_local_tensor(buffer.detach()).cpu().clone()
            )
    return buffers


def _local_model_state(handle: ModelHandle) -> dict[str, torch.Tensor]:
    return {
        **{
            f"parameter.{name}": value
            for name, value in _local_named_params(handle).items()
        },
        **{
            f"persistent_buffer.{name}": value
            for name, value in _local_persistent_buffers(handle).items()
        },
    }


def _seed_persistent_buffers(handle: ModelHandle, cfg) -> int:
    """Make checkpointed buffers non-default so a fresh build cannot fake restore."""
    from megatron.lite.primitive.ckpt.hf_weights import named_persistent_buffers
    from megatron.lite.primitive.optimizers.fsdp2.adamw import to_local_tensor

    local_count = 0
    num_experts = max(
        int(getattr(cfg, "num_experts", getattr(cfg, "n_routed_experts", 2))), 1
    )
    with torch.no_grad():
        for chunk in handle._extras["model_chunks"]:
            for _name, buffer in named_persistent_buffers(chunk):
                local = to_local_tensor(buffer)
                if local.numel() == 0:
                    continue
                values = torch.arange(local.numel(), device=local.device).reshape(
                    local.shape
                )
                if local.dtype == torch.bool:
                    values = values.remainder(2).bool()
                elif local.dtype.is_floating_point:
                    values = values.float().mul_(0.001).add_(0.125).to(local.dtype)
                else:
                    values = values.remainder(num_experts).to(local.dtype)
                local.copy_(values)
                local_count += 1

    counts: list[int | None] = [None] * dist.get_world_size()
    dist.all_gather_object(counts, local_count)
    return sum(int(count or 0) for count in counts)


def _assert_rng_state_equal(lhs: dict, rhs: dict, context: str) -> None:
    assert lhs["random_rng_state"] == rhs["random_rng_state"], context
    lhs_np, rhs_np = lhs["np_rng_state"], rhs["np_rng_state"]
    assert lhs_np[0] == rhs_np[0], context
    np.testing.assert_array_equal(lhs_np[1], rhs_np[1], err_msg=context)
    assert lhs_np[2:] == rhs_np[2:], context
    torch.testing.assert_close(
        lhs["torch_rng_state"], rhs["torch_rng_state"], atol=0, rtol=0, msg=context
    )
    for key in ("cuda_rng_state",):
        if lhs[key] is None or rhs[key] is None:
            assert lhs[key] is rhs[key], context
        else:
            torch.testing.assert_close(lhs[key], rhs[key], atol=0, rtol=0, msg=context)
    assert lhs["rng_tracker_states"], f"{context}: RNG tracker state is empty"
    assert rhs["rng_tracker_states"], f"{context}: RNG tracker state is empty"
    assert lhs["rng_tracker_states"].keys() == rhs["rng_tracker_states"].keys(), context
    for name in lhs["rng_tracker_states"]:
        torch.testing.assert_close(
            lhs["rng_tracker_states"][name],
            rhs["rng_tracker_states"][name],
            atol=0,
            rtol=0,
            msg=f"{context}: tracker={name}",
        )


def _assert_packed_batch_equal(lhs: PackedBatch, rhs: PackedBatch) -> None:
    for name in ("input_ids", "labels", "seq_lens"):
        torch.testing.assert_close(
            getattr(lhs, name), getattr(rhs, name), atol=0, rtol=0, msg=name
        )


def _assert_model_state_bitwise_equal(lhs: ModelHandle, rhs: ModelHandle) -> None:
    lhs_state = _local_model_state(lhs)
    rhs_state = _local_model_state(rhs)
    assert lhs_state.keys() == rhs_state.keys()
    assert lhs_state, "expected at least one local model-state tensor to compare."
    mismatches = []
    for name in lhs_state:
        lhs_tensor, rhs_tensor = lhs_state[name], rhs_state[name]
        if (
            lhs_tensor.shape != rhs_tensor.shape
            or lhs_tensor.dtype != rhs_tensor.dtype
            or not torch.equal(lhs_tensor, rhs_tensor)
        ):
            diff = (
                (lhs_tensor.float() - rhs_tensor.float()).abs().max().item()
                if lhs_tensor.shape == rhs_tensor.shape
                else float("inf")
            )
            mismatches.append(f"{name} (max_abs_diff={diff})")
    assert (
        not mismatches
    ), "save/load not bitwise; mismatched model state:\n" + "\n".join(mismatches)


def _is_valid_hf_export_key(key: str, model_name: str) -> bool:
    """Whether an exported key matches the model's real HF release naming.

    Most models use the ``model.``-rooted HF convention. DeepSeek-V4-Flash ships
    a bare layout (``embed.weight`` / ``head.weight`` / ``norm.weight`` /
    ``layers.N.*`` / ``mtp.N.*`` / ``hc_head_*``), so allow that for deepseek_v4.
    """
    if model_name == "deepseek_v4":
        return key.startswith(("layers.", "mtp.")) or key in (
            "embed.weight",
            "head.weight",
            "norm.weight",
            "hc_head_base",
            "hc_head_fn",
            "hc_head_scale",
        )
    return key.startswith("model.") or key in ("lm_head.weight",)


def _validate_rank0_hf_export_file(
    expected: dict[str, torch.Tensor], cfg, out_dir: str, model_name: str
) -> int:
    """Require every persisted tensor to equal the canonical source export."""
    from safetensors import safe_open

    shards = [f for f in os.listdir(out_dir) if f.endswith(".safetensors")]
    assert shards, f"no safetensors exported to {out_dir}"
    keys: set[str] = set()
    # The shared exporter casts EVERY floating tensor to bf16 (``_cast_export_tensor``
    # with export_dtype=bf16) and leaves integers untouched.  So every float weight
    # — including the router correction bias — must be bf16, while integer auxiliary
    # buffers (e.g. DS4 hash-routing ``tid2eid`` index tables; casting them would
    # corrupt the indices) keep their native integral dtype.  Excluding such buffers
    # from the export would be a de-scope; we keep them and check their dtype.
    for shard in shards:
        with safe_open(os.path.join(out_dir, shard), framework="pt") as fh:
            for key in fh.keys():
                tensor = fh.get_tensor(key)
                assert key not in keys, f"duplicate HF export key across files: {key}"
                assert key in expected, f"file contains unexpected HF export key {key}"
                source_tensor = expected[key].detach().cpu().contiguous()
                assert tensor.shape == source_tensor.shape, (
                    f"{key} file shape={tuple(tensor.shape)}, "
                    f"canonical export shape={tuple(source_tensor.shape)}"
                )
                assert tensor.dtype == source_tensor.dtype, (
                    f"{key} file dtype={tensor.dtype}, "
                    f"canonical export dtype={source_tensor.dtype}"
                )
                torch.testing.assert_close(
                    tensor,
                    source_tensor,
                    atol=0,
                    rtol=0,
                    msg=f"{key}: safetensors differs from canonical source export",
                )
                if tensor.dtype.is_floating_point:
                    assert (
                        tensor.dtype == torch.bfloat16
                    ), f"{key} exported as {tensor.dtype}, want bf16"
                else:
                    assert tensor.dtype in (
                        torch.int64,
                        torch.int32,
                        torch.bool,
                    ), f"{key} exported as unexpected non-float dtype {tensor.dtype}"
                assert torch.isfinite(
                    tensor.float()
                ).all(), f"{key} has non-finite values"
                assert _is_valid_hf_export_key(
                    key, model_name
                ), f"unexpected non-HF export key: {key}"
                keys.add(key)
    assert keys == expected.keys() and keys, (
        "safetensors key set differs from the canonical source export: "
        f"missing={sorted(expected.keys() - keys)} unexpected={sorted(keys - expected.keys())}"
    )
    # PP-gather completeness: the rank-0 export must carry EVERY decoder layer's
    # weights (all pipeline stages gathered), not just the first stage's — guards
    # against a pp-blind export silently dropping later stages' layers.
    num_layers = int(getattr(cfg, "num_hidden_layers"))

    def _has_layer(i: int) -> bool:
        # Match both the ``model.``-rooted convention (``model.layers.{i}.``) and
        # the bare DeepSeek-V4-Flash layout (``layers.{i}.``), which has no prefix.
        prefix = f"layers.{i}."
        return any(k.startswith(prefix) or f".{prefix}" in k for k in keys)

    missing = [i for i in range(num_layers) if not _has_layer(i)]
    assert not missing, (
        f"export missing decoder layers {missing} (PP gather incomplete); "
        f"sample keys: {sorted(keys)[:6]}"
    )
    return len(expected)


def _export_and_reload(
    handle: ModelHandle, cfg, protocol, out_dir: str, model_name: str
) -> int:
    """Export HF weights once and verify the persisted key/value state exactly.

    The integration gate deliberately requests BF16 from the canonical export
    generator. Materialization uses the production distributed writer so
    duplicate keys, generator errors, empty rank-0 output, and write failures
    are propagated to every rank instead of being hidden by ``dict(generator)``.
    """
    chunks = handle._extras["model_chunks"]
    ps = handle._parallel_state

    if not hasattr(protocol, "export_hf_weights"):
        raise AssertionError(f"{protocol.__name__} exposes no HF export path.")
    from megatron.lite.primitive.ckpt.hf_weights import (
        materialize_hf_weights_distributed,
        save_hf_weight_pairs_distributed,
    )

    # Consume the collective-bearing canonical generator exactly once. Keep
    # rank 0's materialized key/value oracle, then give the writer only the
    # side-effect-free dict view so file validation cannot accidentally execute
    # TP/EP/PP/FSDP collectives a second time.
    expected = materialize_hf_weights_distributed(
        protocol.export_hf_weights(
            chunks, cfg, ps, rank0_only=True, export_dtype=torch.bfloat16
        )
    )
    save_hf_weight_pairs_distributed(expected.items(), out_dir)

    outcome: list[str | int | None] = [None, None]
    if dist.get_rank() == 0:
        try:
            outcome[1] = _validate_rank0_hf_export_file(
                expected, cfg, out_dir, model_name
            )
        except Exception as exc:
            outcome[0] = f"{type(exc).__name__}: {exc}"
    # A rank-0 validation failure must release peers instead of stranding them
    # at a barrier until the external torchrun timeout fires.
    dist.broadcast_object_list(outcome, src=0)
    if outcome[0] is not None:
        raise AssertionError(f"synchronized HF export validation failed: {outcome[0]}")
    assert isinstance(outcome[1], int) and outcome[1] > 0
    return outcome[1]


def _assert_live_fsdp2_dtensors_materialize_exactly(
    handle: ModelHandle,
) -> dict[str, int | bool]:
    """Prove that live FSDP2 parameters are DTensors and reconstruct exactly.

    ``DTensor.full_tensor()`` is the production HF-export boundary.  Checking
    only that an export file is finite would let a no-op FSDP wrapper or a
    duplicated/misordered shard reconstruction pass.  For every live DTensor,
    independently all-gather its local shards over the DTensor mesh, trim any
    FSDP padding using the public ``Shard`` metadata, and require bitwise
    equality with ``full_tensor()``.
    """
    from torch.distributed.tensor import DTensor, Shard

    local_dtensor_count = 0
    local_exact_count = 0
    for chunk in handle._extras["model_chunks"]:
        for name, parameter in chunk.named_parameters():
            if not isinstance(parameter, DTensor):
                continue
            local_dtensor_count += 1
            placements = tuple(parameter.placements)
            mesh = parameter.device_mesh
            assert mesh.ndim == 1 and len(placements) == 1, (
                f"{name}: FSDP2 export expected one-dimensional DTensor sharding, "
                f"got mesh_ndim={mesh.ndim} placements={placements}"
            )
            placement = placements[0]
            assert isinstance(
                placement, Shard
            ), f"{name}: FSDP2 export expected a Shard placement, got {placement}"

            local = parameter.to_local().detach()
            mesh_size = int(mesh.size())
            group = mesh.get_group()
            shard_dim = int(placement.dim)
            global_dim = int(parameter.shape[shard_dim])
            # Uneven dim-0 FSDP shards expose their logical (possibly empty)
            # local shape through DTensor, while the collective itself uses
            # equal padded chunks. Recreate that neutral padding explicitly so
            # the independent all-gather also covers dimensions smaller than
            # the DP mesh.
            padded_dim = (global_dim + mesh_size - 1) // mesh_size
            padded_shape = list(local.shape)
            padded_shape[shard_dim] = padded_dim
            padded_local = local.new_zeros(padded_shape)
            if local.numel() > 0:
                padded_local.narrow(shard_dim, 0, local.shape[shard_dim]).copy_(local)
            gathered = [torch.empty_like(padded_local) for _ in range(mesh_size)]
            dist.all_gather(gathered, padded_local.contiguous(), group=group)

            logical_parts = []
            for mesh_rank, shard in enumerate(gathered):
                logical_size, _offset = Shard.local_shard_size_and_offset(
                    global_dim, mesh_size, mesh_rank
                )
                logical_parts.append(shard.narrow(shard_dim, 0, int(logical_size)))
            independently_gathered = torch.cat(logical_parts, dim=shard_dim)
            materialized = parameter.detach().full_tensor()

            assert not isinstance(
                materialized, DTensor
            ), f"{name}: full_tensor() returned another DTensor"
            assert tuple(materialized.shape) == tuple(parameter.shape), (
                f"{name}: full_tensor shape={tuple(materialized.shape)}, "
                f"global shape={tuple(parameter.shape)}"
            )
            assert tuple(independently_gathered.shape) == tuple(parameter.shape), (
                f"{name}: independent shard gather shape="
                f"{tuple(independently_gathered.shape)}, global shape={tuple(parameter.shape)}"
            )
            torch.testing.assert_close(
                materialized,
                independently_gathered,
                atol=0,
                rtol=0,
                msg=f"{name}: full_tensor() differs from independently gathered FSDP shards",
            )
            local_exact_count += 1

    local_counts: list[tuple[int, int] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(
        local_counts, (local_dtensor_count, local_exact_count), group=dist.group.WORLD
    )
    assert all(counts is not None and counts[0] > 0 for counts in local_counts), (
        "FSDP2 export expected every rank to own live DTensor parameters, "
        f"got per-rank counts={local_counts}"
    )
    assert all(
        counts[0] == counts[1] for counts in local_counts if counts is not None
    ), (
        "not every observed FSDP2 DTensor passed exact full-tensor reconstruction: "
        f"{local_counts}"
    )
    global_dtensor_count = sum(
        counts[0] for counts in local_counts if counts is not None
    )
    global_exact_count = sum(counts[1] for counts in local_counts if counts is not None)
    return {
        "global_dtensor_count": global_dtensor_count,
        "global_exact_count": global_exact_count,
        "materialized_exactly": global_dtensor_count > 0
        and global_exact_count == global_dtensor_count,
    }


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
    if model_name == "glm5":
        _assert_glm52_index_share_model_contract(saved, cfg)
    _train_step(saved, backend, cfg)
    seeded_buffer_count = _seed_persistent_buffers(saved, cfg)
    if model_name in {"kimi_k2", "glm5", "deepseek_v4"}:
        assert (
            seeded_buffer_count > 0
        ), f"{model_name} proxy unexpectedly exposes no persistent checkpoint buffers"

    ckpt_dir = _shared_tmp_path(tmp_path, "ckpt")
    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)
    runtime.save_checkpoint(saved, ckpt_dir, step=1)
    from megatron.lite.primitive.ckpt import dcp as dcp_impl

    checkpoint_rng_state = copy.deepcopy(dcp_impl._get_rng_state())

    loaded, _cfg2, _proto2 = _build_handle(model_name, backend, seed=9999)
    if model_name == "glm5":
        _assert_glm52_index_share_model_contract(loaded, _cfg2)
    assert runtime.load_checkpoint(loaded, ckpt_dir) == 1
    _assert_model_state_bitwise_equal(saved, loaded)
    restored_rng_state = copy.deepcopy(dcp_impl._get_rng_state())
    _assert_rng_state_equal(
        checkpoint_rng_state,
        restored_rng_state,
        f"{model_name}/{backend} RNG state after load",
    )

    # A checkpoint is not resume-ready merely because model tensors reload.
    # Require non-empty optimizer moments/steps and, where the backend exposes
    # them, non-empty distributed-optimizer master weights to match before the
    # resumed update. Then prove the exact next batch remains bitwise identical.
    _assert_nonempty_named_bitwise_equal(
        _opt_state_snapshot(saved, backend),
        _opt_state_snapshot(loaded, backend),
        f"{model_name}/{backend} optimizer state after load",
    )
    saved_master = _optimizer_master_snapshot(saved, backend)
    loaded_master = _optimizer_master_snapshot(loaded, backend)
    if backend == "dist_opt":
        _assert_nonempty_named_bitwise_equal(
            saved_master,
            loaded_master,
            f"{model_name}/{backend} optimizer master weights after load",
        )
        master_weights_evidence = "nonempty_exact"
    else:
        assert saved_master == loaded_master == {}
        master_weights_evidence = "not_applicable"
    dcp_impl._restore_rng_state(copy.deepcopy(restored_rng_state))
    saved_resume_batch = _random_packed_batch(cfg.vocab_size)
    _train_step(saved, backend, cfg, batch=saved_resume_batch)
    saved_post_step_rng = copy.deepcopy(dcp_impl._get_rng_state())

    dcp_impl._restore_rng_state(copy.deepcopy(restored_rng_state))
    loaded_resume_batch = _random_packed_batch(cfg.vocab_size)
    _assert_packed_batch_equal(saved_resume_batch, loaded_resume_batch)
    _train_step(loaded, backend, cfg, batch=loaded_resume_batch)
    loaded_post_step_rng = copy.deepcopy(dcp_impl._get_rng_state())
    _assert_rng_state_equal(
        saved_post_step_rng,
        loaded_post_step_rng,
        f"{model_name}/{backend} RNG state after resumed step",
    )
    _assert_model_state_bitwise_equal(saved, loaded)
    _assert_nonempty_named_bitwise_equal(
        _opt_state_snapshot(saved, backend),
        _opt_state_snapshot(loaded, backend),
        f"{model_name}/{backend} optimizer state after resumed step",
    )
    resumed_saved_master = _optimizer_master_snapshot(saved, backend)
    resumed_loaded_master = _optimizer_master_snapshot(loaded, backend)
    if backend == "dist_opt":
        _assert_nonempty_named_bitwise_equal(
            resumed_saved_master,
            resumed_loaded_master,
            f"{model_name}/{backend} optimizer master weights after resumed step",
        )
    else:
        assert resumed_saved_master == resumed_loaded_master == {}
    from megatron.lite.primitive.parallel import (
        validate_mtp_embedding_parameter_replicas,
    )

    for handle in (saved, loaded):
        ps = handle._parallel_state
        validate_mtp_embedding_parameter_replicas(
            handle._extras["model_chunks"],
            ps,
            enabled=bool(ps.embedding_groups_initialized),
        )
    if dist.get_rank() == 0:
        dsa_index_share_evidence = (
            "full_shared_exact" if model_name == "glm5" else "not_applicable"
        )
        print(
            "NON_SKIP_DISTRIBUTED_CHECKPOINT_RESUME_EXACT "
            f"model={model_name} backend={backend} model_state=nonempty_exact "
            f"seeded_persistent_buffers_global={seeded_buffer_count} "
            "optimizer_state=nonempty_exact "
            f"master_weights={master_weights_evidence} next_step=bitwise_exact "
            "rng_sidecar=exact resume_batch_from_rng=bitwise_exact "
            "mtp_embedding_replicas=validated_when_enabled "
            f"dsa_index_share={dsa_index_share_evidence}"
        )


_EXPORT_CASES = [
    *((model_name, "dist_opt") for model_name in MODELS),
    ("deepseek_v4", "fsdp2"),
    ("qwen3_5", "fsdp2"),
]


@pytest.mark.parametrize(("model_name", "backend"), _EXPORT_CASES)
def test_export_hf_bf16_reload(model_name, backend, tmp_path):
    """Export HF weights (bf16) and reload the safetensors shards.

    The five dist-opt cases exercise TP/EP/PP gathering. DeepSeek-V4 and
    Qwen3.5 additionally exercise live FSDP2 DTensor materialization over the
    PP2 x DP4 topology; these are the release-regression models for PR #66.
    """
    if dist.get_world_size() != 8:
        pytest.skip("export proxy smoke requires exactly 8 GPUs.")

    set_deterministic(2026)

    handle, cfg, protocol = _build_handle(model_name, backend, seed=4242)
    _train_step(handle, backend, cfg)

    fsdp_evidence: dict[str, int | bool] | None = None
    if backend == "fsdp2":
        fsdp_evidence = _assert_live_fsdp2_dtensors_materialize_exactly(handle)

    export_dir = _shared_tmp_path(tmp_path, "hf_export")
    exact_exported_tensors = _export_and_reload(
        handle, cfg, protocol, export_dir, model_name
    )
    if dist.get_rank() == 0:
        observed_dtensors = (
            int(fsdp_evidence["global_dtensor_count"])
            if fsdp_evidence is not None
            else 0
        )
        exact_dtensors = (
            int(fsdp_evidence["global_exact_count"]) if fsdp_evidence is not None else 0
        )
        fsdp_materialized = (
            bool(fsdp_evidence["materialized_exactly"])
            if fsdp_evidence is not None
            else False
        )
        print(
            "NON_SKIP_HF_EXPORT_RELOAD "
            f"model={model_name} backend={backend} pp=2 "
            f"fsdp_dtensor_materialized={fsdp_materialized} "
            f"observed_dtensor_params_global={observed_dtensors} "
            f"full_tensor_exact_params_global={exact_dtensors} "
            f"canonical_export_file_exact_tensors={exact_exported_tensors}"
        )


# ──────────────────────────────────────────────────────────────────────────
# Runtime offload / onload roundtrip (RL Best tier: runtime.to(cpu/cuda))
#
# Exercises the real runtime.to() used to reclaim GPU between train and rollout:
# param + optimizer state move CPU<->GPU as a whole (NOT offload_fraction).
# 3 delivery models x 2 optimizers on the 8-GPU proxy2 topology.
# ──────────────────────────────────────────────────────────────────────────

# qwen3_5 offload is validated separately (its GatedDeltaNet linear-attention
# path and run env differ); the others run here.
DELIVERY_MODELS = ("deepseek_v4", "glm5", "kimi_k2")


def _offload_topology(model_name: str) -> ParallelConfig:
    # proxy2 = 8-GPU pp2/ep2/cp2.  CSA/DSA (glm5, ds4) are TP=1 only, so they
    # fill 8 ranks with dp2 (tp1·cp2·pp2·dp2=8); TP-capable MoE models use tp2
    # (tp2·cp2·pp2·dp1=8).
    forced = os.environ.get("MLITE_FORCE_TOPO")
    if forced:
        tp, ep, etp, pp, cp = (int(x) for x in forced.split(","))
        return ParallelConfig(tp=tp, ep=ep, etp=etp, pp=pp, cp=cp)
    if model_name in _TP1_ONLY:  # glm5, deepseek_v4: CSA/DSA are TP=1 only
        return ParallelConfig(tp=1, ep=2, etp=1, pp=2, cp=2)
    return ParallelConfig(tp=2, ep=2, etp=1, pp=2, cp=2)


def _iter_opt_state_tensors(handle: ModelHandle, backend: str):
    """Yield (key, tensor) over optimizer state for either backend — the same
    tensors runtime.to() offloads, so device / value can be inspected."""
    opt = handle._optimizer
    if backend == "fsdp2":
        from megatron.lite.primitive.optimizers.fsdp2.adamw import iter_torch_optimizers

        for ci, child in enumerate(iter_torch_optimizers(opt.optimizer)):
            for pi, st in enumerate(getattr(child, "state", {}).values()):
                if isinstance(st, dict):
                    for k, v in st.items():
                        if isinstance(v, torch.Tensor):
                            yield f"{ci}.{pi}.{k}", v
    else:
        from megatron.core.optimizer import ChainedOptimizer

        opts = opt.chained_optimizers if isinstance(opt, ChainedOptimizer) else [opt]
        for oi, sub in enumerate(opts):
            inner = getattr(sub, "optimizer", None)
            if inner is None:
                continue
            for pi, st in enumerate(inner.state.values()):
                for k, v in st.items():
                    if isinstance(v, torch.Tensor):
                        yield f"{oi}.{pi}.{k}", v


def _opt_state_devices(handle: ModelHandle, backend: str) -> set[str]:
    from megatron.lite.primitive.optimizers.fsdp2.adamw import to_local_tensor

    return {
        to_local_tensor(v).device.type
        for _, v in _iter_opt_state_tensors(handle, backend)
    }


def _opt_state_snapshot(handle: ModelHandle, backend: str) -> dict[str, torch.Tensor]:
    from megatron.lite.primitive.optimizers.fsdp2.adamw import to_local_tensor

    return {
        k: to_local_tensor(v.detach()).cpu().clone()
        for k, v in _iter_opt_state_tensors(handle, backend)
    }


def _optimizer_master_snapshot(
    handle: ModelHandle, backend: str
) -> dict[str, torch.Tensor]:
    if backend != "dist_opt":
        return {}
    parameters = list(handle._optimizer.get_parameters())
    assert parameters, "distributed optimizer exposes no local master parameters"
    return {
        str(index): parameter.detach().cpu().clone()
        for index, parameter in enumerate(parameters)
    }


def _local_param_devices(handle: ModelHandle) -> set[str]:
    from megatron.lite.primitive.optimizers.fsdp2.adamw import to_local_tensor

    return {
        to_local_tensor(p.detach()).device.type
        for chunk in handle._extras["model_chunks"]
        for p in chunk.parameters()
    }


def _assert_named_bitwise_equal(lhs: dict, rhs: dict, label: str) -> None:
    assert lhs.keys() == rhs.keys(), f"{label} keys differ across offload roundtrip."
    mismatches = []
    for name in lhs:
        lhs_tensor, rhs_tensor = lhs[name], rhs[name]
        if (
            lhs_tensor.shape != rhs_tensor.shape
            or lhs_tensor.dtype != rhs_tensor.dtype
            or not torch.equal(lhs_tensor, rhs_tensor)
        ):
            diff = (
                (lhs_tensor.float() - rhs_tensor.float()).abs().max().item()
                if lhs_tensor.shape == rhs_tensor.shape
                else float("inf")
            )
            mismatches.append(f"{name} (max_abs_diff={diff})")
    assert not mismatches, f"{label} not bitwise after offload/onload:\n" + "\n".join(
        mismatches
    )


def _assert_nonempty_named_bitwise_equal(lhs: dict, rhs: dict, label: str) -> None:
    assert lhs, f"{label} left snapshot is empty; exactness would be vacuous."
    assert rhs, f"{label} right snapshot is empty; exactness would be vacuous."
    _assert_named_bitwise_equal(lhs, rhs, label)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("model_name", DELIVERY_MODELS)
def test_offload_onload_roundtrip(model_name, backend, tmp_path):
    """runtime.to(cpu) -> to(cuda) restores params + optimizer state exactly and
    training continues — the RL train<->rollout GPU-reclaim path."""
    if dist.get_world_size() != 8:
        pytest.skip("offload/onload proxy smoke requires exactly 8 GPUs.")

    set_deterministic(2026)
    handle, cfg, _ = _build_handle(
        model_name, backend, seed=4242, topology=_offload_topology(model_name)
    )
    _train_step(handle, backend, cfg)  # populate optimizer (exp_avg) state

    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)
    params_before = _local_named_params(handle)
    opt_before = _opt_state_snapshot(handle, backend)
    assert opt_before, "expected optimizer state after a train step."
    assert _opt_state_devices(handle, backend) == {"cuda"}

    runtime.to(handle, "cpu", model=True, optimizer=True, grad=True)
    assert _opt_state_devices(handle, backend) == {
        "cpu"
    }, "optimizer state not offloaded to CPU."
    if backend == "fsdp2":
        # fsdp2 moves params to CPU directly; dist_opt instead frees the GPU
        # buffer storage (params keep a 0-size cuda handle), so assert only fsdp2.
        assert _local_param_devices(handle) == {"cpu"}, "params not offloaded to CPU."

    runtime.to(handle, "cuda", model=True, optimizer=True, grad=True)
    assert _opt_state_devices(handle, backend) == {
        "cuda"
    }, "optimizer state not back on GPU."
    assert _local_param_devices(handle) == {"cuda"}, "params not back on GPU."

    _assert_named_bitwise_equal(params_before, _local_named_params(handle), "param")
    _assert_named_bitwise_equal(
        opt_before, _opt_state_snapshot(handle, backend), "optimizer-state"
    )

    _train_step(handle, backend, cfg)  # continues training after onload


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("model_name", ["kimi_k2"])
def test_offload_fraction_keeps_optimizer_state_on_cpu(model_name, backend, tmp_path):
    """offload_fraction>0 keeps the optimizer update state on CPU and still
    trains.  One delivery model is enough to guard the real-model wiring
    (generic TinyModel coverage already exists)."""
    if dist.get_world_size() != 8:
        pytest.skip("offload_fraction proxy smoke requires exactly 8 GPUs.")

    set_deterministic(2026)
    handle, cfg, _ = _build_handle(
        model_name,
        backend,
        seed=4242,
        topology=_offload_topology(model_name),
        offload_fraction=1.0,
    )
    _train_step(handle, backend, cfg)
    assert "cpu" in _opt_state_devices(
        handle, backend
    ), "offload_fraction=1.0 should keep optimizer update state on CPU."
    _train_step(handle, backend, cfg)  # trains with the offloaded update state
