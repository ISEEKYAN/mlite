# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Pipeline-layout smoke: real models build through the layout and train on PP>1.

Builds {qwen3_5, qwen3_moe, kimi_k2, glm5, deepseek_v4} with ``num_hidden_layers=9``
on pp=4 (not divisible) and asserts the layout drives a real model, not just the
layout math: each builds, trains a finite-loss step over the uneven split, and HF
export reconstructs every decoder layer. This exercises the auto layout mode on real
models; the custom ``pp_layout`` string mode and the exact per-stage splits (incl.
the real 43/61/78 counts) are pinned in the deterministic unit tests.

Topology follows the save/load/export smoke's validated dist_opt capability:
tp2/ep2 for TP-capable models, tp1/ep2 for glm5 / deepseek_v4 (native lite TP=1).
CP=1: cp>1 on these tiny proxy seqs breaks TE attention backend selection
(orthogonal to layout; covered by the dedicated CP smokes).

Run: torchrun --nproc_per_node=8 -m pytest --mlite-smoke, selecting per-env subsets
with -k (qwen3_5 on the qwen3.5 site; the rest on the DSA overlay). Models also
gate themselves with importorskip.
"""

from __future__ import annotations

import os
import re
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from megatron.lite.primitive.ckpt.hf_weights import unwrap_model
from megatron.lite.primitive.deterministic import set_deterministic
from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime
from megatron.lite.runtime.contracts.config import OptimizerConfig, ParallelConfig
from megatron.lite.runtime.contracts.data import PackedBatch

pytestmark = [
    pytest.mark.mlite,
    pytest.mark.smoke,
    pytest.mark.gpu,
    pytest.mark.distributed,
]

# 9 decoders over pp=4 is non-divisible; 9 is small enough to stay fast yet leaves no
# empty stage even for the MTP model (deepseek_v4 adds a slot on the last stage).
_NUM_LAYERS = 9
_PP = 4

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


def _random_packed_batch(vocab_size: int, *, num_tokens: int = 2048) -> PackedBatch:
    return PackedBatch(
        input_ids=torch.randint(0, vocab_size, (num_tokens,), device="cuda"),
        labels=torch.randint(0, vocab_size, (num_tokens,), device="cuda"),
        seq_lens=torch.full((1,), num_tokens, dtype=torch.int64, device="cuda"),
    )


# ──────────────────────────────────────────────────────────────────────────
# Tiny non-divisible model configs (mirror the save/load smoke, num_hidden_layers=9).
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
        # Deliberately not TP-divisible: export must trim padded vocab rows.
        vocab_size=65,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        linear_num_key_heads=2,
        linear_key_head_dim=4,
        linear_num_value_heads=2,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        layer_types=(["full_attention", "linear_attention"] * ((_NUM_LAYERS + 1) // 2))[
            :_NUM_LAYERS
        ],
        num_nextn_predict_layers=1,
        mtp_layer_types=["full_attention"],
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
        layer_types=["full_attention"] * _NUM_LAYERS,
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
        dsa_indexer_loss_coeff=1.0e-2,
        intermediate_size=20,
        moe_intermediate_size=6,
        first_k_dense_replace=1,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        # GLM-5/5.1 publishes these flags, but without an IndexShare schedule
        # MLite must preserve its pre-PR half-split behavior.
        rope_interleave=True,
        indexer_rope_interleave=True,
    )
    return cfg, protocol


def _glm52_indexer_types(num_layers=78):
    full_layers = {0, 1, 2, *range(6, num_layers, 4)}
    return ["full" if idx in full_layers else "shared" for idx in range(num_layers)]


def _glm52_indexshare():
    pytest.importorskip("cudnn", reason="glm5 fused DSA needs the cudnn DSA stack.")
    _require_te()
    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.model.glm5.lite import protocol

    cfg = Glm5Config(
        num_hidden_layers=78,
        hidden_size=128,
        num_attention_heads=64,
        num_key_value_heads=64,
        head_dim=256,
        vocab_size=32,
        max_position_embeddings=1024,
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
        dsa_indexer_loss_coeff=1.0e-2,
        intermediate_size=20,
        moe_intermediate_size=6,
        first_k_dense_replace=1,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        num_nextn_predict_layers=1,
        index_topk_freq=4,
        index_skip_topk_offset=3,
        indexer_types=_glm52_indexer_types(),
        rope_interleave=True,
        indexer_rope_interleave=True,
    )
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


# Process groups lite's init_parallel creates per build, recorded so the autouse
# teardown can free them (mcore's destroy_model_parallel frees only mcore's groups).
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
    yield
    import gc

    from megatron.core import parallel_state as mpu

    if mpu.is_initialized():
        mpu.destroy_model_parallel()
    # Finalize the just-built model first so DDP releases its group references, then
    # explicitly destroy the process groups lite's init_parallel created for this
    # test. Without this, lite's ~10 fresh NCCL/gloo groups per build leak across the
    # ~6 builds in one torchrun process and make a later PP P2P / export collective
    # fail nondeterministically (mcore's destroy_model_parallel frees only its own).
    gc.collect()
    for ps in _BUILT_PARALLEL_STATES:
        for attr in _PS_GROUP_ATTRS:
            group = getattr(ps, attr, None)
            if group is not None:
                try:
                    dist.destroy_process_group(group)
                except Exception:
                    pass
    _BUILT_PARALLEL_STATES.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _proxy_topology(model_name: str, pp_layout=None) -> ParallelConfig:
    # PP is held at 4 (the layout axis under test); the rest follow the save/load
    # smoke's validated dist_opt capability. CP=1 — see module docstring.
    # pp_layout=None -> auto mode; a string -> custom mode (advanced explicit layout).
    if model_name in _TP1_ONLY:  # glm5 / deepseek_v4: native lite is TP=1 only.
        return ParallelConfig(tp=1, ep=2, etp=1, pp=_PP, cp=1, pp_layout=pp_layout)
    return ParallelConfig(tp=2, ep=2, etp=1, pp=_PP, cp=1, pp_layout=pp_layout)


def _build_handle_from_config(
    model_name: str,
    cfg,
    protocol,
    *,
    seed: int,
    parallel: ParallelConfig | None = None,
    pp_layout=None,
    impl_overrides: dict | None = None,
):
    from types import SimpleNamespace

    from megatron.lite.runtime.contracts.handle import ModelHandle

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if parallel is None:
        parallel = _proxy_topology(model_name, pp_layout=pp_layout)
    impl_kwargs = dict(
        parallel=parallel,
        optimizer="dist_opt",
        optimizer_config=_optimizer_config(),
        use_deepep=False,
        deterministic=True,
    )
    if impl_overrides:
        impl_kwargs.update(impl_overrides)
    impl_cfg = protocol.ImplConfig(**impl_kwargs)
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
    _BUILT_PARALLEL_STATES.append(bundle.parallel_state)
    return handle, cfg


def _build_handle(model_name: str, *, seed: int, pp_layout=None):
    cfg, protocol = MODELS[model_name]()
    impl_overrides = (
        {"mtp_enable": True, "mtp_enable_train": True}
        if model_name in {"qwen3_5", "glm5", "deepseek_v4"}
        else None
    )
    return _build_handle_from_config(
        model_name,
        cfg,
        protocol,
        seed=seed,
        pp_layout=pp_layout,
        impl_overrides=impl_overrides,
    )


def _local_layer_indices(handle) -> list[int]:
    chunk = unwrap_model(handle._extras["model_chunks"][0])
    return list(chunk.layer_indices)


# Decoder layer ids in exported HF names: matches both `model.layers.N.` (HF-rooted,
# most models) and bare `layers.N.` (deepseek_v4-flash release naming).
_HF_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def _hf_decoder_layer_indices(names) -> set[int]:
    """Decoder layer ids referenced by exported HF weight names (``...layers.N...``)."""
    return {int(m.group(1)) for n in names if (m := _HF_LAYER_RE.search(n))}


def _named_local_parameters(handle):
    """Return unique local model parameters with stable chunk-qualified names."""
    named = []
    seen: set[int] = set()
    for chunk_idx, chunk in enumerate(handle._extras["model_chunks"]):
        for name, parameter in unwrap_model(chunk).named_parameters():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            named.append((f"chunk{chunk_idx}.{name}", parameter))
    return named


def _parameter_grad(parameter: torch.nn.Parameter) -> torch.Tensor | None:
    grad = getattr(parameter, "main_grad", None)
    return parameter.grad if grad is None else grad


def _assert_finite_nonzero_gradients(named_parameters, *, label: str) -> list[str]:
    """Require every selected parameter to have a finite gradient and some signal."""
    named_parameters = list(named_parameters)
    assert named_parameters, f"{label}: selected no parameters"
    nonzero = []
    for name, parameter in named_parameters:
        grad = _parameter_grad(parameter)
        assert grad is not None, f"{label}: missing gradient for {name}"
        grad = grad.detach().float()
        assert torch.isfinite(grad).all(), f"{label}: non-finite gradient for {name}"
        if torch.count_nonzero(grad).item() > 0:
            nonzero.append(name)
    assert nonzero, f"{label}: every selected gradient is exactly zero"
    return nonzero


def _assert_finite_nonzero_gradient_signal(
    named_parameters, *, label: str
) -> list[str]:
    """Require finite, nonzero local training signal without banning unused parameters."""
    named_parameters = list(named_parameters)
    assert named_parameters, f"{label}: selected no parameters"
    with_grad = []
    nonzero = []
    for name, parameter in named_parameters:
        grad = _parameter_grad(parameter)
        if grad is None:
            continue
        with_grad.append(name)
        grad = grad.detach().float()
        assert torch.isfinite(grad).all(), f"{label}: non-finite gradient for {name}"
        if torch.count_nonzero(grad).item() > 0:
            nonzero.append(name)
    assert with_grad, f"{label}: no local parameter received a gradient"
    assert nonzero, f"{label}: every local gradient is exactly zero"
    return nonzero


def _snapshot_named_parameters(named_parameters) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone() for name, parameter in named_parameters
    }


def _assert_snapshot_changed(
    before: dict[str, torch.Tensor], named_parameters, *, label: str
) -> list[str]:
    after = {name: parameter.detach().cpu() for name, parameter in named_parameters}
    assert (
        before.keys() == after.keys()
    ), f"{label}: parameter set changed during optimizer step"
    changed = [name for name in before if not torch.equal(before[name], after[name])]
    assert changed, f"{label}: optimizer step did not change any selected parameter"
    return changed


def _optimizer_master_snapshot(handle) -> list[torch.Tensor]:
    """Snapshot FP32/main parameters owned by the distributed optimizer shard."""
    parameters = list(handle._optimizer.get_parameters())
    assert parameters, "distributed optimizer exposes no local main parameters"
    return [parameter.detach().cpu().clone() for parameter in parameters]


def _assert_optimizer_master_changed(handle, before: list[torch.Tensor]) -> int:
    after = list(handle._optimizer.get_parameters())
    assert len(before) == len(
        after
    ), "optimizer main-parameter count changed during step"
    changed = sum(
        not torch.equal(old, new.detach().cpu())
        for old, new in zip(before, after, strict=True)
    )
    assert changed > 0, "optimizer reported success but no FP32/main parameter changed"
    return changed


def _assert_optimizer_state_initialized(handle) -> int:
    """Require Adam state with finite, nonzero moving averages after the first step."""
    from megatron.core.optimizer import ChainedOptimizer

    optimizer = handle._optimizer
    children = (
        optimizer.chained_optimizers
        if isinstance(optimizer, ChainedOptimizer)
        else [optimizer]
    )
    state_tensors = []
    moving_average_tensors = []
    for child in children:
        inner = getattr(child, "optimizer", None)
        assert inner is not None, "distributed optimizer child has no inner optimizer"
        for state in inner.state.values():
            for key, value in state.items():
                if not isinstance(value, torch.Tensor):
                    continue
                assert torch.isfinite(
                    value.float()
                ).all(), f"optimizer state {key} contains non-finite values"
                state_tensors.append(value)
                if key in {"exp_avg", "exp_avg_sq"}:
                    moving_average_tensors.append(value)
    assert state_tensors, "optimizer step created no tensor state"
    assert moving_average_tensors, "Adam optimizer state has no moving averages"
    assert any(
        torch.count_nonzero(value).item() > 0 for value in moving_average_tensors
    ), "Adam moving averages are all exactly zero after a nonzero-gradient step"
    return len(state_tensors)


def _assert_successful_optimizer_step(
    runtime, handle, *, label: str, master_before
) -> dict:
    update_successful, grad_norm, num_zeros = runtime.optimizer_step(handle)
    assert update_successful is True, f"{label}: optimizer rejected/skipped the update"
    assert torch.isfinite(
        torch.tensor(grad_norm)
    ), f"{label}: non-finite grad_norm={grad_norm}"
    assert grad_norm > 0.0, f"{label}: zero grad norm permits a no-op optimizer pass"
    changed_main = _assert_optimizer_master_changed(handle, master_before)
    optimizer_state_tensors = _assert_optimizer_state_initialized(handle)
    return {
        "grad_norm": grad_norm,
        "num_zeros": num_zeros,
        "changed_main": changed_main,
        "optimizer_state_tensors": optimizer_state_tensors,
    }


def _gather_and_validate_pipeline_layout(
    handle, cfg
) -> tuple[list[list[int]], list[dict]]:
    """Prove the global PP layer union and peer ownership, not only local continuity."""
    ps = handle._parallel_state
    local = _local_layer_indices(handle)
    stage = {
        "global_rank": dist.get_rank(),
        "pp_rank": ps.pp_rank,
        "tp_rank": ps.tp_rank,
        "ep_rank": ps.ep_rank,
        "layers": local,
    }
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, stage)

    by_stage: dict[int, list[dict]] = {}
    for item in gathered:
        by_stage.setdefault(item["pp_rank"], []).append(item)
    assert sorted(by_stage) == list(
        range(ps.pp_size)
    ), f"PP ranks are incomplete or out of range: {sorted(by_stage)}"
    expected_peers = dist.get_world_size() // ps.pp_size
    splits = []
    for pp_rank in range(ps.pp_size):
        peers = by_stage[pp_rank]
        assert (
            len(peers) == expected_peers
        ), f"PP stage {pp_rank} has {len(peers)} peers, want {expected_peers}"
        ownership = {tuple(item["layers"]) for item in peers}
        assert (
            len(ownership) == 1
        ), f"TP/EP/DP peers disagree on PP stage {pp_rank} layer ownership: {ownership}"
        splits.append(list(next(iter(ownership))))

    flat = [layer for split in splits for layer in split]
    expected = list(range(cfg.num_hidden_layers))
    assert flat == expected, (
        "global pipeline layout has duplicated, missing, or reordered decoder layers: "
        f"got {flat}, want {expected}"
    )
    return splits, gathered


def _is_valid_hf_export_key(key: str, model_name: str) -> bool:
    if model_name == "deepseek_v4":
        return key.startswith(("layers.", "mtp.")) or key in {
            "embed.weight",
            "head.weight",
            "norm.weight",
            "hc_head_base",
            "hc_head_fn",
            "hc_head_scale",
        }
    if model_name == "qwen3_5":
        return key.startswith(("model.", "mtp.")) or key == "lm_head.weight"
    return key.startswith("model.") or key == "lm_head.weight"


def _required_hf_roots(model_name: str) -> tuple[str, str, str]:
    if model_name == "deepseek_v4":
        return "embed.weight", "norm.weight", "head.weight"
    if model_name == "qwen3_5":
        return (
            "model.language_model.embed_tokens.weight",
            "model.language_model.norm.weight",
            "lm_head.weight",
        )
    return "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"


def _assert_exact_export_manifest(
    model_name: str,
    weights: dict[str, torch.Tensor],
    expected_shapes: dict[str, tuple[int, ...]],
) -> None:
    actual_keys = set(weights)
    expected_keys = set(expected_shapes)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    assert not missing and not unexpected, (
        f"{model_name}: export schema mismatch; missing={missing}, "
        f"unexpected={unexpected}"
    )
    for key, expected_shape in expected_shapes.items():
        actual_shape = tuple(weights[key].shape)
        assert (
            actual_shape == expected_shape
        ), f"{model_name}: {key} shape={actual_shape}, want {expected_shape}"
        if key.endswith(".tid2eid"):
            assert (
                weights[key].dtype == torch.int64
            ), f"{model_name}: {key} dtype={weights[key].dtype}, want torch.int64"


def _glm5_expected_export_shapes(cfg) -> dict[str, tuple[int, ...]]:
    hidden = cfg.hidden_size
    expected = {
        "model.embed_tokens.weight": (cfg.vocab_size, hidden),
        "model.norm.weight": (hidden,),
        "lm_head.weight": (cfg.vocab_size, hidden),
    }

    def add_layer(layer_idx: int, *, mtp: bool = False) -> None:
        prefix = f"model.layers.{layer_idx}."
        expected.update(
            {
                f"{prefix}input_layernorm.weight": (hidden,),
                f"{prefix}post_attention_layernorm.weight": (hidden,),
                f"{prefix}self_attn.q_a_proj.weight": (cfg.q_lora_rank, hidden),
                f"{prefix}self_attn.q_a_layernorm.weight": (cfg.q_lora_rank,),
                f"{prefix}self_attn.q_b_proj.weight": (
                    cfg.num_attention_heads * cfg.qk_head_dim,
                    cfg.q_lora_rank,
                ),
                f"{prefix}self_attn.kv_a_proj_with_mqa.weight": (
                    cfg.kv_lora_rank + cfg.qk_rope_head_dim,
                    hidden,
                ),
                f"{prefix}self_attn.kv_a_layernorm.weight": (cfg.kv_lora_rank,),
                f"{prefix}self_attn.kv_b_proj.weight": (
                    cfg.num_attention_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim),
                    cfg.kv_lora_rank,
                ),
                f"{prefix}self_attn.o_proj.weight": (
                    hidden,
                    cfg.num_attention_heads * cfg.v_head_dim,
                ),
            }
        )
        indexer_prefix = f"{prefix}self_attn.indexer."
        if cfg.builds_dsa_indexer(layer_idx):
            expected.update(
                {
                    f"{indexer_prefix}wq_b.weight": (
                        cfg.index_n_heads * cfg.index_head_dim,
                        cfg.q_lora_rank,
                    ),
                    f"{indexer_prefix}wk.weight": (cfg.index_head_dim, hidden),
                    f"{indexer_prefix}k_norm.weight": (cfg.index_head_dim,),
                    f"{indexer_prefix}k_norm.bias": (cfg.index_head_dim,),
                    f"{indexer_prefix}weights_proj.weight": (
                        cfg.index_n_heads,
                        cfg.index_head_dim,
                    ),
                }
            )

        mlp = f"{prefix}mlp."
        if cfg.is_moe_layer(layer_idx):
            shared_intermediate = cfg.n_shared_experts * cfg.moe_intermediate_size
            expected.update(
                {
                    f"{mlp}gate.weight": (cfg.num_experts, hidden),
                    f"{mlp}gate.e_score_correction_bias": (cfg.num_experts,),
                    f"{mlp}shared_experts.gate_proj.weight": (
                        shared_intermediate,
                        hidden,
                    ),
                    f"{mlp}shared_experts.up_proj.weight": (
                        shared_intermediate,
                        hidden,
                    ),
                    f"{mlp}shared_experts.down_proj.weight": (
                        hidden,
                        shared_intermediate,
                    ),
                }
            )
            for expert_idx in range(cfg.num_experts):
                expert = f"{mlp}experts.{expert_idx}."
                expected.update(
                    {
                        f"{expert}gate_proj.weight": (
                            cfg.moe_intermediate_size,
                            hidden,
                        ),
                        f"{expert}up_proj.weight": (cfg.moe_intermediate_size, hidden),
                        f"{expert}down_proj.weight": (
                            hidden,
                            cfg.moe_intermediate_size,
                        ),
                    }
                )
        else:
            expected.update(
                {
                    f"{mlp}gate_proj.weight": (cfg.intermediate_size, hidden),
                    f"{mlp}up_proj.weight": (cfg.intermediate_size, hidden),
                    f"{mlp}down_proj.weight": (hidden, cfg.intermediate_size),
                }
            )
        if mtp:
            expected.update(
                {
                    f"{prefix}enorm.weight": (hidden,),
                    f"{prefix}hnorm.weight": (hidden,),
                    f"{prefix}eh_proj.weight": (hidden, 2 * hidden),
                    f"{prefix}shared_head.norm.weight": (hidden,),
                }
            )

    for layer_idx in range(cfg.num_hidden_layers):
        add_layer(layer_idx)
    for mtp_idx in range(cfg.num_nextn_predict_layers):
        add_layer(cfg.num_hidden_layers + mtp_idx, mtp=True)
    return expected


def _deepseek_v4_expected_export_shapes(cfg) -> dict[str, tuple[int, ...]]:
    hidden = cfg.hidden_size
    expected = {
        "embed.weight": (cfg.vocab_size, hidden),
        "norm.weight": (hidden,),
        "head.weight": (cfg.vocab_size, hidden),
        "hc_head_fn": (cfg.hc_mult, cfg.hc_mult * hidden),
        "hc_head_base": (cfg.hc_mult,),
        "hc_head_scale": (1,),
    }

    def add_block(prefix: str, layer_idx: int, *, mtp: bool = False) -> None:
        ratio = (
            cfg.compress_ratios[min(layer_idx, len(cfg.compress_ratios) - 1)]
            if cfg.compress_ratios
            else 0
        )
        num_heads_per_group = cfg.num_attention_heads // cfg.o_groups
        hc_mix = (2 + cfg.hc_mult) * cfg.hc_mult
        expected.update(
            {
                f"{prefix}attn_norm.weight": (hidden,),
                f"{prefix}ffn_norm.weight": (hidden,),
                f"{prefix}attn.wq_a.weight": (cfg.q_lora_rank, hidden),
                f"{prefix}attn.q_norm.weight": (cfg.q_lora_rank,),
                f"{prefix}attn.wq_b.weight": (
                    cfg.num_attention_heads * cfg.head_dim,
                    cfg.q_lora_rank,
                ),
                f"{prefix}attn.wkv.weight": (cfg.head_dim, hidden),
                f"{prefix}attn.kv_norm.weight": (cfg.head_dim,),
                f"{prefix}attn.wo_a.weight": (
                    cfg.o_groups * cfg.o_lora_rank,
                    num_heads_per_group * cfg.head_dim,
                ),
                f"{prefix}attn.wo_b.weight": (hidden, cfg.o_groups * cfg.o_lora_rank),
                f"{prefix}attn.attn_sink": (cfg.num_attention_heads,),
                f"{prefix}hc_attn_fn": (hc_mix, cfg.hc_mult * hidden),
                f"{prefix}hc_attn_base": (hc_mix,),
                f"{prefix}hc_attn_scale": (3,),
                f"{prefix}hc_ffn_fn": (hc_mix, cfg.hc_mult * hidden),
                f"{prefix}hc_ffn_base": (hc_mix,),
                f"{prefix}hc_ffn_scale": (3,),
                f"{prefix}ffn.gate.weight": (cfg.num_experts, hidden),
                f"{prefix}ffn.shared_experts.w1.weight": (
                    cfg.n_shared_experts * cfg.moe_intermediate_size,
                    hidden,
                ),
                f"{prefix}ffn.shared_experts.w2.weight": (
                    hidden,
                    cfg.n_shared_experts * cfg.moe_intermediate_size,
                ),
                f"{prefix}ffn.shared_experts.w3.weight": (
                    cfg.n_shared_experts * cfg.moe_intermediate_size,
                    hidden,
                ),
            }
        )
        for expert_idx in range(cfg.num_experts):
            expert = f"{prefix}ffn.experts.{expert_idx}."
            expected.update(
                {
                    f"{expert}w1.weight": (cfg.moe_intermediate_size, hidden),
                    f"{expert}w2.weight": (hidden, cfg.moe_intermediate_size),
                    f"{expert}w3.weight": (cfg.moe_intermediate_size, hidden),
                }
            )
        if layer_idx < cfg.num_hash_layers and not mtp:
            expected[f"{prefix}ffn.gate.tid2eid"] = (
                cfg.vocab_size,
                cfg.num_experts_per_tok,
            )
        else:
            expected[f"{prefix}ffn.gate.bias"] = (cfg.num_experts,)

        if ratio > 1:
            compressor_width = (2 if ratio == 4 else 1) * cfg.head_dim
            compressor = f"{prefix}attn.compressor."
            expected.update(
                {
                    f"{compressor}wkv.weight": (compressor_width, hidden),
                    f"{compressor}wgate.weight": (compressor_width, hidden),
                    f"{compressor}ape": (ratio, compressor_width),
                    f"{compressor}norm.weight": (cfg.head_dim,),
                }
            )
        if ratio == 4:
            indexer = f"{prefix}attn.indexer."
            indexer_compressor_width = 2 * cfg.index_head_dim
            expected.update(
                {
                    f"{indexer}wq_b.weight": (
                        cfg.index_n_heads * cfg.index_head_dim,
                        cfg.q_lora_rank,
                    ),
                    f"{indexer}weights_proj.weight": (cfg.index_n_heads, hidden),
                    f"{indexer}compressor.wkv.weight": (
                        indexer_compressor_width,
                        hidden,
                    ),
                    f"{indexer}compressor.wgate.weight": (
                        indexer_compressor_width,
                        hidden,
                    ),
                    f"{indexer}compressor.ape": (ratio, indexer_compressor_width),
                    f"{indexer}compressor.norm.weight": (cfg.index_head_dim,),
                }
            )
        if mtp:
            expected.update(
                {
                    f"{prefix}e_proj.weight": (hidden, hidden),
                    f"{prefix}h_proj.weight": (hidden, hidden),
                    f"{prefix}enorm.weight": (hidden,),
                    f"{prefix}hnorm.weight": (hidden,),
                    f"{prefix}norm.weight": (hidden,),
                    f"{prefix}hc_head_fn": (cfg.hc_mult, cfg.hc_mult * hidden),
                    f"{prefix}hc_head_base": (cfg.hc_mult,),
                    f"{prefix}hc_head_scale": (1,),
                }
            )

    for layer_idx in range(cfg.num_hidden_layers):
        add_block(f"layers.{layer_idx}.", layer_idx)
    for mtp_idx in range(cfg.num_nextn_predict_layers):
        add_block(f"mtp.{mtp_idx}.", cfg.num_hidden_layers + mtp_idx, mtp=True)
    return expected


def _qwen35_expected_export_shapes(cfg) -> dict[str, tuple[int, ...]]:
    hidden = cfg.hidden_size
    expected = {
        "model.language_model.embed_tokens.weight": (cfg.vocab_size, hidden),
        "model.language_model.norm.weight": (hidden,),
        "lm_head.weight": (cfg.vocab_size, hidden),
    }
    qk_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim
    value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim
    for layer_idx in range(cfg.num_hidden_layers):
        prefix = f"model.language_model.layers.{layer_idx}."
        expected.update(
            {
                f"{prefix}input_layernorm.weight": (hidden,),
                f"{prefix}post_attention_layernorm.weight": (hidden,),
                f"{prefix}mlp.gate.weight": (cfg.num_experts, hidden),
                f"{prefix}mlp.shared_expert.gate_proj.weight": (
                    cfg.shared_expert_intermediate_size,
                    hidden,
                ),
                f"{prefix}mlp.shared_expert.up_proj.weight": (
                    cfg.shared_expert_intermediate_size,
                    hidden,
                ),
                f"{prefix}mlp.shared_expert.down_proj.weight": (
                    hidden,
                    cfg.shared_expert_intermediate_size,
                ),
                f"{prefix}mlp.shared_expert_gate.weight": (1, hidden),
                f"{prefix}mlp.experts.gate_up_proj": (
                    cfg.num_experts,
                    2 * cfg.moe_intermediate_size,
                    hidden,
                ),
                f"{prefix}mlp.experts.down_proj": (
                    cfg.num_experts,
                    hidden,
                    cfg.moe_intermediate_size,
                ),
            }
        )
        if cfg.layer_type_at(layer_idx) == "full_attention":
            attn = f"{prefix}self_attn."
            expected.update(
                {
                    f"{attn}q_proj.weight": (
                        2 * cfg.num_attention_heads * cfg.head_dim,
                        hidden,
                    ),
                    f"{attn}k_proj.weight": (
                        cfg.num_key_value_heads * cfg.head_dim,
                        hidden,
                    ),
                    f"{attn}v_proj.weight": (
                        cfg.num_key_value_heads * cfg.head_dim,
                        hidden,
                    ),
                    f"{attn}q_norm.weight": (cfg.head_dim,),
                    f"{attn}k_norm.weight": (cfg.head_dim,),
                    f"{attn}o_proj.weight": (
                        hidden,
                        cfg.num_attention_heads * cfg.head_dim,
                    ),
                }
            )
        else:
            attn = f"{prefix}linear_attn."
            expected.update(
                {
                    f"{attn}in_proj_qkv.weight": (2 * qk_dim + value_dim, hidden),
                    f"{attn}in_proj_z.weight": (value_dim, hidden),
                    f"{attn}in_proj_b.weight": (cfg.linear_num_value_heads, hidden),
                    f"{attn}in_proj_a.weight": (cfg.linear_num_value_heads, hidden),
                    f"{attn}conv1d.weight": (
                        2 * qk_dim + value_dim,
                        1,
                        cfg.linear_conv_kernel_dim,
                    ),
                    f"{attn}dt_bias": (cfg.linear_num_value_heads,),
                    f"{attn}A_log": (cfg.linear_num_value_heads,),
                    f"{attn}norm.weight": (cfg.linear_value_head_dim,),
                    f"{attn}out_proj.weight": (hidden, value_dim),
                }
            )
    if cfg.num_nextn_predict_layers:
        assert cfg.num_nextn_predict_layers == 1, (
            "the released Qwen3.5 checkpoint schema has exactly one physical MTP "
            f"predictor, got {cfg.num_nextn_predict_layers}"
        )
        expected.update(
            {
                "mtp.pre_fc_norm_embedding.weight": (hidden,),
                "mtp.pre_fc_norm_hidden.weight": (hidden,),
                "mtp.fc.weight": (hidden, 2 * hidden),
                "mtp.norm.weight": (hidden,),
            }
        )
        for mtp_idx in range(cfg.num_nextn_predict_layers):
            layer_type = cfg.layer_type_at(cfg.num_hidden_layers + mtp_idx)
            assert layer_type == "full_attention", (
                "the released Qwen3.5 MTP predictor must use full attention, got "
                f"mtp_layer_types[{mtp_idx}]={layer_type!r}"
            )
            prefix = f"mtp.layers.{mtp_idx}."
            expected.update(
                {
                    f"{prefix}input_layernorm.weight": (hidden,),
                    f"{prefix}post_attention_layernorm.weight": (hidden,),
                    f"{prefix}self_attn.q_proj.weight": (
                        2 * cfg.num_attention_heads * cfg.head_dim,
                        hidden,
                    ),
                    f"{prefix}self_attn.k_proj.weight": (
                        cfg.num_key_value_heads * cfg.head_dim,
                        hidden,
                    ),
                    f"{prefix}self_attn.v_proj.weight": (
                        cfg.num_key_value_heads * cfg.head_dim,
                        hidden,
                    ),
                    f"{prefix}self_attn.q_norm.weight": (cfg.head_dim,),
                    f"{prefix}self_attn.k_norm.weight": (cfg.head_dim,),
                    f"{prefix}self_attn.o_proj.weight": (
                        hidden,
                        cfg.num_attention_heads * cfg.head_dim,
                    ),
                    f"{prefix}mlp.gate.weight": (cfg.num_experts, hidden),
                    f"{prefix}mlp.shared_expert.gate_proj.weight": (
                        cfg.shared_expert_intermediate_size,
                        hidden,
                    ),
                    f"{prefix}mlp.shared_expert.up_proj.weight": (
                        cfg.shared_expert_intermediate_size,
                        hidden,
                    ),
                    f"{prefix}mlp.shared_expert.down_proj.weight": (
                        hidden,
                        cfg.shared_expert_intermediate_size,
                    ),
                    f"{prefix}mlp.shared_expert_gate.weight": (1, hidden),
                }
            )
            for expert_idx in range(cfg.num_experts):
                expert = f"{prefix}mlp.experts.{expert_idx}."
                expected.update(
                    {
                        f"{expert}gate_proj.weight": (
                            cfg.moe_intermediate_size,
                            hidden,
                        ),
                        f"{expert}up_proj.weight": (cfg.moe_intermediate_size, hidden),
                        f"{expert}down_proj.weight": (
                            hidden,
                            cfg.moe_intermediate_size,
                        ),
                    }
                )
    return expected


_QWEN35_FP32_CHECKPOINT_SUFFIXES = (".linear_attn.A_log", ".linear_attn.norm.weight")


def _expected_floating_export_dtype(model_name: str, key: str) -> torch.dtype:
    if model_name == "qwen3_5" and key.endswith(_QWEN35_FP32_CHECKPOINT_SUFFIXES):
        return torch.float32
    return torch.bfloat16


def _validate_model_specific_hf_keys(
    model_name: str, cfg, weights: dict[str, torch.Tensor]
) -> None:
    keys = set(weights)
    embed_key, norm_key, head_key = _required_hf_roots(model_name)
    for required in (embed_key, norm_key, head_key):
        assert (
            required in keys
        ), f"{model_name}: HF export is missing required tensor {required}"
    assert tuple(weights[embed_key].shape) == (cfg.vocab_size, cfg.hidden_size)
    assert tuple(weights[norm_key].shape) == (cfg.hidden_size,)
    assert tuple(weights[head_key].shape) == (cfg.vocab_size, cfg.hidden_size)

    if model_name == "glm5":
        _assert_exact_export_manifest(
            model_name, weights, _glm5_expected_export_shapes(cfg)
        )
        return
    if model_name == "deepseek_v4":
        _assert_exact_export_manifest(
            "deepseek_v4_v4flash_release",
            weights,
            _deepseek_v4_expected_export_shapes(cfg),
        )
        return
    if model_name == "qwen3_5":
        _assert_exact_export_manifest(
            model_name, weights, _qwen35_expected_export_shapes(cfg)
        )
        mtp_keys = {key for key in keys if key.startswith("mtp.")}
        assert (
            len(mtp_keys) == 17 + 3 * cfg.num_experts
        ), f"qwen3_5: incomplete released MTP schema; got {len(mtp_keys)} keys"
        return

    if model_name == "kimi_k2":
        for layer_idx in range(cfg.num_hidden_layers):
            if not cfg.is_moe_layer(layer_idx):
                continue
            bias_key = f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias"
            assert (
                bias_key in keys
            ), f"kimi_k2: PP export is missing router correction buffer {bias_key}"


def _validate_rank0_export_and_reload(exported, cfg, model_name: str, tmp_path) -> int:
    """Validate and persist the rank-0 materialization; caller synchronizes failures."""
    names = [name for name, _ in exported]
    seen: set[str] = set()
    duplicate_names: set[str] = set()
    for name in names:
        if name in seen:
            duplicate_names.add(name)
        seen.add(name)
    duplicates = sorted(duplicate_names)
    assert not duplicates, f"{model_name}: duplicate HF export keys: {duplicates}"
    weights = {name: tensor.detach().cpu().contiguous() for name, tensor in exported}
    assert weights, f"{model_name}: export returned zero tensors"
    for name, tensor in weights.items():
        assert _is_valid_hf_export_key(
            name, model_name
        ), f"{model_name}: invalid HF key schema {name}"
        assert tensor.numel() > 0, f"{model_name}: empty exported tensor {name}"
        assert torch.isfinite(
            tensor.float()
        ).all(), f"{model_name}: non-finite exported tensor {name}"
        if tensor.dtype.is_floating_point:
            expected_dtype = _expected_floating_export_dtype(model_name, name)
            assert tensor.dtype == expected_dtype, (
                f"{model_name}: {name} exported as {tensor.dtype}, "
                f"want {expected_dtype}"
            )
        else:
            assert tensor.dtype in (
                torch.int64,
                torch.int32,
                torch.bool,
            ), f"{model_name}: {name} has unexpected integer dtype {tensor.dtype}"

    present = _hf_decoder_layer_indices(weights)
    expected = set(range(cfg.num_hidden_layers))
    assert expected.issubset(
        present
    ), f"{model_name}: uneven-PP export dropped decoder layers {sorted(expected - present)}"
    _validate_model_specific_hf_keys(model_name, cfg, weights)

    from megatron.lite.primitive.ckpt.hf_weights import save_safetensors
    from safetensors import safe_open

    export_dir = os.path.join(str(tmp_path), f"{model_name}-hf-export")
    save_safetensors(weights, export_dir)
    reloaded = {}
    with safe_open(
        os.path.join(export_dir, "model.safetensors"), framework="pt"
    ) as reader:
        for name in reader.keys():
            reloaded[name] = reader.get_tensor(name)
    assert (
        reloaded.keys() == weights.keys()
    ), f"{model_name}: safetensors reload key mismatch"
    for name in weights:
        assert torch.equal(
            reloaded[name], weights[name]
        ), f"{model_name}: safetensors reload changed tensor {name}"
    return len(weights)


def _export_validate_and_reload(
    handle, cfg, protocol, model_name: str, tmp_path
) -> int:
    """Collectively export once and make every rank fail together on validation errors."""
    exported = list(
        protocol.export_hf_weights(
            handle._extras["model_chunks"],
            cfg,
            handle._parallel_state,
            rank0_only=True,
            # Preserve the official Qwen3.5 mixed-dtype checkpoint contract:
            # GDN A_log and norm stay FP32 while dt_bias and all other text/MTP
            # tensors stay BF16. Other proxy models intentionally force BF16.
            export_dtype=None if model_name == "qwen3_5" else torch.bfloat16,
        )
    )
    rank = dist.get_rank()
    materialized_counts = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(materialized_counts, len(exported))

    outcome: list[object] = [None, 0]
    if rank == 0:
        try:
            assert materialized_counts[0] > 0, "rank 0 materialized no HF tensors"
            assert all(count == 0 for count in materialized_counts[1:]), (
                "rank0_only HF export materialized tensors off rank 0: "
                f"counts={materialized_counts}"
            )
            outcome[1] = _validate_rank0_export_and_reload(
                exported, cfg, model_name, tmp_path
            )
        except Exception as exc:  # synchronize expected assertion failures
            outcome[0] = f"{type(exc).__name__}: {exc}"
    dist.broadcast_object_list(outcome, src=0)
    if outcome[0] is not None:
        raise AssertionError(
            f"{model_name}: synchronized HF export failure: {outcome[0]}"
        )
    return int(outcome[1]) if rank == 0 else 0


@pytest.mark.parametrize("model_name", list(MODELS))
def test_uneven_pp_builds_trains_and_exports(model_name, tmp_path):
    """A non-divisible layer count builds an uneven PP split, trains a step, and
    exports every layer. Build + train + export share one model build (one heavy
    distributed build per test keeps the builds in one torchrun process bounded).

    - build: the non-divisible count produces a valid balanced uneven split (not
      "not divisible"); this stage owns a sorted, in-range, contiguous decoder run.
    - train: one real step over the uneven pipeline yields a finite loss (proves PP
      P2P across stages of unequal depth actually runs the built model).
    - export: HF export all-gathers the per-stage shards (keyed on global layer
      names) back to a complete state — every decoder layer 0..N-1 reconstructed,
      every tensor finite. Regression guard for the deepseek_v4 local-vs-global
      layer-key bug; a dropped/duplicated stage would silently corrupt exports.
    """
    if dist.get_world_size() != 8:
        pytest.skip("auto pipeline-layout proxy smoke requires exactly 8 GPUs.")

    set_deterministic(2026)

    handle, cfg = _build_handle(model_name, seed=4242)
    ps = handle._parallel_state
    assert ps.pp_size == _PP
    assert ps.tp_size == (1 if model_name in _TP1_ONLY else 2)
    assert ps.ep_size == 2
    assert ps.cp_size == 1
    assert cfg.num_hidden_layers == _NUM_LAYERS

    local = _local_layer_indices(handle)
    assert local == sorted(local) and len(set(local)) == len(local), local
    assert local and all(0 <= i < _NUM_LAYERS for i in local), local
    assert local == list(range(local[0], local[0] + len(local))), local
    splits, _layout = _gather_and_validate_pipeline_layout(handle, cfg)

    local_model = unwrap_model(handle._extras["model_chunks"][0])
    if model_name == "glm5":
        assert cfg.rope_interleave is True
        assert cfg.indexer_rope_interleave is True
        assert cfg.dsa_rope_layout_revision == "legacy"
        assert cfg.uses_configured_dsa_rope_layout is False
        for layer in local_model.layers:
            dsa = layer.self_attention.self_attention
            assert dsa.rope_interleaved is False
            assert dsa.indexer is not None
            assert dsa.indexer.rope_interleaved is False

    captured_mtp_losses: list[float] = []
    mtp_hook = None
    mtp_enabled_model = model_name in {"qwen3_5", "glm5", "deepseek_v4"}
    local_has_mtp = mtp_enabled_model and (
        len(local_model.mtp) > 0
        if model_name == "deepseek_v4"
        else local_model.mtp is not None
    )
    if local_has_mtp:

        def _capture_mtp_loss(_module, _inputs, output):
            mtp_loss = output.get("mtp_loss") if isinstance(output, dict) else None
            if mtp_loss is not None:
                captured_mtp_losses.append(float(mtp_loss.detach().float().item()))

        mtp_hook = local_model.register_forward_hook(_capture_mtp_loss)

    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)
    batch = _random_packed_batch(cfg.vocab_size)
    runtime.zero_grad(handle)
    try:
        result = runtime.forward_backward(
            handle, iter([batch]), None, num_microbatches=1
        )
    finally:
        if mtp_hook is not None:
            mtp_hook.remove()
    loss = result.model_output.loss
    assert (
        loss is not None and torch.isfinite(loss).all()
    ), f"{model_name}: non-finite loss {loss} on uneven PP layout"
    named_parameters = _named_local_parameters(handle)
    nonzero_grads = _assert_finite_nonzero_gradient_signal(
        named_parameters, label=f"{model_name} uneven-PP model"
    )
    indexer_parameters = (
        [
            (name, parameter)
            for name, parameter in named_parameters
            if ".indexer." in name
        ]
        if model_name == "glm5"
        else []
    )
    nonzero_indexer_grads = (
        _assert_finite_nonzero_gradients(
            indexer_parameters, label=f"{model_name} full indexers"
        )
        if model_name == "glm5"
        else []
    )
    mtp_parameters = (
        [
            (name, parameter)
            for name, parameter in named_parameters
            if ".mtp." in name or ".mtp_embed." in name
        ]
        if mtp_enabled_model
        else []
    )
    if local_has_mtp:
        assert captured_mtp_losses, f"{model_name} MTP stage produced no mtp_loss"
        assert all(
            torch.isfinite(torch.tensor(value)) and value > 0.0
            for value in captured_mtp_losses
        )
        nonzero_mtp_grads = _assert_finite_nonzero_gradients(
            mtp_parameters, label=f"{model_name} MTP block"
        )
    else:
        assert (
            not mtp_parameters
        ), f"non-MTP stage unexpectedly owns {model_name} MTP parameters"
        assert not captured_mtp_losses
        nonzero_mtp_grads = []
    model_before = _snapshot_named_parameters(named_parameters)
    indexer_before = _snapshot_named_parameters(indexer_parameters)
    mtp_before = _snapshot_named_parameters(mtp_parameters)
    master_before = _optimizer_master_snapshot(handle)
    optimizer_stats = _assert_successful_optimizer_step(
        runtime,
        handle,
        label=f"{model_name} uneven-PP model",
        master_before=master_before,
    )
    changed_model = _assert_snapshot_changed(
        model_before, named_parameters, label=f"{model_name} uneven-PP model"
    )
    changed_indexers = (
        _assert_snapshot_changed(
            indexer_before, indexer_parameters, label=f"{model_name} full indexers"
        )
        if indexer_parameters
        else []
    )
    changed_mtp = (
        _assert_snapshot_changed(
            mtp_before, mtp_parameters, label=f"{model_name} MTP block"
        )
        if mtp_parameters
        else []
    )
    changed_mtp_embedding = [name for name in changed_mtp if ".mtp_embed." in name]
    if local_has_mtp and ps.pp_size > 1:
        assert any(
            ".mtp_embed." in name for name in nonzero_mtp_grads
        ), f"{model_name}: PP MTP embedding replica received no nonzero gradient"
        assert (
            changed_mtp_embedding
        ), f"{model_name}: optimizer did not update the PP MTP embedding replica"
    # Export across the uneven PP split, persist it, and exact-reload it. This is
    # deliberately stronger than merely finding one finite key per decoder layer.
    protocol = handle._extras["protocol"]
    exported_key_count = _export_validate_and_reload(
        handle, cfg, protocol, model_name, tmp_path
    )
    local_evidence = {
        "pp_rank": ps.pp_rank,
        "has_mtp": local_has_mtp,
        "mtp_losses": captured_mtp_losses,
        "nonzero_mtp_grads": len(nonzero_mtp_grads),
        "changed_mtp": len(changed_mtp),
        "changed_mtp_embedding": len(changed_mtp_embedding),
    }
    gathered_evidence = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_evidence, local_evidence)
    mtp_peers = [item for item in gathered_evidence if item["has_mtp"]]
    if mtp_enabled_model:
        expected_mtp_peers = dist.get_world_size() // ps.pp_size
        assert len(mtp_peers) == expected_mtp_peers
        assert {item["pp_rank"] for item in mtp_peers} == {ps.pp_size - 1}
        assert all(item["mtp_losses"] for item in mtp_peers)
        assert all(item["nonzero_mtp_grads"] > 0 for item in mtp_peers)
        assert all(item["changed_mtp"] > 0 for item in mtp_peers)
        assert all(item["changed_mtp_embedding"] > 0 for item in mtp_peers)
        min_mtp_loss = min(value for item in mtp_peers for value in item["mtp_losses"])
        min_mtp_grads = min(item["nonzero_mtp_grads"] for item in mtp_peers)
        min_changed_mtp = min(item["changed_mtp"] for item in mtp_peers)
        min_changed_mtp_embedding = min(
            item["changed_mtp_embedding"] for item in mtp_peers
        )
    else:
        assert not mtp_peers
        min_mtp_loss = 0.0
        min_mtp_grads = 0
        min_changed_mtp = 0
        min_changed_mtp_embedding = 0
    if dist.get_rank() == 0:
        print(
            "NON_SKIP_UNEVEN_PP_BUILD_TRAIN_EXPORT "
            f"model={model_name} num_layers={cfg.num_hidden_layers} "
            f"tp={ps.tp_size} ep={ps.ep_size} pp={ps.pp_size} splits={splits} "
            f"nonzero_grad_params={len(nonzero_grads)} "
            f"nonzero_indexer_grad_params={len(nonzero_indexer_grads)} "
            f"changed_model_params={len(changed_model)} "
            f"changed_indexer_params={len(changed_indexers)} "
            f"changed_main_params={optimizer_stats['changed_main']} "
            f"grad_norm={optimizer_stats['grad_norm']:.6e} "
            f"optimizer_state_tensors={optimizer_stats['optimizer_state_tensors']} "
            f"mtp_train_peers={len(mtp_peers)} mtp_loss_min={min_mtp_loss:.6e} "
            f"mtp_nonzero_grad_params_min={min_mtp_grads} "
            f"mtp_changed_params_min={min_changed_mtp} "
            f"mtp_changed_embedding_params_min={min_changed_mtp_embedding} "
            f"mtp_embedding_export_exact={mtp_enabled_model} "
            f"qwen_mtp_schema_keys={17 + 3 * cfg.num_experts if model_name == 'qwen3_5' else 0} "
            f"qwen_mixed_dtype_exact={model_name == 'qwen3_5'} "
            f"exported_keys={exported_key_count} safetensors_reload_exact=True"
        )


def test_glm52_indexshare_78_layer_uneven_pp_builds_trains_and_keeps_share_groups():
    """GLM5.2 has 78 trunk layers plus one MTP layer and reuses DSA indexer
    top-k across groups. The real PP build must use the auto layout while keeping
    each full source layer on the same stage as its shared layers."""
    if dist.get_world_size() != 8:
        pytest.skip("GLM5.2 IndexShare uneven-PP smoke requires exactly 8 GPUs.")

    from megatron.lite.primitive.modules.attention.dsa import (
        validate_dsa_index_share_pipeline_split,
    )

    set_deterministic(2026)

    cfg, protocol = _glm52_indexshare()
    assert cfg.rope_interleave is True
    assert cfg.indexer_rope_interleave is True
    assert cfg.dsa_rope_layout_revision == "configured"
    assert cfg.uses_configured_dsa_rope_layout is True
    parallel = ParallelConfig(tp=1, ep=1, etp=1, pp=8, cp=1)
    handle, cfg = _build_handle_from_config(
        "glm5",
        cfg,
        protocol,
        seed=5252,
        parallel=parallel,
        impl_overrides={"mtp_enable": True, "mtp_enable_train": True},
    )
    ps = handle._parallel_state
    assert ps.pp_size == 8
    assert cfg.num_hidden_layers == 78
    assert cfg.num_nextn_predict_layers == 1
    assert cfg.uses_dsa_index_share is True
    assert cfg.dsa_indexer_loss_coeff == pytest.approx(1.0e-2)

    local = _local_layer_indices(handle)
    assert local == sorted(local) and len(set(local)) == len(local), local
    validate_dsa_index_share_pipeline_split(
        local,
        topk_freq=cfg.index_topk_freq,
        skip_topk_offset=cfg.index_skip_topk_offset,
        indexer_types=cfg.indexer_types,
    )
    local_shared_sources = [
        (layer_idx, cfg.dsa_indexer_source_layer(layer_idx))
        for layer_idx in local
        if cfg.dsa_indexer_type(layer_idx) == "shared"
    ]
    assert all(
        source_idx in local for _, source_idx in local_shared_sources
    ), local_shared_sources
    validated_splits, _layout = _gather_and_validate_pipeline_layout(handle, cfg)

    local_model = unwrap_model(handle._extras["model_chunks"][0])
    for layer_idx, layer in zip(
        local_model.layer_indices, local_model.layers, strict=True
    ):
        dsa = layer.self_attention.self_attention
        assert dsa.rope_interleaved is True
        assert (dsa.indexer is not None) is (cfg.dsa_indexer_type(layer_idx) == "full")
        if dsa.indexer is not None:
            assert dsa.indexer.rope_interleaved is True
    captured_mtp_losses: list[float] = []
    mtp_hook = None
    if local_model.mtp is not None:
        for mtp_layer in local_model.mtp.layers:
            mtp_dsa = mtp_layer.transformer_layer.self_attention.self_attention
            assert mtp_dsa.rope_interleaved is True
            assert mtp_dsa.indexer is not None
            assert mtp_dsa.indexer.rope_interleaved is True

        def _capture_mtp_loss(_module, _inputs, output):
            mtp_loss = output.get("mtp_loss") if isinstance(output, dict) else None
            if mtp_loss is not None:
                captured_mtp_losses.append(float(mtp_loss.detach().float().item()))

        mtp_hook = local_model.register_forward_hook(_capture_mtp_loss)

    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)
    batch = _random_packed_batch(cfg.vocab_size, num_tokens=1024)
    assert cfg.index_topk < batch.input_ids.numel()
    runtime.zero_grad(handle)
    try:
        result = runtime.forward_backward(
            handle, iter([batch]), None, num_microbatches=1
        )
    finally:
        if mtp_hook is not None:
            mtp_hook.remove()
    loss = result.model_output.loss
    assert (
        loss is not None and torch.isfinite(loss).all()
    ), f"glm52_indexshare: non-finite loss {loss} on 78-layer uneven PP layout"

    named_parameters = _named_local_parameters(handle)
    _assert_finite_nonzero_gradient_signal(
        named_parameters, label="glm52 full local model"
    )
    trunk_indexer_parameters = [
        (name, parameter)
        for name, parameter in named_parameters
        if ".indexer." in name and ".mtp." not in name
    ]
    nonzero_trunk_indexer_grads = _assert_finite_nonzero_gradients(
        trunk_indexer_parameters, label="glm52 full trunk indexers"
    )
    mtp_parameters = [
        (name, parameter)
        for name, parameter in named_parameters
        if ".mtp." in name or ".mtp_embed." in name
    ]
    boundary_embedding = None
    if ps.pp_is_first:
        assert local_model.embed is not None
        boundary_embedding = local_model.embed.embedding.weight
    elif ps.pp_is_last:
        assert local_model.mtp_embed is not None
        boundary_embedding = local_model.mtp_embed.embedding.weight
    boundary_embedding_before = (
        boundary_embedding.detach().cpu().clone()
        if boundary_embedding is not None
        else None
    )
    mtp_indexer_parameters = [
        (name, parameter) for name, parameter in mtp_parameters if ".indexer." in name
    ]
    if local_model.mtp is not None:
        assert captured_mtp_losses, "glm52 MTP rank did not produce mtp_loss"
        assert all(
            torch.isfinite(torch.tensor(value)) and value > 0.0
            for value in captured_mtp_losses
        )
        nonzero_mtp_grads = _assert_finite_nonzero_gradients(
            mtp_parameters, label="glm52 MTP block"
        )
        nonzero_mtp_indexer_grads = _assert_finite_nonzero_gradients(
            mtp_indexer_parameters, label="glm52 MTP full indexer"
        )
    else:
        assert not mtp_parameters, "non-MTP PP stage unexpectedly owns MTP parameters"
        assert not captured_mtp_losses
        nonzero_mtp_grads = []
        nonzero_mtp_indexer_grads = []

    trunk_indexer_before = _snapshot_named_parameters(trunk_indexer_parameters)
    mtp_before = _snapshot_named_parameters(mtp_parameters)
    master_before = _optimizer_master_snapshot(handle)
    optimizer_stats = _assert_successful_optimizer_step(
        runtime, handle, label="glm52 IndexShare PP8", master_before=master_before
    )
    changed_trunk_indexers = _assert_snapshot_changed(
        trunk_indexer_before,
        trunk_indexer_parameters,
        label="glm52 full trunk indexers",
    )
    changed_mtp = (
        _assert_snapshot_changed(mtp_before, mtp_parameters, label="glm52 MTP block")
        if mtp_parameters
        else []
    )
    changed_mtp_embedding = [name for name in changed_mtp if ".mtp_embed." in name]
    if local_model.mtp is not None and ps.pp_size > 1:
        assert any(".mtp_embed." in name for name in nonzero_mtp_grads)
        assert changed_mtp_embedding

    boundary_embedding_after = (
        boundary_embedding.detach().cpu().clone()
        if boundary_embedding is not None
        else None
    )
    boundary_embedding_changed = False
    if boundary_embedding_before is not None:
        assert boundary_embedding_after is not None
        boundary_embedding_changed = not torch.equal(
            boundary_embedding_before, boundary_embedding_after
        )
        assert (
            boundary_embedding_changed
        ), f"GLM5.2 PP boundary embedding did not update on pp_rank={ps.pp_rank}"

    stage_info = {
        "pp_rank": ps.pp_rank,
        "layers": local,
        "shared_sources": local_shared_sources,
        "has_mtp": bool(
            unwrap_model(handle._extras["model_chunks"][0]).mtp is not None
        ),
        "mtp_losses": captured_mtp_losses,
        "nonzero_trunk_indexer_grads": len(nonzero_trunk_indexer_grads),
        "nonzero_mtp_grads": len(nonzero_mtp_grads),
        "nonzero_mtp_indexer_grads": len(nonzero_mtp_indexer_grads),
        "changed_trunk_indexers": len(changed_trunk_indexers),
        "changed_mtp": len(changed_mtp),
        "changed_mtp_embedding": len(changed_mtp_embedding),
        "boundary_embedding": boundary_embedding_after,
        "boundary_embedding_changed": boundary_embedding_changed,
        "grad_norm": optimizer_stats["grad_norm"],
    }
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, stage_info)
    ordered = sorted(gathered, key=lambda item: item["pp_rank"])
    splits = [item["layers"] for item in ordered]
    shared_pairs = sum(len(item["shared_sources"]) for item in ordered)
    mtp_ranks = [item["pp_rank"] for item in ordered if item["has_mtp"]]
    mtp_losses = [value for item in ordered for value in item["mtp_losses"]]
    assert [item["pp_rank"] for item in ordered] == list(range(8))
    assert splits == validated_splits
    assert shared_pairs == 57
    assert mtp_ranks == [7]
    assert len(mtp_losses) == 1 and mtp_losses[0] > 0.0
    assert all(item["nonzero_trunk_indexer_grads"] > 0 for item in ordered)
    assert ordered[7]["nonzero_mtp_grads"] > 0
    assert ordered[7]["nonzero_mtp_indexer_grads"] > 0
    assert ordered[7]["changed_mtp"] > 0
    assert ordered[7]["changed_mtp_embedding"] > 0
    assert ordered[0]["boundary_embedding_changed"] is True
    assert ordered[7]["boundary_embedding_changed"] is True
    assert torch.equal(
        ordered[0]["boundary_embedding"], ordered[7]["boundary_embedding"]
    ), "canonical and MTP-stage embedding diverged after the optimizer step"
    assert all(
        item["boundary_embedding"] is None for item in ordered[1:7]
    ), "a middle PP stage unexpectedly reported a shared embedding replica"
    assert all(item["changed_trunk_indexers"] > 0 for item in ordered)
    if dist.get_rank() == 0:
        print(
            "NON_SKIP_GLM52_INDEXSHARE_UNEVEN_PP_PASSED "
            f"pp=8 num_layers=78 mtp_layers=1 splits={splits} "
            f"shared_pairs={shared_pairs} mtp_ranks={mtp_ranks} "
            f"topk={cfg.index_topk} seq={batch.input_ids.numel()} "
            f"indexer_loss_coeff={cfg.dsa_indexer_loss_coeff:.1e} "
            f"mtp_loss={mtp_losses[0]:.6e} "
            f"min_nonzero_trunk_indexer_grads="
            f"{min(item['nonzero_trunk_indexer_grads'] for item in ordered)} "
            f"mtp_nonzero_grads={ordered[7]['nonzero_mtp_grads']} "
            f"mtp_nonzero_indexer_grads={ordered[7]['nonzero_mtp_indexer_grads']} "
            f"min_changed_trunk_indexers="
            f"{min(item['changed_trunk_indexers'] for item in ordered)} "
            f"mtp_changed_params={ordered[7]['changed_mtp']} "
            f"mtp_changed_embedding_params={ordered[7]['changed_mtp_embedding']} "
            "mtp_embedding_pair_updated=True mtp_embedding_pair_exact=True "
            f"min_grad_norm={min(item['grad_norm'] for item in ordered):.6e} "
            f"loss={float(loss.detach().item()):.6e}"
        )


# Custom-string mode (``ParallelConfig.pp_layout``) is covered by the deterministic
# unit tests (test_parallel_unit.py: the explicit string is honored verbatim, differs
# from the auto split, and a >pp-stage string raises). It shares the build path with
# the auto matrix above (only ``ps.pp_layout`` differs), so it is not given a separate
# GPU build here: keeping this smoke to one heavy distributed build per model avoids a
# trailing build tripping NCCL P2P comm-init timeouts from process-group churn.
