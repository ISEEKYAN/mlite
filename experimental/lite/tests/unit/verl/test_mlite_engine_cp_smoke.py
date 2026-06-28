# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

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
        pytest.skip("CUDA is required for MLite VERL CP smoke.")
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("Run with torchrun so CP ranks are available.")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    if dist.get_world_size() < 2:
        pytest.skip("MLite VERL CP smoke requires at least 2 ranks.")
    return torch.device("cuda", local_rank)


def _write_kimi_config(path) -> None:
    config = {
        "model_type": "deepseek_v3",
        "num_hidden_layers": 2,
        "hidden_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "vocab_size": 128,
        "intermediate_size": 96,
        "moe_intermediate_size": 16,
        "n_routed_experts": 4,
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


def _write_glm52_config(path) -> None:
    config = {
        "model_type": "glm_moe_dsa",
        "num_hidden_layers": 2,
        "hidden_size": 128,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "head_dim": 256,
        "vocab_size": 32,
        "max_position_embeddings": 1024,
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
        "index_topk_freq": 2,
        "index_skip_topk_offset": 1,
        "indexer_types": ["full", "shared"],
        "rope_interleave": True,
        "indexer_rope_interleave": True,
        "dsa_indexer_loss_coeff": 1.0e-2,
        "intermediate_size": 20,
        "moe_intermediate_size": 6,
        "first_k_dense_replace": 1,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "n_group": 1,
        "topk_group": 1,
        "num_nextn_predict_layers": 1,
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
        "n_routed_experts": 4,
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


def _write_qwen35_config(path) -> None:
    # Keep the released nested text-config shape while making both Qwen3.5
    # attention families small enough for an 8-rank CP integration smoke.
    config = {
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe",
            "num_hidden_layers": 2,
            "hidden_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "vocab_size": 128,
            "rms_norm_eps": 1.0e-6,
            "max_position_embeddings": 128,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 16,
            "shared_expert_intermediate_size": 16,
            "linear_num_key_heads": 4,
            "linear_key_head_dim": 8,
            "linear_num_value_heads": 4,
            "linear_value_head_dim": 8,
            "linear_conv_kernel_dim": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "partial_rotary_factor": 1.0,
            "mrope_section": [1, 1, 2],
            "num_nextn_predict_layers": 0,
            "rope_parameters": {
                "rope_theta": 10_000_000.0,
                "partial_rotary_factor": 1.0,
                "mrope_section": [1, 1, 2],
            },
        },
    }
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")


def _assert_built_model_contract(engine, model_name: str) -> str:
    """Prove the requested release path was built, not only serialized."""

    assert engine._mlite_config.model_name == model_name
    model = _unwrap_primary_model(engine)
    config = model.config
    assert model.ps.cp_size == engine.engine_config.cp
    assert model.ps.cp_size > 1

    if model_name == "glm5":
        assert config.resolved_dsa_indexer_types == ("full", "shared")
        assert config.uses_configured_dsa_rope_layout is True
        assert config.rope_interleave is True
        assert config.indexer_rope_interleave is True
        assert config.dsa_indexer_loss_coeff == pytest.approx(1.0e-2)
        assert config.mlp_layer_types == ["dense", "sparse"]
        full_dsa = model.layers[0].self_attention.self_attention
        shared_dsa = model.layers[1].self_attention.self_attention
        assert full_dsa.skip_topk is False
        assert full_dsa.indexer is not None
        assert full_dsa.rope_interleaved is True
        assert full_dsa.indexer.rope_interleaved is True
        assert shared_dsa.skip_topk is True
        assert shared_dsa.indexer is None
        assert shared_dsa.rope_interleaved is True
        return "variant=glm52 dsa_schedule=full,shared rope_layout=interleaved"

    if model_name == "qwen3_5":
        assert config.layer_types == ["linear_attention", "full_attention"]
        assert config.mrope_section == [1, 1, 2]
        assert model.layers[0].linear_attn is not None
        assert model.layers[0].full_attn is None
        assert model.layers[0].linear_attn.cp_mode == "fla_allgather"
        assert model.layers[1].linear_attn is None
        assert model.layers[1].full_attn is not None
        return "variant=qwen3.5 attention_schedule=linear,full"

    if model_name == "deepseek_v4":
        import torch
        from megatron.lite.model.deepseek_v4.lite.model import DeepseekV4CSAAttention
        from megatron.lite.primitive.modules.attention.csa import (
            CompressedSparseAttention,
        )

        first_layer = model.layers["0"]
        assert isinstance(first_layer.self_attn, DeepseekV4CSAAttention)
        csa = first_layer.self_attn.self_attn
        assert isinstance(csa, CompressedSparseAttention)
        assert csa.compress_ratio == config.compress_ratios[0] == 4
        assert csa.compressor is not None
        assert csa.indexer is not None
        assert first_layer.attn_hc is not None
        assert first_layer.ffn_hc is not None
        assert first_layer.attn_hc.hc_mult == config.hc_mult
        assert first_layer.ffn_hc.hc_mult == config.hc_mult
        assert model.hc_mult == config.hc_mult
        assert first_layer.mlp.is_hash_layer is True
        tid2eid = first_layer.mlp.gate.get_buffer("tid2eid")
        assert tid2eid.shape == (config.vocab_size, config.num_experts_per_tok)
        assert tid2eid.dtype == torch.int64
        return "variant=deepseek-v4 csa=true mhc=true hash_moe=true"
    return "variant=kimi-k2 optional=true"


def _unwrap_primary_model(engine):
    model = engine.module
    while not all(hasattr(model, name) for name in ("config", "layers", "ps")):
        wrapped = getattr(model, "module", None)
        if wrapped is None or wrapped is model:
            raise AssertionError(
                f"Cannot resolve primary model through {type(model).__name__}."
            )
        model = wrapped
    return model


def _parameter_gradient(parameter):
    # Megatron-Core DDP accumulates the optimizer-facing gradient in
    # ``main_grad``; fall back to the ordinary autograd slot for unwrapped
    # implementations.
    gradient = getattr(parameter, "main_grad", None)
    if gradient is None:
        gradient = parameter.grad
    return gradient


def _optimizer_master_snapshot(engine) -> list:
    """Snapshot the FP32/main parameters owned by this dist-opt rank."""

    parameters = list(engine.handle._optimizer.get_parameters())
    assert parameters, "distributed optimizer exposes no local main parameters"
    return [parameter.detach().cpu().clone() for parameter in parameters]


def _assert_optimizer_master_changed(engine, before: list) -> tuple[int, float]:
    """Reject a successful return code when no optimizer-owned value changed."""

    import torch

    after = list(engine.handle._optimizer.get_parameters())
    assert len(before) == len(
        after
    ), "optimizer main-parameter count changed during step"
    deltas = [
        (old - new.detach().cpu()).abs().max().float()
        for old, new in zip(before, after, strict=True)
    ]
    assert deltas, "distributed optimizer exposes no post-step main parameters"
    max_abs_delta = torch.stack(deltas).max()
    assert torch.isfinite(max_abs_delta), "optimizer master delta is non-finite"
    changed = sum(delta.item() > 0.0 for delta in deltas)
    assert changed > 0, "optimizer reported success but no FP32/main parameter changed"
    assert max_abs_delta.item() > 0.0
    return changed, float(max_abs_delta.item())


def _assert_optimizer_state_initialized(engine) -> int:
    """Require finite, nonzero Adam moving averages after the first step."""

    import torch
    from megatron.core.optimizer import ChainedOptimizer

    optimizer = engine.handle._optimizer
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


def _assert_model_specific_grad_contract(engine, model_name: str) -> str:
    """Require gradients through the model-specific path on every CP rank."""

    import torch

    def assert_module_has_nonzero_grads(module, *, label: str) -> None:
        gradients = [
            gradient.detach().float()
            for parameter in module.parameters()
            if (gradient := _parameter_gradient(parameter)) is not None
        ]
        assert gradients, f"{label} has no gradients"
        grad_norm = sum(gradient.norm() for gradient in gradients)
        assert torch.isfinite(grad_norm), f"{label} has non-finite gradients"
        assert grad_norm.item() > 0.0, f"{label} has only zero gradients"

    model = _unwrap_primary_model(engine)
    if model_name == "glm5":
        indexer = model.layers[0].self_attention.self_attention.indexer
        assert indexer is not None
        assert_module_has_nonzero_grads(indexer, label="GLM-5.2 full DSA indexer")
        return "indexer_grad=nonzero"

    if model_name == "qwen3_5":
        linear_attention = model.layers[0].linear_attn
        full_attention = model.layers[1].full_attn
        assert linear_attention is not None
        assert full_attention is not None
        assert_module_has_nonzero_grads(
            linear_attention, label="Qwen3.5 linear attention"
        )
        assert_module_has_nonzero_grads(full_attention, label="Qwen3.5 full attention")
        return "linear_grad=nonzero full_grad=nonzero"

    if model_name == "deepseek_v4":
        first_layer = model.layers["0"]
        csa = first_layer.self_attn.self_attn
        assert_module_has_nonzero_grads(csa, label="DeepSeek-V4 CSA")
        assert_module_has_nonzero_grads(
            csa.compressor, label="DeepSeek-V4 ratio-4 CSA compressor"
        )
        assert_module_has_nonzero_grads(
            first_layer.attn_hc, label="DeepSeek-V4 attention mHC"
        )
        assert_module_has_nonzero_grads(first_layer.ffn_hc, label="DeepSeek-V4 FFN mHC")
        assert_module_has_nonzero_grads(
            first_layer.mlp.experts, label="DeepSeek-V4 hash-MoE experts"
        )
        return (
            "csa_grad=nonzero compressor_grad=nonzero "
            "mhc_grad=nonzero hash_moe_grad=nonzero"
        )

    return "model_grad=nonzero"


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


@pytest.mark.parametrize(
    ("model_name", "model_type", "write_config", "vocab_size", "lengths"),
    [
        pytest.param(
            "deepseek_v4",
            "deepseek_v4",
            _write_deepseek_v4_config,
            128,
            [16, 20, 24],
            id="deepseek-v4-release-gate",
        ),
        pytest.param(
            "glm5",
            "glm_moe_dsa",
            _write_glm52_config,
            32,
            [640, 64, 80],
            id="glm-5.2-release-gate",
        ),
        pytest.param(
            "qwen3_5",
            "qwen3_5_moe",
            _write_qwen35_config,
            128,
            [16, 20, 24],
            id="qwen-3.5-linear-full-release-gate",
        ),
        # Retain Kimi coverage, but release manifests should select only the
        # three explicit release-gate nodeids above.
        pytest.param(
            "kimi_k2",
            "deepseek_v3",
            _write_kimi_config,
            128,
            [5, 7, 9],
            id="kimi-k2-optional",
        ),
    ],
)
def test_mlite_engine_runtime_thd_cp_uses_typed_packed_batch(
    tmp_path, model_name, model_type, write_config, vocab_size, lengths
):
    """Validate the external-engine runtime+optimizer atom, not a Ray worker E2E."""

    torch = pytest.importorskip("torch")
    dist = pytest.importorskip("torch.distributed")
    TensorDict = pytest.importorskip("tensordict").TensorDict
    from verl_mlite.compat import apply_runtime_patches

    apply_runtime_patches()
    from megatron.lite.runtime.contracts import PackedBatch
    from verl_mlite.engine.config import MegatronLiteEngineConfig
    from verl_mlite.engine.mlite_engine import MegatronLiteEngine

    device = _init_dist_or_skip()
    world = dist.get_world_size()
    rank = dist.get_rank()
    case_seed = {
        "deepseek_v4": 20260401,
        "glm5": 20260502,
        "qwen3_5": 20260305,
        "kimi_k2": 20260201,
    }[model_name]
    # CP ranks must start from identical model parameters and the same global
    # packed batch. Rank-local RNG streams would create a meaningless mixture
    # of unrelated sequences after the model protocol performs the CP split.
    torch.manual_seed(case_seed)
    torch.cuda.manual_seed_all(case_seed)
    # torchrun ranks can share pytest's TMPDIR. Give each writer a private
    # config directory, then synchronize before any rank starts model build.
    hf_path = tmp_path / f"rank-{rank}" / f"tiny-{model_name}"
    write_config(hf_path)
    dist.barrier()

    engine = MegatronLiteEngine(
        model_config=SimpleNamespace(
            local_path=str(hf_path),
            hf_config={"model_type": model_type},
            mtp=None,
        ),
        engine_config=MegatronLiteEngineConfig(
            model_name=model_name,
            cp=world,
            impl_cfg={
                "use_thd": True,
                "deterministic": False,
                "mtp_enable": False,
            },
            use_fused_kernels=False,
        ),
        optimizer_config=_optimizer_config(),
        checkpoint_config={},
    )
    original_build_config = engine._build_mlite_config

    def _build_config_without_loading_weights(self):
        config = original_build_config()
        config.load_hf_weights = False
        return config

    engine._build_mlite_config = MethodType(
        _build_config_without_loading_weights, engine
    )
    engine.initialize()
    engine.optimizer_zero_grad()
    assert engine.handle._optimizer is not None
    assert engine.handle._lr_scheduler is not None
    built_contract = _assert_built_model_contract(engine, model_name)
    parameter_probe_name, parameter_probe_parameter = next(
        engine.module.named_parameters()
    )
    parameter_probe = parameter_probe_parameter.detach().contiguous()
    parameter_replicas = [torch.empty_like(parameter_probe) for _ in range(world)]
    dist.all_gather(parameter_replicas, parameter_probe)
    assert all(
        torch.equal(replica, parameter_replicas[0])
        for replica in parameter_replicas[1:]
    )

    torch.manual_seed(case_seed + 1)
    torch.cuda.manual_seed_all(case_seed + 1)
    input_ids = torch.nested.as_nested_tensor(
        [
            torch.randint(0, vocab_size, (length,), device=device, dtype=torch.long)
            for length in lengths
        ],
        layout=torch.jagged,
    )
    loss_mask = torch.nested.as_nested_tensor(
        [torch.ones(length, device=device, dtype=torch.float32) for length in lengths],
        layout=torch.jagged,
    )
    micro_batch = TensorDict(
        {"input_ids": input_ids, "loss_mask": loss_mask},
        batch_size=[len(lengths)],
        device=device,
    )

    runtime_batch = engine._make_runtime_batch(micro_batch)
    loss_context = engine._make_runtime_loss_context(micro_batch, loss_scale=1.0)
    assert isinstance(runtime_batch, PackedBatch)
    # Connector batch is model-agnostic: true seq lengths, no padding, no extras.
    assert runtime_batch.seq_lens.tolist() == lengths
    assert int(runtime_batch.input_ids.numel()) == sum(lengths)
    assert not runtime_batch.extras
    input_replicas = [torch.empty_like(runtime_batch.input_ids) for _ in range(world)]
    dist.all_gather(input_replicas, runtime_batch.input_ids)
    assert all(
        torch.equal(replica, input_replicas[0]) for replica in input_replicas[1:]
    )

    with engine.train_mode():
        result = engine.runtime.forward_backward(
            engine.handle,
            iter([(runtime_batch, loss_context)]),
            loss_fn=None,
            num_microbatches=1,
            forward_only=False,
        )

    assert torch.isfinite(result.model_output.loss)
    assert result.model_output.log_probs is not None
    local_grad_norm = torch.zeros((), dtype=torch.float32, device=device)
    for param in engine.module.parameters():
        gradient = _parameter_gradient(param)
        if gradient is not None:
            local_grad_norm = local_grad_norm + gradient.detach().float().norm()
    grad_norm_min = local_grad_norm.clone()
    grad_norm_sum = local_grad_norm.clone()
    dist.all_reduce(grad_norm_min, op=dist.ReduceOp.MIN)
    dist.all_reduce(grad_norm_sum, op=dist.ReduceOp.SUM)
    assert torch.isfinite(grad_norm_min)
    assert torch.isfinite(grad_norm_sum)
    assert grad_norm_min.item() > 0.0

    verl_output = engine._build_verl_model_output(
        raw_output={"log_probs": result.model_output.log_probs},
        runtime_batch=runtime_batch,
    )
    nested_log_probs = verl_output["log_probs"]
    assert [int(x) for x in nested_log_probs.offsets().diff().cpu()] == lengths
    grad_contract = _assert_model_specific_grad_contract(engine, model_name)

    optimizer_master_before = _optimizer_master_snapshot(engine)
    scheduler_steps_before = int(engine.handle._lr_scheduler.state_dict()["num_steps"])
    optimizer_step_receipt = {}
    runtime_optimizer_step = engine.runtime.optimizer_step

    def _recording_runtime_optimizer_step(self, handle):
        del self
        receipt = runtime_optimizer_step(handle)
        optimizer_step_receipt["value"] = receipt
        return receipt

    engine.runtime.optimizer_step = MethodType(
        _recording_runtime_optimizer_step, engine.runtime
    )
    public_optimizer_grad_norm = float(engine.optimizer_step())
    update_successful, runtime_grad_norm, _num_zeros = optimizer_step_receipt["value"]
    assert update_successful is True
    assert public_optimizer_grad_norm == pytest.approx(float(runtime_grad_norm))
    changed_master_params, master_max_abs_delta = _assert_optimizer_master_changed(
        engine, optimizer_master_before
    )
    optimizer_state_tensors = _assert_optimizer_state_initialized(engine)
    optimizer_grad_norm = torch.tensor(
        public_optimizer_grad_norm, dtype=torch.float32, device=device
    )
    optimizer_grad_norm_min = optimizer_grad_norm.clone()
    optimizer_grad_norm_sum = optimizer_grad_norm.clone()
    dist.all_reduce(optimizer_grad_norm_min, op=dist.ReduceOp.MIN)
    dist.all_reduce(optimizer_grad_norm_sum, op=dist.ReduceOp.SUM)
    assert torch.isfinite(optimizer_grad_norm_min)
    assert torch.isfinite(optimizer_grad_norm_sum)
    assert optimizer_grad_norm_min.item() > 0.0

    changed_master_count = torch.tensor(
        changed_master_params, dtype=torch.int64, device=device
    )
    changed_master_count_min = changed_master_count.clone()
    dist.all_reduce(changed_master_count_min, op=dist.ReduceOp.MIN)
    assert changed_master_count_min.item() > 0
    master_delta = torch.tensor(
        master_max_abs_delta, dtype=torch.float32, device=device
    )
    master_delta_min = master_delta.clone()
    master_delta_max = master_delta.clone()
    dist.all_reduce(master_delta_min, op=dist.ReduceOp.MIN)
    dist.all_reduce(master_delta_max, op=dist.ReduceOp.MAX)
    assert torch.isfinite(master_delta_min)
    assert torch.isfinite(master_delta_max)
    assert master_delta_min.item() > 0.0

    post_step_probe_name, post_step_probe_parameter = next(
        engine.module.named_parameters()
    )
    assert post_step_probe_name == parameter_probe_name
    post_step_probe = post_step_probe_parameter.detach().contiguous()
    post_step_replicas = [torch.empty_like(post_step_probe) for _ in range(world)]
    dist.all_gather(post_step_replicas, post_step_probe)
    assert all(
        torch.equal(replica, post_step_replicas[0])
        for replica in post_step_replicas[1:]
    )

    learning_rate = torch.tensor(
        float(engine.lr_scheduler_step()), dtype=torch.float32, device=device
    )
    learning_rate_min = learning_rate.clone()
    learning_rate_max = learning_rate.clone()
    dist.all_reduce(learning_rate_min, op=dist.ReduceOp.MIN)
    dist.all_reduce(learning_rate_max, op=dist.ReduceOp.MAX)
    assert torch.isfinite(learning_rate_min)
    assert torch.isfinite(learning_rate_max)
    assert learning_rate_min.item() > 0.0
    assert learning_rate_min.item() == pytest.approx(learning_rate_max.item())
    scheduler_steps_after = int(engine.handle._lr_scheduler.state_dict()["num_steps"])
    assert scheduler_steps_after == scheduler_steps_before + 1

    if rank == 0:
        print(
            "NON_SKIP_VERL_MLITE_RUNTIME_THD_CP_SMOKE_PASSED "
            "scope=external-engine-runtime-optimizer ray_worker_e2e=false "
            f"model={model_name} "
            f"{built_contract} "
            f"{grad_contract} "
            "cp_parameter_probe_replicated_pre_post=true cp_inputs=replicated "
            "optimizer_step=true optimizer_update_successful=true "
            "optimizer_master_changed=true optimizer_state_initialized=true "
            "lr_scheduler_step=true scheduler_step_delta=1 "
            f"world_size={world} lengths={lengths} "
            f"loss={float(result.model_output.loss.detach().item()):.6e} "
            f"grad_norm_min={float(grad_norm_min.item()):.6e} "
            f"grad_norm_sum={float(grad_norm_sum.item()):.6e} "
            f"optimizer_grad_norm_min={float(optimizer_grad_norm_min.item()):.6e} "
            f"optimizer_grad_norm_sum={float(optimizer_grad_norm_sum.item()):.6e} "
            f"changed_master_params_min={int(changed_master_count_min.item())} "
            f"master_delta_min={float(master_delta_min.item()):.6e} "
            f"master_delta_max={float(master_delta_max.item()):.6e} "
            f"optimizer_state_tensors={optimizer_state_tensors} "
            f"lr={float(learning_rate_min.item()):.6e}"
        )
