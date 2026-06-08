from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from megatron.lite.primitive.optimizers import get_optimizer_backend
from megatron.lite.primitive.optimizers.mfsdp import config as mfsdp_config
from megatron.lite.primitive.optimizers.mfsdp import grad_norm as mfsdp_grad_norm
from megatron.lite.primitive.optimizers.mfsdp import optimizer as mfsdp_optimizer
from megatron.lite.primitive.optimizers.mfsdp.patches import should_skip_tp_duplicate_sync
from megatron.lite.runtime.contracts.config import ParallelConfig


@dataclass
class _FakeDDPConfig:
    use_distributed_optimizer: bool = False
    use_megatron_fsdp: bool = False
    data_parallel_sharding_strategy: str = "no_shard"
    bucket_size: int | None = None
    overlap_grad_reduce: bool = False
    overlap_param_gather: bool = False
    num_distributed_optimizer_instances: int = 1
    nccl_ub: bool = False
    fsdp_double_buffer: bool = False
    megatron_fsdp_main_params_dtype: torch.dtype | None = None
    megatron_fsdp_main_grads_dtype: torch.dtype | None = None
    megatron_fsdp_grad_comm_dtype: torch.dtype | None = None


def _engine_cfg(**overrides):
    cfg = SimpleNamespace(
        model_name="qwen3_5",
        parallel=ParallelConfig(tp=1, ep=1, etp=1, pp=1, vpp=1, cp=1),
        optimizer=SimpleNamespace(optimizer="adam", override_optimizer_config={}),
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_mfsdp_backend_registry_resolves_backend():
    backend = get_optimizer_backend("mfsdp")

    assert backend.name == "megatron_fsdp"
    assert backend.runtime_backend == "megatron_fsdp"


def test_mfsdp_config_lowers_aliases_and_preserves_optimizer_keys():
    opt = SimpleNamespace(
        override_optimizer_config={
            "mfsdp_sharding_strategy": "optim_grads",
            "bucket_size": 1234,
            "adam_beta1": 0.8,
        }
    )

    opt_overrides, ddp_overrides = mfsdp_config.split_mfsdp_overrides(
        opt,
        _FakeDDPConfig,
    )

    assert opt_overrides == {"adam_beta1": 0.8}
    assert ddp_overrides["data_parallel_sharding_strategy"] == "optim_grads"
    assert ddp_overrides["bucket_size"] == 1234

    cfg = mfsdp_config.build_mfsdp_ddp_config(
        _FakeDDPConfig,
        {
            "megatron_fsdp_main_params_dtype": "bf16",
            "megatron_fsdp_grad_comm_dtype": "torch.float16",
        },
    )

    assert cfg.use_distributed_optimizer is True
    assert cfg.use_megatron_fsdp is True
    assert cfg.data_parallel_sharding_strategy == "optim_grads_params"
    assert cfg.megatron_fsdp_main_params_dtype is torch.bfloat16
    assert cfg.megatron_fsdp_main_grads_dtype is None
    assert cfg.megatron_fsdp_grad_comm_dtype is torch.float16


def test_mfsdp_config_rejects_unsupported_optimizer():
    engine_cfg = _engine_cfg(
        optimizer=SimpleNamespace(optimizer="adamw", override_optimizer_config={})
    )
    with pytest.raises(ValueError, match="adam/sgd"):
        mfsdp_config.validate_mfsdp_config(engine_cfg)


def test_mfsdp_grad_clip_scales_unique_grads_and_decoupled_grads():
    shared = torch.nn.Parameter(torch.ones(2))
    shared.grad = torch.ones(2)
    other = torch.nn.Parameter(torch.ones(2))
    other.grad = torch.ones(2)
    decoupled = torch.nn.Parameter(torch.ones(2))
    decoupled.grad = torch.ones(2)
    decoupled.decoupled_grad = torch.ones(2)

    class _Leaf:
        def __init__(self, params, *, use_decoupled_grad=False):
            self.is_stub_optimizer = False
            self.config = SimpleNamespace(
                clip_grad=1.0,
                use_precision_aware_optimizer_no_fp8_or_ds_fp8=use_decoupled_grad,
            )
            self.param_groups = [{"params": params}]

        def get_parameters(self):
            return []

    optimizer = SimpleNamespace(
        chained_optimizers=[
            _Leaf([shared, other]),
            _Leaf([shared]),
            _Leaf([decoupled], use_decoupled_grad=True),
        ],
    )

    mfsdp_grad_norm._clip_mfsdp_grads_by_total_norm(optimizer, grad_norm=4.0)

    assert torch.allclose(shared.grad, torch.full_like(shared.grad, 0.25))
    assert torch.allclose(other.grad, torch.full_like(other.grad, 0.25))
    assert torch.equal(decoupled.grad, torch.ones(2))
    assert torch.allclose(decoupled.decoupled_grad, torch.full_like(decoupled.grad, 0.25))


def test_mfsdp_metadata_infers_tp_partition_attrs_for_known_weights():
    class _Leaf(torch.nn.Module):
        def __init__(self, *, param_name: str = "weight"):
            super().__init__()
            param = torch.nn.Parameter(torch.randn(4, 3))
            param.tensor_model_parallel = True
            setattr(self, param_name, param)

    class _Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = torch.nn.Module()
            self.attn.qkv = torch.nn.Module()
            self.attn.qkv.linear = _Leaf()
            self.attn.proj = torch.nn.Module()
            self.attn.proj.linear = _Leaf()
            self.moe = torch.nn.Module()
            self.moe.experts = torch.nn.Module()
            self.moe.experts.fc1 = _Leaf(param_name="weight0")
            self.moe.experts.fc2 = _Leaf(param_name="weight0")

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([_Layer()])

    mfsdp_optimizer.ensure_mfsdp_tp_partition_attrs(model)

    layer = model.layers[0]
    assert layer.attn.qkv.linear.weight.partition_dim == 0
    assert layer.attn.proj.linear.weight.partition_dim == 1
    assert layer.moe.experts.fc1.weight0.partition_dim == 0
    assert layer.moe.experts.fc2.weight0.partition_dim == 1
    assert should_skip_tp_duplicate_sync(layer.attn.qkv.linear.weight)
