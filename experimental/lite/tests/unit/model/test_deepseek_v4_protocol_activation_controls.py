# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fail-closed DeepSeek-V4 activation recompute/offload protocol coverage."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.mlite


class _Unit(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Linear(2, 2)
        self.mlp = SimpleNamespace(experts=nn.Linear(2, 2), gate=nn.Linear(2, 2))
        self.input_layernorm = nn.LayerNorm(2)
        self.post_attention_layernorm = nn.LayerNorm(2)


class _BareChunk(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleDict({"0": _Unit(), "1": _Unit()})
        self.mtp = nn.ModuleList([_Unit()])


class _WrappedChunk(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _BareChunk()


class _EmptyPipelineChunk(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleDict()
        self.mtp = nn.ModuleList()
        self.layer_indices = []
        self.ps = SimpleNamespace(pp_size=2)


class _MTPUnit(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(3.0))
        self.calls = 0

    def forward(self, hidden_states, *, input_ids, position_ids):
        self.calls += 1
        del input_ids, position_ids
        return hidden_states * self.weight


def test_iter_transformer_units_supports_bare_and_legacy_wrapped_chunks() -> None:
    from megatron.lite.model.deepseek_v4.lite import protocol

    bare = _BareChunk()
    wrapped = _WrappedChunk()

    assert protocol._iter_transformer_units(bare) == [*bare.layers.values(), *bare.mtp]
    assert protocol._iter_transformer_units(wrapped) == [
        *wrapped.model.layers.values(),
        *wrapped.model.mtp,
    ]


def test_activation_controls_wire_each_nonempty_chunk(monkeypatch) -> None:
    from megatron.lite.model.deepseek_v4.lite import protocol

    bare = _BareChunk()
    wrapped = _WrappedChunk()
    recompute_calls = []

    monkeypatch.setattr(
        protocol,
        "apply_recompute",
        lambda units, names, module_map: recompute_calls.append(
            (list(units), list(names), module_map)
        ),
    )

    protocol._apply_activation_memory_controls(
        [bare, wrapped], recompute_spec=["core_attn"], offload_spec=[]
    )

    assert [call[0] for call in recompute_calls] == [
        protocol._iter_transformer_units(bare),
        protocol._iter_transformer_units(wrapped),
    ]
    assert [call[1] for call in recompute_calls] == [["core_attn"], ["core_attn"]]


def test_activation_controls_really_wrap_bare_model_modules() -> None:
    from megatron.lite.model.deepseek_v4.lite import protocol

    recomputed = _BareChunk()
    protocol._apply_activation_memory_controls(
        [recomputed], recompute_spec=["core_attn"], offload_spec=[]
    )
    assert {
        unit.self_attn.forward.__name__
        for unit in protocol._iter_transformer_units(recomputed)
    } == {"_checkpointed_forward"}


@pytest.mark.parametrize(
    ("recompute_spec", "offload_spec", "match"),
    [
        (["unknown"], [], "Unsupported.*recompute"),
        ([], ["unknown"], "Unsupported.*offload"),
        (["moe", "moe"], [], "recompute.*duplicate"),
        (["full", "moe"], [], "'full' must be used alone"),
        (["attn", "core_attn"], [], "target the same module"),
        (["moe", "experts"], [], "parent and child"),
    ],
)
def test_activation_controls_reject_ignored_or_duplicate_selectors(
    recompute_spec, offload_spec, match
) -> None:
    from megatron.lite.model.deepseek_v4.lite import protocol

    with pytest.raises(ValueError, match=match):
        protocol._apply_activation_memory_controls(
            [_BareChunk()], recompute_spec=recompute_spec, offload_spec=offload_spec
        )


def test_activation_controls_fail_closed_for_malformed_or_invalid_empty_chunks(
    monkeypatch,
) -> None:
    from megatron.lite.model.deepseek_v4.lite import protocol

    recompute_calls = []
    monkeypatch.setattr(
        protocol,
        "apply_recompute",
        lambda *args, **kwargs: recompute_calls.append((args, kwargs)),
    )
    with pytest.raises(TypeError, match="expose a layers container"):
        protocol._apply_activation_memory_controls(
            [nn.Module()], recompute_spec=["full"], offload_spec=[]
        )
    with pytest.raises(TypeError, match="expose a layers container"):
        protocol._apply_activation_memory_controls(
            [_BareChunk(), nn.Module()], recompute_spec=["full"], offload_spec=[]
        )
    assert recompute_calls == []
    with pytest.raises(RuntimeError, match="non-pipeline-empty"):
        protocol._apply_activation_memory_controls(
            [SimpleNamespace(layers=nn.ModuleDict(), mtp=nn.ModuleList())],
            recompute_spec=["full"],
            offload_spec=[],
        )
    with pytest.raises(RuntimeError, match="no model chunks"):
        protocol._apply_activation_memory_controls(
            [], recompute_spec=["full"], offload_spec=[]
        )


def test_activation_controls_allow_explicit_empty_pp_stage(monkeypatch) -> None:
    from megatron.lite.model.deepseek_v4.lite import protocol

    monkeypatch.setattr(
        protocol,
        "apply_recompute",
        lambda *_args, **_kwargs: pytest.fail("empty PP stage must not be wrapped"),
    )
    protocol._apply_activation_memory_controls(
        [_EmptyPipelineChunk()], recompute_spec=["full"], offload_spec=[]
    )


@pytest.mark.parametrize("offload_spec", [["core_attn"], ["full"]])
def test_activation_offload_fails_explicitly_before_fake_primitive(
    offload_spec,
) -> None:
    from megatron.lite.model.deepseek_v4.lite import protocol

    with pytest.raises(NotImplementedError, match="performs activation recompute"):
        protocol._apply_activation_memory_controls(
            [_BareChunk()], recompute_spec=[], offload_spec=offload_spec
        )


def test_mtp_full_recompute_keeps_hidden_gradient_positional(
    monkeypatch, transformer_engine_import_stub
) -> None:
    from megatron.lite.model.deepseek_v4.lite import protocol

    transformer_engine_import_stub()
    from megatron.lite.model.deepseek_v4.lite.model import DeepseekV4MTPLayer

    hidden_parameter = inspect.signature(DeepseekV4MTPLayer.forward).parameters[
        "hidden_states"
    ]
    assert hidden_parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD

    monkeypatch.setattr(
        torch.cuda, "get_rng_state", lambda: torch.zeros(1, dtype=torch.uint8)
    )
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda _state: None)
    mtp = _MTPUnit()
    chunk = SimpleNamespace(
        layers=nn.ModuleDict(),
        mtp=nn.ModuleList([mtp]),
        layer_indices=[],
        ps=SimpleNamespace(pp_size=1),
    )
    protocol._apply_activation_memory_controls(
        [chunk], recompute_spec=["full"], offload_spec=[]
    )
    hidden_states = torch.tensor([2.0, -4.0], requires_grad=True)
    mtp(
        hidden_states,
        input_ids=torch.ones(2, dtype=torch.long),
        position_ids=torch.arange(2),
    ).sum().backward()

    assert mtp.calls == 2
    torch.testing.assert_close(hidden_states.grad, torch.full_like(hidden_states, 3.0))
    torch.testing.assert_close(mtp.weight.grad, torch.tensor(-2.0))


def test_build_model_rejects_invalid_policy_before_parallel_or_cuda(
    monkeypatch,
) -> None:
    from megatron.lite.model.deepseek_v4.lite import protocol

    monkeypatch.setattr(
        protocol,
        "init_parallel",
        lambda *_args, **_kwargs: pytest.fail("parallel init must not run"),
    )
    with pytest.raises(NotImplementedError, match="performs activation recompute"):
        protocol.build_model(
            object(),
            impl_cfg=protocol.ImplConfig(optimizer=None, offload=["core_attn"]),
        )


def test_parallel_scope_rejects_vpp_before_parallel_init() -> None:
    from megatron.lite.model.deepseek_v4.lite import protocol
    from megatron.lite.runtime.contracts import ParallelConfig

    with pytest.raises(NotImplementedError, match="virtual pipeline"):
        protocol._validate_parallel_scope(ParallelConfig(vpp=2))


@pytest.mark.smoke
@pytest.mark.gpu
@pytest.mark.distributed
def test_ds4_mtp_full_recompute_matches_real_cuda_backward(tmp_path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the DS4 full-recompute smoke.")
    pytest.importorskip("cudnn", reason="DS4 model import needs the cuDNN DSA stack.")

    import torch.distributed as dist
    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
    from megatron.lite.model.deepseek_v4.lite import protocol
    from megatron.lite.runtime.contracts import ParallelConfig

    initialized_here = False
    if not dist.is_initialized():
        dist.init_process_group(
            "nccl", init_method=f"file://{tmp_path / 'nccl-init'}", rank=0, world_size=1
        )
        initialized_here = True

    def make_config() -> DeepseekV4Config:
        return DeepseekV4Config(
            vocab_size=64,
            hidden_size=128,
            moe_intermediate_size=16,
            num_hidden_layers=1,
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
            sliding_window=16,
            num_hash_layers=2,
            hc_mult=2,
            index_head_dim=64,
            index_n_heads=8,
            index_topk=16,
            num_nextn_predict_layers=1,
            rms_norm_eps=1.0e-6,
        )

    def build(recompute: list[str]):
        torch.manual_seed(20260628)
        torch.cuda.manual_seed_all(20260628)
        bundle = protocol.build_model(
            make_config(),
            impl_cfg=protocol.ImplConfig(
                parallel=ParallelConfig(),
                optimizer=None,
                recompute=recompute,
                attention_backend_override="local",
                mtp_enable=True,
                mtp_enable_train=True,
            ),
        )
        model = bundle.chunks[0]
        assert len(model.layers) == 1
        assert len(model.mtp) == 1
        assert model.layers["0"].self_attn.self_attn.attention_backend == "local"
        assert model.mtp[0].self_attn.self_attn.attention_backend == "local"
        return model

    try:
        reference = build([])
        recomputed = build(["full"])
        recomputed.load_state_dict(reference.state_dict(), strict=True)
        assert recomputed.layers["0"].forward.__name__ == "_checkpointed_forward"
        assert recomputed.mtp[0].forward.__name__ == "_checkpointed_forward"

        torch.manual_seed(20260629)
        input_ids = torch.randint(0, 64, (1, 64), device="cuda")
        labels = torch.randint(0, 64, (1, 64), device="cuda")
        loss_mask = torch.ones_like(labels, dtype=torch.float32)
        position_ids = torch.arange(64, device="cuda").unsqueeze(0)

        def run(model):
            torch.manual_seed(20260630)
            torch.cuda.manual_seed_all(20260630)
            model.zero_grad(set_to_none=True)
            output = model(
                input_ids=input_ids,
                labels=labels,
                loss_mask=loss_mask,
                position_ids=position_ids,
                enable_mtp=True,
            )
            loss = output["loss"]
            assert torch.isfinite(loss)
            assert torch.isfinite(output["mtp_loss"])
            loss.backward()
            gradients = {
                name: parameter.grad.detach().clone()
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            }
            assert gradients
            assert any(name.startswith("mtp.0.") for name in gradients)
            assert all(
                torch.isfinite(gradient).all() for gradient in gradients.values()
            )
            return loss.detach(), output["mtp_loss"].detach(), gradients

        reference_loss, reference_mtp_loss, reference_gradients = run(reference)
        recomputed_loss, recomputed_mtp_loss, recomputed_gradients = run(recomputed)
        torch.testing.assert_close(
            recomputed_loss, reference_loss, rtol=1.0e-5, atol=1.0e-6
        )
        torch.testing.assert_close(
            recomputed_mtp_loss, reference_mtp_loss, rtol=1.0e-5, atol=1.0e-6
        )
        assert recomputed_gradients.keys() == reference_gradients.keys()
        for name in reference_gradients:
            torch.testing.assert_close(
                recomputed_gradients[name],
                reference_gradients[name],
                rtol=1.0e-2,
                atol=1.0e-5,
                msg=lambda message, name=name: f"{name}: {message}",
            )
        assert any(
            gradient.abs().max().item() > 0
            for name, gradient in recomputed_gradients.items()
            if name.startswith("mtp.0.")
        )
        print(
            "DS4_MTP_FULL_RECOMPUTE_CUDA_PASS "
            f"loss={float(recomputed_loss):.8f} gradients={len(recomputed_gradients)}",
            flush=True,
        )
    finally:
        if initialized_here:
            dist.destroy_process_group()
