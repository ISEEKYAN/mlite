# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.mlite


@pytest.fixture(autouse=True)
def _te_import_stub(transformer_engine_import_stub):
    transformer_engine_import_stub()


def _split_grouped_qkvg():
    from megatron.lite.primitive.modules import split_grouped_qkvg

    return split_grouped_qkvg


def _moe_aux_scaler():
    from megatron.lite.primitive.modules.moe import MoEAuxLossAutoScaler

    return MoEAuxLossAutoScaler


def _router_and_parallel_state(monkeypatch):
    from megatron.lite.primitive.modules.router import TopKRouter
    from megatron.lite.primitive.parallel import ParallelState

    del monkeypatch
    return TopKRouter, ParallelState


def _router_config():
    return SimpleNamespace(
        hidden_size=4, num_experts=4, num_experts_per_tok=2, router_aux_loss_coef=0.1
    )


def _walk_grad_fn_names(tensor: torch.Tensor) -> set[str]:
    names: set[str] = set()
    stack = [tensor.grad_fn]
    while stack:
        fn = stack.pop()
        if fn is None:
            continue
        names.add(type(fn).__name__)
        stack.extend(parent for parent, _idx in fn.next_functions)
    return names


def test_attention_public_api_is_narrow():
    from megatron.lite.primitive.modules import attention

    assert attention.__all__ == [
        "DSAIndexShareState",
        "DynamicSparseAttention",
        "MultiLatentAttention",
        "RMSNorm",
        "build_rope_cache",
        "build_rotary_embeddings",
    ]
    assert attention.DynamicSparseAttention is attention.dsa.DynamicSparseAttention
    assert attention.DSAIndexShareState is attention.dsa.DSAIndexShareState
    for internal_name in (
        "dsa_indexer_type_for_layer",
        "is_dsa_skip_topk_layer",
        "source_dsa_compute_layer",
        "validate_dsa_index_share_pipeline_split",
    ):
        assert not hasattr(attention, internal_name)
    for internal_name in (
        "dsa_indexer_type_for_layer",
        "is_dsa_skip_topk_layer",
        "source_dsa_compute_layer",
    ):
        assert internal_name not in attention.dsa.__all__
    assert "validate_dsa_index_share_pipeline_split" in attention.dsa.__all__


def test_gqa_split_grouped_qkvg_preserves_q_gate_kv_order():
    split_grouped_qkvg = _split_grouped_qkvg()
    qkv = torch.arange(24).reshape(1, 24)

    query, gate, key, value = split_grouped_qkvg(
        qkv, num_heads=4, num_kv_heads=2, head_dim=2
    )

    assert query.shape == (1, 4, 2)
    assert gate.shape == (1, 4, 2)
    assert key.shape == (1, 2, 2)
    assert value.shape == (1, 2, 2)
    assert torch.equal(query, torch.tensor([[[0, 1], [2, 3], [12, 13], [14, 15]]]))
    assert torch.equal(gate, torch.tensor([[[4, 5], [6, 7], [16, 17], [18, 19]]]))
    assert torch.equal(key, torch.tensor([[[8, 9], [20, 21]]]))
    assert torch.equal(value, torch.tensor([[[10, 11], [22, 23]]]))


def test_moe_aux_loss_auto_scaler_threads_scaled_aux_gradient():
    MoEAuxLossAutoScaler = _moe_aux_scaler()
    MoEAuxLossAutoScaler.set_loss_scale(torch.tensor([0.25]))
    output = torch.randn(3, requires_grad=True)
    aux_loss = torch.tensor(2.0, requires_grad=True)

    scaled_output = MoEAuxLossAutoScaler.apply(output * 2.0, aux_loss)
    scaled_output.sum().backward()

    torch.testing.assert_close(output.grad, torch.full_like(output, 2.0))
    torch.testing.assert_close(aux_loss.grad, torch.tensor(0.25))
    MoEAuxLossAutoScaler.main_loss_backward_scale = None


def test_topk_router_returns_finite_scores_and_valid_expert_indices(monkeypatch):
    TopKRouter, ParallelState = _router_and_parallel_state(monkeypatch)
    config = _router_config()
    router = TopKRouter(config, ParallelState(), compute_aux_loss=False)
    hidden = torch.randn(5, 4)

    scores, indices = router(hidden)

    assert scores.shape == (5, 2)
    assert indices.shape == (5, 2)
    assert scores.dtype == hidden.dtype
    assert torch.isfinite(scores).all()
    assert indices.min().item() >= 0
    assert indices.max().item() < config.num_experts


def test_topk_router_scores_are_normalized_and_deterministic_in_eval(monkeypatch):
    TopKRouter, ParallelState = _router_and_parallel_state(monkeypatch)
    config = _router_config()
    router = TopKRouter(config, ParallelState(), compute_aux_loss=False)
    hidden = torch.randn(5, config.hidden_size)

    router.eval()
    scores_1, indices_1 = router(hidden)
    scores_2, indices_2 = router(hidden)

    torch.testing.assert_close(scores_1.sum(dim=-1), torch.ones(hidden.size(0)))
    torch.testing.assert_close(scores_1, scores_2, atol=0, rtol=0)
    assert torch.equal(indices_1, indices_2)


def test_topk_router_does_not_attach_aux_scaler_in_eval(monkeypatch):
    TopKRouter, ParallelState = _router_and_parallel_state(monkeypatch)
    config = _router_config()
    router = TopKRouter(config, ParallelState(), compute_aux_loss=True)
    hidden = torch.randn(5, config.hidden_size)

    router.eval()
    scores, _indices = router(hidden)

    assert not any("MoEAuxLoss" in name for name in _walk_grad_fn_names(scores))


def test_dsv4_indexer_is_a_pure_topk_mask_not_an_attention_bias():
    from megatron.lite.primitive.modules.attention.csa import (
        _mask_compressed_scores_with_indexer,
    )

    compressed_scores = torch.tensor(
        [
            [
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
                [[9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]],
            ]
        ]
    )
    index_scores = torch.tensor([[[100.0, -4.0, 3.0, 2.0], [1.0, 8.0, 7.0, 0.0]]])
    actual = _mask_compressed_scores_with_indexer(
        compressed_scores, index_scores, index_topk=2
    )

    selected = torch.zeros_like(index_scores, dtype=torch.bool)
    selected.scatter_(-1, index_scores.topk(2, dim=-1).indices, True)
    selected = selected.unsqueeze(1).expand_as(compressed_scores)
    torch.testing.assert_close(actual[selected], compressed_scores[selected])
    assert torch.isneginf(actual[~selected]).all()
    with pytest.raises(ValueError, match="must be positive"):
        _mask_compressed_scores_with_indexer(
            compressed_scores, index_scores, index_topk=0
        )


def test_dsv4_zero_loss_fused_selector_does_not_create_optimizer_grads():
    from megatron.lite.primitive.modules.attention.csa import (
        _prepare_indexer_inputs_for_fused_loss,
    )

    inputs = tuple(torch.nn.Parameter(torch.randn(2, 3)) for _ in range(3))
    detached = _prepare_indexer_inputs_for_fused_loss(*inputs, loss_coeff=0.0)
    assert all(
        not tensor.requires_grad and tensor.grad_fn is None for tensor in detached
    )
    assert all(
        actual.data_ptr() == source.data_ptr()
        for actual, source in zip(detached, inputs)
    )

    before_step = tuple(parameter.detach().clone() for parameter in inputs)
    optimizer = torch.optim.AdamW(inputs, lr=0.1, weight_decay=0.5)
    optimizer.step()
    for actual, expected in zip(inputs, before_step):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    differentiable = _prepare_indexer_inputs_for_fused_loss(*inputs, loss_coeff=1.0e-2)
    assert all(actual is source for actual, source in zip(differentiable, inputs))


def test_dsv4_torch_indexer_rotates_query_and_compressed_key(monkeypatch):
    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
    from megatron.lite.primitive.modules.attention import csa
    from megatron.lite.primitive.parallel import ParallelState

    monkeypatch.setattr(csa.te, "RMSNorm", torch.nn.RMSNorm)
    rotated_shapes = []

    def record_rotation(tensor):
        rotated_shapes.append(tuple(tensor.shape))
        return tensor

    monkeypatch.setattr(csa, "rotate_activation", record_rotation)
    cfg = DeepseekV4Config(
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        qk_rope_head_dim=2,
        q_lora_rank=4,
        o_lora_rank=4,
        o_groups=1,
        compress_ratios=[4],
        sliding_window=4,
        index_head_dim=4,
        index_n_heads=2,
        index_topk=1,
    )
    module = csa.CompressedSparseAttention(cfg, layer_idx=0, ps=ParallelState())
    hidden = torch.randn(1, 8, cfg.hidden_size)
    position_ids = torch.arange(8).unsqueeze(0)

    output = module(hidden, position_ids=position_ids)

    assert output.shape == hidden.shape
    assert rotated_shapes == [
        (1, 1, 2, cfg.index_head_dim),
        (1, cfg.index_n_heads, 8, cfg.index_head_dim),
    ]


def test_topk_router_aux_loss_contributes_gate_gradient(monkeypatch):
    TopKRouter, ParallelState = _router_and_parallel_state(monkeypatch)
    config = _router_config()
    router = TopKRouter(config, ParallelState(), compute_aux_loss=True)
    hidden = torch.randn(8, config.hidden_size)

    router.train()
    scores, _indices = router(hidden)
    scores.sum().backward()
    grad_with_aux = router.gate.weight.grad.detach().clone()

    router.zero_grad()
    saved_coeff = router.aux_loss_coeff
    router.aux_loss_coeff = 0.0
    scores_no_aux, _indices = router(hidden)
    scores_no_aux.sum().backward()
    grad_no_aux = router.gate.weight.grad.detach().clone()
    router.aux_loss_coeff = saved_coeff

    assert torch.isfinite(grad_with_aux).all()
    assert torch.isfinite(grad_no_aux).all()
    assert (grad_with_aux - grad_no_aux).abs().sum().item() > 0.0


def test_dsa_index_share_schedule_and_state():
    from megatron.lite.primitive.modules.attention.dsa import (
        DSAIndexShareState,
        dsa_indexer_type_for_layer,
        is_dsa_skip_topk_layer,
        source_dsa_compute_layer,
    )

    assert is_dsa_skip_topk_layer(3, skip_topk_offset=3, topk_freq=4) is False
    assert is_dsa_skip_topk_layer(4, skip_topk_offset=3, topk_freq=4) is True
    assert dsa_indexer_type_for_layer(7, skip_topk_offset=3, topk_freq=4) == "full"
    assert source_dsa_compute_layer(6, skip_topk_offset=3, topk_freq=4) == 3

    state = DSAIndexShareState({3: 1})
    topk = torch.tensor([[[0, 1], [1, 2]]], dtype=torch.int32)
    state.save_topk(3, topk, sequence_key=0)
    assert state.cached_tensor_count == 1
    assert torch.equal(state.get_topk(6, 3, sequence_key=0), topk)
    assert state.cached_tensor_count == 0
    with pytest.raises(AssertionError, match="source layer 3"):
        state.get_topk(5, 3, sequence_key=1)


def test_glm5_dsa_wrapper_forwards_explicit_position_ids():
    from megatron.lite.model.glm5.lite.model import Glm5DSAAttention

    class CaptureDSA(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.position_ids = None

        def forward(self, x, **kwargs):
            self.position_ids = kwargs["position_ids"]
            return x

    wrapper = Glm5DSAAttention.__new__(Glm5DSAAttention)
    torch.nn.Module.__init__(wrapper)
    wrapper.ps = SimpleNamespace(cp_size=1, cp_rank=0)
    wrapper.qk_rope_head_dim = 4
    wrapper.rope_theta = 1_000_000.0
    wrapper.self_attention = CaptureDSA()
    hidden = torch.randn(5, 2, 8)
    position_ids = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 0, 1, 2]])

    output = wrapper(hidden, position_ids=position_ids)

    assert output.shape == hidden.shape
    assert torch.equal(wrapper.self_attention.position_ids, position_ids)
    with pytest.raises(ValueError, match="batch dimension"):
        wrapper(hidden, position_ids=torch.zeros(3, 5, dtype=torch.long))


def test_dsa_index_share_state_releases_final_packed_consumer():
    from megatron.lite.primitive.modules.attention.dsa import DSAIndexShareState

    state = DSAIndexShareState({3: 2})
    segment_0 = torch.tensor([[[0, 1]]], dtype=torch.int32)
    segment_1 = torch.tensor([[[1, 0]]], dtype=torch.int32)
    state.save_topk(3, segment_0, sequence_key=0)
    state.save_topk(3, segment_1, sequence_key=1)
    assert state.cached_tensor_count == 2

    assert torch.equal(state.get_topk(4, 3, sequence_key=0), segment_0)
    assert torch.equal(state.get_topk(4, 3, sequence_key=1), segment_1)
    assert state.cached_tensor_count == 2
    assert torch.equal(state.get_topk(5, 3, sequence_key=0), segment_0)
    assert state.cached_tensor_count == 1
    assert torch.equal(state.get_topk(5, 3, sequence_key=1), segment_1)
    assert state.cached_tensor_count == 0


def test_dsa_index_share_state_rejects_unconsumed_source_save():
    from megatron.lite.primitive.modules.attention.dsa import DSAIndexShareState

    state = DSAIndexShareState()
    assert state.needs_topk(3) is False
    with pytest.raises(AssertionError, match="no shared consumer"):
        state.save_topk(3, torch.zeros(1, 1, 1, dtype=torch.int32))


def test_glm5_counts_local_index_share_consumers_and_rejects_activation_replay():
    from megatron.lite.model.glm5.lite.model import (
        _local_dsa_index_share_consumer_counts,
        _validate_dsa_index_share_activation_replay,
    )

    def layer(*, shared: bool, source_layer: int):
        dsa = SimpleNamespace(skip_topk=shared, index_share_source_layer=source_layer)
        return SimpleNamespace(self_attention=SimpleNamespace(self_attention=dsa))

    trunk_layers = [
        layer(shared=False, source_layer=3),
        layer(shared=True, source_layer=3),
        layer(shared=True, source_layer=3),
    ]
    mtp = SimpleNamespace(
        layers=[SimpleNamespace(transformer_layer=layer(shared=False, source_layer=7))],
        repeated_layer=True,
        num_layers=3,
    )

    consumer_counts = _local_dsa_index_share_consumer_counts(trunk_layers, mtp)
    assert consumer_counts == {3: 2}
    _validate_dsa_index_share_activation_replay(
        True,
        recompute_modules=["moe", "attn_proj"],
        offload_modules=["mlp", "attn_proj"],
    )
    _validate_dsa_index_share_activation_replay(
        False,
        recompute_modules=["full", "core_attn", "self_attn", "dsa"],
        offload_modules=["full", "core_attn", "self_attn", "dsa"],
    )
    for replay_kind in ("recompute", "offload"):
        for unsafe_mode in ("full", "core_attn", "self_attn", "dsa"):
            kwargs = {"recompute_modules": [], "offload_modules": []}
            kwargs[f"{replay_kind}_modules"] = [unsafe_mode]
            with pytest.raises(ValueError, match="group-aware"):
                _validate_dsa_index_share_activation_replay(True, **kwargs)


def test_dsv4_fused_dsa_legacy_two_output_api_cpu_mock(monkeypatch):
    """DSv4's public fused wrapper and direct Function keep two outputs."""
    from megatron.lite.primitive.kernels import dsa_kernels

    def fake_forward(ctx, *args):
        query = args[0]
        ctx.input_count = len(args)
        output = query * 2.0
        loss = query.float().sum() * 0.0
        topk = torch.zeros(query.shape[:2] + (1,), dtype=torch.int32)
        return output, loss, topk

    def fake_backward(ctx, grad_output, grad_loss, grad_topk=None):
        del grad_loss, grad_topk
        grads = [None] * ctx.input_count
        grads[0] = grad_output * 2.0
        return tuple(grads)

    with_topk_func = dsa_kernels._FusedIndexerSparseAttnWithTopKFunc
    monkeypatch.setattr(with_topk_func, "forward", staticmethod(fake_forward))
    monkeypatch.setattr(with_topk_func, "backward", staticmethod(fake_backward))

    query = torch.ones(2, 1, 1, 2, requires_grad=True)
    args = (
        query,
        torch.ones(2, 1, 2),
        torch.zeros(1),
        torch.empty(1, 2, 0, dtype=torch.int32),
        torch.ones(2, 1, 1, 2),
        torch.ones(2, 1, 2),
        torch.ones(2, 1, 1),
        1,
        1,
        0.5,
        1.0,
        0.0,
        False,
        0,
        False,
        2,
    )

    direct_result = dsa_kernels.FusedIndexerSparseAttnFunc.apply(*args)
    assert isinstance(direct_result, tuple)
    assert len(direct_result) == 2

    output, indexer_loss = dsa_kernels.fused_indexer_sparse_attn(*args)
    assert output.shape == query.shape
    assert indexer_loss.ndim == 0
    (output.sum() + indexer_loss).backward()
    torch.testing.assert_close(query.grad, torch.full_like(query, 2.0))

    with_topk_result = dsa_kernels.fused_indexer_sparse_attn_with_topk(*args)
    assert len(with_topk_result) == 3
    assert with_topk_result[2].dtype == torch.int32


def test_glm32_indexer_backward_padding_is_zero_and_reversible():
    from megatron.lite.primitive.kernels import dsa_kernels

    query = torch.arange(2 * 3 * 32 * 4, dtype=torch.float32).view(2, 3, 32, 4)
    weights = torch.arange(2 * 3 * 32, dtype=torch.float32).view(2, 3, 32)
    padded_query, padded_weights, original_heads = (
        dsa_kernels._pad_indexer_heads_for_backward(query, weights)
    )

    assert original_heads == 32
    assert padded_query.shape == (2, 3, 64, 4)
    assert padded_weights.shape == (2, 3, 64)
    torch.testing.assert_close(padded_query[:, :, :32], query)
    torch.testing.assert_close(padded_weights[:, :, :32], weights)
    assert torch.count_nonzero(padded_query[:, :, 32:]).item() == 0
    assert torch.count_nonzero(padded_weights[:, :, 32:]).item() == 0

    unchanged_query, unchanged_weights, original_heads = (
        dsa_kernels._pad_indexer_heads_for_backward(padded_query, padded_weights)
    )
    assert original_heads == 64
    assert unchanged_query is padded_query
    assert unchanged_weights is padded_weights


def test_glm5_cp_per_token_indexer_loss_fails_until_global_divisor_is_wired(
    monkeypatch,
):
    from megatron.lite.primitive.modules.attention import dsa

    monkeypatch.setattr(dsa, "RMSNorm", torch.nn.LayerNorm)
    with pytest.raises(NotImplementedError, match="global-token divisor"):
        dsa.DynamicSparseAttention(
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
            rms_norm_eps=1.0e-5,
            indexer_loss_coeff=1.0e-2,
            calculate_per_token_loss=True,
            cp_size=2,
            cp_rank=0,
        )


@pytest.mark.parametrize(
    ("calculate_per_token_loss", "expected_weights"),
    [
        (False, [0.2, 0.3, 0.5]),
        (True, [1.0, 1.0, 1.0]),
    ],
    ids=["mean-kl-token-weighted", "per-token-kl-sum"],
)
def test_glm5_packed_dsa_weights_segment_indexer_losses(
    calculate_per_token_loss, expected_weights
):
    from types import MethodType

    from megatron.lite.primitive.modules.attention import dsa

    attention = dsa.DynamicSparseAttention.__new__(dsa.DynamicSparseAttention)
    torch.nn.Module.__init__(attention)
    attention.calculate_per_token_loss = calculate_per_token_loss
    attention.skip_topk = False
    attention.indexer_loss_coeff = 0.0

    calls = []

    def fake_forward_dense_full(
        self,
        x,
        cos,
        sin,
        position_ids,
        *,
        index_share_state=None,
        index_share_cache_key=None,
        indexer_loss_weight=1.0,
    ):
        del self, cos, sin, position_ids, index_share_state
        calls.append((x.shape[1], index_share_cache_key, indexer_loss_weight))
        return x

    attention._forward_dense_full = MethodType(fake_forward_dense_full, attention)
    x = torch.arange(40, dtype=torch.float32).view(1, 10, 4)
    positions = torch.tensor([[0, 1, 0, 1, 2, 0, 1, 2, 3, 4]])
    cos = torch.zeros(1, 10, 2)
    sin = torch.zeros_like(cos)
    packed_seq_params = SimpleNamespace(
        cu_seqlens_q_padded=torch.tensor([0, 2, 5, 10], dtype=torch.int32)
    )

    actual = attention._forward_packed_full(
        x,
        cos,
        sin,
        positions,
        packed_seq_params,
        index_share_state=None,
    )

    torch.testing.assert_close(actual, x)
    assert [length for length, _key, _weight in calls] == [2, 3, 5]
    assert [key for _length, key, _weight in calls] == [0, 1, 2]
    assert [weight for _length, _key, weight in calls] == pytest.approx(
        expected_weights
    )
    segment_losses = [1.0, 3.0, 5.0]
    aggregated = sum(
        loss * weight
        for loss, (_length, _key, weight) in zip(segment_losses, calls, strict=True)
    )
    expected_aggregate = (
        sum(segment_losses)
        if calculate_per_token_loss
        else (1.0 * 2 + 3.0 * 3 + 5.0 * 5) / 10
    )
    assert aggregated == pytest.approx(expected_aggregate)


def test_glm5_packed_dsa_rejects_padded_indexer_aux_until_query_mask_exists():
    from types import MethodType

    from megatron.lite.primitive.modules.attention import dsa

    attention = dsa.DynamicSparseAttention.__new__(dsa.DynamicSparseAttention)
    torch.nn.Module.__init__(attention)
    attention.calculate_per_token_loss = False
    attention.skip_topk = False
    attention.indexer_loss_coeff = 1.0e-2

    def must_not_run(*args, **kwargs):
        del args, kwargs
        raise AssertionError("padded indexer auxiliary path must fail before DSA")

    attention._forward_dense_full = MethodType(must_not_run, attention)
    x = torch.zeros(1, 8, 4)
    positions = torch.tensor([[0, 1, 0, 0, 0, 1, 2, 0]])
    cos = torch.zeros(1, 8, 2)
    sin = torch.zeros_like(cos)
    packed_seq_params = SimpleNamespace(
        cu_seqlens_q=torch.tensor([0, 2, 5], dtype=torch.int32),
        cu_seqlens_q_padded=torch.tensor([0, 4, 8], dtype=torch.int32),
    )

    with pytest.raises(NotImplementedError, match="query-valid mask"):
        attention._forward_packed_full(
            x,
            cos,
            sin,
            positions,
            packed_seq_params,
            index_share_state=None,
        )


def test_dense_dsa_full_causal_lse_is_chunked_and_topk_independent():
    from megatron.lite.primitive.kernels import dsa_kernels

    torch.manual_seed(123)
    batch, seq_q, seq_k, q_heads, kv_heads, head_dim = 2, 10, 5, 4, 2, 3
    ratio = 2
    scale = 0.37
    q = torch.randn(batch, seq_q, q_heads, head_dim, dtype=torch.bfloat16)
    k = torch.randn(batch, seq_k, kv_heads, head_dim, dtype=torch.bfloat16)

    # Limit the temporary to two query rows so this exercises the bounded
    # block loop rather than accidentally validating one monolithic einsum.
    max_score_bytes = 2 * batch * q_heads * seq_k * 4
    actual = dsa_kernels._compute_full_causal_attn_lse(
        q,
        k,
        scale,
        ratio,
        max_score_bytes=max_score_bytes,
    )
    key_chunked = dsa_kernels._compute_full_causal_attn_lse(
        q,
        k,
        scale,
        ratio,
        max_score_bytes=2 * batch * q_heads * 4,
    )

    expanded_k = k.float().repeat_interleave(q_heads // kv_heads, dim=2)
    scores = torch.einsum("bqhd,bkhd->bqhk", q.float(), expanded_k) * scale
    q_global_start = seq_k * ratio - seq_q
    q_pos = torch.arange(seq_q)
    k_pos = torch.arange(seq_k)
    valid_kv = torch.div(
        q_global_start + q_pos + 1,
        ratio,
        rounding_mode="floor",
    ).clamp(min=0, max=seq_k)
    causal = k_pos.view(1, seq_k) < valid_kv.view(seq_q, 1)
    expected = torch.logsumexp(
        scores.masked_fill(~causal.view(1, seq_q, 1, seq_k), -torch.inf),
        dim=-1,
    )
    expected = torch.where(
        valid_kv.view(1, seq_q, 1) > 0,
        expected,
        torch.full_like(expected, torch.inf),
    )

    torch.testing.assert_close(actual, expected, atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(key_chunked, expected, atol=1.0e-6, rtol=1.0e-6)
    assert actual.shape == (batch, seq_q, q_heads)
    assert torch.isposinf(actual[:, :1]).all()
    assert torch.isfinite(actual[:, 1:]).all()


def test_dsa_score_memory_estimate_exposes_202k_training_boundary():
    from megatron.lite.primitive.kernels import dsa_kernels

    seq = 202_752
    sparse_bytes = dsa_kernels._estimate_dsa_score_peak_bytes(
        1, seq, seq, dense_loss=False
    )
    dense_bytes = dsa_kernels._estimate_dsa_score_peak_bytes(
        1, seq, seq, dense_loss=True
    )
    gib = 1024**3
    assert sparse_bytes / gib > 150.0
    assert dense_bytes / gib > 300.0
    assert dense_bytes > 2 * sparse_bytes


def test_dsa_score_memory_guard_raises_before_predictable_cuda_oom(monkeypatch):
    from megatron.lite.primitive.kernels import dsa_kernels

    gib = 1024**3
    fake_cuda_tensor = SimpleNamespace(
        is_cuda=True,
        device=torch.device("cuda", 0),
    )
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda _device: (80 * gib, 288 * gib),
    )

    with pytest.raises(
        RuntimeError,
        match=r"DSA indexer top-k.*estimated peak of 153\.[0-9] GiB",
    ):
        dsa_kernels._guard_dsa_score_memory(
            fake_cuda_tensor,
            1,
            202_752,
            202_752,
            dense_loss=False,
        )


def test_dsa_score_memory_guard_is_wired_to_indexer_and_fused_paths(monkeypatch):
    from megatron.lite.primitive.kernels import dsa_kernels

    calls = []

    def fail_from_guard(_tensor, batch, seq_q, seq_k, *, dense_loss):
        calls.append((batch, seq_q, seq_k, dense_loss))
        raise RuntimeError("score-memory guard wired")

    monkeypatch.setattr(dsa_kernels, "_ensure_dsa_namespace", lambda: None)
    monkeypatch.setattr(dsa_kernels, "_guard_dsa_score_memory", fail_from_guard)

    q_bshd = torch.zeros(1, 4, 2, 8)
    k_bsd = torch.zeros(1, 1, 8)
    w_bsh = torch.zeros(1, 4, 2)
    with pytest.raises(RuntimeError, match="score-memory guard wired"):
        dsa_kernels._indexer_topk_bshd(q_bshd, k_bsd, w_bsh, 1, ratio=4)

    query = torch.zeros(4, 1, 2, 8)
    kv_full = torch.zeros(5, 1, 8)
    attn_sink = torch.zeros(2)
    window_idxs = torch.zeros(1, 4, 1, dtype=torch.int32)
    q_indexer = torch.zeros(4, 1, 2, 8)
    k_indexer = torch.zeros(1, 1, 8)
    weights = torch.zeros(4, 1, 2)
    with pytest.raises(RuntimeError, match="score-memory guard wired"):
        dsa_kernels.FusedIndexerSparseAttnFunc.apply(
            query,
            kv_full,
            attn_sink,
            window_idxs,
            q_indexer,
            k_indexer,
            weights,
            1,
            4,
            8**-0.5,
            8**-0.5,
            1.0,
            False,
            4,
            False,
            8,
        )

    assert calls == [(1, 4, 1, False), (1, 4, 1, True)]


def test_dsa_bottom_right_topk_lengths_cover_non_square_valid_keys():
    from megatron.lite.primitive.kernels import dsa_kernels

    ratio1 = dsa_kernels._bottom_right_valid_kv_counts(
        seq_q=2, seq_k=4, ratio=1, device=torch.device("cpu")
    )
    assert ratio1.tolist() == [3, 4]

    ratio4 = dsa_kernels._bottom_right_valid_kv_counts(
        seq_q=5, seq_k=2, ratio=4, device=torch.device("cpu")
    )
    assert ratio4.tolist() == [1, 1, 1, 1, 2]

    with pytest.raises(ValueError, match="seq_q <= seq_k"):
        dsa_kernels._bottom_right_valid_kv_counts(
            seq_q=9, seq_k=2, ratio=4, device=torch.device("cpu")
        )


def test_dense_dsa_kl_matches_canonical_epsilon_placement():
    from megatron.lite.primitive.kernels import dsa_kernels

    attn_score = torch.tensor([[[0.7, 0.3, 0.0], [0.0, 0.0, 0.0]]], dtype=torch.float32)
    attn_l1norm = attn_score.sum(dim=-1)
    index_logits = torch.tensor(
        [[[1.2, -0.4, -torch.inf], [-torch.inf, -torch.inf, -torch.inf]]],
        dtype=torch.float32,
    )
    index_lse = torch.logsumexp(index_logits, dim=-1)
    coeff = 0.25

    actual = dsa_kernels._kl_loss_from_dense_scores(
        attn_score,
        attn_l1norm,
        index_logits,
        index_lse,
        coeff,
    )

    target = attn_score[0, 0] / attn_l1norm[0, 0]
    predict = torch.softmax(index_logits[0, 0, :2], dim=-1)
    expected_row = (
        target[:2] * (torch.log(target[:2] + 1.0e-10) - torch.log(predict + 1.0e-10))
    ).sum()
    expected = coeff * expected_row / 2
    torch.testing.assert_close(actual, expected, atol=1.0e-7, rtol=1.0e-7)


def test_dsa_vendor_score_grad_is_adjusted_for_canonical_log_epsilon():
    from megatron.lite.primitive.kernels import dsa_kernels

    logits = torch.tensor([2.0, -30.0, -50.0], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    predict = torch.softmax(logits, dim=-1)
    loss = -(target * torch.log(predict + 1.0e-10)).sum()
    (autograd_grad,) = torch.autograd.grad(loss, logits)

    adjusted_target = dsa_kernels._scale_target_for_canonical_log_eps_backward_(
        target.clone(), predict.detach()
    )
    vendor_equivalent = -adjusted_target + predict.detach() * adjusted_target.sum()
    torch.testing.assert_close(
        vendor_equivalent, autograd_grad, atol=1.0e-12, rtol=1.0e-12
    )
    # The tiny-probability entries must be attenuated; plain -target would be
    # the wrong gradient for log(P + 1e-10).
    assert adjusted_target[-1] < target[-1] * 1.0e-8


def test_glm5_nonpacked_cp_reconstructs_rank3_rotary_in_zigzag_order(monkeypatch):
    from megatron.lite.primitive.modules.attention import dsa
    from megatron.lite.primitive.parallel.cp import zigzag_slice_for_cp

    full_cos = torch.arange(1 * 8 * 4, dtype=torch.float32).view(1, 8, 4)
    full_sin = full_cos + 100.0
    cos_parts = [zigzag_slice_for_cp(full_cos, rank, 2, seq_dim=1) for rank in range(2)]
    sin_parts = [zigzag_slice_for_cp(full_sin, rank, 2, seq_dim=1) for rank in range(2)]

    def fake_all_gather(tensor, *, cp_size, cp_group):
        assert cp_size == 2
        assert cp_group == "cp-group"
        return cos_parts if torch.equal(tensor, cos_parts[0]) else sin_parts

    monkeypatch.setattr(dsa, "_all_gather_cp", fake_all_gather)
    attention = dsa.DynamicSparseAttention.__new__(dsa.DynamicSparseAttention)
    torch.nn.Module.__init__(attention)
    attention.cp_size = 2
    attention.cp_group = "cp-group"

    gathered_cos, gathered_sin = attention._gather_cp_rotary(
        cos_parts[0],
        sin_parts[0],
        local_seq=4,
        full_seq=8,
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(gathered_cos, full_cos)
    torch.testing.assert_close(gathered_sin, full_sin)

    cache_cos = torch.ones(8, 2)
    cache_sin = torch.zeros(8, 2)
    same_cos, same_sin = attention._gather_cp_rotary(
        cache_cos,
        cache_sin,
        local_seq=4,
        full_seq=8,
        device=torch.device("cpu"),
    )
    assert same_cos is cache_cos
    assert same_sin is cache_sin


def test_glm5_packed_cp_rejects_inconsistent_padded_extent():
    from megatron.lite.primitive.modules.attention import dsa

    attention = dsa.DynamicSparseAttention.__new__(dsa.DynamicSparseAttention)
    torch.nn.Module.__init__(attention)
    attention.cp_size = 2
    attention.cp_group = "cp-group"
    packed = SimpleNamespace(
        cu_seqlens_q_padded=torch.tensor([0, 8], dtype=torch.int32)
    )

    with pytest.raises(ValueError, match="padded packed sequence exactly"):
        attention._gather_packed_cp_inputs(
            torch.zeros(1, 3, 8),
            torch.arange(3).unsqueeze(0),
            packed,
        )


@pytest.mark.parametrize(
    ("cos", "sin", "match"),
    [
        (torch.zeros(1, 4, 4), torch.zeros(8, 2), "ranks must match"),
        (torch.zeros(1, 4, 4), torch.zeros(1, 4, 2), "shapes must match"),
        (
            torch.zeros(1, 5, 4),
            torch.zeros(1, 5, 4),
            "local or padded full packed sequence",
        ),
    ],
    ids=["rank", "shape", "coverage"],
)
def test_glm5_packed_cp_rejects_invalid_rotary_representations(cos, sin, match):
    from megatron.lite.primitive.modules.attention import dsa

    attention = dsa.DynamicSparseAttention.__new__(dsa.DynamicSparseAttention)
    torch.nn.Module.__init__(attention)
    attention.cp_size = 2
    attention.cp_group = "cp-group"
    packed = SimpleNamespace(
        cu_seqlens_q_padded=torch.tensor([0, 8], dtype=torch.int32)
    )

    with pytest.raises(ValueError, match=match):
        attention._gather_packed_cp_rotary(
            cos,
            sin,
            packed,
            torch.device("cpu"),
            local_seq=4,
        )


def test_glm5_packed_cp_reconstructs_rank3_rotary(monkeypatch):
    from megatron.lite.primitive.modules.attention import dsa
    from megatron.lite.primitive.parallel.thd import split_packed_to_cp_local

    cu_seqlens = torch.tensor([0, 8], dtype=torch.int32)
    full_cos = torch.arange(1 * 8 * 4, dtype=torch.float32).view(1, 8, 4)
    full_sin = full_cos + 100.0
    cos_parts = [
        split_packed_to_cp_local(
            full_cos,
            cu_seqlens_padded=cu_seqlens,
            cp_size=2,
            cp_rank=rank,
            dim=1,
        )
        for rank in range(2)
    ]
    sin_parts = [
        split_packed_to_cp_local(
            full_sin,
            cu_seqlens_padded=cu_seqlens,
            cp_size=2,
            cp_rank=rank,
            dim=1,
        )
        for rank in range(2)
    ]

    def fake_all_gather(tensor, *, cp_size, cp_group):
        assert cp_size == 2
        assert cp_group == "cp-group"
        return cos_parts if torch.equal(tensor, cos_parts[0]) else sin_parts

    monkeypatch.setattr(dsa, "_all_gather_cp", fake_all_gather)
    attention = dsa.DynamicSparseAttention.__new__(dsa.DynamicSparseAttention)
    torch.nn.Module.__init__(attention)
    attention.cp_size = 2
    attention.cp_group = "cp-group"
    packed = SimpleNamespace(cu_seqlens_q_padded=cu_seqlens)

    gathered_cos, gathered_sin = attention._gather_packed_cp_rotary(
        cos_parts[0],
        sin_parts[0],
        packed,
        torch.device("cpu"),
        local_seq=4,
    )
    torch.testing.assert_close(gathered_cos, full_cos)
    torch.testing.assert_close(gathered_sin, full_sin)


@pytest.mark.parametrize(
    "metadata_index",
    [12, 18],
    ids=["position-local-vs-full", "rotary-local-vs-full"],
)
def test_glm5_cp_rejects_mixed_collective_input_representations(
    monkeypatch, metadata_index
):
    from megatron.lite.primitive.modules.attention import dsa

    attention = dsa.DynamicSparseAttention.__new__(dsa.DynamicSparseAttention)
    torch.nn.Module.__init__(attention)
    attention.cp_size = 2
    attention.cp_group = "cp-group"

    def fake_all_gather(metadata, *, cp_size, cp_group):
        assert cp_size == 2
        assert cp_group == "cp-group"
        peer_metadata = metadata.clone()
        # Metadata layout is header(3), followed by fixed-size shape/dtype
        # records for x, position, cos, and sin. Change one rank's selected
        # sequence axis from local length 4 to full length 8.
        peer_metadata[metadata_index] = 8
        return [metadata, peer_metadata]

    monkeypatch.setattr(dsa, "_all_gather_cp", fake_all_gather)
    with pytest.raises(
        ValueError, match="same local/full position and rotary representation"
    ):
        attention._validate_cp_collective_input_metadata(
            torch.zeros(1, 4, 8),
            torch.arange(4).unsqueeze(0),
            torch.zeros(1, 4, 4),
            torch.zeros(1, 4, 4),
            local_seq=4,
            full_seq=8,
            packed=False,
        )


def test_dsa_index_share_pipeline_guard_rejects_cross_stage_sources():
    from megatron.lite.primitive.modules.attention.dsa import (
        validate_dsa_index_share_pipeline_split,
    )

    validate_dsa_index_share_pipeline_split(
        [0, 1, 2, 3],
        topk_freq=4,
        skip_topk_offset=3,
    )
    with pytest.raises(ValueError, match="cannot cross pipeline stages"):
        validate_dsa_index_share_pipeline_split(
            [3, 4, 5],
            topk_freq=4,
            skip_topk_offset=3,
        )
    with pytest.raises(ValueError, match="must execute before"):
        validate_dsa_index_share_pipeline_split(
            [3, 2],
            topk_freq=4,
            skip_topk_offset=3,
        )


def test_dsa_index_share_pipeline_guard_uses_explicit_nearest_full_source(monkeypatch):
    from megatron.lite.primitive.modules.attention import dsa

    DynamicSparseAttention = dsa.DynamicSparseAttention
    validate_dsa_index_share_pipeline_split = (
        dsa.validate_dsa_index_share_pipeline_split
    )

    indexer_types = ["full", "shared", "full", "shared", "shared", "full"]
    validate_dsa_index_share_pipeline_split(
        [2, 3, 4],
        topk_freq=1,
        skip_topk_offset=0,
        indexer_types=indexer_types,
    )
    with pytest.raises(ValueError, match="cannot cross pipeline stages"):
        validate_dsa_index_share_pipeline_split(
            [3, 4],
            topk_freq=1,
            skip_topk_offset=0,
            indexer_types=indexer_types,
        )

    # A canonical explicit schedule may intentionally contradict freq/offset.
    # The primitive accepts the caller-provided type and source instead of
    # silently recomputing a different schedule.
    monkeypatch.setattr(dsa, "RMSNorm", torch.nn.LayerNorm)
    shared = DynamicSparseAttention(
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
        layer_number=5,
        index_topk_freq=1,
        indexer_type="shared",
        index_share_enabled=True,
        index_share_source_layer=3,
    )
    assert shared.indexer is None
    assert shared.index_share_source_layer == 3
