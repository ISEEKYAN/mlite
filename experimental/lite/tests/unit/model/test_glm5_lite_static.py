"""Static and CPU smoke tests for native GLM-5 lite."""

from __future__ import annotations

from pathlib import Path


def _make_train_config(ps):
    from types import SimpleNamespace

    return SimpleNamespace(
        tp=ps.tp_size,
        ep=ps.ep_size,
        etp=ps.etp_size,
        pp=ps.pp_size,
        cp=ps.cp_size,
        vpp=None,
        use_deepep=False,
        fp8=False,
        recompute_modules=[],
        deterministic=True,
    )


def _make_glm5_model(cfg, ps=None, **kwargs):
    from megatron.lite.model.glm5.lite.model import Glm5Model
    from megatron.lite.primitive.parallel import ParallelState

    ps = ParallelState() if ps is None else ps
    return Glm5Model(cfg, _make_train_config(ps), ps, **kwargs)


def _tiny_config_kwargs():
    return dict(
        num_hidden_layers=2,
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=32,
        max_position_embeddings=16,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_head_dim=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        index_head_dim=8,
        index_n_heads=2,
        index_topk=2,
        intermediate_size=20,
        moe_intermediate_size=6,
        first_k_dense_replace=1,
        n_routed_experts=3,
        n_shared_experts=1,
        num_experts_per_tok=2,
    )


def _glm52_indexer_types(num_layers=78):
    full_layers = {0, 1, 2, *range(6, num_layers, 4)}
    return ["full" if idx in full_layers else "shared" for idx in range(num_layers)]


def test_glm5_registry_resolves_lite():
    from megatron.lite.model.registry import (
        get_train_runtime_module,
        resolve_model_type_from_hf,
        resolve_runtime_model_name,
    )

    runtime_name = resolve_runtime_model_name("glm5", "lite")
    assert runtime_name == "glm5"
    module = get_train_runtime_module(runtime_name)
    assert module.__name__ == "megatron.lite.model.glm5.lite.protocol"
    assert resolve_model_type_from_hf({"model_type": "glm_moe_dsa"}) == "glm5"


def test_glm5_config_reads_hf_architecture_fields():
    from megatron.lite.model.glm5.config import Glm5Config

    cfg = Glm5Config._from_hf_dict(
        {
            "model_type": "glm_moe_dsa",
            "hidden_size": 6144,
            "num_hidden_layers": 78,
            "num_attention_heads": 64,
            "num_key_value_heads": 64,
            "q_lora_rank": 2048,
            "kv_lora_rank": 512,
            "qk_head_dim": 256,
            "qk_nope_head_dim": 192,
            "qk_rope_head_dim": 64,
            "v_head_dim": 256,
            "index_head_dim": 128,
            "index_n_heads": 32,
            "index_topk": 2048,
            "first_k_dense_replace": 3,
            "n_routed_experts": 256,
            "n_shared_experts": 1,
            "num_experts_per_tok": 8,
            "vocab_size": 154880,
            "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
        }
    )

    assert cfg.q_lora_rank == 2048
    assert cfg.kv_lora_rank == 512
    assert cfg.index_topk == 2048
    assert cfg.num_nextn_predict_layers == 1
    assert cfg.rope_theta == 1_000_000.0
    assert cfg.is_moe_layer(2) is False
    assert cfg.is_moe_layer(3) is True


def test_glm5_config_ignores_null_hf_optional_fields():
    from megatron.lite.model.glm5.config import Glm5Config

    cfg = Glm5Config._from_hf_dict(
        {
            "model_type": "glm_moe_dsa",
            "indexer_rope_first": None,
            "indexer_use_hadamard": None,
            "mlp_layer_types": None,
        }
    )

    assert cfg.indexer_rope_first is True
    assert cfg.indexer_use_hadamard is False
    assert cfg.mlp_layer_types is None


def test_glm52_config_preserves_public_positional_signature():
    import inspect

    from megatron.lite.model.glm5.config import Glm5Config

    legacy_names = [
        "num_hidden_layers",
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "max_position_embeddings",
        "rms_norm_eps",
        "attention_dropout",
        "initializer_range",
        "q_lora_rank",
        "kv_lora_rank",
        "qk_head_dim",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "index_head_dim",
        "index_n_heads",
        "index_topk",
        "indexer_layer_norm_eps",
        "indexer_rope_interleave",
        "indexer_rope_first",
        "indexer_use_hadamard",
        "dsa_indexer_loss_coeff",
        "dsa_indexer_use_sparse_loss",
        "calculate_per_token_loss",
        "rope_interleave",
        "rope_theta",
        "latent_rms_norm_eps",
        "intermediate_size",
        "moe_intermediate_size",
        "first_k_dense_replace",
        "n_routed_experts",
        "n_shared_experts",
        "num_experts_per_tok",
        "n_group",
        "topk_group",
        "routed_scaling_factor",
        "norm_topk_prob",
        "num_nextn_predict_layers",
        "mtp_loss_scaling_factor",
        "mtp_use_repeated_layer",
        "mlp_layer_types",
    ]
    parameter_names = list(inspect.signature(Glm5Config).parameters)
    assert parameter_names[: len(legacy_names)] == legacy_names
    assert parameter_names[len(legacy_names) :] == [
        "index_topk_freq",
        "index_skip_topk_offset",
        "index_topk_pattern",
        "indexer_types",
        "index_share_for_mtp_iteration",
        "dsa_rope_layout_revision",
    ]


def test_glm52_config_uses_explicit_indexer_types_as_canonical_schedule():
    import pytest
    from megatron.lite.model.glm5.config import Glm5Config

    indexer_types = _glm52_indexer_types()
    cfg = Glm5Config(
        **{
            **_tiny_config_kwargs(),
            "num_hidden_layers": 78,
            "num_nextn_predict_layers": 1,
        },
        index_topk_freq=4,
        index_skip_topk_offset=3,
        indexer_types=indexer_types,
    )

    assert indexer_types.count("full") == 21
    assert indexer_types.count("shared") == 57
    assert cfg.uses_dsa_index_share is True
    assert cfg.dsa_indexer_type(2) == "full"
    assert cfg.dsa_indexer_type(3) == "shared"
    assert cfg.dsa_indexer_source_layer(3) == 2
    assert cfg.dsa_indexer_type(74) == "full"
    assert cfg.dsa_indexer_type(77) == "shared"
    assert cfg.dsa_indexer_source_layer(77) == 74
    # MTP layer 78 (0-based) is outside the backbone-only indexer_types and is
    # always full.  The serving-only HF index_share_for_mtp_iteration field is
    # not an architecture schedule input in MLite.
    assert cfg.dsa_indexer_type(78) == "full"
    assert cfg.builds_dsa_indexer(78) is True

    custom_types = list(indexer_types)
    custom_types[3] = "full"
    custom_cfg = Glm5Config(
        **{**_tiny_config_kwargs(), "num_hidden_layers": 78},
        index_topk_freq=4,
        index_skip_topk_offset=3,
        indexer_types=custom_types,
    )
    assert custom_cfg.dsa_indexer_type(3) == "full"
    assert custom_cfg.dsa_indexer_source_layer(4) == 3

    with pytest.raises(ValueError, match="shared before any full source"):
        Glm5Config(
            **{**_tiny_config_kwargs(), "num_hidden_layers": 3},
            indexer_types=["shared", "full", "shared"],
        )


def test_glm52_config_pattern_precedence_groups_and_all_full_override():
    from megatron.lite.model.glm5.config import Glm5Config

    pattern_cfg = Glm5Config(
        **{**_tiny_config_kwargs(), "num_hidden_layers": 6},
        index_topk_freq=1,
        index_topk_pattern="FSFSSF",
    )
    assert pattern_cfg.resolved_dsa_indexer_types == (
        "full",
        "shared",
        "full",
        "shared",
        "shared",
        "full",
    )
    assert pattern_cfg.uses_dsa_index_share is True
    assert pattern_cfg.dsa_indexer_source_layer(4) == 2
    assert [pattern_cfg.builds_dsa_indexer(idx) for idx in range(6)] == [
        True,
        False,
        True,
        False,
        False,
        True,
    ]
    assert pattern_cfg.dsa_index_share_decoder_layer_groups() == [
        [0, 1],
        [2, 3, 4],
        [5],
    ]

    # Explicit indexer_types overrides both pattern and a contradictory
    # freq/offset schedule, matching HF's configuration precedence.
    all_full_cfg = Glm5Config(
        **{**_tiny_config_kwargs(), "num_hidden_layers": 6},
        index_topk_freq=4,
        index_skip_topk_offset=3,
        index_topk_pattern="SSSSSS",
        indexer_types=["full"] * 6,
    )
    assert all_full_cfg.uses_dsa_index_share is False
    assert all_full_cfg.has_dsa_index_share_schedule is True
    assert all_full_cfg.uses_configured_dsa_rope_layout is True
    assert all_full_cfg.dsa_index_share_decoder_layer_groups() is None

    # The rotary revision is a load-time compatibility decision, not a live
    # alias of the sharing schedule. Disabling sharing later must not change it.
    all_full_cfg.indexer_types = None
    all_full_cfg.index_topk_pattern = None
    all_full_cfg.index_topk_freq = 1
    all_full_cfg.index_skip_topk_offset = 0
    assert all_full_cfg.has_dsa_index_share_schedule is False
    assert all_full_cfg.uses_configured_dsa_rope_layout is True
    roundtripped = Glm5Config._from_hf_dict(all_full_cfg.to_dict())
    assert roundtripped.has_dsa_index_share_schedule is False
    assert roundtripped.uses_configured_dsa_rope_layout is True


def test_glm5_rope_revision_is_inferred_before_schedule_overrides():
    from megatron.lite.model.glm5.config import Glm5Config

    glm52_source = {
        **_tiny_config_kwargs(),
        "index_topk_freq": 2,
        "index_skip_topk_offset": 1,
        "indexer_types": ["full", "shared"],
    }
    glm52_all_full = Glm5Config._from_hf_dict(
        glm52_source,
        index_topk_freq=1,
        index_skip_topk_offset=0,
        indexer_types=["full", "full"],
    )
    assert glm52_all_full.uses_dsa_index_share is False
    assert glm52_all_full.uses_configured_dsa_rope_layout is True

    glm51_with_local_all_full = Glm5Config._from_hf_dict(
        _tiny_config_kwargs(), indexer_types=["full", "full"]
    )
    assert glm51_with_local_all_full.uses_dsa_index_share is False
    assert glm51_with_local_all_full.uses_configured_dsa_rope_layout is False


def test_glm51_gate_off_preserves_legacy_rope_layout(
    transformer_engine_import_stub, monkeypatch
):
    import torch.nn as nn

    transformer_engine_import_stub()
    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.model.glm5.lite.model import Glm5DSAAttention
    from megatron.lite.primitive.modules.attention import dsa
    from megatron.lite.primitive.parallel import ParallelState

    monkeypatch.setattr(dsa, "RMSNorm", nn.RMSNorm)

    published_rope_fields = {"rope_interleave": True, "indexer_rope_interleave": True}
    legacy_cfg = Glm5Config(**_tiny_config_kwargs(), **published_rope_fields)
    glm52_cfg = Glm5Config(
        **_tiny_config_kwargs(),
        **published_rope_fields,
        indexer_types=["full", "shared"],
    )
    glm52_all_full_cfg = Glm5Config(
        **_tiny_config_kwargs(), **published_rope_fields, indexer_types=["full", "full"]
    )

    assert legacy_cfg.has_dsa_index_share_schedule is False
    assert legacy_cfg.dsa_rope_layout_revision == "legacy"
    # The exported wrapper keeps its pre-PR two-argument constructor contract.
    legacy_dsa = Glm5DSAAttention(legacy_cfg, ParallelState()).self_attention
    assert legacy_dsa.layer_number == 1
    assert legacy_dsa.rope_interleaved is False
    assert legacy_dsa.indexer is not None
    assert legacy_dsa.indexer.rope_interleaved is False

    assert glm52_cfg.dsa_rope_layout_revision == "configured"
    glm52_dsa = Glm5DSAAttention(glm52_cfg, ParallelState(), 0).self_attention
    assert glm52_dsa.rope_interleaved is True
    assert glm52_dsa.indexer is not None
    assert glm52_dsa.indexer.rope_interleaved is True

    assert glm52_all_full_cfg.uses_dsa_index_share is False
    assert glm52_all_full_cfg.uses_configured_dsa_rope_layout is True
    all_full_dsa = Glm5DSAAttention(
        glm52_all_full_cfg, ParallelState(), 0
    ).self_attention
    assert all_full_dsa.rope_interleaved is True
    assert all_full_dsa.indexer is not None
    assert all_full_dsa.indexer.rope_interleaved is True


def test_glm5_dsa_attention_preserves_explicit_packed_position_resets(
    transformer_engine_import_stub,
):
    from types import SimpleNamespace

    import pytest
    import torch
    import torch.nn as nn

    transformer_engine_import_stub()
    from megatron.lite.model.glm5.lite.model import Glm5DSAAttention
    from megatron.lite.primitive.modules.attention import build_rotary_embeddings

    class CaptureDSA(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, x, **kwargs):
            self.calls.append(
                {
                    "position_ids": kwargs["position_ids"].detach().clone(),
                    "cos": kwargs["cos"].detach().clone(),
                    "sin": kwargs["sin"].detach().clone(),
                }
            )
            return torch.zeros_like(x)

    attention = Glm5DSAAttention.__new__(Glm5DSAAttention)
    nn.Module.__init__(attention)
    attention.ps = SimpleNamespace(cp_size=1, cp_rank=0)
    attention.qk_rope_head_dim = 4
    attention.rope_theta = 10_000.0
    capture = CaptureDSA()
    attention.self_attention = capture

    x = torch.zeros(6, 1, 8)
    reset_positions = torch.tensor([[0, 1, 2, 0, 1, 2]], dtype=torch.long)
    out = attention(x, packed_seq_params=object(), position_ids=reset_positions)

    assert out.shape == x.shape
    assert torch.equal(capture.calls[0]["position_ids"], reset_positions)
    segment_cos, segment_sin = build_rotary_embeddings(
        position_ids=reset_positions[:, :3],
        dim=attention.qk_rope_head_dim,
        rope_theta=attention.rope_theta,
        dtype=x.dtype,
    )
    torch.testing.assert_close(
        capture.calls[0]["cos"], torch.cat((segment_cos, segment_cos), dim=1)
    )
    torch.testing.assert_close(
        capture.calls[0]["sin"], torch.cat((segment_sin, segment_sin), dim=1)
    )

    with pytest.raises(
        ValueError, match="packed sequences require explicit position_ids"
    ):
        attention(x, packed_seq_params=object())

    # Direct non-packed legacy callers still get the historical monotonic fallback.
    attention(x)
    assert torch.equal(capture.calls[1]["position_ids"], torch.arange(6).unsqueeze(0))


def test_glm5_dsa_attention_generates_global_zigzag_positions_for_cp(
    transformer_engine_import_stub,
):
    from types import SimpleNamespace

    import torch
    import torch.nn as nn

    transformer_engine_import_stub()
    from megatron.lite.model.glm5.lite.model import Glm5DSAAttention
    from megatron.lite.primitive.parallel import zigzag_position_ids_for_cp

    class CaptureDSA(nn.Module):
        def __init__(self):
            super().__init__()
            self.position_ids = None

        def forward(self, x, **kwargs):
            self.position_ids = kwargs["position_ids"].detach().clone()
            return torch.zeros_like(x)

    attention = Glm5DSAAttention.__new__(Glm5DSAAttention)
    nn.Module.__init__(attention)
    attention.ps = SimpleNamespace(cp_size=2, cp_rank=1)
    attention.qk_rope_head_dim = 4
    attention.rope_theta = 10_000.0
    capture = CaptureDSA()
    attention.self_attention = capture

    local_seq = 4
    x = torch.zeros(local_seq, 1, 8)
    out = attention(x)

    expected = zigzag_position_ids_for_cp(
        local_seq * attention.ps.cp_size,
        attention.ps.cp_rank,
        attention.ps.cp_size,
        x.device,
    )
    assert out.shape == x.shape
    assert torch.equal(capture.position_ids, expected)


def test_glm5_threads_positions_through_trunk_and_mtp(transformer_engine_import_stub):
    from types import SimpleNamespace

    import torch
    import torch.nn as nn

    transformer_engine_import_stub()
    from megatron.lite.model.glm5.lite.model import Glm5Model, Glm5MTPLayer
    from megatron.lite.primitive.parallel import ParallelState

    class RecordingLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.position_ids = None

        def forward(
            self,
            x,
            packed_seq_params=None,
            dsa_index_share_state=None,
            position_ids=None,
        ):
            del packed_seq_params, dsa_index_share_state
            self.position_ids = (
                None if position_ids is None else position_ids.detach().clone()
            )
            return x

    positions = torch.tensor([[0, 1, 2, 0, 1, 2]], dtype=torch.long)
    packed = SimpleNamespace(
        cu_seqlens_q=torch.tensor([0, 3, 6], dtype=torch.int32),
        local_cp_size=None,
        cp_rank=0,
        cp_group=None,
    )

    # Exercise Glm5Model -> Glm5Layer argument propagation without requiring
    # Transformer Engine kernels in this CPU unit test.
    model = Glm5Model.__new__(Glm5Model)
    nn.Module.__init__(model)
    trunk = RecordingLayer()
    model.layers = nn.ModuleList([trunk])
    model.embed = None
    model.norm = None
    model.head = None
    model.mtp = None
    model._input_tensor = None
    model._dsa_index_share_consumer_counts = {}
    model.train_config = SimpleNamespace(fp8=False)
    model.ps = ParallelState()
    hidden = torch.zeros(6, 1, 4)
    model(hidden_states=hidden, position_ids=positions, packed_seq_params=packed)
    assert torch.equal(trunk.position_ids, positions)

    class FakeEmbedding(nn.Module):
        def forward(self, input_ids):
            return input_ids.transpose(0, 1).unsqueeze(-1).expand(-1, -1, 4).float()

    class KeepHiddenWidth(nn.Module):
        def forward(self, value):
            return value[..., :4]

    # MTP rolls token/label positions per packed sequence, while attention
    # deliberately retains the original rotary coordinates (the same contract
    # used by MCore's MTP block).
    mtp = Glm5MTPLayer.__new__(Glm5MTPLayer)
    nn.Module.__init__(mtp)
    mtp.ps = ParallelState()
    mtp.embedding = FakeEmbedding()
    mtp.detach_encoder = False
    mtp.enorm = nn.Identity()
    mtp.hnorm = nn.Identity()
    mtp.eh_proj = KeepHiddenWidth()
    mtp.transformer_layer = RecordingLayer()
    mtp.final_layernorm = nn.Identity()
    _mtp_hidden, rolled_input_ids, rolled_position_ids = mtp(
        input_ids=torch.tensor([[3, 4, 5, 6, 7, 8]], dtype=torch.long),
        position_ids=positions,
        hidden_states=hidden,
        packed_seq_params=packed,
    )
    assert torch.equal(
        rolled_input_ids, torch.tensor([[4, 5, 0, 7, 8, 0]], dtype=torch.long)
    )
    assert torch.equal(
        rolled_position_ids, torch.tensor([[1, 2, 0, 1, 2, 0]], dtype=torch.long)
    )
    assert torch.equal(mtp.transformer_layer.position_ids, positions)


def test_glm52_serving_mtp_share_metadata_is_preserved_but_mtp_is_always_full():
    from megatron.lite.model.glm5.config import Glm5Config

    cfg = Glm5Config._from_hf_dict(
        {
            **_tiny_config_kwargs(),
            "num_hidden_layers": 4,
            "num_nextn_predict_layers": 2,
            "index_topk_freq": 3,
            "index_skip_topk_offset": 1,
            "indexer_types": ["full", "shared", "shared", "full"],
            "index_share_for_mtp_iteration": True,
        }
    )

    # The legacy formula would classify global layers 5 and 6 differently;
    # architecture construction must not apply it to MTP/nextn layers.
    assert [cfg.dsa_indexer_type(idx) for idx in (4, 5)] == ["full", "full"]
    assert [cfg.dsa_indexer_source_layer(idx) for idx in (4, 5)] == [4, 5]
    assert [cfg.builds_dsa_indexer(idx) for idx in (4, 5)] == [True, True]
    assert cfg.index_share_for_mtp_iteration is True
    assert cfg.to_dict()["index_share_for_mtp_iteration"] is True


def test_glm5_config_preserves_mtp_aliases_and_layer_types():
    from megatron.lite.model.glm5.config import Glm5Config

    cfg = Glm5Config._from_hf_dict(
        {
            **_tiny_config_kwargs(),
            "num_nextn_predict": 1,
            "mtp_loss_scaling_factor": 0.2,
            "mlp_layer_types": ["dense", "sparse", "sparse"],
        }
    )

    assert cfg.num_nextn_predict_layers == 1
    assert cfg.mtp_loss_scaling_factor == 0.2
    assert cfg.is_moe_layer(2) is True


def test_glm5_lite_does_not_import_wrappers_or_sibling_models():
    root = (
        Path(__file__).resolve().parents[3]
        / "megatron"
        / "lite"
        / "model"
        / "glm5"
        / "lite"
    )
    for path in root.glob("*.py"):
        text = path.read_text()
        assert "megatron.lite.model.qwen" not in text
        assert "mbridge" not in text
        assert "MCore" not in text
        assert "megatron.core" not in text


def test_glm5_lite_uses_shared_mla_and_dsa_primitive():
    root = Path(__file__).resolve().parents[3] / "megatron" / "lite"
    model_text = (root / "model" / "glm5" / "lite" / "model.py").read_text()
    primitive_text = (
        root / "primitive" / "modules" / "attention" / "dsa.py"
    ).read_text()
    kernel_text = (root / "primitive" / "kernels" / "dsa_kernels.py").read_text()

    assert "DynamicSparseAttention" in model_text
    assert (
        "from megatron.lite.primitive.modules.attention.mla import MultiLatentAttention"
        in primitive_text
    )
    assert "class DynamicSparseAttention" in primitive_text
    assert "class MultiLatentAttention" not in primitive_text
    assert "class DSAIndexer" in primitive_text
    assert "megatron.core" not in primitive_text
    assert "dsa_kernels.fused_indexer_sparse_attn" in primitive_text
    assert "dsa_kernels.dsa_sparse_attn" in primitive_text
    assert "dsa_kernels.indexer_topk" in primitive_text
    assert "value_dim" in kernel_text
    assert "from cudnn.deepseek_sparse_attention import DSA" in kernel_text
    assert "from cudnn import DSA" in kernel_text
    assert (
        "cudnn.deepseek_sparse_attention.indexer_forward._interface_sm90" in kernel_text
    )
    assert "cudnn.deepseek_sparse_attention.indexer_forward._interface" in kernel_text
    assert "torch.cuda.get_device_capability(device)" in kernel_text
    assert "torch.topk" not in primitive_text
    assert "torch.softmax" not in primitive_text
    assert "torch.matmul" not in primitive_text


def test_glm5_dsa_kernel_routes_indexer_forward_by_sm(monkeypatch):
    from megatron.lite.primitive.kernels import dsa_kernels

    sm90_entry = object()
    sm100_entry = object()

    monkeypatch.setattr(dsa_kernels, "_load_indexer_fwd_sm90", lambda: sm90_entry)
    monkeypatch.setattr(dsa_kernels, "_load_indexer_fwd_sm100", lambda: sm100_entry)

    monkeypatch.setattr(
        dsa_kernels.torch.cuda, "get_device_capability", lambda device: (9, 0)
    )
    assert dsa_kernels._select_indexer_forward(None) is sm90_entry

    monkeypatch.setattr(
        dsa_kernels.torch.cuda, "get_device_capability", lambda device: (10, 0)
    )
    assert dsa_kernels._select_indexer_forward(None) is sm100_entry

    monkeypatch.setattr(
        dsa_kernels.torch.cuda, "get_device_capability", lambda device: (8, 0)
    )
    assert dsa_kernels._select_indexer_forward(None) is None


def test_glm5_dsa_training_forward_uses_fused_kernel(monkeypatch):
    import pytest
    import torch
    from megatron.lite.primitive.modules.attention import (
        DynamicSparseAttention,
        build_rope_cache,
        dsa,
    )

    if not torch.cuda.is_available():
        pytest.skip("GLM-5 native attention requires CUDA (Transformer Engine RMSNorm)")
    device = torch.device("cuda")

    calls = {}

    def fake_fused_indexer_sparse_attn(
        query,
        kv_full,
        attn_sink,
        window_idxs,
        q_indexer,
        k_indexer,
        weights,
        indexer_topk,
        ratio,
        softmax_scale,
        indexer_softmax_scale=1.0,
        loss_coeff=0.0,
        sparse_loss=False,
        kv_offset=0,
        calculate_per_token_loss=False,
        value_dim=None,
    ):
        del (
            attn_sink,
            q_indexer,
            k_indexer,
            weights,
            softmax_scale,
            indexer_softmax_scale,
        )
        calls["training"] = {
            "query_shape": tuple(query.shape),
            "kv_shape": tuple(kv_full.shape),
            "window_shape": tuple(window_idxs.shape),
            "indexer_topk": indexer_topk,
            "ratio": ratio,
            "loss_coeff": loss_coeff,
            "sparse_loss": sparse_loss,
            "kv_offset": kv_offset,
            "calculate_per_token_loss": calculate_per_token_loss,
            "value_dim": value_dim,
        }
        return query.new_zeros(
            query.shape[0], query.shape[1], query.shape[2] * value_dim
        ), torch.zeros((), device=query.device, dtype=torch.float32)

    monkeypatch.setattr(
        dsa._dsa_kernels, "fused_indexer_sparse_attn", fake_fused_indexer_sparse_attn
    )

    attn = DynamicSparseAttention(
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
    )
    attn.to(device=device, dtype=torch.bfloat16)
    attn.train()
    x = torch.randn(1, 4, 16, device=device, dtype=torch.bfloat16)
    cos, sin = build_rope_cache(
        dim=4, max_position_embeddings=4, rope_theta=1_000_000.0, device=device
    )
    position_ids = torch.arange(4, device=device).unsqueeze(0)

    out = attn(x, cos=cos, sin=sin, position_ids=position_ids)

    assert out.shape == (1, 4, 16)
    assert calls["training"] == {
        "query_shape": (4, 1, 2, 8),
        "kv_shape": (4, 1, 8),
        "window_shape": (1, 4, 0),
        "indexer_topk": 2,
        "ratio": 1,
        "loss_coeff": 0.0,
        "sparse_loss": False,
        "kv_offset": 0,
        "calculate_per_token_loss": False,
        "value_dim": 4,
    }


def test_glm5_dsa_eval_forward_uses_fused_sparse_attention(monkeypatch):
    import pytest
    import torch
    from megatron.lite.primitive.modules.attention import (
        DynamicSparseAttention,
        build_rope_cache,
        dsa,
    )

    if not torch.cuda.is_available():
        pytest.skip("GLM-5 native attention requires CUDA (Transformer Engine RMSNorm)")
    device = torch.device("cuda")

    calls = {}

    def fake_indexer_topk(
        q_indexer, k_indexer, weights, topk, ratio, indexer_softmax_scale=1.0
    ):
        del q_indexer, k_indexer, weights, indexer_softmax_scale
        calls["indexer"] = {"topk": topk, "ratio": ratio}
        idx = torch.zeros((1, 4, topk), dtype=torch.int32, device=device)
        return idx, torch.full((1, 4), topk, dtype=torch.int32, device=device)

    def fake_dsa_sparse_attn(
        query,
        kv_full,
        attn_sink,
        topk_idxs,
        softmax_scale,
        topk_length=None,
        indexer_topk=0,
        value_dim=None,
    ):
        del kv_full, attn_sink, topk_idxs, softmax_scale, indexer_topk
        calls["sparse"] = {
            "topk_length_is_set": topk_length is not None,
            "value_dim": value_dim,
        }
        return query.new_zeros(
            query.shape[0], query.shape[1], query.shape[2] * value_dim
        )

    monkeypatch.setattr(dsa._dsa_kernels, "indexer_topk", fake_indexer_topk)
    monkeypatch.setattr(dsa._dsa_kernels, "dsa_sparse_attn", fake_dsa_sparse_attn)

    attn = DynamicSparseAttention(
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
    )
    attn.to(device=device, dtype=torch.bfloat16)
    attn.eval()
    x = torch.randn(1, 4, 16, device=device, dtype=torch.bfloat16)
    cos, sin = build_rope_cache(
        dim=4, max_position_embeddings=4, rope_theta=1_000_000.0, device=device
    )
    position_ids = torch.arange(4, device=device).unsqueeze(0)

    with torch.no_grad():
        out = attn(x, cos=cos, sin=sin, position_ids=position_ids)

    assert out.shape == (1, 4, 16)
    assert calls["indexer"] == {"topk": 2, "ratio": 1}
    assert calls["sparse"] == {"topk_length_is_set": True, "value_dim": 4}


def test_glm5_lite_model_exports_native_state_names():
    from megatron.lite.model.glm5.config import Glm5Config

    model = _make_glm5_model(Glm5Config(**_tiny_config_kwargs()))
    keys = set(model.state_dict())

    assert "embed.embedding.weight" in keys
    assert "layers.0.self_attention.self_attention.q_a_proj.weight" in keys
    assert "layers.0.mlp.gate_up.linear.weight" in keys
    assert "layers.1.moe.router.gate.weight" in keys
    assert "layers.1.moe.experts.fc1.weight0" in keys
    assert "layers.1.moe.shared_expert.gate_up.linear.weight" in keys
    assert "head.col.linear.weight" in keys


def test_glm52_index_share_shared_layers_omit_indexer_modules():
    import pytest

    try:
        import transformer_engine.pytorch  # noqa: F401
    except (ModuleNotFoundError, OSError) as exc:
        pytest.skip(f"Transformer Engine is not importable in this environment: {exc}")

    from megatron.lite.model.glm5.config import Glm5Config

    cfg = Glm5Config(
        **{
            **_tiny_config_kwargs(),
            "num_hidden_layers": 6,
            "num_nextn_predict_layers": 1,
        },
        index_topk_freq=4,
        index_skip_topk_offset=3,
        # Deliberately differs from freq/offset: module construction must obey
        # the explicit HF list, not the legacy inferred schedule.
        indexer_types=["full", "shared", "full", "shared", "shared", "full"],
    )
    model = _make_glm5_model(cfg, mtp_enable=True)
    attention_modules = [layer.self_attention.self_attention for layer in model.layers]

    assert [module.indexer is not None for module in attention_modules] == [
        True,
        False,
        True,
        False,
        False,
        True,
    ]
    assert model.mtp is not None
    mtp_attention = model.mtp.layers[0].transformer_layer.self_attention.self_attention
    assert mtp_attention.layer_number == 7
    assert mtp_attention.indexer is not None

    keys = set(model.state_dict())
    assert "layers.1.self_attention.self_attention.indexer.wq_b.weight" not in keys
    assert "layers.2.self_attention.self_attention.indexer.wq_b.weight" in keys
    assert "layers.3.self_attention.self_attention.indexer.wq_b.weight" not in keys
    assert "layers.5.self_attention.self_attention.indexer.wq_b.weight" in keys
    assert (
        "mtp.layers.0.transformer_layer.self_attention.self_attention.indexer.wq_b.weight"
        in keys
    )


def test_glm5_checkpoint_exports_and_saves_hf_style_weights(tmp_path):
    import torch
    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.model.glm5.lite.checkpoint import (
        export_hf_weights,
        save_hf_weights,
    )
    from megatron.lite.primitive.parallel import ParallelState
    from safetensors import safe_open

    cfg = Glm5Config(**_tiny_config_kwargs())
    ps = ParallelState()
    model = _make_glm5_model(cfg, ps=ps)
    model.layers[1].moe.router.expert_bias.copy_(torch.tensor([0.25, -0.5, 1.0]))

    exported = dict(export_hf_weights(model, cfg, ps))
    state = model.state_dict()

    assert torch.equal(
        exported["model.layers.1.mlp.experts.2.gate_proj.weight"],
        state["layers.1.moe.experts.fc1.weight2"][: cfg.moe_intermediate_size]
        .detach()
        .cpu(),
    )
    assert torch.equal(
        exported["model.layers.1.mlp.gate.e_score_correction_bias"],
        state["layers.1.moe.router.expert_bias"].detach().cpu(),
    )
    assert "model.layers.1.mlp.experts.gate_up_proj" not in exported

    hf_dir = tmp_path / "hf"
    save_hf_weights(model, str(hf_dir), cfg, ps)
    with safe_open(
        str(hf_dir / "model.safetensors"), framework="pt", device="cpu"
    ) as handle:
        assert torch.equal(
            handle.get_tensor("model.layers.1.mlp.experts.2.down_proj.weight"),
            state["layers.1.moe.experts.fc2.weight2"].detach().cpu(),
        )
        assert torch.equal(
            handle.get_tensor("model.layers.1.mlp.gate.e_score_correction_bias"),
            state["layers.1.moe.router.expert_bias"].detach().cpu(),
        )

    loaded = _make_glm5_model(cfg, ps=ps)
    from megatron.lite.model.glm5.lite.checkpoint import load_hf_weights

    load_hf_weights(loaded, str(hf_dir), cfg, ps)
    assert torch.equal(
        loaded.state_dict()["layers.1.moe.router.expert_bias"].detach().cpu(),
        state["layers.1.moe.router.expert_bias"].detach().cpu(),
    )

    hf_bf16_dir = tmp_path / "hf_bf16"
    save_hf_weights(model, str(hf_bf16_dir), cfg, ps, export_dtype=torch.bfloat16)
    with safe_open(
        str(hf_bf16_dir / "model.safetensors"), framework="pt", device="cpu"
    ) as handle:
        floating_dtypes = {
            handle.get_tensor(key).dtype
            for key in handle.keys()
            if handle.get_tensor(key).is_floating_point()
        }
        assert floating_dtypes == {torch.bfloat16}

    loaded_bf16 = _make_glm5_model(cfg, ps=ps)
    load_hf_weights(loaded_bf16, str(hf_bf16_dir), cfg, ps)
    assert torch.equal(
        loaded_bf16.state_dict()["layers.1.moe.experts.fc1.weight2"][
            cfg.moe_intermediate_size :
        ]
        .detach()
        .cpu(),
        state["layers.1.moe.experts.fc1.weight2"][cfg.moe_intermediate_size :]
        .detach()
        .cpu()
        .to(torch.bfloat16)
        .to(torch.float32),
    )


def test_glm5_checkpoint_exports_and_loads_mtp_layers(tmp_path):
    import torch
    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.model.glm5.lite.checkpoint import (
        export_hf_weights,
        load_hf_weights,
    )
    from megatron.lite.primitive.ckpt.hf_weights import save_safetensors
    from megatron.lite.primitive.parallel import ParallelState

    cfg = Glm5Config(**_tiny_config_kwargs(), num_nextn_predict_layers=1)
    ps = ParallelState()
    model = _make_glm5_model(cfg, ps=ps, mtp_enable=True)
    state = model.state_dict()

    assert "mtp.layers.0.eh_proj.linear.weight" in state
    assert "mtp.layers.0.transformer_layer.input_layernorm.weight" in state

    exported = dict(export_hf_weights(model, cfg, ps))
    assert "model.layers.2.eh_proj.weight" in exported
    assert "model.layers.2.enorm.weight" in exported
    assert "model.layers.2.hnorm.weight" in exported
    assert "model.layers.2.shared_head.norm.weight" in exported
    assert "model.layers.2.input_layernorm.weight" in exported
    assert "model.layers.2.mlp.gate.weight" in exported
    assert "model.layers.2.mlp.gate.e_score_correction_bias" in exported

    save_safetensors(exported, str(tmp_path))
    loaded = _make_glm5_model(cfg, ps=ps, mtp_enable=True)
    load_hf_weights(loaded, str(tmp_path), cfg, ps)
    assert torch.equal(
        loaded.state_dict()["mtp.layers.0.eh_proj.linear.weight"],
        state["mtp.layers.0.eh_proj.linear.weight"],
    )


def test_glm52_checkpoint_mapping_skips_shared_indexer_without_te():
    import importlib.util

    import pytest
    import torch
    from megatron.lite.model.glm5.config import Glm5Config

    pytest.importorskip("safetensors")

    checkpoint_path = (
        Path(__file__).resolve().parents[3]
        / "megatron"
        / "lite"
        / "model"
        / "glm5"
        / "lite"
        / "checkpoint.py"
    )
    module_spec = importlib.util.spec_from_file_location(
        "_glm5_checkpoint_test", checkpoint_path
    )
    assert module_spec is not None and module_spec.loader is not None
    checkpoint_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(checkpoint_module)

    cfg = Glm5Config(
        **{
            **_tiny_config_kwargs(),
            "num_hidden_layers": 6,
            "num_nextn_predict_layers": 1,
        },
        index_topk_freq=4,
        index_skip_topk_offset=3,
        # Deliberately differs from freq/offset: checkpoint presence must obey
        # the explicit HF list, not the legacy inferred schedule.
        indexer_types=["full", "shared", "full", "shared", "shared", "full"],
    )
    spec = checkpoint_module.Glm5WeightSpec(cfg)
    tensor = torch.ones(1)

    assert spec.native_to_hf(
        "layers.2.self_attention.self_attention.indexer.wq_b.weight", tensor
    ) == [("model.layers.2.self_attn.indexer.wq_b.weight", tensor)]
    assert (
        spec.native_to_hf(
            "layers.3.self_attention.self_attention.indexer.wq_b.weight", tensor
        )
        == []
    )
    assert (
        spec.native_to_hf(
            "layers.1.self_attention.self_attention.indexer.wq_b.weight", tensor
        )
        == []
    )
    assert spec.native_to_hf(
        "layers.5.self_attention.self_attention.indexer.wq_b.weight", tensor
    ) == [("model.layers.5.self_attn.indexer.wq_b.weight", tensor)]
    assert spec.native_to_hf(
        "mtp.layers.0.transformer_layer.self_attention.self_attention.indexer.wq_b.weight",
        tensor,
    ) == [("model.layers.6.self_attn.indexer.wq_b.weight", tensor)]

    base_names = {
        "model.layers.3.self_attn.q_a_proj.weight",
        "model.layers.3.self_attn.q_a_layernorm.weight",
        "model.layers.3.self_attn.q_b_proj.weight",
        "model.layers.3.self_attn.kv_a_proj_with_mqa.weight",
        "model.layers.3.self_attn.kv_a_layernorm.weight",
        "model.layers.3.self_attn.kv_b_proj.weight",
        "model.layers.3.self_attn.o_proj.weight",
    }

    class Reader:
        index = {name: "model.safetensors" for name in base_names}

        def get_tensor(self, name):
            if name not in self.index:
                raise KeyError(name)
            return torch.ones(1)

    out = {}
    checkpoint_module._load_attention(
        out,
        local_prefix="layers.3",
        hf_prefix="model.layers.3.self_attn",
        reader=Reader(),
        ps=object(),
        load_indexer=False,
    )
    assert "layers.3.self_attention.self_attention.q_a_proj.weight" in out
    assert not any(".indexer." in name for name in out)


def test_glm52_fp8_contract_matches_pinned_released_header_authority():
    """Validate GLM-5.2's production FP8 contract without weight payloads."""
    import hashlib
    import json
    import math

    authority_path = Path(__file__).with_name("glm52_fp8_header_authority.json")
    authority = json.loads(authority_path.read_text())
    source = authority["source"]

    assert source["repo"] == "zai-org/GLM-5.2-FP8"
    assert source["revision"] == "31cba24fb749908a485082bdeed6eb1ac6cffc2f"
    assert source["config_sha256"] == (
        "d1539d36be7546a1d827fe9cf74c55874695652efb6a5aaa3e60cde1c76ba819"
    )
    assert source["index_sha256"] == (
        "e0fe7f28c1f853d4824e4d796374e3dacf1fe470988773952c79b063768134bf"
    )
    assert source["tensor_count"] == 118629
    assert source["shard_count"] == 141
    assert source["config_values"] == {
        "hidden_size": 6144,
        "latent_rms_norm_eps": None,
        "q_a_layernorm_effective_eps": 1e-6,
        "q_lora_rank": 2048,
        "rms_norm_eps": 1e-5,
    }

    shard = source["safetensors"]["model-00001-of-00141.safetensors"]
    header_prefix = bytes(shard["header_prefix"])
    assert len(header_prefix) == 8
    assert int.from_bytes(header_prefix, byteorder="little") == shard["header_bytes"]
    assert shard["header_bytes"] == 21840
    assert shard["header_range"] == [8, 21847]
    assert (
        shard["header_range"][1] - shard["header_range"][0] + 1 == shard["header_bytes"]
    )
    assert shard["header_sha256"] == (
        "b438c86e0ba40b96525672d31da75756307c723a3a73efc5ea40bd5893411613"
    )

    tensors = authority["tensors"]
    canonical = json.dumps(
        tensors, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    contract_sha256 = hashlib.sha256(canonical).hexdigest()
    assert authority["canonical_tensor_contract_sha256"] == (
        "5fbeff826bdf608cb6596de4987d6926df613076ed32ff69803007270183f294"
    )
    assert contract_sha256 == authority["canonical_tensor_contract_sha256"]

    weight_name = "model.layers.0.self_attn.q_a_proj.weight"
    scale_name = f"{weight_name}_scale_inv"
    norm_name = "model.layers.0.self_attn.q_a_layernorm.weight"
    assert set(tensors) == {weight_name, scale_name, norm_name}

    weight = tensors[weight_name]
    scale = tensors[scale_name]
    norm = tensors[norm_name]
    assert weight["dtype"] == "F8_E4M3"
    assert weight["shape"] == [2048, 6144]
    assert scale["dtype"] == "F32"
    assert norm == {
        "data_offsets": [3807130368, 3807134464],
        "dtype": "BF16",
        "shape": [2048],
    }

    quantization = authority["quantization_config"]
    assert quantization == {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
    }
    block_rows, block_cols = quantization["weight_block_size"]
    rows, cols = weight["shape"]
    expected_scale_shape = [
        (rows + block_rows - 1) // block_rows,
        (cols + block_cols - 1) // block_cols,
    ]
    assert scale["shape"] == expected_scale_shape == [16, 48]

    dtype_bytes = {"F8_E4M3": 1, "F32": 4, "BF16": 2}
    payload_ranges = shard["payload_ranges"]
    assert payload_ranges.keys() == tensors.keys()
    data_start = 8 + shard["header_bytes"]
    for name, tensor in tensors.items():
        start, end = tensor["data_offsets"]
        assert end > start >= 0
        assert end - start == math.prod(tensor["shape"]) * dtype_bytes[tensor["dtype"]]
        # Safetensors offsets are relative to the data section after the
        # 8-byte length prefix and JSON header.
        assert data_start + end <= shard["file_bytes"]
        assert payload_ranges[name]["file_range"] == [
            data_start + start,
            data_start + end - 1,
        ]
        assert len(payload_ranges[name]["sha256"]) == 64


def test_glm52_fp8_dequant_requires_exact_released_block_scale_contract():
    import pytest
    import torch

    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch float8_e4m3fn is required for the FP8 checkpoint contract")

    from megatron.lite.model.glm5.lite.checkpoint import _dequant_fp8_weight

    weight = torch.ones((129, 130), dtype=torch.float32).to(torch.float8_e4m3fn)

    class Reader:
        index = {"w_scale_inv": "fake.safetensors"}

        def __init__(self, scale):
            self.scale = scale

        def get_tensor(self, name):
            assert name == "w_scale_inv"
            return self.scale

    scale = torch.tensor([[0.5, 1.0], [1.5, 2.0]], dtype=torch.float32)
    actual = _dequant_fp8_weight(Reader(scale), "w", weight)
    expected_scale = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
    expected = weight.float() * expected_scale[:129, :130]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    class MissingScaleReader:
        index = {}

    with pytest.raises(KeyError, match="missing required block scale"):
        _dequant_fp8_weight(MissingScaleReader(), "w", weight)
    with pytest.raises(ValueError, match=r"expected=\(2, 2\)"):
        _dequant_fp8_weight(Reader(torch.ones(1, 2)), "w", weight)
    with pytest.raises(ValueError, match="must be float32"):
        _dequant_fp8_weight(Reader(torch.ones(2, 2, dtype=torch.bfloat16)), "w", weight)
    with pytest.raises(ValueError, match="non-finite"):
        _dequant_fp8_weight(Reader(torch.full((2, 2), float("nan"))), "w", weight)
    with pytest.raises(ValueError, match="strictly positive"):
        _dequant_fp8_weight(Reader(torch.zeros(2, 2)), "w", weight)

    int8_weight = torch.ones((2, 2), dtype=torch.int8)
    assert _dequant_fp8_weight(MissingScaleReader(), "int8", int8_weight) is int8_weight


def test_glm52_checkpoint_skips_shared_indexer_weights_and_loads_full_layers(tmp_path):
    import pytest
    import torch

    try:
        import transformer_engine.pytorch  # noqa: F401
    except (ModuleNotFoundError, OSError) as exc:
        pytest.skip(f"Transformer Engine is not importable in this environment: {exc}")

    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.model.glm5.lite.checkpoint import (
        export_hf_weights,
        load_hf_weights,
    )
    from megatron.lite.primitive.ckpt.hf_weights import save_safetensors
    from megatron.lite.primitive.parallel import ParallelState

    cfg = Glm5Config(
        **{
            **_tiny_config_kwargs(),
            "num_hidden_layers": 6,
            "num_nextn_predict_layers": 1,
        },
        index_topk_freq=4,
        index_skip_topk_offset=3,
        indexer_types=_glm52_indexer_types(num_layers=6),
    )
    ps = ParallelState()
    model = _make_glm5_model(cfg, ps=ps, mtp_enable=True)
    state = model.state_dict()

    exported = dict(export_hf_weights(model, cfg, ps))
    assert "model.layers.2.self_attn.indexer.wq_b.weight" in exported
    assert "model.layers.3.self_attn.indexer.wq_b.weight" not in exported
    assert "model.layers.6.self_attn.indexer.wq_b.weight" in exported

    save_safetensors(exported, str(tmp_path))
    loaded = _make_glm5_model(cfg, ps=ps, mtp_enable=True)
    load_hf_weights(loaded, str(tmp_path), cfg, ps)
    loaded_state = loaded.state_dict()

    assert torch.equal(
        loaded_state["layers.2.self_attention.self_attention.indexer.wq_b.weight"],
        state["layers.2.self_attention.self_attention.indexer.wq_b.weight"],
    )
    assert (
        "layers.3.self_attention.self_attention.indexer.wq_b.weight" not in loaded_state
    )
    assert torch.equal(
        loaded_state[
            "mtp.layers.0.transformer_layer.self_attention.self_attention.indexer.wq_b.weight"
        ],
        state[
            "mtp.layers.0.transformer_layer.self_attention.self_attention.indexer.wq_b.weight"
        ],
    )


def test_glm5_router_modules_use_current_names_and_bias_buffers():
    import torch
    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.model.glm5.lite.model import Glm5SigmoidTopKRouter

    model = _make_glm5_model(
        Glm5Config(**_tiny_config_kwargs(), num_nextn_predict_layers=1), mtp_enable=True
    )
    routers = [
        module
        for module in model.modules()
        if isinstance(module, Glm5SigmoidTopKRouter)
    ]
    assert len(routers) == 2
    for router in routers:
        assert hasattr(router, "gate")
        assert hasattr(router, "expert_bias")
        assert torch.isfinite(router.gate.weight).all()
        assert torch.equal(router.expert_bias, torch.zeros_like(router.expert_bias))


def test_glm5_protocol_allows_cp_only_parallel_scope():
    import pytest
    from megatron.lite.model.glm5.lite.protocol import _validate_parallel_scope
    from megatron.lite.runtime.contracts import ParallelConfig

    # CP-only as well as PP/VPP/EP are supported and must validate cleanly.
    _validate_parallel_scope(ParallelConfig(tp=1, ep=1, etp=1, cp=2, pp=1, vpp=1))
    _validate_parallel_scope(ParallelConfig(tp=1, ep=1, etp=1, cp=1, pp=2, vpp=2))
    # GLM-5 native DSA attention rejects tensor / expert-tensor parallelism.
    with pytest.raises(NotImplementedError):
        _validate_parallel_scope(ParallelConfig(tp=2, ep=1, etp=1, cp=1, pp=1, vpp=1))
    with pytest.raises(NotImplementedError):
        _validate_parallel_scope(ParallelConfig(tp=1, ep=1, etp=2, cp=1, pp=1, vpp=1))


def test_glm52_protocol_rejects_attention_replay_before_parallel_init(
    transformer_engine_import_stub, monkeypatch
):
    import pytest

    transformer_engine_import_stub()
    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.model.glm5.lite import protocol

    cfg = Glm5Config(**_tiny_config_kwargs(), indexer_types=["full", "shared"])

    def unexpected_parallel_init(*args, **kwargs):
        raise AssertionError(
            "IndexShare activation replay must fail before init_parallel"
        )

    monkeypatch.setattr(protocol, "init_parallel", unexpected_parallel_init)
    for impl_cfg in (
        protocol.ImplConfig(optimizer=None, recompute=["core_attn"]),
        protocol.ImplConfig(optimizer=None, offload=["dsa"]),
    ):
        with pytest.raises(ValueError, match="group-aware"):
            protocol.build_model(cfg, impl_cfg=impl_cfg)


def test_glm5_impl_config_accepts_runtime_mtp_fields():
    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.model.glm5.lite.protocol import ImplConfig

    cfg = Glm5Config(**_tiny_config_kwargs(), num_nextn_predict_layers=1)

    assert ImplConfig(mtp_enable=False, mtp_enable_train=False).mtp_enable is False
    assert ImplConfig(mtp_enable=True, mtp_enable_train=True).mtp_enable_train is True
    assert cfg.num_nextn_predict_layers == 1


def test_glm5_protocol_uses_mlite_optimizer_api():
    from megatron.lite.model.glm5.lite.protocol import ImplConfig

    protocol_path = (
        Path(__file__).resolve().parents[3]
        / "megatron"
        / "lite"
        / "model"
        / "glm5"
        / "lite"
        / "protocol.py"
    )
    protocol_text = protocol_path.read_text()

    assert ImplConfig().optimizer == "dist_opt"
    assert "build_dist_opt_training_optimizer" in protocol_text


def test_glm5_lite_tiny_forward_backward(monkeypatch):
    import pytest
    import torch
    from megatron.lite.model.glm5.config import Glm5Config
    from megatron.lite.primitive.modules.attention import dsa

    if not torch.cuda.is_available():
        pytest.skip("GLM-5 native model requires CUDA (Transformer Engine RMSNorm)")
    device = torch.device("cuda")

    def fake_fused_indexer_sparse_attn(
        query,
        kv_full,
        attn_sink,
        window_idxs,
        q_indexer,
        k_indexer,
        weights,
        indexer_topk,
        ratio,
        softmax_scale,
        indexer_softmax_scale=1.0,
        loss_coeff=0.0,
        sparse_loss=False,
        kv_offset=0,
        calculate_per_token_loss=False,
        value_dim=None,
    ):
        del (
            kv_full,
            attn_sink,
            window_idxs,
            q_indexer,
            k_indexer,
            weights,
            indexer_topk,
            ratio,
            softmax_scale,
            indexer_softmax_scale,
            loss_coeff,
            sparse_loss,
            kv_offset,
            calculate_per_token_loss,
        )
        return query.new_zeros(
            query.shape[0], query.shape[1], query.shape[2] * value_dim
        ), torch.zeros((), device=query.device, dtype=torch.float32)

    monkeypatch.setattr(
        dsa._dsa_kernels, "fused_indexer_sparse_attn", fake_fused_indexer_sparse_attn
    )

    torch.manual_seed(1234)
    model = _make_glm5_model(Glm5Config(**_tiny_config_kwargs())).to(
        device=device, dtype=torch.bfloat16
    )
    input_ids = torch.randint(0, model.config.vocab_size, (2, 5), device=device)
    labels = torch.randint(0, model.config.vocab_size, (2, 5), device=device)

    output = model(input_ids=input_ids, labels=labels)

    # hidden_states stays in (seq, batch, hidden) layout; logits are transposed
    # back to (batch, seq, *) inside forward.
    assert output["hidden_states"].shape == (5, 2, model.config.hidden_size)
    assert output["loss"].ndim == 0
    output["loss"].backward()
    grad_norm = sum(
        param.grad.detach().float().norm()
        for param in model.parameters()
        if param.grad is not None
    )
    assert torch.isfinite(grad_norm)

    mtp_model = _make_glm5_model(
        Glm5Config(**_tiny_config_kwargs(), num_nextn_predict_layers=1),
        mtp_enable=True,
        mtp_enable_train=True,
    ).to(device=device, dtype=torch.bfloat16)
    # Inference path (no labels) exposes the per-MTP-head logits.
    mtp_infer = mtp_model(input_ids=input_ids)
    assert len(mtp_infer["mtp_hidden_states"]) == 1
    assert len(mtp_infer["mtp_logits"]) == 1
    assert mtp_infer["mtp_hidden_states"][0].shape == (
        5,
        2,
        mtp_model.config.hidden_size,
    )
    assert mtp_infer["mtp_logits"][0].shape == (2, 5, mtp_model.config.vocab_size)

    # Training path (with labels) returns the MTP loss instead of logits.
    mtp_output = mtp_model(
        input_ids=input_ids,
        labels=labels,
        loss_mask=torch.ones_like(labels, dtype=torch.float32),
    )

    assert len(mtp_output["mtp_hidden_states"]) == 1
    assert mtp_output["mtp_hidden_states"][0].shape == (
        5,
        2,
        mtp_model.config.hidden_size,
    )
    assert mtp_output["mtp_loss"].ndim == 0
    mtp_output["loss"].backward()
    mtp_grad_norm = sum(
        param.grad.detach().float().norm()
        for param in mtp_model.parameters()
        if param.grad is not None
    )
    assert torch.isfinite(mtp_grad_norm)
