# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime
from megatron.lite.runtime.contracts.handle import ModelHandle

pytestmark = pytest.mark.mlite


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def _step(
    model: nn.Module, optimizer: torch.optim.Optimizer, x: torch.Tensor, y: torch.Tensor
):
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    optimizer.step()
    return loss.detach()


def _clone_model_and_optimizer(model: nn.Module):
    clone = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(clone.parameters(), lr=1.0e-3, weight_decay=0.0)
    return clone, optimizer


def _assert_model_close(lhs: nn.Module, rhs: nn.Module):
    for (lhs_name, lhs_param), (rhs_name, rhs_param) in zip(
        lhs.named_parameters(), rhs.named_parameters(), strict=True
    ):
        assert lhs_name == rhs_name
        torch.testing.assert_close(lhs_param, rhs_param, atol=0.0, rtol=0.0)


def test_runtime_checkpoint_load_matches_uninterrupted_training(tmp_path):
    torch.manual_seed(2029)
    base = TinyMLP()
    ckpt_model, ckpt_optimizer = _clone_model_and_optimizer(base)
    direct_model, direct_optimizer = _clone_model_and_optimizer(base)
    loaded_model, loaded_optimizer = _clone_model_and_optimizer(base)
    x0, y0 = torch.randn(3, 4), torch.randn(3, 2)
    x1, y1 = torch.randn(3, 4), torch.randn(3, 2)

    _step(ckpt_model, ckpt_optimizer, x0, y0)
    _step(direct_model, direct_optimizer, x0, y0)

    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)
    ckpt_handle = ModelHandle(
        model=ckpt_model,
        optimizer=ckpt_optimizer,
        _extras={"model_chunks": [ckpt_model]},
    )
    runtime.save_checkpoint(ckpt_handle, str(tmp_path), step=1, use_dcp=False)

    loaded_handle = ModelHandle(
        model=loaded_model,
        optimizer=loaded_optimizer,
        _extras={"model_chunks": [loaded_model]},
    )
    assert runtime.load_checkpoint(loaded_handle, str(tmp_path), use_dcp=False) == 1

    _step(direct_model, direct_optimizer, x1, y1)
    _step(loaded_model, loaded_optimizer, x1, y1)

    _assert_model_close(direct_model, loaded_model)


class DistOptLike:
    """Small optimizer wrapper with the same checkpoint contract as dist_opt."""

    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        self.load_calls = 0
        self.parameter_save_calls = 0
        self.parameter_load_calls = 0

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def step(self):
        self.optimizer.step()
        return True, 0.0, 0

    def state_dict(self):
        state = self.optimizer.state_dict()
        state["dist_opt_like_marker"] = {"load_calls": self.load_calls}
        return state

    def load_state_dict(self, state):
        marker = state.pop("dist_opt_like_marker")
        self.load_calls = int(marker["load_calls"]) + 1
        self.optimizer.load_state_dict(state)

    def save_parameter_state(self, filename: str):
        self.parameter_save_calls += 1
        torch.save({"parameter_save_calls": self.parameter_save_calls}, filename)

    def load_parameter_state(
        self, filename: str, *, update_legacy_format: bool = False
    ):
        state = torch.load(filename)
        self.parameter_load_calls = int(state["parameter_save_calls"])


def test_runtime_checkpoint_uses_optimizer_state_dict_contract(tmp_path):
    torch.manual_seed(2030)
    model = TinyMLP()
    optimizer = DistOptLike(torch.optim.AdamW(model.parameters(), lr=1.0e-3))
    x, y = torch.randn(3, 4), torch.randn(3, 2)
    optimizer.zero_grad()
    torch.nn.functional.mse_loss(model(x), y).backward()
    optimizer.step()

    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)
    runtime.save_checkpoint(
        ModelHandle(
            model=model, optimizer=optimizer, _extras={"model_chunks": [model]}
        ),
        str(tmp_path),
        step=7,
        use_dcp=False,
    )

    loaded_model = TinyMLP()
    loaded_optimizer = DistOptLike(
        torch.optim.AdamW(loaded_model.parameters(), lr=1.0e-3)
    )
    loaded_handle = ModelHandle(
        model=loaded_model,
        optimizer=loaded_optimizer,
        _extras={"model_chunks": [loaded_model]},
    )

    assert runtime.load_checkpoint(loaded_handle, str(tmp_path), use_dcp=False) == 7
    assert loaded_optimizer.load_calls == 1
    assert loaded_optimizer.parameter_load_calls == 1
    assert (tmp_path / "training_state.optimizer_parameter_state.pt").exists()
    _assert_model_close(model, loaded_model)


def test_gather_pipeline_state_dict_collects_later_stage_buffers(monkeypatch):
    from types import SimpleNamespace

    from megatron.lite.primitive.ckpt.hf_weights import gather_pipeline_state_dict

    group = object()
    ps = SimpleNamespace(pp_size=2, pp_group=group)
    local = {"layers.0.router_bias": torch.tensor([1.0])}
    later = {"layers.1.router_bias": torch.tensor([2.0])}

    def fake_all_gather_object(output, value, *, group):
        assert value is local
        assert group is ps.pp_group
        output[:] = [local, later]

    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather_object)
    gathered = gather_pipeline_state_dict(local, ps)
    assert gathered.keys() == local.keys() | later.keys()
    assert torch.equal(gathered["layers.1.router_bias"], later["layers.1.router_bias"])


def test_distributed_error_consensus_uses_explicit_participating_group(monkeypatch):
    from megatron.lite.primitive.ckpt import hf_weights

    participating_group = object()
    events: list[str] = []

    monkeypatch.setattr(hf_weights.dist, "is_initialized", lambda: True)

    def fake_get_backend(group):
        assert group is participating_group
        return "gloo"

    def fake_all_reduce(failed, *, group):
        assert group is participating_group
        assert failed.item() == 1
        events.append("all_reduce")

    def fake_get_world_size(group):
        assert group is participating_group
        return 2

    def fake_all_gather_object(messages, local_error, *, group):
        assert group is participating_group
        assert local_error == "local failure"
        messages[:] = ["local failure", None]
        events.append("all_gather_object")

    monkeypatch.setattr(hf_weights.dist, "get_backend", fake_get_backend)
    monkeypatch.setattr(hf_weights.dist, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(hf_weights.dist, "get_world_size", fake_get_world_size)
    monkeypatch.setattr(hf_weights.dist, "all_gather_object", fake_all_gather_object)

    with pytest.raises(RuntimeError, match=r"^test export: local failure$"):
        hf_weights._distributed_raise_if_error(
            "local failure",
            context="test export",
            participating_group=participating_group,
        )

    assert events == ["all_reduce", "all_gather_object"]


def test_pipeline_buffer_error_consensus_defaults_to_pp_group(monkeypatch):
    from types import SimpleNamespace

    from megatron.lite.primitive.ckpt import hf_weights

    pp_group = object()
    ps = SimpleNamespace(pp_size=2, pp_group=pp_group)
    local = {"layers.0.router_bias": torch.tensor([1.0])}
    later = {"layers.1.router_bias": torch.tensor([2.0])}
    consensus_groups = []

    def fake_all_gather_object(output, value, *, group):
        assert value is local
        assert group is pp_group
        output[:] = [local, later]

    def fake_consensus(
        local_error, *, context, error_type=RuntimeError, participating_group=None
    ):
        assert local_error is None
        assert context == "PP buffer export failed"
        assert error_type is AssertionError
        consensus_groups.append(participating_group)

    monkeypatch.setattr(hf_weights.dist, "all_gather_object", fake_all_gather_object)
    monkeypatch.setattr(hf_weights, "_distributed_raise_if_error", fake_consensus)

    gathered = hf_weights.gather_pipeline_state_dict(local, ps)

    assert gathered.keys() == local.keys() | later.keys()
    assert consensus_groups == [pp_group]


def test_pipeline_buffer_export_rejects_missing_stage_state(monkeypatch):
    from types import SimpleNamespace

    from megatron.lite.primitive.ckpt.hf_weights import gather_pipeline_state_dict

    ps = SimpleNamespace(pp_size=2, pp_group=object())
    local = {"layers.0.router_bias": torch.tensor([1.0])}

    def fake_all_gather_object(output, value, *, group):
        assert value is local
        assert group is ps.pp_group
        output[:] = [local, None]

    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather_object)

    with pytest.raises(
        AssertionError,
        match=(
            r"^PP buffer export failed: pipeline export buffer state missing "
            r"for stage 1$"
        ),
    ):
        gather_pipeline_state_dict(local, ps)


def test_pipeline_parameter_error_consensus_defaults_to_pp_group(monkeypatch):
    from types import SimpleNamespace

    from megatron.lite.primitive.ckpt import hf_weights

    class WeightSpec:
        num_experts = 0

        @staticmethod
        def is_expert(_name):
            return False

        @staticmethod
        def tp_spec(_name):
            return None

        @staticmethod
        def native_to_hf(name, tensor):
            return [(name, tensor)]

    model = nn.Linear(2, 2, bias=False)
    pp_group = object()
    ps = SimpleNamespace(pp_size=2, pp_group=pp_group, tp_size=1)
    consensus_groups = []

    def fake_all_gather_object(output, local_state, *, group):
        assert group is pp_group
        output[:] = [local_state, {"remote.weight": torch.ones(1)}]

    def fake_consensus(
        local_error, *, context, error_type=RuntimeError, participating_group=None
    ):
        assert local_error is None
        assert error_type is AssertionError
        consensus_groups.append((context, participating_group))

    monkeypatch.setattr(hf_weights.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(hf_weights.dist, "all_gather_object", fake_all_gather_object)
    monkeypatch.setattr(hf_weights, "_distributed_raise_if_error", fake_consensus)

    exported = dict(hf_weights.export_hf_weights(model, WeightSpec(), ps))

    assert exported.keys() == {"weight", "remote.weight"}
    assert consensus_groups == [
        ("PP parameter export failed", pp_group),
        ("PP MTP embedding export validation failed", pp_group),
    ]


def test_pipeline_parameter_export_rejects_local_vpp_name_collision(monkeypatch):
    from types import SimpleNamespace

    from megatron.lite.primitive.ckpt import hf_weights

    class WeightSpec:
        num_experts = 0

        @staticmethod
        def is_expert(_name):
            return False

        @staticmethod
        def tp_spec(_name):
            return None

        @staticmethod
        def native_to_hf(name, tensor):
            return [(name, tensor)]

    first_chunk = nn.Linear(2, 2, bias=False)
    second_chunk = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        first_chunk.weight.fill_(1.0)
        second_chunk.weight.fill_(2.0)

    pp_group = object()
    ps = SimpleNamespace(pp_size=2, pp_group=pp_group, tp_size=1)

    def fake_all_gather_object(output, local_state, *, group):
        assert group is pp_group
        # The first chunk remains canonical while the duplicate is reported;
        # the old direct assignment silently replaced it with the second chunk.
        torch.testing.assert_close(local_state["weight"], torch.ones(2, 2))
        output[:] = [local_state, {"remote.weight": torch.ones(1)}]

    monkeypatch.setattr(hf_weights.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(hf_weights.dist, "all_gather_object", fake_all_gather_object)

    with pytest.raises(
        AssertionError,
        match=(
            r"^PP parameter export failed: pipeline export parameter collision "
            r"for weight$"
        ),
    ):
        list(
            hf_weights.export_hf_weights([first_chunk, second_chunk], WeightSpec(), ps)
        )


def test_pipeline_parameter_export_rejects_missing_stage_state(monkeypatch):
    from types import SimpleNamespace

    from megatron.lite.primitive.ckpt import hf_weights

    class WeightSpec:
        num_experts = 0

        @staticmethod
        def is_expert(_name):
            return False

        @staticmethod
        def tp_spec(_name):
            return None

        @staticmethod
        def native_to_hf(name, tensor):
            return [(name, tensor)]

    model = nn.Linear(2, 2, bias=False)
    ps = SimpleNamespace(pp_size=2, pp_group=object(), tp_size=1)

    def fake_all_gather_object(output, local_state, *, group):
        assert group is ps.pp_group
        output[:] = [local_state, None]

    monkeypatch.setattr(hf_weights.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(hf_weights.dist, "all_gather_object", fake_all_gather_object)

    with pytest.raises(
        AssertionError,
        match=(
            r"^PP parameter export failed: pipeline export parameter state missing "
            r"for stage 1$"
        ),
    ):
        list(hf_weights.export_hf_weights(model, WeightSpec(), ps))


@pytest.mark.parametrize("expert", [False, True], ids=["dense", "batched-expert"])
def test_non_pipeline_export_rejects_local_vpp_name_collision(expert):
    from types import SimpleNamespace

    from megatron.lite.primitive.ckpt import hf_weights

    class WeightSpec:
        num_experts = 1

        @staticmethod
        def is_expert(_name):
            return expert

        @staticmethod
        def tp_spec(_name):
            return None

        @staticmethod
        def native_to_hf(name, tensor):
            return [(name, tensor)]

    ps = SimpleNamespace(pp_size=1, tp_size=1)
    chunks = [nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False)]

    with pytest.raises(
        AssertionError,
        match=r"^local export parameter collision for weight$",
    ):
        list(hf_weights.export_hf_weights(chunks, WeightSpec(), ps))


def test_named_persistent_buffers_excludes_runtime_caches():
    from megatron.lite.primitive.ckpt.hf_weights import named_persistent_buffers

    model = nn.Module()
    model.register_buffer("root_cache", torch.zeros(1), persistent=False)
    model.child = nn.Module()
    model.child.register_buffer("router_bias", torch.ones(2), persistent=True)
    model.child.register_buffer("workspace", torch.zeros(3), persistent=False)

    buffers = dict(named_persistent_buffers(model))

    assert buffers.keys() == {"child.router_bias"}
    assert buffers["child.router_bias"] is model.child.router_bias


def test_resolve_param_name_only_accepts_unique_wrapper_suffix():
    from megatron.lite.primitive.ckpt.hf_weights import _resolve_param_name

    state = {
        "layers.0.weight": object(),
        "module.layers.1.weight": object(),
        "prefix.layers.2.weight.suffix": object(),
    }

    assert _resolve_param_name("layers.0.weight", state) == "layers.0.weight"
    assert _resolve_param_name("layers.1.weight", state) == "module.layers.1.weight"
    assert _resolve_param_name("layers.2.weight", state) is None


def test_resolve_param_name_rejects_ambiguous_wrapper_suffixes():
    from megatron.lite.primitive.ckpt.hf_weights import _resolve_param_name

    state = {
        "module.layers.0.weight": object(),
        "_orig_mod.layers.0.weight": object(),
    }

    with pytest.raises(
        ValueError,
        match=(
            r"^ambiguous wrapped parameter match for 'layers\.0\.weight': "
            r"\['_orig_mod\.layers\.0\.weight', 'module\.layers\.0\.weight'\]$"
        ),
    ):
        _resolve_param_name("layers.0.weight", state)


def test_load_hf_weights_rejects_broadcastable_shape_mismatch(monkeypatch):
    from types import SimpleNamespace

    from megatron.lite.primitive.ckpt import hf_weights
    import megatron.lite.primitive.parallel as parallel

    class Reader:
        def __init__(self, _path):
            pass

        @staticmethod
        def get_tensor(name):
            assert name == "hf.weight"
            return torch.ones(1, 2)

    class WeightSpec:
        @staticmethod
        def weight_map():
            return {"weight": ["hf.weight"]}

        @staticmethod
        def expert_global_id(_name):
            return None

        @staticmethod
        def hf_to_native(_name, tensors):
            return tensors[0]

        @staticmethod
        def tp_spec(_name):
            return None

    model = nn.Linear(2, 2, bias=False)
    ps = SimpleNamespace(ep_size=1, ep_rank=0)
    monkeypatch.setattr(hf_weights, "SafeTensorReader", Reader)
    monkeypatch.setitem(parallel.__dict__, "pad_vocab_for_tp", lambda size, _tp: size)

    with pytest.raises(
        ValueError,
        match=(
            r"^HF load shape mismatch for weight: "
            r"checkpoint=\(1, 2\), model=\(2, 2\)$"
        ),
    ):
        hf_weights.load_hf_weights(model, "/unused", WeightSpec(), ps)


def test_to_global_layer_name_never_remaps_mtp_namespace():
    from megatron.lite.primitive.ckpt.hf_weights import to_global_layer_name

    layer_map = {0: 8, 1: 9}
    assert to_global_layer_name("layers.0.attn.weight", layer_map) == (
        "layers.8.attn.weight"
    )
    assert to_global_layer_name("mtp.layers.0.attn.weight", layer_map) == (
        "mtp.layers.0.attn.weight"
    )


def test_gather_pipeline_state_dict_rejects_conflicting_stage_names(monkeypatch):
    from types import SimpleNamespace

    from megatron.lite.primitive.ckpt.hf_weights import gather_pipeline_state_dict

    ps = SimpleNamespace(pp_size=2, pp_group=object())
    local = {"layers.0.router_bias": torch.tensor([1.0])}

    def fake_all_gather_object(output, _value, *, group):
        assert group is ps.pp_group
        output[:] = [local, {"layers.0.router_bias": torch.tensor([9.0])}]

    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather_object)
    with pytest.raises(
        AssertionError,
        match=(
            r"^PP buffer export failed: pipeline export buffer collision for "
            r"layers\.0\.router_bias$"
        ),
    ):
        gather_pipeline_state_dict(local, ps)


def test_gather_pipeline_state_dict_rejects_identical_zero_stage_names(monkeypatch):
    from types import SimpleNamespace

    from megatron.lite.primitive.ckpt.hf_weights import gather_pipeline_state_dict

    ps = SimpleNamespace(pp_size=2, pp_group=object())
    local = {"layers.0.router_bias": torch.zeros(2)}

    def fake_all_gather_object(output, _value, *, group):
        assert group is ps.pp_group
        output[:] = [local, {"layers.0.router_bias": torch.zeros(2)}]

    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather_object)
    with pytest.raises(
        AssertionError,
        match=(
            r"^PP buffer export failed: pipeline export buffer collision for "
            r"layers\.0\.router_bias$"
        ),
    ):
        gather_pipeline_state_dict(local, ps)


@pytest.mark.parametrize("rank", [0, 1], ids=["failing-lane", "peer-lane"])
def test_materialize_hf_weights_propagates_generator_error_to_world(monkeypatch, rank):
    from megatron.lite.primitive.ckpt import hf_weights

    shared_error = "ValueError: lane 0 generator exploded"

    monkeypatch.setattr(hf_weights.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(hf_weights.dist, "get_backend", lambda: "gloo")
    monkeypatch.setattr(hf_weights.dist, "get_world_size", lambda: 2)

    def fake_all_reduce(failed, *, group):
        assert group is hf_weights.dist.group.WORLD
        assert failed.device.type == "cpu"
        assert failed.item() == int(rank == 0)
        failed.fill_(1)

    def fake_all_gather_object(messages, local_error, *, group):
        assert group is hf_weights.dist.group.WORLD
        assert local_error == (shared_error if rank == 0 else None)
        messages[:] = [shared_error, None]

    monkeypatch.setattr(hf_weights.dist, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(hf_weights.dist, "all_gather_object", fake_all_gather_object)

    def weights():
        if rank == 0:
            raise ValueError("lane 0 generator exploded")
        yield "model.weight", torch.ones(1)

    with pytest.raises(
        RuntimeError,
        match=(
            r"^HF weight materialization failed: ValueError: "
            r"lane 0 generator exploded$"
        ),
    ):
        hf_weights.materialize_hf_weights_distributed(weights())


def test_materialize_hf_weights_rejects_duplicate_keys(monkeypatch):
    from megatron.lite.primitive.ckpt import hf_weights

    monkeypatch.setattr(hf_weights.dist, "is_initialized", lambda: False)
    weights = [
        ("model.weight", torch.tensor([1.0])),
        ("model.weight", torch.tensor([2.0])),
    ]

    with pytest.raises(
        RuntimeError,
        match=(
            r"^HF weight materialization failed: AssertionError: "
            r"duplicate HF export keys: \['model\.weight'\]$"
        ),
    ):
        hf_weights.materialize_hf_weights_distributed(iter(weights))


def test_save_hf_weight_pairs_rejects_empty_rank0_export(monkeypatch, tmp_path):
    from megatron.lite.primitive.ckpt import hf_weights

    monkeypatch.setattr(hf_weights.dist, "is_initialized", lambda: False)

    def unexpected_save(*_args, **_kwargs):
        raise AssertionError("an empty export must not create a safetensors file")

    monkeypatch.setattr(hf_weights, "save_safetensors", unexpected_save)

    with pytest.raises(
        RuntimeError,
        match=(
            r"^HF safetensors write failed: ValueError: "
            r"rank 0 materialized no HF weights$"
        ),
    ):
        hf_weights.save_hf_weight_pairs_distributed(iter(()), str(tmp_path))


@pytest.mark.parametrize("rank", [0, 1], ids=["writer-lane", "peer-lane"])
def test_save_hf_weight_pairs_propagates_rank0_write_error_before_barrier(
    monkeypatch, tmp_path, rank
):
    from megatron.lite.primitive.ckpt import hf_weights

    events = []
    write_error = "OSError: disk full"
    consensus_round = 0

    monkeypatch.setattr(hf_weights.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(hf_weights.dist, "get_backend", lambda: "gloo")
    monkeypatch.setattr(hf_weights.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(hf_weights.dist, "get_rank", lambda: rank)

    def fake_all_reduce(failed, *, group):
        nonlocal consensus_round
        assert group is hf_weights.dist.group.WORLD
        assert failed.device.type == "cpu"
        consensus_round += 1
        events.append(f"all_reduce_{consensus_round}")
        if consensus_round == 1:
            assert failed.item() == 0
            return
        assert consensus_round == 2
        assert failed.item() == int(rank == 0)
        failed.fill_(1)

    def fake_all_gather_object(messages, local_error, *, group):
        assert group is hf_weights.dist.group.WORLD
        assert consensus_round == 2
        assert local_error == (write_error if rank == 0 else None)
        events.append("all_gather_error")
        messages[:] = [write_error, None]

    def fake_save(_out, _path):
        assert rank == 0
        events.append("write")
        raise OSError("disk full")

    def fake_barrier():
        events.append("barrier")

    monkeypatch.setattr(hf_weights.dist, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(hf_weights.dist, "all_gather_object", fake_all_gather_object)
    monkeypatch.setattr(hf_weights.dist, "barrier", fake_barrier)
    monkeypatch.setattr(hf_weights, "save_safetensors", fake_save)

    with pytest.raises(
        RuntimeError,
        match=r"^HF safetensors write failed: OSError: disk full$",
    ):
        hf_weights.save_hf_weight_pairs_distributed(
            iter([("model.weight", torch.ones(1))]), str(tmp_path)
        )

    assert events[-1] == "all_gather_error"
    assert "barrier" not in events
    assert events.count("write") == int(rank == 0)


def test_save_hf_weight_pairs_writes_before_success_barrier(monkeypatch, tmp_path):
    from megatron.lite.primitive.ckpt import hf_weights

    events = []

    monkeypatch.setattr(hf_weights.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(hf_weights.dist, "get_backend", lambda: "gloo")
    monkeypatch.setattr(hf_weights.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(hf_weights.dist, "get_rank", lambda: 0)

    def fake_all_reduce(failed, *, group):
        assert group is hf_weights.dist.group.WORLD
        assert failed.device.type == "cpu"
        assert failed.item() == 0
        events.append("all_reduce")

    def unexpected_all_gather_object(*_args, **_kwargs):
        raise AssertionError("a successful consensus must not gather error strings")

    def fake_save(out, path):
        assert path == str(tmp_path)
        assert list(out) == ["model.weight"]
        torch.testing.assert_close(out["model.weight"], torch.tensor([3.0]))
        events.append("write")

    def fake_barrier():
        events.append("barrier")

    monkeypatch.setattr(hf_weights.dist, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(
        hf_weights.dist, "all_gather_object", unexpected_all_gather_object
    )
    monkeypatch.setattr(hf_weights.dist, "barrier", fake_barrier)
    monkeypatch.setattr(hf_weights, "save_safetensors", fake_save)

    hf_weights.save_hf_weight_pairs_distributed(
        iter([("model.weight", torch.tensor([3.0]))]), str(tmp_path)
    )

    assert events == ["all_reduce", "write", "all_reduce", "barrier"]
