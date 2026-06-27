# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import copy
import random
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from megatron.lite.primitive.ckpt import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime
from megatron.lite.runtime.contracts.config import ParallelConfig
from megatron.lite.runtime.contracts.handle import ModelHandle


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


def _assert_model_state_unchanged(model: nn.Module, before: dict[str, torch.Tensor]):
    assert model.state_dict().keys() == before.keys()
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name], atol=0.0, rtol=0.0)


def test_runtime_local_checkpoint_load_matches_uninterrupted_training(tmp_path):
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
    runtime.save_checkpoint(
        ModelHandle(model=ckpt_model, optimizer=ckpt_optimizer),
        str(tmp_path),
        step=1,
        use_dcp=False,
    )

    assert (
        runtime.load_checkpoint(
            ModelHandle(model=loaded_model, optimizer=loaded_optimizer),
            str(tmp_path),
            use_dcp=False,
        )
        == 1
    )

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
        self.update_legacy_format = None

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

    def validate_state_dict(self, state):
        from megatron.lite.primitive.ckpt import dcp

        if not isinstance(state, dict) or set(state) != {
            "state",
            "param_groups",
            "dist_opt_like_marker",
        }:
            raise ValueError("invalid DistOptLike state schema")
        marker = state["dist_opt_like_marker"]
        if not isinstance(marker, dict) or type(marker.get("load_calls")) is not int:
            raise TypeError("invalid DistOptLike marker")
        inner_state = {
            key: value for key, value in state.items() if key != "dist_opt_like_marker"
        }
        dcp._validate_torch_optimizer_checkpoint_state(self.optimizer, inner_state)

    def save_parameter_state(self, filename: str):
        self.parameter_save_calls += 1
        torch.save({"parameter_save_calls": self.parameter_save_calls}, filename)

    def load_parameter_state(
        self, filename: str, *, update_legacy_format: bool = False
    ):
        state = torch.load(filename, weights_only=False)
        self.parameter_load_calls = int(state["parameter_save_calls"])
        self.update_legacy_format = update_legacy_format

    @staticmethod
    def validate_parameter_state(filename: str, *, update_legacy_format: bool = False):
        del update_legacy_format
        state = torch.load(filename, map_location="cpu", weights_only=False)
        if (
            not isinstance(state, dict)
            or set(state) != {"parameter_save_calls"}
            or type(state["parameter_save_calls"]) is not int
            or state["parameter_save_calls"] < 1
        ):
            raise ValueError("invalid DistOptLike parameter state")


def test_runtime_local_checkpoint_uses_optimizer_parameter_state_contract(tmp_path):
    torch.manual_seed(2030)
    model = TinyMLP()
    optimizer = DistOptLike(torch.optim.AdamW(model.parameters(), lr=1.0e-3))
    x, y = torch.randn(3, 4), torch.randn(3, 2)
    optimizer.zero_grad()
    torch.nn.functional.mse_loss(model(x), y).backward()
    optimizer.step()

    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)
    runtime.save_checkpoint(
        ModelHandle(model=model, optimizer=optimizer),
        str(tmp_path),
        step=7,
        use_dcp=False,
    )

    loaded_model = TinyMLP()
    loaded_optimizer = DistOptLike(
        torch.optim.AdamW(loaded_model.parameters(), lr=1.0e-3)
    )

    assert (
        runtime.load_checkpoint(
            ModelHandle(model=loaded_model, optimizer=loaded_optimizer),
            str(tmp_path),
            update_legacy_format=True,
            use_dcp=False,
        )
        == 7
    )
    assert loaded_optimizer.load_calls == 1
    assert loaded_optimizer.parameter_load_calls == 1
    assert loaded_optimizer.update_legacy_format is True
    assert (tmp_path / "training_state.optimizer_parameter_state.pt").exists()
    _assert_model_close(model, loaded_model)


def test_runtime_local_checkpoint_restores_rng_state(tmp_path):
    model = TinyMLP()
    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)

    random.seed(2031)
    np.random.seed(2031)
    torch.manual_seed(2031)

    runtime.save_checkpoint(
        ModelHandle(model=model, optimizer=None), str(tmp_path), step=9, use_dcp=False
    )

    expected_python = random.random()
    expected_numpy = np.random.random(4)
    expected_torch = torch.rand(4)

    random.seed(9999)
    np.random.seed(9999)
    torch.manual_seed(9999)

    assert (
        runtime.load_checkpoint(
            ModelHandle(model=model, optimizer=None), str(tmp_path), use_dcp=False
        )
        == 9
    )
    assert random.random() == expected_python
    np.testing.assert_allclose(np.random.random(4), expected_numpy, atol=0.0, rtol=0.0)
    torch.testing.assert_close(torch.rand(4), expected_torch, atol=0.0, rtol=0.0)


def test_runtime_local_checkpoint_uses_rank_specific_files_when_distributed(tmp_path):
    model = TinyMLP()
    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)

    with (
        patch("megatron.lite.primitive.ckpt.dcp.dist.is_available", return_value=True),
        patch(
            "megatron.lite.primitive.ckpt.dcp.dist.is_initialized", return_value=True
        ),
        patch("megatron.lite.primitive.ckpt.dcp.dist.get_rank", return_value=3),
    ):
        runtime.save_checkpoint(
            ModelHandle(model=model, optimizer=None),
            str(tmp_path),
            step=11,
            use_dcp=False,
        )
        assert (tmp_path / "training_state_rank_00003.pt").exists()
        assert not (tmp_path / "training_state.pt").exists()
        assert (
            runtime.load_checkpoint(
                ModelHandle(model=model, optimizer=None), str(tmp_path), use_dcp=False
            )
            == 11
        )


def test_primitive_local_checkpoint_keeps_optimizer_checkpoints_local(tmp_path):
    model = TinyMLP()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)

    with patch("megatron.lite.primitive.ckpt.dcp.dcp.save") as dcp_save_mock:
        save_training_checkpoint(model, optimizer, 12, str(tmp_path), use_dcp=False)

    dcp_save_mock.assert_not_called()
    assert (tmp_path / "training_state.pt").exists()


@pytest.mark.parametrize(
    ("save_model", "save_optimizer"), ((False, True), (True, False), (False, False))
)
def test_local_checkpoint_save_rejects_partial_component_selection(
    tmp_path, save_model: bool, save_optimizer: bool
):
    model = TinyMLP()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)

    with pytest.raises(
        ValueError, match="Local-format checkpoints do not support partial"
    ):
        save_training_checkpoint(
            model,
            optimizer,
            12,
            str(tmp_path),
            use_dcp=False,
            save_model=save_model,
            save_optimizer=save_optimizer,
        )

    assert not (tmp_path / "training_state.pt").exists()


@pytest.mark.parametrize(
    ("load_model", "load_optimizer"), ((False, True), (True, False), (False, False))
)
def test_local_checkpoint_load_rejects_partial_component_selection_before_mutation(
    tmp_path, load_model: bool, load_optimizer: bool
):
    source = TinyMLP()
    source_optimizer = torch.optim.AdamW(source.parameters(), lr=1.0e-3)
    save_training_checkpoint(source, source_optimizer, 13, str(tmp_path), use_dcp=False)
    target = TinyMLP()
    target_optimizer = torch.optim.AdamW(target.parameters(), lr=1.0e-3)
    before = copy.deepcopy(target.state_dict())

    with pytest.raises(
        ValueError, match="Local-format checkpoints do not support partial"
    ):
        load_training_checkpoint(
            target,
            target_optimizer,
            str(tmp_path),
            use_dcp=False,
            load_model=load_model,
            load_optimizer=load_optimizer,
        )

    for name, tensor in target.state_dict().items():
        torch.testing.assert_close(tensor, before[name], atol=0.0, rtol=0.0)


def test_local_checkpoint_rejects_invalid_step_before_any_mutation(tmp_path):
    source = TinyMLP()
    save_training_checkpoint(source, None, 14, str(tmp_path), use_dcp=False)
    checkpoint_file = tmp_path / "training_state.pt"
    state = torch.load(checkpoint_file, weights_only=False)
    state["step"] = "14"
    torch.save(state, checkpoint_file)

    target = TinyMLP()
    before = copy.deepcopy(target.state_dict())
    rng_before = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="step must be a non-negative integer"):
        load_training_checkpoint(target, None, str(tmp_path), use_dcp=False)

    _assert_model_state_unchanged(target, before)
    torch.testing.assert_close(torch.get_rng_state(), rng_before, atol=0, rtol=0)


@pytest.mark.parametrize("corruption", ["missing", "unexpected", "shape", "dtype"])
def test_local_checkpoint_preflights_every_chunk_before_copy(tmp_path, corruption):
    source_chunks = [nn.Linear(3, 3), nn.Linear(3, 2)]
    save_training_checkpoint(
        source_chunks, None, 15, str(tmp_path), use_dcp=False, save_rng=False
    )
    checkpoint_file = tmp_path / "training_state.pt"
    state = torch.load(checkpoint_file, weights_only=False)
    second_chunk = state["model"][1]
    if corruption == "missing":
        second_chunk.pop("param.weight")
    elif corruption == "unexpected":
        second_chunk["param.removed"] = torch.zeros(1)
    elif corruption == "shape":
        second_chunk["param.weight"] = torch.zeros(1, 3)
    else:
        second_chunk["param.weight"] = second_chunk["param.weight"].double()
    torch.save(state, checkpoint_file)

    target_chunks = [nn.Linear(3, 3), nn.Linear(3, 2)]
    before = [copy.deepcopy(chunk.state_dict()) for chunk in target_chunks]
    with pytest.raises(RuntimeError, match="model chunk 1"):
        load_training_checkpoint(
            target_chunks, None, str(tmp_path), use_dcp=False, load_rng=False
        )

    for chunk, chunk_before in zip(target_chunks, before, strict=True):
        _assert_model_state_unchanged(chunk, chunk_before)


def test_local_checkpoint_rejects_optimizer_state_before_model_copy(tmp_path):
    source = TinyMLP()
    source_optimizer = torch.optim.AdamW(source.parameters(), lr=1.0e-3)
    _step(source, source_optimizer, torch.randn(2, 4), torch.randn(2, 2))
    save_training_checkpoint(source, source_optimizer, 16, str(tmp_path), use_dcp=False)
    checkpoint_file = tmp_path / "training_state.pt"
    state = torch.load(checkpoint_file, weights_only=False)
    state["optimizer"]["param_groups"][0]["lr"] = "corrupt"
    torch.save(state, checkpoint_file)

    target = TinyMLP()
    target_optimizer = torch.optim.AdamW(target.parameters(), lr=1.0e-3)
    before = copy.deepcopy(target.state_dict())
    with pytest.raises(RuntimeError, match="optimizer checkpoint is incompatible"):
        load_training_checkpoint(target, target_optimizer, str(tmp_path), use_dcp=False)

    _assert_model_state_unchanged(target, before)
    assert not target_optimizer.state


def test_local_checkpoint_rejects_incompatible_rng_before_model_copy(tmp_path):
    source = TinyMLP()
    save_training_checkpoint(source, None, 17, str(tmp_path), use_dcp=False)
    checkpoint_file = tmp_path / "training_state.pt"
    state = torch.load(checkpoint_file, weights_only=False)
    state["rng_state"]["torch_rng_state"] = torch.zeros(1, dtype=torch.uint8)
    torch.save(state, checkpoint_file)

    target = TinyMLP()
    before = copy.deepcopy(target.state_dict())
    rng_before = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="RNG checkpoint is incompatible"):
        load_training_checkpoint(target, None, str(tmp_path), use_dcp=False)

    _assert_model_state_unchanged(target, before)
    torch.testing.assert_close(torch.get_rng_state(), rng_before, atol=0, rtol=0)


def test_local_checkpoint_rejects_missing_requested_rng_before_model_copy(tmp_path):
    source = TinyMLP()
    save_training_checkpoint(
        source, None, 17, str(tmp_path), use_dcp=False, save_rng=False
    )
    target = TinyMLP()
    before = copy.deepcopy(target.state_dict())
    rng_before = torch.get_rng_state().clone()

    with pytest.raises(RuntimeError, match="RNG state .* is missing"):
        load_training_checkpoint(target, None, str(tmp_path), use_dcp=False)

    _assert_model_state_unchanged(target, before)
    torch.testing.assert_close(torch.get_rng_state(), rng_before, atol=0, rtol=0)


@pytest.mark.parametrize("saved_has_optimizer", [False, True])
def test_local_checkpoint_rejects_optimizer_presence_mismatch_before_model_copy(
    tmp_path, saved_has_optimizer
):
    source = TinyMLP()
    source_optimizer = (
        torch.optim.AdamW(source.parameters(), lr=1.0e-3)
        if saved_has_optimizer
        else None
    )
    save_training_checkpoint(source, source_optimizer, 18, str(tmp_path), use_dcp=False)
    target = TinyMLP()
    target_optimizer = (
        None
        if saved_has_optimizer
        else torch.optim.AdamW(target.parameters(), lr=1.0e-3)
    )
    before = copy.deepcopy(target.state_dict())

    with pytest.raises(RuntimeError, match="optimizer presence does not match"):
        load_training_checkpoint(target, target_optimizer, str(tmp_path), use_dcp=False)

    _assert_model_state_unchanged(target, before)


def test_local_checkpoint_requires_parameter_state_file_before_model_copy(tmp_path):
    source = TinyMLP()
    source_optimizer = DistOptLike(torch.optim.AdamW(source.parameters(), lr=1.0e-3))
    save_training_checkpoint(source, source_optimizer, 19, str(tmp_path), use_dcp=False)
    (tmp_path / "training_state.optimizer_parameter_state.pt").unlink()

    target = TinyMLP()
    target_optimizer = DistOptLike(torch.optim.AdamW(target.parameters(), lr=1.0e-3))
    before = copy.deepcopy(target.state_dict())
    with pytest.raises(
        RuntimeError,
        match="optimizer parameter-state preflight failed.*FileNotFoundError",
    ):
        load_training_checkpoint(target, target_optimizer, str(tmp_path), use_dcp=False)

    _assert_model_state_unchanged(target, before)
    assert target_optimizer.load_calls == 0
    assert target_optimizer.parameter_load_calls == 0


def test_local_checkpoint_rejects_corrupt_parameter_state_before_any_commit(tmp_path):
    source = TinyMLP()
    source_optimizer = DistOptLike(torch.optim.AdamW(source.parameters(), lr=1.0e-3))
    save_training_checkpoint(source, source_optimizer, 19, str(tmp_path), use_dcp=False)
    parameter_state_path = tmp_path / "training_state.optimizer_parameter_state.pt"
    torch.save({"wrong": 1}, parameter_state_path)

    target = TinyMLP()
    target_optimizer = DistOptLike(torch.optim.AdamW(target.parameters(), lr=1.0e-3))
    before = copy.deepcopy(target.state_dict())
    with pytest.raises(
        RuntimeError, match="optimizer parameter-state preflight failed"
    ):
        load_training_checkpoint(target, target_optimizer, str(tmp_path), use_dcp=False)

    _assert_model_state_unchanged(target, before)
    assert target_optimizer.load_calls == 0
    assert target_optimizer.parameter_load_calls == 0
    assert not list(tmp_path.glob(".*.mlite-load-*"))


def test_local_checkpoint_supports_mcore_parameter_state_schema_adapter(tmp_path):
    class MCoreParameterStateLike:
        def __init__(self):
            self.load_calls = 0
            self.parameter_load_calls = 0
            self.grad_scaler = None
            self.optimizer = SimpleNamespace(
                param_groups=[
                    {
                        "params": [],
                        "lr": 1.0e-3,
                        "wd_mult": 1.0,
                        "lr_mult": 1.0,
                        "is_expert_parallel": False,
                        "is_decoupled_lr": False,
                    }
                ]
            )

        @staticmethod
        def _parameter_state(fill: float):
            tensors = {
                name: torch.full((3,), fill, dtype=torch.float32)
                for name in ("param", "exp_avg", "exp_avg_sq")
            }
            tensors["numel_unpadded"] = 3
            return {
                "buckets_coalesced": True,
                0: {(torch.float32, torch.float32): tensors},
            }

        def state_dict(self):
            return {
                "optimizer": {
                    "param_groups": [
                        {
                            "lr": 1.0e-3,
                            "step": 4,
                            "wd_mult": 1.0,
                            "lr_mult": 1.0,
                            "is_expert_parallel": False,
                            "is_decoupled_lr": False,
                        }
                    ]
                }
            }

        def load_state_dict(self, state):
            assert "optimizer" in state
            self.load_calls += 1

        def save_parameter_state(self, filename):
            torch.save(self._parameter_state(7.0), filename)

        def get_parameter_state_dp_zero(self, *, empty_data=False):
            assert empty_data is True
            return self._parameter_state(0.0)

        def load_parameter_state(self, filename, *, update_legacy_format=False):
            assert update_legacy_format is False
            state = torch.load(filename, map_location="cpu", weights_only=False)
            assert state[0][(torch.float32, torch.float32)]["param"].shape == (3,)
            self.parameter_load_calls += 1

    source = TinyMLP()
    source_optimizer = MCoreParameterStateLike()
    save_training_checkpoint(source, source_optimizer, 20, str(tmp_path), use_dcp=False)
    target = TinyMLP()
    target_optimizer = MCoreParameterStateLike()

    assert (
        load_training_checkpoint(target, target_optimizer, str(tmp_path), use_dcp=False)
        == 20
    )
    assert target_optimizer.load_calls == 1
    assert target_optimizer.parameter_load_calls == 1
    assert not list(tmp_path.glob(".*.mlite-load-*"))


def test_local_parameter_state_preflight_binds_staged_inode_across_replace(tmp_path):
    from megatron.lite.primitive.ckpt import dcp

    model = TinyMLP()
    optimizer = DistOptLike(torch.optim.AdamW(model.parameters(), lr=1.0e-3))
    source = tmp_path / "parameter_state.pt"
    torch.save({"parameter_save_calls": 1}, source)
    load_path, staged_path, fingerprint = (
        dcp._preflight_local_optimizer_parameter_state(
            optimizer, source, update_legacy_format=False
        )
    )
    assert staged_path is not None
    replacement = tmp_path / "replacement.pt"
    torch.save({"parameter_save_calls": 99}, replacement)
    replacement.replace(source)

    try:
        dcp._revalidate_local_optimizer_parameter_state(staged_path, fingerprint)
        optimizer.load_parameter_state(str(load_path))
        assert optimizer.parameter_load_calls == 1
    finally:
        staged_path.unlink(missing_ok=True)


def test_local_checkpoint_requires_exact_parameter_state_capability_before_copy(
    tmp_path,
):
    source = TinyMLP()
    source_optimizer = torch.optim.AdamW(source.parameters(), lr=1.0e-3)
    save_training_checkpoint(source, source_optimizer, 20, str(tmp_path), use_dcp=False)

    target = TinyMLP()
    target_optimizer = DistOptLike(torch.optim.AdamW(target.parameters(), lr=1.0e-3))
    before = copy.deepcopy(target.state_dict())
    with pytest.raises(RuntimeError, match="parameter-state presence does not match"):
        load_training_checkpoint(target, target_optimizer, str(tmp_path), use_dcp=False)

    _assert_model_state_unchanged(target, before)


def test_local_checkpoint_requires_outer_optimizer_apply_contract_before_copy(tmp_path):
    class NoLoadOptimizer:
        pass

    source = TinyMLP()
    source_optimizer = torch.optim.AdamW(source.parameters(), lr=1.0e-3)
    save_training_checkpoint(source, source_optimizer, 21, str(tmp_path), use_dcp=False)
    target = TinyMLP()
    before = copy.deepcopy(target.state_dict())

    with pytest.raises(RuntimeError, match="does not provide load_state_dict"):
        load_training_checkpoint(
            target, NoLoadOptimizer(), str(tmp_path), use_dcp=False
        )

    _assert_model_state_unchanged(target, before)


def test_primitive_explicit_dcp_saves_optimizer_rank_sidecar(tmp_path):
    model = TinyMLP()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    parallel = ParallelConfig(tp=1, ep=1, pp=1, cp=1)

    with (
        patch(
            "megatron.lite.primitive.ckpt.dcp._build_meshes", return_value=(None, None)
        ),
        patch(
            "megatron.lite.primitive.ckpt.dcp.DTensor.from_local",
            side_effect=lambda tensor, *args, **kwargs: tensor,
        ),
        patch("megatron.lite.primitive.ckpt.dcp.dcp.save") as dcp_save_mock,
    ):
        save_training_checkpoint(
            model,
            optimizer,
            12,
            str(tmp_path),
            parallel,
            SimpleNamespace(pp_size=1, pp_rank=0),
            use_dcp=True,
        )

    dcp_save_mock.assert_called_once()
    assert (tmp_path / "step_12" / "optimizer_rank_0.pt").exists()


def test_runtime_dcp_checkpoint_threads_parallel_config_and_protocol_hooks(tmp_path):
    model = TinyMLP()
    parallel = ParallelConfig(tp=2, ep=1, pp=1, cp=1)
    ps = object()

    def placement_fn(name: str):
        return ["placement", name]

    def expert_classifier(name: str):
        return name.endswith("expert")

    proto = SimpleNamespace(
        PLACEMENT_FN=placement_fn, EXPERT_CLASSIFIER=expert_classifier
    )
    handle = ModelHandle(
        model=[model],
        optimizer=None,
        parallel_state=ps,
        config=SimpleNamespace(parallel=parallel),
        _extras={"model_chunks": [model], "protocol": proto},
    )
    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)

    with patch("megatron.lite.primitive.ckpt.save_training_checkpoint") as save_mock:
        runtime.save_checkpoint(handle, str(tmp_path), global_step=13, use_dcp=True)

    save_args = save_mock.call_args.args
    save_kwargs = save_mock.call_args.kwargs
    assert isinstance(save_args[0], nn.ModuleList)
    assert save_args[0][0] is model
    assert save_args[2] == 13
    assert save_args[3] == str(tmp_path)
    assert save_args[4] is parallel
    assert save_args[5] is ps
    assert save_kwargs["get_placements"] is placement_fn
    assert save_kwargs["is_expert"] is expert_classifier
    assert save_kwargs["use_dcp"] is True
    assert save_kwargs["save_rng"] is True

    with patch(
        "megatron.lite.primitive.ckpt.load_training_checkpoint", return_value=13
    ) as load_mock:
        assert runtime.load_checkpoint(handle, str(tmp_path), use_dcp=True) == 13

    load_args = load_mock.call_args.args
    load_kwargs = load_mock.call_args.kwargs
    assert isinstance(load_args[0], nn.ModuleList)
    assert load_args[0][0] is model
    assert load_args[3] is parallel
    assert load_args[4] is ps
    assert load_kwargs["get_placements"] is placement_fn
    assert load_kwargs["is_expert"] is expert_classifier
    assert load_kwargs["use_dcp"] is True
    assert load_kwargs["load_rng"] is True
