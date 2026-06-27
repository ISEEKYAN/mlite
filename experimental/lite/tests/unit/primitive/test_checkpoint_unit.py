# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import builtins
import copy
import datetime
import importlib
import json
import multiprocessing as mp
import random
import sys
import types

import numpy as np
import pytest
import torch
import torch.nn as nn
from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime
from megatron.lite.runtime.contracts.handle import ModelHandle

pytestmark = pytest.mark.mlite


def _gloo_extra_state_target_commit_failure_worker(rank: int, init_path: str) -> None:
    import torch.distributed as dist
    from megatron.lite.primitive.ckpt import dcp

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=20),
    )
    try:

        class Target:
            def __init__(self):
                self.value = rank + 10
                self.apply_calls = 0

            def snapshot(self):
                return self.value

            def apply(self, state):
                self.apply_calls += 1
                self.value = state["value"]
                if self.apply_calls == 2 and rank == 1:
                    raise RuntimeError("rank-1 injected scheduler commit failure")

            def restore(self, snapshot):
                self.value = snapshot

            def fingerprint(self):
                return self.value

        target = Target()
        targets = {"lr_scheduler.pt": target}
        values = {"lr_scheduler.pt": {"value": 99}}
        dcp._preflight_extra_state_targets(targets, values)
        assert target.value == rank + 10

        with pytest.raises(RuntimeError, match="runtime must be poisoned"):
            dcp._commit_extra_state_targets(targets, values)
        assert target.value == rank + 10

        restored: list[int | None] = [None, None]
        dist.all_gather_object(restored, target.value)
        assert restored == [10, 11]
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _checkpoint_transaction_race_worker(
    checkpoint_path: str, start_event, results
) -> None:
    from megatron.lite.primitive.ckpt import dcp

    start_event.wait(timeout=20)
    try:
        dcp._begin_checkpoint_transaction(
            checkpoint_path,
            step=77,
            save_model=True,
            save_optimizer=False,
            save_rng=False,
            payload_format="dcp",
            optimizer_storage="none",
        )
    except Exception as exc:
        results.put(("error", type(exc).__name__, str(exc)))
    else:
        results.put(("ok", None, None))


def _gloo_checkpoint_reservation_failure_worker(
    rank: int, init_path: str, checkpoint_path: str
) -> None:
    import torch.distributed as dist
    from megatron.lite.primitive.ckpt import dcp

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=20),
    )
    try:
        with pytest.raises(RuntimeError, match="atomic reservation failed"):
            dcp._begin_checkpoint_transaction(
                checkpoint_path,
                step=78,
                save_model=True,
                save_optimizer=False,
                save_rng=False,
                payload_format="dcp",
                optimizer_storage="none",
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_gloo_available(), reason="Gloo is unavailable"
)
def test_extra_state_target_rank_failure_gloo_rolls_back_without_hang(tmp_path):
    init_path = str(tmp_path / "gloo-extra-state-target-init")
    ctx = mp.get_context("spawn")
    processes = [
        ctx.Process(
            target=_gloo_extra_state_target_commit_failure_worker,
            args=(rank, init_path),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
    alive = [process for process in processes if process.is_alive()]
    for process in alive:
        process.terminate()
        process.join(timeout=5)
    assert not alive, "extra-state target commit consensus hung"
    assert [process.exitcode for process in processes] == [0, 0]


def test_hf_weight_mapping_import_is_independent_of_safetensors_io(monkeypatch):
    from megatron.lite.primitive.ckpt import hf_weights

    real_import = builtins.__import__

    def import_without_safetensors(
        name, globals=None, locals=None, fromlist=(), level=0
    ):
        if name == "safetensors" or name.startswith("safetensors."):
            raise ModuleNotFoundError("blocked optional dependency", name="safetensors")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_safetensors)
    with pytest.raises(
        ImportError, match="checkpoint I/O requires the optional 'safetensors' package"
    ):
        hf_weights._require_safetensors_io()


def test_checkpoint_resolver_ignores_incomplete_latest_transaction(tmp_path):
    from megatron.lite.primitive.ckpt import dcp

    step1 = tmp_path / "step_1"
    manifest1 = dcp._begin_checkpoint_transaction(
        str(step1),
        step=1,
        save_model=True,
        save_optimizer=False,
        save_rng=False,
        payload_format="dcp",
        optimizer_storage="none",
    )
    dcp._complete_checkpoint_transaction(str(step1), manifest1)
    step2 = tmp_path / "step_2"
    dcp._begin_checkpoint_transaction(
        str(step2),
        step=2,
        save_model=True,
        save_optimizer=False,
        save_rng=False,
        payload_format="dcp",
        optimizer_storage="none",
    )

    assert dcp._resolve_step_checkpoint_path(str(tmp_path)) == str(step1)
    with pytest.raises(
        RuntimeError,
        match="checkpoint completion manifest validation failed: RuntimeError: "
        "checkpoint status is 'incomplete'",
    ):
        dcp._validate_checkpoint_manifest(
            str(step2), load_model=True, load_optimizer=False, load_rng=False
        )


def test_same_step_retry_preserves_existing_complete_checkpoint_bytes(tmp_path):
    from megatron.lite.primitive.ckpt import dcp

    step = tmp_path / "step_2"
    manifest = dcp._begin_checkpoint_transaction(
        str(step),
        step=2,
        save_model=True,
        save_optimizer=False,
        save_rng=False,
        payload_format="dcp",
        optimizer_storage="none",
    )
    payload_path = step / "payload.distcp"
    payload_path.write_bytes(b"immutable-complete-payload")
    dcp._complete_checkpoint_transaction(str(step), manifest)
    manifest_path = step / dcp._CHECKPOINT_MANIFEST
    before = {
        entry.name: entry.read_bytes() for entry in step.iterdir() if entry.is_file()
    }

    with pytest.raises(
        RuntimeError,
        match="checkpoint destination is non-empty and will not be overwritten",
    ):
        dcp._begin_checkpoint_transaction(
            str(step),
            step=2,
            save_model=True,
            save_optimizer=False,
            save_rng=False,
            payload_format="dcp",
            optimizer_storage="none",
        )

    assert json.loads(manifest_path.read_text())["status"] == "complete"
    assert {
        entry.name: entry.read_bytes() for entry in step.iterdir() if entry.is_file()
    } == before


def test_checkpoint_transaction_atomically_reserves_one_same_step_writer(tmp_path):
    checkpoint_path = str(tmp_path / "step_77")
    ctx = mp.get_context("spawn")
    start_event = ctx.Event()
    results = ctx.Queue()
    processes = [
        ctx.Process(
            target=_checkpoint_transaction_race_worker,
            args=(checkpoint_path, start_event, results),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    start_event.set()
    for process in processes:
        process.join(timeout=30)
    alive = [process for process in processes if process.is_alive()]
    for process in alive:
        process.terminate()
        process.join(timeout=5)
    assert not alive, "same-step checkpoint reservation race hung"
    assert [process.exitcode for process in processes] == [0, 0]
    outcomes = [results.get(timeout=5) for _ in processes]
    assert [outcome[0] for outcome in outcomes].count("ok") == 1
    assert [outcome[0] for outcome in outcomes].count("error") == 1
    error = next(outcome[2] for outcome in outcomes if outcome[0] == "error")
    assert "atomically reserved" in error or "will not be overwritten" in error


@pytest.mark.parametrize("existing_kind", ["file", "malformed-manifest"])
@pytest.mark.skipif(
    not torch.distributed.is_gloo_available(), reason="Gloo is unavailable"
)
def test_checkpoint_atomic_reservation_rank0_failure_propagates_without_hang(
    tmp_path, existing_kind
):
    checkpoint_path = tmp_path / "step_78"
    if existing_kind == "file":
        checkpoint_path.write_text("not-a-directory")
    else:
        checkpoint_path.mkdir()
        (checkpoint_path / "mlite_checkpoint_manifest.json").write_text("{broken")
    init_path = str(tmp_path / f"gloo-reservation-{existing_kind}")
    ctx = mp.get_context("spawn")
    processes = [
        ctx.Process(
            target=_gloo_checkpoint_reservation_failure_worker,
            args=(rank, init_path, str(checkpoint_path)),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
    alive = [process for process in processes if process.is_alive()]
    for process in alive:
        process.terminate()
        process.join(timeout=5)
    assert not alive, f"{existing_kind} reservation failure propagation hung"
    assert [process.exitcode for process in processes] == [0, 0]


def test_legacy_dcp_checkpoint_requires_explicit_migration_opt_in(tmp_path):
    from megatron.lite.primitive.ckpt import dcp

    legacy = tmp_path / "step_4"
    legacy.mkdir()

    with pytest.raises(
        RuntimeError, match="Legacy DCP checkpoints are rejected by default"
    ):
        dcp._validate_checkpoint_manifest(
            str(legacy), load_model=True, load_optimizer=False, load_rng=False
        )

    assert (
        dcp._validate_checkpoint_manifest(
            str(legacy),
            load_model=True,
            load_optimizer=False,
            load_rng=False,
            allow_legacy_checkpoint=True,
        )
        is None
    )


def test_checkpoint_resolver_ignores_newer_legacy_directory_by_default(tmp_path):
    from megatron.lite.primitive.ckpt import dcp

    current = tmp_path / "step_3"
    manifest = dcp._begin_checkpoint_transaction(
        str(current),
        step=3,
        save_model=True,
        save_optimizer=False,
        save_rng=False,
        payload_format="dcp",
        optimizer_storage="none",
    )
    dcp._complete_checkpoint_transaction(str(current), manifest)
    legacy = tmp_path / "step_9"
    legacy.mkdir()

    assert dcp._resolve_step_checkpoint_path(str(tmp_path)) == str(current)
    assert dcp._resolve_step_checkpoint_path(
        str(tmp_path), allow_legacy_checkpoint=True
    ) == str(legacy)


def test_completed_checkpoint_manifest_requires_declared_rank_sidecars(tmp_path):
    from megatron.lite.primitive.ckpt import dcp

    step = tmp_path / "step_3"
    manifest = dcp._begin_checkpoint_transaction(
        str(step),
        step=3,
        save_model=True,
        save_optimizer=True,
        save_rng=True,
        payload_format="dcp",
        optimizer_storage="rank_sidecar",
    )
    dcp._complete_checkpoint_transaction(str(step), manifest)

    with pytest.raises(
        RuntimeError, match="completed checkpoint is missing rank-local RNG sidecar"
    ):
        dcp._validate_checkpoint_manifest(
            str(step), load_model=True, load_optimizer=True, load_rng=True
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda manifest: manifest.__setitem__("unknown", 1),
            "top-level schema mismatch",
        ),
        (lambda manifest: manifest.pop("world_size"), "top-level schema mismatch"),
        (lambda manifest: manifest.__setitem__("step", -1), "non-negative integer"),
        (lambda manifest: manifest.__setitem__("world_size", 0), "positive integer"),
        (
            lambda manifest: manifest["components"].__setitem__("model", 1),
            "component values must be booleans",
        ),
        (
            lambda manifest: manifest["components"].__setitem__("unknown", False),
            "components schema mismatch",
        ),
        (
            lambda manifest: manifest.__setitem__("optimizer_storage", "rank_sidecar"),
            "optimizer component does not match optimizer_storage",
        ),
        (
            lambda manifest: manifest["components"].__setitem__("extra_states", True),
            "extra_states component does not match extra_state_files",
        ),
    ],
)
def test_checkpoint_manifest_v2_uses_exact_schema(mutation, match):
    from megatron.lite.primitive.ckpt import dcp

    manifest = {
        "format": dcp._CHECKPOINT_MANIFEST_FORMAT,
        "status": "complete",
        "step": 3,
        "world_size": 1,
        "components": {
            "model": True,
            "optimizer": False,
            "rng": False,
            "extra_states": False,
        },
        "payload_format": "dcp",
        "optimizer_storage": "none",
        "extra_state_files": [],
    }
    mutation(manifest)

    with pytest.raises(RuntimeError, match=match):
        dcp._validate_checkpoint_manifest_schema(manifest, expected_status="complete")


def test_checkpoint_resolver_rejects_rank_path_disagreement(monkeypatch, tmp_path):
    from megatron.lite.primitive.ckpt import dcp

    step = tmp_path / "step_3"
    manifest = dcp._begin_checkpoint_transaction(
        str(step),
        step=3,
        save_model=True,
        save_optimizer=False,
        save_rng=False,
        payload_format="dcp",
        optimizer_storage="none",
    )
    dcp._complete_checkpoint_transaction(str(step), manifest)

    def disagree(value):
        other = copy.deepcopy(value)
        other["selected_path"] = f"{other['selected_path']}.rank1"
        return [value, other]

    monkeypatch.setattr(dcp, "_gather_world_objects", disagree)
    with pytest.raises(
        RuntimeError, match="checkpoint step resolution differs across ranks"
    ):
        dcp._resolve_step_checkpoint_path(str(tmp_path))


def test_checkpoint_manifest_rejects_rank_content_disagreement(monkeypatch, tmp_path):
    from megatron.lite.primitive.ckpt import dcp

    step = tmp_path / "step_4"
    manifest = dcp._begin_checkpoint_transaction(
        str(step),
        step=4,
        save_model=True,
        save_optimizer=False,
        save_rng=False,
        payload_format="dcp",
        optimizer_storage="none",
    )
    dcp._complete_checkpoint_transaction(str(step), manifest)

    def disagree(value):
        other = copy.deepcopy(value)
        other["manifest"]["step"] = 5
        return [value, other]

    monkeypatch.setattr(dcp, "_gather_world_objects", disagree)
    with pytest.raises(RuntimeError, match="checkpoint manifest differs across ranks"):
        dcp._validate_checkpoint_manifest(
            str(step), load_model=True, load_optimizer=False, load_rng=False
        )


def test_checkpoint_manifest_rejects_rank_presence_disagreement(monkeypatch, tmp_path):
    from megatron.lite.primitive.ckpt import dcp

    step = tmp_path / "step_5"
    manifest = dcp._begin_checkpoint_transaction(
        str(step),
        step=5,
        save_model=True,
        save_optimizer=False,
        save_rng=False,
        payload_format="dcp",
        optimizer_storage="none",
    )
    dcp._complete_checkpoint_transaction(str(step), manifest)

    def disagree(value):
        other = copy.deepcopy(value)
        other["manifest_present"] = False
        other["manifest"] = None
        return [value, other]

    monkeypatch.setattr(dcp, "_gather_world_objects", disagree)
    with pytest.raises(RuntimeError, match="checkpoint manifest differs across ranks"):
        dcp._validate_checkpoint_manifest(
            str(step), load_model=True, load_optimizer=False, load_rng=False
        )


def test_checkpoint_resolver_wraps_local_directory_read_error(monkeypatch, tmp_path):
    from megatron.lite.primitive.ckpt import dcp

    def fail_listdir(_path):
        raise PermissionError("injected directory read denial")

    monkeypatch.setattr(dcp.os, "listdir", fail_listdir)
    with pytest.raises(
        RuntimeError,
        match=(
            "checkpoint step resolution failed: PermissionError: "
            "injected directory read denial"
        ),
    ):
        dcp._resolve_step_checkpoint_path(str(tmp_path))


def test_dcp_tensor_contract_includes_only_persistent_buffers():
    from megatron.lite.primitive.ckpt import dcp

    class BufferedModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2))
            self.register_buffer("router_bias", torch.arange(2), persistent=True)
            self.register_buffer("workspace", torch.zeros(4), persistent=False)

    items = dict(dcp._named_model_checkpoint_tensors(BufferedModule()))

    assert set(items) == {"weight", "router_bias"}


def test_dcp_model_metadata_rejects_unexpected_current_stage_tensor():
    from megatron.lite.primitive.ckpt import dcp

    model = nn.Linear(2, 2)
    parameters = list(model.named_parameters())
    metadata_keys = {
        dcp._DCP_MODEL_SCHEMA_KEY,
        "model_pp1.weight",
        "model_pp1.bias",
        "model_pp1.removed_parameter",
        # A different pipeline stage owns a disjoint namespace and must not be
        # mistaken for an unexpected tensor on this rank.
        "model_pp0.weight",
    }

    with pytest.raises(
        RuntimeError, match=r"unexpected model tensors.*model_pp1\.removed_parameter"
    ):
        dcp._validate_dcp_model_metadata(
            metadata_keys,
            model_prefix="model_pp1",
            parameter_items=parameters,
            buffer_items=[],
            require_schema=True,
        )


def test_completed_dcp_model_metadata_requires_schema_key():
    from megatron.lite.primitive.ckpt import dcp

    model = nn.Linear(2, 2)
    with pytest.raises(RuntimeError, match="missing the required model schema key"):
        dcp._validate_dcp_model_metadata(
            {"model.weight", "model.bias"},
            model_prefix="model",
            parameter_items=list(model.named_parameters()),
            buffer_items=[],
            require_schema=True,
        )


def test_requested_rank_local_optimizer_checkpoint_must_exist(tmp_path):
    from megatron.lite.primitive.ckpt import dcp

    optimizer = torch.optim.AdamW(nn.Linear(2, 2).parameters(), lr=0.01)

    with pytest.raises(
        FileNotFoundError,
        match="optimizer checkpoint requested by load_optimizer=True is missing",
    ):
        dcp._load_optimizer_checkpoint(optimizer, str(tmp_path))


def test_dcp_load_optimizer_request_requires_optimizer_object(tmp_path):
    from megatron.lite.primitive.ckpt import dcp

    with pytest.raises(
        ValueError, match="load_optimizer=True requires a non-None optimizer"
    ):
        dcp.load_training_checkpoint(
            nn.Linear(2, 2), None, str(tmp_path), use_dcp=True, load_optimizer=True
        )


def test_rng_tracker_save_converts_graph_safe_generator_to_tensor(monkeypatch):
    from megatron.lite.primitive.ckpt import dcp

    generator = torch.Generator().manual_seed(20260628)

    class Tracker:
        @staticmethod
        def get_states():
            return {"model-parallel-rng": generator}

    tensor_parallel = types.SimpleNamespace(
        get_cuda_rng_tracker=lambda: Tracker(),
        convert_cuda_rng_state=lambda state, *, to_graphable: (
            state
            if to_graphable or isinstance(state, torch.Tensor)
            else state.get_state()
        ),
    )
    core = types.ModuleType("megatron.core")
    core.tensor_parallel = tensor_parallel
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)

    states = dcp._get_cuda_rng_tracker_states()

    assert states.keys() == {"model-parallel-rng"}
    assert isinstance(states["model-parallel-rng"], torch.Tensor)
    torch.testing.assert_close(
        states["model-parallel-rng"], generator.get_state(), atol=0, rtol=0
    )


def _assert_nested_state_equal(actual, expected):
    if torch.is_tensor(expected):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            _assert_nested_state_equal(actual[key], value)
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_value, expected_value in zip(actual, expected, strict=True):
            _assert_nested_state_equal(actual_value, expected_value)
    else:
        assert actual == expected


def _initialized_adam(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model(torch.ones(2, model.in_features)).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return optimizer


def _initialized_fp32_adamw():
    from megatron.lite.primitive.optimizers.fsdp2.adamw import FP32AdamW

    param = nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
    optimizer = FP32AdamW(
        [param], lr=0.05, weight_decay=0.1, betas=(0.8, 0.95), eps=1e-6
    )
    param.grad = torch.tensor([0.25, -0.5])
    optimizer.step()
    param.grad = None
    return param, optimizer


def _legacy_fp32_adamw_state(optimizer):
    current = copy.deepcopy(optimizer.state_dict())
    return {
        "type": "fp32_adamw",
        "step_count": current["step_count"],
        "master_params": current["master_params"],
        "exp_avgs": current["exp_avgs"],
        "exp_avg_sqs": current["exp_avg_sqs"],
        "steps": current["steps"],
        "weight_decays": [
            float(optimizer.state[param].get("weight_decay", optimizer.weight_decay))
            for param in optimizer.params
        ],
    }


def test_fp32_adamw_preflight_is_non_mutating_and_never_calls_load(monkeypatch):
    from megatron.lite.primitive.ckpt import dcp

    _param, optimizer = _initialized_fp32_adamw()
    candidate = copy.deepcopy(optimizer.state_dict())
    candidate["master_params"][0].add_(7.0)
    candidate["exp_avgs"][0].add_(3.0)
    candidate["exp_avg_sqs"][0].add_(2.0)
    live_before = {
        name: optimizer.state[optimizer.params[0]][name].clone()
        for name in ("master_param", "exp_avg", "exp_avg_sq")
    }
    monkeypatch.setattr(
        optimizer,
        "load_state_dict",
        lambda _state: pytest.fail("preflight must not call load_state_dict"),
    )

    dcp._preflight_optimizer_checkpoint_state(optimizer, candidate)

    for name, expected in live_before.items():
        torch.testing.assert_close(
            optimizer.state[optimizer.params[0]][name], expected, atol=0, rtol=0
        )


@pytest.mark.parametrize("corruption", ["shape", "dtype"])
def test_fp32_adamw_preflight_rejects_tensor_contract_mismatch(corruption):
    from megatron.lite.primitive.ckpt import dcp

    _param, optimizer = _initialized_fp32_adamw()
    candidate = copy.deepcopy(optimizer.state_dict())
    if corruption == "shape":
        candidate["master_params"][0] = torch.zeros(3, dtype=torch.float32)
    else:
        candidate["master_params"][0] = candidate["master_params"][0].double()

    with pytest.raises(RuntimeError, match="optimizer checkpoint is incompatible"):
        dcp._preflight_optimizer_checkpoint_state(optimizer, candidate)


def test_fp32_adamw_checkpoint_restores_complete_optimizer_configuration():
    _param, source = _initialized_fp32_adamw()
    state = copy.deepcopy(source.state_dict())
    target_param = nn.Parameter(torch.tensor([3.0, 4.0], dtype=torch.float32))
    from megatron.lite.primitive.optimizers.fsdp2.adamw import FP32AdamW

    target = FP32AdamW(
        [target_param], lr=0.2, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8
    )

    target.load_state_dict(state)

    assert target.lr == source.lr
    assert target.weight_decay == source.weight_decay
    assert target.betas == source.betas
    assert target.eps == source.eps
    assert target.step_count == source.step_count
    assert target.param_groups[0]["lr"] == source.param_groups[0]["lr"]
    assert (
        target.param_groups[0]["weight_decay"] == source.param_groups[0]["weight_decay"]
    )


def test_fp32_adamw_explicit_legacy_migration_upgrades_to_strict_v2():
    _param, optimizer = _initialized_fp32_adamw()
    legacy = _legacy_fp32_adamw_state(optimizer)
    legacy["weight_decays"] = [0.125 for _param in optimizer.params]

    migrated = optimizer.migrate_legacy_state_dict(legacy)

    assert migrated["version"] == 2
    assert migrated["config"]["betas"] == optimizer.betas
    assert migrated["param_groups"][0]["options"]["weight_decay"] == 0.125
    optimizer.validate_state_dict(migrated)


@pytest.mark.parametrize("wrapper_kind", ["fsdp2", "chained"])
def test_legacy_optimizer_sidecar_migrates_only_on_explicit_unmanifested_load(
    monkeypatch, tmp_path, wrapper_kind
):
    from megatron.lite.primitive.ckpt import dcp
    from megatron.lite.primitive.optimizers.fsdp2.adamw import ChainedOptimizer

    _param, fp32_optimizer = _initialized_fp32_adamw()
    legacy = _legacy_fp32_adamw_state(fp32_optimizer)
    expected_master = legacy["master_params"][0].clone()
    expected_step = legacy["steps"][0]

    class FSDP2Like:
        def __init__(self, optimizer):
            self.optimizer = optimizer

        def state_dict(self):
            return self.optimizer.state_dict()

        def load_state_dict(self, state):
            self.optimizer.load_state_dict(state)

    if wrapper_kind == "fsdp2":
        optimizer = FSDP2Like(fp32_optimizer)
        sidecar_state = legacy
    else:
        optimizer = ChainedOptimizer([fp32_optimizer])
        sidecar_state = {"type": "chained_torch_optimizer", "optimizers": [legacy]}

    fp32_optimizer.step_count = 0
    for state in fp32_optimizer.state.values():
        state["master_param"].zero_()
        state["exp_avg"].zero_()
        state["exp_avg_sq"].zero_()
        state["step"] = 0

    step = tmp_path / "step_31"
    step.mkdir()
    torch.save(sidecar_state, step / "optimizer_rank_0.pt")
    monkeypatch.setattr(dcp, "_supports_dist_opt_distckpt", lambda *_args: False)
    monkeypatch.setattr(dcp, "_build_meshes", lambda _config: (None, None))

    def fake_dcp_load(state_dict, *, checkpoint_id):
        assert checkpoint_id == str(step)
        state_dict["step"] = 31

    monkeypatch.setattr(dcp.dcp, "load", fake_dcp_load)

    with pytest.raises(RuntimeError, match="Legacy DCP checkpoints are rejected"):
        dcp.load_training_checkpoint(
            nn.Linear(2, 2),
            optimizer,
            str(step),
            config=object(),
            ps=types.SimpleNamespace(pp_size=1, pp_rank=0),
            use_dcp=True,
            load_model=False,
            load_optimizer=True,
            load_rng=False,
        )

    assert (
        dcp.load_training_checkpoint(
            nn.Linear(2, 2),
            optimizer,
            str(step),
            config=object(),
            ps=types.SimpleNamespace(pp_size=1, pp_rank=0),
            use_dcp=True,
            load_model=False,
            load_optimizer=True,
            load_rng=False,
            allow_legacy_checkpoint=True,
        )
        == 31
    )
    assert fp32_optimizer.step_count == legacy["step_count"]
    assert fp32_optimizer.state[fp32_optimizer.params[0]]["step"] == expected_step
    torch.testing.assert_close(
        fp32_optimizer.state[fp32_optimizer.params[0]]["master_param"],
        expected_master,
        atol=0,
        rtol=0,
    )


def test_manifested_checkpoint_never_enables_legacy_optimizer_migration(
    monkeypatch, tmp_path
):
    from megatron.lite.primitive.ckpt import dcp

    _param, optimizer = _initialized_fp32_adamw()
    legacy = _legacy_fp32_adamw_state(optimizer)
    step = tmp_path / "step_32"
    _complete_manifest(dcp, step, save_optimizer=True, optimizer_storage="rank_sidecar")
    torch.save(legacy, step / "optimizer_rank_0.pt")
    monkeypatch.setattr(dcp, "_supports_dist_opt_distckpt", lambda *_args: False)

    with pytest.raises(RuntimeError, match="optimizer checkpoint is incompatible"):
        dcp.load_training_checkpoint(
            nn.Linear(2, 2),
            optimizer,
            str(step),
            config=object(),
            ps=types.SimpleNamespace(pp_size=1, pp_rank=0),
            use_dcp=True,
            load_model=False,
            load_optimizer=True,
            load_rng=False,
            allow_legacy_checkpoint=True,
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda state: state["config"].__setitem__("weight_decay", float("nan")),
        lambda state: state["config"].pop("betas"),
        lambda state: state["config"].__setitem__("betas", "bad"),
    ],
)
def test_fp32_adamw_preflight_rejects_invalid_hyperparameters(mutate):
    from megatron.lite.primitive.ckpt import dcp

    _param, optimizer = _initialized_fp32_adamw()
    candidate = copy.deepcopy(optimizer.state_dict())
    mutate(candidate)

    with pytest.raises(RuntimeError, match="optimizer checkpoint is incompatible"):
        dcp._preflight_optimizer_checkpoint_state(optimizer, candidate)


def test_torch_adamw_preflight_rejects_missing_initialized_moment_key():
    from megatron.lite.primitive.ckpt import dcp

    model = nn.Linear(2, 2)
    optimizer = _initialized_adam(model)
    candidate = copy.deepcopy(optimizer.state_dict())
    first_state = next(iter(candidate["state"].values()))
    first_state.pop("exp_avg")

    with pytest.raises(RuntimeError, match="Optimizer state key mismatch"):
        dcp._preflight_optimizer_checkpoint_state(optimizer, candidate)


def test_torch_adamw_preflight_rejects_missing_moment_with_fresh_runtime():
    from megatron.lite.primitive.ckpt import dcp

    source_model = nn.Linear(2, 2)
    source_optimizer = _initialized_adam(source_model)
    candidate = copy.deepcopy(source_optimizer.state_dict())
    next(iter(candidate["state"].values())).pop("exp_avg")
    target_model = nn.Linear(2, 2)
    target_optimizer = torch.optim.AdamW(target_model.parameters(), lr=0.01)
    assert target_optimizer.state == {}

    with pytest.raises(RuntimeError, match="Optimizer state key mismatch"):
        dcp._preflight_optimizer_checkpoint_state(target_optimizer, candidate)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda state: state["param_groups"][0].pop("betas"),
        lambda state: state["param_groups"][0].__setitem__("betas", "bad"),
        lambda state: state["param_groups"][0].__setitem__("lr", float("nan")),
    ],
)
def test_torch_adamw_preflight_rejects_invalid_group_schema_or_values(mutate):
    from megatron.lite.primitive.ckpt import dcp

    model = nn.Linear(2, 2)
    optimizer = _initialized_adam(model)
    candidate = copy.deepcopy(optimizer.state_dict())
    mutate(candidate)

    with pytest.raises(RuntimeError, match="optimizer checkpoint is incompatible"):
        dcp._preflight_optimizer_checkpoint_state(optimizer, candidate)


def test_fresh_mcore_chained_optimizer_preflight_uses_pure_inner_schema():
    from megatron.lite.primitive.ckpt import dcp

    class FreshMCoreDistOpt:
        def __init__(self, lr_mult):
            self.grad_scaler = None
            self.optimizer = types.SimpleNamespace(
                param_groups=[
                    {
                        "params": [],
                        "lr": 1.0e-3,
                        "wd_mult": 1.0,
                        "lr_mult": lr_mult,
                        "is_expert_parallel": False,
                        "is_decoupled_lr": False,
                    }
                ]
            )
            self.lr_mult = lr_mult

        def state_dict(self):
            raise AssertionError("fresh MCore outer state_dict must not be called")

        def get_parameter_state_dp_zero(self, *, empty_data=False):
            assert empty_data is True
            return {
                "buckets_coalesced": True,
                0: {
                    (torch.float32, torch.float32): {
                        "param": torch.empty(2),
                        "exp_avg": torch.empty(2),
                        "exp_avg_sq": torch.empty(2),
                        "numel_unpadded": 2,
                    }
                },
            }

    children = [FreshMCoreDistOpt(1.0), FreshMCoreDistOpt(2.0)]

    class FreshMCoreChain:
        chained_optimizers = children

        @staticmethod
        def load_state_dict(_state):
            raise AssertionError("preflight must not apply optimizer state")

    def saved_state(child):
        group = {
            key: value
            for key, value in child.optimizer.param_groups[0].items()
            if key != "params"
        }
        group["step"] = 8
        return {"optimizer": {"param_groups": [group]}}

    candidate = [saved_state(child) for child in children]
    dcp._preflight_optimizer_checkpoint_state(FreshMCoreChain(), candidate)
    templates = dcp._optimizer_parameter_state_template(FreshMCoreChain())
    assert isinstance(templates, list) and len(templates) == 2
    assert templates[0][0][(torch.float32, torch.float32)]["param"].shape == (2,)

    candidate[1]["optimizer"]["param_groups"][0]["lr_mult"] = 3.0
    with pytest.raises(RuntimeError, match="group identifier 'lr_mult' mismatch"):
        dcp._preflight_optimizer_checkpoint_state(FreshMCoreChain(), candidate)


def _complete_manifest(
    dcp,
    step,
    *,
    save_optimizer=False,
    save_rng=False,
    optimizer_storage="none",
    payload_format="dcp",
    extra_state_files=(),
):
    manifest = dcp._begin_checkpoint_transaction(
        str(step),
        step=int(step.name.removeprefix("step_")),
        save_model=True,
        save_optimizer=save_optimizer,
        save_rng=save_rng,
        payload_format=payload_format,
        optimizer_storage=optimizer_storage,
        extra_state_files=extra_state_files,
    )
    dcp._complete_checkpoint_transaction(str(step), manifest)
    return manifest


def test_manifest_routes_model_only_distckpt_without_optimizer(monkeypatch, tmp_path):
    from megatron.lite.primitive.ckpt import dcp

    step = tmp_path / "step_33"
    _complete_manifest(dcp, step, payload_format="distckpt")
    calls = []
    monkeypatch.setattr(
        dcp, "_supports_dist_opt_distckpt", lambda _model, optimizer: optimizer is None
    )

    def load_dist(*_args, **kwargs):
        assert kwargs["load_model"] is True
        assert kwargs["load_optimizer"] is False
        calls.append("distckpt")
        return 33

    monkeypatch.setattr(dcp, "_load_dist_opt_checkpoint", load_dist)
    monkeypatch.setattr(
        dcp.dcp,
        "load",
        lambda *_args, **_kwargs: pytest.fail("generic DCP must not run"),
    )

    assert (
        dcp.load_training_checkpoint(
            nn.Linear(2, 2),
            None,
            str(step),
            use_dcp=True,
            load_model=True,
            load_optimizer=False,
            load_rng=False,
        )
        == 33
    )
    assert calls == ["distckpt"]


def test_manifest_routes_dcp_even_with_current_dist_optimizer_unrequested(
    monkeypatch, tmp_path
):
    from megatron.lite.primitive.ckpt import dcp

    step = tmp_path / "step_34"
    _complete_manifest(dcp, step, payload_format="dcp")
    optimizer = types.SimpleNamespace(sharded_state_dict=lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        dcp,
        "_supports_dist_opt_distckpt",
        lambda *_args: pytest.fail(
            "strict DCP routing must not use current heuristics"
        ),
    )
    monkeypatch.setattr(
        dcp,
        "_load_dist_opt_checkpoint",
        lambda *_args, **_kwargs: pytest.fail("distckpt must not run"),
    )

    def load_dcp(state_dict, *, checkpoint_id):
        assert checkpoint_id == str(step)
        state_dict["step"] = 34

    monkeypatch.setattr(dcp.dcp, "load", load_dcp)

    assert (
        dcp.load_training_checkpoint(
            nn.Linear(2, 2),
            optimizer,
            str(step),
            use_dcp=True,
            load_model=False,
            load_optimizer=False,
            load_rng=False,
        )
        == 34
    )


@pytest.mark.parametrize("payload_format", ["dcp", "distckpt"])
def test_manifest_extra_only_load_needs_no_model_backend_config(
    monkeypatch, tmp_path, payload_format
):
    from megatron.lite.primitive.ckpt import dcp

    step = tmp_path / "step_35"
    _complete_manifest(
        dcp, step, payload_format=payload_format, extra_state_files=("scheduler.pt",)
    )
    torch.save({"cursor": 9}, step / "scheduler.pt")
    loaded = {}
    monkeypatch.setattr(
        dcp,
        "_supports_dist_opt_distckpt",
        lambda *_args: pytest.fail("extra-only strict routing needs no heuristic"),
    )
    if payload_format == "distckpt":
        monkeypatch.setattr(
            dcp, "_load_dist_opt_checkpoint", lambda *_args, **_kwargs: 35
        )
        monkeypatch.setattr(
            dcp.dcp,
            "load",
            lambda *_args, **_kwargs: pytest.fail("generic DCP must not run"),
        )
    else:

        def load_dcp(state_dict, *, checkpoint_id):
            assert checkpoint_id == str(step)
            state_dict["step"] = 35

        monkeypatch.setattr(dcp.dcp, "load", load_dcp)
        monkeypatch.setattr(
            dcp,
            "_load_dist_opt_checkpoint",
            lambda *_args, **_kwargs: pytest.fail("distckpt must not run"),
        )

    assert (
        dcp.load_training_checkpoint(
            nn.Linear(2, 2),
            None,
            str(step),
            use_dcp=True,
            load_model=False,
            load_optimizer=False,
            load_rng=False,
            load_extra_state_files=("scheduler.pt",),
            loaded_extra_states=loaded,
        )
        == 35
    )
    assert loaded == {"scheduler.pt": {"cursor": 9}}


def test_corrupt_optimizer_sidecar_fails_before_dcp_model_load(monkeypatch, tmp_path):
    from megatron.lite.primitive.ckpt import dcp

    model = nn.Linear(2, 2)
    optimizer = _initialized_adam(model)
    model_before = copy.deepcopy(model.state_dict())
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    rng_before = torch.get_rng_state().clone()
    step = tmp_path / "step_3"
    _complete_manifest(dcp, step, save_optimizer=True, optimizer_storage="rank_sidecar")
    (step / "optimizer_rank_0.pt").write_bytes(b"not-a-torch-checkpoint")
    monkeypatch.setattr(dcp, "_supports_dist_opt_distckpt", lambda *_args: False)
    monkeypatch.setattr(
        dcp.dcp,
        "load",
        lambda *_args, **_kwargs: pytest.fail("DCP model load must not run"),
    )

    with pytest.raises(RuntimeError, match="checkpoint sidecar preflight failed"):
        dcp.load_training_checkpoint(
            model, optimizer, str(step), use_dcp=True, load_rng=False
        )

    _assert_nested_state_equal(model.state_dict(), model_before)
    _assert_nested_state_equal(optimizer.state_dict(), optimizer_before)
    torch.testing.assert_close(torch.get_rng_state(), rng_before, atol=0, rtol=0)


def test_incompatible_optimizer_sidecar_rolls_back_preflight(monkeypatch, tmp_path):
    from megatron.lite.primitive.ckpt import dcp

    model = nn.Linear(2, 2)
    optimizer = _initialized_adam(model)
    model_before = copy.deepcopy(model.state_dict())
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    rng_before = torch.get_rng_state().clone()
    incompatible = copy.deepcopy(optimizer_before)
    incompatible["param_groups"][0]["params"] = []
    step = tmp_path / "step_4"
    _complete_manifest(dcp, step, save_optimizer=True, optimizer_storage="rank_sidecar")
    torch.save(incompatible, step / "optimizer_rank_0.pt")
    monkeypatch.setattr(dcp, "_supports_dist_opt_distckpt", lambda *_args: False)

    with pytest.raises(RuntimeError, match="optimizer checkpoint is incompatible"):
        dcp.load_training_checkpoint(
            model, optimizer, str(step), use_dcp=True, load_rng=False
        )

    _assert_nested_state_equal(model.state_dict(), model_before)
    _assert_nested_state_equal(optimizer.state_dict(), optimizer_before)
    torch.testing.assert_close(torch.get_rng_state(), rng_before, atol=0, rtol=0)


def test_optimizer_sidecar_validator_rejects_without_mutating_live_state(
    monkeypatch, tmp_path
):
    from megatron.lite.primitive.ckpt import dcp

    class PartialOptimizer:
        def __init__(self):
            self.value = 0

        def state_dict(self):
            return {"value": self.value}

        def load_state_dict(self, state):
            self.value = state["value"]

        @staticmethod
        def validate_state_dict(state):
            if state.get("fail"):
                raise RuntimeError("incompatible optimizer state")

    model = nn.Linear(2, 2)
    optimizer = PartialOptimizer()
    model_before = copy.deepcopy(model.state_dict())
    rng_before = torch.get_rng_state().clone()
    step = tmp_path / "step_41"
    _complete_manifest(dcp, step, save_optimizer=True, optimizer_storage="rank_sidecar")
    torch.save({"value": 9, "fail": True}, step / "optimizer_rank_0.pt")
    monkeypatch.setattr(dcp, "_supports_dist_opt_distckpt", lambda *_args: False)

    with pytest.raises(RuntimeError, match="incompatible optimizer state"):
        dcp.load_training_checkpoint(
            model, optimizer, str(step), use_dcp=True, load_rng=False
        )

    _assert_nested_state_equal(model.state_dict(), model_before)
    assert optimizer.value == 0
    torch.testing.assert_close(torch.get_rng_state(), rng_before, atol=0, rtol=0)


def test_sidecar_preflight_never_snapshots_model_or_deepcopies_optimizer(
    monkeypatch, tmp_path
):
    from megatron.lite.primitive.ckpt import dcp

    class NoDeepcopyState(dict):
        def __deepcopy__(self, _memo):
            raise AssertionError("optimizer state must not be deep-copied")

    class Optimizer:
        def __init__(self):
            self.value = 0

        def state_dict(self):
            return NoDeepcopyState(value=self.value)

        def load_state_dict(self, state):
            self.value = state["value"]

        @staticmethod
        def validate_state_dict(state):
            if not isinstance(state.get("value"), int):
                raise TypeError("value must be int")

    optimizer = Optimizer()
    torch.save({"value": 7}, tmp_path / "optimizer_rank_0.pt")
    monkeypatch.setattr(
        dcp,
        "_chunk_tensor_state",
        lambda *_args, **_kwargs: pytest.fail("model snapshot must not run"),
    )

    optimizer_state, rng_state, extra_states = dcp._preload_checkpoint_sidecars(
        str(tmp_path),
        optimizer=optimizer,
        load_optimizer=True,
        load_rng=False,
        rng_required=False,
        extra_state_files=(),
        extra_state_validators={},
        extra_state_targets={},
        checkpoint_step=None,
    )

    assert optimizer.value == 0
    assert optimizer_state == {"value": 7}
    assert rng_state is None
    assert extra_states == {}


def test_distopt_corrupt_rng_sidecar_is_preloaded_before_payload(monkeypatch, tmp_path):
    from megatron.lite.primitive.ckpt import dcp

    model = nn.Linear(2, 2)
    model_before = copy.deepcopy(model.state_dict())
    rng_before = torch.get_rng_state().clone()
    step = tmp_path / "step_5"
    _complete_manifest(dcp, step, save_rng=True, payload_format="distckpt")
    (step / "rng_state_rank_00000.pt").write_bytes(b"corrupt-rng")
    monkeypatch.setattr(dcp, "_supports_dist_opt_distckpt", lambda *_args: True)
    monkeypatch.setattr(
        dcp,
        "_load_dist_opt_checkpoint",
        lambda *_args, **_kwargs: pytest.fail("distckpt payload load must not run"),
    )

    with pytest.raises(RuntimeError, match="checkpoint sidecar preflight failed"):
        dcp.load_training_checkpoint(
            model, None, str(step), use_dcp=True, load_optimizer=False, load_rng=True
        )

    _assert_nested_state_equal(model.state_dict(), model_before)
    torch.testing.assert_close(torch.get_rng_state(), rng_before, atol=0, rtol=0)


def test_partially_applied_rng_preflight_restores_every_rng(monkeypatch, tmp_path):
    from megatron.lite.primitive.ckpt import dcp

    original = copy.deepcopy(dcp._get_rng_state())
    random.seed(991)
    np.random.seed(991)
    candidate = copy.deepcopy(dcp._get_rng_state())
    dcp._restore_rng_state(original)
    candidate["torch_rng_state"] = torch.zeros(1, dtype=torch.uint8)
    model = nn.Linear(2, 2)
    model_before = copy.deepcopy(model.state_dict())
    rng_before = copy.deepcopy(dcp._get_rng_state())
    step = tmp_path / "step_51"
    _complete_manifest(dcp, step, save_rng=True, payload_format="distckpt")
    torch.save(candidate, step / "rng_state_rank_00000.pt")
    monkeypatch.setattr(dcp, "_supports_dist_opt_distckpt", lambda *_args: True)
    monkeypatch.setattr(
        dcp,
        "_load_dist_opt_checkpoint",
        lambda *_args, **_kwargs: pytest.fail("distckpt payload load must not run"),
    )

    with pytest.raises(RuntimeError, match="RNG checkpoint is incompatible"):
        dcp.load_training_checkpoint(
            model, None, str(step), use_dcp=True, load_optimizer=False, load_rng=True
        )

    restored = dcp._get_rng_state()
    assert restored["random_rng_state"] == rng_before["random_rng_state"]
    assert restored["np_rng_state"][0] == rng_before["np_rng_state"][0]
    np.testing.assert_array_equal(
        restored["np_rng_state"][1], rng_before["np_rng_state"][1]
    )
    assert restored["np_rng_state"][2:] == rng_before["np_rng_state"][2:]
    torch.testing.assert_close(
        restored["torch_rng_state"], rng_before["torch_rng_state"], atol=0, rtol=0
    )
    _assert_nested_state_equal(model.state_dict(), model_before)


def test_rank0_extra_state_is_in_completion_manifest(monkeypatch, tmp_path):
    from megatron.lite.primitive.ckpt import dcp

    model = nn.Linear(2, 2)
    monkeypatch.setattr(dcp, "_supports_dist_opt_distckpt", lambda *_args: True)
    monkeypatch.setattr(
        dcp, "_save_dist_opt_checkpoint", lambda *_args, **_kwargs: None
    )

    dcp.save_training_checkpoint(
        model,
        None,
        6,
        str(tmp_path),
        use_dcp=True,
        save_optimizer=False,
        save_rng=False,
        extra_states={"lr_scheduler.pt": {"num_steps": 11}},
    )

    step = tmp_path / "step_6"
    manifest = json.loads((step / dcp._CHECKPOINT_MANIFEST).read_text())
    assert manifest["status"] == "complete"
    assert manifest["components"]["extra_states"] is True
    assert manifest["extra_state_files"] == ["lr_scheduler.pt"]
    assert torch.load(step / "lr_scheduler.pt", weights_only=False) == {"num_steps": 11}


def test_completed_manifest_requires_every_declared_extra_state(tmp_path):
    from megatron.lite.primitive.ckpt import dcp

    step = tmp_path / "step_61"
    _complete_manifest(
        dcp, step, payload_format="distckpt", extra_state_files=("lr_scheduler.pt",)
    )

    with pytest.raises(
        RuntimeError, match="completed checkpoint is missing declared extra-state file"
    ):
        dcp._validate_checkpoint_manifest(
            str(step),
            load_model=True,
            load_optimizer=False,
            load_rng=False,
            load_extra_state_files=("lr_scheduler.pt",),
        )


def test_extra_state_validator_fails_before_distopt_payload(monkeypatch, tmp_path):
    from megatron.lite.primitive.ckpt import dcp

    model = nn.Linear(2, 2)
    model_before = copy.deepcopy(model.state_dict())
    rng_before = torch.get_rng_state().clone()
    loaded = {"existing": "preserve"}
    step = tmp_path / "step_7"
    _complete_manifest(
        dcp, step, payload_format="distckpt", extra_state_files=("lr_scheduler.pt",)
    )
    torch.save({"num_steps": "bad"}, step / "lr_scheduler.pt")
    monkeypatch.setattr(dcp, "_supports_dist_opt_distckpt", lambda *_args: True)
    monkeypatch.setattr(
        dcp,
        "_load_dist_opt_checkpoint",
        lambda *_args, **_kwargs: pytest.fail("distckpt payload load must not run"),
    )

    def validate_scheduler(state):
        if not isinstance(state.get("num_steps"), int):
            raise TypeError("num_steps must be int")

    with pytest.raises(RuntimeError, match="num_steps must be int"):
        dcp.load_training_checkpoint(
            model,
            None,
            str(step),
            use_dcp=True,
            load_optimizer=False,
            load_rng=False,
            load_extra_state_files=("lr_scheduler.pt",),
            loaded_extra_states=loaded,
            extra_state_validators={"lr_scheduler.pt": validate_scheduler},
        )

    _assert_nested_state_equal(model.state_dict(), model_before)
    assert loaded == {"existing": "preserve"}
    torch.testing.assert_close(torch.get_rng_state(), rng_before, atol=0, rtol=0)


def test_requested_extra_state_is_published_only_after_core_commit(
    monkeypatch, tmp_path
):
    from megatron.lite.primitive.ckpt import dcp

    events = []

    class Target:
        def __init__(self):
            self.value = 3

        def snapshot(self):
            return self.value

        def apply(self, state):
            events.append("target_apply")
            self.value = state["num_steps"]

        def restore(self, snapshot):
            events.append("target_restore")
            self.value = snapshot

        def fingerprint(self):
            return self.value

        @staticmethod
        def validate_step(state, expected_step):
            events.append(("validate_step", expected_step))
            assert state["checkpoint_step"] == expected_step

    model = nn.Linear(2, 2)
    step = tmp_path / "step_71"
    _complete_manifest(
        dcp, step, payload_format="distckpt", extra_state_files=("lr_scheduler.pt",)
    )
    torch.save({"num_steps": 17, "checkpoint_step": 71}, step / "lr_scheduler.pt")
    loaded = {}
    target = Target()
    monkeypatch.setattr(dcp, "_supports_dist_opt_distckpt", lambda *_args: True)

    def load_core(*_args, **_kwargs):
        events.append("core")
        return 71

    monkeypatch.setattr(dcp, "_load_dist_opt_checkpoint", load_core)

    def validate_scheduler(state):
        assert state["num_steps"] == 17

    restored_step = dcp.load_training_checkpoint(
        model,
        None,
        str(step),
        use_dcp=True,
        load_optimizer=False,
        load_rng=False,
        load_extra_state_files=("lr_scheduler.pt",),
        loaded_extra_states=loaded,
        extra_state_validators={"lr_scheduler.pt": validate_scheduler},
        extra_state_targets={"lr_scheduler.pt": target},
    )

    assert restored_step == 71
    assert loaded == {"lr_scheduler.pt": {"num_steps": 17, "checkpoint_step": 71}}
    assert target.value == 17
    assert [event for event in events if isinstance(event, str)] == [
        "target_apply",
        "target_restore",
        "core",
        "target_apply",
    ]
    assert ("validate_step", 71) in events


def test_extra_state_target_commit_failure_rolls_back_target_and_poison_fails(
    monkeypatch, tmp_path
):
    from megatron.lite.primitive.ckpt import dcp

    class Target:
        def __init__(self):
            self.value = 4
            self.apply_calls = 0

        def snapshot(self):
            return self.value

        def apply(self, state):
            self.apply_calls += 1
            self.value = state["num_steps"]
            if self.apply_calls == 2:
                raise RuntimeError("injected target commit failure")

        def restore(self, snapshot):
            self.value = snapshot

        def fingerprint(self):
            return self.value

    model = nn.Linear(2, 2)
    step = tmp_path / "step_72"
    _complete_manifest(
        dcp, step, payload_format="distckpt", extra_state_files=("lr_scheduler.pt",)
    )
    torch.save({"num_steps": 18}, step / "lr_scheduler.pt")
    loaded = {}
    target = Target()
    core_called = []
    monkeypatch.setattr(dcp, "_supports_dist_opt_distckpt", lambda *_args: True)

    def load_core(*_args, **_kwargs):
        core_called.append(True)
        return 72

    monkeypatch.setattr(dcp, "_load_dist_opt_checkpoint", load_core)

    with pytest.raises(
        RuntimeError,
        match=r"runtime must be poisoned: .*injected target commit failure",
    ):
        dcp.load_training_checkpoint(
            model,
            None,
            str(step),
            use_dcp=True,
            load_optimizer=False,
            load_rng=False,
            load_extra_state_files=("lr_scheduler.pt",),
            loaded_extra_states=loaded,
            extra_state_targets={"lr_scheduler.pt": target},
        )

    assert core_called == [True]
    assert target.value == 4
    assert loaded == {}


def test_extra_state_target_commit_runs_after_optimizer_and_rng():
    from megatron.lite.primitive.ckpt import dcp

    events = []

    class Optimizer:
        @staticmethod
        def load_state_dict(_state):
            events.append("optimizer")

    class Target:
        value = 0

        def snapshot(self):
            return self.value

        def apply(self, state):
            events.append("target")
            self.value = state["num_steps"]

        def restore(self, snapshot):
            self.value = snapshot

        def fingerprint(self):
            return self.value

    target = Target()
    loaded = {}
    dcp._commit_preloaded_sidecars(
        optimizer=Optimizer(),
        optimizer_state={"state": "ready"},
        rng_state=None,
        loaded_extra_states=loaded,
        extra_state_values={"lr_scheduler.pt": {"num_steps": 19}},
        extra_state_targets={"lr_scheduler.pt": target},
    )

    assert events == ["optimizer", "target"]
    assert target.value == 19
    assert loaded == {"lr_scheduler.pt": {"num_steps": 19}}


def test_distckpt_preflights_on_disk_keys_before_mcore_load_can_mutate_model(
    monkeypatch, tmp_path
):
    target_module = "megatron.lite.primitive.ckpt.distckpt"
    parent_module = importlib.import_module("megatron.lite.primitive.ckpt")
    missing_parent_attr = object()
    previous_parent_attr = parent_module.__dict__.get("distckpt", missing_parent_attr)
    monkeypatch.delitem(sys.modules, target_module, raising=False)

    strict_sentinel = object()

    class StrictHandling:
        RAISE_UNEXPECTED = strict_sentinel

    class ShardedBase:
        def __init__(self, key):
            self.key = key

    class ShardedTensor(ShardedBase):
        def __init__(self, key):
            super().__init__(key)
            self.global_shape = (1,)
            self.dtype = torch.float32
            self.allow_shape_mismatch = False

    class ShardedTensorFactory(ShardedBase):
        def build(self):
            raise AssertionError("factory expansion is not expected in this test")

    class LocalNonpersistentObject:
        pass

    dist_checkpointing = types.ModuleType("megatron.core.dist_checkpointing")

    def unexpected_load(*_args, **_kwargs):
        raise AssertionError("MCore load must not run after metadata mismatch")

    dist_checkpointing.load = unexpected_load
    mapping = types.ModuleType("megatron.core.dist_checkpointing.mapping")
    mapping.LocalNonpersistentObject = LocalNonpersistentObject
    mapping.ShardedBase = ShardedBase
    mapping.ShardedTensor = ShardedTensor
    mapping.ShardedTensorFactory = ShardedTensorFactory
    validation = types.ModuleType("megatron.core.dist_checkpointing.validation")
    validation.StrictHandling = StrictHandling
    core = types.ModuleType("megatron.core")
    core.dist_checkpointing = dist_checkpointing
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(
        sys.modules, "megatron.core.dist_checkpointing", dist_checkpointing
    )
    monkeypatch.setitem(
        sys.modules, "megatron.core.dist_checkpointing.mapping", mapping
    )
    monkeypatch.setitem(
        sys.modules, "megatron.core.dist_checkpointing.validation", validation
    )

    distckpt = importlib.import_module(target_module)
    model = nn.Linear(2, 2)
    before = copy.deepcopy(model.state_dict())
    monkeypatch.setattr(
        distckpt,
        "_model_sharded_state_dict",
        lambda _model: {
            "model": {"weight": ShardedTensor("weight"), "bias": ShardedTensor("bias")}
        },
    )
    monkeypatch.setattr(
        distckpt,
        "_load_checkpoint_sharded_metadata",
        lambda _path: {
            "weight": ShardedTensor("weight"),
            "bias": ShardedTensor("bias"),
            "removed": ShardedTensor("removed_parameter"),
        },
    )
    monkeypatch.setattr(
        distckpt, "_load_checkpoint_common_state", lambda _path: {"step": 0}
    )

    with pytest.raises(
        RuntimeError,
        match=r"distckpt checkpoint metadata preflight failed: .*removed_parameter",
    ):
        distckpt.load_dist_opt_checkpoint(
            model, None, str(tmp_path), load_model=True, load_optimizer=False
        )

    _assert_nested_state_equal(model.state_dict(), before)
    if previous_parent_attr is missing_parent_attr:
        parent_module.__dict__.pop("distckpt", None)
    else:
        parent_module.distckpt = previous_parent_attr


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
        state = torch.load(filename)
        self.parameter_load_calls = int(state["parameter_save_calls"])

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
        AssertionError, match=r"^local export parameter collision for weight$"
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

    state = {"module.layers.0.weight": object(), "_orig_mod.layers.0.weight": object()}

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

    import megatron.lite.primitive.parallel as parallel
    from megatron.lite.primitive.ckpt import hf_weights

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
        RuntimeError,
        match=(
            r"WeightSpec HF load preflight failed: RuntimeError: chunk0: "
            r"checkpoint shape mismatch for weight: source=\(1, 2\) target=\(2, 2\)"
        ),
    ):
        hf_weights.load_hf_weights(model, "/unused", WeightSpec(), ps)


def test_generic_hf_loader_scopes_atomic_transaction_to_explicit_group(monkeypatch):
    from types import SimpleNamespace

    from megatron.lite.primitive.ckpt import hf_weights

    calls = []
    participating_group = object()

    def fake_atomic_load(
        model,
        builder,
        *,
        context,
        participating_group=None,
        allow_missing_parameter=None,
    ):
        del builder, allow_missing_parameter
        calls.append((model, context, participating_group))

    class WeightSpec:
        pass

    model = nn.Linear(2, 2, bias=False)
    monkeypatch.setattr(hf_weights, "load_hf_model_chunks_atomically", fake_atomic_load)

    hf_weights.load_hf_weights(
        model,
        "/unused",
        WeightSpec(),
        SimpleNamespace(),
        participating_group=participating_group,
    )

    assert calls == [(model, "WeightSpec HF load", participating_group)]


def test_generic_hf_loader_rejects_stale_local_layer_mapping(monkeypatch):
    from types import SimpleNamespace

    import megatron.lite.primitive.parallel as parallel
    from megatron.lite.primitive.ckpt import hf_weights

    class Reader:
        def __init__(self, _path):
            pass

        @staticmethod
        def get_tensor(_name):
            return torch.ones(2, 2)

    class WeightSpec:
        @staticmethod
        def weight_map():
            return {
                "layers.0.weight": ["hf.weight"],
                "layers.0.removed_weight": ["hf.removed_weight"],
            }

        @staticmethod
        def expert_global_id(_name):
            return None

        @staticmethod
        def hf_to_native(_name, tensors):
            return tensors[0]

        @staticmethod
        def tp_spec(_name):
            return None

    class LocalStage(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_indices = [0]
            self.layers = nn.ModuleList([nn.Linear(2, 2, bias=False)])

    monkeypatch.setattr(hf_weights, "SafeTensorReader", Reader)
    monkeypatch.setitem(parallel.__dict__, "pad_vocab_for_tp", lambda size, _tp: size)

    with pytest.raises(
        RuntimeError,
        match=r"maps local layer tensor 'layers\.0\.removed_weight'.*no matching",
    ):
        hf_weights.load_hf_weights(
            LocalStage(), "/unused", WeightSpec(), SimpleNamespace(ep_size=1, ep_rank=0)
        )


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
        RuntimeError, match=r"^HF safetensors write failed: OSError: disk full$"
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
