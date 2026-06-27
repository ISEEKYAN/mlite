# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import copy
import math
from types import SimpleNamespace

import pytest
import torch
from verl_mlite.engine.config import MegatronLiteEngineConfig
from verl_mlite.engine.mlite_engine import (
    _LR_SCHEDULER_CONFIG_FIELDS,
    _LR_SCHEDULER_FORMAT,
    _LR_SCHEDULER_PAYLOAD_FORMAT,
    _LR_SCHEDULER_STATE,
    MegatronLiteEngine,
    _checkpoint_components_with_consensus,
    _content_set,
    _LRSchedulerCheckpointTarget,
    _MegatronLiteLRScheduler,
    _scheduler_payload,
    _scheduler_state_with_consensus,
    _validate_lr_scheduler_payload,
    _validate_lr_scheduler_state,
)


def _scheduler_state(*, num_steps=7, max_lr=0.25):
    config = {
        "init_lr": 0.0,
        "max_lr": max_lr,
        "min_lr": 0.0,
        "lr_warmup_steps": 0,
        "lr_decay_steps": 10,
        "lr_decay_style": "constant",
        "start_wd": 0.1,
        "end_wd": 0.1,
        "wd_incr_steps": 10,
        "wd_incr_style": "constant",
        "wsd_decay_steps": None,
        "lr_wsd_decay_style": "exponential",
    }
    assert set(config) == set(_LR_SCHEDULER_CONFIG_FIELDS)
    return {"format": _LR_SCHEDULER_FORMAT, "num_steps": num_steps, "config": config}


def _scheduler_payload_state(*, checkpoint_step=13, num_steps=None, max_lr=0.25):
    if num_steps is None:
        num_steps = checkpoint_step
    return {
        "format": _LR_SCHEDULER_PAYLOAD_FORMAT,
        "checkpoint_step": checkpoint_step,
        "scheduler_state": _scheduler_state(num_steps=num_steps, max_lr=max_lr),
    }


class _Scheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self._state = _scheduler_state()
        self.loaded_state = None

    def state_dict(self):
        return copy.deepcopy(self._state)

    def load_state_dict(self, state):
        _validate_lr_scheduler_state(state)
        self._state = copy.deepcopy(state)
        self.loaded_state = copy.deepcopy(state)
        for group in self.optimizer.param_groups:
            group["lr"] = state["config"]["max_lr"]
            group["weight_decay"] = state["config"]["end_wd"]


def _real_scheduler(*, max_lr=0.25, use_checkpoint_config=False):
    optimizer = SimpleNamespace(
        param_groups=[{"lr": max_lr, "weight_decay": 0.1, "min_lr": 0.0}]
    )
    scheduler = _MegatronLiteLRScheduler(
        optimizer,
        init_lr=0.0,
        max_lr=max_lr,
        min_lr=0.0,
        lr_warmup_steps=0,
        lr_decay_steps=10,
        lr_decay_style="constant",
        start_wd=0.1,
        end_wd=0.1,
        wd_incr_steps=10,
        wd_incr_style="constant",
        wsd_decay_steps=None,
        lr_wsd_decay_style="exponential",
        use_checkpoint_config=use_checkpoint_config,
    )
    return scheduler, optimizer


def _group_scheduler(param_groups, *, use_checkpoint_config=False):
    optimizer = SimpleNamespace(param_groups=param_groups)
    scheduler = _MegatronLiteLRScheduler(
        optimizer,
        init_lr=0.01,
        max_lr=0.2,
        min_lr=0.02,
        lr_warmup_steps=2,
        lr_decay_steps=6,
        lr_decay_style="linear",
        start_wd=0.1,
        end_wd=0.2,
        wd_incr_steps=4,
        wd_incr_style="linear",
        wsd_decay_steps=None,
        lr_wsd_decay_style="exponential",
        use_checkpoint_config=use_checkpoint_config,
    )
    return scheduler, optimizer


@pytest.fixture(autouse=True)
def _single_process_dist(monkeypatch):
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.dist.is_initialized", lambda: False
    )


def _optimizer_config() -> SimpleNamespace:
    return SimpleNamespace(
        optimizer="adam",
        lr=1e-6,
        min_lr=None,
        min_lr_ratio=None,
        clip_grad=1.0,
        weight_decay=0.1,
        lr_warmup_steps_ratio=0.0,
        total_training_steps=10,
        lr_warmup_steps=0,
        override_optimizer_config={},
    )


def _engine_config(**kwargs) -> MegatronLiteEngineConfig:
    values = {"custom_backend_module": None, "impl_cfg": {"use_thd": True}}
    values.update(kwargs)
    return MegatronLiteEngineConfig(**values)


def _initialized_engine(*, checkpoint_config=None, param_offload=False):
    engine = MegatronLiteEngine(
        model_config=SimpleNamespace(
            local_path="/tmp/qwen35", hf_config={"model_type": "qwen3_5_moe"}, mtp=None
        ),
        engine_config=_engine_config(param_offload=param_offload),
        optimizer_config=_optimizer_config(),
        checkpoint_config=checkpoint_config or {},
    )

    def placement_fn(name):
        return ["placement", name]

    def expert_classifier(name):
        return name.endswith("expert")

    parallel = SimpleNamespace(tp=1, cp=1, pp=1)
    parallel_state = SimpleNamespace(dp_rank=0)
    module = torch.nn.Linear(2, 2)
    optimizer = SimpleNamespace(
        param_groups=[{"lr": 0.25, "weight_decay": 0.1, "min_lr": 0.0}]
    )
    scheduler = _Scheduler(optimizer)
    engine.module = module
    engine.handle = SimpleNamespace(
        _optimizer=optimizer,
        _lr_scheduler=scheduler,
        _config=SimpleNamespace(parallel=parallel),
        _parallel_state=parallel_state,
        _extras={
            "protocol": SimpleNamespace(
                PLACEMENT_FN=placement_fn, EXPERT_CLASSIFIER=expert_classifier
            )
        },
    )
    engine.runtime = object()
    return (
        engine,
        module,
        optimizer,
        scheduler,
        parallel,
        parallel_state,
        placement_fn,
        expert_classifier,
    )


def test_checkpoint_content_set_uses_exact_keys_and_rejects_unknown_values():
    assert _content_set(None) == set()
    assert _content_set("model") == {"model"}
    assert _content_set("[model, optimizer, extra]") == {"model", "optimizer", "extra"}
    assert _content_set((" 'model' ", '"extra"')) == {"model", "extra"}
    with pytest.raises(ValueError, match="unsupported entries"):
        _content_set("not_model", key="checkpoint_config.save_contents")
    with pytest.raises(TypeError, match="entries must be strings"):
        _content_set(["model", 7], key="checkpoint_config.save_contents")


def test_save_checkpoint_forwards_contents_scheduler_and_param_offload_reload(
    tmp_path, monkeypatch
):
    (
        engine,
        module,
        optimizer,
        scheduler,
        parallel,
        parallel_state,
        placement_fn,
        expert_classifier,
    ) = _initialized_engine(
        checkpoint_config={"save_contents": ["model", "extra"]}, param_offload=True
    )
    to_calls = []
    save_calls = []
    sync_calls = []
    monkeypatch.setattr(engine, "to", lambda **kwargs: to_calls.append(kwargs))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: sync_calls.append(True))
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.save_training_checkpoint",
        lambda *args, **kwargs: save_calls.append((args, kwargs)),
    )
    scheduler._state["num_steps"] = 13

    engine.save_checkpoint(str(tmp_path), global_step=13)

    assert to_calls == [
        {"device": "cuda", "model": True, "optimizer": False, "grad": False},
        {"device": "cpu", "model": True, "optimizer": False, "grad": False},
    ]
    assert sync_calls == [True]
    assert len(save_calls) == 1
    save_args, save_kwargs = save_calls[0]
    assert save_args == (module, optimizer, 13, str(tmp_path), parallel, parallel_state)
    assert save_kwargs["get_placements"] is placement_fn
    assert save_kwargs["is_expert"] is expert_classifier
    assert save_kwargs["save_model"] is True
    assert save_kwargs["save_optimizer"] is False
    assert save_kwargs["save_rng"] is True
    assert save_kwargs["extra_states"] == {
        _LR_SCHEDULER_STATE: _scheduler_payload_state(checkpoint_step=13)
    }
    assert not (tmp_path / _LR_SCHEDULER_STATE).exists()


def test_save_checkpoint_skips_when_contents_exclude_model_and_optimizer(
    tmp_path, monkeypatch
):
    engine, *_ = _initialized_engine(checkpoint_config={"save_contents": []})
    checkpoint_path = tmp_path / "ckpt"
    save_calls = []
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.save_training_checkpoint",
        lambda *args, **kwargs: save_calls.append((args, kwargs)),
    )

    engine.save_checkpoint(str(checkpoint_path), global_step=13)

    assert save_calls == []
    assert not checkpoint_path.exists()


def test_save_setup_failure_reaches_consensus_before_checkpoint_collectives(
    tmp_path, monkeypatch
):
    engine, _module, _optimizer, scheduler, *_ = _initialized_engine()
    scheduler._state["num_steps"] = 3
    monkeypatch.setattr(
        engine, "_checkpoint_hooks", lambda: (_ for _ in ()).throw(OSError("hook boom"))
    )
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.save_training_checkpoint",
        lambda *args, **kwargs: pytest.fail("checkpoint save must not run"),
    )
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.dist.is_initialized", lambda: True
    )
    monkeypatch.setattr("verl_mlite.engine.mlite_engine.dist.get_world_size", lambda: 2)
    gathered_errors = []

    def fake_all_gather_object(output, value):
        if isinstance(value, str):
            gathered_errors.append(value)
            output[:] = [value, None]
        else:
            output[:] = [copy.deepcopy(value), copy.deepcopy(value)]

    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.dist.all_gather_object", fake_all_gather_object
    )

    with pytest.raises(RuntimeError, match="checkpoint save setup failed.*hook boom"):
        engine.save_checkpoint(str(tmp_path), global_step=3)

    assert any("hook boom" in error for error in gathered_errors)


def test_save_offload_cleanup_failure_is_not_silently_ignored(tmp_path, monkeypatch):
    engine, _module, _optimizer, scheduler, *_ = _initialized_engine(param_offload=True)
    scheduler._state["num_steps"] = 3

    def fake_to(**kwargs):
        if kwargs["device"] == "cpu":
            raise RuntimeError("offload cleanup boom")

    monkeypatch.setattr(engine, "to", fake_to)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.save_training_checkpoint",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(RuntimeError, match="offload cleanup.*offload cleanup boom"):
        engine.save_checkpoint(str(tmp_path), global_step=3)


def test_save_checkpoint_treats_rng_and_scheduler_as_extra(tmp_path, monkeypatch):
    engine, _module, _optimizer, scheduler, *_ = _initialized_engine(
        checkpoint_config={"save_contents": ["extra"]}
    )
    save_calls = []
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.save_training_checkpoint",
        lambda *args, **kwargs: save_calls.append((args, kwargs)),
    )
    scheduler._state["num_steps"] = 9

    engine.save_checkpoint(str(tmp_path), global_step=9)

    assert len(save_calls) == 1
    _args, kwargs = save_calls[0]
    assert kwargs["save_model"] is False
    assert kwargs["save_optimizer"] is False
    assert kwargs["save_rng"] is True
    assert kwargs["extra_states"] == {
        _LR_SCHEDULER_STATE: {
            "format": _LR_SCHEDULER_PAYLOAD_FORMAT,
            "checkpoint_step": 9,
            "scheduler_state": scheduler.state_dict(),
        }
    }


def test_save_checkpoint_model_only_does_not_write_rng_or_scheduler(
    tmp_path, monkeypatch
):
    engine, *_ = _initialized_engine(checkpoint_config={"save_contents": ["model"]})
    save_calls = []
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.save_training_checkpoint",
        lambda *args, **kwargs: save_calls.append((args, kwargs)),
    )

    engine.save_checkpoint(str(tmp_path), global_step=9)

    _args, kwargs = save_calls[0]
    assert kwargs["save_model"] is True
    assert kwargs["save_optimizer"] is False
    assert kwargs["save_rng"] is False
    assert kwargs["extra_states"] is None


@pytest.mark.parametrize(
    "contents",
    [
        [],
        ["model"],
        ["optimizer"],
        ["extra"],
        ["model", "optimizer"],
        ["model", "extra"],
        ["optimizer", "extra"],
        ["model", "optimizer", "extra"],
    ],
)
def test_save_checkpoint_all_component_combinations(tmp_path, monkeypatch, contents):
    engine, _module, _optimizer, scheduler, *_ = _initialized_engine(
        checkpoint_config={"save_contents": contents}
    )
    calls = []
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.save_training_checkpoint",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    scheduler._state["num_steps"] = 6
    if "optimizer" in contents and "extra" not in contents:
        with pytest.raises(ValueError, match="exact-resume checkpoint"):
            engine.save_checkpoint(str(tmp_path / "ckpt"), global_step=6)
        assert calls == []
        return

    engine.save_checkpoint(str(tmp_path / "ckpt"), global_step=6)

    if not contents:
        assert calls == []
        return
    _args, kwargs = calls[0]
    assert kwargs["save_model"] is ("model" in contents)
    assert kwargs["save_optimizer"] is ("optimizer" in contents)
    assert kwargs["save_rng"] is ("extra" in contents)
    assert bool(kwargs["extra_states"]) is ("extra" in contents)


def test_load_checkpoint_restores_scheduler_and_param_offload_reload(
    tmp_path, monkeypatch
):
    (
        engine,
        module,
        optimizer,
        scheduler,
        parallel,
        parallel_state,
        placement_fn,
        expert_classifier,
    ) = _initialized_engine(param_offload=True)
    saved_scheduler_payload = _scheduler_payload_state(
        checkpoint_step=23, num_steps=23, max_lr=0.125
    )
    to_calls = []
    load_calls = []
    sync_calls = []
    monkeypatch.setattr(engine, "to", lambda **kwargs: to_calls.append(kwargs))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: sync_calls.append(True))

    def fake_load(*args, **kwargs):
        load_calls.append((args, kwargs))
        kwargs["extra_state_validators"][_LR_SCHEDULER_STATE](saved_scheduler_payload)
        target = kwargs["extra_state_targets"][_LR_SCHEDULER_STATE]
        target.validate_step(saved_scheduler_payload, 23)
        snapshot = target.snapshot()
        target.apply(saved_scheduler_payload)
        assert target.fingerprint()["checkpoint_step"] == 23
        target.restore(snapshot)
        optimizer.param_groups[0]["lr"] = 999.0  # core optimizer commit happens first
        target.apply(saved_scheduler_payload)
        kwargs["loaded_extra_states"][_LR_SCHEDULER_STATE] = saved_scheduler_payload
        return 23

    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.load_training_checkpoint", fake_load
    )

    engine.load_checkpoint(str(tmp_path))

    assert to_calls == [
        {"device": "cuda", "model": True, "optimizer": False, "grad": False},
        {"device": "cpu", "model": True, "optimizer": False, "grad": False},
    ]
    assert sync_calls == [True]
    assert scheduler.loaded_state == saved_scheduler_payload["scheduler_state"]
    assert optimizer.param_groups[0]["lr"] == 0.125
    assert len(load_calls) == 1
    load_args, load_kwargs = load_calls[0]
    assert load_args == (module, optimizer, str(tmp_path), parallel, parallel_state)
    assert load_kwargs["get_placements"] is placement_fn
    assert load_kwargs["is_expert"] is expert_classifier
    assert load_kwargs["load_model"] is True
    assert load_kwargs["load_optimizer"] is True
    assert load_kwargs["load_rng"] is True
    assert load_kwargs["allow_legacy_checkpoint"] is False
    assert load_kwargs["load_extra_state_files"] == (_LR_SCHEDULER_STATE,)
    assert load_kwargs["loaded_extra_states"] == {
        _LR_SCHEDULER_STATE: saved_scheduler_payload
    }
    assert load_kwargs["extra_state_validators"] == {
        _LR_SCHEDULER_STATE: _validate_lr_scheduler_payload
    }
    assert isinstance(
        load_kwargs["extra_state_targets"][_LR_SCHEDULER_STATE],
        _LRSchedulerCheckpointTarget,
    )


@pytest.mark.parametrize(
    ("load_contents", "expected"),
    [
        ([], (False, False, False, False)),
        (["model"], (True, False, False, False)),
        (["optimizer"], (False, True, False, False)),
        (["extra"], (False, False, True, True)),
        (["model", "optimizer"], (True, True, False, False)),
        (["model", "extra"], (True, False, True, True)),
        (["optimizer", "extra"], (False, True, True, True)),
        (["model", "optimizer", "extra"], (True, True, True, True)),
    ],
)
def test_load_checkpoint_component_policy_is_symmetric(
    tmp_path, monkeypatch, load_contents, expected
):
    engine, *_ = _initialized_engine(checkpoint_config={"load_contents": load_contents})
    payload = _scheduler_payload_state(checkpoint_step=5, num_steps=5)
    calls = []

    def fake_load(*args, **kwargs):
        calls.append((args, kwargs))
        target_map = kwargs.get("extra_state_targets")
        if target_map:
            target = target_map[_LR_SCHEDULER_STATE]
            target.validate_step(payload, 5)
            target.apply(payload)
            kwargs["loaded_extra_states"][_LR_SCHEDULER_STATE] = payload
        return 5

    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.load_training_checkpoint", fake_load
    )

    load_model, load_optimizer, load_rng, has_scheduler_target = expected
    if load_optimizer and not load_rng:
        with pytest.raises(ValueError, match="matching LR scheduler progress"):
            engine.load_checkpoint(str(tmp_path))
        assert calls == []
        return

    engine.load_checkpoint(str(tmp_path))

    if not any(expected):
        assert calls == []
        return
    _args, kwargs = calls[0]
    assert kwargs["load_model"] is load_model
    assert kwargs["load_optimizer"] is load_optimizer
    assert kwargs["load_rng"] is load_rng
    assert bool(kwargs["extra_state_targets"]) is has_scheduler_target


def test_load_contents_defaults_to_saved_partial_component_policy(
    tmp_path, monkeypatch
):
    engine, *_ = _initialized_engine(checkpoint_config={"save_contents": ["model"]})
    calls = []
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.load_training_checkpoint",
        lambda *args, **kwargs: calls.append((args, kwargs)) or 3,
    )

    engine.load_checkpoint(str(tmp_path))

    _args, kwargs = calls[0]
    assert kwargs["load_model"] is True
    assert kwargs["load_optimizer"] is False
    assert kwargs["load_rng"] is False
    assert kwargs["extra_state_targets"] is None


def test_load_setup_failure_reaches_consensus_before_checkpoint_collectives(
    tmp_path, monkeypatch
):
    engine, *_ = _initialized_engine()
    monkeypatch.setattr(
        engine, "_checkpoint_hooks", lambda: (_ for _ in ()).throw(OSError("hook boom"))
    )
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.load_training_checkpoint",
        lambda *args, **kwargs: pytest.fail("checkpoint load must not run"),
    )
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.dist.is_initialized", lambda: True
    )
    monkeypatch.setattr("verl_mlite.engine.mlite_engine.dist.get_world_size", lambda: 2)
    gathered_errors = []

    def fake_all_gather_object(output, value):
        if isinstance(value, str):
            gathered_errors.append(value)
            output[:] = [value, None]
        else:
            output[:] = [copy.deepcopy(value), copy.deepcopy(value)]

    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.dist.all_gather_object", fake_all_gather_object
    )

    with pytest.raises(RuntimeError, match="checkpoint load setup failed.*hook boom"):
        engine.load_checkpoint(str(tmp_path))

    assert any("hook boom" in error for error in gathered_errors)
    assert engine._checkpoint_load_poisoned is True


@pytest.mark.parametrize(
    "state, error",
    [
        (None, TypeError),
        ({}, ValueError),
        ({**_scheduler_state(), "format": "unsupported"}, ValueError),
        ({**_scheduler_state(), "num_steps": True}, TypeError),
        ({**_scheduler_state(), "num_steps": 1.5}, TypeError),
        ({**_scheduler_state(), "num_steps": -1}, ValueError),
        (
            {
                **_scheduler_state(),
                "config": {**_scheduler_state()["config"], "max_lr": float("nan")},
            },
            ValueError,
        ),
        (
            {
                **_scheduler_state(),
                "config": {**_scheduler_state()["config"], "min_lr": -0.1},
            },
            ValueError,
        ),
        (
            {
                **_scheduler_state(),
                "config": {
                    **_scheduler_state()["config"],
                    "min_lr": 0.5,
                    "max_lr": 0.25,
                },
            },
            ValueError,
        ),
        (
            {
                **_scheduler_state(),
                "config": {
                    **_scheduler_state()["config"],
                    "wd_incr_style": "exponential",
                },
            },
            ValueError,
        ),
        (
            {
                **_scheduler_state(),
                "config": {**_scheduler_state()["config"], "init_lr": 0.5},
            },
            ValueError,
        ),
        (
            {
                **_scheduler_state(),
                "config": {
                    **_scheduler_state()["config"],
                    "start_wd": 0.2,
                    "end_wd": 0.1,
                },
            },
            ValueError,
        ),
    ],
)
def test_lr_scheduler_state_validation_rejects_incompatible_sidecars(state, error):
    with pytest.raises(error):
        _validate_lr_scheduler_state(state)


def test_lr_scheduler_payload_binds_core_checkpoint_step():
    payload = _scheduler_payload_state(checkpoint_step=12)

    _validate_lr_scheduler_payload(payload, expected_step=12)
    with pytest.raises(RuntimeError, match="sidecar/core step mismatch"):
        _validate_lr_scheduler_payload(payload, expected_step=13)
    with pytest.raises(RuntimeError, match="progress/core step mismatch"):
        _validate_lr_scheduler_payload(
            _scheduler_payload_state(checkpoint_step=12, num_steps=11), expected_step=12
        )


def test_lr_scheduler_runtime_config_override_matches_verl_wrapper_policy():
    scheduler, _optimizer = _real_scheduler(max_lr=0.25)
    checkpoint_state = _scheduler_state(num_steps=4, max_lr=0.125)

    scheduler.load_state_dict(checkpoint_state)

    assert scheduler.max_lr == 0.25
    assert scheduler.num_steps == 4


def test_lr_scheduler_checkpoint_config_policy_is_explicit():
    scheduler, optimizer = _real_scheduler(max_lr=0.25, use_checkpoint_config=True)
    checkpoint_state = _scheduler_state(num_steps=4, max_lr=0.125)

    scheduler.load_state_dict(checkpoint_state)

    assert scheduler.max_lr == 0.125
    assert optimizer.param_groups[0]["lr"] == 0.125


def test_lr_scheduler_preserves_group_lr_wd_and_tensor_lr_across_resume():
    lr_tensor = torch.tensor(-1.0)
    direct, direct_optimizer = _group_scheduler(
        [
            {
                "lr": lr_tensor,
                "weight_decay": -1.0,
                "max_lr": 0.2,
                "min_lr": 0.02,
                "wd_mult": 1.0,
            },
            {
                "lr": -1.0,
                "weight_decay": -1.0,
                "max_lr": 0.05,
                "min_lr": 0.005,
                "wd_mult": 0.0,
            },
        ]
    )

    assert direct_optimizer.param_groups[0]["lr"] is lr_tensor
    assert lr_tensor.item() == pytest.approx(0.01)
    assert direct_optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.1)
    assert direct_optimizer.param_groups[1]["lr"] == pytest.approx(0.01)
    assert direct_optimizer.param_groups[1]["weight_decay"] == 0.0

    direct.step(2)
    assert lr_tensor.item() == pytest.approx(0.2)
    assert direct_optimizer.param_groups[1]["lr"] == pytest.approx(0.05)
    assert direct_optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.15)
    assert direct_optimizer.param_groups[1]["weight_decay"] == 0.0
    checkpoint_state = direct.state_dict()

    resumed_lr_tensor = torch.tensor(-1.0)
    resumed, resumed_optimizer = _group_scheduler(
        [
            {
                "lr": resumed_lr_tensor,
                "weight_decay": -1.0,
                "max_lr": 0.2,
                "min_lr": 0.02,
                "wd_mult": 1.0,
            },
            {
                "lr": -1.0,
                "weight_decay": -1.0,
                "max_lr": 0.05,
                "min_lr": 0.005,
                "wd_mult": 0.0,
            },
        ]
    )
    resumed.load_state_dict(checkpoint_state)

    assert resumed_optimizer.param_groups[0]["lr"] is resumed_lr_tensor
    for _ in range(3):
        direct.step()
        resumed.step()
        assert resumed.state_dict() == direct.state_dict()
        for direct_group, resumed_group in zip(
            direct_optimizer.param_groups, resumed_optimizer.param_groups, strict=True
        ):
            assert float(resumed_group["lr"]) == pytest.approx(
                float(direct_group["lr"])
            )
            assert resumed_group["weight_decay"] == pytest.approx(
                direct_group["weight_decay"]
            )


def test_lr_scheduler_supports_unambiguous_legacy_lr_multiplier():
    scheduler, optimizer = _group_scheduler(
        [{"lr": -1.0, "weight_decay": 0.1, "lr_mult": 0.5, "wd_mult": 1.0}]
    )

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.005)
    scheduler.step(2)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)


def test_lr_scheduler_rejects_ambiguous_explicit_bounds_and_lr_multiplier():
    with pytest.raises(ValueError, match="mixes explicit max_lr/min_lr"):
        _group_scheduler(
            [
                {
                    "lr": 0.0,
                    "weight_decay": 0.1,
                    "max_lr": 0.1,
                    "min_lr": 0.01,
                    "lr_mult": 0.5,
                }
            ]
        )


@pytest.mark.parametrize(
    ("style", "expected"),
    [
        ("linear", 0.5),
        ("cosine", 0.5),
        ("exponential", 2.0 * math.sqrt(0.5) - 1.0),
        ("minus_sqrt", 1.0 - math.sqrt(0.5)),
    ],
)
def test_wsd_lr_decay_matches_mcore_coefficients(style, expected):
    optimizer = SimpleNamespace(param_groups=[{"lr": 0.0, "weight_decay": 0.1}])
    scheduler = _MegatronLiteLRScheduler(
        optimizer,
        init_lr=0.0,
        max_lr=1.0,
        min_lr=0.0,
        lr_warmup_steps=0,
        lr_decay_steps=4,
        lr_decay_style="wsd",
        start_wd=0.1,
        end_wd=0.1,
        wd_incr_steps=4,
        wd_incr_style="constant",
        wsd_decay_steps=4,
        lr_wsd_decay_style=style,
        use_checkpoint_config=False,
    )

    scheduler.step(2)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(expected)


def test_inverse_square_root_pins_min_lr_after_decay_horizon():
    optimizer = SimpleNamespace(param_groups=[{"lr": 0.0, "weight_decay": 0.1}])
    scheduler = _MegatronLiteLRScheduler(
        optimizer,
        init_lr=0.0,
        max_lr=1.0,
        min_lr=0.1,
        lr_warmup_steps=1,
        lr_decay_steps=4,
        lr_decay_style="inverse-square-root",
        start_wd=0.1,
        end_wd=0.1,
        wd_incr_steps=4,
        wd_incr_style="constant",
        wsd_decay_steps=None,
        lr_wsd_decay_style="exponential",
        use_checkpoint_config=False,
    )

    scheduler.step(5)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)


@pytest.mark.parametrize("wsd_decay_steps", [None, 0, 5])
def test_wsd_scheduler_rejects_invalid_decay_span(wsd_decay_steps):
    optimizer = SimpleNamespace(param_groups=[{"lr": 0.0, "weight_decay": 0.1}])
    with pytest.raises(ValueError, match="requires 1 <= wsd_decay_steps"):
        _MegatronLiteLRScheduler(
            optimizer,
            init_lr=0.0,
            max_lr=1.0,
            min_lr=0.0,
            lr_warmup_steps=0,
            lr_decay_steps=4,
            lr_decay_style="wsd",
            start_wd=0.1,
            end_wd=0.1,
            wd_incr_steps=4,
            wd_incr_style="constant",
            wsd_decay_steps=wsd_decay_steps,
            lr_wsd_decay_style="exponential",
            use_checkpoint_config=False,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lr_warmup_steps", -1),
        ("lr_warmup_steps", 1.5),
        ("lr_decay_steps", 0),
        ("wd_incr_steps", 0),
    ],
)
def test_lr_scheduler_constructor_rejects_instead_of_coercing(field, value):
    kwargs = {
        "init_lr": 0.0,
        "max_lr": 1.0,
        "min_lr": 0.0,
        "lr_warmup_steps": 0,
        "lr_decay_steps": 4,
        "lr_decay_style": "linear",
        "start_wd": 0.1,
        "end_wd": 0.1,
        "wd_incr_steps": 4,
        "wd_incr_style": "constant",
        "wsd_decay_steps": None,
        "lr_wsd_decay_style": "exponential",
        "use_checkpoint_config": False,
    }
    kwargs[field] = value
    optimizer = SimpleNamespace(param_groups=[{"lr": 0.0, "weight_decay": 0.1}])

    with pytest.raises((TypeError, ValueError)):
        _MegatronLiteLRScheduler(optimizer, **kwargs)


@pytest.mark.parametrize("increment", [-1, 1.5, True])
def test_lr_scheduler_step_rejects_invalid_increment(increment):
    scheduler, _optimizer = _real_scheduler()

    with pytest.raises(ValueError, match="non-negative integer"):
        scheduler.step(increment)


def test_lr_scheduler_target_restores_tensor_lr_identity_after_preflight():
    lr_tensor = torch.tensor(0.0)
    scheduler, optimizer = _group_scheduler(
        [{"lr": lr_tensor, "weight_decay": 0.1, "max_lr": 0.2, "min_lr": 0.02}],
        use_checkpoint_config=True,
    )
    target = _LRSchedulerCheckpointTarget(scheduler)
    baseline = target.snapshot()
    candidate = _scheduler_payload_state(checkpoint_step=2, num_steps=2, max_lr=0.1)

    target.apply(candidate)
    target.restore(baseline)

    assert optimizer.param_groups[0]["lr"] is lr_tensor
    assert lr_tensor.item() == pytest.approx(0.01)


def test_lr_scheduler_resume_preserves_future_lr_and_weight_decay_sequence():
    direct, direct_optimizer = _real_scheduler(max_lr=0.25)
    for _ in range(4):
        direct.step()
    checkpoint_state = direct.state_dict()

    resumed, resumed_optimizer = _real_scheduler(max_lr=0.25)
    resumed.load_state_dict(checkpoint_state)

    for _ in range(5):
        direct.step()
        resumed.step()
        assert resumed.state_dict() == direct.state_dict()
        assert (
            resumed_optimizer.param_groups[0]["lr"]
            == direct_optimizer.param_groups[0]["lr"]
        )
        assert resumed_optimizer.param_groups[0]["weight_decay"] == (
            direct_optimizer.param_groups[0]["weight_decay"]
        )


def test_lr_scheduler_target_preflight_can_restore_small_state():
    scheduler, optimizer = _real_scheduler(max_lr=0.25, use_checkpoint_config=True)
    target = _LRSchedulerCheckpointTarget(scheduler)
    baseline = target.snapshot()
    payload = _scheduler_payload_state(checkpoint_step=8, num_steps=8, max_lr=0.125)

    target.validate_step(payload, 8)
    target.apply(payload)
    assert optimizer.param_groups[0]["lr"] == 0.125
    assert target.fingerprint()["checkpoint_step"] == 8
    target.restore(baseline)

    assert optimizer.param_groups[0]["lr"] == 0.25
    assert target.fingerprint() == _LRSchedulerCheckpointTarget(scheduler).fingerprint()


def test_scheduler_save_rejects_cross_rank_step_mismatch(monkeypatch):
    optimizer = SimpleNamespace(
        param_groups=[{"lr": 0.25, "weight_decay": 0.1, "min_lr": 0.0}]
    )
    scheduler = _Scheduler(optimizer)
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.dist.is_initialized", lambda: True
    )
    monkeypatch.setattr("verl_mlite.engine.mlite_engine.dist.get_world_size", lambda: 2)

    def fake_all_gather_object(output, value):
        if value is None:
            output[:] = [None, None]
        else:
            other = copy.deepcopy(value)
            other["num_steps"] += 1
            output[:] = [value, other]

    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.dist.all_gather_object", fake_all_gather_object
    )

    with pytest.raises(RuntimeError, match="differs across ranks"):
        _scheduler_state_with_consensus(scheduler)


def test_scheduler_payload_validation_reaches_error_consensus_before_payload_gather(
    monkeypatch,
):
    scheduler, _optimizer = _real_scheduler()
    state = scheduler.state_dict()
    state["num_steps"] = 7
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine._scheduler_state_with_consensus",
        lambda _scheduler: state,
    )
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.dist.is_initialized", lambda: True
    )
    monkeypatch.setattr("verl_mlite.engine.mlite_engine.dist.get_world_size", lambda: 2)
    gathered = []

    def fake_all_gather_object(output, value):
        gathered.append(value)
        if isinstance(value, str):
            output[:] = [value, None]
        else:
            pytest.fail("payload gather must not run after local validation failure")

    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.dist.all_gather_object", fake_all_gather_object
    )

    with pytest.raises(RuntimeError, match="progress/core step mismatch"):
        _scheduler_payload(scheduler, checkpoint_step=8)

    assert len(gathered) == 1


def test_checkpoint_component_presence_mismatch_fails_before_branch(monkeypatch):
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.dist.is_initialized", lambda: True
    )
    monkeypatch.setattr("verl_mlite.engine.mlite_engine.dist.get_world_size", lambda: 2)

    def fake_all_gather_object(output, value):
        output[:] = [value, (*value[:3], not value[3])]

    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.dist.all_gather_object", fake_all_gather_object
    )

    with pytest.raises(RuntimeError, match="component policy differs across ranks"):
        _checkpoint_components_with_consensus(
            model=True,
            optimizer=True,
            extra=True,
            scheduler_present=True,
            context="checkpoint save",
        )


def test_load_checkpoint_rejects_missing_required_scheduler_extra_state(
    tmp_path, monkeypatch
):
    engine, *_ = _initialized_engine()
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.load_training_checkpoint",
        lambda *args, **kwargs: 17,
    )

    with pytest.raises(RuntimeError, match="required lr_scheduler.pt extra state"):
        engine.load_checkpoint(str(tmp_path))

    assert engine._checkpoint_load_poisoned is True
    with pytest.raises(RuntimeError, match="poisoned by a failed checkpoint load"):
        engine.optimizer_step()


def test_post_load_validation_reaches_error_consensus_before_barrier(
    tmp_path, monkeypatch
):
    engine, *_ = _initialized_engine()
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.load_training_checkpoint",
        lambda *args, **kwargs: 17,
    )
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.dist.is_initialized", lambda: True
    )
    monkeypatch.setattr("verl_mlite.engine.mlite_engine.dist.get_world_size", lambda: 2)
    gathered_errors = []

    def fake_all_gather_object(output, value):
        if isinstance(value, str):
            gathered_errors.append(value)
        output[:] = [copy.deepcopy(value), copy.deepcopy(value)]

    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.dist.all_gather_object", fake_all_gather_object
    )
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.dist.barrier",
        lambda: pytest.fail("barrier must not run after post-load validation failure"),
    )

    with pytest.raises(RuntimeError, match="required lr_scheduler.pt extra state"):
        engine.load_checkpoint(str(tmp_path))

    assert any(
        "required lr_scheduler.pt extra state" in error for error in gathered_errors
    )
    assert engine._checkpoint_load_poisoned is True


def test_legacy_step_validation_reaches_consensus_before_scheduler_commit(
    tmp_path, monkeypatch
):
    from megatron.lite.primitive.ckpt import dcp as dcp_impl

    engine, *_ = _initialized_engine()
    payload = _scheduler_payload_state(checkpoint_step=8, num_steps=8)
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine._legacy_scheduler_payload_with_consensus",
        lambda *_args, **_kwargs: payload,
    )
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.load_training_checkpoint",
        lambda *args, **kwargs: 9,
    )
    monkeypatch.setattr(dcp_impl, "_preflight_extra_state_targets", lambda *_args: None)
    monkeypatch.setattr(
        dcp_impl,
        "_commit_extra_state_targets",
        lambda *_args: pytest.fail(
            "legacy target commit must not run on step mismatch"
        ),
    )
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.dist.is_initialized", lambda: True
    )
    monkeypatch.setattr("verl_mlite.engine.mlite_engine.dist.get_world_size", lambda: 2)
    gathered_errors = []

    def fake_all_gather_object(output, value):
        if isinstance(value, str):
            gathered_errors.append(value)
        output[:] = [copy.deepcopy(value), copy.deepcopy(value)]

    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.dist.all_gather_object", fake_all_gather_object
    )

    with pytest.raises(RuntimeError, match="sidecar/core step mismatch"):
        engine.load_checkpoint(str(tmp_path), allow_legacy_checkpoint=True)

    assert any("sidecar/core step mismatch" in error for error in gathered_errors)
    assert engine._checkpoint_load_poisoned is True


def test_explicit_legacy_scheduler_migration_reads_root_sidecar(tmp_path, monkeypatch):
    engine, _module, _optimizer, scheduler, *_ = _initialized_engine()
    (tmp_path / "step_4").mkdir()
    torch.save({"num_steps": 4}, tmp_path / _LR_SCHEDULER_STATE)
    load_calls = []

    def fake_load(*args, **kwargs):
        load_calls.append((args, kwargs))
        return 4

    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.load_training_checkpoint", fake_load
    )

    engine.load_checkpoint(str(tmp_path), allow_legacy_checkpoint=True)

    assert scheduler.state_dict()["num_steps"] == 4
    _args, kwargs = load_calls[0]
    assert kwargs["load_extra_state_files"] is None
    assert kwargs["extra_state_targets"] is None
    assert kwargs["allow_legacy_checkpoint"] is True


def test_legacy_scheduler_migration_rejects_root_and_step_sidecars(
    tmp_path, monkeypatch
):
    engine, *_ = _initialized_engine()
    step = tmp_path / "step_4"
    step.mkdir()
    torch.save({"num_steps": 4}, tmp_path / _LR_SCHEDULER_STATE)
    torch.save({"num_steps": 4}, step / _LR_SCHEDULER_STATE)
    core_called = []
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.load_training_checkpoint",
        lambda *args, **kwargs: core_called.append(True) or 4,
    )

    with pytest.raises(RuntimeError, match="requires exactly one recognized"):
        engine.load_checkpoint(str(tmp_path), allow_legacy_checkpoint=True)

    assert core_called == []
    assert engine._checkpoint_load_poisoned is True
