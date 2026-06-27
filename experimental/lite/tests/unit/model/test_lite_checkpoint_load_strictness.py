# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import datetime
import importlib
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

pytestmark = pytest.mark.mlite


def _weight_module(shape: tuple[int, ...]) -> nn.Module:
    module = nn.Module()
    module.register_parameter("weight", nn.Parameter(torch.zeros(shape)))
    return module


def _wrapped_target(shape: tuple[int, ...]) -> nn.Module:
    model = nn.Module()
    model.wrapper = nn.Module()
    model.wrapper.shared_gate = _weight_module(shape)
    return model


_COPY_MODULES = (
    "megatron.lite.model.glm5.lite.checkpoint",
    "megatron.lite.model.kimi_k2.lite.checkpoint",
    "megatron.lite.model.deepseek_v4.lite.checkpoint",
    "megatron.lite.model.qwen3_5.lite.checkpoint",
)

_PROTOCOL_MODULES = tuple(
    module_name.removesuffix(".checkpoint") + ".protocol"
    for module_name in _COPY_MODULES
)


@pytest.mark.parametrize("module_name", _COPY_MODULES)
def test_model_hf_copy_allows_only_unique_wrapper_suffix(
    module_name: str, transformer_engine_import_stub
) -> None:
    transformer_engine_import_stub()
    copy_loaded_state = importlib.import_module(module_name)._copy_loaded_state
    model = _wrapped_target((1, 4))
    expected = torch.arange(4, dtype=torch.float32).reshape(1, 4)

    copy_loaded_state(model, {"shared_gate.weight": expected})

    torch.testing.assert_close(
        model.wrapper.shared_gate.weight.detach(), expected, atol=0.0, rtol=0.0
    )


@pytest.mark.parametrize("module_name", _COPY_MODULES)
def test_model_hf_copy_rejects_broadcastable_shape(
    module_name: str, transformer_engine_import_stub
) -> None:
    transformer_engine_import_stub()
    copy_loaded_state = importlib.import_module(module_name)._copy_loaded_state
    model = _wrapped_target((1, 4))

    with pytest.raises(RuntimeError, match=r"checkpoint shape mismatch"):
        copy_loaded_state(model, {"shared_gate.weight": torch.ones(1)})


@pytest.mark.parametrize("module_name", _COPY_MODULES)
def test_model_hf_copy_rejects_ambiguous_wrapper_suffix(
    module_name: str, transformer_engine_import_stub
) -> None:
    transformer_engine_import_stub()
    copy_loaded_state = importlib.import_module(module_name)._copy_loaded_state
    model = nn.Module()
    model.left = _wrapped_target((1, 4))
    model.right = _wrapped_target((1, 4))

    with pytest.raises(RuntimeError, match=r"Ambiguous .* checkpoint target"):
        copy_loaded_state(model, {"shared_gate.weight": torch.ones(1, 4)})


def _two_parameter_model() -> nn.Module:
    model = nn.Module()
    model.first = _weight_module((2, 2))
    model.last = _weight_module((2, 2))
    return model


@pytest.mark.parametrize("module_name", _COPY_MODULES)
def test_model_hf_copy_rejects_missing_native_parameter_without_mutation(
    module_name: str, transformer_engine_import_stub
) -> None:
    transformer_engine_import_stub()
    copy_loaded_state = importlib.import_module(module_name)._copy_loaded_state
    model = _two_parameter_model()
    before = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }

    with pytest.raises(RuntimeError, match=r"does not cover all required native state"):
        copy_loaded_state(model, {"first.weight": torch.ones(2, 2)})

    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name], atol=0.0, rtol=0.0)


@pytest.mark.parametrize("module_name", _COPY_MODULES)
def test_model_hf_copy_late_shape_mismatch_does_not_partially_write(
    module_name: str, transformer_engine_import_stub
) -> None:
    transformer_engine_import_stub()
    copy_loaded_state = importlib.import_module(module_name)._copy_loaded_state
    model = _two_parameter_model()

    with pytest.raises(
        RuntimeError, match=r"checkpoint shape mismatch for last.weight"
    ):
        copy_loaded_state(
            model,
            {"first.weight": torch.full((2, 2), 7.0), "last.weight": torch.ones(1)},
        )

    torch.testing.assert_close(
        model.first.weight.detach(), torch.zeros(2, 2), atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        model.last.weight.detach(), torch.zeros(2, 2), atol=0.0, rtol=0.0
    )


@pytest.mark.parametrize("module_name", _COPY_MODULES)
def test_model_hf_copy_rejects_unmapped_native_key(
    module_name: str, transformer_engine_import_stub
) -> None:
    transformer_engine_import_stub()
    copy_loaded_state = importlib.import_module(module_name)._copy_loaded_state
    model = _weight_module((2, 2))

    with pytest.raises(RuntimeError, match=r"has no native target: stale.weight"):
        copy_loaded_state(
            model, {"weight": torch.ones(2, 2), "stale.weight": torch.ones(2, 2)}
        )

    torch.testing.assert_close(
        model.weight.detach(), torch.zeros(2, 2), atol=0.0, rtol=0.0
    )


@pytest.mark.parametrize("module_name", _COPY_MODULES)
def test_model_hf_loader_forwards_explicit_participating_group(
    module_name: str, monkeypatch, transformer_engine_import_stub
) -> None:
    transformer_engine_import_stub()
    checkpoint = importlib.import_module(module_name)
    participating_group = object()
    calls = []

    def fake_load_chunks(
        models,
        builder,
        *,
        context,
        participating_group=None,
        allow_missing_parameter=None,
    ) -> int:
        del builder, context, allow_missing_parameter
        calls.append((models, participating_group))
        return 0

    monkeypatch.setattr(checkpoint, "load_hf_model_chunks_atomically", fake_load_chunks)
    model = _weight_module((1,))
    checkpoint.load_hf_weights(
        model, "/unused", object(), object(), participating_group=participating_group
    )

    assert calls == [(model, participating_group)]


@pytest.mark.parametrize(
    ("checkpoint_module_name", "protocol_module_name"),
    zip(_COPY_MODULES, _PROTOCOL_MODULES, strict=True),
)
def test_model_protocol_hf_loaders_forward_explicit_participating_group(
    checkpoint_module_name: str,
    protocol_module_name: str,
    monkeypatch,
    transformer_engine_import_stub,
) -> None:
    transformer_engine_import_stub()
    checkpoint = importlib.import_module(checkpoint_module_name)
    protocol = importlib.import_module(protocol_module_name)
    participating_group = object()
    calls = []

    def fake_load(models, path, model_cfg, ps, *, participating_group=None) -> None:
        calls.append((models, path, model_cfg, ps, participating_group))

    monkeypatch.setattr(checkpoint, "load_hf_weights", fake_load)
    if hasattr(protocol, "_load_hf_weights_impl"):
        monkeypatch.setattr(protocol, "_load_hf_weights_impl", fake_load)

    chunk = _weight_module((1,))
    model_cfg = object()
    ps = object()
    protocol.load_hf_weights(
        chunk, "/unused", model_cfg, ps, participating_group=participating_group
    )
    protocol.load_hf_weights_many(
        [chunk], "/unused", model_cfg, ps, participating_group=participating_group
    )

    assert calls == [
        (chunk, "/unused", model_cfg, ps, participating_group),
        ([chunk], "/unused", model_cfg, ps, participating_group),
    ]


def test_atomic_hf_copy_requires_persistent_but_not_derived_buffers() -> None:
    from megatron.lite.primitive.ckpt.hf_weights import copy_hf_state_atomically

    model = _weight_module((2, 2))
    model.register_buffer("schema", torch.zeros(2, dtype=torch.int64), persistent=True)
    model.register_buffer("cache", torch.ones(2), persistent=False)

    with pytest.raises(RuntimeError, match=r"persistent_buffers=\['schema'\]"):
        copy_hf_state_atomically(
            model, {"weight": torch.ones(2, 2)}, context="test HF load"
        )

    copy_hf_state_atomically(
        model,
        {"weight": torch.ones(2, 2), "schema": torch.tensor([3, 4], dtype=torch.int64)},
        context="test HF load",
    )
    torch.testing.assert_close(model.cache, torch.ones(2), atol=0.0, rtol=0.0)


def test_atomic_hf_copy_rolls_back_on_unexpected_copy_failure(monkeypatch) -> None:
    from megatron.lite.primitive.ckpt import hf_weights

    model = _two_parameter_model()
    model.first.weight.data.fill_(11.0)
    model.last.weight.data.fill_(13.0)
    before = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }
    real_copy = hf_weights._copy_tensor_for_hf_load
    calls = 0

    def fail_second_copy(target: torch.Tensor, source: torch.Tensor) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected copy failure")
        real_copy(target, source)

    monkeypatch.setattr(hf_weights, "_copy_tensor_for_hf_load", fail_second_copy)
    with pytest.raises(
        RuntimeError, match=r"atomic copy failed.*injected copy failure"
    ):
        hf_weights.copy_hf_state_atomically(
            model,
            {
                "first.weight": torch.full((2, 2), 17.0),
                "last.weight": torch.full((2, 2), 19.0),
            },
            context="test HF load",
        )

    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name], atol=0.0, rtol=0.0)


def test_atomic_hf_vpp_preflight_rejects_later_chunk_before_any_mutation() -> None:
    from megatron.lite.primitive.ckpt.hf_weights import copy_hf_states_atomically

    first = _weight_module((2, 2))
    second = _weight_module((2, 2))
    first.weight.data.fill_(11.0)
    second.weight.data.fill_(13.0)

    with pytest.raises(RuntimeError, match=r"preflight failed.*chunk1"):
        copy_hf_states_atomically(
            [(first, {"weight": torch.full((2, 2), 17.0)}), (second, {})],
            context="VPP HF load",
        )

    torch.testing.assert_close(
        first.weight, torch.full((2, 2), 11.0), atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        second.weight, torch.full((2, 2), 13.0), atol=0.0, rtol=0.0
    )


def test_atomic_hf_vpp_copy_failure_rolls_back_earlier_chunks(monkeypatch) -> None:
    from megatron.lite.primitive.ckpt import hf_weights

    first = _weight_module((2, 2))
    second = _weight_module((2, 2))
    first.weight.data.fill_(11.0)
    second.weight.data.fill_(13.0)
    real_copy = hf_weights._copy_tensor_for_hf_load
    calls = 0

    def fail_second_chunk(target: torch.Tensor, source: torch.Tensor) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected VPP copy failure")
        real_copy(target, source)

    monkeypatch.setattr(hf_weights, "_copy_tensor_for_hf_load", fail_second_chunk)
    with pytest.raises(RuntimeError, match=r"atomic copy failed.*chunk1.weight"):
        hf_weights.copy_hf_states_atomically(
            [
                (first, {"weight": torch.full((2, 2), 17.0)}),
                (second, {"weight": torch.full((2, 2), 19.0)}),
            ],
            context="VPP HF load",
        )

    torch.testing.assert_close(
        first.weight, torch.full((2, 2), 11.0), atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        second.weight, torch.full((2, 2), 13.0), atol=0.0, rtol=0.0
    )


def _gloo_corrupt_one_stage_worker(rank: int, init_path: str) -> None:
    from megatron.lite.primitive.ckpt import hf_weights

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=20),
    )
    try:

        def build_state() -> dict[str, torch.Tensor]:
            if rank == 1:
                raise KeyError("rank-1 checkpoint shard is corrupt")
            return {"weight": torch.ones(2, 2)}

        try:
            hf_weights.materialize_hf_load_state(
                build_state, context="distributed test load"
            )
        except RuntimeError as exc:
            assert "rank-1 checkpoint shard is corrupt" in str(exc)
        else:
            raise AssertionError("rank-local materialization error was not propagated")

        model = _weight_module((2, 2))
        loaded = {"weight": torch.ones(2, 2) if rank == 0 else torch.ones(1)}
        try:
            hf_weights.copy_hf_state_atomically(
                model, loaded, context="distributed test load"
            )
        except RuntimeError as exc:
            assert "checkpoint shape mismatch" in str(exc)
        else:
            raise AssertionError("rank-local preflight error was not propagated")
        torch.testing.assert_close(
            model.weight.detach(), torch.zeros(2, 2), atol=0.0, rtol=0.0
        )

        # Both ranks pass preflight. Rank 1 then fails its second physical copy;
        # rank 0 must learn about that failure and roll back its fully-copied
        # model rather than keeping a mixed distributed checkpoint state.
        model = _two_parameter_model()
        model.first.weight.data.fill_(5.0)
        model.last.weight.data.fill_(6.0)
        before = {
            name: tensor.detach().clone() for name, tensor in model.state_dict().items()
        }
        real_copy = hf_weights._copy_tensor_for_hf_load
        calls = 0

        def fail_rank_one_second_copy(
            target: torch.Tensor, source: torch.Tensor
        ) -> None:
            nonlocal calls
            calls += 1
            if rank == 1 and calls == 2:
                raise RuntimeError("rank-1 injected copy failure")
            real_copy(target, source)

        hf_weights._copy_tensor_for_hf_load = fail_rank_one_second_copy
        try:
            try:
                hf_weights.copy_hf_state_atomically(
                    model,
                    {
                        "first.weight": torch.full((2, 2), 7.0),
                        "last.weight": torch.full((2, 2), 8.0),
                    },
                    context="distributed test load",
                )
            except RuntimeError as exc:
                assert "rank-1 injected copy failure" in str(exc)
            else:
                raise AssertionError("rank-local copy error was not propagated")
        finally:
            hf_weights._copy_tensor_for_hf_load = real_copy
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, before[name], atol=0.0, rtol=0.0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is unavailable")
def test_atomic_hf_load_corrupt_one_stage_gloo_consensus_no_hang(tmp_path) -> None:
    init_path = str(tmp_path / "gloo-init")
    ctx = mp.get_context("spawn")
    processes = [
        ctx.Process(target=_gloo_corrupt_one_stage_worker, args=(rank, init_path))
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
    assert not alive, "distributed HF load consensus hung"
    assert [process.exitcode for process in processes] == [0, 0]


def _gloo_vpp_copy_failure_worker(rank: int, init_path: str) -> None:
    from megatron.lite.primitive.ckpt import hf_weights

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=20),
    )
    try:
        real_copy = hf_weights._copy_tensor_for_hf_load

        # Exercise both possible failing ranks in one process-group lifetime.
        # The injected failure is in chunk 2, after every rank has already
        # mutated chunks 0 and 1. The transaction must restore all chunks on
        # both ranks before it reports the peer failure.
        for failing_rank in (0, 1):
            chunks = [_weight_module((2, 2)) for _ in range(3)]
            before: list[torch.Tensor] = []
            model_states: list[tuple[nn.Module, dict[str, torch.Tensor]]] = []
            for chunk_idx, chunk in enumerate(chunks):
                baseline = float(100 * rank + 10 * failing_rank + chunk_idx)
                chunk.weight.data.fill_(baseline)
                before.append(chunk.weight.detach().clone())
                model_states.append(
                    (
                        chunk,
                        {
                            "weight": torch.full(
                                (2, 2), 1000.0 + baseline, dtype=torch.float32
                            )
                        },
                    )
                )

            calls = 0

            def fail_later_chunk(target: torch.Tensor, source: torch.Tensor) -> None:
                nonlocal calls
                calls += 1
                if rank == failing_rank and calls == 3:
                    raise RuntimeError(
                        f"rank-{failing_rank} injected VPP chunk-2 copy failure"
                    )
                real_copy(target, source)

            hf_weights._copy_tensor_for_hf_load = fail_later_chunk
            try:
                try:
                    hf_weights.copy_hf_states_atomically(
                        model_states,
                        context=f"distributed VPP test load failing rank {failing_rank}",
                    )
                except RuntimeError as exc:
                    message = str(exc)
                    assert f"rank-{failing_rank} injected VPP" in message
                    assert "copying chunk2.weight" in message
                else:
                    raise AssertionError(
                        "rank-local later-chunk copy error was not propagated"
                    )
            finally:
                hf_weights._copy_tensor_for_hf_load = real_copy

            local_restored = []
            for chunk_idx, chunk in enumerate(chunks):
                torch.testing.assert_close(
                    chunk.weight.detach(), before[chunk_idx], atol=0.0, rtol=0.0
                )
                local_restored.append(chunk.weight.detach().clone())

            # A collective readback explicitly covers both ranks x all chunks.
            gathered: list[list[torch.Tensor] | None] = [None, None]
            dist.all_gather_object(gathered, local_restored)
            for peer_rank, peer_chunks in enumerate(gathered):
                assert peer_chunks is not None
                for chunk_idx, restored in enumerate(peer_chunks):
                    expected = torch.full(
                        (2, 2), float(100 * peer_rank + 10 * failing_rank + chunk_idx)
                    )
                    torch.testing.assert_close(restored, expected, atol=0.0, rtol=0.0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is unavailable")
def test_atomic_hf_vpp_copy_failure_gloo_rolls_back_all_chunks_on_all_ranks(
    tmp_path,
) -> None:
    init_path = str(tmp_path / "gloo-vpp-init")
    ctx = mp.get_context("spawn")
    processes = [
        ctx.Process(target=_gloo_vpp_copy_failure_worker, args=(rank, init_path))
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
    assert not alive, "distributed VPP HF load rollback consensus hung"
    assert [process.exitcode for process in processes] == [0, 0]


def _run_four_rank_gloo_worker(worker, init_path: str) -> None:
    ctx = mp.get_context("spawn")
    processes = [
        ctx.Process(target=worker, args=(rank, init_path)) for rank in range(4)
    ]
    for process in processes:
        process.start()

    deadline = time.monotonic() + 30
    for process in processes:
        process.join(timeout=max(0.0, deadline - time.monotonic()))
    alive = [process for process in processes if process.is_alive()]
    for process in alive:
        process.terminate()
        process.join(timeout=5)

    assert not alive, "independent subgroup HF load consensus hung"
    assert [process.exitcode for process in processes] == [0, 0, 0, 0]


def _gloo_independent_success_subgroups_worker(rank: int, init_path: str) -> None:
    from megatron.lite.primitive.ckpt import hf_weights

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=4,
        timeout=datetime.timedelta(seconds=20),
    )
    try:
        # Every WORLD rank must create groups in the same order.  Each rank then
        # enters collectives only on the subgroup to which it belongs.
        groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
        subgroup_index = rank // 2
        participating_group = groups[subgroup_index]
        model = _weight_module((2, 2))

        # The second subgroup performs one extra transaction.  A hidden WORLD
        # collective would require ranks 0/1 to enter that operation and either
        # hang or fail when they independently finish after the first load.
        for step in range(subgroup_index + 1):
            expected = torch.full((2, 2), float(10 * subgroup_index + step + 1))
            hf_weights.copy_hf_state_atomically(
                model,
                {"weight": expected.clone()},
                context=f"subgroup {subgroup_index} success load {step}",
                participating_group=participating_group,
            )
            torch.testing.assert_close(
                model.weight.detach(), expected, atol=0.0, rtol=0.0
            )

        dist.barrier(group=participating_group)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is unavailable")
def test_atomic_hf_load_independent_success_subgroups_do_not_wait_for_world(
    tmp_path,
) -> None:
    _run_four_rank_gloo_worker(
        _gloo_independent_success_subgroups_worker,
        str(tmp_path / "gloo-independent-success-init"),
    )


def _gloo_failure_isolated_from_success_subgroup_worker(
    rank: int, init_path: str
) -> None:
    from megatron.lite.primitive.ckpt import hf_weights

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=4,
        timeout=datetime.timedelta(seconds=20),
    )
    try:
        groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
        subgroup_index = rank // 2
        participating_group = groups[subgroup_index]
        model = _weight_module((2, 2))
        baseline = torch.full((2, 2), float(rank + 1))
        model.weight.data.copy_(baseline)
        expected = torch.full((2, 2), float(rank + 101))
        real_copy = hf_weights._copy_tensor_for_hf_load

        def fail_on_rank_zero(target: torch.Tensor, source: torch.Tensor) -> None:
            if rank == 0:
                raise RuntimeError("rank-0 subgroup-local injected copy failure")
            real_copy(target, source)

        if subgroup_index == 0:
            hf_weights._copy_tensor_for_hf_load = fail_on_rank_zero
        try:
            if subgroup_index == 0:
                with pytest.raises(
                    RuntimeError, match="rank-0 subgroup-local injected copy failure"
                ):
                    hf_weights.copy_hf_state_atomically(
                        model,
                        {"weight": expected},
                        context="failing subgroup load",
                        participating_group=participating_group,
                    )
                torch.testing.assert_close(
                    model.weight.detach(), baseline, atol=0.0, rtol=0.0
                )
            else:
                hf_weights.copy_hf_state_atomically(
                    model,
                    {"weight": expected},
                    context="successful subgroup load",
                    participating_group=participating_group,
                )
                torch.testing.assert_close(
                    model.weight.detach(), expected, atol=0.0, rtol=0.0
                )
        finally:
            hf_weights._copy_tensor_for_hf_load = real_copy

        # No WORLD synchronization is needed for either subgroup to exit.
        dist.barrier(group=participating_group)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is unavailable")
def test_atomic_hf_load_failure_consensus_is_isolated_from_success_subgroup(
    tmp_path,
) -> None:
    _run_four_rank_gloo_worker(
        _gloo_failure_isolated_from_success_subgroup_worker,
        str(tmp_path / "gloo-failure-isolation-init"),
    )
