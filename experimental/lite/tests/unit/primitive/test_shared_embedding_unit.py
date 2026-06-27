# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import datetime
import time
import traceback
import uuid
from collections.abc import Callable
from queue import Empty
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

import megatron.lite.primitive.parallel.shared_embedding as shared_embedding
from megatron.lite.primitive.parallel.shared_embedding import (
    allreduce_mtp_embedding_grads,
    synchronize_mtp_embedding_parameters,
    validate_mtp_embedding_parameter_replicas,
)
from megatron.lite.primitive.parallel.state import (
    ParallelState,
    init_mtp_embedding_group,
    init_parallel,
)

pytestmark = pytest.mark.mlite


@pytest.fixture(autouse=True)
def _mock_collective_tests_start_uninitialized(monkeypatch) -> None:
    # Phase-A may invoke this unit file from an already-initialized outer rank.
    # Mock process groups must still use the deterministic emulated-Gloo path.
    # Tests that exercise real initialization override this locally or spawn a
    # clean child interpreter.
    monkeypatch.setattr(dist, "is_initialized", lambda: False)


class _EmbeddingOwner(nn.Module):
    def __init__(
        self,
        *,
        canonical_attr: str | None = None,
        canonical_weight: torch.Tensor | None = None,
        mtp_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if canonical_attr is not None:
            if canonical_weight is None:
                # Last-stage models retain the canonical attribute name even
                # though they do not own the first-stage parameter.
                setattr(self, canonical_attr, None)
            else:
                setattr(self, canonical_attr, _Embedding(canonical_weight))
        if mtp_weight is not None:
            self.mtp_embed = _Embedding(mtp_weight)


class _Embedding(nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            weight.shape[0],
            weight.shape[1],
            device=weight.device,
            dtype=weight.dtype,
        )
        with torch.no_grad():
            self.embedding.weight.copy_(weight)


def _boundary_states(group: object) -> tuple[ParallelState, ParallelState]:
    common = {
        "pp_size": 2,
        "embedding_group": group,
        "embedding_global_ranks": [3, 11],
    }
    return (
        ParallelState(
            **common,
            pp_rank=0,
            pp_is_first=True,
            pp_is_last=False,
        ),
        ParallelState(
            **common,
            pp_rank=1,
            pp_is_first=False,
            pp_is_last=True,
        ),
    )


def _run_for_both_boundaries(
    fn: Callable,
    first: nn.Module,
    last: nn.Module,
    first_ps: ParallelState,
    last_ps: ParallelState,
) -> None:
    fn([first], first_ps, enabled=True)
    fn([last], last_ps, enabled=True)


def _prime_shared_embedding_cache(
    monkeypatch,
    first: nn.Module,
    last: nn.Module,
    first_ps: ParallelState,
    last_ps: ParallelState,
) -> None:
    """Run the real initialization protocol with an emulated matching pair."""
    reduced: list[torch.Tensor] = []

    def fake_all_gather(
        outputs: list[torch.Tensor], tensor: torch.Tensor, *, group: object
    ) -> None:
        assert group is first_ps.embedding_group
        assert tensor.dtype == torch.int64
        for output in outputs:
            output.copy_(tensor)

    def fake_all_reduce(tensor: torch.Tensor, *, group: object) -> None:
        if group is torch.distributed.group.WORLD:
            assert tensor.dtype == torch.int32
            tensor.zero_()
            return
        assert group is first_ps.embedding_group
        reduced.append(tensor)
        if len(reduced) == 2:
            total = sum((item.clone() for item in reduced), torch.zeros_like(tensor))
            for item in reduced:
                item.copy_(total)

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    _run_for_both_boundaries(
        synchronize_mtp_embedding_parameters,
        first,
        last,
        first_ps,
        last_ps,
    )
    assert first_ps.mtp_embedding_preflight_cache is not None
    assert last_ps.mtp_embedding_preflight_cache is not None


def _distributed_shared_embedding_worker(
    rank: int,
    world_size: int,
    init_method: str,
    case: str,
    results,
) -> None:
    """Real Gloo endpoint used by the no-hang metadata regression test."""
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=20),
        )
        first_rank = 0
        last_rank = world_size - 1
        boundary = rank in {first_rank, last_rank}
        pair_group = (
            dist.new_group(ranks=[first_rank, last_rank])
            if world_size > 2
            else dist.group.WORLD
        )
        if case == "shape":
            weight = torch.zeros(3 + rank, 2, dtype=torch.float32)
        elif case == "dtype":
            dtype = torch.float32 if rank == 0 else torch.float64
            weight = torch.zeros(3, 2, dtype=dtype)
        else:
            assert case in {
                "missing_cache",
                "asymmetric_cache",
                "missing_group",
                "missing_group_pp4",
            }
            weight = torch.zeros(3, 2, dtype=torch.float32)
        if rank == first_rank:
            model = _EmbeddingOwner(canonical_attr="embed", canonical_weight=weight)
        elif rank == last_rank:
            model = _EmbeddingOwner(mtp_weight=weight)
        else:
            model = _EmbeddingOwner()
        missing_group_rank = (
            case in {"missing_group", "missing_group_pp4"} and rank == last_rank
        )
        ps = ParallelState(
            pp_size=world_size,
            pp_rank=rank,
            pp_is_first=rank == first_rank,
            pp_is_last=rank == last_rank,
            pp_global_ranks=list(range(world_size)),
            embedding_group=(
                None if missing_group_rank or not boundary else pair_group
            ),
            embedding_global_ranks=([first_rank, last_rank] if boundary else None),
            embedding_groups_initialized=True,
        )
        try:
            if case in {"missing_cache", "missing_group", "missing_group_pp4"}:
                allreduce_mtp_embedding_grads([model], ps, enabled=True)
            elif case == "asymmetric_cache":
                synchronize_mtp_embedding_parameters([model], ps, enabled=True)
                local_weight = (
                    model.embed.embedding.weight
                    if rank == 0
                    else model.mtp_embed.embedding.weight
                )
                local_weight.grad = torch.ones_like(local_weight)
                if rank == 1:
                    ps.mtp_embedding_preflight_cache = None
                allreduce_mtp_embedding_grads([model], ps, enabled=True)
            else:
                synchronize_mtp_embedding_parameters([model], ps, enabled=True)
        except RuntimeError as exc:
            results.put((rank, "runtime_error", str(exc)))
        else:
            results.put((rank, "unexpected_success", ""))
    except BaseException:  # pragma: no cover - surfaced with the remote traceback
        results.put((rank, "worker_error", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_real_gloo_case(
    tmp_path, case: str, *, world_size: int = 2
) -> dict[int, tuple[str, str]]:
    init_method = (tmp_path / f"gloo-{case}-{uuid.uuid4().hex}").as_uri()
    context = mp.get_context("spawn")
    results = context.Queue()
    processes = [
        context.Process(
            target=_distributed_shared_embedding_worker,
            args=(rank, world_size, init_method, case, results),
        )
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()

    deadline = time.monotonic() + 30
    for process in processes:
        process.join(max(0.0, deadline - time.monotonic()))
    hung = [process.pid for process in processes if process.is_alive()]
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(5)
    if hung:
        pytest.fail(f"Gloo metadata preflight hung worker processes: {hung}")

    payloads = []
    try:
        for _ in range(world_size):
            payloads.append(results.get(timeout=5))
    except Empty:
        pytest.fail(
            "Gloo metadata preflight workers exited without both rank results; "
            f"exitcodes={[process.exitcode for process in processes]}"
        )
    finally:
        results.close()
        results.join_thread()

    assert [process.exitcode for process in processes] == [0] * world_size
    by_rank = {rank: (status, message) for rank, status, message in payloads}
    assert set(by_rank) == set(range(world_size))
    return by_rank


@pytest.mark.parametrize("canonical_attr", ["embed", "embed_tokens"])
def test_shared_embedding_initialization_broadcasts_first_stage_and_marks_replica(
    monkeypatch, canonical_attr: str
) -> None:
    embedding_group = object()
    first_ps, last_ps = _boundary_states(embedding_group)
    canonical = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    first = _EmbeddingOwner(
        canonical_attr=canonical_attr,
        canonical_weight=canonical,
    )
    last = _EmbeddingOwner(
        canonical_attr=canonical_attr,
        mtp_weight=torch.full_like(canonical, -9.0),
    )
    reduced: list[torch.Tensor] = []
    collective_order: list[str] = []

    def fake_all_gather(
        outputs: list[torch.Tensor], tensor: torch.Tensor, *, group: object
    ) -> None:
        assert group is embedding_group
        assert tensor.dtype == torch.int64
        collective_order.append("pair_parameter_metadata")
        for output in outputs:
            output.copy_(tensor)

    def fake_all_reduce(tensor: torch.Tensor, *, group: object) -> None:
        if group is torch.distributed.group.WORLD:
            assert tensor.dtype == torch.int32
            collective_order.append("world")
            return
        assert group is embedding_group
        collective_order.append("pair_data")
        reduced.append(tensor)
        if len(reduced) == 2:
            total = sum((item.clone() for item in reduced), torch.zeros_like(tensor))
            for item in reduced:
                item.copy_(total)

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    _run_for_both_boundaries(
        synchronize_mtp_embedding_parameters,
        first,
        last,
        first_ps,
        last_ps,
    )

    assert (
        collective_order
        == [
            "world",
            "pair_parameter_metadata",
            "world",
            "pair_data",
        ]
        * 2
    )
    first_weight = getattr(first, canonical_attr).embedding.weight
    last_weight = last.mtp_embed.embedding.weight
    torch.testing.assert_close(first_weight, canonical, atol=0.0, rtol=0.0)
    torch.testing.assert_close(last_weight, canonical, atol=0.0, rtol=0.0)
    assert first_weight.shared_embedding is True
    assert last_weight.shared_embedding is True
    assert not getattr(first_weight, "shared", False)
    assert last_weight.shared is True
    assert last._mlite_tied_checkpoint_keys == {
        "mtp_embed.embedding.weight": f"{canonical_attr}.embedding.weight"
    }
    assert first_ps.mtp_embedding_preflight_cache is not None
    assert last_ps.mtp_embedding_preflight_cache is not None


def test_shared_embedding_gradient_allreduce_sums_main_and_parameter_grads(
    monkeypatch,
) -> None:
    embedding_group = object()
    first_ps, last_ps = _boundary_states(embedding_group)
    zeros = torch.zeros(3, 2)
    first = _EmbeddingOwner(canonical_attr="embed", canonical_weight=zeros)
    last = _EmbeddingOwner(mtp_weight=zeros)
    _prime_shared_embedding_cache(
        monkeypatch,
        first,
        last,
        first_ps,
        last_ps,
    )
    first_weight = first.embed.embedding.weight
    last_weight = last.mtp_embed.embedding.weight
    first_weight.grad = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    last_weight.main_grad = torch.tensor([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
    last_weight.grad = torch.full_like(last_weight, -99.0)
    expected = first_weight.grad.clone() + last_weight.main_grad.clone()
    collective_order: list[str] = []

    def fake_all_gather(
        outputs: list[torch.Tensor], tensor: torch.Tensor, *, group: object
    ) -> None:
        assert group is embedding_group
        assert tensor.dtype == torch.int64
        collective_order.append("pair_gradient_metadata")
        for output in outputs:
            output.copy_(tensor)

    def fake_all_reduce(tensor: torch.Tensor, *, group: object) -> None:
        if group is torch.distributed.group.WORLD:
            assert tensor.dtype == torch.int32
            collective_order.append("world")
            return
        assert group is embedding_group
        collective_order.append("pair_data")
        tensor.copy_(expected)

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    _run_for_both_boundaries(
        allreduce_mtp_embedding_grads,
        first,
        last,
        first_ps,
        last_ps,
    )

    assert (
        collective_order
        == [
            "world",
            "pair_gradient_metadata",
            "world",
            "pair_data",
        ]
        * 2
    )
    torch.testing.assert_close(first_weight.grad, expected, atol=0.0, rtol=0.0)
    torch.testing.assert_close(last_weight.main_grad, expected, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        last_weight.grad,
        torch.full_like(last_weight, -99.0),
        atol=0.0,
        rtol=0.0,
    )


def test_incomplete_shared_embedding_grads_fail_both_ranks_before_data_allreduce(
    monkeypatch,
) -> None:
    embedding_group = object()
    first_ps, last_ps = _boundary_states(embedding_group)
    zeros = torch.zeros(3, 2)
    first = _EmbeddingOwner(canonical_attr="embed", canonical_weight=zeros)
    last = _EmbeddingOwner(mtp_weight=zeros)
    _prime_shared_embedding_cache(
        monkeypatch,
        first,
        last,
        first_ps,
        last_ps,
    )
    first.embed.embedding.weight.grad = torch.ones_like(first.embed.embedding.weight)
    collective_order: list[str] = []

    def fake_all_gather(
        outputs: list[torch.Tensor], tensor: torch.Tensor, *, group: object
    ) -> None:
        assert group is embedding_group
        assert tensor.dtype == torch.int64
        collective_order.append("pair_gradient_metadata")
        for output in outputs:
            output.copy_(tensor)
        # Always expose the real 1/2 pair state to both emulated endpoints.
        outputs[0][1] = 1
        outputs[1].zero_()
        outputs[1][0] = tensor[0]
        outputs[1][2] = 1

    def fake_all_reduce(tensor: torch.Tensor, *, group: object) -> None:
        if group is torch.distributed.group.WORLD:
            assert tensor.dtype == torch.int32
            collective_order.append("world")
            return
        raise AssertionError("gradient data collective must not run")

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    for model, ps in ((first, first_ps), (last, last_ps)):
        with pytest.raises(
            RuntimeError,
            match=r"gradient metadata preflight failed.*local_pair_presence=1/2",
        ):
            allreduce_mtp_embedding_grads([model], ps, enabled=True)

    assert (
        collective_order
        == [
            "world",
            "pair_gradient_metadata",
            "world",
        ]
        * 2
    )


def test_absent_shared_embedding_grads_return_after_presence_collective(
    monkeypatch,
) -> None:
    embedding_group = object()
    first_ps, last_ps = _boundary_states(embedding_group)
    zeros = torch.zeros(3, 2)
    first = _EmbeddingOwner(canonical_attr="embed", canonical_weight=zeros)
    last = _EmbeddingOwner(mtp_weight=zeros)
    _prime_shared_embedding_cache(
        monkeypatch,
        first,
        last,
        first_ps,
        last_ps,
    )
    collective_order: list[str] = []

    def fake_all_gather(
        outputs: list[torch.Tensor], tensor: torch.Tensor, *, group: object
    ) -> None:
        assert group is embedding_group
        assert tensor.dtype == torch.int64
        collective_order.append("pair_gradient_metadata")
        for output in outputs:
            output.copy_(tensor)

    def fake_all_reduce(tensor: torch.Tensor, *, group: object) -> None:
        if group is torch.distributed.group.WORLD:
            assert tensor.dtype == torch.int32
            collective_order.append("world")
            return
        raise AssertionError("gradient data collective must not run")

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    _run_for_both_boundaries(
        allreduce_mtp_embedding_grads,
        first,
        last,
        first_ps,
        last_ps,
    )

    # A layer can legitimately be unused on both endpoints. Both ranks still
    # compare fixed-size metadata and reach WORLD consensus, then skip data.
    assert (
        collective_order
        == [
            "world",
            "pair_gradient_metadata",
            "world",
        ]
        * 2
    )


@pytest.mark.parametrize(
    ("mismatch", "first_weight", "last_weight"),
    [
        ("shape", torch.zeros(3, 2), torch.zeros(4, 2)),
        (
            "dtype",
            torch.zeros(3, 2, dtype=torch.float32),
            torch.zeros(3, 2, dtype=torch.float64),
        ),
    ],
)
def test_parameter_metadata_mismatch_fails_both_endpoints_before_data_collective(
    monkeypatch,
    mismatch: str,
    first_weight: torch.Tensor,
    last_weight: torch.Tensor,
) -> None:
    embedding_group = object()
    first_ps, last_ps = _boundary_states(embedding_group)
    first = _EmbeddingOwner(canonical_attr="embed", canonical_weight=first_weight)
    last = _EmbeddingOwner(mtp_weight=last_weight)
    collective_order: list[str] = []

    def fake_all_gather(
        outputs: list[torch.Tensor], tensor: torch.Tensor, *, group: object
    ) -> None:
        assert group is embedding_group
        assert tensor.dtype == torch.int64
        collective_order.append("pair_parameter_metadata")
        for output in outputs:
            output.copy_(tensor)
        # Return the same deterministic first/last records to both emulated
        # callers so both observe the cross-rank mismatch.
        outputs[0][2] = 1
        outputs[1][2] = 1
        if mismatch == "shape":
            outputs[0][3] = outputs[1][3] = 2
            outputs[0][8:10] = torch.tensor([3, 2])
            outputs[1][8:10] = torch.tensor([4, 2])
        else:
            outputs[0][4] = 8
            outputs[1][4] = 9

    def fake_all_reduce(tensor: torch.Tensor, *, group: object) -> None:
        assert group is torch.distributed.group.WORLD
        assert tensor.dtype == torch.int32
        collective_order.append("world")

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    for model, ps in ((first, first_ps), (last, last_ps)):
        with pytest.raises(RuntimeError, match="parameter metadata preflight failed"):
            synchronize_mtp_embedding_parameters([model], ps, enabled=True)

    assert (
        collective_order
        == [
            "world",
            "pair_parameter_metadata",
            "world",
        ]
        * 2
    )
    assert first_ps.mtp_embedding_preflight_cache is None
    assert last_ps.mtp_embedding_preflight_cache is None


@pytest.mark.parametrize("mismatch", ["shape", "dtype"])
def test_real_gloo_parameter_metadata_mismatch_fails_both_ranks_without_hang(
    tmp_path,
    mismatch: str,
) -> None:
    # Each outer Phase-A pytest rank receives its own tmp_path. UUID also keeps
    # repeated parameter cases from sharing a FileStore rendezvous path.
    by_rank = _run_real_gloo_case(tmp_path, mismatch)
    for rank in (0, 1):
        status, message = by_rank[rank]
        assert status == "runtime_error", message
        assert "parameter metadata preflight failed" in message


def test_real_gloo_missing_cache_fails_both_ranks_without_hang(tmp_path) -> None:
    by_rank = _run_real_gloo_case(tmp_path, "missing_cache")
    for rank in (0, 1):
        status, message = by_rank[rank]
        assert status == "runtime_error", message
        assert "gradient metadata preflight failed" in message
        assert "local_cache_missing=True" in message


def test_real_gloo_missing_group_fails_all_ranks_before_pair_collective(
    tmp_path,
) -> None:
    by_rank = _run_real_gloo_case(tmp_path, "missing_group_pp4", world_size=4)
    for rank in range(4):
        status, message = by_rank[rank]
        assert status == "runtime_error", message
        assert "gradient group preflight failed" in message


def test_real_gloo_asymmetric_cache_invalidation_fails_without_hang(tmp_path) -> None:
    by_rank = _run_real_gloo_case(tmp_path, "asymmetric_cache")
    for rank in (0, 1):
        status, message = by_rank[rank]
        assert status == "runtime_error", message
        assert "gradient metadata preflight failed" in message
    assert "local_cache_missing=False" in by_rank[0][1]
    assert "local_cache_missing=True" in by_rank[1][1]


def test_gradient_shape_mismatch_fails_both_endpoints_before_data_collective(
    monkeypatch,
) -> None:
    embedding_group = object()
    first_ps, last_ps = _boundary_states(embedding_group)
    zeros = torch.zeros(3, 2)
    first = _EmbeddingOwner(canonical_attr="embed", canonical_weight=zeros)
    last = _EmbeddingOwner(mtp_weight=zeros)
    _prime_shared_embedding_cache(
        monkeypatch,
        first,
        last,
        first_ps,
        last_ps,
    )
    first.embed.embedding.weight.main_grad = torch.ones(3, 2)
    last.mtp_embed.embedding.weight.main_grad = torch.ones(4, 2)
    collective_order: list[str] = []

    def fake_all_gather(
        outputs: list[torch.Tensor], tensor: torch.Tensor, *, group: object
    ) -> None:
        assert group is embedding_group
        assert tensor.dtype == torch.int64
        collective_order.append("pair_gradient_metadata")
        for output in outputs:
            output.copy_(tensor)
            output[1] = 1
            output[2] = 1
            output[3] = 2
        outputs[0][8:10] = torch.tensor([3, 2])
        outputs[1][8:10] = torch.tensor([4, 2])

    def fake_all_reduce(tensor: torch.Tensor, *, group: object) -> None:
        if group is torch.distributed.group.WORLD:
            assert tensor.dtype == torch.int32
            collective_order.append("world")
            return
        raise AssertionError("gradient data collective must not run")

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    for model, ps in ((first, first_ps), (last, last_ps)):
        with pytest.raises(
            RuntimeError,
            match=r"gradient metadata preflight failed.*local_pair_presence=2/2",
        ):
            allreduce_mtp_embedding_grads([model], ps, enabled=True)

    assert collective_order == ["world", "pair_gradient_metadata", "world"] * 2


def test_cached_parameter_signature_change_fails_before_gradient_data_collective(
    monkeypatch,
) -> None:
    embedding_group = object()
    first_ps, last_ps = _boundary_states(embedding_group)
    zeros = torch.zeros(3, 2)
    first = _EmbeddingOwner(canonical_attr="embed", canonical_weight=zeros)
    last = _EmbeddingOwner(mtp_weight=zeros)
    _prime_shared_embedding_cache(
        monkeypatch,
        first,
        last,
        first_ps,
        last_ps,
    )
    # Preserve Parameter identity while changing the facts proven at init. Both
    # endpoints use the same new metadata so only the cached fingerprint can
    # catch this stale-preflight condition.
    first_weight = first.embed.embedding.weight
    last_weight = last.mtp_embed.embedding.weight
    first_weight.data = torch.zeros(4, 2)
    last_weight.data = torch.zeros(4, 2)
    first_weight.main_grad = torch.ones(4, 2)
    last_weight.main_grad = torch.ones(4, 2)
    collective_order: list[str] = []

    def fake_all_gather(
        outputs: list[torch.Tensor], tensor: torch.Tensor, *, group: object
    ) -> None:
        assert group is embedding_group
        collective_order.append("pair_gradient_metadata")
        for output in outputs:
            output.copy_(tensor)

    def fake_all_reduce(tensor: torch.Tensor, *, group: object) -> None:
        if group is dist.group.WORLD:
            collective_order.append("world")
            return
        raise AssertionError("gradient data collective must not run")

    monkeypatch.setattr(dist, "all_gather", fake_all_gather)
    monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)

    for model, ps in ((first, first_ps), (last, last_ps)):
        with pytest.raises(RuntimeError, match="gradient metadata preflight failed"):
            allreduce_mtp_embedding_grads([model], ps, enabled=True)

    assert collective_order == ["world", "pair_gradient_metadata", "world"] * 2


def test_nccl_cpu_parameter_fails_metadata_preflight_before_data_collective(
    monkeypatch,
) -> None:
    embedding_group = object()
    first_ps, last_ps = _boundary_states(embedding_group)
    zeros = torch.zeros(3, 2)
    first = _EmbeddingOwner(canonical_attr="embed", canonical_weight=zeros)
    last = _EmbeddingOwner(mtp_weight=zeros)
    collective_order: list[str] = []

    def fake_backend(group: object) -> str:
        return "gloo" if group is torch.distributed.group.WORLD else "nccl"

    def fake_all_gather(
        outputs: list[torch.Tensor], tensor: torch.Tensor, *, group: object
    ) -> None:
        assert group is embedding_group
        collective_order.append("pair_parameter_metadata")
        for output in outputs:
            output.copy_(tensor)

    def fake_all_reduce(tensor: torch.Tensor, *, group: object) -> None:
        assert group is torch.distributed.group.WORLD
        collective_order.append("world")

    monkeypatch.setattr(shared_embedding, "_group_backend_name", fake_backend)
    # The collectives are emulated on this CPU-only unit-test host. Production
    # chooses CUDA for NCCL control metadata; only the data tensor stays CPU and
    # must therefore be rejected by the encoded compatibility bit.
    monkeypatch.setattr(
        shared_embedding,
        "_control_device_for_group",
        lambda group, model_chunks=(): torch.device("cpu"),
    )
    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    for model, ps in ((first, first_ps), (last, last_ps)):
        with pytest.raises(RuntimeError, match="parameter metadata preflight failed"):
            synchronize_mtp_embedding_parameters([model], ps, enabled=True)

    assert (
        collective_order
        == [
            "world",
            "pair_parameter_metadata",
            "world",
        ]
        * 2
    )


def test_declared_embedding_ranks_must_match_actual_group_membership(
    monkeypatch,
) -> None:
    embedding_group = object()
    first_ps, last_ps = _boundary_states(embedding_group)
    zeros = torch.zeros(3, 2)
    first = _EmbeddingOwner(canonical_attr="embed", canonical_weight=zeros)
    last = _EmbeddingOwner(mtp_weight=zeros)
    current_rank = 3
    collective_order: list[str] = []

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_backend", lambda group: "gloo")
    monkeypatch.setattr(dist, "get_rank", lambda: current_rank)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(
        dist,
        "get_process_group_ranks",
        lambda group: [3, 12],
    )

    def fake_all_reduce(tensor: torch.Tensor, *, group: object) -> None:
        assert group is dist.group.WORLD
        collective_order.append("world")

    def forbidden_all_gather(*args, **kwargs) -> None:
        raise AssertionError("pair metadata must not run for invalid membership")

    monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(dist, "all_gather", forbidden_all_gather)

    for rank, model, ps in (
        (3, first, first_ps),
        (11, last, last_ps),
    ):
        current_rank = rank
        with pytest.raises(RuntimeError, match="group preflight failed"):
            synchronize_mtp_embedding_parameters([model], ps, enabled=True)

    assert collective_order == ["world", "world"]


@pytest.mark.parametrize(
    ("fn", "expected_world_collectives"),
    [
        (synchronize_mtp_embedding_parameters, 2),
        (allreduce_mtp_embedding_grads, 2),
        (validate_mtp_embedding_parameter_replicas, 3),
    ],
)
def test_middle_pipeline_rank_participates_only_in_world_consensus(
    monkeypatch,
    fn: Callable,
    expected_world_collectives: int,
) -> None:
    middle = _EmbeddingOwner()
    ps = ParallelState(
        pp_size=4,
        pp_rank=1,
        pp_is_first=False,
        pp_is_last=False,
        embedding_group=None,
        embedding_global_ranks=None,
    )
    collective_order: list[str] = []

    def fake_all_reduce(tensor: torch.Tensor, *, group: object) -> None:
        assert group is torch.distributed.group.WORLD
        assert tensor.dtype == torch.int32
        collective_order.append("world")

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    if fn is allreduce_mtp_embedding_grads:
        # Initialization caches static parameter facts on every rank. The
        # optimizer hot path revalidates the active group through WORLD before
        # its dynamic pair-metadata -> WORLD sequence.
        synchronize_mtp_embedding_parameters([middle], ps, enabled=True)
        collective_order.clear()

    fn([middle], ps, enabled=True)

    assert collective_order == ["world"] * expected_world_collectives


def test_mtp_embedding_parameter_replica_validation_accepts_exact_equality(
    monkeypatch,
) -> None:
    embedding_group = object()
    first_ps, last_ps = _boundary_states(embedding_group)
    canonical = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    first = _EmbeddingOwner(canonical_attr="embed", canonical_weight=canonical)
    last = _EmbeddingOwner(mtp_weight=canonical.clone())
    collective_order: list[str] = []

    def fake_all_gather(
        outputs: list[torch.Tensor], tensor: torch.Tensor, *, group: object
    ) -> None:
        assert group is embedding_group
        assert tensor.dtype == torch.int64
        collective_order.append("pair_parameter_metadata")
        for output in outputs:
            output.copy_(tensor)

    def fake_all_reduce(tensor: torch.Tensor, *, group: object) -> None:
        assert tensor.dtype == torch.int32
        assert group is torch.distributed.group.WORLD
        collective_order.append("world")
        tensor.zero_()

    def fake_broadcast(
        tensor: torch.Tensor,
        *,
        src: int,
        group: object,
    ) -> None:
        assert src == first_ps.embedding_global_ranks[0]
        assert group is embedding_group
        collective_order.append("pair_broadcast")
        tensor.copy_(canonical)

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(torch.distributed, "broadcast", fake_broadcast)

    _run_for_both_boundaries(
        validate_mtp_embedding_parameter_replicas,
        first,
        last,
        first_ps,
        last_ps,
    )

    assert (
        collective_order
        == [
            "world",
            "pair_parameter_metadata",
            "world",
            "pair_broadcast",
            "world",
        ]
        * 2
    )


def test_save_validation_rechecks_parameters_instead_of_trusting_init_cache(
    monkeypatch,
) -> None:
    embedding_group = object()
    first_ps, last_ps = _boundary_states(embedding_group)
    canonical = torch.zeros(3, 2)
    first = _EmbeddingOwner(canonical_attr="embed", canonical_weight=canonical)
    last = _EmbeddingOwner(mtp_weight=canonical)
    _prime_shared_embedding_cache(
        monkeypatch,
        first,
        last,
        first_ps,
        last_ps,
    )
    # Replace the registered parameter after initialization. A validator that
    # trusted only the static optimizer cache would miss this incompatible
    # replica and could hang in broadcast.
    last.mtp_embed = _Embedding(torch.zeros(4, 2))
    collective_order: list[str] = []

    def fake_all_gather(
        outputs: list[torch.Tensor], tensor: torch.Tensor, *, group: object
    ) -> None:
        assert group is embedding_group
        collective_order.append("pair_parameter_metadata")
        for output in outputs:
            output.copy_(tensor)
            output[2] = 1
            output[3] = 2
        outputs[0][8:10] = torch.tensor([3, 2])
        outputs[1][8:10] = torch.tensor([4, 2])

    def fake_all_reduce(tensor: torch.Tensor, *, group: object) -> None:
        assert group is torch.distributed.group.WORLD
        collective_order.append("world")

    def forbidden_broadcast(*args, **kwargs) -> None:
        raise AssertionError("parameter broadcast must not run")

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(torch.distributed, "broadcast", forbidden_broadcast)

    for model, ps in ((first, first_ps), (last, last_ps)):
        with pytest.raises(RuntimeError, match="parameter metadata preflight failed"):
            validate_mtp_embedding_parameter_replicas([model], ps, enabled=True)

    assert (
        collective_order
        == [
            "world",
            "pair_parameter_metadata",
            "world",
        ]
        * 2
    )


def test_mtp_embedding_parameter_replica_divergence_fails_all_ranks_via_world(
    monkeypatch,
) -> None:
    embedding_group = object()
    first_ps, last_ps = _boundary_states(embedding_group)
    canonical = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    first = _EmbeddingOwner(canonical_attr="embed", canonical_weight=canonical)
    perturbed = canonical.clone()
    perturbed[1, 0] += 0.25
    last = _EmbeddingOwner(mtp_weight=perturbed)
    collective_order: list[str] = []
    world_calls = 0

    def fake_all_gather(
        outputs: list[torch.Tensor], tensor: torch.Tensor, *, group: object
    ) -> None:
        assert group is embedding_group
        assert tensor.dtype == torch.int64
        collective_order.append("pair_parameter_metadata")
        for output in outputs:
            output.copy_(tensor)

    def fake_all_reduce(tensor: torch.Tensor, *, group: object) -> None:
        nonlocal world_calls
        assert tensor.dtype == torch.int32
        if group is torch.distributed.group.WORLD:
            phase = world_calls % 3
            collective_order.append("world")
            # The final WORLD consensus carries the last-stage mismatch to
            # every rank, including the locally-equal canonical owner.
            tensor.fill_(int(phase == 2))
            world_calls += 1
            return
        raise AssertionError("only WORLD error consensus uses all_reduce")

    def fake_broadcast(
        tensor: torch.Tensor,
        *,
        src: int,
        group: object,
    ) -> None:
        assert src == first_ps.embedding_global_ranks[0]
        assert group is embedding_group
        collective_order.append("pair_broadcast")
        tensor.copy_(canonical)

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(torch.distributed, "broadcast", fake_broadcast)

    errors: list[str] = []
    for model, ps in ((first, first_ps), (last, last_ps)):
        with pytest.raises(RuntimeError, match="replica diverged") as exc_info:
            validate_mtp_embedding_parameter_replicas([model], ps, enabled=True)
        errors.append(str(exc_info.value))

    assert "local_max_abs=0.000000e+00" in errors[0]
    assert "local_max_abs=2.500000e-01" in errors[1]
    assert (
        collective_order
        == [
            "world",
            "pair_parameter_metadata",
            "world",
            "pair_broadcast",
            "world",
        ]
        * 2
    )


def test_dist_opt_finalize_runs_dp_sync_before_mtp_pair_sync(monkeypatch) -> None:
    import megatron.lite.primitive.optimizers.megatron_wrap as megatron_wrap

    original = _EmbeddingOwner(
        canonical_attr="embed", canonical_weight=torch.zeros(3, 2)
    )
    wrapped = nn.Module()
    chunks = [original]
    ps = ParallelState(pp_size=2)
    optimizer = object()
    call_order: list[str] = []

    def fake_build_dist_opt_stack(actual_chunks, **kwargs):
        assert actual_chunks == [original]
        return [wrapped], optimizer

    def fake_finalize_dist_opt_grads(actual_chunks, actual_optimizer) -> None:
        assert actual_chunks == [wrapped]
        assert actual_optimizer is optimizer
        call_order.append("dp_finalize")

    def fake_allreduce_mtp_embedding_grads(
        actual_chunks, actual_ps, *, enabled
    ) -> None:
        assert actual_chunks == [wrapped]
        assert actual_ps is ps
        assert enabled is True
        call_order.append("mtp_pair")

    monkeypatch.setattr(
        megatron_wrap,
        "build_dist_opt_stack",
        fake_build_dist_opt_stack,
    )
    monkeypatch.setattr(
        megatron_wrap,
        "finalize_dist_opt_grads",
        fake_finalize_dist_opt_grads,
    )
    monkeypatch.setattr(
        shared_embedding,
        "allreduce_mtp_embedding_grads",
        fake_allreduce_mtp_embedding_grads,
    )

    actual_optimizer, finalize_grads = megatron_wrap.build_dist_opt_training_optimizer(
        chunks,
        model_cfg=SimpleNamespace(),
        impl_cfg=SimpleNamespace(
            optimizer_config=SimpleNamespace(),
            parallel=SimpleNamespace(),
        ),
        ps=ps,
        model_name="unit",
        mtp_enabled=True,
    )
    assert actual_optimizer is optimizer
    assert chunks == [wrapped]

    finalize_grads()

    assert call_order == ["dp_finalize", "mtp_pair"]


@pytest.mark.parametrize("rank", [0, 1, 3])
def test_init_parallel_defers_mtp_embedding_pair_until_lazy_init(
    monkeypatch,
    rank: int,
) -> None:
    config = SimpleNamespace(tp=1, ep=1, etp=1, cp=1, pp=4)
    created_groups: list[tuple[tuple[int, ...], str | None, object]] = []

    def fake_new_group(ranks, backend=None):
        marker = object()
        created_groups.append((tuple(ranks), backend, marker))
        return marker

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 4)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(torch.distributed, "new_group", fake_new_group)

    ps = init_parallel(config)

    assert ps.embedding_groups_initialized is False
    assert ps.embedding_group is None
    assert ps.embedding_global_ranks is None
    assert not [entry for entry in created_groups if entry[0] == (0, 3)]

    before_lazy_init = len(created_groups)
    init_mtp_embedding_group(ps)
    lazy_groups = created_groups[before_lazy_init:]

    assert len(lazy_groups) == 1
    pair_ranks, backend, pair_group = lazy_groups[0]
    assert pair_ranks == (0, 3)
    assert backend is None
    assert ps.embedding_groups_initialized is True
    if rank in pair_ranks:
        assert ps.embedding_group is pair_group
        assert ps.embedding_global_ranks == [0, 3]
    else:
        assert ps.embedding_group is None
        assert ps.embedding_global_ranks is None

    # Lazy initialization is idempotent and never creates a second pair.
    init_mtp_embedding_group(ps)
    assert len(created_groups) == before_lazy_init + 1


def test_distckpt_tied_embedding_uses_canonical_logical_key_and_replica_id() -> None:
    pytest.importorskip("megatron.core.dist_checkpointing")
    from megatron.lite.primitive.ckpt.distckpt import (
        _model_sharded_state_dict,
        attach_model_sharded_state_dict,
    )

    weight = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    first = _EmbeddingOwner(
        canonical_attr="embed_tokens",
        canonical_weight=weight,
    )
    last = _EmbeddingOwner(mtp_weight=weight)
    last._mlite_tied_checkpoint_keys = {
        "mtp_embed.embedding.weight": "embed_tokens.embedding.weight"
    }
    first_ps = ParallelState(
        pp_size=2,
        pp_rank=0,
        pp_is_first=True,
        pp_is_last=False,
        tp_size=4,
        tp_rank=2,
        dp_cp_rank=3,
    )
    last_ps = ParallelState(
        pp_size=2,
        pp_rank=1,
        pp_is_first=False,
        pp_is_last=True,
        tp_size=4,
        tp_rank=2,
        dp_cp_rank=3,
    )
    attach_model_sharded_state_dict([first], first_ps)
    attach_model_sharded_state_dict([last], last_ps)

    first_state = _model_sharded_state_dict(first)["model_pp0"]
    last_state = _model_sharded_state_dict(last)["model_pp1"]
    canonical = first_state["embed_tokens.embedding.weight"]
    replica = last_state["mtp_embed.embedding.weight"]

    assert canonical.key == replica.key == "model_pp0.embed_tokens.embedding.weight"
    assert canonical.replica_id == (0, 2, 3)
    assert replica.replica_id == (1, 2, 3)


def test_distckpt_save_validates_tied_embedding_before_logical_collapse(
    monkeypatch,
    tmp_path,
) -> None:
    pytest.importorskip("megatron.core.dist_checkpointing")
    import megatron.lite.primitive.ckpt.distckpt as distckpt
    import megatron.lite.primitive.parallel as parallel

    model = _EmbeddingOwner(mtp_weight=torch.zeros(3, 2))
    ps = ParallelState(
        pp_size=2,
        pp_rank=1,
        pp_is_first=False,
        pp_is_last=True,
        embedding_groups_initialized=True,
    )
    model._mlite_dist_opt_parallel_state = ps
    model._mlite_tied_checkpoint_keys = {
        "mtp_embed.embedding.weight": "embed.embedding.weight"
    }
    call_order: list[str] = []

    def fake_validate(chunks, actual_ps, *, enabled: bool) -> None:
        assert chunks == [model]
        assert actual_ps is ps
        assert enabled is True
        call_order.append("validate")

    def fake_model_sharded_state_dict(actual_model):
        assert actual_model is model
        call_order.append("logical_collapse")
        return {"model_pp1": {}}

    def fake_save(state_dict, checkpoint_dir, **kwargs) -> None:
        assert state_dict == {"step": 7, "model_pp1": {}}
        assert checkpoint_dir == str(tmp_path)
        call_order.append("save")

    monkeypatch.setattr(
        parallel,
        "validate_mtp_embedding_parameter_replicas",
        fake_validate,
    )
    monkeypatch.setattr(
        distckpt,
        "_model_sharded_state_dict",
        fake_model_sharded_state_dict,
    )
    monkeypatch.setattr(distckpt.dist_checkpointing, "save", fake_save)

    distckpt.save_dist_opt_checkpoint(
        model,
        optimizer=None,
        step=7,
        checkpoint_dir=str(tmp_path),
        save_model=True,
        save_optimizer=False,
    )

    assert call_order == ["validate", "logical_collapse", "save"]


@pytest.mark.parametrize(
    "canonical_name",
    ["embed.embedding.weight", "embed_tokens.embedding.weight"],
)
def test_hf_export_accepts_equal_mtp_embedding_replica(canonical_name: str) -> None:
    from megatron.lite.primitive.ckpt.hf_weights import (
        _validate_mtp_embedding_replica,
    )

    canonical = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    _validate_mtp_embedding_replica(
        {
            canonical_name: canonical,
            "mtp_embed.embedding.weight": canonical.clone(),
        }
    )


def test_hf_export_rejects_perturbed_mtp_embedding_replica() -> None:
    from megatron.lite.primitive.ckpt.hf_weights import (
        _validate_mtp_embedding_replica,
    )

    canonical = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    replica = canonical.clone()
    replica[1, 0] += 0.25

    with pytest.raises(AssertionError, match="divergent PP MTP embedding replica"):
        _validate_mtp_embedding_replica(
            {
                "embed.embedding.weight": canonical,
                "mtp_embed.embedding.weight": replica,
            }
        )


def test_hf_export_rejects_mtp_embedding_replica_without_canonical() -> None:
    from megatron.lite.primitive.ckpt.hf_weights import (
        _validate_mtp_embedding_replica,
    )

    with pytest.raises(AssertionError, match="without exactly one canonical"):
        _validate_mtp_embedding_replica(
            {"mtp_embed.embedding.weight": torch.zeros(3, 2)}
        )
