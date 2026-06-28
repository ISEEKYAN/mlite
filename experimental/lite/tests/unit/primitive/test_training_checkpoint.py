# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import copy
import datetime
import hashlib
import multiprocessing as mp
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.distributed.tensor import Replicate, Shard

pytest.importorskip("megatron.core.dist_checkpointing")

from megatron.core.dist_checkpointing.strategies.torch import (
    _replace_state_dict_keys_with_sharded_keys,
)
from megatron.lite.primitive.ckpt import dcp
from megatron.lite.primitive.ckpt.distckpt import (
    _iter_sharded_bases,
    _load_model_state_dict,
    _model_sharded_state_dict,
    _rank_offsets_and_replica_id,
    _sharded_logical_keys,
    _single_or_all_model_state,
    _synchronize_native_optimizer_steps,
    attach_model_sharded_state_dict,
    load_dist_opt_checkpoint,
    save_dist_opt_checkpoint,
)
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.protocols import (
    default_expert_classifier,
    default_placement_fn,
)
from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime
from megatron.lite.runtime.contracts.handle import ModelHandle


def _assert_state_equal(actual, expected) -> None:
    if torch.is_tensor(expected):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            _assert_state_equal(actual[key], value)
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_state_equal(actual_item, expected_item)
    else:
        assert actual == expected


def _gloo_common_state_semantic_consensus_worker(rank: int, init_path: str) -> None:
    import megatron.lite.primitive.ckpt.distckpt as distckpt
    import torch.distributed as worker_dist

    worker_dist.init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=20),
    )
    try:
        tensor = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3).t()
        if rank == 0:
            common = {"step": 3, "nested": {"tensor": tensor, "flag": True}}
        else:
            common = {"nested": {"flag": True, "tensor": tensor.clone()}, "step": 3}
        distckpt._assert_world_consensus(
            distckpt._common_state_semantic_fingerprint(common),
            context="semantic common state differs",
        )
        worker_dist.barrier()

        if rank == 1:
            common["nested"]["tensor"][0, 0] += 1
        with pytest.raises(RuntimeError, match="semantic common state differs"):
            distckpt._assert_world_consensus(
                distckpt._common_state_semantic_fingerprint(common),
                context="semantic common state differs",
            )
        worker_dist.barrier()
    finally:
        worker_dist.destroy_process_group()


def test_optimizer_checkpoint_roundtrips_rank_local_state(tmp_path) -> None:
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    loss = model(torch.ones(3, 4)).sum()
    loss.backward()
    optimizer.step()

    expected = copy.deepcopy(optimizer.state_dict())
    dcp._save_optimizer_checkpoint(optimizer, str(tmp_path))

    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                value.zero_()

    dcp._load_optimizer_checkpoint(optimizer, str(tmp_path))

    assert (tmp_path / "optimizer_rank_0.pt").exists()
    _assert_state_equal(optimizer.state_dict(), expected)


def test_optimizer_checkpoint_load_fails_when_requested_state_is_missing(
    tmp_path,
) -> None:
    optimizer = torch.optim.AdamW(torch.nn.Linear(2, 2).parameters(), lr=0.01)

    with pytest.raises(
        FileNotFoundError,
        match="optimizer checkpoint requested by load_optimizer=True is missing",
    ):
        dcp._load_optimizer_checkpoint(optimizer, str(tmp_path))


def test_dcp_model_tensor_contract_includes_only_persistent_buffers() -> None:
    class BufferedModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2))
            self.register_buffer("router_bias", torch.arange(2), persistent=True)
            self.register_buffer("workspace", torch.zeros(4), persistent=False)

    items = dict(dcp._named_model_checkpoint_tensors(BufferedModule()))

    assert set(items) == {"weight", "router_bias"}


class FakeDistOpt:
    def __init__(self):
        self.save_model_sd = None
        self.load_model_sd = None
        self.loaded_state = None

    def sharded_state_dict(self, model_sd, is_loading: bool = False, metadata=None):
        assert metadata == DISTOPT_METADATA
        if is_loading:
            self.load_model_sd = model_sd
        else:
            self.save_model_sd = model_sd
        return {"is_loading": is_loading}

    def load_state_dict(self, state):
        self.loaded_state = (
            state["loaded_state"] if set(state) == {"loaded_state"} else state
        )

    def state_dict(self):
        return {"loaded_state": copy.deepcopy(self.loaded_state)}


class FakeWrapper(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.wrapper_load_called = False

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        self.wrapper_load_called = True
        return super().load_state_dict(*args, **kwargs)


DISTOPT_METADATA = {
    "distrib_optim_sharding_type": "fully_reshardable",
    "distrib_optim_fully_reshardable_mem_efficient": False,
    "chained_optim_avoid_prefix": True,
}


def _fake_checkpoint_metadata(*trees):
    entries = [entry for tree in trees for entry in _iter_sharded_bases(tree)]
    return {
        f"entry_{index}": entry.without_data() for index, entry in enumerate(entries)
    }


def _mock_distckpt_common_state(monkeypatch, state) -> None:
    import megatron.lite.primitive.ckpt.distckpt as distckpt

    monkeypatch.setattr(distckpt, "_load_checkpoint_common_state", lambda _path: state)
    monkeypatch.setattr(
        distckpt,
        "_common_state_file_fingerprint",
        lambda _path: (4096, "fixture-common-state-sha256"),
    )


def test_dist_opt_checkpoint_dispatches_to_mcore_distckpt(
    monkeypatch, tmp_path
) -> None:
    model = torch.nn.Linear(4, 2)
    optimizer = FakeDistOpt()
    ps = ParallelState(pp_rank=1, tp_rank=2, dp_cp_rank=3)
    attach_model_sharded_state_dict([model], ps)
    saved = {}

    def fake_save(state_dict, checkpoint_dir, **kwargs):
        saved["state_dict"] = state_dict
        saved["checkpoint_dir"] = checkpoint_dir
        saved["kwargs"] = kwargs

    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.distckpt.dist_checkpointing.save", fake_save
    )

    dcp.save_training_checkpoint(model, optimizer, 5, str(tmp_path), use_dcp=True)

    model_sd = saved["state_dict"]["model"]
    assert set(model_sd) == {"weight", "bias"}
    assert model_sd["weight"].replica_id == (0, 2, 3)
    assert optimizer.save_model_sd is model_sd
    assert saved["state_dict"]["optimizer"] == {"is_loading": False}
    assert saved["state_dict"]["step"] == 5
    assert saved["checkpoint_dir"] == str(tmp_path / "step_5")
    assert saved["kwargs"]["validate_access_integrity"] is False
    assert saved["kwargs"]["content_metadata"] == DISTOPT_METADATA
    assert not (tmp_path / "step_5" / "optimizer_rank_0.pt").exists()


def test_dist_opt_checkpoint_offsets_cover_tp_pp_ep_etp_topology() -> None:
    ps = ParallelState(
        pp_size=2,
        pp_rank=1,
        tp_size=2,
        tp_rank=1,
        ep_size=2,
        ep_rank=1,
        etp_size=2,
        etp_rank=1,
        dp_size=2,
        dp_rank=0,
        cp_size=1,
        cp_rank=0,
        dp_cp_rank=0,
        expert_dp_size=1,
        expert_dp_rank=0,
    )

    dense_offsets, dense_replica = _rank_offsets_and_replica_id(
        [Replicate(), Replicate(), Replicate(), Shard(0)], ps, expert=False
    )
    expert_offsets, expert_replica = _rank_offsets_and_replica_id(
        [Replicate(), Replicate(), Shard(0), Shard(0)], ps, expert=True
    )

    assert dense_offsets == ((0, 1, 2),)
    assert dense_replica == (0, 0, 0)
    assert expert_offsets == ((0, 3, 4),)
    assert expert_replica == (0, 0, 0)


def test_dist_opt_replica_id_groups_sharded_axes_by_placement() -> None:
    placements = [Replicate(), Replicate(), Replicate(), Shard(0)]
    rank_offsets0, replica_id0 = _rank_offsets_and_replica_id(
        placements, ParallelState(tp_size=2, tp_rank=0), expert=False
    )
    rank_offsets1, replica_id1 = _rank_offsets_and_replica_id(
        placements, ParallelState(tp_size=2, tp_rank=1), expert=False
    )

    assert rank_offsets0 == ((0, 0, 2),)
    assert rank_offsets1 == ((0, 1, 2),)
    assert replica_id0 == replica_id1 == (0, 0, 0)

    expert_offsets, expert_replica_id = _rank_offsets_and_replica_id(
        [Replicate(), Replicate(), Shard(0), Shard(1)],
        ParallelState(ep_size=2, ep_rank=1, etp_size=2, etp_rank=1),
        expert=True,
    )

    assert expert_offsets == ((0, 1, 2), (1, 1, 2))
    assert expert_replica_id == (0, 0, 0)


def test_dist_opt_replica_id_does_not_treat_pp_as_a_replica_axis() -> None:
    rank_offsets, replica_id = _rank_offsets_and_replica_id(
        [Replicate(), Replicate(), Replicate(), Shard(0)],
        ParallelState(pp_size=2, pp_rank=1, tp_size=2, tp_rank=1),
        expert=False,
    )

    assert rank_offsets == ((0, 1, 2),)
    assert replica_id == (0, 0, 0)

    _rank_offsets, replica_id = _rank_offsets_and_replica_id(
        [Replicate(), Replicate(), Replicate(), Replicate()],
        ParallelState(pp_size=2, pp_rank=1, tp_size=2, tp_rank=0),
        expert=False,
    )

    assert replica_id == (0, 0, 0)


def test_dist_opt_pp_rank_one_model_keys_survive_torch_dist_main_replica_filter() -> (
    None
):
    ps = ParallelState(pp_size=2, pp_rank=1, pp_is_first=False, pp_is_last=True)
    model = torch.nn.Linear(4, 2)
    attach_model_sharded_state_dict([model], ps)

    model_sd = _model_sharded_state_dict(model)
    filtered_sd, _flat_mapping, _rename_mapping = (
        _replace_state_dict_keys_with_sharded_keys(
            model_sd, keep_only_main_replica=True
        )
    )

    assert set(filtered_sd) == {"model_pp1.weight", "model_pp1.bias"}


def test_dist_opt_model_state_keys_are_pp_and_vpp_aware() -> None:
    ps = ParallelState(pp_size=2, pp_rank=1, pp_is_first=False, pp_is_last=True)
    single_chunk = torch.nn.Linear(4, 2)
    attach_model_sharded_state_dict([single_chunk], ps)

    single_sd = _model_sharded_state_dict(single_chunk)

    assert set(single_sd) == {"model_pp1"}
    assert set(single_sd["model_pp1"]) == {"weight", "bias"}
    assert single_sd["model_pp1"]["weight"].key == "model_pp1.weight"
    assert _single_or_all_model_state(single_sd) is single_sd

    chunks = [torch.nn.Linear(4, 2), torch.nn.Linear(4, 2)]
    attach_model_sharded_state_dict(chunks, ps)

    vpp_sd = _model_sharded_state_dict(chunks)

    assert set(vpp_sd) == {"model_pp1_vpp0", "model_pp1_vpp1"}
    assert set(vpp_sd["model_pp1_vpp0"]) == {"weight", "bias"}
    assert set(vpp_sd["model_pp1_vpp1"]) == {"weight", "bias"}
    assert vpp_sd["model_pp1_vpp0"]["weight"].key == "model_pp1_vpp0.weight"
    assert vpp_sd["model_pp1_vpp1"]["weight"].key == "model_pp1_vpp1.weight"
    assert _single_or_all_model_state(vpp_sd) is vpp_sd


def test_dist_opt_model_load_rejects_missing_pp_subtree_before_mutation() -> None:
    ps = ParallelState(pp_size=2, pp_rank=1, pp_is_first=False, pp_is_last=True)
    model = torch.nn.Linear(2, 2)
    attach_model_sharded_state_dict([model], ps)
    before = copy.deepcopy(model.state_dict())

    with pytest.raises(
        RuntimeError, match=r"missing required model subtree 'model_pp1'"
    ):
        _load_model_state_dict(model, {"step": 7, "model_pp0": {}})

    _assert_state_equal(model.state_dict(), before)


def test_dist_opt_vpp_load_preflights_every_chunk_before_mutation() -> None:
    ps = ParallelState(pp_size=2, pp_rank=1, pp_is_first=False, pp_is_last=True)
    chunks = [torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)]
    attach_model_sharded_state_dict(chunks, ps)
    before = [copy.deepcopy(chunk.state_dict()) for chunk in chunks]
    first_state = {
        name: torch.full_like(tensor, 17)
        for name, tensor in chunks[0].state_dict().items()
    }

    with pytest.raises(
        RuntimeError, match=r"missing required model subtree 'model_pp1_vpp1'"
    ):
        _load_model_state_dict(chunks, {"step": 7, "model_pp1_vpp0": first_state})

    for chunk, expected in zip(chunks, before, strict=True):
        _assert_state_equal(chunk.state_dict(), expected)


def test_dist_opt_model_load_allows_only_tied_aliases_outside_sharded_template() -> (
    None
):
    class TiedModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.canonical = torch.nn.Parameter(torch.zeros(2))
            self.alias = self.canonical

    model = TiedModule()
    attach_model_sharded_state_dict([model], ParallelState())
    loaded = torch.tensor([3.0, 5.0])

    _load_model_state_dict(model, {"model": {"canonical": loaded}})

    assert model.canonical is model.alias
    torch.testing.assert_close(model.canonical, loaded)


def test_dist_opt_model_load_rejects_incomplete_subtree_before_mutation() -> None:
    model = torch.nn.Linear(2, 2)
    attach_model_sharded_state_dict([model], ParallelState())
    before = copy.deepcopy(model.state_dict())

    with pytest.raises(RuntimeError, match=r"key mismatch: missing=\['bias'\]"):
        _load_model_state_dict(
            model, {"model": {"weight": torch.full_like(model.weight, 9)}}
        )

    _assert_state_equal(model.state_dict(), before)


def test_dist_opt_vpp_rejects_duplicate_logical_shards() -> None:
    ps = ParallelState(pp_size=2, pp_rank=1, pp_is_first=False, pp_is_last=True)
    chunks = [torch.nn.Linear(2, 2, bias=False), torch.nn.Linear(2, 2, bias=False)]
    for chunk in chunks:
        chunk._mlite_tied_checkpoint_keys = {"weight": "embed.embedding.weight"}
    attach_model_sharded_state_dict(chunks, ps)

    with pytest.raises(
        RuntimeError,
        match=(
            r"^distckpt logical shard collision for "
            r"key='model_pp0\.embed\.embedding\.weight', replica_id=\(1, 0, 0\): "
            r"model_pp1_vpp0\.weight and model_pp1_vpp1\.weight$"
        ),
    ):
        _model_sharded_state_dict(chunks)


def test_dist_opt_vpp_mtp_embedding_replica_uses_canonical_first_stage_key() -> None:
    class Embedding(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 4)

    class EmbeddingOwner(torch.nn.Module):
        def __init__(self, attr: str) -> None:
            super().__init__()
            setattr(self, attr, Embedding())

    first_ps = ParallelState(pp_size=2, pp_rank=0, pp_is_first=True, pp_is_last=False)
    first_chunks = [torch.nn.Linear(4, 4), EmbeddingOwner("embed")]
    attach_model_sharded_state_dict(first_chunks, first_ps)
    first_state = _model_sharded_state_dict(first_chunks)
    canonical = first_state["model_pp0_vpp1"]["embed.embedding.weight"]

    last_ps = ParallelState(pp_size=2, pp_rank=1, pp_is_first=False, pp_is_last=True)
    last_chunks = [torch.nn.Linear(4, 4), EmbeddingOwner("mtp_embed")]
    last_chunks[1]._mlite_tied_checkpoint_keys = {
        "mtp_embed.embedding.weight": "embed.embedding.weight"
    }
    attach_model_sharded_state_dict(last_chunks, last_ps)
    last_state = _model_sharded_state_dict(last_chunks)
    replica = last_state["model_pp1_vpp1"]["mtp_embed.embedding.weight"]

    assert canonical.key == replica.key == "model_pp0.embed.embedding.weight"
    assert canonical.replica_id[0] == 0
    assert replica.replica_id[0] == 1


def test_dist_opt_sharded_state_excludes_nonpersistent_buffers() -> None:
    class BufferedModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2))
            self.register_buffer("router_bias", torch.arange(2), persistent=True)
            self.register_buffer("workspace", torch.zeros(4), persistent=False)

    model = BufferedModule()
    attach_model_sharded_state_dict([model], ParallelState())

    model_sd = _model_sharded_state_dict(model)["model"]

    assert set(model_sd) == {"weight", "router_bias"}
    assert "workspace" not in model_sd


def test_dist_opt_save_wraps_model_preflight_failure_before_mcore(
    monkeypatch, tmp_path
) -> None:
    import megatron.lite.primitive.ckpt.distckpt as distckpt

    model = torch.nn.Linear(2, 2, bias=False)
    attach_model_sharded_state_dict([model], ParallelState())

    def fail_model_state(_model):
        raise ValueError("broken VPP metadata")

    def unexpected_save(*_args, **_kwargs):
        raise AssertionError("MCore save must not run after a preflight failure")

    monkeypatch.setattr(distckpt, "_model_sharded_state_dict", fail_model_state)
    monkeypatch.setattr(distckpt.dist_checkpointing, "save", unexpected_save)

    with pytest.raises(
        RuntimeError,
        match=(
            r"^distckpt model state construction failed: ValueError: "
            r"broken VPP metadata$"
        ),
    ):
        distckpt.save_dist_opt_checkpoint(
            model,
            optimizer=None,
            step=3,
            checkpoint_dir=str(tmp_path),
            save_model=True,
            save_optimizer=False,
        )


def test_dist_opt_save_consensus_wraps_checkpoint_directory_failure(
    monkeypatch, tmp_path
) -> None:
    import megatron.lite.primitive.ckpt.distckpt as distckpt

    model = torch.nn.Linear(2, 2, bias=False)
    attach_model_sharded_state_dict([model], ParallelState())

    def fail_makedirs(*_args, **_kwargs):
        raise PermissionError("read-only checkpoint root")

    def unexpected_model_state(*_args, **_kwargs):
        raise AssertionError("model state construction must not run")

    def unexpected_save(*_args, **_kwargs):
        raise AssertionError("MCore save must not run")

    monkeypatch.setattr(distckpt.os, "makedirs", fail_makedirs)
    monkeypatch.setattr(distckpt, "_model_sharded_state_dict", unexpected_model_state)
    monkeypatch.setattr(distckpt.dist_checkpointing, "save", unexpected_save)

    with pytest.raises(
        RuntimeError,
        match=(
            r"^distckpt checkpoint directory creation failed: PermissionError: "
            r"read-only checkpoint root$"
        ),
    ):
        distckpt.save_dist_opt_checkpoint(
            model,
            optimizer=None,
            step=3,
            checkpoint_dir=str(tmp_path),
            save_model=True,
            save_optimizer=False,
        )


def test_distckpt_world_consensus_rejects_rank_metadata_disagreement(
    monkeypatch,
) -> None:
    import megatron.lite.primitive.ckpt.distckpt as distckpt

    monkeypatch.setattr(distckpt.dist, "is_available", lambda: True)
    monkeypatch.setattr(distckpt.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(distckpt.dist, "get_world_size", lambda: 2)

    def fake_distributed_raise_if_error(local_error, *, context):
        assert local_error is None
        assert context.endswith(" exchange failed")

    monkeypatch.setattr(
        distckpt, "_distributed_raise_if_error", fake_distributed_raise_if_error
    )

    def fake_all_gather_object(output, value, *, group):
        assert group is distckpt.dist.group.WORLD
        output[:] = [value, {"step": 8}]

    monkeypatch.setattr(distckpt.dist, "all_gather_object", fake_all_gather_object)

    with pytest.raises(
        RuntimeError,
        match=(
            r"^distckpt common-state metadata differs across ranks: "
            r"rank values=\[\{'step': 7\}, \{'step': 8\}\]$"
        ),
    ):
        distckpt._assert_world_consensus(
            {"step": 7}, context="distckpt common-state metadata differs across ranks"
        )


def test_distckpt_common_state_fingerprint_hashes_exact_file_bytes(tmp_path) -> None:
    import megatron.lite.primitive.ckpt.distckpt as distckpt

    payload = b"common-state\x00with-bf16-and-nested-payload\xff"
    (tmp_path / "common.pt").write_bytes(payload)

    assert distckpt._common_state_file_fingerprint(str(tmp_path)) == (
        len(payload),
        hashlib.sha256(payload).hexdigest(),
    )


def test_distckpt_common_state_semantic_fingerprint_is_canonical_and_exact() -> None:
    import megatron.lite.primitive.ckpt.distckpt as distckpt

    tensor = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3).t()
    left = {
        "step": 7,
        "nested": {
            "tensor": tensor,
            "empty": torch.empty(0, dtype=torch.bfloat16),
            "scalar": torch.tensor(2, dtype=torch.int64),
            "values": [True, 1, 1.0, -0.0],
        },
    }
    right = {
        "nested": {
            "values": [True, 1, 1.0, -0.0],
            "scalar": torch.tensor(2, dtype=torch.int64),
            "empty": torch.empty(0, dtype=torch.bfloat16),
            "tensor": tensor.clone(),
        },
        "step": 7,
    }
    fingerprint = distckpt._common_state_semantic_fingerprint(left)
    assert distckpt._common_state_semantic_fingerprint(right) == fingerprint

    right["nested"]["tensor"][0, 0] += 1
    assert distckpt._common_state_semantic_fingerprint(right) != fingerprint
    right["nested"]["tensor"][0, 0] -= 1
    right["nested"]["values"][-1] = 0.0
    assert distckpt._common_state_semantic_fingerprint(right) != fingerprint
    right["nested"]["values"][-1] = -0.0
    right["nested"]["values"] = tuple(right["nested"]["values"])
    assert distckpt._common_state_semantic_fingerprint(right) != fingerprint


def test_distckpt_common_state_semantic_fingerprint_fails_closed() -> None:
    import megatron.lite.primitive.ckpt.distckpt as distckpt

    cycle = []
    cycle.append(cycle)
    with pytest.raises(ValueError, match="cycle detected"):
        distckpt._common_state_semantic_fingerprint(cycle)
    with pytest.raises(TypeError, match="unsupported common-state value"):
        distckpt._common_state_semantic_fingerprint({"bad": object()})
    with pytest.raises(TypeError, match="unsupported common-state mapping key"):
        distckpt._common_state_semantic_fingerprint({object(): 1})
    sparse = torch.ones(1, 1).to_sparse()
    with pytest.raises(TypeError, match="unsupported common-state tensor"):
        distckpt._common_state_semantic_fingerprint({"sparse": sparse})


@pytest.mark.skipif(
    not torch.distributed.is_gloo_available(), reason="Gloo is unavailable"
)
def test_distckpt_common_state_semantic_world_consensus_gloo(tmp_path) -> None:
    init_path = str(tmp_path / "gloo-common-semantic-init")
    ctx = mp.get_context("spawn")
    processes = [
        ctx.Process(
            target=_gloo_common_state_semantic_consensus_worker, args=(rank, init_path)
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
    assert not alive, "common-state semantic WORLD consensus hung"
    assert [process.exitcode for process in processes] == [0, 0]


def test_distckpt_common_state_preflight_detects_file_change_during_read(
    monkeypatch,
) -> None:
    import megatron.lite.primitive.ckpt.distckpt as distckpt

    fingerprints = iter([(10, "before"), (11, "after")])
    monkeypatch.setattr(
        distckpt, "_common_state_file_fingerprint", lambda _path: next(fingerprints)
    )
    monkeypatch.setattr(
        distckpt,
        "_load_checkpoint_common_state",
        lambda _path: {"step": 1, "content_metadata": DISTOPT_METADATA},
    )

    with pytest.raises(
        RuntimeError, match="common.pt changed while it was being preflighted"
    ):
        distckpt._preflight_distckpt_common_state(
            {"step": 0},
            "/checkpoint",
            load_optimizer=False,
            requested_sharded_keys=set(),
            checkpoint_sharded_keys=set(),
            expected_step=1,
            allow_legacy_checkpoint=False,
        )


def test_distckpt_common_state_revalidation_detects_post_preflight_change(
    monkeypatch,
) -> None:
    import megatron.lite.primitive.ckpt.distckpt as distckpt

    monkeypatch.setattr(
        distckpt, "_common_state_file_fingerprint", lambda _path: (12, "changed")
    )
    with pytest.raises(RuntimeError, match="common.pt changed after preflight"):
        distckpt._revalidate_distckpt_common_state_file("/checkpoint", (12, "original"))


def test_distckpt_mcore_load_binds_only_owner_to_prevalidated_common(
    monkeypatch, tmp_path
) -> None:
    import megatron.lite.primitive.ckpt.distckpt as distckpt
    from megatron.core.dist_checkpointing import serialization

    checkpoint_path = str(tmp_path / "target")
    other_path = str(tmp_path / "other")
    common_state = {"step": 9, "content_metadata": DISTOPT_METADATA}
    original_calls = []

    def original_load_common(path):
        original_calls.append(str(path))
        return {"source": str(path)}

    monkeypatch.setattr(serialization, "load_common", original_load_common)

    def fake_distckpt_load(load_sd, actual_path, **kwargs):
        assert load_sd == {"step": 0}
        assert actual_path == checkpoint_path
        assert kwargs["validate_access_integrity"] is False
        assert serialization.load_common(actual_path) is common_state
        assert serialization.load_common(other_path) == {"source": other_path}
        other_thread_result = []
        thread = threading.Thread(
            target=lambda: other_thread_result.append(
                serialization.load_common(actual_path)
            )
        )
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert other_thread_result == [{"source": actual_path}]
        return {"step": 9}

    monkeypatch.setattr(distckpt.dist_checkpointing, "load", fake_distckpt_load)
    assert distckpt._load_distckpt_with_preloaded_common(
        {"step": 0}, checkpoint_path, common_state, validate_access_integrity=False
    ) == {"step": 9}
    assert serialization.load_common is original_load_common
    assert original_calls == [other_path, checkpoint_path]


def test_distckpt_legacy_model_only_ignores_optimizer_format_metadata() -> None:
    import megatron.lite.primitive.ckpt.distckpt as distckpt

    legacy_metadata = {
        **DISTOPT_METADATA,
        "distrib_optim_sharding_type": "dp_reshardable",
    }
    distckpt._validate_distckpt_content_metadata(
        {"content_metadata": legacy_metadata},
        load_optimizer=False,
        allow_legacy_checkpoint=True,
    )
    with pytest.raises(RuntimeError, match="incompatible with the MLite distckpt"):
        distckpt._validate_distckpt_content_metadata(
            {"content_metadata": legacy_metadata},
            load_optimizer=True,
            allow_legacy_checkpoint=True,
        )
    with pytest.raises(RuntimeError, match="cannot safely infer its sharding format"):
        distckpt._validate_distckpt_content_metadata(
            {}, load_optimizer=True, allow_legacy_checkpoint=True
        )


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        (None, "missing required content_metadata"),
        (
            {**DISTOPT_METADATA, "distrib_optim_sharding_type": "dp_reshardable"},
            "mismatched=\\['distrib_optim_sharding_type'\\]",
        ),
        (
            {**DISTOPT_METADATA, "distrib_optim_fully_reshardable_mem_efficient": 0},
            "mismatched=\\['distrib_optim_fully_reshardable_mem_efficient'\\]",
        ),
        (
            {**DISTOPT_METADATA, "unknown_format_flag": True},
            "unexpected=\\['unknown_format_flag'\\]",
        ),
    ],
)
def test_dist_opt_load_rejects_incompatible_content_metadata_before_mcore_load(
    monkeypatch, tmp_path, metadata, message
) -> None:
    import megatron.lite.primitive.ckpt.distckpt as distckpt

    model = torch.nn.Linear(2, 2)
    attach_model_sharded_state_dict([model], ParallelState())
    before = copy.deepcopy(model.state_dict())
    checkpoint_metadata = _fake_checkpoint_metadata(_model_sharded_state_dict(model))
    common_state = {"step": 4}
    if metadata is not None:
        common_state["content_metadata"] = metadata
    monkeypatch.setattr(
        distckpt, "_load_checkpoint_sharded_metadata", lambda _path: checkpoint_metadata
    )
    _mock_distckpt_common_state(monkeypatch, common_state)

    def unexpected_load(*_args, **_kwargs):
        with torch.no_grad():
            model.weight.fill_(99)
        raise AssertionError("MCore load must not run after metadata mismatch")

    monkeypatch.setattr(distckpt.dist_checkpointing, "load", unexpected_load)

    with pytest.raises(RuntimeError, match=message):
        load_dist_opt_checkpoint(
            model, None, str(tmp_path), load_model=True, load_optimizer=False
        )

    _assert_state_equal(model.state_dict(), before)


def test_dist_opt_load_rejects_optimizer_format_discriminator_before_mcore_load(
    monkeypatch, tmp_path
) -> None:
    import megatron.lite.primitive.ckpt.distckpt as distckpt

    class FormatDistOpt(FakeDistOpt):
        def sharded_state_dict(self, model_sd, is_loading=False, metadata=None):
            super().sharded_state_dict(
                model_sd, is_loading=is_loading, metadata=metadata
            )
            source = next(iter(_iter_sharded_bases(model_sd)))
            return {
                "param_state": replace(
                    source,
                    key=f"optimizer.exp_avg.{source.key}",
                    data=torch.zeros_like(source.data),
                ),
                "param_state_sharding_type": "fully_reshardable",
            }

    model = torch.nn.Linear(2, 2)
    optimizer = FormatDistOpt()
    attach_model_sharded_state_dict([model], ParallelState())
    before = copy.deepcopy(model.state_dict())
    model_sd = _model_sharded_state_dict(model)
    optimizer_sd = optimizer.sharded_state_dict(
        _single_or_all_model_state(model_sd), is_loading=True, metadata=DISTOPT_METADATA
    )
    monkeypatch.setattr(
        distckpt,
        "_load_checkpoint_sharded_metadata",
        lambda _path: _fake_checkpoint_metadata(model_sd, optimizer_sd),
    )
    _mock_distckpt_common_state(
        monkeypatch,
        {
            "step": 4,
            "optimizer": {"param_state_sharding_type": "dp_reshardable"},
            "content_metadata": DISTOPT_METADATA,
        },
    )
    monkeypatch.setattr(
        distckpt.dist_checkpointing,
        "load",
        lambda *_args, **_kwargs: pytest.fail("MCore load must not run"),
    )

    with pytest.raises(
        RuntimeError, match="checkpoint optimizer format discriminator mismatch"
    ):
        load_dist_opt_checkpoint(
            model, optimizer, str(tmp_path), load_model=True, load_optimizer=True
        )

    _assert_state_equal(model.state_dict(), before)


def test_dist_opt_save_rejects_model_keys_in_reserved_optimizer_namespace(
    monkeypatch, tmp_path
) -> None:
    class AmbiguousModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.optimizer = torch.nn.Linear(2, 2, bias=False)

    model = AmbiguousModel()
    attach_model_sharded_state_dict([model], ParallelState())
    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.distckpt.dist_checkpointing.save",
        lambda *_args, **_kwargs: pytest.fail("MCore save must not run"),
    )

    with pytest.raises(
        RuntimeError,
        match=(
            r"distckpt model state construction failed: RuntimeError: model "
            r"checkpoint keys use the reserved 'optimizer\.' prefix"
        ),
    ):
        save_dist_opt_checkpoint(
            model, None, 3, str(tmp_path), save_model=True, save_optimizer=False
        )


def test_dist_opt_checkpoint_loads_from_mcore_distckpt(monkeypatch, tmp_path) -> None:
    wrapped_module = torch.nn.Linear(4, 2)
    model = FakeWrapper(wrapped_module)
    optimizer = FakeDistOpt()
    attach_model_sharded_state_dict([model], ParallelState())
    checkpoint_metadata = _fake_checkpoint_metadata(_model_sharded_state_dict(model))
    expected_weight = torch.full_like(wrapped_module.weight, 3.0)
    expected_bias = torch.full_like(wrapped_module.bias, -2.0)

    def fake_load(sharded_state_dict, checkpoint_dir, **kwargs):
        from megatron.core.dist_checkpointing.validation import StrictHandling

        assert set(sharded_state_dict["model"]) == {"weight", "bias"}
        assert optimizer.load_model_sd is sharded_state_dict["model"]
        assert sharded_state_dict["optimizer"] == {"is_loading": True}
        assert checkpoint_dir == str(tmp_path / "step_5")
        assert kwargs["validate_access_integrity"] is False
        assert kwargs["strict"] is StrictHandling.RAISE_UNEXPECTED
        return {
            "step": 5,
            "model": {"weight": expected_weight, "bias": expected_bias},
            "optimizer": {"loaded": True},
        }

    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.distckpt.dist_checkpointing.load", fake_load
    )
    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.distckpt._load_checkpoint_sharded_metadata",
        lambda _path: checkpoint_metadata,
    )
    _mock_distckpt_common_state(
        monkeypatch,
        {
            "step": 5,
            "optimizer": {"is_loading": False},
            "content_metadata": DISTOPT_METADATA,
        },
    )

    step = dcp.load_training_checkpoint(
        model,
        optimizer,
        str(tmp_path / "step_5"),
        use_dcp=True,
        allow_legacy_checkpoint=True,
    )

    assert step == 5
    assert not model.wrapper_load_called
    torch.testing.assert_close(wrapped_module.weight, expected_weight)
    torch.testing.assert_close(wrapped_module.bias, expected_bias)
    assert optimizer.loaded_state == {"loaded": True}


@pytest.mark.parametrize(
    ("load_model", "load_optimizer"),
    ((True, True), (True, False), (False, True), (False, False)),
)
def test_dist_opt_full_checkpoint_supports_component_partial_loads(
    monkeypatch, tmp_path, load_model: bool, load_optimizer: bool
) -> None:
    from megatron.core.dist_checkpointing.validation import StrictHandling

    class ShardedFakeDistOpt(FakeDistOpt):
        def sharded_state_dict(self, model_sd, is_loading: bool = False, metadata=None):
            super().sharded_state_dict(
                model_sd, is_loading=is_loading, metadata=metadata
            )
            source = next(iter(_iter_sharded_bases(model_sd)))
            return {
                "exp_avg": replace(
                    source,
                    key=f"optimizer.exp_avg.{source.key}",
                    data=torch.zeros_like(source.data),
                )
            }

    model = torch.nn.Linear(2, 2)
    optimizer = ShardedFakeDistOpt()
    attach_model_sharded_state_dict([model], ParallelState())
    model_sd = _model_sharded_state_dict(model)
    optimizer_sd = optimizer.sharded_state_dict(
        _single_or_all_model_state(model_sd), is_loading=True, metadata=DISTOPT_METADATA
    )
    checkpoint_metadata = _fake_checkpoint_metadata(model_sd, optimizer_sd)
    expected_weight = torch.full_like(model.weight, 7)
    expected_bias = torch.full_like(model.bias, -3)
    load_calls = []

    def fake_load(load_sd, _checkpoint_dir, **kwargs):
        assert kwargs["strict"] is StrictHandling.RAISE_UNEXPECTED
        assert _sharded_logical_keys(load_sd) == (
            ({"weight", "bias"} if load_model else set())
            | ({"optimizer.exp_avg.weight"} if load_optimizer else set())
        )
        load_calls.append(True)
        loaded = {"step": 11}
        if load_model:
            loaded["model"] = {"weight": expected_weight, "bias": expected_bias}
        if load_optimizer:
            loaded["optimizer"] = {"loaded": True}
        return loaded

    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.distckpt._load_checkpoint_sharded_metadata",
        lambda _path: checkpoint_metadata,
    )
    _mock_distckpt_common_state(
        monkeypatch, {"step": 11, "content_metadata": DISTOPT_METADATA}
    )
    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.distckpt.dist_checkpointing.load", fake_load
    )
    before = copy.deepcopy(model.state_dict())

    step = load_dist_opt_checkpoint(
        model,
        optimizer,
        str(tmp_path),
        load_model=load_model,
        load_optimizer=load_optimizer,
    )

    assert step == 11
    assert load_calls == [True]
    if load_model:
        torch.testing.assert_close(model.weight, expected_weight)
        torch.testing.assert_close(model.bias, expected_bias)
    else:
        _assert_state_equal(model.state_dict(), before)
    if load_optimizer:
        assert optimizer.loaded_state == {"loaded": True}


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="MCore distckpt requires CUDA"
)
def test_real_mcore_full_checkpoint_roundtrips_all_partial_component_loads(
    tmp_path,
) -> None:
    class TinyShardedOptimizer:
        def __init__(self, model: torch.nn.Linear, value: float):
            self.exp_avg = torch.full_like(model.weight, value)

        def sharded_state_dict(self, model_sd, is_loading: bool = False, metadata=None):
            del is_loading
            assert metadata == DISTOPT_METADATA
            source = next(iter(_iter_sharded_bases(model_sd)))
            return {
                "exp_avg": replace(
                    source, key=f"optimizer.exp_avg.{source.key}", data=self.exp_avg
                )
            }

        def load_state_dict(self, state):
            self.exp_avg.copy_(state["exp_avg"])

    initialized_here = False
    if not dist.is_initialized():
        dist.init_process_group(
            "nccl", init_method=f"file://{tmp_path / 'nccl-init'}", rank=0, world_size=1
        )
        initialized_here = True
    try:
        source = torch.nn.Linear(2, 2, device="cuda")
        with torch.no_grad():
            source.weight.fill_(5)
            source.bias.fill_(-2)
        source_optimizer = TinyShardedOptimizer(source, 7)
        attach_model_sharded_state_dict([source], ParallelState())
        checkpoint_dir = str(tmp_path / "distckpt")
        save_dist_opt_checkpoint(
            source,
            source_optimizer,
            23,
            checkpoint_dir,
            save_model=True,
            save_optimizer=True,
        )

        for load_model, load_optimizer in (
            (True, True),
            (True, False),
            (False, True),
            (False, False),
        ):
            target = torch.nn.Linear(2, 2, device="cuda")
            with torch.no_grad():
                target.weight.zero_()
                target.bias.zero_()
            target_optimizer = TinyShardedOptimizer(target, -4)
            attach_model_sharded_state_dict([target], ParallelState())

            assert (
                load_dist_opt_checkpoint(
                    target,
                    target_optimizer,
                    checkpoint_dir,
                    load_model=load_model,
                    load_optimizer=load_optimizer,
                )
                == 23
            )
            expected_weight = (
                source.weight if load_model else torch.zeros_like(source.weight)
            )
            expected_bias = source.bias if load_model else torch.zeros_like(source.bias)
            expected_exp_avg = (
                source_optimizer.exp_avg
                if load_optimizer
                else torch.full_like(source_optimizer.exp_avg, -4)
            )
            torch.testing.assert_close(target.weight, expected_weight)
            torch.testing.assert_close(target.bias, expected_bias)
            torch.testing.assert_close(target_optimizer.exp_avg, expected_exp_avg)
    finally:
        if initialized_here:
            dist.destroy_process_group()


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="MCore distckpt requires CUDA"
)
def test_real_mcore_legacy_model_only_ignores_nonpersistent_buffer(tmp_path) -> None:
    from megatron.core import dist_checkpointing

    class BufferedModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0], device="cuda"))
            self.register_buffer(
                "router_bias", torch.tensor([3.0, 4.0], device="cuda"), persistent=True
            )
            self.register_buffer(
                "workspace", torch.tensor([5.0, 6.0], device="cuda"), persistent=False
            )

    initialized_here = False
    if not dist.is_initialized():
        dist.init_process_group(
            "nccl", init_method=f"file://{tmp_path / 'nccl-init'}", rank=0, world_size=1
        )
        initialized_here = True
    try:
        model = BufferedModule()
        attach_model_sharded_state_dict([model], ParallelState())
        current = _model_sharded_state_dict(model)
        source = next(
            entry for entry in _iter_sharded_bases(current) if entry.key == "weight"
        )
        legacy_model = dict(current["model"])
        legacy_model["workspace"] = replace(
            source, key="workspace", data=model.workspace
        )
        checkpoint_dir = tmp_path / "legacy-distckpt"
        checkpoint_dir.mkdir()
        dist_checkpointing.save(
            {"step": 31, "model": legacy_model},
            str(checkpoint_dir),
            validate_access_integrity=False,
        )

        with torch.no_grad():
            model.weight.zero_()
            model.router_bias.zero_()
            model.workspace.fill_(99)

        assert (
            load_dist_opt_checkpoint(
                model,
                None,
                str(checkpoint_dir),
                load_model=True,
                load_optimizer=False,
                allow_legacy_checkpoint=True,
            )
            == 31
        )
        torch.testing.assert_close(
            model.weight, torch.tensor([1.0, 2.0], device="cuda")
        )
        torch.testing.assert_close(
            model.router_bias, torch.tensor([3.0, 4.0], device="cuda")
        )
        torch.testing.assert_close(
            model.workspace, torch.tensor([99.0, 99.0], device="cuda")
        )
    finally:
        if initialized_here:
            dist.destroy_process_group()


@pytest.fixture
def legacy_dist_opt_nonpersistent_fixture():
    class BufferedModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2))
            self.register_buffer("router_bias", torch.arange(2.0), persistent=True)
            self.register_buffer("workspace", torch.full((2,), -7.0), persistent=False)

    model = BufferedModule()
    attach_model_sharded_state_dict([model], ParallelState())
    model_sd = _model_sharded_state_dict(model)
    checkpoint_metadata = _fake_checkpoint_metadata(model_sd)
    source = next(
        entry for entry in checkpoint_metadata.values() if entry.key == "weight"
    )
    # This is the metadata shape emitted by the pre-fix writer, which traversed
    # all named_buffers() instead of honoring persistent=False.
    checkpoint_metadata["legacy_workspace"] = replace(source, key="workspace")
    return model, checkpoint_metadata


def test_mcore_raise_unexpected_ignores_checkpoint_only_legacy_buffer(
    legacy_dist_opt_nonpersistent_fixture,
) -> None:
    from megatron.core.dist_checkpointing.validation import (
        StrictHandling,
        validate_integrity_and_strict_load,
    )

    model, checkpoint_metadata = legacy_dist_opt_nonpersistent_fixture
    load_sd = _model_sharded_state_dict(model)

    validated, missing, unexpected = validate_integrity_and_strict_load(
        load_sd,
        StrictHandling.RAISE_UNEXPECTED,
        validate_access_integrity=False,
        ckpt_sharded_metadata=checkpoint_metadata,
    )

    assert _sharded_logical_keys(validated) == {"weight", "router_bias"}
    assert missing == set()
    assert unexpected == set()


def test_dist_opt_legacy_model_only_allows_declared_nonpersistent_buffer_preflight(
    monkeypatch, tmp_path, legacy_dist_opt_nonpersistent_fixture
) -> None:
    import megatron.lite.primitive.ckpt.distckpt as distckpt
    from megatron.core.dist_checkpointing.validation import StrictHandling

    model, checkpoint_metadata = legacy_dist_opt_nonpersistent_fixture
    expected_weight = torch.full_like(model.weight, 5)
    expected_router_bias = torch.full_like(model.router_bias, 3)
    workspace_before = model.workspace.clone()

    monkeypatch.setattr(
        distckpt, "_load_checkpoint_sharded_metadata", lambda _path: checkpoint_metadata
    )
    _mock_distckpt_common_state(monkeypatch, {"step": 8})

    def fake_load(load_sd, _checkpoint_dir, **kwargs):
        assert _sharded_logical_keys(load_sd) == {"weight", "router_bias"}
        assert kwargs["strict"] is StrictHandling.RAISE_UNEXPECTED
        return {
            "step": 8,
            "model": {"weight": expected_weight, "router_bias": expected_router_bias},
        }

    monkeypatch.setattr(distckpt.dist_checkpointing, "load", fake_load)

    assert (
        load_dist_opt_checkpoint(
            model,
            None,
            str(tmp_path),
            load_model=True,
            load_optimizer=False,
            allow_legacy_checkpoint=True,
        )
        == 8
    )
    torch.testing.assert_close(model.weight, expected_weight)
    torch.testing.assert_close(model.router_bias, expected_router_bias)
    torch.testing.assert_close(model.workspace, workspace_before)


def test_dist_opt_nonpersistent_legacy_key_requires_explicit_opt_in(
    monkeypatch, tmp_path, legacy_dist_opt_nonpersistent_fixture
) -> None:
    import megatron.lite.primitive.ckpt.distckpt as distckpt

    model, checkpoint_metadata = legacy_dist_opt_nonpersistent_fixture
    before = copy.deepcopy(model.state_dict())
    workspace_before = model.workspace.clone()
    monkeypatch.setattr(
        distckpt, "_load_checkpoint_sharded_metadata", lambda _path: checkpoint_metadata
    )
    monkeypatch.setattr(
        distckpt.dist_checkpointing,
        "load",
        lambda *_args, **_kwargs: pytest.fail("MCore load must not run"),
    )

    with pytest.raises(
        RuntimeError,
        match=r"saved_but_unrequested_for_requested_components=\['workspace'\]",
    ):
        load_dist_opt_checkpoint(
            model,
            None,
            str(tmp_path),
            load_model=True,
            load_optimizer=False,
            allow_legacy_checkpoint=False,
        )

    _assert_state_equal(model.state_dict(), before)
    torch.testing.assert_close(model.workspace, workspace_before)


def test_dist_opt_legacy_model_only_rejects_unknown_model_tensor_before_mcore_load(
    monkeypatch, tmp_path, legacy_dist_opt_nonpersistent_fixture
) -> None:
    import megatron.lite.primitive.ckpt.distckpt as distckpt

    model, checkpoint_metadata = legacy_dist_opt_nonpersistent_fixture
    source = next(iter(checkpoint_metadata.values()))
    checkpoint_metadata["unknown"] = replace(source, key="removed_parameter")
    before = copy.deepcopy(model.state_dict())
    workspace_before = model.workspace.clone()
    monkeypatch.setattr(
        distckpt, "_load_checkpoint_sharded_metadata", lambda _path: checkpoint_metadata
    )
    monkeypatch.setattr(
        distckpt.dist_checkpointing,
        "load",
        lambda *_args, **_kwargs: pytest.fail("MCore load must not run"),
    )

    with pytest.raises(
        RuntimeError,
        match=(
            r"saved_but_unrequested_for_requested_components="
            r"\['removed_parameter'\]"
        ),
    ):
        load_dist_opt_checkpoint(
            model,
            None,
            str(tmp_path),
            load_model=True,
            load_optimizer=False,
            allow_legacy_checkpoint=True,
        )

    _assert_state_equal(model.state_dict(), before)
    torch.testing.assert_close(model.workspace, workspace_before)


def test_dist_opt_saved_stale_model_key_fails_metadata_preflight_before_load(
    monkeypatch, tmp_path
) -> None:
    wrapped_module = torch.nn.Linear(2, 2)
    model = FakeWrapper(wrapped_module)
    optimizer = FakeDistOpt()
    attach_model_sharded_state_dict([model], ParallelState())
    model_before = copy.deepcopy(wrapped_module.state_dict())
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    model_sd = _model_sharded_state_dict(model)
    checkpoint_metadata = _fake_checkpoint_metadata(model_sd)
    first_entry = next(iter(checkpoint_metadata.values()))
    checkpoint_metadata["stale"] = replace(first_entry, key="removed_parameter")

    def unexpected_load(*_args, **_kwargs):
        raise AssertionError("MCore load must not run after metadata mismatch")

    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.distckpt.dist_checkpointing.load", unexpected_load
    )
    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.distckpt._load_checkpoint_sharded_metadata",
        lambda _path: checkpoint_metadata,
    )
    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.distckpt._load_checkpoint_common_state",
        lambda _path: {"step": 8, "optimizer": {"is_loading": False}},
    )

    with pytest.raises(
        RuntimeError,
        match=(
            r"distckpt checkpoint metadata preflight failed: sharded key mismatch "
            r"before load: .*removed_parameter"
        ),
    ):
        dcp.load_training_checkpoint(
            model,
            optimizer,
            str(tmp_path / "step_8"),
            use_dcp=True,
            load_rng=False,
            allow_legacy_checkpoint=True,
        )

    _assert_state_equal(wrapped_module.state_dict(), model_before)
    _assert_state_equal(optimizer.state_dict(), optimizer_before)


def test_dist_opt_requested_stale_model_key_fails_before_mcore_load_or_mutation(
    monkeypatch, tmp_path
) -> None:
    model = torch.nn.Linear(2, 2)
    attach_model_sharded_state_dict([model], ParallelState())
    before = copy.deepcopy(model.state_dict())
    model_sd = _model_sharded_state_dict(model)
    checkpoint_metadata = _fake_checkpoint_metadata(model_sd)
    checkpoint_metadata = {
        metadata_key: entry
        for metadata_key, entry in checkpoint_metadata.items()
        if entry.key != "weight"
    }

    def unexpected_load(*_args, **_kwargs):
        with torch.no_grad():
            model.weight.fill_(99)
        raise AssertionError("MCore load must not run after metadata mismatch")

    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.distckpt._load_checkpoint_sharded_metadata",
        lambda _path: checkpoint_metadata,
    )
    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.distckpt._load_checkpoint_common_state",
        lambda _path: {"step": 0},
    )
    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.distckpt.dist_checkpointing.load", unexpected_load
    )

    with pytest.raises(
        RuntimeError,
        match=(
            r"distckpt checkpoint metadata preflight failed: sharded key mismatch "
            r"before load: requested_but_absent=\['weight'\]"
        ),
    ):
        load_dist_opt_checkpoint(
            model, None, str(tmp_path), load_model=True, load_optimizer=False
        )

    _assert_state_equal(model.state_dict(), before)


def test_dist_opt_checkpoint_load_rejects_missing_optimizer_before_model_mutation(
    monkeypatch, tmp_path
) -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = FakeDistOpt()
    attach_model_sharded_state_dict([model], ParallelState())
    before = copy.deepcopy(model.state_dict())
    checkpoint_metadata = _fake_checkpoint_metadata(_model_sharded_state_dict(model))

    def fake_load(_sharded_state_dict, _checkpoint_dir, **_kwargs):
        return {
            "step": 5,
            "model": {
                "weight": torch.full_like(model.weight, 9),
                "bias": torch.full_like(model.bias, -4),
            },
        }

    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.distckpt.dist_checkpointing.load", fake_load
    )
    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.distckpt._load_checkpoint_sharded_metadata",
        lambda _path: checkpoint_metadata,
    )
    _mock_distckpt_common_state(
        monkeypatch, {"step": 5, "content_metadata": DISTOPT_METADATA}
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "distckpt common-state preflight failed: RuntimeError: checkpoint is missing "
            "optimizer state requested by load_optimizer=True"
        ),
    ):
        dcp.load_training_checkpoint(
            model,
            optimizer,
            str(tmp_path / "step_5"),
            use_dcp=True,
            allow_legacy_checkpoint=True,
        )

    _assert_state_equal(model.state_dict(), before)


def test_dist_opt_step_sync_traverses_multi_optimizer_chain_without_optimizer_property() -> (
    None
):
    class FakeTorchOptimizer:
        def __init__(self, steps):
            self.state = {
                object(): {"step": torch.tensor(step, dtype=torch.int64)}
                for step in steps
            }

    class FakeDistOpt:
        def __init__(self, steps):
            self.optimizer = FakeTorchOptimizer(steps)

    class FakeChainedOptimizer:
        def __init__(self):
            self.chained_optimizers = [FakeDistOpt([1, 3]), FakeDistOpt([2, 4])]

        @property
        def optimizer(self):
            raise AssertionError(
                "ChainedOptimizer has more than one optimizer when accessing self.optimizer"
            )

    chained = FakeChainedOptimizer()

    _synchronize_native_optimizer_steps(chained)

    for child in chained.chained_optimizers:
        steps = [int(state["step"].item()) for state in child.optimizer.state.values()]
        assert steps == [max(steps)] * len(steps)


def test_runtime_checkpoint_api_passes_current_training_checkpoint_signature(
    monkeypatch, tmp_path
) -> None:
    calls = {}

    def fake_save(model, optimizer, step, path, config, ps, **kwargs):
        calls["save"] = (model, optimizer, step, path, config, ps, kwargs)

    def fake_load(model, optimizer, path, config, ps, **kwargs):
        calls["load"] = (model, optimizer, path, config, ps, kwargs)
        return 7

    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.save_training_checkpoint", fake_save
    )
    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.load_training_checkpoint", fake_load
    )

    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)
    model = torch.nn.Linear(1, 1)
    optimizer = object()
    parallel = SimpleNamespace(tp=1, etp=1, ep=1, pp=1, cp=1)
    ps = object()
    handle = ModelHandle(
        model=model,
        optimizer=optimizer,
        parallel_state=ps,
        config=SimpleNamespace(parallel=parallel),
    )

    runtime.save_checkpoint(
        handle, str(tmp_path), global_step=7, save_model=True, save_optimizer=False
    )
    loaded_step = runtime.load_checkpoint(
        handle,
        str(tmp_path),
        load_model=False,
        load_optimizer=True,
        allow_legacy_checkpoint=True,
    )

    assert calls["save"] == (
        model,
        optimizer,
        7,
        str(tmp_path),
        parallel,
        ps,
        {
            "get_placements": default_placement_fn,
            "is_expert": default_expert_classifier,
            "use_dcp": True,
            "save_rng": True,
            "save_model": True,
            "save_optimizer": False,
        },
    )
    assert calls["load"] == (
        model,
        optimizer,
        str(tmp_path),
        parallel,
        ps,
        {
            "get_placements": default_placement_fn,
            "is_expert": default_expert_classifier,
            "use_dcp": True,
            "load_rng": True,
            "load_parameter_state_update_legacy_format": False,
            "load_model": False,
            "load_optimizer": True,
            "allow_legacy_checkpoint": True,
        },
    )
    assert loaded_step == 7
