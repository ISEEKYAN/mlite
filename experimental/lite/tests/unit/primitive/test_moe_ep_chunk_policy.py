from __future__ import annotations

import pytest

from megatron.lite.primitive.moe_ep_chunk_policy import (
    ep_chunk_ranges,
    parse_ep_chunk_spec,
    resolve_ep_chunk_count,
)


def test_parse_ep_chunk_spec_accepts_auto_and_positive_ints():
    assert parse_ep_chunk_spec(None) == "auto"
    assert parse_ep_chunk_spec("auto") == "auto"
    assert parse_ep_chunk_spec("3") == 3
    assert parse_ep_chunk_spec(2) == 2

    with pytest.raises(ValueError, match=">= 1"):
        parse_ep_chunk_spec(0)
    with pytest.raises(ValueError, match="integer or 'auto'"):
        parse_ep_chunk_spec("bad")


def test_resolve_ep_chunk_count_matches_bumblebee_defaults():
    assert resolve_ep_chunk_count(1024, ep_size=8, hidden_size=4096) == 1
    assert resolve_ep_chunk_count(16_384, ep_size=1, hidden_size=4096) == 1
    assert resolve_ep_chunk_count(16_384, ep_size=8, hidden_size=4096) == 2
    assert resolve_ep_chunk_count(32_768, ep_size=8, hidden_size=4096) == 3
    assert (
        resolve_ep_chunk_count(
            32_768,
            ep_size=8,
            hidden_size=4096,
            direction="fused_backward",
        )
        == 2
    )
    assert resolve_ep_chunk_count(32_768, ep_size=8, hidden_size=4096, spec=4) == 4


def test_ep_chunk_ranges_are_contiguous_and_non_empty():
    ranges = ep_chunk_ranges(10, 3)
    assert ranges == [(0, 4), (4, 7), (7, 10)]
    assert ranges[0][0] == 0
    assert ranges[-1][-1] == 10
    assert all(start < end for start, end in ranges)


def test_ep_chunk_ranges_honor_mlite_weights(monkeypatch):
    monkeypatch.setenv("MEGATRON_LITE_EP_CHUNK_WEIGHTS", "1,2,1")
    assert ep_chunk_ranges(10, 3) == [(0, 3), (3, 8), (8, 10)]


def test_ep_chunk_ranges_accept_bumblebee_legacy_weights(monkeypatch):
    monkeypatch.delenv("MEGATRON_LITE_EP_CHUNK_WEIGHTS", raising=False)
    monkeypatch.setenv("BUMBLEBEE_EP_CHUNK_WEIGHTS", "1,1")
    assert ep_chunk_ranges(5, 2) == [(0, 3), (3, 5)]


def test_ep_chunk_ranges_keep_weighted_chunks_non_empty(monkeypatch):
    monkeypatch.setenv("MEGATRON_LITE_EP_CHUNK_WEIGHTS", "100,1,1")
    ranges = ep_chunk_ranges(3, 3)

    assert ranges[0][0] == 0
    assert ranges[-1][-1] == 3
    assert all(start < end for start, end in ranges)


def test_ep_chunk_ranges_reject_bad_weights(monkeypatch):
    monkeypatch.setenv("MEGATRON_LITE_EP_CHUNK_WEIGHTS", "1,0")
    with pytest.raises(ValueError, match="finite positive"):
        ep_chunk_ranges(5, 2)
