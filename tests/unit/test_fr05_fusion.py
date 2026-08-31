"""
FR-05 — Fuse: RRF default, optional min-max weighted fusion.

Written test-first per GOV-04. D-07: rank-based and scale-free is the point —
FR-04's channel scores are not comparable across channels, so fusion may use
only the *ranks* by default.
"""
from __future__ import annotations

import pytest

from kse_memory.core.fusion import FusedItem, fuse_rrf, fuse_weighted

pytestmark = pytest.mark.unit


CHANNELS = {
    "vector": (("a", 0.92), ("b", 0.40), ("c", 0.11)),
    "conceptual": (("b", 0.71), ("a", 0.70), ("d", 0.02)),
    "graph": (("a", 2.0), ("d", 1.0)),
}


def test_rrf_returns_ranked_fused_items():
    fused = fuse_rrf(CHANNELS)
    assert all(isinstance(item, FusedItem) for item in fused)
    scores = [item.fused for item in fused]
    assert scores == sorted(scores, reverse=True)
    # 'a' is ranked 1st, 2nd and 1st — nothing can beat it
    assert fused[0].entity_id == "a"


def test_rrf_scores_follow_the_formula():
    """score(e) = sum over channels of 1 / (k + rank_e), rank 1-based."""
    fused = {item.entity_id: item.fused for item in fuse_rrf(CHANNELS, k=60)}
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 63)


def test_fused_items_carry_per_channel_ranks_and_scores():
    """FR-06 needs receipts: ranks and raw scores per channel, None where absent."""
    item = {i.entity_id: i for i in fuse_rrf(CHANNELS)}["d"]
    assert item.ranks == {"vector": None, "conceptual": 3, "graph": 2}
    assert item.scores["conceptual"] == pytest.approx(0.02)
    assert item.scores["vector"] is None


def test_rrf_ignores_scores_entirely():
    """Rank-based means the numbers cannot matter, only the order."""
    rescaled = {
        name: tuple((e, s * 1000 + 5) for e, s in rows)
        for name, rows in CHANNELS.items()
    }
    original = [(i.entity_id, i.fused) for i in fuse_rrf(CHANNELS)]
    scaled = [(i.entity_id, i.fused) for i in fuse_rrf(rescaled)]
    assert original == scaled


def test_rrf_weights_bias_channels():
    heavy_graph = fuse_rrf(CHANNELS, weights={"vector": 0.1, "conceptual": 0.1, "graph": 10.0})
    assert heavy_graph[0].entity_id == "a"           # top of the heavy channel
    assert heavy_graph[1].entity_id == "d"           # second in graph outranks the rest


def test_rrf_deterministic_tiebreak_by_id():
    tied = {"vector": (("z", 1.0),), "graph": (("m", 1.0),)}
    fused = fuse_rrf(tied)
    assert [i.entity_id for i in fused] == ["m", "z"]  # equal fused score -> id order


def test_rrf_rejects_bad_k():
    for bad in (0, -3):
        with pytest.raises(ValueError, match="k"):
            fuse_rrf(CHANNELS, k=bad)


def test_rrf_empty_channels_fuse_to_empty():
    assert fuse_rrf({}) == ()
    assert fuse_rrf({"vector": (), "graph": ()}) == ()


def test_top_k_truncates_after_fusion():
    fused = fuse_rrf(CHANNELS, top_k=2)
    assert len(fused) == 2
    assert fused[0].entity_id == "a"


# ------------------------------------------------------------ weighted path
def test_weighted_normalises_min_max_within_channels():
    """A channel's scale must not leak: min-max maps each channel onto [0,1]."""
    fused = {i.entity_id: i.fused for i in fuse_weighted(CHANNELS)}
    # vector: a=1.0, b=(0.40-0.11)/(0.92-0.11), c=0.0
    # conceptual: b=1.0, a=(0.70-0.02)/(0.71-0.02), d=0.0
    # graph: a=1.0, d=0.0
    expected_a = 1.0 + (0.70 - 0.02) / (0.71 - 0.02) + 1.0
    assert fused["a"] == pytest.approx(expected_a, abs=1e-9)


def test_weighted_single_score_channel_counts_full():
    """A one-item channel has no range; its item scores 1.0, not 0/0."""
    fused = {i.entity_id: i.fused for i in fuse_weighted({"vector": (("only", 0.3),)})}
    assert fused["only"] == pytest.approx(1.0)


def test_weighted_respects_weights():
    fused = fuse_weighted(CHANNELS, weights={"vector": 0.0, "conceptual": 0.0, "graph": 1.0})
    assert fused[0].entity_id == "a"  # the weighted channel's max wins outright


def test_weighted_channel_minimum_collapses_to_zero():
    """A real wart of min-max, pinned deliberately: the lowest-scored item in
    a channel normalises to 0.0 — indistinguishable from items the channel
    never returned. RRF has no such cliff, which is part of why it is the
    default (D-07)."""
    fused = {i.entity_id: i.fused for i in
             fuse_weighted(CHANNELS, weights={"vector": 0.0, "conceptual": 0.0, "graph": 1.0})}
    assert fused["d"] == 0.0          # graph returned it, ranked last
    assert fused["b"] == 0.0          # graph never saw it — same number
