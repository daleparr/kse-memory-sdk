"""
FR-06 — Explain: per-channel ranks, scores and per-dimension breakdown.

Written test-first per GOV-04.

Everything an explanation needs already exists — FusedItem carries channel
ranks and raw scores, ParsedQuery carries targets and replay identity,
RetrievalResult carries degraded-channel errors, the concept store holds the
entity's dimension scores. FR-06 is the discipline of assembling those into
one attachable object per result, losing nothing.

explain_results is pure: the caller fetches dimension scores; explanation
itself does no I/O, so it lives in the unit lane.
"""
from __future__ import annotations

import pytest

from tests.conftest import StubEmbedder
from kse_memory.core.explain import DimensionRow, Explanation, explain_results
from kse_memory.core.fusion import fuse_rrf
from kse_memory.core.query import parse_query
from kse_memory.core.retrieval import RetrievalResult
from kse_memory.core.schema import load_schema

pytestmark = pytest.mark.unit

SCHEMA = load_schema({
    "name": "e", "version": "1.0.0",
    "dimensions": [
        {"name": "depth", "description": "", "anchors": ["deep technical detail"]},
        {"name": "clarity", "description": "", "anchors": ["plainly explained"]},
    ],
})


@pytest.fixture
def parsed():
    return parse_query("deep technical detail", SCHEMA, StubEmbedder())


@pytest.fixture
def retrieval():
    return RetrievalResult(
        vector=(("a", 0.9), ("b", 0.4)),
        conceptual=(("b", 0.8), ("a", 0.7)),
        graph=(("a", 2.0),),
        errors={},
    )


@pytest.fixture
def fused(retrieval):
    return fuse_rrf({
        "vector": retrieval.vector,
        "conceptual": retrieval.conceptual,
        "graph": retrieval.graph,
    })


DIMENSION_SCORES = {
    "a": {"depth": 0.9, "clarity": 0.3},
    "b": {"depth": 0.2, "clarity": 0.8},
}


def test_one_explanation_per_fused_result_in_order(parsed, retrieval, fused):
    explanations = explain_results(parsed, retrieval, fused, DIMENSION_SCORES)
    assert [e.entity_id for e in explanations] == [i.entity_id for i in fused]
    assert all(isinstance(e, Explanation) for e in explanations)


def test_channel_ranks_and_scores_survive_intact(parsed, retrieval, fused):
    by_id = {e.entity_id: e for e in explain_results(parsed, retrieval, fused, DIMENSION_SCORES)}
    a = by_id["a"]
    assert a.ranks == {"vector": 1, "conceptual": 2, "graph": 1}
    assert a.channel_scores["vector"] == pytest.approx(0.9)
    assert by_id["b"].ranks["graph"] is None  # absence is information


def test_dimension_breakdown_pairs_target_with_score(parsed, retrieval, fused):
    by_id = {e.entity_id: e for e in explain_results(parsed, retrieval, fused, DIMENSION_SCORES)}
    rows = {r.name: r for r in by_id["a"].dimensions}
    assert set(rows) == {"depth", "clarity"}
    assert rows["depth"].target == parsed.targets["depth"]
    assert rows["depth"].score == pytest.approx(0.9)
    assert isinstance(rows["depth"], DimensionRow)


def test_alignment_is_closeness_on_the_unit_interval(parsed, retrieval, fused):
    """alignment = 1 - |target - score|: bounded, symmetric, interpretable."""
    by_id = {e.entity_id: e for e in explain_results(parsed, retrieval, fused, DIMENSION_SCORES)}
    for explanation in by_id.values():
        for row in explanation.dimensions:
            assert row.alignment == pytest.approx(1.0 - abs(row.target - row.score))
            assert 0.0 <= row.alignment <= 1.0


def test_breakdown_follows_schema_order(parsed, retrieval, fused):
    explanation = explain_results(parsed, retrieval, fused, DIMENSION_SCORES)[0]
    assert [r.name for r in explanation.dimensions] == ["depth", "clarity"]


def test_replay_identity_is_attached(parsed, retrieval, fused):
    explanation = explain_results(parsed, retrieval, fused, DIMENSION_SCORES)[0]
    assert explanation.schema_name == "e"
    assert explanation.schema_version == "1.0.0"
    assert explanation.model_id == StubEmbedder.model_id
    assert explanation.query == "deep technical detail"


def test_degraded_channels_are_surfaced(parsed, fused):
    degraded = RetrievalResult(
        vector=(("a", 0.9), ("b", 0.4)),
        conceptual=(("b", 0.8), ("a", 0.7)),
        graph=(),
        errors={"graph": "graph backend down"},
    )
    explanation = explain_results(parsed, degraded, fused, DIMENSION_SCORES)[0]
    assert explanation.degraded == {"graph": "graph backend down"}


def test_missing_dimension_scores_degrade_to_empty_breakdown(parsed, retrieval, fused):
    """An entity the concept store has no scores for still gets explained."""
    explanations = explain_results(parsed, retrieval, fused, {"a": DIMENSION_SCORES["a"]})
    by_id = {e.entity_id: e for e in explanations}
    assert by_id["b"].dimensions == ()
    assert by_id["b"].ranks["vector"] == 2  # the rest of the explanation stands


def test_explanations_are_deterministic(parsed, retrieval, fused):
    a = explain_results(parsed, retrieval, fused, DIMENSION_SCORES)
    b = explain_results(parsed, retrieval, fused, DIMENSION_SCORES)
    assert a == b
