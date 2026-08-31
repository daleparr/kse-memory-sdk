"""
FR-03 — Query parse: embed the query, map it to dimension targets.

Written test-first per GOV-04.

The v2 path extracted "conceptual weights" by keyword-matching against the
hardcoded fashion vocabulary (BD3 debt list, item 1). FR-03 replaces that:
a query is embedded once and its similarity to each dimension's anchor
centroid becomes that dimension's target weight. Same geometry as FR-02
scoring, so queries and items live in one space by construction.

The v2 SearchService consumer is rewired at FR-04; this is the parse itself.
"""
from __future__ import annotations

import pytest

from tests.conftest import StubEmbedder
from kse_memory.core.projection import anchor_centroids
from kse_memory.core.query import ParsedQuery, parse_query
from kse_memory.core.schema import load_schema

pytestmark = pytest.mark.unit


class CountingStub(StubEmbedder):
    def __init__(self):
        self.calls = 0
        self.texts = []

    def embed(self, texts):
        self.calls += 1
        self.texts.extend(texts)
        return super().embed(texts)


SCHEMA = {
    "name": "q", "version": "1.0.0",
    "dimensions": [
        {"name": "alpha", "description": "", "anchors": ["exact anchor text"]},
        {"name": "beta", "description": "", "anchors": ["something else entirely"]},
        {"name": "gamma", "description": "", "anchors": ["third topic", "another third topic"]},
    ],
}


@pytest.fixture
def schema():
    return load_schema(SCHEMA)


def test_parse_returns_targets_for_every_dimension(schema):
    parsed = parse_query("a question", schema, StubEmbedder())
    assert isinstance(parsed, ParsedQuery)
    assert set(parsed.targets) == {"alpha", "beta", "gamma"}
    assert len(parsed.vector) == StubEmbedder.dim


def test_targets_are_bounded_unit_interval(schema):
    parsed = parse_query("anything at all", schema, StubEmbedder())
    assert all(0.0 <= v <= 1.0 for v in parsed.targets.values())


def test_query_matching_an_anchor_targets_that_dimension_hardest(schema):
    """The FR-03 contract itself: similarity to anchors drives the mapping."""
    parsed = parse_query("exact anchor text", schema, StubEmbedder())
    assert parsed.targets["alpha"] == 1.0  # identical text -> cosine 1 -> unit 1.0
    assert parsed.targets["alpha"] >= max(parsed.targets["beta"], parsed.targets["gamma"])


def test_parse_carries_replay_identity(schema):
    """A parsed query must be reproducible: schema + model pin it down."""
    parsed = parse_query("q", schema, StubEmbedder())
    assert parsed.schema_name == "q"
    assert parsed.schema_version == "1.0.0"
    assert parsed.model_id == StubEmbedder.model_id
    assert parsed.text == "q"


def test_parse_is_deterministic(schema):
    assert parse_query("q", schema, StubEmbedder()) == parse_query("q", schema, StubEmbedder())


def test_precomputed_centroids_are_reused_not_recomputed(schema):
    """A long-lived caller passes its cached centroids; only the query embeds."""
    embedder = CountingStub()
    centroids = anchor_centroids(schema, embedder)
    calls_before = embedder.calls

    parsed = parse_query("q", schema, embedder, centroids=centroids)

    assert embedder.calls == calls_before + 1
    assert embedder.texts[-1] == "q"
    assert set(parsed.targets) == {"alpha", "beta", "gamma"}


def test_centroid_and_fresh_paths_agree(schema):
    embedder = StubEmbedder()
    fresh = parse_query("q", schema, embedder)
    reused = parse_query("q", schema, embedder, centroids=anchor_centroids(schema, embedder))
    assert fresh == reused


def test_blank_query_is_rejected(schema):
    for bad in ("", "   ", "\n"):
        with pytest.raises(ValueError, match="empty"):
            parse_query(bad, schema, StubEmbedder())


def test_parse_makes_no_network_calls(no_network, schema):
    assert parse_query("offline", schema, StubEmbedder()).targets
