"""
US7 — local scorer parity: no key, schema-conformant, quickstart-identical.

Written test-first per GOV-04. TC-07's clauses, made executable:
- scoring runs with every *_API_KEY scrubbed from the environment, offline;
- scores are schema-conformant for ARBITRARY schemas (Hypothesis, not three
  examples);
- API keys are inert on the default path: quickstart output is byte-identical
  whether keys are absent or present-but-bogus.
"""
from __future__ import annotations

import os

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.conftest import StubEmbedder
from kse_memory.core.ingest import normalise_record
from kse_memory.core.projection import SCORE_PRECISION, project, score_dimensions
from kse_memory.core.schema import load_schema

pytestmark = pytest.mark.unit


@pytest.fixture
def no_api_keys(monkeypatch):
    """TC-07's premise, enforced: no key of any kind is reachable."""
    for name in list(os.environ):
        if "API_KEY" in name or "_TOKEN" in name:
            monkeypatch.delenv(name, raising=False)
    yield


SCHEMA = load_schema({
    "name": "s", "version": "1.0.0",
    "dimensions": [
        {"name": "a", "description": "", "anchors": ["alpha anchor"]},
        {"name": "b", "description": "", "anchors": ["beta anchor", "second beta"]},
    ],
})


def test_scoring_needs_no_key_and_no_network(no_api_keys, no_network):
    entity = normalise_record({"title": "t", "description": "d"})
    scores = score_dimensions(entity, SCHEMA, StubEmbedder())
    assert set(scores) == {"a", "b"}


# ------------------------------------------------- conformance, generalised
name = st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=12)
anchor = st.text(min_size=1, max_size=30).filter(lambda s: s.strip())
nonblank = st.text(min_size=1, max_size=30).filter(lambda s: s.strip())


@st.composite
def schemas(draw):
    names = draw(st.lists(name, min_size=1, max_size=5, unique=True))
    return load_schema({
        "name": "gen", "version": "1.0.0",
        "dimensions": [
            {"name": n, "description": "", "anchors": draw(st.lists(anchor, min_size=1, max_size=3))}
            for n in names
        ],
    })


@st.composite
def entities(draw):
    return normalise_record({
        "title": draw(nonblank), "description": draw(nonblank),
        "tags": draw(st.lists(st.text(min_size=1, max_size=8), max_size=4)),
    })


@given(schemas(), entities())
@settings(max_examples=150)
def test_scores_are_schema_conformant_for_arbitrary_schemas(schema, entity):
    """Every dimension present, every value in [0,1] at SCORE_PRECISION —
    for whatever schema and entity arrive, not for three hand-picked ones."""
    scores = score_dimensions(entity, schema, StubEmbedder())
    assert set(scores) == set(schema.names())
    for value in scores.values():
        assert 0.0 <= value <= 1.0
        assert round(value, SCORE_PRECISION) == value


@given(schemas(), entities())
@settings(max_examples=100)
def test_projection_replay_identity_holds_for_arbitrary_schemas(schema, entity):
    p1 = project(entity, schema, StubEmbedder())
    p2 = project(entity, schema, StubEmbedder())
    assert p1 == p2


# ------------------------------------------------------------------ parity
@pytest.mark.asyncio
@pytest.mark.component
async def test_api_keys_are_inert_on_the_default_path(monkeypatch, stub_embedder):
    """Quickstart output must be identical whether keys are scrubbed or
    present-but-bogus — a default path that behaves differently when a key
    happens to exist is a covert dependency."""
    from kse_memory.quickstart.v3 import run_quickstart

    for key in list(os.environ):
        if "API_KEY" in key:
            monkeypatch.delenv(key, raising=False)
    without = await run_quickstart(embedder=stub_embedder)

    monkeypatch.setenv("OPENAI_API_KEY", "bogus-key-should-be-ignored")
    monkeypatch.setenv("PINECONE_API_KEY", "bogus-key-should-be-ignored")
    with_bogus = await run_quickstart(embedder=StubEmbedder())

    assert {q: [(h.entity_id, h.similarity, dict(h.scores)) for h in hits]
            for q, hits in without.searches.items()} == \
           {q: [(h.entity_id, h.similarity, dict(h.scores)) for h in hits]
            for q, hits in with_bogus.searches.items()}
