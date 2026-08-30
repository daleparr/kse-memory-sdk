"""
FR-04 — Retrieve concurrently: vector · conceptual · graph channels.

Written test-first per GOV-04.

Consumes a ParsedQuery (FR-03). Because query targets and item scores share
one geometry, the conceptual channel is find_similar_dimensions on the
targets, directly — no translation layer.

FR-07 groundwork is built in rather than bolted on: a missing store is an
empty channel, a raising store is an empty channel plus an errors entry, and
neither disturbs the other channels. FR-05 fuses; FR-07 adds the fused
confidence threshold.
"""
from __future__ import annotations

import asyncio

import pytest

from tests.conftest import StubEmbedder
from kse_memory.core.dimension_store import DimensionScores, InMemoryDimensionStore
from kse_memory.core.projection import SCORED_AS, dimension_node_id
from kse_memory.core.query import parse_query
from kse_memory.core.retrieval import RetrievalResult, retrieve
from kse_memory.core.schema import load_schema

pytestmark = [pytest.mark.asyncio, pytest.mark.component]

SCHEMA = load_schema({
    "name": "r", "version": "1.0.0",
    "dimensions": [
        {"name": "alpha", "description": "", "anchors": ["exact anchor text"]},
        {"name": "beta", "description": "", "anchors": ["something else entirely"]},
    ],
})


class FakeVectorStore:
    def __init__(self, rows=None):
        self.rows = rows or []  # [(id, score, meta)] pre-ranked

    async def search_vectors(self, query_vector, top_k=10, filters=None):
        return self.rows[:top_k]


class FakeGraphStore:
    """Holds entity -> dimension SCORED_AS edges; neighbors are undirected."""

    def __init__(self, edges=None):
        self.edges = edges or []  # [(entity_id, dimension_node_id)]

    async def get_neighbors(self, node_id, relationship_types=None):
        out = []
        for source, target in self.edges:
            if source == node_id:
                out.append({"id": target})
            elif target == node_id:
                out.append({"id": source})
        return out


class SlowStore:
    """Records concurrency: every channel must have STARTED before any may
    finish, which is only possible if they run concurrently."""

    def __init__(self, barrier, results):
        self.barrier = barrier
        self.results = results

    async def _wait(self):
        self.barrier.arrived += 1
        while self.barrier.arrived < self.barrier.expected:
            await asyncio.sleep(0)

    async def search_vectors(self, query_vector, top_k=10, filters=None):
        await self._wait()
        return self.results

    async def find_similar_dimensions(self, scores, threshold=0.0, limit=10):
        await self._wait()
        return []

    async def get_neighbors(self, node_id, relationship_types=None):
        await self._wait()
        return []


class Barrier:
    def __init__(self, expected):
        self.expected = expected
        self.arrived = 0


class RaisingStore:
    async def search_vectors(self, *a, **k):
        raise RuntimeError("vector backend down")

    async def find_similar_dimensions(self, *a, **k):
        raise RuntimeError("concept backend down")

    async def get_neighbors(self, *a, **k):
        raise RuntimeError("graph backend down")


@pytest.fixture
def parsed():
    return parse_query("exact anchor text", SCHEMA, StubEmbedder())


async def seeded_concept_store():
    store = InMemoryDimensionStore()
    await store.store_dimensions("close", DimensionScores(
        schema_name="r", schema_version="1.0.0",
        scores={"alpha": 1.0, "beta": 0.45}))
    await store.store_dimensions("far", DimensionScores(
        schema_name="r", schema_version="1.0.0",
        scores={"alpha": 0.1, "beta": 0.9}))
    return store


# ---------------------------------------------------------------- channels
async def test_vector_channel_returns_ranked_ids(parsed):
    vectors = FakeVectorStore([("a", 0.9, {}), ("b", 0.5, {})])
    result = await retrieve(parsed, vector_store=vectors)
    assert result.vector == (("a", 0.9), ("b", 0.5))


async def test_conceptual_channel_searches_with_the_query_targets(parsed):
    """FR-03's geometry pays off: targets are valid DimensionScores."""
    store = await seeded_concept_store()
    result = await retrieve(parsed, concept_store=store)
    ids = [entity_id for entity_id, _ in result.conceptual]
    assert ids[0] == "close"


async def test_conceptual_channel_scopes_to_schema(parsed):
    store = await seeded_concept_store()
    await store.store_dimensions("other", DimensionScores(
        schema_name="different", schema_version="9.9.9",
        scores={"alpha": 1.0, "beta": 0.45}))
    result = await retrieve(parsed, concept_store=store)
    assert "other" not in [e for e, _ in result.conceptual]


async def test_graph_channel_ranks_by_target_dimension_coverage(parsed):
    """Entities touching more of the query's top dimensions rank higher;
    ties break deterministically by id."""
    alpha, beta = dimension_node_id("r", "alpha"), dimension_node_id("r", "beta")
    graph = FakeGraphStore(edges=[
        ("both", alpha), ("both", beta),
        ("only-alpha", alpha),
        ("only-beta", beta),
    ])
    result = await retrieve(parsed, graph_store=graph)
    ids = [entity_id for entity_id, _ in result.graph]
    assert ids[0] == "both"
    assert set(ids) == {"both", "only-alpha", "only-beta"}
    assert ids == sorted(ids, key=lambda i: (-dict(result.graph)[i], i))


async def test_top_k_bounds_every_channel(parsed):
    vectors = FakeVectorStore([(f"v{i}", 1.0 - i / 10, {}) for i in range(8)])
    store = await seeded_concept_store()
    result = await retrieve(parsed, vector_store=vectors, concept_store=store, top_k=1)
    assert len(result.vector) == 1
    assert len(result.conceptual) == 1


# ------------------------------------------------------------- concurrency
async def test_channels_run_concurrently(parsed):
    """All three channels must be in flight at once: each blocks until every
    one has started, so sequential execution would deadlock (timeout)."""
    barrier = Barrier(expected=3)
    store = SlowStore(barrier, [("a", 1.0, {})])
    result = await asyncio.wait_for(
        retrieve(parsed, vector_store=store, concept_store=store, graph_store=store),
        timeout=5.0,
    )
    assert result.vector == (("a", 1.0),)


# ------------------------------------------------------- degradation (FR-07)
async def test_missing_stores_yield_empty_channels_not_errors(parsed):
    result = await retrieve(parsed)
    assert result.vector == () and result.conceptual == () and result.graph == ()
    assert result.errors == {}


async def test_raising_channel_degrades_and_is_flagged(parsed):
    vectors = FakeVectorStore([("a", 0.9, {})])
    result = await retrieve(parsed, vector_store=vectors,
                            concept_store=RaisingStore(), graph_store=RaisingStore())
    assert result.vector == (("a", 0.9),)  # healthy channel unaffected
    assert result.conceptual == () and result.graph == ()
    assert set(result.errors) == {"conceptual", "graph"}
    assert "down" in result.errors["conceptual"]


async def test_result_is_deterministic(parsed):
    store = await seeded_concept_store()
    a = await retrieve(parsed, concept_store=store)
    b = await retrieve(parsed, concept_store=store)
    assert a == b


async def test_retrieve_makes_no_network_calls(no_network, parsed):
    result = await retrieve(parsed, concept_store=await seeded_concept_store())
    assert result.conceptual
