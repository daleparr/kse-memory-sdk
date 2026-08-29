"""
Wiring FR-01 ingest to FR-02 projection: the v3 ingest path.

Written test-first per GOV-04.

This is the seam the two features were built either side of. A raw adapter
record goes in; a normalised entity, a projection under the user's schema, and
incremental writes to the configured stores come out — with the replay identity
carried end to end, so re-ingesting unchanged content costs nothing.

Scope note: the concept store is deliberately NOT written. ConceptStoreInterface
is typed against the legacy ConceptualDimensions class, so it cannot hold
schema-driven scores without an interface change (TC-04 follow-up). Dimension
scores live in the graph as scored edges, which is where upsert_projection
puts them.
"""
from __future__ import annotations

import pytest

from kse_memory.core.pipeline import IngestPipeline, IngestResult
from kse_memory.core.schema import load_schema

pytestmark = pytest.mark.asyncio


class CountingEmbedder:
    """Counts every text it is asked to embed, so batching can be asserted."""

    model_id = "counting-v1"

    def __init__(self):
        self.calls = 0
        self.texts = []

    def embed(self, texts):
        self.calls += 1
        self.texts.extend(texts)
        return [[float(len(t) % 5) + 1.0, 2.0, 3.0] for t in texts]


class FakeGraphStore:
    def __init__(self):
        self.nodes, self.relationships, self.calls = {}, {}, []

    async def create_node(self, node_id, labels, properties):
        self.calls.append(("create_node", node_id))
        self.nodes[node_id] = {"labels": list(labels), "properties": dict(properties)}
        return True

    async def update_node(self, node_id, properties):
        self.calls.append(("update_node", node_id))
        self.nodes.setdefault(node_id, {"labels": [], "properties": {}})
        self.nodes[node_id]["properties"].update(properties)
        return True

    async def get_node(self, node_id):
        return self.nodes.get(node_id)

    async def get_neighbors(self, node_id, relationship_types=None):
        return [{"id": t} for (s, t, r) in self.relationships
                if s == node_id and (relationship_types is None or r in relationship_types)]

    async def create_relationship(self, source_id, target_id, relationship_type, properties=None):
        self.calls.append(("create_relationship", source_id, target_id))
        self.relationships[(source_id, target_id, relationship_type)] = dict(properties or {})
        return True

    async def delete_relationship(self, source_id, target_id, relationship_type):
        self.calls.append(("delete_relationship", source_id, target_id))
        self.relationships.pop((source_id, target_id, relationship_type), None)
        return True

    def writes(self):
        return [c for c in self.calls]


class FakeVectorStore:
    def __init__(self):
        self.vectors = {}
        self.calls = 0

    async def upsert_vectors(self, vectors):
        self.calls += 1
        for vid, vec, meta in vectors:
            self.vectors[vid] = (list(vec), dict(meta))
        return True


SCHEMA = {
    "name": "s", "version": "1.0.0",
    "dimensions": [
        {"name": "alpha", "description": "", "anchors": ["a one", "a two"]},
        {"name": "beta", "description": "", "anchors": ["b one"]},
    ],
}

RECORD = {"title": "vector index", "description": "hnsw graph", "tags": ["ann", "search"]}


@pytest.fixture
def schema():
    return load_schema(SCHEMA)


@pytest.fixture
def stores():
    return FakeGraphStore(), FakeVectorStore()


@pytest.fixture
def pipeline(schema, stores):
    graph, vectors = stores
    return IngestPipeline(schema, CountingEmbedder(), graph_store=graph, vector_store=vectors)


async def test_ingest_returns_entity_and_projection(pipeline):
    result = await pipeline.ingest(RECORD)
    assert isinstance(result, IngestResult)
    assert result.entity.title == "vector index"
    assert set(result.projection.scores) == {"alpha", "beta"}
    assert result.written is True


async def test_entity_id_is_the_deterministic_fr01_id(pipeline):
    a = await pipeline.ingest(RECORD)
    b = await pipeline.ingest(dict(RECORD))
    assert a.entity.id == b.entity.id
    assert a.entity.id.startswith("kse-")


async def test_graph_receives_node_and_scored_edges(pipeline, stores):
    graph, _ = stores
    result = await pipeline.ingest(RECORD)
    assert result.entity.id in graph.nodes
    assert len([k for k in graph.relationships if k[0] == result.entity.id]) == 2


async def test_vector_store_receives_the_entity_embedding(pipeline, stores):
    _, vectors = stores
    result = await pipeline.ingest(RECORD)
    assert result.entity.id in vectors.vectors
    vec, meta = vectors.vectors[result.entity.id]
    assert len(vec) == 3
    assert meta["content_hash"] == result.projection.content_hash


async def test_reingesting_unchanged_content_writes_nothing(pipeline, stores):
    graph, vectors = stores
    await pipeline.ingest(RECORD)
    graph_writes, vector_writes = len(graph.writes()), vectors.calls

    result = await pipeline.ingest(dict(RECORD))

    assert result.written is False
    assert len(graph.writes()) == graph_writes
    assert vectors.calls == vector_writes


async def test_changed_content_is_rewritten(pipeline, stores):
    graph, _ = stores
    await pipeline.ingest(RECORD)
    before = len(graph.writes())
    result = await pipeline.ingest({**RECORD, "description": "different"})
    assert result.written is True
    assert len(graph.writes()) > before


async def test_tag_reorder_does_not_rewrite(pipeline):
    """FR-01's set-like tags and FR-02's sorted text, verified end to end."""
    await pipeline.ingest(RECORD)
    result = await pipeline.ingest({**RECORD, "tags": ["search", "ann"]})
    assert result.written is False


async def test_anchors_are_embedded_once_not_per_record(schema, stores):
    """Anchors are schema-level and constant. Re-embedding them per record
    would make ingest cost scale with the schema size for no reason."""
    graph, vectors = stores
    embedder = CountingEmbedder()
    p = IngestPipeline(schema, embedder, graph_store=graph, vector_store=vectors)

    for i in range(4):
        await p.ingest({**RECORD, "description": f"body {i}"})

    anchor_texts = [a for d in schema.dimensions for a in d.anchors]
    for anchor in anchor_texts:
        assert embedder.texts.count(anchor) == 1, f"{anchor!r} embedded more than once"


async def test_malformed_record_raises_before_any_write(pipeline, stores):
    """A bad record must not leave a half-written graph behind it."""
    graph, vectors = stores
    with pytest.raises(ValueError):
        await pipeline.ingest({"title": "no description"})
    assert graph.nodes == {}
    assert vectors.vectors == {}


async def test_pipeline_works_with_no_stores(schema):
    """Projection alone is a valid use — stores are optional."""
    p = IngestPipeline(schema, CountingEmbedder())
    result = await p.ingest(RECORD)
    assert result.projection.scores
    assert result.written is False  # nothing to write to


async def test_ingest_many_returns_one_result_per_record(pipeline):
    results = await pipeline.ingest_many([RECORD, {**RECORD, "description": "other"}])
    assert len(results) == 2
    assert results[0].entity.id != results[1].entity.id


async def test_ingest_makes_no_network_calls(no_network, pipeline):
    assert (await pipeline.ingest(RECORD)).written is True


async def test_vector_only_pipeline_still_writes(schema):
    """A pipeline with a vector store but no graph must still store vectors.

    Regression: the first implementation gated the vector write on the graph's
    verdict and then re-checked for a graph, so this configuration wrote
    nothing at all and reported success.
    """
    vectors = FakeVectorStore()
    p = IngestPipeline(schema, CountingEmbedder(), vector_store=vectors)
    result = await p.ingest(RECORD)
    assert result.written is True
    assert result.entity.id in vectors.vectors
