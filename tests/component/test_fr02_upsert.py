"""
T-008, third limb — incremental graph-edge upsert.

Written test-first per GOV-04.

"Incremental" is the whole point: re-projecting unchanged content must not
write. A projection already carries its replay identity (content hash + schema
version + model id), so the store can be asked what it has and the write
skipped when nothing moved. Without that, every re-ingest rewrites the graph
and "incremental" is a word in a spec rather than a property of the system.
"""
from __future__ import annotations

import pytest

from kse_memory.core.ingest import normalise_record
from kse_memory.core.projection import project, upsert_projection
from kse_memory.core.schema import load_schema

pytestmark = [pytest.mark.asyncio, pytest.mark.component]


class FakeGraphStore:
    """Records every mutation so tests can assert on writes, not just state."""

    def __init__(self):
        self.nodes = {}
        self.relationships = {}
        self.calls = []

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
        self.calls.append(("get_node", node_id))
        return self.nodes.get(node_id)

    async def get_neighbors(self, node_id, relationship_types=None):
        self.calls.append(("get_neighbors", node_id))
        return [
            {"id": target}
            for (src, target, rel) in self.relationships
            if src == node_id and (relationship_types is None or rel in relationship_types)
        ]

    async def create_relationship(self, source_id, target_id, relationship_type, properties=None):
        self.calls.append(("create_relationship", source_id, target_id))
        self.relationships[(source_id, target_id, relationship_type)] = dict(properties or {})
        return True

    async def delete_relationship(self, source_id, target_id, relationship_type):
        self.calls.append(("delete_relationship", source_id, target_id))
        self.relationships.pop((source_id, target_id, relationship_type), None)
        return True

    def writes(self):
        return [c for c in self.calls if c[0] != "get_node"]


class StubEmbedder:
    model_id = "stub-v1"

    def embed(self, texts):
        return [[float(len(t) % 7) + 1.0, 2.0, 3.0] for t in texts]


SCHEMA = {
    "name": "s",
    "version": "1.0.0",
    "dimensions": [
        {"name": "alpha", "description": "", "anchors": ["a"]},
        {"name": "beta", "description": "", "anchors": ["b"]},
    ],
}


@pytest.fixture
def schema():
    return load_schema(SCHEMA)


@pytest.fixture
def entity():
    return normalise_record({"title": "t", "description": "d"})


async def test_first_upsert_creates_node_and_one_edge_per_dimension(schema, entity):
    store = FakeGraphStore()
    p = project(entity, schema, StubEmbedder())
    written = await upsert_projection(p, store)

    assert written is True
    assert entity.id in store.nodes
    edges = [k for k in store.relationships if k[0] == entity.id]
    assert len(edges) == len(schema.names())


async def test_edge_weight_carries_the_dimension_score(schema, entity):
    store = FakeGraphStore()
    p = project(entity, schema, StubEmbedder())
    await upsert_projection(p, store)

    for (src, target, _), props in store.relationships.items():
        dimension = target.rsplit(":", 1)[-1]
        assert props["score"] == pytest.approx(p.scores[dimension])


async def test_reupsert_of_identical_projection_writes_nothing(schema, entity):
    """The incremental guarantee: unchanged content must cost zero writes."""
    store = FakeGraphStore()
    p = project(entity, schema, StubEmbedder())
    await upsert_projection(p, store)
    before = len(store.writes())

    written = await upsert_projection(p, store)

    assert written is False
    assert len(store.writes()) == before


async def test_changed_content_triggers_a_rewrite(schema):
    store = FakeGraphStore()
    a = project(normalise_record({"title": "t", "description": "one"}), schema, StubEmbedder())
    b = project(normalise_record({"title": "t", "description": "two"}), schema, StubEmbedder())
    await upsert_projection(a, store)
    before = len(store.writes())

    assert await upsert_projection(b, store) is True
    assert len(store.writes()) > before


async def test_schema_bump_triggers_a_rewrite(schema, entity):
    """A schema change invalidates projections computed under the old one."""
    store = FakeGraphStore()
    await upsert_projection(project(entity, schema, StubEmbedder()), store)
    before = len(store.writes())

    bumped = load_schema({**SCHEMA, "version": "1.1.0"})
    assert await upsert_projection(project(entity, bumped, StubEmbedder()), store) is True
    assert len(store.writes()) > before


async def test_stale_edges_are_removed_when_dimensions_disappear(schema, entity):
    """A narrowed schema must not leave orphaned edges scoring dead dimensions."""
    store = FakeGraphStore()
    await upsert_projection(project(entity, schema, StubEmbedder()), store)

    narrowed = load_schema({
        "name": "s", "version": "2.0.0",
        "dimensions": [{"name": "alpha", "description": "", "anchors": ["a"]}],
    })
    await upsert_projection(project(entity, narrowed, StubEmbedder()), store)

    remaining = {k[1].rsplit(":", 1)[-1] for k in store.relationships if k[0] == entity.id}
    assert remaining == {"alpha"}


async def test_upsert_makes_no_network_calls(no_network, schema, entity):
    store = FakeGraphStore()
    assert await upsert_projection(project(entity, schema, StubEmbedder()), store) is True
