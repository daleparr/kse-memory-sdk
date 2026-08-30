"""
GraphStoreInterface conformance (T-066) — the contract as used.

FR-02's upsert and FR-04's traversal rely on: node create/get/update,
relationship create/delete, and get_neighbors returning nodes connected in
EITHER direction with the type filter honoured. Every wired backend must
agree on those semantics or fusion built on them silently diverges.
"""
from __future__ import annotations

import pytest

pytestmark = [pytest.mark.asyncio, pytest.mark.conformance]


async def test_create_then_get_node(graph_store):
    await graph_store.create_node("n1", ["Entity"], {"content_hash": "abc"})
    node = await graph_store.get_node("n1")
    assert node is not None
    assert node["properties"]["content_hash"] == "abc"


async def test_get_unknown_node_returns_none(graph_store):
    assert await graph_store.get_node("missing") is None


async def test_update_merges_properties(graph_store):
    await graph_store.create_node("n1", ["Entity"], {"a": 1})
    await graph_store.update_node("n1", {"b": 2})
    node = await graph_store.get_node("n1")
    assert node["properties"] == {"a": 1, "b": 2}


async def test_neighbors_are_undirected(graph_store):
    """The FR-04 contract: edges written entity->dimension must be walkable
    dimension->entity, or the graph channel goes blind."""
    await graph_store.create_node("e", ["Entity"], {})
    await graph_store.create_node("d", ["Dimension"], {})
    await graph_store.create_relationship("e", "d", "SCORED_AS", {"score": 0.5})

    from_entity = await graph_store.get_neighbors("e", ["SCORED_AS"])
    from_dimension = await graph_store.get_neighbors("d", ["SCORED_AS"])
    assert [n["id"] for n in from_entity] == ["d"]
    assert [n["id"] for n in from_dimension] == ["e"]


async def test_neighbor_type_filter_is_honoured(graph_store):
    await graph_store.create_node("a", ["Entity"], {})
    await graph_store.create_node("b", ["Entity"], {})
    await graph_store.create_relationship("a", "b", "OTHER", {})
    assert await graph_store.get_neighbors("a", ["SCORED_AS"]) == []
    assert [n["id"] for n in await graph_store.get_neighbors("a", ["OTHER"])] == ["b"]


async def test_delete_relationship_removes_the_edge(graph_store):
    await graph_store.create_node("a", ["Entity"], {})
    await graph_store.create_node("b", ["Entity"], {})
    await graph_store.create_relationship("a", "b", "SCORED_AS", {})
    await graph_store.delete_relationship("a", "b", "SCORED_AS")
    assert await graph_store.get_neighbors("a", ["SCORED_AS"]) == []
