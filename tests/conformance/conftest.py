"""
Conformance lane (T-066): shared behavioural suites per interface.

Each registry maps a backend name to an async factory returning a connected
instance, or raising pytest.skip when the backend is unavailable — server
backends carry requires_backend markers and skip in CI until a live instance
is configured (D-16: Tier 1 in CI, Tier 2 runnable locally).
"""
from __future__ import annotations

import pytest

from kse_memory.backends.mock import MockVectorStore
from kse_memory.core.config import VectorStoreConfig
from kse_memory.quickstart.v3 import _GraphStore, _VectorIndex


async def _mock_vector():
    store = MockVectorStore(VectorStoreConfig())
    await store.connect()
    return store


async def _quickstart_vector():
    return _VectorIndex()


async def _quickstart_graph():
    return _GraphStore()


async def _networkx_graph():
    from kse_memory.backends.networkx_graph import NetworkXGraphStore

    store = NetworkXGraphStore()
    await store.connect()
    return store


async def _arangodb_graph():
    pytest.skip("no live ArangoDB configured (requires_backend: arangodb)")


async def _neo4j_graph():
    try:
        import neo4j  # noqa: F401
    except ImportError:
        pytest.skip("neo4j driver not installed (requires_backend: neo4j)")
    pytest.skip("no live Neo4j configured (requires_backend: neo4j)")


#: name -> (factory, supports_full_interface)
VECTOR_BACKENDS = {
    "mock": (_mock_vector, True),
    "quickstart-index": (_quickstart_vector, False),
}

GRAPH_BACKENDS = {
    "quickstart-memory": (_quickstart_graph, True),
    "networkx": (_networkx_graph, True),
    "neo4j": (_neo4j_graph, True),
    "arangodb": (_arangodb_graph, True),
}


@pytest.fixture(params=list(VECTOR_BACKENDS))
async def vector_store(request):
    factory, full = VECTOR_BACKENDS[request.param]
    store = await factory()
    store._conformance_full = full
    return store


@pytest.fixture(params=list(GRAPH_BACKENDS))
async def graph_store(request):
    factory, _ = GRAPH_BACKENDS[request.param]
    return await factory()


def full_interface(store) -> bool:
    return getattr(store, "_conformance_full", False)
