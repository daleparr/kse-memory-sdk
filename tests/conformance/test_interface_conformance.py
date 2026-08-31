"""
Static interface conformance (US9 / TC-09, AR-05).

Behavioural conformance needs a live backend; THIS suite needs only the
class. It catches the defect family found twice on this branch — a backend
declaring an interface while implementing the right behaviour under the
wrong method names, leaving it impossible to instantiate (MongoDBBackend,
then ArangoDBBackend) — and it runs for every registered backend, always,
server or no server.
"""
from __future__ import annotations

import importlib
import inspect

import pytest

pytestmark = pytest.mark.conformance

#: Every backend class the factories can hand out, by interface.
REGISTERED = {
    "graph": [
        ("kse_memory.backends.memory_graph", "MemoryGraphStore"),
        ("kse_memory.backends.networkx_graph", "NetworkXGraphStore"),
        ("kse_memory.backends.neo4j", "Neo4jBackend"),
        ("kse_memory.backends.arangodb", "ArangoDBBackend"),
    ],
    "vector": [
        ("kse_memory.backends.mock", "MockVectorStore"),
    ],
}

GRAPH_CONTRACT = {
    "connect", "disconnect", "create_node", "update_node", "delete_node",
    "get_node", "get_neighbors", "create_relationship", "delete_relationship",
    "find_path", "execute_query",
}


def _load(module, name):
    try:
        return getattr(importlib.import_module(module), name)
    except ImportError as exc:
        pytest.skip(f"{module} not importable here: {exc}")


@pytest.mark.parametrize("module,name", REGISTERED["graph"])
def test_graph_backend_satisfies_the_abc(module, name):
    """Zero unimplemented abstract methods — or the class is a lie."""
    cls = _load(module, name)
    missing = sorted(getattr(cls, "__abstractmethods__", frozenset()))
    assert not missing, f"{name} cannot be instantiated; unimplemented: {missing}"


@pytest.mark.parametrize("module,name", REGISTERED["graph"])
def test_graph_backend_methods_are_coroutines_with_compatible_arity(module, name):
    cls = _load(module, name)
    for method_name in sorted(GRAPH_CONTRACT):
        method = getattr(cls, method_name, None)
        assert method is not None, f"{name} lacks {method_name}"
        assert inspect.iscoroutinefunction(method), f"{name}.{method_name} is not async"
    # spot-check the traversal signature FR-04 depends on
    params = list(inspect.signature(cls.get_neighbors).parameters)
    assert params[1] == "node_id" and "relationship_types" in params
