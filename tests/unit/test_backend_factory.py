"""
Backend factory unit tests — fresh coverage replacing tests/test_backends.py.

The legacy file's cloud-backend interface tests all errored at setup on
obsolete fixtures, and its configuration assertions enforced the pre-v3
cloud defaults that TC-02 overturned. This covers the factory contract as
it stands; per-backend behaviour belongs to the conformance lane.
"""
from __future__ import annotations

import pytest

from kse_memory.backends import get_concept_store, get_graph_store, get_vector_store
from kse_memory.backends.memory_graph import MemoryGraphStore
from kse_memory.backends.mock import MockVectorStore
from kse_memory.core.config import ConceptStoreConfig, GraphStoreConfig, VectorStoreConfig
from kse_memory.core.dimension_store import InMemoryDimensionStore
from kse_memory.exceptions import BackendError

pytestmark = pytest.mark.unit


def test_default_configs_resolve_to_in_process_stores():
    """TC-02 at the factory layer: defaults need no service and no key."""
    assert isinstance(get_vector_store(VectorStoreConfig()), MockVectorStore)
    assert isinstance(get_graph_store(GraphStoreConfig()), MemoryGraphStore)
    assert isinstance(get_concept_store(ConceptStoreConfig()), InMemoryDimensionStore)


def test_memory_and_mock_are_aliases():
    for name in ("memory", "mock", "MEMORY"):
        assert isinstance(get_vector_store(VectorStoreConfig(backend=name)), MockVectorStore)
        assert isinstance(get_graph_store(GraphStoreConfig(backend=name)), MemoryGraphStore)


def test_unsupported_backends_raise_naming_the_backend():
    with pytest.raises(BackendError, match="nonsense"):
        get_vector_store(VectorStoreConfig(backend="nonsense"))
    with pytest.raises(BackendError, match="nonsense"):
        get_graph_store(GraphStoreConfig(backend="nonsense"))
    with pytest.raises(BackendError, match="nonsense"):
        get_concept_store(ConceptStoreConfig(backend="nonsense"))


def test_backend_error_carries_backend_type():
    error = BackendError("boom", "somebackend")
    assert "boom" in str(error)
    assert error.details["backend_type"] == "somebackend"
