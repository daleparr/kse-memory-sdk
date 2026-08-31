"""
Model and config unit tests salvaged from tests/test_core.py (T-068 map).

The rest of that file — KSEMemory CRUD/search against store fixtures — is
covered by the component pipeline suites and the integration lane, and was
deleted per the retirement map once T-025 closed. These survived because
they are genuine unit tests of surfaces that still exist.
"""
from __future__ import annotations

import pytest

from kse_memory.core.config import KSEConfig
from kse_memory.core.memory import KSEMemory
from kse_memory.core.models import SearchQuery, SearchType

pytestmark = pytest.mark.unit


def test_config_round_trips_backend_choices():
    config = KSEConfig.from_dict({
        "debug": True,
        "vector_store": {"backend": "pinecone"},
        "graph_store": {"backend": "neo4j"},
        "concept_store": {"backend": "postgresql"},
    })
    assert config.debug is True
    assert config.vector_store.backend == "pinecone"
    assert config.graph_store.backend == "neo4j"
    assert config.concept_store.backend == "postgresql"


def test_default_config_is_local_everywhere():
    """TC-02 at the config layer: the default stack is in-process."""
    config = KSEConfig()
    assert config.vector_store.backend == "memory"
    assert config.graph_store.backend == "memory"
    assert config.concept_store.backend == "memory"


def test_memory_constructs_uninitialised():
    kse = KSEMemory(KSEConfig())
    assert kse._initialized is False
    assert kse._connected is False


def test_search_query_creation():
    query = SearchQuery(query="comfortable shoes", search_type=SearchType.HYBRID, limit=5)
    assert query.query == "comfortable shoes"
    assert query.search_type == SearchType.HYBRID
    assert query.limit == 5


def test_search_query_with_filters():
    query = SearchQuery(
        query="athletic wear", search_type=SearchType.SEMANTIC, limit=10,
        filters={"category": "Athletic", "price_max": 200},
    )
    assert query.filters["category"] == "Athletic"
    assert query.filters["price_max"] == 200


def test_search_type_enum_values():
    assert SearchType.SEMANTIC.value == "semantic"
    assert SearchType.CONCEPTUAL.value == "conceptual"
    assert SearchType.KNOWLEDGE_GRAPH.value == "knowledge_graph"
    assert SearchType.HYBRID.value == "hybrid"
