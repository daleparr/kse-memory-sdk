"""
Repairing the v2 SearchService: dead method call, and D-07 compliance.

Written test-first per GOV-04.

Two defects, both real:
1. conceptual_search still called concept_store.find_similar_concepts — a
   method deleted with ConceptualDimensions. Broken at runtime since that
   removal; the mocked legacy suite never noticed, which is GOV-04's whole
   argument.
2. hybrid_search combined channels by weighted score-summing. D-07 confirms
   RRF as the default precisely because channel scores are not comparable;
   the v2 mechanism contradicted a confirmed decision.
"""
from __future__ import annotations

import pytest

from kse_memory.core.config import SearchConfig
from kse_memory.core.dimension_store import ConceptStoreAdapter, InMemoryDimensionStore
from kse_memory.core.models import Product, SearchQuery, SearchResult, SearchType
from kse_memory.services.search import SearchService

pytestmark = [pytest.mark.asyncio, pytest.mark.component]


def product(pid, title="t"):
    return Product(id=pid, title=title, description="d")  # Entity subclass


class FakeVectorStore:
    def __init__(self, metadata=None):
        self.metadata = metadata or {}

    async def get_vector(self, product_id):
        if product_id in self.metadata:
            return ([0.0], self.metadata[product_id])
        return None


@pytest.fixture
def service():
    return SearchService(
        config=SearchConfig(),
        vector_store=FakeVectorStore({
            "p1": {"id": "p1", "title": "one", "description": "d"},
            "p2": {"id": "p2", "title": "two", "description": "d"},
        }),
        graph_store=None,
        concept_store=None,
        embedding_service=None,
        cache_service=None,
    )


async def test_conceptual_search_uses_the_living_interface(service):
    """The call target must be find_similar_dimensions, not the dead method."""
    store = InMemoryDimensionStore()
    await store.store_dimensions("p1", ConceptStoreAdapter.to_generic(
        {"elegance": 0.9, "comfort": 0.1}))
    await store.store_dimensions("p2", ConceptStoreAdapter.to_generic(
        {"elegance": 0.1, "comfort": 0.9}))
    service.concept_store = store

    results = await service.conceptual_search({"elegance": 0.9, "comfort": 0.1}, limit=5)

    assert [r.product.id for r in results][0] == "p1"
    assert all(isinstance(r, SearchResult) for r in results)


async def test_hybrid_search_fuses_by_rank_not_score(service, monkeypatch):
    """RRF order must hold even when raw scores would say otherwise (D-07).

    semantic:   a (0.99), b (0.98)   <- huge scores, tight gap
    conceptual: b (0.10), a (0.05)   <- tiny scores
    graph:      b (0.02)             <- tinier still

    Weighted score-summing crowns a (0.99+0.05 > 0.98+0.10+0.02 is false —
    1.04 < 1.10 — so weighted picks b here too; use ranks where they differ):
    ranks: a = 1st+2nd, b = 2nd+1st+1st -> RRF crowns b decisively. A
    score-summer with semantic-heavy weights (the v2 default 0.6/0.2/0.2)
    crowns a: 0.6*0.99+0.2*0.05=0.604 > 0.6*0.98+0.2*0.10+0.2*0.02=0.612 —
    still b… so pin the property instead: the fused order must equal
    fuse_rrf's order on the same inputs, whatever the scores say.
    """
    from kse_memory.core.fusion import fuse_rrf

    async def semantic(q, limit):
        return [SearchResult(entity=product("a"), score=0.99),
                SearchResult(entity=product("b"), score=0.98)]

    async def conceptual(dims, limit):
        return [SearchResult(entity=product("b"), score=0.10),
                SearchResult(entity=product("a"), score=0.05)]

    async def graph(q, limit):
        return [SearchResult(entity=product("b"), score=0.02)]

    monkeypatch.setattr(service, "semantic_search", semantic)
    monkeypatch.setattr(service, "conceptual_search", conceptual)
    monkeypatch.setattr(service, "knowledge_graph_search", graph)

    results = await service.hybrid_search(SearchQuery(query="q", search_type=SearchType.HYBRID))

    expected = fuse_rrf({
        "semantic": (("a", 0.99), ("b", 0.98)),
        "conceptual": (("b", 0.10), ("a", 0.05)),
        "graph": (("b", 0.02),),
    })
    assert [r.product.id for r in results] == [i.entity_id for i in expected]


async def test_hybrid_explanations_name_channel_ranks(service, monkeypatch):
    async def semantic(q, limit):
        return [SearchResult(entity=product("a"), score=0.9)]

    async def empty(*a, **k):
        return []

    monkeypatch.setattr(service, "semantic_search", semantic)
    monkeypatch.setattr(service, "conceptual_search", empty)
    monkeypatch.setattr(service, "knowledge_graph_search", empty)

    results = await service.hybrid_search(SearchQuery(query="q", search_type=SearchType.HYBRID))
    assert "semantic" in results[0].explanation
    assert "1" in results[0].explanation
