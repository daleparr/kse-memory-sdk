"""
The v3-native search service: parse -> retrieve -> answer -> explain, boxed.

Written test-first per GOV-04.

Apps needed a single object owning the schema, the embedder and the cached
centroids, so the FR-03..FR-07 spine stops being assembly instructions. The
quickstart is rewired onto it in the same commit — the demo runs the object
apps run, or the demo proves nothing.
"""
from __future__ import annotations

import pytest

from tests.conftest import StubEmbedder
from kse_memory.core.answer import HybridAnswer
from kse_memory.core.dimension_store import InMemoryDimensionStore
from kse_memory.core.pipeline import IngestPipeline
from kse_memory.core.schema import load_schema
from kse_memory.services.hybrid import HybridSearchResponse, HybridSearchService
from kse_memory.quickstart.v3 import _GraphStore, _VectorIndex

pytestmark = [pytest.mark.asyncio, pytest.mark.component]

SCHEMA = {
    "name": "svc", "version": "1.0.0",
    "dimensions": [
        {"name": "depth", "description": "", "anchors": ["deep detail"]},
        {"name": "clarity", "description": "", "anchors": ["plainly said"]},
    ],
}

RECORDS = [
    {"title": "deep dive", "description": "deep detail on internals"},
    {"title": "primer", "description": "plainly said introduction"},
    {"title": "misc", "description": "unrelated notes"},
]


@pytest.fixture
async def service(stub_embedder):
    schema = load_schema(SCHEMA)
    stores = dict(graph_store=_GraphStore(), vector_store=_VectorIndex(),
                  concept_store=InMemoryDimensionStore())
    pipeline = IngestPipeline(schema, stub_embedder, **stores)
    await pipeline.ingest_many(RECORDS)
    return HybridSearchService(schema, stub_embedder, **stores)


async def test_search_returns_a_full_response(service):
    response = await service.search("deep detail")
    assert isinstance(response, HybridSearchResponse)
    assert isinstance(response.answer, HybridAnswer)
    assert response.parsed.text == "deep detail"
    assert len(response.explanations) == len(response.answer.items)
    assert response.answer.items


async def test_results_carry_titles(service):
    response = await service.search("deep detail")
    assert response.titles[response.answer.items[0].entity_id]


async def test_centroids_computed_once_across_searches(schema_service_counter):
    service, embedder = schema_service_counter
    await service.search("first query")
    calls_after_first = embedder.calls
    await service.search("second query")
    # exactly one extra embed call: the second query itself
    assert embedder.calls == calls_after_first + 1


@pytest.fixture
async def schema_service_counter():
    class Counting(StubEmbedder):
        def __init__(self):
            self.calls = 0

        def embed(self, texts):
            self.calls += 1
            return super().embed(texts)

    embedder = Counting()
    schema = load_schema(SCHEMA)
    stores = dict(graph_store=_GraphStore(), vector_store=_VectorIndex(),
                  concept_store=InMemoryDimensionStore())
    pipeline = IngestPipeline(schema, embedder, **stores)
    await pipeline.ingest_many(RECORDS)
    return HybridSearchService(schema, embedder, **stores,
                               centroids=pipeline.centroids), embedder


async def test_missing_stores_degrade_not_raise(stub_embedder):
    service = HybridSearchService(load_schema(SCHEMA), stub_embedder)
    response = await service.search("anything")
    assert response.answer.items == ()
    assert response.answer.confidence == 0.0


async def test_search_is_deterministic(service):
    a = await service.search("deep detail")
    b = await service.search("deep detail")
    assert [i.entity_id for i in a.answer.items] == [i.entity_id for i in b.answer.items]
    assert a.answer.confidence == b.answer.confidence


async def test_no_network(no_network, service):
    assert (await service.search("deep detail")).answer.items
