"""
TC-02 (partial) — the v3 quickstart path over IngestPipeline.

Written test-first per GOV-04.

Scope honesty: FR-03..FR-05 (query parse, concurrent retrieval, fusion) are
unbuilt, so quickstart demonstrates ingest -> project -> DENSE retrieval with
per-dimension receipts, and says so. TC-02's "hybrid results" clause cannot
close until those FRs land; what closes here is: no API key, no network, no
CUDA, sample corpus in, ranked results with dimension scores out.
"""
from __future__ import annotations

import pytest

from tests.conftest import StubEmbedder
from kse_memory.quickstart.v3 import (
    DEFAULT_QUERIES,
    DEFAULT_RECORDS,
    DEFAULT_SCHEMA,
    QuickstartResult,
    run_quickstart,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.component]


async def test_quickstart_runs_offline_with_stub(no_network, stub_embedder):
    """The core TC-02 clauses: no key, no network, results come back."""
    result = await run_quickstart(embedder=stub_embedder)
    assert isinstance(result, QuickstartResult)
    assert result.ingested == len(DEFAULT_RECORDS)
    assert set(result.searches) == set(DEFAULT_QUERIES)
    assert all(hits for hits in result.searches.values())


async def test_results_carry_dimension_receipts(stub_embedder):
    """D-12's aha needs receipts: every hit shows its per-dimension scores."""
    result = await run_quickstart(embedder=stub_embedder)
    for hits in result.searches.values():
        for hit in hits:
            assert set(hit.scores) == set(d["name"] for d in DEFAULT_SCHEMA["dimensions"])
            assert all(0.0 <= v <= 1.0 for v in hit.scores.values())


async def test_results_are_ranked_and_bounded(stub_embedder):
    result = await run_quickstart(embedder=stub_embedder, top_k=3)
    for hits in result.searches.values():
        assert len(hits) <= 3
        sims = [h.similarity for h in hits]
        assert sims == sorted(sims, reverse=True)


async def test_rerun_is_incremental(stub_embedder):
    """Second ingest of the same corpus must write nothing (FR-01/FR-02)."""
    first = await run_quickstart(embedder=stub_embedder)
    assert first.written == len(DEFAULT_RECORDS)
    second = await run_quickstart(embedder=stub_embedder, pipeline=first.pipeline)
    assert second.written == 0


async def test_custom_records_and_queries(stub_embedder):
    records = [
        {"title": "alpha doc", "description": "about widgets"},
        {"title": "beta doc", "description": "about sprockets"},
    ]
    result = await run_quickstart(
        embedder=stub_embedder, records=records, queries=["widgets"]
    )
    assert result.ingested == 2
    assert list(result.searches) == ["widgets"]


async def test_default_schema_is_domain_neutral():
    """TC-04 adjacent: the demo schema must not smuggle retail vocabulary back."""
    names = {d["name"] for d in DEFAULT_SCHEMA["dimensions"]}
    forbidden = {"elegance", "comfort", "boldness", "modernity", "minimalism",
                 "luxury", "seasonality"}
    assert not (names & forbidden)


async def test_deterministic_across_runs(stub_embedder):
    a = await run_quickstart(embedder=stub_embedder)
    b = await run_quickstart(embedder=StubEmbedder())
    for query in a.searches:
        assert [h.entity_id for h in a.searches[query]] == [h.entity_id for h in b.searches[query]]
