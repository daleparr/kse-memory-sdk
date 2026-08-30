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


async def test_every_result_carries_a_full_explanation(stub_embedder):
    """FR-06's contract, end to end through the quickstart path."""
    result = await run_quickstart(embedder=stub_embedder)
    for query, hits in result.searches.items():
        explanations = result.explanations[query]
        assert [e.entity_id for e in explanations] == [h.entity_id for h in hits]
        for explanation in explanations:
            assert explanation.query == query
            assert set(explanation.ranks) == {"vector", "conceptual", "graph"}
            assert explanation.dimensions  # every ingested entity has scores
            assert explanation.degraded == {}


async def test_answers_state_what_they_are(stub_embedder):
    """FR-07 through the quickstart: every query gets a verdict."""
    result = await run_quickstart(embedder=stub_embedder)
    for query in result.searches:
        verdict = result.answers[query]
        assert 0.0 <= verdict.confidence <= 1.0
        assert verdict.hybrid or verdict.dense_only or verdict.fallback_reason
        # the displayed hits ARE the verdict's ranking
        assert [h.entity_id for h in result.searches[query]] == \
               [i.entity_id for i in verdict.items]
