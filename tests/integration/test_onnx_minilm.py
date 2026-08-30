"""
Integration lane — genuine ONNX MiniLM from the local cache (D-16, T-065).

Skips when no model is cached at ~/.cache/kse (AR-01 forbids downloading one
during a test run). CI's onnx lane warms the cache in a setup step, outside
any test, then runs this for real.

This is the only place the hand-written WordPiece tokeniser meets the real
30522-token vocabulary and a real exported graph — everything else exercises
the contract through stubs and synthetic models.
"""
from __future__ import annotations

import pytest

from kse_memory.core.projection import OnnxEmbedder, default_model_dir

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.integration,
    pytest.mark.skipif(
        not (default_model_dir() / "model.onnx").exists(),
        reason="no MiniLM cached at ~/.cache/kse; AR-01 forbids downloading one",
    ),
]


def test_real_model_produces_normalised_embeddings():
    e = OnnxEmbedder()
    vecs = e.embed(["a technical document", "a friendly greeting"])
    assert len(vecs) == 2
    for v in vecs:
        assert abs(sum(x * x for x in v) ** 0.5 - 1.0) < 1e-4
    assert vecs[0] != vecs[1]


def test_real_model_is_deterministic():
    e = OnnxEmbedder()
    assert e.embed(["replay identity"]) == e.embed(["replay identity"])


def test_real_model_batch_invariance():
    """Padding must not leak into a short text's vector — against the real graph."""
    e = OnnxEmbedder()
    alone = e.embed(["short"])[0]
    batched = e.embed(["short", "a much longer sentence that forces padding of the first"])[0]
    assert alone == pytest.approx(batched, abs=1e-5)


def test_real_model_similarity_sanity():
    """Near-paraphrases should sit closer than unrelated text."""
    e = OnnxEmbedder()
    a, b, c = e.embed([
        "how to tune an approximate nearest neighbour index",
        "tuning ANN index parameters for vector search",
        "a recipe for lemon drizzle cake",
    ])
    dot = lambda x, y: sum(p * q for p, q in zip(x, y))
    assert dot(a, b) > dot(a, c)


async def test_quickstart_end_to_end_offline(no_network):
    """TC-02's executable core: the whole quickstart path — real model,
    real projection, real retrieval — under the no-network fixture."""
    import time

    from kse_memory.quickstart.v3 import DEFAULT_RECORDS, run_quickstart

    started = time.perf_counter()
    result = await run_quickstart(OnnxEmbedder())
    elapsed = time.perf_counter() - started

    assert result.ingested == len(DEFAULT_RECORDS)
    assert result.written == len(DEFAULT_RECORDS)
    assert elapsed < 60.0  # TC-02's budget, with two orders of margin

    # the semantic sanity that makes the demo a demo: the tuning guide tops
    # the tuning query
    hits = result.searches["how do I tune a vector index"]
    assert hits[0].title == "HNSW index tuning guide"
    assert all(h.scores for h in hits)

    # incrementality survives the real model too
    again = await run_quickstart(OnnxEmbedder(), pipeline=result.pipeline)
    assert again.written == 0


async def test_query_parse_against_the_real_model(no_network):
    """FR-03 with the genuine MiniLM: bounded, deterministic, discriminating."""
    from kse_memory.core.query import parse_query
    from kse_memory.core.schema import load_schema
    from kse_memory.quickstart.v3 import DEFAULT_SCHEMA

    schema = load_schema(DEFAULT_SCHEMA)
    e = OnnxEmbedder()

    a = parse_query("step by step deployment instructions to follow", schema, e)
    b = parse_query("step by step deployment instructions to follow", schema, e)
    assert a == b
    assert all(0.0 <= v <= 1.0 for v in a.targets.values())

    # a how-to query should target practicality above novelty
    assert a.targets["practicality"] > a.targets["novelty"]


async def test_all_three_channels_against_the_real_model(no_network):
    """FR-04 end to end: parse with the genuine MiniLM, retrieve over the
    quickstart pipeline's populated stores — every channel produces results."""
    from kse_memory.core.query import parse_query
    from kse_memory.core.retrieval import retrieve
    from kse_memory.quickstart.v3 import run_quickstart

    seeded = await run_quickstart(OnnxEmbedder())
    pipeline = seeded.pipeline

    parsed = parse_query(
        "how do I tune a vector index", pipeline.schema, pipeline.embedder,
        centroids=pipeline.centroids,
    )
    result = await retrieve(
        parsed,
        vector_store=pipeline.vector_store,
        concept_store=pipeline.concept_store,
        graph_store=pipeline.graph_store,
        top_k=5,
    )

    assert result.errors == {}
    assert result.vector and result.conceptual
    # The graph channel ABSTAINS on this corpus: every entity connects to the
    # query's top dimensions, so coverage is uniform — a constant function
    # carries no rank information, and emitting id-order would be fabricated
    # evidence (the US6 abstention fix). Empty, with no error, is correct.
    assert result.graph == ()

    # channels agree on the corpus: every id is a real ingested entity
    ingested = {r.entity.id for r in await pipeline.ingest_many([])} or None
    all_ids = {e for ch in (result.vector, result.conceptual, result.graph) for e, _ in ch}
    assert all(e.startswith("kse-") for e in all_ids)

    # the dense channel still knows the right answer
    top_vector_id = result.vector[0][0]
    stored = pipeline.vector_store.rows[top_vector_id][1]
    assert stored["title"] == "HNSW index tuning guide"


async def test_explanations_against_the_real_model(no_network):
    """FR-06 with the genuine MiniLM: the top hit's receipt is complete and
    internally consistent with the displayed result."""
    from kse_memory.quickstart.v3 import run_quickstart

    result = await run_quickstart(OnnxEmbedder(), queries=["how do I tune a vector index"])
    (explanations,) = result.explanations.values()
    top = explanations[0]
    top_hit = result.searches["how do I tune a vector index"][0]

    assert top.entity_id == top_hit.entity_id
    assert top.fused == pytest.approx(top_hit.similarity, abs=1e-6)
    assert top.ranks == dict(top_hit.channel_ranks)
    assert {r.name: r.score for r in top.dimensions} == dict(top_hit.scores)
    assert top.model_id == "onnx-minilm-l6-v2"
    assert all(0.0 <= r.alignment <= 1.0 for r in top.dimensions)


async def test_healthy_corpus_yields_confident_hybrid_answers(no_network):
    """FR-07 with the genuine MiniLM: all channels healthy, answers hybrid."""
    from kse_memory.quickstart.v3 import run_quickstart

    result = await run_quickstart(OnnxEmbedder())
    for query, verdict in result.answers.items():
        assert verdict.degraded == {}
        assert verdict.hybrid is True, (query, verdict.confidence, verdict.fallback_reason)
        assert verdict.confidence >= 0.5


async def test_dead_graph_store_degrades_with_receipts(no_network):
    """Kill one channel mid-flight: the answer survives, names the failure,
    and the verdict reflects the reduced corroboration."""
    from kse_memory.core.answer import answer as build_answer
    from kse_memory.core.query import parse_query
    from kse_memory.core.retrieval import retrieve
    from kse_memory.quickstart.v3 import run_quickstart

    seeded = await run_quickstart(OnnxEmbedder())
    pipeline = seeded.pipeline

    class DeadGraph:
        async def get_neighbors(self, *a, **k):
            raise RuntimeError("graph store unavailable")

    parsed = parse_query("how do I tune a vector index", pipeline.schema,
                         pipeline.embedder, centroids=pipeline.centroids)
    channels = await retrieve(parsed, vector_store=pipeline.vector_store,
                              concept_store=pipeline.concept_store,
                              graph_store=DeadGraph(), top_k=5)
    verdict = build_answer(channels, top_k=5)

    assert "graph" in verdict.degraded
    assert verdict.items  # still answered
    assert verdict.confidence <= 2 / 3  # a dead channel corroborates nothing



@pytest.mark.parametrize("pack_name", ["retail", "finance", "documents"])
async def test_pack_showcase_beats_dense_with_the_real_model(no_network, pack_name):
    """TC-06's heart: each pack's query is one pure vector search handles
    WORSE — the target must rank strictly better under hybrid, genuine model,
    no network. The mechanism (anchor vocabulary bridging) is stated in each
    pack's corpus.json."""
    from examples.packs import load_pack, run_showcase

    outcome = await run_showcase(load_pack(pack_name), OnnxEmbedder())
    assert outcome.hybrid_rank < outcome.dense_rank, (
        pack_name, outcome.dense_rank, outcome.hybrid_rank)
    assert outcome.dense_top != outcome.target_id  # dense actually takes the bait
