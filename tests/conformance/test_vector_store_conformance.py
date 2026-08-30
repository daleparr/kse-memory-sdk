"""
VectorStoreInterface conformance (T-066).

The core contract — what IngestPipeline and FR-04 retrieval actually rely
on — must hold for every wired backend. The extended contract applies to
backends implementing the full interface.
"""
from __future__ import annotations

import pytest

from tests.conformance.conftest import full_interface

pytestmark = [pytest.mark.asyncio, pytest.mark.conformance]

V = {
    "up": [1.0, 0.0, 0.0],
    "up2": [0.9, 0.1, 0.0],
    "side": [0.0, 1.0, 0.0],
}


async def seed(store):
    # Deliberately seeded in the WRONG order for a [1,0,0] query: a store
    # that returns insertion order instead of computing similarity fails.
    await store.upsert_vectors([
        ("side", V["side"], {"title": "side"}),
        ("up2", V["up2"], {"title": "up2"}),
        ("up", V["up"], {"title": "up"}),
    ])


# ------------------------------------------------------------ core contract
async def test_search_ranks_by_similarity_to_the_query(vector_store):
    """The contract that makes retrieval mean anything: the most similar
    vector comes first. A store returning arbitrary scores does not conform."""
    await seed(vector_store)
    rows = await vector_store.search_vectors([1.0, 0.0, 0.0], top_k=3)
    ids = [r[0] for r in rows]
    assert ids[0] == "up"
    assert ids.index("up2") < ids.index("side")


async def test_search_scores_descend(vector_store):
    await seed(vector_store)
    rows = await vector_store.search_vectors([1.0, 0.0, 0.0], top_k=3)
    scores = [r[1] for r in rows]
    assert scores == sorted(scores, reverse=True)


async def test_search_respects_top_k(vector_store):
    await seed(vector_store)
    assert len(await vector_store.search_vectors([1.0, 0.0, 0.0], top_k=2)) == 2


async def test_upsert_overwrites_by_id(vector_store):
    await seed(vector_store)
    await vector_store.upsert_vectors([("up", V["side"], {"title": "moved"})])
    rows = await vector_store.search_vectors([0.0, 1.0, 0.0], top_k=1)
    assert rows[0][0] in ("up", "side")  # 'up' now points sideways
    rows_all = await vector_store.search_vectors([0.0, 1.0, 0.0], top_k=3)
    moved = [r for r in rows_all if r[0] == "up"]
    assert moved and moved[0][2].get("title") == "moved"


async def test_search_returns_metadata(vector_store):
    await seed(vector_store)
    rows = await vector_store.search_vectors([1.0, 0.0, 0.0], top_k=1)
    assert rows[0][2].get("title") == "up"


# -------------------------------------------------------- extended contract
async def test_get_vector_round_trips(vector_store):
    if not full_interface(vector_store):
        pytest.skip("core-subset backend")
    await seed(vector_store)
    got = await vector_store.get_vector("up")
    assert got is not None
    vector, metadata = got
    assert list(vector) == V["up"]
    assert metadata.get("title") == "up"


async def test_get_unknown_returns_none(vector_store):
    if not full_interface(vector_store):
        pytest.skip("core-subset backend")
    assert await vector_store.get_vector("missing") is None


async def test_delete_removes(vector_store):
    if not full_interface(vector_store):
        pytest.skip("core-subset backend")
    await seed(vector_store)
    await vector_store.delete_vectors(["up"])
    assert await vector_store.get_vector("up") is None
