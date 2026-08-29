"""
Generalising the concept store: schema-driven dimension scores.

Written test-first per GOV-04.

ConceptStoreInterface was typed against ConceptualDimensions — ten hardcoded
fashion axes. A user's schema can name any dimensions it likes, so that type
made the concept store structurally incapable of holding a v3 projection, and
was the last blocker on TC-04.

The generalisation keeps the legacy surface working: the ConceptualDimensions
methods become adapters over the generic ones, so existing callers are not
broken by the change.
"""
from __future__ import annotations

import pytest

from kse_memory.core.dimension_store import (
    DimensionScores,
    InMemoryDimensionStore,
)
from kse_memory.core.interfaces import ConceptStoreInterface

pytestmark = pytest.mark.asyncio


SCORES = DimensionScores(
    schema_name="generic-v1",
    schema_version="1.0.0",
    scores={"technical_depth": 0.8, "accessibility": 0.2},
)


@pytest.fixture
def store():
    return InMemoryDimensionStore()


# --------------------------------------------------------------- value type
def test_scores_carry_schema_identity():
    """Scores without their schema are uninterpretable — 0.8 of what?"""
    assert SCORES.schema_name == "generic-v1"
    assert SCORES.schema_version == "1.0.0"
    assert SCORES["technical_depth"] == 0.8


def test_scores_reject_out_of_range_values():
    with pytest.raises(ValueError, match="0.0"):
        DimensionScores(schema_name="s", schema_version="1.0.0", scores={"d": 1.5})


def test_scores_are_comparable_by_value():
    other = DimensionScores(
        schema_name="generic-v1", schema_version="1.0.0",
        scores={"technical_depth": 0.8, "accessibility": 0.2},
    )
    assert SCORES == other


# ------------------------------------------------------------ generic store
async def test_store_and_get_round_trip(store):
    await store.store_dimensions("e1", SCORES)
    assert await store.get_dimensions("e1") == SCORES


async def test_get_unknown_entity_returns_none(store):
    assert await store.get_dimensions("nope") is None


async def test_delete_removes(store):
    await store.store_dimensions("e1", SCORES)
    assert await store.delete_dimensions("e1") is True
    assert await store.get_dimensions("e1") is None


async def test_arbitrary_dimension_names_are_accepted(store):
    """The whole point: no fixed vocabulary."""
    exotic = DimensionScores(
        schema_name="pharma", schema_version="2.1.0",
        scores={"trial_phase_maturity": 0.9, "regulatory_burden": 0.4},
    )
    await store.store_dimensions("drug-1", exotic)
    assert await store.get_dimensions("drug-1") == exotic


async def test_find_similar_ranks_and_respects_threshold(store):
    await store.store_dimensions("near", DimensionScores(
        schema_name="generic-v1", schema_version="1.0.0",
        scores={"technical_depth": 0.79, "accessibility": 0.21}))
    await store.store_dimensions("far", DimensionScores(
        schema_name="generic-v1", schema_version="1.0.0",
        scores={"technical_depth": 0.05, "accessibility": 0.95}))

    hits = await store.find_similar_dimensions(SCORES, threshold=0.5, limit=10)
    ids = [h[0] for h in hits]
    assert ids[0] == "near"
    assert "far" not in ids


async def test_find_similar_only_compares_within_the_same_schema(store):
    """Two schemas may both have a 'risk' dimension meaning different things."""
    await store.store_dimensions("other-schema", DimensionScores(
        schema_name="different", schema_version="1.0.0",
        scores={"technical_depth": 0.8, "accessibility": 0.2}))
    hits = await store.find_similar_dimensions(SCORES, threshold=0.0, limit=10)
    assert [h[0] for h in hits] == []


async def test_find_similar_respects_limit(store):
    for i in range(5):
        await store.store_dimensions(f"e{i}", DimensionScores(
            schema_name="generic-v1", schema_version="1.0.0",
            scores={"technical_depth": 0.8, "accessibility": 0.2}))
    assert len(await store.find_similar_dimensions(SCORES, threshold=0.0, limit=2)) == 2


# ------------------------------------------------------ legacy compatibility
async def test_legacy_conceptual_dimensions_still_store(store):
    """Existing callers must not break on the generalisation."""
    from kse_memory.core.models import ConceptualDimensions

    legacy = ConceptualDimensions(elegance=0.7, comfort=0.3)
    await store.store_conceptual_dimensions("p1", legacy)

    generic = await store.get_dimensions("p1")
    assert generic is not None
    assert generic["elegance"] == pytest.approx(0.7)


async def test_legacy_getter_returns_a_legacy_object(store):
    from kse_memory.core.models import ConceptualDimensions

    await store.store_conceptual_dimensions("p1", ConceptualDimensions(elegance=0.7))
    back = await store.get_conceptual_dimensions("p1")
    assert isinstance(back, ConceptualDimensions)
    assert back.elegance == pytest.approx(0.7)


async def test_unmigrated_backend_fails_loudly_not_silently():
    """A backend that has not implemented the generic surface must say so."""

    class LegacyOnlyBackend(ConceptStoreInterface):
        async def connect(self): return True
        async def disconnect(self): return True
        async def store_conceptual_dimensions(self, product_id, dimensions): return True
        async def get_conceptual_dimensions(self, product_id): return None
        async def delete_conceptual_dimensions(self, product_id): return True
        async def find_similar_concepts(self, dimensions, threshold=0.8, limit=10): return []
        async def get_dimension_statistics(self): return {}

    backend = LegacyOnlyBackend()
    with pytest.raises(NotImplementedError, match="LegacyOnlyBackend"):
        await backend.store_dimensions("e1", SCORES)


async def test_no_network(no_network, store):
    await store.store_dimensions("e1", SCORES)
    assert await store.get_dimensions("e1") == SCORES
