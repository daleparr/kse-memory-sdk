"""
MongoDBBackend: conform to ConceptStoreInterface, and to the TC-04 surface.

Written test-first per GOV-04.

The backend declared ConceptStoreInterface but implemented its methods under
different names — store_product_concepts rather than store_conceptual_dimensions,
and so on. Five abstract methods were therefore never satisfied, so the class
could not be instantiated at all: any attempt raised TypeError. The
functionality existed; the contract was never met.

No MongoDB server is required here. Motor is exercised through a fake
collection, which is what these tests can honestly cover: document shape and
query construction. Behaviour against a real server is not claimed.
"""
from __future__ import annotations

import pytest

from kse_memory.core.dimension_store import DimensionScores

motor_backend = pytest.importorskip("kse_memory.backends.mongodb")

pytestmark = [pytest.mark.asyncio, pytest.mark.component]


class FakeCursor:
    def __init__(self, docs):
        self._docs = list(docs)

    def __aiter__(self):
        async def gen():
            for d in self._docs:
                yield d
        return gen()


class FakeCollection:
    """Just enough motor surface for the concept-store methods."""

    def __init__(self):
        self.docs = {}

    async def replace_one(self, flt, doc, upsert=False):
        key = flt.get("product_id")
        self.docs[key] = dict(doc)
        return type("R", (), {"upserted_id": key, "modified_count": 1})()

    async def find_one(self, flt):
        return self.docs.get(flt.get("product_id"))

    async def delete_one(self, flt):
        existed = flt.get("product_id") in self.docs
        self.docs.pop(flt.get("product_id"), None)
        return type("R", (), {"deleted_count": 1 if existed else 0})()

    def find(self, flt=None):
        docs = list(self.docs.values())
        if flt:
            for k, v in flt.items():
                docs = [d for d in docs if d.get(k) == v]
        return FakeCursor(docs)

    async def count_documents(self, flt):
        return len(self.docs)


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setattr(motor_backend, "MONGODB_AVAILABLE", True)
    cfg = type("Cfg", (), {"uri": "mongodb://x", "database": "kse"})()
    b = motor_backend.MongoDBBackend(cfg)
    b.products_collection = FakeCollection()
    b.spaces_collection = FakeCollection()
    b._connected = True
    return b


SCORES = DimensionScores(
    schema_name="pharma", schema_version="2.1.0",
    scores={"trial_phase_maturity": 0.9, "regulatory_burden": 0.4},
)


# ------------------------------------------------------------- the regression
def test_backend_can_be_instantiated(monkeypatch):
    """Previously TypeError: 5 abstract methods were never implemented."""
    monkeypatch.setattr(motor_backend, "MONGODB_AVAILABLE", True)
    cfg = type("Cfg", (), {"uri": "mongodb://x", "database": "kse"})()
    assert motor_backend.MongoDBBackend(cfg) is not None


def test_no_unimplemented_abstract_methods():
    assert not getattr(motor_backend.MongoDBBackend, "__abstractmethods__", frozenset())


# ------------------------------------------------------------ generic surface
async def test_store_and_get_dimensions(backend):
    await backend.store_dimensions("e1", SCORES)
    assert await backend.get_dimensions("e1") == SCORES


async def test_arbitrary_dimension_names_survive_the_round_trip(backend):
    """TC-04: Mongo must not impose a vocabulary any more than the interface does."""
    await backend.store_dimensions("e1", SCORES)
    got = await backend.get_dimensions("e1")
    assert set(got.scores) == {"trial_phase_maturity", "regulatory_burden"}


async def test_get_unknown_returns_none(backend):
    assert await backend.get_dimensions("missing") is None


async def test_delete_dimensions(backend):
    await backend.store_dimensions("e1", SCORES)
    assert await backend.delete_dimensions("e1") is True
    assert await backend.get_dimensions("e1") is None
    assert await backend.delete_dimensions("e1") is False


async def test_find_similar_is_scoped_to_schema(backend):
    await backend.store_dimensions("same", SCORES)
    await backend.store_dimensions("other", DimensionScores(
        schema_name="different", schema_version="1.0.0",
        scores={"trial_phase_maturity": 0.9, "regulatory_burden": 0.4}))

    hits = await backend.find_similar_dimensions(SCORES, threshold=0.0, limit=10)
    assert [h[0] for h in hits] == ["same"]


async def test_find_similar_respects_threshold_and_limit(backend):
    await backend.store_dimensions("near", SCORES)
    await backend.store_dimensions("far", DimensionScores(
        schema_name="pharma", schema_version="2.1.0",
        scores={"trial_phase_maturity": 0.0, "regulatory_burden": 1.0}))

    assert [h[0] for h in await backend.find_similar_dimensions(SCORES, threshold=0.99, limit=10)] == ["near"]
    assert len(await backend.find_similar_dimensions(SCORES, threshold=0.0, limit=1)) == 1


async def test_dimension_statistics_are_per_dimension(backend):
    await backend.store_dimensions("a", SCORES)
    await backend.store_dimensions("b", DimensionScores(
        schema_name="pharma", schema_version="2.1.0",
        scores={"trial_phase_maturity": 0.5, "regulatory_burden": 0.2}))

    stats = await backend.get_dimension_statistics()
    assert set(stats) == {"trial_phase_maturity", "regulatory_burden"}
    assert stats["trial_phase_maturity"]["mean"] == pytest.approx(0.7)
    assert stats["trial_phase_maturity"]["max"] == pytest.approx(0.9)


async def test_operations_require_a_connection(backend):
    backend._connected = False
    with pytest.raises(Exception, match="[Nn]ot connected"):
        await backend.store_dimensions("e1", SCORES)
