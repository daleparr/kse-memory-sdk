"""
TC — FR-01: Ingest — normalise raw records to universal Entity with SHA-256
content hash for dedupe and deterministic replay. (BD3 FR-01; BD4 TC-02 slice;
US2 acceptance path step 1.)

Written FIRST per GOV-04. Targets kse_memory.core.ingest (new module).
All tests run under the AR-01 no-network fixture.
"""
import socket

import pytest

pytestmark = pytest.mark.unit

from kse_memory.core.ingest import content_hash, normalise_record
from kse_memory.core.models import Entity


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """AR-01: the ingest path must never touch the network."""

    def _blocked(*a, **k):  # pragma: no cover
        raise AssertionError("AR-01 violated: ingest attempted a network call")

    monkeypatch.setattr(socket.socket, "connect", _blocked)


RAW = {
    "title": "Trail Runner X",
    "description": "Cushioned minimalist trail shoe",
    "entity_type": "product",
    "category": "footwear",
    "tags": ["running", "trail"],
    "metadata": {"domain": {"retail": {"price": 129.0, "currency": "GBP"}}},
}


# ---------------------------------------------------------------- normalise
def test_normalise_returns_entity_with_stable_given_id():
    e = normalise_record({**RAW, "id": "sku-123"})
    assert isinstance(e, Entity)
    assert e.id == "sku-123"
    assert e.title == "Trail Runner X"


def test_normalise_without_id_derives_deterministic_id_from_content():
    """No id supplied → id derived from content hash, so replay is stable
    and re-ingesting the same record cannot create a duplicate identity."""
    e1 = normalise_record(dict(RAW))
    e2 = normalise_record(dict(RAW))
    assert e1.id == e2.id
    assert e1.id.startswith("kse-"), "derived ids are namespaced to avoid uuid collision"


def test_normalise_rejects_records_missing_required_fields():
    with pytest.raises(ValueError, match="title"):
        normalise_record({"description": "no title here"})


def test_normalise_ignores_unknown_keys_but_preserves_them_in_metadata():
    e = normalise_record({**RAW, "warehouse_zone": "B4"})
    assert e.metadata.get("extra", {}).get("warehouse_zone") == "B4"


# ---------------------------------------------------------------- content_hash
def test_content_hash_is_sha256_hex():
    h = content_hash(normalise_record(dict(RAW)))
    assert len(h) == 64 and int(h, 16) >= 0


def test_content_hash_is_key_order_and_timestamp_invariant():
    """Same content → same hash: dedupe must not depend on dict ordering
    or volatile fields (id, created_at/updated_at, embeddings)."""
    reordered = {
        "metadata": {"domain": {"retail": {"currency": "GBP", "price": 129.0}}},
        "tags": ["running", "trail"],
        "category": "footwear",
        "entity_type": "product",
        "description": "Cushioned minimalist trail shoe",
        "title": "Trail Runner X",
    }
    e1, e2 = normalise_record(dict(RAW)), normalise_record(reordered)
    e2.updated_at = e2.created_at.replace(year=2001)  # volatile field differs
    assert content_hash(e1) == content_hash(e2)


def test_content_hash_changes_when_content_changes():
    e1 = normalise_record(dict(RAW))
    e2 = normalise_record({**RAW, "description": "Cushioned minimalist trail shoe v2"})
    assert content_hash(e1) != content_hash(e2)


def test_content_hash_survives_roundtrip_replay():
    """Replay determinism: hash(entity) == hash(from_dict(to_dict(entity)))."""
    e = normalise_record(dict(RAW))
    replayed = Entity.from_dict(e.to_dict())
    assert content_hash(e) == content_hash(replayed)


# ------------------------------------------------- review findings (session 3)
def test_content_hash_is_tag_order_invariant():
    """Tags are set-like for identity: reordering must not trigger re-projection."""
    e1 = normalise_record({**RAW, "tags": ["running", "trail"]})
    e2 = normalise_record({**RAW, "tags": ["trail", "running"]})
    assert content_hash(e1) == content_hash(e2)


def test_content_hash_rejects_non_serialisable_content():
    """Silent str() fallback embeds memory addresses → non-deterministic hashes.
    FR-01 must fail loudly instead of corrupting replay identity."""

    class Weird:
        pass

    # With no explicit id, identity derives from the hash — so bad content
    # fails at the ingest door, which is exactly where we want it.
    with pytest.raises(ValueError, match="not deterministically serialisable"):
        normalise_record({**RAW, "gadget": Weird()})

    # With an explicit id, ingest defers hashing; the hash call itself must fail.
    e = normalise_record({**RAW, "id": "sku-9", "gadget": Weird()})
    with pytest.raises(ValueError, match="not deterministically serialisable"):
        content_hash(e)
