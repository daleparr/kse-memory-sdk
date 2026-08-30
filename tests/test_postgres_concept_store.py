"""
PostgreSQLBackend: migrate to schema-driven dimensions without touching data.

Written test-first per GOV-04.

Postgres is the hard case. Its legacy surface *works* and its table carries the
fashion vocabulary in the DDL itself — elegance, comfort, luxury as literal
REAL columns — so real deployments may hold real rows there. Unlike MongoDB,
whose concept store could never even be constructed, nothing here is broken.

So this migration is strictly additive:
- a new `entity_dimensions` table holds schema-driven scores as JSONB
- the legacy table and every legacy method are left exactly as they were
- copying legacy rows forward is an explicit, non-destructive opt-in

Repointing the legacy methods at the new table would have been tidier and would
have silently orphaned existing rows.

No PostgreSQL server is involved. asyncpg is exercised through a fake pool, so
these tests cover SQL construction, parameter binding and result mapping.
Behaviour against a real server is not claimed.
"""
from __future__ import annotations

import json

import pytest

from kse_memory.core.dimension_store import DimensionScores

pg = pytest.importorskip("kse_memory.backends.postgresql")

pytestmark = pytest.mark.asyncio


class FakeConn:
    def __init__(self):
        self.executed = []
        self.fetchrow_result = None
        self.fetch_result = []
        self.execute_result = "INSERT 0 1"

    async def execute(self, sql, *args):
        self.executed.append((" ".join(sql.split()), args))
        return self.execute_result

    async def fetchrow(self, sql, *args):
        self.executed.append((" ".join(sql.split()), args))
        return self.fetchrow_result

    async def fetch(self, sql, *args):
        self.executed.append((" ".join(sql.split()), args))
        return self.fetch_result


class FakeAcquire:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, *exc):
        return False


class FakePool:
    def __init__(self, conn):
        self.conn = conn

    def acquire(self):
        return FakeAcquire(self.conn)


@pytest.fixture
def conn():
    return FakeConn()


@pytest.fixture
def backend(monkeypatch, conn):
    monkeypatch.setattr(pg, "ASYNCPG_AVAILABLE", True, raising=False)
    cfg = type("Cfg", (), {"uri": "postgresql://x", "database": "kse"})()
    b = pg.PostgreSQLBackend(cfg)
    b.pool = FakePool(conn)
    b._connected = True
    return b


SCORES = DimensionScores(
    schema_name="pharma", schema_version="2.1.0",
    scores={"trial_phase_maturity": 0.9, "regulatory_burden": 0.4},
)


def sql_of(conn):
    return " || ".join(s for s, _ in conn.executed)


# ------------------------------------------------------------------- safety
async def test_new_table_ddl_never_drops_or_rewrites_the_legacy_table(backend, conn):
    """The legacy table may hold real rows. Additive DDL only."""
    await backend._create_dimension_tables(conn)
    sql = sql_of(conn).upper()

    assert "ENTITY_DIMENSIONS" in sql
    assert "DROP TABLE" not in sql
    assert "DROP COLUMN" not in sql
    assert "TRUNCATE" not in sql
    # the legacy table must not be altered at all
    assert "ALTER TABLE CONCEPTUAL_DIMENSIONS" not in sql


async def test_legacy_methods_are_untouched(backend):
    """Legacy calls must still hit the legacy table, not the new one."""
    import inspect

    src = inspect.getsource(backend.store_conceptual_dimensions)
    assert "conceptual_dimensions" in src
    assert "entity_dimensions" not in src


# ------------------------------------------------------------ generic surface
async def test_store_dimensions_upserts_with_json_payload(backend, conn):
    await backend.store_dimensions("e1", SCORES)
    sql, args = conn.executed[-1]

    assert "entity_dimensions" in sql
    assert "ON CONFLICT" in sql.upper()
    assert args[0] == "e1"
    assert args[1] == "pharma"
    assert args[2] == "2.1.0"
    assert json.loads(args[3]) == {"trial_phase_maturity": 0.9, "regulatory_burden": 0.4}


async def test_get_dimensions_maps_a_row(backend, conn):
    conn.fetchrow_result = {
        "schema_name": "pharma", "schema_version": "2.1.0",
        "scores": json.dumps({"trial_phase_maturity": 0.9, "regulatory_burden": 0.4}),
    }
    assert await backend.get_dimensions("e1") == SCORES


async def test_get_dimensions_accepts_a_decoded_jsonb_dict(backend, conn):
    """asyncpg may return JSONB already decoded, depending on codec setup."""
    conn.fetchrow_result = {
        "schema_name": "pharma", "schema_version": "2.1.0",
        "scores": {"trial_phase_maturity": 0.9, "regulatory_burden": 0.4},
    }
    assert await backend.get_dimensions("e1") == SCORES


async def test_get_dimensions_returns_none_when_absent(backend, conn):
    conn.fetchrow_result = None
    assert await backend.get_dimensions("nope") is None


async def test_delete_dimensions_reads_the_status_tag(backend, conn):
    conn.execute_result = "DELETE 1"
    assert await backend.delete_dimensions("e1") is True
    conn.execute_result = "DELETE 0"
    assert await backend.delete_dimensions("e1") is False


async def test_find_similar_is_scoped_to_schema_in_sql(backend, conn):
    conn.fetch_result = [
        {"entity_id": "near", "scores": json.dumps({"trial_phase_maturity": 0.89, "regulatory_burden": 0.41})},
        {"entity_id": "far", "scores": json.dumps({"trial_phase_maturity": 0.0, "regulatory_burden": 1.0})},
    ]
    hits = await backend.find_similar_dimensions(SCORES, threshold=0.9, limit=10)

    sql, args = conn.executed[-1]
    assert "schema_name" in sql and "schema_version" in sql
    assert args[0] == "pharma" and args[1] == "2.1.0"
    assert [h[0] for h in hits] == ["near"]


async def test_arbitrary_dimension_names_survive(backend, conn):
    exotic = DimensionScores(schema_name="legal", schema_version="1.0.0",
                             scores={"precedent_density": 0.3, "jurisdictional_reach": 0.7})
    await backend.store_dimensions("e1", exotic)
    _, args = conn.executed[-1]
    assert set(json.loads(args[3])) == {"precedent_density", "jurisdictional_reach"}


# ---------------------------------------------------------------- migration
async def test_legacy_migration_copies_without_deleting(backend, conn):
    conn.fetch_result = [
        {"product_id": "p1", "elegance": 0.7, "comfort": 0.3, "boldness": 0.0,
         "modernity": 0.0, "minimalism": 0.0, "luxury": 0.0, "functionality": 0.0,
         "versatility": 0.0, "seasonality": 0.0, "innovation": 0.0},
    ]
    count = await backend.migrate_legacy_dimensions()

    assert count == 1
    sql = sql_of(conn).upper()
    assert "DELETE" not in sql
    assert "DROP" not in sql
    inserts = [a for s, a in conn.executed if "entity_dimensions" in s]
    assert inserts and inserts[-1][0] == "p1"


async def test_migration_is_idempotent_by_upsert(backend, conn):
    conn.fetch_result = [
        {"product_id": "p1", "elegance": 0.7, "comfort": 0.0, "boldness": 0.0,
         "modernity": 0.0, "minimalism": 0.0, "luxury": 0.0, "functionality": 0.0,
         "versatility": 0.0, "seasonality": 0.0, "innovation": 0.0},
    ]
    await backend.migrate_legacy_dimensions()
    inserts = [s for s, _ in conn.executed if "entity_dimensions" in s]
    assert any("ON CONFLICT" in s.upper() for s in inserts)


async def test_operations_require_a_connection(backend):
    backend._connected = False
    with pytest.raises(Exception):
        await backend.store_dimensions("e1", SCORES)


async def test_connect_creates_the_dimension_table(backend, conn):
    """Regression: the DDL method existed but nothing invoked it, so the
    table would never have been created in a real deployment."""
    # mirror what connect() does
    await backend._create_tables(conn)
    await backend._create_dimension_tables(conn)
    assert "entity_dimensions" in sql_of(conn)

    import inspect
    connect_src = inspect.getsource(backend.connect)
    assert "_create_dimension_tables" in connect_src, "connect() must invoke the DDL"
