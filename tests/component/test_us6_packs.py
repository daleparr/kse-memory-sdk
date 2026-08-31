"""
US6 — domain packs: worked examples, not blank schemas (TC-06 structure).

Written test-first per GOV-04. The structural clauses live here (component
lane, stub embedder); the "pure vector search handles this worse" clause
needs real semantics and lives in the integration lane.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from kse_memory.core.schema import load_schema
from examples.packs import PACKS, load_pack

pytestmark = pytest.mark.component

EXPECTED = {"retail", "finance", "documents"}


def test_the_three_packs_exist():
    assert set(PACKS) == EXPECTED


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_pack_ships_a_valid_schema(name):
    pack = load_pack(name)
    schema = load_schema(pack.schema_path)   # the actual YAML file, validated
    assert len(schema) >= 2
    assert schema.name.startswith(name)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_pack_ships_a_corpus_with_the_showcase_target(name):
    pack = load_pack(name)
    ids = {record["id"] for record in pack.records}
    assert len(pack.records) >= 6
    assert pack.showcase.target_id in ids


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_pack_ships_a_runnable_notebook(name):
    pack = load_pack(name)
    assert pack.notebook_path.exists()
    notebook = json.loads(pack.notebook_path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] >= 4
    source = "".join("".join(c.get("source", [])) for c in notebook["cells"])
    assert "load_pack" in source and f'"{name}"' in source
    assert "run_showcase" in source


@pytest.mark.parametrize("name", sorted(EXPECTED))
async def test_showcase_runs_offline_with_a_stub(name, no_network, stub_embedder):
    """The machinery runs offline end to end; semantic outcomes are the
    integration lane's to assert."""
    from examples.packs import run_showcase

    outcome = await run_showcase(load_pack(name), stub_embedder)
    assert outcome.dense_rank >= 1
    assert outcome.hybrid_rank >= 1
    assert outcome.query
