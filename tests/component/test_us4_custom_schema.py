"""
US4 — "As a developer in any domain, I define my own conceptual dimensions."

Written test-first per GOV-04. TC-04's clauses, exercised the way a developer
would: a YAML schema *file* in their own domain vocabulary, end to end
through ingest, scoring, querying and receipts — plus the CLI flag that makes
the story real outside a Python session.
"""
from __future__ import annotations

import pytest

from tests.conftest import StubEmbedder
from kse_memory.core.dimension_store import DimensionScores
from kse_memory.core.schema import load_schema
from kse_memory.quickstart.v3 import run_quickstart

pytestmark = [pytest.mark.asyncio, pytest.mark.component]

FINANCE_YAML = """\
name: finance-instruments
version: 1.0.0
dimensions:
  - name: liquidity
    description: How quickly the instrument converts to cash
    anchors:
      - traded continuously with deep order books
      - convertible to cash within a day
  - name: volatility
    description: Price variability
    anchors:
      - large daily price swings
      - stable value over long horizons
  - name: regulatory_burden
    description: Compliance overhead
    anchors:
      - extensive filing and disclosure requirements
"""

RECORDS = [
    {"title": "blue-chip equity", "description": "traded continuously with deep order books, moderate swings"},
    {"title": "private placement", "description": "illiquid holding with extensive filing and disclosure requirements"},
    {"title": "money market fund", "description": "stable value, convertible to cash within a day"},
]


@pytest.fixture
def schema_file(tmp_path):
    path = tmp_path / "finance.yaml"
    path.write_text(FINANCE_YAML, encoding="utf-8")
    return path


async def test_a_yaml_file_drives_the_whole_story(schema_file, stub_embedder):
    """TC-04 end to end: file in, scored + queryable + receipted out."""
    schema = load_schema(schema_file)
    assert schema.names() == ("liquidity", "volatility", "regulatory_burden")

    result = await run_quickstart(
        embedder=stub_embedder,
        schema={"name": schema.name, "version": schema.version,
                "dimensions": [{"name": d.name, "description": d.description,
                                "anchors": list(d.anchors)} for d in schema]},
        records=RECORDS,
        queries=["something easy to sell fast"],
    )

    # scored: every ingested item carries the developer's dimensions
    (hits,) = result.searches.values()
    for hit in hits:
        assert set(hit.scores) == {"liquidity", "volatility", "regulatory_burden"}

    # parsed: the query targets are keyed by the developer's vocabulary
    (parsed,) = result.parses.values()
    assert set(parsed.targets) == {"liquidity", "volatility", "regulatory_burden"}

    # queryable: the concept store answers similarity in the same schema
    store = result.pipeline.concept_store
    probe = DimensionScores(schema_name="finance-instruments", schema_version="1.0.0",
                            scores={"liquidity": 0.9, "volatility": 0.5, "regulatory_burden": 0.2})
    assert await store.find_similar_dimensions(probe, threshold=0.0, limit=3)


async def test_run_quickstart_accepts_a_schema_path_directly(schema_file, stub_embedder):
    """A developer should hand over the file, not transcribe it to a dict."""
    result = await run_quickstart(
        embedder=stub_embedder, schema=schema_file,
        records=RECORDS, queries=["stable and safe"],
    )
    (hits,) = result.searches.values()
    assert set(hits[0].scores) == {"liquidity", "volatility", "regulatory_burden"}


def test_cli_quickstart_exposes_a_schema_flag():
    """The story must be reachable from the command line (D-14 layer 1)."""
    from click.testing import CliRunner

    from kse_memory.cli import quickstart

    result = CliRunner().invoke(quickstart, ["--help"])
    assert "--schema" in result.output


def test_cli_explain_exposes_a_schema_flag():
    from click.testing import CliRunner

    from kse_memory.cli import explain

    result = CliRunner().invoke(explain, ["--help"])
    assert "--schema" in result.output
