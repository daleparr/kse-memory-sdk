"""
The pinned ESCI slice (D-103 closure; TC-05's remaining dataset clause).

Written test-first per GOV-04. The slice is a COMMITTED artefact in BEIR
format, so the benchmark harness loads it through the same code path as
scifact/nfcorpus — no second loader, no giant download for users. These
tests pin the slice's shape, provenance manifest and label mapping.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
SLICE = ROOT / "benchmarks" / "esci_slice" / "esci-slice"


def test_the_slice_is_committed_in_beir_format():
    assert (SLICE / "corpus.jsonl").is_file()
    assert (SLICE / "queries.jsonl").is_file()
    assert (SLICE / "qrels" / "test.tsv").is_file()


def test_the_slice_loads_through_the_standard_loader():
    from benchmarks.harness import load_beir_dir

    data = load_beir_dir(SLICE)
    assert len(data["queries"]) == 200          # D-104: first 200 by query_id
    assert len(data["corpus"]) >= 1000
    assert set(data["queries"]) == set(data["qrels"])


def test_qrels_use_the_documented_graded_mapping():
    """D-104: E→3, S→2, C→1, I→0. Nothing outside that range may appear."""
    from benchmarks.harness import load_beir_dir

    data = load_beir_dir(SLICE)
    values = {rel for qrels in data["qrels"].values() for rel in qrels.values()}
    assert values <= {0, 1, 2, 3}
    assert 3 in values                          # exact matches exist
    assert any(                                  # every query has ≥1 relevant doc
        any(rel > 0 for rel in qrels.values()) for qrels in data["qrels"].values()
    )


def test_every_judged_document_is_in_the_corpus():
    from benchmarks.harness import load_beir_dir

    data = load_beir_dir(SLICE)
    corpus_ids = set(data["corpus"])
    for qrels in data["qrels"].values():
        assert set(qrels) <= corpus_ids


def test_the_provenance_manifest_pins_the_sources():
    manifest = json.loads((SLICE / "MANIFEST.json").read_text(encoding="utf-8"))
    for key in ("source_repo", "source_sha256", "filter", "label_map", "decision"):
        assert key in manifest, key
    assert manifest["label_map"] == {"E": 3, "S": 2, "C": 1, "I": 0}
    assert manifest["decision"] == "D-104"
    assert len(manifest["source_sha256"]) == 2   # both parquets pinned
