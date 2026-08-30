"""
US5 — the benchmark harness's arithmetic, unit-tested (TC-05).

Written test-first per GOV-04. Published numbers come from this code, so its
metrics are pinned against hand-computed examples — a benchmark harness with
untested arithmetic is a rumour generator.
"""
from __future__ import annotations

import pytest

from benchmarks.harness import format_results_table, ndcg_at_k, recall_at_k

pytestmark = pytest.mark.unit


# qrels: {doc_id: relevance}; ranking: ordered doc ids
def test_ndcg_perfect_ranking_is_one():
    qrels = {"a": 2, "b": 1}
    assert ndcg_at_k(["a", "b", "x"], qrels, k=10) == pytest.approx(1.0)


def test_ndcg_hand_computed_example():
    """ranking [x, a, b] with rels a=2, b=1:
    DCG  = 0 + 3/log2(3) + 1/log2(4)   (gain 2^rel - 1)
    IDCG = 3/log2(2) + 1/log2(3)
    """
    import math

    qrels = {"a": 2, "b": 1}
    dcg = 3 / math.log2(3) + 1 / math.log2(4)
    idcg = 3 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(["x", "a", "b"], qrels, k=10) == pytest.approx(dcg / idcg)


def test_ndcg_k_truncates():
    qrels = {"a": 1}
    assert ndcg_at_k(["x", "y", "a"], qrels, k=2) == 0.0


def test_ndcg_no_relevant_docs_is_zero():
    assert ndcg_at_k(["x", "y"], {}, k=10) == 0.0


def test_recall_counts_relevant_in_top_k():
    qrels = {"a": 1, "b": 2, "c": 1}
    assert recall_at_k(["a", "x", "b"], qrels, k=3) == pytest.approx(2 / 3)
    assert recall_at_k(["a", "x", "b"], qrels, k=1) == pytest.approx(1 / 3)


def test_recall_ignores_zero_relevance_qrels():
    """BEIR qrels can carry explicit 0 judgements; they are not relevant."""
    qrels = {"a": 1, "z": 0}
    assert recall_at_k(["a", "z"], qrels, k=2) == pytest.approx(1.0)


# ------------------------------------------------------------- the table
def test_results_table_reports_losses_plainly():
    """TC-05's honesty clause: a losing system is printed as a loss, with a
    delta — never omitted, never softened."""
    rows = [
        {"dataset": "scifact", "system": "dense", "ndcg@10": 0.600, "recall@100": 0.900},
        {"dataset": "scifact", "system": "hybrid", "ndcg@10": 0.550, "recall@100": 0.910},
    ]
    table = format_results_table(rows, baseline="dense")
    assert "scifact" in table and "hybrid" in table
    assert "-0.050" in table          # the nDCG loss, signed, visible
    assert "+0.010" in table          # and the recall win, same treatment


def test_results_table_is_deterministic():
    rows = [
        {"dataset": "d", "system": "dense", "ndcg@10": 0.5, "recall@100": 0.5},
        {"dataset": "d", "system": "hybrid", "ndcg@10": 0.6, "recall@100": 0.6},
    ]
    assert format_results_table(rows, baseline="dense") == \
           format_results_table(rows, baseline="dense")
