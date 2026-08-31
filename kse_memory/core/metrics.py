"""
Ranking metrics — the single, unit-tested implementation (US5/US10).

Moved here from benchmarks/harness.py so that learned fusion (a packaged
capability) can evaluate itself without the package depending on the
unpackaged benchmarks/ tree. benchmarks imports FROM here; never the
reverse. The arithmetic is pinned to hand-computed values in
tests/unit/test_bench_harness.py.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

__all__ = ["ndcg_at_k", "recall_at_k"]


def ndcg_at_k(ranking: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    """nDCG@k with exponential gain (2^rel - 1), the BEIR convention."""
    ideal = sorted((r for r in qrels.values() if r > 0), reverse=True)[:k]
    if not ideal:
        return 0.0
    idcg = sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal))
    dcg = sum(
        (2 ** qrels.get(doc_id, 0) - 1) / math.log2(i + 2)
        for i, doc_id in enumerate(ranking[:k])
    )
    return dcg / idcg


def recall_at_k(ranking: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    relevant = {doc_id for doc_id, rel in qrels.items() if rel > 0}
    if not relevant:
        return 0.0
    return len(relevant & set(ranking[:k])) / len(relevant)
