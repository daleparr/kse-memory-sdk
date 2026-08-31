"""
FR-07 — Fallback: confidence-gated hybrid answers with an explicit flag.

Written test-first per GOV-04.

Confidence is corroboration: the mean, over the fused top results, of the
fraction of configured channels that returned each result. It is bounded,
deterministic, and explainable in one sentence — "the top results are each
backed by 2.3 of 3 channels on average". When it falls below the threshold,
the answer falls back to the dense ranking and SAYS SO; an unflagged
fallback would be the same lie the dense-only quickstart label existed to
avoid.
"""
from __future__ import annotations

import pytest

from kse_memory.core.answer import HybridAnswer, answer, assess_confidence
from kse_memory.core.fusion import fuse_rrf
from kse_memory.core.retrieval import RetrievalResult

pytestmark = pytest.mark.unit


def rr(vector=(), conceptual=(), graph=(), errors=None):
    return RetrievalResult(vector=tuple(vector), conceptual=tuple(conceptual),
                           graph=tuple(graph), errors=errors or {})


AGREEING = rr(
    vector=(("a", 0.9), ("b", 0.5)),
    conceptual=(("a", 0.8), ("b", 0.6)),
    graph=(("a", 2.0), ("b", 1.0)),
)

DISAGREEING = rr(
    vector=(("a", 0.9), ("b", 0.5)),
    conceptual=(("c", 0.8), ("d", 0.6)),
    graph=(("e", 2.0), ("f", 1.0)),
)


# ------------------------------------------------------------- confidence
def test_full_agreement_is_full_confidence():
    fused = fuse_rrf({"vector": AGREEING.vector, "conceptual": AGREEING.conceptual,
                      "graph": AGREEING.graph})
    assert assess_confidence(AGREEING, fused, top_k=2) == pytest.approx(1.0)


def test_total_disagreement_is_one_third_confidence():
    """Every fused entity is backed by exactly 1 of 3 channels."""
    fused = fuse_rrf({"vector": DISAGREEING.vector, "conceptual": DISAGREEING.conceptual,
                      "graph": DISAGREEING.graph})
    assert assess_confidence(DISAGREEING, fused, top_k=6) == pytest.approx(1 / 3)


def test_empty_channel_lowers_confidence():
    partial = rr(vector=(("a", 0.9),), conceptual=(("a", 0.8),), graph=())
    fused = fuse_rrf({"vector": partial.vector, "conceptual": partial.conceptual,
                      "graph": partial.graph})
    assert assess_confidence(partial, fused, top_k=1) == pytest.approx(2 / 3)


def test_confidence_of_empty_fusion_is_zero():
    empty = rr()
    assert assess_confidence(empty, (), top_k=5) == 0.0


# ---------------------------------------------------------------- answers
def test_confident_answer_stays_hybrid():
    result = answer(AGREEING, confidence_threshold=0.5)
    assert isinstance(result, HybridAnswer)
    assert result.hybrid is True
    assert result.dense_only is False
    assert result.fallback_reason is None
    assert result.items[0].entity_id == "a"
    assert result.confidence == pytest.approx(1.0)


def test_low_confidence_falls_back_to_dense_with_explicit_flag():
    result = answer(DISAGREEING, confidence_threshold=0.5)
    assert result.dense_only is True
    assert result.hybrid is False
    assert [i.entity_id for i in result.items] == ["a", "b"]  # the vector ranking
    assert "confidence" in result.fallback_reason
    assert result.confidence == pytest.approx(1 / 3)


def test_fallback_items_still_carry_receipts():
    """Dense-only is a ranking choice, not an amnesty from FR-06."""
    result = answer(DISAGREEING, confidence_threshold=0.5)
    top = result.items[0]
    assert top.ranks["vector"] == 1
    assert top.ranks["conceptual"] is None


def test_no_dense_channel_means_no_fallback_but_still_flagged():
    """You cannot fall back to a channel that returned nothing."""
    no_vector = rr(conceptual=(("c", 0.8),), graph=(("e", 2.0),))
    result = answer(no_vector, confidence_threshold=0.9)
    assert result.dense_only is False
    assert result.hybrid is False           # low confidence: do not claim it
    assert result.fallback_reason is not None
    assert "vector" in result.fallback_reason
    assert result.items                     # fused ranking retained


def test_degraded_channels_are_carried():
    degraded = rr(vector=(("a", 0.9),), errors={"graph": "down"})
    result = answer(degraded)
    assert result.degraded == {"graph": "down"}


def test_single_channel_answer_is_not_called_hybrid():
    solo = rr(vector=(("a", 0.9), ("b", 0.5)))
    result = answer(solo, confidence_threshold=0.0)
    assert result.hybrid is False
    assert result.dense_only is True
    assert result.fallback_reason is None   # nothing fell back; it just IS dense


def test_threshold_is_validated():
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="threshold"):
            answer(AGREEING, confidence_threshold=bad)


def test_answers_are_deterministic():
    assert answer(DISAGREEING) == answer(DISAGREEING)
