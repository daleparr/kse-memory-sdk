"""
FR-07 — Fallback: confidence-gated hybrid answers with an explicit flag.

Design (BD3; D-07, D-12):
- Confidence is *corroboration*: the mean, over the fused top results, of the
  fraction of configured channels that returned each result. One sentence
  explains it — "the top results are each backed by 2.3 of 3 channels on
  average" — and D-12's receipts culture demands nothing less legible. Empty
  and failed channels count in the denominator: a channel that produced
  nothing corroborated nothing.
- Below the threshold, the answer falls back to the dense ranking and SAYS
  SO. The explicit flag is the point of the FR: an unflagged fallback would
  be the same lie the quickstart's dense-only label existed to avoid, in the
  opposite direction.
- Fallback is a ranking choice, not an amnesty from FR-06: dense-only items
  still carry their per-channel receipts.
- When the dense channel itself has nothing to offer, there is nothing to
  fall back TO: the fused ranking is retained, ``hybrid`` is still refused,
  and the reason states why. A single-configured-channel answer is reported
  as dense-only without a fallback reason — it did not fall back, it simply
  is what it is.

Guardrails honoured: AR-01 (pure computation), AR-05 (typed surface).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence, Tuple

from .fusion import DEFAULT_RRF_K, FusedItem, fuse_rrf
from .retrieval import RetrievalResult

__all__ = ["DEFAULT_CONFIDENCE_THRESHOLD", "HybridAnswer", "answer", "assess_confidence"]

DEFAULT_CONFIDENCE_THRESHOLD = 0.5


@dataclass(frozen=True)
class HybridAnswer:
    """A ranked answer that states what it is.

    Exactly one of these claims holds and the fields say which:
    - ``hybrid``: the fused ranking, confidently corroborated.
    - ``dense_only``: the vector ranking — either by fallback (see
      ``fallback_reason``) or because vector was the only channel.
    - neither: a low-confidence fused ranking kept for want of a dense
      channel to fall back to; ``fallback_reason`` explains.
    """

    items: Tuple[FusedItem, ...]
    hybrid: bool
    dense_only: bool
    confidence: float
    fallback_reason: Optional[str] = None
    degraded: Mapping[str, str] = field(default_factory=dict)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HybridAnswer):
            return NotImplemented
        return (
            self.items == other.items
            and self.hybrid == other.hybrid
            and self.dense_only == other.dense_only
            and self.confidence == other.confidence
            and self.fallback_reason == other.fallback_reason
            and dict(self.degraded) == dict(other.degraded)
        )


def _channels_of(retrieval: RetrievalResult) -> Mapping[str, Tuple[Tuple[str, float], ...]]:
    return {
        "vector": retrieval.vector,
        "conceptual": retrieval.conceptual,
        "graph": retrieval.graph,
    }


def assess_confidence(
    retrieval: RetrievalResult,
    fused: Sequence[FusedItem],
    top_k: int = 10,
) -> float:
    """Mean corroboration of the fused top results, in [0, 1].

    For each of the top ``top_k`` fused entities: the fraction of channels
    (all three — empty and failed ones corroborate nothing) that returned it.
    Averaged. Empty fusion is zero confidence by definition.
    """
    considered = list(fused)[:top_k]
    if not considered:
        return 0.0
    total_channels = len(_channels_of(retrieval))
    support = [
        sum(1 for rank in item.ranks.values() if rank is not None) / total_channels
        for item in considered
    ]
    return sum(support) / len(support)


def answer(
    retrieval: RetrievalResult,
    *,
    k: int = DEFAULT_RRF_K,
    top_k: int = 10,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    weights: Optional[Mapping[str, float]] = None,
) -> HybridAnswer:
    """Fuse the channels and gate the result on corroboration (FR-07).

    Raises:
        ValueError: if the threshold lies outside [0, 1].
    """
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError(
            f"confidence threshold must lie in [0, 1], got {confidence_threshold}"
        )

    channels = _channels_of(retrieval)
    fused = fuse_rrf(channels, k=k, weights=weights, top_k=top_k)
    confidence = assess_confidence(retrieval, fused, top_k=top_k)

    populated = [name for name, rows in channels.items() if rows]
    only_vector = populated == ["vector"]

    if only_vector:
        # Nothing fused against: the answer is dense, plainly, no fallback story.
        return HybridAnswer(
            items=fused, hybrid=False, dense_only=True,
            confidence=confidence, fallback_reason=None,
            degraded=dict(retrieval.errors),
        )

    if confidence >= confidence_threshold:
        return HybridAnswer(
            items=fused, hybrid=len(populated) >= 2, dense_only=False,
            confidence=confidence, fallback_reason=None,
            degraded=dict(retrieval.errors),
        )

    if retrieval.vector:
        # Re-rank by the dense channel alone; receipts stay attached (FR-06).
        by_id = {item.entity_id: item for item in fuse_rrf(channels, k=k, weights=weights)}
        dense_items = tuple(by_id[entity_id] for entity_id, _ in retrieval.vector[:top_k])
        return HybridAnswer(
            items=dense_items, hybrid=False, dense_only=True,
            confidence=confidence,
            fallback_reason=(
                f"fused confidence {confidence:.2f} below threshold "
                f"{confidence_threshold:.2f}; fell back to the dense ranking"
            ),
            degraded=dict(retrieval.errors),
        )

    return HybridAnswer(
        items=fused, hybrid=False, dense_only=False,
        confidence=confidence,
        fallback_reason=(
            f"fused confidence {confidence:.2f} below threshold "
            f"{confidence_threshold:.2f}, and the vector channel returned "
            "nothing to fall back to; fused ranking retained"
        ),
        degraded=dict(retrieval.errors),
    )
