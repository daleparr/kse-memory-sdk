"""
FR-05 — Fuse channel rankings: RRF default, min-max weighted optional.

Design (BD3; decision D-07):
- FR-04's channel scores are not comparable across channels — cosine, cosine
  over a different space, and a coverage count. Reciprocal Rank Fusion uses
  only the *ranks*, so it is scale-free by construction; the property suite
  proves invariance under positive rescaling rather than assuming it.
- ``fused(e) = Σ_c w_c / (k + rank_c(e))`` with 1-based ranks and the
  conventional k=60. Higher k flattens the curve (rank 1 vs 2 matters less);
  it must be positive or the formula degenerates.
- Every fused item carries its per-channel rank and raw score — FR-06's
  explanations are receipts, and receipts are collected at fusion time or
  never.
- Ties break by entity id: replay demands that equal evidence yields one
  ordering, not an arbitrary one (BD4 "fusion seedable").
- ``fuse_weighted`` is the optional path: min-max normalisation inside each
  channel, then a weighted sum. It exists for callers who genuinely know
  their channels' score distributions; RRF stays the default precisely
  because most callers do not.

Guardrails honoured: AR-01 (pure computation), AR-05 (typed surface).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Tuple

__all__ = ["DEFAULT_RRF_K", "FusedItem", "fuse_rrf", "fuse_weighted"]

DEFAULT_RRF_K = 60


@dataclass(frozen=True)
class FusedItem:
    """One entity's fused standing, with the evidence that produced it.

    ``ranks``/``scores`` hold an entry per input channel; ``None`` marks a
    channel that did not return the entity at all — absence is information
    and FR-06 will want to say so.
    """

    entity_id: str
    fused: float
    ranks: Mapping[str, Optional[int]] = field(default_factory=dict)
    scores: Mapping[str, Optional[float]] = field(default_factory=dict)


def _evidence(
    channels: Mapping[str, Sequence[Tuple[str, float]]],
) -> Dict[str, Dict[str, Tuple[int, float]]]:
    """Per entity: {channel: (rank, raw score)} for channels that returned it."""
    seen: Dict[str, Dict[str, Tuple[int, float]]] = {}
    for name, rows in channels.items():
        for position, (entity_id, score) in enumerate(rows, start=1):
            seen.setdefault(entity_id, {})[name] = (position, float(score))
    return seen


def _build(
    channels: Mapping[str, Sequence[Tuple[str, float]]],
    fused_scores: Mapping[str, float],
    evidence: Mapping[str, Mapping[str, Tuple[int, float]]],
    top_k: Optional[int],
) -> Tuple[FusedItem, ...]:
    ordered = sorted(fused_scores.items(), key=lambda item: (-item[1], item[0]))
    items = tuple(
        FusedItem(
            entity_id=entity_id,
            fused=score,
            ranks={name: (evidence[entity_id].get(name) or (None,))[0] for name in channels},
            scores={
                name: evidence[entity_id][name][1] if name in evidence[entity_id] else None
                for name in channels
            },
        )
        for entity_id, score in ordered
    )
    return items[:top_k] if top_k is not None else items


def fuse_rrf(
    channels: Mapping[str, Sequence[Tuple[str, float]]],
    k: int = DEFAULT_RRF_K,
    weights: Optional[Mapping[str, float]] = None,
    top_k: Optional[int] = None,
) -> Tuple[FusedItem, ...]:
    """Reciprocal Rank Fusion over per-channel rankings (D-07 default).

    Args:
        channels: channel name -> rank-ordered ``(entity_id, score)`` rows,
            exactly as :class:`RetrievalResult` provides them. Scores are
            carried through as receipts; only the order is used.
        k: the RRF constant; must be positive.
        weights: optional per-channel multipliers (missing names weigh 1.0).
        top_k: truncate the fused ranking after fusion.

    Raises:
        ValueError: if ``k`` is not positive.
    """
    if k <= 0:
        raise ValueError(f"RRF k must be positive, got {k}")

    evidence = _evidence(channels)
    fused: Dict[str, float] = {
        entity_id: sum(
            (weights or {}).get(name, 1.0) / (k + rank)
            for name, (rank, _) in per_channel.items()
        )
        for entity_id, per_channel in evidence.items()
    }
    return _build(channels, fused, evidence, top_k)


def fuse_weighted(
    channels: Mapping[str, Sequence[Tuple[str, float]]],
    weights: Optional[Mapping[str, float]] = None,
    top_k: Optional[int] = None,
) -> Tuple[FusedItem, ...]:
    """Min-max normalised weighted fusion — the optional score-aware path.

    Each channel's scores are mapped onto [0, 1] by min-max within that
    channel (a single-score channel maps to 1.0: it has no range, and zero
    would erase the channel's only vote). The fused score is the weighted sum.
    """
    evidence = _evidence(channels)

    spans: Dict[str, Tuple[float, float]] = {}
    for name, rows in channels.items():
        if rows:
            values = [score for _, score in rows]
            spans[name] = (min(values), max(values))

    fused: Dict[str, float] = {}
    for entity_id, per_channel in evidence.items():
        total = 0.0
        for name, (_, score) in per_channel.items():
            low, high = spans[name]
            unit = 1.0 if high == low else (score - low) / (high - low)
            total += (weights or {}).get(name, 1.0) * unit
        fused[entity_id] = total
    return _build(channels, fused, evidence, top_k)
