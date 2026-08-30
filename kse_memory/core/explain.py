"""
FR-06 — Explain: per-channel ranks, scores and per-dimension breakdown.

Design (BD3; D-12 "judgement with receipts", D-14 inspection):
- Everything an explanation needs already exists by the time fusion returns:
  FusedItem carries channel ranks and raw scores, ParsedQuery carries targets
  and replay identity, RetrievalResult carries degraded-channel errors, and
  the concept store holds each entity's dimension scores. FR-06 assembles
  them into one attachable object per result, losing nothing on the way.
- The dimension breakdown pairs the query's target with the entity's score on
  each axis and derives ``alignment = 1 - |target - score|``: bounded on
  [0, 1], symmetric, and readable as "how close this item sits to what the
  query asked for on this dimension". No cleverness — an explanation that
  needs explaining has failed.
- ``explain_results`` is pure. The caller fetches dimension scores (it is the
  one holding a store); explanation itself does no I/O, embeds nothing and
  raises on nothing — a result that cannot be fully explained is explained as
  far as the evidence goes.

Guardrails honoured: AR-01 (pure computation), AR-05 (typed surface).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence, Tuple

from .fusion import FusedItem
from .query import ParsedQuery
from .retrieval import RetrievalResult

__all__ = ["DimensionRow", "Explanation", "explain_results"]


@dataclass(frozen=True)
class DimensionRow:
    """One dimension's contribution to one result."""

    name: str
    target: float      # what the query asked for (FR-03)
    score: float       # what the entity carries (FR-02)
    alignment: float   # 1 - |target - score|


@dataclass(frozen=True)
class Explanation:
    """The full receipt for one fused result.

    ``ranks``/``channel_scores`` hold an entry per channel with ``None`` for
    a channel that did not return the entity — absence is information.
    ``degraded`` names channels that failed during retrieval (FR-07 will add
    the fused-confidence flag on top of this).
    """

    entity_id: str
    fused: float
    query: str
    schema_name: str
    schema_version: str
    model_id: str
    ranks: Mapping[str, Optional[int]] = field(default_factory=dict)
    channel_scores: Mapping[str, Optional[float]] = field(default_factory=dict)
    dimensions: Tuple[DimensionRow, ...] = ()
    degraded: Mapping[str, str] = field(default_factory=dict)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Explanation):
            return NotImplemented
        return (
            self.entity_id == other.entity_id
            and self.fused == other.fused
            and self.query == other.query
            and self.schema_name == other.schema_name
            and self.schema_version == other.schema_version
            and self.model_id == other.model_id
            and dict(self.ranks) == dict(other.ranks)
            and dict(self.channel_scores) == dict(other.channel_scores)
            and self.dimensions == other.dimensions
            and dict(self.degraded) == dict(other.degraded)
        )


def explain_results(
    parsed: ParsedQuery,
    retrieval: RetrievalResult,
    fused: Sequence[FusedItem],
    dimension_scores: Mapping[str, Mapping[str, float]],
) -> Tuple[Explanation, ...]:
    """Attach a full explanation to every fused result, in fused order.

    Args:
        dimension_scores: entity id -> that entity's dimension scores, as the
            caller's concept store returned them. An entity absent here gets
            an empty breakdown but keeps every other part of its explanation.
    """
    explanations = []
    for item in fused:
        entity_scores = dimension_scores.get(item.entity_id)
        rows: Tuple[DimensionRow, ...] = ()
        if entity_scores is not None:
            rows = tuple(
                DimensionRow(
                    name=name,
                    target=target,
                    score=float(entity_scores.get(name, 0.0)),
                    alignment=round(
                        1.0 - abs(target - float(entity_scores.get(name, 0.0))), 6
                    ),
                )
                for name, target in parsed.targets.items()
            )
        explanations.append(
            Explanation(
                entity_id=item.entity_id,
                fused=item.fused,
                query=parsed.text,
                schema_name=parsed.schema_name,
                schema_version=parsed.schema_version,
                model_id=parsed.model_id,
                ranks=dict(item.ranks),
                channel_scores=dict(item.scores),
                dimensions=rows,
                degraded=dict(retrieval.errors),
            )
        )
    return tuple(explanations)
