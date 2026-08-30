"""
The v3-native search service: FR-03..FR-07 in one app-facing object.

Design:
- Owns the schema, the embedder and the anchor centroids (computed once), so
  the parse -> retrieve -> answer -> explain spine stops being assembly
  instructions and becomes a method call.
- Stores are optional, exactly as in retrieval: a missing store is a missing
  channel, and the FR-07 verdict tells the caller what its answer is made of.
- This object is what replaces the v2 SearchService's hybrid path for
  schema-driven applications. The v2 service survives for legacy stored data;
  its fusion now defers to FR-05 as well.

Guardrails honoured: AR-01 (no network), AR-04 (no GPU), AR-05 (typed surface).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

from ..core.answer import DEFAULT_CONFIDENCE_THRESHOLD, HybridAnswer, answer
from ..core.explain import Explanation, explain_results
from ..core.projection import anchor_centroids
from ..core.query import ParsedQuery, parse_query
from ..core.retrieval import retrieve
from ..core.schema import DimensionSchema

__all__ = ["HybridSearchResponse", "HybridSearchService"]


@dataclass(frozen=True)
class HybridSearchResponse:
    """Everything one search produced: the verdict, the receipts, the parse."""

    answer: HybridAnswer
    explanations: Tuple[Explanation, ...]
    parsed: ParsedQuery
    titles: Mapping[str, str] = field(default_factory=dict)


class HybridSearchService:
    """Schema-driven hybrid search over the FR-03..FR-07 spine."""

    def __init__(
        self,
        schema: DimensionSchema,
        embedder,
        *,
        vector_store=None,
        concept_store=None,
        graph_store=None,
        centroids: Optional[Mapping[str, List[float]]] = None,
    ) -> None:
        self.schema = schema
        self.embedder = embedder
        self.vector_store = vector_store
        self.concept_store = concept_store
        self.graph_store = graph_store
        self._centroids = dict(centroids) if centroids is not None else None

    @property
    def centroids(self) -> Mapping[str, List[float]]:
        """Anchor centroids, embedded once on first use (or adopted from a
        pipeline's cache via the constructor)."""
        if self._centroids is None:
            self._centroids = anchor_centroids(self.schema, self.embedder)
        return self._centroids

    async def search(
        self,
        query: str,
        top_k: int = 10,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        weights: Optional[Mapping[str, float]] = None,
    ) -> HybridSearchResponse:
        """Parse, retrieve concurrently, fuse under the confidence gate,
        and attach the full receipt to every result."""
        parsed = parse_query(query, self.schema, self.embedder, centroids=self.centroids)

        channels = await retrieve(
            parsed,
            vector_store=self.vector_store,
            concept_store=self.concept_store,
            graph_store=self.graph_store,
            top_k=top_k,
        )
        verdict = answer(
            channels, top_k=top_k,
            confidence_threshold=confidence_threshold, weights=weights,
        )

        dimension_scores: Dict[str, Mapping[str, float]] = {}
        titles: Dict[str, str] = {}
        for item in verdict.items:
            if self.concept_store is not None:
                stored = await self.concept_store.get_dimensions(item.entity_id)
                if stored:
                    dimension_scores[item.entity_id] = dict(stored.scores)
            rows = getattr(self.vector_store, "rows", None)
            if rows and item.entity_id in rows:
                titles[item.entity_id] = rows[item.entity_id][1].get("title", item.entity_id)

        return HybridSearchResponse(
            answer=verdict,
            explanations=explain_results(parsed, channels, verdict.items, dimension_scores),
            parsed=parsed,
            titles=titles,
        )
