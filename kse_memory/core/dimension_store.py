"""
Schema-driven dimension scores and a reference store (TC-04).

Why this exists:
``ConceptStoreInterface`` was typed against ``ConceptualDimensions`` — ten
hardcoded fashion axes. A user's schema may name any dimensions it likes, so
that type made the concept store structurally incapable of holding a v3
projection. It was the last blocker on TC-04's "no hardcoded fashion
vocabulary in the default path".

Scores travel with their schema identity. A bare mapping of numbers is not
interpretable — 0.8 of what, under whose definition? — and without the version
a schema bump cannot invalidate stale scores, which would quietly break the
replay guarantee (BD4).

Guardrails honoured: AR-01 (no network), AR-05 (typed public surface).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

__all__ = ["ConceptStoreAdapter", "DimensionScores", "InMemoryDimensionStore", "cosine_similarity"]


@dataclass(frozen=True)
class DimensionScores:
    """One entity's scores under one schema version.

    Raises:
        ValueError: if any score falls outside [0, 1]. Scores are similarities
            mapped onto the unit interval; anything else means the producer is
            not the scorer this store expects, and storing it would corrupt
            every later comparison.
    """

    schema_name: str
    schema_version: str
    scores: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in self.scores.items():
            if not (0.0 <= float(value) <= 1.0):
                raise ValueError(
                    f"dimension {name!r} scored {value!r}; scores must lie in [0.0, 1.0]"
                )

    def __getitem__(self, name: str) -> float:
        return self.scores[name]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DimensionScores):
            return NotImplemented
        return (
            self.schema_name == other.schema_name
            and self.schema_version == other.schema_version
            and dict(self.scores) == dict(other.scores)
        )

    @property
    def schema_key(self) -> Tuple[str, str]:
        """Identity of the schema these scores are expressed in."""
        return (self.schema_name, self.schema_version)


def cosine_similarity(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    """Cosine similarity over two sparse score mappings.

    Public because backends need it to rank consistently with the reference
    store; a private import across module boundaries is how implementations
    drift apart.
    """
    keys = set(a) | set(b)
    dot = sum(float(a.get(k, 0.0)) * float(b.get(k, 0.0)) for k in keys)
    na = math.sqrt(sum(float(v) ** 2 for v in a.values()))
    nb = math.sqrt(sum(float(v) ** 2 for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class InMemoryDimensionStore:
    """Reference implementation of the generalised concept store.

    Process-local and non-durable: it exists so the default CPU-only path has
    a working store with no service to run (AR-01), and so backends have a
    behavioural reference to conform to.
    """

    def __init__(self) -> None:
        self._scores: Dict[str, DimensionScores] = {}

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> bool:
        return True

    async def store_dimensions(self, entity_id: str, scores: DimensionScores) -> bool:
        self._scores[entity_id] = scores
        return True

    async def get_dimensions(self, entity_id: str) -> Optional[DimensionScores]:
        return self._scores.get(entity_id)

    async def delete_dimensions(self, entity_id: str) -> bool:
        return self._scores.pop(entity_id, None) is not None

    async def find_similar_dimensions(
        self, scores: DimensionScores, threshold: float = 0.8, limit: int = 10
    ) -> List[Tuple[str, float]]:
        """Rank stored entities by cosine similarity within the same schema.

        Comparison is scoped to ``schema_key``: two schemas may both define a
        dimension called "risk" and mean unrelated things by it, so scores from
        different schemas are not commensurable and are never compared.
        """
        hits = [
            (entity_id, cosine_similarity(scores.scores, stored.scores))
            for entity_id, stored in self._scores.items()
            if stored.schema_key == scores.schema_key
        ]
        hits = [h for h in hits if h[1] >= threshold]
        hits.sort(key=lambda h: h[1], reverse=True)
        return hits[:limit]

    async def get_dimension_statistics(self) -> Dict[str, Dict[str, float]]:
        by_dimension: Dict[str, List[float]] = {}
        for stored in self._scores.values():
            for name, value in stored.scores.items():
                by_dimension.setdefault(name, []).append(float(value))
        return {
            name: {
                "count": float(len(values)),
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
            }
            for name, values in by_dimension.items()
        }

    # ---------------------------------------------------- legacy compatibility
    async def store_conceptual_dimensions(self, product_id: str, dimensions) -> bool:
        return await self.store_dimensions(
            product_id, ConceptStoreAdapter.to_generic(dimensions)
        )

    async def get_conceptual_dimensions(self, product_id: str):
        stored = await self.get_dimensions(product_id)
        return None if stored is None else ConceptStoreAdapter.to_legacy(stored)

    async def delete_conceptual_dimensions(self, product_id: str) -> bool:
        return await self.delete_dimensions(product_id)

    async def find_similar_concepts(self, dimensions, threshold: float = 0.8, limit: int = 10):
        return await self.find_similar_dimensions(
            ConceptStoreAdapter.to_generic(dimensions), threshold, limit
        )


class ConceptStoreAdapter:
    """Translates between the legacy ConceptualDimensions and generic scores.

    Kept in one place so the deprecation has a single seam to delete when
    ConceptualDimensions finally goes.
    """

    LEGACY_SCHEMA = ("legacy-conceptual-dimensions", "2.0.0")

    @staticmethod
    def to_generic(dimensions) -> DimensionScores:
        raw = dimensions.to_dict() if hasattr(dimensions, "to_dict") else dict(dimensions)
        name, version = ConceptStoreAdapter.LEGACY_SCHEMA
        return DimensionScores(
            schema_name=name,
            schema_version=version,
            scores={k: float(v) for k, v in raw.items()},
        )

    @staticmethod
    def to_legacy(scores: DimensionScores):
        from .models import ConceptualDimensions

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the class warns on construction
            legacy = ConceptualDimensions()
        for name, value in scores.scores.items():
            if hasattr(legacy, name):
                setattr(legacy, name, float(value))
        return legacy
