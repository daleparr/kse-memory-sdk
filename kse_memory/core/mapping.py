"""
US8 — cross-domain dimension mapping via anchor geometry (TC-08).

The mapping is derived FROM the schemas rather than declared beside them:
every dimension already has an anchor centroid in the shared embedding
space (FR-02/FR-03), so a target dimension's value is the similarity-
weighted combination of the source values —

    value[t] = Σ_s w(s, t) · value[s],
    w(s, t) ∝ max(0, cos(centroid_s, centroid_t)) ** SHARPNESS,

row-normalised per target. Sharpening concentrates weight on genuinely
aligned dimensions instead of smearing it across weak similarities; the
weight matrix rides along in the result, so a mapping can explain itself
(the FR-06 receipts culture applies to transforms too).

This replaces the v2 identity stub ("For now, return the same values" —
BD3 debt item 2). Same-schema mapping is the identity by definition, and
is short-circuited as such.

Guardrails honoured: AR-01 (no network), AR-05 (typed public surface).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from .projection import SCORE_PRECISION, anchor_centroids, vector_cosine
from .schema import DimensionSchema

__all__ = ["MappedDimensions", "SHARPNESS", "map_dimensions"]

#: Exponent applied to non-negative centroid similarities before row
#: normalisation. Higher values concentrate transfer on the best-aligned
#: source dimension; 4 keeps a same-text alignment dominant (~0.8+ of the
#: row) against typical unrelated-text cosines.
SHARPNESS = 4


@dataclass(frozen=True)
class MappedDimensions:
    """A transform that can explain itself.

    ``weights[target][source]`` is the fraction of ``values[target]`` that
    came from each source dimension — rows sum to 1.
    """

    values: Mapping[str, float]
    weights: Mapping[str, Mapping[str, float]]
    source_schema: Tuple[str, str]
    target_schema: Tuple[str, str]
    model_id: str

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MappedDimensions):
            return NotImplemented
        return (
            dict(self.values) == dict(other.values)
            and {k: dict(v) for k, v in self.weights.items()}
            == {k: dict(v) for k, v in other.weights.items()}
            and self.source_schema == other.source_schema
            and self.target_schema == other.target_schema
            and self.model_id == other.model_id
        )


def map_dimensions(
    source_schema: DimensionSchema,
    target_schema: DimensionSchema,
    values: Mapping[str, float],
    embedder: Any,
    centroids: Optional[Mapping[str, Any]] = None,
) -> MappedDimensions:
    """Transform ``values`` from the source schema into the target schema.

    Raises:
        ValueError: if ``values`` does not cover every source dimension —
            a partial profile would silently misweight the transfer.
    """
    missing = [name for name in source_schema.names() if name not in values]
    if missing:
        raise ValueError(f"values missing source dimensions: {missing}")

    identity = (
        source_schema.name == target_schema.name
        and source_schema.version == target_schema.version
        and source_schema.names() == target_schema.names()
    )
    if identity:
        return MappedDimensions(
            values={n: float(values[n]) for n in source_schema.names()},
            weights={n: {n: 1.0} for n in source_schema.names()},
            source_schema=(source_schema.name, source_schema.version),
            target_schema=(target_schema.name, target_schema.version),
            model_id=getattr(embedder, "model_id", "unknown"),
        )

    source_centroids = anchor_centroids(source_schema, embedder)
    target_centroids = anchor_centroids(target_schema, embedder)

    weights: Dict[str, Dict[str, float]] = {}
    mapped: Dict[str, float] = {}
    for target_name in target_schema.names():
        raw = {
            source_name: max(0.0, vector_cosine(
                source_centroids[source_name], target_centroids[target_name]
            )) ** SHARPNESS
            for source_name in source_schema.names()
        }
        total = sum(raw.values())
        if total == 0.0:
            # No source dimension points anywhere near this target: transfer
            # nothing rather than inventing a uniform prior.
            row = {name: 0.0 for name in raw}
            value = 0.0
        else:
            row = {name: weight / total for name, weight in raw.items()}
            value = sum(row[name] * float(values[name]) for name in row)
        weights[target_name] = {k: round(v, SCORE_PRECISION) for k, v in row.items()}
        mapped[target_name] = round(min(1.0, max(0.0, value)), SCORE_PRECISION)

    return MappedDimensions(
        values=mapped,
        weights=weights,
        source_schema=(source_schema.name, source_schema.version),
        target_schema=(target_schema.name, target_schema.version),
        model_id=getattr(embedder, "model_id", "unknown"),
    )
