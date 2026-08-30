"""
FR-03 — Query parse: embed the query, map it to dimension targets.

Design (BD3; criteria TC-02, TC-04):
- One embedding call per query. The query's similarity to each dimension's
  anchor centroid — the same centroids FR-02 scores items against — becomes
  that dimension's target weight, mapped onto [0, 1] exactly as item scores
  are. Queries and items therefore live in one geometry by construction: a
  query can be compared to an item's scores without any translation layer.
- This replaces the v2 keyword extraction against the hardcoded retail
  vocabulary. Dimensions come from the user's schema; nothing here knows any
  vocabulary of its own (TC-04).
- A ParsedQuery carries its replay identity (schema name + version, model id),
  so FR-06 explanations can state exactly which schema and model produced a
  ranking, and a schema bump visibly changes the parse.

Consumers: FR-04 retrieval takes the vector for the dense channel and the
targets for the conceptual channel. The v2 SearchService keyword path is
retired when FR-04 rewires retrieval, not here.

Guardrails honoured: AR-01 (no network), AR-04 (no GPU), AR-05 (typed surface).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

from .projection import SCORE_PRECISION, anchor_centroids, vector_cosine
from .schema import DimensionSchema

__all__ = ["ParsedQuery", "parse_query"]


@dataclass(frozen=True)
class ParsedQuery:
    """A query embedded and mapped into one schema's dimension space.

    Equality is value equality across the replay identity and the numbers,
    so two parses compare equal only when genuinely interchangeable.
    """

    text: str
    vector: Sequence[float]
    schema_name: str
    schema_version: str
    model_id: str
    targets: Mapping[str, float] = field(default_factory=dict)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ParsedQuery):
            return NotImplemented
        return (
            self.text == other.text
            and list(self.vector) == list(other.vector)
            and self.schema_name == other.schema_name
            and self.schema_version == other.schema_version
            and self.model_id == other.model_id
            and dict(self.targets) == dict(other.targets)
        )


def parse_query(
    text: str,
    schema: DimensionSchema,
    embedder,
    centroids: Optional[Mapping[str, Sequence[float]]] = None,
) -> ParsedQuery:
    """Parse ``text`` against ``schema`` using ``embedder``.

    Args:
        centroids: precomputed anchor centroids (e.g. an IngestPipeline's
            cache). When supplied, only the query itself is embedded — anchors
            are schema-level constants and re-embedding them per query would
            make query cost scale with schema size.

    Raises:
        ValueError: if the query is empty or whitespace.
    """
    if not text or not text.strip():
        raise ValueError("query text is empty")

    if centroids is None:
        centroids = anchor_centroids(schema, embedder)

    vector: List[float] = embedder.embed([text])[0]

    targets: Dict[str, float] = {}
    for dimension in schema.dimensions:
        unit = (vector_cosine(vector, centroids[dimension.name]) + 1.0) / 2.0
        targets[dimension.name] = round(min(1.0, max(0.0, unit)), SCORE_PRECISION)

    return ParsedQuery(
        text=text,
        vector=vector,
        schema_name=schema.name,
        schema_version=schema.version,
        model_id=getattr(embedder, "model_id", "unknown"),
        targets=targets,
    )
