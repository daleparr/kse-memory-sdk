"""
The v3 ingest path: FR-01 normalisation wired to FR-02 projection.

Design (BD2/BD3; criteria TC-02, TC-04; decisions D-03, D-08):
- One call takes a raw adapter record to a normalised entity, a projection
  under the user's schema, and incremental writes to whichever stores are
  configured. Both halves already carried the replay identity; this is the
  seam that makes it flow end to end.
- Incremental by default. ``upsert_projection`` skips when the stored identity
  already matches, and this pipeline skips the vector write on the same
  signal, so re-ingesting unchanged content costs nothing anywhere.
- Anchor centroids are computed once per pipeline, not once per record.
  Anchors are schema-level constants; embedding them per item would make
  ingest cost scale with schema size.
- Normalisation happens before any store is touched, so a malformed record
  raises at the door rather than leaving a half-written graph.

Not written here: the concept store. ``ConceptStoreInterface`` is typed
against the legacy ``ConceptualDimensions``, so it cannot hold schema-driven
scores without an interface change. Scores live in the graph as scored edges.

Guardrails honoured: AR-01 (no network), AR-04 (no GPU dependency),
AR-05 (typed public surface).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .ingest import content_hash, normalise_record
from .models import Entity
from .projection import (
    Projection,
    anchor_centroids,
    entity_text,
    score_from_vectors,
    upsert_projection,
)
from .schema import DimensionSchema

__all__ = ["IngestPipeline", "IngestResult"]


@dataclass(frozen=True)
class IngestResult:
    """What one record produced.

    ``written`` is False when the stores already held this exact projection —
    the signal that the incremental path did its job.
    """

    entity: Entity
    projection: Projection
    written: bool


class IngestPipeline:
    """Raw records in; normalised, projected, incrementally stored entities out."""

    def __init__(
        self,
        schema: DimensionSchema,
        embedder,
        *,
        graph_store=None,
        vector_store=None,
    ) -> None:
        self.schema = schema
        self.embedder = embedder
        self.graph_store = graph_store
        self.vector_store = vector_store
        self._centroids: Optional[Dict[str, List[float]]] = None

    @property
    def centroids(self) -> Dict[str, List[float]]:
        """Dimension centroids, embedded once on first use."""
        if self._centroids is None:
            self._centroids = anchor_centroids(self.schema, self.embedder)
        return self._centroids

    async def ingest(self, raw: Mapping[str, Any]) -> IngestResult:
        """Normalise, project and incrementally store one record.

        Raises:
            ValueError: if the record is malformed. Raised before any store is
                touched, so a bad record cannot leave partial state behind.
        """
        entity = normalise_record(raw)

        vector = self.embedder.embed([entity_text(entity)])[0]
        projection = Projection(
            entity_id=entity.id,
            content_hash=content_hash(entity),
            schema_name=self.schema.name,
            schema_version=self.schema.version,
            model_id=getattr(self.embedder, "model_id", "unknown"),
            scores=score_from_vectors(vector, self.centroids),
        )

        # The graph knows whether this projection is already current. Without
        # one there is nothing to ask, so treat every record as changed rather
        # than silently skipping the write.
        if self.graph_store is not None:
            changed = await upsert_projection(projection, self.graph_store)
        else:
            changed = self.vector_store is not None

        if changed and self.vector_store is not None:
            await self.vector_store.upsert_vectors(
                [(entity.id, vector, self._metadata(entity, projection))]
            )

        written = changed

        return IngestResult(entity=entity, projection=projection, written=written)

    async def ingest_many(
        self, raws: Sequence[Mapping[str, Any]]
    ) -> List[IngestResult]:
        """Ingest a sequence of records, preserving order."""
        return [await self.ingest(raw) for raw in raws]

    @staticmethod
    def _metadata(entity: Entity, projection: Projection) -> Dict[str, Any]:
        return {
            "title": entity.title,
            "content_hash": projection.content_hash,
            "schema_name": projection.schema_name,
            "schema_version": projection.schema_version,
            "model_id": projection.model_id,
            **{f"dim_{k}": v for k, v in projection.scores.items()},
        }
