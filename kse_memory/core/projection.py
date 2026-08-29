"""
FR-02 — Projection: embed an entity's text and score it against a user schema.

Design (BD2/BD3, criteria TC-02, TC-04, TC-07; decisions D-03, D-04):
- The default embedder is a local ONNX MiniLM resolved from a **local cache
  path**. It never downloads: a missing model is a loud, actionable error, not
  a silent fetch. That is what makes AR-01 (zero network on the default path)
  true by construction rather than by convention.
- Scoring is similarity to anchor centroids, not keyword matching. A dimension
  means what its anchors say it means, so the library ships no vocabulary of
  its own (TC-04).
- A projection carries its own replay identity: content hash, schema name and
  version, and embedding model id. Per BD4 those three reproduce any
  projection; if any changes, the projection is a different artefact.
- Scores are rounded to ``_PRECISION`` decimals so replay is stable across
  platforms whose float reductions differ in the last bits.

Scope: FR-02's incremental graph-edge upsert is specified separately (TC-09)
and is not implemented here.

Guardrails honoured: AR-01 (no network), AR-04 (no GPU dependency),
AR-05 (typed public surface).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from .ingest import content_hash
from .models import Entity
from .schema import DimensionSchema

__all__ = [
    "DEFAULT_MODEL_ID",
    "ModelNotAvailableError",
    "OnnxEmbedder",
    "Projection",
    "TEXT_FIELDS",
    "anchor_centroids",
    "entity_text",
    "score_from_vectors",
    "default_cache_dir",
    "default_model_dir",
    "default_model_path",
    "project",
    "score_dimensions",
    "upsert_projection",
]

#: Default local embedding model (D-04: all-MiniLM-L6-v2, ONNX int8).
DEFAULT_MODEL_ID = "onnx-minilm-l6-v2"

#: Entity fields that carry natural language worth embedding. Deliberately a
#: subset of ingest.CONTENT_FIELDS: media paths, variations and free-form
#: metadata are content for *hashing* but noise for *meaning*.
TEXT_FIELDS = ("title", "description", "entity_type", "category", "source", "tags")

_PRECISION = 6


def default_cache_dir() -> Path:
    """Resolve KSE's local cache directory (D-101).

    Precedence: ``KSE_CACHE_DIR`` (explicit override) → ``XDG_CACHE_HOME/kse``
    → ``~/.cache/kse``. XDG is honoured because the convention exists and
    users who set it mean it; the fallback is the documented default.

    Resolution is pure — nothing is created. Writing into the cache is the job
    of whatever populates it, never of a default-path read (AR-01).
    """
    explicit = os.environ.get("KSE_CACHE_DIR")
    if explicit:
        return Path(explicit)
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "kse"
    return Path.home() / ".cache" / "kse"


def default_model_dir(model_id: str = DEFAULT_MODEL_ID) -> Path:
    """The directory holding one model's artefacts (``model.onnx``, ``vocab.txt``).

    Namespaced by model id so several models can coexist and so a model
    upgrade cannot silently reuse a stale artefact — the id is part of a
    projection's replay identity.
    """
    return default_cache_dir() / "models" / model_id


def default_model_path(model_id: str = DEFAULT_MODEL_ID) -> Path:
    """Where a given model's ONNX graph is expected to live."""
    return default_model_dir(model_id) / "model.onnx"


class ModelNotAvailableError(RuntimeError):
    """Raised when the local embedding model cannot be resolved.

    Deliberately not a network error: reaching this state means the model is
    absent, and the correct response is to fetch it *out of band*, never as a
    side effect of a default-path call (AR-01).
    """


@dataclass(frozen=True)
class Projection:
    """An entity's scores under one schema and one embedding model.

    Equality is value equality across the full replay identity, so two
    projections compare equal only if they are genuinely reproducible from the
    same inputs.
    """

    entity_id: str
    content_hash: str
    schema_name: str
    schema_version: str
    model_id: str
    scores: Mapping[str, float] = field(default_factory=dict)


class OnnxEmbedder:
    """Default CPU-only text embedder, backed by a locally cached ONNX model.

    The model is *never* downloaded. ``model_path`` defaults to the local cache
    (D-101: ``~/.cache/kse``) and must already exist; if it does not,
    construction fails immediately with a message naming the path and the
    override, so a CPU-only, offline setup is diagnosable rather than
    mysterious.
    """

    def __init__(self, model_path=None, model_id: str = DEFAULT_MODEL_ID, model_dir=None) -> None:
        if model_dir is not None:
            directory = Path(model_dir)
            path = directory / "model.onnx"
        elif model_path is not None:
            path = Path(model_path)
            directory = path.parent
        else:
            directory = default_model_dir(model_id)
            path = directory / "model.onnx"

        if not path.exists():
            raise ModelNotAvailableError(
                f"embedding model not found at local path: {path}\n"
                "KSE never downloads models on the default path (AR-01). Fetch "
                "the model out of band and place it at that path, or set "
                "KSE_CACHE_DIR to a cache that already contains it."
            )
        self.model_path = path
        self.model_dir = directory
        self.model_id = model_id
        self._session = None
        self._tokenizer = None

    def _ensure_session(self):
        if self._session is None:
            import onnxruntime  # imported lazily: absent in doc/lint contexts

            self._session = onnxruntime.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"]
            )
        return self._session

    @property
    def tokenizer(self):
        """The WordPiece tokeniser, loaded from ``vocab.txt`` on first use."""
        if self._tokenizer is None:
            from .tokenizer import WordPieceTokenizer

            self._tokenizer = WordPieceTokenizer.from_model_dir(self.model_dir)
        return self._tokenizer

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed ``texts`` into L2-normalised sentence vectors.

        Mean-pools the final hidden states over real tokens only. The
        attention mask is applied before averaging, so a short text batched
        with a long one is unaffected by the padding between them — batching
        is a performance decision and must never change a result.
        """
        import numpy as np

        if not texts:
            return []

        encoded = self.tokenizer.encode_batch(list(texts))
        session = self._ensure_session()

        # Models differ on whether they accept token_type_ids; feed only what
        # this graph declares, or onnxruntime rejects the call.
        declared = {i.name for i in session.get_inputs()}
        feed = {k: v for k, v in encoded.items() if k in declared}
        missing = declared - set(feed)
        if missing:
            raise ValueError(
                f"model expects inputs this tokeniser does not provide: {sorted(missing)}"
            )

        hidden = np.asarray(session.run(None, feed)[0], dtype=np.float64)
        mask = encoded["attention_mask"].astype(np.float64)[..., None]

        summed = (hidden * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), 1e-9, None)
        pooled = summed / counts

        norms = np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
        return (pooled / norms).tolist()


def _entity_text(entity: Entity) -> str:
    """Render an entity's language-bearing fields deterministically.

    List fields are sorted, matching FR-01's set-like treatment of tags, so a
    cosmetic reorder cannot change a projection any more than it can change a
    content hash.
    """
    parts: List[str] = []
    for name in TEXT_FIELDS:
        value = getattr(entity, name, None)
        if value in (None, [], {}, ""):
            continue
        if isinstance(value, (list, tuple)):
            value = " ".join(sorted(str(v) for v in value))
        parts.append(f"{name}: {value}")
    return "\n".join(parts)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _centroid(vectors: Sequence[Sequence[float]]) -> List[float]:
    count = len(vectors)
    return [sum(col) / count for col in zip(*vectors)]


def anchor_centroids(schema: DimensionSchema, embedder) -> Dict[str, List[float]]:
    """Embed every dimension's anchors once and average them per dimension.

    Anchors are schema-level and constant, so a long-lived caller should
    compute this once and reuse it. Re-deriving it per item would make ingest
    cost scale with schema size for no benefit.
    """
    texts: List[str] = []
    spans = []
    for dimension in schema.dimensions:
        start = len(texts)
        texts.extend(dimension.anchors)
        spans.append((start, len(texts)))

    vectors = embedder.embed(texts)
    return {
        dimension.name: _centroid(vectors[start:end])
        for dimension, (start, end) in zip(schema.dimensions, spans)
    }


def score_from_vectors(
    entity_vector: Sequence[float], centroids: Mapping[str, Sequence[float]]
) -> Dict[str, float]:
    """Score one entity vector against precomputed dimension centroids."""
    scores: Dict[str, float] = {}
    for name, centroid in centroids.items():
        unit = (_cosine(entity_vector, centroid) + 1.0) / 2.0
        scores[name] = round(min(1.0, max(0.0, unit)), _PRECISION)
    return scores


def entity_text(entity: Entity) -> str:
    """Public alias for the canonical text rendering used when embedding."""
    return _entity_text(entity)


def score_dimensions(
    entity: Entity, schema: DimensionSchema, embedder
) -> Dict[str, float]:
    """Score ``entity`` against every dimension in ``schema``.

    Each dimension's anchors are embedded and averaged; the score is the cosine
    similarity between the entity's text and that centroid, mapped from
    [-1, 1] onto [0, 1].

    One batched ``embed`` call covers the entity and every anchor, so the cost
    is one model invocation per projection rather than one per dimension.
    """
    texts: List[str] = [_entity_text(entity)]
    spans = []
    for dimension in schema.dimensions:
        start = len(texts)
        texts.extend(dimension.anchors)
        spans.append((start, len(texts)))

    vectors = embedder.embed(texts)
    if len(vectors) != len(texts):
        raise ValueError(
            f"embedder returned {len(vectors)} vectors for {len(texts)} texts"
        )

    entity_vector = vectors[0]
    scores: Dict[str, float] = {}
    for dimension, (start, end) in zip(schema.dimensions, spans):
        similarity = _cosine(entity_vector, _centroid(vectors[start:end]))
        unit = (similarity + 1.0) / 2.0
        scores[dimension.name] = round(min(1.0, max(0.0, unit)), _PRECISION)
    return scores


def project(entity: Entity, schema: DimensionSchema, embedder) -> Projection:
    """Project ``entity`` into ``schema``'s dimension space.

    The returned :class:`Projection` carries the full replay identity, so it
    can be recomputed — or invalidated — without consulting the source system.
    """
    return Projection(
        entity_id=entity.id,
        content_hash=content_hash(entity),
        schema_name=schema.name,
        schema_version=schema.version,
        model_id=getattr(embedder, "model_id", "unknown"),
        scores=score_dimensions(entity, schema, embedder),
    )


#: Relationship type linking an entity to one scored dimension.
SCORED_AS = "SCORED_AS"

#: Node-id prefix for dimension nodes, namespaced by schema so two schemas
#: naming the same dimension do not collide in one graph.
def dimension_node_id(schema_name: str, dimension: str) -> str:
    """Stable node id for a dimension within a schema."""
    return f"dim:{schema_name}:{dimension}"


def _identity(projection: "Projection") -> Dict[str, str]:
    """The triple that decides whether stored state is current (BD4 Replay)."""
    return {
        "content_hash": projection.content_hash,
        "schema_version": projection.schema_version,
        "model_id": projection.model_id,
    }


async def upsert_projection(projection: "Projection", graph_store) -> bool:
    """Write ``projection`` into the graph, incrementally.

    Returns ``True`` if anything was written, ``False`` if the stored state was
    already current. The check is the replay identity: if content hash, schema
    version and model id all match what the node carries, nothing about the
    projection can have changed, so the write is skipped entirely.

    Stale edges are removed when a schema narrows, so a dimension dropped from
    the schema does not linger as an orphaned score.

    Guardrails honoured: AR-01 — this touches only the supplied store.
    """
    node_id = projection.entity_id
    existing = await graph_store.get_node(node_id)
    identity = _identity(projection)

    if existing:
        properties = existing.get("properties", {}) or {}
        if all(properties.get(k) == v for k, v in identity.items()):
            return False  # already current — the incremental guarantee

    node_properties = {
        **identity,
        "schema_name": projection.schema_name,
    }
    if existing:
        await graph_store.update_node(node_id, node_properties)
    else:
        await graph_store.create_node(node_id, ["Entity"], node_properties)

    # Drop edges for dimensions this projection no longer scores. Existing
    # edges are discovered through GraphStoreInterface.get_neighbors rather
    # than any backend's internals, so this works against a real store.
    wanted = {
        dimension_node_id(projection.schema_name, name) for name in projection.scores
    }
    for neighbour in await graph_store.get_neighbors(node_id, [SCORED_AS]) or []:
        target = neighbour.get("id") if isinstance(neighbour, Mapping) else neighbour
        if target and target not in wanted:
            await graph_store.delete_relationship(node_id, target, SCORED_AS)

    for name, score in projection.scores.items():
        await graph_store.create_relationship(
            node_id,
            dimension_node_id(projection.schema_name, name),
            SCORED_AS,
            {"score": score, "schema_version": projection.schema_version},
        )
    return True
