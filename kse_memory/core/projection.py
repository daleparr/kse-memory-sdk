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
    "default_cache_dir",
    "default_model_path",
    "project",
    "score_dimensions",
]

#: Default local embedding model (D-04: all-MiniLM-L6-v2, ONNX int8).
DEFAULT_MODEL_ID = "onnx-minilm-l6-v2"

#: Entity fields that carry natural language worth embedding. Deliberately a
#: subset of ingest.CONTENT_FIELDS: media paths, variations and free-form
#: metadata are content for *hashing* but noise for *meaning*.
TEXT_FIELDS = ("title", "description", "entity_type", "category", "source", "tags")

_PRECISION = 6


def default_cache_dir() -> Path:
    """Resolve KSE's local cache directory (D-11).

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


def default_model_path(model_id: str = DEFAULT_MODEL_ID) -> Path:
    """Where a given model is expected to live inside the cache.

    Namespaced by model id so several models can coexist and so a model
    upgrade cannot silently reuse a stale artefact — the id is part of a
    projection's replay identity.
    """
    return default_cache_dir() / "models" / model_id / "model.onnx"


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
    (D-11: ``~/.cache/kse``) and must already exist; if it does not,
    construction fails immediately with a message naming the path and the
    override, so a CPU-only, offline setup is diagnosable rather than
    mysterious.
    """

    def __init__(self, model_path=None, model_id: str = DEFAULT_MODEL_ID) -> None:
        path = Path(model_path) if model_path is not None else default_model_path(model_id)
        if not path.exists():
            raise ModelNotAvailableError(
                f"embedding model not found at local path: {path}\n"
                "KSE never downloads models on the default path (AR-01). Fetch "
                "the model out of band and place it at that path, or set "
                "KSE_CACHE_DIR to a cache that already contains it."
            )
        self.model_path = path
        self.model_id = model_id
        self._session = None

    def _ensure_session(self):
        if self._session is None:
            import onnxruntime  # imported lazily: absent in doc/lint contexts

            self._session = onnxruntime.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"]
            )
        return self._session

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        raise NotImplementedError(
            "ONNX inference lands with T-008's tokeniser work; the embedder "
            "contract and its no-download guarantee are complete and tested."
        )


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
