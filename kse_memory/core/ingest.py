"""
FR-01 — Ingest: normalise raw records to a universal Entity with a SHA-256
content hash for dedupe and deterministic replay.

Design (BD2/BD3, decisions D-08):
- The user's source system stays canonical; this module only *projects* raw
  records into KSE's universal Entity shape. It never mutates the source.
- ``content_hash`` is computed over a canonical serialisation of the entity's
  *content* fields only — volatile fields (id, timestamps, embeddings,
  conceptual scores, graph ids) are excluded, so the hash is stable across
  replays, key orderings, and projection rebuilds.
- When no id is supplied, a deterministic ``kse-<hash16>`` id is derived from
  the content hash, so re-ingesting an identical record cannot mint a new
  identity (dedupe by construction).

Guardrails honoured: AR-01 (no network), AR-05 (typed public surface).
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Mapping

from .models import Entity

__all__ = ["normalise_record", "content_hash", "CONTENT_FIELDS"]

#: Fields that constitute the *content* of an entity for hashing purposes.
#: Everything else (id, timestamps, embeddings, conceptual scores, graph ids)
#: is volatile or derived, and must not affect dedupe/replay identity.
CONTENT_FIELDS = (
    "title",
    "description",
    "entity_type",
    "category",
    "source",
    "tags",
    "media",
    "variations",
    "metadata",
)

_ENTITY_KEYS = frozenset(
    CONTENT_FIELDS + ("id",)
)  # keys normalise_record maps directly onto Entity


def normalise_record(raw: Mapping[str, Any]) -> Entity:
    """Normalise a raw adapter record into a universal :class:`Entity`.

    Args:
        raw: Arbitrary mapping from an adapter (Shopify, generic dict, etc.).
            Must contain at least ``title`` and ``description``. Unknown keys
            are preserved under ``metadata["extra"]`` rather than dropped.

    Returns:
        A validated Entity. If ``raw`` carries no ``id``, a deterministic
        ``kse-<hash16>`` id is derived from the content hash.

    Raises:
        ValueError: if required fields are missing or not strings.
    """
    for required in ("title", "description"):
        value = raw.get(required)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"record is missing required field: {required!r}")

    known: Dict[str, Any] = {k: raw[k] for k in _ENTITY_KEYS if k in raw}
    extra = {k: v for k, v in raw.items() if k not in _ENTITY_KEYS}
    if extra:
        metadata = dict(known.get("metadata") or {})
        metadata["extra"] = {**metadata.get("extra", {}), **extra}
        known["metadata"] = metadata

    entity = Entity(**known)

    if not raw.get("id"):
        # Deterministic identity from content: replay-stable, dedupe-safe.
        entity.id = f"kse-{content_hash(entity)[:16]}"
    return entity


_SET_LIKE_FIELDS = frozenset({"tags"})  # order carries no meaning for identity


def _canonical_payload(entity: Entity) -> Dict[str, Any]:
    """Extract the content fields in a canonical, hash-stable form."""
    payload: Dict[str, Any] = {}
    for field_name in CONTENT_FIELDS:
        value = getattr(entity, field_name, None)
        if value in (None, [], {}, ""):
            continue  # absent and empty are equivalent for identity purposes
        if field_name in _SET_LIKE_FIELDS and isinstance(value, list):
            value = sorted(str(v) for v in value)
        payload[field_name] = value
    return payload


def content_hash(entity: Entity) -> str:
    """SHA-256 hex digest of the entity's canonical content.

    Invariant under: dict key ordering, volatile fields (id, created_at,
    updated_at, embeddings, conceptual scores, knowledge_graph_id), and
    ``to_dict``/``from_dict`` round-trips. Changes iff content changes.
    """
    try:
        canonical = json.dumps(
            _canonical_payload(entity),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            # No default= fallback: str() on arbitrary objects embeds memory
            # addresses, silently corrupting replay identity (session-3 review).
        )
    except TypeError as exc:
        raise ValueError(
            "entity content is not deterministically serialisable for hashing; "
            "convert custom objects to JSON-safe values before ingest "
            f"(offending entity id={entity.id!r}): {exc}"
        ) from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
