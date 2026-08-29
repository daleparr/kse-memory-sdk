"""
US4 / FR-02 — user-supplied dimension schemas.

Design (BD2/BD3, criteria TC-04):
- Dimensions are *the user's*, never the library's. KSE ships no vocabulary of
  its own; a schema names each dimension and supplies anchor descriptions that
  the scorer embeds and compares against. This is what retires the legacy
  hardcoded retail dimensions (``ConceptualDimensions``, deprecated for v3).
- Schemas are versioned with semver because the version participates in
  projection identity: a schema bump must invalidate projections computed
  under the old one, or replay claims are false (BD4 "Replay").
- Loading is pure and offline: YAML parsing only, no resolution of remote
  refs, so AR-01 holds by construction.

Guardrails honoured: AR-01 (no network), AR-05 (typed public surface).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence, Union

import yaml

__all__ = ["Dimension", "DimensionSchema", "SchemaError", "load_schema"]

_SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


class SchemaError(ValueError):
    """Raised when a dimension schema is malformed.

    A subclass of ``ValueError`` so callers can treat it as ordinary bad input
    without importing KSE's exception hierarchy.
    """


@dataclass(frozen=True)
class Dimension:
    """One named axis of meaning, defined by example rather than by a word.

    ``anchors`` are short natural-language descriptions of what a high score
    on this dimension looks like. The scorer embeds them and measures an
    item's similarity to their centroid, so the anchors — not the name — carry
    the semantics.
    """

    name: str
    description: str
    anchors: tuple


@dataclass(frozen=True)
class DimensionSchema:
    """A versioned, ordered set of dimensions supplied by the user."""

    name: str
    version: str
    dimensions: tuple

    def names(self) -> tuple:
        """Dimension names, in schema order."""
        return tuple(d.name for d in self.dimensions)

    def __getitem__(self, name: str) -> Dimension:
        for d in self.dimensions:
            if d.name == name:
                return d
        raise KeyError(name)

    def __iter__(self) -> Iterator:
        return iter(self.dimensions)

    def __len__(self) -> int:
        return len(self.dimensions)


def load_schema(source: Union[str, Path, Mapping[str, Any]]) -> DimensionSchema:
    """Load and validate a dimension schema.

    Args:
        source: a path to a YAML file, or an already-parsed mapping.

    Returns:
        The validated schema.

    Raises:
        SchemaError: if the schema is malformed. Validation is strict and
            eager — a schema that loads is a schema that scores, so failures
            surface at configuration time rather than mid-projection.
    """
    if isinstance(source, Mapping):
        raw: Mapping[str, Any] = source
    else:
        path = Path(source)
        if not path.is_file():
            raise SchemaError(f"schema file not found: {path}")
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, Mapping):
            raise SchemaError(f"schema file did not parse to a mapping: {path}")
        raw = loaded

    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise SchemaError("schema is missing a non-empty 'name'")

    version = raw.get("version")
    if not isinstance(version, str) or not _SEMVER.match(version):
        raise SchemaError(
            f"schema 'version' must be semver (e.g. 1.0.0), got {version!r}"
        )

    raw_dimensions = raw.get("dimensions")
    if not isinstance(raw_dimensions, Sequence) or isinstance(raw_dimensions, str):
        raise SchemaError("schema 'dimensions' must be a list")
    if not raw_dimensions:
        raise SchemaError("schema declares no dimensions; at least one is required")

    dimensions = []
    seen = set()
    for index, entry in enumerate(raw_dimensions):
        if not isinstance(entry, Mapping):
            raise SchemaError(f"dimension #{index} is not a mapping")

        d_name = entry.get("name")
        if not isinstance(d_name, str) or not d_name.strip():
            raise SchemaError(f"dimension #{index} is missing a non-empty 'name'")
        if d_name in seen:
            raise SchemaError(f"duplicate dimension name: {d_name!r}")
        seen.add(d_name)

        anchors = entry.get("anchors") or []
        if isinstance(anchors, str) or not isinstance(anchors, Sequence):
            raise SchemaError(f"dimension {d_name!r}: 'anchors' must be a list")
        anchors = tuple(str(a) for a in anchors if str(a).strip())
        if not anchors:
            raise SchemaError(
                f"dimension {d_name!r} has no anchor descriptions; anchors are "
                "what give a dimension meaning, so at least one is required"
            )

        dimensions.append(
            Dimension(
                name=d_name,
                description=str(entry.get("description") or ""),
                anchors=anchors,
            )
        )

    return DimensionSchema(name=name, version=version, dimensions=tuple(dimensions))
