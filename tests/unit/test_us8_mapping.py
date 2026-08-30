"""
US8 — cross-domain dimension mapping that actually transforms (TC-08).

Written test-first per GOV-04. The legacy map_dimensions was an identity
stub ("For now, return the same values" — BD3 debt item 2). The replacement
derives the mapping FROM the schemas: dimensions already live in one
embedding space through their anchor centroids, so weight transfers by
centroid similarity — no hand-built correspondence tables, any schema pair.
"""
from __future__ import annotations

import pytest

from tests.conftest import StubEmbedder
from kse_memory.core.mapping import MappedDimensions, map_dimensions
from kse_memory.core.schema import load_schema

pytestmark = pytest.mark.unit


def schema(name, dims):
    return load_schema({
        "name": name, "version": "1.0.0",
        "dimensions": [{"name": n, "description": "", "anchors": a} for n, a in dims],
    })


# Engineered alignment for the stub embedder: identical anchor text means
# identical centroid, so "opulence" must transfer to "premium_feel" and
# barely to "portability".
SOURCE = schema("shop", [
    ("opulence", ["made of cashmere and silk"]),
    ("sturdiness", ["survives being dropped daily"]),
])
TARGET = schema("catalogue", [
    ("premium_feel", ["made of cashmere and silk"]),
    ("portability", ["light enough for one hand"]),
])


def test_result_is_keyed_by_the_target_schema():
    mapped = map_dimensions(SOURCE, TARGET, {"opulence": 0.9, "sturdiness": 0.2}, StubEmbedder())
    assert isinstance(mapped, MappedDimensions)
    assert set(mapped.values) == {"premium_feel", "portability"}


def test_non_identity_behaviour_is_proven():
    """TC-08's explicit clause: the transform must be demonstrably not the
    identity — different keys, and values actually recomputed."""
    source_values = {"opulence": 0.9, "sturdiness": 0.2}
    mapped = map_dimensions(SOURCE, TARGET, source_values, StubEmbedder())
    assert mapped.values != source_values
    assert set(mapped.values) != set(source_values)


def test_aligned_dimensions_receive_the_weight():
    """Identical anchors -> identical centroids -> opulence dominates the
    premium_feel row and its value transfers almost whole. (Cross-dimension
    VALUE comparisons are meaningless under a hash embedder — the semantic
    claim lives in the integration lane with the genuine model.)"""
    mapped = map_dimensions(SOURCE, TARGET, {"opulence": 0.9, "sturdiness": 0.1}, StubEmbedder())
    assert mapped.weights["premium_feel"]["opulence"] > 0.8   # dominant contributor
    assert mapped.values["premium_feel"] == pytest.approx(0.9, abs=0.1)


def test_values_stay_on_the_unit_interval():
    mapped = map_dimensions(SOURCE, TARGET, {"opulence": 1.0, "sturdiness": 1.0}, StubEmbedder())
    assert all(0.0 <= v <= 1.0 for v in mapped.values.values())


def test_same_schema_is_the_identity():
    """Mapping a schema onto itself is a no-op by definition, not by luck."""
    values = {"opulence": 0.7, "sturdiness": 0.3}
    mapped = map_dimensions(SOURCE, SOURCE, values, StubEmbedder())
    assert mapped.values == values


def test_mapping_carries_replay_provenance():
    mapped = map_dimensions(SOURCE, TARGET, {"opulence": 0.9, "sturdiness": 0.2}, StubEmbedder())
    assert mapped.source_schema == ("shop", "1.0.0")
    assert mapped.target_schema == ("catalogue", "1.0.0")
    assert mapped.model_id == StubEmbedder.model_id


def test_weights_are_normalised_per_target():
    mapped = map_dimensions(SOURCE, TARGET, {"opulence": 0.5, "sturdiness": 0.5}, StubEmbedder())
    for target_name, row in mapped.weights.items():
        assert sum(row.values()) == pytest.approx(1.0)


def test_missing_source_value_is_an_error():
    with pytest.raises(ValueError, match="sturdiness"):
        map_dimensions(SOURCE, TARGET, {"opulence": 0.9}, StubEmbedder())


def test_mapping_is_deterministic():
    values = {"opulence": 0.9, "sturdiness": 0.2}
    assert map_dimensions(SOURCE, TARGET, values, StubEmbedder()) == \
           map_dimensions(SOURCE, TARGET, values, StubEmbedder())


# ------------------------------------------------------------ legacy stub
def test_legacy_map_dimensions_is_no_longer_the_identity():
    """BD3 debt item 2, closed: the v2 DomainMapper now transforms through
    the same engine (profiles' dimension descriptions become anchors)."""
    from kse_memory.core.domain_mapping import ConceptualSpaceMapper, Domain

    mapper = ConceptualSpaceMapper()
    # Non-uniform values: a uniform vector is a fixed point of every convex
    # combination and proves nothing about the transform.
    source_names = sorted(mapper.domain_profiles[Domain.RETAIL].dimensions)
    source_values = {name: (0.9 if i % 2 == 0 else 0.1)
                     for i, name in enumerate(source_names)}
    mapped = mapper.map_dimensions(
        Domain.RETAIL, Domain.HEALTHCARE, source_values,
        embedder=StubEmbedder(),
    )
    assert mapped != source_values
    assert set(mapped) == set(mapper.domain_profiles[Domain.HEALTHCARE].dimensions)
