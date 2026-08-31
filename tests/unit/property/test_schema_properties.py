"""
T-067 — Hypothesis properties for dimension schema round-trips (US4).

A schema that loads must survive YAML serialisation unchanged: schemas are
versioned replay inputs (BD4), so a lossy round-trip would corrupt replay
identity at the source.
"""
from __future__ import annotations

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from kse_memory.core.schema import SchemaError, load_schema

pytestmark = pytest.mark.unit

name = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz_-", min_size=1, max_size=15
).filter(lambda s: s.strip("_-"))
# YAML normalises the line-break class (NEL \x85, LS \u2028, PS \u2029, CR)
# in scalars — Hypothesis found "0\x85" round-tripping to "0 ". That is a
# property of YAML, not of load_schema: a schema containing those characters
# is not YAML-representable, so the round-trip property is scoped to text
# that is.
_YAML_BREAKS = {"\r", "\x85", "\u2028", "\u2029"}
anchor = st.text(min_size=1, max_size=40).filter(
    lambda s: s.strip() and not (set(s) & _YAML_BREAKS))
semver = st.tuples(
    st.integers(0, 99), st.integers(0, 99), st.integers(0, 99)
).map(lambda t: f"{t[0]}.{t[1]}.{t[2]}")


@st.composite
def schemas(draw):
    names = draw(st.lists(name, min_size=1, max_size=4, unique=True))
    return {
        "name": draw(name),
        "version": draw(semver),
        "dimensions": [
            {
                "name": dimension_name,
                "description": draw(st.text(max_size=30).filter(
                    lambda t: not (set(t) & _YAML_BREAKS))),
                "anchors": draw(st.lists(anchor, min_size=1, max_size=3)),
            }
            for dimension_name in names
        ],
    }


@given(schemas())
@settings(max_examples=200)
def test_loading_is_deterministic(mapping):
    assert load_schema(mapping) == load_schema(mapping)


@given(schemas())
@settings(max_examples=200)
def test_yaml_round_trip_preserves_the_schema(mapping):
    """dump -> parse -> load must equal load of the original mapping."""
    reloaded = yaml.safe_load(yaml.safe_dump(mapping, allow_unicode=True))
    assert load_schema(reloaded) == load_schema(mapping)


@given(schemas())
@settings(max_examples=200)
def test_dimension_order_and_anchors_are_preserved(mapping):
    schema = load_schema(mapping)
    assert list(schema.names()) == [d["name"] for d in mapping["dimensions"]]
    for declared in mapping["dimensions"]:
        assert schema[declared["name"]].anchors == tuple(declared["anchors"])


@given(schemas(), st.text(max_size=10).filter(
    lambda s: not __import__("re").match(r"^\d+\.\d+\.\d+$", s)))
@settings(max_examples=100)
def test_non_semver_versions_are_rejected(mapping, bad_version):
    with pytest.raises(SchemaError, match="version"):
        load_schema({**mapping, "version": bad_version})


@given(schemas())
@settings(max_examples=100)
def test_duplicate_dimension_names_are_rejected(mapping):
    duplicated = dict(mapping)
    duplicated["dimensions"] = mapping["dimensions"] + [mapping["dimensions"][0]]
    with pytest.raises(SchemaError, match="duplicate"):
        load_schema(duplicated)
