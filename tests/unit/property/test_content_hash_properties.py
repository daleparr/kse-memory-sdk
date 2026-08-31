"""
T-067 — Hypothesis properties for content_hash (FR-01).

The Session 3 defects (memory addresses in hashes, tag-order sensitivity)
are the class this lane exists to catch automatically: invariants over the
whole input space, not three hand-picked examples.
"""
from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from kse_memory.core.ingest import content_hash, normalise_record

pytestmark = pytest.mark.unit

nonblank = st.text(min_size=1, max_size=30).filter(lambda s: s.strip())

json_scalars = st.one_of(
    st.none(), st.booleans(), st.integers(-10**6, 10**6),
    st.floats(-1e6, 1e6, allow_nan=False), st.text(max_size=20),
)
json_values = st.recursive(
    json_scalars,
    lambda children: st.one_of(
        st.lists(children, max_size=4),
        st.dictionaries(st.text(min_size=1, max_size=10), children, max_size=4),
    ),
    max_leaves=12,
)
metadata = st.dictionaries(st.text(min_size=1, max_size=10), json_values, max_size=5)


@st.composite
def records(draw):
    return {
        "title": draw(nonblank),
        "description": draw(nonblank),
        "tags": draw(st.lists(st.text(min_size=1, max_size=10), max_size=5)),
        "metadata": draw(metadata),
    }


@given(records())
@settings(max_examples=200)
def test_hash_is_deterministic_across_constructions(record):
    a = content_hash(normalise_record(dict(record)))
    b = content_hash(normalise_record(dict(record)))
    assert a == b


@given(records(), st.randoms())
@settings(max_examples=200)
def test_metadata_key_order_cannot_matter(record, rng):
    """The Session 3 class of defect, generalised: any insertion order of the
    same metadata must hash identically."""
    items = list(record["metadata"].items())
    rng.shuffle(items)
    shuffled = {**record, "metadata": dict(items)}
    assert content_hash(normalise_record(record)) == content_hash(normalise_record(shuffled))


@given(records(), st.randoms())
@settings(max_examples=200)
def test_tag_permutation_cannot_matter(record, rng):
    tags = list(record["tags"])
    rng.shuffle(tags)
    assert content_hash(normalise_record(record)) == \
           content_hash(normalise_record({**record, "tags": tags}))


@given(records())
@settings(max_examples=200)
def test_absent_and_empty_are_equivalent(record):
    without = {k: v for k, v in record.items() if k not in ("tags", "metadata")}
    empty = {**without, "tags": [], "metadata": {}}
    assert content_hash(normalise_record(without)) == content_hash(normalise_record(empty))


@given(records(), nonblank)
@settings(max_examples=200)
def test_supplied_id_never_affects_the_hash(record, explicit_id):
    """Identity is derived FROM content; it must never feed back into it."""
    anonymous = content_hash(normalise_record(record))
    identified = content_hash(normalise_record({**record, "id": explicit_id}))
    assert anonymous == identified


@given(records(), nonblank)
@settings(max_examples=200)
def test_changed_description_changes_the_hash(record, extra):
    changed = {**record, "description": record["description"] + extra}
    assert content_hash(normalise_record(record)) != content_hash(normalise_record(changed))


@given(records())
@settings(max_examples=100)
def test_unicode_content_hashes_cleanly(record):
    """No crashes and 64 lowercase hex chars, whatever unicode arrives."""
    digest = content_hash(normalise_record(record))
    assert len(digest) == 64 and set(digest) <= set("0123456789abcdef")
