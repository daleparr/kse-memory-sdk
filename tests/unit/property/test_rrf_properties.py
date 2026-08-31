"""
T-067 (RRF portion) — Hypothesis property suites for fusion.

D-16 mandates property tests for RRF: scale invariance, rank monotonicity and
k-parameter bounds. These are the invariants that make rank-based fusion safe
to trust across arbitrary channel score distributions — exactly the class of
guarantee example-based tests cannot provide.
"""
from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from kse_memory.core.fusion import fuse_rrf

pytestmark = pytest.mark.unit

entity_ids = st.text(alphabet="abcdefgh", min_size=1, max_size=3)


@st.composite
def channels(draw, min_channels=1, max_channels=3):
    names = draw(st.lists(
        st.sampled_from(["vector", "conceptual", "graph"]),
        min_size=min_channels, max_size=max_channels, unique=True))
    out = {}
    for name in names:
        ids = draw(st.lists(entity_ids, min_size=0, max_size=6, unique=True))
        scores = draw(st.lists(
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
            min_size=len(ids), max_size=len(ids)))
        # rankings arrive rank-ordered from FR-04; enforce that shape
        rows = sorted(zip(ids, scores), key=lambda r: -r[1])
        out[name] = tuple(rows)
    return out


@given(channels(), st.floats(min_value=1e-6, max_value=1e6, allow_nan=False))
@settings(max_examples=200)
def test_scale_invariance(chans, factor):
    """Positive rescaling of any channel's scores cannot change the fusion."""
    rescaled = {
        name: tuple((entity, score * factor) for entity, score in rows)
        for name, rows in chans.items()
    }
    assert [(i.entity_id, i.fused) for i in fuse_rrf(chans)] == \
           [(i.entity_id, i.fused) for i in fuse_rrf(rescaled)]


@given(channels(min_channels=2))
@settings(max_examples=200)
def test_rank_monotonicity(chans):
    """Promoting an entity to rank 1 in one channel never lowers its fused score."""
    name, rows = next(iter(chans.items()))
    if len(rows) < 2:
        return
    entity = rows[-1][0]
    before = {i.entity_id: i.fused for i in fuse_rrf(chans)}[entity]
    promoted = dict(chans)
    promoted[name] = (rows[-1],) + rows[:-1]
    after = {i.entity_id: i.fused for i in fuse_rrf(promoted)}[entity]
    assert after >= before


@given(channels(), st.integers(min_value=1, max_value=1000))
@settings(max_examples=200)
def test_fused_scores_are_bounded(chans, k):
    """0 < fused <= channels / (k + 1) for unweighted RRF."""
    fused = fuse_rrf(chans, k=k)
    upper = len(chans) / (k + 1)
    for item in fused:
        assert 0.0 < item.fused <= upper + 1e-12


@given(channels())
@settings(max_examples=200)
def test_channel_dict_order_is_irrelevant(chans):
    reversed_order = dict(reversed(list(chans.items())))
    assert [i.entity_id for i in fuse_rrf(chans)] == \
           [i.entity_id for i in fuse_rrf(reversed_order)]


@given(channels(min_channels=1, max_channels=1))
@settings(max_examples=200)
def test_single_channel_preserves_its_ranking(chans):
    """With one channel, RRF must be a no-op on the order."""
    (rows,) = chans.values()
    assert [i.entity_id for i in fuse_rrf(chans)] == [entity for entity, _ in rows]


@given(channels())
@settings(max_examples=100)
def test_every_input_entity_appears_exactly_once(chans):
    fused = fuse_rrf(chans)
    ids = [i.entity_id for i in fused]
    assert len(ids) == len(set(ids))
    assert set(ids) == {e for rows in chans.values() for e, _ in rows}
