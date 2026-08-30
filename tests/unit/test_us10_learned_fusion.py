"""
US10 — learned fusion: trained, evaluated against RRF, recommended only
where it wins (TC-11).

Written test-first per GOV-04. The model is a logistic layer over
per-channel reciprocal-rank features — scale-free like RRF itself, so the
comparison is apples to apples. Training is seeded plain gradient descent:
deterministic, dependency-free, replayable (BD4 "fusion seedable").
"""
from __future__ import annotations

import pytest

from kse_memory.core.fusion import FusedItem, fuse_rrf
from kse_memory.core.learned_fusion import (
    FusionEvaluation,
    LabelledQuery,
    LearnedFusion,
    evaluate_vs_rrf,
    train_learned_fusion,
)

pytestmark = pytest.mark.unit


def oracle_queries(n=24):
    """vector channel is an oracle; conceptual and graph are adversarial:
    they rank the relevant doc LAST. RRF (equal weights) gets dragged down;
    a learner that upweights vector wins. Deterministic construction."""
    out = []
    for i in range(n):
        relevant = f"rel{i}"
        decoys = [f"d{i}a", f"d{i}b", f"d{i}c"]
        out.append(LabelledQuery(
            channels={
                "vector": tuple((doc, 1.0 - j * 0.1) for j, doc in enumerate([relevant] + decoys)),
                "conceptual": tuple((doc, 1.0 - j * 0.1) for j, doc in enumerate(decoys + [relevant])),
                "graph": tuple((doc, float(3 - j)) for j, doc in enumerate(decoys + [relevant])),
            },
            qrels={relevant: 1},
        ))
    return out


def symmetric_queries(n=24):
    """Every channel agrees perfectly: nothing to learn beyond RRF."""
    out = []
    for i in range(n):
        docs = [f"rel{i}", f"d{i}a", f"d{i}b"]
        ranking = tuple((doc, 1.0 - j * 0.2) for j, doc in enumerate(docs))
        out.append(LabelledQuery(
            channels={"vector": ranking, "conceptual": ranking, "graph": ranking},
            qrels={docs[0]: 1},
        ))
    return out


# ------------------------------------------------------------------ training
def test_training_is_deterministic():
    data = oracle_queries()
    a = train_learned_fusion(data, seed=7)
    b = train_learned_fusion(data, seed=7)
    assert a.weights == b.weights and a.bias == b.bias


def test_training_learns_to_trust_the_oracle_channel():
    model = train_learned_fusion(oracle_queries(), seed=7)
    assert model.weights["vector"] > model.weights["conceptual"]
    assert model.weights["vector"] > model.weights["graph"]


def test_model_carries_replay_metadata():
    model = train_learned_fusion(oracle_queries(), seed=7)
    assert model.seed == 7
    assert model.examples == 24
    assert set(model.weights) == {"vector", "conceptual", "graph"}


def test_training_without_positives_is_an_error():
    empty = [LabelledQuery(channels={"vector": (("a", 1.0),)}, qrels={})]
    with pytest.raises(ValueError, match="relevant"):
        train_learned_fusion(empty, seed=7)


# ------------------------------------------------------------------- fusing
def test_fuse_returns_receipted_items_ranked_descending():
    model = train_learned_fusion(oracle_queries(), seed=7)
    channels = oracle_queries(1)[0].channels
    items = model.fuse(channels, top_k=4)
    assert all(isinstance(item, FusedItem) for item in items)
    scores = [item.fused for item in items]
    assert scores == sorted(scores, reverse=True)
    assert items[0].ranks["vector"] is not None  # receipts intact


def test_learned_fusion_beats_rrf_on_the_oracle_scenario():
    """The construction RRF cannot win: two adversarial channels outvote the
    oracle under equal weights; the learner reweights."""
    model = train_learned_fusion(oracle_queries(), seed=7)
    example = oracle_queries(1)[0]
    learned_top = model.fuse(example.channels, top_k=1)[0].entity_id
    rrf_top = fuse_rrf(example.channels, top_k=1)[0].entity_id
    assert learned_top == "rel0"
    assert rrf_top != "rel0"


# --------------------------------------------------------------- evaluation
def test_evaluation_recommends_only_where_learned_wins():
    train, holdout = oracle_queries(24), oracle_queries(8)
    model = train_learned_fusion(train, seed=7)
    evaluation = evaluate_vs_rrf(model, holdout, k=3)
    assert isinstance(evaluation, FusionEvaluation)
    assert evaluation.learned_ndcg > evaluation.rrf_ndcg
    assert evaluation.recommended is True


def test_evaluation_refuses_to_recommend_a_tie():
    """TC-11's honesty clause: parity is not a win. When channels agree,
    learned can at best match RRF — recommended must be False."""
    train, holdout = symmetric_queries(24), symmetric_queries(8)
    model = train_learned_fusion(train, seed=7)
    evaluation = evaluate_vs_rrf(model, holdout, k=3)
    assert evaluation.learned_ndcg == pytest.approx(evaluation.rrf_ndcg)
    assert evaluation.recommended is False


def test_evaluation_carries_both_numbers_for_the_receipt():
    model = train_learned_fusion(oracle_queries(), seed=7)
    evaluation = evaluate_vs_rrf(model, oracle_queries(4), k=3)
    assert 0.0 <= evaluation.rrf_ndcg <= 1.0
    assert 0.0 <= evaluation.learned_ndcg <= 1.0
    assert evaluation.queries == 4


# ----------------------------------------------------- metrics move (layering)
def test_ndcg_now_lives_in_the_package():
    """benchmarks/ must depend on the package, never the reverse."""
    from kse_memory.core.metrics import ndcg_at_k, recall_at_k
    from benchmarks.harness import ndcg_at_k as harness_ndcg

    assert harness_ndcg is ndcg_at_k
    assert ndcg_at_k(["a"], {"a": 1}, k=1) == 1.0
    assert recall_at_k(["a", "b"], {"b": 1}, k=2) == 1.0
