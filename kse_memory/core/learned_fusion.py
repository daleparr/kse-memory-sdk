"""
US10 — learned fusion: a logistic layer over channel evidence (TC-11).

Design (D-07: learned fusion is P3 OPT-IN — nothing here touches the RRF
default; a caller must construct and evaluate a model deliberately):
- Features per (query, entity) are per-channel reciprocal ranks,
  ``1 / (RRF_K + rank)`` with 0 for absence — the same scale-free quantity
  RRF sums with equal weights. Learned fusion is therefore exactly "RRF
  with trained weights plus a bias", which makes the mandated comparison
  apples to apples and the learned weights directly interpretable.
- Training is plain batch gradient descent on logistic loss: seeded,
  deterministic, dependency-free, replayable (BD4 "fusion seedable"). The
  model object carries seed, example count and weights — its receipt.
- ``evaluate_vs_rrf`` scores both systems with the package's own tested
  nDCG and recommends the learned model ONLY where it strictly wins.
  Parity is not a win (TC-11's honesty clause).

Guardrails honoured: AR-01 (pure computation), AR-03 (numbers only from
the evaluation itself), AR-05 (typed public surface).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence, Tuple

from .fusion import DEFAULT_RRF_K, FusedItem, fuse_rrf
from .metrics import ndcg_at_k

__all__ = [
    "FusionEvaluation",
    "LabelledQuery",
    "LearnedFusion",
    "evaluate_vs_rrf",
    "train_learned_fusion",
]

Channels = Mapping[str, Sequence[Tuple[str, float]]]


@dataclass(frozen=True)
class LabelledQuery:
    """One training example: the channel rankings a query produced, and the
    relevance judgements for its documents (qrels convention: rel > 0)."""

    channels: Channels
    qrels: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class FusionEvaluation:
    """Both numbers, always — the recommendation is derivable, not asserted."""

    learned_ndcg: float
    rrf_ndcg: float
    recommended: bool
    queries: int
    k: int


def _features(channels: Channels) -> Dict[str, Dict[str, float]]:
    """entity -> {channel: reciprocal rank}, 0.0 where the channel lacks it."""
    names = list(channels)
    out: Dict[str, Dict[str, float]] = {}
    for name in names:
        for position, (entity_id, _) in enumerate(channels[name], start=1):
            out.setdefault(entity_id, {n: 0.0 for n in names})[name] = 1.0 / (
                DEFAULT_RRF_K + position
            )
    return out


@dataclass(frozen=True)
class LearnedFusion:
    """Trained weights over channels; usable wherever fuse_rrf is."""

    weights: Mapping[str, float]
    bias: float
    seed: int
    examples: int

    def score(self, features: Mapping[str, float]) -> float:
        z = self.bias + sum(
            self.weights.get(name, 0.0) * value for name, value in features.items()
        )
        return 1.0 / (1.0 + math.exp(-z))

    def fuse(self, channels: Channels, top_k: int = 10) -> Tuple[FusedItem, ...]:
        """Rank by the learned score; receipts identical in shape to RRF's."""
        by_entity = _features(channels)
        rrf_items = {item.entity_id: item for item in fuse_rrf(channels)}
        ordered = sorted(
            by_entity.items(),
            key=lambda kv: (-self.score(kv[1]), kv[0]),
        )[:top_k]
        return tuple(
            FusedItem(
                entity_id=entity_id,
                fused=round(self.score(features), 6),
                ranks=rrf_items[entity_id].ranks,
                scores=rrf_items[entity_id].scores,
            )
            for entity_id, features in ordered
        )


def train_learned_fusion(
    examples: Sequence[LabelledQuery],
    seed: int = 0,
    epochs: int = 300,
    learning_rate: float = 5.0,
) -> LearnedFusion:
    """Fit the logistic layer on (features, relevant?) pairs.

    The learning rate looks large because reciprocal-rank features live in
    [0, 1/61] — gradients are tiny in feature units.

    Raises:
        ValueError: if no example contains a relevant document — there is
            nothing to learn and a model fit to nothing would still emit
            confident numbers.
    """
    channel_names = sorted({name for ex in examples for name in ex.channels})
    rows: List[Tuple[Dict[str, float], int]] = []
    for example in examples:
        features = _features(example.channels)
        for entity_id, feats in features.items():
            label = 1 if example.qrels.get(entity_id, 0) > 0 else 0
            rows.append((feats, label))
    if not any(label for _, label in rows):
        raise ValueError("no relevant documents in the training examples")

    rng = random.Random(seed)
    weights = {name: rng.uniform(-0.01, 0.01) for name in channel_names}
    bias = 0.0
    for _ in range(epochs):
        grad_w = {name: 0.0 for name in channel_names}
        grad_b = 0.0
        for feats, label in rows:
            z = bias + sum(weights[n] * feats.get(n, 0.0) for n in channel_names)
            p = 1.0 / (1.0 + math.exp(-z))
            error = p - label
            for name in channel_names:
                grad_w[name] += error * feats.get(name, 0.0)
            grad_b += error
        scale = learning_rate / len(rows)
        for name in channel_names:
            weights[name] -= scale * grad_w[name]
        bias -= scale * grad_b

    return LearnedFusion(
        weights={name: round(w, 6) for name, w in weights.items()},
        bias=round(bias, 6),
        seed=seed,
        examples=len(examples),
    )


def evaluate_vs_rrf(
    model: LearnedFusion, holdout: Sequence[LabelledQuery], k: int = 10
) -> FusionEvaluation:
    """Score learned vs RRF on held-out queries; recommend only a strict win."""
    learned_scores = []
    rrf_scores = []
    for example in holdout:
        learned = [item.entity_id for item in model.fuse(example.channels, top_k=k)]
        rrf = [item.entity_id for item in fuse_rrf(example.channels, top_k=k)]
        learned_scores.append(ndcg_at_k(learned, example.qrels, k))
        rrf_scores.append(ndcg_at_k(rrf, example.qrels, k))

    learned_mean = sum(learned_scores) / len(learned_scores)
    rrf_mean = sum(rrf_scores) / len(rrf_scores)
    return FusionEvaluation(
        learned_ndcg=round(learned_mean, 6),
        rrf_ndcg=round(rrf_mean, 6),
        recommended=learned_mean > rrf_mean,
        queries=len(holdout),
        k=k,
    )
