"""
Integration lane — genuine ONNX MiniLM from the local cache (D-16, T-065).

Skips when no model is cached at ~/.cache/kse (AR-01 forbids downloading one
during a test run). CI's onnx lane warms the cache in a setup step, outside
any test, then runs this for real.

This is the only place the hand-written WordPiece tokeniser meets the real
30522-token vocabulary and a real exported graph — everything else exercises
the contract through stubs and synthetic models.
"""
from __future__ import annotations

import pytest

from kse_memory.core.projection import OnnxEmbedder, default_model_dir

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (default_model_dir() / "model.onnx").exists(),
        reason="no MiniLM cached at ~/.cache/kse; AR-01 forbids downloading one",
    ),
]


def test_real_model_produces_normalised_embeddings():
    e = OnnxEmbedder()
    vecs = e.embed(["a technical document", "a friendly greeting"])
    assert len(vecs) == 2
    for v in vecs:
        assert abs(sum(x * x for x in v) ** 0.5 - 1.0) < 1e-4
    assert vecs[0] != vecs[1]


def test_real_model_is_deterministic():
    e = OnnxEmbedder()
    assert e.embed(["replay identity"]) == e.embed(["replay identity"])


def test_real_model_batch_invariance():
    """Padding must not leak into a short text's vector — against the real graph."""
    e = OnnxEmbedder()
    alone = e.embed(["short"])[0]
    batched = e.embed(["short", "a much longer sentence that forces padding of the first"])[0]
    assert alone == pytest.approx(batched, abs=1e-5)


def test_real_model_similarity_sanity():
    """Near-paraphrases should sit closer than unrelated text."""
    e = OnnxEmbedder()
    a, b, c = e.embed([
        "how to tune an approximate nearest neighbour index",
        "tuning ANN index parameters for vector search",
        "a recipe for lemon drizzle cake",
    ])
    dot = lambda x, y: sum(p * q for p, q in zip(x, y))
    assert dot(a, b) > dot(a, c)
