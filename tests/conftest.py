"""
Shared test fixtures (D-16, T-065).

- ``no_network`` — AR-01 enforcement; any socket connect fails the test.
- ``_seed_rngs`` — autouse; every test starts from the same RNG state, so an
  unseeded random source can never make a test flaky (D-16: zero flake).
- ``frozen_clock`` — pins ``time.time``; for timestamp-adjacent assertions.
- ``stub_embedder`` / ``StubEmbedder`` — the D-16 deterministic stub: a real
  implementation of the FR-02 embedder contract (``embed(texts)`` returning
  unit vectors, plus ``model_id``), not a mock. Hash-projects text, so equal
  text means equal vector and different text almost surely differs.
  Note: docs/TESTING.md names EmbeddingServiceInterface; the FR-02 contract
  this implements is the simpler ``embed()`` protocol actually in use.
"""
import hashlib
import math
import random
import socket
import time

import pytest


@pytest.fixture
def no_network(monkeypatch):
    """Fail the test if anything attempts a network connection (AR-01).

    Use on every default-path test. Quickstart and local flows must pass under
    this fixture with zero exemptions.
    """

    def _blocked(*args, **kwargs):  # pragma: no cover - triggered only on violation
        raise AssertionError("AR-01 violated: default path attempted a network call")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    yield


@pytest.fixture(autouse=True)
def _seed_rngs():
    """D-16: every test runs from a fixed RNG state."""
    random.seed(1337)
    try:  # numpy is a default dependency, but stay import-tolerant
        import numpy

        numpy.random.seed(1337)
    except ImportError:  # pragma: no cover
        pass
    yield


@pytest.fixture
def frozen_clock(monkeypatch):
    """Pin time.time to a fixed instant; advance explicitly via .tick()."""

    class Clock:
        now = 1_756_500_000.0  # 2025-08-29T21:20:00Z, arbitrary but fixed

        def tick(self, seconds: float) -> None:
            Clock.now += seconds

    clock = Clock()
    monkeypatch.setattr(time, "time", lambda: Clock.now)
    return clock


class StubEmbedder:
    """Deterministic stand-in for the ONNX embedder (D-16 stub policy).

    A real implementation of the FR-02 embedder contract, not a mock: it
    hash-projects text onto the unit sphere, so results are deterministic,
    offline and instant, and downstream cosine arithmetic stays meaningful.
    """

    model_id = "stub-minilm-v1"
    dim = 16

    def embed(self, texts):
        out = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vec = [(digest[i % len(digest)] / 255.0) - 0.5 for i in range(self.dim)]
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


@pytest.fixture
def stub_embedder():
    """A fresh deterministic embedder (D-16: offline lanes never load a model)."""
    return StubEmbedder()
