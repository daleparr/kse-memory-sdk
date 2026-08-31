"""
Framework story — the LangChain adapter, tested against real local stores.

Written test-first per GOV-04, replacing the mocked tests/test_integrations.py
(retirement map, T-068). Skips cleanly when langchain-core is absent — the
extras-gated lane — and runs live wherever it is installed.
"""
from __future__ import annotations

import pytest

langchain_core = pytest.importorskip(
    "langchain_core", reason="extras-gated: pip install langchain-core"
)

from kse_memory.core.config import KSEConfig
from kse_memory.core.memory import KSEMemory
from kse_memory.core.models import Product
from kse_memory.integrations.langchain import KSEVectorStore

from kse_memory.core.projection import default_model_dir

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.component,
    pytest.mark.skipif(
        not (default_model_dir() / "model.onnx").exists(),
        reason="semantic path needs the cached ONNX model (or the ST extra)",
    ),
]


@pytest.fixture
async def kse():
    memory = KSEMemory(KSEConfig())  # all-memory defaults since T-068
    await memory.initialize("generic", {"data_source": lambda **kw: []})
    for pid, title, description in [
        ("p1", "running shoes", "lightweight cushioned trainers for daily runs"),
        ("p2", "hiking boots", "waterproof leather boots for rough trails"),
        ("p3", "office chair", "ergonomic mesh chair with lumbar support"),
    ]:
        # the real path: the embedding service now serves the default model
        # from the ONNX cache, so add_product can compute embeddings for real
        await memory.add_product(
            Product(id=pid, title=title, description=description),
            compute_embeddings=True, compute_concepts=False,
        )
    yield memory
    await memory.disconnect()


def test_documents_are_real_langchain_documents():
    """With langchain-core installed, the adapter must use IT, not the stub."""
    from langchain_core.documents import Document as RealDocument

    from kse_memory.integrations.langchain import Document

    assert Document is RealDocument


async def test_vector_store_wraps_kse(kse):
    store = KSEVectorStore(kse_memory=kse, search_type="semantic")
    docs = store.similarity_search("running", k=2)
    assert docs
    assert all(hasattr(d, "page_content") and hasattr(d, "metadata") for d in docs)


async def test_similarity_search_with_score_returns_pairs(kse):
    store = KSEVectorStore(kse_memory=kse, search_type="semantic")
    pairs = store.similarity_search_with_score("boots", k=2)
    assert pairs
    for doc, score in pairs:
        assert isinstance(score, float)
        assert hasattr(doc, "page_content")


async def test_search_type_round_trips_from_string(kse):
    """SearchType values are lowercase; the adapter must map strings
    case-insensitively instead of crashing on .upper()."""
    for name in ("semantic", "SEMANTIC", "hybrid"):
        store = KSEVectorStore(kse_memory=kse, search_type=name)
        assert store.search_type is not None
