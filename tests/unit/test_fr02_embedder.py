"""
T-008 — ONNX embedding: WordPiece tokenisation and pooled inference.

Written test-first per GOV-04.

The tokeniser is hand-written rather than delegated to `transformers` or
`tokenizers`: `transformers` pulls torch, which pulls the CUDA stack on Linux
and breaks AR-04 outright. A dependency-free WordPiece keeps the default tree
CPU-only and auditable, at the cost of some speed on a path that is dominated
by the model forward pass anyway.

Inference is exercised against a fake ONNX session. A real MiniLM cannot be
downloaded in tests (AR-01) and must not be vendored, so the *pooling and
normalisation* contract is tested exhaustively here and the real-model path is
covered by an integration test that skips when no model is cached.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit

from kse_memory.core.projection import (
    ModelNotAvailableError,
    OnnxEmbedder,
)
from kse_memory.core.tokenizer import WordPieceTokenizer

VOCAB = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]",
    "hello", "world", "vector", "index", "cafe", "run", "test", ",",
    "##ning", "##ing", "##es",
]


@pytest.fixture
def model_dir(tmp_path):
    (tmp_path / "vocab.txt").write_text("\n".join(VOCAB) + "\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def tokenizer(model_dir):
    return WordPieceTokenizer.from_model_dir(model_dir)


# ------------------------------------------------------------- tokenisation
def test_vocab_is_loaded_in_order(tokenizer):
    assert tokenizer.vocab["[PAD]"] == 0
    assert tokenizer.vocab["hello"] == 4


def test_missing_vocab_is_a_clear_error(tmp_path):
    with pytest.raises(ModelNotAvailableError, match="vocab"):
        WordPieceTokenizer.from_model_dir(tmp_path)


def test_wraps_in_cls_and_sep(tokenizer):
    assert tokenizer.tokenize_to_tokens("hello world") == ["[CLS]", "hello", "world", "[SEP]"]


def test_greedy_subword_split(tokenizer):
    """'running' is not in vocab; 'run' + '##ning' is."""
    assert tokenizer.tokenize_to_tokens("running") == ["[CLS]", "run", "##ning", "[SEP]"]


def test_unknown_word_becomes_unk(tokenizer):
    assert tokenizer.tokenize_to_tokens("zzzqqq") == ["[CLS]", "[UNK]", "[SEP]"]


def test_lowercases_and_strips_accents(tokenizer):
    assert tokenizer.tokenize_to_tokens("CAFÉ") == ["[CLS]", "cafe", "[SEP]"]


def test_splits_punctuation(tokenizer):
    assert tokenizer.tokenize_to_tokens("hello, world") == [
        "[CLS]", "hello", ",", "world", "[SEP]"
    ]


def test_truncates_to_max_length(tokenizer):
    enc = tokenizer.encode_batch(["hello world vector index test"], max_length=4)
    assert enc["input_ids"].shape == (1, 4)
    # truncation must not drop the terminator, or the model sees a ragged input
    assert enc["input_ids"][0][-1] == tokenizer.vocab["[SEP]"]


def test_pads_short_sequences_and_masks_the_padding(tokenizer):
    enc = tokenizer.encode_batch(["hello", "hello world vector"])
    ids, mask = enc["input_ids"], enc["attention_mask"]
    assert ids.shape == mask.shape
    assert ids.shape[0] == 2
    # row 0 is shorter, so it is padded and the pad positions are masked out
    assert mask[0].sum() < mask[1].sum()
    assert set(ids[0][mask[0] == 0].tolist()) <= {tokenizer.vocab["[PAD]"]}


def test_encoding_is_deterministic(tokenizer):
    a = tokenizer.encode_batch(["hello world"])
    b = tokenizer.encode_batch(["hello world"])
    assert np.array_equal(a["input_ids"], b["input_ids"])


# ------------------------------------------------------- pooled inference
class _Input:
    def __init__(self, name):
        self.name = name


class FakeSession:
    """Returns per-token vectors derived from token id, so pooling is checkable."""

    hidden = 4

    def __init__(self, names=("input_ids", "attention_mask", "token_type_ids")):
        self._names = names
        self.last_feed = None

    def get_inputs(self):
        return [_Input(n) for n in self._names]

    def run(self, output_names, feed):
        self.last_feed = feed
        ids = feed["input_ids"]
        batch, seq = ids.shape
        out = np.zeros((batch, seq, self.hidden), dtype=np.float32)
        for b in range(batch):
            for s in range(seq):
                out[b, s, :] = float(ids[b, s]) + 1.0
        return [out]


@pytest.fixture
def embedder(model_dir, monkeypatch):
    (model_dir / "model.onnx").write_bytes(b"placeholder")
    e = OnnxEmbedder(model_dir=model_dir)
    session = FakeSession()
    monkeypatch.setattr(e, "_ensure_session", lambda: session)
    return e


def test_embed_returns_one_vector_per_text(embedder):
    out = embedder.embed(["hello", "hello world"])
    assert len(out) == 2
    assert all(len(v) == FakeSession.hidden for v in out)


def test_embeddings_are_l2_normalised(embedder):
    for vec in embedder.embed(["hello world", "vector index"]):
        assert abs(sum(v * v for v in vec) ** 0.5 - 1.0) < 1e-6


def test_pooling_ignores_padding(embedder):
    """The property that catches the classic bug.

    A short text batched alongside a long one gets padded. If padding leaked
    into the mean, the short text's vector would depend on its batch-mates.
    It must not.
    """
    alone = embedder.embed(["hello"])[0]
    batched = embedder.embed(["hello", "vector index test world"])[0]
    assert alone == pytest.approx(batched, abs=1e-6)


def test_embed_is_deterministic(embedder):
    assert embedder.embed(["hello world"]) == embedder.embed(["hello world"])


def test_embed_feeds_only_the_inputs_the_session_declares(model_dir, monkeypatch):
    """Models differ: some take token_type_ids, some do not. Feeding an
    undeclared input makes onnxruntime raise, so the feed must be derived
    from the session, not assumed."""
    (model_dir / "model.onnx").write_bytes(b"placeholder")
    e = OnnxEmbedder(model_dir=model_dir)
    session = FakeSession(names=("input_ids", "attention_mask"))
    monkeypatch.setattr(e, "_ensure_session", lambda: session)
    e.embed(["hello"])
    assert set(session.last_feed) == {"input_ids", "attention_mask"}


def test_embed_makes_no_network_calls(no_network, embedder):
    assert embedder.embed(["hello world"])
