"""
T-008 — WordPiece tokenisation for the default ONNX embedder.

Why hand-written (D-03, AR-04):
``transformers`` would give this for free, but it requires torch, and the
Linux torch wheel pulls the whole NVIDIA stack — the exact violation AR-04
exists to prevent. ``tokenizers`` avoids torch but adds a compiled dependency
to a tree we are deliberately keeping small. WordPiece is a short, fully
specified algorithm, so we implement it: zero new dependencies, deterministic,
and auditable line by line.

This is BERT-uncased preprocessing: NFD accent stripping, lowercasing,
punctuation splitting, then greedy longest-match-first subword lookup with
``##`` continuations.

Guardrails honoured: AR-01 (no network), AR-04 (no GPU dependency).
"""
from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

__all__ = ["WordPieceTokenizer"]

_PAD, _UNK, _CLS, _SEP = "[PAD]", "[UNK]", "[CLS]", "[SEP]"
_DEFAULT_MAX_LENGTH = 256


def _is_punctuation(char: str) -> bool:
    code = ord(char)
    if (33 <= code <= 47) or (58 <= code <= 64) or (91 <= code <= 96) or (123 <= code <= 126):
        return True
    return unicodedata.category(char).startswith("P")


class WordPieceTokenizer:
    """BERT-uncased WordPiece over a ``vocab.txt``.

    Token ids are line numbers in the vocabulary file, which is the format
    every MiniLM export ships.
    """

    def __init__(self, vocab: Dict[str, int]) -> None:
        for required in (_PAD, _UNK, _CLS, _SEP):
            if required not in vocab:
                raise ValueError(f"vocabulary is missing the {required} token")
        self.vocab = vocab
        self.pad_id = vocab[_PAD]
        self.unk_id = vocab[_UNK]
        self.cls_id = vocab[_CLS]
        self.sep_id = vocab[_SEP]

    @classmethod
    def from_model_dir(cls, model_dir: "str | Path") -> "WordPieceTokenizer":
        """Load ``vocab.txt`` from a cached model directory.

        Raises:
            ModelNotAvailableError: if the vocabulary is absent. Like the model
                itself this is never fetched — a missing vocabulary is a setup
                problem to be reported, not resolved over the network (AR-01).
        """
        from .projection import ModelNotAvailableError  # circular at import time

        path = Path(model_dir) / "vocab.txt"
        if not path.is_file():
            raise ModelNotAvailableError(
                f"tokeniser vocab not found at local path: {path}\n"
                "KSE never downloads model artefacts (AR-01). The vocab.txt "
                "shipped with the ONNX export belongs beside model.onnx."
            )
        vocab: Dict[str, int] = {}
        with path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                token = line.rstrip("\n")
                if token and token not in vocab:
                    vocab[token] = index
        return cls(vocab)

    # ---------------------------------------------------------------- pieces
    def _basic_tokenize(self, text: str) -> List[str]:
        text = unicodedata.normalize("NFD", text.strip())
        text = "".join(c for c in text if unicodedata.category(c) != "Mn")
        text = text.lower()

        tokens: List[str] = []
        current: List[str] = []
        for char in text:
            if char.isspace():
                if current:
                    tokens.append("".join(current))
                    current = []
            elif _is_punctuation(char):
                if current:
                    tokens.append("".join(current))
                    current = []
                tokens.append(char)
            else:
                current.append(char)
        if current:
            tokens.append("".join(current))
        return tokens

    def _wordpiece(self, word: str) -> List[str]:
        """Greedy longest-match-first. An unmatchable word is a single [UNK]."""
        if word in self.vocab:
            return [word]

        pieces: List[str] = []
        start = 0
        while start < len(word):
            end = len(word)
            match = None
            while start < end:
                candidate = word[start:end]
                if start > 0:
                    candidate = "##" + candidate
                if candidate in self.vocab:
                    match = candidate
                    break
                end -= 1
            if match is None:
                return [_UNK]  # partial matches are worse than none
            pieces.append(match)
            start = end
        return pieces

    def tokenize_to_tokens(self, text: str, max_length: int = _DEFAULT_MAX_LENGTH) -> List[str]:
        """Tokenise to strings, including the [CLS]/[SEP] terminators."""
        pieces: List[str] = []
        for word in self._basic_tokenize(text):
            pieces.extend(self._wordpiece(word))
        # reserve two slots so truncation can never cost us the terminators
        pieces = pieces[: max(0, max_length - 2)]
        return [_CLS] + pieces + [_SEP]

    def encode_batch(
        self, texts: Sequence[str], max_length: int = _DEFAULT_MAX_LENGTH
    ) -> Dict[str, np.ndarray]:
        """Encode a batch to padded ``int64`` arrays.

        Returns ``input_ids``, ``attention_mask`` and ``token_type_ids``. The
        mask is what lets pooling ignore padding; without it a short text's
        embedding would depend on its batch-mates.
        """
        rows = [
            [self.vocab.get(t, self.unk_id) for t in self.tokenize_to_tokens(t, max_length)]
            for t in texts
        ]
        width = max((len(r) for r in rows), default=2)

        input_ids = np.full((len(rows), width), self.pad_id, dtype=np.int64)
        attention_mask = np.zeros((len(rows), width), dtype=np.int64)
        for i, row in enumerate(rows):
            input_ids[i, : len(row)] = row
            attention_mask[i, : len(row)] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": np.zeros_like(input_ids),
        }
