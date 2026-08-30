"""
US6 — domain packs: retail, finance, documents (TC-06).

Each pack ships a dimension schema (YAML), a small corpus, and one showcase
query engineered to demonstrate the legitimate hybrid mechanism: the schema's
anchors bridge a vocabulary gap between how the query is phrased and how the
right document is written, so the conceptual channel finds what pure vector
search ranks worse. The notebooks are thin callers over this module — the
same code the tests execute, so a green test means a runnable notebook.

Offline by construction (AR-01): in-memory stores, the locally cached model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

PACKS_DIR = Path(__file__).parent
PACKS: Tuple[str, ...] = ("retail", "finance", "documents")


@dataclass(frozen=True)
class Showcase:
    """The engineered demonstration: one query, one intended winner."""

    query: str
    target_id: str
    why: str  # the mechanism, stated — a demo that can't explain itself is a trick


@dataclass(frozen=True)
class Pack:
    name: str
    schema_path: Path
    notebook_path: Path
    records: List[Dict[str, Any]]
    showcase: Showcase


@dataclass(frozen=True)
class ShowcaseOutcome:
    query: str
    target_id: str
    dense_rank: int    # 1-based rank of the target under pure vector search
    hybrid_rank: int   # 1-based rank under the full hybrid answer
    dense_top: str
    hybrid_top: str


def load_pack(name: str) -> Pack:
    if name not in PACKS:
        raise KeyError(f"unknown pack {name!r}; available: {PACKS}")
    root = PACKS_DIR / name
    manifest = json.loads((root / "corpus.json").read_text(encoding="utf-8"))
    return Pack(
        name=name,
        schema_path=root / "schema.yaml",
        notebook_path=root / f"{name}_pack.ipynb",
        records=manifest["records"],
        showcase=Showcase(**manifest["showcase"]),
    )


async def run_showcase(pack: Pack, embedder: Any, top_k: int = 10) -> ShowcaseOutcome:
    """Ingest the pack and rank the showcase query, dense vs hybrid."""
    from kse_memory.core.query import parse_query
    from kse_memory.core.schema import load_schema
    from kse_memory.quickstart.v3 import build_pipeline
    from kse_memory.services.hybrid import HybridSearchService

    schema = load_schema(pack.schema_path)
    pipeline = build_pipeline(embedder, schema)
    await pipeline.ingest_many(pack.records)

    parsed = parse_query(pack.showcase.query, schema, embedder, centroids=pipeline.centroids)
    dense_rows = await pipeline.vector_store.search_vectors(list(parsed.vector), top_k=top_k)
    dense_ids = [row[0] for row in dense_rows]

    service = HybridSearchService(
        schema, embedder,
        vector_store=pipeline.vector_store,
        concept_store=pipeline.concept_store,
        graph_store=pipeline.graph_store,
        centroids=pipeline.centroids,
    )
    response = await service.search(pack.showcase.query, top_k=top_k)
    hybrid_ids = [item.entity_id for item in response.answer.items]

    def rank_of(ids: List[str]) -> int:
        return ids.index(pack.showcase.target_id) + 1 if pack.showcase.target_id in ids else top_k + 1

    return ShowcaseOutcome(
        query=pack.showcase.query,
        target_id=pack.showcase.target_id,
        dense_rank=rank_of(dense_ids),
        hybrid_rank=rank_of(hybrid_ids),
        dense_top=dense_ids[0] if dense_ids else "",
        hybrid_top=hybrid_ids[0] if hybrid_ids else "",
    )
