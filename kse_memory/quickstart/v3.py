"""
The v3 quickstart: ingest -> project -> dense retrieval, offline (TC-02 partial).

What this demonstrates, honestly:
- FR-01/FR-02 for real: records normalised, projected under a user schema by a
  local embedder, written incrementally to in-memory stores. Re-running against
  the same pipeline writes nothing — the replay identity at work.
- HYBRID retrieval, earned and honest about itself: FR-03 parses the query,
  FR-04 runs the three channels concurrently, FR-05 fuses with RRF, and
  FR-07 gates the fusion on corroboration — a low-confidence fusion falls
  back to the dense ranking WITH an explicit flag. Every result carries its
  per-dimension scores and per-channel ranks as receipts.

No API key, no network call, no CUDA — the default path promises of TC-02.

Guardrails honoured: AR-01, AR-04, AR-05.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..core.dimension_store import InMemoryDimensionStore
from ..core.pipeline import IngestPipeline
from ..core.answer import HybridAnswer
from ..core.explain import Explanation, explain_results
from ..core.fusion import FusedItem
from ..core.query import ParsedQuery
from ..services.hybrid import HybridSearchService
from ..core.schema import DimensionSchema, load_schema

__all__ = [
    "DEFAULT_QUERIES",
    "DEFAULT_RECORDS",
    "DEFAULT_SCHEMA",
    "Hit",
    "QuickstartResult",
    "run_quickstart",
]

#: Demo schema. Deliberately domain-neutral: dimensions describe *material*,
#: not merchandise, so the demo cannot smuggle the retired retail vocabulary
#: back into the default path (TC-04).
DEFAULT_SCHEMA: Dict[str, Any] = {
    "name": "quickstart-docs",
    "version": "1.0.0",
    "dimensions": [
        {
            "name": "technical_depth",
            "description": "How technical the material is",
            "anchors": [
                "dense technical specification with precise terminology",
                "implementation detail aimed at engineers",
            ],
        },
        {
            "name": "practicality",
            "description": "How directly actionable it is",
            "anchors": [
                "step by step instructions you can follow immediately",
                "a worked example with commands to run",
            ],
        },
        {
            "name": "novelty",
            "description": "How new the ideas are",
            "anchors": [
                "a newly proposed method not seen before",
                "original research findings",
            ],
        },
    ],
}

#: Small mixed corpus — enough to make ranking and receipts visible.
DEFAULT_RECORDS: List[Dict[str, Any]] = [
    {"title": "HNSW index tuning guide", "description": "Practical walkthrough of ef_search and M parameters for approximate nearest neighbour indexes, with benchmarks to run.", "tags": ["ann", "vectors", "howto"]},
    {"title": "Why cosine similarity works", "description": "An intuitive explanation of angular distance for text embeddings, aimed at newcomers.", "tags": ["embeddings", "intro"]},
    {"title": "A novel fusion operator for hybrid retrieval", "description": "We propose a rank-based fusion method combining dense, sparse and graph channels, with proofs.", "tags": ["research", "fusion"]},
    {"title": "Deploying models on CPU-only hardware", "description": "Step by step: quantise to ONNX int8, cache locally, serve with zero GPU dependencies.", "tags": ["onnx", "deployment", "howto"]},
    {"title": "The history of information retrieval", "description": "From card catalogues to neural rankers — a survey of how search evolved.", "tags": ["survey", "history"]},
    {"title": "Debugging a non-deterministic hash", "description": "A war story: object memory addresses leaking into content hashes, and the failing test that caught it.", "tags": ["testing", "war-story"]},
]

DEFAULT_QUERIES: Tuple[str, ...] = (
    "how do I tune a vector index",
    "new research on combining search results",
    "explain embeddings simply",
)


@dataclass(frozen=True)
class Hit:
    """One fused result with its receipts.

    ``similarity`` is the fused RRF score; ``channel_ranks`` shows where each
    channel placed the entity (None = channel did not return it) — the FR-06
    explanation surface, collected at fusion time.
    """

    entity_id: str
    title: str
    similarity: float
    scores: Mapping[str, float] = field(default_factory=dict)
    channel_ranks: Mapping[str, Optional[int]] = field(default_factory=dict)


@dataclass
class QuickstartResult:
    ingested: int
    written: int
    searches: Dict[str, List[Hit]]
    parses: Dict[str, ParsedQuery]
    explanations: Dict[str, Tuple[Explanation, ...]]
    answers: Dict[str, HybridAnswer]
    pipeline: IngestPipeline


class _VectorIndex:
    """Minimal in-memory vector store satisfying the pipeline's upsert side."""

    def __init__(self) -> None:
        self.rows: Dict[str, Tuple[List[float], Dict[str, Any]]] = {}

    async def upsert_vectors(self, vectors) -> bool:
        for entity_id, vector, metadata in vectors:
            self.rows[entity_id] = (list(vector), dict(metadata))
        return True

    async def search_vectors(self, query_vector, top_k: int = 10, filters=None):
        """The VectorStoreInterface search contract, over in-memory rows."""
        from ..core.projection import vector_cosine

        ranked = sorted(
            ((eid, vector_cosine(query_vector, vec), meta)
             for eid, (vec, meta) in self.rows.items()),
            key=lambda r: r[1],
            reverse=True,
        )
        return ranked[:top_k]


class _GraphStore:
    """Minimal in-memory graph store for the incremental upsert."""

    def __init__(self) -> None:
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.relationships: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    async def create_node(self, node_id, labels, properties) -> bool:
        self.nodes[node_id] = {"labels": list(labels), "properties": dict(properties)}
        return True

    async def update_node(self, node_id, properties) -> bool:
        self.nodes.setdefault(node_id, {"labels": [], "properties": {}})
        self.nodes[node_id]["properties"].update(properties)
        return True

    async def get_node(self, node_id):
        return self.nodes.get(node_id)

    async def get_neighbors(self, node_id, relationship_types=None):
        # Neighbours are connected-in-either-direction, matching every graph
        # backend's own traversal semantics — FR-04's graph channel walks
        # dimension -> entity against edges written entity -> dimension.
        out = []
        for (source, target, rel) in self.relationships:
            if relationship_types is not None and rel not in relationship_types:
                continue
            if source == node_id:
                out.append({"id": target})
            elif target == node_id:
                out.append({"id": source})
        return out

    async def create_relationship(self, source_id, target_id, relationship_type, properties=None) -> bool:
        self.relationships[(source_id, target_id, relationship_type)] = dict(properties or {})
        return True

    async def delete_relationship(self, source_id, target_id, relationship_type) -> bool:
        self.relationships.pop((source_id, target_id, relationship_type), None)
        return True


def build_pipeline(embedder, schema: Optional[DimensionSchema] = None) -> IngestPipeline:
    """An IngestPipeline over fresh in-memory stores."""
    return IngestPipeline(
        schema or load_schema(DEFAULT_SCHEMA),
        embedder,
        graph_store=_GraphStore(),
        vector_store=_VectorIndex(),
        concept_store=InMemoryDimensionStore(),
    )


async def run_quickstart(
    embedder,
    schema: Optional[Mapping[str, Any]] = None,
    records: Optional[Sequence[Mapping[str, Any]]] = None,
    queries: Optional[Sequence[str]] = None,
    top_k: int = 5,
    pipeline: Optional[IngestPipeline] = None,
) -> QuickstartResult:
    """Ingest the corpus and run dense searches with dimension receipts.

    Pass ``pipeline`` from a previous result to demonstrate incrementality:
    re-ingesting unchanged content writes nothing.
    """
    if pipeline is None:
        loaded = load_schema(dict(schema)) if schema else load_schema(DEFAULT_SCHEMA)
        pipeline = build_pipeline(embedder, loaded)

    results = await pipeline.ingest_many(list(records or DEFAULT_RECORDS))
    written = sum(1 for r in results if r.written)

    searches: Dict[str, List[Hit]] = {}
    parses: Dict[str, ParsedQuery] = {}
    explanations: Dict[str, Tuple[Explanation, ...]] = {}
    answers: Dict[str, HybridAnswer] = {}
    index: _VectorIndex = pipeline.vector_store
    concept_store = pipeline.concept_store

    # The demo runs the object applications run — or it proves nothing.
    service = HybridSearchService(
        pipeline.schema, pipeline.embedder,
        vector_store=index, concept_store=concept_store,
        graph_store=pipeline.graph_store, centroids=pipeline.centroids,
    )

    for query in queries or DEFAULT_QUERIES:
        response = await service.search(query, top_k=top_k)
        parses[query] = response.parsed
        answers[query] = response.answer
        explanations[query] = response.explanations

        hits: List[Hit] = []
        for item, explanation in zip(response.answer.items, response.explanations):
            hits.append(
                Hit(
                    entity_id=item.entity_id,
                    title=response.titles.get(item.entity_id, item.entity_id),
                    similarity=round(item.fused, 6),
                    scores={row.name: row.score for row in explanation.dimensions},
                    channel_ranks=dict(item.ranks),
                )
            )
        searches[query] = hits

    return QuickstartResult(
        ingested=len(results),
        written=written,
        searches=searches,
        parses=parses,
        explanations=explanations,
        answers=answers,
        pipeline=pipeline,
    )
