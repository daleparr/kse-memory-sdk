"""
FR-04 — Retrieve concurrently: vector · conceptual · graph channels.

Design (BD3; criteria TC-02; decisions D-03, D-07):
- Three channels run under ``asyncio.gather``, each against whichever store is
  configured. The input is a :class:`ParsedQuery` (FR-03); nothing here embeds.
- Vector: the store's own ``search_vectors`` top-k.
- Conceptual: the query's dimension targets ARE a valid score vector under the
  schema — FR-02 and FR-03 share one geometry by construction — so this
  channel is ``find_similar_dimensions`` on the targets, directly.
- Graph: traversal from the query's strongest target dimensions to the
  entities scored on them (the SCORED_AS edges FR-02's upsert wrote), ranked
  by how many of those dimensions an entity touches, ties broken by id.
  Uses only the portable ``get_neighbors`` contract; neighbours are read as
  connected-in-either-direction, which is what "neighbours" means in every
  graph backend's own traversal semantics.
- Degradation is built in rather than bolted on (FR-07 groundwork): a missing
  store is an empty channel; a raising store is an empty channel plus an
  ``errors`` entry; healthy channels are never disturbed. FR-05 fuses these
  rankings; FR-07 adds the fused-confidence threshold and the explicit
  dense-only flag.

Guardrails honoured: AR-01 (no network of its own), AR-05 (typed surface).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .dimension_store import DimensionScores
from .projection import SCORED_AS, dimension_node_id
from .query import ParsedQuery

__all__ = ["RetrievalResult", "retrieve"]

#: How many of the query's strongest dimensions seed the graph traversal.
GRAPH_SEED_DIMENSIONS = 2


@dataclass(frozen=True)
class RetrievalResult:
    """Per-channel rankings, ready for FR-05 fusion.

    Each channel is an ordered tuple of ``(entity_id, score)``; scores are
    channel-local and NOT comparable across channels — that is precisely why
    fusion is rank-based (D-07).
    """

    vector: Tuple[Tuple[str, float], ...] = ()
    conceptual: Tuple[Tuple[str, float], ...] = ()
    graph: Tuple[Tuple[str, float], ...] = ()
    errors: Mapping[str, str] = field(default_factory=dict)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RetrievalResult):
            return NotImplemented
        return (
            self.vector == other.vector
            and self.conceptual == other.conceptual
            and self.graph == other.graph
            and dict(self.errors) == dict(other.errors)
        )


async def _vector_channel(parsed: ParsedQuery, store: Any, top_k: int) -> Tuple[Tuple[str, float], ...]:
    rows = await store.search_vectors(list(parsed.vector), top_k=top_k)
    return tuple((entity_id, float(score)) for entity_id, score, *_ in rows[:top_k])


async def _conceptual_channel(parsed: ParsedQuery, store: Any, top_k: int) -> Tuple[Tuple[str, float], ...]:
    targets = DimensionScores(
        schema_name=parsed.schema_name,
        schema_version=parsed.schema_version,
        scores=dict(parsed.targets),
    )
    hits = await store.find_similar_dimensions(targets, threshold=0.0, limit=top_k)
    return tuple((entity_id, float(score)) for entity_id, score in hits[:top_k])


async def _graph_channel(parsed: ParsedQuery, store: Any, top_k: int) -> Tuple[Tuple[str, float], ...]:
    seeds = sorted(parsed.targets, key=lambda name: (-parsed.targets[name], name))
    seeds = seeds[:GRAPH_SEED_DIMENSIONS]

    coverage: Dict[str, int] = {}
    for name in seeds:
        node = dimension_node_id(parsed.schema_name, name)
        for neighbour in await store.get_neighbors(node, [SCORED_AS]) or []:
            entity_id = neighbour.get("id") if isinstance(neighbour, Mapping) else neighbour
            if entity_id:
                coverage[entity_id] = coverage.get(entity_id, 0) + 1

    # A constant coverage function carries no rank information: emitting an
    # id-ordered list would inject alphabetical noise into fusion at full
    # channel weight (US6's pack lab caught exactly that). Abstain instead —
    # an empty channel is honest and FR-07 already prices it into confidence.
    if len(set(coverage.values())) <= 1:
        return ()

    ranked = sorted(coverage.items(), key=lambda item: (-item[1], item[0]))
    return tuple((entity_id, float(count)) for entity_id, count in ranked[:top_k])


async def retrieve(
    parsed: ParsedQuery,
    *,
    vector_store: Any = None,
    concept_store: Any = None,
    graph_store: Any = None,
    top_k: int = 10,
) -> RetrievalResult:
    """Run every configured channel concurrently and collect the rankings.

    A channel with no store returns empty. A channel whose store raises
    returns empty and files the failure under ``errors`` keyed by channel
    name — retrieval itself never raises for a backend fault, because a
    degraded answer with a flag beats no answer (FR-07).
    """
    channels = {
        "vector": _vector_channel(parsed, vector_store, top_k) if vector_store else None,
        "conceptual": _conceptual_channel(parsed, concept_store, top_k) if concept_store else None,
        "graph": _graph_channel(parsed, graph_store, top_k) if graph_store else None,
    }

    live = {name: coro for name, coro in channels.items() if coro is not None}
    names = list(live)
    outcomes = await asyncio.gather(*live.values(), return_exceptions=True)

    results: Dict[str, Tuple[Tuple[str, float], ...]] = {
        "vector": (), "conceptual": (), "graph": ()
    }
    errors: Dict[str, str] = {}
    for name, outcome in zip(names, outcomes):
        if isinstance(outcome, BaseException):
            errors[name] = str(outcome)
        else:
            results[name] = outcome

    return RetrievalResult(
        vector=results["vector"],
        conceptual=results["conceptual"],
        graph=results["graph"],
        errors=errors,
    )
