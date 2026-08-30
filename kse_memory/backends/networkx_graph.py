"""
NetworkX graph store — the D-06 Tier-1 in-process backend (US9).

A genuinely different implementation from MemoryGraphStore (real graph
library, real traversal algorithms) passing the same conformance suite —
which is the US9 pluggability claim demonstrated rather than asserted.
networkx is already a default dependency; no service, no key (TC-02).

Semantics pinned by the behavioural conformance suite: neighbours are
connected-in-either-direction (FR-04's traversal contract), relationship
type filtering, update merges properties.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import networkx as nx

__all__ = ["NetworkXGraphStore"]


class NetworkXGraphStore:
    """GraphStoreInterface over a ``networkx.MultiDiGraph``."""

    def __init__(self, config: Any = None) -> None:
        self.config = config
        self.graph = nx.MultiDiGraph()
        self._connected = False

    async def connect(self) -> bool:
        self._connected = True
        return True

    async def disconnect(self) -> bool:
        self._connected = False
        return True

    async def create_node(self, node_id: str, labels: Sequence[str], properties: Dict[str, Any]) -> bool:
        self.graph.add_node(node_id, labels=list(labels), properties=dict(properties))
        return True

    async def update_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        if node_id not in self.graph:
            self.graph.add_node(node_id, labels=[], properties={})
        self.graph.nodes[node_id].setdefault("properties", {}).update(properties)
        return True

    async def delete_node(self, node_id: str) -> bool:
        if node_id not in self.graph:
            return False
        self.graph.remove_node(node_id)  # networkx drops incident edges itself
        return True

    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        if node_id not in self.graph:
            return None
        data = self.graph.nodes[node_id]
        return {"labels": list(data.get("labels", [])), "properties": dict(data.get("properties", {}))}

    async def get_neighbors(
        self, node_id: str, relationship_types: Optional[Sequence[str]] = None
    ) -> List[Dict[str, Any]]:
        if node_id not in self.graph:
            return []
        out: List[Dict[str, Any]] = []
        seen = set()
        for _, target, key in self.graph.out_edges(node_id, keys=True):
            if (relationship_types is None or key in relationship_types) and target not in seen:
                seen.add(target)
                out.append({"id": target})
        for source, _, key in self.graph.in_edges(node_id, keys=True):
            if (relationship_types is None or key in relationship_types) and source not in seen:
                seen.add(source)
                out.append({"id": source})
        return out

    async def create_relationship(
        self, source_id: str, target_id: str, relationship_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        # keyed by type: one edge per (source, target, type), upsert semantics
        self.graph.add_edge(source_id, target_id, key=relationship_type,
                            properties=dict(properties or {}))
        return True

    async def delete_relationship(self, source_id: str, target_id: str, relationship_type: str) -> bool:
        if self.graph.has_edge(source_id, target_id, key=relationship_type):
            self.graph.remove_edge(source_id, target_id, key=relationship_type)
            return True
        return False

    async def find_path(self, source_id: str, target_id: str, max_depth: int = 3) -> Optional[List[Dict[str, Any]]]:
        if source_id not in self.graph or target_id not in self.graph:
            return None
        try:
            path = nx.shortest_path(self.graph.to_undirected(as_view=True), source_id, target_id)
        except nx.NetworkXNoPath:
            return None
        if len(path) - 1 > max_depth:
            return None
        return [{"id": node} for node in path]

    async def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError(
            "NetworkXGraphStore has no query language; use the typed methods."
        )
