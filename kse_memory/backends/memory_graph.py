"""
In-process graph store: the zero-service default (TC-02, D-03).

Promoted from the quickstart's internal store once the conformance lane
(T-066) pinned its semantics — notably get_neighbors returning nodes
connected in EITHER direction, which FR-04's dimension->entity traversal
depends on. Process-local and non-durable by design; durable graphs are the
tiered backends' job (D-06).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

__all__ = ["MemoryGraphStore"]


class MemoryGraphStore:
    """Dict-backed graph store implementing the contract-as-used."""

    def __init__(self, config=None) -> None:
        self.config = config
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.relationships: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self._connected = False

    async def connect(self) -> bool:
        self._connected = True
        return True

    async def disconnect(self) -> bool:
        self._connected = False
        return True

    async def create_node(self, node_id, labels, properties) -> bool:
        self.nodes[node_id] = {"labels": list(labels), "properties": dict(properties)}
        return True

    async def update_node(self, node_id, properties) -> bool:
        self.nodes.setdefault(node_id, {"labels": [], "properties": {}})
        self.nodes[node_id]["properties"].update(properties)
        return True

    async def delete_node(self, node_id) -> bool:
        existed = self.nodes.pop(node_id, None) is not None
        self.relationships = {
            key: value for key, value in self.relationships.items()
            if node_id not in (key[0], key[1])
        }
        return existed

    async def get_node(self, node_id) -> Optional[Dict[str, Any]]:
        return self.nodes.get(node_id)

    async def get_neighbors(self, node_id, relationship_types=None):
        # Either direction: FR-04 walks dimension -> entity over edges
        # written entity -> dimension. Pinned by the conformance suite.
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
        return self.relationships.pop((source_id, target_id, relationship_type), None) is not None

    async def find_path(self, source_id, target_id, max_depth: int = 3):
        # Breadth-first over undirected edges; enough for the in-process tier.
        frontier = [[source_id]]
        seen = {source_id}
        while frontier:
            path = frontier.pop(0)
            if len(path) > max_depth + 1:
                continue
            for neighbour in await self.get_neighbors(path[-1]):
                node = neighbour["id"]
                if node == target_id:
                    return [{"id": n} for n in path + [node]]
                if node not in seen:
                    seen.add(node)
                    frontier.append(path + [node])
        return None

    async def execute_query(self, query, parameters=None):
        raise NotImplementedError(
            "MemoryGraphStore has no query language; use the typed methods."
        )
