"""
Temporal knowledge graph implementation for KSE Memory SDK
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
from collections import defaultdict
import networkx as nx

from ..core.models import Product, ConceptualDimensions
from .temporal_models import (
    TemporalKnowledgeItem, TemporalRelationship, TemporalQuery,
    TemporalEvent, TimeInterval, TemporalPattern
)


logger = logging.getLogger(__name__)


@dataclass
class TemporalGraphNode:
    """Node in temporal knowledge graph"""
    
    entity_id: str
    entity_type: str
    properties: Dict[str, Any]
    
    # Temporal information
    creation_time: datetime
    last_modified: datetime
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    
    # Versioning
    version: int = 1
    previous_versions: List[str] = field(default_factory=list)
    
    # Temporal embeddings
    temporal_embedding: Optional[torch.Tensor] = None
    time_encoding: Optional[torch.Tensor] = None
    
    def is_valid_at(self, timestamp: datetime) -> bool:
        """Check if node is valid at given timestamp"""
        if self.valid_from and timestamp < self.valid_from:
            return False
        if self.valid_to and timestamp > self.valid_to:
            return False
        return True
    
    def get_temporal_features(self) -> torch.Tensor:
        """Get temporal features for this node"""
        features = []
        
        # Time since creation (normalized)
        time_since_creation = (datetime.now() - self.creation_time).total_seconds() / 86400  # days
        features.append(time_since_creation)
        
        # Time since last modification
        time_since_modified = (datetime.now() - self.last_modified).total_seconds() / 86400
        features.append(time_since_modified)
        
        # Validity duration
        if self.valid_from and self.valid_to:
            validity_duration = (self.valid_to - self.valid_from).total_seconds() / 86400
        else:
            validity_duration = 0.0
        features.append(validity_duration)
        
        # Version information
        features.append(float(self.version))
        features.append(float(len(self.previous_versions)))
        
        return torch.tensor(features, dtype=torch.float32)


@dataclass
class TemporalGraphEdge:
    """Edge in temporal knowledge graph"""
    
    source_id: str
    target_id: str
    relation_type: str
    properties: Dict[str, Any]
    
    # Temporal information
    start_time: datetime
    end_time: Optional[datetime] = None
    confidence: float = 1.0
    
    # Temporal patterns
    is_recurring: bool = False
    recurrence_pattern: Optional[str] = None  # "daily", "weekly", "monthly", etc.
    
    # Causal information
    is_causal: bool = False
    causal_delay: Optional[timedelta] = None
    
    def is_active_at(self, timestamp: datetime) -> bool:
        """Check if edge is active at given timestamp"""
        if timestamp < self.start_time:
            return False
        if self.end_time and timestamp > self.end_time:
            return False
        return True
    
    def get_duration(self) -> Optional[timedelta]:
        """Get duration of the relationship"""
        if self.end_time:
            return self.end_time - self.start_time
        return None
    
    def overlaps_with(self, other: 'TemporalGraphEdge') -> bool:
        """Check if this edge overlaps temporally with another"""
        # Check if intervals overlap
        start1, end1 = self.start_time, self.end_time or datetime.max
        start2, end2 = other.start_time, other.end_time or datetime.max
        
        return start1 <= end2 and start2 <= end1


class TemporalKnowledgeGraph:
    """Temporal knowledge graph for KSE Memory"""
    
    def __init__(self, time_encoding_dim: int = 64):
        self.time_encoding_dim = time_encoding_dim
        
        # Graph storage
        self.nodes: Dict[str, TemporalGraphNode] = {}
        self.edges: Dict[str, TemporalGraphEdge] = {}
        self.temporal_index: Dict[datetime, Set[str]] = defaultdict(set)
        
        # NetworkX graph for analysis
        self.graph = nx.MultiDiGraph()
        
        # Time encoding
        self.time_encoder = Time2VecEncoder(time_encoding_dim)
        
        # Temporal patterns
        self.temporal_patterns: List[TemporalPattern] = []
        self.pattern_index: Dict[str, List[TemporalPattern]] = defaultdict(list)
        
        # Statistics
        self.stats = {
            "total_nodes": 0,
            "total_edges": 0,
            "temporal_events": 0,
            "active_patterns": 0
        }
    
    def add_temporal_node(self, entity_id: str, entity_type: str,
                         properties: Dict[str, Any],
                         timestamp: Optional[datetime] = None,
                         valid_from: Optional[datetime] = None,
                         valid_to: Optional[datetime] = None) -> TemporalGraphNode:
        """Add a temporal node to the graph"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Create temporal embedding
        temporal_embedding = self._create_temporal_embedding(properties, timestamp)
        time_encoding = self.time_encoder.encode(timestamp)
        
        node = TemporalGraphNode(
            entity_id=entity_id,
            entity_type=entity_type,
            properties=properties,
            creation_time=timestamp,
            last_modified=timestamp,
            valid_from=valid_from,
            valid_to=valid_to,
            temporal_embedding=temporal_embedding,
            time_encoding=time_encoding
        )
        
        # Store node
        self.nodes[entity_id] = node
        self.temporal_index[timestamp].add(entity_id)
        
        # Add to NetworkX graph
        self.graph.add_node(entity_id, **properties, timestamp=timestamp)
        
        self.stats["total_nodes"] += 1
        logger.info(f"Added temporal node: {entity_id} at {timestamp}")
        
        return node
    
    def add_temporal_edge(self, source_id: str, target_id: str,
                         relation_type: str, properties: Dict[str, Any],
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None,
                         confidence: float = 1.0,
                         is_causal: bool = False,
                         causal_delay: Optional[timedelta] = None) -> TemporalGraphEdge:
        """Add a temporal edge to the graph"""
        if start_time is None:
            start_time = datetime.now()
        
        edge_id = f"{source_id}_{target_id}_{relation_type}_{start_time.isoformat()}"
        
        edge = TemporalGraphEdge(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            properties=properties,
            start_time=start_time,
            end_time=end_time,
            confidence=confidence,
            is_causal=is_causal,
            causal_delay=causal_delay
        )
        
        # Store edge
        self.edges[edge_id] = edge
        self.temporal_index[start_time].add(edge_id)
        
        # Add to NetworkX graph
        self.graph.add_edge(
            source_id, target_id,
            key=edge_id,
            relation_type=relation_type,
            start_time=start_time,
            end_time=end_time,
            confidence=confidence,
            **properties
        )
        
        self.stats["total_edges"] += 1
        logger.info(f"Added temporal edge: {source_id} -> {target_id} ({relation_type}) at {start_time}")
        
        return edge
    
    def query_temporal_neighborhood(self, entity_id: str,
                                  timestamp: datetime,
                                  max_hops: int = 2,
                                  relation_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Query temporal neighborhood of an entity at a specific time"""
        if entity_id not in self.nodes:
            return {"nodes": [], "edges": [], "timestamp": timestamp}
        
        # Find nodes and edges valid at timestamp
        valid_nodes = []
        valid_edges = []
        visited = set()
        queue = [(entity_id, 0)]
        
        while queue:
            current_id, hops = queue.pop(0)
            
            if current_id in visited or hops > max_hops:
                continue
            
            visited.add(current_id)
            
            # Check if node is valid at timestamp
            if current_id in self.nodes:
                node = self.nodes[current_id]
                if node.is_valid_at(timestamp):
                    valid_nodes.append(node)
            
            # Find connected edges
            for edge_id, edge in self.edges.items():
                if edge.source_id == current_id or edge.target_id == current_id:
                    if edge.is_active_at(timestamp):
                        if relation_types is None or edge.relation_type in relation_types:
                            valid_edges.append(edge)
                            
                            # Add connected node to queue
                            next_id = edge.target_id if edge.source_id == current_id else edge.source_id
                            if next_id not in visited:
                                queue.append((next_id, hops + 1))
        
        return {
            "nodes": valid_nodes,
            "edges": valid_edges,
            "timestamp": timestamp,
            "neighborhood_size": len(valid_nodes)
        }
    
    def find_temporal_paths(self, source_id: str, target_id: str,
                           start_time: datetime, end_time: datetime,
                           max_path_length: int = 5) -> List[List[TemporalGraphEdge]]:
        """Find temporal paths between two entities within a time window"""
        paths = []
        
        def dfs_temporal_path(current_id: str, path: List[TemporalGraphEdge],
                            current_time: datetime, visited: Set[str]):
            if len(path) > max_path_length:
                return
            
            if current_id == target_id and current_time <= end_time:
                paths.append(path.copy())
                return
            
            # Find outgoing edges from current node
            for edge_id, edge in self.edges.items():
                if (edge.source_id == current_id and 
                    edge.target_id not in visited and
                    edge.start_time >= current_time and
                    edge.start_time <= end_time):
                    
                    visited.add(edge.target_id)
                    path.append(edge)
                    
                    # Continue DFS with edge end time
                    next_time = edge.end_time or end_time
                    dfs_temporal_path(edge.target_id, path, next_time, visited)
                    
                    path.pop()
                    visited.remove(edge.target_id)
        
        dfs_temporal_path(source_id, [], start_time, {source_id})
        return paths
    
    def detect_temporal_patterns(self, pattern_type: str = "recurring",
                               min_support: int = 3,
                               time_window: timedelta = timedelta(days=30)) -> List[TemporalPattern]:
        """Detect temporal patterns in the graph"""
        patterns = []
        
        if pattern_type == "recurring":
            patterns.extend(self._detect_recurring_patterns(min_support, time_window))
        elif pattern_type == "causal":
            patterns.extend(self._detect_causal_patterns(min_support, time_window))
        elif pattern_type == "seasonal":
            patterns.extend(self._detect_seasonal_patterns(min_support))
        
        # Store patterns
        for pattern in patterns:
            self.temporal_patterns.append(pattern)
            self.pattern_index[pattern.pattern_type].append(pattern)
        
        self.stats["active_patterns"] = len(self.temporal_patterns)
        return patterns
    
    def _detect_recurring_patterns(self, min_support: int,
                                 time_window: timedelta) -> List[TemporalPattern]:
        """Detect recurring temporal patterns"""
        patterns = []
        
        # Group edges by relation type
        relation_groups = defaultdict(list)
        for edge in self.edges.values():
            relation_groups[edge.relation_type].append(edge)
        
        for relation_type, edges in relation_groups.items():
            if len(edges) < min_support:
                continue
            
            # Sort edges by start time
            edges.sort(key=lambda e: e.start_time)
            
            # Find recurring intervals
            intervals = []
            for i in range(len(edges) - 1):
                interval = edges[i + 1].start_time - edges[i].start_time
                intervals.append(interval)
            
            # Check for consistent intervals
            if len(intervals) >= min_support - 1:
                avg_interval = sum(intervals, timedelta()) / len(intervals)
                
                # Check if intervals are consistent (within 10% variance)
                variance = sum((interval - avg_interval).total_seconds() ** 2 for interval in intervals) / len(intervals)
                std_dev = variance ** 0.5
                
                if std_dev < avg_interval.total_seconds() * 0.1:  # 10% variance threshold
                    pattern = TemporalPattern(
                        pattern_id=f"recurring_{relation_type}_{len(patterns)}",
                        pattern_type="recurring",
                        entities=[edge.source_id for edge in edges[:min_support]],
                        relations=[relation_type],
                        time_intervals=[TimeInterval(
                            start=edges[0].start_time,
                            end=edges[-1].start_time,
                            duration=avg_interval
                        )],
                        confidence=1.0 - (std_dev / avg_interval.total_seconds()),
                        support=len(edges),
                        properties={
                            "average_interval": avg_interval.total_seconds(),
                            "variance": variance,
                            "relation_type": relation_type
                        }
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_causal_patterns(self, min_support: int,
                              time_window: timedelta) -> List[TemporalPattern]:
        """Detect causal temporal patterns"""
        patterns = []
        
        # Find potential causal relationships
        causal_candidates = []
        
        for edge1_id, edge1 in self.edges.items():
            for edge2_id, edge2 in self.edges.items():
                if edge1_id != edge2_id:
                    # Check temporal ordering
                    if edge1.start_time < edge2.start_time:
                        delay = edge2.start_time - edge1.start_time
                        
                        # Check if delay is within reasonable bounds
                        if delay <= time_window:
                            causal_candidates.append((edge1, edge2, delay))
        
        # Group by delay patterns
        delay_groups = defaultdict(list)
        for edge1, edge2, delay in causal_candidates:
            delay_key = int(delay.total_seconds() / 3600)  # Group by hour
            delay_groups[delay_key].append((edge1, edge2, delay))
        
        # Find patterns with sufficient support
        for delay_key, candidates in delay_groups.items():
            if len(candidates) >= min_support:
                # Extract common relation types
                relation_pairs = defaultdict(int)
                for edge1, edge2, _ in candidates:
                    pair = (edge1.relation_type, edge2.relation_type)
                    relation_pairs[pair] += 1
                
                for (rel1, rel2), count in relation_pairs.items():
                    if count >= min_support:
                        avg_delay = sum(delay for _, _, delay in candidates) / len(candidates)
                        
                        pattern = TemporalPattern(
                            pattern_id=f"causal_{rel1}_{rel2}_{len(patterns)}",
                            pattern_type="causal",
                            entities=[],  # Would need more sophisticated entity extraction
                            relations=[rel1, rel2],
                            time_intervals=[TimeInterval(
                                start=min(edge1.start_time for edge1, _, _ in candidates),
                                end=max(edge2.start_time for _, edge2, _ in candidates),
                                duration=avg_delay
                            )],
                            confidence=count / len(candidates),
                            support=count,
                            properties={
                                "cause_relation": rel1,
                                "effect_relation": rel2,
                                "average_delay": avg_delay.total_seconds(),
                                "delay_variance": np.var([d.total_seconds() for _, _, d in candidates])
                            }
                        )
                        patterns.append(pattern)
        
        return patterns
    
    def _detect_seasonal_patterns(self, min_support: int) -> List[TemporalPattern]:
        """Detect seasonal temporal patterns"""
        patterns = []
        
        # Group events by time of day, day of week, month, etc.
        time_groups = {
            "hour": defaultdict(list),
            "day_of_week": defaultdict(list),
            "month": defaultdict(list)
        }
        
        for edge in self.edges.values():
            time_groups["hour"][edge.start_time.hour].append(edge)
            time_groups["day_of_week"][edge.start_time.weekday()].append(edge)
            time_groups["month"][edge.start_time.month].append(edge)
        
        for time_unit, groups in time_groups.items():
            for time_value, edges in groups.items():
                if len(edges) >= min_support:
                    # Check for relation type consistency
                    relation_counts = defaultdict(int)
                    for edge in edges:
                        relation_counts[edge.relation_type] += 1
                    
                    for relation_type, count in relation_counts.items():
                        if count >= min_support:
                            pattern = TemporalPattern(
                                pattern_id=f"seasonal_{time_unit}_{time_value}_{relation_type}_{len(patterns)}",
                                pattern_type="seasonal",
                                entities=[],
                                relations=[relation_type],
                                time_intervals=[],
                                confidence=count / len(edges),
                                support=count,
                                properties={
                                    "time_unit": time_unit,
                                    "time_value": time_value,
                                    "relation_type": relation_type,
                                    "seasonal_strength": count / len(edges)
                                }
                            )
                            patterns.append(pattern)
        
        return patterns
    
    def _create_temporal_embedding(self, properties: Dict[str, Any],
                                 timestamp: datetime) -> torch.Tensor:
        """Create temporal embedding for entity"""
        # Combine property features with temporal features
        property_features = []
        
        # Extract numerical features from properties
        for key, value in properties.items():
            if isinstance(value, (int, float)):
                property_features.append(float(value))
            elif isinstance(value, str):
                # Simple string hash feature
                property_features.append(float(hash(value) % 1000) / 1000.0)
        
        # Pad or truncate to fixed size
        target_size = 32
        if len(property_features) < target_size:
            property_features.extend([0.0] * (target_size - len(property_features)))
        else:
            property_features = property_features[:target_size]
        
        # Add temporal features
        time_encoding = self.time_encoder.encode(timestamp)
        
        # Combine features
        combined_features = torch.cat([
            torch.tensor(property_features, dtype=torch.float32),
            time_encoding
        ])
        
        return combined_features
    
    def get_temporal_statistics(self) -> Dict[str, Any]:
        """Get comprehensive temporal graph statistics"""
        now = datetime.now()
        
        # Node statistics
        active_nodes = sum(1 for node in self.nodes.values() if node.is_valid_at(now))
        
        # Edge statistics
        active_edges = sum(1 for edge in self.edges.values() if edge.is_active_at(now))
        
        # Temporal span
        all_times = []
        for node in self.nodes.values():
            all_times.append(node.creation_time)
            if node.valid_from:
                all_times.append(node.valid_from)
            if node.valid_to:
                all_times.append(node.valid_to)
        
        for edge in self.edges.values():
            all_times.append(edge.start_time)
            if edge.end_time:
                all_times.append(edge.end_time)
        
        temporal_span = max(all_times) - min(all_times) if all_times else timedelta()
        
        return {
            "total_nodes": len(self.nodes),
            "active_nodes": active_nodes,
            "total_edges": len(self.edges),
            "active_edges": active_edges,
            "temporal_patterns": len(self.temporal_patterns),
            "temporal_span_days": temporal_span.days,
            "graph_density": len(self.edges) / (len(self.nodes) * (len(self.nodes) - 1)) if len(self.nodes) > 1 else 0,
            "average_node_degree": 2 * len(self.edges) / len(self.nodes) if len(self.nodes) > 0 else 0,
            "pattern_types": list(self.pattern_index.keys()),
            "time_encoding_dimension": self.time_encoding_dim
        }
    
    def export_temporal_snapshot(self, timestamp: datetime) -> Dict[str, Any]:
        """Export a snapshot of the graph at a specific timestamp"""
        snapshot_nodes = []
        snapshot_edges = []
        
        for node in self.nodes.values():
            if node.is_valid_at(timestamp):
                snapshot_nodes.append({
                    "entity_id": node.entity_id,
                    "entity_type": node.entity_type,
                    "properties": node.properties,
                    "temporal_features": node.get_temporal_features().tolist()
                })
        
        for edge in self.edges.values():
            if edge.is_active_at(timestamp):
                snapshot_edges.append({
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "relation_type": edge.relation_type,
                    "properties": edge.properties,
                    "confidence": edge.confidence,
                    "duration": edge.get_duration().total_seconds() if edge.get_duration() else None
                })
        
        return {
            "timestamp": timestamp.isoformat(),
            "nodes": snapshot_nodes,
            "edges": snapshot_edges,
            "statistics": {
                "node_count": len(snapshot_nodes),
                "edge_count": len(snapshot_edges)
            }
        }


class Time2VecEncoder:
    """Time2Vec encoding for temporal features"""
    
    def __init__(self, encoding_dim: int = 64):
        self.encoding_dim = encoding_dim
        self.linear_weights = torch.randn(1, encoding_dim // 2)
        self.periodic_weights = torch.randn(1, encoding_dim // 2)
    
    def encode(self, timestamp: datetime) -> torch.Tensor:
        """Encode timestamp using Time2Vec"""
        # Convert timestamp to Unix timestamp
        unix_time = timestamp.timestamp()
        time_tensor = torch.tensor([[unix_time]], dtype=torch.float32)
        
        # Linear component
        linear_component = torch.matmul(time_tensor, self.linear_weights)
        
        # Periodic component
        periodic_component = torch.sin(torch.matmul(time_tensor, self.periodic_weights))
        
        # Concatenate components
        encoding = torch.cat([linear_component, periodic_component], dim=1)
        
        return encoding.squeeze(0)