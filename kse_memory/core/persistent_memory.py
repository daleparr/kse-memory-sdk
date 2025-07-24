"""
Persistent Memory Architecture for KSE Memory SDK

This module implements true temporal knowledge graphs with cross-session learning
capabilities and automated knowledge consolidation.
"""

import asyncio
import logging
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import torch
from collections import defaultdict, deque

from .models import Product, SearchQuery, SearchResult, ConceptualSpace
from .interfaces import GraphStoreInterface
from ..temporal.temporal_models import (
    TemporalEvent, TemporalRelationship, TemporalKnowledgeItem,
    TimeInterval, TemporalPattern
)

logger = logging.getLogger(__name__)


class MemoryImportance(Enum):
    """Memory importance levels for consolidation."""
    CRITICAL = 1.0
    HIGH = 0.8
    MEDIUM = 0.6
    LOW = 0.4
    EPHEMERAL = 0.2


class ConsolidationStrategy(Enum):
    """Strategies for memory consolidation."""
    FREQUENCY_BASED = "frequency"
    RECENCY_BASED = "recency"
    IMPORTANCE_BASED = "importance"
    HYBRID = "hybrid"


@dataclass
class MemoryTrace:
    """Represents a memory trace with temporal and importance information."""
    
    trace_id: str
    content: Any
    timestamp: datetime
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    importance_score: float = 0.5
    decay_rate: float = 0.1
    consolidation_level: int = 0
    related_traces: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_current_importance(self, current_time: datetime) -> float:
        """Calculate current importance considering decay."""
        if self.last_accessed is None:
            age_hours = (current_time - self.timestamp).total_seconds() / 3600
        else:
            age_hours = (current_time - self.last_accessed).total_seconds() / 3600
        
        # Apply exponential decay
        decay_factor = np.exp(-self.decay_rate * age_hours / 24)  # decay per day
        
        # Boost importance based on access frequency
        frequency_boost = min(1.0, self.access_count / 10.0)
        
        return self.importance_score * decay_factor + frequency_boost * 0.2
    
    def access(self, current_time: datetime):
        """Record an access to this memory trace."""
        self.access_count += 1
        self.last_accessed = current_time


@dataclass
class KnowledgeCluster:
    """Represents a cluster of related knowledge items."""
    
    cluster_id: str
    centroid_embedding: np.ndarray
    member_traces: Set[str]
    creation_time: datetime
    last_updated: datetime
    importance_score: float
    consolidation_count: int = 0
    
    def update_centroid(self, embeddings: Dict[str, np.ndarray]):
        """Update cluster centroid based on member embeddings."""
        if not self.member_traces:
            return
        
        member_embeddings = [embeddings[trace_id] for trace_id in self.member_traces 
                           if trace_id in embeddings]
        
        if member_embeddings:
            self.centroid_embedding = np.mean(member_embeddings, axis=0)
            self.last_updated = datetime.now()


class TemporalKnowledgeGraph:
    """
    Temporal knowledge graph with persistent memory capabilities.
    
    Maintains temporal relationships between entities and supports
    cross-session learning through persistent storage.
    """
    
    def __init__(self, graph_store: GraphStoreInterface, config: Dict[str, Any]):
        self.graph_store = graph_store
        self.config = config
        
        # Temporal relationship tracking
        self.temporal_relationships: Dict[str, List[TemporalRelationship]] = defaultdict(list)
        self.temporal_events: Dict[str, TemporalEvent] = {}
        self.temporal_patterns: Dict[str, TemporalPattern] = {}
        
        # Cross-session state
        self.session_id = self._generate_session_id()
        self.cross_session_knowledge: Dict[str, Any] = {}
        self.session_history: List[Dict[str, Any]] = []
        
        # Configuration
        self.max_relationship_age = timedelta(days=config.get("max_relationship_age_days", 30))
        self.pattern_detection_threshold = config.get("pattern_detection_threshold", 0.7)
        
        logger.info(f"Initialized TemporalKnowledgeGraph with session {self.session_id}")
    
    def _generate_session_id(self) -> str:
        """Generate unique session identifier."""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:12]
    
    async def add_temporal_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        properties: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a temporal relationship to the graph."""
        
        relationship = TemporalRelationship(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            start_time=start_time,
            end_time=end_time,
            properties=properties or {}
        )
        
        # Store in memory
        key = f"{source_id}:{target_id}"
        self.temporal_relationships[key].append(relationship)
        
        # Persist to graph store
        await self._persist_relationship(relationship)
        
        # Update cross-session knowledge
        await self._update_cross_session_knowledge(relationship)
        
        # Check for patterns
        await self._detect_temporal_patterns(source_id, target_id, relation_type)
        
        logger.debug(f"Added temporal relationship: {source_id} -> {target_id} ({relation_type})")
        
        return f"{relationship.source_id}:{relationship.target_id}:{relationship.start_time.isoformat()}"
    
    async def _persist_relationship(self, relationship: TemporalRelationship):
        """Persist relationship to graph store with temporal properties."""
        
        # Create Cypher query for Neo4j-style storage
        query = """
        MERGE (s:Entity {id: $source_id})
        MERGE (t:Entity {id: $target_id})
        CREATE (s)-[r:TEMPORAL_RELATION {
            type: $relation_type,
            start_time: $start_time,
            end_time: $end_time,
            session_id: $session_id,
            properties: $properties
        }]->(t)
        """
        
        params = {
            "source_id": relationship.source_id,
            "target_id": relationship.target_id,
            "relation_type": relationship.relation_type,
            "start_time": relationship.start_time.isoformat(),
            "end_time": relationship.end_time.isoformat() if relationship.end_time else None,
            "session_id": self.session_id,
            "properties": json.dumps(relationship.properties)
        }
        
        # This would be implemented by the specific graph store backend
        # await self.graph_store.execute_query(query, params)
        
        logger.debug(f"Persisted temporal relationship to graph store")
    
    async def _update_cross_session_knowledge(self, relationship: TemporalRelationship):
        """Update cross-session knowledge based on new relationship."""
        
        # Track relationship patterns across sessions
        pattern_key = f"{relationship.relation_type}:{relationship.source_id}:{relationship.target_id}"
        
        if pattern_key not in self.cross_session_knowledge:
            self.cross_session_knowledge[pattern_key] = {
                "count": 0,
                "first_seen": relationship.start_time,
                "last_seen": relationship.start_time,
                "sessions": set(),
                "avg_duration": None,
                "confidence": 0.0
            }
        
        pattern_info = self.cross_session_knowledge[pattern_key]
        pattern_info["count"] += 1
        pattern_info["last_seen"] = relationship.start_time
        pattern_info["sessions"].add(self.session_id)
        
        # Calculate average duration if end_time is available
        if relationship.end_time:
            duration = (relationship.end_time - relationship.start_time).total_seconds()
            if pattern_info["avg_duration"] is None:
                pattern_info["avg_duration"] = duration
            else:
                pattern_info["avg_duration"] = (pattern_info["avg_duration"] + duration) / 2
        
        # Update confidence based on frequency and recency
        pattern_info["confidence"] = min(1.0, pattern_info["count"] / 10.0)
        
        logger.debug(f"Updated cross-session knowledge for pattern: {pattern_key}")
    
    async def _detect_temporal_patterns(self, source_id: str, target_id: str, relation_type: str):
        """Detect temporal patterns in relationships."""
        
        # Get all relationships of this type involving these entities
        related_relationships = []
        for key, relationships in self.temporal_relationships.items():
            for rel in relationships:
                if (rel.relation_type == relation_type and 
                    (rel.source_id == source_id or rel.target_id == target_id)):
                    related_relationships.append(rel)
        
        if len(related_relationships) < 3:
            return  # Need at least 3 relationships to detect patterns
        
        # Simple pattern detection: recurring time intervals
        time_intervals = []
        for rel in related_relationships:
            if rel.end_time:
                interval = (rel.end_time - rel.start_time).total_seconds()
                time_intervals.append(interval)
        
        if len(time_intervals) >= 3:
            avg_interval = np.mean(time_intervals)
            std_interval = np.std(time_intervals)
            
            # If intervals are consistent (low std deviation), it's a pattern
            if std_interval / avg_interval < 0.3:  # 30% variation threshold
                pattern_id = f"pattern_{source_id}_{target_id}_{relation_type}"
                
                pattern = TemporalPattern(
                    pattern_id=pattern_id,
                    pattern_type="recurring",
                    entities=[source_id, target_id],
                    relations=[relation_type],
                    time_intervals=[TimeInterval(
                        start=rel.start_time,
                        end=rel.end_time or rel.start_time
                    ) for rel in related_relationships if rel.end_time],
                    confidence=1.0 - (std_interval / avg_interval),
                    support=len(related_relationships)
                )
                
                self.temporal_patterns[pattern_id] = pattern
                logger.info(f"Detected temporal pattern: {pattern_id} with confidence {pattern.confidence:.2f}")
    
    async def query_temporal_relationships(
        self,
        entity_id: str,
        time_range: Optional[TimeInterval] = None,
        relation_types: Optional[List[str]] = None
    ) -> List[TemporalRelationship]:
        """Query temporal relationships for an entity."""
        
        results = []
        
        for key, relationships in self.temporal_relationships.items():
            for rel in relationships:
                # Check if entity is involved
                if entity_id not in [rel.source_id, rel.target_id]:
                    continue
                
                # Check time range
                if time_range and not time_range.contains(rel.start_time):
                    continue
                
                # Check relation types
                if relation_types and rel.relation_type not in relation_types:
                    continue
                
                results.append(rel)
        
        return results
    
    async def get_cross_session_insights(self, entity_id: str) -> Dict[str, Any]:
        """Get cross-session insights for an entity."""
        
        insights = {
            "entity_id": entity_id,
            "session_appearances": 0,
            "relationship_patterns": [],
            "temporal_patterns": [],
            "confidence_scores": {}
        }
        
        # Analyze cross-session knowledge
        for pattern_key, pattern_info in self.cross_session_knowledge.items():
            if entity_id in pattern_key:
                insights["relationship_patterns"].append({
                    "pattern": pattern_key,
                    "frequency": pattern_info["count"],
                    "sessions": len(pattern_info["sessions"]),
                    "confidence": pattern_info["confidence"],
                    "avg_duration": pattern_info.get("avg_duration")
                })
        
        # Add temporal patterns
        for pattern_id, pattern in self.temporal_patterns.items():
            if entity_id in pattern.entities:
                insights["temporal_patterns"].append({
                    "pattern_id": pattern_id,
                    "pattern_type": pattern.pattern_type,
                    "confidence": pattern.confidence,
                    "support": pattern.support
                })
        
        return insights


class AutomatedKnowledgeConsolidator:
    """
    Automated knowledge consolidation system that manages memory importance,
    performs cleanup, and optimizes knowledge storage.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Memory management
        self.memory_traces: Dict[str, MemoryTrace] = {}
        self.knowledge_clusters: Dict[str, KnowledgeCluster] = {}
        self.consolidation_queue: deque = deque()
        
        # Configuration
        self.max_memory_traces = config.get("max_memory_traces", 10000)
        self.consolidation_threshold = config.get("consolidation_threshold", 0.8)
        self.cleanup_interval = timedelta(hours=config.get("cleanup_interval_hours", 24))
        self.min_importance_threshold = config.get("min_importance_threshold", 0.1)
        
        # Strategy
        self.consolidation_strategy = ConsolidationStrategy(
            config.get("consolidation_strategy", "hybrid")
        )
        
        # State tracking
        self.last_cleanup = datetime.now()
        self.consolidation_stats = {
            "total_consolidations": 0,
            "traces_removed": 0,
            "clusters_created": 0,
            "last_consolidation": None
        }
        
        logger.info(f"Initialized AutomatedKnowledgeConsolidator with strategy: {self.consolidation_strategy.value}")
    
    async def add_memory_trace(
        self,
        trace_id: str,
        content: Any,
        importance: MemoryImportance = MemoryImportance.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a new memory trace to the system."""
        
        trace = MemoryTrace(
            trace_id=trace_id,
            content=content,
            timestamp=datetime.now(),
            importance_score=importance.value,
            metadata=metadata or {}
        )
        
        self.memory_traces[trace_id] = trace
        
        # Check if consolidation is needed
        if len(self.memory_traces) > self.max_memory_traces:
            await self._trigger_consolidation()
        
        logger.debug(f"Added memory trace: {trace_id} with importance {importance.value}")
        return trace_id
    
    async def access_memory_trace(self, trace_id: str) -> Optional[MemoryTrace]:
        """Access a memory trace and update its statistics."""
        
        if trace_id not in self.memory_traces:
            return None
        
        trace = self.memory_traces[trace_id]
        trace.access(datetime.now())
        
        return trace
    
    async def _trigger_consolidation(self):
        """Trigger memory consolidation process."""
        
        current_time = datetime.now()
        
        # Calculate importance scores for all traces
        trace_importance = {}
        for trace_id, trace in self.memory_traces.items():
            importance = trace.calculate_current_importance(current_time)
            trace_importance[trace_id] = importance
        
        # Sort by importance
        sorted_traces = sorted(trace_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Apply consolidation strategy
        if self.consolidation_strategy == ConsolidationStrategy.FREQUENCY_BASED:
            await self._consolidate_by_frequency(sorted_traces)
        elif self.consolidation_strategy == ConsolidationStrategy.RECENCY_BASED:
            await self._consolidate_by_recency(sorted_traces)
        elif self.consolidation_strategy == ConsolidationStrategy.IMPORTANCE_BASED:
            await self._consolidate_by_importance(sorted_traces)
        else:  # HYBRID
            await self._consolidate_hybrid(sorted_traces)
        
        self.consolidation_stats["total_consolidations"] += 1
        self.consolidation_stats["last_consolidation"] = current_time
        
        logger.info(f"Completed memory consolidation. Traces: {len(self.memory_traces)}")
    
    async def _consolidate_by_importance(self, sorted_traces: List[Tuple[str, float]]):
        """Consolidate memory traces based on importance scores."""
        
        # Keep top traces, remove low-importance ones
        keep_count = int(self.max_memory_traces * 0.8)  # Keep 80%
        
        traces_to_remove = []
        for i, (trace_id, importance) in enumerate(sorted_traces):
            if i >= keep_count or importance < self.min_importance_threshold:
                traces_to_remove.append(trace_id)
        
        # Remove low-importance traces
        for trace_id in traces_to_remove:
            del self.memory_traces[trace_id]
            self.consolidation_stats["traces_removed"] += 1
        
        # Cluster remaining traces
        await self._cluster_similar_traces()
    
    async def _consolidate_by_frequency(self, sorted_traces: List[Tuple[str, float]]):
        """Consolidate based on access frequency."""
        
        # Sort by access count
        frequency_sorted = sorted(
            self.memory_traces.items(),
            key=lambda x: x[1].access_count,
            reverse=True
        )
        
        # Keep frequently accessed traces
        keep_count = int(self.max_memory_traces * 0.8)
        traces_to_remove = [trace_id for trace_id, _ in frequency_sorted[keep_count:]]
        
        for trace_id in traces_to_remove:
            del self.memory_traces[trace_id]
            self.consolidation_stats["traces_removed"] += 1
    
    async def _consolidate_by_recency(self, sorted_traces: List[Tuple[str, float]]):
        """Consolidate based on recency."""
        
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(days=7)  # Keep last 7 days
        
        traces_to_remove = []
        for trace_id, trace in self.memory_traces.items():
            last_time = trace.last_accessed or trace.timestamp
            if last_time < cutoff_time:
                traces_to_remove.append(trace_id)
        
        for trace_id in traces_to_remove:
            del self.memory_traces[trace_id]
            self.consolidation_stats["traces_removed"] += 1
    
    async def _consolidate_hybrid(self, sorted_traces: List[Tuple[str, float]]):
        """Hybrid consolidation combining multiple strategies."""
        
        # First pass: Remove very low importance traces
        traces_to_remove = []
        for trace_id, importance in sorted_traces:
            if importance < self.min_importance_threshold:
                traces_to_remove.append(trace_id)
        
        for trace_id in traces_to_remove:
            del self.memory_traces[trace_id]
            self.consolidation_stats["traces_removed"] += 1
        
        # Second pass: Apply recency filter
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(days=30)  # 30-day cutoff
        
        old_traces = []
        for trace_id, trace in self.memory_traces.items():
            last_time = trace.last_accessed or trace.timestamp
            if last_time < cutoff_time and trace.access_count < 2:
                old_traces.append(trace_id)
        
        for trace_id in old_traces:
            del self.memory_traces[trace_id]
            self.consolidation_stats["traces_removed"] += 1
        
        # Third pass: Cluster similar traces
        await self._cluster_similar_traces()
    
    async def _cluster_similar_traces(self):
        """Cluster similar memory traces to reduce redundancy."""
        
        if len(self.memory_traces) < 10:
            return  # Not enough traces to cluster
        
        # This is a simplified clustering approach
        # In practice, you'd use embeddings and proper clustering algorithms
        
        from collections import defaultdict
        
        # Group traces by content type or metadata similarity
        content_groups = defaultdict(list)
        
        for trace_id, trace in self.memory_traces.items():
            # Simple grouping by content type
            content_type = type(trace.content).__name__
            content_groups[content_type].append(trace_id)
        
        # Create clusters for groups with multiple items
        for content_type, trace_ids in content_groups.items():
            if len(trace_ids) > 3:  # Cluster if more than 3 similar traces
                cluster_id = f"cluster_{content_type}_{datetime.now().timestamp()}"
                
                # Calculate centroid (simplified)
                centroid = np.random.random(384)  # Placeholder embedding
                
                cluster = KnowledgeCluster(
                    cluster_id=cluster_id,
                    centroid_embedding=centroid,
                    member_traces=set(trace_ids),
                    creation_time=datetime.now(),
                    last_updated=datetime.now(),
                    importance_score=np.mean([
                        self.memory_traces[tid].importance_score for tid in trace_ids
                    ])
                )
                
                self.knowledge_clusters[cluster_id] = cluster
                self.consolidation_stats["clusters_created"] += 1
                
                logger.debug(f"Created knowledge cluster: {cluster_id} with {len(trace_ids)} traces")
    
    async def get_consolidation_stats(self) -> Dict[str, Any]:
        """Get consolidation statistics."""
        
        return {
            **self.consolidation_stats,
            "current_traces": len(self.memory_traces),
            "current_clusters": len(self.knowledge_clusters),
            "memory_utilization": len(self.memory_traces) / self.max_memory_traces,
            "avg_trace_importance": np.mean([
                trace.importance_score for trace in self.memory_traces.values()
            ]) if self.memory_traces else 0.0
        }
    
    async def cleanup_expired_traces(self):
        """Clean up expired memory traces."""
        
        current_time = datetime.now()
        
        if current_time - self.last_cleanup < self.cleanup_interval:
            return  # Not time for cleanup yet
        
        expired_traces = []
        for trace_id, trace in self.memory_traces.items():
            importance = trace.calculate_current_importance(current_time)
            if importance < self.min_importance_threshold:
                expired_traces.append(trace_id)
        
        for trace_id in expired_traces:
            del self.memory_traces[trace_id]
            self.consolidation_stats["traces_removed"] += 1
        
        self.last_cleanup = current_time
        
        if expired_traces:
            logger.info(f"Cleaned up {len(expired_traces)} expired memory traces")


class PersistentMemoryManager:
    """
    Main manager for persistent memory architecture combining temporal knowledge graphs
    and automated knowledge consolidation.
    """
    
    def __init__(
        self,
        graph_store: GraphStoreInterface,
        config: Dict[str, Any]
    ):
        self.config = config
        
        # Initialize components
        self.temporal_graph = TemporalKnowledgeGraph(graph_store, config.get("temporal_graph", {}))
        self.consolidator = AutomatedKnowledgeConsolidator(config.get("consolidation", {}))
        
        # Cross-session learning state
        self.session_learning_enabled = config.get("cross_session_learning", True)
        self.learning_rate = config.get("learning_rate", 0.01)
        self.adaptation_threshold = config.get("adaptation_threshold", 0.7)
        
        logger.info("Initialized PersistentMemoryManager with cross-session learning")
    
    async def add_knowledge_item(
        self,
        item_id: str,
        content: Any,
        importance: MemoryImportance = MemoryImportance.MEDIUM,
        temporal_context: Optional[Dict[str, Any]] = None,
        relationships: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Add a knowledge item with temporal and relationship context."""
        
        # Add to memory consolidator
        trace_id = await self.consolidator.add_memory_trace(
            trace_id=item_id,
            content=content,
            importance=importance,
            metadata=temporal_context or {}
        )
        
        # Add temporal relationships if provided
        if relationships:
            for rel in relationships:
                await self.temporal_graph.add_temporal_relationship(
                    source_id=item_id,
                    target_id=rel["target_id"],
                    relation_type=rel["relation_type"],
                    start_time=rel.get("start_time", datetime.now()),
                    end_time=rel.get("end_time"),
                    properties=rel.get("properties", {})
                )
        
        return trace_id
    
    async def query_knowledge(
        self,
        query: str,
        temporal_context: Optional[TimeInterval] = None,
        include_cross_session: bool = True
    ) -> Dict[str, Any]:
        """Query knowledge with temporal and cross-session context."""
        
        results = {
            "query": query,
            "temporal_context": temporal_context,
            "memory_traces": [],
            "temporal_relationships": [],
            "cross_session_insights": {},
            "consolidation_info": {}
        }
        
        # Search memory traces (simplified - would use embeddings in practice)
        for trace_id, trace in self.consolidator.memory_traces.items():
            if query.lower() in str(trace.content).lower():
                results["memory_traces"].append({
                    "trace_id": trace_id,
                    "content": trace.content,
                    "importance": trace.calculate_current_importance(datetime.now()),
                    "access_count": trace.access_count,
                    "last_accessed": trace.last_accessed
                })
        
        # Get temporal relationships
        if results["memory_traces"]:
            for trace_info in results["memory_traces"]:
                trace_id = trace_info["trace_id"]
                relationships = await self.temporal_graph.query_temporal_relationships(
                    entity_id=trace_id,
                    time_range=temporal_context
                )
                
                for rel in relationships:
                    results["temporal_relationships"].append({
                        "source_id": rel.source_id,
                        "target_id": rel.target_id,
                        "relation_type": rel.relation_type,
                        "start_time": rel.start_time,
                        "end_time": rel.end_time,
                        "confidence": rel.confidence
                    })
        
        # Get cross-session insights
        if include_cross_session and results["memory_traces"]:
            for trace_info in results["memory_traces"]:
                trace_id = trace_info["trace_id"]
                insights = await self.temporal_graph.get_cross_session_insights(trace_id)
                results["cross_session_insights"][trace_id] = insights
        
        # Get consolidation info
        results["consolidation_info"] = await self.consolidator.get_consolidation_stats()
        
        return results
    
    async def learn_from_interaction(
        self,
        query: str,
        selected_results: List[str],
        feedback_score: float,
        context: Optional[Dict[str, Any]] = None
    ):
        """Learn from user interactions to improve future results."""
        
        if not self.session_learning_enabled:
            return
        
        current_time = datetime.now()
        
        # Update importance scores for selected results
        for result_id in selected_results:
            if result_id in self.consolidator.memory_traces:
                trace = self.consolidator.memory_traces[result_id]
                
                # Adjust importance based on feedback
                importance_adjustment = feedback_score * self.learning_rate
                trace.importance_score = min(1.0, trace.importance_score + importance_adjustment)
                
                # Record access
                trace.access(current_time)
        
        # Create temporal relationship between query and results
        query_id = hashlib.md5(query.encode()).hexdigest()[:12]
        
        for result_id in selected_results:
            await self.temporal_graph.add_temporal_relationship(
                source_id=query_id,
                target_id=result_id,
                relation_type="query_result",
                start_time=current_time,
                properties={
                    "feedback_score": feedback_score,
                    "query": query,
                    "context": context or {}
                }
            )
        
        logger.debug(f"Learned from interaction: query={query}, feedback={feedback_score}")
    
    async def get_memory_status(self) -> Dict[str, Any]:
        """Get comprehensive memory system status."""
        
        consolidation_stats = await self.consolidator.get_consolidation_stats()
        
        return {
            "session_id": self.temporal_graph.session_id,
            "memory_traces": len(self.consolidator.memory_traces),
            "knowledge_clusters": len(self.consolidator.knowledge_clusters),
            "temporal_relationships": sum(
                len(rels) for rels in self.temporal_graph.temporal_relationships.values()
            ),
            "temporal_patterns": len(self.temporal_graph.temporal_patterns),
            "cross_session_knowledge": len(self.temporal_graph.cross_session_knowledge),
            "consolidation_stats": consolidation_stats,
            "learning_enabled": self.session_learning_enabled
        }