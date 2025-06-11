"""
Temporal KSE implementation with time-aware reasoning
"""

import torch
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging

from ..core.memory import KSEMemory
from ..core.config import KSEConfig
from ..core.models import SearchQuery, SearchResult
from .temporal_models import (
    TemporalEvent, TemporalQuery, TemporalQueryResult, 
    TemporalEntity, TemporalConstraint
)
from .time_embeddings import Time2VecEncoder, temporal_similarity_function
from .temporal_graph import TemporalNeo4jLayer
from .temporal_conceptual import TemporalConceptualSpace

logger = logging.getLogger(__name__)


class TemporalKSE(KSEMemory):
    """Extended KSE with temporal reasoning capabilities"""
    
    def __init__(self, config: KSEConfig):
        super().__init__(config)
        
        # Initialize temporal components
        self.temporal_graph = TemporalNeo4jLayer(
            uri=config.graph_store.get("uri", "bolt://localhost:7687"),
            auth=(
                config.graph_store.get("username", "neo4j"),
                config.graph_store.get("password", "password")
            )
        )
        
        self.time_aware_embeddings = Time2VecEncoder(
            embedding_dim=config.embedding.get("dimension", 512),
            time_dim=config.temporal.get("time_dimension", 64)
        )
        
        self.temporal_conceptual_space = TemporalConceptualSpace(
            dimensions=config.conceptual_space.get("dimensions", []),
            temporal_decay=config.temporal.get("decay_rate", 0.1)
        )
        
        # Temporal fusion weights
        self.temporal_fusion_weights = {
            "temporal_graph": config.temporal.get("graph_weight", 0.3),
            "temporal_conceptual": config.temporal.get("conceptual_weight", 0.4),
            "temporal_embedding": config.temporal.get("embedding_weight", 0.3)
        }
        
        logger.info("Temporal KSE initialized with time-aware reasoning")
    
    async def process_temporal_query(self, temporal_query: TemporalQuery) -> TemporalQueryResult:
        """Process queries involving temporal reasoning"""
        
        start_time = datetime.now()
        reasoning_path = []
        
        try:
            # Extract temporal features from query
            temporal_features = self._extract_temporal_features(temporal_query)
            reasoning_path.append(f"Extracted temporal features: {len(temporal_features)} dimensions")
            
            # Temporal graph traversal
            temporal_kg_score = await self._compute_temporal_graph_similarity(
                temporal_query, temporal_features
            )
            reasoning_path.append(f"Temporal graph similarity computed: {temporal_kg_score:.3f}")
            
            # Time-aware conceptual similarity
            temporal_cs_score = await self._compute_temporal_conceptual_similarity(
                temporal_query, temporal_features
            )
            reasoning_path.append(f"Temporal conceptual similarity: {temporal_cs_score:.3f}")
            
            # Time-aware embeddings
            temporal_ne_score = await self._compute_temporal_embedding_similarity(
                temporal_query, temporal_features
            )
            reasoning_path.append(f"Temporal embedding similarity: {temporal_ne_score:.3f}")
            
            # Temporal fusion
            results = await self._temporal_fusion(
                temporal_kg_score, temporal_cs_score, temporal_ne_score, 
                temporal_query, temporal_features
            )
            reasoning_path.append(f"Temporal fusion completed: {len(results)} results")
            
            # Calculate confidence score
            confidence_score = self._calculate_temporal_confidence(
                temporal_kg_score, temporal_cs_score, temporal_ne_score
            )
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return TemporalQueryResult(
                query=temporal_query,
                results=results,
                temporal_reasoning_path=reasoning_path,
                confidence_score=confidence_score,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error processing temporal query: {e}")
            raise
    
    def _extract_temporal_features(self, temporal_query: TemporalQuery) -> Dict[str, Any]:
        """Extract temporal features from query"""
        
        features = {
            "has_time_constraint": temporal_query.time_constraint is not None,
            "temporal_focus": temporal_query.temporal_focus,
            "granularity": temporal_query.temporal_granularity,
            "query_length": len(temporal_query.query_text.split()),
            "temporal_keywords": self._count_temporal_keywords(temporal_query.query_text)
        }
        
        if temporal_query.time_constraint:
            features.update({
                "time_span_days": (
                    temporal_query.time_constraint.max_time - 
                    temporal_query.time_constraint.min_time
                ).days,
                "is_historical": temporal_query.time_constraint.max_time < datetime.now(),
                "is_future": temporal_query.time_constraint.min_time > datetime.now()
            })
        
        return features
    
    def _count_temporal_keywords(self, query_text: str) -> int:
        """Count temporal keywords in query"""
        temporal_keywords = [
            "before", "after", "during", "when", "since", "until", "while",
            "yesterday", "today", "tomorrow", "last", "next", "recent",
            "historical", "past", "future", "current", "now", "then"
        ]
        
        words = query_text.lower().split()
        return sum(1 for word in words if word in temporal_keywords)
    
    async def _compute_temporal_graph_similarity(self, temporal_query: TemporalQuery, 
                                               temporal_features: Dict[str, Any]) -> float:
        """Compute similarity using temporal graph traversal"""
        
        try:
            # Find temporal paths in the graph
            temporal_paths = await self.temporal_graph.find_temporal_paths(
                query_text=temporal_query.query_text,
                time_constraint=temporal_query.time_constraint
            )
            
            if not temporal_paths:
                return 0.0
            
            # Calculate weighted similarity based on path confidence and temporal consistency
            total_score = 0.0
            for path in temporal_paths:
                path_score = path.total_confidence * path.temporal_consistency
                total_score += path_score
            
            # Normalize by number of paths
            return min(total_score / len(temporal_paths), 1.0)
            
        except Exception as e:
            logger.warning(f"Error in temporal graph similarity: {e}")
            return 0.0
    
    async def _compute_temporal_conceptual_similarity(self, temporal_query: TemporalQuery,
                                                    temporal_features: Dict[str, Any]) -> float:
        """Compute similarity using temporal conceptual space"""
        
        try:
            # Get query time for conceptual space lookup
            query_time = datetime.now()
            if temporal_query.time_constraint:
                # Use midpoint of time constraint
                query_time = temporal_query.time_constraint.min_time + (
                    temporal_query.time_constraint.max_time - 
                    temporal_query.time_constraint.min_time
                ) / 2
            
            # Compute temporal conceptual similarity
            similarity_score = await self.temporal_conceptual_space.compute_temporal_similarity(
                query_text=temporal_query.query_text,
                query_time=query_time,
                temporal_features=temporal_features
            )
            
            return similarity_score
            
        except Exception as e:
            logger.warning(f"Error in temporal conceptual similarity: {e}")
            return 0.0
    
    async def _compute_temporal_embedding_similarity(self, temporal_query: TemporalQuery,
                                                   temporal_features: Dict[str, Any]) -> float:
        """Compute similarity using time-aware embeddings"""
        
        try:
            # Create temporal events from query
            query_events = self._query_to_temporal_events(temporal_query)
            
            if not query_events:
                return 0.0
            
            # Encode temporal sequence
            query_encoding = self.time_aware_embeddings.encode_temporal_sequence(query_events)
            
            # Compare with stored temporal embeddings (simplified)
            # In practice, this would compare against a database of temporal embeddings
            similarity_score = self._compare_with_stored_embeddings(query_encoding)
            
            return similarity_score
            
        except Exception as e:
            logger.warning(f"Error in temporal embedding similarity: {e}")
            return 0.0
    
    def _query_to_temporal_events(self, temporal_query: TemporalQuery) -> List[TemporalEvent]:
        """Convert temporal query to sequence of temporal events"""
        
        events = []
        base_time = datetime.now()
        
        if temporal_query.time_constraint:
            base_time = temporal_query.time_constraint.min_time
        
        # Create event from query
        event = TemporalEvent(
            id="query_event",
            content=temporal_query.query_text,
            timestamp=base_time,
            event_type="query",
            metadata={"temporal_focus": temporal_query.temporal_focus}
        )
        
        events.append(event)
        return events
    
    def _compare_with_stored_embeddings(self, query_encoding: torch.Tensor) -> float:
        """Compare query encoding with stored temporal embeddings"""
        # Simplified implementation - in practice would use vector database
        # Return a dummy similarity score for now
        return 0.75
    
    async def _temporal_fusion(self, kg_score: float, cs_score: float, ne_score: float,
                             temporal_query: TemporalQuery, temporal_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fuse temporal similarity scores and generate results"""
        
        # Adaptive weighting based on temporal features
        weights = self._compute_adaptive_temporal_weights(temporal_features)
        
        # Compute fused score
        fused_score = (
            weights["temporal_graph"] * kg_score +
            weights["temporal_conceptual"] * cs_score +
            weights["temporal_embedding"] * ne_score
        )
        
        # Generate results based on fused score
        results = []
        
        # Simplified result generation - in practice would query actual data
        for i in range(min(5, int(fused_score * 10))):
            result = {
                "id": f"temporal_result_{i}",
                "title": f"Temporal Result {i+1}",
                "description": f"Result matching temporal query with score {fused_score:.3f}",
                "temporal_relevance": fused_score,
                "kg_contribution": weights["temporal_graph"] * kg_score,
                "cs_contribution": weights["temporal_conceptual"] * cs_score,
                "ne_contribution": weights["temporal_embedding"] * ne_score,
                "timestamp": datetime.now().isoformat()
            }
            results.append(result)
        
        return results
    
    def _compute_adaptive_temporal_weights(self, temporal_features: Dict[str, Any]) -> Dict[str, float]:
        """Compute adaptive weights based on temporal features"""
        
        base_weights = self.temporal_fusion_weights.copy()
        
        # Adjust weights based on temporal characteristics
        if temporal_features.get("has_time_constraint", False):
            # Increase graph weight for constrained queries
            base_weights["temporal_graph"] *= 1.2
        
        if temporal_features.get("temporal_keywords", 0) > 2:
            # Increase conceptual weight for keyword-rich queries
            base_weights["temporal_conceptual"] *= 1.1
        
        if temporal_features.get("is_historical", False):
            # Increase embedding weight for historical queries
            base_weights["temporal_embedding"] *= 1.15
        
        # Normalize weights
        total_weight = sum(base_weights.values())
        return {k: v / total_weight for k, v in base_weights.items()}
    
    def _calculate_temporal_confidence(self, kg_score: float, cs_score: float, ne_score: float) -> float:
        """Calculate overall confidence in temporal reasoning"""
        
        # Confidence based on score consistency
        scores = [kg_score, cs_score, ne_score]
        mean_score = np.mean(scores)
        score_variance = np.var(scores)
        
        # High confidence when scores are consistent and high
        consistency_factor = 1.0 - min(score_variance, 0.5)
        quality_factor = mean_score
        
        return (consistency_factor * quality_factor) ** 0.5
    
    async def add_temporal_entity(self, entity: TemporalEntity):
        """Add entity with temporal information"""
        
        try:
            # Store in temporal graph
            await self.temporal_graph.store_temporal_entity(entity)
            
            # Update temporal conceptual space
            await self.temporal_conceptual_space.update_temporal_dimensions(
                entity.id, entity.temporal_dimensions, entity.timestamp
            )
            
            # Store temporal embedding
            temporal_events = [TemporalEvent(
                id=f"{entity.id}_event",
                content=entity.name,
                timestamp=entity.timestamp,
                metadata=entity.properties
            )]
            
            encoding = self.time_aware_embeddings.encode_temporal_sequence(temporal_events)
            # Store encoding in vector database (implementation depends on backend)
            
            logger.info(f"Added temporal entity: {entity.id}")
            
        except Exception as e:
            logger.error(f"Error adding temporal entity: {e}")
            raise
    
    async def temporal_search(self, query_text: str, time_constraint: Optional[TemporalConstraint] = None,
                            temporal_focus: str = "current") -> TemporalQueryResult:
        """Convenience method for temporal search"""
        
        temporal_query = TemporalQuery(
            query_text=query_text,
            time_constraint=time_constraint,
            temporal_focus=temporal_focus
        )
        
        return await self.process_temporal_query(temporal_query)
    
    def get_temporal_statistics(self) -> Dict[str, Any]:
        """Get statistics about temporal data and performance"""
        
        return {
            "temporal_entities_count": self.temporal_graph.get_entity_count(),
            "temporal_relations_count": self.temporal_graph.get_relation_count(),
            "conceptual_dimensions": len(self.temporal_conceptual_space.dimensions),
            "embedding_dimension": self.time_aware_embeddings.embedding_dim,
            "fusion_weights": self.temporal_fusion_weights
        }