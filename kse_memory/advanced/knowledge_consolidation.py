"""
Knowledge Consolidation Engine for KSE Memory SDK

This module implements advanced knowledge consolidation capabilities including:
- Automated importance ranking with multi-factor analysis
- Memory retention policies with intelligent archival
- Knowledge graph pruning and optimization
- Conflict resolution and knowledge synthesis
"""

import asyncio
import logging
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, Counter
import uuid
import hashlib
import math

logger = logging.getLogger(__name__)


class ImportanceLevel(Enum):
    """Knowledge importance levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    EPHEMERAL = "ephemeral"


class RetentionPolicy(Enum):
    """Memory retention policies."""
    PERMANENT = "permanent"
    LONG_TERM = "long_term"  # Years
    MEDIUM_TERM = "medium_term"  # Months
    SHORT_TERM = "short_term"  # Weeks
    TEMPORARY = "temporary"  # Days


class ConsolidationStrategy(Enum):
    """Knowledge consolidation strategies."""
    FREQUENCY_BASED = "frequency_based"
    RECENCY_BASED = "recency_based"
    IMPORTANCE_BASED = "importance_based"
    NETWORK_BASED = "network_based"
    HYBRID = "hybrid"


@dataclass
class KnowledgeImportanceFactors:
    """Factors contributing to knowledge importance."""
    
    # Usage factors
    access_frequency: float = 0.0
    recent_access: float = 0.0
    user_engagement: float = 0.0
    
    # Content factors
    information_density: float = 0.0
    uniqueness: float = 0.0
    completeness: float = 0.0
    
    # Network factors
    connectivity: float = 0.0
    centrality: float = 0.0
    bridge_score: float = 0.0
    
    # Contextual factors
    domain_relevance: float = 0.0
    temporal_relevance: float = 0.0
    user_preference: float = 0.0
    
    # Meta factors
    source_reliability: float = 0.0
    validation_score: float = 0.0
    expert_rating: float = 0.0
    
    def calculate_composite_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate composite importance score."""
        
        if weights is None:
            weights = {
                "usage": 0.25,
                "content": 0.25,
                "network": 0.20,
                "contextual": 0.15,
                "meta": 0.15
            }
        
        # Usage component
        usage_score = (self.access_frequency + self.recent_access + self.user_engagement) / 3
        
        # Content component
        content_score = (self.information_density + self.uniqueness + self.completeness) / 3
        
        # Network component
        network_score = (self.connectivity + self.centrality + self.bridge_score) / 3
        
        # Contextual component
        contextual_score = (self.domain_relevance + self.temporal_relevance + self.user_preference) / 3
        
        # Meta component
        meta_score = (self.source_reliability + self.validation_score + self.expert_rating) / 3
        
        # Weighted composite
        composite = (
            usage_score * weights["usage"] +
            content_score * weights["content"] +
            network_score * weights["network"] +
            contextual_score * weights["contextual"] +
            meta_score * weights["meta"]
        )
        
        return min(composite, 1.0)


@dataclass
class ConsolidatedKnowledge:
    """Represents consolidated knowledge item."""
    
    consolidated_id: str
    source_items: List[str]
    consolidation_strategy: ConsolidationStrategy
    created_at: datetime
    
    # Consolidated content
    title: str
    summary: str
    key_concepts: List[str]
    relationships: List[Dict[str, Any]]
    
    # Quality metrics
    consolidation_confidence: float
    information_gain: float
    coherence_score: float
    
    # Retention
    importance_level: ImportanceLevel
    retention_policy: RetentionPolicy
    expires_at: Optional[datetime] = None
    
    def is_expired(self) -> bool:
        """Check if consolidated knowledge has expired."""
        if not self.expires_at:
            return False
        return datetime.now() > self.expires_at


class ImportanceRanker:
    """Automated importance ranking system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Ranking models and weights
        self.importance_weights = self.config.get("importance_weights", {
            "usage": 0.25,
            "content": 0.25,
            "network": 0.20,
            "contextual": 0.15,
            "meta": 0.15
        })
        
        # Thresholds for importance levels
        self.importance_thresholds = {
            ImportanceLevel.CRITICAL: 0.9,
            ImportanceLevel.HIGH: 0.7,
            ImportanceLevel.MEDIUM: 0.5,
            ImportanceLevel.LOW: 0.3,
            ImportanceLevel.EPHEMERAL: 0.0
        }
        
        # Caching for performance
        self.importance_cache: Dict[str, Tuple[float, datetime]] = {}
        self.cache_ttl = timedelta(hours=1)
    
    async def rank_knowledge_items(
        self,
        knowledge_items: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, ImportanceLevel]]:
        """Rank knowledge items by importance."""
        
        rankings = []
        
        for item in knowledge_items:
            item_id = item.get("id", "")
            
            # Check cache first
            cached_result = self._get_cached_importance(item_id)
            if cached_result:
                importance_score, importance_level = cached_result
            else:
                # Calculate importance factors
                factors = await self._calculate_importance_factors(item, knowledge_items, context)
                
                # Calculate composite score
                importance_score = factors.calculate_composite_score(self.importance_weights)
                
                # Determine importance level
                importance_level = self._determine_importance_level(importance_score)
                
                # Cache result
                self._cache_importance(item_id, importance_score, importance_level)
            
            rankings.append((item_id, importance_score, importance_level))
        
        # Sort by importance score (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Ranked {len(rankings)} knowledge items by importance")
        
        return rankings
    
    async def _calculate_importance_factors(
        self,
        item: Dict[str, Any],
        all_items: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> KnowledgeImportanceFactors:
        """Calculate importance factors for a knowledge item."""
        
        factors = KnowledgeImportanceFactors()
        
        # Usage factors
        factors.access_frequency = await self._calculate_access_frequency(item)
        factors.recent_access = await self._calculate_recent_access(item)
        factors.user_engagement = await self._calculate_user_engagement(item)
        
        # Content factors
        factors.information_density = await self._calculate_information_density(item)
        factors.uniqueness = await self._calculate_uniqueness(item, all_items)
        factors.completeness = await self._calculate_completeness(item)
        
        # Network factors
        factors.connectivity = await self._calculate_connectivity(item, all_items)
        factors.centrality = await self._calculate_centrality(item, all_items)
        factors.bridge_score = await self._calculate_bridge_score(item, all_items)
        
        # Contextual factors
        factors.domain_relevance = await self._calculate_domain_relevance(item, context)
        factors.temporal_relevance = await self._calculate_temporal_relevance(item)
        factors.user_preference = await self._calculate_user_preference(item, context)
        
        # Meta factors
        factors.source_reliability = await self._calculate_source_reliability(item)
        factors.validation_score = await self._calculate_validation_score(item)
        factors.expert_rating = await self._calculate_expert_rating(item)
        
        return factors
    
    async def _calculate_access_frequency(self, item: Dict[str, Any]) -> float:
        """Calculate access frequency score."""
        
        access_count = item.get("access_count", 0)
        created_at = item.get("created_at", datetime.now())
        age_days = (datetime.now() - created_at).days + 1
        
        # Normalize by age to get frequency per day
        frequency = access_count / age_days
        
        # Apply logarithmic scaling to prevent extreme values
        return min(math.log(frequency + 1) / math.log(10), 1.0)
    
    async def _calculate_recent_access(self, item: Dict[str, Any]) -> float:
        """Calculate recent access score."""
        
        last_accessed = item.get("last_accessed")
        if not last_accessed:
            return 0.0
        
        # Calculate days since last access
        days_since = (datetime.now() - last_accessed).days
        
        # Exponential decay (half-life of 7 days)
        return math.exp(-days_since * math.log(2) / 7)
    
    async def _calculate_information_density(self, item: Dict[str, Any]) -> float:
        """Calculate information density score."""
        
        content = item.get("content", "")
        
        # Simple metrics (in production, use advanced NLP)
        word_count = len(content.split())
        unique_words = len(set(content.lower().split()))
        
        if word_count == 0:
            return 0.0
        
        # Density based on unique word ratio and length
        density = (unique_words / word_count) * min(word_count / 100, 1.0)
        
        return min(density, 1.0)
    
    async def _calculate_uniqueness(
        self,
        item: Dict[str, Any],
        all_items: List[Dict[str, Any]]
    ) -> float:
        """Calculate uniqueness score compared to other items."""
        
        item_content = item.get("content", "").lower()
        if not item_content:
            return 0.0
        
        # Calculate similarity with other items
        similarities = []
        
        for other_item in all_items:
            if other_item.get("id") == item.get("id"):
                continue
            
            other_content = other_item.get("content", "").lower()
            if not other_content:
                continue
            
            # Simple word overlap similarity
            item_words = set(item_content.split())
            other_words = set(other_content.split())
            
            if not item_words or not other_words:
                continue
            
            intersection = len(item_words & other_words)
            union = len(item_words | other_words)
            
            similarity = intersection / union if union > 0 else 0.0
            similarities.append(similarity)
        
        if not similarities:
            return 1.0  # Unique if no other items to compare
        
        # Uniqueness is inverse of maximum similarity
        max_similarity = max(similarities)
        return 1.0 - max_similarity
    
    async def _calculate_connectivity(
        self,
        item: Dict[str, Any],
        all_items: List[Dict[str, Any]]
    ) -> float:
        """Calculate network connectivity score."""
        
        item_id = item.get("id", "")
        connections = 0
        
        # Count relationships with other items
        for other_item in all_items:
            if other_item.get("id") == item_id:
                continue
            
            # Check for relationships
            item_relations = item.get("relationships", [])
            for relation in item_relations:
                if relation.get("target_id") == other_item.get("id"):
                    connections += 1
        
        # Normalize by total possible connections
        max_connections = len(all_items) - 1
        if max_connections == 0:
            return 0.0
        
        return min(connections / max_connections, 1.0)
    
    def _determine_importance_level(self, score: float) -> ImportanceLevel:
        """Determine importance level from score."""
        
        for level, threshold in self.importance_thresholds.items():
            if score >= threshold:
                return level
        
        return ImportanceLevel.EPHEMERAL
    
    def _get_cached_importance(self, item_id: str) -> Optional[Tuple[float, ImportanceLevel]]:
        """Get cached importance if still valid."""
        
        if item_id not in self.importance_cache:
            return None
        
        score, level, cached_at = self.importance_cache[item_id]
        
        if datetime.now() - cached_at > self.cache_ttl:
            del self.importance_cache[item_id]
            return None
        
        return score, level
    
    def _cache_importance(self, item_id: str, score: float, level: ImportanceLevel):
        """Cache importance calculation."""
        
        self.importance_cache[item_id] = (score, level, datetime.now())


class RetentionPolicyManager:
    """Manages memory retention policies and lifecycle."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Retention periods
        self.retention_periods = {
            RetentionPolicy.PERMANENT: None,  # Never expires
            RetentionPolicy.LONG_TERM: timedelta(days=365 * 2),  # 2 years
            RetentionPolicy.MEDIUM_TERM: timedelta(days=90),  # 3 months
            RetentionPolicy.SHORT_TERM: timedelta(days=30),  # 1 month
            RetentionPolicy.TEMPORARY: timedelta(days=7),  # 1 week
        }
        
        # Policy mappings
        self.importance_to_policy = {
            ImportanceLevel.CRITICAL: RetentionPolicy.PERMANENT,
            ImportanceLevel.HIGH: RetentionPolicy.LONG_TERM,
            ImportanceLevel.MEDIUM: RetentionPolicy.MEDIUM_TERM,
            ImportanceLevel.LOW: RetentionPolicy.SHORT_TERM,
            ImportanceLevel.EPHEMERAL: RetentionPolicy.TEMPORARY,
        }
        
        # Archive storage
        self.archived_items: Dict[str, Dict[str, Any]] = {}
        self.pending_deletion: List[str] = []
    
    async def apply_retention_policies(
        self,
        knowledge_items: List[Dict[str, Any]],
        importance_rankings: List[Tuple[str, float, ImportanceLevel]]
    ) -> Dict[str, Any]:
        """Apply retention policies to knowledge items."""
        
        # Create importance lookup
        importance_map = {item_id: (score, level) for item_id, score, level in importance_rankings}
        
        policy_actions = {
            "retained": [],
            "archived": [],
            "scheduled_deletion": [],
            "policy_updates": []
        }
        
        for item in knowledge_items:
            item_id = item.get("id", "")
            
            if item_id not in importance_map:
                continue
            
            score, importance_level = importance_map[item_id]
            
            # Determine retention policy
            retention_policy = self.importance_to_policy.get(
                importance_level,
                RetentionPolicy.TEMPORARY
            )
            
            # Apply policy
            action = await self._apply_item_policy(item, retention_policy, importance_level)
            
            # Record action
            if action["type"] == "retain":
                policy_actions["retained"].append(item_id)
            elif action["type"] == "archive":
                policy_actions["archived"].append(item_id)
                await self._archive_item(item, action["details"])
            elif action["type"] == "schedule_deletion":
                policy_actions["scheduled_deletion"].append(item_id)
                self.pending_deletion.append(item_id)
            elif action["type"] == "update_policy":
                policy_actions["policy_updates"].append({
                    "item_id": item_id,
                    "old_policy": action["old_policy"],
                    "new_policy": action["new_policy"]
                })
        
        logger.info(f"Applied retention policies: {len(policy_actions['retained'])} retained, "
                   f"{len(policy_actions['archived'])} archived, "
                   f"{len(policy_actions['scheduled_deletion'])} scheduled for deletion")
        
        return policy_actions
    
    async def _apply_item_policy(
        self,
        item: Dict[str, Any],
        retention_policy: RetentionPolicy,
        importance_level: ImportanceLevel
    ) -> Dict[str, Any]:
        """Apply retention policy to a specific item."""
        
        item_id = item.get("id", "")
        created_at = item.get("created_at", datetime.now())
        current_policy = item.get("retention_policy")
        
        # Check if item has expired
        if await self._is_item_expired(item, retention_policy):
            if importance_level in [ImportanceLevel.LOW, ImportanceLevel.EPHEMERAL]:
                return {
                    "type": "schedule_deletion",
                    "reason": "expired_low_importance"
                }
            else:
                return {
                    "type": "archive",
                    "reason": "expired_but_important",
                    "details": {
                        "archive_reason": "retention_policy_expired",
                        "original_importance": importance_level.value
                    }
                }
        
        # Check if policy needs updating
        if current_policy != retention_policy.value:
            # Update item policy
            item["retention_policy"] = retention_policy.value
            item["policy_updated_at"] = datetime.now()
            
            # Set expiration date if applicable
            retention_period = self.retention_periods.get(retention_policy)
            if retention_period:
                item["expires_at"] = created_at + retention_period
            else:
                item["expires_at"] = None
            
            return {
                "type": "update_policy",
                "old_policy": current_policy,
                "new_policy": retention_policy.value
            }
        
        return {
            "type": "retain",
            "reason": "within_policy"
        }
    
    async def _is_item_expired(
        self,
        item: Dict[str, Any],
        retention_policy: RetentionPolicy
    ) -> bool:
        """Check if item has expired based on retention policy."""
        
        if retention_policy == RetentionPolicy.PERMANENT:
            return False
        
        created_at = item.get("created_at", datetime.now())
        retention_period = self.retention_periods.get(retention_policy)
        
        if not retention_period:
            return False
        
        expiry_date = created_at + retention_period
        return datetime.now() > expiry_date
    
    async def _archive_item(self, item: Dict[str, Any], archive_details: Dict[str, Any]):
        """Archive a knowledge item."""
        
        item_id = item.get("id", "")
        
        archived_item = {
            **item,
            "archived_at": datetime.now(),
            "archive_reason": archive_details.get("archive_reason", "unknown"),
            "original_importance": archive_details.get("original_importance", "unknown")
        }
        
        self.archived_items[item_id] = archived_item
        
        logger.info(f"Archived item {item_id}: {archive_details.get('archive_reason')}")
    
    async def cleanup_expired_items(self) -> Dict[str, Any]:
        """Clean up expired items scheduled for deletion."""
        
        deleted_items = []
        
        for item_id in self.pending_deletion[:]:  # Copy to avoid modification during iteration
            # In production, this would actually delete from storage
            deleted_items.append(item_id)
            self.pending_deletion.remove(item_id)
            
            logger.info(f"Deleted expired item: {item_id}")
        
        return {
            "deleted_count": len(deleted_items),
            "deleted_items": deleted_items,
            "cleanup_timestamp": datetime.now()
        }
    
    def get_retention_statistics(self) -> Dict[str, Any]:
        """Get retention policy statistics."""
        
        return {
            "archived_items": len(self.archived_items),
            "pending_deletion": len(self.pending_deletion),
            "retention_policies": {
                policy.value: {
                    "period_days": period.days if period else None,
                    "description": f"{policy.value.replace('_', ' ').title()} retention"
                }
                for policy, period in self.retention_periods.items()
            }
        }


class KnowledgeConsolidator:
    """Consolidates related knowledge items into coherent units."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Consolidation parameters
        self.similarity_threshold = self.config.get("similarity_threshold", 0.7)
        self.min_consolidation_items = self.config.get("min_consolidation_items", 2)
        self.max_consolidation_items = self.config.get("max_consolidation_items", 10)
        
        # Consolidated knowledge storage
        self.consolidated_items: Dict[str, ConsolidatedKnowledge] = {}
        
        # Consolidation strategies
        self.strategies = {
            ConsolidationStrategy.FREQUENCY_BASED: self._consolidate_by_frequency,
            ConsolidationStrategy.RECENCY_BASED: self._consolidate_by_recency,
            ConsolidationStrategy.IMPORTANCE_BASED: self._consolidate_by_importance,
            ConsolidationStrategy.NETWORK_BASED: self._consolidate_by_network,
            ConsolidationStrategy.HYBRID: self._consolidate_hybrid
        }
    
    async def consolidate_knowledge(
        self,
        knowledge_items: List[Dict[str, Any]],
        strategy: ConsolidationStrategy = ConsolidationStrategy.HYBRID,
        context: Optional[Dict[str, Any]] = None
    ) -> List[ConsolidatedKnowledge]:
        """Consolidate knowledge items using specified strategy."""
        
        # Get consolidation strategy function
        strategy_func = self.strategies.get(strategy, self._consolidate_hybrid)
        
        # Find consolidation candidates
        consolidation_groups = await strategy_func(knowledge_items, context)
        
        # Create consolidated knowledge items
        consolidated_items = []
        
        for group in consolidation_groups:
            if len(group) < self.min_consolidation_items:
                continue
            
            consolidated = await self._create_consolidated_item(group, strategy, context)
            consolidated_items.append(consolidated)
            
            # Store consolidated item
            self.consolidated_items[consolidated.consolidated_id] = consolidated
        
        logger.info(f"Consolidated {len(knowledge_items)} items into {len(consolidated_items)} consolidated items")
        
        return consolidated_items
    
    async def _consolidate_by_frequency(
        self,
        items: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> List[List[Dict[str, Any]]]:
        """Consolidate based on access frequency patterns."""
        
        # Group items by similar access patterns
        frequency_groups = defaultdict(list)
        
        for item in items:
            access_count = item.get("access_count", 0)
            frequency_bucket = self._get_frequency_bucket(access_count)
            frequency_groups[frequency_bucket].append(item)
        
        # Further group by content similarity within frequency buckets
        consolidation_groups = []
        
        for bucket, bucket_items in frequency_groups.items():
            if len(bucket_items) < self.min_consolidation_items:
                continue
            
            similarity_groups = await self._group_by_similarity(bucket_items)
            consolidation_groups.extend(similarity_groups)
        
        return consolidation_groups
    
    async def _consolidate_by_importance(
        self,
        items: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> List[List[Dict[str, Any]]]:
        """Consolidate based on importance levels."""
        
        # Group items by importance level
        importance_groups = defaultdict(list)
        
        for item in items:
            importance = item.get("importance_level", ImportanceLevel.MEDIUM.value)
            importance_groups[importance].append(item)
        
        # Group by similarity within importance levels
        consolidation_groups = []
        
        for level, level_items in importance_groups.items():
            if len(level_items) < self.min_consolidation_items:
                continue
            
            similarity_groups = await self._group_by_similarity(level_items)
            consolidation_groups.extend(similarity_groups)
        
        return consolidation_groups
    
    async def _consolidate_hybrid(
        self,
        items: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> List[List[Dict[str, Any]]]:
        """Hybrid consolidation using multiple factors."""
        
        # Calculate composite similarity scores
        similarity_matrix = await self._calculate_hybrid_similarity_matrix(items)
        
        # Use clustering algorithm to group similar items
        clusters = await self._cluster_items(items, similarity_matrix)
        
        return clusters
    
    async def _group_by_similarity(
        self,
        items: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """Group items by content similarity."""
        
        if len(items) < 2:
            return [items] if items else []
        
        # Calculate pairwise similarities
        similarity_matrix = []
        
        for i, item1 in enumerate(items):
            row = []
            for j, item2 in enumerate(items):
                if i == j:
                    similarity = 1.0
                else:
                    similarity = await self._calculate_content_similarity(item1, item2)
                row.append(similarity)
            similarity_matrix.append(row)
        
        # Group items with high similarity
        groups = []
        used_indices = set()
        
        for i, item in enumerate(items):
            if i in used_indices:
                continue
            
            group = [item]
            used_indices.add(i)
            
            for j, other_item in enumerate(items):
                if j in used_indices or i == j:
                    continue
                
                if similarity_matrix[i][j] >= self.similarity_threshold:
                    group.append(other_item)
                    used_indices.add(j)
            
            if len(group) >= self.min_consolidation_items:
                groups.append(group)
        
        return groups
    
    async def _create_consolidated_item(
        self,
        source_items: List[Dict[str, Any]],
        strategy: ConsolidationStrategy,
        context: Optional[Dict[str, Any]] = None
    ) -> ConsolidatedKnowledge:
        """Create a consolidated knowledge item from source items."""
        
        consolidated_id = f"consolidated_{uuid.uuid4().hex[:8]}"
        
        # Extract key information from source items
        titles = [item.get("title", "") for item in source_items if item.get("title")]
        contents = [item.get("content", "") for item in source_items if item.get("content")]
        concepts = []
        
        for item in source_items:
            item_concepts = item.get("concepts", [])
            concepts.extend(item_concepts)
        
        # Generate consolidated title
        consolidated_title = await self._generate_consolidated_title(titles, contents)
        
        # Generate consolidated summary
        consolidated_summary = await self._generate_consolidated_summary(contents)
        
        # Extract key concepts
        key_concepts = await self._extract_key_concepts(concepts, contents)
        
        # Merge relationships
        relationships = await self._merge_relationships(source_items)
        
        # Calculate quality metrics
        consolidation_confidence = await self._calculate_consolidation_confidence(source_items)
        information_gain = await self._calculate_information_gain(source_items, consolidated_summary)
        coherence_score = await self._calculate_coherence_score(source_items, consolidated_summary)
        
        # Determine importance and retention
        max_importance = max(
            (ImportanceLevel(item.get("importance_level", ImportanceLevel.MEDIUM.value)) 
             for item in source_items),
            default=ImportanceLevel.MEDIUM
        )
        
        retention_policy = RetentionPolicy.MEDIUM_TERM
        if max_importance in [ImportanceLevel.CRITICAL, ImportanceLevel.HIGH]:
            retention_policy = RetentionPolicy.LONG_TERM
        
        consolidated = ConsolidatedKnowledge(
            consolidated_id=consolidated_id,
            source_items=[item.get("id", "") for item in source_items],
            consolidation_strategy=strategy,
            created_at=datetime.now(),
            title=consolidated_title,
            summary=consolidated_summary,
            key_concepts=key_concepts,
            relationships=relationships,
            consolidation_confidence=consolidation_confidence,
            information_gain=information_gain,
            coherence_score=coherence_score,
            importance_level=max_importance,
            retention_policy=retention_policy
        )
        
        return consolidated
    
    async def _generate_consolidated_title(
        self,
        titles: List[str],
        contents: List[str]
    ) -> str:
        """Generate title for consolidated knowledge."""
        
        if not titles and not contents:
            return "Consolidated Knowledge"
        
        # Use most common words from titles
        if titles:
            all_title_words = []
            for title in titles:
                all_title_words.extend(title.lower().split())
            
            word_counts = Counter(all_title_words)
            common_words = [word for word, count in word_counts.most_common(3)]
            
            if common_words:
                return " ".join(common_words).title()
        
        # Fallback to content analysis
        if contents:
            all_content = " ".join(contents)
            words = all_content.lower().split()
            word_counts = Counter(words)
            
            # Filter out common stop words (simplified)
            stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
            meaningful_words = [word for word, count in word_counts.most_common(10) if word not in stop_words]
            
            if meaningful_words:
                return " ".join(meaningful_words[:3]).title()
        
        return "Consolidated Knowledge"
    
    async def _generate_consolidated_summary(self, contents: List[str]) -> str:
        """Generate summary for consolidated knowledge."""
        
        if not contents:
            return "No content available for summary."
        
        # Simple extractive summarization (in production, use advanced NLP)
        all_sentences = []
        
        for content in contents:
            # Split into sentences (simplified)
            sentences = [s.strip() for s in content.split('.') if s.strip()]
            all_sentences.extend(sentences)
        
        if not all_sentences:
            return "Content consolidation completed."
        
        # Take first few sentences as summary
        summary_sentences = all_sentences[:3]
        summary = ". ".join(summary_sentences)
        
        if not summary.endswith('.'):
            summary += "."
        
        return summary
    
    def get_consolidation_statistics(self) -> Dict[str, Any]:
        """Get consolidation statistics."""
        
        total_consolidated = len(self.consolidated_items)
        
        # Group by strategy
        strategy_counts = defaultdict(int)
        for item in self.consolidated_items.values():
            strategy_counts[item.consolidation_strategy.value] += 1
        
        # Group by importance
        importance_counts = defaultdict(int)
        for item in self.consolidated_items.values():
            importance_counts[item.importance_level.value] += 1
        
        # Calculate average metrics
        if total_consolidated > 0:
            avg_confidence = sum(item.consolidation_confidence for item in self.consolidated_items.values()) / total_consolidated
            avg_coherence = sum(item.coherence_score for item in self.consolidated_items.values()) / total_consolidated
            avg_info_gain = sum(item.information_gain for item in self.consolidated_items.values()) / total_consolidated
        else:
            avg_confidence = avg_coherence = avg_info_gain = 0.0
        
        return {
            "total_consolidated_items": total_consolidated,
            "consolidation_strategies": dict(strategy_counts),
            "importance_distribution": dict(importance_counts),
            "average_metrics": {
                "consolidation_confidence": avg_confidence,
                "coherence_score": avg_coherence,
                "information_gain": avg_info_gain
            }
        }


class KnowledgeConsolidationEngine:
    """Main knowledge consolidation engine coordinating all components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self.importance_ranker = ImportanceRanker(self.config.get("importance_ranker", {}))
        self.retention_manager = RetentionPolicyManager(self.config.get("retention_manager", {}))
        self.consolidator = KnowledgeConsolidator(self.config.get("consolidator", {}))
        
        # Performance tracking
        self.consolidation_metrics = {
            "items_processed": 0,
            "items_consolidated": 0,
            "items_archived": 0,
            "items_deleted": 0,
            "consolidation_runs": 0,
            "average_importance_score": 0.0
        }
        
        logger.info("Initialized KnowledgeConsolidationEngine")
    
    async def consolidate_knowledge_base(
        self,
        knowledge_items: List[Dict[str, Any]],
        strategy: ConsolidationStrategy = ConsolidationStrategy.HYBRID,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Perform comprehensive knowledge base consolidation."""
        
        # Step 1: Rank items by importance
        importance_rankings = await self.importance_ranker.rank_knowledge_items(
            knowledge_items, context
        )
        
        # Step 2: Apply retention policies
        retention_actions = await self.retention_manager.apply_retention_policies(
            knowledge_items, importance_rankings
        )
        
        # Step 3: Consolidate related knowledge
        retained_items = [
            item for item in knowledge_items
            if item.get("id") in retention_actions["retained"] or
            item.get("id") in retention_actions["policy_updates"]
        ]
        
        consolidated_items = await self.consolidator.consolidate_knowledge(
            retained_items, strategy, context
        )
        
        # Step 4: Clean up expired items
        cleanup_result = await self.retention_manager.cleanup_expired_items()
        
        # Update metrics
        self.consolidation_metrics["items_processed"] += len(knowledge_items)
        self.consolidation_metrics["items_consolidated"] += len(consolidated_items)
        self.consolidation_metrics["items_archived"] += len(retention_actions["archived"])
        self.consolidation_metrics["items_deleted"] += cleanup_result["deleted_count"]
        self.consolidation_metrics["consolidation_runs"] += 1
        
        if importance_rankings:
            avg_importance = sum(score for _, score, _ in importance_rankings) / len(importance_rankings)
            self.consolidation_metrics["average_importance_score"] = avg_importance
        
        result = {
            "consolidation_id": f"consolidation_{uuid.uuid4().hex[:8]}",
            "processed_at": datetime.now(),
            "input_items": len(knowledge_items),
            "importance_rankings": len(importance_rankings),
            "retention_actions": retention_actions,
            "consolidated_items": len(consolidated_items),
            "cleanup_result": cleanup_result,
            "consolidation_strategy": strategy.value,
            "performance_metrics": {
                "processing_time": "calculated_in_production",
                "memory_usage": "calculated_in_production",
                "consolidation_ratio": len(consolidated_items) / max(len(knowledge_items), 1)
            }
        }
        
        logger.info(f"Consolidated knowledge base: {len(knowledge_items)} → {len(consolidated_items)} items")
        
        return result
    
    async def get_knowledge_quality_report(
        self,
        knowledge_items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate knowledge quality and consolidation report."""
        
        # Analyze current knowledge base
        importance_rankings = await self.importance_ranker.rank_knowledge_items(knowledge_items)
        
        # Quality metrics
        importance_distribution = defaultdict(int)
        total_score = 0.0
        
        for item_id, score, level in importance_rankings:
            importance_distribution[level.value] += 1
            total_score += score
        
        avg_importance = total_score / len(importance_rankings) if importance_rankings else 0.0
        
        # Consolidation potential
        consolidation_candidates = []
        for item in knowledge_items:
            if item.get("importance_level") in [ImportanceLevel.MEDIUM.value, ImportanceLevel.HIGH.value]:
                consolidation_candidates.append(item)
        
        # Retention analysis
        retention_stats = self.retention_manager.get_retention_statistics()
        consolidation_stats = self.consolidator.get_consolidation_statistics()
        
        return {
            "knowledge_base_summary": {
                "total_items": len(knowledge_items),
                "average_importance": avg_importance,
                "importance_distribution": dict(importance_distribution)
            },
            "consolidation_potential": {
                "candidates": len(consolidation_candidates),
                "estimated_reduction": f"{len(consolidation_candidates) * 0.3:.0f} items"
            },
            "retention_analysis": retention_stats,
            "consolidation_history": consolidation_stats,
            "recommendations": await self._generate_consolidation_recommendations(
                knowledge_items, importance_rankings
            ),
            "engine_metrics": self.consolidation_metrics
        }
    
    async def _generate_consolidation_recommendations(
        self,
        knowledge_items: List[Dict[str, Any]],
        importance_rankings: List[Tuple[str, float, ImportanceLevel]]
    ) -> List[str]:
        """Generate recommendations for knowledge consolidation."""
        
        recommendations = []
        
        # Importance-based recommendations
        low_importance_count = len([
            level for _, _, level in importance_rankings
            if level == ImportanceLevel.LOW
        ])
        
        if low_importance_count > 10:
            recommendations.append(
                f"Consider archiving {low_importance_count} low-importance items to reduce clutter"
            )
        
        # Consolidation recommendations
        high_importance_count = len([
            level for _, _, level in importance_rankings
            if level in [ImportanceLevel.HIGH, ImportanceLevel.CRITICAL]
        ])
        
        if high_importance_count > 20:
            recommendations.append(
                "High number of important items detected - consider consolidation to improve organization"
            )
        
        # Retention recommendations
        if len(self.retention_manager.pending_deletion) > 0:
            recommendations.append(
                f"Clean up {len(self.retention_manager.pending_deletion)} items scheduled for deletion"
            )
        
        # Performance recommendations
        if len(knowledge_items) > 1000:
            recommendations.append(
                "Large knowledge base detected - regular consolidation recommended for optimal performance"
            )
        
        return recommendations
    
    def get_consolidation_status(self) -> Dict[str, Any]:
        """Get knowledge consolidation engine status."""
        
        return {
            "engine_status": "active",
            "metrics": self.consolidation_metrics,
            "components": {
                "importance_ranker": {
                    "status": "active",
                    "cached_items": len(self.importance_ranker.importance_cache)
                },
                "retention_manager": {
                    "status": "active",
                    "archived_items": len(self.retention_manager.archived_items),
                    "pending_deletion": len(self.retention_manager.pending_deletion)
                },
                "consolidator": {
                    "status": "active",
                    "consolidated_items": len(self.consolidator.consolidated_items)
                }
            }
        }