"""
Foundation Integration Module for KSE Memory SDK

This module integrates the persistent memory architecture, multi-modal embeddings,
and meta-learning core into a unified foundation layer that serves as the universal
substrate for domain customizations.
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from .persistent_memory import (
    PersistentMemoryManager, MemoryImportance, TemporalKnowledgeGraph,
    AutomatedKnowledgeConsolidator
)
from .multimodal_embeddings import (
    UniversalMultiModalEncoder, ModalityType, MultiModalEmbedding,
    ModalityEmbedding, FusionStrategy
)
from .meta_learning import (
    TransferLearningEngine, AdaptationStrategy, MetaTask, TaskType,
    AdaptationResult
)
from .interfaces import GraphStoreInterface
from .models import Product, SearchQuery, SearchResult, ConceptualSpace
from ..temporal.temporal_models import TimeInterval

logger = logging.getLogger(__name__)


class FoundationCapability(Enum):
    """Core capabilities of the foundation layer."""
    PERSISTENT_MEMORY = "persistent_memory"
    MULTIMODAL_EMBEDDING = "multimodal_embedding"
    META_LEARNING = "meta_learning"
    CROSS_MODAL_SEARCH = "cross_modal_search"
    TEMPORAL_REASONING = "temporal_reasoning"
    DOMAIN_ADAPTATION = "domain_adaptation"


@dataclass
class FoundationConfig:
    """Configuration for the foundation layer."""
    
    # Persistent Memory Configuration
    persistent_memory: Dict[str, Any] = field(default_factory=lambda: {
        "temporal_graph": {
            "max_relationship_age_days": 30,
            "pattern_detection_threshold": 0.7
        },
        "consolidation": {
            "max_memory_traces": 10000,
            "consolidation_strategy": "hybrid",
            "min_importance_threshold": 0.1,
            "cleanup_interval_hours": 24
        },
        "cross_session_learning": True,
        "learning_rate": 0.01,
        "adaptation_threshold": 0.7
    })
    
    # Multi-Modal Embedding Configuration
    multimodal_embedding: Dict[str, Any] = field(default_factory=lambda: {
        "enable_text": True,
        "enable_image": True,
        "enable_audio": True,
        "enable_structured": True,
        "text_model": "sentence-transformers/all-MiniLM-L6-v2",
        "image_model": "openai/clip-vit-base-patch32",
        "audio_model": "facebook/wav2vec2-base",
        "structured_dim": 384,
        "fusion": {
            "fusion_strategy": "attention_weighted",
            "target_dimension": 512
        },
        "cache_size": 1000
    })
    
    # Meta-Learning Configuration
    meta_learning: Dict[str, Any] = field(default_factory=lambda: {
        "enable_maml": True,
        "enable_prototypical": True,
        "default_strategy": "maml",
        "min_examples_per_class": 3,
        "max_adaptation_time": 300,
        "maml": {
            "inner_lr": 0.01,
            "outer_lr": 0.001,
            "inner_steps": 5,
            "meta_batch_size": 4,
            "input_dim": 512,
            "hidden_dim": 256,
            "output_dim": 10
        },
        "prototypical": {
            "input_dim": 512,
            "hidden_dim": 256,
            "embedding_dim": 128,
            "learning_rate": 0.001
        }
    })
    
    # Integration Configuration
    integration: Dict[str, Any] = field(default_factory=lambda: {
        "enable_cross_modal_memory": True,
        "enable_temporal_multimodal": True,
        "enable_meta_memory_consolidation": True,
        "similarity_threshold": 0.5,
        "max_concurrent_adaptations": 3,
        "foundation_cache_size": 5000
    })


@dataclass
class UniversalKnowledgeItem:
    """Universal knowledge item that combines all foundation capabilities."""
    
    item_id: str
    content: Any
    modality_embeddings: Dict[ModalityType, ModalityEmbedding]
    fused_embedding: MultiModalEmbedding
    conceptual_coordinates: np.ndarray
    importance: MemoryImportance
    domain: str
    temporal_context: Optional[TimeInterval] = None
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    adaptation_metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_primary_modality(self) -> ModalityType:
        """Get the primary modality based on confidence scores."""
        if not self.modality_embeddings:
            return ModalityType.TEXT  # Default
        
        return max(
            self.modality_embeddings.items(),
            key=lambda x: x[1].confidence
        )[0]
    
    def get_cross_modal_features(self) -> Dict[str, float]:
        """Extract cross-modal features for analysis."""
        features = {}
        
        # Modality presence
        for modality in ModalityType:
            features[f"has_{modality.value}"] = float(modality in self.modality_embeddings)
        
        # Confidence scores
        for modality, embedding in self.modality_embeddings.items():
            features[f"{modality.value}_confidence"] = embedding.confidence
        
        # Alignment scores from fused embedding
        if hasattr(self.fused_embedding, 'alignment_scores'):
            for (mod1, mod2), score in self.fused_embedding.alignment_scores.items():
                features[f"alignment_{mod1.value}_{mod2.value}"] = score
        
        return features


class UniversalFoundationLayer:
    """
    Universal Foundation Layer that integrates all core capabilities into
    a unified substrate for domain customizations.
    """
    
    def __init__(
        self,
        graph_store: GraphStoreInterface,
        config: Optional[FoundationConfig] = None
    ):
        self.config = config or FoundationConfig()
        self.graph_store = graph_store
        
        # Initialize core components
        self._initialize_components()
        
        # Integration state
        self.universal_knowledge: Dict[str, UniversalKnowledgeItem] = {}
        self.domain_adaptations: Dict[str, AdaptationResult] = {}
        self.cross_modal_cache: Dict[str, Any] = {}
        
        # Performance tracking
        self.performance_metrics = {
            "total_items": 0,
            "successful_adaptations": 0,
            "cross_modal_queries": 0,
            "temporal_patterns_detected": 0,
            "memory_consolidations": 0
        }
        
        # Active capabilities
        self.active_capabilities = set()
        self._validate_capabilities()
        
        logger.info(f"Initialized UniversalFoundationLayer with {len(self.active_capabilities)} capabilities")
    
    def _initialize_components(self):
        """Initialize all foundation components."""
        
        # Persistent Memory
        self.memory_manager = PersistentMemoryManager(
            graph_store=self.graph_store,
            config=self.config.persistent_memory
        )
        self.active_capabilities.add(FoundationCapability.PERSISTENT_MEMORY)
        
        # Multi-Modal Embeddings
        try:
            self.multimodal_encoder = UniversalMultiModalEncoder(
                config=self.config.multimodal_embedding
            )
            self.active_capabilities.add(FoundationCapability.MULTIMODAL_EMBEDDING)
            self.active_capabilities.add(FoundationCapability.CROSS_MODAL_SEARCH)
        except Exception as e:
            logger.warning(f"Failed to initialize multimodal encoder: {e}")
            self.multimodal_encoder = None
        
        # Meta-Learning
        try:
            self.meta_learning_engine = TransferLearningEngine(
                config=self.config.meta_learning
            )
            self.active_capabilities.add(FoundationCapability.META_LEARNING)
            self.active_capabilities.add(FoundationCapability.DOMAIN_ADAPTATION)
        except Exception as e:
            logger.warning(f"Failed to initialize meta-learning engine: {e}")
            self.meta_learning_engine = None
        
        # Temporal reasoning (from memory manager)
        if hasattr(self.memory_manager, 'temporal_graph'):
            self.active_capabilities.add(FoundationCapability.TEMPORAL_REASONING)
    
    def _validate_capabilities(self):
        """Validate that required capabilities are available."""
        
        required_capabilities = [
            FoundationCapability.PERSISTENT_MEMORY,
            FoundationCapability.MULTIMODAL_EMBEDDING,
            FoundationCapability.META_LEARNING
        ]
        
        missing_capabilities = set(required_capabilities) - self.active_capabilities
        
        if missing_capabilities:
            logger.warning(f"Missing required capabilities: {missing_capabilities}")
        else:
            logger.info("All required foundation capabilities are active")
    
    async def add_universal_knowledge_item(
        self,
        item_id: str,
        content: Any,
        modality_data: Dict[ModalityType, Any],
        domain: str,
        importance: MemoryImportance = MemoryImportance.MEDIUM,
        temporal_context: Optional[TimeInterval] = None,
        relationships: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Add a universal knowledge item with full foundation capabilities."""
        
        try:
            # Multi-modal encoding
            if self.multimodal_encoder:
                multimodal_result = await self.multimodal_encoder.encode_multimodal(
                    modality_data, return_individual=False
                )
                
                if isinstance(multimodal_result, dict):
                    # Individual embeddings returned
                    modality_embeddings = multimodal_result
                    fused_embedding = None
                else:
                    # Fused embedding returned
                    modality_embeddings = multimodal_result.modality_embeddings
                    fused_embedding = multimodal_result
            else:
                # Fallback: create dummy embeddings
                modality_embeddings = {}
                fused_embedding = None
            
            # Generate conceptual coordinates (simplified)
            conceptual_coords = np.random.randn(10)  # 10-dimensional conceptual space
            
            # Create universal knowledge item
            universal_item = UniversalKnowledgeItem(
                item_id=item_id,
                content=content,
                modality_embeddings=modality_embeddings,
                fused_embedding=fused_embedding,
                conceptual_coordinates=conceptual_coords,
                importance=importance,
                domain=domain,
                temporal_context=temporal_context,
                relationships=relationships or []
            )
            
            # Store in universal knowledge
            self.universal_knowledge[item_id] = universal_item
            
            # Add to persistent memory
            await self.memory_manager.add_knowledge_item(
                item_id=item_id,
                content=content,
                importance=importance,
                temporal_context={"domain": domain, "temporal_context": temporal_context},
                relationships=relationships
            )
            
            # Update performance metrics
            self.performance_metrics["total_items"] += 1
            
            logger.debug(f"Added universal knowledge item: {item_id} in domain {domain}")
            
            return item_id
            
        except Exception as e:
            logger.error(f"Failed to add universal knowledge item {item_id}: {e}")
            raise
    
    async def adapt_to_domain(
        self,
        domain: str,
        examples: List[Dict[str, Any]],
        strategy: Optional[AdaptationStrategy] = None
    ) -> AdaptationResult:
        """Adapt the foundation to a new domain using meta-learning."""
        
        if not self.meta_learning_engine:
            raise RuntimeError("Meta-learning engine not available")
        
        # Enhance examples with multi-modal embeddings if available
        enhanced_examples = []
        
        for example in examples:
            enhanced_example = example.copy()
            
            # Add embeddings if content is available
            if 'content' in example and self.multimodal_encoder:
                try:
                    # Determine modality
                    modality_data = self._infer_modality_data(example['content'])
                    
                    if modality_data:
                        embeddings = await self.multimodal_encoder.encode_multimodal(
                            modality_data, return_individual=True
                        )
                        
                        # Use primary embedding
                        if embeddings:
                            primary_modality = max(embeddings.keys(), key=lambda k: embeddings[k].confidence)
                            enhanced_example['embedding'] = embeddings[primary_modality].embedding
                
                except Exception as e:
                    logger.warning(f"Failed to enhance example with embeddings: {e}")
            
            enhanced_examples.append(enhanced_example)
        
        # Perform domain adaptation
        result = await self.meta_learning_engine.learn_domain(
            domain=domain,
            examples=enhanced_examples,
            strategy=strategy
        )
        
        # Store adaptation result
        self.domain_adaptations[domain] = result
        
        # Update performance metrics
        if result.is_successful():
            self.performance_metrics["successful_adaptations"] += 1
        
        logger.info(f"Adapted foundation to domain '{domain}': success={result.is_successful()}")
        
        return result
    
    def _infer_modality_data(self, content: Any) -> Dict[ModalityType, Any]:
        """Infer modality data from content."""
        
        modality_data = {}
        
        if isinstance(content, str):
            modality_data[ModalityType.TEXT] = content
        elif isinstance(content, dict):
            modality_data[ModalityType.STRUCTURED] = content
            
            # Check for embedded media
            if 'image' in content:
                modality_data[ModalityType.IMAGE] = content['image']
            if 'audio' in content:
                modality_data[ModalityType.AUDIO] = content['audio']
            if 'text' in content:
                modality_data[ModalityType.TEXT] = content['text']
        
        return modality_data
    
    async def cross_modal_search(
        self,
        query_data: Dict[ModalityType, Any],
        target_domains: Optional[List[str]] = None,
        similarity_threshold: float = 0.5,
        top_k: int = 10,
        include_temporal: bool = True
    ) -> List[Tuple[str, Dict[str, float]]]:
        """Perform cross-modal search across the universal knowledge base."""
        
        if not self.multimodal_encoder:
            raise RuntimeError("Multimodal encoder not available")
        
        # Encode query
        query_embedding = await self.multimodal_encoder.encode_multimodal(query_data)
        
        # Filter candidates by domain if specified
        candidates = []
        for item_id, item in self.universal_knowledge.items():
            if target_domains and item.domain not in target_domains:
                continue
            
            # Temporal filtering if requested
            if include_temporal and item.temporal_context:
                # Add temporal logic here
                pass
            
            candidates.append((item_id, item))
        
        # Compute similarities
        matches = []
        
        for item_id, item in candidates:
            if item.fused_embedding:
                similarities = await self.multimodal_encoder.compute_cross_modal_similarity(
                    query_embedding, item.fused_embedding
                )
                
                overall_score = similarities.get("fused_similarity", 0.0)
                
                # Add cross-modal boost
                cross_modal_scores = [
                    score for key, score in similarities.items()
                    if "to" in key and key.split("_to_")[0] != key.split("_to_")[1]
                ]
                
                if cross_modal_scores:
                    cross_modal_boost = max(cross_modal_scores) * 0.2
                    overall_score += cross_modal_boost
                
                if overall_score >= similarity_threshold:
                    matches.append((item_id, {
                        "overall_score": overall_score,
                        **similarities
                    }))
        
        # Sort and return top-k
        matches.sort(key=lambda x: x[1]["overall_score"], reverse=True)
        
        # Update performance metrics
        self.performance_metrics["cross_modal_queries"] += 1
        
        return matches[:top_k]
    
    async def temporal_reasoning_query(
        self,
        query: str,
        time_range: Optional[TimeInterval] = None,
        include_patterns: bool = True
    ) -> Dict[str, Any]:
        """Perform temporal reasoning query using persistent memory."""
        
        result = await self.memory_manager.query_knowledge(
            query=query,
            temporal_context=time_range,
            include_cross_session=True
        )
        
        if include_patterns:
            # Add temporal pattern information
            temporal_graph = self.memory_manager.temporal_graph
            
            result["temporal_patterns"] = [
                {
                    "pattern_id": pattern_id,
                    "pattern_type": pattern.pattern_type,
                    "confidence": pattern.confidence,
                    "support": pattern.support
                }
                for pattern_id, pattern in temporal_graph.temporal_patterns.items()
            ]
        
        return result
    
    async def transfer_knowledge_between_domains(
        self,
        source_domain: str,
        target_domain: str,
        target_examples: List[Dict[str, Any]],
        strategy: Optional[AdaptationStrategy] = None
    ) -> AdaptationResult:
        """Transfer knowledge from source domain to target domain."""
        
        if not self.meta_learning_engine:
            raise RuntimeError("Meta-learning engine not available")
        
        # Enhance target examples with embeddings
        enhanced_examples = []
        
        for example in target_examples:
            enhanced_example = example.copy()
            
            if 'content' in example and self.multimodal_encoder:
                try:
                    modality_data = self._infer_modality_data(example['content'])
                    
                    if modality_data:
                        embeddings = await self.multimodal_encoder.encode_multimodal(
                            modality_data, return_individual=True
                        )
                        
                        if embeddings:
                            primary_modality = max(embeddings.keys(), key=lambda k: embeddings[k].confidence)
                            enhanced_example['embedding'] = embeddings[primary_modality].embedding
                
                except Exception as e:
                    logger.warning(f"Failed to enhance transfer example: {e}")
            
            enhanced_examples.append(enhanced_example)
        
        # Perform knowledge transfer
        result = await self.meta_learning_engine.transfer_knowledge(
            source_domain=source_domain,
            target_domain=target_domain,
            target_examples=enhanced_examples,
            strategy=strategy
        )
        
        # Store adaptation result
        self.domain_adaptations[target_domain] = result
        
        logger.info(f"Transferred knowledge from '{source_domain}' to '{target_domain}'")
        
        return result
    
    async def consolidate_memory(self, force: bool = False) -> Dict[str, Any]:
        """Trigger memory consolidation across all components."""
        
        consolidation_results = {}
        
        # Consolidate persistent memory
        if hasattr(self.memory_manager, 'consolidator'):
            await self.memory_manager.consolidator.cleanup_expired_traces()
            stats = await self.memory_manager.consolidator.get_consolidation_stats()
            consolidation_results["persistent_memory"] = stats
        
        # Update performance metrics
        self.performance_metrics["memory_consolidations"] += 1
        
        logger.info("Completed memory consolidation across foundation layer")
        
        return consolidation_results
    
    async def get_foundation_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the foundation layer."""
        
        status = {
            "active_capabilities": [cap.value for cap in self.active_capabilities],
            "performance_metrics": self.performance_metrics.copy(),
            "universal_knowledge_items": len(self.universal_knowledge),
            "domain_adaptations": len(self.domain_adaptations),
            "cross_modal_cache_size": len(self.cross_modal_cache)
        }
        
        # Add component-specific status
        if self.memory_manager:
            memory_status = await self.memory_manager.get_memory_status()
            status["persistent_memory"] = memory_status
        
        if self.multimodal_encoder:
            multimodal_stats = await self.multimodal_encoder.get_modality_statistics()
            status["multimodal_encoder"] = multimodal_stats
        
        if self.meta_learning_engine:
            meta_stats = await self.meta_learning_engine.get_domain_statistics()
            status["meta_learning"] = meta_stats
        
        return status
    
    async def learn_from_interaction(
        self,
        query_data: Dict[ModalityType, Any],
        selected_results: List[str],
        feedback_score: float,
        context: Optional[Dict[str, Any]] = None
    ):
        """Learn from user interactions across all foundation components."""
        
        # Learn in persistent memory
        query_text = str(query_data.get(ModalityType.TEXT, "multimodal_query"))
        await self.memory_manager.learn_from_interaction(
            query=query_text,
            selected_results=selected_results,
            feedback_score=feedback_score,
            context=context
        )
        
        # Update importance scores for selected items
        for result_id in selected_results:
            if result_id in self.universal_knowledge:
                item = self.universal_knowledge[result_id]
                
                # Boost importance based on feedback
                if feedback_score > 0.7:
                    if item.importance.value < 1.0:
                        # Promote importance level
                        new_importance_value = min(1.0, item.importance.value + 0.1)
                        item.importance = MemoryImportance(new_importance_value)
        
        logger.debug(f"Learned from interaction: feedback={feedback_score}")
    
    def has_capability(self, capability: FoundationCapability) -> bool:
        """Check if a specific capability is available."""
        return capability in self.active_capabilities
    
    async def save_foundation_state(self, base_path: str):
        """Save the complete foundation state."""
        
        # Save meta-learning models
        if self.meta_learning_engine:
            self.meta_learning_engine.save_all_models(f"{base_path}_meta")
        
        # Save universal knowledge (simplified)
        import pickle
        
        with open(f"{base_path}_universal_knowledge.pkl", 'wb') as f:
            # Note: This is simplified - in production, you'd use proper serialization
            serializable_knowledge = {
                item_id: {
                    "item_id": item.item_id,
                    "content": item.content,
                    "domain": item.domain,
                    "importance": item.importance.value,
                    "created_at": item.created_at
                }
                for item_id, item in self.universal_knowledge.items()
            }
            pickle.dump(serializable_knowledge, f)
        
        logger.info(f"Saved foundation state to {base_path}_*")
    
    async def load_foundation_state(self, base_path: str):
        """Load the complete foundation state."""
        
        # Load meta-learning models
        if self.meta_learning_engine:
            self.meta_learning_engine.load_all_models(f"{base_path}_meta")
        
        # Load universal knowledge (simplified)
        import pickle
        
        try:
            with open(f"{base_path}_universal_knowledge.pkl", 'rb') as f:
                serializable_knowledge = pickle.load(f)
                
                # Reconstruct universal knowledge items (simplified)
                for item_id, data in serializable_knowledge.items():
                    # This is a simplified reconstruction
                    # In production, you'd properly deserialize all components
                    pass
        
        except FileNotFoundError:
            logger.warning(f"Universal knowledge file not found: {base_path}_universal_knowledge.pkl")
        
        logger.info(f"Loaded foundation state from {base_path}_*")