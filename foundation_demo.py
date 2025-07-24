#!/usr/bin/env python3
"""
KSE Memory SDK - Foundation Layer Demonstration

This script demonstrates the new Foundation Layer capabilities:
1. Persistent Memory Architecture with temporal knowledge graphs
2. Multi-Modal Embedding Engine with cross-modal similarity
3. Meta-Learning Core with MAML-based domain adaptation

The foundation layer serves as the universal substrate for domain customizations.
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Foundation Layer Imports
from kse_memory.core.foundation_integration import (
    UniversalFoundationLayer, FoundationConfig, FoundationCapability,
    UniversalKnowledgeItem
)
from kse_memory.core.persistent_memory import MemoryImportance
from kse_memory.core.multimodal_embeddings import ModalityType
from kse_memory.core.meta_learning import AdaptationStrategy, TaskType
from kse_memory.temporal.temporal_models import TimeInterval

# Mock graph store for demonstration
class MockGraphStore:
    """Mock graph store for demonstration purposes."""
    
    async def connect(self):
        return True
    
    async def disconnect(self):
        return True
    
    async def execute_query(self, query: str, params: Dict[str, Any]):
        return {"status": "success"}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FoundationDemo:
    """Comprehensive demonstration of the Foundation Layer."""
    
    def __init__(self):
        self.graph_store = MockGraphStore()
        self.foundation = None
        
    async def initialize_foundation(self):
        """Initialize the foundation layer with all capabilities."""
        
        print("🚀 Initializing Universal Foundation Layer...")
        
        # Create configuration
        config = FoundationConfig()
        
        # Reduce model requirements for demo
        config.multimodal_embedding.update({
            "enable_image": False,  # Disable to avoid model downloads
            "enable_audio": False,  # Disable to avoid model downloads
            "enable_text": True,
            "enable_structured": True
        })
        
        # Initialize foundation
        self.foundation = UniversalFoundationLayer(
            graph_store=self.graph_store,
            config=config
        )
        
        # Check capabilities
        capabilities = [cap.value for cap in self.foundation.active_capabilities]
        print(f"✅ Active capabilities: {capabilities}")
        
        return self.foundation

    async def demo_persistent_memory(self):
        """Demonstrate persistent memory with temporal knowledge graphs."""
        
        print("\n🧠 === PERSISTENT MEMORY DEMONSTRATION ===")
        
        # Add knowledge items with temporal context
        items = [
            {
                "id": "product_1",
                "content": "Premium wireless headphones with noise cancellation",
                "domain": "electronics",
                "importance": MemoryImportance.HIGH,
                "relationships": [
                    {
                        "target_id": "product_2",
                        "relation_type": "similar_to",
                        "start_time": datetime.now(),
                        "properties": {"similarity_score": 0.8}
                    }
                ]
            },
            {
                "id": "product_2", 
                "content": "Professional studio headphones for audio production",
                "domain": "electronics",
                "importance": MemoryImportance.MEDIUM
            },
            {
                "id": "query_1",
                "content": "Best headphones for music production",
                "domain": "electronics",
                "importance": MemoryImportance.LOW
            }
        ]
        
        # Add items to persistent memory
        for item in items:
            await self.foundation.memory_manager.add_knowledge_item(
                item_id=item["id"],
                content=item["content"],
                importance=item["importance"],
                temporal_context={"domain": item["domain"]},
                relationships=item.get("relationships", [])
            )
            print(f"📝 Added: {item['id']} - {item['content'][:50]}...")
        
        # Query with temporal context
        print("\n🔍 Querying persistent memory...")
        
        query_result = await self.foundation.memory_manager.query_knowledge(
            query="headphones",
            include_cross_session=True
        )
        
        print(f"Found {len(query_result['memory_traces'])} memory traces")
        print(f"Found {len(query_result['temporal_relationships'])} temporal relationships")
        
        # Demonstrate cross-session learning
        print("\n🔄 Learning from interaction...")
        
        await self.foundation.memory_manager.learn_from_interaction(
            query="headphones for music",
            selected_results=["product_2"],
            feedback_score=0.9,
            context={"user_intent": "professional_audio"}
        )
        
        # Get memory status
        memory_status = await self.foundation.memory_manager.get_memory_status()
        print(f"📊 Memory Status:")
        print(f"  - Memory traces: {memory_status['memory_traces']}")
        print(f"  - Temporal relationships: {memory_status['temporal_relationships']}")
        print(f"  - Cross-session knowledge: {memory_status['cross_session_knowledge']}")

    async def demo_multimodal_embeddings(self):
        """Demonstrate multi-modal embedding engine."""
        
        print("\n🎨 === MULTI-MODAL EMBEDDING DEMONSTRATION ===")
        
        if not self.foundation.multimodal_encoder:
            print("⚠️ Multimodal encoder not available - skipping demo")
            return
        
        # Test different modalities
        test_items = [
            {
                "id": "item_text",
                "modalities": {
                    ModalityType.TEXT: "High-quality wireless bluetooth headphones"
                }
            },
            {
                "id": "item_structured",
                "modalities": {
                    ModalityType.STRUCTURED: {
                        "product_name": "Wireless Headphones",
                        "category": "Electronics",
                        "price": 299.99,
                        "rating": 4.5,
                        "features": ["bluetooth", "noise_cancellation", "wireless"]
                    }
                }
            },
            {
                "id": "item_mixed",
                "modalities": {
                    ModalityType.TEXT: "Professional studio monitor headphones",
                    ModalityType.STRUCTURED: {
                        "type": "studio_monitor",
                        "impedance": "32_ohm",
                        "frequency_response": "20Hz-20kHz"
                    }
                }
            }
        ]
        
        embeddings = []
        
        # Encode each item
        for item in test_items:
            print(f"🔧 Encoding {item['id']}...")
            
            try:
                embedding = await self.foundation.multimodal_encoder.encode_multimodal(
                    item["modalities"]
                )
                embeddings.append((item["id"], embedding))
                
                print(f"  ✅ Encoded with modalities: {list(item['modalities'].keys())}")
                print(f"  📊 Fused embedding dimension: {len(embedding.fused_embedding)}")
                
            except Exception as e:
                print(f"  ❌ Failed to encode: {e}")
        
        # Demonstrate cross-modal similarity
        if len(embeddings) >= 2:
            print("\n🔍 Computing cross-modal similarities...")
            
            emb1_id, emb1 = embeddings[0]
            emb2_id, emb2 = embeddings[1]
            
            similarities = await self.foundation.multimodal_encoder.compute_cross_modal_similarity(
                emb1, emb2
            )
            
            print(f"Similarity between {emb1_id} and {emb2_id}:")
            for sim_type, score in similarities.items():
                print(f"  - {sim_type}: {score:.3f}")
        
        # Get modality statistics
        stats = await self.foundation.multimodal_encoder.get_modality_statistics()
        print(f"\n📈 Multimodal Statistics:")
        print(f"  - Available modalities: {stats['available_modalities']}")
        print(f"  - Fusion strategy: {stats['fusion_strategy']}")
        print(f"  - Target dimension: {stats['target_dimension']}")

    async def demo_meta_learning(self):
        """Demonstrate meta-learning and domain adaptation."""
        
        print("\n🎓 === META-LEARNING DEMONSTRATION ===")
        
        if not self.foundation.meta_learning_engine:
            print("⚠️ Meta-learning engine not available - skipping demo")
            return
        
        # Create training examples for different domains
        electronics_examples = [
            {
                "content": "Wireless bluetooth headphones with active noise cancellation",
                "class": "headphones",
                "embedding": np.random.randn(512).tolist(),  # Mock embedding
                "conceptual_coords": [0.8, 0.6, 0.9, 0.7, 0.5, 0.8, 0.6, 0.7, 0.9, 0.5]
            },
            {
                "content": "Professional studio monitor speakers for audio production",
                "class": "speakers", 
                "embedding": np.random.randn(512).tolist(),
                "conceptual_coords": [0.9, 0.8, 0.7, 0.8, 0.6, 0.9, 0.7, 0.8, 0.8, 0.6]
            },
            {
                "content": "High-resolution audio interface for recording",
                "class": "audio_interface",
                "embedding": np.random.randn(512).tolist(),
                "conceptual_coords": [0.7, 0.9, 0.8, 0.9, 0.7, 0.8, 0.8, 0.9, 0.7, 0.8]
            }
        ]
        
        # Learn electronics domain
        print("📚 Learning electronics domain...")
        
        electronics_result = await self.foundation.adapt_to_domain(
            domain="electronics",
            examples=electronics_examples,
            strategy=AdaptationStrategy.MAML
        )
        
        print(f"✅ Electronics adaptation:")
        print(f"  - Accuracy: {electronics_result.validation_accuracy:.3f}")
        print(f"  - Loss: {electronics_result.adaptation_loss:.3f}")
        print(f"  - Success: {electronics_result.is_successful()}")
        
        # Create examples for a related domain (audio equipment)
        audio_examples = [
            {
                "content": "Professional microphone for podcast recording",
                "class": "microphone",
                "embedding": np.random.randn(512).tolist(),
                "conceptual_coords": [0.8, 0.7, 0.9, 0.8, 0.6, 0.7, 0.8, 0.9, 0.7, 0.6]
            },
            {
                "content": "Digital audio workstation software for music production",
                "class": "software",
                "embedding": np.random.randn(512).tolist(),
                "conceptual_coords": [0.6, 0.8, 0.7, 0.9, 0.8, 0.6, 0.9, 0.8, 0.7, 0.9]
            }
        ]
        
        # Transfer knowledge from electronics to audio domain
        print("\n🔄 Transferring knowledge to audio domain...")
        
        transfer_result = await self.foundation.transfer_knowledge_between_domains(
            source_domain="electronics",
            target_domain="audio_equipment", 
            target_examples=audio_examples,
            strategy=AdaptationStrategy.MAML
        )
        
        print(f"✅ Knowledge transfer:")
        print(f"  - Accuracy: {transfer_result.validation_accuracy:.3f}")
        print(f"  - Transfer success: {transfer_result.performance_metrics.get('transfer_success', False)}")
        
        # Demonstrate few-shot classification
        print("\n🎯 Few-shot classification...")
        
        query_embedding = np.random.randn(512)
        
        classifications = await self.foundation.meta_learning_engine.few_shot_classify(
            query_embedding=query_embedding,
            domain="electronics",
            top_k=3
        )
        
        print("Classification results:")
        for class_name, confidence in classifications:
            print(f"  - {class_name}: {confidence:.3f}")
        
        # Get domain statistics
        domain_stats = await self.foundation.meta_learning_engine.get_domain_statistics()
        print(f"\n📊 Domain Statistics:")
        print(f"  - Total domains: {domain_stats['total_domains']}")
        print(f"  - Domains: {domain_stats['domains']}")
        print(f"  - Average accuracy: {domain_stats['average_accuracy']:.3f}")

    async def demo_universal_integration(self):
        """Demonstrate the integrated universal foundation capabilities."""
        
        print("\n🌐 === UNIVERSAL INTEGRATION DEMONSTRATION ===")
        
        # Add universal knowledge items that combine all capabilities
        universal_items = [
            {
                "id": "universal_1",
                "content": "Premium noise-cancelling headphones for travel",
                "modality_data": {
                    ModalityType.TEXT: "Premium noise-cancelling headphones for travel",
                    ModalityType.STRUCTURED: {
                        "category": "headphones",
                        "features": ["noise_cancellation", "travel", "premium"],
                        "price_range": "high"
                    }
                },
                "domain": "travel_electronics"
            },
            {
                "id": "universal_2", 
                "content": "Portable bluetooth speaker for outdoor activities",
                "modality_data": {
                    ModalityType.TEXT: "Portable bluetooth speaker for outdoor activities",
                    ModalityType.STRUCTURED: {
                        "category": "speakers",
                        "features": ["portable", "bluetooth", "outdoor", "waterproof"],
                        "use_case": "outdoor"
                    }
                },
                "domain": "outdoor_electronics"
            }
        ]
        
        # Add universal knowledge items
        print("📝 Adding universal knowledge items...")
        
        for item in universal_items:
            try:
                await self.foundation.add_universal_knowledge_item(
                    item_id=item["id"],
                    content=item["content"],
                    modality_data=item["modality_data"],
                    domain=item["domain"],
                    importance=MemoryImportance.HIGH
                )
                print(f"✅ Added universal item: {item['id']}")
                
            except Exception as e:
                print(f"❌ Failed to add {item['id']}: {e}")
        
        # Perform cross-modal search
        print("\n🔍 Cross-modal search...")
        
        try:
            search_results = await self.foundation.cross_modal_search(
                query_data={
                    ModalityType.TEXT: "wireless audio device for travel",
                    ModalityType.STRUCTURED: {
                        "use_case": "travel",
                        "type": "audio"
                    }
                },
                similarity_threshold=0.3,
                top_k=5
            )
            
            print(f"Found {len(search_results)} cross-modal matches:")
            for item_id, scores in search_results:
                print(f"  - {item_id}: overall={scores['overall_score']:.3f}")
                
        except Exception as e:
            print(f"❌ Cross-modal search failed: {e}")
        
        # Demonstrate temporal reasoning
        print("\n⏰ Temporal reasoning query...")
        
        temporal_result = await self.foundation.temporal_reasoning_query(
            query="headphones",
            include_patterns=True
        )
        
        print(f"Temporal query results:")
        print(f"  - Memory traces: {len(temporal_result.get('memory_traces', []))}")
        print(f"  - Temporal relationships: {len(temporal_result.get('temporal_relationships', []))}")
        print(f"  - Temporal patterns: {len(temporal_result.get('temporal_patterns', []))}")
        
        # Learn from interaction
        print("\n🎓 Learning from interaction...")
        
        await self.foundation.learn_from_interaction(
            query_data={ModalityType.TEXT: "travel headphones"},
            selected_results=["universal_1"],
            feedback_score=0.95,
            context={"interaction_type": "cross_modal_search"}
        )
        
        print("✅ Learned from user interaction")

    async def demo_foundation_status(self):
        """Show comprehensive foundation status."""
        
        print("\n📊 === FOUNDATION STATUS ===")
        
        status = await self.foundation.get_foundation_status()
        
        print("🏗️ Foundation Layer Status:")
        print(f"  - Active capabilities: {status['active_capabilities']}")
        print(f"  - Universal knowledge items: {status['universal_knowledge_items']}")
        print(f"  - Domain adaptations: {status['domain_adaptations']}")
        
        print("\n📈 Performance Metrics:")
        for metric, value in status['performance_metrics'].items():
            print(f"  - {metric}: {value}")
        
        if 'persistent_memory' in status:
            print("\n🧠 Persistent Memory:")
            pm_status = status['persistent_memory']
            print(f"  - Session ID: {pm_status['session_id']}")
            print(f"  - Memory traces: {pm_status['memory_traces']}")
            print(f"  - Temporal patterns: {pm_status['temporal_patterns']}")
        
        if 'multimodal_encoder' in status:
            print("\n🎨 Multimodal Encoder:")
            mm_status = status['multimodal_encoder']
            print(f"  - Available modalities: {mm_status['available_modalities']}")
            print(f"  - Fusion strategy: {mm_status['fusion_strategy']}")
        
        if 'meta_learning' in status:
            print("\n🎓 Meta-Learning:")
            ml_status = status['meta_learning']
            print(f"  - Total domains: {ml_status['total_domains']}")
            print(f"  - Average accuracy: {ml_status.get('average_accuracy', 0):.3f}")

    async def run_complete_demo(self):
        """Run the complete foundation layer demonstration."""
        
        print("🌟 KSE Memory SDK - Foundation Layer Demo")
        print("=" * 50)
        
        try:
            # Initialize foundation
            await self.initialize_foundation()
            
            # Run individual demonstrations
            await self.demo_persistent_memory()
            await self.demo_multimodal_embeddings()
            await self.demo_meta_learning()
            await self.demo_universal_integration()
            await self.demo_foundation_status()
            
            print("\n✅ === DEMO COMPLETED SUCCESSFULLY ===")
            print("\n🎯 Key Achievements:")
            print("  ✅ Persistent memory with temporal knowledge graphs")
            print("  ✅ Multi-modal embeddings with cross-modal similarity")
            print("  ✅ Meta-learning with MAML-based domain adaptation")
            print("  ✅ Universal integration of all foundation capabilities")
            print("  ✅ Cross-session learning and knowledge consolidation")
            
            print("\n🚀 The Foundation Layer is ready for domain customizations!")
            
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()


async def main():
    """Main demonstration function."""
    
    demo = FoundationDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())