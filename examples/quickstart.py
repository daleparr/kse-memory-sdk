"""
KSE Memory SDK - Quick Start Example

This example demonstrates the basic usage of the KSE Memory SDK
for creating an intelligent product memory system.
"""

import asyncio
import logging
from typing import List

from kse_memory import (
    KSEMemory,
    KSEConfig,
    Product,
    SearchQuery,
    SearchType,
    ConceptualDimensions,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Main example function."""
    
    # 1. Create configuration
    config = KSEConfig(
        debug=True,
    )
    
    # Update configuration components
    # Note: API keys are automatically loaded from environment variables
    # Copy .env.example to .env and fill in your actual values
    
    config.vector_store.backend = "pinecone"
    # config.vector_store.api_key is loaded from PINECONE_API_KEY env var
    config.vector_store.environment = "us-west1-gcp"
    config.vector_store.index_name = "kse-quickstart"
    config.vector_store.dimension = 384
    
    config.graph_store.backend = "neo4j"
    config.graph_store.uri = "bolt://localhost:7687"
    config.graph_store.username = "neo4j"
    # config.graph_store.password is loaded from NEO4J_PASSWORD env var
    
    config.concept_store.backend = "postgresql"
    config.concept_store.host = "localhost"
    config.concept_store.port = 5432
    config.concept_store.database = "kse_concepts"
    config.concept_store.username = "postgres"
    # config.concept_store.password is loaded from POSTGRES_PASSWORD env var
    
    config.embedding.text_model = "sentence-transformers/all-MiniLM-L6-v2"
    # config.embedding.openai_api_key is loaded from OPENAI_API_KEY env var
    
    config.conceptual.auto_compute = True
    # config.conceptual.llm_api_key is loaded from OPENAI_API_KEY env var
    
    # 2. Initialize KSE Memory
    kse = KSEMemory(config)
    
    # 3. Connect to a data source (using generic adapter with sample data)
    sample_products = create_sample_products()
    
    def sample_data_source(limit=100, offset=0, **kwargs):
        """Sample data source function."""
        end_idx = offset + limit
        return sample_products[offset:end_idx]
    
    adapter_config = {
        "data_source": sample_data_source,
    }
    
    await kse.initialize("generic", adapter_config)
    
    # 4. Add products to the memory system
    logger.info("Adding products to KSE Memory...")
    
    for product in sample_products:
        await kse.add_product(product, compute_embeddings=True, compute_concepts=True)
        logger.info(f"Added product: {product.title}")
    
    # 5. Perform different types of searches
    logger.info("\n=== Search Examples ===")
    
    # Semantic search
    logger.info("\n1. Semantic Search:")
    semantic_results = await kse.search("comfortable running shoes")
    for result in semantic_results[:3]:
        logger.info(f"  - {result.product.title} (score: {result.score:.3f})")
    
    # Conceptual search
    logger.info("\n2. Conceptual Search:")
    conceptual_query = SearchQuery(
        query="elegant and luxurious items",
        search_type=SearchType.CONCEPTUAL,
        conceptual_weights={"elegance": 0.4, "luxury": 0.6}
    )
    conceptual_results = await kse.search(conceptual_query)
    for result in conceptual_results[:3]:
        logger.info(f"  - {result.product.title} (score: {result.score:.3f})")
    
    # Hybrid search (combines all approaches)
    logger.info("\n3. Hybrid Search:")
    hybrid_query = SearchQuery(
        query="stylish casual wear",
        search_type=SearchType.HYBRID,
        limit=5
    )
    hybrid_results = await kse.search(hybrid_query)
    for result in hybrid_results:
        logger.info(f"  - {result.product.title} (score: {result.score:.3f})")
        if result.explanation:
            logger.info(f"    Explanation: {result.explanation}")
    
    # 6. Get product recommendations
    logger.info("\n=== Recommendation Example ===")
    
    if sample_products:
        product_id = sample_products[0].id
        recommendations = await kse.get_recommendations(product_id, limit=3)
        
        logger.info(f"\nRecommendations for '{sample_products[0].title}':")
        for rec in recommendations:
            logger.info(f"  - {rec.product.title} (score: {rec.score:.3f})")
    
    # 7. Health check
    logger.info("\n=== System Health ===")
    health = await kse.health_check()
    logger.info(f"System Status: {'Healthy' if health['connected'] else 'Unhealthy'}")
    for component, status in health.get('components', {}).items():
        logger.info(f"  {component}: {status}")
    
    # 8. Cleanup
    await kse.disconnect()
    logger.info("\nKSE Memory disconnected successfully!")


def create_sample_products() -> List[Product]:
    """Create sample products for the demonstration."""
    
    return [
        Product(
            id="prod_001",
            title="Nike Air Max 270",
            description="Comfortable running shoes with excellent cushioning and breathable mesh upper. Perfect for daily runs and casual wear.",
            price=150.00,
            currency="USD",
            category="Footwear",
            brand="Nike",
            tags=["running", "comfortable", "breathable", "casual", "athletic"],
            images=["https://example.com/nike-air-max-270.jpg"],
            conceptual_dimensions=ConceptualDimensions(
                comfort=0.9,
                functionality=0.8,
                modernity=0.7,
                versatility=0.8,
                boldness=0.6
            )
        ),
        
        Product(
            id="prod_002", 
            title="Elegant Silk Dress",
            description="Luxurious silk dress with elegant draping and sophisticated design. Perfect for formal events and special occasions.",
            price=299.99,
            currency="USD",
            category="Clothing",
            brand="Luxury Fashion Co",
            tags=["elegant", "silk", "formal", "luxury", "sophisticated"],
            images=["https://example.com/silk-dress.jpg"],
            conceptual_dimensions=ConceptualDimensions(
                elegance=0.95,
                luxury=0.9,
                modernity=0.7,
                minimalism=0.8,
                seasonality=0.6
            )
        ),
        
        Product(
            id="prod_003",
            title="Minimalist Watch",
            description="Clean, minimalist watch design with premium materials. Timeless style that complements any outfit.",
            price=199.00,
            currency="USD", 
            category="Accessories",
            brand="Modern Time",
            tags=["minimalist", "watch", "timeless", "premium", "accessories"],
            images=["https://example.com/minimalist-watch.jpg"],
            conceptual_dimensions=ConceptualDimensions(
                minimalism=0.95,
                elegance=0.8,
                modernity=0.9,
                luxury=0.7,
                versatility=0.9
            )
        ),
        
        Product(
            id="prod_004",
            title="Bold Statement Jacket",
            description="Eye-catching jacket with unique design elements. Makes a bold fashion statement while maintaining functionality.",
            price=179.99,
            currency="USD",
            category="Clothing", 
            brand="Urban Style",
            tags=["bold", "statement", "unique", "fashion", "jacket"],
            images=["https://example.com/statement-jacket.jpg"],
            conceptual_dimensions=ConceptualDimensions(
                boldness=0.95,
                modernity=0.8,
                innovation=0.7,
                functionality=0.6,
                versatility=0.5
            )
        ),
        
        Product(
            id="prod_005",
            title="Comfortable Lounge Set",
            description="Ultra-soft lounge set perfect for relaxation. Combines comfort with style for the ultimate at-home experience.",
            price=89.99,
            currency="USD",
            category="Clothing",
            brand="Comfort Plus",
            tags=["comfortable", "lounge", "soft", "relaxation", "casual"],
            images=["https://example.com/lounge-set.jpg"],
            conceptual_dimensions=ConceptualDimensions(
                comfort=0.95,
                functionality=0.8,
                versatility=0.7,
                minimalism=0.6,
                seasonality=0.8
            )
        )
    ]


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())