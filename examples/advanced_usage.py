"""
KSE Memory SDK - Advanced Usage Examples

This example demonstrates advanced features of the KSE Memory SDK
including custom adapters, advanced search, and performance optimization.
"""

import asyncio
import logging
from typing import List, Dict, Any
from datetime import datetime

from kse_memory import (
    KSEMemory,
    KSEConfig,
    Product,
    SearchQuery,
    SearchType,
    ConceptualDimensions,
)
from kse_memory.adapters import GenericAdapter
from kse_memory.core.config import (
    VectorStoreConfig,
    EmbeddingConfig,
    ConceptualConfig,
    SearchConfig,
    CacheConfig,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def advanced_configuration_example():
    """Demonstrate advanced configuration options."""
    logger.info("=== Advanced Configuration Example ===")
    
    # Create advanced configuration
    config = KSEConfig(
        app_name="Advanced KSE Demo",
        debug=True,
        
        # Vector store with Weaviate
        vector_store=VectorStoreConfig(
            backend="weaviate",
            host="localhost",
            port=8080,
            index_name="advanced-products",
            dimension=768,  # Larger dimension for better accuracy
            metric="cosine"
        ),
        
        # Advanced embedding configuration
        embedding=EmbeddingConfig(
            text_model="sentence-transformers/all-mpnet-base-v2",  # Better model
            batch_size=16,
            max_length=512,
            normalize=True
        ),
        
        # Conceptual configuration with LLM
        conceptual=ConceptualConfig(
            auto_compute=True,
            llm_model="gpt-4",
            llm_api_key="your-openai-api-key"
        ),
        
        # Advanced search configuration
        search=SearchConfig(
            default_limit=20,
            max_limit=200,
            hybrid_weights={
                "embedding": 0.5,
                "conceptual": 0.3,
                "knowledge_graph": 0.2
            },
            similarity_threshold=0.75,
            enable_reranking=True
        ),
        
        # Redis cache for performance
        cache=CacheConfig(
            enabled=True,
            backend="redis",
            host="localhost",
            port=6379,
            ttl=1800  # 30 minutes
        ),
        
        # Performance settings
        max_workers=8,
        batch_processing=True
    )
    
    logger.info(f"Configuration created: {config.app_name}")
    logger.info(f"Vector store: {config.vector_store.backend}")
    logger.info(f"Embedding model: {config.embedding.text_model}")
    logger.info(f"Cache enabled: {config.cache.enabled}")


async def custom_adapter_example():
    """Demonstrate creating and using a custom adapter."""
    logger.info("=== Custom Adapter Example ===")
    
    # Simulate an external API
    class MockProductAPI:
        def __init__(self):
            self.products = [
                {
                    "id": "api_001",
                    "name": "Premium Headphones",
                    "description": "High-quality wireless headphones with noise cancellation",
                    "price": 299.99,
                    "category": "Electronics",
                    "brand": "AudioTech",
                    "attributes": {
                        "wireless": True,
                        "noise_cancelling": True,
                        "battery_life": "30 hours"
                    }
                },
                {
                    "id": "api_002",
                    "name": "Ergonomic Office Chair",
                    "description": "Comfortable office chair with lumbar support and adjustable height",
                    "price": 449.99,
                    "category": "Furniture",
                    "brand": "OfficeComfort",
                    "attributes": {
                        "adjustable": True,
                        "lumbar_support": True,
                        "material": "mesh"
                    }
                }
            ]
        
        async def get_products(self, limit=100, offset=0):
            """Simulate API call to get products."""
            await asyncio.sleep(0.1)  # Simulate network delay
            end_idx = offset + limit
            return self.products[offset:end_idx]
        
        async def get_product(self, product_id):
            """Simulate API call to get specific product."""
            await asyncio.sleep(0.05)
            for product in self.products:
                if product["id"] == product_id:
                    return product
            return None
    
    # Create custom data source using the API
    api = MockProductAPI()
    
    async def custom_data_source(limit=100, offset=0, product_id=None, **kwargs):
        """Custom data source that converts API data to KSE format."""
        if product_id:
            api_product = await api.get_product(product_id)
            if api_product:
                return [convert_api_product(api_product)]
            return []
        else:
            api_products = await api.get_products(limit, offset)
            return [convert_api_product(p) for p in api_products]
    
    def convert_api_product(api_product: Dict[str, Any]) -> Dict[str, Any]:
        """Convert API product format to KSE Product format."""
        return {
            "id": api_product["id"],
            "title": api_product["name"],
            "description": api_product["description"],
            "price": api_product["price"],
            "currency": "USD",
            "category": api_product["category"],
            "brand": api_product["brand"],
            "tags": list(api_product.get("attributes", {}).keys()),
            "metadata": {
                "api_source": True,
                "attributes": api_product.get("attributes", {})
            }
        }
    
    # Use custom adapter
    config = KSEConfig(debug=True)
    kse = KSEMemory(config)
    
    adapter_config = {
        "data_source": custom_data_source,
    }
    
    await kse.initialize("generic", adapter_config)
    
    try:
        # Sync products from custom API
        synced_count = await kse.sync_products()
        logger.info(f"Synced {synced_count} products from custom API")
        
        # Search the synced products
        results = await kse.search("comfortable chair")
        logger.info(f"Found {len(results)} results for 'comfortable chair'")
        
        for result in results:
            logger.info(f"  - {result.product.title} (score: {result.score:.3f})")
    
    finally:
        await kse.disconnect()


async def advanced_search_example():
    """Demonstrate advanced search capabilities."""
    logger.info("=== Advanced Search Example ===")
    
    # Create products with rich conceptual dimensions
    products = [
        Product(
            id="adv_001",
            title="Luxury Silk Scarf",
            description="Handcrafted silk scarf with intricate patterns. Perfect for elegant occasions.",
            price=189.99,
            category="Accessories",
            brand="LuxuryBrand",
            tags=["silk", "handcrafted", "elegant", "luxury"],
            conceptual_dimensions=ConceptualDimensions(
                elegance=0.95,
                luxury=0.9,
                minimalism=0.3,
                boldness=0.7,
                versatility=0.8
            )
        ),
        Product(
            id="adv_002",
            title="Minimalist Desk Lamp",
            description="Clean, modern desk lamp with adjustable brightness. Perfect for any workspace.",
            price=79.99,
            category="Lighting",
            brand="ModernDesign",
            tags=["minimalist", "modern", "adjustable", "workspace"],
            conceptual_dimensions=ConceptualDimensions(
                minimalism=0.95,
                modernity=0.9,
                functionality=0.85,
                elegance=0.7,
                innovation=0.6
            )
        ),
        Product(
            id="adv_003",
            title="Bold Statement Necklace",
            description="Eye-catching statement necklace that makes a bold fashion statement.",
            price=129.99,
            category="Jewelry",
            brand="BoldFashion",
            tags=["bold", "statement", "jewelry", "fashion"],
            conceptual_dimensions=ConceptualDimensions(
                boldness=0.95,
                elegance=0.6,
                luxury=0.7,
                modernity=0.8,
                versatility=0.4
            )
        )
    ]
    
    # Setup KSE Memory
    config = KSEConfig(debug=True)
    kse = KSEMemory(config)
    
    def sample_data_source(limit=100, offset=0, **kwargs):
        end_idx = offset + limit
        return [p.to_dict() for p in products[offset:end_idx]]
    
    await kse.initialize("generic", {"data_source": sample_data_source})
    
    try:
        # Add products
        for product in products:
            await kse.add_product(product, compute_embeddings=False, compute_concepts=False)
        
        # 1. Semantic Search
        logger.info("\n1. Semantic Search:")
        semantic_results = await kse.search(SearchQuery(
            query="elegant accessories for special occasions",
            search_type=SearchType.SEMANTIC,
            limit=5
        ))
        
        for result in semantic_results:
            logger.info(f"  - {result.product.title} (score: {result.score:.3f})")
        
        # 2. Conceptual Search with Custom Weights
        logger.info("\n2. Conceptual Search (Elegance + Luxury):")
        conceptual_results = await kse.search(SearchQuery(
            query="elegant luxury items",
            search_type=SearchType.CONCEPTUAL,
            conceptual_weights={
                "elegance": 0.6,
                "luxury": 0.8,
                "boldness": 0.2
            },
            limit=5
        ))
        
        for result in conceptual_results:
            logger.info(f"  - {result.product.title} (score: {result.score:.3f})")
            if result.conceptual_similarity:
                logger.info(f"    Conceptual similarity: {result.conceptual_similarity:.3f}")
        
        # 3. Filtered Search
        logger.info("\n3. Filtered Search (Accessories under $150):")
        filtered_results = await kse.search(SearchQuery(
            query="stylish accessories",
            search_type=SearchType.HYBRID,
            filters={
                "category": "Accessories",
                "price_max": 150.0
            },
            limit=5
        ))
        
        for result in filtered_results:
            logger.info(f"  - {result.product.title} (${result.product.price}) (score: {result.score:.3f})")
        
        # 4. Multi-Category Search
        logger.info("\n4. Multi-Category Search:")
        multi_results = await kse.search(SearchQuery(
            query="modern design items",
            search_type=SearchType.HYBRID,
            filters={
                "category": ["Lighting", "Accessories", "Jewelry"]
            },
            limit=5
        ))
        
        for result in multi_results:
            logger.info(f"  - {result.product.title} ({result.product.category}) (score: {result.score:.3f})")
    
    finally:
        await kse.disconnect()


async def performance_optimization_example():
    """Demonstrate performance optimization techniques."""
    logger.info("=== Performance Optimization Example ===")
    
    # Create a larger dataset for performance testing
    def generate_large_dataset(size: int) -> List[Product]:
        """Generate a large dataset for performance testing."""
        products = []
        categories = ["Electronics", "Clothing", "Accessories", "Home", "Sports"]
        brands = ["BrandA", "BrandB", "BrandC", "BrandD", "BrandE"]
        
        for i in range(size):
            product = Product(
                id=f"perf_{i:04d}",
                title=f"Product {i}",
                description=f"Description for product {i} with various features and benefits.",
                price=round(10.0 + (i % 500), 2),
                category=categories[i % len(categories)],
                brand=brands[i % len(brands)],
                tags=[f"tag{i%10}", f"feature{i%5}", f"type{i%3}"],
                conceptual_dimensions=ConceptualDimensions(
                    elegance=round((i % 100) / 100.0, 2),
                    comfort=round(((i + 10) % 100) / 100.0, 2),
                    luxury=round(((i + 20) % 100) / 100.0, 2),
                    functionality=round(((i + 30) % 100) / 100.0, 2)
                )
            )
            products.append(product)
        
        return products
    
    # Generate dataset
    dataset_size = 1000
    large_dataset = generate_large_dataset(dataset_size)
    logger.info(f"Generated dataset with {len(large_dataset)} products")
    
    # Configure for performance
    config = KSEConfig(
        debug=False,  # Disable debug for performance
        max_workers=8,
        batch_processing=True,
        embedding=EmbeddingConfig(
            batch_size=32,  # Larger batch size
        ),
        cache=CacheConfig(
            enabled=True,
            backend="memory",  # Use memory cache for demo
            ttl=600
        )
    )
    
    kse = KSEMemory(config)
    
    def large_data_source(limit=100, offset=0, **kwargs):
        end_idx = offset + limit
        return [p.to_dict() for p in large_dataset[offset:end_idx]]
    
    await kse.initialize("generic", {"data_source": large_data_source})
    
    try:
        # Batch product addition
        logger.info("Adding products in batches...")
        start_time = datetime.now()
        
        batch_size = 50
        for i in range(0, len(large_dataset), batch_size):
            batch = large_dataset[i:i + batch_size]
            
            # Add batch concurrently
            tasks = [
                kse.add_product(product, compute_embeddings=False, compute_concepts=False)
                for product in batch
            ]
            await asyncio.gather(*tasks)
            
            if (i + batch_size) % 200 == 0:
                logger.info(f"Added {i + batch_size} products...")
        
        add_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Added {len(large_dataset)} products in {add_time:.2f} seconds")
        
        # Performance testing with different search types
        search_queries = [
            "comfortable products",
            "luxury items",
            "modern design",
            "functional accessories",
            "elegant clothing"
        ]
        
        logger.info("\nPerformance testing searches...")
        
        for query in search_queries:
            start_time = datetime.now()
            
            results = await kse.search(SearchQuery(
                query=query,
                search_type=SearchType.HYBRID,
                limit=20
            ))
            
            search_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Query '{query}': {len(results)} results in {search_time:.3f}s")
        
        # Test caching performance
        logger.info("\nTesting cache performance...")
        
        # First search (cache miss)
        start_time = datetime.now()
        results1 = await kse.search("test query for caching")
        first_time = (datetime.now() - start_time).total_seconds()
        
        # Second search (cache hit)
        start_time = datetime.now()
        results2 = await kse.search("test query for caching")
        second_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"First search: {first_time:.3f}s")
        logger.info(f"Second search (cached): {second_time:.3f}s")
        logger.info(f"Cache speedup: {first_time/second_time:.1f}x")
    
    finally:
        await kse.disconnect()


async def real_time_updates_example():
    """Demonstrate real-time product updates and webhooks."""
    logger.info("=== Real-Time Updates Example ===")
    
    # Simulate a dynamic product catalog
    class DynamicCatalog:
        def __init__(self):
            self.products = {}
            self.version = 0
        
        def add_product(self, product: Product):
            self.products[product.id] = product
            self.version += 1
            logger.info(f"Added product: {product.title}")
        
        def update_product(self, product_id: str, updates: Dict[str, Any]):
            if product_id in self.products:
                product = self.products[product_id]
                for key, value in updates.items():
                    if hasattr(product, key):
                        setattr(product, key, value)
                self.version += 1
                logger.info(f"Updated product: {product.title}")
        
        def remove_product(self, product_id: str):
            if product_id in self.products:
                product = self.products.pop(product_id)
                self.version += 1
                logger.info(f"Removed product: {product.title}")
        
        def get_products(self, limit=100, offset=0):
            products_list = list(self.products.values())
            end_idx = offset + limit
            return [p.to_dict() for p in products_list[offset:end_idx]]
    
    # Create dynamic catalog
    catalog = DynamicCatalog()
    
    # Add initial products
    initial_products = [
        Product(
            id="rt_001",
            title="Smart Watch",
            description="Advanced smartwatch with health monitoring",
            price=299.99,
            category="Electronics",
            brand="TechBrand"
        ),
        Product(
            id="rt_002",
            title="Wireless Earbuds",
            description="High-quality wireless earbuds with noise cancellation",
            price=149.99,
            category="Electronics",
            brand="AudioBrand"
        )
    ]
    
    for product in initial_products:
        catalog.add_product(product)
    
    # Setup KSE Memory
    config = KSEConfig(debug=True)
    kse = KSEMemory(config)
    
    await kse.initialize("generic", {"data_source": catalog.get_products})
    
    try:
        # Initial sync
        synced_count = await kse.sync_products()
        logger.info(f"Initial sync: {synced_count} products")
        
        # Simulate real-time updates
        logger.info("\nSimulating real-time updates...")
        
        # Add new product
        new_product = Product(
            id="rt_003",
            title="Bluetooth Speaker",
            description="Portable Bluetooth speaker with excellent sound quality",
            price=79.99,
            category="Electronics",
            brand="AudioBrand"
        )
        catalog.add_product(new_product)
        await kse.add_product(new_product)
        
        # Update existing product
        catalog.update_product("rt_001", {"price": 279.99})
        updated_product = Product.from_dict(catalog.products["rt_001"].to_dict())
        await kse.add_product(updated_product)  # Re-add with updates
        
        # Search after updates
        results = await kse.search("bluetooth audio")
        logger.info(f"\nSearch results after updates:")
        for result in results:
            logger.info(f"  - {result.product.title} (${result.product.price})")
        
        # Simulate webhook handling
        webhook_events = [
            {"type": "product.created", "product_id": "rt_003"},
            {"type": "product.updated", "product_id": "rt_001"},
        ]
        
        logger.info("\nProcessing webhook events...")
        for event in webhook_events:
            # In a real scenario, this would be called by webhook handler
            logger.info(f"Processed webhook: {event['type']} for {event['product_id']}")
    
    finally:
        await kse.disconnect()


async def main():
    """Run all advanced examples."""
    logger.info("KSE Memory SDK - Advanced Usage Examples")
    logger.info("=" * 50)
    
    try:
        await advanced_configuration_example()
        await custom_adapter_example()
        await advanced_search_example()
        await performance_optimization_example()
        await real_time_updates_example()
        
        logger.info("\n" + "=" * 50)
        logger.info("All advanced examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Example failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())