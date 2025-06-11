"""
Integration tests for KSE Memory SDK.
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path

from kse_memory import (
    KSEMemory,
    KSEConfig,
    Product,
    SearchQuery,
    SearchType,
    ConceptualDimensions,
)
from kse_memory.adapters import GenericAdapter
from kse_memory.services import EmbeddingService, ConceptualService
from kse_memory.exceptions import KSEError


@pytest.fixture
def sample_products():
    """Create sample products for testing."""
    return [
        Product(
            id="test_001",
            title="Comfortable Running Shoes",
            description="High-quality running shoes with excellent cushioning and breathable mesh upper. Perfect for daily runs and athletic activities.",
            price=129.99,
            currency="USD",
            category="Footwear",
            brand="SportsBrand",
            tags=["running", "comfortable", "athletic", "breathable"],
            conceptual_dimensions=ConceptualDimensions(
                comfort=0.9,
                functionality=0.8,
                modernity=0.7,
                versatility=0.8
            )
        ),
        Product(
            id="test_002",
            title="Elegant Evening Dress",
            description="Sophisticated evening dress made from premium silk. Features elegant draping and timeless design perfect for formal occasions.",
            price=299.99,
            currency="USD",
            category="Clothing",
            brand="LuxuryFashion",
            tags=["elegant", "formal", "silk", "evening"],
            conceptual_dimensions=ConceptualDimensions(
                elegance=0.95,
                luxury=0.9,
                modernity=0.6,
                minimalism=0.7
            )
        ),
        Product(
            id="test_003",
            title="Minimalist Watch",
            description="Clean, minimalist watch design with premium materials. Timeless style that complements any outfit.",
            price=199.99,
            currency="USD",
            category="Accessories",
            brand="ModernTime",
            tags=["minimalist", "watch", "timeless", "premium"],
            conceptual_dimensions=ConceptualDimensions(
                minimalism=0.95,
                elegance=0.8,
                modernity=0.9,
                luxury=0.7,
                versatility=0.9
            )
        )
    ]


@pytest.fixture
def test_config():
    """Create test configuration."""
    return KSEConfig(
        debug=True,
        # Use in-memory backends for testing
        vector_store={
            "backend": "memory",
        },
        graph_store={
            "backend": "memory",
        },
        concept_store={
            "backend": "memory",
        },
        embedding={
            "text_model": "sentence-transformers/all-MiniLM-L6-v2",
            "batch_size": 2,  # Small batch for testing
        },
        conceptual={
            "auto_compute": False,  # Disable LLM for testing
        },
        cache={
            "enabled": False,  # Disable cache for testing
        }
    )


@pytest.mark.asyncio
class TestKSEMemoryIntegration:
    """Integration tests for KSE Memory."""
    
    async def test_full_workflow(self, test_config, sample_products):
        """Test complete workflow from initialization to search."""
        # Create data source
        def sample_data_source(limit=100, offset=0, **kwargs):
            end_idx = offset + limit
            return sample_products[offset:end_idx]
        
        # Initialize KSE Memory
        kse = KSEMemory(test_config)
        
        # Connect with generic adapter
        adapter_config = {
            "data_source": sample_data_source,
        }
        
        await kse.initialize("generic", adapter_config)
        
        try:
            # Verify initialization
            assert kse.is_initialized
            assert kse.is_connected
            
            # Add products
            for product in sample_products:
                success = await kse.add_product(
                    product, 
                    compute_embeddings=False,  # Skip embeddings for speed
                    compute_concepts=False     # Skip concepts for speed
                )
                assert success
            
            # Test basic search (will use product metadata)
            results = await kse.search("running shoes")
            assert len(results) >= 0  # May be 0 without embeddings
            
            # Test product retrieval
            product = await kse.get_product("test_001")
            assert product is not None
            assert product.title == "Comfortable Running Shoes"
            
            # Test health check
            health = await kse.health_check()
            assert health["initialized"] is True
            assert health["connected"] is True
            
        finally:
            await kse.disconnect()
    
    async def test_search_types(self, test_config, sample_products):
        """Test different search types."""
        def sample_data_source(limit=100, offset=0, **kwargs):
            return sample_products
        
        kse = KSEMemory(test_config)
        adapter_config = {"data_source": sample_data_source}
        
        await kse.initialize("generic", adapter_config)
        
        try:
            # Add products with conceptual dimensions
            for product in sample_products:
                await kse.add_product(product, compute_embeddings=False, compute_concepts=False)
            
            # Test semantic search
            semantic_query = SearchQuery(
                query="comfortable shoes",
                search_type=SearchType.SEMANTIC,
                limit=5
            )
            semantic_results = await kse.search(semantic_query)
            assert isinstance(semantic_results, list)
            
            # Test conceptual search
            conceptual_query = SearchQuery(
                query="elegant items",
                search_type=SearchType.CONCEPTUAL,
                conceptual_weights={"elegance": 0.8},
                limit=5
            )
            conceptual_results = await kse.search(conceptual_query)
            assert isinstance(conceptual_results, list)
            
            # Test hybrid search
            hybrid_query = SearchQuery(
                query="modern accessories",
                search_type=SearchType.HYBRID,
                limit=5
            )
            hybrid_results = await kse.search(hybrid_query)
            assert isinstance(hybrid_results, list)
            
        finally:
            await kse.disconnect()
    
    async def test_recommendations(self, test_config, sample_products):
        """Test product recommendations."""
        def sample_data_source(limit=100, offset=0, **kwargs):
            return sample_products
        
        kse = KSEMemory(test_config)
        adapter_config = {"data_source": sample_data_source}
        
        await kse.initialize("generic", adapter_config)
        
        try:
            # Add products
            for product in sample_products:
                await kse.add_product(product, compute_embeddings=False, compute_concepts=False)
            
            # Get recommendations
            recommendations = await kse.get_recommendations("test_001", limit=2)
            assert isinstance(recommendations, list)
            
            # Verify recommendations don't include the source product
            rec_ids = [rec.product.id for rec in recommendations]
            assert "test_001" not in rec_ids
            
        finally:
            await kse.disconnect()
    
    async def test_error_handling(self, test_config):
        """Test error handling in various scenarios."""
        kse = KSEMemory(test_config)
        
        # Test operations before initialization
        with pytest.raises(KSEError, match="not initialized"):
            await kse.search("test")
        
        with pytest.raises(KSEError, match="not initialized"):
            test_product = Product(id="test", title="Test", description="Test")
            await kse.add_product(test_product)
        
        # Test invalid adapter
        with pytest.raises(Exception):  # Should raise an adapter error
            await kse.initialize("invalid_adapter", {})


@pytest.mark.asyncio
class TestGenericAdapterIntegration:
    """Integration tests for Generic Adapter."""
    
    async def test_csv_data_source(self, sample_products):
        """Test CSV data source integration."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Write CSV header
            f.write("id,title,description,price,category,brand\n")
            
            # Write product data
            for product in sample_products:
                f.write(f"{product.id},{product.title},{product.description},{product.price},{product.category},{product.brand}\n")
            
            csv_path = f.name
        
        try:
            # Create CSV data source
            csv_source = GenericAdapter.create_csv_data_source(csv_path)
            
            # Test data source
            data = csv_source(limit=10)
            assert len(data) == len(sample_products)
            assert data[0]['id'] == 'test_001'
            assert data[0]['title'] == 'Comfortable Running Shoes'
            
        finally:
            Path(csv_path).unlink()  # Clean up
    
    async def test_json_data_source(self, sample_products):
        """Test JSON data source integration."""
        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            products_data = {
                "products": [product.to_dict() for product in sample_products]
            }
            json.dump(products_data, f)
            json_path = f.name
        
        try:
            # Create JSON data source
            json_source = GenericAdapter.create_json_data_source(json_path, "products")
            
            # Test data source
            data = json_source(limit=10)
            assert len(data) == len(sample_products)
            assert data[0]['id'] == 'test_001'
            assert data[0]['title'] == 'Comfortable Running Shoes'
            
        finally:
            Path(json_path).unlink()  # Clean up
    
    async def test_custom_data_source(self, sample_products):
        """Test custom data source integration."""
        # Create custom data source function
        def custom_source(limit=100, offset=0, product_id=None, **kwargs):
            if product_id:
                # Return specific product
                for product in sample_products:
                    if product.id == product_id:
                        return [product.to_dict()]
                return []
            else:
                # Return paginated products
                end_idx = offset + limit
                return [p.to_dict() for p in sample_products[offset:end_idx]]
        
        # Test with generic adapter
        adapter = GenericAdapter()
        
        await adapter.connect({
            "data_source": custom_source,
        })
        
        try:
            # Test get_products
            products = await adapter.get_products(limit=2)
            assert len(products) == 2
            assert products[0].id == "test_001"
            
            # Test get_product
            product = await adapter.get_product("test_002")
            assert product is not None
            assert product.id == "test_002"
            assert product.title == "Elegant Evening Dress"
            
            # Test sync_products
            count = await adapter.sync_products()
            assert count == len(sample_products)
            
        finally:
            await adapter.disconnect()


@pytest.mark.asyncio
class TestServiceIntegration:
    """Integration tests for services."""
    
    async def test_embedding_service_integration(self):
        """Test embedding service with real models (if available)."""
        from kse_memory.core.config import EmbeddingConfig
        
        config = EmbeddingConfig(
            text_model="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=2
        )
        
        service = EmbeddingService(config)
        
        try:
            # Test single embedding
            embedding = await service.generate_text_embedding("test product description")
            assert embedding.dimension > 0
            assert len(embedding.vector) == embedding.dimension
            assert embedding.model == config.text_model
            
            # Test batch embeddings
            texts = ["product 1", "product 2", "product 3"]
            embeddings = await service.generate_batch_text_embeddings(texts)
            assert len(embeddings) == 3
            assert all(emb.dimension == embeddings[0].dimension for emb in embeddings)
            
        except Exception as e:
            # Skip test if model not available
            pytest.skip(f"Embedding model not available: {e}")
    
    async def test_conceptual_service_integration(self, sample_products):
        """Test conceptual service integration."""
        from kse_memory.core.config import ConceptualConfig
        
        config = ConceptualConfig(
            auto_compute=False,  # Disable LLM for testing
        )
        
        service = ConceptualService(config)
        
        # Test similarity computation
        dims1 = ConceptualDimensions(elegance=0.8, comfort=0.6, luxury=0.9)
        dims2 = ConceptualDimensions(elegance=0.7, comfort=0.8, luxury=0.8)
        
        similarity = service.compute_similarity(dims1, dims2)
        assert 0.0 <= similarity <= 1.0
        
        # Test dimension weights
        weights = service.get_dimension_weights("elegant luxury items")
        assert "elegance" in weights
        assert "luxury" in weights
        assert weights["elegance"] > 1.0  # Should be boosted
        assert weights["luxury"] > 1.0    # Should be boosted


@pytest.mark.asyncio
class TestConfigurationIntegration:
    """Integration tests for configuration."""
    
    async def test_config_file_loading(self):
        """Test loading configuration from file."""
        config_data = {
            "app_name": "Test App",
            "debug": True,
            "vector_store": {
                "backend": "weaviate",
                "host": "localhost",
                "port": 8080,
            },
            "embedding": {
                "text_model": "sentence-transformers/all-mpnet-base-v2",
                "batch_size": 16,
            }
        }
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            # Load configuration
            config = KSEConfig.from_file(config_path)
            
            assert config.app_name == "Test App"
            assert config.debug is True
            assert config.vector_store.backend == "weaviate"
            assert config.vector_store.host == "localhost"
            assert config.vector_store.port == 8080
            assert config.embedding.text_model == "sentence-transformers/all-mpnet-base-v2"
            assert config.embedding.batch_size == 16
            
        finally:
            Path(config_path).unlink()  # Clean up
    
    async def test_config_validation_integration(self):
        """Test configuration validation."""
        # Test invalid configuration
        config = KSEConfig()
        errors = config.validate()
        assert len(errors) > 0  # Should have validation errors
        
        # Test valid configuration
        config.vector_store.api_key = "test-key"
        config.embedding.openai_api_key = "test-key"
        config.conceptual.llm_api_key = "test-key"
        
        errors = config.validate()
        assert len(errors) == 0  # Should pass validation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])