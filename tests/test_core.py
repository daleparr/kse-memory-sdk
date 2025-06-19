"""
KSE Memory SDK - Core Tests

Comprehensive test suite for the core hybrid knowledge retrieval engine.
"""

import pytest
import asyncio
from typing import List

from kse_memory.core.memory import KSEMemory
from kse_memory.core.config import KSEConfig
from kse_memory.core.models import Product, SearchQuery, SearchType, ConceptualDimensions


class TestKSEMemoryCore:
    """Test suite for core KSE Memory functionality."""
    
    @pytest.fixture
    async def kse_memory(self):
        """Create KSE Memory instance for testing."""
        config = KSEConfig.from_dict({
            "debug": True,
            "vector_store": {"backend": "mock"},
            "graph_store": {"backend": "mock"},
            "concept_store": {"backend": "mock"}
        })
        kse = KSEMemory(config)
        
        # Mock data source for testing
        async def mock_data_source(**kwargs):
            return []
        
        def mock_converter(data):
            return data
        
        adapter_config = {
            "data_source": mock_data_source,
            "product_converter": mock_converter
        }
        
        await kse.initialize("generic", adapter_config)
        yield kse
        await kse.disconnect()
    
    @pytest.fixture
    def sample_products(self):
        """Create sample products for testing."""
        return [
            Product(
                id="test_001",
                title="Premium Running Shoes",
                description="Lightweight athletic shoes with responsive cushioning",
                price=129.99,
                category="Athletic Footwear",
                tags=["running", "athletic", "comfortable"],
                conceptual_dimensions=ConceptualDimensions(
                    comfort=0.9, functionality=0.95, modernity=0.8
                )
            ),
            Product(
                id="test_002",
                title="Elegant Evening Dress",
                description="Sophisticated silk dress for formal occasions",
                price=299.99,
                category="Formal Wear",
                tags=["elegant", "formal", "silk"],
                conceptual_dimensions=ConceptualDimensions(
                    elegance=0.95, luxury=0.9, comfort=0.6
                )
            ),
            Product(
                id="test_003",
                title="Casual Cotton T-Shirt",
                description="Basic cotton t-shirt for everyday wear",
                price=29.99,
                category="Casual Wear",
                tags=["casual", "cotton", "basic"],
                conceptual_dimensions=ConceptualDimensions(
                    comfort=0.8, versatility=0.95, minimalism=0.9
                )
            )
        ]
    
    @pytest.mark.unit
    def test_config_creation(self):
        """Test KSE Memory configuration creation."""
        config = KSEConfig.from_dict({
            "debug": True,
            "vector_store": {"backend": "pinecone"},
            "graph_store": {"backend": "neo4j"},
            "concept_store": {"backend": "postgresql"}
        })
        
        assert config is not None
        assert config.debug is True
        assert config.vector_store.backend == "pinecone"
        assert config.graph_store.backend == "neo4j"
        assert config.concept_store.backend == "postgresql"
    
    @pytest.mark.unit
    def test_memory_creation(self):
        """Test KSE Memory instance creation."""
        config = KSEConfig.from_dict({
            "vector_store": {"backend": "pinecone", "api_key": "test-key"},
            "embedding": {"openai_api_key": "test-key"},
            "conceptual": {"auto_compute": False}
        })
        kse = KSEMemory(config)
        
        assert kse is not None
        assert kse.config is not None
        assert kse._initialized is False
        assert kse._connected is False
    
    @pytest.mark.unit
    async def test_add_product(self, kse_memory, sample_products):
        """Test adding products to KSE Memory."""
        product = sample_products[0]
        
        result = await kse_memory.add_product(product)
        assert result is True
        
        # Verify product was added
        stored_product = await kse_memory.get_product(product.id)
        assert stored_product is not None
        assert stored_product.id == product.id
        assert stored_product.title == product.title
    
    @pytest.mark.unit
    async def test_vector_search(self, kse_memory, sample_products):
        """Test vector-only search functionality."""
        # Add products
        for product in sample_products:
            await kse_memory.add_product(product)
        
        # Perform vector search
        query = SearchQuery(
            query="comfortable athletic shoes",
            search_type=SearchType.VECTOR,
            limit=3
        )
        
        results = await kse_memory.search(query)
        
        assert len(results) > 0
        assert all(result.score >= 0 for result in results)
        assert all(result.product is not None for result in results)
        
        # Running shoes should be top result for athletic query
        top_result = results[0]
        assert "running" in top_result.product.title.lower() or "athletic" in top_result.product.title.lower()
    
    @pytest.mark.unit
    async def test_conceptual_search(self, kse_memory, sample_products):
        """Test conceptual space search functionality."""
        # Add products
        for product in sample_products:
            await kse_memory.add_product(product)
        
        # Perform conceptual search
        query = SearchQuery(
            query="comfortable everyday wear",
            search_type=SearchType.CONCEPTUAL,
            limit=3
        )
        
        results = await kse_memory.search(query)
        
        assert len(results) > 0
        assert all(result.score >= 0 for result in results)
        
        # Should prioritize products with high comfort scores
        for result in results:
            if result.product.conceptual_dimensions:
                comfort_score = result.product.conceptual_dimensions.comfort
                assert comfort_score >= 0.5  # Should find reasonably comfortable items
    
    @pytest.mark.unit
    async def test_graph_search(self, kse_memory, sample_products):
        """Test knowledge graph search functionality."""
        # Add products
        for product in sample_products:
            await kse_memory.add_product(product)
        
        # Perform graph search
        query = SearchQuery(
            query="athletic footwear",
            search_type=SearchType.GRAPH,
            limit=3
        )
        
        results = await kse_memory.search(query)
        
        assert len(results) > 0
        assert all(result.score >= 0 for result in results)
        
        # Should find products in athletic category
        athletic_found = any(
            "athletic" in result.product.category.lower() or
            "athletic" in result.product.tags
            for result in results
        )
        assert athletic_found
    
    @pytest.mark.unit
    async def test_hybrid_search(self, kse_memory, sample_products):
        """Test hybrid search combining all approaches."""
        # Add products
        for product in sample_products:
            await kse_memory.add_product(product)
        
        # Perform hybrid search
        query = SearchQuery(
            query="comfortable athletic wear",
            search_type=SearchType.HYBRID,
            limit=3
        )
        
        results = await kse_memory.search(query)
        
        assert len(results) > 0
        assert all(result.score >= 0 for result in results)
        
        # Hybrid should generally perform better than individual approaches
        # Test by comparing with vector-only search
        vector_query = SearchQuery(
            query="comfortable athletic wear",
            search_type=SearchType.VECTOR,
            limit=3
        )
        vector_results = await kse_memory.search(vector_query)
        
        if len(vector_results) > 0 and len(results) > 0:
            # Hybrid should have competitive or better scores
            hybrid_avg = sum(r.score for r in results) / len(results)
            vector_avg = sum(r.score for r in vector_results) / len(vector_results)
            
            # Allow for some variance, but hybrid should be competitive
            assert hybrid_avg >= vector_avg * 0.9
    
    @pytest.mark.unit
    async def test_search_with_filters(self, kse_memory, sample_products):
        """Test search with category and price filters."""
        # Add products
        for product in sample_products:
            await kse_memory.add_product(product)
        
        # Search with category filter
        query = SearchQuery(
            query="comfortable",
            search_type=SearchType.HYBRID,
            limit=10,
            filters={"category": "Athletic Footwear"}
        )
        
        results = await kse_memory.search(query)
        
        # Should only return products from Athletic Footwear category
        for result in results:
            assert result.product.category == "Athletic Footwear"
    
    @pytest.mark.unit
    async def test_empty_search(self, kse_memory):
        """Test search with no products in memory."""
        query = SearchQuery(
            query="any query",
            search_type=SearchType.HYBRID,
            limit=5
        )
        
        results = await kse_memory.search(query)
        assert len(results) == 0
    
    @pytest.mark.unit
    async def test_product_update(self, kse_memory, sample_products):
        """Test updating existing products."""
        product = sample_products[0]
        await kse_memory.add_product(product)
        
        # Update product
        updated_product = Product(
            id=product.id,
            title="Updated Premium Running Shoes",
            description="Updated description with new features",
            price=149.99,
            category=product.category,
            tags=product.tags + ["updated"],
            conceptual_dimensions=product.conceptual_dimensions
        )
        
        result = await kse_memory.update_product(updated_product)
        assert result is True
        
        # Verify update
        stored_product = await kse_memory.get_product(product.id)
        assert stored_product.title == "Updated Premium Running Shoes"
        assert stored_product.price == 149.99
        assert "updated" in stored_product.tags
    
    @pytest.mark.unit
    async def test_product_deletion(self, kse_memory, sample_products):
        """Test deleting products from memory."""
        product = sample_products[0]
        await kse_memory.add_product(product)
        
        # Verify product exists
        stored_product = await kse_memory.get_product(product.id)
        assert stored_product is not None
        
        # Delete product
        result = await kse_memory.delete_product(product.id)
        assert result is True
        
        # Verify deletion
        deleted_product = await kse_memory.get_product(product.id)
        assert deleted_product is None
    
    @pytest.mark.unit
    async def test_bulk_operations(self, kse_memory, sample_products):
        """Test bulk product operations."""
        # Bulk add
        result = await kse_memory.add_products(sample_products)
        assert result is True
        
        # Verify all products were added
        for product in sample_products:
            stored_product = await kse_memory.get_product(product.id)
            assert stored_product is not None
        
        # Bulk search should find all products
        query = SearchQuery(
            query="",  # Empty query should return all
            search_type=SearchType.HYBRID,
            limit=10
        )
        
        results = await kse_memory.search(query)
        assert len(results) == len(sample_products)


class TestConceptualDimensions:
    """Test suite for conceptual dimensions functionality."""
    
    @pytest.mark.unit
    def test_conceptual_dimensions_creation(self):
        """Test creating conceptual dimensions."""
        dims = ConceptualDimensions(
            elegance=0.8,
            comfort=0.9,
            boldness=0.3
        )
        
        assert dims.elegance == 0.8
        assert dims.comfort == 0.9
        assert dims.boldness == 0.3
        
        # Test default values
        assert dims.modernity == 0.0  # Default value
    
    @pytest.mark.unit
    def test_conceptual_dimensions_validation(self):
        """Test conceptual dimensions validation."""
        # Valid dimensions (0-1 range)
        dims = ConceptualDimensions(elegance=0.5, comfort=1.0, boldness=0.0)
        assert dims.elegance == 0.5
        
        # Test boundary values
        dims = ConceptualDimensions(elegance=0.0, comfort=1.0)
        assert dims.elegance == 0.0
        assert dims.comfort == 1.0
    
    @pytest.mark.unit
    def test_conceptual_dimensions_to_dict(self):
        """Test converting conceptual dimensions to dictionary."""
        dims = ConceptualDimensions(
            elegance=0.8,
            comfort=0.9,
            boldness=0.3
        )
        
        dims_dict = dims.to_dict()
        
        assert isinstance(dims_dict, dict)
        assert dims_dict["elegance"] == 0.8
        assert dims_dict["comfort"] == 0.9
        assert dims_dict["boldness"] == 0.3
        assert "modernity" in dims_dict  # Should include defaults


class TestSearchQuery:
    """Test suite for search query functionality."""
    
    @pytest.mark.unit
    def test_search_query_creation(self):
        """Test creating search queries."""
        query = SearchQuery(
            query="comfortable shoes",
            search_type=SearchType.HYBRID,
            limit=5
        )
        
        assert query.query == "comfortable shoes"
        assert query.search_type == SearchType.HYBRID
        assert query.limit == 5
    
    @pytest.mark.unit
    def test_search_query_with_filters(self):
        """Test search queries with filters."""
        query = SearchQuery(
            query="athletic wear",
            search_type=SearchType.SEMANTIC,
            limit=10,
            filters={"category": "Athletic", "price_max": 200}
        )
        
        assert query.query == "athletic wear"
        assert query.search_type == SearchType.SEMANTIC
        assert query.limit == 10
        assert query.filters["category"] == "Athletic"
        assert query.filters["price_max"] == 200
    
    @pytest.mark.unit
    def test_search_type_enum(self):
        """Test search type enumeration."""
        assert SearchType.SEMANTIC.value == "semantic"
        assert SearchType.CONCEPTUAL.value == "conceptual"
        assert SearchType.KNOWLEDGE_GRAPH.value == "knowledge_graph"
        assert SearchType.HYBRID.value == "hybrid"


@pytest.mark.integration
class TestKSEMemoryIntegration:
    """Integration tests for KSE Memory with real backends."""
    
    @pytest.fixture
    async def kse_memory_with_backends(self):
        """Create KSE Memory with real backend configurations."""
        config = KSEConfig.from_dict({
            "vector_store": {
                "backend": "memory",  # Use memory for testing
            },
            "graph_store": {
                "backend": "memory",
            },
            "concept_store": {
                "backend": "memory",
            }
        })
        kse = KSEMemory(config)
        await kse.initialize("generic", {})
        yield kse
        await kse.disconnect()
    
    @pytest.mark.integration
    async def test_end_to_end_workflow(self, kse_memory_with_backends):
        """Test complete end-to-end workflow."""
        kse = kse_memory_with_backends
        
        # Create test product
        product = Product(
            id="e2e_test_001",
            title="Test Product for E2E",
            description="This is a test product for end-to-end testing",
            price=99.99,
            category="Test Category",
            tags=["test", "e2e", "product"],
            conceptual_dimensions=ConceptualDimensions(
                functionality=0.8,
                innovation=0.7,
                comfort=0.6
            )
        )
        
        # Add product
        add_result = await kse.add_product(product)
        assert add_result is True
        
        # Search for product
        query = SearchQuery(
            query="test product",
            search_type=SearchType.HYBRID,
            limit=5
        )
        
        search_results = await kse.search(query)
        assert len(search_results) > 0
        
        # Verify we found our product
        found_product = None
        for result in search_results:
            if result.product.id == product.id:
                found_product = result.product
                break
        
        assert found_product is not None
        assert found_product.title == product.title
        
        # Update product
        updated_product = Product(
            id=product.id,
            title="Updated Test Product",
            description=product.description,
            price=119.99,
            category=product.category,
            tags=product.tags,
            conceptual_dimensions=product.conceptual_dimensions
        )
        
        update_result = await kse.update_product(updated_product)
        assert update_result is True
        
        # Verify update
        retrieved_product = await kse.get_product(product.id)
        assert retrieved_product.title == "Updated Test Product"
        assert retrieved_product.price == 119.99
        
        # Delete product
        delete_result = await kse.delete_product(product.id)
        assert delete_result is True
        
        # Verify deletion
        deleted_product = await kse.get_product(product.id)
        assert deleted_product is None


@pytest.mark.performance
class TestPerformance:
    """Performance tests for KSE Memory."""
    
    @pytest.mark.performance
    async def test_search_performance(self):
        """Test search performance with larger dataset."""
        config = KSEConfig.from_dict({
            "vector_store": {"backend": "memory"},
            "graph_store": {"backend": "memory"},
            "concept_store": {"backend": "memory"}
        })
        kse = KSEMemory(config)
        await kse.initialize("generic", {})
        
        try:
            # Create larger dataset
            products = []
            for i in range(100):
                product = Product(
                    id=f"perf_test_{i:03d}",
                    title=f"Test Product {i}",
                    description=f"Description for test product {i}",
                    price=float(i * 10),
                    category="Test Category",
                    tags=["test", f"product_{i}"],
                    conceptual_dimensions=ConceptualDimensions(
                        functionality=min(1.0, i / 100.0),
                        innovation=min(1.0, (i + 10) / 100.0)
                    )
                )
                products.append(product)
            
            # Bulk add products
            import time
            start_time = time.time()
            await kse.add_products(products)
            add_time = time.time() - start_time
            
            # Test search performance
            query = SearchQuery(
                query="test product",
                search_type=SearchType.HYBRID,
                limit=10
            )
            
            start_time = time.time()
            results = await kse.search(query)
            search_time = time.time() - start_time
            
            # Performance assertions
            assert add_time < 10.0  # Should add 100 products in under 10 seconds
            assert search_time < 1.0  # Should search in under 1 second
            assert len(results) > 0
            
        finally:
            await kse.disconnect()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])