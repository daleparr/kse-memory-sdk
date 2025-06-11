"""
KSE Memory SDK - Backend Tests

Test suite for all backend implementations.
"""

import pytest
import asyncio
from typing import List, Dict, Any

from kse_memory.core.models import Product, ConceptualDimensions
from kse_memory.core.config import VectorStoreConfig, GraphStoreConfig, ConceptStoreConfig
from kse_memory.backends import (
    get_vector_store, get_graph_store, get_concept_store,
    ChromaDBBackend, MongoDBBackend, ArangoDBBackend, MilvusBackend
)
from kse_memory.exceptions import BackendError


class TestBackendFactory:
    """Test backend factory functions."""
    
    def test_get_vector_store_chromadb(self):
        """Test ChromaDB vector store creation."""
        config = VectorStoreConfig(
            backend="chromadb",
            uri="memory://",
            dimension=384
        )
        
        try:
            store = get_vector_store(config)
            assert isinstance(store, ChromaDBBackend)
        except BackendError as e:
            # ChromaDB might not be installed
            assert "chromadb package is required" in str(e)
    
    def test_get_vector_store_milvus(self):
        """Test Milvus vector store creation."""
        config = VectorStoreConfig(
            backend="milvus",
            uri="milvus://localhost:19530",
            dimension=1536
        )
        
        try:
            store = get_vector_store(config)
            assert isinstance(store, MilvusBackend)
        except BackendError as e:
            # Milvus might not be installed
            assert "pymilvus package is required" in str(e)
    
    def test_get_graph_store_arangodb(self):
        """Test ArangoDB graph store creation."""
        config = GraphStoreConfig(
            backend="arangodb",
            uri="http://localhost:8529",
            database="kse_test",
            username="root",
            password="test"
        )
        
        try:
            store = get_graph_store(config)
            assert isinstance(store, ArangoDBBackend)
        except BackendError as e:
            # ArangoDB might not be installed
            assert "python-arango package is required" in str(e)
    
    def test_get_concept_store_mongodb(self):
        """Test MongoDB concept store creation."""
        config = ConceptStoreConfig(
            backend="mongodb",
            uri="mongodb://localhost:27017",
            database="kse_test"
        )
        
        try:
            store = get_concept_store(config)
            assert isinstance(store, MongoDBBackend)
        except BackendError as e:
            # MongoDB might not be installed
            assert "motor package is required" in str(e)
    
    def test_unsupported_vector_backend(self):
        """Test unsupported vector backend raises error."""
        config = VectorStoreConfig(
            backend="unsupported",
            uri="test://",
            dimension=384
        )
        
        with pytest.raises(BackendError) as exc_info:
            get_vector_store(config)
        
        assert "Unsupported vector store backend: unsupported" in str(exc_info.value)
    
    def test_unsupported_graph_backend(self):
        """Test unsupported graph backend raises error."""
        config = GraphStoreConfig(
            backend="unsupported",
            uri="test://",
            database="test"
        )
        
        with pytest.raises(BackendError) as exc_info:
            get_graph_store(config)
        
        assert "Unsupported graph store backend: unsupported" in str(exc_info.value)
    
    def test_unsupported_concept_backend(self):
        """Test unsupported concept backend raises error."""
        config = ConceptStoreConfig(
            backend="unsupported",
            uri="test://",
            database="test"
        )
        
        with pytest.raises(BackendError) as exc_info:
            get_concept_store(config)
        
        assert "Unsupported concept store backend: unsupported" in str(exc_info.value)


class TestBackendInterfaces:
    """Test backend interface compliance."""
    
    @pytest.fixture
    def sample_product(self):
        """Create a sample product for testing."""
        return Product(
            id="test-product-1",
            title="Test Product",
            description="A test product for backend testing",
            price=99.99,
            currency="USD",
            category="Electronics",
            brand="TestBrand",
            tags=["test", "electronics"],
            conceptual_dimensions=ConceptualDimensions(
                elegance=0.7,
                comfort=0.8,
                boldness=0.6,
                sustainability=0.9,
                luxury=0.5,
                innovation=0.8,
                versatility=0.7,
                durability=0.9,
                affordability=0.6,
                trendiness=0.7
            )
        )
    
    @pytest.fixture
    def sample_embedding(self):
        """Create a sample embedding vector."""
        return [0.1] * 384  # 384-dimensional vector
    
    def test_chromadb_backend_interface(self, sample_product, sample_embedding):
        """Test ChromaDB backend interface methods."""
        config = VectorStoreConfig(
            backend="chromadb",
            uri="memory://",
            dimension=384
        )
        
        try:
            backend = ChromaDBBackend(config)
            
            # Test interface methods exist
            assert hasattr(backend, 'connect')
            assert hasattr(backend, 'disconnect')
            assert hasattr(backend, 'upsert_product')
            assert hasattr(backend, 'search_similar')
            assert hasattr(backend, 'get_product_by_id')
            assert hasattr(backend, 'delete_product')
            assert hasattr(backend, 'list_products')
            assert hasattr(backend, 'get_collection_stats')
            
        except BackendError:
            # ChromaDB not installed, skip test
            pytest.skip("ChromaDB not available")
    
    def test_mongodb_backend_interface(self, sample_product):
        """Test MongoDB backend interface methods."""
        config = ConceptStoreConfig(
            backend="mongodb",
            uri="mongodb://localhost:27017",
            database="kse_test"
        )
        
        try:
            backend = MongoDBBackend(config)
            
            # Test interface methods exist
            assert hasattr(backend, 'connect')
            assert hasattr(backend, 'disconnect')
            assert hasattr(backend, 'store_product_concepts')
            assert hasattr(backend, 'get_product_concepts')
            assert hasattr(backend, 'find_similar_products')
            assert hasattr(backend, 'get_concept_distribution')
            assert hasattr(backend, 'create_conceptual_space')
            assert hasattr(backend, 'get_conceptual_space')
            assert hasattr(backend, 'list_conceptual_spaces')
            assert hasattr(backend, 'delete_product_concepts')
            assert hasattr(backend, 'get_concept_statistics')
            
        except BackendError:
            # MongoDB not installed, skip test
            pytest.skip("MongoDB not available")
    
    def test_arangodb_backend_interface(self, sample_product):
        """Test ArangoDB backend interface methods."""
        config = GraphStoreConfig(
            backend="arangodb",
            uri="http://localhost:8529",
            database="kse_test",
            username="root",
            password="test"
        )
        
        try:
            backend = ArangoDBBackend(config)
            
            # Test interface methods exist
            assert hasattr(backend, 'connect')
            assert hasattr(backend, 'disconnect')
            assert hasattr(backend, 'add_product_node')
            assert hasattr(backend, 'add_relationship')
            assert hasattr(backend, 'find_related_products')
            assert hasattr(backend, 'get_product_relationships')
            assert hasattr(backend, 'delete_product_node')
            assert hasattr(backend, 'execute_graph_query')
            assert hasattr(backend, 'get_graph_statistics')
            
        except BackendError:
            # ArangoDB not installed, skip test
            pytest.skip("ArangoDB not available")
    
    def test_milvus_backend_interface(self, sample_product, sample_embedding):
        """Test Milvus backend interface methods."""
        config = VectorStoreConfig(
            backend="milvus",
            uri="milvus://localhost:19530",
            dimension=384
        )
        
        try:
            backend = MilvusBackend(config)
            
            # Test interface methods exist
            assert hasattr(backend, 'connect')
            assert hasattr(backend, 'disconnect')
            assert hasattr(backend, 'upsert_product')
            assert hasattr(backend, 'search_similar')
            assert hasattr(backend, 'get_product_by_id')
            assert hasattr(backend, 'delete_product')
            assert hasattr(backend, 'list_products')
            assert hasattr(backend, 'get_collection_stats')
            assert hasattr(backend, 'create_index')
            
        except BackendError:
            # Milvus not installed, skip test
            pytest.skip("Milvus not available")


class TestBackendConfiguration:
    """Test backend configuration handling."""
    
    def test_vector_store_config_validation(self):
        """Test vector store configuration validation."""
        # Valid configuration
        config = VectorStoreConfig(
            backend="chromadb",
            uri="memory://",
            dimension=384
        )
        assert config.backend == "chromadb"
        assert config.uri == "memory://"
        assert config.dimension == 384
    
    def test_graph_store_config_validation(self):
        """Test graph store configuration validation."""
        # Valid configuration
        config = GraphStoreConfig(
            backend="arangodb",
            uri="http://localhost:8529",
            database="kse_test",
            username="root",
            password="test"
        )
        assert config.backend == "arangodb"
        assert config.uri == "http://localhost:8529"
        assert config.database == "kse_test"
        assert config.username == "root"
        assert config.password == "test"
    
    def test_concept_store_config_validation(self):
        """Test concept store configuration validation."""
        # Valid configuration
        config = ConceptStoreConfig(
            backend="mongodb",
            uri="mongodb://localhost:27017",
            database="kse_test"
        )
        assert config.backend == "mongodb"
        assert config.uri == "mongodb://localhost:27017"
        assert config.database == "kse_test"


class TestBackendErrorHandling:
    """Test backend error handling."""
    
    def test_backend_error_creation(self):
        """Test BackendError creation and attributes."""
        error = BackendError("Test error message", "test_backend")
        
        assert str(error) == "Test error message"
        assert error.backend_type == "test_backend"
    
    def test_backend_error_without_backend_type(self):
        """Test BackendError creation without backend type."""
        error = BackendError("Test error message")
        
        assert str(error) == "Test error message"
        assert error.backend_type is None


@pytest.mark.asyncio
class TestBackendIntegration:
    """Integration tests for backends (requires actual services)."""
    
    async def test_mock_backends_integration(self):
        """Test integration with mock backends."""
        from kse_memory.backends.mock import MockVectorStore, MockGraphStore, MockConceptStore
        
        # Create mock backends
        vector_config = VectorStoreConfig(backend="mock", uri="memory://", dimension=384)
        graph_config = GraphStoreConfig(backend="mock", uri="memory://", database="test")
        concept_config = ConceptStoreConfig(backend="mock", uri="memory://", database="test")
        
        vector_store = MockVectorStore(vector_config)
        graph_store = MockGraphStore(graph_config)
        concept_store = MockConceptStore(concept_config)
        
        # Test connections
        assert await vector_store.connect()
        assert await graph_store.connect()
        assert await concept_store.connect()
        
        # Test basic operations
        sample_product = Product(
            id="test-1",
            title="Test Product",
            description="Test description",
            price=99.99,
            currency="USD",
            category="Test",
            brand="TestBrand",
            conceptual_dimensions=ConceptualDimensions(
                elegance=0.7,
                comfort=0.8,
                boldness=0.6
            )
        )
        
        sample_embedding = [0.1] * 384
        
        # Vector store operations
        assert await vector_store.upsert_product(sample_product, sample_embedding)
        results = await vector_store.search_similar(sample_embedding, limit=5)
        assert isinstance(results, list)
        
        # Graph store operations
        assert await graph_store.add_product_node(sample_product)
        relationships = await graph_store.get_product_relationships("test-1")
        assert isinstance(relationships, list)
        
        # Concept store operations
        assert await concept_store.store_product_concepts(sample_product)
        concepts = await concept_store.get_product_concepts("test-1")
        assert concepts is not None
        
        # Cleanup
        assert await vector_store.disconnect()
        assert await graph_store.disconnect()
        assert await concept_store.disconnect()