"""
KSE Memory SDK - Integration Tests

Test suite for framework integrations (LangChain, LlamaIndex).
"""

import pytest
import asyncio
from typing import List

from kse_memory.core.memory import KSEMemory
from kse_memory.core.config import KSEConfig
from kse_memory.core.models import Product, SearchQuery, SearchType, ConceptualDimensions


class TestLangChainIntegration:
    """Test suite for LangChain integration."""
    
    @pytest.fixture
    async def kse_memory(self):
        """Create KSE Memory instance for testing."""
        config = KSEConfig(
            debug=True,
            vector_store={"backend": "memory"},
            graph_store={"backend": "memory"},
            concept_store={"backend": "memory"}
        )
        kse = KSEMemory(config)
        await kse.initialize("generic", {})
        yield kse
        await kse.disconnect()
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            "Premium athletic running shoes with responsive cushioning and breathable design.",
            "Elegant silk evening dress perfect for formal occasions and special events.",
            "Comfortable cotton t-shirt for everyday casual wear and relaxation.",
            "Modern wireless headphones with noise cancellation and long battery life.",
            "Luxury leather handbag with premium craftsmanship and timeless design."
        ]
    
    @pytest.mark.integration
    async def test_kse_vector_store_creation(self, kse_memory, sample_documents):
        """Test creating KSE vector store from texts."""
        try:
            from kse_memory.integrations.langchain import KSEVectorStore
            
            # Create vector store from texts
            vectorstore = KSEVectorStore.from_texts(
                texts=sample_documents,
                search_type="hybrid"
            )
            
            assert vectorstore is not None
            assert vectorstore.search_type.value == "HYBRID"
            
        except ImportError:
            pytest.skip("LangChain not installed")
    
    @pytest.mark.integration
    async def test_kse_vector_store_similarity_search(self, sample_documents):
        """Test similarity search with KSE vector store."""
        try:
            from kse_memory.integrations.langchain import KSEVectorStore
            
            # Create and populate vector store
            vectorstore = KSEVectorStore.from_texts(
                texts=sample_documents,
                search_type="hybrid"
            )
            
            # Perform similarity search
            results = vectorstore.similarity_search("comfortable athletic wear", k=3)
            
            assert len(results) > 0
            assert len(results) <= 3
            
            # Check that results are Document objects
            for doc in results:
                assert hasattr(doc, 'page_content')
                assert hasattr(doc, 'metadata')
                assert isinstance(doc.page_content, str)
                assert isinstance(doc.metadata, dict)
            
            # Athletic shoes should be in top results
            athletic_found = any("athletic" in doc.page_content.lower() for doc in results)
            assert athletic_found
            
        except ImportError:
            pytest.skip("LangChain not installed")
    
    @pytest.mark.integration
    async def test_kse_vector_store_with_scores(self, sample_documents):
        """Test similarity search with scores."""
        try:
            from kse_memory.integrations.langchain import KSEVectorStore
            
            vectorstore = KSEVectorStore.from_texts(
                texts=sample_documents,
                search_type="hybrid"
            )
            
            # Perform similarity search with scores
            results = vectorstore.similarity_search_with_score("elegant formal wear", k=2)
            
            assert len(results) > 0
            assert len(results) <= 2
            
            # Check result format
            for doc, score in results:
                assert hasattr(doc, 'page_content')
                assert isinstance(score, (int, float))
                assert 0 <= score <= 1
            
            # Elegant dress should be top result
            top_doc, top_score = results[0]
            assert "elegant" in top_doc.page_content.lower() or "dress" in top_doc.page_content.lower()
            
        except ImportError:
            pytest.skip("LangChain not installed")
    
    @pytest.mark.integration
    async def test_kse_langchain_retriever(self, kse_memory):
        """Test KSE LangChain retriever."""
        try:
            from kse_memory.integrations.langchain import KSELangChainRetriever
            
            # Create retriever
            retriever = KSELangChainRetriever(
                kse_memory=kse_memory,
                search_type="hybrid",
                k=3
            )
            
            assert retriever is not None
            assert retriever.search_type.value == "HYBRID"
            assert retriever.k == 3
            
            # Add some test products first
            products = [
                Product(
                    id="test_001",
                    title="Athletic Running Shoes",
                    description="Comfortable shoes for running and athletics",
                    category="Footwear",
                    tags=["athletic", "running", "comfortable"]
                ),
                Product(
                    id="test_002",
                    title="Formal Evening Dress",
                    description="Elegant dress for formal occasions",
                    category="Clothing",
                    tags=["formal", "elegant", "dress"]
                )
            ]
            
            for product in products:
                await kse_memory.add_product(product)
            
            # Test retrieval
            docs = retriever.get_relevant_documents("comfortable athletic footwear")
            
            assert len(docs) > 0
            
            # Check document format
            for doc in docs:
                assert hasattr(doc, 'page_content')
                assert hasattr(doc, 'metadata')
                assert 'retriever' in doc.metadata
                assert doc.metadata['retriever'] == 'KSE_hybrid_ai'
            
        except ImportError:
            pytest.skip("LangChain not installed")


class TestLlamaIndexIntegration:
    """Test suite for LlamaIndex integration."""
    
    @pytest.fixture
    async def kse_memory(self):
        """Create KSE Memory instance for testing."""
        config = KSEConfig(
            debug=True,
            vector_store={"backend": "memory"},
            graph_store={"backend": "memory"},
            concept_store={"backend": "memory"}
        )
        kse = KSEMemory(config)
        await kse.initialize("generic", {})
        yield kse
        await kse.disconnect()
    
    @pytest.mark.integration
    async def test_kse_llamaindex_retriever(self, kse_memory):
        """Test KSE LlamaIndex retriever."""
        try:
            from kse_memory.integrations.llamaindex import KSELlamaIndexRetriever
            
            # Create retriever
            retriever = KSELlamaIndexRetriever(
                kse_memory=kse_memory,
                search_type="hybrid",
                similarity_top_k=3
            )
            
            assert retriever is not None
            assert retriever.search_type.value == "HYBRID"
            assert retriever.similarity_top_k == 3
            
            # Add test products
            products = [
                Product(
                    id="llama_test_001",
                    title="Smart Fitness Tracker",
                    description="Advanced fitness tracking with health monitoring",
                    category="Wearables",
                    tags=["fitness", "smart", "health"]
                ),
                Product(
                    id="llama_test_002",
                    title="Wireless Earbuds",
                    description="Premium wireless earbuds with noise cancellation",
                    category="Audio",
                    tags=["wireless", "audio", "premium"]
                )
            ]
            
            for product in products:
                await kse_memory.add_product(product)
            
            # Create mock query bundle
            class MockQueryBundle:
                def __init__(self, query_str):
                    self.query_str = query_str
            
            query_bundle = MockQueryBundle("fitness tracking device")
            
            # Test retrieval
            nodes_with_scores = retriever._retrieve(query_bundle)
            
            assert len(nodes_with_scores) > 0
            
            # Check node format
            for node_with_score in nodes_with_scores:
                assert hasattr(node_with_score, 'node')
                assert hasattr(node_with_score, 'score')
                assert isinstance(node_with_score.score, (int, float))
                
                node = node_with_score.node
                assert hasattr(node, 'text')
                assert hasattr(node, 'metadata')
                assert 'retriever' in node.metadata
                assert node.metadata['retriever'] == 'KSE_hybrid_ai'
            
        except ImportError:
            pytest.skip("LlamaIndex not installed")
    
    @pytest.mark.integration
    async def test_kse_vector_store_index(self):
        """Test KSE Vector Store Index for LlamaIndex."""
        try:
            from kse_memory.integrations.llamaindex import KSEVectorStoreIndex
            
            # Create vector store index
            vector_store = KSEVectorStoreIndex(
                search_type="hybrid"
            )
            
            assert vector_store is not None
            assert vector_store.search_type.value == "HYBRID"
            
            # Create mock documents
            class MockDocument:
                def __init__(self, text, metadata=None):
                    self.text = text
                    self.metadata = metadata or {}
            
            documents = [
                MockDocument("High-performance laptop for gaming and productivity"),
                MockDocument("Comfortable office chair with ergonomic design"),
                MockDocument("Professional camera for photography enthusiasts")
            ]
            
            # Add documents
            doc_ids = vector_store.add_documents(documents)
            
            assert len(doc_ids) == len(documents)
            assert all(isinstance(doc_id, str) for doc_id in doc_ids)
            
            # Query the vector store
            results = vector_store.query("gaming laptop", similarity_top_k=2)
            
            assert len(results) > 0
            assert len(results) <= 2
            
            # Check result format
            for node_with_score in results:
                assert hasattr(node_with_score, 'node')
                assert hasattr(node_with_score, 'score')
                
                node = node_with_score.node
                assert hasattr(node, 'text')
                assert hasattr(node, 'metadata')
            
        except ImportError:
            pytest.skip("LlamaIndex not installed")


class TestIntegrationCompatibility:
    """Test compatibility between different integrations."""
    
    @pytest.mark.integration
    async def test_framework_interoperability(self):
        """Test that both frameworks can work with the same KSE instance."""
        config = KSEConfig(
            vector_store={"backend": "memory"},
            graph_store={"backend": "memory"},
            concept_store={"backend": "memory"}
        )
        kse = KSEMemory(config)
        await kse.initialize("generic", {})
        
        try:
            # Add test data
            products = [
                Product(
                    id="interop_001",
                    title="Multi-Framework Test Product",
                    description="Product for testing framework interoperability",
                    category="Test",
                    tags=["test", "interop"]
                )
            ]
            
            for product in products:
                await kse.add_product(product)
            
            # Test with both frameworks if available
            langchain_results = None
            llamaindex_results = None
            
            try:
                from kse_memory.integrations.langchain import KSELangChainRetriever
                langchain_retriever = KSELangChainRetriever(kse_memory=kse, k=1)
                langchain_results = langchain_retriever.get_relevant_documents("test product")
            except ImportError:
                pass
            
            try:
                from kse_memory.integrations.llamaindex import KSELlamaIndexRetriever
                llamaindex_retriever = KSELlamaIndexRetriever(kse_memory=kse, similarity_top_k=1)
                
                class MockQueryBundle:
                    def __init__(self, query_str):
                        self.query_str = query_str
                
                query_bundle = MockQueryBundle("test product")
                llamaindex_results = llamaindex_retriever._retrieve(query_bundle)
            except ImportError:
                pass
            
            # If both frameworks are available, compare results
            if langchain_results is not None and llamaindex_results is not None:
                assert len(langchain_results) > 0
                assert len(llamaindex_results) > 0
                
                # Both should find the test product
                langchain_found = any("test" in doc.page_content.lower() for doc in langchain_results)
                llamaindex_found = any("test" in node.node.text.lower() for node in llamaindex_results)
                
                assert langchain_found
                assert llamaindex_found
            
        finally:
            await kse.disconnect()
    
    @pytest.mark.integration
    async def test_migration_compatibility(self):
        """Test that migration from traditional stores works correctly."""
        # This test would verify that migrating from traditional vector stores
        # to KSE Memory maintains functionality while improving performance
        
        sample_texts = [
            "Traditional vector store document about machine learning",
            "Another document about artificial intelligence and deep learning",
            "Document about natural language processing and transformers"
        ]
        
        # Test LangChain migration if available
        try:
            from kse_memory.integrations.langchain import KSEVectorStore
            
            # Create KSE vector store (migration target)
            kse_store = KSEVectorStore.from_texts(
                texts=sample_texts,
                search_type="hybrid"
            )
            
            # Test that it works like a traditional vector store
            results = kse_store.similarity_search("machine learning", k=2)
            
            assert len(results) > 0
            assert any("machine learning" in doc.page_content.lower() for doc in results)
            
            # Test with scores
            results_with_scores = kse_store.similarity_search_with_score("artificial intelligence", k=2)
            
            assert len(results_with_scores) > 0
            for doc, score in results_with_scores:
                assert isinstance(score, (int, float))
                assert 0 <= score <= 1
            
        except ImportError:
            pytest.skip("LangChain not available for migration test")


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-m", "integration"])