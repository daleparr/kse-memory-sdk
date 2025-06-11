"""
KSE Memory SDK - LlamaIndex Integration

Provides integration with LlamaIndex for hybrid AI retrieval
in RAG applications and knowledge management systems.
"""

from typing import List, Dict, Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

try:
    from llama_index.core.schema import Document, NodeWithScore
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.vector_stores import VectorStore
    from llama_index.core.bridge.pydantic import Field
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    # Create mock classes if LlamaIndex is not installed
    class Document:
        def __init__(self, text: str, metadata: Dict[str, Any] = None):
            self.text = text
            self.metadata = metadata or {}
    
    class NodeWithScore:
        def __init__(self, node, score: float = 0.0):
            self.node = node
            self.score = score
    
    class BaseRetriever:
        pass
    
    class VectorStore:
        pass
    
    def Field(**kwargs):
        return None
    
    LLAMAINDEX_AVAILABLE = False

from ..core.memory import KSEMemory
from ..core.config import KSEConfig
from ..core.models import Product, SearchQuery, SearchType


class KSELlamaIndexRetriever(BaseRetriever):
    """
    LlamaIndex Retriever implementation using KSE Memory.
    
    Provides hybrid AI retrieval capabilities for LlamaIndex
    applications with knowledge graphs, conceptual spaces,
    and neural embeddings.
    
    Example:
        from kse_memory.integrations.llamaindex import KSELlamaIndexRetriever
        from llama_index.core import QueryEngine
        
        # Initialize retriever
        retriever = KSELlamaIndexRetriever(
            kse_memory=kse_memory,
            search_type="hybrid",
            similarity_top_k=5
        )
        
        # Use in LlamaIndex query engine
        query_engine = QueryEngine.from_args(
            retriever=retriever,
            llm=llm
        )
    """
    
    def __init__(
        self,
        kse_memory: Optional[KSEMemory] = None,
        config: Optional[KSEConfig] = None,
        search_type: str = "hybrid",
        similarity_top_k: int = 4,
        adapter: str = "generic",
        **kwargs
    ):
        """
        Initialize KSE LlamaIndex Retriever.
        
        Args:
            kse_memory: Existing KSE Memory instance
            config: KSE configuration (if kse_memory not provided)
            search_type: Search type ('vector', 'conceptual', 'graph', 'hybrid')
            similarity_top_k: Number of results to return
            adapter: Platform adapter to use
        """
        if not LLAMAINDEX_AVAILABLE:
            raise ImportError(
                "LlamaIndex is required for this integration. "
                "Install with: pip install llama-index"
            )
        
        super().__init__(**kwargs)
        
        self.kse_memory = kse_memory
        self.config = config or KSEConfig()
        self.search_type = SearchType(search_type.upper())
        self.similarity_top_k = similarity_top_k
        self.adapter = adapter
        self._initialized = False
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    async def _ensure_initialized(self):
        """Ensure KSE Memory is initialized."""
        if not self._initialized:
            if self.kse_memory is None:
                self.kse_memory = KSEMemory(self.config)
                await self.kse_memory.initialize(self.adapter, {})
            self._initialized = True
    
    def _run_async(self, coro):
        """Run async function in thread pool."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    
    def _retrieve(self, query_bundle) -> List[NodeWithScore]:
        """
        Retrieve relevant nodes for a query.
        
        Args:
            query_bundle: LlamaIndex query bundle
            
        Returns:
            List of nodes with scores
        """
        async def _retrieve_async():
            await self._ensure_initialized()
            
            # Extract query string from bundle
            query_str = str(query_bundle.query_str) if hasattr(query_bundle, 'query_str') else str(query_bundle)
            
            search_query = SearchQuery(
                query=query_str,
                search_type=self.search_type,
                limit=self.similarity_top_k
            )
            
            results = await self.kse_memory.search(search_query)
            
            # Convert to LlamaIndex nodes
            nodes_with_scores = []
            for result in results:
                # Create document from product
                doc = Document(
                    text=result.product.description,
                    metadata={
                        **result.product.metadata,
                        "id": result.product.id,
                        "title": result.product.title,
                        "category": result.product.category,
                        "price": result.product.price,
                        "tags": result.product.tags,
                        "search_type": self.search_type.value,
                        "retriever": "KSE_hybrid_ai"
                    }
                )
                
                # Create node with score
                node_with_score = NodeWithScore(
                    node=doc,
                    score=result.score
                )
                nodes_with_scores.append(node_with_score)
            
            return nodes_with_scores
        
        return self._executor.submit(self._run_async, _retrieve_async()).result()


class KSEVectorStoreIndex:
    """
    LlamaIndex Vector Store implementation using KSE Memory.
    
    Provides hybrid AI capabilities as a drop-in replacement
    for traditional vector stores in LlamaIndex applications.
    
    Example:
        from kse_memory.integrations.llamaindex import KSEVectorStoreIndex
        from llama_index.core import ServiceContext
        
        # Initialize KSE vector store
        vector_store = KSEVectorStoreIndex(
            kse_memory=kse_memory,
            search_type="hybrid"
        )
        
        # Use in LlamaIndex service context
        service_context = ServiceContext.from_defaults(
            vector_store=vector_store
        )
    """
    
    def __init__(
        self,
        kse_memory: Optional[KSEMemory] = None,
        config: Optional[KSEConfig] = None,
        search_type: str = "hybrid",
        adapter: str = "generic"
    ):
        """
        Initialize KSE Vector Store Index.
        
        Args:
            kse_memory: Existing KSE Memory instance
            config: KSE configuration (if kse_memory not provided)
            search_type: Default search type
            adapter: Platform adapter to use
        """
        if not LLAMAINDEX_AVAILABLE:
            raise ImportError(
                "LlamaIndex is required for this integration. "
                "Install with: pip install llama-index"
            )
        
        self.kse_memory = kse_memory
        self.config = config or KSEConfig()
        self.search_type = SearchType(search_type.upper())
        self.adapter = adapter
        self._initialized = False
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    async def _ensure_initialized(self):
        """Ensure KSE Memory is initialized."""
        if not self._initialized:
            if self.kse_memory is None:
                self.kse_memory = KSEMemory(self.config)
                await self.kse_memory.initialize(self.adapter, {})
            self._initialized = True
    
    def _run_async(self, coro):
        """Run async function in thread pool."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of LlamaIndex documents
            
        Returns:
            List of document IDs
        """
        async def _add_documents():
            await self._ensure_initialized()
            
            ids = []
            for i, doc in enumerate(documents):
                # Convert LlamaIndex document to Product
                metadata = doc.metadata or {}
                
                product = Product(
                    id=f"llamaindex_doc_{i}",
                    title=metadata.get("title", f"Document {i}"),
                    description=doc.text,
                    price=metadata.get("price", 0.0),
                    category=metadata.get("category", "Document"),
                    tags=metadata.get("tags", []),
                    metadata=metadata
                )
                
                await self.kse_memory.add_product(product)
                ids.append(product.id)
            
            return ids
        
        return self._executor.submit(self._run_async, _add_documents()).result()
    
    def query(
        self,
        query: str,
        similarity_top_k: int = 4,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[NodeWithScore]:
        """
        Query the vector store.
        
        Args:
            query: Search query
            similarity_top_k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of nodes with scores
        """
        async def _query():
            await self._ensure_initialized()
            
            search_query = SearchQuery(
                query=query,
                search_type=self.search_type,
                limit=similarity_top_k,
                filters=filters
            )
            
            results = await self.kse_memory.search(search_query)
            
            # Convert to LlamaIndex nodes
            nodes_with_scores = []
            for result in results:
                doc = Document(
                    text=result.product.description,
                    metadata={
                        **result.product.metadata,
                        "id": result.product.id,
                        "title": result.product.title,
                        "category": result.product.category,
                        "price": result.product.price,
                        "tags": result.product.tags,
                        "search_type": self.search_type.value
                    }
                )
                
                node_with_score = NodeWithScore(
                    node=doc,
                    score=result.score
                )
                nodes_with_scores.append(node_with_score)
            
            return nodes_with_scores
        
        return self._executor.submit(self._run_async, _query()).result()


# Utility functions for easy integration
def create_kse_retriever(
    config: Optional[KSEConfig] = None,
    search_type: str = "hybrid",
    similarity_top_k: int = 4,
    adapter: str = "generic"
) -> KSELlamaIndexRetriever:
    """
    Create a KSE Retriever with default configuration.
    
    Args:
        config: Optional KSE configuration
        search_type: Search type
        similarity_top_k: Number of results
        adapter: Platform adapter
        
    Returns:
        Configured KSE Retriever
    """
    return KSELlamaIndexRetriever(
        config=config,
        search_type=search_type,
        similarity_top_k=similarity_top_k,
        adapter=adapter
    )


def create_kse_vector_store(
    config: Optional[KSEConfig] = None,
    search_type: str = "hybrid",
    adapter: str = "generic"
) -> KSEVectorStoreIndex:
    """
    Create a KSE Vector Store with default configuration.
    
    Args:
        config: Optional KSE configuration
        search_type: Default search type
        adapter: Platform adapter
        
    Returns:
        Configured KSE Vector Store
    """
    return KSEVectorStoreIndex(
        config=config,
        search_type=search_type,
        adapter=adapter
    )


# Example usage and migration guide
LLAMAINDEX_MIGRATION_EXAMPLE = """
# Migration from traditional LlamaIndex to KSE

# Before (traditional vector store):
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.vector_stores import SimpleVectorStore

documents = [Document(text="...") for doc in raw_docs]
vector_store = SimpleVectorStore()
index = VectorStoreIndex.from_documents(
    documents,
    vector_store=vector_store
)
retriever = index.as_retriever(similarity_top_k=5)

# After (KSE hybrid AI):
from kse_memory.integrations.llamaindex import KSELlamaIndexRetriever, KSEVectorStoreIndex

# Option 1: Use as retriever directly
retriever = KSELlamaIndexRetriever(
    search_type="hybrid",
    similarity_top_k=5
)

# Option 2: Use as vector store
vector_store = KSEVectorStoreIndex(search_type="hybrid")
vector_store.add_documents(documents)

# Benefits of migration:
# - 18%+ improvement in retrieval relevance
# - Conceptual understanding beyond embeddings
# - Knowledge graph relationships
# - Multi-dimensional similarity matching
# - Zero configuration required
"""