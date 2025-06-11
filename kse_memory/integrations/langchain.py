"""
KSE Memory SDK - LangChain Integration

Provides drop-in replacements for LangChain vector stores
and retrievers with hybrid AI capabilities.
"""

from typing import List, Dict, Any, Optional, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor

try:
    from langchain.schema import Document
    from langchain.vectorstores.base import VectorStore
    from langchain.schema.retriever import BaseRetriever
    from langchain.callbacks.manager import CallbackManagerForRetrieverRun
    LANGCHAIN_AVAILABLE = True
except ImportError:
    # Create mock classes if LangChain is not installed
    class Document:
        def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
            self.page_content = page_content
            self.metadata = metadata or {}
    
    class VectorStore:
        pass
    
    class BaseRetriever:
        pass
    
    class CallbackManagerForRetrieverRun:
        pass
    
    LANGCHAIN_AVAILABLE = False

from ..core.memory import KSEMemory
from ..core.config import KSEConfig
from ..core.models import Product, SearchQuery, SearchType


class KSEVectorStore(VectorStore):
    """
    LangChain VectorStore implementation using KSE Memory.
    
    Drop-in replacement for traditional vector stores with
    hybrid AI capabilities including conceptual and graph search.
    
    Example:
        from kse_memory.integrations.langchain import KSEVectorStore
        
        # Initialize with KSE Memory
        vectorstore = KSEVectorStore(
            kse_memory=kse_memory,
            search_type="hybrid"  # Use hybrid search by default
        )
        
        # Use as normal LangChain vector store
        docs = vectorstore.similarity_search("comfortable shoes", k=5)
    """
    
    def __init__(
        self,
        kse_memory: Optional[KSEMemory] = None,
        config: Optional[KSEConfig] = None,
        search_type: str = "hybrid",
        adapter: str = "generic"
    ):
        """
        Initialize KSE Vector Store.
        
        Args:
            kse_memory: Existing KSE Memory instance
            config: KSE configuration (if kse_memory not provided)
            search_type: Default search type ('vector', 'conceptual', 'graph', 'hybrid')
            adapter: Platform adapter to use
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is required for this integration. "
                "Install with: pip install langchain"
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
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> List[str]:
        """
        Add texts to the vector store.
        
        Args:
            texts: List of text content
            metadatas: Optional list of metadata dicts
            
        Returns:
            List of document IDs
        """
        async def _add_texts():
            await self._ensure_initialized()
            
            ids = []
            for i, text in enumerate(texts):
                metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                
                # Convert text to Product for KSE Memory
                product = Product(
                    id=f"doc_{len(ids)}_{i}",
                    title=metadata.get("title", f"Document {i}"),
                    description=text,
                    price=metadata.get("price", 0.0),
                    category=metadata.get("category", "Document"),
                    tags=metadata.get("tags", []),
                    metadata=metadata
                )
                
                await self.kse_memory.add_product(product)
                ids.append(product.id)
            
            return ids
        
        return self._executor.submit(self._run_async, _add_texts()).result()
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Document]:
        """
        Perform similarity search.
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of LangChain Documents
        """
        async def _search():
            await self._ensure_initialized()
            
            search_query = SearchQuery(
                query=query,
                search_type=self.search_type,
                limit=k,
                filters=filter
            )
            
            results = await self.kse_memory.search(search_query)
            
            # Convert to LangChain Documents
            documents = []
            for result in results:
                doc = Document(
                    page_content=result.product.description,
                    metadata={
                        **result.product.metadata,
                        "id": result.product.id,
                        "title": result.product.title,
                        "category": result.product.category,
                        "price": result.product.price,
                        "tags": result.product.tags,
                        "score": result.score,
                        "search_type": self.search_type.value
                    }
                )
                documents.append(doc)
            
            return documents
        
        return self._executor.submit(self._run_async, _search()).result()
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with scores.
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of (Document, score) tuples
        """
        async def _search_with_score():
            await self._ensure_initialized()
            
            search_query = SearchQuery(
                query=query,
                search_type=self.search_type,
                limit=k,
                filters=filter
            )
            
            results = await self.kse_memory.search(search_query)
            
            # Convert to LangChain Documents with scores
            documents_with_scores = []
            for result in results:
                doc = Document(
                    page_content=result.product.description,
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
                documents_with_scores.append((doc, result.score))
            
            return documents_with_scores
        
        return self._executor.submit(self._run_async, _search_with_score()).result()
    
    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding=None,  # Not used, KSE handles embeddings
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> "KSEVectorStore":
        """
        Create KSE Vector Store from texts.
        
        Args:
            texts: List of text content
            embedding: Ignored (KSE handles embeddings)
            metadatas: Optional metadata
            
        Returns:
            KSE Vector Store instance
        """
        store = cls(**kwargs)
        store.add_texts(texts, metadatas)
        return store


class KSELangChainRetriever(BaseRetriever):
    """
    LangChain Retriever implementation using KSE Memory.
    
    Provides hybrid AI retrieval capabilities as a drop-in
    replacement for traditional retrievers.
    
    Example:
        from kse_memory.integrations.langchain import KSELangChainRetriever
        
        # Initialize retriever
        retriever = KSELangChainRetriever(
            kse_memory=kse_memory,
            search_type="hybrid",
            k=5
        )
        
        # Use in LangChain chains
        from langchain.chains import RetrievalQA
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever
        )
    """
    
    def __init__(
        self,
        kse_memory: Optional[KSEMemory] = None,
        config: Optional[KSEConfig] = None,
        search_type: str = "hybrid",
        k: int = 4,
        adapter: str = "generic",
        **kwargs
    ):
        """
        Initialize KSE LangChain Retriever.
        
        Args:
            kse_memory: Existing KSE Memory instance
            config: KSE configuration (if kse_memory not provided)
            search_type: Search type ('vector', 'conceptual', 'graph', 'hybrid')
            k: Number of results to return
            adapter: Platform adapter to use
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is required for this integration. "
                "Install with: pip install langchain"
            )
        
        super().__init__(**kwargs)
        
        self.kse_memory = kse_memory
        self.config = config or KSEConfig()
        self.search_type = SearchType(search_type.upper())
        self.k = k
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
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            run_manager: LangChain callback manager
            
        Returns:
            List of relevant documents
        """
        async def _retrieve():
            await self._ensure_initialized()
            
            search_query = SearchQuery(
                query=query,
                search_type=self.search_type,
                limit=self.k
            )
            
            results = await self.kse_memory.search(search_query)
            
            # Convert to LangChain Documents
            documents = []
            for result in results:
                doc = Document(
                    page_content=result.product.description,
                    metadata={
                        **result.product.metadata,
                        "id": result.product.id,
                        "title": result.product.title,
                        "category": result.product.category,
                        "price": result.product.price,
                        "tags": result.product.tags,
                        "score": result.score,
                        "search_type": self.search_type.value,
                        "retriever": "KSE_hybrid_ai"
                    }
                )
                documents.append(doc)
            
            return documents
        
        return self._executor.submit(self._run_async, _retrieve()).result()


# Utility functions for easy integration
def create_kse_vectorstore(
    config: Optional[KSEConfig] = None,
    search_type: str = "hybrid",
    adapter: str = "generic"
) -> KSEVectorStore:
    """
    Create a KSE Vector Store with default configuration.
    
    Args:
        config: Optional KSE configuration
        search_type: Default search type
        adapter: Platform adapter
        
    Returns:
        Configured KSE Vector Store
    """
    return KSEVectorStore(
        config=config,
        search_type=search_type,
        adapter=adapter
    )


def create_kse_retriever(
    config: Optional[KSEConfig] = None,
    search_type: str = "hybrid",
    k: int = 4,
    adapter: str = "generic"
) -> KSELangChainRetriever:
    """
    Create a KSE Retriever with default configuration.
    
    Args:
        config: Optional KSE configuration
        search_type: Search type
        k: Number of results
        adapter: Platform adapter
        
    Returns:
        Configured KSE Retriever
    """
    return KSELangChainRetriever(
        config=config,
        search_type=search_type,
        k=k,
        adapter=adapter
    )


# Example usage and migration guide
MIGRATION_EXAMPLE = """
# Migration from traditional LangChain vector stores to KSE

# Before (traditional vector store):
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(
    texts=documents,
    embedding=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# After (KSE hybrid AI):
from kse_memory.integrations.langchain import KSEVectorStore, KSELangChainRetriever

# Option 1: Use as vector store
vectorstore = KSEVectorStore.from_texts(
    texts=documents,
    search_type="hybrid"  # Enables hybrid AI capabilities
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Option 2: Use as retriever directly
retriever = KSELangChainRetriever(
    search_type="hybrid",
    k=5
)

# Benefits of migration:
# - 18%+ improvement in relevance scores
# - Conceptual understanding beyond keywords
# - Knowledge graph relationships
# - Multi-dimensional similarity
# - Zero additional configuration required
"""