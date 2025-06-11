"""
Core interfaces for KSE Memory SDK components.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from .models import Product, SearchQuery, SearchResult, ConceptualDimensions, EmbeddingVector, KnowledgeGraph


class AdapterInterface(ABC):
    """Interface for platform adapters (Shopify, WooCommerce, etc.)."""
    
    @abstractmethod
    async def connect(self, config: Dict[str, Any]) -> bool:
        """Connect to the platform."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from the platform."""
        pass
    
    @abstractmethod
    async def get_products(self, limit: int = 100, offset: int = 0) -> List[Product]:
        """Retrieve products from the platform."""
        pass
    
    @abstractmethod
    async def get_product(self, product_id: str) -> Optional[Product]:
        """Retrieve a specific product by ID."""
        pass
    
    @abstractmethod
    async def sync_products(self) -> int:
        """Sync all products from the platform."""
        pass
    
    @abstractmethod
    async def webhook_handler(self, event_type: str, payload: Dict[str, Any]) -> bool:
        """Handle webhook events from the platform."""
        pass


class VectorStoreInterface(ABC):
    """Interface for vector storage backends (Pinecone, Weaviate, etc.)."""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the vector store."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from the vector store."""
        pass
    
    @abstractmethod
    async def create_index(self, dimension: int, metric: str = "cosine") -> bool:
        """Create a new vector index."""
        pass
    
    @abstractmethod
    async def delete_index(self) -> bool:
        """Delete the vector index."""
        pass
    
    @abstractmethod
    async def upsert_vectors(self, vectors: List[Tuple[str, List[float], Dict[str, Any]]]) -> bool:
        """Insert or update vectors with metadata."""
        pass
    
    @abstractmethod
    async def delete_vectors(self, vector_ids: List[str]) -> bool:
        """Delete vectors by IDs."""
        pass
    
    @abstractmethod
    async def search_vectors(
        self, 
        query_vector: List[float], 
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    async def get_vector(self, vector_id: str) -> Optional[Tuple[List[float], Dict[str, Any]]]:
        """Get a specific vector by ID."""
        pass


class GraphStoreInterface(ABC):
    """Interface for graph storage backends (Neo4j, ArangoDB, etc.)."""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the graph store."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from the graph store."""
        pass
    
    @abstractmethod
    async def create_node(self, node_id: str, labels: List[str], properties: Dict[str, Any]) -> bool:
        """Create a new node."""
        pass
    
    @abstractmethod
    async def update_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """Update node properties."""
        pass
    
    @abstractmethod
    async def delete_node(self, node_id: str) -> bool:
        """Delete a node."""
        pass
    
    @abstractmethod
    async def create_relationship(
        self, 
        source_id: str, 
        target_id: str, 
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create a relationship between nodes."""
        pass
    
    @abstractmethod
    async def delete_relationship(self, source_id: str, target_id: str, relationship_type: str) -> bool:
        """Delete a relationship."""
        pass
    
    @abstractmethod
    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a node by ID."""
        pass
    
    @abstractmethod
    async def get_neighbors(self, node_id: str, relationship_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get neighboring nodes."""
        pass
    
    @abstractmethod
    async def find_path(self, source_id: str, target_id: str, max_depth: int = 3) -> Optional[List[Dict[str, Any]]]:
        """Find path between two nodes."""
        pass
    
    @abstractmethod
    async def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a custom graph query."""
        pass


class ConceptStoreInterface(ABC):
    """Interface for conceptual space storage backends (PostgreSQL, MongoDB, etc.)."""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the concept store."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from the concept store."""
        pass
    
    @abstractmethod
    async def store_conceptual_dimensions(self, product_id: str, dimensions: ConceptualDimensions) -> bool:
        """Store conceptual dimensions for a product."""
        pass
    
    @abstractmethod
    async def get_conceptual_dimensions(self, product_id: str) -> Optional[ConceptualDimensions]:
        """Get conceptual dimensions for a product."""
        pass
    
    @abstractmethod
    async def delete_conceptual_dimensions(self, product_id: str) -> bool:
        """Delete conceptual dimensions for a product."""
        pass
    
    @abstractmethod
    async def find_similar_concepts(
        self, 
        dimensions: ConceptualDimensions, 
        threshold: float = 0.8,
        limit: int = 10
    ) -> List[Tuple[str, float]]:
        """Find products with similar conceptual dimensions."""
        pass
    
    @abstractmethod
    async def get_dimension_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for each conceptual dimension."""
        pass


class EmbeddingServiceInterface(ABC):
    """Interface for embedding generation services."""
    
    @abstractmethod
    async def generate_text_embedding(self, text: str) -> EmbeddingVector:
        """Generate text embedding."""
        pass
    
    @abstractmethod
    async def generate_image_embedding(self, image_url: str) -> EmbeddingVector:
        """Generate image embedding."""
        pass
    
    @abstractmethod
    async def generate_batch_text_embeddings(self, texts: List[str]) -> List[EmbeddingVector]:
        """Generate embeddings for multiple texts."""
        pass
    
    @abstractmethod
    async def generate_batch_image_embeddings(self, image_urls: List[str]) -> List[EmbeddingVector]:
        """Generate embeddings for multiple images."""
        pass


class ConceptualServiceInterface(ABC):
    """Interface for conceptual dimension computation services."""
    
    @abstractmethod
    async def compute_dimensions(self, product: Product) -> ConceptualDimensions:
        """Compute conceptual dimensions for a product."""
        pass
    
    @abstractmethod
    async def compute_batch_dimensions(self, products: List[Product]) -> List[ConceptualDimensions]:
        """Compute conceptual dimensions for multiple products."""
        pass
    
    @abstractmethod
    async def explain_dimensions(self, product: Product, dimensions: ConceptualDimensions) -> str:
        """Explain why specific dimensions were assigned."""
        pass


class SearchServiceInterface(ABC):
    """Interface for search services."""
    
    @abstractmethod
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform a search query."""
        pass
    
    @abstractmethod
    async def semantic_search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Perform semantic search using embeddings."""
        pass
    
    @abstractmethod
    async def conceptual_search(self, dimensions: ConceptualDimensions, limit: int = 10) -> List[SearchResult]:
        """Perform conceptual search using conceptual dimensions."""
        pass
    
    @abstractmethod
    async def knowledge_graph_search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Perform search using knowledge graph relationships."""
        pass
    
    @abstractmethod
    async def hybrid_search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform hybrid search combining multiple approaches."""
        pass


class CacheInterface(ABC):
    """Interface for caching services."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries."""
        pass