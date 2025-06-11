"""
Mock vector store backend for testing KSE Memory SDK.
"""

from typing import List, Dict, Any, Optional, Tuple
from ..core.interfaces import VectorStoreInterface
from ..core.models import Product, EmbeddingVector
from ..core.config import VectorStoreConfig
from ..exceptions import BackendError


class MockVectorStore(VectorStoreInterface):
    """Mock vector store implementation for testing."""
    
    def __init__(self, config: VectorStoreConfig):
        """Initialize mock backend."""
        self.config = config
        self._connected = False
        self._vectors = {}  # Store vectors in memory
        self._metadata = {}  # Store metadata in memory
    
    async def connect(self) -> bool:
        """Connect to mock store."""
        self._connected = True
        return True
    
    async def disconnect(self) -> bool:
        """Disconnect from mock store."""
        self._connected = False
        return True
    
    async def create_index(self, dimension: int, metric: str = "cosine") -> bool:
        """Create a mock index."""
        if not self._connected:
            raise BackendError("Not connected to mock store", "mock")
        return True
    
    async def delete_index(self) -> bool:
        """Delete the mock index."""
        if not self._connected:
            raise BackendError("Not connected to mock store", "mock")
        self._vectors.clear()
        self._metadata.clear()
        return True
    
    async def upsert_vectors(self, vectors: List[Tuple[str, List[float], Dict[str, Any]]]) -> bool:
        """Insert or update vectors with metadata."""
        if not self._connected:
            raise BackendError("Not connected to mock store", "mock")
        
        for vector_id, vector, metadata in vectors:
            self._vectors[vector_id] = vector
            self._metadata[vector_id] = metadata
        
        return True
    
    async def delete_vectors(self, vector_ids: List[str]) -> bool:
        """Delete vectors by IDs."""
        if not self._connected:
            raise BackendError("Not connected to mock store", "mock")
        
        for vector_id in vector_ids:
            self._vectors.pop(vector_id, None)
            self._metadata.pop(vector_id, None)
        
        return True
    
    async def search_vectors(
        self, 
        query_vector: List[float], 
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar vectors."""
        if not self._connected:
            raise BackendError("Not connected to mock store", "mock")
        
        # Simple mock search - return stored vectors with mock scores
        results = []
        for vector_id, vector in list(self._vectors.items())[:top_k]:
            # Mock similarity score
            score = 0.8 - (len(results) * 0.1)  # Decreasing scores
            metadata = self._metadata.get(vector_id, {})
            
            # Apply simple filters if provided
            if filters:
                match = True
                for key, value in filters.items():
                    if key in metadata and metadata[key] != value:
                        match = False
                        break
                if not match:
                    continue
            
            results.append((vector_id, score, metadata))
        
        return results
    
    async def get_vector(self, vector_id: str) -> Optional[Tuple[List[float], Dict[str, Any]]]:
        """Get a specific vector by ID."""
        if not self._connected:
            raise BackendError("Not connected to mock store", "mock")
        
        if vector_id in self._vectors:
            return (self._vectors[vector_id], self._metadata[vector_id])
        
        return None
    
    # Additional methods for compatibility with existing code
    async def add_product(self, product: Product) -> bool:
        """Add product to mock store."""
        if not self._connected:
            raise BackendError("Not connected to mock store", "mock")
        
        if not product.text_embedding:
            # Create a mock embedding
            mock_vector = [0.1] * self.config.dimension
            product.text_embedding = EmbeddingVector(
                vector=mock_vector,
                model="mock-model",
                dimension=self.config.dimension
            )
        
        metadata = {
            "title": product.title,
            "description": product.description,
            "category": product.category,
            "price": product.price,
            "tags": product.tags,
        }
        
        self._vectors[product.id] = product.text_embedding.vector
        self._metadata[product.id] = metadata
        
        return True
    
    async def search_similar(self, query_vector: List[float], limit: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        results = await self.search_vectors(query_vector, limit, filters)
        
        return [
            {
                "id": vector_id,
                "score": score,
                "payload": metadata
            }
            for vector_id, score, metadata in results
        ]
    
    async def get_product(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Get product by ID."""
        result = await self.get_vector(product_id)
        if result:
            vector, metadata = result
            return {
                "id": product_id,
                "vector": vector,
                "payload": metadata
            }
        return None
    
    async def update_product(self, product: Product) -> bool:
        """Update product."""
        return await self.add_product(product)
    
    async def delete_product(self, product_id: str) -> bool:
        """Delete product."""
        return await self.delete_vectors([product_id])
    
    async def bulk_add_products(self, products: List[Product]) -> bool:
        """Bulk add products."""
        for product in products:
            await self.add_product(product)
        return True
    
    async def clear_collection(self) -> bool:
        """Clear all data."""
        self._vectors.clear()
        self._metadata.clear()
        return True