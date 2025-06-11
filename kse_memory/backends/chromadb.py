"""
ChromaDB vector store backend for KSE Memory SDK.
"""

from typing import List, Dict, Any, Optional, Tuple
import asyncio
from ..core.interfaces import VectorStoreInterface
from ..core.models import Product, EmbeddingVector
from ..core.config import VectorStoreConfig
from ..exceptions import BackendError

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


class ChromaDBBackend(VectorStoreInterface):
    """ChromaDB vector store implementation."""
    
    def __init__(self, config: VectorStoreConfig):
        """Initialize ChromaDB backend."""
        if not CHROMADB_AVAILABLE:
            raise BackendError("chromadb package is required for ChromaDB backend", "chromadb")
        
        self.config = config
        self.client = None
        self.collection = None
        self._connected = False
    
    async def connect(self) -> bool:
        """Connect to ChromaDB instance."""
        try:
            # Create ChromaDB client
            if self.config.host and self.config.port:
                # Remote ChromaDB instance
                self.client = chromadb.HttpClient(
                    host=self.config.host,
                    port=self.config.port
                )
            else:
                # Local persistent ChromaDB
                self.client = chromadb.PersistentClient(
                    path="./chroma_db"
                )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name=self.config.collection_name or self.config.index_name
                )
            except Exception:
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=self.config.collection_name or self.config.index_name,
                    metadata={"hnsw:space": self.config.metric}
                )
            
            self._connected = True
            return True
            
        except Exception as e:
            raise BackendError(f"Failed to connect to ChromaDB: {str(e)}", "chromadb")
    
    async def disconnect(self) -> bool:
        """Disconnect from ChromaDB."""
        self._connected = False
        return True
    
    async def create_index(self, dimension: int, metric: str = "cosine") -> bool:
        """Create a new collection in ChromaDB."""
        if not self._connected:
            raise BackendError("Not connected to ChromaDB", "chromadb")
        
        try:
            collection_name = f"{self.config.index_name}_{dimension}d"
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": metric}
            )
            return True
            
        except Exception as e:
            raise BackendError(f"Failed to create ChromaDB collection: {str(e)}", "chromadb")
    
    async def delete_index(self) -> bool:
        """Delete the ChromaDB collection."""
        if not self._connected:
            raise BackendError("Not connected to ChromaDB", "chromadb")
        
        try:
            if self.collection:
                self.client.delete_collection(self.collection.name)
                self.collection = None
            return True
            
        except Exception as e:
            raise BackendError(f"Failed to delete ChromaDB collection: {str(e)}", "chromadb")
    
    async def upsert_vectors(self, vectors: List[Tuple[str, List[float], Dict[str, Any]]]) -> bool:
        """Insert or update vectors with metadata."""
        if not self._connected or not self.collection:
            raise BackendError("Not connected to ChromaDB or no collection", "chromadb")
        
        try:
            ids = []
            embeddings = []
            metadatas = []
            
            for vector_id, vector, metadata in vectors:
                ids.append(vector_id)
                embeddings.append(vector)
                metadatas.append(metadata)
            
            # ChromaDB upsert operation
            await asyncio.to_thread(
                self.collection.upsert,
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            return True
            
        except Exception as e:
            raise BackendError(f"Failed to upsert vectors to ChromaDB: {str(e)}", "chromadb")
    
    async def delete_vectors(self, vector_ids: List[str]) -> bool:
        """Delete vectors by IDs."""
        if not self._connected or not self.collection:
            raise BackendError("Not connected to ChromaDB or no collection", "chromadb")
        
        try:
            await asyncio.to_thread(
                self.collection.delete,
                ids=vector_ids
            )
            return True
            
        except Exception as e:
            raise BackendError(f"Failed to delete vectors from ChromaDB: {str(e)}", "chromadb")
    
    async def search_vectors(
        self, 
        query_vector: List[float], 
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar vectors."""
        if not self._connected or not self.collection:
            raise BackendError("Not connected to ChromaDB or no collection", "chromadb")
        
        try:
            # Convert filters to ChromaDB format
            where_clause = None
            if filters:
                where_clause = {}
                for key, value in filters.items():
                    where_clause[key] = {"$eq": value}
            
            results = await asyncio.to_thread(
                self.collection.query,
                query_embeddings=[query_vector],
                n_results=top_k,
                where=where_clause,
                include=["metadatas", "distances"]
            )
            
            # Convert results to expected format
            output = []
            if results["ids"] and len(results["ids"]) > 0:
                for i, vector_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i]
                    # Convert distance to similarity score (ChromaDB returns distances)
                    score = 1.0 / (1.0 + distance)
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    output.append((vector_id, score, metadata))
            
            return output
            
        except Exception as e:
            raise BackendError(f"Failed to search vectors in ChromaDB: {str(e)}", "chromadb")
    
    async def get_vector(self, vector_id: str) -> Optional[Tuple[List[float], Dict[str, Any]]]:
        """Get a specific vector by ID."""
        if not self._connected or not self.collection:
            raise BackendError("Not connected to ChromaDB or no collection", "chromadb")
        
        try:
            results = await asyncio.to_thread(
                self.collection.get,
                ids=[vector_id],
                include=["embeddings", "metadatas"]
            )
            
            if results["ids"] and len(results["ids"]) > 0:
                embedding = results["embeddings"][0] if results["embeddings"] else []
                metadata = results["metadatas"][0] if results["metadatas"] else {}
                return (embedding, metadata)
            
            return None
            
        except Exception as e:
            raise BackendError(f"Failed to get vector from ChromaDB: {str(e)}", "chromadb")
    
    # Additional methods for compatibility
    async def add_product(self, product: Product) -> bool:
        """Add product to ChromaDB."""
        if not product.text_embedding:
            raise BackendError("Product must have text embedding", "chromadb")
        
        metadata = {
            "title": product.title,
            "description": product.description,
            "category": product.category,
            "price": product.price,
            "tags": product.tags,
        }
        
        return await self.upsert_vectors([(product.id, product.text_embedding.vector, metadata)])
    
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
        vectors = []
        for product in products:
            if product.text_embedding:
                metadata = {
                    "title": product.title,
                    "description": product.description,
                    "category": product.category,
                    "price": product.price,
                    "tags": product.tags,
                }
                vectors.append((product.id, product.text_embedding.vector, metadata))
        
        if vectors:
            return await self.upsert_vectors(vectors)
        return True
    
    async def clear_collection(self) -> bool:
        """Clear all data from ChromaDB collection."""
        if not self._connected or not self.collection:
            raise BackendError("Not connected to ChromaDB or no collection", "chromadb")
        
        try:
            # Get all IDs and delete them
            results = await asyncio.to_thread(
                self.collection.get,
                include=[]
            )
            
            if results["ids"]:
                await self.delete_vectors(results["ids"])
            
            return True
            
        except Exception as e:
            raise BackendError(f"Failed to clear ChromaDB collection: {str(e)}", "chromadb")