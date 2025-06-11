"""
Pinecone vector store backend for KSE Memory SDK.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple

try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

from ..core.interfaces import VectorStoreInterface
from ..core.config import VectorStoreConfig
from ..exceptions import VectorStoreError, AuthenticationError


logger = logging.getLogger(__name__)


class PineconeBackend(VectorStoreInterface):
    """
    Pinecone vector store backend for KSE Memory SDK.
    
    Provides vector storage and similarity search using Pinecone's
    managed vector database service.
    """
    
    def __init__(self, config: VectorStoreConfig):
        """
        Initialize Pinecone backend.
        
        Args:
            config: Vector store configuration
        """
        if not PINECONE_AVAILABLE:
            raise VectorStoreError(
                "Pinecone dependencies not installed. Install with: pip install kse-memory[pinecone]",
                operation="initialization"
            )
        
        self.config = config
        self.index = None
        self._connected = False
        
        logger.info("Pinecone backend initialized")
    
    async def connect(self) -> bool:
        """
        Connect to Pinecone service.
        
        Returns:
            True if connection successful
            
        Raises:
            VectorStoreError: If connection fails
            AuthenticationError: If authentication fails
        """
        try:
            if not self.config.api_key:
                raise AuthenticationError(
                    "Pinecone API key is required",
                    service="pinecone"
                )
            
            # Initialize Pinecone
            pinecone.init(
                api_key=self.config.api_key,
                environment=self.config.environment or "us-west1-gcp"
            )
            
            # Check if index exists
            existing_indexes = pinecone.list_indexes()
            
            if self.config.index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index: {self.config.index_name}")
                await self.create_index(self.config.dimension, self.config.metric)
            
            # Connect to index
            self.index = pinecone.Index(self.config.index_name)
            
            # Test connection
            stats = self.index.describe_index_stats()
            logger.info(f"Connected to Pinecone index: {self.config.index_name} ({stats.get('total_vector_count', 0)} vectors)")
            
            self._connected = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone: {str(e)}")
            if "api key" in str(e).lower() or "unauthorized" in str(e).lower():
                raise AuthenticationError(f"Pinecone authentication failed: {str(e)}", service="pinecone")
            raise VectorStoreError(f"Connection failed: {str(e)}", operation="connect")
    
    async def disconnect(self) -> bool:
        """
        Disconnect from Pinecone service.
        
        Returns:
            True if disconnection successful
        """
        try:
            self.index = None
            self._connected = False
            logger.info("Disconnected from Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"Error during Pinecone disconnection: {str(e)}")
            return False
    
    async def create_index(self, dimension: int, metric: str = "cosine") -> bool:
        """
        Create a new Pinecone index.
        
        Args:
            dimension: Vector dimension
            metric: Distance metric (cosine, euclidean, dotproduct)
            
        Returns:
            True if index created successfully
            
        Raises:
            VectorStoreError: If index creation fails
        """
        try:
            pinecone.create_index(
                name=self.config.index_name,
                dimension=dimension,
                metric=metric
            )
            
            # Wait for index to be ready
            await asyncio.sleep(5)
            
            logger.info(f"Created Pinecone index: {self.config.index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create Pinecone index: {str(e)}")
            raise VectorStoreError(f"Index creation failed: {str(e)}", operation="create_index")
    
    async def delete_index(self) -> bool:
        """
        Delete the Pinecone index.
        
        Returns:
            True if index deleted successfully
            
        Raises:
            VectorStoreError: If index deletion fails
        """
        try:
            pinecone.delete_index(self.config.index_name)
            self.index = None
            self._connected = False
            
            logger.info(f"Deleted Pinecone index: {self.config.index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete Pinecone index: {str(e)}")
            raise VectorStoreError(f"Index deletion failed: {str(e)}", operation="delete_index")
    
    async def upsert_vectors(self, vectors: List[Tuple[str, List[float], Dict[str, Any]]]) -> bool:
        """
        Insert or update vectors in Pinecone.
        
        Args:
            vectors: List of (id, vector, metadata) tuples
            
        Returns:
            True if upsert successful
            
        Raises:
            VectorStoreError: If upsert fails
        """
        self._ensure_connected()
        
        try:
            # Convert to Pinecone format
            pinecone_vectors = []
            for vector_id, vector, metadata in vectors:
                # Pinecone has metadata size limits, so we need to be selective
                filtered_metadata = self._filter_metadata(metadata)
                
                pinecone_vectors.append({
                    "id": vector_id,
                    "values": vector,
                    "metadata": filtered_metadata
                })
            
            # Upsert in batches (Pinecone has batch size limits)
            batch_size = 100
            for i in range(0, len(pinecone_vectors), batch_size):
                batch = pinecone_vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
            
            logger.debug(f"Upserted {len(vectors)} vectors to Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upsert vectors to Pinecone: {str(e)}")
            raise VectorStoreError(f"Vector upsert failed: {str(e)}", operation="upsert")
    
    async def delete_vectors(self, vector_ids: List[str]) -> bool:
        """
        Delete vectors from Pinecone.
        
        Args:
            vector_ids: List of vector IDs to delete
            
        Returns:
            True if deletion successful
            
        Raises:
            VectorStoreError: If deletion fails
        """
        self._ensure_connected()
        
        try:
            # Delete in batches
            batch_size = 1000  # Pinecone delete batch limit
            for i in range(0, len(vector_ids), batch_size):
                batch = vector_ids[i:i + batch_size]
                self.index.delete(ids=batch)
            
            logger.debug(f"Deleted {len(vector_ids)} vectors from Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete vectors from Pinecone: {str(e)}")
            raise VectorStoreError(f"Vector deletion failed: {str(e)}", operation="delete")
    
    async def search_vectors(
        self, 
        query_vector: List[float], 
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar vectors in Pinecone.
        
        Args:
            query_vector: Query vector
            top_k: Number of results to return
            filters: Metadata filters
            
        Returns:
            List of (id, score, metadata) tuples
            
        Raises:
            VectorStoreError: If search fails
        """
        self._ensure_connected()
        
        try:
            # Convert filters to Pinecone format
            pinecone_filter = self._convert_filters(filters) if filters else None
            
            # Perform search
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                filter=pinecone_filter,
                include_metadata=True
            )
            
            # Convert results
            search_results = []
            for match in results.get("matches", []):
                search_results.append((
                    match["id"],
                    float(match["score"]),
                    match.get("metadata", {})
                ))
            
            logger.debug(f"Pinecone search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to search vectors in Pinecone: {str(e)}")
            raise VectorStoreError(f"Vector search failed: {str(e)}", operation="search")
    
    async def get_vector(self, vector_id: str) -> Optional[Tuple[List[float], Dict[str, Any]]]:
        """
        Get a specific vector from Pinecone.
        
        Args:
            vector_id: Vector ID
            
        Returns:
            (vector, metadata) tuple if found, None otherwise
            
        Raises:
            VectorStoreError: If retrieval fails
        """
        self._ensure_connected()
        
        try:
            # Fetch vector
            results = self.index.fetch(ids=[vector_id])
            
            if vector_id in results.get("vectors", {}):
                vector_data = results["vectors"][vector_id]
                return (
                    vector_data.get("values", []),
                    vector_data.get("metadata", {})
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get vector {vector_id} from Pinecone: {str(e)}")
            raise VectorStoreError(f"Vector retrieval failed: {str(e)}", operation="get")
    
    def _filter_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter metadata to comply with Pinecone limitations.
        
        Args:
            metadata: Original metadata
            
        Returns:
            Filtered metadata
        """
        # Pinecone has limitations on metadata:
        # - Max 40KB per vector
        # - Only certain data types supported
        # - Nested objects have limited depth
        
        filtered = {}
        
        # Include only essential fields that are likely to be used for filtering
        essential_fields = [
            "id", "title", "category", "brand", "price", "currency",
            "tags", "created_at", "updated_at"
        ]
        
        for field in essential_fields:
            if field in metadata:
                value = metadata[field]
                
                # Convert to Pinecone-compatible types
                if isinstance(value, (str, int, float, bool)):
                    filtered[field] = value
                elif isinstance(value, list):
                    # Convert list to string for Pinecone
                    if all(isinstance(item, str) for item in value):
                        filtered[field] = ",".join(value[:10])  # Limit to 10 items
                elif isinstance(value, dict):
                    # Flatten simple dicts
                    for k, v in value.items():
                        if isinstance(v, (str, int, float, bool)):
                            filtered[f"{field}_{k}"] = v
        
        return filtered
    
    def _convert_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert filters to Pinecone format.
        
        Args:
            filters: Original filters
            
        Returns:
            Pinecone-compatible filters
        """
        # Pinecone uses a specific filter format
        # This is a simplified conversion
        pinecone_filter = {}
        
        for key, value in filters.items():
            if isinstance(value, (str, int, float, bool)):
                pinecone_filter[key] = {"$eq": value}
            elif isinstance(value, list):
                pinecone_filter[key] = {"$in": value}
            elif isinstance(value, dict):
                if "$gt" in value or "$lt" in value or "$gte" in value or "$lte" in value:
                    pinecone_filter[key] = value
                else:
                    pinecone_filter[key] = {"$eq": value}
        
        return pinecone_filter
    
    def _ensure_connected(self):
        """Ensure the backend is connected."""
        if not self._connected:
            raise VectorStoreError("Not connected to Pinecone. Call connect() first.", operation="check_connection")