"""
Weaviate vector store backend for KSE Memory SDK.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple

try:
    import weaviate
    from weaviate import Client
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False

from ..core.interfaces import VectorStoreInterface
from ..core.config import VectorStoreConfig
from ..exceptions import VectorStoreError, AuthenticationError


logger = logging.getLogger(__name__)


class WeaviateBackend(VectorStoreInterface):
    """
    Weaviate vector store backend for KSE Memory SDK.
    
    Provides vector storage and similarity search using Weaviate's
    open-source vector database.
    """
    
    def __init__(self, config: VectorStoreConfig):
        """
        Initialize Weaviate backend.
        
        Args:
            config: Vector store configuration
        """
        if not WEAVIATE_AVAILABLE:
            raise VectorStoreError(
                "Weaviate dependencies not installed. Install with: pip install kse-memory[weaviate]",
                operation="initialization"
            )
        
        self.config = config
        self.client: Optional[Client] = None
        self._connected = False
        self.class_name = "KSEProduct"
        
        logger.info("Weaviate backend initialized")
    
    async def connect(self) -> bool:
        """
        Connect to Weaviate instance.
        
        Returns:
            True if connection successful
            
        Raises:
            VectorStoreError: If connection fails
            AuthenticationError: If authentication fails
        """
        try:
            # Build Weaviate URL
            url = f"http://{self.config.host}:{self.config.port}"
            
            # Create client with optional authentication
            auth_config = None
            if self.config.api_key:
                auth_config = weaviate.AuthApiKey(api_key=self.config.api_key)
            
            self.client = weaviate.Client(
                url=url,
                auth_client_secret=auth_config,
                timeout_config=(5, 15)  # (connect, read) timeouts
            )
            
            # Test connection
            if not self.client.is_ready():
                raise VectorStoreError("Weaviate instance is not ready", operation="connect")
            
            # Create schema if it doesn't exist
            await self._create_schema()
            
            self._connected = True
            logger.info(f"Connected to Weaviate at {url}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {str(e)}")
            if "authentication" in str(e).lower() or "unauthorized" in str(e).lower():
                raise AuthenticationError(f"Weaviate authentication failed: {str(e)}", service="weaviate")
            raise VectorStoreError(f"Connection failed: {str(e)}", operation="connect")
    
    async def disconnect(self) -> bool:
        """
        Disconnect from Weaviate instance.
        
        Returns:
            True if disconnection successful
        """
        try:
            if self.client:
                # Weaviate client doesn't have explicit close method
                self.client = None
            
            self._connected = False
            logger.info("Disconnected from Weaviate")
            return True
            
        except Exception as e:
            logger.error(f"Error during Weaviate disconnection: {str(e)}")
            return False
    
    async def create_index(self, dimension: int, metric: str = "cosine") -> bool:
        """
        Create a new Weaviate class (index).
        
        Args:
            dimension: Vector dimension
            metric: Distance metric (cosine, euclidean, dot)
            
        Returns:
            True if index created successfully
            
        Raises:
            VectorStoreError: If index creation fails
        """
        try:
            # Delete existing class if it exists
            if self.client.schema.exists(self.class_name):
                self.client.schema.delete_class(self.class_name)
            
            # Map metric to Weaviate format
            distance_metric = {
                "cosine": "cosine",
                "euclidean": "l2-squared",
                "dotproduct": "dot"
            }.get(metric, "cosine")
            
            # Create class schema
            class_schema = {
                "class": self.class_name,
                "description": "KSE Memory products with vector embeddings",
                "vectorizer": "none",  # We provide our own vectors
                "moduleConfig": {
                    "text2vec-transformers": {
                        "skip": True
                    }
                },
                "vectorIndexConfig": {
                    "distance": distance_metric,
                    "ef": 64,
                    "efConstruction": 128,
                    "maxConnections": 64
                },
                "properties": [
                    {
                        "name": "productId",
                        "dataType": ["string"],
                        "description": "Unique product identifier"
                    },
                    {
                        "name": "title",
                        "dataType": ["string"],
                        "description": "Product title"
                    },
                    {
                        "name": "description",
                        "dataType": ["text"],
                        "description": "Product description"
                    },
                    {
                        "name": "category",
                        "dataType": ["string"],
                        "description": "Product category"
                    },
                    {
                        "name": "brand",
                        "dataType": ["string"],
                        "description": "Product brand"
                    },
                    {
                        "name": "price",
                        "dataType": ["number"],
                        "description": "Product price"
                    },
                    {
                        "name": "currency",
                        "dataType": ["string"],
                        "description": "Price currency"
                    },
                    {
                        "name": "tags",
                        "dataType": ["string[]"],
                        "description": "Product tags"
                    },
                    {
                        "name": "metadata",
                        "dataType": ["object"],
                        "description": "Additional product metadata"
                    }
                ]
            }
            
            self.client.schema.create_class(class_schema)
            
            logger.info(f"Created Weaviate class: {self.class_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create Weaviate class: {str(e)}")
            raise VectorStoreError(f"Index creation failed: {str(e)}", operation="create_index")
    
    async def delete_index(self) -> bool:
        """
        Delete the Weaviate class (index).
        
        Returns:
            True if index deleted successfully
            
        Raises:
            VectorStoreError: If index deletion fails
        """
        try:
            if self.client.schema.exists(self.class_name):
                self.client.schema.delete_class(self.class_name)
                logger.info(f"Deleted Weaviate class: {self.class_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete Weaviate class: {str(e)}")
            raise VectorStoreError(f"Index deletion failed: {str(e)}", operation="delete_index")
    
    async def upsert_vectors(self, vectors: List[Tuple[str, List[float], Dict[str, Any]]]) -> bool:
        """
        Insert or update vectors in Weaviate.
        
        Args:
            vectors: List of (id, vector, metadata) tuples
            
        Returns:
            True if upsert successful
            
        Raises:
            VectorStoreError: If upsert fails
        """
        self._ensure_connected()
        
        try:
            # Prepare objects for batch import
            objects = []
            for vector_id, vector, metadata in vectors:
                # Filter metadata for Weaviate properties
                filtered_metadata = self._filter_metadata(metadata)
                
                obj = {
                    "class": self.class_name,
                    "id": vector_id,
                    "vector": vector,
                    "properties": filtered_metadata
                }
                objects.append(obj)
            
            # Batch import
            with self.client.batch as batch:
                batch.batch_size = 100
                for obj in objects:
                    batch.add_data_object(
                        data_object=obj["properties"],
                        class_name=obj["class"],
                        uuid=obj["id"],
                        vector=obj["vector"]
                    )
            
            logger.debug(f"Upserted {len(vectors)} vectors to Weaviate")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upsert vectors to Weaviate: {str(e)}")
            raise VectorStoreError(f"Vector upsert failed: {str(e)}", operation="upsert")
    
    async def delete_vectors(self, vector_ids: List[str]) -> bool:
        """
        Delete vectors from Weaviate.
        
        Args:
            vector_ids: List of vector IDs to delete
            
        Returns:
            True if deletion successful
            
        Raises:
            VectorStoreError: If deletion fails
        """
        self._ensure_connected()
        
        try:
            for vector_id in vector_ids:
                self.client.data_object.delete(
                    uuid=vector_id,
                    class_name=self.class_name
                )
            
            logger.debug(f"Deleted {len(vector_ids)} vectors from Weaviate")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete vectors from Weaviate: {str(e)}")
            raise VectorStoreError(f"Vector deletion failed: {str(e)}", operation="delete")
    
    async def search_vectors(
        self, 
        query_vector: List[float], 
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar vectors in Weaviate.
        
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
            # Build query
            query = (
                self.client.query
                .get(self.class_name, ["productId", "title", "description", "category", "brand", "price", "currency", "tags", "metadata"])
                .with_near_vector({"vector": query_vector})
                .with_limit(top_k)
                .with_additional(["certainty", "distance"])
            )
            
            # Add filters if provided
            if filters:
                where_filter = self._build_where_filter(filters)
                if where_filter:
                    query = query.with_where(where_filter)
            
            # Execute query
            result = query.do()
            
            # Process results
            search_results = []
            if "data" in result and "Get" in result["data"] and self.class_name in result["data"]["Get"]:
                for item in result["data"]["Get"][self.class_name]:
                    # Extract metadata
                    metadata = {
                        "id": item.get("productId"),
                        "title": item.get("title"),
                        "description": item.get("description"),
                        "category": item.get("category"),
                        "brand": item.get("brand"),
                        "price": item.get("price"),
                        "currency": item.get("currency"),
                        "tags": item.get("tags", []),
                    }
                    
                    # Add custom metadata
                    if item.get("metadata"):
                        metadata.update(item["metadata"])
                    
                    # Get similarity score (certainty in Weaviate)
                    additional = item.get("_additional", {})
                    score = additional.get("certainty", 0.0)
                    
                    search_results.append((
                        item.get("productId"),
                        float(score),
                        metadata
                    ))
            
            logger.debug(f"Weaviate search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to search vectors in Weaviate: {str(e)}")
            raise VectorStoreError(f"Vector search failed: {str(e)}", operation="search")
    
    async def get_vector(self, vector_id: str) -> Optional[Tuple[List[float], Dict[str, Any]]]:
        """
        Get a specific vector from Weaviate.
        
        Args:
            vector_id: Vector ID
            
        Returns:
            (vector, metadata) tuple if found, None otherwise
            
        Raises:
            VectorStoreError: If retrieval fails
        """
        self._ensure_connected()
        
        try:
            # Get object by ID
            result = self.client.data_object.get_by_id(
                uuid=vector_id,
                class_name=self.class_name,
                with_vector=True
            )
            
            if result:
                vector = result.get("vector", [])
                properties = result.get("properties", {})
                
                # Build metadata
                metadata = {
                    "id": properties.get("productId"),
                    "title": properties.get("title"),
                    "description": properties.get("description"),
                    "category": properties.get("category"),
                    "brand": properties.get("brand"),
                    "price": properties.get("price"),
                    "currency": properties.get("currency"),
                    "tags": properties.get("tags", []),
                }
                
                if properties.get("metadata"):
                    metadata.update(properties["metadata"])
                
                return (vector, metadata)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get vector {vector_id} from Weaviate: {str(e)}")
            raise VectorStoreError(f"Vector retrieval failed: {str(e)}", operation="get")
    
    async def _create_schema(self):
        """Create Weaviate schema if it doesn't exist."""
        try:
            if not self.client.schema.exists(self.class_name):
                await self.create_index(self.config.dimension, self.config.metric)
            else:
                logger.debug(f"Weaviate class {self.class_name} already exists")
                
        except Exception as e:
            logger.warning(f"Failed to create Weaviate schema: {str(e)}")
    
    def _filter_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Filter metadata to match Weaviate schema."""
        filtered = {}
        
        # Map to Weaviate properties
        property_mapping = {
            "id": "productId",
            "title": "title",
            "description": "description",
            "category": "category",
            "brand": "brand",
            "price": "price",
            "currency": "currency",
            "tags": "tags",
        }
        
        for key, weaviate_key in property_mapping.items():
            if key in metadata and metadata[key] is not None:
                value = metadata[key]
                
                # Handle different data types
                if weaviate_key == "price" and isinstance(value, (int, float)):
                    filtered[weaviate_key] = float(value)
                elif weaviate_key == "tags" and isinstance(value, list):
                    filtered[weaviate_key] = [str(tag) for tag in value]
                elif isinstance(value, str):
                    filtered[weaviate_key] = value
        
        # Add remaining metadata as nested object
        remaining_metadata = {}
        for key, value in metadata.items():
            if key not in property_mapping and isinstance(value, (str, int, float, bool)):
                remaining_metadata[key] = value
        
        if remaining_metadata:
            filtered["metadata"] = remaining_metadata
        
        return filtered
    
    def _build_where_filter(self, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build Weaviate where filter from filters dict."""
        conditions = []
        
        for key, value in filters.items():
            # Map filter keys to Weaviate properties
            weaviate_key = {
                "category": "category",
                "brand": "brand",
                "price_min": "price",
                "price_max": "price",
                "tags": "tags"
            }.get(key)
            
            if not weaviate_key:
                continue
            
            if key == "price_min":
                conditions.append({
                    "path": [weaviate_key],
                    "operator": "GreaterThanEqual",
                    "valueNumber": float(value)
                })
            elif key == "price_max":
                conditions.append({
                    "path": [weaviate_key],
                    "operator": "LessThanEqual", 
                    "valueNumber": float(value)
                })
            elif key == "tags":
                if isinstance(value, list):
                    for tag in value:
                        conditions.append({
                            "path": [weaviate_key],
                            "operator": "ContainsAny",
                            "valueText": [str(tag)]
                        })
                else:
                    conditions.append({
                        "path": [weaviate_key],
                        "operator": "ContainsAny",
                        "valueText": [str(value)]
                    })
            else:
                if isinstance(value, list):
                    conditions.append({
                        "path": [weaviate_key],
                        "operator": "ContainsAny",
                        "valueText": [str(v) for v in value]
                    })
                else:
                    conditions.append({
                        "path": [weaviate_key],
                        "operator": "Equal",
                        "valueText": str(value)
                    })
        
        if not conditions:
            return None
        
        if len(conditions) == 1:
            return conditions[0]
        
        # Combine multiple conditions with AND
        return {
            "operator": "And",
            "operands": conditions
        }
    
    def _ensure_connected(self):
        """Ensure the backend is connected."""
        if not self._connected:
            raise VectorStoreError("Not connected to Weaviate. Call connect() first.", operation="check_connection")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get Weaviate statistics.
        
        Returns:
            Dictionary with database statistics
        """
        self._ensure_connected()
        
        try:
            stats = {}
            
            # Get object count
            result = self.client.query.aggregate(self.class_name).with_meta_count().do()
            
            if "data" in result and "Aggregate" in result["data"]:
                aggregate_data = result["data"]["Aggregate"]
                if self.class_name in aggregate_data:
                    meta = aggregate_data[self.class_name][0].get("meta", {})
                    stats["object_count"] = meta.get("count", 0)
            
            # Get cluster status
            cluster_status = self.client.cluster.get_nodes_status()
            stats["cluster_status"] = cluster_status
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get Weaviate statistics: {str(e)}")
            return {}