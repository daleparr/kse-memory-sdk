"""
Milvus vector store backend for KSE Memory SDK.
"""

from typing import List, Dict, Any, Optional
import asyncio
import numpy as np
from ..core.interfaces import VectorStoreInterface
from ..core.models import Product
from ..core.config import VectorStoreConfig
from ..exceptions import BackendError, VectorStoreError

try:
    from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False


class MilvusBackend(VectorStoreInterface):
    """Milvus vector store implementation."""
    
    def __init__(self, config: VectorStoreConfig):
        """Initialize Milvus backend."""
        if not MILVUS_AVAILABLE:
            raise BackendError("pymilvus package is required for Milvus backend", "milvus")
        
        self.config = config
        self.collection = None
        self._connected = False
        self.collection_name = "kse_products"
        self.dimension = config.dimension or 1536  # Default OpenAI embedding dimension
    
    async def connect(self) -> bool:
        """Connect to Milvus instance."""
        try:
            # Parse connection parameters from URI
            if self.config.uri.startswith("milvus://"):
                # Remove protocol
                uri_parts = self.config.uri[9:].split(":")
                host = uri_parts[0]
                port = int(uri_parts[1]) if len(uri_parts) > 1 else 19530
            else:
                host = "localhost"
                port = 19530
            
            # Connect to Milvus
            await asyncio.to_thread(
                connections.connect,
                alias="default",
                host=host,
                port=port
            )
            
            # Create collection if it doesn't exist
            if not await asyncio.to_thread(utility.has_collection, self.collection_name):
                await self._create_collection()
            
            # Get collection
            self.collection = Collection(self.collection_name)
            
            # Load collection into memory
            await asyncio.to_thread(self.collection.load)
            
            self._connected = True
            return True
            
        except Exception as e:
            raise VectorStoreError(f"Failed to connect to Milvus: {str(e)}", "connect")
    
    async def disconnect(self) -> bool:
        """Disconnect from Milvus."""
        try:
            if self.collection:
                await asyncio.to_thread(self.collection.release)
            await asyncio.to_thread(connections.disconnect, "default")
            self._connected = False
            return True
        except Exception:
            return False
    
    async def _create_collection(self):
        """Create the products collection with schema."""
        try:
            # Define schema
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
                FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=2000),
                FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="brand", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="price", dtype=DataType.FLOAT),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension)
            ]
            
            schema = CollectionSchema(
                fields=fields,
                description="KSE product embeddings collection"
            )
            
            # Create collection
            collection = Collection(
                name=self.collection_name,
                schema=schema
            )
            
            # Create index for vector field
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            
            await asyncio.to_thread(
                collection.create_index,
                field_name="embedding",
                index_params=index_params
            )
            
        except Exception as e:
            raise VectorStoreError(f"Failed to create Milvus collection: {str(e)}", "create_collection")
    
    async def upsert_product(self, product: Product, embedding: List[float]) -> bool:
        """Upsert product with embedding."""
        if not self._connected:
            raise VectorStoreError("Not connected to Milvus", "upsert_product")
        
        try:
            # Prepare data
            data = [
                [product.id],  # id
                [product.title],  # title
                [product.description or ""],  # description
                [product.category or ""],  # category
                [product.brand or ""],  # brand
                [float(product.price) if product.price else 0.0],  # price
                [embedding]  # embedding
            ]
            
            # Delete existing record if it exists
            await self._delete_by_id(product.id)
            
            # Insert new record
            await asyncio.to_thread(self.collection.insert, data)
            
            # Flush to ensure data is written
            await asyncio.to_thread(self.collection.flush)
            
            return True
            
        except Exception as e:
            raise VectorStoreError(f"Failed to upsert product: {str(e)}", "upsert_product")
    
    async def _delete_by_id(self, product_id: str):
        """Delete product by ID."""
        try:
            expr = f'id == "{product_id}"'
            await asyncio.to_thread(self.collection.delete, expr)
        except Exception:
            pass  # Product might not exist
    
    async def search_similar(self, query_embedding: List[float], limit: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar products."""
        if not self._connected:
            raise VectorStoreError("Not connected to Milvus", "search_similar")
        
        try:
            # Prepare search parameters
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            # Build filter expression
            filter_expr = None
            if filters:
                filter_conditions = []
                for key, value in filters.items():
                    if isinstance(value, str):
                        filter_conditions.append(f'{key} == "{value}"')
                    elif isinstance(value, (int, float)):
                        filter_conditions.append(f'{key} == {value}')
                    elif isinstance(value, list) and len(value) == 2:
                        # Range filter [min, max]
                        filter_conditions.append(f'{key} >= {value[0]} and {key} <= {value[1]}')
                
                if filter_conditions:
                    filter_expr = " and ".join(filter_conditions)
            
            # Perform search
            results = await asyncio.to_thread(
                self.collection.search,
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                expr=filter_expr,
                output_fields=["id", "title", "description", "category", "brand", "price"]
            )
            
            # Format results
            formatted_results = []
            for hits in results:
                for hit in hits:
                    formatted_results.append({
                        "id": hit.entity.get("id"),
                        "title": hit.entity.get("title"),
                        "description": hit.entity.get("description"),
                        "category": hit.entity.get("category"),
                        "brand": hit.entity.get("brand"),
                        "price": hit.entity.get("price"),
                        "score": float(hit.score),
                        "distance": float(hit.distance)
                    })
            
            return formatted_results
            
        except Exception as e:
            raise VectorStoreError(f"Failed to search similar products: {str(e)}", "search_similar")
    
    async def get_product_by_id(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Get product by ID."""
        if not self._connected:
            raise VectorStoreError("Not connected to Milvus", "get_product_by_id")
        
        try:
            expr = f'id == "{product_id}"'
            results = await asyncio.to_thread(
                self.collection.query,
                expr=expr,
                output_fields=["id", "title", "description", "category", "brand", "price", "embedding"]
            )
            
            if results:
                result = results[0]
                return {
                    "id": result.get("id"),
                    "title": result.get("title"),
                    "description": result.get("description"),
                    "category": result.get("category"),
                    "brand": result.get("brand"),
                    "price": result.get("price"),
                    "embedding": result.get("embedding")
                }
            
            return None
            
        except Exception as e:
            raise VectorStoreError(f"Failed to get product by ID: {str(e)}", "get_product_by_id")
    
    async def delete_product(self, product_id: str) -> bool:
        """Delete product by ID."""
        if not self._connected:
            raise VectorStoreError("Not connected to Milvus", "delete_product")
        
        try:
            expr = f'id == "{product_id}"'
            await asyncio.to_thread(self.collection.delete, expr)
            await asyncio.to_thread(self.collection.flush)
            return True
            
        except Exception as e:
            raise VectorStoreError(f"Failed to delete product: {str(e)}", "delete_product")
    
    async def list_products(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """List products with pagination."""
        if not self._connected:
            raise VectorStoreError("Not connected to Milvus", "list_products")
        
        try:
            # Milvus doesn't support offset directly, so we'll use a workaround
            # Get all results and slice them (not ideal for large datasets)
            results = await asyncio.to_thread(
                self.collection.query,
                expr="",  # Empty expression to get all
                output_fields=["id", "title", "description", "category", "brand", "price"],
                limit=limit + offset
            )
            
            # Apply offset manually
            sliced_results = results[offset:offset + limit] if len(results) > offset else []
            
            formatted_results = []
            for result in sliced_results:
                formatted_results.append({
                    "id": result.get("id"),
                    "title": result.get("title"),
                    "description": result.get("description"),
                    "category": result.get("category"),
                    "brand": result.get("brand"),
                    "price": result.get("price")
                })
            
            return formatted_results
            
        except Exception as e:
            raise VectorStoreError(f"Failed to list products: {str(e)}", "list_products")
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        if not self._connected:
            raise VectorStoreError("Not connected to Milvus", "get_collection_stats")
        
        try:
            # Get collection info
            await asyncio.to_thread(self.collection.load)
            
            stats = await asyncio.to_thread(self.collection.get_stats)
            
            return {
                "total_count": stats.get("row_count", 0),
                "collection_name": self.collection_name,
                "dimension": self.dimension,
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE"
            }
            
        except Exception as e:
            raise VectorStoreError(f"Failed to get collection stats: {str(e)}", "get_collection_stats")
    
    async def create_index(self, index_params: Optional[Dict[str, Any]] = None) -> bool:
        """Create or update index on the embedding field."""
        if not self._connected:
            raise VectorStoreError("Not connected to Milvus", "create_index")
        
        try:
            if index_params is None:
                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 1024}
                }
            
            # Drop existing index if it exists
            try:
                await asyncio.to_thread(self.collection.drop_index)
            except Exception:
                pass  # Index might not exist
            
            # Create new index
            await asyncio.to_thread(
                self.collection.create_index,
                field_name="embedding",
                index_params=index_params
            )
            
            # Load collection with new index
            await asyncio.to_thread(self.collection.load)
            
            return True
            
        except Exception as e:
            raise VectorStoreError(f"Failed to create index: {str(e)}", "create_index")