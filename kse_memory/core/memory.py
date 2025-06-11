"""
Main KSE Memory class - the primary interface for the SDK.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

from .config import KSEConfig
from .models import Product, SearchQuery, SearchResult, ConceptualDimensions, EmbeddingVector
from .interfaces import (
    AdapterInterface,
    VectorStoreInterface,
    GraphStoreInterface,
    ConceptStoreInterface,
    EmbeddingServiceInterface,
    ConceptualServiceInterface,
    SearchServiceInterface,
    CacheInterface,
)
from ..exceptions import KSEError, ConfigurationError
from ..adapters import get_adapter
from ..backends import get_vector_store, get_graph_store, get_concept_store
from ..services import (
    EmbeddingService,
    ConceptualService,
    SearchService,
    CacheService,
)


logger = logging.getLogger(__name__)


class KSEMemory:
    """
    Main KSE Memory class providing hybrid AI memory capabilities.
    
    This class combines Knowledge Graphs, Conceptual Spaces, and Neural Embeddings
    to create an intelligent memory system for product understanding and discovery.
    """
    
    def __init__(self, config: Optional[KSEConfig] = None):
        """
        Initialize KSE Memory system.
        
        Args:
            config: Configuration object. If None, uses default configuration.
        """
        self.config = config or KSEConfig()
        self._validate_config()
        
        # Initialize components
        self.adapter: Optional[AdapterInterface] = None
        self.vector_store: Optional[VectorStoreInterface] = None
        self.graph_store: Optional[GraphStoreInterface] = None
        self.concept_store: Optional[ConceptStoreInterface] = None
        self.embedding_service: Optional[EmbeddingServiceInterface] = None
        self.conceptual_service: Optional[ConceptualServiceInterface] = None
        self.search_service: Optional[SearchServiceInterface] = None
        self.cache_service: Optional[CacheInterface] = None
        
        # State tracking
        self._initialized = False
        self._connected = False
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, self.config.log_level.value))
        logger.info(f"KSE Memory initialized with config: {self.config.app_name} v{self.config.version}")
    
    def _validate_config(self):
        """Validate configuration and raise errors if invalid."""
        errors = self.config.validate()
        if errors:
            raise ConfigurationError(f"Configuration validation failed: {', '.join(errors)}")
    
    async def initialize(self, adapter_type: str, adapter_config: Dict[str, Any]) -> bool:
        """
        Initialize the KSE Memory system with a specific adapter.
        
        Args:
            adapter_type: Type of adapter (e.g., 'shopify', 'woocommerce', 'generic')
            adapter_config: Configuration for the adapter
            
        Returns:
            True if initialization successful
            
        Raises:
            KSEError: If initialization fails
        """
        try:
            logger.info(f"Initializing KSE Memory with {adapter_type} adapter")
            
            # Initialize adapter
            self.adapter = get_adapter(adapter_type)
            await self.adapter.connect(adapter_config)
            
            # Initialize storage backends
            self.vector_store = get_vector_store(self.config.vector_store)
            await self.vector_store.connect()
            
            self.graph_store = get_graph_store(self.config.graph_store)
            await self.graph_store.connect()
            
            self.concept_store = get_concept_store(self.config.concept_store)
            await self.concept_store.connect()
            
            # Initialize services
            self.embedding_service = EmbeddingService(self.config.embedding)
            self.conceptual_service = ConceptualService(self.config.conceptual)
            self.cache_service = CacheService(self.config.cache)
            
            # Initialize search service with all components
            self.search_service = SearchService(
                config=self.config.search,
                vector_store=self.vector_store,
                graph_store=self.graph_store,
                concept_store=self.concept_store,
                embedding_service=self.embedding_service,
                cache_service=self.cache_service,
            )
            
            self._initialized = True
            self._connected = True
            
            logger.info("KSE Memory initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize KSE Memory: {str(e)}")
            raise KSEError(f"Initialization failed: {str(e)}") from e
    
    async def disconnect(self) -> bool:
        """
        Disconnect from all services and clean up resources.
        
        Returns:
            True if disconnection successful
        """
        try:
            logger.info("Disconnecting KSE Memory")
            
            # Disconnect from all services
            if self.adapter:
                await self.adapter.disconnect()
            
            if self.vector_store:
                await self.vector_store.disconnect()
            
            if self.graph_store:
                await self.graph_store.disconnect()
            
            if self.concept_store:
                await self.concept_store.disconnect()
            
            self._connected = False
            logger.info("KSE Memory disconnected successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during disconnection: {str(e)}")
            return False
    
    async def add_product(self, product: Product, compute_embeddings: bool = True, compute_concepts: bool = True) -> bool:
        """
        Add a product to the KSE Memory system.
        
        Args:
            product: Product to add
            compute_embeddings: Whether to compute embeddings
            compute_concepts: Whether to compute conceptual dimensions
            
        Returns:
            True if product added successfully
            
        Raises:
            KSEError: If adding product fails
        """
        self._ensure_initialized()
        
        try:
            logger.debug(f"Adding product: {product.id}")
            
            # Compute embeddings if requested
            if compute_embeddings and self.embedding_service:
                # Generate text embedding
                text_content = f"{product.title} {product.description} {' '.join(product.tags)}"
                product.text_embedding = await self.embedding_service.generate_text_embedding(text_content)
                
                # Generate image embedding if images available
                if product.images and len(product.images) > 0:
                    product.image_embedding = await self.embedding_service.generate_image_embedding(product.images[0])
            
            # Compute conceptual dimensions if requested
            if compute_concepts and self.conceptual_service:
                product.conceptual_dimensions = await self.conceptual_service.compute_dimensions(product)
            
            # Store in vector store
            if product.text_embedding and self.vector_store:
                await self.vector_store.upsert_vectors([
                    (product.id, product.text_embedding.vector, product.to_dict())
                ])
            
            # Store in concept store
            if product.conceptual_dimensions and self.concept_store:
                await self.concept_store.store_conceptual_dimensions(product.id, product.conceptual_dimensions)
            
            # Create knowledge graph node
            if self.graph_store:
                await self.graph_store.create_node(
                    product.id,
                    ["Product"],
                    {
                        "title": product.title,
                        "category": product.category,
                        "brand": product.brand,
                        "price": product.price,
                        "tags": product.tags,
                    }
                )
                
                # Create relationships based on category, brand, tags
                if product.category:
                    category_id = f"category_{product.category.lower().replace(' ', '_')}"
                    await self.graph_store.create_node(category_id, ["Category"], {"name": product.category})
                    await self.graph_store.create_relationship(product.id, category_id, "BELONGS_TO")
                
                if product.brand:
                    brand_id = f"brand_{product.brand.lower().replace(' ', '_')}"
                    await self.graph_store.create_node(brand_id, ["Brand"], {"name": product.brand})
                    await self.graph_store.create_relationship(product.id, brand_id, "MADE_BY")
            
            logger.debug(f"Product {product.id} added successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add product {product.id}: {str(e)}")
            raise KSEError(f"Failed to add product: {str(e)}") from e
    
    async def get_product(self, product_id: str) -> Optional[Product]:
        """
        Retrieve a product by ID.
        
        Args:
            product_id: ID of the product to retrieve
            
        Returns:
            Product if found, None otherwise
        """
        self._ensure_initialized()
        
        try:
            # Try cache first
            if self.cache_service:
                cached_product = await self.cache_service.get(f"product:{product_id}")
                if cached_product:
                    return Product.from_dict(cached_product)
            
            # Get from vector store (which contains full product data)
            if self.vector_store:
                vector_data = await self.vector_store.get_vector(product_id)
                if vector_data:
                    _, metadata = vector_data
                    product = Product.from_dict(metadata)
                    
                    # Cache the result
                    if self.cache_service:
                        await self.cache_service.set(f"product:{product_id}", product.to_dict())
                    
                    return product
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get product {product_id}: {str(e)}")
            return None
    
    async def search(self, query: Union[str, SearchQuery]) -> List[SearchResult]:
        """
        Search for products using the KSE Memory system.
        
        Args:
            query: Search query (string or SearchQuery object)
            
        Returns:
            List of search results
            
        Raises:
            KSEError: If search fails
        """
        self._ensure_initialized()
        
        if not self.search_service:
            raise KSEError("Search service not initialized")
        
        try:
            # Convert string query to SearchQuery object
            if isinstance(query, str):
                search_query = SearchQuery(query=query)
            else:
                search_query = query
            
            logger.debug(f"Performing search: {search_query.query}")
            
            # Perform search
            results = await self.search_service.search(search_query)
            
            logger.debug(f"Search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise KSEError(f"Search failed: {str(e)}") from e
    
    async def sync_products(self) -> int:
        """
        Sync products from the connected platform.
        
        Returns:
            Number of products synced
            
        Raises:
            KSEError: If sync fails
        """
        self._ensure_initialized()
        
        if not self.adapter:
            raise KSEError("No adapter connected")
        
        try:
            logger.info("Starting product sync")
            
            # Get products from adapter
            products = await self.adapter.get_products()
            
            # Add each product to KSE Memory
            synced_count = 0
            for product in products:
                try:
                    await self.add_product(product)
                    synced_count += 1
                except Exception as e:
                    logger.warning(f"Failed to sync product {product.id}: {str(e)}")
            
            logger.info(f"Synced {synced_count} products")
            return synced_count
            
        except Exception as e:
            logger.error(f"Product sync failed: {str(e)}")
            raise KSEError(f"Product sync failed: {str(e)}") from e
    
    async def get_recommendations(self, product_id: str, limit: int = 10) -> List[SearchResult]:
        """
        Get product recommendations based on a given product.
        
        Args:
            product_id: ID of the product to base recommendations on
            limit: Maximum number of recommendations
            
        Returns:
            List of recommended products
        """
        self._ensure_initialized()
        
        try:
            # Get the source product
            product = await self.get_product(product_id)
            if not product:
                return []
            
            # Use conceptual dimensions for recommendations if available
            if product.conceptual_dimensions and self.concept_store:
                similar_products = await self.concept_store.find_similar_concepts(
                    product.conceptual_dimensions,
                    threshold=0.7,
                    limit=limit + 1  # +1 to exclude the source product
                )
                
                # Convert to SearchResult objects
                results = []
                for similar_id, similarity in similar_products:
                    if similar_id != product_id:  # Exclude the source product
                        similar_product = await self.get_product(similar_id)
                        if similar_product:
                            results.append(SearchResult(
                                product=similar_product,
                                score=similarity,
                                conceptual_similarity=similarity,
                                explanation=f"Similar conceptual profile to {product.title}"
                            ))
                
                return results[:limit]
            
            # Fallback to embedding-based recommendations
            elif product.text_embedding and self.vector_store:
                similar_vectors = await self.vector_store.search_vectors(
                    product.text_embedding.vector,
                    top_k=limit + 1
                )
                
                results = []
                for vector_id, similarity, metadata in similar_vectors:
                    if vector_id != product_id:  # Exclude the source product
                        similar_product = Product.from_dict(metadata)
                        results.append(SearchResult(
                            product=similar_product,
                            score=similarity,
                            embedding_similarity=similarity,
                            explanation=f"Similar semantic profile to {product.title}"
                        ))
                
                return results[:limit]
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get recommendations for {product_id}: {str(e)}")
            return []
    
    def _ensure_initialized(self):
        """Ensure the system is initialized."""
        if not self._initialized:
            raise KSEError("KSE Memory not initialized. Call initialize() first.")
        
        if not self._connected:
            raise KSEError("KSE Memory not connected. Call initialize() first.")
    
    @property
    def is_initialized(self) -> bool:
        """Check if the system is initialized."""
        return self._initialized
    
    @property
    def is_connected(self) -> bool:
        """Check if the system is connected."""
        return self._connected
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on all components.
        
        Returns:
            Dictionary with health status of each component
        """
        health = {
            "initialized": self._initialized,
            "connected": self._connected,
            "timestamp": datetime.utcnow().isoformat(),
            "components": {}
        }
        
        if self._initialized:
            # Check each component
            try:
                if self.vector_store:
                    # Try a simple operation
                    health["components"]["vector_store"] = "healthy"
            except Exception:
                health["components"]["vector_store"] = "unhealthy"
            
            try:
                if self.graph_store:
                    health["components"]["graph_store"] = "healthy"
            except Exception:
                health["components"]["graph_store"] = "unhealthy"
            
            try:
                if self.concept_store:
                    health["components"]["concept_store"] = "healthy"
            except Exception:
                health["components"]["concept_store"] = "unhealthy"
            
            try:
                if self.adapter:
                    health["components"]["adapter"] = "healthy"
            except Exception:
                health["components"]["adapter"] = "unhealthy"
        
        return health