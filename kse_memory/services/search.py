"""
Search service for KSE Memory SDK.
"""

import asyncio
import logging
import math
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from ..core.interfaces import (
    SearchServiceInterface,
    VectorStoreInterface,
    GraphStoreInterface,
    ConceptStoreInterface,
    EmbeddingServiceInterface,
    CacheInterface,
)
from ..core.config import SearchConfig
from ..core.models import SearchQuery, SearchResult, Product, SearchType, ConceptualDimensions
from ..exceptions import SearchError
from .conceptual import ConceptualService


logger = logging.getLogger(__name__)


class SearchService(SearchServiceInterface):
    """
    Hybrid search service for KSE Memory SDK.
    
    Combines semantic search (embeddings), conceptual search (conceptual spaces),
    and knowledge graph search to provide comprehensive product discovery.
    """
    
    def __init__(
        self,
        config: SearchConfig,
        vector_store: VectorStoreInterface,
        graph_store: GraphStoreInterface,
        concept_store: ConceptStoreInterface,
        embedding_service: EmbeddingServiceInterface,
        cache_service: Optional[CacheInterface] = None,
    ):
        """
        Initialize search service.
        
        Args:
            config: Search configuration
            vector_store: Vector store backend
            graph_store: Graph store backend
            concept_store: Concept store backend
            embedding_service: Embedding service
            cache_service: Optional cache service
        """
        self.config = config
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.concept_store = concept_store
        self.embedding_service = embedding_service
        self.cache_service = cache_service
        
        # Initialize conceptual service for similarity computation
        from ..core.config import ConceptualConfig
        conceptual_config = ConceptualConfig(auto_compute=False)  # No LLM needed for similarity
        self.conceptual_service = ConceptualService(conceptual_config)
        
        logger.info("Search service initialized")
    
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """
        Perform a search query.
        
        Args:
            query: Search query object
            
        Returns:
            List of search results
            
        Raises:
            SearchError: If search fails
        """
        try:
            # Check cache first
            cache_key = self._generate_cache_key(query)
            if self.cache_service:
                cached_results = await self.cache_service.get(cache_key)
                if cached_results:
                    logger.debug(f"Returning cached search results for query: {query.query}")
                    return [SearchResult.from_dict(result) for result in cached_results]
            
            # Route to appropriate search method
            if query.search_type == SearchType.SEMANTIC:
                results = await self.semantic_search(query.query, query.limit)
            elif query.search_type == SearchType.CONCEPTUAL:
                # Extract conceptual dimensions from query
                dimensions = self._extract_conceptual_dimensions(query)
                results = await self.conceptual_search(dimensions, query.limit)
            elif query.search_type == SearchType.KNOWLEDGE_GRAPH:
                results = await self.knowledge_graph_search(query.query, query.limit)
            elif query.search_type == SearchType.HYBRID:
                results = await self.hybrid_search(query)
            else:
                raise SearchError(f"Unsupported search type: {query.search_type}", query=query.query)
            
            # Apply filters
            if query.filters:
                results = self._apply_filters(results, query.filters)
            
            # Apply pagination
            start_idx = query.offset
            end_idx = start_idx + query.limit
            results = results[start_idx:end_idx]
            
            # Cache results
            if self.cache_service and results:
                await self.cache_service.set(
                    cache_key,
                    [result.to_dict() for result in results],
                    ttl=300  # 5 minutes
                )
            
            logger.debug(f"Search completed: {len(results)} results for query '{query.query}'")
            return results
            
        except Exception as e:
            logger.error(f"Search failed for query '{query.query}': {str(e)}")
            raise SearchError(f"Search failed: {str(e)}", query=query.query)
    
    async def semantic_search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """
        Perform semantic search using embeddings.
        
        Args:
            query: Search query text
            limit: Maximum number of results
            
        Returns:
            List of search results
        """
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.generate_text_embedding(query)
            
            # Search vector store
            vector_results = await self.vector_store.search_vectors(
                query_embedding.vector,
                top_k=limit * 2,  # Get more results for reranking
                filters=None
            )
            
            # Convert to SearchResult objects
            results = []
            for vector_id, similarity, metadata in vector_results:
                try:
                    product = Product.from_dict(metadata)
                    result = SearchResult(
                        product=product,
                        score=similarity,
                        embedding_similarity=similarity,
                        explanation=f"Semantic similarity to '{query}'"
                    )
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to convert vector result {vector_id}: {str(e)}")
            
            # Apply similarity threshold
            results = [r for r in results if r.score >= self.config.similarity_threshold]
            
            # Sort by score and limit
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Semantic search failed: {str(e)}")
            raise SearchError(f"Semantic search failed: {str(e)}", query=query)
    
    async def conceptual_search(self, dimensions: ConceptualDimensions, limit: int = 10) -> List[SearchResult]:
        """
        Perform conceptual search using conceptual dimensions.
        
        Args:
            dimensions: Target conceptual dimensions
            limit: Maximum number of results
            
        Returns:
            List of search results
        """
        try:
            # Find similar concepts
            similar_concepts = await self.concept_store.find_similar_concepts(
                dimensions,
                threshold=self.config.similarity_threshold,
                limit=limit * 2
            )
            
            # Convert to SearchResult objects
            results = []
            for product_id, similarity in similar_concepts:
                try:
                    # Get product from vector store (which has full metadata)
                    vector_data = await self.vector_store.get_vector(product_id)
                    if vector_data:
                        _, metadata = vector_data
                        product = Product.from_dict(metadata)
                        
                        result = SearchResult(
                            product=product,
                            score=similarity,
                            conceptual_similarity=similarity,
                            explanation=f"Conceptual similarity match"
                        )
                        results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to convert conceptual result {product_id}: {str(e)}")
            
            # Sort by score and limit
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Conceptual search failed: {str(e)}")
            raise SearchError(f"Conceptual search failed: {str(e)}")
    
    async def knowledge_graph_search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """
        Perform search using knowledge graph relationships.
        
        Args:
            query: Search query text
            limit: Maximum number of results
            
        Returns:
            List of search results
        """
        try:
            # Extract entities from query (simplified approach)
            entities = self._extract_entities(query)
            
            results = []
            
            # Search for products related to extracted entities
            for entity in entities:
                try:
                    # Find nodes matching the entity
                    cypher_query = """
                    MATCH (p:Product)-[r]-(n)
                    WHERE toLower(n.name) CONTAINS toLower($entity)
                       OR toLower(p.title) CONTAINS toLower($entity)
                       OR toLower(p.category) CONTAINS toLower($entity)
                       OR toLower(p.brand) CONTAINS toLower($entity)
                    RETURN p.id as product_id, type(r) as relationship, n.name as related_entity
                    LIMIT $limit
                    """
                    
                    graph_results = await self.graph_store.execute_query(
                        cypher_query,
                        {"entity": entity, "limit": limit}
                    )
                    
                    # Convert graph results to SearchResult objects
                    for result in graph_results:
                        product_id = result.get("product_id")
                        relationship = result.get("relationship")
                        related_entity = result.get("related_entity")
                        
                        if product_id:
                            # Get full product data
                            vector_data = await self.vector_store.get_vector(product_id)
                            if vector_data:
                                _, metadata = vector_data
                                product = Product.from_dict(metadata)
                                
                                # Calculate relevance score based on relationship type
                                score = self._calculate_graph_relevance(relationship, entity, related_entity)
                                
                                search_result = SearchResult(
                                    product=product,
                                    score=score,
                                    knowledge_graph_similarity=score,
                                    explanation=f"Connected to '{related_entity}' via {relationship}"
                                )
                                results.append(search_result)
                                
                except Exception as e:
                    logger.warning(f"Failed to process entity '{entity}': {str(e)}")
            
            # Remove duplicates and sort
            unique_results = {}
            for result in results:
                if result.product.id not in unique_results or result.score > unique_results[result.product.id].score:
                    unique_results[result.product.id] = result
            
            final_results = list(unique_results.values())
            final_results.sort(key=lambda x: x.score, reverse=True)
            
            return final_results[:limit]
            
        except Exception as e:
            logger.error(f"Knowledge graph search failed: {str(e)}")
            raise SearchError(f"Knowledge graph search failed: {str(e)}", query=query)
    
    async def hybrid_search(self, query: SearchQuery) -> List[SearchResult]:
        """
        Perform hybrid search combining multiple approaches.
        
        Args:
            query: Search query object
            
        Returns:
            List of search results
        """
        try:
            # Get results from each search method
            search_tasks = []
            
            if query.include_embeddings:
                search_tasks.append(("semantic", self.semantic_search(query.query, query.limit * 2)))
            
            if query.include_conceptual:
                dimensions = self._extract_conceptual_dimensions(query)
                search_tasks.append(("conceptual", self.conceptual_search(dimensions, query.limit * 2)))
            
            if query.include_knowledge_graph:
                search_tasks.append(("graph", self.knowledge_graph_search(query.query, query.limit * 2)))
            
            # Execute searches concurrently
            search_results = await asyncio.gather(*[task[1] for task in search_tasks], return_exceptions=True)
            
            # Combine results
            all_results = {}
            weights = self.config.hybrid_weights
            
            for i, (search_type, results) in enumerate(zip([task[0] for task in search_tasks], search_results)):
                if isinstance(results, Exception):
                    logger.warning(f"{search_type} search failed: {str(results)}")
                    continue
                
                weight = weights.get(search_type, weights.get("embedding", 0.33))
                
                for result in results:
                    product_id = result.product.id
                    
                    if product_id in all_results:
                        # Combine scores
                        existing = all_results[product_id]
                        combined_score = existing.score + (result.score * weight)
                        
                        # Update individual similarity scores
                        if search_type == "semantic":
                            existing.embedding_similarity = result.score
                        elif search_type == "conceptual":
                            existing.conceptual_similarity = result.score
                        elif search_type == "graph":
                            existing.knowledge_graph_similarity = result.score
                        
                        existing.score = combined_score
                        existing.explanation = self._combine_explanations(existing.explanation, result.explanation)
                        
                    else:
                        # New result
                        weighted_score = result.score * weight
                        result.score = weighted_score
                        all_results[product_id] = result
            
            # Convert to list and sort
            final_results = list(all_results.values())
            final_results.sort(key=lambda x: x.score, reverse=True)
            
            # Apply reranking if enabled
            if self.config.enable_reranking and len(final_results) > 1:
                final_results = await self._rerank_results(query, final_results)
            
            return final_results[:query.limit]
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {str(e)}")
            raise SearchError(f"Hybrid search failed: {str(e)}", query=query.query)
    
    def _extract_conceptual_dimensions(self, query: SearchQuery) -> ConceptualDimensions:
        """Extract conceptual dimensions from search query."""
        # Use provided weights or extract from query text
        if query.conceptual_weights:
            dimensions_dict = {}
            for dim in ["elegance", "comfort", "boldness", "modernity", "minimalism", 
                       "luxury", "functionality", "versatility", "seasonality", "innovation"]:
                dimensions_dict[dim] = query.conceptual_weights.get(dim, 0.5)
            return ConceptualDimensions(**dimensions_dict)
        else:
            # Extract from query text using keyword matching
            weights = self.conceptual_service.get_dimension_weights(query.query)
            
            # Normalize weights to 0-1 range
            max_weight = max(weights.values()) if weights else 1.0
            normalized_weights = {dim: weight / max_weight for dim, weight in weights.items()}
            
            return ConceptualDimensions(**normalized_weights)
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract entities from query text (simplified approach)."""
        # This is a simplified entity extraction
        # In a production system, you might use NLP libraries like spaCy
        
        words = query.lower().split()
        
        # Filter out common stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        entities = [word for word in words if word not in stop_words and len(word) > 2]
        
        return entities[:5]  # Limit to 5 entities
    
    def _calculate_graph_relevance(self, relationship: str, query_entity: str, related_entity: str) -> float:
        """Calculate relevance score for graph search results."""
        # Base score
        score = 0.5
        
        # Boost score based on relationship type
        relationship_weights = {
            "BELONGS_TO": 0.8,
            "MADE_BY": 0.7,
            "SIMILAR_TO": 0.9,
            "RELATED_TO": 0.6,
            "TAGGED_WITH": 0.5,
        }
        
        score *= relationship_weights.get(relationship, 0.5)
        
        # Boost if exact match
        if query_entity.lower() == related_entity.lower():
            score *= 1.5
        
        # Ensure score is in valid range
        return min(1.0, score)
    
    def _apply_filters(self, results: List[SearchResult], filters: Dict[str, Any]) -> List[SearchResult]:
        """Apply filters to search results."""
        filtered_results = []
        
        for result in results:
            product = result.product
            include = True
            
            # Apply each filter
            for filter_key, filter_value in filters.items():
                if filter_key == "category" and product.category:
                    if isinstance(filter_value, list):
                        include = product.category.lower() in [v.lower() for v in filter_value]
                    else:
                        include = product.category.lower() == filter_value.lower()
                
                elif filter_key == "brand" and product.brand:
                    if isinstance(filter_value, list):
                        include = product.brand.lower() in [v.lower() for v in filter_value]
                    else:
                        include = product.brand.lower() == filter_value.lower()
                
                elif filter_key == "price_min" and product.price:
                    include = product.price >= filter_value
                
                elif filter_key == "price_max" and product.price:
                    include = product.price <= filter_value
                
                elif filter_key == "tags" and product.tags:
                    if isinstance(filter_value, list):
                        include = any(tag.lower() in [t.lower() for t in product.tags] for tag in filter_value)
                    else:
                        include = filter_value.lower() in [t.lower() for t in product.tags]
                
                if not include:
                    break
            
            if include:
                filtered_results.append(result)
        
        return filtered_results
    
    async def _rerank_results(self, query: SearchQuery, results: List[SearchResult]) -> List[SearchResult]:
        """Rerank results using additional signals."""
        # This is a placeholder for more sophisticated reranking
        # In a production system, you might use learning-to-rank models
        
        try:
            # Simple reranking based on multiple factors
            for result in results:
                product = result.product
                
                # Boost recent products
                if product.created_at:
                    days_old = (datetime.utcnow() - product.created_at).days
                    recency_boost = max(0, 1 - (days_old / 365))  # Decay over a year
                    result.score *= (1 + recency_boost * 0.1)
                
                # Boost products with more complete information
                completeness = 0
                if product.description: completeness += 1
                if product.images: completeness += 1
                if product.tags: completeness += 1
                if product.category: completeness += 1
                if product.brand: completeness += 1
                
                completeness_boost = completeness / 5.0
                result.score *= (1 + completeness_boost * 0.05)
            
            # Re-sort after reranking
            results.sort(key=lambda x: x.score, reverse=True)
            return results
            
        except Exception as e:
            logger.warning(f"Reranking failed: {str(e)}")
            return results
    
    def _combine_explanations(self, explanation1: str, explanation2: str) -> str:
        """Combine multiple explanations."""
        if not explanation1:
            return explanation2
        if not explanation2:
            return explanation1
        
        return f"{explanation1}; {explanation2}"
    
    def _generate_cache_key(self, query: SearchQuery) -> str:
        """Generate cache key for search query."""
        key_parts = [
            f"search:{query.search_type.value}",
            f"q:{hash(query.query)}",
            f"limit:{query.limit}",
            f"offset:{query.offset}",
        ]
        
        if query.filters:
            filters_str = str(sorted(query.filters.items()))
            key_parts.append(f"filters:{hash(filters_str)}")
        
        if query.conceptual_weights:
            weights_str = str(sorted(query.conceptual_weights.items()))
            key_parts.append(f"weights:{hash(weights_str)}")
        
        return ":".join(key_parts)