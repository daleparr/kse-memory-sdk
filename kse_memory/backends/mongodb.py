"""
MongoDB concept store backend for KSE Memory SDK.
"""

from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime
from ..core.interfaces import ConceptStoreInterface
from ..core.models import Product
from ..core.config import ConceptStoreConfig
from ..exceptions import BackendError, ConceptStoreError

try:
    from motor.motor_asyncio import AsyncIOMotorClient
    from pymongo import IndexModel, ASCENDING, DESCENDING
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False


class MongoDBBackend(ConceptStoreInterface):
    """MongoDB concept store implementation."""
    
    def __init__(self, config: ConceptStoreConfig):
        """Initialize MongoDB backend."""
        if not MONGODB_AVAILABLE:
            raise BackendError("motor package is required for MongoDB backend", "mongodb")
        
        self.config = config
        self.client = None
        self.db = None
        self.products_collection = None
        self.concepts_collection = None
        self.spaces_collection = None
        self._connected = False
    
    async def connect(self) -> bool:
        """Connect to MongoDB instance."""
        try:
            # Create MongoDB client
            self.client = AsyncIOMotorClient(self.config.uri)
            
            # Get database
            self.db = self.client[self.config.database]
            
            # Test connection
            await self.client.admin.command('ping')
            
            # Get collections
            self.products_collection = self.db.products
            self.concepts_collection = self.db.concepts
            self.spaces_collection = self.db.conceptual_spaces
            
            # Create indexes
            await self._create_indexes()
            
            self._connected = True
            return True
            
        except Exception as e:
            raise ConceptStoreError(f"Failed to connect to MongoDB: {str(e)}", "connect")
    
    async def disconnect(self) -> bool:
        """Disconnect from MongoDB."""
        if self.client:
            self.client.close()
        self._connected = False
        return True
    
    async def _create_indexes(self):
        """Create necessary indexes for efficient querying."""
        try:
            # Product indexes
            product_indexes = [
                IndexModel([("product_id", ASCENDING)], unique=True),
                IndexModel([("category", ASCENDING)]),
                IndexModel([("brand", ASCENDING)]),
                IndexModel([("conceptual_dimensions.elegance", ASCENDING)]),
                IndexModel([("conceptual_dimensions.comfort", ASCENDING)]),
                IndexModel([("conceptual_dimensions.boldness", ASCENDING)]),
                IndexModel([("conceptual_dimensions.sustainability", ASCENDING)]),
                IndexModel([("conceptual_dimensions.luxury", ASCENDING)]),
                IndexModel([("conceptual_dimensions.innovation", ASCENDING)]),
                IndexModel([("conceptual_dimensions.versatility", ASCENDING)]),
                IndexModel([("conceptual_dimensions.durability", ASCENDING)]),
                IndexModel([("conceptual_dimensions.affordability", ASCENDING)]),
                IndexModel([("conceptual_dimensions.trendiness", ASCENDING)]),
                IndexModel([("updated_at", DESCENDING)])
            ]
            
            await self.products_collection.create_indexes(product_indexes)
            
            # Concept indexes
            concept_indexes = [
                IndexModel([("concept_name", ASCENDING)], unique=True),
                IndexModel([("domain", ASCENDING)]),
                IndexModel([("dimensions", ASCENDING)])
            ]
            
            await self.concepts_collection.create_indexes(concept_indexes)
            
            # Conceptual space indexes
            space_indexes = [
                IndexModel([("space_name", ASCENDING)], unique=True),
                IndexModel([("domain", ASCENDING)]),
                IndexModel([("created_at", DESCENDING)])
            ]
            
            await self.spaces_collection.create_indexes(space_indexes)
            
        except Exception as e:
            # Indexes might already exist, continue
            pass
    
    async def store_product_concepts(self, product: Product) -> bool:
        """Store product conceptual dimensions."""
        if not self._connected:
            raise ConceptStoreError("Not connected to MongoDB", "store_product_concepts")
        
        if not product.conceptual_dimensions:
            return False
        
        try:
            # Prepare product concept document
            concept_doc = {
                "product_id": product.id,
                "title": product.title,
                "category": product.category,
                "brand": product.brand,
                "conceptual_dimensions": dict(product.conceptual_dimensions),
                "metadata": product.metadata or {},
                "updated_at": datetime.utcnow()
            }
            
            # Upsert product concepts
            await self.products_collection.replace_one(
                {"product_id": product.id},
                concept_doc,
                upsert=True
            )
            
            return True
            
        except Exception as e:
            raise ConceptStoreError(f"Failed to store product concepts: {str(e)}", "store_product_concepts")
    
    async def get_product_concepts(self, product_id: str) -> Optional[Dict[str, float]]:
        """Get conceptual dimensions for a product."""
        if not self._connected:
            raise ConceptStoreError("Not connected to MongoDB", "get_product_concepts")
        
        try:
            doc = await self.products_collection.find_one({"product_id": product_id})
            
            if doc and "conceptual_dimensions" in doc:
                return {k: float(v) for k, v in doc["conceptual_dimensions"].items()}
            
            return None
            
        except Exception as e:
            raise ConceptStoreError(f"Failed to get product concepts: {str(e)}", "get_product_concepts")
    
    async def find_similar_products(self, target_dimensions: Dict[str, float], threshold: float = 0.8, limit: int = 10) -> List[Dict[str, Any]]:
        """Find products with similar conceptual dimensions."""
        if not self._connected:
            raise ConceptStoreError("Not connected to MongoDB", "find_similar_products")
        
        try:
            target_dict = (target_dimensions.to_dict() if hasattr(target_dimensions, 'to_dict') else dict(target_dimensions))
            
            # Build aggregation pipeline for similarity search
            pipeline = [
                {
                    "$addFields": {
                        "similarity": {
                            "$let": {
                                "vars": {
                                    "target": target_dict,
                                    "current": "$conceptual_dimensions"
                                },
                                "in": {
                                    "$divide": [
                                        {
                                            "$sum": [
                                                {"$multiply": [
                                                    {"$ifNull": ["$$current.elegance", 0]},
                                                    {"$ifNull": ["$$target.elegance", 0]}
                                                ]},
                                                {"$multiply": [
                                                    {"$ifNull": ["$$current.comfort", 0]},
                                                    {"$ifNull": ["$$target.comfort", 0]}
                                                ]},
                                                {"$multiply": [
                                                    {"$ifNull": ["$$current.boldness", 0]},
                                                    {"$ifNull": ["$$target.boldness", 0]}
                                                ]},
                                                {"$multiply": [
                                                    {"$ifNull": ["$$current.sustainability", 0]},
                                                    {"$ifNull": ["$$target.sustainability", 0]}
                                                ]},
                                                {"$multiply": [
                                                    {"$ifNull": ["$$current.luxury", 0]},
                                                    {"$ifNull": ["$$target.luxury", 0]}
                                                ]},
                                                {"$multiply": [
                                                    {"$ifNull": ["$$current.innovation", 0]},
                                                    {"$ifNull": ["$$target.innovation", 0]}
                                                ]},
                                                {"$multiply": [
                                                    {"$ifNull": ["$$current.versatility", 0]},
                                                    {"$ifNull": ["$$target.versatility", 0]}
                                                ]},
                                                {"$multiply": [
                                                    {"$ifNull": ["$$current.durability", 0]},
                                                    {"$ifNull": ["$$target.durability", 0]}
                                                ]},
                                                {"$multiply": [
                                                    {"$ifNull": ["$$current.affordability", 0]},
                                                    {"$ifNull": ["$$target.affordability", 0]}
                                                ]},
                                                {"$multiply": [
                                                    {"$ifNull": ["$$current.trendiness", 0]},
                                                    {"$ifNull": ["$$target.trendiness", 0]}
                                                ]}
                                            ]
                                        },
                                        {
                                            "$multiply": [
                                                {
                                                    "$sqrt": {
                                                        "$sum": [
                                                            {"$pow": [{"$ifNull": ["$$current.elegance", 0]}, 2]},
                                                            {"$pow": [{"$ifNull": ["$$current.comfort", 0]}, 2]},
                                                            {"$pow": [{"$ifNull": ["$$current.boldness", 0]}, 2]},
                                                            {"$pow": [{"$ifNull": ["$$current.sustainability", 0]}, 2]},
                                                            {"$pow": [{"$ifNull": ["$$current.luxury", 0]}, 2]},
                                                            {"$pow": [{"$ifNull": ["$$current.innovation", 0]}, 2]},
                                                            {"$pow": [{"$ifNull": ["$$current.versatility", 0]}, 2]},
                                                            {"$pow": [{"$ifNull": ["$$current.durability", 0]}, 2]},
                                                            {"$pow": [{"$ifNull": ["$$current.affordability", 0]}, 2]},
                                                            {"$pow": [{"$ifNull": ["$$current.trendiness", 0]}, 2]}
                                                        ]
                                                    }
                                                },
                                                {
                                                    "$sqrt": {
                                                        "$sum": [
                                                            {"$pow": [{"$ifNull": ["$$target.elegance", 0]}, 2]},
                                                            {"$pow": [{"$ifNull": ["$$target.comfort", 0]}, 2]},
                                                            {"$pow": [{"$ifNull": ["$$target.boldness", 0]}, 2]},
                                                            {"$pow": [{"$ifNull": ["$$target.sustainability", 0]}, 2]},
                                                            {"$pow": [{"$ifNull": ["$$target.luxury", 0]}, 2]},
                                                            {"$pow": [{"$ifNull": ["$$target.innovation", 0]}, 2]},
                                                            {"$pow": [{"$ifNull": ["$$target.versatility", 0]}, 2]},
                                                            {"$pow": [{"$ifNull": ["$$target.durability", 0]}, 2]},
                                                            {"$pow": [{"$ifNull": ["$$target.affordability", 0]}, 2]},
                                                            {"$pow": [{"$ifNull": ["$$target.trendiness", 0]}, 2]}
                                                        ]
                                                    }
                                                }
                                            ]
                                        }
                                    ]
                                }
                            }
                        }
                    }
                },
                {"$match": {"similarity": {"$gte": threshold}}},
                {"$sort": {"similarity": -1}},
                {"$limit": limit},
                {
                    "$project": {
                        "product_id": 1,
                        "title": 1,
                        "category": 1,
                        "brand": 1,
                        "conceptual_dimensions": 1,
                        "similarity": 1
                    }
                }
            ]
            
            cursor = self.products_collection.aggregate(pipeline)
            results = []
            
            async for doc in cursor:
                results.append({
                    "product_id": doc["product_id"],
                    "title": doc["title"],
                    "category": doc.get("category"),
                    "brand": doc.get("brand"),
                    "conceptual_dimensions": doc["conceptual_dimensions"],
                    "similarity": doc["similarity"]
                })
            
            return results
            
        except Exception as e:
            raise ConceptStoreError(f"Failed to find similar products: {str(e)}", "find_similar_products")
    
    async def get_concept_distribution(self, dimension: str, category: Optional[str] = None) -> Dict[str, Any]:
        """Get distribution statistics for a conceptual dimension."""
        if not self._connected:
            raise ConceptStoreError("Not connected to MongoDB", "get_concept_distribution")
        
        try:
            match_stage = {}
            if category:
                match_stage["category"] = category
            
            pipeline = [
                {"$match": match_stage},
                {
                    "$group": {
                        "_id": None,
                        "avg": {"$avg": f"$conceptual_dimensions.{dimension}"},
                        "min": {"$min": f"$conceptual_dimensions.{dimension}"},
                        "max": {"$max": f"$conceptual_dimensions.{dimension}"},
                        "count": {"$sum": 1},
                        "values": {"$push": f"$conceptual_dimensions.{dimension}"}
                    }
                },
                {
                    "$addFields": {
                        "std_dev": {
                            "$stdDevPop": "$values"
                        }
                    }
                }
            ]
            
            cursor = self.products_collection.aggregate(pipeline)
            result = await cursor.to_list(length=1)
            
            if result:
                stats = result[0]
                return {
                    "dimension": dimension,
                    "category": category,
                    "average": stats.get("avg", 0),
                    "minimum": stats.get("min", 0),
                    "maximum": stats.get("max", 0),
                    "count": stats.get("count", 0),
                    "standard_deviation": stats.get("std_dev", 0)
                }
            
            return {
                "dimension": dimension,
                "category": category,
                "average": 0,
                "minimum": 0,
                "maximum": 0,
                "count": 0,
                "standard_deviation": 0
            }
            
        except Exception as e:
            raise ConceptStoreError(f"Failed to get concept distribution: {str(e)}", "get_concept_distribution")
    
    async def create_conceptual_space(self, space_name: str, domain: str, dimensions: List[str], metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Create a new conceptual space definition."""
        if not self._connected:
            raise ConceptStoreError("Not connected to MongoDB", "create_conceptual_space")
        
        try:
            space_doc = {
                "space_name": space_name,
                "domain": domain,
                "dimensions": dimensions,
                "metadata": metadata or {},
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            await self.spaces_collection.replace_one(
                {"space_name": space_name},
                space_doc,
                upsert=True
            )
            
            return True
            
        except Exception as e:
            raise ConceptStoreError(f"Failed to create conceptual space: {str(e)}", "create_conceptual_space")
    
    async def get_conceptual_space(self, space_name: str) -> Optional[Dict[str, Any]]:
        """Get conceptual space definition."""
        if not self._connected:
            raise ConceptStoreError("Not connected to MongoDB", "get_conceptual_space")
        
        try:
            doc = await self.spaces_collection.find_one({"space_name": space_name})
            
            if doc:
                # Remove MongoDB ObjectId for JSON serialization
                doc.pop("_id", None)
                return doc
            
            return None
            
        except Exception as e:
            raise ConceptStoreError(f"Failed to get conceptual space: {str(e)}", "get_conceptual_space")
    
    async def list_conceptual_spaces(self, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all conceptual spaces, optionally filtered by domain."""
        if not self._connected:
            raise ConceptStoreError("Not connected to MongoDB", "list_conceptual_spaces")
        
        try:
            query = {}
            if domain:
                query["domain"] = domain
            
            cursor = self.spaces_collection.find(query).sort("created_at", -1)
            spaces = []
            
            async for doc in cursor:
                doc.pop("_id", None)  # Remove MongoDB ObjectId
                spaces.append(doc)
            
            return spaces
            
        except Exception as e:
            raise ConceptStoreError(f"Failed to list conceptual spaces: {str(e)}", "list_conceptual_spaces")
    
    async def delete_product_concepts(self, product_id: str) -> bool:
        """Delete conceptual dimensions for a product."""
        if not self._connected:
            raise ConceptStoreError("Not connected to MongoDB", "delete_product_concepts")
        
        try:
            result = await self.products_collection.delete_one({"product_id": product_id})
            return result.deleted_count > 0
            
        except Exception as e:
            raise ConceptStoreError(f"Failed to delete product concepts: {str(e)}", "delete_product_concepts")
    
    async def get_concept_statistics(self) -> Dict[str, Any]:
        """Get concept store statistics."""
        if not self._connected:
            raise ConceptStoreError("Not connected to MongoDB", "get_concept_statistics")
        
        try:
            stats = {}
            
            # Count products with concepts
            stats["products_with_concepts"] = await self.products_collection.count_documents({})
            
            # Count conceptual spaces
            stats["conceptual_spaces"] = await self.spaces_collection.count_documents({})
            
            # Get category distribution
            pipeline = [
                {"$group": {"_id": "$category", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
            
            cursor = self.products_collection.aggregate(pipeline)
            category_distribution = []
            async for doc in cursor:
                category_distribution.append({
                    "category": doc["_id"],
                    "count": doc["count"]
                })
            
            stats["category_distribution"] = category_distribution
            
            return stats
            
        except Exception as e:
            raise ConceptStoreError(f"Failed to get concept statistics: {str(e)}", "get_concept_statistics")
    # ------------------------------------------------------------------ TC-04
    # ConceptStoreInterface conformance.
    #
    # This class previously implemented the right behaviour under the wrong
    # names — store_product_concepts rather than the interface's store method,
    # and so on — leaving five abstract methods unsatisfied. The class could
    # therefore never be instantiated: construction raised TypeError.
    #
    # The generic, schema-driven surface is implemented natively; MongoDB's
    # free-form documents suit arbitrary dimension names with no schema
    # migration. The legacy fixed-dimension methods were thin adapters
    # over it, so there is one implementation rather than two.

    _DIMENSION_FIELD = "conceptual_dimensions"

    def _require_connection(self, operation: str) -> None:
        if not self._connected:
            raise ConceptStoreError("Not connected to MongoDB", operation)

    async def store_dimensions(self, entity_id: str, scores) -> bool:
        """Store schema-driven dimension scores."""
        self._require_connection("store_dimensions")
        try:
            await self.products_collection.replace_one(
                {"product_id": entity_id},
                {
                    "product_id": entity_id,
                    self._DIMENSION_FIELD: {k: float(v) for k, v in scores.scores.items()},
                    "schema_name": scores.schema_name,
                    "schema_version": scores.schema_version,
                    "updated_at": datetime.utcnow(),
                },
                upsert=True,
            )
            return True
        except ConceptStoreError:
            raise
        except Exception as e:
            raise ConceptStoreError(f"Failed to store dimensions: {str(e)}", "store_dimensions")

    async def get_dimensions(self, entity_id: str):
        """Get schema-driven dimension scores, or None."""
        self._require_connection("get_dimensions")
        from ..core.dimension_store import DimensionScores

        try:
            doc = await self.products_collection.find_one({"product_id": entity_id})
            if not doc or self._DIMENSION_FIELD not in doc:
                return None
            return DimensionScores(
                schema_name=doc.get("schema_name", ""),
                schema_version=doc.get("schema_version", ""),
                scores={k: float(v) for k, v in doc[self._DIMENSION_FIELD].items()},
            )
        except ConceptStoreError:
            raise
        except Exception as e:
            raise ConceptStoreError(f"Failed to get dimensions: {str(e)}", "get_dimensions")

    async def delete_dimensions(self, entity_id: str) -> bool:
        """Delete stored scores. False if there was nothing to delete."""
        self._require_connection("delete_dimensions")
        try:
            result = await self.products_collection.delete_one({"product_id": entity_id})
            return getattr(result, "deleted_count", 0) > 0
        except ConceptStoreError:
            raise
        except Exception as e:
            raise ConceptStoreError(f"Failed to delete dimensions: {str(e)}", "delete_dimensions")

    async def find_similar_dimensions(self, scores, threshold: float = 0.8, limit: int = 10):
        """Rank entities by cosine similarity, within the same schema.

        Similarity is computed in Python rather than in an aggregation
        pipeline: dimension names come from the user's schema, so the
        arithmetic cannot be written into a fixed pipeline the way the legacy
        ten-column version was. The query narrows to the schema first; for
        very large collections this wants a server-side $function or a
        materialised vector, which is a performance change, not a correctness
        one.
        """
        self._require_connection("find_similar_dimensions")
        from ..core.dimension_store import cosine_similarity

        try:
            cursor = self.products_collection.find(
                {"schema_name": scores.schema_name, "schema_version": scores.schema_version}
            )
            hits = []
            async for doc in cursor:
                stored = doc.get(self._DIMENSION_FIELD) or {}
                similarity = cosine_similarity(scores.scores, {k: float(v) for k, v in stored.items()})
                if similarity >= threshold:
                    hits.append((doc["product_id"], similarity))
            hits.sort(key=lambda h: h[1], reverse=True)
            return hits[:limit]
        except ConceptStoreError:
            raise
        except Exception as e:
            raise ConceptStoreError(f"Failed to find similar dimensions: {str(e)}", "find_similar_dimensions")

    async def get_dimension_statistics(self):
        """Per-dimension count/mean/min/max across stored entities.

        Distinct from get_concept_statistics, which reports collection-level
        counts. The interface asks for per-dimension figures.
        """
        self._require_connection("get_dimension_statistics")
        try:
            by_dimension = {}
            cursor = self.products_collection.find({})
            async for doc in cursor:
                for name, value in (doc.get(self._DIMENSION_FIELD) or {}).items():
                    by_dimension.setdefault(name, []).append(float(value))
            return {
                name: {
                    "count": float(len(values)),
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                }
                for name, values in by_dimension.items()
            }
        except ConceptStoreError:
            raise
        except Exception as e:
            raise ConceptStoreError(f"Failed to get dimension statistics: {str(e)}", "get_dimension_statistics")
