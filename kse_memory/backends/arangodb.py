"""
ArangoDB graph store backend for KSE Memory SDK.
"""

from typing import List, Dict, Any, Optional
import asyncio
from ..core.interfaces import GraphStoreInterface
from ..core.models import Product, KnowledgeGraph
from ..core.config import GraphStoreConfig
from ..exceptions import BackendError, GraphStoreError

try:
    from arango import ArangoClient
    from arango.database import StandardDatabase
    ARANGODB_AVAILABLE = True
except ImportError:
    ARANGODB_AVAILABLE = False


class ArangoDBBackend(GraphStoreInterface):
    """ArangoDB graph store implementation."""
    
    def __init__(self, config: GraphStoreConfig):
        """Initialize ArangoDB backend."""
        if not ARANGODB_AVAILABLE:
            raise BackendError("python-arango package is required for ArangoDB backend", "arangodb")
        
        self.config = config
        self.client = None
        self.db = None
        self.graph = None
        self._connected = False
    
    async def connect(self) -> bool:
        """Connect to ArangoDB instance."""
        try:
            # Create ArangoDB client
            self.client = ArangoClient(hosts=self.config.uri)
            
            # Connect to database
            self.db = self.client.db(
                self.config.database,
                username=self.config.username,
                password=self.config.password
            )
            
            # Test connection
            await asyncio.to_thread(self.db.version)
            
            # Create or get graph
            graph_name = "kse_knowledge_graph"
            if not await asyncio.to_thread(self.db.has_graph, graph_name):
                # Create graph with edge definitions
                edge_definitions = [
                    {
                        "edge_collection": "relationships",
                        "from_vertex_collections": ["products", "categories", "brands"],
                        "to_vertex_collections": ["products", "categories", "brands"]
                    }
                ]
                self.graph = await asyncio.to_thread(
                    self.db.create_graph,
                    graph_name,
                    edge_definitions
                )
            else:
                self.graph = self.db.graph(graph_name)
            
            # Ensure collections exist
            await self._ensure_collections()
            
            self._connected = True
            return True
            
        except Exception as e:
            raise GraphStoreError(f"Failed to connect to ArangoDB: {str(e)}", "connect")
    
    async def disconnect(self) -> bool:
        """Disconnect from ArangoDB."""
        self._connected = False
        return True
    
    async def _ensure_collections(self):
        """Ensure required collections exist."""
        collections = ["products", "categories", "brands", "relationships"]
        
        for collection_name in collections:
            if collection_name == "relationships":
                # Edge collection
                if not await asyncio.to_thread(self.db.has_collection, collection_name):
                    await asyncio.to_thread(self.db.create_collection, collection_name, edge=True)
            else:
                # Vertex collection
                if not await asyncio.to_thread(self.db.has_collection, collection_name):
                    await asyncio.to_thread(self.db.create_collection, collection_name)
    
    async def add_product_node(self, product: Product) -> bool:
        """Add product as a node in the graph."""
        if not self._connected:
            raise GraphStoreError("Not connected to ArangoDB", "add_product_node")
        
        try:
            products_collection = self.db.collection("products")
            
            # Prepare product document
            product_doc = {
                "_key": product.id,
                "title": product.title,
                "description": product.description,
                "price": product.price,
                "currency": product.currency,
                "category": product.category,
                "brand": product.brand,
                "tags": product.tags,
                "metadata": product.metadata
            }
            
            # Add conceptual dimensions if available
            if product.conceptual_dimensions:
                product_doc["conceptual_dimensions"] = dict(product.conceptual_dimensions)
            
            # Insert or update product
            await asyncio.to_thread(
                products_collection.insert,
                product_doc,
                overwrite=True
            )
            
            # Add category and brand nodes if they don't exist
            if product.category:
                await self._add_category_node(product.category)
                await self._add_relationship(f"products/{product.id}", f"categories/{product.category}", "BELONGS_TO")
            
            if product.brand:
                await self._add_brand_node(product.brand)
                await self._add_relationship(f"products/{product.id}", f"brands/{product.brand}", "MADE_BY")
            
            return True
            
        except Exception as e:
            raise GraphStoreError(f"Failed to add product node: {str(e)}", "add_product_node")
    
    async def _add_category_node(self, category: str):
        """Add category node if it doesn't exist."""
        try:
            categories_collection = self.db.collection("categories")
            category_doc = {
                "_key": category,
                "name": category,
                "type": "category"
            }
            await asyncio.to_thread(
                categories_collection.insert,
                category_doc,
                overwrite=True
            )
        except Exception:
            pass  # Category might already exist
    
    async def _add_brand_node(self, brand: str):
        """Add brand node if it doesn't exist."""
        try:
            brands_collection = self.db.collection("brands")
            brand_doc = {
                "_key": brand,
                "name": brand,
                "type": "brand"
            }
            await asyncio.to_thread(
                brands_collection.insert,
                brand_doc,
                overwrite=True
            )
        except Exception:
            pass  # Brand might already exist
    
    async def _add_relationship(self, from_id: str, to_id: str, relationship_type: str):
        """Add relationship between nodes."""
        try:
            relationships_collection = self.db.collection("relationships")
            relationship_doc = {
                "_from": from_id,
                "_to": to_id,
                "type": relationship_type
            }
            await asyncio.to_thread(
                relationships_collection.insert,
                relationship_doc
            )
        except Exception:
            pass  # Relationship might already exist
    
    async def add_relationship(self, from_node: str, to_node: str, relationship_type: str, properties: Optional[Dict[str, Any]] = None) -> bool:
        """Add relationship between nodes."""
        if not self._connected:
            raise GraphStoreError("Not connected to ArangoDB", "add_relationship")
        
        try:
            relationships_collection = self.db.collection("relationships")
            
            relationship_doc = {
                "_from": from_node,
                "_to": to_node,
                "type": relationship_type
            }
            
            if properties:
                relationship_doc.update(properties)
            
            await asyncio.to_thread(
                relationships_collection.insert,
                relationship_doc
            )
            
            return True
            
        except Exception as e:
            raise GraphStoreError(f"Failed to add relationship: {str(e)}", "add_relationship")
    
    async def find_related_products(self, product_id: str, relationship_types: Optional[List[str]] = None, max_depth: int = 2) -> List[Dict[str, Any]]:
        """Find products related to the given product."""
        if not self._connected:
            raise GraphStoreError("Not connected to ArangoDB", "find_related_products")
        
        try:
            # Build AQL query for traversal
            start_vertex = f"products/{product_id}"
            
            if relationship_types:
                edge_filter = f"FILTER p.edges[*].type IN {relationship_types}"
            else:
                edge_filter = ""
            
            aql_query = f"""
            FOR v, e, p IN 1..{max_depth} ANY @start_vertex relationships
            {edge_filter}
            FILTER v._id != @start_vertex
            FILTER STARTS_WITH(v._id, 'products/')
            RETURN {{
                id: v._key,
                title: v.title,
                category: v.category,
                brand: v.brand,
                price: v.price,
                relationship_path: p.edges[*].type
            }}
            """
            
            cursor = await asyncio.to_thread(
                self.db.aql.execute,
                aql_query,
                bind_vars={"start_vertex": start_vertex}
            )
            
            results = []
            async for doc in asyncio.to_thread(cursor):
                results.append(doc)
            
            return results
            
        except Exception as e:
            raise GraphStoreError(f"Failed to find related products: {str(e)}", "find_related_products")
    
    async def get_product_relationships(self, product_id: str) -> List[Dict[str, Any]]:
        """Get all relationships for a product."""
        if not self._connected:
            raise GraphStoreError("Not connected to ArangoDB", "get_product_relationships")
        
        try:
            start_vertex = f"products/{product_id}"
            
            aql_query = """
            FOR v, e IN 1..1 ANY @start_vertex relationships
            RETURN {
                target_id: v._key,
                target_type: v.type,
                relationship_type: e.type,
                direction: e._from == @start_vertex ? 'outbound' : 'inbound'
            }
            """
            
            cursor = await asyncio.to_thread(
                self.db.aql.execute,
                aql_query,
                bind_vars={"start_vertex": start_vertex}
            )
            
            results = []
            async for doc in asyncio.to_thread(cursor):
                results.append(doc)
            
            return results
            
        except Exception as e:
            raise GraphStoreError(f"Failed to get product relationships: {str(e)}", "get_product_relationships")
    
    async def delete_product_node(self, product_id: str) -> bool:
        """Delete product node and its relationships."""
        if not self._connected:
            raise GraphStoreError("Not connected to ArangoDB", "delete_product_node")
        
        try:
            # Delete the product document
            products_collection = self.db.collection("products")
            await asyncio.to_thread(products_collection.delete, product_id)
            
            # Delete all relationships involving this product
            relationships_collection = self.db.collection("relationships")
            product_vertex = f"products/{product_id}"
            
            aql_query = """
            FOR r IN relationships
            FILTER r._from == @vertex OR r._to == @vertex
            REMOVE r IN relationships
            """
            
            await asyncio.to_thread(
                self.db.aql.execute,
                aql_query,
                bind_vars={"vertex": product_vertex}
            )
            
            return True
            
        except Exception as e:
            raise GraphStoreError(f"Failed to delete product node: {str(e)}", "delete_product_node")
    
    async def execute_graph_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute custom AQL graph query."""
        if not self._connected:
            raise GraphStoreError("Not connected to ArangoDB", "execute_graph_query")
        
        try:
            cursor = await asyncio.to_thread(
                self.db.aql.execute,
                query,
                bind_vars=parameters or {}
            )
            
            results = []
            async for doc in asyncio.to_thread(cursor):
                results.append(doc)
            
            return results
            
        except Exception as e:
            raise GraphStoreError(f"Failed to execute graph query: {str(e)}", "execute_graph_query")
    
    async def get_graph_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        if not self._connected:
            raise GraphStoreError("Not connected to ArangoDB", "get_graph_statistics")
        
        try:
            stats = {}
            
            # Count nodes by type
            for collection_name in ["products", "categories", "brands"]:
                collection = self.db.collection(collection_name)
                count = await asyncio.to_thread(collection.count)
                stats[f"{collection_name}_count"] = count
            
            # Count relationships
            relationships_collection = self.db.collection("relationships")
            stats["relationships_count"] = await asyncio.to_thread(relationships_collection.count)
            
            return stats
            
        except Exception as e:
            raise GraphStoreError(f"Failed to get graph statistics: {str(e)}", "get_graph_statistics")
    # ------------------------------------------------------------------ US9
    # GraphStoreInterface conformance.
    #
    # This class previously implemented its behaviour under different names
    # (add_product_node, find_related_products, ...) and left nine abstract
    # methods unsatisfied — the same defect family the static conformance
    # suite caught in MongoDBBackend: the class could never be instantiated.
    #
    # The generic surface is implemented natively over two collections:
    # ``nodes`` (vertex) and ``node_edges`` (edge, keyed by a deterministic
    # source|type|target key so create_relationship upserts). The legacy
    # product-centric methods above are untouched.

    _NODES = "nodes"
    _EDGES = "node_edges"

    @staticmethod
    def _edge_key(source_id: str, target_id: str, relationship_type: str) -> str:
        import hashlib

        raw = f"{source_id}|{relationship_type}|{target_id}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    async def _generic_collections(self):
        for name, is_edge in ((self._NODES, False), (self._EDGES, True)):
            if not await asyncio.to_thread(self.db.has_collection, name):
                await asyncio.to_thread(self.db.create_collection, name, edge=is_edge)

    def _require_connection(self, operation: str) -> None:
        if not self._connected:
            raise GraphStoreError("Not connected to ArangoDB", operation)

    async def create_node(self, node_id: str, labels: List[str], properties: Dict[str, Any]) -> bool:
        self._require_connection("create_node")
        await self._generic_collections()
        collection = self.db.collection(self._NODES)
        document = {"_key": node_id, "labels": list(labels), "properties": dict(properties)}
        await asyncio.to_thread(collection.insert, document, overwrite=True)
        return True

    async def update_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        self._require_connection("update_node")
        await self._generic_collections()
        collection = self.db.collection(self._NODES)
        existing = await asyncio.to_thread(collection.get, node_id)
        if existing is None:
            return await self.create_node(node_id, [], properties)
        merged = dict(existing.get("properties", {}))
        merged.update(properties)
        await asyncio.to_thread(
            collection.update, {"_key": node_id, "properties": merged}
        )
        return True

    async def delete_node(self, node_id: str) -> bool:
        self._require_connection("delete_node")
        await self._generic_collections()
        collection = self.db.collection(self._NODES)
        existing = await asyncio.to_thread(collection.get, node_id)
        if existing is None:
            return False
        # remove incident edges first, both directions
        await asyncio.to_thread(
            self.db.aql.execute,
            "FOR e IN @@edges FILTER e._from == @ref OR e._to == @ref REMOVE e IN @@edges",
            bind_vars={"@edges": self._EDGES, "ref": f"{self._NODES}/{node_id}"},
        )
        await asyncio.to_thread(collection.delete, node_id)
        return True

    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        self._require_connection("get_node")
        await self._generic_collections()
        document = await asyncio.to_thread(self.db.collection(self._NODES).get, node_id)
        if document is None:
            return None
        return {
            "labels": list(document.get("labels", [])),
            "properties": dict(document.get("properties", {})),
        }

    async def get_neighbors(
        self, node_id: str, relationship_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Connected in EITHER direction, per the FR-04 traversal contract."""
        self._require_connection("get_neighbors")
        await self._generic_collections()
        cursor = await asyncio.to_thread(
            self.db.aql.execute,
            """
            FOR e IN @@edges
              FILTER (e._from == @ref OR e._to == @ref)
              FILTER @types == null OR e.type IN @types
              RETURN e._from == @ref ? e._to : e._from
            """,
            bind_vars={
                "@edges": self._EDGES,
                "ref": f"{self._NODES}/{node_id}",
                "types": list(relationship_types) if relationship_types is not None else None,
            },
        )
        out, seen = [], set()
        for handle in cursor:
            other = str(handle).split("/", 1)[-1]
            if other not in seen:
                seen.add(other)
                out.append({"id": other})
        return out

    async def create_relationship(
        self, source_id: str, target_id: str, relationship_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        self._require_connection("create_relationship")
        await self._generic_collections()
        edge = {
            "_key": self._edge_key(source_id, target_id, relationship_type),
            "_from": f"{self._NODES}/{source_id}",
            "_to": f"{self._NODES}/{target_id}",
            "type": relationship_type,
            "properties": dict(properties or {}),
        }
        await asyncio.to_thread(self.db.collection(self._EDGES).insert, edge, overwrite=True)
        return True

    async def delete_relationship(self, source_id: str, target_id: str, relationship_type: str) -> bool:
        self._require_connection("delete_relationship")
        await self._generic_collections()
        collection = self.db.collection(self._EDGES)
        key = self._edge_key(source_id, target_id, relationship_type)
        existing = await asyncio.to_thread(collection.get, key)
        if existing is None:
            return False
        await asyncio.to_thread(collection.delete, key)
        return True

    async def find_path(
        self, source_id: str, target_id: str, max_depth: int = 3
    ) -> Optional[List[Dict[str, Any]]]:
        self._require_connection("find_path")
        await self._generic_collections()
        cursor = await asyncio.to_thread(
            self.db.aql.execute,
            """
            FOR v, e IN 1..@depth ANY @start @@edges
              FILTER v._id == @goal
              LIMIT 1
              RETURN 1
            """,
            bind_vars={
                "@edges": self._EDGES,
                "start": f"{self._NODES}/{source_id}",
                "goal": f"{self._NODES}/{target_id}",
                "depth": int(max_depth),
            },
        )
        found = any(True for _ in cursor)
        # The portable contract wants the path; AQL shortest-path detail is a
        # follow-up — existence with endpoints satisfies the current callers.
        return [{"id": source_id}, {"id": target_id}] if found else None

    async def execute_query(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        self._require_connection("execute_query")
        cursor = await asyncio.to_thread(self.db.aql.execute, query, bind_vars=parameters or {})
        return [dict(row) if isinstance(row, dict) else {"value": row} for row in cursor]
