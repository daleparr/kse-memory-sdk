"""
Neo4j graph store backend for KSE Memory SDK.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional

try:
    from neo4j import AsyncGraphDatabase, AsyncDriver
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

from ..core.interfaces import GraphStoreInterface
from ..core.config import GraphStoreConfig
from ..exceptions import GraphStoreError, AuthenticationError


logger = logging.getLogger(__name__)


class Neo4jBackend(GraphStoreInterface):
    """
    Neo4j graph store backend for KSE Memory SDK.
    
    Provides graph storage and relationship management using Neo4j
    for knowledge graph functionality.
    """
    
    def __init__(self, config: GraphStoreConfig):
        """
        Initialize Neo4j backend.
        
        Args:
            config: Graph store configuration
        """
        if not NEO4J_AVAILABLE:
            raise GraphStoreError(
                "Neo4j dependencies not installed. Install with: pip install kse-memory[neo4j]",
                operation="initialization"
            )
        
        self.config = config
        self.driver: Optional[AsyncDriver] = None
        self._connected = False
        
        logger.info("Neo4j backend initialized")
    
    async def connect(self) -> bool:
        """
        Connect to Neo4j database.
        
        Returns:
            True if connection successful
            
        Raises:
            GraphStoreError: If connection fails
            AuthenticationError: If authentication fails
        """
        try:
            # Create Neo4j driver
            self.driver = AsyncGraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password),
                database=self.config.database
            )
            
            # Test connection
            await self.driver.verify_connectivity()
            
            # Create indexes for better performance
            await self._create_indexes()
            
            self._connected = True
            logger.info(f"Connected to Neo4j at {self.config.uri}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            if "authentication" in str(e).lower() or "unauthorized" in str(e).lower():
                raise AuthenticationError(f"Neo4j authentication failed: {str(e)}", service="neo4j")
            raise GraphStoreError(f"Connection failed: {str(e)}", operation="connect")
    
    async def disconnect(self) -> bool:
        """
        Disconnect from Neo4j database.
        
        Returns:
            True if disconnection successful
        """
        try:
            if self.driver:
                await self.driver.close()
                self.driver = None
            
            self._connected = False
            logger.info("Disconnected from Neo4j")
            return True
            
        except Exception as e:
            logger.error(f"Error during Neo4j disconnection: {str(e)}")
            return False
    
    async def create_node(self, node_id: str, labels: List[str], properties: Dict[str, Any]) -> bool:
        """
        Create a new node in Neo4j.
        
        Args:
            node_id: Unique node identifier
            labels: List of node labels
            properties: Node properties
            
        Returns:
            True if node created successfully
            
        Raises:
            GraphStoreError: If node creation fails
        """
        self._ensure_connected()
        
        try:
            # Prepare labels string
            labels_str = ":".join(labels)
            
            # Add ID to properties
            properties = properties.copy()
            properties["id"] = node_id
            
            # Create Cypher query
            cypher = f"""
            MERGE (n:{labels_str} {{id: $node_id}})
            SET n += $properties
            RETURN n
            """
            
            async with self.driver.session() as session:
                result = await session.run(cypher, node_id=node_id, properties=properties)
                await result.consume()
            
            logger.debug(f"Created node {node_id} with labels {labels}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create node {node_id}: {str(e)}")
            raise GraphStoreError(f"Node creation failed: {str(e)}", operation="create_node")
    
    async def update_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """
        Update node properties in Neo4j.
        
        Args:
            node_id: Node identifier
            properties: Properties to update
            
        Returns:
            True if node updated successfully
            
        Raises:
            GraphStoreError: If node update fails
        """
        self._ensure_connected()
        
        try:
            cypher = """
            MATCH (n {id: $node_id})
            SET n += $properties
            RETURN n
            """
            
            async with self.driver.session() as session:
                result = await session.run(cypher, node_id=node_id, properties=properties)
                records = await result.data()
                
                if not records:
                    raise GraphStoreError(f"Node {node_id} not found", operation="update_node")
            
            logger.debug(f"Updated node {node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update node {node_id}: {str(e)}")
            raise GraphStoreError(f"Node update failed: {str(e)}", operation="update_node")
    
    async def delete_node(self, node_id: str) -> bool:
        """
        Delete a node from Neo4j.
        
        Args:
            node_id: Node identifier
            
        Returns:
            True if node deleted successfully
            
        Raises:
            GraphStoreError: If node deletion fails
        """
        self._ensure_connected()
        
        try:
            cypher = """
            MATCH (n {id: $node_id})
            DETACH DELETE n
            """
            
            async with self.driver.session() as session:
                result = await session.run(cypher, node_id=node_id)
                await result.consume()
            
            logger.debug(f"Deleted node {node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete node {node_id}: {str(e)}")
            raise GraphStoreError(f"Node deletion failed: {str(e)}", operation="delete_node")
    
    async def create_relationship(
        self, 
        source_id: str, 
        target_id: str, 
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Create a relationship between nodes in Neo4j.
        
        Args:
            source_id: Source node identifier
            target_id: Target node identifier
            relationship_type: Type of relationship
            properties: Optional relationship properties
            
        Returns:
            True if relationship created successfully
            
        Raises:
            GraphStoreError: If relationship creation fails
        """
        self._ensure_connected()
        
        try:
            properties = properties or {}
            
            cypher = f"""
            MATCH (a {{id: $source_id}}), (b {{id: $target_id}})
            MERGE (a)-[r:{relationship_type}]->(b)
            SET r += $properties
            RETURN r
            """
            
            async with self.driver.session() as session:
                result = await session.run(
                    cypher, 
                    source_id=source_id, 
                    target_id=target_id, 
                    properties=properties
                )
                records = await result.data()
                
                if not records:
                    raise GraphStoreError(
                        f"Failed to create relationship: nodes {source_id} or {target_id} not found",
                        operation="create_relationship"
                    )
            
            logger.debug(f"Created relationship {source_id} -[{relationship_type}]-> {target_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create relationship {source_id} -> {target_id}: {str(e)}")
            raise GraphStoreError(f"Relationship creation failed: {str(e)}", operation="create_relationship")
    
    async def delete_relationship(self, source_id: str, target_id: str, relationship_type: str) -> bool:
        """
        Delete a relationship between nodes in Neo4j.
        
        Args:
            source_id: Source node identifier
            target_id: Target node identifier
            relationship_type: Type of relationship
            
        Returns:
            True if relationship deleted successfully
            
        Raises:
            GraphStoreError: If relationship deletion fails
        """
        self._ensure_connected()
        
        try:
            cypher = f"""
            MATCH (a {{id: $source_id}})-[r:{relationship_type}]->(b {{id: $target_id}})
            DELETE r
            """
            
            async with self.driver.session() as session:
                result = await session.run(cypher, source_id=source_id, target_id=target_id)
                await result.consume()
            
            logger.debug(f"Deleted relationship {source_id} -[{relationship_type}]-> {target_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete relationship {source_id} -> {target_id}: {str(e)}")
            raise GraphStoreError(f"Relationship deletion failed: {str(e)}", operation="delete_relationship")
    
    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a node by ID from Neo4j.
        
        Args:
            node_id: Node identifier
            
        Returns:
            Node data if found, None otherwise
            
        Raises:
            GraphStoreError: If retrieval fails
        """
        self._ensure_connected()
        
        try:
            cypher = """
            MATCH (n {id: $node_id})
            RETURN n, labels(n) as labels
            """
            
            async with self.driver.session() as session:
                result = await session.run(cypher, node_id=node_id)
                records = await result.data()
                
                if records:
                    record = records[0]
                    node_data = dict(record["n"])
                    node_data["labels"] = record["labels"]
                    return node_data
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get node {node_id}: {str(e)}")
            raise GraphStoreError(f"Node retrieval failed: {str(e)}", operation="get_node")
    
    async def get_neighbors(self, node_id: str, relationship_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get neighboring nodes in Neo4j.
        
        Args:
            node_id: Node identifier
            relationship_types: Optional list of relationship types to filter
            
        Returns:
            List of neighboring nodes with relationship information
            
        Raises:
            GraphStoreError: If retrieval fails
        """
        self._ensure_connected()
        
        try:
            if relationship_types:
                rel_filter = "|".join(relationship_types)
                cypher = f"""
                MATCH (n {{id: $node_id}})-[r:{rel_filter}]-(neighbor)
                RETURN neighbor, labels(neighbor) as labels, type(r) as relationship_type, r as relationship
                """
            else:
                cypher = """
                MATCH (n {id: $node_id})-[r]-(neighbor)
                RETURN neighbor, labels(neighbor) as labels, type(r) as relationship_type, r as relationship
                """
            
            async with self.driver.session() as session:
                result = await session.run(cypher, node_id=node_id)
                records = await result.data()
                
                neighbors = []
                for record in records:
                    neighbor_data = dict(record["neighbor"])
                    neighbor_data["labels"] = record["labels"]
                    neighbor_data["relationship_type"] = record["relationship_type"]
                    neighbor_data["relationship_properties"] = dict(record["relationship"])
                    neighbors.append(neighbor_data)
                
                return neighbors
                
        except Exception as e:
            logger.error(f"Failed to get neighbors for node {node_id}: {str(e)}")
            raise GraphStoreError(f"Neighbor retrieval failed: {str(e)}", operation="get_neighbors")
    
    async def find_path(self, source_id: str, target_id: str, max_depth: int = 3) -> Optional[List[Dict[str, Any]]]:
        """
        Find path between two nodes in Neo4j.
        
        Args:
            source_id: Source node identifier
            target_id: Target node identifier
            max_depth: Maximum path depth
            
        Returns:
            Path information if found, None otherwise
            
        Raises:
            GraphStoreError: If path finding fails
        """
        self._ensure_connected()
        
        try:
            cypher = f"""
            MATCH path = shortestPath((a {{id: $source_id}})-[*1..{max_depth}]-(b {{id: $target_id}}))
            RETURN path, length(path) as path_length
            """
            
            async with self.driver.session() as session:
                result = await session.run(cypher, source_id=source_id, target_id=target_id)
                records = await result.data()
                
                if records:
                    record = records[0]
                    path_data = {
                        "length": record["path_length"],
                        "nodes": [],
                        "relationships": []
                    }
                    
                    # Extract path information
                    path = record["path"]
                    for node in path.nodes:
                        node_data = dict(node)
                        node_data["labels"] = list(node.labels)
                        path_data["nodes"].append(node_data)
                    
                    for rel in path.relationships:
                        rel_data = dict(rel)
                        rel_data["type"] = rel.type
                        rel_data["start_node"] = rel.start_node.element_id
                        rel_data["end_node"] = rel.end_node.element_id
                        path_data["relationships"].append(rel_data)
                    
                    return [path_data]
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to find path {source_id} -> {target_id}: {str(e)}")
            raise GraphStoreError(f"Path finding failed: {str(e)}", operation="find_path")
    
    async def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a custom Cypher query in Neo4j.
        
        Args:
            query: Cypher query string
            parameters: Optional query parameters
            
        Returns:
            Query results
            
        Raises:
            GraphStoreError: If query execution fails
        """
        self._ensure_connected()
        
        try:
            parameters = parameters or {}
            
            async with self.driver.session() as session:
                result = await session.run(query, **parameters)
                records = await result.data()
                return records
                
        except Exception as e:
            logger.error(f"Failed to execute query: {str(e)}")
            raise GraphStoreError(f"Query execution failed: {str(e)}", operation="execute_query")
    
    async def _create_indexes(self):
        """Create indexes for better performance."""
        try:
            indexes = [
                "CREATE INDEX product_id_index IF NOT EXISTS FOR (p:Product) ON (p.id)",
                "CREATE INDEX category_name_index IF NOT EXISTS FOR (c:Category) ON (c.name)",
                "CREATE INDEX brand_name_index IF NOT EXISTS FOR (b:Brand) ON (b.name)",
                "CREATE INDEX tag_name_index IF NOT EXISTS FOR (t:Tag) ON (t.name)",
            ]
            
            async with self.driver.session() as session:
                for index_query in indexes:
                    try:
                        await session.run(index_query)
                    except Exception as e:
                        # Index might already exist, which is fine
                        logger.debug(f"Index creation note: {str(e)}")
            
            logger.debug("Neo4j indexes created/verified")
            
        except Exception as e:
            logger.warning(f"Failed to create Neo4j indexes: {str(e)}")
    
    def _ensure_connected(self):
        """Ensure the backend is connected."""
        if not self._connected:
            raise GraphStoreError("Not connected to Neo4j. Call connect() first.", operation="check_connection")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get graph statistics.
        
        Returns:
            Dictionary with graph statistics
        """
        self._ensure_connected()
        
        try:
            stats_queries = {
                "node_count": "MATCH (n) RETURN count(n) as count",
                "relationship_count": "MATCH ()-[r]->() RETURN count(r) as count",
                "product_count": "MATCH (p:Product) RETURN count(p) as count",
                "category_count": "MATCH (c:Category) RETURN count(c) as count",
                "brand_count": "MATCH (b:Brand) RETURN count(b) as count",
            }
            
            stats = {}
            async with self.driver.session() as session:
                for stat_name, query in stats_queries.items():
                    try:
                        result = await session.run(query)
                        record = await result.single()
                        stats[stat_name] = record["count"] if record else 0
                    except Exception as e:
                        logger.warning(f"Failed to get {stat_name}: {str(e)}")
                        stats[stat_name] = 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get graph statistics: {str(e)}")
            return {}