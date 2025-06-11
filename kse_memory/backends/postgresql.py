"""
PostgreSQL concept store backend for KSE Memory SDK.
"""

import asyncio
import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

from ..core.interfaces import ConceptStoreInterface
from ..core.config import ConceptStoreConfig
from ..core.models import ConceptualDimensions
from ..exceptions import ConceptStoreError, AuthenticationError


logger = logging.getLogger(__name__)


class PostgreSQLBackend(ConceptStoreInterface):
    """
    PostgreSQL concept store backend for KSE Memory SDK.
    
    Provides storage and similarity search for conceptual dimensions
    using PostgreSQL with vector similarity extensions.
    """
    
    def __init__(self, config: ConceptStoreConfig):
        """
        Initialize PostgreSQL backend.
        
        Args:
            config: Concept store configuration
        """
        if not ASYNCPG_AVAILABLE:
            raise ConceptStoreError(
                "AsyncPG dependencies not installed. Install with: pip install kse-memory[postgresql]",
                operation="initialization"
            )
        
        self.config = config
        self.pool = None
        self._connected = False
        
        # Standard conceptual dimensions
        self.dimensions = [
            "elegance", "comfort", "boldness", "modernity", "minimalism",
            "luxury", "functionality", "versatility", "seasonality", "innovation"
        ]
        
        logger.info("PostgreSQL backend initialized")
    
    async def connect(self) -> bool:
        """
        Connect to PostgreSQL database.
        
        Returns:
            True if connection successful
            
        Raises:
            ConceptStoreError: If connection fails
            AuthenticationError: If authentication fails
        """
        try:
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.username,
                password=self.config.password,
                min_size=1,
                max_size=10,
                command_timeout=30
            )
            
            # Test connection and create tables
            async with self.pool.acquire() as conn:
                await conn.execute("SELECT 1")
                await self._create_tables(conn)
                await self._create_indexes(conn)
            
            self._connected = True
            logger.info(f"Connected to PostgreSQL at {self.config.host}:{self.config.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {str(e)}")
            if "authentication" in str(e).lower() or "password" in str(e).lower():
                raise AuthenticationError(f"PostgreSQL authentication failed: {str(e)}", service="postgresql")
            raise ConceptStoreError(f"Connection failed: {str(e)}", operation="connect")
    
    async def disconnect(self) -> bool:
        """
        Disconnect from PostgreSQL database.
        
        Returns:
            True if disconnection successful
        """
        try:
            if self.pool:
                await self.pool.close()
                self.pool = None
            
            self._connected = False
            logger.info("Disconnected from PostgreSQL")
            return True
            
        except Exception as e:
            logger.error(f"Error during PostgreSQL disconnection: {str(e)}")
            return False
    
    async def store_conceptual_dimensions(self, product_id: str, dimensions: ConceptualDimensions) -> bool:
        """
        Store conceptual dimensions for a product.
        
        Args:
            product_id: Product identifier
            dimensions: Conceptual dimensions to store
            
        Returns:
            True if storage successful
            
        Raises:
            ConceptStoreError: If storage fails
        """
        self._ensure_connected()
        
        try:
            dim_dict = dimensions.to_dict()
            
            # Create vector representation
            vector = [dim_dict.get(dim, 0.0) for dim in self.dimensions]
            
            async with self.pool.acquire() as conn:
                # Upsert conceptual dimensions
                await conn.execute("""
                    INSERT INTO conceptual_dimensions (
                        product_id, elegance, comfort, boldness, modernity, minimalism,
                        luxury, functionality, versatility, seasonality, innovation,
                        vector, created_at, updated_at
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14
                    )
                    ON CONFLICT (product_id) DO UPDATE SET
                        elegance = EXCLUDED.elegance,
                        comfort = EXCLUDED.comfort,
                        boldness = EXCLUDED.boldness,
                        modernity = EXCLUDED.modernity,
                        minimalism = EXCLUDED.minimalism,
                        luxury = EXCLUDED.luxury,
                        functionality = EXCLUDED.functionality,
                        versatility = EXCLUDED.versatility,
                        seasonality = EXCLUDED.seasonality,
                        innovation = EXCLUDED.innovation,
                        vector = EXCLUDED.vector,
                        updated_at = EXCLUDED.updated_at
                """, 
                    product_id,
                    dim_dict["elegance"],
                    dim_dict["comfort"],
                    dim_dict["boldness"],
                    dim_dict["modernity"],
                    dim_dict["minimalism"],
                    dim_dict["luxury"],
                    dim_dict["functionality"],
                    dim_dict["versatility"],
                    dim_dict["seasonality"],
                    dim_dict["innovation"],
                    vector,
                    datetime.utcnow(),
                    datetime.utcnow()
                )
            
            logger.debug(f"Stored conceptual dimensions for product {product_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store conceptual dimensions for product {product_id}: {str(e)}")
            raise ConceptStoreError(f"Storage failed: {str(e)}", operation="store")
    
    async def get_conceptual_dimensions(self, product_id: str) -> Optional[ConceptualDimensions]:
        """
        Get conceptual dimensions for a product.
        
        Args:
            product_id: Product identifier
            
        Returns:
            ConceptualDimensions if found, None otherwise
            
        Raises:
            ConceptStoreError: If retrieval fails
        """
        self._ensure_connected()
        
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT elegance, comfort, boldness, modernity, minimalism,
                           luxury, functionality, versatility, seasonality, innovation
                    FROM conceptual_dimensions
                    WHERE product_id = $1
                """, product_id)
                
                if row:
                    return ConceptualDimensions(
                        elegance=row["elegance"],
                        comfort=row["comfort"],
                        boldness=row["boldness"],
                        modernity=row["modernity"],
                        minimalism=row["minimalism"],
                        luxury=row["luxury"],
                        functionality=row["functionality"],
                        versatility=row["versatility"],
                        seasonality=row["seasonality"],
                        innovation=row["innovation"]
                    )
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get conceptual dimensions for product {product_id}: {str(e)}")
            raise ConceptStoreError(f"Retrieval failed: {str(e)}", operation="get")
    
    async def delete_conceptual_dimensions(self, product_id: str) -> bool:
        """
        Delete conceptual dimensions for a product.
        
        Args:
            product_id: Product identifier
            
        Returns:
            True if deletion successful
            
        Raises:
            ConceptStoreError: If deletion fails
        """
        self._ensure_connected()
        
        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute("""
                    DELETE FROM conceptual_dimensions
                    WHERE product_id = $1
                """, product_id)
                
                # Check if any rows were deleted
                rows_deleted = int(result.split()[-1])
                
            logger.debug(f"Deleted conceptual dimensions for product {product_id}")
            return rows_deleted > 0
            
        except Exception as e:
            logger.error(f"Failed to delete conceptual dimensions for product {product_id}: {str(e)}")
            raise ConceptStoreError(f"Deletion failed: {str(e)}", operation="delete")
    
    async def find_similar_concepts(
        self, 
        dimensions: ConceptualDimensions, 
        threshold: float = 0.8,
        limit: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find products with similar conceptual dimensions.
        
        Args:
            dimensions: Target conceptual dimensions
            threshold: Similarity threshold (0.0 to 1.0)
            limit: Maximum number of results
            
        Returns:
            List of (product_id, similarity_score) tuples
            
        Raises:
            ConceptStoreError: If search fails
        """
        self._ensure_connected()
        
        try:
            dim_dict = dimensions.to_dict()
            target_vector = [dim_dict.get(dim, 0.0) for dim in self.dimensions]
            
            async with self.pool.acquire() as conn:
                # Calculate cosine similarity
                rows = await conn.fetch("""
                    SELECT 
                        product_id,
                        (
                            (vector[1] * $1 + vector[2] * $2 + vector[3] * $3 + vector[4] * $4 + vector[5] * $5 +
                             vector[6] * $6 + vector[7] * $7 + vector[8] * $8 + vector[9] * $9 + vector[10] * $10)
                            /
                            (
                                sqrt(vector[1]^2 + vector[2]^2 + vector[3]^2 + vector[4]^2 + vector[5]^2 +
                                     vector[6]^2 + vector[7]^2 + vector[8]^2 + vector[9]^2 + vector[10]^2) *
                                sqrt($1^2 + $2^2 + $3^2 + $4^2 + $5^2 + $6^2 + $7^2 + $8^2 + $9^2 + $10^2)
                            )
                        ) as similarity
                    FROM conceptual_dimensions
                    WHERE 
                        sqrt(vector[1]^2 + vector[2]^2 + vector[3]^2 + vector[4]^2 + vector[5]^2 +
                             vector[6]^2 + vector[7]^2 + vector[8]^2 + vector[9]^2 + vector[10]^2) > 0
                        AND sqrt($1^2 + $2^2 + $3^2 + $4^2 + $5^2 + $6^2 + $7^2 + $8^2 + $9^2 + $10^2) > 0
                    HAVING similarity >= $11
                    ORDER BY similarity DESC
                    LIMIT $12
                """, *target_vector, threshold, limit)
                
                results = [(row["product_id"], float(row["similarity"])) for row in rows]
                
            logger.debug(f"Found {len(results)} similar concepts")
            return results
            
        except Exception as e:
            logger.error(f"Failed to find similar concepts: {str(e)}")
            raise ConceptStoreError(f"Similarity search failed: {str(e)}", operation="find_similar")
    
    async def get_dimension_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for each conceptual dimension.
        
        Returns:
            Dictionary with dimension statistics
            
        Raises:
            ConceptStoreError: If statistics retrieval fails
        """
        self._ensure_connected()
        
        try:
            stats = {}
            
            async with self.pool.acquire() as conn:
                for dimension in self.dimensions:
                    row = await conn.fetchrow(f"""
                        SELECT 
                            COUNT(*) as count,
                            AVG({dimension}) as mean,
                            STDDEV({dimension}) as stddev,
                            MIN({dimension}) as min_val,
                            MAX({dimension}) as max_val,
                            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {dimension}) as q25,
                            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {dimension}) as median,
                            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {dimension}) as q75
                        FROM conceptual_dimensions
                        WHERE {dimension} IS NOT NULL
                    """)
                    
                    if row and row["count"] > 0:
                        stats[dimension] = {
                            "count": int(row["count"]),
                            "mean": float(row["mean"]) if row["mean"] else 0.0,
                            "stddev": float(row["stddev"]) if row["stddev"] else 0.0,
                            "min": float(row["min_val"]) if row["min_val"] else 0.0,
                            "max": float(row["max_val"]) if row["max_val"] else 0.0,
                            "q25": float(row["q25"]) if row["q25"] else 0.0,
                            "median": float(row["median"]) if row["median"] else 0.0,
                            "q75": float(row["q75"]) if row["q75"] else 0.0,
                        }
                    else:
                        stats[dimension] = {
                            "count": 0,
                            "mean": 0.0,
                            "stddev": 0.0,
                            "min": 0.0,
                            "max": 0.0,
                            "q25": 0.0,
                            "median": 0.0,
                            "q75": 0.0,
                        }
            
            logger.debug("Retrieved dimension statistics")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get dimension statistics: {str(e)}")
            raise ConceptStoreError(f"Statistics retrieval failed: {str(e)}", operation="get_statistics")
    
    async def _create_tables(self, conn):
        """Create necessary tables."""
        try:
            # Create conceptual dimensions table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS conceptual_dimensions (
                    product_id VARCHAR(255) PRIMARY KEY,
                    elegance REAL NOT NULL DEFAULT 0.0,
                    comfort REAL NOT NULL DEFAULT 0.0,
                    boldness REAL NOT NULL DEFAULT 0.0,
                    modernity REAL NOT NULL DEFAULT 0.0,
                    minimalism REAL NOT NULL DEFAULT 0.0,
                    luxury REAL NOT NULL DEFAULT 0.0,
                    functionality REAL NOT NULL DEFAULT 0.0,
                    versatility REAL NOT NULL DEFAULT 0.0,
                    seasonality REAL NOT NULL DEFAULT 0.0,
                    innovation REAL NOT NULL DEFAULT 0.0,
                    vector REAL[] NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
                )
            """)
            
            # Add constraints
            await conn.execute("""
                ALTER TABLE conceptual_dimensions 
                ADD CONSTRAINT IF NOT EXISTS check_elegance 
                CHECK (elegance >= 0.0 AND elegance <= 1.0)
            """)
            
            await conn.execute("""
                ALTER TABLE conceptual_dimensions 
                ADD CONSTRAINT IF NOT EXISTS check_comfort 
                CHECK (comfort >= 0.0 AND comfort <= 1.0)
            """)
            
            await conn.execute("""
                ALTER TABLE conceptual_dimensions 
                ADD CONSTRAINT IF NOT EXISTS check_boldness 
                CHECK (boldness >= 0.0 AND boldness <= 1.0)
            """)
            
            await conn.execute("""
                ALTER TABLE conceptual_dimensions 
                ADD CONSTRAINT IF NOT EXISTS check_modernity 
                CHECK (modernity >= 0.0 AND modernity <= 1.0)
            """)
            
            await conn.execute("""
                ALTER TABLE conceptual_dimensions 
                ADD CONSTRAINT IF NOT EXISTS check_minimalism 
                CHECK (minimalism >= 0.0 AND minimalism <= 1.0)
            """)
            
            await conn.execute("""
                ALTER TABLE conceptual_dimensions 
                ADD CONSTRAINT IF NOT EXISTS check_luxury 
                CHECK (luxury >= 0.0 AND luxury <= 1.0)
            """)
            
            await conn.execute("""
                ALTER TABLE conceptual_dimensions 
                ADD CONSTRAINT IF NOT EXISTS check_functionality 
                CHECK (functionality >= 0.0 AND functionality <= 1.0)
            """)
            
            await conn.execute("""
                ALTER TABLE conceptual_dimensions 
                ADD CONSTRAINT IF NOT EXISTS check_versatility 
                CHECK (versatility >= 0.0 AND versatility <= 1.0)
            """)
            
            await conn.execute("""
                ALTER TABLE conceptual_dimensions 
                ADD CONSTRAINT IF NOT EXISTS check_seasonality 
                CHECK (seasonality >= 0.0 AND seasonality <= 1.0)
            """)
            
            await conn.execute("""
                ALTER TABLE conceptual_dimensions 
                ADD CONSTRAINT IF NOT EXISTS check_innovation 
                CHECK (innovation >= 0.0 AND innovation <= 1.0)
            """)
            
            logger.debug("PostgreSQL tables created/verified")
            
        except Exception as e:
            logger.warning(f"Failed to create PostgreSQL tables: {str(e)}")
            raise
    
    async def _create_indexes(self, conn):
        """Create indexes for better performance."""
        try:
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_conceptual_dimensions_product_id ON conceptual_dimensions (product_id)",
                "CREATE INDEX IF NOT EXISTS idx_conceptual_dimensions_elegance ON conceptual_dimensions (elegance)",
                "CREATE INDEX IF NOT EXISTS idx_conceptual_dimensions_comfort ON conceptual_dimensions (comfort)",
                "CREATE INDEX IF NOT EXISTS idx_conceptual_dimensions_luxury ON conceptual_dimensions (luxury)",
                "CREATE INDEX IF NOT EXISTS idx_conceptual_dimensions_created_at ON conceptual_dimensions (created_at)",
                "CREATE INDEX IF NOT EXISTS idx_conceptual_dimensions_updated_at ON conceptual_dimensions (updated_at)",
            ]
            
            for index_query in indexes:
                try:
                    await conn.execute(index_query)
                except Exception as e:
                    logger.debug(f"Index creation note: {str(e)}")
            
            logger.debug("PostgreSQL indexes created/verified")
            
        except Exception as e:
            logger.warning(f"Failed to create PostgreSQL indexes: {str(e)}")
    
    def _ensure_connected(self):
        """Ensure the backend is connected."""
        if not self._connected:
            raise ConceptStoreError("Not connected to PostgreSQL. Call connect() first.", operation="check_connection")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get general database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        self._ensure_connected()
        
        try:
            stats = {}
            
            async with self.pool.acquire() as conn:
                # Get total count
                row = await conn.fetchrow("SELECT COUNT(*) as total FROM conceptual_dimensions")
                stats["total_products"] = int(row["total"]) if row else 0
                
                # Get recent additions
                row = await conn.fetchrow("""
                    SELECT COUNT(*) as recent 
                    FROM conceptual_dimensions 
                    WHERE created_at >= NOW() - INTERVAL '24 hours'
                """)
                stats["recent_additions"] = int(row["recent"]) if row else 0
                
                # Get database size
                row = await conn.fetchrow("""
                    SELECT pg_size_pretty(pg_total_relation_size('conceptual_dimensions')) as size
                """)
                stats["table_size"] = row["size"] if row else "Unknown"
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get PostgreSQL statistics: {str(e)}")
            return {}