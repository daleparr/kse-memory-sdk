"""
Storage backends for KSE Memory SDK.
"""

from typing import Any
from ..core.interfaces import VectorStoreInterface, GraphStoreInterface, ConceptStoreInterface
from ..core.config import VectorStoreConfig, GraphStoreConfig, ConceptStoreConfig
from ..exceptions import BackendError


def get_vector_store(config: VectorStoreConfig) -> VectorStoreInterface:
    """
    Factory function to get the appropriate vector store backend.

    Args:
        config: Vector store configuration

    Returns:
        Vector store instance

    Raises:
        BackendError: If backend type is not supported
    """
    backend_type = config.backend.lower()

    if backend_type == "pinecone":
        from .pinecone import PineconeBackend
        return PineconeBackend(config)
    elif backend_type == "weaviate":
        from .weaviate import WeaviateBackend
        return WeaviateBackend(config)
    elif backend_type == "qdrant":
        from .qdrant import QdrantBackend
        return QdrantBackend(config)
    elif backend_type == "chromadb":
        from .chromadb import ChromaDBBackend
        return ChromaDBBackend(config)
    elif backend_type == "milvus":
        from .milvus import MilvusBackend
        return MilvusBackend(config)
    else:
        raise BackendError(f"Unsupported vector store backend: {backend_type}", "vector_store")


def get_graph_store(config: GraphStoreConfig) -> GraphStoreInterface:
    """
    Factory function to get the appropriate graph store backend.
    
    Args:
        config: Graph store configuration
        
    Returns:
        Graph store instance
        
    Raises:
        BackendError: If backend type is not supported
    """
    backend_type = config.backend.lower()
    
    if backend_type == "neo4j":
        from .neo4j import Neo4jBackend
        return Neo4jBackend(config)
    elif backend_type == "arangodb":
        from .arangodb import ArangoDBBackend
        return ArangoDBBackend(config)
    else:
        raise BackendError(f"Unsupported graph store backend: {backend_type}", "graph_store")


def get_concept_store(config: ConceptStoreConfig) -> ConceptStoreInterface:
    """
    Factory function to get the appropriate concept store backend.
    
    Args:
        config: Concept store configuration
        
    Returns:
        Concept store instance
        
    Raises:
        BackendError: If backend type is not supported
    """
    backend_type = config.backend.lower()
    
    if backend_type == "postgresql":
        from .postgresql import PostgreSQLBackend
        return PostgreSQLBackend(config)
    elif backend_type == "mongodb":
        from .mongodb import MongoDBBackend
        return MongoDBBackend(config)
    else:
        raise BackendError(f"Unsupported concept store backend: {backend_type}", "concept_store")


# Import main backend classes for direct access
from .pinecone import PineconeBackend
from .weaviate import WeaviateBackend
from .qdrant import QdrantBackend
from .chromadb import ChromaDBBackend
from .milvus import MilvusBackend
from .neo4j import Neo4jBackend
from .arangodb import ArangoDBBackend
from .postgresql import PostgreSQLBackend
from .mongodb import MongoDBBackend

__all__ = [
    "get_vector_store",
    "get_graph_store",
    "get_concept_store",
    "PineconeBackend",
    "WeaviateBackend",
    "QdrantBackend",
    "ChromaDBBackend",
    "MilvusBackend",
    "Neo4jBackend",
    "ArangoDBBackend",
    "PostgreSQLBackend",
    "MongoDBBackend",
]