"""
KSE Memory SDK - Pure-Play Knowledge Space Embeddings for Any Application

A hybrid AI memory system combining Knowledge Graphs, Conceptual Spaces, 
and Neural Embeddings for intelligent product understanding and discovery.
"""

__version__ = "1.1.0"
__author__ = "KSE System"
__email__ = "dev@kse-system.com"

from .core import (
    KSEMemory,
    KSEConfig,
    Product,
    SearchQuery,
    SearchResult,
    SearchType,
    ConceptualDimensions,
)

from .adapters import (
    ShopifyAdapter,
    WooCommerceAdapter,
    GenericAdapter,
)

from .backends import (
    PineconeBackend,
    WeaviateBackend,
    QdrantBackend,
    ChromaDBBackend,
    MilvusBackend,
    Neo4jBackend,
    ArangoDBBackend,
    PostgreSQLBackend,
    MongoDBBackend,
)

from .exceptions import (
    KSEError,
    ConfigurationError,
    AdapterError,
    BackendError,
    SearchError,
)

# Main SDK interface
__all__ = [
    # Core classes
    "KSEMemory",
    "KSEConfig",
    "Product",
    "SearchQuery",
    "SearchResult",
    "SearchType",
    "ConceptualDimensions",
    
    # Adapters
    "ShopifyAdapter",
    "WooCommerceAdapter",
    "GenericAdapter",
    
    # Backends
    "PineconeBackend",
    "WeaviateBackend",
    "QdrantBackend",
    "ChromaDBBackend",
    "MilvusBackend",
    "Neo4jBackend",
    "ArangoDBBackend",
    "PostgreSQLBackend",
    "MongoDBBackend",
    
    # Exceptions
    "KSEError",
    "ConfigurationError",
    "AdapterError",
    "BackendError",
    "SearchError",
]

# SDK metadata
SDK_INFO = {
    "name": "kse-memory",
    "version": __version__,
    "description": "Pure-Play Knowledge Space Embeddings for Any Application",
    "author": __author__,
    "email": __email__,
    "license": "MIT",
    "python_requires": ">=3.9",
    "homepage": "https://github.com/kse-system/kse-memory-sdk",
    "documentation": "https://docs.kse-memory.com",
}

def get_version() -> str:
    """Get the current SDK version."""
    return __version__

def get_info() -> dict:
    """Get SDK metadata information."""
    return SDK_INFO.copy()