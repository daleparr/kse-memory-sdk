"""
Core KSE Memory components and interfaces.
"""

from .memory import KSEMemory
from .config import KSEConfig
from .models import (
    Entity,
    ConceptualSpace,
    Product,  # Deprecated - use Entity instead
    SearchQuery,
    SearchResult,
    SearchType,
    ConceptualDimensions,  # Deprecated - use ConceptualSpace instead
    KnowledgeGraph,
    EmbeddingVector,
)
from .interfaces import (
    AdapterInterface,
    VectorStoreInterface,
    GraphStoreInterface,
    ConceptStoreInterface,
)

__all__ = [
    "KSEMemory",
    "KSEConfig",
    "Entity",
    "ConceptualSpace",
    "Product",  # Deprecated - use Entity instead
    "SearchQuery",
    "SearchResult",
    "SearchType",
    "ConceptualDimensions",  # Deprecated - use ConceptualSpace instead
    "KnowledgeGraph",
    "EmbeddingVector",
    "AdapterInterface",
    "VectorStoreInterface",
    "GraphStoreInterface",
    "ConceptStoreInterface",
]