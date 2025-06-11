"""
Core data models for KSE Memory SDK.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import uuid


class SearchType(Enum):
    """Types of search operations."""
    SEMANTIC = "semantic"
    CONCEPTUAL = "conceptual"
    HYBRID = "hybrid"
    KNOWLEDGE_GRAPH = "knowledge_graph"


class ConceptualDimension(Enum):
    """Standard conceptual dimensions for product analysis."""
    ELEGANCE = "elegance"
    COMFORT = "comfort"
    BOLDNESS = "boldness"
    MODERNITY = "modernity"
    MINIMALISM = "minimalism"
    LUXURY = "luxury"
    FUNCTIONALITY = "functionality"
    VERSATILITY = "versatility"
    SEASONALITY = "seasonality"
    INNOVATION = "innovation"


@dataclass
class ConceptualDimensions:
    """Conceptual space coordinates for a product."""
    elegance: float = 0.0
    comfort: float = 0.0
    boldness: float = 0.0
    modernity: float = 0.0
    minimalism: float = 0.0
    luxury: float = 0.0
    functionality: float = 0.0
    versatility: float = 0.0
    seasonality: float = 0.0
    innovation: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary representation."""
        return {
            "elegance": self.elegance,
            "comfort": self.comfort,
            "boldness": self.boldness,
            "modernity": self.modernity,
            "minimalism": self.minimalism,
            "luxury": self.luxury,
            "functionality": self.functionality,
            "versatility": self.versatility,
            "seasonality": self.seasonality,
            "innovation": self.innovation,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "ConceptualDimensions":
        """Create from dictionary representation."""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


@dataclass
class EmbeddingVector:
    """Neural embedding representation."""
    vector: List[float]
    model: str
    dimension: int
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate embedding vector."""
        if len(self.vector) != self.dimension:
            raise ValueError(f"Vector length {len(self.vector)} doesn't match dimension {self.dimension}")


@dataclass
class Product:
    """Core product representation in KSE Memory."""
    id: str
    title: str
    description: str
    price: Optional[float] = None
    currency: Optional[str] = None
    category: Optional[str] = None
    brand: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)
    variants: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # KSE-specific fields
    conceptual_dimensions: Optional[ConceptualDimensions] = None
    text_embedding: Optional[EmbeddingVector] = None
    image_embedding: Optional[EmbeddingVector] = None
    knowledge_graph_id: Optional[str] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Post-initialization validation."""
        if not self.id:
            self.id = str(uuid.uuid4())
        
        # Ensure tags are strings
        self.tags = [str(tag) for tag in self.tags]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "price": self.price,
            "currency": self.currency,
            "category": self.category,
            "brand": self.brand,
            "tags": self.tags,
            "images": self.images,
            "variants": self.variants,
            "metadata": self.metadata,
            "knowledge_graph_id": self.knowledge_graph_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        
        if self.conceptual_dimensions:
            result["conceptual_dimensions"] = self.conceptual_dimensions.to_dict()
        
        if self.text_embedding:
            result["text_embedding"] = {
                "vector": self.text_embedding.vector,
                "model": self.text_embedding.model,
                "dimension": self.text_embedding.dimension,
                "created_at": self.text_embedding.created_at.isoformat(),
            }
        
        if self.image_embedding:
            result["image_embedding"] = {
                "vector": self.image_embedding.vector,
                "model": self.image_embedding.model,
                "dimension": self.image_embedding.dimension,
                "created_at": self.image_embedding.created_at.isoformat(),
            }
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Product":
        """Create from dictionary representation."""
        # Handle datetime fields
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        
        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
        
        # Handle conceptual dimensions
        conceptual_dims = None
        if "conceptual_dimensions" in data:
            conceptual_dims = ConceptualDimensions.from_dict(data["conceptual_dimensions"])
        
        # Handle embeddings
        text_embedding = None
        if "text_embedding" in data:
            emb_data = data["text_embedding"]
            emb_created_at = emb_data.get("created_at")
            if isinstance(emb_created_at, str):
                emb_created_at = datetime.fromisoformat(emb_created_at.replace('Z', '+00:00'))
            text_embedding = EmbeddingVector(
                vector=emb_data["vector"],
                model=emb_data["model"],
                dimension=emb_data["dimension"],
                created_at=emb_created_at or datetime.utcnow(),
            )
        
        image_embedding = None
        if "image_embedding" in data:
            emb_data = data["image_embedding"]
            emb_created_at = emb_data.get("created_at")
            if isinstance(emb_created_at, str):
                emb_created_at = datetime.fromisoformat(emb_created_at.replace('Z', '+00:00'))
            image_embedding = EmbeddingVector(
                vector=emb_data["vector"],
                model=emb_data["model"],
                dimension=emb_data["dimension"],
                created_at=emb_created_at or datetime.utcnow(),
            )
        
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            title=data["title"],
            description=data["description"],
            price=data.get("price"),
            currency=data.get("currency"),
            category=data.get("category"),
            brand=data.get("brand"),
            tags=data.get("tags", []),
            images=data.get("images", []),
            variants=data.get("variants", []),
            metadata=data.get("metadata", {}),
            conceptual_dimensions=conceptual_dims,
            text_embedding=text_embedding,
            image_embedding=image_embedding,
            knowledge_graph_id=data.get("knowledge_graph_id"),
            created_at=created_at or datetime.utcnow(),
            updated_at=updated_at or datetime.utcnow(),
        )


@dataclass
class SearchQuery:
    """Search query representation."""
    query: str
    search_type: SearchType = SearchType.HYBRID
    filters: Dict[str, Any] = field(default_factory=dict)
    limit: int = 10
    offset: int = 0
    include_conceptual: bool = True
    include_embeddings: bool = True
    include_knowledge_graph: bool = True
    conceptual_weights: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "query": self.query,
            "search_type": self.search_type.value,
            "filters": self.filters,
            "limit": self.limit,
            "offset": self.offset,
            "include_conceptual": self.include_conceptual,
            "include_embeddings": self.include_embeddings,
            "include_knowledge_graph": self.include_knowledge_graph,
            "conceptual_weights": self.conceptual_weights,
        }


@dataclass
class SearchResult:
    """Search result representation."""
    product: Product
    score: float
    explanation: Optional[str] = None
    conceptual_similarity: Optional[float] = None
    embedding_similarity: Optional[float] = None
    knowledge_graph_similarity: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "product": self.product.to_dict(),
            "score": self.score,
            "explanation": self.explanation,
            "conceptual_similarity": self.conceptual_similarity,
            "embedding_similarity": self.embedding_similarity,
            "knowledge_graph_similarity": self.knowledge_graph_similarity,
        }


@dataclass
class KnowledgeGraph:
    """Knowledge graph representation."""
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_node(self, node_id: str, node_type: str, properties: Dict[str, Any] = None):
        """Add a node to the knowledge graph."""
        node = {
            "id": node_id,
            "type": node_type,
            "properties": properties or {},
        }
        self.nodes.append(node)
    
    def add_edge(self, source_id: str, target_id: str, relationship: str, properties: Dict[str, Any] = None):
        """Add an edge to the knowledge graph."""
        edge = {
            "source": source_id,
            "target": target_id,
            "relationship": relationship,
            "properties": properties or {},
        }
        self.edges.append(edge)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "metadata": self.metadata,
        }