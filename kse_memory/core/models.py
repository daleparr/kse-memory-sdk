"""
Core data models for KSE Memory SDK.

Universal AI memory system for any domain - healthcare, finance, enterprise, research, and more.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import uuid
import warnings


class SearchType(Enum):
    """Types of search operations."""
    SEMANTIC = "semantic"
    CONCEPTUAL = "conceptual"
    HYBRID = "hybrid"
    KNOWLEDGE_GRAPH = "knowledge_graph"


# Domain-specific conceptual dimension templates
DOMAIN_DIMENSIONS = {
    "healthcare": {
        "urgency": "How urgent or time-sensitive the entity is",
        "complexity": "Technical or procedural complexity level",
        "invasiveness": "How invasive or disruptive the entity is",
        "cost_effectiveness": "Economic efficiency and value",
        "safety": "Safety profile and risk level",
        "accessibility": "How accessible or available the entity is"
    },
    "finance": {
        "risk": "Risk level and volatility",
        "liquidity": "How easily convertible to cash",
        "growth_potential": "Expected growth and returns",
        "stability": "Consistency and reliability",
        "complexity": "Structural and operational complexity",
        "regulatory_compliance": "Adherence to regulations"
    },
    "real_estate": {
        "location_quality": "Desirability and convenience of location",
        "condition": "Physical state and maintenance level",
        "investment_potential": "Expected appreciation and returns",
        "size_efficiency": "Space utilization and layout quality",
        "amenities": "Available features and facilities",
        "market_demand": "Current and projected demand"
    },
    "enterprise": {
        "importance": "Strategic importance and priority",
        "complexity": "Technical or operational complexity",
        "urgency": "Time sensitivity and deadlines",
        "impact": "Potential business impact",
        "resource_intensity": "Required resources and effort",
        "stakeholder_value": "Value to stakeholders"
    },
    "research": {
        "novelty": "Originality and innovation level",
        "impact": "Potential scientific or practical impact",
        "rigor": "Methodological quality and reliability",
        "accessibility": "Ease of understanding and application",
        "reproducibility": "Ability to replicate results",
        "interdisciplinary": "Cross-domain applicability"
    },
    "retail": {
        "elegance": "Aesthetic appeal and sophistication",
        "comfort": "User comfort and ease of use",
        "boldness": "Distinctive and attention-grabbing qualities",
        "modernity": "Contemporary style and innovation",
        "minimalism": "Simplicity and clean design",
        "luxury": "Premium quality and exclusivity",
        "functionality": "Practical utility and performance",
        "versatility": "Adaptability to different uses",
        "seasonality": "Time-specific relevance",
        "innovation": "Technological or design innovation"
    }
}


@dataclass
class ConceptualSpace:
    """
    Flexible conceptual space for any domain.
    
    Replaces the hardcoded retail-specific ConceptualDimensions with a dynamic system
    that can adapt to any industry or use case.
    """
    dimensions: Dict[str, float] = field(default_factory=dict)
    domain: Optional[str] = None
    dimension_descriptions: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize with domain-specific dimensions if domain is specified."""
        if self.domain and self.domain in DOMAIN_DIMENSIONS and not self.dimensions:
            domain_dims = DOMAIN_DIMENSIONS[self.domain]
            self.dimensions = {dim: 0.0 for dim in domain_dims.keys()}
            self.dimension_descriptions = domain_dims.copy()
    
    @classmethod
    def create_for_domain(cls, domain: str, custom_dimensions: Optional[Dict[str, str]] = None) -> "ConceptualSpace":
        """
        Create a conceptual space for a specific domain.
        
        Args:
            domain: Domain name (healthcare, finance, retail, etc.)
            custom_dimensions: Optional custom dimensions to override defaults
            
        Returns:
            ConceptualSpace configured for the domain
        """
        if domain in DOMAIN_DIMENSIONS:
            dimensions_def = DOMAIN_DIMENSIONS[domain].copy()
            if custom_dimensions:
                dimensions_def.update(custom_dimensions)
        else:
            dimensions_def = custom_dimensions or {}
        
        return cls(
            dimensions={dim: 0.0 for dim in dimensions_def.keys()},
            domain=domain,
            dimension_descriptions=dimensions_def
        )
    
    @classmethod
    def create_custom(cls, dimensions: Dict[str, str], domain: Optional[str] = None) -> "ConceptualSpace":
        """
        Create a custom conceptual space with arbitrary dimensions.
        
        Args:
            dimensions: Dict mapping dimension names to descriptions
            domain: Optional domain identifier
            
        Returns:
            ConceptualSpace with custom dimensions
        """
        return cls(
            dimensions={dim: 0.0 for dim in dimensions.keys()},
            domain=domain,
            dimension_descriptions=dimensions
        )
    
    def set_dimension(self, dimension: str, value: float, description: Optional[str] = None):
        """Set a dimension value and optionally its description."""
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Dimension value must be between 0.0 and 1.0, got {value}")
        
        self.dimensions[dimension] = value
        if description:
            self.dimension_descriptions[dimension] = description
    
    def get_dimension(self, dimension: str) -> Optional[float]:
        """Get a dimension value."""
        return self.dimensions.get(dimension)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "dimensions": self.dimensions,
            "domain": self.domain,
            "dimension_descriptions": self.dimension_descriptions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConceptualSpace":
        """Create from dictionary representation."""
        return cls(
            dimensions=data.get("dimensions", {}),
            domain=data.get("domain"),
            dimension_descriptions=data.get("dimension_descriptions", {})
        )


# Backward compatibility - deprecated retail-specific class
@dataclass
class ConceptualDimensions:
    """
    DEPRECATED: Use ConceptualSpace instead.
    
    Legacy retail-specific conceptual dimensions. This class is maintained for
    backward compatibility but will be removed in v3.0.0.
    """
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
    
    def __post_init__(self):
        """Issue deprecation warning."""
        warnings.warn(
            "ConceptualDimensions is deprecated and will be removed in v3.0.0. "
            "Use ConceptualSpace.create_for_domain('retail') instead.",
            DeprecationWarning,
            stacklevel=2
        )
    
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
    
    def to_conceptual_space(self) -> ConceptualSpace:
        """Convert to new ConceptualSpace format."""
        space = ConceptualSpace.create_for_domain("retail")
        space.dimensions.update(self.to_dict())
        return space


# Backward compatibility enum - deprecated
class ConceptualDimension(Enum):
    """
    DEPRECATED: Standard conceptual dimensions for product analysis.
    
    This enum is deprecated in favor of the flexible ConceptualSpace system.
    Will be removed in v3.0.0.
    """
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
class Entity:
    """
    Universal entity representation in KSE Memory.
    
    This replaces the retail-specific Product class with a flexible entity model
    that can represent any type of data across domains:
    - Healthcare: Patients, treatments, medications, procedures
    - Finance: Assets, transactions, portfolios, instruments
    - Real Estate: Properties, listings, locations, amenities
    - Enterprise: Documents, projects, resources, processes
    - Research: Papers, datasets, experiments, findings
    - Retail: Products, services, brands, categories
    """
    title: str
    description: str
    id: Optional[str] = None
    entity_type: Optional[str] = None  # e.g., "patient", "asset", "property", "document", "product"
    category: Optional[str] = None
    source: Optional[str] = None  # Replaces "brand" - could be provider, institution, etc.
    tags: List[str] = field(default_factory=list)
    media: List[str] = field(default_factory=list)  # Replaces "images" - could be images, documents, etc.
    variations: List[Dict[str, Any]] = field(default_factory=list)  # Replaces "variants" - more generic
    metadata: Dict[str, Any] = field(default_factory=dict)  # Flexible storage for domain-specific fields
    
    # KSE-specific fields
    conceptual_space: Optional[ConceptualSpace] = None  # New flexible conceptual representation
    text_embedding: Optional[EmbeddingVector] = None
    media_embedding: Optional[EmbeddingVector] = None  # Replaces "image_embedding" - more generic
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
    
    def set_domain_metadata(self, domain: str, **kwargs):
        """
        Set domain-specific metadata fields.
        
        Examples:
        - Healthcare: entity.set_domain_metadata("healthcare", patient_id="12345", urgency="high")
        - Finance: entity.set_domain_metadata("finance", price=100.50, currency="USD", risk_level="medium")
        - Real Estate: entity.set_domain_metadata("real_estate", price=500000, bedrooms=3, location="downtown")
        """
        if "domain" not in self.metadata:
            self.metadata["domain"] = {}
        self.metadata["domain"][domain] = kwargs
    
    def get_domain_metadata(self, domain: str) -> Dict[str, Any]:
        """Get domain-specific metadata."""
        return self.metadata.get("domain", {}).get(domain, {})
    
    # Convenience methods for common retail fields (backward compatibility)
    @property
    def price(self) -> Optional[float]:
        """Get price from retail domain metadata."""
        return self.get_domain_metadata("retail").get("price")
    
    @price.setter
    def price(self, value: Optional[float]):
        """Set price in retail domain metadata."""
        if value is not None:
            self.set_domain_metadata("retail", price=value)
    
    @property
    def currency(self) -> Optional[str]:
        """Get currency from retail domain metadata."""
        return self.get_domain_metadata("retail").get("currency")
    
    @currency.setter
    def currency(self, value: Optional[str]):
        """Set currency in retail domain metadata."""
        if value is not None:
            self.set_domain_metadata("retail", currency=value)
    
    @property
    def brand(self) -> Optional[str]:
        """Get brand (alias for source) for backward compatibility."""
        return self.source
    
    @brand.setter
    def brand(self, value: Optional[str]):
        """Set brand (alias for source) for backward compatibility."""
        self.source = value
    
    @property
    def images(self) -> List[str]:
        """Get images (alias for media) for backward compatibility."""
        return self.media
    
    @images.setter
    def images(self, value: List[str]):
        """Set images (alias for media) for backward compatibility."""
        self.media = value
    
    @property
    def variants(self) -> List[Dict[str, Any]]:
        """Get variants (alias for variations) for backward compatibility."""
        return self.variations
    
    @variants.setter
    def variants(self, value: List[Dict[str, Any]]):
        """Set variants (alias for variations) for backward compatibility."""
        self.variations = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "entity_type": self.entity_type,
            "category": self.category,
            "source": self.source,
            "tags": self.tags,
            "media": self.media,
            "variations": self.variations,
            "metadata": self.metadata,
            "knowledge_graph_id": self.knowledge_graph_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        
        if self.conceptual_space:
            result["conceptual_space"] = self.conceptual_space.to_dict()
        
        if self.text_embedding:
            result["text_embedding"] = {
                "vector": self.text_embedding.vector,
                "model": self.text_embedding.model,
                "dimension": self.text_embedding.dimension,
                "created_at": self.text_embedding.created_at.isoformat(),
            }
        
        if self.media_embedding:
            result["media_embedding"] = {
                "vector": self.media_embedding.vector,
                "model": self.media_embedding.model,
                "dimension": self.media_embedding.dimension,
                "created_at": self.media_embedding.created_at.isoformat(),
            }
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        """Create from dictionary representation."""
        # Handle datetime fields
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        
        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
        
        # Handle conceptual space (new format) or conceptual dimensions (legacy)
        conceptual_space = None
        if "conceptual_space" in data:
            conceptual_space = ConceptualSpace.from_dict(data["conceptual_space"])
        elif "conceptual_dimensions" in data:
            # Legacy support - convert old format to new
            legacy_dims = ConceptualDimensions.from_dict(data["conceptual_dimensions"])
            conceptual_space = legacy_dims.to_conceptual_space()
        
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
        
        # Handle media embedding (new) or image embedding (legacy)
        media_embedding = None
        embedding_key = "media_embedding" if "media_embedding" in data else "image_embedding"
        if embedding_key in data:
            emb_data = data[embedding_key]
            emb_created_at = emb_data.get("created_at")
            if isinstance(emb_created_at, str):
                emb_created_at = datetime.fromisoformat(emb_created_at.replace('Z', '+00:00'))
            media_embedding = EmbeddingVector(
                vector=emb_data["vector"],
                model=emb_data["model"],
                dimension=emb_data["dimension"],
                created_at=emb_created_at or datetime.utcnow(),
            )
        
        # Handle legacy field mappings
        entity_type = data.get("entity_type")
        source = data.get("source") or data.get("brand")  # brand -> source mapping
        media = data.get("media") or data.get("images", [])  # images -> media mapping
        variations = data.get("variations") or data.get("variants", [])  # variants -> variations mapping
        
        # Handle legacy price/currency fields
        metadata = data.get("metadata", {}).copy()
        if "price" in data or "currency" in data:
            if "domain" not in metadata:
                metadata["domain"] = {}
            if "retail" not in metadata["domain"]:
                metadata["domain"]["retail"] = {}
            if "price" in data:
                metadata["domain"]["retail"]["price"] = data["price"]
            if "currency" in data:
                metadata["domain"]["retail"]["currency"] = data["currency"]
        
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            title=data["title"],
            description=data["description"],
            entity_type=entity_type,
            category=data.get("category"),
            source=source,
            tags=data.get("tags", []),
            media=media,
            variations=variations,
            metadata=metadata,
            conceptual_space=conceptual_space,
            text_embedding=text_embedding,
            media_embedding=media_embedding,
            knowledge_graph_id=data.get("knowledge_graph_id"),
            created_at=created_at or datetime.utcnow(),
            updated_at=updated_at or datetime.utcnow(),
        )
    
    @classmethod
    def create_healthcare_entity(
        cls,
        title: str,
        description: str,
        entity_type: str = "patient",
        **kwargs
    ) -> "Entity":
        """Create a healthcare entity with appropriate conceptual space."""
        entity = cls(
            title=title,
            description=description,
            entity_type=entity_type,
            conceptual_space=ConceptualSpace.create_for_domain("healthcare"),
            **kwargs
        )
        return entity
    
    @classmethod
    def create_finance_entity(
        cls,
        title: str,
        description: str,
        entity_type: str = "asset",
        **kwargs
    ) -> "Entity":
        """Create a finance entity with appropriate conceptual space."""
        entity = cls(
            title=title,
            description=description,
            entity_type=entity_type,
            conceptual_space=ConceptualSpace.create_for_domain("finance"),
            **kwargs
        )
        return entity
    
    @classmethod
    def create_real_estate_entity(
        cls,
        title: str,
        description: str,
        entity_type: str = "property",
        **kwargs
    ) -> "Entity":
        """Create a real estate entity with appropriate conceptual space."""
        entity = cls(
            title=title,
            description=description,
            entity_type=entity_type,
            conceptual_space=ConceptualSpace.create_for_domain("real_estate"),
            **kwargs
        )
        return entity
    
    @classmethod
    def create_enterprise_entity(
        cls,
        title: str,
        description: str,
        entity_type: str = "document",
        **kwargs
    ) -> "Entity":
        """Create an enterprise entity with appropriate conceptual space."""
        entity = cls(
            title=title,
            description=description,
            entity_type=entity_type,
            conceptual_space=ConceptualSpace.create_for_domain("enterprise"),
            **kwargs
        )
        return entity
    
    @classmethod
    def create_research_entity(
        cls,
        title: str,
        description: str,
        entity_type: str = "research_paper",
        **kwargs
    ) -> "Entity":
        """Create a research entity with appropriate conceptual space."""
        entity = cls(
            title=title,
            description=description,
            entity_type=entity_type,
            conceptual_space=ConceptualSpace.create_for_domain("research"),
            **kwargs
        )
        return entity
    
    @classmethod
    def create_retail_entity(
        cls,
        title: str,
        description: str,
        entity_type: str = "product",
        **kwargs
    ) -> "Entity":
        """Create a retail entity with appropriate conceptual space."""
        entity = cls(
            title=title,
            description=description,
            entity_type=entity_type,
            conceptual_space=ConceptualSpace.create_for_domain("retail"),
            **kwargs
        )
        return entity


# Backward compatibility alias
@dataclass
class Product(Entity):
    """
    DEPRECATED: Use Entity instead.
    
    Legacy product class maintained for backward compatibility.
    Will be removed in v3.0.0.
    """
    
    def __post_init__(self):
        """Issue deprecation warning and set retail defaults."""
        warnings.warn(
            "Product class is deprecated and will be removed in v3.0.0. "
            "Use Entity.create_retail_entity() or Entity with entity_type='product' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Set retail-specific defaults
        if not self.entity_type:
            self.entity_type = "product"
        if not self.conceptual_space:
            self.conceptual_space = ConceptualSpace.create_for_domain("retail")
        
        super().__post_init__()
    
    # Override conceptual_dimensions property for backward compatibility
    @property
    def conceptual_dimensions(self) -> Optional[ConceptualDimensions]:
        """Get conceptual dimensions in legacy format."""
        if not self.conceptual_space or self.conceptual_space.domain != "retail":
            return None
        
        # Convert ConceptualSpace back to ConceptualDimensions
        dims = ConceptualDimensions()
        for dim_name, value in self.conceptual_space.dimensions.items():
            if hasattr(dims, dim_name):
                setattr(dims, dim_name, value)
        return dims
    
    @conceptual_dimensions.setter
    def conceptual_dimensions(self, value: Optional[ConceptualDimensions]):
        """Set conceptual dimensions from legacy format."""
        if value is None:
            self.conceptual_space = None
        else:
            self.conceptual_space = value.to_conceptual_space()


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
    entity: Entity
    score: float
    explanation: Optional[str] = None
    conceptual_similarity: Optional[float] = None
    embedding_similarity: Optional[float] = None
    knowledge_graph_similarity: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "entity": self.entity.to_dict(),
            "score": self.score,
            "explanation": self.explanation,
            "conceptual_similarity": self.conceptual_similarity,
            "embedding_similarity": self.embedding_similarity,
            "knowledge_graph_similarity": self.knowledge_graph_similarity,
        }
    
    # Backward compatibility property
    @property
    def product(self) -> Entity:
        """
        DEPRECATED: Use 'entity' instead.
        
        Backward compatibility property for legacy code.
        Will be removed in v3.0.0.
        """
        warnings.warn(
            "SearchResult.product is deprecated and will be removed in v3.0.0. "
            "Use SearchResult.entity instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.entity
    
    @product.setter
    def product(self, value: Entity):
        """Backward compatibility setter for product property."""
        warnings.warn(
            "SearchResult.product is deprecated and will be removed in v3.0.0. "
            "Use SearchResult.entity instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.entity = value


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