"""
Configuration management for KSE Memory SDK.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import yaml
import os
from enum import Enum


class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EmbeddingModel(Enum):
    """Supported embedding models."""
    SENTENCE_BERT = "sentence-transformers/all-MiniLM-L6-v2"
    SENTENCE_BERT_LARGE = "sentence-transformers/all-mpnet-base-v2"
    OPENAI_ADA = "text-embedding-ada-002"
    OPENAI_3_SMALL = "text-embedding-3-small"
    OPENAI_3_LARGE = "text-embedding-3-large"


@dataclass
class VectorStoreConfig:
    """Vector store configuration."""
    backend: str = "pinecone"  # pinecone, weaviate, qdrant
    api_key: Optional[str] = None
    environment: Optional[str] = None
    index_name: str = "kse-products"
    dimension: int = 384
    metric: str = "cosine"
    host: Optional[str] = None
    port: Optional[int] = None
    collection_name: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.backend == "pinecone" and not self.api_key:
            self.api_key = os.getenv("PINECONE_API_KEY")
        elif self.backend == "weaviate":
            self.host = self.host or os.getenv("WEAVIATE_HOST", "localhost")
            self.port = self.port or int(os.getenv("WEAVIATE_PORT", "8080"))
        elif self.backend == "qdrant":
            self.host = self.host or os.getenv("QDRANT_HOST", "localhost")
            self.port = self.port or int(os.getenv("QDRANT_PORT", "6333"))
            self.collection_name = self.collection_name or "kse_products"


@dataclass
class GraphStoreConfig:
    """Graph store configuration."""
    backend: str = "neo4j"  # neo4j, arangodb
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: Optional[str] = None
    database: str = "neo4j"
    
    def __post_init__(self):
        """Post-initialization validation."""
        if not self.password:
            self.password = os.getenv("NEO4J_PASSWORD", "password")


@dataclass
class ConceptStoreConfig:
    """Concept store configuration."""
    backend: str = "postgresql"  # postgresql, mongodb
    host: str = "localhost"
    port: int = 5432
    database: str = "kse_concepts"
    username: str = "postgres"
    password: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization validation."""
        if not self.password:
            if self.backend == "postgresql":
                self.password = os.getenv("POSTGRES_PASSWORD", "password")
            elif self.backend == "mongodb":
                self.password = os.getenv("MONGODB_PASSWORD", "password")
                self.port = self.port if self.port != 5432 else 27017


@dataclass
class EmbeddingConfig:
    """Embedding configuration."""
    text_model: str = EmbeddingModel.SENTENCE_BERT.value
    image_model: str = "openai/clip-vit-base-patch32"
    openai_api_key: Optional[str] = None
    batch_size: int = 32
    max_length: int = 512
    normalize: bool = True
    
    def __post_init__(self):
        """Post-initialization validation."""
        if not self.openai_api_key:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")


@dataclass
class ConceptualConfig:
    """Conceptual spaces configuration."""
    dimensions: List[str] = field(default_factory=lambda: [
        "elegance", "comfort", "boldness", "modernity", "minimalism",
        "luxury", "functionality", "versatility", "seasonality", "innovation"
    ])
    auto_compute: bool = True
    llm_model: str = "gpt-4"
    llm_api_key: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization validation."""
        if not self.llm_api_key:
            self.llm_api_key = os.getenv("OPENAI_API_KEY")


@dataclass
class SearchConfig:
    """Search configuration."""
    default_limit: int = 10
    max_limit: int = 100
    hybrid_weights: Dict[str, float] = field(default_factory=lambda: {
        "embedding": 0.4,
        "conceptual": 0.3,
        "knowledge_graph": 0.3
    })
    similarity_threshold: float = 0.7
    enable_reranking: bool = True
    rerank_model: Optional[str] = None


@dataclass
class CacheConfig:
    """Cache configuration."""
    enabled: bool = True
    backend: str = "redis"  # redis, memory
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    ttl: int = 3600  # seconds
    
    def __post_init__(self):
        """Post-initialization validation."""
        if not self.password:
            self.password = os.getenv("REDIS_PASSWORD")


@dataclass
class KSEConfig:
    """Main KSE Memory configuration."""
    # Core settings
    app_name: str = "KSE Memory"
    version: str = "1.0.0"
    debug: bool = False
    log_level: LogLevel = LogLevel.WARNING
    
    # Component configurations
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    graph_store: GraphStoreConfig = field(default_factory=GraphStoreConfig)
    concept_store: ConceptStoreConfig = field(default_factory=ConceptStoreConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    conceptual: ConceptualConfig = field(default_factory=ConceptualConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    
    # Adapter settings
    adapter_config: Dict[str, Any] = field(default_factory=dict)
    
    # Performance settings
    max_workers: int = 4
    request_timeout: int = 30
    batch_processing: bool = True
    
    @classmethod
    def from_file(cls, config_path: str) -> "KSEConfig":
        """Load configuration from YAML file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KSEConfig":
        """Create configuration from dictionary."""
        config = cls()
        
        # Update basic fields
        for key, value in data.items():
            if hasattr(config, key) and not isinstance(getattr(config, key), (VectorStoreConfig, GraphStoreConfig, ConceptStoreConfig, EmbeddingConfig, ConceptualConfig, SearchConfig, CacheConfig)):
                setattr(config, key, value)
        
        # Update component configurations
        if "vector_store" in data:
            config.vector_store = VectorStoreConfig(**data["vector_store"])
        
        if "graph_store" in data:
            config.graph_store = GraphStoreConfig(**data["graph_store"])
        
        if "concept_store" in data:
            config.concept_store = ConceptStoreConfig(**data["concept_store"])
        
        if "embedding" in data:
            config.embedding = EmbeddingConfig(**data["embedding"])
        
        if "conceptual" in data:
            config.conceptual = ConceptualConfig(**data["conceptual"])
        
        if "search" in data:
            config.search = SearchConfig(**data["search"])
        
        if "cache" in data:
            config.cache = CacheConfig(**data["cache"])
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "app_name": self.app_name,
            "version": self.version,
            "debug": self.debug,
            "log_level": self.log_level.value,
            "vector_store": {
                "backend": self.vector_store.backend,
                "api_key": self.vector_store.api_key,
                "environment": self.vector_store.environment,
                "index_name": self.vector_store.index_name,
                "dimension": self.vector_store.dimension,
                "metric": self.vector_store.metric,
                "host": self.vector_store.host,
                "port": self.vector_store.port,
                "collection_name": self.vector_store.collection_name,
            },
            "graph_store": {
                "backend": self.graph_store.backend,
                "uri": self.graph_store.uri,
                "username": self.graph_store.username,
                "password": self.graph_store.password,
                "database": self.graph_store.database,
            },
            "concept_store": {
                "backend": self.concept_store.backend,
                "host": self.concept_store.host,
                "port": self.concept_store.port,
                "database": self.concept_store.database,
                "username": self.concept_store.username,
                "password": self.concept_store.password,
            },
            "embedding": {
                "text_model": self.embedding.text_model,
                "image_model": self.embedding.image_model,
                "openai_api_key": self.embedding.openai_api_key,
                "batch_size": self.embedding.batch_size,
                "max_length": self.embedding.max_length,
                "normalize": self.embedding.normalize,
            },
            "conceptual": {
                "dimensions": self.conceptual.dimensions,
                "auto_compute": self.conceptual.auto_compute,
                "llm_model": self.conceptual.llm_model,
                "llm_api_key": self.conceptual.llm_api_key,
            },
            "search": {
                "default_limit": self.search.default_limit,
                "max_limit": self.search.max_limit,
                "hybrid_weights": self.search.hybrid_weights,
                "similarity_threshold": self.search.similarity_threshold,
                "enable_reranking": self.search.enable_reranking,
                "rerank_model": self.search.rerank_model,
            },
            "cache": {
                "enabled": self.cache.enabled,
                "backend": self.cache.backend,
                "host": self.cache.host,
                "port": self.cache.port,
                "password": self.cache.password,
                "db": self.cache.db,
                "ttl": self.cache.ttl,
            },
            "adapter_config": self.adapter_config,
            "max_workers": self.max_workers,
            "request_timeout": self.request_timeout,
            "batch_processing": self.batch_processing,
        }
    
    def save_to_file(self, config_path: str):
        """Save configuration to YAML file."""
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Validate vector store
        if self.vector_store.backend == "pinecone" and not self.vector_store.api_key:
            errors.append("Pinecone API key is required")
        
        # Validate embedding config
        if self.embedding.text_model.startswith("text-embedding") and not self.embedding.openai_api_key:
            errors.append("OpenAI API key is required for OpenAI embedding models")
        
        # Validate conceptual config
        if self.conceptual.auto_compute and not self.conceptual.llm_api_key:
            errors.append("LLM API key is required for automatic conceptual dimension computation")
        
        # Validate search weights
        total_weight = sum(self.search.hybrid_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            errors.append(f"Hybrid search weights must sum to 1.0, got {total_weight}")
        
        return errors