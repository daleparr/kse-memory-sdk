"""
Production readiness validator for KSE Memory SDK.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from .config import KSEConfig
from ..exceptions import ConfigurationError


@dataclass
class ValidationResult:
    """Result of production validation check."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    recommendations: List[str]


class ProductionValidator:
    """Validates KSE Memory configuration for production readiness."""
    
    def __init__(self):
        """Initialize production validator."""
        self.required_backends = {
            "vector_store": ["pinecone", "weaviate", "qdrant", "chromadb", "milvus"],
            "graph_store": ["neo4j", "arangodb"],
            "concept_store": ["postgresql", "mongodb"]
        }
    
    def validate_config(self, config: KSEConfig) -> ValidationResult:
        """
        Validate configuration for production readiness.
        
        Args:
            config: KSE configuration to validate
            
        Returns:
            Validation result with errors, warnings, and recommendations
        """
        errors = []
        warnings = []
        recommendations = []
        
        # Check debug mode
        if config.debug:
            warnings.append("Debug mode is enabled - should be disabled in production")
            recommendations.append("Set debug=False for production deployment")
        
        # Validate vector store configuration
        vector_errors, vector_warnings, vector_recs = self._validate_vector_store(config)
        errors.extend(vector_errors)
        warnings.extend(vector_warnings)
        recommendations.extend(vector_recs)
        
        # Validate graph store configuration
        graph_errors, graph_warnings, graph_recs = self._validate_graph_store(config)
        errors.extend(graph_errors)
        warnings.extend(graph_warnings)
        recommendations.extend(graph_recs)
        
        # Validate concept store configuration
        concept_errors, concept_warnings, concept_recs = self._validate_concept_store(config)
        errors.extend(concept_errors)
        warnings.extend(concept_warnings)
        recommendations.extend(concept_recs)
        
        # Validate embedding configuration
        embed_errors, embed_warnings, embed_recs = self._validate_embedding_config(config)
        errors.extend(embed_errors)
        warnings.extend(embed_warnings)
        recommendations.extend(embed_recs)
        
        # Validate security settings
        sec_errors, sec_warnings, sec_recs = self._validate_security(config)
        errors.extend(sec_errors)
        warnings.extend(sec_warnings)
        recommendations.extend(sec_recs)
        
        # Validate performance settings
        perf_warnings, perf_recs = self._validate_performance(config)
        warnings.extend(perf_warnings)
        recommendations.extend(perf_recs)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            recommendations=recommendations
        )
    
    def _validate_vector_store(self, config: KSEConfig) -> tuple:
        """Validate vector store configuration."""
        errors = []
        warnings = []
        recommendations = []
        
        backend = config.vector_store.backend.lower()
        
        # Check if backend is production-ready
        if backend not in self.required_backends["vector_store"]:
            errors.append(f"Vector store backend '{backend}' is not supported for production")
            recommendations.append(f"Use one of: {', '.join(self.required_backends['vector_store'])}")
        
        # Backend-specific validation
        if backend == "pinecone":
            if not config.vector_store.api_key:
                errors.append("Pinecone API key is required for production")
                recommendations.append("Set PINECONE_API_KEY environment variable or provide api_key in config")
            
            if not config.vector_store.environment:
                warnings.append("Pinecone environment not specified")
                recommendations.append("Specify Pinecone environment for better performance")
        
        elif backend == "weaviate":
            if not config.vector_store.host:
                errors.append("Weaviate host is required for production")
                recommendations.append("Provide Weaviate host URL in configuration")
        
        elif backend == "qdrant":
            if not config.vector_store.host:
                errors.append("Qdrant host is required for production")
                recommendations.append("Provide Qdrant host URL in configuration")
        
        # Check dimension settings
        if config.vector_store.dimension < 384:
            warnings.append("Vector dimension is quite low - may impact search quality")
            recommendations.append("Consider using at least 384 dimensions for better semantic understanding")
        
        return errors, warnings, recommendations
    
    def _validate_graph_store(self, config: KSEConfig) -> tuple:
        """Validate graph store configuration."""
        errors = []
        warnings = []
        recommendations = []
        
        backend = config.graph_store.backend.lower()
        
        # Check if backend is production-ready
        if backend not in self.required_backends["graph_store"]:
            errors.append(f"Graph store backend '{backend}' is not supported for production")
            recommendations.append(f"Use one of: {', '.join(self.required_backends['graph_store'])}")
        
        # Backend-specific validation
        if backend == "neo4j":
            if not config.graph_store.uri:
                errors.append("Neo4j URI is required for production")
                recommendations.append("Provide Neo4j connection URI")
            
            if not config.graph_store.username or not config.graph_store.password:
                errors.append("Neo4j credentials are required for production")
                recommendations.append("Provide Neo4j username and password")
        
        elif backend == "arangodb":
            if not config.graph_store.uri:
                errors.append("ArangoDB URI is required for production")
                recommendations.append("Provide ArangoDB connection URI")
            
            if not config.graph_store.database:
                errors.append("ArangoDB database name is required")
                recommendations.append("Specify ArangoDB database name")
        
        return errors, warnings, recommendations
    
    def _validate_concept_store(self, config: KSEConfig) -> tuple:
        """Validate concept store configuration."""
        errors = []
        warnings = []
        recommendations = []
        
        backend = config.concept_store.backend.lower()
        
        # Check if backend is production-ready
        if backend not in self.required_backends["concept_store"]:
            errors.append(f"Concept store backend '{backend}' is not supported for production")
            recommendations.append(f"Use one of: {', '.join(self.required_backends['concept_store'])}")
        
        # Backend-specific validation
        if backend == "postgresql":
            if not config.concept_store.uri:
                errors.append("PostgreSQL URI is required for production")
                recommendations.append("Provide PostgreSQL connection URI")
        
        elif backend == "mongodb":
            if not config.concept_store.uri:
                errors.append("MongoDB URI is required for production")
                recommendations.append("Provide MongoDB connection URI")
            
            if not config.concept_store.database:
                errors.append("MongoDB database name is required")
                recommendations.append("Specify MongoDB database name")
        
        return errors, warnings, recommendations
    
    def _validate_embedding_config(self, config: KSEConfig) -> tuple:
        """Validate embedding configuration."""
        errors = []
        warnings = []
        recommendations = []
        
        # Check for API keys if using OpenAI models
        if "openai" in config.embedding.text_model.lower():
            if not config.embedding.openai_api_key:
                errors.append("OpenAI API key is required for OpenAI embedding models")
                recommendations.append("Set OPENAI_API_KEY environment variable or provide in config")
        
        # Check batch size for production
        if config.embedding.batch_size > 100:
            warnings.append("Large embedding batch size may cause memory issues")
            recommendations.append("Consider reducing batch_size to 32-64 for production stability")
        
        if config.embedding.batch_size < 8:
            warnings.append("Small embedding batch size may be inefficient")
            recommendations.append("Consider increasing batch_size to 16-32 for better throughput")
        
        return errors, warnings, recommendations
    
    def _validate_security(self, config: KSEConfig) -> tuple:
        """Validate security settings."""
        errors = []
        warnings = []
        recommendations = []
        
        # Check for hardcoded credentials
        if hasattr(config.vector_store, 'api_key') and config.vector_store.api_key:
            if config.vector_store.api_key.startswith(('test', 'demo', 'sample')):
                errors.append("Test/demo API key detected in vector store configuration")
                recommendations.append("Use production API keys and store them securely")
        
        # Check for insecure connections
        if hasattr(config.vector_store, 'host') and config.vector_store.host:
            if config.vector_store.host.startswith('http://'):
                warnings.append("Insecure HTTP connection detected for vector store")
                recommendations.append("Use HTTPS for production deployments")
        
        if hasattr(config.graph_store, 'uri') and config.graph_store.uri:
            if config.graph_store.uri.startswith('http://'):
                warnings.append("Insecure HTTP connection detected for graph store")
                recommendations.append("Use secure connections (bolt+s://, https://) for production")
        
        return errors, warnings, recommendations
    
    def _validate_performance(self, config: KSEConfig) -> tuple:
        """Validate performance settings."""
        warnings = []
        recommendations = []
        
        # Check cache configuration
        if not config.cache.enabled:
            warnings.append("Caching is disabled - may impact performance")
            recommendations.append("Enable caching for production to improve response times")
        
        # Check worker configuration
        if hasattr(config, 'max_workers') and config.max_workers < 4:
            warnings.append("Low worker count may limit throughput")
            recommendations.append("Consider increasing max_workers for production workloads")
        
        # Check conceptual service settings
        if config.conceptual.auto_compute:
            if not config.conceptual.llm_api_key:
                warnings.append("Auto-compute enabled but no LLM API key provided")
                recommendations.append("Provide LLM API key or disable auto_compute for production")
        
        return warnings, recommendations
    
    def generate_production_config_template(self) -> Dict[str, Any]:
        """Generate a production-ready configuration template."""
        return {
            "app_name": "KSE Memory Production",
            "version": "1.0.0",
            "debug": False,
            "log_level": "WARNING",
            
            "vector_store": {
                "backend": "pinecone",  # or weaviate, qdrant, chromadb, milvus
                "api_key": "${PINECONE_API_KEY}",
                "environment": "${PINECONE_ENVIRONMENT}",
                "index_name": "kse-products-prod",
                "dimension": 1536,
                "metric": "cosine"
            },
            
            "graph_store": {
                "backend": "neo4j",  # or arangodb
                "uri": "${NEO4J_URI}",
                "username": "${NEO4J_USERNAME}",
                "password": "${NEO4J_PASSWORD}",
                "database": "kse-prod"
            },
            
            "concept_store": {
                "backend": "postgresql",  # or mongodb
                "uri": "${POSTGRESQL_URI}",
                "database": "kse_concepts_prod"
            },
            
            "embedding": {
                "text_model": "text-embedding-3-small",
                "openai_api_key": "${OPENAI_API_KEY}",
                "batch_size": 32,
                "max_retries": 3,
                "timeout": 30
            },
            
            "conceptual": {
                "auto_compute": True,
                "llm_model": "gpt-4o-mini",
                "llm_api_key": "${OPENAI_API_KEY}",
                "batch_size": 16
            },
            
            "cache": {
                "enabled": True,
                "backend": "redis",
                "uri": "${REDIS_URI}",
                "ttl": 3600,
                "max_size": 10000
            },
            
            "performance": {
                "max_workers": 8,
                "request_timeout": 30,
                "connection_pool_size": 20
            }
        }


def validate_production_readiness(config: KSEConfig) -> ValidationResult:
    """
    Convenience function to validate production readiness.
    
    Args:
        config: KSE configuration to validate
        
    Returns:
        Validation result
    """
    validator = ProductionValidator()
    return validator.validate_config(config)