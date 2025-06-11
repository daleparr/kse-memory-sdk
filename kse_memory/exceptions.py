"""
Custom exceptions for KSE Memory SDK.
"""


class KSEError(Exception):
    """Base exception for all KSE Memory errors."""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self):
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ConfigurationError(KSEError):
    """Raised when there's a configuration error."""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "CONFIG_ERROR", details)


class AdapterError(KSEError):
    """Raised when there's an adapter-related error."""
    
    def __init__(self, message: str, adapter_type: str = None, details: dict = None):
        error_details = details or {}
        if adapter_type:
            error_details["adapter_type"] = adapter_type
        super().__init__(message, "ADAPTER_ERROR", error_details)


class BackendError(KSEError):
    """Raised when there's a backend storage error."""
    
    def __init__(self, message: str, backend_type: str = None, details: dict = None):
        error_details = details or {}
        if backend_type:
            error_details["backend_type"] = backend_type
        super().__init__(message, "BACKEND_ERROR", error_details)


class SearchError(KSEError):
    """Raised when there's a search-related error."""
    
    def __init__(self, message: str, query: str = None, details: dict = None):
        error_details = details or {}
        if query:
            error_details["query"] = query
        super().__init__(message, "SEARCH_ERROR", error_details)


class EmbeddingError(KSEError):
    """Raised when there's an embedding generation error."""
    
    def __init__(self, message: str, model: str = None, details: dict = None):
        error_details = details or {}
        if model:
            error_details["model"] = model
        super().__init__(message, "EMBEDDING_ERROR", error_details)


class ConceptualError(KSEError):
    """Raised when there's a conceptual dimension computation error."""
    
    def __init__(self, message: str, product_id: str = None, details: dict = None):
        error_details = details or {}
        if product_id:
            error_details["product_id"] = product_id
        super().__init__(message, "CONCEPTUAL_ERROR", error_details)


class VectorStoreError(BackendError):
    """Raised when there's a vector store error."""
    
    def __init__(self, message: str, operation: str = None, details: dict = None):
        error_details = details or {}
        if operation:
            error_details["operation"] = operation
        super().__init__(message, "vector_store", error_details)


class GraphStoreError(BackendError):
    """Raised when there's a graph store error."""
    
    def __init__(self, message: str, operation: str = None, details: dict = None):
        error_details = details or {}
        if operation:
            error_details["operation"] = operation
        super().__init__(message, "graph_store", error_details)


class ConceptStoreError(BackendError):
    """Raised when there's a concept store error."""
    
    def __init__(self, message: str, operation: str = None, details: dict = None):
        error_details = details or {}
        if operation:
            error_details["operation"] = operation
        super().__init__(message, "concept_store", error_details)


class CacheError(KSEError):
    """Raised when there's a cache-related error."""
    
    def __init__(self, message: str, operation: str = None, details: dict = None):
        error_details = details or {}
        if operation:
            error_details["operation"] = operation
        super().__init__(message, "CACHE_ERROR", error_details)


class ValidationError(KSEError):
    """Raised when there's a data validation error."""
    
    def __init__(self, message: str, field: str = None, value: str = None, details: dict = None):
        error_details = details or {}
        if field:
            error_details["field"] = field
        if value:
            error_details["value"] = value
        super().__init__(message, "VALIDATION_ERROR", error_details)


class AuthenticationError(KSEError):
    """Raised when there's an authentication error."""
    
    def __init__(self, message: str, service: str = None, details: dict = None):
        error_details = details or {}
        if service:
            error_details["service"] = service
        super().__init__(message, "AUTH_ERROR", error_details)


class RateLimitError(KSEError):
    """Raised when rate limits are exceeded."""
    
    def __init__(self, message: str, service: str = None, retry_after: int = None, details: dict = None):
        error_details = details or {}
        if service:
            error_details["service"] = service
        if retry_after:
            error_details["retry_after"] = retry_after
        super().__init__(message, "RATE_LIMIT_ERROR", error_details)


class TimeoutError(KSEError):
    """Raised when operations timeout."""
    
    def __init__(self, message: str, operation: str = None, timeout: int = None, details: dict = None):
        error_details = details or {}
        if operation:
            error_details["operation"] = operation
        if timeout:
            error_details["timeout"] = timeout
        super().__init__(message, "TIMEOUT_ERROR", error_details)


class NotFoundError(KSEError):
    """Raised when a requested resource is not found."""
    
    def __init__(self, message: str, resource_type: str = None, resource_id: str = None, details: dict = None):
        error_details = details or {}
        if resource_type:
            error_details["resource_type"] = resource_type
        if resource_id:
            error_details["resource_id"] = resource_id
        super().__init__(message, "NOT_FOUND_ERROR", error_details)


class DuplicateError(KSEError):
    """Raised when trying to create a resource that already exists."""
    
    def __init__(self, message: str, resource_type: str = None, resource_id: str = None, details: dict = None):
        error_details = details or {}
        if resource_type:
            error_details["resource_type"] = resource_type
        if resource_id:
            error_details["resource_id"] = resource_id
        super().__init__(message, "DUPLICATE_ERROR", error_details)