"""
Test suite for KSE Memory SDK.
"""

# Test configuration
TEST_CONFIG = {
    "vector_store": {
        "backend": "memory",  # Use in-memory for testing
    },
    "graph_store": {
        "backend": "memory",  # Use in-memory for testing
    },
    "concept_store": {
        "backend": "memory",  # Use in-memory for testing
    },
    "embedding": {
        "text_model": "sentence-transformers/all-MiniLM-L6-v2",
        "batch_size": 4,  # Small batch for testing
    },
    "conceptual": {
        "auto_compute": False,  # Disable LLM for testing
    },
    "cache": {
        "enabled": False,  # Disable cache for testing
    },
}