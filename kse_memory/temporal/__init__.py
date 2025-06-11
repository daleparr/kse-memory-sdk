"""
Temporal reasoning module for KSE Memory SDK

This module provides comprehensive temporal reasoning capabilities including
temporal knowledge graphs, time-aware conceptual spaces, and temporal pattern detection.
"""

from .temporal_models import (
    TemporalKnowledgeItem,
    TemporalRelationship,
    TemporalQuery,
    TemporalEvent,
    TimeInterval,
    TemporalPattern,
    TemporalConceptualSpace,
    TemporalConcept,
    TemporalMemoryConfig
)

from .temporal_graph import (
    TemporalGraphNode,
    TemporalGraphEdge,
    TemporalKnowledgeGraph,
    Time2VecEncoder
)

from .temporal_conceptual import (
    TemporalDimension,
    TemporalConceptualSpaceManager,
    TemporalEncoder,
    TemporalPatternDetector
)

__all__ = [
    # Core temporal models
    "TemporalKnowledgeItem",
    "TemporalRelationship", 
    "TemporalQuery",
    "TemporalEvent",
    "TimeInterval",
    "TemporalPattern",
    "TemporalConceptualSpace",
    "TemporalConcept",
    "TemporalMemoryConfig",
    
    # Temporal knowledge graph
    "TemporalGraphNode",
    "TemporalGraphEdge", 
    "TemporalKnowledgeGraph",
    "Time2VecEncoder",
    
    # Temporal conceptual spaces
    "TemporalDimension",
    "TemporalConceptualSpaceManager",
    "TemporalEncoder",
    "TemporalPatternDetector"
]

# Version info
__version__ = "1.0.0"
__author__ = "KSE Memory SDK Team"
__description__ = "Temporal reasoning for Knowledge Space Embeddings"

# Module-level configuration
DEFAULT_TIME_ENCODING_DIM = 64
DEFAULT_TEMPORAL_WINDOW_DAYS = 30
DEFAULT_PATTERN_MIN_SUPPORT = 3
DEFAULT_TEMPORAL_DECAY = 0.1

# Temporal pattern types
TEMPORAL_PATTERN_TYPES = [
    "recurring",
    "causal", 
    "seasonal",
    "drift",
    "oscillation",
    "clustering",
    "emergence"
]

# Time encoding methods
TIME_ENCODING_METHODS = [
    "time2vec",
    "sinusoidal",
    "learned",
    "absolute"
]


def create_temporal_config(
    time_encoding_dim: int = DEFAULT_TIME_ENCODING_DIM,
    temporal_window_days: int = DEFAULT_TEMPORAL_WINDOW_DAYS,
    pattern_min_support: int = DEFAULT_PATTERN_MIN_SUPPORT,
    temporal_decay: float = DEFAULT_TEMPORAL_DECAY,
    enable_pattern_detection: bool = True,
    enable_prediction: bool = True,
    time_encoding_method: str = "time2vec",
    **kwargs
) -> TemporalMemoryConfig:
    """
    Create a temporal memory configuration with sensible defaults.
    
    Args:
        time_encoding_dim: Dimension of temporal encodings
        temporal_window_days: Default temporal window in days
        pattern_min_support: Minimum support for pattern detection
        temporal_decay: Temporal decay rate for relevance
        enable_pattern_detection: Whether to enable pattern detection
        enable_prediction: Whether to enable temporal prediction
        time_encoding_method: Method for encoding time
        **kwargs: Additional configuration parameters
    
    Returns:
        TemporalMemoryConfig: Configured temporal settings
    """
    return TemporalMemoryConfig(
        time_encoding_dim=time_encoding_dim,
        temporal_window_days=temporal_window_days,
        pattern_min_support=pattern_min_support,
        temporal_decay=temporal_decay,
        enable_pattern_detection=enable_pattern_detection,
        enable_prediction=enable_prediction,
        time_encoding_method=time_encoding_method,
        **kwargs
    )


def create_temporal_knowledge_graph(
    time_encoding_dim: int = DEFAULT_TIME_ENCODING_DIM
) -> TemporalKnowledgeGraph:
    """
    Create a temporal knowledge graph with default configuration.
    
    Args:
        time_encoding_dim: Dimension of temporal encodings
    
    Returns:
        TemporalKnowledgeGraph: Configured temporal knowledge graph
    """
    return TemporalKnowledgeGraph(time_encoding_dim=time_encoding_dim)


def create_temporal_conceptual_manager(
    base_dimensions: int = 10,
    time_encoding_dim: int = DEFAULT_TIME_ENCODING_DIM
) -> TemporalConceptualSpaceManager:
    """
    Create a temporal conceptual space manager.
    
    Args:
        base_dimensions: Number of base conceptual dimensions
        time_encoding_dim: Dimension of temporal encodings
    
    Returns:
        TemporalConceptualSpaceManager: Configured manager
    """
    return TemporalConceptualSpaceManager(
        base_dimensions=base_dimensions,
        time_encoding_dim=time_encoding_dim
    )


# Convenience functions for common temporal operations

def encode_timestamp(timestamp, method: str = "time2vec", 
                    encoding_dim: int = DEFAULT_TIME_ENCODING_DIM):
    """
    Encode a timestamp using the specified method.
    
    Args:
        timestamp: Datetime object to encode
        method: Encoding method ("time2vec", "sinusoidal", etc.)
        encoding_dim: Dimension of the encoding
    
    Returns:
        torch.Tensor: Encoded timestamp
    """
    if method == "time2vec":
        encoder = Time2VecEncoder(encoding_dim)
        return encoder.encode(timestamp)
    elif method == "sinusoidal":
        encoder = TemporalEncoder(encoding_dim)
        return encoder.encode(timestamp)
    else:
        raise ValueError(f"Unknown encoding method: {method}")


def calculate_temporal_similarity(timestamp1, timestamp2, 
                                window_days: int = DEFAULT_TEMPORAL_WINDOW_DAYS) -> float:
    """
    Calculate temporal similarity between two timestamps.
    
    Args:
        timestamp1: First timestamp
        timestamp2: Second timestamp  
        window_days: Temporal window in days
    
    Returns:
        float: Similarity score between 0 and 1
    """
    import math
    from datetime import timedelta
    
    time_diff = abs((timestamp1 - timestamp2).total_seconds())
    window_seconds = window_days * 24 * 3600
    
    # Exponential decay similarity
    similarity = math.exp(-time_diff / window_seconds)
    return similarity


def detect_temporal_anomalies(temporal_data, threshold: float = 2.0) -> list:
    """
    Detect temporal anomalies in a sequence of temporal data.
    
    Args:
        temporal_data: List of (timestamp, value) tuples
        threshold: Standard deviation threshold for anomaly detection
    
    Returns:
        list: List of anomalous data points
    """
    import numpy as np
    
    if len(temporal_data) < 3:
        return []
    
    values = [value for _, value in temporal_data]
    mean_val = np.mean(values)
    std_val = np.std(values)
    
    anomalies = []
    for timestamp, value in temporal_data:
        if abs(value - mean_val) > threshold * std_val:
            anomalies.append((timestamp, value))
    
    return anomalies


def interpolate_temporal_value(temporal_data, target_timestamp):
    """
    Interpolate a value at a target timestamp from temporal data.
    
    Args:
        temporal_data: List of (timestamp, value) tuples
        target_timestamp: Target timestamp for interpolation
    
    Returns:
        float: Interpolated value
    """
    if not temporal_data:
        return None
    
    # Sort by timestamp
    sorted_data = sorted(temporal_data, key=lambda x: x[0])
    
    # Find surrounding points
    before = None
    after = None
    
    for timestamp, value in sorted_data:
        if timestamp <= target_timestamp:
            before = (timestamp, value)
        elif timestamp > target_timestamp and after is None:
            after = (timestamp, value)
            break
    
    # Interpolate
    if before and after:
        time_diff = (after[0] - before[0]).total_seconds()
        if time_diff > 0:
            weight = (target_timestamp - before[0]).total_seconds() / time_diff
            return before[1] + weight * (after[1] - before[1])
    elif before:
        return before[1]
    elif after:
        return after[1]
    
    return None


def get_temporal_summary(pattern_types: list = None) -> dict:
    """
    Get a summary of temporal reasoning capabilities.
    
    Args:
        pattern_types: List of pattern types to include
    
    Returns:
        dict: Summary of temporal features
    """
    if pattern_types is None:
        pattern_types = TEMPORAL_PATTERN_TYPES
    
    return {
        "temporal_patterns": pattern_types,
        "encoding_methods": TIME_ENCODING_METHODS,
        "default_encoding_dim": DEFAULT_TIME_ENCODING_DIM,
        "default_window_days": DEFAULT_TEMPORAL_WINDOW_DAYS,
        "features": {
            "temporal_knowledge_graphs": True,
            "temporal_conceptual_spaces": True,
            "pattern_detection": True,
            "temporal_prediction": True,
            "anomaly_detection": True,
            "temporal_interpolation": True,
            "causal_reasoning": True,
            "seasonal_analysis": True
        },
        "use_cases": [
            "Time-aware information retrieval",
            "Temporal pattern mining",
            "Predictive analytics",
            "Causal relationship discovery",
            "Seasonal trend analysis",
            "Temporal anomaly detection",
            "Dynamic knowledge evolution",
            "Time-sensitive recommendations"
        ]
    }


# Module initialization
def _initialize_temporal_module():
    """Initialize the temporal reasoning module"""
    import logging
    
    logger = logging.getLogger(__name__)
    logger.info("KSE Temporal Reasoning Module initialized")
    logger.info(f"Default time encoding dimension: {DEFAULT_TIME_ENCODING_DIM}")
    logger.info(f"Supported pattern types: {TEMPORAL_PATTERN_TYPES}")
    logger.info(f"Supported encoding methods: {TIME_ENCODING_METHODS}")


# Initialize on import
_initialize_temporal_module()