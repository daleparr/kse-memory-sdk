"""
Time2Vec encoding for temporal embeddings in KSE Memory SDK
"""

import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from typing import List, Optional, Tuple
from .temporal_models import TemporalEvent


class Time2VecEncoder(nn.Module):
    """Time2Vec encoding for temporal embeddings"""
    
    def __init__(self, embedding_dim: int = 512, time_dim: int = 64):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.time_dim = time_dim
        
        # Learnable parameters for Time2Vec
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.randn(1))
        self.W = nn.Parameter(torch.randn(time_dim - 1))
        self.B = nn.Parameter(torch.randn(time_dim - 1))
        
        # Projection layer
        self.time_projection = nn.Linear(time_dim, embedding_dim)
        
        # Content encoder (simplified - in practice would use transformer)
        self.content_encoder = nn.Sequential(
            nn.Linear(768, embedding_dim),  # Assuming BERT-like input
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Temporal attention mechanism
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
    
    def time2vec(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Convert timestamps to Time2Vec representation
        
        Args:
            timestamps: Tensor of Unix timestamps
            
        Returns:
            Time2Vec encoded representations
        """
        # Normalize timestamps to reasonable range
        normalized_timestamps = timestamps / 86400.0  # Convert to days
        
        # Linear component
        linear_component = self.w0 * normalized_timestamps + self.b0
        
        # Periodic components
        periodic_components = torch.sin(
            torch.outer(normalized_timestamps, self.W) + self.B
        )
        
        # Concatenate linear and periodic components
        time_encoding = torch.cat([
            linear_component.unsqueeze(-1), 
            periodic_components
        ], dim=-1)
        
        return self.time_projection(time_encoding)
    
    def encode_content(self, content: str) -> torch.Tensor:
        """Encode content text (simplified implementation)"""
        # In practice, this would use a proper text encoder like BERT
        # For now, create a dummy encoding
        content_hash = hash(content) % 1000000
        dummy_encoding = torch.randn(768) * (content_hash / 1000000.0)
        return self.content_encoder(dummy_encoding.unsqueeze(0)).squeeze(0)
    
    def encode_temporal_sequence(self, event_sequence: List[TemporalEvent]) -> torch.Tensor:
        """Encode sequence of temporal events
        
        Args:
            event_sequence: List of temporal events
            
        Returns:
            Sequence encoding with temporal information
        """
        if not event_sequence:
            return torch.zeros(self.embedding_dim)
        
        # Extract timestamps and content
        timestamps = torch.tensor([
            event.timestamp.timestamp() for event in event_sequence
        ], dtype=torch.float32)
        
        content_embeddings = torch.stack([
            self.encode_content(event.content) for event in event_sequence
        ])
        
        # Time encodings
        time_encodings = self.time2vec(timestamps)
        
        # Combine content and time
        temporal_embeddings = content_embeddings + time_encodings
        
        # Apply temporal attention
        attended_embeddings, attention_weights = self.temporal_attention(
            temporal_embeddings.unsqueeze(0),
            temporal_embeddings.unsqueeze(0),
            temporal_embeddings.unsqueeze(0)
        )
        
        # Return mean-pooled representation
        return attended_embeddings.squeeze(0).mean(dim=0)
    
    def compute_temporal_similarity(self, events1: List[TemporalEvent], 
                                   events2: List[TemporalEvent]) -> float:
        """Compute temporal similarity between two event sequences"""
        
        if not events1 or not events2:
            return 0.0
        
        # Encode both sequences
        encoding1 = self.encode_temporal_sequence(events1)
        encoding2 = self.encode_temporal_sequence(events2)
        
        # Compute cosine similarity
        similarity = torch.cosine_similarity(
            encoding1.unsqueeze(0), 
            encoding2.unsqueeze(0)
        )
        
        return similarity.item()
    
    def get_temporal_features(self, timestamp: datetime) -> torch.Tensor:
        """Extract temporal features from a single timestamp"""
        
        # Convert to various time scales
        unix_time = timestamp.timestamp()
        hour_of_day = timestamp.hour / 24.0
        day_of_week = timestamp.weekday() / 7.0
        day_of_month = timestamp.day / 31.0
        month_of_year = timestamp.month / 12.0
        
        # Create feature vector
        temporal_features = torch.tensor([
            unix_time / 86400.0,  # Normalized days since epoch
            hour_of_day,
            day_of_week,
            day_of_month,
            month_of_year,
            np.sin(2 * np.pi * hour_of_day),  # Cyclical hour
            np.cos(2 * np.pi * hour_of_day),
            np.sin(2 * np.pi * day_of_week),  # Cyclical day of week
            np.cos(2 * np.pi * day_of_week),
            np.sin(2 * np.pi * month_of_year),  # Cyclical month
            np.cos(2 * np.pi * month_of_year)
        ], dtype=torch.float32)
        
        return temporal_features


class TemporalEmbeddingCache:
    """Cache for temporal embeddings to improve performance"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.cache = {}
        self.access_order = []
    
    def get(self, key: str) -> Optional[torch.Tensor]:
        """Get cached embedding"""
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, embedding: torch.Tensor):
        """Store embedding in cache"""
        if len(self.cache) >= self.max_size:
            # Remove least recently used
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]
        
        self.cache[key] = embedding
        self.access_order.append(key)
    
    def clear(self):
        """Clear the cache"""
        self.cache.clear()
        self.access_order.clear()


class TemporalSimilarityFunction:
    """Implements the temporal similarity function from the research paper"""
    
    def __init__(self, gamma_t: float = 0.3):
        self.gamma_t = gamma_t
    
    def compute_similarity(self, entity1, entity2) -> float:
        """
        Compute temporal similarity using the formula:
        Sim_T(p1, p2) = γ_t * e^(-|t1 - t2|) + (1 - γ_t) * Sim_base(p1, p2)
        """
        # Temporal component
        time_diff = abs((entity1.timestamp - entity2.timestamp).total_seconds())
        temporal_component = self.gamma_t * np.exp(-time_diff / 3600)  # Decay per hour
        
        # Base similarity component (simplified)
        base_similarity = self._compute_base_similarity(entity1, entity2)
        base_component = (1 - self.gamma_t) * base_similarity
        
        return temporal_component + base_component
    
    def _compute_base_similarity(self, entity1, entity2) -> float:
        """Compute base similarity between entities (simplified implementation)"""
        # In practice, this would use the full KSE similarity computation
        # For now, use a simple overlap measure
        
        if not hasattr(entity1, 'properties') or not hasattr(entity2, 'properties'):
            return 0.5  # Default similarity
        
        # Simple Jaccard similarity on property keys
        keys1 = set(entity1.properties.keys())
        keys2 = set(entity2.properties.keys())
        
        if not keys1 and not keys2:
            return 1.0
        
        intersection = len(keys1.intersection(keys2))
        union = len(keys1.union(keys2))
        
        return intersection / union if union > 0 else 0.0


def temporal_similarity_function(p1, p2, gamma_t: float = 0.3) -> float:
    """
    Standalone temporal similarity function as specified in the research paper
    
    Args:
        p1, p2: Temporal entities with timestamp attributes
        gamma_t: Temporal weight parameter
        
    Returns:
        Temporal similarity score
    """
    similarity_fn = TemporalSimilarityFunction(gamma_t)
    return similarity_fn.compute_similarity(p1, p2)