"""
Temporal data models for KSE Memory SDK
"""

import torch
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from ..core.models import Product, ConceptualDimensions


class TemporalRelationType(Enum):
    """Types of temporal relationships"""
    PRECEDES = "precedes"
    FOLLOWS = "follows"
    OVERLAPS = "overlaps"
    CONTAINS = "contains"
    SIMULTANEOUS = "simultaneous"
    CAUSES = "causes"


@dataclass
class TimeInterval:
    """Represents a time interval"""
    
    start: datetime
    end: datetime
    duration: Optional[timedelta] = None
    
    def __post_init__(self):
        if self.duration is None:
            self.duration = self.end - self.start
    
    def contains(self, timestamp: datetime) -> bool:
        """Check if timestamp is within this interval"""
        return self.start <= timestamp <= self.end
    
    def overlaps_with(self, other: 'TimeInterval') -> bool:
        """Check if this interval overlaps with another"""
        return self.start <= other.end and other.start <= self.end
    
    def get_overlap(self, other: 'TimeInterval') -> Optional['TimeInterval']:
        """Get the overlapping interval with another interval"""
        if not self.overlaps_with(other):
            return None
        
        overlap_start = max(self.start, other.start)
        overlap_end = min(self.end, other.end)
        
        return TimeInterval(start=overlap_start, end=overlap_end)


@dataclass
class TemporalKnowledgeItem:
    """Knowledge item with temporal validity"""
    
    item_id: str
    content: str
    timestamp: datetime
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    confidence: float = 1.0
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_valid_at(self, timestamp: datetime) -> bool:
        """Check if item is valid at given timestamp"""
        if self.valid_from and timestamp < self.valid_from:
            return False
        if self.valid_to and timestamp > self.valid_to:
            return False
        return True
    
    def get_validity_duration(self) -> Optional[timedelta]:
        """Get the duration of validity"""
        if self.valid_from and self.valid_to:
            return self.valid_to - self.valid_from
        return None


@dataclass
class TemporalRelationship:
    """Temporal relationship between entities"""
    
    source_id: str
    target_id: str
    relation_type: str
    start_time: datetime
    end_time: Optional[datetime] = None
    confidence: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def is_active_at(self, timestamp: datetime) -> bool:
        """Check if relationship is active at given timestamp"""
        if timestamp < self.start_time:
            return False
        if self.end_time and timestamp > self.end_time:
            return False
        return True
    
    def overlaps_with(self, other: 'TemporalRelationship') -> bool:
        """Check if this relationship overlaps temporally with another"""
        start1, end1 = self.start_time, self.end_time or datetime.max
        start2, end2 = other.start_time, other.end_time or datetime.max
        
        return start1 <= end2 and start2 <= end1


@dataclass
class TemporalEvent:
    """Represents a temporal event"""
    
    event_id: str
    event_type: str
    timestamp: datetime
    duration: Optional[timedelta] = None
    participants: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    
    def get_end_time(self) -> datetime:
        """Get the end time of the event"""
        if self.duration:
            return self.timestamp + self.duration
        return self.timestamp


@dataclass
class TemporalPattern:
    """Represents a detected temporal pattern"""
    
    pattern_id: str
    pattern_type: str  # "recurring", "causal", "seasonal", etc.
    entities: List[str]
    relations: List[str]
    time_intervals: List[TimeInterval]
    confidence: float
    support: int  # Number of instances supporting this pattern
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def get_pattern_duration(self) -> Optional[timedelta]:
        """Get the total duration covered by this pattern"""
        if not self.time_intervals:
            return None
        
        start = min(interval.start for interval in self.time_intervals)
        end = max(interval.end for interval in self.time_intervals)
        return end - start


@dataclass
class TemporalQuery:
    """Query with temporal constraints"""
    
    query_text: str
    timestamp: Optional[datetime] = None
    time_range: Optional[TimeInterval] = None
    temporal_weight: float = 0.5  # Weight for temporal vs semantic similarity
    max_age: Optional[timedelta] = None
    require_valid: bool = True  # Only return items valid at query time
    
    def is_within_constraints(self, item_timestamp: datetime) -> bool:
        """Check if an item timestamp satisfies query constraints"""
        if self.timestamp and self.max_age:
            if abs((item_timestamp - self.timestamp).total_seconds()) > self.max_age.total_seconds():
                return False
        
        if self.time_range:
            if not self.time_range.contains(item_timestamp):
                return False
        
        return True


@dataclass
class TemporalConcept:
    """Concept in temporal conceptual space"""
    
    concept_id: str
    space_id: str
    coordinates: torch.Tensor
    timestamp: datetime
    properties: Dict[str, Any] = field(default_factory=dict)
    temporal_embedding: Optional[torch.Tensor] = None
    
    def get_temporal_features(self) -> torch.Tensor:
        """Get temporal features for this concept"""
        features = []
        
        # Time-based features
        now = datetime.now()
        age_hours = (now - self.timestamp).total_seconds() / 3600
        features.append(age_hours)
        
        # Cyclical time features
        hour_of_day = self.timestamp.hour / 24.0
        day_of_week = self.timestamp.weekday() / 7.0
        day_of_year = self.timestamp.timetuple().tm_yday / 365.0
        
        features.extend([hour_of_day, day_of_week, day_of_year])
        
        return torch.tensor(features, dtype=torch.float32)


@dataclass
class TemporalConceptualSpace:
    """Conceptual space with temporal awareness"""
    
    space_id: str
    domain: str
    dimensions: List[str]
    temporal_dimensions: Dict[str, Any] = field(default_factory=dict)
    concepts: Dict[str, TemporalConcept] = field(default_factory=dict)
    creation_time: datetime = field(default_factory=datetime.now)
    temporal_config: Dict[str, Any] = field(default_factory=dict)
    
    def add_concept(self, concept: TemporalConcept):
        """Add a temporal concept to this space"""
        self.concepts[concept.concept_id] = concept
    
    def get_concepts_at_time(self, timestamp: datetime, 
                           time_window: Optional[timedelta] = None) -> List[TemporalConcept]:
        """Get concepts that were active at a specific time"""
        if time_window is None:
            time_window = timedelta(hours=24)
        
        start_time = timestamp - time_window
        end_time = timestamp + time_window
        
        return [
            concept for concept in self.concepts.values()
            if start_time <= concept.timestamp <= end_time
        ]


@dataclass
class TemporalMemoryConfig:
    """Configuration for temporal memory operations"""
    
    time_encoding_dim: int = 64
    temporal_window_days: int = 30
    pattern_min_support: int = 3
    temporal_decay: float = 0.1
    enable_pattern_detection: bool = True
    enable_prediction: bool = True
    time_encoding_method: str = "time2vec"
    max_temporal_patterns: int = 1000
    pattern_cache_size: int = 100
    temporal_similarity_threshold: float = 0.5
    
    def get_temporal_window(self) -> timedelta:
        """Get temporal window as timedelta"""
        return timedelta(days=self.temporal_window_days)
    
    def get_decay_factor(self, age: timedelta) -> float:
        """Calculate decay factor based on age"""
        import math
        age_days = age.total_seconds() / (24 * 3600)
        return math.exp(-self.temporal_decay * age_days)