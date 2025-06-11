"""
Temporal conceptual spaces implementation for KSE Memory SDK
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
from collections import defaultdict
import math

from ..core.models import ConceptualDimensions
from .temporal_models import (
    TemporalConceptualSpace, TemporalConcept, TemporalQuery,
    TimeInterval, TemporalPattern, TemporalEvent
)


logger = logging.getLogger(__name__)


@dataclass
class TemporalDimension:
    """Temporal dimension in conceptual space"""
    
    name: str
    dimension_type: str  # "quality", "temporal", "hybrid"
    
    # Temporal properties
    temporal_weight: float = 1.0
    temporal_decay: float = 0.1  # How quickly temporal influence decays
    seasonal_period: Optional[timedelta] = None
    
    # Value evolution
    value_history: List[Tuple[datetime, float]] = field(default_factory=list)
    trend_direction: float = 0.0  # -1 (decreasing), 0 (stable), 1 (increasing)
    volatility: float = 0.0  # Measure of value stability
    
    # Prediction
    predicted_values: Dict[datetime, float] = field(default_factory=dict)
    prediction_confidence: Dict[datetime, float] = field(default_factory=dict)
    
    def add_value(self, timestamp: datetime, value: float):
        """Add a new value observation"""
        self.value_history.append((timestamp, value))
        self.value_history.sort(key=lambda x: x[0])  # Keep sorted by time
        
        # Update trend and volatility
        self._update_statistics()
    
    def get_value_at(self, timestamp: datetime) -> Optional[float]:
        """Get interpolated value at specific timestamp"""
        if not self.value_history:
            return None
        
        # Find surrounding values
        before = None
        after = None
        
        for ts, val in self.value_history:
            if ts <= timestamp:
                before = (ts, val)
            elif ts > timestamp and after is None:
                after = (ts, val)
                break
        
        # Interpolate
        if before and after:
            time_diff = (after[0] - before[0]).total_seconds()
            if time_diff > 0:
                weight = (timestamp - before[0]).total_seconds() / time_diff
                return before[1] + weight * (after[1] - before[1])
        elif before:
            return before[1]
        elif after:
            return after[1]
        
        return None
    
    def predict_value(self, timestamp: datetime) -> Tuple[float, float]:
        """Predict value at future timestamp with confidence"""
        if timestamp in self.predicted_values:
            return self.predicted_values[timestamp], self.prediction_confidence[timestamp]
        
        if not self.value_history:
            return 0.0, 0.0
        
        # Simple linear trend prediction
        if len(self.value_history) >= 2:
            recent_values = self.value_history[-5:]  # Use last 5 values
            
            # Calculate trend
            x_values = [(ts - recent_values[0][0]).total_seconds() for ts, _ in recent_values]
            y_values = [val for _, val in recent_values]
            
            # Linear regression
            n = len(recent_values)
            sum_x = sum(x_values)
            sum_y = sum(y_values)
            sum_xy = sum(x * y for x, y in zip(x_values, y_values))
            sum_x2 = sum(x * x for x in x_values)
            
            if n * sum_x2 - sum_x * sum_x != 0:
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                intercept = (sum_y - slope * sum_x) / n
                
                # Predict
                time_delta = (timestamp - recent_values[0][0]).total_seconds()
                predicted_value = intercept + slope * time_delta
                
                # Confidence based on volatility and time distance
                time_distance = (timestamp - recent_values[-1][0]).total_seconds()
                confidence = max(0.1, 1.0 - (self.volatility * time_distance / 86400))  # Decay over days
                
                self.predicted_values[timestamp] = predicted_value
                self.prediction_confidence[timestamp] = confidence
                
                return predicted_value, confidence
        
        # Fallback to last known value
        last_value = self.value_history[-1][1]
        return last_value, 0.5
    
    def _update_statistics(self):
        """Update trend and volatility statistics"""
        if len(self.value_history) < 2:
            return
        
        # Calculate trend (slope of recent values)
        recent_values = self.value_history[-10:]  # Use last 10 values
        if len(recent_values) >= 2:
            time_diffs = [(recent_values[i][0] - recent_values[i-1][0]).total_seconds() 
                         for i in range(1, len(recent_values))]
            value_diffs = [recent_values[i][1] - recent_values[i-1][1] 
                          for i in range(1, len(recent_values))]
            
            if sum(time_diffs) > 0:
                avg_slope = sum(vd / td for vd, td in zip(value_diffs, time_diffs) if td > 0) / len(value_diffs)
                self.trend_direction = np.tanh(avg_slope)  # Normalize to [-1, 1]
        
        # Calculate volatility (standard deviation of recent changes)
        if len(recent_values) >= 3:
            values = [val for _, val in recent_values]
            self.volatility = np.std(values)


class TemporalConceptualSpaceManager:
    """Manages temporal conceptual spaces for KSE Memory"""
    
    def __init__(self, base_dimensions: int = 10, time_encoding_dim: int = 32):
        self.base_dimensions = base_dimensions
        self.time_encoding_dim = time_encoding_dim
        
        # Temporal conceptual spaces
        self.temporal_spaces: Dict[str, TemporalConceptualSpace] = {}
        self.temporal_dimensions: Dict[str, TemporalDimension] = {}
        
        # Time encoding
        self.time_encoder = TemporalEncoder(time_encoding_dim)
        
        # Concept evolution tracking
        self.concept_trajectories: Dict[str, List[Tuple[datetime, torch.Tensor]]] = defaultdict(list)
        self.concept_predictions: Dict[str, Dict[datetime, torch.Tensor]] = defaultdict(dict)
        
        # Temporal patterns
        self.temporal_patterns: List[TemporalPattern] = []
        self.pattern_detector = TemporalPatternDetector()
        
        # Statistics
        self.stats = {
            "total_spaces": 0,
            "total_concepts": 0,
            "temporal_events": 0,
            "active_patterns": 0
        }
    
    def create_temporal_space(self, space_id: str, domain: str,
                            dimensions: Dict[str, Dict[str, Any]],
                            temporal_config: Optional[Dict[str, Any]] = None) -> TemporalConceptualSpace:
        """Create a new temporal conceptual space"""
        
        # Create temporal dimensions
        temporal_dims = {}
        for dim_name, dim_config in dimensions.items():
            temporal_dim = TemporalDimension(
                name=dim_name,
                dimension_type=dim_config.get("type", "quality"),
                temporal_weight=dim_config.get("temporal_weight", 1.0),
                temporal_decay=dim_config.get("temporal_decay", 0.1),
                seasonal_period=dim_config.get("seasonal_period")
            )
            temporal_dims[dim_name] = temporal_dim
            self.temporal_dimensions[f"{space_id}_{dim_name}"] = temporal_dim
        
        # Create temporal space
        temporal_space = TemporalConceptualSpace(
            space_id=space_id,
            domain=domain,
            dimensions=list(dimensions.keys()),
            temporal_dimensions=temporal_dims,
            creation_time=datetime.now(),
            temporal_config=temporal_config or {}
        )
        
        self.temporal_spaces[space_id] = temporal_space
        self.stats["total_spaces"] += 1
        
        logger.info(f"Created temporal conceptual space: {space_id} for domain: {domain}")
        return temporal_space
    
    def add_temporal_concept(self, space_id: str, concept_id: str,
                           coordinates: torch.Tensor, timestamp: Optional[datetime] = None,
                           properties: Optional[Dict[str, Any]] = None) -> TemporalConcept:
        """Add a temporal concept to a space"""
        if space_id not in self.temporal_spaces:
            raise ValueError(f"Temporal space {space_id} not found")
        
        if timestamp is None:
            timestamp = datetime.now()
        
        space = self.temporal_spaces[space_id]
        
        # Create temporal concept
        temporal_concept = TemporalConcept(
            concept_id=concept_id,
            space_id=space_id,
            coordinates=coordinates,
            timestamp=timestamp,
            properties=properties or {},
            temporal_embedding=self._create_temporal_embedding(coordinates, timestamp)
        )
        
        # Add to space
        space.add_concept(temporal_concept)
        
        # Track concept trajectory
        self.concept_trajectories[concept_id].append((timestamp, coordinates))
        
        # Update dimension values
        for i, dim_name in enumerate(space.dimensions):
            if i < len(coordinates):
                dim_key = f"{space_id}_{dim_name}"
                if dim_key in self.temporal_dimensions:
                    self.temporal_dimensions[dim_key].add_value(timestamp, coordinates[i].item())
        
        self.stats["total_concepts"] += 1
        logger.info(f"Added temporal concept: {concept_id} to space: {space_id} at {timestamp}")
        
        return temporal_concept
    
    def query_temporal_concepts(self, space_id: str, query_point: torch.Tensor,
                              timestamp: datetime, time_window: Optional[timedelta] = None,
                              k: int = 10, temporal_weight: float = 0.3) -> List[Tuple[TemporalConcept, float]]:
        """Query temporal concepts with time-aware similarity"""
        if space_id not in self.temporal_spaces:
            return []
        
        space = self.temporal_spaces[space_id]
        results = []
        
        # Define time window
        if time_window is None:
            time_window = timedelta(days=30)
        
        start_time = timestamp - time_window
        end_time = timestamp + time_window
        
        for concept in space.concepts.values():
            # Check if concept is within time window
            if start_time <= concept.timestamp <= end_time:
                # Calculate spatial similarity
                spatial_distance = torch.norm(query_point - concept.coordinates).item()
                spatial_similarity = 1.0 / (1.0 + spatial_distance)
                
                # Calculate temporal similarity
                time_diff = abs((timestamp - concept.timestamp).total_seconds())
                temporal_similarity = math.exp(-time_diff / (time_window.total_seconds() / 2))
                
                # Combine similarities
                combined_similarity = (
                    (1 - temporal_weight) * spatial_similarity +
                    temporal_weight * temporal_similarity
                )
                
                results.append((concept, combined_similarity))
        
        # Sort by similarity and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def predict_concept_evolution(self, concept_id: str, future_timestamp: datetime) -> Tuple[torch.Tensor, float]:
        """Predict how a concept will evolve to a future timestamp"""
        if concept_id not in self.concept_trajectories:
            raise ValueError(f"Concept {concept_id} not found in trajectories")
        
        trajectory = self.concept_trajectories[concept_id]
        if len(trajectory) < 2:
            # Not enough data for prediction
            return trajectory[0][1], 0.1
        
        # Use linear extrapolation for each dimension
        timestamps = [ts for ts, _ in trajectory]
        coordinates_list = [coords for _, coords in trajectory]
        
        # Convert timestamps to numerical values
        base_time = timestamps[0]
        time_values = [(ts - base_time).total_seconds() for ts in timestamps]
        future_time_value = (future_timestamp - base_time).total_seconds()
        
        predicted_coords = []
        confidence_scores = []
        
        for dim in range(coordinates_list[0].shape[0]):
            dim_values = [coords[dim].item() for coords in coordinates_list]
            
            # Linear regression for this dimension
            if len(time_values) >= 2:
                # Calculate slope and intercept
                n = len(time_values)
                sum_t = sum(time_values)
                sum_v = sum(dim_values)
                sum_tv = sum(t * v for t, v in zip(time_values, dim_values))
                sum_t2 = sum(t * t for t in time_values)
                
                if n * sum_t2 - sum_t * sum_t != 0:
                    slope = (n * sum_tv - sum_t * sum_v) / (n * sum_t2 - sum_t * sum_t)
                    intercept = (sum_v - slope * sum_t) / n
                    
                    # Predict value
                    predicted_value = intercept + slope * future_time_value
                    
                    # Calculate confidence based on fit quality
                    predicted_values = [intercept + slope * t for t in time_values]
                    mse = sum((pred - actual) ** 2 for pred, actual in zip(predicted_values, dim_values)) / n
                    confidence = max(0.1, 1.0 / (1.0 + mse))
                    
                    predicted_coords.append(predicted_value)
                    confidence_scores.append(confidence)
                else:
                    # Fallback to last value
                    predicted_coords.append(dim_values[-1])
                    confidence_scores.append(0.5)
            else:
                predicted_coords.append(dim_values[-1])
                confidence_scores.append(0.5)
        
        predicted_tensor = torch.tensor(predicted_coords, dtype=torch.float32)
        overall_confidence = np.mean(confidence_scores)
        
        # Cache prediction
        self.concept_predictions[concept_id][future_timestamp] = predicted_tensor
        
        return predicted_tensor, overall_confidence
    
    def detect_temporal_patterns(self, space_id: str, pattern_types: List[str] = None) -> List[TemporalPattern]:
        """Detect temporal patterns in conceptual space"""
        if space_id not in self.temporal_spaces:
            return []
        
        if pattern_types is None:
            pattern_types = ["drift", "oscillation", "clustering", "emergence"]
        
        space = self.temporal_spaces[space_id]
        detected_patterns = []
        
        for pattern_type in pattern_types:
            if pattern_type == "drift":
                patterns = self._detect_concept_drift(space)
            elif pattern_type == "oscillation":
                patterns = self._detect_oscillations(space)
            elif pattern_type == "clustering":
                patterns = self._detect_temporal_clustering(space)
            elif pattern_type == "emergence":
                patterns = self._detect_concept_emergence(space)
            else:
                continue
            
            detected_patterns.extend(patterns)
        
        # Store patterns
        self.temporal_patterns.extend(detected_patterns)
        self.stats["active_patterns"] = len(self.temporal_patterns)
        
        return detected_patterns
    
    def _detect_concept_drift(self, space: TemporalConceptualSpace) -> List[TemporalPattern]:
        """Detect concept drift patterns"""
        patterns = []
        
        # Group concepts by time windows
        time_windows = defaultdict(list)
        window_size = timedelta(days=7)  # Weekly windows
        
        for concept in space.concepts.values():
            window_start = concept.timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            window_start = window_start - timedelta(days=window_start.weekday())  # Start of week
            time_windows[window_start].append(concept)
        
        # Calculate centroid drift between consecutive windows
        sorted_windows = sorted(time_windows.keys())
        
        for i in range(len(sorted_windows) - 1):
            current_window = time_windows[sorted_windows[i]]
            next_window = time_windows[sorted_windows[i + 1]]
            
            if len(current_window) >= 3 and len(next_window) >= 3:
                # Calculate centroids
                current_centroid = torch.mean(torch.stack([c.coordinates for c in current_window]), dim=0)
                next_centroid = torch.mean(torch.stack([c.coordinates for c in next_window]), dim=0)
                
                # Calculate drift magnitude
                drift_magnitude = torch.norm(next_centroid - current_centroid).item()
                
                if drift_magnitude > 0.1:  # Threshold for significant drift
                    pattern = TemporalPattern(
                        pattern_id=f"drift_{space.space_id}_{i}",
                        pattern_type="drift",
                        entities=[c.concept_id for c in current_window + next_window],
                        relations=[],
                        time_intervals=[TimeInterval(
                            start=sorted_windows[i],
                            end=sorted_windows[i + 1],
                            duration=sorted_windows[i + 1] - sorted_windows[i]
                        )],
                        confidence=min(1.0, drift_magnitude),
                        support=len(current_window) + len(next_window),
                        properties={
                            "drift_magnitude": drift_magnitude,
                            "space_id": space.space_id,
                            "centroid_before": current_centroid.tolist(),
                            "centroid_after": next_centroid.tolist()
                        }
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_oscillations(self, space: TemporalConceptualSpace) -> List[TemporalPattern]:
        """Detect oscillation patterns in concept positions"""
        patterns = []
        
        # Track concept movements over time
        for concept_id, trajectory in self.concept_trajectories.items():
            if len(trajectory) < 6:  # Need sufficient data points
                continue
            
            # Extract coordinates for each dimension
            for dim in range(trajectory[0][1].shape[0]):
                values = [coords[dim].item() for _, coords in trajectory]
                timestamps = [ts for ts, _ in trajectory]
                
                # Detect oscillations using autocorrelation
                if self._is_oscillating(values):
                    # Calculate period
                    period = self._estimate_period(values, timestamps)
                    
                    if period:
                        pattern = TemporalPattern(
                            pattern_id=f"oscillation_{concept_id}_{dim}",
                            pattern_type="oscillation",
                            entities=[concept_id],
                            relations=[],
                            time_intervals=[TimeInterval(
                                start=timestamps[0],
                                end=timestamps[-1],
                                duration=period
                            )],
                            confidence=0.8,  # Could be improved with better analysis
                            support=len(trajectory),
                            properties={
                                "dimension": dim,
                                "period": period.total_seconds(),
                                "amplitude": max(values) - min(values),
                                "space_id": space.space_id
                            }
                        )
                        patterns.append(pattern)
        
        return patterns
    
    def _detect_temporal_clustering(self, space: TemporalConceptualSpace) -> List[TemporalPattern]:
        """Detect temporal clustering patterns"""
        patterns = []
        
        # Group concepts by time periods
        time_clusters = defaultdict(list)
        cluster_window = timedelta(hours=1)  # 1-hour clustering window
        
        sorted_concepts = sorted(space.concepts.values(), key=lambda c: c.timestamp)
        
        current_cluster = []
        current_cluster_start = None
        
        for concept in sorted_concepts:
            if not current_cluster:
                current_cluster = [concept]
                current_cluster_start = concept.timestamp
            elif concept.timestamp - current_cluster_start <= cluster_window:
                current_cluster.append(concept)
            else:
                # Process current cluster
                if len(current_cluster) >= 3:  # Minimum cluster size
                    cluster_id = f"cluster_{current_cluster_start.isoformat()}"
                    time_clusters[cluster_id] = current_cluster
                
                # Start new cluster
                current_cluster = [concept]
                current_cluster_start = concept.timestamp
        
        # Process last cluster
        if len(current_cluster) >= 3:
            cluster_id = f"cluster_{current_cluster_start.isoformat()}"
            time_clusters[cluster_id] = current_cluster
        
        # Analyze clusters for patterns
        for cluster_id, concepts in time_clusters.items():
            if len(concepts) >= 3:
                # Calculate spatial clustering
                coordinates = torch.stack([c.coordinates for c in concepts])
                centroid = torch.mean(coordinates, dim=0)
                distances = [torch.norm(coords - centroid).item() for coords in coordinates]
                avg_distance = np.mean(distances)
                
                if avg_distance < 0.5:  # Threshold for spatial clustering
                    pattern = TemporalPattern(
                        pattern_id=f"temporal_cluster_{cluster_id}",
                        pattern_type="clustering",
                        entities=[c.concept_id for c in concepts],
                        relations=[],
                        time_intervals=[TimeInterval(
                            start=concepts[0].timestamp,
                            end=concepts[-1].timestamp,
                            duration=concepts[-1].timestamp - concepts[0].timestamp
                        )],
                        confidence=1.0 - (avg_distance / 1.0),  # Normalize confidence
                        support=len(concepts),
                        properties={
                            "spatial_compactness": avg_distance,
                            "temporal_compactness": (concepts[-1].timestamp - concepts[0].timestamp).total_seconds(),
                            "centroid": centroid.tolist(),
                            "space_id": space.space_id
                        }
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_concept_emergence(self, space: TemporalConceptualSpace) -> List[TemporalPattern]:
        """Detect concept emergence patterns"""
        patterns = []
        
        # Analyze concept density over time
        time_windows = defaultdict(list)
        window_size = timedelta(days=1)  # Daily windows
        
        for concept in space.concepts.values():
            window_start = concept.timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            time_windows[window_start].append(concept)
        
        # Look for sudden increases in concept density
        sorted_windows = sorted(time_windows.keys())
        
        for i in range(1, len(sorted_windows)):
            current_count = len(time_windows[sorted_windows[i]])
            previous_count = len(time_windows[sorted_windows[i - 1]])
            
            # Check for significant increase
            if current_count > previous_count * 2 and current_count >= 5:
                pattern = TemporalPattern(
                    pattern_id=f"emergence_{space.space_id}_{i}",
                    pattern_type="emergence",
                    entities=[c.concept_id for c in time_windows[sorted_windows[i]]],
                    relations=[],
                    time_intervals=[TimeInterval(
                        start=sorted_windows[i],
                        end=sorted_windows[i] + window_size,
                        duration=window_size
                    )],
                    confidence=min(1.0, current_count / (previous_count + 1)),
                    support=current_count,
                    properties={
                        "emergence_factor": current_count / (previous_count + 1),
                        "concept_count": current_count,
                        "previous_count": previous_count,
                        "space_id": space.space_id
                    }
                )
                patterns.append(pattern)
        
        return patterns
    
    def _create_temporal_embedding(self, coordinates: torch.Tensor, timestamp: datetime) -> torch.Tensor:
        """Create temporal embedding combining spatial and temporal features"""
        # Encode timestamp
        time_encoding = self.time_encoder.encode(timestamp)
        
        # Combine spatial coordinates with temporal encoding
        temporal_embedding = torch.cat([coordinates, time_encoding])
        
        return temporal_embedding
    
    def _is_oscillating(self, values: List[float], threshold: float = 0.3) -> bool:
        """Check if a sequence of values shows oscillating behavior"""
        if len(values) < 6:
            return False
        
        # Calculate autocorrelation
        n = len(values)
        mean_val = np.mean(values)
        variance = np.var(values)
        
        if variance == 0:
            return False
        
        # Check for periodic behavior
        autocorr_values = []
        for lag in range(1, min(n // 2, 10)):
            autocorr = sum((values[i] - mean_val) * (values[i + lag] - mean_val) 
                          for i in range(n - lag)) / ((n - lag) * variance)
            autocorr_values.append(autocorr)
        
        # Look for peaks in autocorrelation
        max_autocorr = max(autocorr_values) if autocorr_values else 0
        return max_autocorr > threshold
    
    def _estimate_period(self, values: List[float], timestamps: List[datetime]) -> Optional[timedelta]:
        """Estimate the period of oscillation"""
        if len(values) < 6:
            return None
        
        # Find peaks in the signal
        peaks = []
        for i in range(1, len(values) - 1):
            if values[i] > values[i - 1] and values[i] > values[i + 1]:
                peaks.append(i)
        
        if len(peaks) < 2:
            return None
        
        # Calculate average time between peaks
        peak_intervals = []
        for i in range(1, len(peaks)):
            interval = timestamps[peaks[i]] - timestamps[peaks[i - 1]]
            peak_intervals.append(interval)
        
        if peak_intervals:
            avg_interval = sum(peak_intervals, timedelta()) / len(peak_intervals)
            return avg_interval
        
        return None
    
    def get_temporal_space_summary(self, space_id: str) -> Dict[str, Any]:
        """Get comprehensive summary of temporal conceptual space"""
        if space_id not in self.temporal_spaces:
            return {}
        
        space = self.temporal_spaces[space_id]
        
        # Calculate statistics
        concept_count = len(space.concepts)
        if concept_count == 0:
            return {"space_id": space_id, "concept_count": 0}
        
        # Temporal span
        timestamps = [c.timestamp for c in space.concepts.values()]
        temporal_span = max(timestamps) - min(timestamps)
        
        # Dimension statistics
        dimension_stats = {}
        for dim_name in space.dimensions:
            dim_key = f"{space_id}_{dim_name}"
            if dim_key in self.temporal_dimensions:
                dim = self.temporal_dimensions[dim_key]
                dimension_stats[dim_name] = {
                    "trend_direction": dim.trend_direction,
                    "volatility": dim.volatility,
                    "value_count": len(dim.value_history),
                    "latest_value": dim.value_history[-1][1] if dim.value_history else None
                }
        
        # Pattern statistics
        space_patterns = [p for p in self.temporal_patterns 
                         if p.properties.get("space_id") == space_id]
        
        return {
            "space_id": space_id,
            "domain": space.domain,
            "concept_count": concept_count,
            "dimension_count": len(space.dimensions),
            "temporal_span_days": temporal_span.days,
            "creation_time": space.creation_time.isoformat(),
            "dimension_statistics": dimension_stats,
            "pattern_count": len(space_patterns),
            "pattern_types": list(set(p.pattern_type for p in space_patterns)),
            "trajectory_count": len([t for t in self.concept_trajectories.keys() 
                                   if any(c.concept_id == t for c in space.concepts.values())])
        }


class TemporalEncoder:
    """Encodes temporal information for conceptual spaces"""
    
    def __init__(self, encoding_dim: int = 32):
        self.encoding_dim = encoding_dim
        
        # Learnable parameters for different time scales
        self.hour_weights = torch.randn(1, encoding_dim // 4)
        self.day_weights = torch.randn(1, encoding_dim // 4)
        self.month_weights = torch.randn(1, encoding_dim // 4)
        self.year_weights = torch.randn(1, encoding_dim // 4)
    
    def encode(self, timestamp: datetime) -> torch.Tensor:
        """Encode timestamp into temporal features"""
        # Extract time components
        hour = timestamp.hour / 24.0
        day = timestamp.day / 31.0
        month = timestamp.month / 12.0
        year = (timestamp.year - 2020) / 10.0  # Normalize around 2020
        
        # Create component tensors
        hour_tensor = torch.tensor([[hour]], dtype=torch.float32)
        day_tensor = torch.tensor([[day]], dtype=torch.float32)
        month_tensor = torch.tensor([[month]], dtype=torch.float32)
        year_tensor = torch.tensor([[year]], dtype=torch.float32)
        
        # Apply transformations
        hour_encoding = torch.sin(torch.matmul(hour_tensor, self.hour_weights))
        day_encoding = torch.sin(torch.matmul(day_tensor, self.day_weights))
        month_encoding = torch.sin(torch.matmul(month_tensor, self.month_weights))
        year_encoding = torch.sin(torch.matmul(year_tensor, self.year_weights))
        
        # Concatenate encodings
        temporal_encoding = torch.cat([
            hour_encoding, day_encoding, month_encoding, year_encoding
        ], dim=1)
        
        return temporal_encoding.squeeze(0)


class TemporalPatternDetector:
    """Detects temporal patterns in conceptual spaces"""
    
    def __init__(self):
        self.pattern_cache = {}
        self.detection_threshold = 0.5
        self.min_pattern_length = 3