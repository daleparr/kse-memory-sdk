"""
Data models for federated learning in KSE Memory SDK
"""

import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum


class FederationRole(Enum):
    """Roles in federated learning"""
    COORDINATOR = "coordinator"
    PARTICIPANT = "participant"
    AGGREGATOR = "aggregator"


class PrivacyLevel(Enum):
    """Privacy levels for federated learning"""
    NONE = "none"
    BASIC = "basic"
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    SECURE_AGGREGATION = "secure_aggregation"
    FULL_PRIVACY = "full_privacy"


@dataclass
class FederationConfig:
    """Configuration for federated KSE setup"""
    
    # Node identification
    node_id: str
    federation_id: str
    role: FederationRole
    
    # Privacy settings
    privacy_level: PrivacyLevel = PrivacyLevel.DIFFERENTIAL_PRIVACY
    privacy_epsilon: float = 0.3
    privacy_delta: float = 1e-5
    
    # Communication settings
    coordinator_endpoint: str = "http://localhost:8080"
    communication_rounds: int = 50
    local_epochs: int = 5
    
    # Model settings
    model_architecture: Dict[str, Any] = field(default_factory=dict)
    aggregation_method: str = "fedavg"
    
    # Security settings
    use_encryption: bool = True
    encryption_scheme: str = "paillier"
    key_size: int = 2048
    
    # Performance settings
    batch_size: int = 32
    learning_rate: float = 0.001
    max_gradient_norm: float = 1.0
    
    # Local KSE configuration
    local_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelUpdate:
    """Represents a model update from local training"""
    
    # Update components
    kg_update: torch.Tensor
    cs_update: torch.Tensor
    embedding_update: torch.Tensor
    
    # Metadata
    node_id: str
    round_number: int
    local_epochs: int
    sample_count: int
    loss_value: float
    
    # Timing information
    training_time_ms: float
    communication_time_ms: float = 0.0
    
    # Quality metrics
    local_accuracy: float = 0.0
    gradient_norm: float = 0.0
    
    def get_total_parameters(self) -> int:
        """Get total number of parameters in update"""
        return (self.kg_update.numel() + 
                self.cs_update.numel() + 
                self.embedding_update.numel())
    
    def get_update_size_mb(self) -> float:
        """Get size of update in megabytes"""
        total_elements = self.get_total_parameters()
        # Assuming float32 (4 bytes per parameter)
        size_bytes = total_elements * 4
        return size_bytes / (1024 * 1024)


@dataclass
class PrivateModelUpdate:
    """Model update with differential privacy applied"""
    
    # Privatized updates
    kg_update: torch.Tensor
    cs_update: torch.Tensor
    embedding_update: torch.Tensor
    
    # Privacy metadata
    privacy_budget_used: float
    noise_scale: float
    clipping_norm: float
    
    # Original metadata
    node_id: str
    round_number: int
    sample_count: int
    
    # Privacy guarantees
    epsilon: float
    delta: float
    
    def get_privacy_cost(self) -> Dict[str, float]:
        """Get privacy cost breakdown"""
        return {
            "epsilon_used": self.privacy_budget_used,
            "total_epsilon": self.epsilon,
            "remaining_budget": self.epsilon - self.privacy_budget_used,
            "delta": self.delta
        }


@dataclass
class EncryptedUpdate:
    """Encrypted model update for secure aggregation"""
    
    # Encrypted components
    encrypted_kg: bytes
    encrypted_cs: bytes
    encrypted_embedding: bytes
    
    # Encryption metadata
    node_id: str
    encryption_scheme: str
    public_key_hash: str
    
    # Integrity verification
    checksum: str
    signature: Optional[str] = None
    
    # Size information
    original_size_mb: float = 0.0
    encrypted_size_mb: float = 0.0
    
    def get_compression_ratio(self) -> float:
        """Get compression ratio of encryption"""
        if self.original_size_mb == 0:
            return 1.0
        return self.encrypted_size_mb / self.original_size_mb


@dataclass
class EncryptedAggregate:
    """Aggregated encrypted updates"""
    
    # Aggregated encrypted components
    aggregated_kg: bytes
    aggregated_cs: bytes
    aggregated_embedding: bytes
    
    # Aggregation metadata
    participant_count: int
    round_number: int
    aggregation_method: str
    
    # Verification
    aggregate_checksum: str
    participant_checksums: List[str]


@dataclass
class FederationRoundResult:
    """Result of a federation training round"""
    
    # Round information
    round_number: int
    participants: int
    successful_updates: int
    
    # Performance metrics
    global_loss: float
    global_accuracy: float
    convergence_metric: float
    
    # Communication metrics
    total_communication_mb: float
    round_duration_seconds: float
    
    # Privacy metrics
    privacy_budget_remaining: float
    privacy_violations: int = 0
    
    # Model quality
    model_drift: float = 0.0
    participant_similarity: float = 0.0
    
    def get_success_rate(self) -> float:
        """Get success rate of the round"""
        if self.participants == 0:
            return 0.0
        return self.successful_updates / self.participants


@dataclass
class FederatedKnowledgeTransfer:
    """Represents knowledge transfer between federated nodes"""
    
    # Transfer information
    source_node: str
    target_node: str
    knowledge_type: str  # "kg_patterns", "cs_dimensions", "embeddings"
    
    # Transfer content
    transferred_knowledge: Dict[str, Any]
    transfer_quality: float
    
    # Privacy preservation
    privacy_preserved: bool
    anonymization_level: float
    
    # Performance impact
    transfer_time_ms: float
    knowledge_improvement: float


@dataclass
class FederatedBenchmarkResult:
    """Results from federated learning benchmarks"""
    
    # Test configuration
    test_name: str
    participant_count: int
    rounds_completed: int
    
    # Performance results
    final_accuracy: float
    convergence_round: int
    communication_efficiency: float
    
    # Privacy results
    privacy_preservation_score: float
    membership_inference_resistance: float
    attribute_inference_resistance: float
    
    # Comparison with centralized
    centralized_accuracy: float
    federated_advantage: float
    
    def get_privacy_summary(self) -> Dict[str, float]:
        """Get summary of privacy metrics"""
        return {
            "overall_privacy": self.privacy_preservation_score,
            "membership_resistance": self.membership_inference_resistance,
            "attribute_resistance": self.attribute_inference_resistance,
            "average_resistance": (
                self.membership_inference_resistance + 
                self.attribute_inference_resistance
            ) / 2
        }


class FederatedDataset:
    """Represents a federated dataset partition"""
    
    def __init__(self, node_id: str, data: List[Dict[str, Any]], 
                 is_iid: bool = False):
        self.node_id = node_id
        self.data = data
        self.is_iid = is_iid
        self.created_at = datetime.now()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        if not self.data:
            return {"size": 0, "categories": []}
        
        categories = set()
        for item in self.data:
            if "category" in item:
                categories.add(item["category"])
        
        return {
            "size": len(self.data),
            "categories": list(categories),
            "is_iid": self.is_iid,
            "created_at": self.created_at.isoformat()
        }
    
    def get_sample(self, n: int) -> List[Dict[str, Any]]:
        """Get a sample of n items from the dataset"""
        import random
        return random.sample(self.data, min(n, len(self.data)))


@dataclass
class FederatedMetrics:
    """Comprehensive metrics for federated learning"""
    
    # Training metrics
    local_loss: float
    local_accuracy: float
    global_loss: float
    global_accuracy: float
    
    # Communication metrics
    bytes_sent: int
    bytes_received: int
    round_trip_time_ms: float
    
    # Privacy metrics
    epsilon_consumed: float
    delta_consumed: float
    privacy_budget_remaining: float
    
    # Efficiency metrics
    training_time_ms: float
    communication_time_ms: float
    total_time_ms: float
    
    # Quality metrics
    model_similarity: float
    knowledge_transfer_rate: float
    convergence_speed: float
    
    def get_efficiency_ratio(self) -> float:
        """Get ratio of training time to total time"""
        if self.total_time_ms == 0:
            return 0.0
        return self.training_time_ms / self.total_time_ms
    
    def get_communication_overhead(self) -> float:
        """Get communication overhead as percentage"""
        if self.total_time_ms == 0:
            return 0.0
        return (self.communication_time_ms / self.total_time_ms) * 100


class FederatedSecurityAudit:
    """Security audit results for federated learning"""
    
    def __init__(self):
        self.audit_timestamp = datetime.now()
        self.vulnerabilities = []
        self.privacy_violations = []
        self.security_score = 0.0
    
    def add_vulnerability(self, severity: str, description: str, 
                         recommendation: str):
        """Add a security vulnerability"""
        self.vulnerabilities.append({
            "severity": severity,
            "description": description,
            "recommendation": recommendation,
            "timestamp": datetime.now().isoformat()
        })
    
    def add_privacy_violation(self, violation_type: str, details: str):
        """Add a privacy violation"""
        self.privacy_violations.append({
            "type": violation_type,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })
    
    def calculate_security_score(self) -> float:
        """Calculate overall security score"""
        base_score = 100.0
        
        # Deduct points for vulnerabilities
        for vuln in self.vulnerabilities:
            if vuln["severity"] == "critical":
                base_score -= 20
            elif vuln["severity"] == "high":
                base_score -= 10
            elif vuln["severity"] == "medium":
                base_score -= 5
            elif vuln["severity"] == "low":
                base_score -= 2
        
        # Deduct points for privacy violations
        base_score -= len(self.privacy_violations) * 15
        
        self.security_score = max(0.0, base_score)
        return self.security_score
    
    def get_audit_report(self) -> Dict[str, Any]:
        """Get comprehensive audit report"""
        return {
            "audit_timestamp": self.audit_timestamp.isoformat(),
            "security_score": self.calculate_security_score(),
            "vulnerability_count": len(self.vulnerabilities),
            "privacy_violation_count": len(self.privacy_violations),
            "vulnerabilities": self.vulnerabilities,
            "privacy_violations": self.privacy_violations,
            "recommendations": self._get_recommendations()
        }
    
    def _get_recommendations(self) -> List[str]:
        """Get security recommendations"""
        recommendations = []
        
        if len(self.vulnerabilities) > 0:
            recommendations.append("Address identified security vulnerabilities")
        
        if len(self.privacy_violations) > 0:
            recommendations.append("Implement stronger privacy protections")
        
        if self.security_score < 80:
            recommendations.append("Conduct comprehensive security review")
        
        return recommendations