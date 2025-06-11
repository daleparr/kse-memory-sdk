"""
Federated learning module for KSE Memory SDK

This module provides comprehensive federated learning capabilities for KSE,
including differential privacy, secure aggregation, and distributed training.
"""

from .federated_models import (
    FederationConfig,
    FederationRole,
    PrivacyLevel,
    ModelUpdate,
    PrivateModelUpdate,
    EncryptedUpdate,
    EncryptedAggregate,
    FederationRoundResult,
    FederatedKnowledgeTransfer,
    FederatedBenchmarkResult,
    FederatedDataset,
    FederatedMetrics,
    FederatedSecurityAudit
)

from .privacy import (
    PrivacyBudget,
    DifferentialPrivacyMechanism,
    SecureAggregation,
    PrivacyAccountant,
    PrivacyAuditor,
    PrivacyPreservingKSE,
    create_private_update
)

from .federated_client import FederatedKSEClient
from .federated_coordinator import FederatedCoordinator, create_coordinator

__all__ = [
    # Core models
    "FederationConfig",
    "FederationRole", 
    "PrivacyLevel",
    "ModelUpdate",
    "PrivateModelUpdate",
    "EncryptedUpdate",
    "EncryptedAggregate",
    "FederationRoundResult",
    "FederatedKnowledgeTransfer",
    "FederatedBenchmarkResult",
    "FederatedDataset",
    "FederatedMetrics",
    "FederatedSecurityAudit",
    
    # Privacy components
    "PrivacyBudget",
    "DifferentialPrivacyMechanism",
    "SecureAggregation",
    "PrivacyAccountant",
    "PrivacyAuditor",
    "PrivacyPreservingKSE",
    "create_private_update",
    
    # Federated components
    "FederatedKSEClient",
    "FederatedCoordinator",
    "create_coordinator"
]

# Version info
__version__ = "1.0.0"
__author__ = "KSE Memory SDK Team"
__description__ = "Federated learning for Knowledge Space Embeddings"

# Module-level configuration
DEFAULT_PRIVACY_EPSILON = 0.3
DEFAULT_PRIVACY_DELTA = 1e-5
DEFAULT_FEDERATION_ROUNDS = 50
DEFAULT_LOCAL_EPOCHS = 5
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001

# Privacy levels mapping
PRIVACY_LEVELS = {
    "none": PrivacyLevel.NONE,
    "basic": PrivacyLevel.BASIC,
    "differential_privacy": PrivacyLevel.DIFFERENTIAL_PRIVACY,
    "secure_aggregation": PrivacyLevel.SECURE_AGGREGATION,
    "full_privacy": PrivacyLevel.FULL_PRIVACY
}

# Federation roles mapping
FEDERATION_ROLES = {
    "coordinator": FederationRole.COORDINATOR,
    "participant": FederationRole.PARTICIPANT,
    "aggregator": FederationRole.AGGREGATOR
}


def create_federation_config(
    node_id: str,
    federation_id: str,
    role: str = "participant",
    privacy_level: str = "differential_privacy",
    privacy_epsilon: float = DEFAULT_PRIVACY_EPSILON,
    privacy_delta: float = DEFAULT_PRIVACY_DELTA,
    coordinator_endpoint: str = "http://localhost:8080",
    communication_rounds: int = DEFAULT_FEDERATION_ROUNDS,
    local_epochs: int = DEFAULT_LOCAL_EPOCHS,
    **kwargs
) -> FederationConfig:
    """
    Create a federation configuration with sensible defaults.
    
    Args:
        node_id: Unique identifier for this node
        federation_id: Identifier for the federation
        role: Role in federation ("coordinator", "participant", "aggregator")
        privacy_level: Privacy level ("none", "basic", "differential_privacy", 
                      "secure_aggregation", "full_privacy")
        privacy_epsilon: Privacy epsilon parameter
        privacy_delta: Privacy delta parameter
        coordinator_endpoint: URL of the federation coordinator
        communication_rounds: Number of communication rounds
        local_epochs: Number of local training epochs per round
        **kwargs: Additional configuration parameters
    
    Returns:
        FederationConfig: Configured federation settings
    """
    return FederationConfig(
        node_id=node_id,
        federation_id=federation_id,
        role=FEDERATION_ROLES[role],
        privacy_level=PRIVACY_LEVELS[privacy_level],
        privacy_epsilon=privacy_epsilon,
        privacy_delta=privacy_delta,
        coordinator_endpoint=coordinator_endpoint,
        communication_rounds=communication_rounds,
        local_epochs=local_epochs,
        **kwargs
    )


def create_privacy_mechanism(
    epsilon: float = DEFAULT_PRIVACY_EPSILON,
    delta: float = DEFAULT_PRIVACY_DELTA,
    sensitivity: float = 1.0
) -> DifferentialPrivacyMechanism:
    """
    Create a differential privacy mechanism with default parameters.
    
    Args:
        epsilon: Privacy epsilon parameter
        delta: Privacy delta parameter  
        sensitivity: Sensitivity of the function
    
    Returns:
        DifferentialPrivacyMechanism: Configured privacy mechanism
    """
    return DifferentialPrivacyMechanism(epsilon, delta, sensitivity)


def create_secure_aggregation(key_size: int = 2048) -> SecureAggregation:
    """
    Create a secure aggregation mechanism.
    
    Args:
        key_size: RSA key size for encryption
    
    Returns:
        SecureAggregation: Configured secure aggregation
    """
    return SecureAggregation(key_size)


# Convenience functions for common federated learning scenarios

async def quick_federated_training(
    kse_memory,
    node_id: str,
    federation_id: str,
    coordinator_url: str = "http://localhost:8080",
    privacy_level: str = "differential_privacy",
    rounds: int = 10
) -> dict:
    """
    Quick setup for federated training with minimal configuration.
    
    Args:
        kse_memory: KSE Memory instance
        node_id: Unique node identifier
        federation_id: Federation identifier
        coordinator_url: Coordinator endpoint URL
        privacy_level: Privacy protection level
        rounds: Number of training rounds
    
    Returns:
        dict: Training results
    """
    config = create_federation_config(
        node_id=node_id,
        federation_id=federation_id,
        coordinator_endpoint=coordinator_url,
        privacy_level=privacy_level,
        communication_rounds=rounds
    )
    
    async with FederatedKSEClient(config, kse_memory) as client:
        return await client.start_federated_training()


def get_privacy_summary(privacy_level: str) -> dict:
    """
    Get a summary of privacy guarantees for a given privacy level.
    
    Args:
        privacy_level: Privacy level string
    
    Returns:
        dict: Privacy guarantees and features
    """
    level = PRIVACY_LEVELS.get(privacy_level, PrivacyLevel.NONE)
    
    summaries = {
        PrivacyLevel.NONE: {
            "differential_privacy": False,
            "secure_aggregation": False,
            "encryption": False,
            "gradient_clipping": False,
            "privacy_guarantee": "No privacy protection",
            "use_case": "Public data or testing environments"
        },
        PrivacyLevel.BASIC: {
            "differential_privacy": False,
            "secure_aggregation": False,
            "encryption": True,
            "gradient_clipping": True,
            "privacy_guarantee": "Basic encryption and gradient clipping",
            "use_case": "Semi-trusted environments"
        },
        PrivacyLevel.DIFFERENTIAL_PRIVACY: {
            "differential_privacy": True,
            "secure_aggregation": False,
            "encryption": True,
            "gradient_clipping": True,
            "privacy_guarantee": "Formal differential privacy guarantees",
            "use_case": "Sensitive data with trusted aggregator"
        },
        PrivacyLevel.SECURE_AGGREGATION: {
            "differential_privacy": False,
            "secure_aggregation": True,
            "encryption": True,
            "gradient_clipping": True,
            "privacy_guarantee": "Secure multi-party computation",
            "use_case": "Untrusted aggregator environments"
        },
        PrivacyLevel.FULL_PRIVACY: {
            "differential_privacy": True,
            "secure_aggregation": True,
            "encryption": True,
            "gradient_clipping": True,
            "privacy_guarantee": "Maximum privacy protection",
            "use_case": "Highly sensitive data and untrusted parties"
        }
    }
    
    return summaries.get(level, summaries[PrivacyLevel.NONE])


def estimate_privacy_budget(
    num_rounds: int,
    epsilon_total: float = DEFAULT_PRIVACY_EPSILON,
    delta_total: float = DEFAULT_PRIVACY_DELTA
) -> dict:
    """
    Estimate privacy budget allocation for federated training.
    
    Args:
        num_rounds: Number of training rounds
        epsilon_total: Total privacy epsilon budget
        delta_total: Total privacy delta budget
    
    Returns:
        dict: Privacy budget breakdown
    """
    epsilon_per_round = epsilon_total / num_rounds
    delta_per_round = delta_total / num_rounds
    
    return {
        "total_epsilon": epsilon_total,
        "total_delta": delta_total,
        "epsilon_per_round": epsilon_per_round,
        "delta_per_round": delta_per_round,
        "num_rounds": num_rounds,
        "privacy_guarantee": f"({epsilon_total}, {delta_total})-differential privacy",
        "recommendation": (
            "Strong privacy" if epsilon_total <= 1.0 else
            "Moderate privacy" if epsilon_total <= 5.0 else
            "Weak privacy"
        )
    }


# Module initialization
def _initialize_module():
    """Initialize the federated learning module"""
    import logging
    
    logger = logging.getLogger(__name__)
    logger.info("KSE Federated Learning Module initialized")
    logger.info(f"Default privacy: ε={DEFAULT_PRIVACY_EPSILON}, δ={DEFAULT_PRIVACY_DELTA}")
    logger.info(f"Supported privacy levels: {list(PRIVACY_LEVELS.keys())}")
    logger.info(f"Supported federation roles: {list(FEDERATION_ROLES.keys())}")


# Initialize on import
_initialize_module()