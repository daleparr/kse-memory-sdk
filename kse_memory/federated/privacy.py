"""
Differential privacy implementation for federated KSE learning
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization
import secrets
import hashlib

from .federated_models import PrivateModelUpdate, PrivacyLevel


logger = logging.getLogger(__name__)


@dataclass
class PrivacyBudget:
    """Manages privacy budget for differential privacy"""
    
    total_epsilon: float
    total_delta: float
    consumed_epsilon: float = 0.0
    consumed_delta: float = 0.0
    
    def can_spend(self, epsilon: float, delta: float) -> bool:
        """Check if we can spend the given privacy budget"""
        return (self.consumed_epsilon + epsilon <= self.total_epsilon and
                self.consumed_delta + delta <= self.total_delta)
    
    def spend(self, epsilon: float, delta: float) -> bool:
        """Spend privacy budget if available"""
        if self.can_spend(epsilon, delta):
            self.consumed_epsilon += epsilon
            self.consumed_delta += delta
            return True
        return False
    
    def remaining_budget(self) -> Tuple[float, float]:
        """Get remaining privacy budget"""
        return (self.total_epsilon - self.consumed_epsilon,
                self.total_delta - self.consumed_delta)
    
    def budget_fraction_used(self) -> float:
        """Get fraction of budget used (0-1)"""
        epsilon_fraction = self.consumed_epsilon / self.total_epsilon
        delta_fraction = self.consumed_delta / self.total_delta
        return max(epsilon_fraction, delta_fraction)


class DifferentialPrivacyMechanism:
    """Implements differential privacy mechanisms for KSE"""
    
    def __init__(self, epsilon: float = 0.3, delta: float = 1e-5,
                 sensitivity: float = 1.0):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.privacy_budget = PrivacyBudget(epsilon, delta)
        
    def gaussian_mechanism(self, data: torch.Tensor, 
                          epsilon: float, delta: float) -> torch.Tensor:
        """Apply Gaussian mechanism for differential privacy"""
        if not self.privacy_budget.can_spend(epsilon, delta):
            raise ValueError("Insufficient privacy budget")
        
        # Calculate noise scale for Gaussian mechanism
        # σ ≥ √(2 ln(1.25/δ)) * Δf / ε
        noise_scale = np.sqrt(2 * np.log(1.25 / delta)) * self.sensitivity / epsilon
        
        # Add Gaussian noise
        noise = torch.normal(0, noise_scale, data.shape)
        private_data = data + noise
        
        # Update privacy budget
        self.privacy_budget.spend(epsilon, delta)
        
        logger.info(f"Applied Gaussian mechanism with σ={noise_scale:.4f}")
        return private_data
    
    def laplace_mechanism(self, data: torch.Tensor, epsilon: float) -> torch.Tensor:
        """Apply Laplace mechanism for differential privacy"""
        if not self.privacy_budget.can_spend(epsilon, 0):
            raise ValueError("Insufficient privacy budget")
        
        # Calculate noise scale for Laplace mechanism
        # b = Δf / ε
        noise_scale = self.sensitivity / epsilon
        
        # Add Laplace noise
        noise = torch.distributions.Laplace(0, noise_scale).sample(data.shape)
        private_data = data + noise
        
        # Update privacy budget
        self.privacy_budget.spend(epsilon, 0)
        
        logger.info(f"Applied Laplace mechanism with b={noise_scale:.4f}")
        return private_data
    
    def gradient_clipping(self, gradients: torch.Tensor, 
                         max_norm: float) -> torch.Tensor:
        """Clip gradients to bound sensitivity"""
        grad_norm = torch.norm(gradients)
        if grad_norm > max_norm:
            gradients = gradients * (max_norm / grad_norm)
        return gradients
    
    def private_aggregation(self, updates: List[torch.Tensor],
                           epsilon: float, delta: float) -> torch.Tensor:
        """Aggregate updates with differential privacy"""
        # Stack updates
        stacked_updates = torch.stack(updates)
        
        # Compute mean
        mean_update = torch.mean(stacked_updates, dim=0)
        
        # Apply privacy mechanism
        private_mean = self.gaussian_mechanism(mean_update, epsilon, delta)
        
        return private_mean


class SecureAggregation:
    """Implements secure aggregation for federated learning"""
    
    def __init__(self, key_size: int = 2048):
        self.key_size = key_size
        self.private_key = None
        self.public_key = None
        self._generate_keys()
    
    def _generate_keys(self):
        """Generate RSA key pair"""
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.key_size
        )
        self.public_key = self.private_key.public_key()
    
    def get_public_key_pem(self) -> bytes:
        """Get public key in PEM format"""
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
    
    def encrypt_tensor(self, tensor: torch.Tensor) -> bytes:
        """Encrypt tensor using public key"""
        # Convert tensor to bytes
        tensor_bytes = tensor.numpy().tobytes()
        
        # Encrypt in chunks (RSA has size limitations)
        chunk_size = self.key_size // 8 - 42  # Account for padding
        encrypted_chunks = []
        
        for i in range(0, len(tensor_bytes), chunk_size):
            chunk = tensor_bytes[i:i + chunk_size]
            encrypted_chunk = self.public_key.encrypt(
                chunk,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            encrypted_chunks.append(encrypted_chunk)
        
        return b''.join(encrypted_chunks)
    
    def decrypt_tensor(self, encrypted_data: bytes, 
                      original_shape: Tuple[int, ...],
                      dtype: torch.dtype) -> torch.Tensor:
        """Decrypt tensor using private key"""
        # Decrypt in chunks
        chunk_size = self.key_size // 8
        decrypted_chunks = []
        
        for i in range(0, len(encrypted_data), chunk_size):
            chunk = encrypted_data[i:i + chunk_size]
            decrypted_chunk = self.private_key.decrypt(
                chunk,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            decrypted_chunks.append(decrypted_chunk)
        
        # Reconstruct tensor
        tensor_bytes = b''.join(decrypted_chunks)
        tensor_array = np.frombuffer(tensor_bytes, dtype=dtype.numpy())
        tensor = torch.from_numpy(tensor_array).reshape(original_shape)
        
        return tensor
    
    def compute_checksum(self, data: bytes) -> str:
        """Compute SHA-256 checksum"""
        return hashlib.sha256(data).hexdigest()
    
    def verify_integrity(self, data: bytes, checksum: str) -> bool:
        """Verify data integrity using checksum"""
        computed_checksum = self.compute_checksum(data)
        return computed_checksum == checksum


class PrivacyAccountant:
    """Tracks privacy expenditure across federated learning"""
    
    def __init__(self, total_epsilon: float, total_delta: float):
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.expenditures = []
        self.current_epsilon = 0.0
        self.current_delta = 0.0
    
    def spend_budget(self, epsilon: float, delta: float, 
                    operation: str) -> bool:
        """Spend privacy budget for an operation"""
        if self.current_epsilon + epsilon > self.total_epsilon:
            logger.warning(f"Epsilon budget exceeded for {operation}")
            return False
        
        if self.current_delta + delta > self.total_delta:
            logger.warning(f"Delta budget exceeded for {operation}")
            return False
        
        # Record expenditure
        self.expenditures.append({
            "operation": operation,
            "epsilon": epsilon,
            "delta": delta,
            "timestamp": torch.tensor(0.0)  # Placeholder for timestamp
        })
        
        self.current_epsilon += epsilon
        self.current_delta += delta
        
        logger.info(f"Privacy budget spent: ε={epsilon:.4f}, δ={delta:.6f} "
                   f"for {operation}")
        return True
    
    def get_remaining_budget(self) -> Tuple[float, float]:
        """Get remaining privacy budget"""
        return (self.total_epsilon - self.current_epsilon,
                self.total_delta - self.current_delta)
    
    def get_budget_summary(self) -> Dict[str, Any]:
        """Get comprehensive budget summary"""
        remaining_eps, remaining_delta = self.get_remaining_budget()
        
        return {
            "total_epsilon": self.total_epsilon,
            "total_delta": self.total_delta,
            "consumed_epsilon": self.current_epsilon,
            "consumed_delta": self.current_delta,
            "remaining_epsilon": remaining_eps,
            "remaining_delta": remaining_delta,
            "budget_fraction_used": self.current_epsilon / self.total_epsilon,
            "expenditure_count": len(self.expenditures),
            "expenditures": self.expenditures
        }


class PrivacyAuditor:
    """Audits privacy guarantees in federated learning"""
    
    def __init__(self):
        self.audit_log = []
    
    def audit_model_update(self, update: PrivateModelUpdate) -> Dict[str, Any]:
        """Audit a private model update"""
        audit_result = {
            "node_id": update.node_id,
            "round_number": update.round_number,
            "privacy_budget_used": update.privacy_budget_used,
            "epsilon": update.epsilon,
            "delta": update.delta,
            "noise_scale": update.noise_scale,
            "clipping_norm": update.clipping_norm,
            "privacy_violations": []
        }
        
        # Check for privacy violations
        if update.privacy_budget_used > update.epsilon:
            audit_result["privacy_violations"].append(
                "Privacy budget exceeded epsilon limit"
            )
        
        if update.noise_scale < self._minimum_noise_scale(update.epsilon, update.delta):
            audit_result["privacy_violations"].append(
                "Insufficient noise for privacy guarantee"
            )
        
        # Check for potential privacy leaks
        if self._detect_privacy_leak(update):
            audit_result["privacy_violations"].append(
                "Potential privacy leak detected"
            )
        
        self.audit_log.append(audit_result)
        return audit_result
    
    def _minimum_noise_scale(self, epsilon: float, delta: float) -> float:
        """Calculate minimum required noise scale"""
        return np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    
    def _detect_privacy_leak(self, update: PrivateModelUpdate) -> bool:
        """Detect potential privacy leaks in update"""
        # Simple heuristic: check if update values are suspiciously large
        kg_max = torch.max(torch.abs(update.kg_update))
        cs_max = torch.max(torch.abs(update.cs_update))
        emb_max = torch.max(torch.abs(update.embedding_update))
        
        # If any component has extremely large values, it might indicate
        # insufficient noise or clipping
        threshold = 100.0  # Configurable threshold
        return kg_max > threshold or cs_max > threshold or emb_max > threshold
    
    def generate_privacy_report(self) -> Dict[str, Any]:
        """Generate comprehensive privacy audit report"""
        total_audits = len(self.audit_log)
        violations = sum(1 for audit in self.audit_log 
                        if audit["privacy_violations"])
        
        return {
            "total_audits": total_audits,
            "privacy_violations": violations,
            "violation_rate": violations / total_audits if total_audits > 0 else 0,
            "audit_log": self.audit_log,
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate privacy recommendations based on audit results"""
        recommendations = []
        
        violation_rate = sum(1 for audit in self.audit_log 
                           if audit["privacy_violations"]) / len(self.audit_log)
        
        if violation_rate > 0.1:
            recommendations.append(
                "High privacy violation rate detected. "
                "Review privacy parameters and mechanisms."
            )
        
        if any("budget exceeded" in str(audit["privacy_violations"]) 
               for audit in self.audit_log):
            recommendations.append(
                "Privacy budget management needs improvement. "
                "Consider increasing total budget or reducing per-round consumption."
            )
        
        return recommendations


def create_private_update(kg_update: torch.Tensor,
                         cs_update: torch.Tensor,
                         embedding_update: torch.Tensor,
                         node_id: str,
                         round_number: int,
                         sample_count: int,
                         privacy_mechanism: DifferentialPrivacyMechanism,
                         epsilon: float,
                         delta: float,
                         clipping_norm: float = 1.0) -> PrivateModelUpdate:
    """Create a private model update with differential privacy"""
    
    # Clip gradients
    kg_clipped = privacy_mechanism.gradient_clipping(kg_update, clipping_norm)
    cs_clipped = privacy_mechanism.gradient_clipping(cs_update, clipping_norm)
    emb_clipped = privacy_mechanism.gradient_clipping(embedding_update, clipping_norm)
    
    # Apply differential privacy
    kg_private = privacy_mechanism.gaussian_mechanism(kg_clipped, epsilon/3, delta/3)
    cs_private = privacy_mechanism.gaussian_mechanism(cs_clipped, epsilon/3, delta/3)
    emb_private = privacy_mechanism.gaussian_mechanism(emb_clipped, epsilon/3, delta/3)
    
    # Calculate noise scale
    noise_scale = np.sqrt(2 * np.log(1.25 / (delta/3))) * clipping_norm / (epsilon/3)
    
    return PrivateModelUpdate(
        kg_update=kg_private,
        cs_update=cs_private,
        embedding_update=emb_private,
        privacy_budget_used=epsilon,
        noise_scale=noise_scale,
        clipping_norm=clipping_norm,
        node_id=node_id,
        round_number=round_number,
        sample_count=sample_count,
        epsilon=epsilon,
        delta=delta
    )


class PrivacyPreservingKSE:
    """Privacy-preserving wrapper for KSE operations"""
    
    def __init__(self, privacy_level: PrivacyLevel = PrivacyLevel.DIFFERENTIAL_PRIVACY,
                 epsilon: float = 0.3, delta: float = 1e-5):
        self.privacy_level = privacy_level
        self.privacy_mechanism = DifferentialPrivacyMechanism(epsilon, delta)
        self.secure_aggregation = SecureAggregation()
        self.privacy_accountant = PrivacyAccountant(epsilon, delta)
        self.privacy_auditor = PrivacyAuditor()
    
    def privatize_query(self, query_embedding: torch.Tensor,
                       epsilon: float) -> torch.Tensor:
        """Add privacy noise to query embedding"""
        if self.privacy_level == PrivacyLevel.NONE:
            return query_embedding
        
        return self.privacy_mechanism.laplace_mechanism(query_embedding, epsilon)
    
    def privatize_results(self, results: List[Dict[str, Any]],
                         k: int, epsilon: float) -> List[Dict[str, Any]]:
        """Privatize search results using exponential mechanism"""
        if self.privacy_level == PrivacyLevel.NONE:
            return results[:k]
        
        # Simplified exponential mechanism for top-k selection
        # In practice, this would be more sophisticated
        scores = [result.get("score", 0.0) for result in results]
        
        # Add noise to scores
        noisy_scores = []
        for score in scores:
            noise = np.random.laplace(0, 2.0 / epsilon)  # Sensitivity = 2
            noisy_scores.append(score + noise)
        
        # Sort by noisy scores and return top-k
        sorted_indices = np.argsort(noisy_scores)[::-1]
        return [results[i] for i in sorted_indices[:k]]
    
    def get_privacy_summary(self) -> Dict[str, Any]:
        """Get comprehensive privacy summary"""
        return {
            "privacy_level": self.privacy_level.value,
            "budget_summary": self.privacy_accountant.get_budget_summary(),
            "audit_report": self.privacy_auditor.generate_privacy_report(),
            "security_features": {
                "differential_privacy": self.privacy_level != PrivacyLevel.NONE,
                "secure_aggregation": self.privacy_level in [
                    PrivacyLevel.SECURE_AGGREGATION, PrivacyLevel.FULL_PRIVACY
                ],
                "encryption": True,
                "gradient_clipping": True
            }
        }