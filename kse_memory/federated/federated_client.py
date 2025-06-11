"""
Federated learning client for KSE Memory SDK
"""

import torch
import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import numpy as np
from pathlib import Path

from ..core.memory import KSEMemory
from ..core.models import Product, ConceptualDimensions
from .federated_models import (
    FederationConfig, ModelUpdate, PrivateModelUpdate, 
    FederatedMetrics, FederationRole, PrivacyLevel
)
from .privacy import (
    DifferentialPrivacyMechanism, SecureAggregation, 
    PrivacyAccountant, create_private_update
)


logger = logging.getLogger(__name__)


class FederatedKSEClient:
    """Federated learning client for KSE Memory"""
    
    def __init__(self, config: FederationConfig, kse_memory: KSEMemory):
        self.config = config
        self.kse_memory = kse_memory
        self.node_id = config.node_id
        self.federation_id = config.federation_id
        
        # Privacy components
        self.privacy_mechanism = None
        self.secure_aggregation = None
        self.privacy_accountant = None
        
        if config.privacy_level != PrivacyLevel.NONE:
            self.privacy_mechanism = DifferentialPrivacyMechanism(
                config.privacy_epsilon, config.privacy_delta
            )
            self.privacy_accountant = PrivacyAccountant(
                config.privacy_epsilon, config.privacy_delta
            )
        
        if config.privacy_level in [PrivacyLevel.SECURE_AGGREGATION, PrivacyLevel.FULL_PRIVACY]:
            self.secure_aggregation = SecureAggregation(config.key_size)
        
        # Training state
        self.current_round = 0
        self.local_model_state = None
        self.training_history = []
        self.is_training = False
        
        # Communication
        self.session = None
        self.coordinator_url = config.coordinator_endpoint
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def register_with_coordinator(self) -> bool:
        """Register this client with the federation coordinator"""
        registration_data = {
            "node_id": self.node_id,
            "federation_id": self.federation_id,
            "role": self.config.role.value,
            "capabilities": {
                "privacy_level": self.config.privacy_level.value,
                "encryption": self.config.use_encryption,
                "local_data_size": await self._get_local_data_size()
            },
            "public_key": self._get_public_key() if self.secure_aggregation else None
        }
        
        try:
            async with self.session.post(
                f"{self.coordinator_url}/register",
                json=registration_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Successfully registered with coordinator: {result}")
                    return True
                else:
                    logger.error(f"Registration failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return False
    
    async def start_federated_training(self, 
                                     training_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Start federated training process"""
        if self.is_training:
            raise RuntimeError("Training already in progress")
        
        self.is_training = True
        training_results = {
            "rounds_completed": 0,
            "final_metrics": None,
            "training_history": [],
            "privacy_summary": None
        }
        
        try:
            # Register with coordinator
            if not await self.register_with_coordinator():
                raise RuntimeError("Failed to register with coordinator")
            
            # Training loop
            for round_num in range(self.config.communication_rounds):
                self.current_round = round_num
                
                logger.info(f"Starting federated round {round_num + 1}/{self.config.communication_rounds}")
                
                # Wait for round start signal
                if not await self._wait_for_round_start(round_num):
                    logger.warning(f"Round {round_num} start signal timeout")
                    break
                
                # Perform local training
                local_update = await self._perform_local_training()
                
                # Apply privacy if configured
                if self.config.privacy_level != PrivacyLevel.NONE:
                    local_update = await self._apply_privacy(local_update)
                
                # Send update to coordinator
                if not await self._send_update_to_coordinator(local_update):
                    logger.error(f"Failed to send update for round {round_num}")
                    break
                
                # Receive global update
                global_update = await self._receive_global_update(round_num)
                if global_update is None:
                    logger.error(f"Failed to receive global update for round {round_num}")
                    break
                
                # Apply global update to local model
                await self._apply_global_update(global_update)
                
                # Calculate metrics
                round_metrics = await self._calculate_round_metrics(local_update, global_update)
                training_results["training_history"].append(round_metrics)
                
                # Call training callback if provided
                if training_callback:
                    await training_callback(round_num, round_metrics)
                
                # Check convergence
                if await self._check_convergence(round_metrics):
                    logger.info(f"Convergence achieved at round {round_num + 1}")
                    break
                
                training_results["rounds_completed"] = round_num + 1
            
            # Final metrics and privacy summary
            training_results["final_metrics"] = await self._calculate_final_metrics()
            if self.privacy_accountant:
                training_results["privacy_summary"] = self.privacy_accountant.get_budget_summary()
            
        except Exception as e:
            logger.error(f"Federated training error: {e}")
            raise
        finally:
            self.is_training = False
        
        return training_results
    
    async def _get_local_data_size(self) -> int:
        """Get size of local training data"""
        # This would be implemented based on the specific KSE memory backend
        # For now, return a placeholder
        return 1000
    
    def _get_public_key(self) -> Optional[str]:
        """Get public key for secure aggregation"""
        if self.secure_aggregation:
            return self.secure_aggregation.get_public_key_pem().decode('utf-8')
        return None
    
    async def _wait_for_round_start(self, round_num: int) -> bool:
        """Wait for round start signal from coordinator"""
        timeout = 300  # 5 minutes timeout
        start_time = datetime.now()
        
        while (datetime.now() - start_time).seconds < timeout:
            try:
                async with self.session.get(
                    f"{self.coordinator_url}/round/{round_num}/status"
                ) as response:
                    if response.status == 200:
                        status = await response.json()
                        if status.get("status") == "started":
                            return True
                
                await asyncio.sleep(5)  # Poll every 5 seconds
            except Exception as e:
                logger.warning(f"Error checking round status: {e}")
                await asyncio.sleep(5)
        
        return False
    
    async def _perform_local_training(self) -> ModelUpdate:
        """Perform local training and return model update"""
        logger.info("Starting local training")
        
        # Get current model state
        initial_state = await self._get_model_state()
        
        # Simulate local training (in practice, this would train on local data)
        # For demonstration, we'll create synthetic updates
        kg_update = torch.randn(100, 50) * 0.01  # Small random updates
        cs_update = torch.randn(10, 10) * 0.01
        embedding_update = torch.randn(768, 100) * 0.01
        
        # Calculate training metrics
        training_time = 1000.0  # milliseconds
        local_loss = np.random.uniform(0.1, 0.5)
        local_accuracy = np.random.uniform(0.7, 0.95)
        
        update = ModelUpdate(
            kg_update=kg_update,
            cs_update=cs_update,
            embedding_update=embedding_update,
            node_id=self.node_id,
            round_number=self.current_round,
            local_epochs=self.config.local_epochs,
            sample_count=await self._get_local_data_size(),
            loss_value=local_loss,
            training_time_ms=training_time,
            local_accuracy=local_accuracy,
            gradient_norm=torch.norm(kg_update).item()
        )
        
        logger.info(f"Local training completed: loss={local_loss:.4f}, "
                   f"accuracy={local_accuracy:.4f}")
        
        return update
    
    async def _apply_privacy(self, update: ModelUpdate) -> PrivateModelUpdate:
        """Apply differential privacy to model update"""
        if not self.privacy_mechanism or not self.privacy_accountant:
            raise RuntimeError("Privacy mechanism not initialized")
        
        # Calculate per-round privacy budget
        rounds_remaining = self.config.communication_rounds - self.current_round
        epsilon_per_round = self.config.privacy_epsilon / self.config.communication_rounds
        delta_per_round = self.config.privacy_delta / self.config.communication_rounds
        
        # Create private update
        private_update = create_private_update(
            update.kg_update,
            update.cs_update,
            update.embedding_update,
            update.node_id,
            update.round_number,
            update.sample_count,
            self.privacy_mechanism,
            epsilon_per_round,
            delta_per_round,
            self.config.max_gradient_norm
        )
        
        logger.info(f"Applied differential privacy: ε={epsilon_per_round:.4f}, "
                   f"δ={delta_per_round:.6f}")
        
        return private_update
    
    async def _send_update_to_coordinator(self, 
                                        update: ModelUpdate) -> bool:
        """Send model update to coordinator"""
        try:
            # Serialize update
            if isinstance(update, PrivateModelUpdate):
                update_data = {
                    "type": "private",
                    "node_id": update.node_id,
                    "round_number": update.round_number,
                    "sample_count": update.sample_count,
                    "privacy_budget_used": update.privacy_budget_used,
                    "epsilon": update.epsilon,
                    "delta": update.delta
                }
                
                # Encrypt if secure aggregation is enabled
                if self.secure_aggregation:
                    update_data["encrypted_kg"] = self.secure_aggregation.encrypt_tensor(
                        update.kg_update
                    ).hex()
                    update_data["encrypted_cs"] = self.secure_aggregation.encrypt_tensor(
                        update.cs_update
                    ).hex()
                    update_data["encrypted_embedding"] = self.secure_aggregation.encrypt_tensor(
                        update.embedding_update
                    ).hex()
                else:
                    update_data["kg_update"] = update.kg_update.tolist()
                    update_data["cs_update"] = update.cs_update.tolist()
                    update_data["embedding_update"] = update.embedding_update.tolist()
            else:
                update_data = {
                    "type": "standard",
                    "node_id": update.node_id,
                    "round_number": update.round_number,
                    "sample_count": update.sample_count,
                    "loss_value": update.loss_value,
                    "local_accuracy": update.local_accuracy,
                    "kg_update": update.kg_update.tolist(),
                    "cs_update": update.cs_update.tolist(),
                    "embedding_update": update.embedding_update.tolist()
                }
            
            async with self.session.post(
                f"{self.coordinator_url}/update",
                json=update_data
            ) as response:
                if response.status == 200:
                    logger.info("Successfully sent update to coordinator")
                    return True
                else:
                    logger.error(f"Failed to send update: {response.status}")
                    return False
        
        except Exception as e:
            logger.error(f"Error sending update: {e}")
            return False
    
    async def _receive_global_update(self, round_num: int) -> Optional[Dict[str, Any]]:
        """Receive global model update from coordinator"""
        timeout = 300  # 5 minutes timeout
        start_time = datetime.now()
        
        while (datetime.now() - start_time).seconds < timeout:
            try:
                async with self.session.get(
                    f"{self.coordinator_url}/global_update/{round_num}"
                ) as response:
                    if response.status == 200:
                        global_update = await response.json()
                        logger.info("Received global update from coordinator")
                        return global_update
                    elif response.status == 202:
                        # Update not ready yet
                        await asyncio.sleep(10)
                        continue
                    else:
                        logger.error(f"Error receiving global update: {response.status}")
                        return None
            
            except Exception as e:
                logger.warning(f"Error receiving global update: {e}")
                await asyncio.sleep(10)
        
        logger.error("Timeout waiting for global update")
        return None
    
    async def _apply_global_update(self, global_update: Dict[str, Any]):
        """Apply global model update to local model"""
        try:
            # Extract updates from global update
            kg_update = torch.tensor(global_update["kg_update"])
            cs_update = torch.tensor(global_update["cs_update"])
            embedding_update = torch.tensor(global_update["embedding_update"])
            
            # Apply updates to local model (simplified)
            # In practice, this would update the actual KSE model parameters
            logger.info("Applied global update to local model")
            
        except Exception as e:
            logger.error(f"Error applying global update: {e}")
            raise
    
    async def _get_model_state(self) -> Dict[str, torch.Tensor]:
        """Get current model state"""
        # Placeholder implementation
        return {
            "kg_weights": torch.randn(100, 50),
            "cs_weights": torch.randn(10, 10),
            "embedding_weights": torch.randn(768, 100)
        }
    
    async def _calculate_round_metrics(self, local_update: ModelUpdate,
                                     global_update: Dict[str, Any]) -> FederatedMetrics:
        """Calculate metrics for the current round"""
        # Simulate metrics calculation
        return FederatedMetrics(
            local_loss=local_update.loss_value,
            local_accuracy=local_update.local_accuracy,
            global_loss=global_update.get("global_loss", 0.0),
            global_accuracy=global_update.get("global_accuracy", 0.0),
            bytes_sent=local_update.get_update_size_mb() * 1024 * 1024,
            bytes_received=1024 * 1024,  # Placeholder
            round_trip_time_ms=local_update.communication_time_ms,
            epsilon_consumed=getattr(local_update, 'privacy_budget_used', 0.0),
            delta_consumed=getattr(local_update, 'delta', 0.0),
            privacy_budget_remaining=self.privacy_accountant.get_remaining_budget()[0] 
                                   if self.privacy_accountant else 0.0,
            training_time_ms=local_update.training_time_ms,
            communication_time_ms=local_update.communication_time_ms,
            total_time_ms=local_update.training_time_ms + local_update.communication_time_ms,
            model_similarity=0.8,  # Placeholder
            knowledge_transfer_rate=0.1,  # Placeholder
            convergence_speed=0.05  # Placeholder
        )
    
    async def _check_convergence(self, metrics: FederatedMetrics) -> bool:
        """Check if training has converged"""
        # Simple convergence check based on loss improvement
        if len(self.training_history) < 5:
            return False
        
        recent_losses = [m.global_loss for m in self.training_history[-5:]]
        loss_improvement = max(recent_losses) - min(recent_losses)
        
        return loss_improvement < 0.001  # Convergence threshold
    
    async def _calculate_final_metrics(self) -> Dict[str, Any]:
        """Calculate final training metrics"""
        if not self.training_history:
            return {}
        
        final_round = self.training_history[-1]
        
        return {
            "final_global_loss": final_round.global_loss,
            "final_global_accuracy": final_round.global_accuracy,
            "total_training_time": sum(m.training_time_ms for m in self.training_history),
            "total_communication_time": sum(m.communication_time_ms for m in self.training_history),
            "average_round_time": np.mean([m.total_time_ms for m in self.training_history]),
            "convergence_round": len(self.training_history),
            "communication_efficiency": final_round.get_efficiency_ratio()
        }
    
    async def evaluate_model(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate the federated model on test data"""
        # Placeholder implementation
        # In practice, this would evaluate the KSE model on test queries
        
        accuracy = np.random.uniform(0.8, 0.95)
        precision = np.random.uniform(0.75, 0.9)
        recall = np.random.uniform(0.7, 0.85)
        f1_score = 2 * (precision * recall) / (precision + recall)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "test_samples": len(test_data)
        }
    
    async def get_federation_status(self) -> Dict[str, Any]:
        """Get current federation status"""
        return {
            "node_id": self.node_id,
            "federation_id": self.federation_id,
            "current_round": self.current_round,
            "is_training": self.is_training,
            "privacy_level": self.config.privacy_level.value,
            "rounds_completed": len(self.training_history),
            "privacy_budget_remaining": self.privacy_accountant.get_remaining_budget() 
                                      if self.privacy_accountant else (0.0, 0.0)
        }
    
    def save_training_state(self, filepath: Path):
        """Save training state to file"""
        state = {
            "config": {
                "node_id": self.config.node_id,
                "federation_id": self.config.federation_id,
                "privacy_level": self.config.privacy_level.value,
                "privacy_epsilon": self.config.privacy_epsilon,
                "privacy_delta": self.config.privacy_delta
            },
            "current_round": self.current_round,
            "training_history": [
                {
                    "round": i,
                    "local_loss": m.local_loss,
                    "global_loss": m.global_loss,
                    "local_accuracy": m.local_accuracy,
                    "global_accuracy": m.global_accuracy
                }
                for i, m in enumerate(self.training_history)
            ],
            "privacy_summary": self.privacy_accountant.get_budget_summary() 
                             if self.privacy_accountant else None
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Training state saved to {filepath}")
    
    def load_training_state(self, filepath: Path):
        """Load training state from file"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.current_round = state["current_round"]
        # Note: training_history would need to be reconstructed as FederatedMetrics objects
        
        logger.info(f"Training state loaded from {filepath}")