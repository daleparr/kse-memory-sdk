"""
Federated learning coordinator for KSE Memory SDK
"""

import asyncio
import aiohttp
from aiohttp import web
import json
import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
import torch
import numpy as np
from dataclasses import asdict
import uuid

from .federated_models import (
    FederationConfig, ModelUpdate, PrivateModelUpdate, 
    FederationRoundResult, FederationRole, PrivacyLevel,
    EncryptedUpdate, EncryptedAggregate
)
from .privacy import SecureAggregation, PrivacyAuditor


logger = logging.getLogger(__name__)


class FederatedCoordinator:
    """Coordinates federated learning across KSE clients"""
    
    def __init__(self, federation_id: str, config: Dict[str, Any]):
        self.federation_id = federation_id
        self.config = config
        
        # Participant management
        self.registered_participants: Dict[str, Dict[str, Any]] = {}
        self.active_participants: Set[str] = set()
        self.participant_updates: Dict[int, Dict[str, Any]] = {}
        
        # Round management
        self.current_round = 0
        self.max_rounds = config.get("max_rounds", 100)
        self.min_participants = config.get("min_participants", 2)
        self.round_timeout = config.get("round_timeout", 300)  # 5 minutes
        
        # Aggregation
        self.aggregation_method = config.get("aggregation_method", "fedavg")
        self.global_model_state = None
        self.round_results: List[FederationRoundResult] = []
        
        # Privacy and security
        self.privacy_level = PrivacyLevel(config.get("privacy_level", "differential_privacy"))
        self.secure_aggregation = SecureAggregation() if self.privacy_level in [
            PrivacyLevel.SECURE_AGGREGATION, PrivacyLevel.FULL_PRIVACY
        ] else None
        self.privacy_auditor = PrivacyAuditor()
        
        # Communication
        self.app = web.Application()
        self.setup_routes()
        
        # State management
        self.is_training = False
        self.round_start_times: Dict[int, datetime] = {}
        self.federation_metrics = {
            "total_rounds": 0,
            "successful_rounds": 0,
            "total_participants": 0,
            "average_participation": 0.0
        }
    
    def setup_routes(self):
        """Setup HTTP routes for coordinator API"""
        self.app.router.add_post('/register', self.handle_registration)
        self.app.router.add_get('/round/{round_num}/status', self.handle_round_status)
        self.app.router.add_post('/update', self.handle_update)
        self.app.router.add_get('/global_update/{round_num}', self.handle_global_update)
        self.app.router.add_get('/federation/status', self.handle_federation_status)
        self.app.router.add_get('/federation/metrics', self.handle_federation_metrics)
        self.app.router.add_post('/federation/start', self.handle_start_training)
        self.app.router.add_post('/federation/stop', self.handle_stop_training)
    
    async def handle_registration(self, request: web.Request) -> web.Response:
        """Handle participant registration"""
        try:
            data = await request.json()
            node_id = data["node_id"]
            federation_id = data["federation_id"]
            
            if federation_id != self.federation_id:
                return web.Response(
                    text=json.dumps({"error": "Invalid federation ID"}),
                    status=400,
                    content_type='application/json'
                )
            
            # Register participant
            self.registered_participants[node_id] = {
                "node_id": node_id,
                "role": data["role"],
                "capabilities": data["capabilities"],
                "public_key": data.get("public_key"),
                "registered_at": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat()
            }
            
            self.active_participants.add(node_id)
            
            logger.info(f"Registered participant {node_id}")
            
            return web.Response(
                text=json.dumps({
                    "status": "registered",
                    "federation_id": self.federation_id,
                    "participant_count": len(self.registered_participants)
                }),
                content_type='application/json'
            )
        
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return web.Response(
                text=json.dumps({"error": str(e)}),
                status=500,
                content_type='application/json'
            )
    
    async def handle_round_status(self, request: web.Request) -> web.Response:
        """Handle round status requests"""
        try:
            round_num = int(request.match_info['round_num'])
            
            if round_num == self.current_round and self.is_training:
                status = "started"
            elif round_num < self.current_round:
                status = "completed"
            else:
                status = "pending"
            
            return web.Response(
                text=json.dumps({
                    "round": round_num,
                    "status": status,
                    "current_round": self.current_round,
                    "participants_ready": len(self.active_participants)
                }),
                content_type='application/json'
            )
        
        except Exception as e:
            logger.error(f"Round status error: {e}")
            return web.Response(
                text=json.dumps({"error": str(e)}),
                status=500,
                content_type='application/json'
            )
    
    async def handle_update(self, request: web.Request) -> web.Response:
        """Handle model update from participant"""
        try:
            data = await request.json()
            node_id = data["node_id"]
            round_number = data["round_number"]
            
            # Validate participant
            if node_id not in self.registered_participants:
                return web.Response(
                    text=json.dumps({"error": "Participant not registered"}),
                    status=400,
                    content_type='application/json'
                )
            
            # Validate round
            if round_number != self.current_round:
                return web.Response(
                    text=json.dumps({"error": "Invalid round number"}),
                    status=400,
                    content_type='application/json'
                )
            
            # Store update
            if round_number not in self.participant_updates:
                self.participant_updates[round_number] = {}
            
            self.participant_updates[round_number][node_id] = data
            
            # Update participant last seen
            self.registered_participants[node_id]["last_seen"] = datetime.now().isoformat()
            
            logger.info(f"Received update from {node_id} for round {round_number}")
            
            return web.Response(
                text=json.dumps({"status": "received"}),
                content_type='application/json'
            )
        
        except Exception as e:
            logger.error(f"Update handling error: {e}")
            return web.Response(
                text=json.dumps({"error": str(e)}),
                status=500,
                content_type='application/json'
            )
    
    async def handle_global_update(self, request: web.Request) -> web.Response:
        """Handle global update requests"""
        try:
            round_num = int(request.match_info['round_num'])
            
            # Check if global update is ready
            if round_num not in self.participant_updates:
                return web.Response(
                    text=json.dumps({"error": "Round not found"}),
                    status=404,
                    content_type='application/json'
                )
            
            # Check if aggregation is complete
            if not hasattr(self, f'global_update_{round_num}'):
                return web.Response(
                    text=json.dumps({"status": "aggregating"}),
                    status=202,
                    content_type='application/json'
                )
            
            global_update = getattr(self, f'global_update_{round_num}')
            
            return web.Response(
                text=json.dumps(global_update),
                content_type='application/json'
            )
        
        except Exception as e:
            logger.error(f"Global update error: {e}")
            return web.Response(
                text=json.dumps({"error": str(e)}),
                status=500,
                content_type='application/json'
            )
    
    async def handle_federation_status(self, request: web.Request) -> web.Response:
        """Handle federation status requests"""
        try:
            status = {
                "federation_id": self.federation_id,
                "is_training": self.is_training,
                "current_round": self.current_round,
                "max_rounds": self.max_rounds,
                "registered_participants": len(self.registered_participants),
                "active_participants": len(self.active_participants),
                "privacy_level": self.privacy_level.value,
                "aggregation_method": self.aggregation_method
            }
            
            return web.Response(
                text=json.dumps(status),
                content_type='application/json'
            )
        
        except Exception as e:
            logger.error(f"Status error: {e}")
            return web.Response(
                text=json.dumps({"error": str(e)}),
                status=500,
                content_type='application/json'
            )
    
    async def handle_federation_metrics(self, request: web.Request) -> web.Response:
        """Handle federation metrics requests"""
        try:
            metrics = {
                **self.federation_metrics,
                "round_results": [asdict(result) for result in self.round_results[-10:]],  # Last 10 rounds
                "privacy_audit": self.privacy_auditor.generate_privacy_report()
            }
            
            return web.Response(
                text=json.dumps(metrics),
                content_type='application/json'
            )
        
        except Exception as e:
            logger.error(f"Metrics error: {e}")
            return web.Response(
                text=json.dumps({"error": str(e)}),
                status=500,
                content_type='application/json'
            )
    
    async def handle_start_training(self, request: web.Request) -> web.Response:
        """Handle start training requests"""
        try:
            if self.is_training:
                return web.Response(
                    text=json.dumps({"error": "Training already in progress"}),
                    status=400,
                    content_type='application/json'
                )
            
            if len(self.active_participants) < self.min_participants:
                return web.Response(
                    text=json.dumps({
                        "error": f"Insufficient participants. Need {self.min_participants}, have {len(self.active_participants)}"
                    }),
                    status=400,
                    content_type='application/json'
                )
            
            # Start training
            asyncio.create_task(self.run_federated_training())
            
            return web.Response(
                text=json.dumps({"status": "training_started"}),
                content_type='application/json'
            )
        
        except Exception as e:
            logger.error(f"Start training error: {e}")
            return web.Response(
                text=json.dumps({"error": str(e)}),
                status=500,
                content_type='application/json'
            )
    
    async def handle_stop_training(self, request: web.Request) -> web.Response:
        """Handle stop training requests"""
        try:
            self.is_training = False
            
            return web.Response(
                text=json.dumps({"status": "training_stopped"}),
                content_type='application/json'
            )
        
        except Exception as e:
            logger.error(f"Stop training error: {e}")
            return web.Response(
                text=json.dumps({"error": str(e)}),
                status=500,
                content_type='application/json'
            )
    
    async def run_federated_training(self):
        """Run the federated training process"""
        self.is_training = True
        logger.info("Starting federated training")
        
        try:
            for round_num in range(self.max_rounds):
                if not self.is_training:
                    break
                
                self.current_round = round_num
                self.round_start_times[round_num] = datetime.now()
                
                logger.info(f"Starting round {round_num + 1}/{self.max_rounds}")
                
                # Wait for participant updates
                await self._wait_for_updates(round_num)
                
                # Aggregate updates
                global_update = await self._aggregate_updates(round_num)
                
                if global_update is None:
                    logger.error(f"Failed to aggregate updates for round {round_num}")
                    break
                
                # Store global update
                setattr(self, f'global_update_{round_num}', global_update)
                
                # Calculate round results
                round_result = await self._calculate_round_result(round_num, global_update)
                self.round_results.append(round_result)
                
                # Check convergence
                if await self._check_convergence(round_result):
                    logger.info(f"Convergence achieved at round {round_num + 1}")
                    break
                
                # Update federation metrics
                self._update_federation_metrics(round_result)
        
        except Exception as e:
            logger.error(f"Federated training error: {e}")
        finally:
            self.is_training = False
            logger.info("Federated training completed")
    
    async def _wait_for_updates(self, round_num: int):
        """Wait for participant updates for a round"""
        timeout = timedelta(seconds=self.round_timeout)
        start_time = datetime.now()
        
        while datetime.now() - start_time < timeout:
            if round_num in self.participant_updates:
                received_updates = len(self.participant_updates[round_num])
                if received_updates >= self.min_participants:
                    logger.info(f"Received {received_updates} updates for round {round_num}")
                    return
            
            await asyncio.sleep(5)  # Check every 5 seconds
        
        logger.warning(f"Timeout waiting for updates in round {round_num}")
    
    async def _aggregate_updates(self, round_num: int) -> Optional[Dict[str, Any]]:
        """Aggregate participant updates for a round"""
        if round_num not in self.participant_updates:
            return None
        
        updates = self.participant_updates[round_num]
        if len(updates) == 0:
            return None
        
        logger.info(f"Aggregating {len(updates)} updates for round {round_num}")
        
        try:
            if self.aggregation_method == "fedavg":
                return await self._federated_averaging(updates)
            elif self.aggregation_method == "weighted_avg":
                return await self._weighted_averaging(updates)
            else:
                raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
        
        except Exception as e:
            logger.error(f"Aggregation error: {e}")
            return None
    
    async def _federated_averaging(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Perform federated averaging of updates"""
        kg_updates = []
        cs_updates = []
        embedding_updates = []
        total_samples = 0
        
        for node_id, update_data in updates.items():
            if update_data["type"] == "private":
                # Handle private updates
                if "encrypted_kg" in update_data:
                    # Decrypt if encrypted
                    kg_update = self._decrypt_update(update_data["encrypted_kg"])
                    cs_update = self._decrypt_update(update_data["encrypted_cs"])
                    embedding_update = self._decrypt_update(update_data["encrypted_embedding"])
                else:
                    kg_update = torch.tensor(update_data["kg_update"])
                    cs_update = torch.tensor(update_data["cs_update"])
                    embedding_update = torch.tensor(update_data["embedding_update"])
            else:
                # Handle standard updates
                kg_update = torch.tensor(update_data["kg_update"])
                cs_update = torch.tensor(update_data["cs_update"])
                embedding_update = torch.tensor(update_data["embedding_update"])
            
            kg_updates.append(kg_update)
            cs_updates.append(cs_update)
            embedding_updates.append(embedding_update)
            total_samples += update_data["sample_count"]
        
        # Compute averages
        avg_kg = torch.mean(torch.stack(kg_updates), dim=0)
        avg_cs = torch.mean(torch.stack(cs_updates), dim=0)
        avg_embedding = torch.mean(torch.stack(embedding_updates), dim=0)
        
        # Calculate global metrics
        global_loss = np.mean([
            update_data.get("loss_value", 0.0) 
            for update_data in updates.values()
        ])
        global_accuracy = np.mean([
            update_data.get("local_accuracy", 0.0) 
            for update_data in updates.values()
        ])
        
        return {
            "kg_update": avg_kg.tolist(),
            "cs_update": avg_cs.tolist(),
            "embedding_update": avg_embedding.tolist(),
            "global_loss": global_loss,
            "global_accuracy": global_accuracy,
            "participant_count": len(updates),
            "total_samples": total_samples,
            "aggregation_method": "fedavg"
        }
    
    async def _weighted_averaging(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Perform weighted averaging based on sample counts"""
        kg_updates = []
        cs_updates = []
        embedding_updates = []
        weights = []
        total_samples = 0
        
        for node_id, update_data in updates.items():
            sample_count = update_data["sample_count"]
            total_samples += sample_count
            weights.append(sample_count)
            
            if update_data["type"] == "private":
                if "encrypted_kg" in update_data:
                    kg_update = self._decrypt_update(update_data["encrypted_kg"])
                    cs_update = self._decrypt_update(update_data["encrypted_cs"])
                    embedding_update = self._decrypt_update(update_data["encrypted_embedding"])
                else:
                    kg_update = torch.tensor(update_data["kg_update"])
                    cs_update = torch.tensor(update_data["cs_update"])
                    embedding_update = torch.tensor(update_data["embedding_update"])
            else:
                kg_update = torch.tensor(update_data["kg_update"])
                cs_update = torch.tensor(update_data["cs_update"])
                embedding_update = torch.tensor(update_data["embedding_update"])
            
            kg_updates.append(kg_update)
            cs_updates.append(cs_update)
            embedding_updates.append(embedding_update)
        
        # Normalize weights
        weights = torch.tensor(weights, dtype=torch.float32)
        weights = weights / weights.sum()
        
        # Compute weighted averages
        weighted_kg = sum(w * update for w, update in zip(weights, kg_updates))
        weighted_cs = sum(w * update for w, update in zip(weights, cs_updates))
        weighted_embedding = sum(w * update for w, update in zip(weights, embedding_updates))
        
        # Calculate global metrics
        weighted_loss = sum(
            w * update_data.get("loss_value", 0.0)
            for w, update_data in zip(weights, updates.values())
        )
        weighted_accuracy = sum(
            w * update_data.get("local_accuracy", 0.0)
            for w, update_data in zip(weights, updates.values())
        )
        
        return {
            "kg_update": weighted_kg.tolist(),
            "cs_update": weighted_cs.tolist(),
            "embedding_update": weighted_embedding.tolist(),
            "global_loss": float(weighted_loss),
            "global_accuracy": float(weighted_accuracy),
            "participant_count": len(updates),
            "total_samples": total_samples,
            "aggregation_method": "weighted_avg"
        }
    
    def _decrypt_update(self, encrypted_data: str) -> torch.Tensor:
        """Decrypt an encrypted update"""
        if not self.secure_aggregation:
            raise RuntimeError("Secure aggregation not initialized")
        
        encrypted_bytes = bytes.fromhex(encrypted_data)
        # Note: This is simplified - in practice, we'd need shape and dtype info
        # For now, return a placeholder tensor
        return torch.randn(10, 10)  # Placeholder
    
    async def _calculate_round_result(self, round_num: int, 
                                    global_update: Dict[str, Any]) -> FederationRoundResult:
        """Calculate results for a completed round"""
        updates = self.participant_updates.get(round_num, {})
        
        # Calculate communication metrics
        total_communication_mb = sum(
            len(json.dumps(update).encode()) / (1024 * 1024)
            for update in updates.values()
        )
        
        # Calculate round duration
        start_time = self.round_start_times.get(round_num, datetime.now())
        round_duration = (datetime.now() - start_time).total_seconds()
        
        # Calculate convergence metric (simplified)
        convergence_metric = 1.0 / (global_update["global_loss"] + 1e-6)
        
        return FederationRoundResult(
            round_number=round_num,
            participants=len(self.active_participants),
            successful_updates=len(updates),
            global_loss=global_update["global_loss"],
            global_accuracy=global_update["global_accuracy"],
            convergence_metric=convergence_metric,
            total_communication_mb=total_communication_mb,
            round_duration_seconds=round_duration,
            privacy_budget_remaining=1.0,  # Placeholder
            privacy_violations=0,
            model_drift=0.1,  # Placeholder
            participant_similarity=0.8  # Placeholder
        )
    
    async def _check_convergence(self, round_result: FederationRoundResult) -> bool:
        """Check if training has converged"""
        if len(self.round_results) < 5:
            return False
        
        # Check loss improvement over last 5 rounds
        recent_losses = [r.global_loss for r in self.round_results[-5:]]
        loss_improvement = max(recent_losses) - min(recent_losses)
        
        return loss_improvement < 0.001  # Convergence threshold
    
    def _update_federation_metrics(self, round_result: FederationRoundResult):
        """Update federation-wide metrics"""
        self.federation_metrics["total_rounds"] += 1
        
        if round_result.successful_updates > 0:
            self.federation_metrics["successful_rounds"] += 1
        
        self.federation_metrics["average_participation"] = (
            self.federation_metrics["average_participation"] * 
            (self.federation_metrics["total_rounds"] - 1) +
            round_result.get_success_rate()
        ) / self.federation_metrics["total_rounds"]
    
    async def start_server(self, host: str = "0.0.0.0", port: int = 8080):
        """Start the coordinator server"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, host, port)
        await site.start()
        
        logger.info(f"Federated coordinator started on {host}:{port}")
        logger.info(f"Federation ID: {self.federation_id}")
        
        return runner
    
    def get_federation_summary(self) -> Dict[str, Any]:
        """Get comprehensive federation summary"""
        return {
            "federation_id": self.federation_id,
            "configuration": {
                "max_rounds": self.max_rounds,
                "min_participants": self.min_participants,
                "aggregation_method": self.aggregation_method,
                "privacy_level": self.privacy_level.value,
                "round_timeout": self.round_timeout
            },
            "participants": {
                "registered": len(self.registered_participants),
                "active": len(self.active_participants),
                "details": list(self.registered_participants.values())
            },
            "training_status": {
                "is_training": self.is_training,
                "current_round": self.current_round,
                "completed_rounds": len(self.round_results)
            },
            "metrics": self.federation_metrics,
            "recent_results": [
                asdict(result) for result in self.round_results[-5:]
            ]
        }


async def create_coordinator(federation_id: str, config: Dict[str, Any]) -> FederatedCoordinator:
    """Create and initialize a federated coordinator"""
    coordinator = FederatedCoordinator(federation_id, config)
    return coordinator


# Example usage
async def main():
    """Example coordinator setup"""
    config = {
        "max_rounds": 50,
        "min_participants": 2,
        "round_timeout": 300,
        "aggregation_method": "fedavg",
        "privacy_level": "differential_privacy"
    }
    
    federation_id = str(uuid.uuid4())
    coordinator = await create_coordinator(federation_id, config)
    
    # Start server
    runner = await coordinator.start_server(host="localhost", port=8080)
    
    try:
        # Keep server running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down coordinator")
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())