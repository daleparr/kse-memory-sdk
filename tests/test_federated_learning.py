"""
Comprehensive test suite for federated learning functionality
"""

import pytest
import torch
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

from kse_memory.federated import (
    FederationConfig, FederationRole, PrivacyLevel,
    ModelUpdate, PrivateModelUpdate, FederatedMetrics,
    FederatedKSEClient, FederatedCoordinator,
    DifferentialPrivacyMechanism, SecureAggregation,
    PrivacyAccountant, create_private_update,
    create_federation_config, create_coordinator
)
from kse_memory.core.memory import KSEMemory


class TestFederationConfig:
    """Test federation configuration"""
    
    def test_create_federation_config(self):
        """Test creating federation configuration"""
        config = create_federation_config(
            node_id="test_node",
            federation_id="test_federation",
            role="participant",
            privacy_level="differential_privacy"
        )
        
        assert config.node_id == "test_node"
        assert config.federation_id == "test_federation"
        assert config.role == FederationRole.PARTICIPANT
        assert config.privacy_level == PrivacyLevel.DIFFERENTIAL_PRIVACY
        assert config.privacy_epsilon == 0.3
        assert config.privacy_delta == 1e-5
    
    def test_federation_config_validation(self):
        """Test federation configuration validation"""
        # Valid configuration
        config = FederationConfig(
            node_id="node1",
            federation_id="fed1",
            role=FederationRole.PARTICIPANT
        )
        assert config.node_id == "node1"
        
        # Test privacy settings
        config.privacy_level = PrivacyLevel.FULL_PRIVACY
        assert config.privacy_level == PrivacyLevel.FULL_PRIVACY


class TestModelUpdates:
    """Test model update structures"""
    
    def test_model_update_creation(self):
        """Test creating model updates"""
        kg_update = torch.randn(10, 5)
        cs_update = torch.randn(5, 5)
        embedding_update = torch.randn(100, 50)
        
        update = ModelUpdate(
            kg_update=kg_update,
            cs_update=cs_update,
            embedding_update=embedding_update,
            node_id="test_node",
            round_number=1,
            local_epochs=5,
            sample_count=100,
            loss_value=0.5,
            training_time_ms=1000.0,
            local_accuracy=0.85
        )
        
        assert update.node_id == "test_node"
        assert update.round_number == 1
        assert update.sample_count == 100
        assert update.get_total_parameters() == kg_update.numel() + cs_update.numel() + embedding_update.numel()
        
        # Test update size calculation
        size_mb = update.get_update_size_mb()
        assert size_mb > 0
    
    def test_private_model_update(self):
        """Test private model updates"""
        kg_update = torch.randn(10, 5)
        cs_update = torch.randn(5, 5)
        embedding_update = torch.randn(100, 50)
        
        private_update = PrivateModelUpdate(
            kg_update=kg_update,
            cs_update=cs_update,
            embedding_update=embedding_update,
            privacy_budget_used=0.1,
            noise_scale=0.5,
            clipping_norm=1.0,
            node_id="test_node",
            round_number=1,
            sample_count=100,
            epsilon=0.3,
            delta=1e-5
        )
        
        privacy_cost = private_update.get_privacy_cost()
        assert privacy_cost["epsilon_used"] == 0.1
        assert privacy_cost["total_epsilon"] == 0.3
        assert abs(privacy_cost["remaining_budget"] - 0.2) < 1e-10  # Handle floating point precision


class TestDifferentialPrivacy:
    """Test differential privacy mechanisms"""
    
    def test_privacy_budget(self):
        """Test privacy budget management"""
        from kse_memory.federated.privacy import PrivacyBudget
        
        budget = PrivacyBudget(total_epsilon=1.0, total_delta=1e-5)
        
        # Test spending budget
        assert budget.can_spend(0.3, 1e-6)
        assert budget.spend(0.3, 1e-6)
        
        remaining = budget.remaining_budget()
        assert remaining[0] == 0.7
        assert remaining[1] == 9e-6
        
        # Test budget exhaustion
        assert not budget.can_spend(0.8, 0)
    
    def test_differential_privacy_mechanism(self):
        """Test differential privacy mechanisms"""
        mechanism = DifferentialPrivacyMechanism(epsilon=1.0, delta=1e-5)
        
        # Test Gaussian mechanism
        data = torch.randn(10, 5)
        private_data = mechanism.gaussian_mechanism(data, 0.5, 1e-6)
        
        assert private_data.shape == data.shape
        assert not torch.equal(data, private_data)  # Should be different due to noise
        
        # Test Laplace mechanism
        private_data_laplace = mechanism.laplace_mechanism(data, 0.5)
        assert private_data_laplace.shape == data.shape
    
    def test_gradient_clipping(self):
        """Test gradient clipping"""
        mechanism = DifferentialPrivacyMechanism()
        
        # Test with large gradients
        large_gradients = torch.randn(10, 5) * 10
        clipped = mechanism.gradient_clipping(large_gradients, max_norm=1.0)
        
        clipped_norm = torch.norm(clipped)
        assert clipped_norm <= 1.0
    
    def test_create_private_update(self):
        """Test creating private updates"""
        mechanism = DifferentialPrivacyMechanism(epsilon=1.0, delta=1e-5)
        
        kg_update = torch.randn(10, 5)
        cs_update = torch.randn(5, 5)
        embedding_update = torch.randn(100, 50)
        
        private_update = create_private_update(
            kg_update, cs_update, embedding_update,
            "test_node", 1, 100, mechanism, 0.3, 1e-6
        )
        
        assert isinstance(private_update, PrivateModelUpdate)
        assert private_update.epsilon == 0.3
        assert private_update.delta == 1e-6


class TestSecureAggregation:
    """Test secure aggregation"""
    
    def test_key_generation(self):
        """Test RSA key generation"""
        secure_agg = SecureAggregation(key_size=1024)  # Smaller key for testing
        
        public_key_pem = secure_agg.get_public_key_pem()
        assert b"BEGIN PUBLIC KEY" in public_key_pem
    
    def test_encryption_decryption(self):
        """Test tensor encryption and decryption"""
        secure_agg = SecureAggregation(key_size=1024)
        
        # Test with small tensor due to RSA size limitations
        original_tensor = torch.randn(2, 2)
        
        # Encrypt
        encrypted_data = secure_agg.encrypt_tensor(original_tensor)
        assert isinstance(encrypted_data, bytes)
        
        # Decrypt
        decrypted_tensor = secure_agg.decrypt_tensor(
            encrypted_data, original_tensor.shape, original_tensor.dtype
        )
        
        # Should be approximately equal (floating point precision)
        assert torch.allclose(original_tensor, decrypted_tensor, atol=1e-6)
    
    def test_checksum_verification(self):
        """Test data integrity verification"""
        secure_agg = SecureAggregation()
        
        data = b"test data"
        checksum = secure_agg.compute_checksum(data)
        
        assert secure_agg.verify_integrity(data, checksum)
        assert not secure_agg.verify_integrity(b"modified data", checksum)


class TestPrivacyAccountant:
    """Test privacy accounting"""
    
    def test_privacy_accounting(self):
        """Test privacy budget accounting"""
        accountant = PrivacyAccountant(total_epsilon=1.0, total_delta=1e-5)
        
        # Spend some budget
        assert accountant.spend_budget(0.3, 1e-6, "training_round_1")
        assert accountant.spend_budget(0.2, 1e-6, "training_round_2")
        
        # Check remaining budget
        remaining = accountant.get_remaining_budget()
        assert remaining[0] == 0.5
        assert remaining[1] == 8e-6
        
        # Try to exceed budget
        assert not accountant.spend_budget(0.6, 0, "training_round_3")
        
        # Get budget summary
        summary = accountant.get_budget_summary()
        assert summary["consumed_epsilon"] == 0.5
        assert summary["expenditure_count"] == 2


@pytest.mark.asyncio
class TestFederatedClient:
    """Test federated learning client"""
    
    async def test_client_initialization(self):
        """Test client initialization"""
        config = create_federation_config(
            node_id="test_client",
            federation_id="test_fed"
        )
        
        # Mock KSE Memory
        mock_kse = Mock(spec=KSEMemory)
        
        client = FederatedKSEClient(config, mock_kse)
        
        assert client.node_id == "test_client"
        assert client.federation_id == "test_fed"
        assert client.privacy_mechanism is not None
        assert client.privacy_accountant is not None
    
    async def test_client_registration(self):
        """Test client registration with coordinator"""
        config = create_federation_config(
            node_id="test_client",
            federation_id="test_fed"
        )
        
        mock_kse = Mock(spec=KSEMemory)
        
        async with FederatedKSEClient(config, mock_kse) as client:
            # Mock successful registration response
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.json = AsyncMock(return_value={"status": "registered"})
                mock_post.return_value.__aenter__.return_value = mock_response
                
                result = await client.register_with_coordinator()
                assert result is True
    
    async def test_local_training(self):
        """Test local training simulation"""
        config = create_federation_config(
            node_id="test_client",
            federation_id="test_fed"
        )
        
        mock_kse = Mock(spec=KSEMemory)
        
        async with FederatedKSEClient(config, mock_kse) as client:
            update = await client._perform_local_training()
            
            assert isinstance(update, ModelUpdate)
            assert update.node_id == "test_client"
            assert update.round_number == 0
            assert update.loss_value > 0
    
    async def test_privacy_application(self):
        """Test applying privacy to updates"""
        config = create_federation_config(
            node_id="test_client",
            federation_id="test_fed",
            privacy_level="differential_privacy"
        )
        
        mock_kse = Mock(spec=KSEMemory)
        
        async with FederatedKSEClient(config, mock_kse) as client:
            # Create a standard update
            update = ModelUpdate(
                kg_update=torch.randn(10, 5),
                cs_update=torch.randn(5, 5),
                embedding_update=torch.randn(100, 50),
                node_id="test_client",
                round_number=0,
                local_epochs=5,
                sample_count=100,
                loss_value=0.5,
                training_time_ms=1000.0
            )
            
            # Apply privacy
            private_update = await client._apply_privacy(update)
            
            assert isinstance(private_update, PrivateModelUpdate)
            assert private_update.epsilon > 0
            assert private_update.delta > 0


@pytest.mark.asyncio
class TestFederatedCoordinator:
    """Test federated learning coordinator"""
    
    async def test_coordinator_creation(self):
        """Test coordinator creation"""
        config = {
            "max_rounds": 10,
            "min_participants": 2,
            "aggregation_method": "fedavg"
        }
        
        coordinator = await create_coordinator("test_federation", config)
        
        assert coordinator.federation_id == "test_federation"
        assert coordinator.max_rounds == 10
        assert coordinator.min_participants == 2
    
    async def test_participant_registration(self):
        """Test participant registration"""
        config = {
            "max_rounds": 10,
            "min_participants": 2
        }
        
        coordinator = await create_coordinator("test_federation", config)
        
        # Mock registration request
        from aiohttp import web
        from aiohttp.test_utils import make_mocked_request
        
        registration_data = {
            "node_id": "test_node",
            "federation_id": "test_federation",
            "role": "participant",
            "capabilities": {"privacy_level": "differential_privacy"}
        }
        
        request = make_mocked_request(
            'POST', '/register',
            payload=json.dumps(registration_data).encode()
        )
        
        # Mock request.json()
        request.json = AsyncMock(return_value=registration_data)
        
        response = await coordinator.handle_registration(request)
        
        assert response.status == 200
        assert "test_node" in coordinator.registered_participants
    
    async def test_federated_averaging(self):
        """Test federated averaging aggregation"""
        config = {
            "aggregation_method": "fedavg"
        }
        
        coordinator = await create_coordinator("test_federation", config)
        
        # Create mock updates
        updates = {
            "node1": {
                "type": "standard",
                "kg_update": torch.randn(5, 3).tolist(),
                "cs_update": torch.randn(3, 3).tolist(),
                "embedding_update": torch.randn(10, 5).tolist(),
                "sample_count": 100,
                "loss_value": 0.4,
                "local_accuracy": 0.8
            },
            "node2": {
                "type": "standard",
                "kg_update": torch.randn(5, 3).tolist(),
                "cs_update": torch.randn(3, 3).tolist(),
                "embedding_update": torch.randn(10, 5).tolist(),
                "sample_count": 150,
                "loss_value": 0.3,
                "local_accuracy": 0.85
            }
        }
        
        global_update = await coordinator._federated_averaging(updates)
        
        assert "kg_update" in global_update
        assert "cs_update" in global_update
        assert "embedding_update" in global_update
        assert global_update["participant_count"] == 2
        assert global_update["aggregation_method"] == "fedavg"


class TestFederatedMetrics:
    """Test federated learning metrics"""
    
    def test_metrics_calculation(self):
        """Test metrics calculation"""
        metrics = FederatedMetrics(
            local_loss=0.5,
            local_accuracy=0.8,
            global_loss=0.4,
            global_accuracy=0.85,
            bytes_sent=1024,
            bytes_received=2048,
            round_trip_time_ms=100.0,
            epsilon_consumed=0.1,
            delta_consumed=1e-6,
            privacy_budget_remaining=0.2,
            training_time_ms=5000.0,
            communication_time_ms=500.0,
            total_time_ms=5500.0,
            model_similarity=0.9,
            knowledge_transfer_rate=0.1,
            convergence_speed=0.05
        )
        
        efficiency_ratio = metrics.get_efficiency_ratio()
        assert 0 <= efficiency_ratio <= 1
        
        comm_overhead = metrics.get_communication_overhead()
        assert comm_overhead >= 0


class TestFederatedIntegration:
    """Integration tests for federated learning"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_federation(self):
        """Test end-to-end federated learning simulation"""
        # This would be a more complex integration test
        # For now, we'll test the basic flow
        
        # Create coordinator
        coordinator_config = {
            "max_rounds": 2,
            "min_participants": 1,
            "aggregation_method": "fedavg"
        }
        
        coordinator = await create_coordinator("integration_test", coordinator_config)
        
        # Create client
        client_config = create_federation_config(
            node_id="integration_client",
            federation_id="integration_test",
            communication_rounds=2
        )
        
        mock_kse = Mock(spec=KSEMemory)
        
        # Test basic initialization
        async with FederatedKSEClient(client_config, mock_kse) as client:
            assert client.node_id == "integration_client"
            assert coordinator.federation_id == "integration_test"
    
    def test_federation_state_persistence(self):
        """Test saving and loading federation state"""
        config = create_federation_config(
            node_id="persistence_test",
            federation_id="test_fed"
        )
        
        mock_kse = Mock(spec=KSEMemory)
        client = FederatedKSEClient(config, mock_kse)
        
        # Add some training history
        mock_metrics = FederatedMetrics(
            local_loss=0.5, local_accuracy=0.8, global_loss=0.4, global_accuracy=0.85,
            bytes_sent=1024, bytes_received=2048, round_trip_time_ms=100.0,
            epsilon_consumed=0.1, delta_consumed=1e-6, privacy_budget_remaining=0.2,
            training_time_ms=5000.0, communication_time_ms=500.0, total_time_ms=5500.0,
            model_similarity=0.9, knowledge_transfer_rate=0.1, convergence_speed=0.05
        )
        client.training_history.append(mock_metrics)
        
        # Test state saving
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = Path(f.name)
        
        try:
            client.save_training_state(temp_path)
            assert temp_path.exists()
            
            # Verify saved content
            with open(temp_path, 'r') as f:
                saved_state = json.load(f)
            
            assert saved_state["config"]["node_id"] == "persistence_test"
            assert len(saved_state["training_history"]) == 1
        finally:
            temp_path.unlink()


class TestFederatedSecurity:
    """Test federated learning security features"""
    
    def test_privacy_auditing(self):
        """Test privacy auditing functionality"""
        from kse_memory.federated.privacy import PrivacyAuditor
        
        auditor = PrivacyAuditor()
        
        # Create a private update for auditing
        private_update = PrivateModelUpdate(
            kg_update=torch.randn(10, 5),
            cs_update=torch.randn(5, 5),
            embedding_update=torch.randn(100, 50),
            privacy_budget_used=0.1,
            noise_scale=0.5,
            clipping_norm=1.0,
            node_id="test_node",
            round_number=1,
            sample_count=100,
            epsilon=0.3,
            delta=1e-5
        )
        
        audit_result = auditor.audit_model_update(private_update)
        
        assert audit_result["node_id"] == "test_node"
        assert audit_result["epsilon"] == 0.3
        assert isinstance(audit_result["privacy_violations"], list)
    
    def test_security_audit(self):
        """Test security audit functionality"""
        from kse_memory.federated.federated_models import FederatedSecurityAudit
        
        audit = FederatedSecurityAudit()
        
        # Add some vulnerabilities
        audit.add_vulnerability("high", "Test vulnerability", "Fix recommendation")
        audit.add_privacy_violation("data_leak", "Test privacy violation")
        
        # Calculate security score
        score = audit.calculate_security_score()
        assert 0 <= score <= 100
        
        # Get audit report
        report = audit.get_audit_report()
        assert "security_score" in report
        assert "vulnerability_count" in report
        assert len(report["vulnerabilities"]) == 1
        assert len(report["privacy_violations"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])