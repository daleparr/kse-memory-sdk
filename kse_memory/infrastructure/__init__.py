"""
KSE Memory SDK Infrastructure Layer

This package provides production-ready infrastructure components for the KSE Memory SDK:

- API Gateway: Multi-tenant API gateway with rate limiting and tenant isolation
- Billing System: Usage-based billing with automated invoicing and payment processing
- Security Layer: Enterprise security with RBAC, authentication, and audit trails
- Service Management: Health monitoring, metrics collection, and service orchestration

Usage:
    from kse_memory.infrastructure import KSEServiceManager
    
    # Initialize with foundation layer
    service_manager = KSEServiceManager(foundation_layer)
    
    # Start all infrastructure services
    await service_manager.start_services()
    
    # Create tenants, manage billing, monitor health, etc.
"""

from .service_management import KSEServiceManager
from .api_gateway import KSEAPIGateway, TenantTier, TenantConfig
from .billing_system import BillingEngine, BillingEvent, BillingPeriod
from .security import SecurityManager, Role, Permission, AuditEventType

__version__ = "1.0.0"

__all__ = [
    # Main service manager
    "KSEServiceManager",
    
    # API Gateway
    "KSEAPIGateway",
    "TenantTier", 
    "TenantConfig",
    
    # Billing System
    "BillingEngine",
    "BillingEvent",
    "BillingPeriod",
    
    # Security Layer
    "SecurityManager",
    "Role",
    "Permission", 
    "AuditEventType",
]