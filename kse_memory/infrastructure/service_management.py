"""
Service Management and Integration for KSE Memory SDK

This module provides comprehensive service management, health monitoring,
admin APIs, and orchestration of all infrastructure components.
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import psutil
import aiohttp

from .api_gateway import KSEAPIGateway, TenantTier
from .billing_system import BillingEngine, BillingEvent
from .security import SecurityManager, Role, Permission

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service status states."""
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class HealthCheckType(Enum):
    """Types of health checks."""
    LIVENESS = "liveness"
    READINESS = "readiness"
    STARTUP = "startup"


@dataclass
class HealthCheck:
    """Health check configuration and results."""
    
    check_id: str
    name: str
    check_type: HealthCheckType
    endpoint: Optional[str] = None
    interval: int = 30  # seconds
    timeout: int = 5  # seconds
    retries: int = 3
    
    # Results
    last_check: Optional[datetime] = None
    status: ServiceStatus = ServiceStatus.STARTING
    response_time: float = 0.0
    error_message: Optional[str] = None
    consecutive_failures: int = 0
    
    # Thresholds
    failure_threshold: int = 3
    success_threshold: int = 1
    
    def is_healthy(self) -> bool:
        """Check if this health check is passing."""
        return self.status == ServiceStatus.HEALTHY
    
    def should_run(self) -> bool:
        """Check if health check should run now."""
        if not self.last_check:
            return True
        
        return datetime.now() - self.last_check >= timedelta(seconds=self.interval)


@dataclass
class ServiceMetrics:
    """Service performance metrics."""
    
    # Request metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    
    # Resource metrics
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    
    # Business metrics
    active_tenants: int = 0
    total_knowledge_items: int = 0
    searches_per_minute: float = 0.0
    
    # Billing metrics
    revenue_today: float = 0.0
    pending_invoices: int = 0
    
    # Timestamp
    collected_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.successful_requests / max(self.total_requests, 1),
            "avg_response_time": self.avg_response_time,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "disk_usage": self.disk_usage,
            "active_tenants": self.active_tenants,
            "total_knowledge_items": self.total_knowledge_items,
            "searches_per_minute": self.searches_per_minute,
            "revenue_today": self.revenue_today,
            "pending_invoices": self.pending_invoices,
            "collected_at": self.collected_at.isoformat()
        }


class HealthMonitor:
    """Monitors service health and performance."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.health_checks: Dict[str, HealthCheck] = {}
        self.metrics_history: List[ServiceMetrics] = []
        self.max_history_size = self.config.get("max_history_size", 1000)
        
        # Initialize default health checks
        self._initialize_default_checks()
        
        # Monitoring task
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False
    
    def _initialize_default_checks(self):
        """Initialize default health checks."""
        
        # Foundation layer health
        self.add_health_check(HealthCheck(
            check_id="foundation_layer",
            name="Foundation Layer",
            check_type=HealthCheckType.READINESS,
            interval=60
        ))
        
        # API Gateway health
        self.add_health_check(HealthCheck(
            check_id="api_gateway",
            name="API Gateway",
            check_type=HealthCheckType.LIVENESS,
            endpoint="/health",
            interval=30
        ))
        
        # Database connectivity
        self.add_health_check(HealthCheck(
            check_id="database",
            name="Database Connection",
            check_type=HealthCheckType.READINESS,
            interval=45
        ))
        
        # Memory usage
        self.add_health_check(HealthCheck(
            check_id="memory_usage",
            name="Memory Usage",
            check_type=HealthCheckType.LIVENESS,
            interval=30
        ))
    
    def add_health_check(self, health_check: HealthCheck):
        """Add a health check."""
        self.health_checks[health_check.check_id] = health_check
        logger.info(f"Added health check: {health_check.name}")
    
    async def start_monitoring(self):
        """Start health monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started health monitoring")
    
    async def stop_monitoring(self):
        """Stop health monitoring."""
        self.is_monitoring = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped health monitoring")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        
        while self.is_monitoring:
            try:
                # Run health checks
                await self._run_health_checks()
                
                # Collect metrics
                await self._collect_metrics()
                
                # Wait before next cycle
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(30)  # Wait longer on error
    
    async def _run_health_checks(self):
        """Run all health checks that are due."""
        
        for check in self.health_checks.values():
            if check.should_run():
                await self._run_single_health_check(check)
    
    async def _run_single_health_check(self, check: HealthCheck):
        """Run a single health check."""
        
        start_time = time.time()
        
        try:
            # Run the actual check based on type
            if check.check_id == "foundation_layer":
                success = await self._check_foundation_layer()
            elif check.check_id == "api_gateway":
                success = await self._check_api_gateway()
            elif check.check_id == "database":
                success = await self._check_database()
            elif check.check_id == "memory_usage":
                success = await self._check_memory_usage()
            else:
                success = await self._check_http_endpoint(check)
            
            # Calculate response time
            response_time = time.time() - start_time
            check.response_time = response_time
            
            # Update check status
            if success:
                check.consecutive_failures = 0
                if check.consecutive_failures <= check.success_threshold:
                    check.status = ServiceStatus.HEALTHY
                check.error_message = None
            else:
                check.consecutive_failures += 1
                if check.consecutive_failures >= check.failure_threshold:
                    check.status = ServiceStatus.UNHEALTHY
                else:
                    check.status = ServiceStatus.DEGRADED
            
            check.last_check = datetime.now()
            
        except Exception as e:
            check.consecutive_failures += 1
            check.status = ServiceStatus.ERROR
            check.error_message = str(e)
            check.last_check = datetime.now()
            
            logger.error(f"Health check {check.name} failed: {e}")
    
    async def _check_foundation_layer(self) -> bool:
        """Check foundation layer health."""
        # This would check if foundation components are responding
        return True  # Simplified for demo
    
    async def _check_api_gateway(self) -> bool:
        """Check API gateway health."""
        # This would make an HTTP request to the gateway health endpoint
        return True  # Simplified for demo
    
    async def _check_database(self) -> bool:
        """Check database connectivity."""
        # This would test database connection
        return True  # Simplified for demo
    
    async def _check_memory_usage(self) -> bool:
        """Check memory usage health."""
        memory = psutil.virtual_memory()
        return memory.percent < 90  # Fail if memory usage > 90%
    
    async def _check_http_endpoint(self, check: HealthCheck) -> bool:
        """Check HTTP endpoint health."""
        if not check.endpoint:
            return True
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    check.endpoint,
                    timeout=aiohttp.ClientTimeout(total=check.timeout)
                ) as response:
                    return response.status == 200
        except:
            return False
    
    async def _collect_metrics(self):
        """Collect system and business metrics."""
        
        # System metrics
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Create metrics object
        metrics = ServiceMetrics(
            cpu_usage=cpu_percent,
            memory_usage=memory.percent,
            disk_usage=disk.percent / 1024**3  # Convert to GB
        )
        
        # Add to history
        self.metrics_history.append(metrics)
        
        # Trim history if too large
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size:]
    
    def get_overall_health(self) -> ServiceStatus:
        """Get overall service health status."""
        if not self.health_checks:
            return ServiceStatus.STARTING
        
        statuses = [check.status for check in self.health_checks.values()]
        
        if any(status == ServiceStatus.ERROR for status in statuses):
            return ServiceStatus.ERROR
        elif any(status == ServiceStatus.UNHEALTHY for status in statuses):
            return ServiceStatus.UNHEALTHY
        elif any(status == ServiceStatus.DEGRADED for status in statuses):
            return ServiceStatus.DEGRADED
        elif all(status == ServiceStatus.HEALTHY for status in statuses):
            return ServiceStatus.HEALTHY
        else:
            return ServiceStatus.STARTING
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary."""
        
        overall_status = self.get_overall_health()
        
        return {
            "overall_status": overall_status.value,
            "checks": {
                check_id: {
                    "name": check.name,
                    "status": check.status.value,
                    "last_check": check.last_check.isoformat() if check.last_check else None,
                    "response_time": check.response_time,
                    "consecutive_failures": check.consecutive_failures,
                    "error_message": check.error_message
                }
                for check_id, check in self.health_checks.items()
            },
            "generated_at": datetime.now().isoformat()
        }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        
        if not self.metrics_history:
            return {"error": "No metrics available"}
        
        latest = self.metrics_history[-1]
        
        # Calculate averages over last hour
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_metrics = [
            m for m in self.metrics_history
            if m.collected_at >= one_hour_ago
        ]
        
        if recent_metrics:
            avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
            avg_response_time = sum(m.avg_response_time for m in recent_metrics) / len(recent_metrics)
        else:
            avg_cpu = latest.cpu_usage
            avg_memory = latest.memory_usage
            avg_response_time = latest.avg_response_time
        
        return {
            "current": latest.to_dict(),
            "last_hour_averages": {
                "cpu_usage": avg_cpu,
                "memory_usage": avg_memory,
                "avg_response_time": avg_response_time
            },
            "total_metrics_collected": len(self.metrics_history)
        }


class AdminAPIManager:
    """Manages administrative APIs and operations."""
    
    def __init__(
        self,
        api_gateway: KSEAPIGateway,
        billing_engine: BillingEngine,
        security_manager: SecurityManager,
        health_monitor: HealthMonitor
    ):
        self.api_gateway = api_gateway
        self.billing_engine = billing_engine
        self.security_manager = security_manager
        self.health_monitor = health_monitor
    
    async def create_tenant(
        self,
        admin_token: str,
        name: str,
        tier: str,
        billing_email: str,
        initial_user: Dict[str, str]
    ) -> Dict[str, Any]:
        """Create a new tenant with initial user."""
        
        # Authorize admin action
        authorized, admin_user = await self.security_manager.authorize_action(
            session_token=admin_token,
            permission=Permission.ADMIN_TENANTS
        )
        
        if not authorized:
            raise PermissionError("Insufficient permissions to create tenant")
        
        try:
            # Create tenant
            tenant_tier = TenantTier(tier.lower())
            tenant = self.api_gateway.create_tenant(
                name=name,
                tier=tenant_tier,
                billing_email=billing_email
            )
            
            # Create initial user
            user_id = await self.security_manager.create_user(
                username=initial_user["username"],
                email=initial_user["email"],
                password=initial_user["password"],
                tenant_id=tenant.tenant_id,
                roles=[Role.ADMIN],
                created_by=admin_user.user_id
            )
            
            # Create billing cycle
            cycle_id = await self.billing_engine.create_billing_cycle(
                tenant_id=tenant.tenant_id
            )
            
            return {
                "tenant": {
                    "tenant_id": tenant.tenant_id,
                    "name": tenant.name,
                    "tier": tenant.tier.value,
                    "api_key": tenant.api_key,
                    "created_at": tenant.created_at.isoformat()
                },
                "initial_user": {
                    "user_id": user_id,
                    "username": initial_user["username"],
                    "email": initial_user["email"]
                },
                "billing_cycle_id": cycle_id,
                "status": "created"
            }
            
        except Exception as e:
            logger.error(f"Failed to create tenant: {e}")
            raise
    
    async def get_system_overview(self, admin_token: str) -> Dict[str, Any]:
        """Get comprehensive system overview."""
        
        # Authorize admin action
        authorized, admin_user = await self.security_manager.authorize_action(
            session_token=admin_token,
            permission=Permission.ADMIN_SYSTEM
        )
        
        if not authorized:
            raise PermissionError("Insufficient permissions to view system overview")
        
        # Gather system information
        gateway_stats = self.api_gateway.get_gateway_stats()
        billing_analytics = await self.billing_engine.get_billing_analytics()
        health_summary = self.health_monitor.get_health_summary()
        metrics_summary = self.health_monitor.get_metrics_summary()
        
        return {
            "system_status": {
                "overall_health": health_summary["overall_status"],
                "uptime": "calculating...",  # Would calculate actual uptime
                "version": "1.0.0"
            },
            "tenants": gateway_stats,
            "billing": billing_analytics,
            "health": health_summary,
            "metrics": metrics_summary,
            "generated_at": datetime.now().isoformat(),
            "generated_by": admin_user.username
        }
    
    async def manage_tenant_billing(
        self,
        admin_token: str,
        tenant_id: str,
        action: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Manage tenant billing operations."""
        
        # Authorize admin action
        authorized, admin_user = await self.security_manager.authorize_action(
            session_token=admin_token,
            permission=Permission.ADMIN_BILLING
        )
        
        if not authorized:
            raise PermissionError("Insufficient permissions to manage billing")
        
        if action == "finalize_cycle":
            cycle_id = kwargs.get("cycle_id")
            tenant_tier = kwargs.get("tenant_tier", "basic")
            
            invoice_id = await self.billing_engine.finalize_billing_cycle(
                cycle_id=cycle_id,
                tenant_tier=tenant_tier
            )
            
            return {
                "action": "finalize_cycle",
                "cycle_id": cycle_id,
                "invoice_id": invoice_id,
                "status": "completed"
            }
        
        elif action == "process_payment":
            invoice_id = kwargs.get("invoice_id")
            amount = kwargs.get("amount")
            
            payment_result = await self.billing_engine.process_payment(
                invoice_id=invoice_id,
                amount=amount
            )
            
            return {
                "action": "process_payment",
                "invoice_id": invoice_id,
                "payment_result": payment_result,
                "status": "completed"
            }
        
        elif action == "get_summary":
            summary = await self.billing_engine.get_tenant_billing_summary(
                tenant_id=tenant_id
            )
            
            return {
                "action": "get_summary",
                "tenant_id": tenant_id,
                "summary": summary
            }
        
        else:
            raise ValueError(f"Unknown billing action: {action}")
    
    async def manage_security(
        self,
        admin_token: str,
        action: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Manage security operations."""
        
        # Authorize admin action
        authorized, admin_user = await self.security_manager.authorize_action(
            session_token=admin_token,
            permission=Permission.ADMIN_SECURITY
        )
        
        if not authorized:
            raise PermissionError("Insufficient permissions to manage security")
        
        if action == "get_dashboard":
            tenant_id = kwargs.get("tenant_id", admin_user.tenant_id)
            dashboard = await self.security_manager.get_security_dashboard(tenant_id)
            
            return {
                "action": "get_dashboard",
                "dashboard": dashboard
            }
        
        elif action == "create_user":
            user_id = await self.security_manager.create_user(
                username=kwargs["username"],
                email=kwargs["email"],
                password=kwargs["password"],
                tenant_id=kwargs["tenant_id"],
                roles=[Role(r) for r in kwargs.get("roles", ["user"])],
                created_by=admin_user.user_id
            )
            
            return {
                "action": "create_user",
                "user_id": user_id,
                "status": "created"
            }
        
        elif action == "revoke_sessions":
            user_id = kwargs["user_id"]
            self.security_manager.session_manager.revoke_user_sessions(user_id)
            
            return {
                "action": "revoke_sessions",
                "user_id": user_id,
                "status": "completed"
            }
        
        else:
            raise ValueError(f"Unknown security action: {action}")


class KSEServiceManager:
    """Main service manager that orchestrates all infrastructure components."""
    
    def __init__(self, foundation_layer, config: Optional[Dict[str, Any]] = None):
        self.foundation_layer = foundation_layer
        self.config = config or {}
        
        # Initialize infrastructure components
        self.api_gateway = KSEAPIGateway(foundation_layer, self.config.get("api_gateway", {}))
        self.billing_engine = BillingEngine(self.config.get("billing", {}))
        self.security_manager = SecurityManager(self.config.get("security", {}))
        self.health_monitor = HealthMonitor(self.config.get("health_monitor", {}))
        
        # Initialize admin API manager
        self.admin_api = AdminAPIManager(
            self.api_gateway,
            self.billing_engine,
            self.security_manager,
            self.health_monitor
        )
        
        # Service state
        self.status = ServiceStatus.STOPPED
        self.start_time: Optional[datetime] = None
        
        logger.info("Initialized KSEServiceManager")
    
    async def start_services(self):
        """Start all services."""
        
        logger.info("Starting KSE Memory SDK services...")
        self.status = ServiceStatus.STARTING
        
        try:
            # Start health monitoring
            await self.health_monitor.start_monitoring()
            
            # Initialize security (create default admin user if needed)
            await self._initialize_security()
            
            # Start API gateway server (if FastAPI is available)
            if self.api_gateway.app:
                # This would start the server in production
                logger.info("API Gateway ready to start")
            
            self.status = ServiceStatus.HEALTHY
            self.start_time = datetime.now()
            
            logger.info("✅ All KSE Memory SDK services started successfully")
            
        except Exception as e:
            self.status = ServiceStatus.ERROR
            logger.error(f"❌ Failed to start services: {e}")
            raise
    
    async def stop_services(self):
        """Stop all services gracefully."""
        
        logger.info("Stopping KSE Memory SDK services...")
        self.status = ServiceStatus.STOPPING
        
        try:
            # Stop health monitoring
            await self.health_monitor.stop_monitoring()
            
            # Flush any pending data
            await self.billing_engine.usage_tracker.flush_usage_buffer()
            await self.security_manager.audit_logger.flush_events()
            
            self.status = ServiceStatus.STOPPED
            
            logger.info("✅ All KSE Memory SDK services stopped gracefully")
            
        except Exception as e:
            self.status = ServiceStatus.ERROR
            logger.error(f"❌ Error stopping services: {e}")
            raise
    
    async def _initialize_security(self):
        """Initialize security with default admin user."""
        
        # Check if admin user exists
        admin_user = self.security_manager._find_user_by_username("admin")
        
        if not admin_user:
            # Create default admin user
            admin_password = self.security_manager.password_manager.generate_secure_password()
            
            await self.security_manager.create_user(
                username="admin",
                email="admin@kse-memory-sdk.com",
                password=admin_password,
                tenant_id="admin",
                roles=[Role.SUPER_ADMIN]
            )
            
            logger.info(f"Created default admin user with password: {admin_password}")
            logger.warning("Please change the default admin password immediately!")
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status."""
        
        uptime = None
        if self.start_time:
            uptime = str(datetime.now() - self.start_time)
        
        return {
            "service_manager": {
                "status": self.status.value,
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "uptime": uptime
            },
            "components": {
                "api_gateway": {
                    "status": "active" if self.api_gateway.app else "inactive",
                    "stats": self.api_gateway.get_gateway_stats()
                },
                "billing_engine": {
                    "status": "active",
                    "usage_buffer_size": len(self.billing_engine.usage_tracker.usage_buffer)
                },
                "security_manager": {
                    "status": "active",
                    "total_users": len(self.security_manager.users),
                    "active_sessions": len(self.security_manager.session_manager.active_sessions)
                },
                "health_monitor": {
                    "status": "monitoring" if self.health_monitor.is_monitoring else "stopped",
                    "overall_health": self.health_monitor.get_overall_health().value
                }
            },
            "foundation_layer": {
                "status": "active" if self.foundation_layer else "inactive"
            },
            "generated_at": datetime.now().isoformat()
        }
    
    async def create_demo_tenant(self) -> Dict[str, Any]:
        """Create a demo tenant for testing."""
        
        # Create demo tenant
        demo_tenant = self.api_gateway.create_tenant(
            name="Demo Company",
            tier=TenantTier.PROFESSIONAL,
            billing_email="demo@example.com"
        )
        
        # Create demo user
        demo_user_id = await self.security_manager.create_user(
            username="demo_user",
            email="demo@example.com",
            password="DemoPassword123!",
            tenant_id=demo_tenant.tenant_id,
            roles=[Role.DEVELOPER]
        )
        
        # Create billing cycle
        cycle_id = await self.billing_engine.create_billing_cycle(
            tenant_id=demo_tenant.tenant_id
        )
        
        # Record some demo usage
        await self.billing_engine.record_billable_event(
            tenant_id=demo_tenant.tenant_id,
            event_type=BillingEvent.SEARCH_REQUEST,
            tenant_tier="professional",
            quantity=10
        )
        
        await self.billing_engine.record_billable_event(
            tenant_id=demo_tenant.tenant_id,
            event_type=BillingEvent.ADD_ITEM,
            tenant_tier="professional",
            quantity=5
        )
        
        return {
            "tenant": {
                "tenant_id": demo_tenant.tenant_id,
                "name": demo_tenant.name,
                "tier": demo_tenant.tier.value,
                "api_key": demo_tenant.api_key
            },
            "user": {
                "user_id": demo_user_id,
                "username": "demo_user",
                "email": "demo@example.com",
                "password": "DemoPassword123!"
            },
            "billing_cycle_id": cycle_id,
            "status": "created"
        }
    
    async def run_health_check(self) -> Dict[str, Any]:
        """Run immediate health check of all services."""
        
        return {
            "service_status": self.get_service_status(),
            "health_summary": self.health_monitor.get_health_summary(),
            "metrics_summary": self.health_monitor.get_metrics_summary()
        }