"""
API Gateway and Multi-Tenant Architecture for KSE Memory SDK

This module implements a production-ready API gateway with multi-tenant support,
request routing, rate limiting, authentication, and tenant isolation.
"""

import asyncio
import logging
import time
import hashlib
import json
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import uuid
from collections import defaultdict, deque

# FastAPI and related imports
try:
    from fastapi import FastAPI, HTTPException, Depends, Request, Response
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from starlette.middleware.base import BaseHTTPMiddleware
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logging.warning("FastAPI not available. Install with: pip install fastapi uvicorn")

logger = logging.getLogger(__name__)


class TenantTier(Enum):
    """Tenant subscription tiers."""
    FREE = "free"
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


class RequestType(Enum):
    """Types of API requests for billing."""
    SEARCH = "search"
    ADD_ITEM = "add_item"
    ADAPT_DOMAIN = "adapt_domain"
    CROSS_MODAL_SEARCH = "cross_modal_search"
    TEMPORAL_QUERY = "temporal_query"
    TRANSFER_LEARNING = "transfer_learning"


@dataclass
class TenantConfig:
    """Configuration for a tenant."""
    
    tenant_id: str
    name: str
    tier: TenantTier
    api_key: str
    created_at: datetime = field(default_factory=datetime.now)
    
    # Rate limiting
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    
    # Resource limits
    max_knowledge_items: int = 1000
    max_domains: int = 5
    max_concurrent_requests: int = 10
    
    # Features
    enabled_features: List[str] = field(default_factory=lambda: [
        "basic_search", "add_items", "temporal_queries"
    ])
    
    # Billing
    billing_plan: str = "usage_based"
    billing_email: Optional[str] = None
    
    # Security
    allowed_origins: List[str] = field(default_factory=list)
    ip_whitelist: List[str] = field(default_factory=list)
    
    # Status
    is_active: bool = True
    suspended_reason: Optional[str] = None
    
    def __post_init__(self):
        """Set tier-specific defaults."""
        if self.tier == TenantTier.FREE:
            self.requests_per_minute = 10
            self.requests_per_hour = 100
            self.requests_per_day = 1000
            self.max_knowledge_items = 100
            self.max_domains = 1
            self.enabled_features = ["basic_search", "add_items"]
            
        elif self.tier == TenantTier.BASIC:
            self.requests_per_minute = 60
            self.requests_per_hour = 1000
            self.requests_per_day = 10000
            self.max_knowledge_items = 1000
            self.max_domains = 3
            self.enabled_features = ["basic_search", "add_items", "temporal_queries", "cross_modal_search"]
            
        elif self.tier == TenantTier.PROFESSIONAL:
            self.requests_per_minute = 300
            self.requests_per_hour = 5000
            self.requests_per_day = 50000
            self.max_knowledge_items = 10000
            self.max_domains = 10
            self.enabled_features = [
                "basic_search", "add_items", "temporal_queries", 
                "cross_modal_search", "domain_adaptation", "transfer_learning"
            ]
            
        elif self.tier == TenantTier.ENTERPRISE:
            self.requests_per_minute = 1000
            self.requests_per_hour = 20000
            self.requests_per_day = 200000
            self.max_knowledge_items = 100000
            self.max_domains = 50
            self.enabled_features = [
                "basic_search", "add_items", "temporal_queries", "cross_modal_search",
                "domain_adaptation", "transfer_learning", "advanced_analytics", "custom_models"
            ]
    
    def has_feature(self, feature: str) -> bool:
        """Check if tenant has access to a feature."""
        return feature in self.enabled_features
    
    def get_rate_limit_key(self, window: str) -> str:
        """Get rate limit key for a time window."""
        return f"rate_limit:{self.tenant_id}:{window}"


@dataclass
class RateLimitWindow:
    """Rate limiting window tracking."""
    
    requests: deque = field(default_factory=deque)
    window_size: timedelta = field(default_factory=lambda: timedelta(minutes=1))
    max_requests: int = 60
    
    def is_allowed(self) -> bool:
        """Check if request is allowed within rate limit."""
        now = datetime.now()
        
        # Remove old requests outside the window
        while self.requests and now - self.requests[0] > self.window_size:
            self.requests.popleft()
        
        # Check if under limit
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        
        return False
    
    def get_reset_time(self) -> Optional[datetime]:
        """Get when the rate limit resets."""
        if not self.requests:
            return None
        return self.requests[0] + self.window_size


class TenantManager:
    """Manages tenant configurations and isolation."""
    
    def __init__(self):
        self.tenants: Dict[str, TenantConfig] = {}
        self.api_key_to_tenant: Dict[str, str] = {}
        self.rate_limits: Dict[str, Dict[str, RateLimitWindow]] = defaultdict(dict)
        
        # Create default admin tenant
        self._create_admin_tenant()
    
    def _create_admin_tenant(self):
        """Create default admin tenant."""
        admin_tenant = TenantConfig(
            tenant_id="admin",
            name="System Administrator",
            tier=TenantTier.ENTERPRISE,
            api_key=self._generate_api_key("admin"),
            enabled_features=["*"]  # All features
        )
        self.add_tenant(admin_tenant)
    
    def _generate_api_key(self, tenant_id: str) -> str:
        """Generate API key for tenant."""
        timestamp = str(int(time.time()))
        data = f"{tenant_id}:{timestamp}:{uuid.uuid4()}"
        return f"kse_{hashlib.sha256(data.encode()).hexdigest()[:32]}"
    
    def add_tenant(self, tenant: TenantConfig) -> str:
        """Add a new tenant."""
        self.tenants[tenant.tenant_id] = tenant
        self.api_key_to_tenant[tenant.api_key] = tenant.tenant_id
        
        # Initialize rate limit windows
        self.rate_limits[tenant.tenant_id] = {
            "minute": RateLimitWindow(
                window_size=timedelta(minutes=1),
                max_requests=tenant.requests_per_minute
            ),
            "hour": RateLimitWindow(
                window_size=timedelta(hours=1),
                max_requests=tenant.requests_per_hour
            ),
            "day": RateLimitWindow(
                window_size=timedelta(days=1),
                max_requests=tenant.requests_per_day
            )
        }
        
        logger.info(f"Added tenant: {tenant.tenant_id} ({tenant.tier.value})")
        return tenant.tenant_id
    
    def get_tenant_by_api_key(self, api_key: str) -> Optional[TenantConfig]:
        """Get tenant by API key."""
        tenant_id = self.api_key_to_tenant.get(api_key)
        if tenant_id:
            return self.tenants.get(tenant_id)
        return None
    
    def get_tenant(self, tenant_id: str) -> Optional[TenantConfig]:
        """Get tenant by ID."""
        return self.tenants.get(tenant_id)
    
    def check_rate_limit(self, tenant_id: str) -> Tuple[bool, Dict[str, Any]]:
        """Check if tenant is within rate limits."""
        if tenant_id not in self.rate_limits:
            return False, {"error": "Tenant not found"}
        
        windows = self.rate_limits[tenant_id]
        
        # Check all windows
        for window_name, window in windows.items():
            if not window.is_allowed():
                reset_time = window.get_reset_time()
                return False, {
                    "error": f"Rate limit exceeded for {window_name}",
                    "window": window_name,
                    "reset_time": reset_time.isoformat() if reset_time else None,
                    "max_requests": window.max_requests
                }
        
        return True, {}
    
    def suspend_tenant(self, tenant_id: str, reason: str):
        """Suspend a tenant."""
        if tenant_id in self.tenants:
            self.tenants[tenant_id].is_active = False
            self.tenants[tenant_id].suspended_reason = reason
            logger.warning(f"Suspended tenant {tenant_id}: {reason}")
    
    def activate_tenant(self, tenant_id: str):
        """Activate a suspended tenant."""
        if tenant_id in self.tenants:
            self.tenants[tenant_id].is_active = True
            self.tenants[tenant_id].suspended_reason = None
            logger.info(f"Activated tenant {tenant_id}")
    
    def get_tenant_stats(self, tenant_id: str) -> Dict[str, Any]:
        """Get tenant statistics."""
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return {}
        
        windows = self.rate_limits.get(tenant_id, {})
        
        return {
            "tenant_id": tenant_id,
            "tier": tenant.tier.value,
            "is_active": tenant.is_active,
            "current_usage": {
                "minute": len(windows.get("minute", RateLimitWindow()).requests),
                "hour": len(windows.get("hour", RateLimitWindow()).requests),
                "day": len(windows.get("day", RateLimitWindow()).requests)
            },
            "limits": {
                "minute": tenant.requests_per_minute,
                "hour": tenant.requests_per_hour,
                "day": tenant.requests_per_day
            },
            "features": tenant.enabled_features
        }


class RequestRouter:
    """Routes API requests to appropriate handlers with tenant isolation."""
    
    def __init__(self, foundation_layer):
        self.foundation_layer = foundation_layer
        self.tenant_contexts: Dict[str, Any] = {}
    
    async def get_tenant_context(self, tenant_id: str) -> Dict[str, Any]:
        """Get or create tenant-specific context."""
        if tenant_id not in self.tenant_contexts:
            # Create isolated context for tenant
            self.tenant_contexts[tenant_id] = {
                "knowledge_items": {},
                "domains": {},
                "search_history": [],
                "created_at": datetime.now()
            }
        
        return self.tenant_contexts[tenant_id]
    
    async def route_search_request(
        self,
        tenant_id: str,
        query: str,
        search_type: str = "hybrid",
        limit: int = 10
    ) -> Dict[str, Any]:
        """Route search request with tenant isolation."""
        
        context = await self.get_tenant_context(tenant_id)
        
        # Perform search within tenant context
        # This would integrate with the foundation layer
        results = {
            "query": query,
            "search_type": search_type,
            "results": [],
            "tenant_id": tenant_id,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to tenant's search history
        context["search_history"].append({
            "query": query,
            "timestamp": datetime.now(),
            "results_count": len(results["results"])
        })
        
        return results
    
    async def route_add_item_request(
        self,
        tenant_id: str,
        item_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Route add item request with tenant isolation."""
        
        context = await self.get_tenant_context(tenant_id)
        
        # Add item to tenant's isolated knowledge base
        item_id = f"{tenant_id}_{uuid.uuid4().hex[:8]}"
        
        context["knowledge_items"][item_id] = {
            **item_data,
            "item_id": item_id,
            "tenant_id": tenant_id,
            "created_at": datetime.now()
        }
        
        return {
            "item_id": item_id,
            "status": "created",
            "tenant_id": tenant_id
        }
    
    async def route_domain_adaptation_request(
        self,
        tenant_id: str,
        domain: str,
        examples: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Route domain adaptation request."""
        
        context = await self.get_tenant_context(tenant_id)
        
        # Perform domain adaptation within tenant context
        adaptation_id = f"{tenant_id}_{domain}_{uuid.uuid4().hex[:8]}"
        
        # This would integrate with the foundation layer's meta-learning
        context["domains"][domain] = {
            "adaptation_id": adaptation_id,
            "examples_count": len(examples),
            "created_at": datetime.now(),
            "status": "training"
        }
        
        return {
            "adaptation_id": adaptation_id,
            "domain": domain,
            "status": "started",
            "tenant_id": tenant_id
        }


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Middleware for API key authentication and tenant identification."""
    
    def __init__(self, app, tenant_manager: TenantManager):
        super().__init__(app)
        self.tenant_manager = tenant_manager
    
    async def dispatch(self, request: Request, call_next):
        """Process request with authentication."""
        
        # Skip auth for health checks and docs
        if request.url.path in ["/health", "/docs", "/openapi.json"]:
            return await call_next(request)
        
        # Extract API key
        api_key = None
        
        # Check Authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            api_key = auth_header.split(" ", 1)[1]
        
        # Check X-API-Key header
        if not api_key:
            api_key = request.headers.get("X-API-Key")
        
        if not api_key:
            return Response(
                content=json.dumps({"error": "API key required"}),
                status_code=401,
                media_type="application/json"
            )
        
        # Validate API key and get tenant
        tenant = self.tenant_manager.get_tenant_by_api_key(api_key)
        if not tenant:
            return Response(
                content=json.dumps({"error": "Invalid API key"}),
                status_code=401,
                media_type="application/json"
            )
        
        # Check if tenant is active
        if not tenant.is_active:
            return Response(
                content=json.dumps({
                    "error": "Tenant suspended",
                    "reason": tenant.suspended_reason
                }),
                status_code=403,
                media_type="application/json"
            )
        
        # Check rate limits
        allowed, limit_info = self.tenant_manager.check_rate_limit(tenant.tenant_id)
        if not allowed:
            return Response(
                content=json.dumps(limit_info),
                status_code=429,
                media_type="application/json"
            )
        
        # Add tenant info to request state
        request.state.tenant = tenant
        request.state.tenant_id = tenant.tenant_id
        
        # Continue with request
        response = await call_next(request)
        
        # Add tenant info to response headers
        response.headers["X-Tenant-ID"] = tenant.tenant_id
        response.headers["X-Rate-Limit-Remaining-Minute"] = str(
            tenant.requests_per_minute - len(
                self.tenant_manager.rate_limits[tenant.tenant_id]["minute"].requests
            )
        )
        
        return response


class KSEAPIGateway:
    """Main API Gateway for KSE Memory SDK with multi-tenant support."""
    
    def __init__(self, foundation_layer, config: Optional[Dict[str, Any]] = None):
        self.foundation_layer = foundation_layer
        self.config = config or {}
        
        # Initialize components
        self.tenant_manager = TenantManager()
        self.request_router = RequestRouter(foundation_layer)
        
        # Initialize FastAPI app if available
        if FASTAPI_AVAILABLE:
            self.app = self._create_fastapi_app()
        else:
            self.app = None
            logger.warning("FastAPI not available - API gateway disabled")
    
    def _create_fastapi_app(self) -> FastAPI:
        """Create and configure FastAPI application."""
        
        app = FastAPI(
            title="KSE Memory SDK API",
            description="Multi-tenant API Gateway for KSE Memory SDK",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.get("allowed_origins", ["*"]),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"]
        )
        
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=self.config.get("allowed_hosts", ["*"])
        )
        
        app.add_middleware(AuthenticationMiddleware, self.tenant_manager)
        
        # Register routes
        self._register_routes(app)
        
        return app
    
    def _register_routes(self, app: FastAPI):
        """Register API routes."""
        
        @app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0"
            }
        
        @app.post("/api/v1/search")
        async def search(request: Request, query_data: Dict[str, Any]):
            """Search endpoint."""
            tenant_id = request.state.tenant_id
            
            # Check feature access
            if not request.state.tenant.has_feature("basic_search"):
                raise HTTPException(status_code=403, detail="Feature not available in your plan")
            
            return await self.request_router.route_search_request(
                tenant_id=tenant_id,
                query=query_data.get("query", ""),
                search_type=query_data.get("search_type", "hybrid"),
                limit=query_data.get("limit", 10)
            )
        
        @app.post("/api/v1/items")
        async def add_item(request: Request, item_data: Dict[str, Any]):
            """Add knowledge item endpoint."""
            tenant_id = request.state.tenant_id
            
            if not request.state.tenant.has_feature("add_items"):
                raise HTTPException(status_code=403, detail="Feature not available in your plan")
            
            return await self.request_router.route_add_item_request(
                tenant_id=tenant_id,
                item_data=item_data
            )
        
        @app.post("/api/v1/domains/{domain}/adapt")
        async def adapt_domain(request: Request, domain: str, examples: List[Dict[str, Any]]):
            """Domain adaptation endpoint."""
            tenant_id = request.state.tenant_id
            
            if not request.state.tenant.has_feature("domain_adaptation"):
                raise HTTPException(status_code=403, detail="Feature not available in your plan")
            
            return await self.request_router.route_domain_adaptation_request(
                tenant_id=tenant_id,
                domain=domain,
                examples=examples
            )
        
        @app.get("/api/v1/tenant/stats")
        async def get_tenant_stats(request: Request):
            """Get tenant statistics."""
            tenant_id = request.state.tenant_id
            return self.tenant_manager.get_tenant_stats(tenant_id)
        
        @app.get("/api/v1/tenant/usage")
        async def get_tenant_usage(request: Request):
            """Get tenant usage information."""
            tenant_id = request.state.tenant_id
            context = await self.request_router.get_tenant_context(tenant_id)
            
            return {
                "tenant_id": tenant_id,
                "knowledge_items": len(context.get("knowledge_items", {})),
                "domains": len(context.get("domains", {})),
                "search_history_count": len(context.get("search_history", [])),
                "created_at": context.get("created_at", datetime.now()).isoformat()
            }
    
    def create_tenant(
        self,
        name: str,
        tier: TenantTier,
        billing_email: Optional[str] = None,
        **kwargs
    ) -> TenantConfig:
        """Create a new tenant."""
        
        tenant_id = f"tenant_{uuid.uuid4().hex[:8]}"
        api_key = self.tenant_manager._generate_api_key(tenant_id)
        
        tenant = TenantConfig(
            tenant_id=tenant_id,
            name=name,
            tier=tier,
            api_key=api_key,
            billing_email=billing_email,
            **kwargs
        )
        
        self.tenant_manager.add_tenant(tenant)
        return tenant
    
    def get_tenant_info(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Get tenant information."""
        tenant = self.tenant_manager.get_tenant(tenant_id)
        if not tenant:
            return None
        
        return {
            "tenant_id": tenant.tenant_id,
            "name": tenant.name,
            "tier": tenant.tier.value,
            "created_at": tenant.created_at.isoformat(),
            "is_active": tenant.is_active,
            "features": tenant.enabled_features,
            "limits": {
                "requests_per_minute": tenant.requests_per_minute,
                "requests_per_hour": tenant.requests_per_hour,
                "requests_per_day": tenant.requests_per_day,
                "max_knowledge_items": tenant.max_knowledge_items,
                "max_domains": tenant.max_domains
            }
        }
    
    async def start_server(self, host: str = "0.0.0.0", port: int = 8000):
        """Start the API gateway server."""
        
        if not self.app:
            raise RuntimeError("FastAPI not available - cannot start server")
        
        logger.info(f"Starting KSE API Gateway on {host}:{port}")
        
        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )
        
        server = uvicorn.Server(config)
        await server.serve()
    
    def get_gateway_stats(self) -> Dict[str, Any]:
        """Get API gateway statistics."""
        
        return {
            "total_tenants": len(self.tenant_manager.tenants),
            "active_tenants": sum(1 for t in self.tenant_manager.tenants.values() if t.is_active),
            "tenant_tiers": {
                tier.value: sum(1 for t in self.tenant_manager.tenants.values() if t.tier == tier)
                for tier in TenantTier
            },
            "total_contexts": len(self.request_router.tenant_contexts),
            "foundation_status": "active" if self.foundation_layer else "inactive"
        }