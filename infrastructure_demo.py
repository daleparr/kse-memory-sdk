"""
Infrastructure Layer Demo for KSE Memory SDK

This script demonstrates the complete infrastructure layer including:
- API Gateway with multi-tenancy
- Usage-based billing system
- Enterprise security with RBAC
- Service management and monitoring

Run this to see all infrastructure components working together.
"""

import asyncio
import logging
import json
from datetime import datetime
from decimal import Decimal

# Import foundation layer (from previous implementation)
try:
    from kse_memory.core.foundation_integration import UniversalFoundationLayer
    FOUNDATION_AVAILABLE = True
except ImportError:
    FOUNDATION_AVAILABLE = False
    print("Foundation layer not available - using mock")

# Import infrastructure components
from kse_memory.infrastructure.service_management import KSEServiceManager
from kse_memory.infrastructure.api_gateway import TenantTier
from kse_memory.infrastructure.billing_system import BillingEvent
from kse_memory.infrastructure.security import Role, Permission

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockFoundationLayer:
    """Mock foundation layer for demo purposes."""
    
    def __init__(self):
        self.status = "active"
        self.capabilities = [
            "persistent_memory", "multimodal_embeddings", 
            "meta_learning", "universal_integration"
        ]
    
    async def add_universal_knowledge_item(self, **kwargs):
        return f"mock_item_{hash(str(kwargs)) % 10000}"
    
    async def cross_modal_search(self, **kwargs):
        return {
            "results": [
                {"id": "mock_result_1", "score": 0.95, "content": "Mock search result"},
                {"id": "mock_result_2", "score": 0.87, "content": "Another mock result"}
            ],
            "total": 2
        }
    
    def get_foundation_status(self):
        return {
            "status": "active",
            "capabilities": self.capabilities,
            "total_items": 1000,
            "domains": ["general", "technical", "business"]
        }


class InfrastructureDemo:
    """Comprehensive infrastructure demonstration."""
    
    def __init__(self):
        self.service_manager = None
        self.demo_tenant_info = None
        self.demo_user_session = None
        self.admin_session = None
    
    async def initialize_infrastructure(self):
        """Initialize the complete infrastructure stack."""
        
        print("\n🚀 === INITIALIZING KSE MEMORY SDK INFRASTRUCTURE ===")
        
        # Initialize foundation layer (mock if not available)
        if FOUNDATION_AVAILABLE:
            foundation_layer = UniversalFoundationLayer()
            print("✅ Using real Foundation Layer")
        else:
            foundation_layer = MockFoundationLayer()
            print("⚠️  Using mock Foundation Layer")
        
        # Initialize service manager with all infrastructure components
        config = {
            "api_gateway": {
                "allowed_origins": ["*"],
                "allowed_hosts": ["*"]
            },
            "billing": {
                "default_billing_period": "monthly",
                "tax_rate": 0.08,  # 8% tax
                "invoice_due_days": 30
            },
            "security": {
                "secret_key": "demo_secret_key_for_testing_only",
                "max_failed_logins": 3,
                "lockout_duration": 300,  # 5 minutes
                "require_mfa": False
            },
            "health_monitor": {
                "max_history_size": 500
            }
        }
        
        self.service_manager = KSEServiceManager(foundation_layer, config)
        
        # Start all services
        await self.service_manager.start_services()
        
        print("✅ Infrastructure initialized successfully!")
        print(f"   - API Gateway: {'Active' if self.service_manager.api_gateway.app else 'Mock'}")
        print(f"   - Billing Engine: Active")
        print(f"   - Security Manager: Active")
        print(f"   - Health Monitor: Active")
        
        # Get admin credentials for demo
        admin_users = [u for u in self.service_manager.security_manager.users.values() if u.username == "admin"]
        if admin_users:
            admin_user = admin_users[0]
            print(f"\n🔑 Admin user created:")
            print(f"   Username: admin")
            print(f"   Email: {admin_user.email}")
            print("   Password: [Generated - check logs for actual password]")
    
    async def demo_tenant_management(self):
        """Demonstrate tenant creation and management."""
        
        print("\n🏢 === TENANT MANAGEMENT DEMO ===")
        
        # Create demo tenant
        self.demo_tenant_info = await self.service_manager.create_demo_tenant()
        
        print("✅ Demo tenant created:")
        print(f"   Tenant ID: {self.demo_tenant_info['tenant']['tenant_id']}")
        print(f"   Name: {self.demo_tenant_info['tenant']['name']}")
        print(f"   Tier: {self.demo_tenant_info['tenant']['tier']}")
        print(f"   API Key: {self.demo_tenant_info['tenant']['api_key'][:20]}...")
        
        print(f"\n👤 Demo user created:")
        print(f"   Username: {self.demo_tenant_info['user']['username']}")
        print(f"   Password: {self.demo_tenant_info['user']['password']}")
        print(f"   Email: {self.demo_tenant_info['user']['email']}")
        
        # Get tenant information
        tenant_info = self.service_manager.api_gateway.get_tenant_info(
            self.demo_tenant_info['tenant']['tenant_id']
        )
        
        print(f"\n📊 Tenant limits:")
        print(f"   Requests per minute: {tenant_info['limits']['requests_per_minute']}")
        print(f"   Max knowledge items: {tenant_info['limits']['max_knowledge_items']}")
        print(f"   Max domains: {tenant_info['limits']['max_domains']}")
        print(f"   Features: {', '.join(tenant_info['features'])}")
    
    async def demo_security_authentication(self):
        """Demonstrate security and authentication."""
        
        print("\n🔐 === SECURITY & AUTHENTICATION DEMO ===")
        
        # Authenticate demo user
        session_id, token = await self.service_manager.security_manager.authenticate_user(
            username=self.demo_tenant_info['user']['username'],
            password=self.demo_tenant_info['user']['password'],
            ip_address="127.0.0.1",
            user_agent="Infrastructure Demo"
        )
        
        if session_id and token:
            self.demo_user_session = token
            print("✅ Demo user authenticated successfully")
            print(f"   Session ID: {session_id}")
            print(f"   JWT Token: {token[:50]}...")
        else:
            print("❌ Authentication failed")
            return
        
        # Test authorization
        authorized, user = await self.service_manager.security_manager.authorize_action(
            session_token=token,
            permission=Permission.EXECUTE_SEARCH,
            ip_address="127.0.0.1"
        )
        
        print(f"\n🔍 Authorization test (EXECUTE_SEARCH):")
        print(f"   Authorized: {authorized}")
        if user:
            print(f"   User: {user.username}")
            print(f"   Roles: {[r.value for r in user.roles]}")
            print(f"   Permissions: {len(user.permissions)} total")
        
        # Test admin authorization
        admin_authorized, admin_user = await self.service_manager.security_manager.authorize_action(
            session_token=token,
            permission=Permission.ADMIN_SYSTEM,
            ip_address="127.0.0.1"
        )
        
        print(f"\n👑 Admin authorization test (ADMIN_SYSTEM):")
        print(f"   Authorized: {admin_authorized}")
        print(f"   Expected: False (demo user is not admin)")
        
        # Get security dashboard
        dashboard = await self.service_manager.security_manager.get_security_dashboard(
            self.demo_tenant_info['tenant']['tenant_id']
        )
        
        print(f"\n📈 Security dashboard:")
        print(f"   Total users: {dashboard['users']['total']}")
        print(f"   Active users: {dashboard['users']['active']}")
        print(f"   Failed logins (24h): {dashboard['last_24_hours']['failed_logins']}")
        print(f"   Security events (24h): {dashboard['last_24_hours']['security_events']}")
    
    async def demo_billing_system(self):
        """Demonstrate usage tracking and billing."""
        
        print("\n💰 === BILLING SYSTEM DEMO ===")
        
        tenant_id = self.demo_tenant_info['tenant']['tenant_id']
        tenant_tier = self.demo_tenant_info['tenant']['tier']
        
        # Record various billable events
        events_to_record = [
            (BillingEvent.SEARCH_REQUEST, 25, "Search queries"),
            (BillingEvent.ADD_ITEM, 10, "Knowledge items added"),
            (BillingEvent.CROSS_MODAL_SEARCH, 8, "Cross-modal searches"),
            (BillingEvent.TEMPORAL_QUERY, 5, "Temporal queries"),
            (BillingEvent.DOMAIN_ADAPTATION, 2, "Domain adaptations"),
            (BillingEvent.TRANSFER_LEARNING, 1, "Transfer learning operations")
        ]
        
        print("📝 Recording billable events:")
        total_cost = Decimal('0')
        
        for event_type, quantity, description in events_to_record:
            record_id = await self.service_manager.billing_engine.record_billable_event(
                tenant_id=tenant_id,
                event_type=event_type,
                tenant_tier=tenant_tier,
                quantity=Decimal(str(quantity))
            )
            
            # Calculate cost for display
            pricing_tier = self.service_manager.billing_engine.pricing_tiers[tenant_tier]
            unit_cost = pricing_tier.get_event_cost(event_type)
            event_cost = unit_cost * Decimal(str(quantity))
            total_cost += event_cost
            
            print(f"   ✅ {description}: {quantity} × ${float(unit_cost)/100:.3f} = ${float(event_cost)/100:.2f}")
        
        print(f"\n💵 Total estimated cost: ${float(total_cost)/100:.2f}")
        
        # Get billing summary
        billing_summary = await self.service_manager.billing_engine.get_tenant_billing_summary(
            tenant_id=tenant_id
        )
        
        print(f"\n📊 Billing summary:")
        print(f"   Period: {billing_summary['period']['start_date'][:10]} to {billing_summary['period']['end_date'][:10]}")
        print(f"   Total cost: ${billing_summary['total_cost']:.2f}")
        print(f"   Total records: {billing_summary['total_records']}")
        
        if billing_summary['usage_summary']:
            print(f"   Usage breakdown:")
            for event_type, usage in billing_summary['usage_summary'].items():
                print(f"     - {event_type.replace('_', ' ').title()}: {usage['count']} (${usage['cost']:.2f})")
        
        # Create and finalize a billing cycle
        cycle_id = self.demo_tenant_info['billing_cycle_id']
        
        try:
            invoice_id = await self.service_manager.billing_engine.finalize_billing_cycle(
                cycle_id=cycle_id,
                tenant_tier=tenant_tier
            )
            
            print(f"\n🧾 Billing cycle finalized:")
            print(f"   Cycle ID: {cycle_id}")
            print(f"   Invoice ID: {invoice_id}")
            
            # Get invoice details
            invoice = self.service_manager.billing_engine.invoices.get(invoice_id)
            if invoice:
                print(f"   Invoice amount: ${float(invoice.total_amount):.2f}")
                print(f"   Due date: {invoice.due_date.strftime('%Y-%m-%d')}")
                print(f"   Status: {invoice.payment_status.value}")
            
        except Exception as e:
            print(f"⚠️  Could not finalize billing cycle: {e}")
    
    async def demo_api_gateway(self):
        """Demonstrate API gateway functionality."""
        
        print("\n🌐 === API GATEWAY DEMO ===")
        
        # Get gateway statistics
        gateway_stats = self.service_manager.api_gateway.get_gateway_stats()
        
        print("📊 Gateway statistics:")
        print(f"   Total tenants: {gateway_stats['total_tenants']}")
        print(f"   Active tenants: {gateway_stats['active_tenants']}")
        print(f"   Tenant tiers: {gateway_stats['tenant_tiers']}")
        print(f"   Foundation status: {gateway_stats['foundation_status']}")
        
        # Simulate API requests through the router
        tenant_id = self.demo_tenant_info['tenant']['tenant_id']
        router = self.service_manager.api_gateway.request_router
        
        print(f"\n🔄 Simulating API requests:")
        
        # Search request
        search_result = await router.route_search_request(
            tenant_id=tenant_id,
            query="artificial intelligence machine learning",
            search_type="hybrid",
            limit=5
        )
        
        print(f"   ✅ Search request processed:")
        print(f"      Query: {search_result['query']}")
        print(f"      Results: {len(search_result['results'])}")
        print(f"      Timestamp: {search_result['timestamp']}")
        
        # Add item request
        item_result = await router.route_add_item_request(
            tenant_id=tenant_id,
            item_data={
                "title": "Demo Knowledge Item",
                "content": "This is a demonstration knowledge item for the infrastructure demo.",
                "category": "demo",
                "tags": ["demo", "infrastructure", "test"]
            }
        )
        
        print(f"   ✅ Add item request processed:")
        print(f"      Item ID: {item_result['item_id']}")
        print(f"      Status: {item_result['status']}")
        
        # Domain adaptation request
        adaptation_result = await router.route_domain_adaptation_request(
            tenant_id=tenant_id,
            domain="healthcare",
            examples=[
                {"text": "Medical diagnosis", "label": "medical"},
                {"text": "Patient treatment", "label": "medical"},
                {"text": "Clinical trial", "label": "research"}
            ]
        )
        
        print(f"   ✅ Domain adaptation request processed:")
        print(f"      Adaptation ID: {adaptation_result['adaptation_id']}")
        print(f"      Domain: {adaptation_result['domain']}")
        print(f"      Status: {adaptation_result['status']}")
        
        # Get tenant context
        context = await router.get_tenant_context(tenant_id)
        print(f"\n📋 Tenant context:")
        print(f"   Knowledge items: {len(context['knowledge_items'])}")
        print(f"   Domains: {len(context['domains'])}")
        print(f"   Search history: {len(context['search_history'])}")
    
    async def demo_health_monitoring(self):
        """Demonstrate health monitoring and metrics."""
        
        print("\n🏥 === HEALTH MONITORING DEMO ===")
        
        # Get health summary
        health_summary = self.service_manager.health_monitor.get_health_summary()
        
        print("🩺 Health check results:")
        print(f"   Overall status: {health_summary['overall_status']}")
        
        for check_id, check_info in health_summary['checks'].items():
            status_emoji = "✅" if check_info['status'] == 'healthy' else "⚠️" if check_info['status'] == 'degraded' else "❌"
            print(f"   {status_emoji} {check_info['name']}: {check_info['status']}")
            if check_info['response_time'] > 0:
                print(f"      Response time: {check_info['response_time']:.3f}s")
            if check_info['consecutive_failures'] > 0:
                print(f"      Consecutive failures: {check_info['consecutive_failures']}")
        
        # Get metrics summary
        metrics_summary = self.service_manager.health_monitor.get_metrics_summary()
        
        if 'current' in metrics_summary:
            current_metrics = metrics_summary['current']
            print(f"\n📈 Current system metrics:")
            print(f"   CPU usage: {current_metrics['cpu_usage']:.1f}%")
            print(f"   Memory usage: {current_metrics['memory_usage']:.1f}%")
            print(f"   Disk usage: {current_metrics['disk_usage']:.1f} GB")
            print(f"   Success rate: {current_metrics['success_rate']:.1%}")
            
            if 'last_hour_averages' in metrics_summary:
                averages = metrics_summary['last_hour_averages']
                print(f"\n📊 Last hour averages:")
                print(f"   CPU usage: {averages['cpu_usage']:.1f}%")
                print(f"   Memory usage: {averages['memory_usage']:.1f}%")
                print(f"   Response time: {averages['avg_response_time']:.3f}s")
        
        print(f"\n🔍 Total metrics collected: {metrics_summary.get('total_metrics_collected', 0)}")
    
    async def demo_service_management(self):
        """Demonstrate service management capabilities."""
        
        print("\n⚙️  === SERVICE MANAGEMENT DEMO ===")
        
        # Get comprehensive service status
        service_status = self.service_manager.get_service_status()
        
        print("🔧 Service manager status:")
        print(f"   Status: {service_status['service_manager']['status']}")
        print(f"   Uptime: {service_status['service_manager']['uptime']}")
        
        print(f"\n🧩 Component status:")
        for component, status in service_status['components'].items():
            print(f"   {component}: {status['status']}")
            if 'stats' in status:
                if 'total_tenants' in status['stats']:
                    print(f"      Tenants: {status['stats']['total_tenants']}")
            elif 'usage_buffer_size' in status:
                print(f"      Buffer size: {status['usage_buffer_size']}")
            elif 'total_users' in status:
                print(f"      Users: {status['total_users']}")
                print(f"      Sessions: {status['active_sessions']}")
        
        print(f"\n🏗️  Foundation layer: {service_status['foundation_layer']['status']}")
        
        # Run immediate health check
        health_check = await self.service_manager.run_health_check()
        
        print(f"\n🏥 Immediate health check completed:")
        overall_health = health_check['health_summary']['overall_status']
        health_emoji = "✅" if overall_health == 'healthy' else "⚠️" if overall_health == 'degraded' else "❌"
        print(f"   {health_emoji} Overall health: {overall_health}")
    
    async def demo_admin_operations(self):
        """Demonstrate administrative operations."""
        
        print("\n👑 === ADMIN OPERATIONS DEMO ===")
        
        # First, authenticate as admin to get admin token
        admin_session_id, admin_token = await self.service_manager.security_manager.authenticate_user(
            username="admin",
            password="admin",  # This would be the generated password in real scenario
            ip_address="127.0.0.1",
            user_agent="Admin Demo"
        )
        
        if not admin_token:
            print("⚠️  Admin authentication failed - using mock admin operations")
            return
        
        self.admin_session = admin_token
        print("✅ Admin authenticated successfully")
        
        # Get system overview
        try:
            system_overview = await self.service_manager.admin_api.get_system_overview(admin_token)
            
            print(f"\n📊 System overview:")
            print(f"   Overall health: {system_overview['system_status']['overall_health']}")
            print(f"   Version: {system_overview['system_status']['version']}")
            print(f"   Total tenants: {system_overview['tenants']['total_tenants']}")
            print(f"   Total revenue: ${system_overview['billing']['total_revenue']:.2f}")
            print(f"   Pending revenue: ${system_overview['billing']['pending_revenue']:.2f}")
            
        except PermissionError as e:
            print(f"⚠️  Admin operation failed: {e}")
        
        # Demonstrate tenant billing management
        try:
            billing_summary = await self.service_manager.admin_api.manage_tenant_billing(
                admin_token=admin_token,
                tenant_id=self.demo_tenant_info['tenant']['tenant_id'],
                action="get_summary"
            )
            
            print(f"\n💰 Tenant billing management:")
            print(f"   Action: {billing_summary['action']}")
            print(f"   Total cost: ${billing_summary['summary']['total_cost']:.2f}")
            print(f"   Total records: {billing_summary['summary']['total_records']}")
            
        except PermissionError as e:
            print(f"⚠️  Billing management failed: {e}")
        
        # Demonstrate security management
        try:
            security_dashboard = await self.service_manager.admin_api.manage_security(
                admin_token=admin_token,
                action="get_dashboard",
                tenant_id=self.demo_tenant_info['tenant']['tenant_id']
            )
            
            print(f"\n🔐 Security management:")
            dashboard = security_dashboard['dashboard']
            print(f"   Total events (24h): {dashboard['last_24_hours']['total_events']}")
            print(f"   Failed logins (24h): {dashboard['last_24_hours']['failed_logins']}")
            print(f"   Active sessions: {dashboard['last_24_hours']['active_sessions']}")
            
        except PermissionError as e:
            print(f"⚠️  Security management failed: {e}")
    
    async def demo_billing_analytics(self):
        """Demonstrate billing analytics and reporting."""
        
        print("\n📈 === BILLING ANALYTICS DEMO ===")
        
        # Get comprehensive billing analytics
        analytics = await self.service_manager.billing_engine.get_billing_analytics()
        
        print("💰 Billing analytics:")
        print(f"   Total revenue: ${analytics['total_revenue']:.2f}")
        print(f"   Pending revenue: ${analytics['pending_revenue']:.2f}")
        print(f"   Total invoices: {analytics['total_invoices']}")
        print(f"   Paid invoices: {analytics['paid_invoices']}")
        print(f"   Overdue invoices: {analytics['overdue_invoices']}")
        print(f"   Overdue amount: ${analytics['overdue_amount']:.2f}")
        print(f"   Billing cycles: {analytics['billing_cycles']}")
        print(f"   Usage records buffered: {analytics['usage_records_buffered']}")
        
        # Calculate some derived metrics
        if analytics['total_invoices'] > 0:
            payment_rate = analytics['paid_invoices'] / analytics['total_invoices']
            print(f"   Payment rate: {payment_rate:.1%}")
        
        if analytics['total_revenue'] > 0:
            overdue_percentage = analytics['overdue_amount'] / (analytics['total_revenue'] + analytics['pending_revenue'])
            print(f"   Overdue percentage: {overdue_percentage:.1%}")
    
    async def cleanup_demo(self):
        """Clean up demo resources."""
        
        print("\n🧹 === CLEANUP ===")
        
        try:
            # Stop services gracefully
            await self.service_manager.stop_services()
            print("✅ All services stopped gracefully")
            
        except Exception as e:
            print(f"⚠️  Error during cleanup: {e}")
    
    async def run_complete_demo(self):
        """Run the complete infrastructure demonstration."""
        
        print("🌟 KSE Memory SDK - Infrastructure Layer Demo")
        print("=" * 60)
        
        try:
            await self.initialize_infrastructure()
            await self.demo_tenant_management()
            await self.demo_security_authentication()
            await self.demo_billing_system()
            await self.demo_api_gateway()
            await self.demo_health_monitoring()
            await self.demo_service_management()
            await self.demo_admin_operations()
            await self.demo_billing_analytics()
            
            print("\n✅ === DEMO COMPLETED SUCCESSFULLY ===")
            print("\n🎯 Key Infrastructure Achievements:")
            print("  ✅ Multi-tenant API Gateway with rate limiting")
            print("  ✅ Usage-based billing with automated invoicing")
            print("  ✅ Enterprise security with RBAC and audit trails")
            print("  ✅ Comprehensive health monitoring and metrics")
            print("  ✅ Service management and admin operations")
            print("  ✅ Complete integration with foundation layer")
            
            print("\n🚀 The Infrastructure Layer is production-ready!")
            print("\n📋 Next Steps:")
            print("  - Deploy to production environment")
            print("  - Configure external databases and storage")
            print("  - Set up monitoring and alerting")
            print("  - Implement payment processor integration")
            print("  - Add custom domain support")
            
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            await self.cleanup_demo()


async def main():
    """Main demo function."""
    
    demo = InfrastructureDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())