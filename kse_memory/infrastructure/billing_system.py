"""
Usage-Based Billing System for KSE Memory SDK

This module implements a comprehensive billing system that tracks API usage,
calculates costs, generates invoices, and handles payment processing.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from decimal import Decimal, ROUND_HALF_UP
import uuid
from collections import defaultdict

logger = logging.getLogger(__name__)


class BillingEvent(Enum):
    """Types of billable events."""
    SEARCH_REQUEST = "search_request"
    ADD_ITEM = "add_item"
    DOMAIN_ADAPTATION = "domain_adaptation"
    CROSS_MODAL_SEARCH = "cross_modal_search"
    TEMPORAL_QUERY = "temporal_query"
    TRANSFER_LEARNING = "transfer_learning"
    STORAGE_GB_HOUR = "storage_gb_hour"
    COMPUTE_HOUR = "compute_hour"
    BANDWIDTH_GB = "bandwidth_gb"


class BillingPeriod(Enum):
    """Billing periods."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"


class PaymentStatus(Enum):
    """Payment statuses."""
    PENDING = "pending"
    PROCESSING = "processing"
    PAID = "paid"
    FAILED = "failed"
    REFUNDED = "refunded"
    DISPUTED = "disputed"


@dataclass
class PricingTier:
    """Pricing configuration for a tenant tier."""
    
    tier_name: str
    
    # Per-request pricing (in cents)
    search_request_cost: Decimal = Decimal('5')  # 5 cents per search
    add_item_cost: Decimal = Decimal('2')  # 2 cents per item
    domain_adaptation_cost: Decimal = Decimal('100')  # $1 per domain adaptation
    cross_modal_search_cost: Decimal = Decimal('8')  # 8 cents per cross-modal search
    temporal_query_cost: Decimal = Decimal('6')  # 6 cents per temporal query
    transfer_learning_cost: Decimal = Decimal('50')  # 50 cents per transfer
    
    # Resource-based pricing (per hour/GB)
    storage_cost_per_gb_hour: Decimal = Decimal('0.1')  # 0.1 cents per GB-hour
    compute_cost_per_hour: Decimal = Decimal('10')  # 10 cents per compute hour
    bandwidth_cost_per_gb: Decimal = Decimal('9')  # 9 cents per GB bandwidth
    
    # Volume discounts (percentage off at thresholds)
    volume_discounts: Dict[int, Decimal] = field(default_factory=lambda: {
        1000: Decimal('0.05'),    # 5% off after 1000 requests
        10000: Decimal('0.10'),   # 10% off after 10000 requests
        100000: Decimal('0.15'),  # 15% off after 100000 requests
    })
    
    # Monthly minimums and maximums
    monthly_minimum: Decimal = Decimal('0')  # No minimum by default
    monthly_maximum: Optional[Decimal] = None  # No maximum by default
    
    def get_event_cost(self, event_type: BillingEvent) -> Decimal:
        """Get cost for a billing event type."""
        cost_mapping = {
            BillingEvent.SEARCH_REQUEST: self.search_request_cost,
            BillingEvent.ADD_ITEM: self.add_item_cost,
            BillingEvent.DOMAIN_ADAPTATION: self.domain_adaptation_cost,
            BillingEvent.CROSS_MODAL_SEARCH: self.cross_modal_search_cost,
            BillingEvent.TEMPORAL_QUERY: self.temporal_query_cost,
            BillingEvent.TRANSFER_LEARNING: self.transfer_learning_cost,
            BillingEvent.STORAGE_GB_HOUR: self.storage_cost_per_gb_hour,
            BillingEvent.COMPUTE_HOUR: self.compute_cost_per_hour,
            BillingEvent.BANDWIDTH_GB: self.bandwidth_cost_per_gb,
        }
        return cost_mapping.get(event_type, Decimal('0'))
    
    def calculate_volume_discount(self, total_requests: int) -> Decimal:
        """Calculate volume discount percentage."""
        discount = Decimal('0')
        
        for threshold, discount_rate in sorted(self.volume_discounts.items(), reverse=True):
            if total_requests >= threshold:
                discount = discount_rate
                break
        
        return discount


@dataclass
class UsageRecord:
    """Individual usage record for billing."""
    
    record_id: str
    tenant_id: str
    event_type: BillingEvent
    timestamp: datetime
    quantity: Decimal = Decimal('1')
    unit_cost: Decimal = Decimal('0')
    total_cost: Decimal = Decimal('0')
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate total cost."""
        self.total_cost = self.quantity * self.unit_cost


@dataclass
class BillingCycle:
    """Billing cycle for a tenant."""
    
    cycle_id: str
    tenant_id: str
    start_date: datetime
    end_date: datetime
    period: BillingPeriod
    
    # Usage tracking
    usage_records: List[UsageRecord] = field(default_factory=list)
    
    # Cost calculations
    subtotal: Decimal = Decimal('0')
    discount_amount: Decimal = Decimal('0')
    tax_amount: Decimal = Decimal('0')
    total_amount: Decimal = Decimal('0')
    
    # Status
    is_finalized: bool = False
    finalized_at: Optional[datetime] = None
    
    def add_usage_record(self, record: UsageRecord):
        """Add a usage record to this cycle."""
        if self.is_finalized:
            raise ValueError("Cannot add usage to finalized billing cycle")
        
        self.usage_records.append(record)
        self.subtotal += record.total_cost
    
    def finalize_cycle(self, tax_rate: Decimal = Decimal('0')):
        """Finalize the billing cycle and calculate final amounts."""
        if self.is_finalized:
            return
        
        # Calculate tax
        self.tax_amount = (self.subtotal - self.discount_amount) * tax_rate
        
        # Calculate total
        self.total_amount = self.subtotal - self.discount_amount + self.tax_amount
        
        # Mark as finalized
        self.is_finalized = True
        self.finalized_at = datetime.now()
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get usage summary for this cycle."""
        summary = defaultdict(lambda: {"count": 0, "cost": Decimal('0')})
        
        for record in self.usage_records:
            event_type = record.event_type.value
            summary[event_type]["count"] += int(record.quantity)
            summary[event_type]["cost"] += record.total_cost
        
        return dict(summary)


@dataclass
class Invoice:
    """Invoice for a billing cycle."""
    
    invoice_id: str
    tenant_id: str
    billing_cycle_id: str
    issue_date: datetime
    due_date: datetime
    
    # Invoice details
    line_items: List[Dict[str, Any]] = field(default_factory=list)
    subtotal: Decimal = Decimal('0')
    discount_amount: Decimal = Decimal('0')
    tax_amount: Decimal = Decimal('0')
    total_amount: Decimal = Decimal('0')
    
    # Payment tracking
    payment_status: PaymentStatus = PaymentStatus.PENDING
    paid_amount: Decimal = Decimal('0')
    paid_at: Optional[datetime] = None
    
    # Metadata
    currency: str = "USD"
    notes: Optional[str] = None
    
    def mark_as_paid(self, amount: Decimal, payment_date: Optional[datetime] = None):
        """Mark invoice as paid."""
        self.paid_amount = amount
        self.paid_at = payment_date or datetime.now()
        
        if amount >= self.total_amount:
            self.payment_status = PaymentStatus.PAID
        else:
            self.payment_status = PaymentStatus.PROCESSING  # Partial payment
    
    def is_overdue(self) -> bool:
        """Check if invoice is overdue."""
        return (
            self.payment_status == PaymentStatus.PENDING and
            datetime.now() > self.due_date
        )


class UsageTracker:
    """Tracks usage events for billing purposes."""
    
    def __init__(self):
        self.usage_buffer: List[UsageRecord] = []
        self.buffer_size = 1000
        self.flush_interval = timedelta(minutes=5)
        self.last_flush = datetime.now()
    
    async def record_usage(
        self,
        tenant_id: str,
        event_type: BillingEvent,
        quantity: Decimal = Decimal('1'),
        unit_cost: Optional[Decimal] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Record a usage event."""
        
        record_id = f"usage_{uuid.uuid4().hex[:12]}"
        
        record = UsageRecord(
            record_id=record_id,
            tenant_id=tenant_id,
            event_type=event_type,
            timestamp=datetime.now(),
            quantity=quantity,
            unit_cost=unit_cost or Decimal('0'),
            metadata=metadata or {}
        )
        
        self.usage_buffer.append(record)
        
        # Auto-flush if buffer is full or time interval exceeded
        if (len(self.usage_buffer) >= self.buffer_size or 
            datetime.now() - self.last_flush > self.flush_interval):
            await self.flush_usage_buffer()
        
        return record_id
    
    async def flush_usage_buffer(self):
        """Flush usage buffer to persistent storage."""
        if not self.usage_buffer:
            return
        
        # In production, this would write to a database
        logger.info(f"Flushing {len(self.usage_buffer)} usage records")
        
        # Clear buffer
        self.usage_buffer.clear()
        self.last_flush = datetime.now()
    
    async def get_usage_for_period(
        self,
        tenant_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[UsageRecord]:
        """Get usage records for a specific period."""
        
        # In production, this would query a database
        # For now, return from buffer (simplified)
        return [
            record for record in self.usage_buffer
            if (record.tenant_id == tenant_id and
                start_date <= record.timestamp <= end_date)
        ]


class BillingEngine:
    """Main billing engine that handles usage tracking, pricing, and invoice generation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Components
        self.usage_tracker = UsageTracker()
        
        # Pricing tiers
        self.pricing_tiers = self._initialize_pricing_tiers()
        
        # Billing cycles and invoices
        self.billing_cycles: Dict[str, BillingCycle] = {}
        self.invoices: Dict[str, Invoice] = {}
        
        # Configuration
        self.default_billing_period = BillingPeriod(
            self.config.get("default_billing_period", "monthly")
        )
        self.tax_rate = Decimal(str(self.config.get("tax_rate", 0.0)))
        self.invoice_due_days = self.config.get("invoice_due_days", 30)
        
        logger.info("Initialized BillingEngine")
    
    def _initialize_pricing_tiers(self) -> Dict[str, PricingTier]:
        """Initialize pricing tiers."""
        
        return {
            "free": PricingTier(
                tier_name="free",
                search_request_cost=Decimal('0'),  # Free tier
                add_item_cost=Decimal('0'),
                domain_adaptation_cost=Decimal('0'),
                monthly_maximum=Decimal('0')  # No charges for free tier
            ),
            "basic": PricingTier(
                tier_name="basic",
                search_request_cost=Decimal('3'),
                add_item_cost=Decimal('1'),
                domain_adaptation_cost=Decimal('50'),
                monthly_minimum=Decimal('1000')  # $10 minimum
            ),
            "professional": PricingTier(
                tier_name="professional",
                search_request_cost=Decimal('2'),
                add_item_cost=Decimal('1'),
                domain_adaptation_cost=Decimal('30'),
                cross_modal_search_cost=Decimal('5'),
                temporal_query_cost=Decimal('4'),
                transfer_learning_cost=Decimal('25'),
                monthly_minimum=Decimal('2500')  # $25 minimum
            ),
            "enterprise": PricingTier(
                tier_name="enterprise",
                search_request_cost=Decimal('1'),
                add_item_cost=Decimal('0.5'),
                domain_adaptation_cost=Decimal('20'),
                cross_modal_search_cost=Decimal('3'),
                temporal_query_cost=Decimal('2'),
                transfer_learning_cost=Decimal('15'),
                volume_discounts={
                    10000: Decimal('0.10'),
                    50000: Decimal('0.20'),
                    100000: Decimal('0.30'),
                },
                monthly_minimum=Decimal('10000')  # $100 minimum
            )
        }
    
    async def record_billable_event(
        self,
        tenant_id: str,
        event_type: BillingEvent,
        tenant_tier: str = "basic",
        quantity: Decimal = Decimal('1'),
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Record a billable event."""
        
        # Get pricing for tenant tier
        pricing_tier = self.pricing_tiers.get(tenant_tier, self.pricing_tiers["basic"])
        unit_cost = pricing_tier.get_event_cost(event_type)
        
        # Record usage
        record_id = await self.usage_tracker.record_usage(
            tenant_id=tenant_id,
            event_type=event_type,
            quantity=quantity,
            unit_cost=unit_cost,
            metadata=metadata
        )
        
        logger.debug(f"Recorded billable event: {event_type.value} for tenant {tenant_id}")
        
        return record_id
    
    async def create_billing_cycle(
        self,
        tenant_id: str,
        period: BillingPeriod = None,
        start_date: Optional[datetime] = None
    ) -> str:
        """Create a new billing cycle."""
        
        period = period or self.default_billing_period
        start_date = start_date or datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        # Calculate end date based on period
        if period == BillingPeriod.MONTHLY:
            # Next month, same day
            if start_date.month == 12:
                end_date = start_date.replace(year=start_date.year + 1, month=1)
            else:
                end_date = start_date.replace(month=start_date.month + 1)
        elif period == BillingPeriod.DAILY:
            end_date = start_date + timedelta(days=1)
        elif period == BillingPeriod.WEEKLY:
            end_date = start_date + timedelta(weeks=1)
        elif period == BillingPeriod.YEARLY:
            end_date = start_date.replace(year=start_date.year + 1)
        else:
            end_date = start_date + timedelta(hours=1)  # Default to hourly
        
        cycle_id = f"cycle_{tenant_id}_{start_date.strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"
        
        cycle = BillingCycle(
            cycle_id=cycle_id,
            tenant_id=tenant_id,
            start_date=start_date,
            end_date=end_date,
            period=period
        )
        
        self.billing_cycles[cycle_id] = cycle
        
        logger.info(f"Created billing cycle {cycle_id} for tenant {tenant_id}")
        
        return cycle_id
    
    async def finalize_billing_cycle(
        self,
        cycle_id: str,
        tenant_tier: str = "basic"
    ) -> str:
        """Finalize a billing cycle and generate invoice."""
        
        cycle = self.billing_cycles.get(cycle_id)
        if not cycle:
            raise ValueError(f"Billing cycle {cycle_id} not found")
        
        if cycle.is_finalized:
            raise ValueError(f"Billing cycle {cycle_id} already finalized")
        
        # Get usage records for the period
        usage_records = await self.usage_tracker.get_usage_for_period(
            tenant_id=cycle.tenant_id,
            start_date=cycle.start_date,
            end_date=cycle.end_date
        )
        
        # Add usage records to cycle
        for record in usage_records:
            cycle.add_usage_record(record)
        
        # Apply volume discounts
        pricing_tier = self.pricing_tiers.get(tenant_tier, self.pricing_tiers["basic"])
        total_requests = len([r for r in usage_records if r.event_type in [
            BillingEvent.SEARCH_REQUEST, BillingEvent.CROSS_MODAL_SEARCH, 
            BillingEvent.TEMPORAL_QUERY
        ]])
        
        discount_rate = pricing_tier.calculate_volume_discount(total_requests)
        cycle.discount_amount = cycle.subtotal * discount_rate
        
        # Apply monthly minimum/maximum
        if pricing_tier.monthly_minimum and cycle.subtotal < pricing_tier.monthly_minimum:
            cycle.subtotal = pricing_tier.monthly_minimum
        
        if pricing_tier.monthly_maximum and cycle.subtotal > pricing_tier.monthly_maximum:
            cycle.subtotal = pricing_tier.monthly_maximum
        
        # Finalize cycle
        cycle.finalize_cycle(self.tax_rate)
        
        # Generate invoice
        invoice_id = await self.generate_invoice(cycle_id)
        
        logger.info(f"Finalized billing cycle {cycle_id}, generated invoice {invoice_id}")
        
        return invoice_id
    
    async def generate_invoice(self, cycle_id: str) -> str:
        """Generate an invoice for a billing cycle."""
        
        cycle = self.billing_cycles.get(cycle_id)
        if not cycle or not cycle.is_finalized:
            raise ValueError(f"Billing cycle {cycle_id} not found or not finalized")
        
        invoice_id = f"inv_{cycle.tenant_id}_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"
        
        # Create line items from usage summary
        usage_summary = cycle.get_usage_summary()
        line_items = []
        
        for event_type, summary in usage_summary.items():
            line_items.append({
                "description": f"{event_type.replace('_', ' ').title()}",
                "quantity": summary["count"],
                "unit_cost": summary["cost"] / summary["count"] if summary["count"] > 0 else Decimal('0'),
                "total_cost": summary["cost"]
            })
        
        invoice = Invoice(
            invoice_id=invoice_id,
            tenant_id=cycle.tenant_id,
            billing_cycle_id=cycle_id,
            issue_date=datetime.now(),
            due_date=datetime.now() + timedelta(days=self.invoice_due_days),
            line_items=line_items,
            subtotal=cycle.subtotal,
            discount_amount=cycle.discount_amount,
            tax_amount=cycle.tax_amount,
            total_amount=cycle.total_amount
        )
        
        self.invoices[invoice_id] = invoice
        
        logger.info(f"Generated invoice {invoice_id} for ${float(invoice.total_amount):.2f}")
        
        return invoice_id
    
    async def process_payment(
        self,
        invoice_id: str,
        amount: Decimal,
        payment_method: str = "credit_card",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a payment for an invoice."""
        
        invoice = self.invoices.get(invoice_id)
        if not invoice:
            raise ValueError(f"Invoice {invoice_id} not found")
        
        # In production, this would integrate with payment processors like Stripe
        payment_result = {
            "payment_id": f"pay_{uuid.uuid4().hex[:12]}",
            "status": "success",
            "amount": amount,
            "method": payment_method,
            "processed_at": datetime.now(),
            "metadata": metadata or {}
        }
        
        # Update invoice
        invoice.mark_as_paid(amount)
        
        logger.info(f"Processed payment of ${float(amount):.2f} for invoice {invoice_id}")
        
        return payment_result
    
    async def get_tenant_billing_summary(
        self,
        tenant_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get billing summary for a tenant."""
        
        # Default to current month
        if not start_date:
            start_date = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        if not end_date:
            end_date = datetime.now()
        
        # Get usage records
        usage_records = await self.usage_tracker.get_usage_for_period(
            tenant_id, start_date, end_date
        )
        
        # Calculate totals
        total_cost = sum(record.total_cost for record in usage_records)
        usage_by_type = defaultdict(lambda: {"count": 0, "cost": Decimal('0')})
        
        for record in usage_records:
            event_type = record.event_type.value
            usage_by_type[event_type]["count"] += int(record.quantity)
            usage_by_type[event_type]["cost"] += record.total_cost
        
        # Get invoices
        tenant_invoices = [
            invoice for invoice in self.invoices.values()
            if invoice.tenant_id == tenant_id and start_date <= invoice.issue_date <= end_date
        ]
        
        return {
            "tenant_id": tenant_id,
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "usage_summary": dict(usage_by_type),
            "total_cost": float(total_cost),
            "total_records": len(usage_records),
            "invoices": [
                {
                    "invoice_id": inv.invoice_id,
                    "amount": float(inv.total_amount),
                    "status": inv.payment_status.value,
                    "due_date": inv.due_date.isoformat(),
                    "is_overdue": inv.is_overdue()
                }
                for inv in tenant_invoices
            ]
        }
    
    async def get_billing_analytics(self) -> Dict[str, Any]:
        """Get billing analytics across all tenants."""
        
        total_revenue = sum(
            invoice.paid_amount for invoice in self.invoices.values()
            if invoice.payment_status == PaymentStatus.PAID
        )
        
        pending_revenue = sum(
            invoice.total_amount for invoice in self.invoices.values()
            if invoice.payment_status == PaymentStatus.PENDING
        )
        
        overdue_invoices = [
            invoice for invoice in self.invoices.values()
            if invoice.is_overdue()
        ]
        
        return {
            "total_revenue": float(total_revenue),
            "pending_revenue": float(pending_revenue),
            "total_invoices": len(self.invoices),
            "paid_invoices": len([
                inv for inv in self.invoices.values()
                if inv.payment_status == PaymentStatus.PAID
            ]),
            "overdue_invoices": len(overdue_invoices),
            "overdue_amount": float(sum(inv.total_amount for inv in overdue_invoices)),
            "billing_cycles": len(self.billing_cycles),
            "usage_records_buffered": len(self.usage_tracker.usage_buffer)
        }