"""
Budget Sentinel Service - Real-time cost tracking and budget enforcement.

Monitors spending across all pipeline operations with automatic alerts and emergency shutdown
capabilities. Enforces strict budget limits of ≤$62 per cycle.
"""

import logging
import time
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
import asyncio
from contextlib import asynccontextmanager

from pipeline.config.settings import get_settings


class BudgetCategory(Enum):
    """Budget allocation categories."""
    OPENAI_TOKENS = "openai_tokens"
    GOOGLE_ADS = "google_ads"
    INFRASTRUCTURE = "infrastructure"
    EXTERNAL_APIS = "external_apis"


class AlertLevel(Enum):
    """Budget alert severity levels."""
    WARNING = "warning"      # 80% threshold
    CRITICAL = "critical"    # 95% threshold
    EMERGENCY = "emergency"  # 100% threshold


@dataclass
class BudgetAllocation:
    """Per-category budget allocation."""
    category: BudgetCategory
    allocated: Decimal
    spent: Decimal = field(default_factory=lambda: Decimal('0.00'))
    warning_threshold: Decimal = field(default_factory=lambda: Decimal('0.80'))
    critical_threshold: Decimal = field(default_factory=lambda: Decimal('0.95'))
    
    @property
    def remaining(self) -> Decimal:
        """Calculate remaining budget."""
        return self.allocated - self.spent
    
    @property
    def usage_percentage(self) -> Decimal:
        """Calculate usage as percentage."""
        if self.allocated == 0:
            return Decimal('0.00')
        return (self.spent / self.allocated) * 100
    
    def can_spend(self, amount: Decimal) -> bool:
        """Check if amount can be spent without exceeding allocation."""
        return (self.spent + amount) <= self.allocated


@dataclass
class CostTrackingRecord:
    """Individual cost tracking record."""
    timestamp: datetime
    service: str
    operation: str
    category: BudgetCategory
    amount: Decimal
    cycle_id: str
    metadata: Dict = field(default_factory=dict)


class BudgetSentinelService:
    """Real-time budget monitoring and enforcement service."""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)
        
        # Initialize budget allocations
        self.allocations = self._initialize_allocations()
        self.cost_records: List[CostTrackingRecord] = []
        self.alert_callbacks: Dict[AlertLevel, List[Callable]] = {
            AlertLevel.WARNING: [],
            AlertLevel.CRITICAL: [],
            AlertLevel.EMERGENCY: []
        }
        
        # Circuit breaker state
        self.emergency_shutdown = False
        self.circuit_breaker_active = False
        
        # Current cycle tracking
        self.current_cycle_id = self._generate_cycle_id()
        self.cycle_start_time = datetime.utcnow()
    
    def _initialize_allocations(self) -> Dict[BudgetCategory, BudgetAllocation]:
        """Initialize budget allocations from settings."""
        total_budget = Decimal(str(self.settings.budget.total_cycle_budget))
        
        allocations = {
            BudgetCategory.OPENAI_TOKENS: BudgetAllocation(
                category=BudgetCategory.OPENAI_TOKENS,
                allocated=Decimal(str(self.settings.budget.openai_budget))
            ),
            BudgetCategory.GOOGLE_ADS: BudgetAllocation(
                category=BudgetCategory.GOOGLE_ADS,
                allocated=Decimal(str(self.settings.budget.google_ads_budget))
            ),
            BudgetCategory.INFRASTRUCTURE: BudgetAllocation(
                category=BudgetCategory.INFRASTRUCTURE,
                allocated=Decimal(str(self.settings.budget.infrastructure_budget))
            ),
            BudgetCategory.EXTERNAL_APIS: BudgetAllocation(
                category=BudgetCategory.EXTERNAL_APIS,
                allocated=total_budget - Decimal(str(self.settings.budget.openai_budget))
                - Decimal(str(self.settings.budget.google_ads_budget))
                - Decimal(str(self.settings.budget.infrastructure_budget))
            )
        }
        
        return allocations
    
    def _generate_cycle_id(self) -> str:
        """Generate unique cycle identifier."""
        timestamp = int(time.time())
        return f"cycle_{timestamp}"
    
    def _mask_amount(self, amount: Decimal) -> str:
        """Mask financial amounts for secure logging."""
        if amount == 0:
            return "$0.00"
        elif amount < Decimal('1.00'):
            return "<$1"
        elif amount < Decimal('10.00'):
            return "<$10"
        elif amount < Decimal('100.00'):
            return "<$100"
        else:
            return ">=$100"
    
    async def track_cost(
        self, 
        service: str, 
        operation: str, 
        category: BudgetCategory, 
        amount: Decimal,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Track cost for operation and enforce budget constraints.
        
        Returns:
            bool: True if operation allowed, False if blocked by budget
        """
        if self.emergency_shutdown:
            self.logger.error(f"Operation blocked: Emergency shutdown active")
            return False
        
        # Round amount to 2 decimal places
        amount = amount.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        
        allocation = self.allocations[category]
        
        # Check if operation would exceed budget
        if not allocation.can_spend(amount):
            self.logger.warning(
                f"Operation blocked: Would exceed {category.value} budget "
                f"({self._mask_amount(allocation.remaining)} remaining, {self._mask_amount(amount)} requested)"
            )
            return False
        
        # Track the cost
        record = CostTrackingRecord(
            timestamp=datetime.utcnow(),
            service=service,
            operation=operation,
            category=category,
            amount=amount,
            cycle_id=self.current_cycle_id,
            metadata=metadata or {}
        )
        
        self.cost_records.append(record)
        allocation.spent += amount
        
        # Check alert thresholds
        await self._check_alert_thresholds(allocation)
        
        self.logger.info(
            f"Cost tracked: {service}.{operation} = {self._mask_amount(amount)} "
            f"({category.value}: {allocation.usage_percentage:.1f}% used)"
        )
        
        return True
    
    async def _check_alert_thresholds(self, allocation: BudgetAllocation):
        """Check if allocation has crossed alert thresholds."""
        usage_pct = allocation.usage_percentage / 100
        
        if usage_pct >= 1.0:  # Emergency: 100%
            await self._trigger_alert(AlertLevel.EMERGENCY, allocation)
        elif usage_pct >= allocation.critical_threshold:  # Critical: 95%
            await self._trigger_alert(AlertLevel.CRITICAL, allocation)
        elif usage_pct >= allocation.warning_threshold:  # Warning: 80%
            await self._trigger_alert(AlertLevel.WARNING, allocation)
    
    async def _trigger_alert(self, level: AlertLevel, allocation: BudgetAllocation):
        """Trigger budget alert and execute enforcement actions."""
        self.logger.warning(
            f"Budget alert ({level.value}): {allocation.category.value} "
            f"at {allocation.usage_percentage:.1f}% usage (threshold exceeded)"
        )
        
        # Execute alert callbacks
        for callback in self.alert_callbacks[level]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(level, allocation)
                else:
                    callback(level, allocation)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
        
        # Execute enforcement actions
        if level == AlertLevel.EMERGENCY:
            await self._emergency_shutdown()
        elif level == AlertLevel.CRITICAL:
            await self._throttle_operations()
    
    async def _emergency_shutdown(self):
        """Activate emergency shutdown - block all operations."""
        self.emergency_shutdown = True
        self.circuit_breaker_active = True
        
        self.logger.critical(
            f"EMERGENCY SHUTDOWN ACTIVATED - Budget exceeded for cycle {self.current_cycle_id}"
        )
        
        # Could trigger external notifications here (Slack, email, etc.)
    
    async def _throttle_operations(self):
        """Throttle non-critical operations."""
        self.circuit_breaker_active = True
        self.logger.warning("Throttling non-critical operations due to budget constraints")
    
    def register_alert_callback(self, level: AlertLevel, callback: Callable):
        """Register callback for budget alerts."""
        self.alert_callbacks[level].append(callback)
    
    def get_budget_status(self) -> Dict:
        """Get current budget status for all categories."""
        total_allocated = sum(a.allocated for a in self.allocations.values())
        total_spent = sum(a.spent for a in self.allocations.values())
        
        return {
            "cycle_id": self.current_cycle_id,
            "cycle_start": self.cycle_start_time.isoformat(),
            "total_budget": {
                "allocated": float(total_allocated),
                "spent": float(total_spent),
                "remaining": float(total_allocated - total_spent),
                "usage_percentage": float((total_spent / total_allocated) * 100)
            },
            "categories": {
                category.value: {
                    "allocated": float(allocation.allocated),
                    "spent": float(allocation.spent),
                    "remaining": float(allocation.remaining),
                    "usage_percentage": float(allocation.usage_percentage)
                }
                for category, allocation in self.allocations.items()
            },
            "emergency_shutdown": self.emergency_shutdown,
            "circuit_breaker_active": self.circuit_breaker_active
        }
    
    def reset_cycle(self):
        """Reset budget tracking for new cycle."""
        old_cycle = self.current_cycle_id
        self.current_cycle_id = self._generate_cycle_id()
        self.cycle_start_time = datetime.utcnow()
        
        # Reset allocations
        for allocation in self.allocations.values():
            allocation.spent = Decimal('0.00')
        
        # Archive old records (in production, persist to database)
        self.cost_records = []
        
        # Reset emergency states
        self.emergency_shutdown = False
        self.circuit_breaker_active = False
        
        self.logger.info(f"Budget cycle reset: {old_cycle} -> {self.current_cycle_id}")
    
    @asynccontextmanager
    async def track_operation(
        self, 
        service: str, 
        operation: str, 
        category: BudgetCategory,
        estimated_cost: Decimal
    ):
        """Context manager for tracking operation costs."""
        # Pre-check budget availability
        if not await self.track_cost(service, operation, category, estimated_cost):
            raise BudgetExceededException(
                f"Operation {service}.{operation} blocked by budget constraints"
            )
        
        try:
            yield
        except Exception as e:
            # Could implement cost rollback logic here if needed
            self.logger.error(f"Operation {service}.{operation} failed: {e}")
            raise


class BudgetExceededException(Exception):
    """Exception raised when budget constraints are violated."""
    pass


# Singleton instance
_budget_sentinel = None


def get_budget_sentinel() -> BudgetSentinelService:
    """Get singleton Budget Sentinel instance."""
    global _budget_sentinel
    if _budget_sentinel is None:
        _budget_sentinel = BudgetSentinelService()
    return _budget_sentinel