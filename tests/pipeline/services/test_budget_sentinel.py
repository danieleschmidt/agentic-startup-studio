"""
Comprehensive test suite for the Budget Sentinel Service.

Tests cover real-time cost tracking, budget enforcement, alert mechanisms,
emergency shutdown capabilities, and financial security measures.
"""

import pytest
import asyncio
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Any

from pipeline.services.budget_sentinel import (
    BudgetCategory,
    AlertLevel,
    BudgetAllocation,
    CostTrackingRecord,
    BudgetSentinelService,
    get_budget_sentinel
)


@pytest.fixture
def sample_budget_allocation():
    """Create a sample budget allocation for testing."""
    return BudgetAllocation(
        category=BudgetCategory.OPENAI_TOKENS,
        allocated=Decimal('10.00'),
        spent=Decimal('3.00'),
        warning_threshold=Decimal('0.80'),
        critical_threshold=Decimal('0.95')
    )


@pytest.fixture
def sample_cost_record():
    """Create a sample cost tracking record."""
    return CostTrackingRecord(
        timestamp=datetime.utcnow(),
        service="openai_api",
        operation="chat_completion",
        category=BudgetCategory.OPENAI_TOKENS,
        amount=Decimal('0.25'),
        cycle_id="cycle_123456",
        metadata={"model": "gpt-4", "tokens": 1000}
    )


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = Mock()
    settings.budget = Mock()
    settings.budget.total_cycle_budget = 62.00
    settings.budget.openai_budget = 10.00
    settings.budget.google_ads_budget = 45.00
    settings.budget.infrastructure_budget = 5.00
    return settings


@pytest.fixture
def budget_sentinel_service(mock_settings):
    """Create budget sentinel service with mock settings."""
    with patch('pipeline.services.budget_sentinel.get_settings', return_value=mock_settings):
        return BudgetSentinelService()


class TestBudgetCategory:
    """Test BudgetCategory enum."""
    
    def test_budget_category_values(self):
        """Test budget category enum values."""
        assert BudgetCategory.OPENAI_TOKENS.value == "openai_tokens"
        assert BudgetCategory.GOOGLE_ADS.value == "google_ads"
        assert BudgetCategory.INFRASTRUCTURE.value == "infrastructure"
        assert BudgetCategory.EXTERNAL_APIS.value == "external_apis"
    
    def test_budget_category_coverage(self):
        """Test that all major cost categories are covered."""
        categories = [cat.value for cat in BudgetCategory]
        
        # Essential categories for startup pipeline
        assert "openai_tokens" in categories
        assert "google_ads" in categories
        assert "infrastructure" in categories
        assert "external_apis" in categories


class TestAlertLevel:
    """Test AlertLevel enum."""
    
    def test_alert_level_values(self):
        """Test alert level enum values."""
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.CRITICAL.value == "critical"
        assert AlertLevel.EMERGENCY.value == "emergency"
    
    def test_alert_level_hierarchy(self):
        """Test alert level logical hierarchy."""
        levels = [level.value for level in AlertLevel]
        
        # Should have escalating severity levels
        assert "warning" in levels
        assert "critical" in levels
        assert "emergency" in levels


class TestBudgetAllocation:
    """Test BudgetAllocation data class."""
    
    def test_budget_allocation_creation(self, sample_budget_allocation):
        """Test budget allocation creation."""
        allocation = sample_budget_allocation
        
        assert allocation.category == BudgetCategory.OPENAI_TOKENS
        assert allocation.allocated == Decimal('10.00')
        assert allocation.spent == Decimal('3.00')
        assert allocation.warning_threshold == Decimal('0.80')
        assert allocation.critical_threshold == Decimal('0.95')
    
    def test_remaining_calculation(self, sample_budget_allocation):
        """Test remaining budget calculation."""
        allocation = sample_budget_allocation
        
        remaining = allocation.remaining
        assert remaining == Decimal('7.00')  # 10.00 - 3.00
    
    def test_usage_percentage_calculation(self, sample_budget_allocation):
        """Test usage percentage calculation."""
        allocation = sample_budget_allocation
        
        percentage = allocation.usage_percentage
        assert percentage == Decimal('30.00')  # (3.00 / 10.00) * 100
    
    def test_usage_percentage_zero_allocation(self):
        """Test usage percentage with zero allocation."""
        allocation = BudgetAllocation(
            category=BudgetCategory.OPENAI_TOKENS,
            allocated=Decimal('0.00'),
            spent=Decimal('0.00')
        )
        
        percentage = allocation.usage_percentage
        assert percentage == Decimal('0.00')
    
    def test_can_spend_within_budget(self, sample_budget_allocation):
        """Test can_spend method within budget."""
        allocation = sample_budget_allocation
        
        # Can spend $5 (remaining = $7)
        assert allocation.can_spend(Decimal('5.00')) is True
        
        # Can spend exactly remaining amount
        assert allocation.can_spend(Decimal('7.00')) is True
        
        # Cannot spend more than remaining
        assert allocation.can_spend(Decimal('8.00')) is False
    
    def test_can_spend_edge_cases(self):
        """Test can_spend edge cases."""
        # Fully spent allocation
        allocation = BudgetAllocation(
            category=BudgetCategory.OPENAI_TOKENS,
            allocated=Decimal('10.00'),
            spent=Decimal('10.00')
        )
        
        assert allocation.can_spend(Decimal('0.01')) is False
        assert allocation.can_spend(Decimal('0.00')) is True
        
        # Over-spent allocation (shouldn't happen but test defensive)
        allocation.spent = Decimal('12.00')
        assert allocation.can_spend(Decimal('0.00')) is True
        assert allocation.can_spend(Decimal('0.01')) is False


class TestCostTrackingRecord:
    """Test CostTrackingRecord data class."""
    
    def test_cost_tracking_record_creation(self, sample_cost_record):
        """Test cost tracking record creation."""
        record = sample_cost_record
        
        assert isinstance(record.timestamp, datetime)
        assert record.service == "openai_api"
        assert record.operation == "chat_completion"
        assert record.category == BudgetCategory.OPENAI_TOKENS
        assert record.amount == Decimal('0.25')
        assert record.cycle_id == "cycle_123456"
        assert record.metadata["model"] == "gpt-4"
        assert record.metadata["tokens"] == 1000
    
    def test_cost_tracking_record_defaults(self):
        """Test cost tracking record with minimal data."""
        record = CostTrackingRecord(
            timestamp=datetime.utcnow(),
            service="test_service",
            operation="test_operation",
            category=BudgetCategory.INFRASTRUCTURE,
            amount=Decimal('1.50'),
            cycle_id="test_cycle"
        )
        
        assert record.metadata == {}
    
    def test_cost_tracking_record_with_metadata(self):
        """Test cost tracking record with extensive metadata."""
        metadata = {
            "request_id": "req_123",
            "user_id": "user_456",
            "tokens_used": 2500,
            "model": "gpt-4-turbo",
            "completion_time_ms": 1234,
            "cache_hit": False
        }
        
        record = CostTrackingRecord(
            timestamp=datetime.utcnow(),
            service="openai_completion",
            operation="chat_completion",
            category=BudgetCategory.OPENAI_TOKENS,
            amount=Decimal('0.75'),
            cycle_id="cycle_789",
            metadata=metadata
        )
        
        assert record.metadata == metadata
        assert record.metadata["request_id"] == "req_123"
        assert record.metadata["tokens_used"] == 2500


class TestBudgetSentinelService:
    """Test BudgetSentinelService main functionality."""
    
    def test_service_initialization(self, budget_sentinel_service):
        """Test budget sentinel service initialization."""
        service = budget_sentinel_service
        
        # Check allocations were initialized
        assert len(service.allocations) == 4
        assert BudgetCategory.OPENAI_TOKENS in service.allocations
        assert BudgetCategory.GOOGLE_ADS in service.allocations
        assert BudgetCategory.INFRASTRUCTURE in service.allocations
        assert BudgetCategory.EXTERNAL_APIS in service.allocations
        
        # Check initial state
        assert service.emergency_shutdown is False
        assert service.circuit_breaker_active is False
        assert len(service.cost_records) == 0
        assert service.current_cycle_id.startswith("cycle_")
        assert isinstance(service.cycle_start_time, datetime)
    
    def test_allocation_initialization_amounts(self, budget_sentinel_service):
        """Test budget allocation amounts are correct."""
        service = budget_sentinel_service
        
        # Check specific allocations
        openai_allocation = service.allocations[BudgetCategory.OPENAI_TOKENS]
        assert openai_allocation.allocated == Decimal('10.00')
        
        google_ads_allocation = service.allocations[BudgetCategory.GOOGLE_ADS]
        assert google_ads_allocation.allocated == Decimal('45.00')
        
        infrastructure_allocation = service.allocations[BudgetCategory.INFRASTRUCTURE]
        assert infrastructure_allocation.allocated == Decimal('5.00')
        
        # External APIs should get remainder (62 - 10 - 45 - 5 = 2)
        external_apis_allocation = service.allocations[BudgetCategory.EXTERNAL_APIS]
        assert external_apis_allocation.allocated == Decimal('2.00')
        
        # Total should equal total budget
        total_allocated = sum(alloc.allocated for alloc in service.allocations.values())
        assert total_allocated == Decimal('62.00')
    
    def test_generate_cycle_id(self, budget_sentinel_service):
        """Test cycle ID generation."""
        service = budget_sentinel_service
        
        cycle_id = service._generate_cycle_id()
        assert cycle_id.startswith("cycle_")
        assert len(cycle_id) > 6  # More than just "cycle_"
        
        # Should be unique
        cycle_id_2 = service._generate_cycle_id()
        assert cycle_id != cycle_id_2
    
    def test_mask_amount_security(self, budget_sentinel_service):
        """Test amount masking for security logging."""
        service = budget_sentinel_service
        
        # Test various amount ranges
        assert service._mask_amount(Decimal('0.00')) == "$0.00"
        assert service._mask_amount(Decimal('0.50')) == "<$1"
        assert service._mask_amount(Decimal('0.99')) == "<$1"
        assert service._mask_amount(Decimal('5.00')) == "<$10"
        assert service._mask_amount(Decimal('9.99')) == "<$10"
        assert service._mask_amount(Decimal('50.00')) == "<$100"
        assert service._mask_amount(Decimal('99.99')) == "<$100"
        assert service._mask_amount(Decimal('100.00')) == ">=$100"
        assert service._mask_amount(Decimal('500.00')) == ">=$100"
    
    @pytest.mark.asyncio
    async def test_track_cost_basic(self, budget_sentinel_service):
        """Test basic cost tracking functionality."""
        service = budget_sentinel_service
        
        # Track a cost
        amount = Decimal('2.50')
        await service.track_cost(
            service="openai_api",
            operation="completion",
            category=BudgetCategory.OPENAI_TOKENS,
            amount=amount
        )
        
        # Check cost was recorded
        assert len(service.cost_records) == 1
        record = service.cost_records[0]
        assert record.service == "openai_api"
        assert record.operation == "completion"
        assert record.category == BudgetCategory.OPENAI_TOKENS
        assert record.amount == amount
        assert record.cycle_id == service.current_cycle_id
        
        # Check allocation was updated
        allocation = service.allocations[BudgetCategory.OPENAI_TOKENS]
        assert allocation.spent == amount
        assert allocation.remaining == Decimal('7.50')  # 10.00 - 2.50
    
    @pytest.mark.asyncio
    async def test_track_cost_with_metadata(self, budget_sentinel_service):
        """Test cost tracking with metadata."""
        service = budget_sentinel_service
        
        metadata = {
            "model": "gpt-4",
            "tokens": 1500,
            "user_session": "session_123"
        }
        
        await service.track_cost(
            service="openai_completion",
            operation="chat",
            category=BudgetCategory.OPENAI_TOKENS,
            amount=Decimal('1.25'),
            metadata=metadata
        )
        
        record = service.cost_records[0]
        assert record.metadata == metadata
        assert record.metadata["model"] == "gpt-4"
        assert record.metadata["tokens"] == 1500
    
    @pytest.mark.asyncio
    async def test_track_cost_multiple_categories(self, budget_sentinel_service):
        """Test tracking costs across multiple categories."""
        service = budget_sentinel_service
        
        # Track OpenAI cost
        await service.track_cost(
            service="openai",
            operation="completion",
            category=BudgetCategory.OPENAI_TOKENS,
            amount=Decimal('3.00')
        )
        
        # Track Google Ads cost
        await service.track_cost(
            service="google_ads",
            operation="campaign",
            category=BudgetCategory.GOOGLE_ADS,
            amount=Decimal('15.00')
        )
        
        # Track Infrastructure cost
        await service.track_cost(
            service="aws",
            operation="compute",
            category=BudgetCategory.INFRASTRUCTURE,
            amount=Decimal('2.00')
        )
        
        # Check all allocations updated
        assert service.allocations[BudgetCategory.OPENAI_TOKENS].spent == Decimal('3.00')
        assert service.allocations[BudgetCategory.GOOGLE_ADS].spent == Decimal('15.00')
        assert service.allocations[BudgetCategory.INFRASTRUCTURE].spent == Decimal('2.00')
        assert service.allocations[BudgetCategory.EXTERNAL_APIS].spent == Decimal('0.00')
        
        # Check total records
        assert len(service.cost_records) == 3
    
    @pytest.mark.asyncio
    async def test_budget_warning_threshold(self, budget_sentinel_service):
        """Test budget warning threshold triggering."""
        service = budget_sentinel_service
        
        # Set up warning callback
        warning_triggered = []
        def warning_callback(category, allocation, amount):
            warning_triggered.append((category, allocation, amount))
        
        service.register_alert_callback(AlertLevel.WARNING, warning_callback)
        
        # Spend up to warning threshold (80% of $10 = $8)
        await service.track_cost(
            service="openai",
            operation="completion",
            category=BudgetCategory.OPENAI_TOKENS,
            amount=Decimal('8.50')  # Over 80% threshold
        )
        
        # Check warning was triggered
        assert len(warning_triggered) == 1
        category, allocation, amount = warning_triggered[0]
        assert category == BudgetCategory.OPENAI_TOKENS
        assert amount == Decimal('8.50')
    
    @pytest.mark.asyncio
    async def test_budget_critical_threshold(self, budget_sentinel_service):
        """Test budget critical threshold triggering."""
        service = budget_sentinel_service
        
        # Set up critical callback
        critical_triggered = []
        def critical_callback(category, allocation, amount):
            critical_triggered.append((category, allocation, amount))
        
        service.register_alert_callback(AlertLevel.CRITICAL, critical_callback)
        
        # Spend up to critical threshold (95% of $10 = $9.50)
        await service.track_cost(
            service="openai",
            operation="completion",
            category=BudgetCategory.OPENAI_TOKENS,
            amount=Decimal('9.75')  # Over 95% threshold
        )
        
        # Check critical alert was triggered
        assert len(critical_triggered) == 1
        category, allocation, amount = critical_triggered[0]
        assert category == BudgetCategory.OPENAI_TOKENS
        assert amount == Decimal('9.75')
    
    @pytest.mark.asyncio
    async def test_budget_emergency_shutdown(self, budget_sentinel_service):
        """Test emergency shutdown on budget exceeded."""
        service = budget_sentinel_service
        
        # Set up emergency callback
        emergency_triggered = []
        def emergency_callback(category, allocation, amount):
            emergency_triggered.append((category, allocation, amount))
        
        service.register_alert_callback(AlertLevel.EMERGENCY, emergency_callback)
        
        # Exceed budget (spend $12 on $10 allocation)
        await service.track_cost(
            service="openai",
            operation="completion",
            category=BudgetCategory.OPENAI_TOKENS,
            amount=Decimal('12.00')  # Exceeds allocation
        )
        
        # Check emergency shutdown was triggered
        assert len(emergency_triggered) == 1
        assert service.emergency_shutdown is True
        assert service.circuit_breaker_active is True
    
    @pytest.mark.asyncio
    async def test_can_spend_check(self, budget_sentinel_service):
        """Test can_spend pre-flight check."""
        service = budget_sentinel_service
        
        # Should be able to spend within budget
        assert await service.can_spend(BudgetCategory.OPENAI_TOKENS, Decimal('5.00')) is True
        assert await service.can_spend(BudgetCategory.OPENAI_TOKENS, Decimal('10.00')) is True
        
        # Should not be able to exceed budget
        assert await service.can_spend(BudgetCategory.OPENAI_TOKENS, Decimal('15.00')) is False
        
        # After spending some, check remaining
        await service.track_cost(
            service="openai",
            operation="test",
            category=BudgetCategory.OPENAI_TOKENS,
            amount=Decimal('6.00')
        )
        
        assert await service.can_spend(BudgetCategory.OPENAI_TOKENS, Decimal('4.00')) is True
        assert await service.can_spend(BudgetCategory.OPENAI_TOKENS, Decimal('5.00')) is False
    
    @pytest.mark.asyncio
    async def test_emergency_shutdown_blocks_spending(self, budget_sentinel_service):
        """Test that emergency shutdown blocks further spending."""
        service = budget_sentinel_service
        
        # Trigger emergency shutdown
        service.emergency_shutdown = True
        service.circuit_breaker_active = True
        
        # Should not be able to spend anything
        assert await service.can_spend(BudgetCategory.OPENAI_TOKENS, Decimal('0.01')) is False
        assert await service.can_spend(BudgetCategory.GOOGLE_ADS, Decimal('1.00')) is False
        
        # Try to track cost - should raise exception or be blocked
        with pytest.raises(Exception):  # Should raise budget exceeded exception
            await service.track_cost(
                service="openai",
                operation="blocked",
                category=BudgetCategory.OPENAI_TOKENS,
                amount=Decimal('1.00')
            )
    
    def test_register_alert_callback(self, budget_sentinel_service):
        """Test registering alert callbacks."""
        service = budget_sentinel_service
        
        def warning_callback(category, allocation, amount):
            pass
        
        def critical_callback(category, allocation, amount):
            pass
        
        # Register callbacks
        service.register_alert_callback(AlertLevel.WARNING, warning_callback)
        service.register_alert_callback(AlertLevel.CRITICAL, critical_callback)
        
        # Check callbacks were registered
        assert len(service.alert_callbacks[AlertLevel.WARNING]) == 1
        assert len(service.alert_callbacks[AlertLevel.CRITICAL]) == 1
        assert len(service.alert_callbacks[AlertLevel.EMERGENCY]) == 0
        
        assert service.alert_callbacks[AlertLevel.WARNING][0] == warning_callback
        assert service.alert_callbacks[AlertLevel.CRITICAL][0] == critical_callback
    
    @pytest.mark.asyncio
    async def test_get_current_usage_report(self, budget_sentinel_service):
        """Test getting current usage report."""
        service = budget_sentinel_service
        
        # Track some costs
        await service.track_cost("openai", "completion", BudgetCategory.OPENAI_TOKENS, Decimal('3.00'))
        await service.track_cost("google_ads", "campaign", BudgetCategory.GOOGLE_ADS, Decimal('20.00'))
        await service.track_cost("aws", "compute", BudgetCategory.INFRASTRUCTURE, Decimal('1.50'))
        
        # Get usage report
        report = await service.get_current_usage_report()
        
        # Check report structure
        assert "cycle_id" in report
        assert "total_budget" in report
        assert "total_spent" in report
        assert "categories" in report
        assert "alerts_triggered" in report
        
        # Check totals
        assert report["total_budget"] == Decimal('62.00')
        assert report["total_spent"] == Decimal('24.50')  # 3 + 20 + 1.5
        
        # Check category details
        categories = report["categories"]
        assert len(categories) == 4
        
        openai_cat = next(cat for cat in categories if cat["category"] == BudgetCategory.OPENAI_TOKENS.value)
        assert openai_cat["spent"] == Decimal('3.00')
        assert openai_cat["allocated"] == Decimal('10.00')
    
    @pytest.mark.asyncio
    async def test_reset_cycle(self, budget_sentinel_service):
        """Test resetting budget cycle."""
        service = budget_sentinel_service
        
        # Track some costs and trigger alerts
        await service.track_cost("openai", "completion", BudgetCategory.OPENAI_TOKENS, Decimal('8.00'))
        
        original_cycle_id = service.current_cycle_id
        
        # Reset cycle
        await service.reset_cycle()
        
        # Check cycle was reset
        assert service.current_cycle_id != original_cycle_id
        assert service.current_cycle_id.startswith("cycle_")
        assert len(service.cost_records) == 0
        assert service.emergency_shutdown is False
        assert service.circuit_breaker_active is False
        
        # Check allocations were reset
        for allocation in service.allocations.values():
            assert allocation.spent == Decimal('0.00')
    
    @pytest.mark.asyncio
    async def test_concurrent_cost_tracking(self, budget_sentinel_service):
        """Test concurrent cost tracking operations."""
        service = budget_sentinel_service
        
        # Track multiple costs concurrently
        tasks = []
        for i in range(10):
            task = service.track_cost(
                service=f"service_{i}",
                operation=f"operation_{i}",
                category=BudgetCategory.OPENAI_TOKENS,
                amount=Decimal('0.50')
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks)
        
        # Check all costs were tracked
        assert len(service.cost_records) == 10
        
        # Check total spent
        total_spent = service.allocations[BudgetCategory.OPENAI_TOKENS].spent
        assert total_spent == Decimal('5.00')  # 10 * 0.50
    
    @pytest.mark.asyncio
    async def test_budget_context_manager(self, budget_sentinel_service):
        """Test budget context manager for safe spending."""
        service = budget_sentinel_service
        
        # Mock the context manager method
        @asynccontextmanager
        async def spend_with_budget_check(category: BudgetCategory, estimated_amount: Decimal):
            if not await service.can_spend(category, estimated_amount):
                raise ValueError(f"Insufficient budget for {category.value}")
            try:
                yield
            finally:
                # Would track actual cost here
                pass
        
        # Add method to service for testing
        service.spend_with_budget_check = spend_with_budget_check
        
        # Test successful spending within budget
        async with service.spend_with_budget_check(BudgetCategory.OPENAI_TOKENS, Decimal('5.00')):
            # Simulate work that costs money
            pass
        
        # Test spending that exceeds budget
        with pytest.raises(ValueError, match="Insufficient budget"):
            async with service.spend_with_budget_check(BudgetCategory.OPENAI_TOKENS, Decimal('15.00')):
                pass


class TestBudgetSentinelSingleton:
    """Test Budget Sentinel singleton functionality."""
    
    def test_get_budget_sentinel_singleton(self, mock_settings):
        """Test singleton pattern for budget sentinel."""
        with patch('pipeline.services.budget_sentinel.get_settings', return_value=mock_settings):
            # Reset singleton
            import pipeline.services.budget_sentinel
            pipeline.services.budget_sentinel._budget_sentinel = None
            
            # Get sentinel instances
            sentinel1 = get_budget_sentinel()
            sentinel2 = get_budget_sentinel()
            
            # Should be the same instance
            assert sentinel1 is sentinel2
            assert isinstance(sentinel1, BudgetSentinelService)


class TestBudgetSentinelIntegration:
    """Integration tests for Budget Sentinel Service."""
    
    @pytest.mark.asyncio
    async def test_realistic_startup_pipeline_budget_usage(self, budget_sentinel_service):
        """Test realistic budget usage pattern for startup pipeline."""
        service = budget_sentinel_service
        
        # Simulate typical startup validation pipeline costs
        
        # 1. Initial idea processing (OpenAI)
        await service.track_cost(
            service="openai_gpt4",
            operation="idea_analysis",
            category=BudgetCategory.OPENAI_TOKENS,
            amount=Decimal('2.50'),
            metadata={"stage": "ideation", "tokens": 5000}
        )
        
        # 2. Market research (External APIs)
        await service.track_cost(
            service="semantic_scholar",
            operation="market_research",
            category=BudgetCategory.EXTERNAL_APIS,
            amount=Decimal('0.50'),
            metadata={"papers_analyzed": 20}
        )
        
        # 3. Competitive analysis (Google Ads for keyword research)
        await service.track_cost(
            service="google_ads_api",
            operation="keyword_research",
            category=BudgetCategory.GOOGLE_ADS,
            amount=Decimal('15.00'),
            metadata={"keywords": 100, "campaigns": 5}
        )
        
        # 4. Technical feasibility (More OpenAI)
        await service.track_cost(
            service="openai_gpt4",
            operation="technical_analysis",
            category=BudgetCategory.OPENAI_TOKENS,
            amount=Decimal('3.00'),
            metadata={"stage": "technical_validation", "tokens": 6000}
        )
        
        # 5. Pitch deck generation (OpenAI)
        await service.track_cost(
            service="openai_gpt4",
            operation="pitch_deck_generation",
            category=BudgetCategory.OPENAI_TOKENS,
            amount=Decimal('1.50'),
            metadata={"stage": "deck_generation", "slides": 12}
        )
        
        # 6. Infrastructure for smoke test
        await service.track_cost(
            service="fly_io",
            operation="app_deployment",
            category=BudgetCategory.INFRASTRUCTURE,
            amount=Decimal('3.00'),
            metadata={"duration_hours": 24}
        )
        
        # 7. Additional Google Ads for smoke test
        await service.track_cost(
            service="google_ads_campaign",
            operation="smoke_test_ads",
            category=BudgetCategory.GOOGLE_ADS,
            amount=Decimal('25.00'),
            metadata={"impressions": 10000, "clicks": 150}
        )
        
        # Check usage patterns
        report = await service.get_current_usage_report()
        
        # Verify realistic distribution
        openai_spent = service.allocations[BudgetCategory.OPENAI_TOKENS].spent
        google_ads_spent = service.allocations[BudgetCategory.GOOGLE_ADS].spent
        infrastructure_spent = service.allocations[BudgetCategory.INFRASTRUCTURE].spent
        external_apis_spent = service.allocations[BudgetCategory.EXTERNAL_APIS].spent
        
        assert openai_spent == Decimal('7.00')  # Within $10 limit
        assert google_ads_spent == Decimal('40.00')  # Within $45 limit
        assert infrastructure_spent == Decimal('3.00')  # Within $5 limit
        assert external_apis_spent == Decimal('0.50')  # Within $2 limit
        
        total_spent = openai_spent + google_ads_spent + infrastructure_spent + external_apis_spent
        assert total_spent == Decimal('50.50')  # Within $62 total budget
        
        # Should not have triggered emergency shutdown
        assert service.emergency_shutdown is False
        
        # Check that we're efficiently using budget categories
        openai_utilization = (openai_spent / Decimal('10.00')) * 100
        google_ads_utilization = (google_ads_spent / Decimal('45.00')) * 100
        
        assert openai_utilization == Decimal('70.00')  # 70% utilization
        assert google_ads_utilization < Decimal('90.00')  # Under 90% utilization
    
    @pytest.mark.asyncio
    async def test_budget_exceeded_scenario(self, budget_sentinel_service):
        """Test scenario where budget limits are exceeded."""
        service = budget_sentinel_service
        
        # Set up alert tracking
        alerts_triggered = []
        def alert_handler(category, allocation, amount):
            alerts_triggered.append({
                "level": "emergency",
                "category": category,
                "amount": amount,
                "timestamp": datetime.utcnow()
            })
        
        service.register_alert_callback(AlertLevel.EMERGENCY, alert_handler)
        
        # Exceed OpenAI budget dramatically
        with pytest.raises(Exception):  # Should raise budget exceeded
            await service.track_cost(
                service="runaway_openai_usage",
                operation="excessive_generation",
                category=BudgetCategory.OPENAI_TOKENS,
                amount=Decimal('50.00'),  # Way over $10 limit
                metadata={"reason": "infinite_loop_bug"}
            )
        
        # Should have triggered emergency
        assert len(alerts_triggered) > 0
        assert service.emergency_shutdown is True
        
        # Further spending should be blocked
        assert await service.can_spend(BudgetCategory.GOOGLE_ADS, Decimal('1.00')) is False
    
    @pytest.mark.asyncio
    async def test_multi_cycle_budget_management(self, budget_sentinel_service):
        """Test budget management across multiple cycles."""
        service = budget_sentinel_service
        
        # Cycle 1: Normal usage
        await service.track_cost("openai", "cycle1", BudgetCategory.OPENAI_TOKENS, Decimal('5.00'))
        await service.track_cost("google_ads", "cycle1", BudgetCategory.GOOGLE_ADS, Decimal('20.00'))
        
        cycle1_id = service.current_cycle_id
        cycle1_total = sum(alloc.spent for alloc in service.allocations.values())
        
        # Reset for cycle 2
        await service.reset_cycle()
        cycle2_id = service.current_cycle_id
        
        # Cycle 2: Different usage pattern
        await service.track_cost("openai", "cycle2", BudgetCategory.OPENAI_TOKENS, Decimal('8.00'))
        await service.track_cost("infrastructure", "cycle2", BudgetCategory.INFRASTRUCTURE, Decimal('4.00'))
        
        cycle2_total = sum(alloc.spent for alloc in service.allocations.values())
        
        # Verify cycles are independent
        assert cycle1_id != cycle2_id
        assert cycle1_total == Decimal('25.00')
        assert cycle2_total == Decimal('12.00')
        
        # Verify cycle 2 started fresh
        assert len(service.cost_records) == 2  # Only cycle 2 records remain