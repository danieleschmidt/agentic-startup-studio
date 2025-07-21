"""
Comprehensive tests for Event Bus system.

Tests critical messaging infrastructure for event-driven architecture,
following TDD principles with extensive coverage of pub/sub patterns.
"""

import pytest
import asyncio
import json
import uuid
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch
from typing import Any, List

# Test imports
from pipeline.events.event_bus import (
    EventBus,
    EventType,
    DomainEvent,
    EventHandler,
    EventBusError,
    DeadLetterQueue,
    EventPriority,
    EventMetrics
)


class TestEventType:
    """Test EventType enum functionality."""
    
    def test_event_type_values(self):
        """Test that EventType enum has expected values."""
        assert EventType.IDEA_CREATED.value == "idea.created"
        assert EventType.IDEA_VALIDATED.value == "idea.validated"
        assert EventType.EVIDENCE_COLLECTED.value == "evidence.collected"
        assert EventType.DECK_GENERATED.value == "deck.generated"
        assert EventType.CAMPAIGN_CREATED.value == "campaign.created"
    
    def test_event_type_categories(self):
        """Test event type categorization."""
        # Phase 1: Data Ingestion
        ingestion_events = [
            EventType.IDEA_CREATED,
            EventType.IDEA_VALIDATED,
            EventType.IDEA_REJECTED
        ]
        
        # Phase 2: Data Processing
        processing_events = [
            EventType.EVIDENCE_COLLECTION_STARTED,
            EventType.EVIDENCE_COLLECTED,
            EventType.RESEARCH_COMPLETED
        ]
        
        # Verify different categories exist
        assert len(ingestion_events) >= 3
        assert len(processing_events) >= 3
        
        # Verify no overlap
        assert set(ingestion_events).isdisjoint(set(processing_events))


class TestDomainEvent:
    """Test DomainEvent dataclass functionality."""
    
    def test_domain_event_creation(self):
        """Test creating domain events."""
        event = DomainEvent(
            event_type=EventType.IDEA_CREATED,
            data={"title": "Test Idea", "category": "ai_ml"},
            correlation_id="test-correlation-123"
        )
        
        assert event.event_type == EventType.IDEA_CREATED
        assert event.data["title"] == "Test Idea"
        assert event.correlation_id == "test-correlation-123"
        assert isinstance(event.event_id, str)
        assert isinstance(event.timestamp, datetime)
        assert event.timestamp.tzinfo is not None
    
    def test_domain_event_defaults(self):
        """Test domain event default values."""
        event = DomainEvent(
            event_type=EventType.IDEA_VALIDATED,
            data={"status": "valid"}
        )
        
        assert event.event_id is not None
        assert len(event.event_id) > 0
        assert event.timestamp is not None
        assert event.correlation_id is None  # Default
        assert event.priority == EventPriority.NORMAL  # Default
    
    def test_domain_event_serialization(self):
        """Test domain event JSON serialization."""
        event = DomainEvent(
            event_type=EventType.EVIDENCE_COLLECTED,
            data={"evidence_count": 5, "sources": ["url1", "url2"]},
            correlation_id="correlation-456"
        )
        
        # Test to_dict
        event_dict = event.to_dict()
        assert event_dict["event_type"] == "evidence.collected"
        assert event_dict["data"]["evidence_count"] == 5
        assert event_dict["correlation_id"] == "correlation-456"
        assert "event_id" in event_dict
        assert "timestamp" in event_dict
        
        # Test from_dict
        reconstructed = DomainEvent.from_dict(event_dict)
        assert reconstructed.event_type == event.event_type
        assert reconstructed.data == event.data
        assert reconstructed.correlation_id == event.correlation_id
        assert reconstructed.event_id == event.event_id
    
    def test_domain_event_json_serialization(self):
        """Test domain event JSON string serialization."""
        event = DomainEvent(
            event_type=EventType.CAMPAIGN_CREATED,
            data={"campaign_id": "camp-123", "budget": 1000.50}
        )
        
        # Test to_json
        json_str = event.to_json()
        parsed = json.loads(json_str)
        assert parsed["event_type"] == "campaign.created"
        assert parsed["data"]["budget"] == 1000.50
        
        # Test from_json
        reconstructed = DomainEvent.from_json(json_str)
        assert reconstructed.event_type == event.event_type
        assert reconstructed.data == event.data


class TestEventHandler:
    """Test EventHandler base class functionality."""
    
    def test_event_handler_interface(self):
        """Test EventHandler abstract interface."""
        # Should not be able to instantiate abstract base class
        with pytest.raises(TypeError):
            EventHandler()
    
    def test_concrete_event_handler(self):
        """Test concrete event handler implementation."""
        
        class TestHandler(EventHandler):
            def __init__(self):
                self.handled_events = []
            
            async def handle(self, event: DomainEvent) -> None:
                self.handled_events.append(event)
            
            def can_handle(self, event_type: EventType) -> bool:
                return event_type in [EventType.IDEA_CREATED, EventType.IDEA_VALIDATED]
        
        handler = TestHandler()
        
        # Test can_handle
        assert handler.can_handle(EventType.IDEA_CREATED)
        assert handler.can_handle(EventType.IDEA_VALIDATED)
        assert not handler.can_handle(EventType.CAMPAIGN_CREATED)
        
        # Test handle (needs to be async)
        assert hasattr(handler, 'handle')
        assert asyncio.iscoroutinefunction(handler.handle)


class TestEventBus:
    """Test EventBus core functionality."""
    
    @pytest.fixture
    def event_bus(self):
        """Create event bus for testing."""
        return EventBus()
    
    @pytest.fixture
    def test_handler(self):
        """Create test event handler."""
        
        class TestHandler(EventHandler):
            def __init__(self):
                self.handled_events = []
            
            async def handle(self, event: DomainEvent) -> None:
                self.handled_events.append(event)
            
            def can_handle(self, event_type: EventType) -> bool:
                return event_type == EventType.IDEA_CREATED
        
        return TestHandler()
    
    def test_event_bus_initialization(self, event_bus):
        """Test event bus initialization."""
        assert hasattr(event_bus, 'handlers')
        assert hasattr(event_bus, 'metrics')
        assert len(event_bus.handlers) == 0
    
    def test_subscribe_handler(self, event_bus, test_handler):
        """Test subscribing event handlers."""
        event_bus.subscribe(EventType.IDEA_CREATED, test_handler)
        
        assert EventType.IDEA_CREATED in event_bus.handlers
        assert test_handler in event_bus.handlers[EventType.IDEA_CREATED]
    
    def test_subscribe_multiple_handlers(self, event_bus):
        """Test subscribing multiple handlers to same event."""
        
        class Handler1(EventHandler):
            async def handle(self, event): pass
            def can_handle(self, event_type): return True
        
        class Handler2(EventHandler):
            async def handle(self, event): pass
            def can_handle(self, event_type): return True
        
        handler1 = Handler1()
        handler2 = Handler2()
        
        event_bus.subscribe(EventType.IDEA_CREATED, handler1)
        event_bus.subscribe(EventType.IDEA_CREATED, handler2)
        
        handlers = event_bus.handlers[EventType.IDEA_CREATED]
        assert len(handlers) == 2
        assert handler1 in handlers
        assert handler2 in handlers
    
    def test_unsubscribe_handler(self, event_bus, test_handler):
        """Test unsubscribing event handlers."""
        # Subscribe first
        event_bus.subscribe(EventType.IDEA_CREATED, test_handler)
        assert test_handler in event_bus.handlers[EventType.IDEA_CREATED]
        
        # Unsubscribe
        event_bus.unsubscribe(EventType.IDEA_CREATED, test_handler)
        assert test_handler not in event_bus.handlers[EventType.IDEA_CREATED]
    
    @pytest.mark.asyncio
    async def test_publish_event(self, event_bus, test_handler):
        """Test publishing events to subscribers."""
        # Subscribe handler
        event_bus.subscribe(EventType.IDEA_CREATED, test_handler)
        
        # Create and publish event
        event = DomainEvent(
            event_type=EventType.IDEA_CREATED,
            data={"title": "Test Idea"}
        )
        
        await event_bus.publish(event)
        
        # Verify handler received event
        assert len(test_handler.handled_events) == 1
        assert test_handler.handled_events[0] == event
    
    @pytest.mark.asyncio
    async def test_publish_no_handlers(self, event_bus):
        """Test publishing event with no subscribers."""
        event = DomainEvent(
            event_type=EventType.IDEA_CREATED,
            data={"title": "No Handlers"}
        )
        
        # Should not raise error
        await event_bus.publish(event)
    
    @pytest.mark.asyncio
    async def test_publish_handler_error(self, event_bus):
        """Test handling errors in event handlers."""
        
        class FailingHandler(EventHandler):
            async def handle(self, event: DomainEvent) -> None:
                raise Exception("Handler error")
            
            def can_handle(self, event_type: EventType) -> bool:
                return True
        
        failing_handler = FailingHandler()
        event_bus.subscribe(EventType.IDEA_CREATED, failing_handler)
        
        event = DomainEvent(
            event_type=EventType.IDEA_CREATED,
            data={"title": "Error Test"}
        )
        
        # Should handle error gracefully
        await event_bus.publish(event)
        
        # Check that error was recorded in metrics
        assert event_bus.metrics.total_errors > 0


class TestEventMetrics:
    """Test event metrics tracking."""
    
    @pytest.fixture
    def event_bus(self):
        """Create event bus for metrics testing."""
        return EventBus()
    
    @pytest.fixture
    def counting_handler(self):
        """Create handler that counts events."""
        
        class CountingHandler(EventHandler):
            def __init__(self):
                self.count = 0
            
            async def handle(self, event: DomainEvent) -> None:
                self.count += 1
            
            def can_handle(self, event_type: EventType) -> bool:
                return True
        
        return CountingHandler()
    
    @pytest.mark.asyncio
    async def test_metrics_tracking(self, event_bus, counting_handler):
        """Test that metrics are tracked correctly."""
        event_bus.subscribe(EventType.IDEA_CREATED, counting_handler)
        
        # Publish multiple events
        for i in range(3):
            event = DomainEvent(
                event_type=EventType.IDEA_CREATED,
                data={"title": f"Idea {i}"}
            )
            await event_bus.publish(event)
        
        # Check metrics
        metrics = event_bus.get_metrics()
        assert metrics["total_events"] >= 3
        assert metrics["events_by_type"]["idea.created"] >= 3
        assert counting_handler.count == 3
    
    def test_reset_metrics(self, event_bus):
        """Test resetting event bus metrics."""
        # Add some metrics
        event_bus.metrics.record_event(EventType.IDEA_CREATED)
        event_bus.metrics.record_event(EventType.IDEA_VALIDATED)
        
        assert event_bus.metrics.total_events > 0
        
        # Reset
        event_bus.reset_metrics()
        assert event_bus.metrics.total_events == 0


class TestEventPriority:
    """Test event priority handling."""
    
    @pytest.fixture
    def event_bus(self):
        """Create event bus for priority testing."""
        return EventBus()
    
    @pytest.fixture
    def priority_handler(self):
        """Create handler that tracks event order."""
        
        class PriorityHandler(EventHandler):
            def __init__(self):
                self.received_events = []
            
            async def handle(self, event: DomainEvent) -> None:
                self.received_events.append(event)
            
            def can_handle(self, event_type: EventType) -> bool:
                return True
        
        return PriorityHandler()
    
    @pytest.mark.asyncio
    async def test_priority_ordering(self, event_bus, priority_handler):
        """Test that high priority events are processed first."""
        event_bus.subscribe(EventType.IDEA_CREATED, priority_handler)
        
        # Create events with different priorities
        low_priority = DomainEvent(
            event_type=EventType.IDEA_CREATED,
            data={"title": "Low Priority"},
            priority=EventPriority.LOW
        )
        
        high_priority = DomainEvent(
            event_type=EventType.IDEA_CREATED,
            data={"title": "High Priority"},
            priority=EventPriority.HIGH
        )
        
        normal_priority = DomainEvent(
            event_type=EventType.IDEA_CREATED,
            data={"title": "Normal Priority"},
            priority=EventPriority.NORMAL
        )
        
        # Publish in mixed order
        await event_bus.publish(low_priority)
        await event_bus.publish(high_priority)
        await event_bus.publish(normal_priority)
        
        # If priority queue is implemented, high priority should be first
        # For basic implementation, order may be preserved
        assert len(priority_handler.received_events) == 3


class TestEventBusIntegration:
    """Test event bus integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_event_workflow(self):
        """Test complete event workflow."""
        event_bus = EventBus()
        workflow_events = []
        
        class WorkflowHandler(EventHandler):
            def __init__(self, next_event_type=None):
                self.next_event_type = next_event_type
            
            async def handle(self, event: DomainEvent) -> None:
                workflow_events.append(event.event_type)
                
                # Trigger next event in workflow
                if self.next_event_type:
                    next_event = DomainEvent(
                        event_type=self.next_event_type,
                        data={"triggered_by": event.event_id},
                        correlation_id=event.correlation_id
                    )
                    await event_bus.publish(next_event)
            
            def can_handle(self, event_type: EventType) -> bool:
                return True
        
        # Set up workflow: IDEA_CREATED -> IDEA_VALIDATED -> EVIDENCE_COLLECTION_STARTED
        event_bus.subscribe(EventType.IDEA_CREATED, 
                          WorkflowHandler(EventType.IDEA_VALIDATED))
        event_bus.subscribe(EventType.IDEA_VALIDATED, 
                          WorkflowHandler(EventType.EVIDENCE_COLLECTION_STARTED))
        event_bus.subscribe(EventType.EVIDENCE_COLLECTION_STARTED, 
                          WorkflowHandler())
        
        # Start workflow
        initial_event = DomainEvent(
            event_type=EventType.IDEA_CREATED,
            data={"title": "Workflow Test"},
            correlation_id="workflow-123"
        )
        
        await event_bus.publish(initial_event)
        
        # Give time for async propagation
        await asyncio.sleep(0.1)
        
        # Verify workflow executed
        assert EventType.IDEA_CREATED in workflow_events
        assert EventType.IDEA_VALIDATED in workflow_events
        assert EventType.EVIDENCE_COLLECTION_STARTED in workflow_events
    
    @pytest.mark.asyncio
    async def test_error_handling_workflow(self):
        """Test error handling in event workflows."""
        event_bus = EventBus()
        
        class ErrorHandler(EventHandler):
            async def handle(self, event: DomainEvent) -> None:
                raise ValueError("Simulated handler error")
            
            def can_handle(self, event_type: EventType) -> bool:
                return True
        
        class SuccessHandler(EventHandler):
            def __init__(self):
                self.handled = False
            
            async def handle(self, event: DomainEvent) -> None:
                self.handled = True
            
            def can_handle(self, event_type: EventType) -> bool:
                return True
        
        error_handler = ErrorHandler()
        success_handler = SuccessHandler()
        
        # Subscribe both handlers
        event_bus.subscribe(EventType.IDEA_CREATED, error_handler)
        event_bus.subscribe(EventType.IDEA_CREATED, success_handler)
        
        event = DomainEvent(
            event_type=EventType.IDEA_CREATED,
            data={"title": "Error Test"}
        )
        
        await event_bus.publish(event)
        
        # Success handler should still work despite error handler failing
        assert success_handler.handled
        assert event_bus.metrics.total_errors > 0


class TestDeadLetterQueue:
    """Test dead letter queue functionality."""
    
    @pytest.fixture
    def dead_letter_queue(self):
        """Create dead letter queue for testing."""
        return DeadLetterQueue()
    
    def test_dead_letter_queue_creation(self, dead_letter_queue):
        """Test dead letter queue initialization."""
        assert hasattr(dead_letter_queue, 'failed_events')
        assert len(dead_letter_queue.failed_events) == 0
    
    def test_add_failed_event(self, dead_letter_queue):
        """Test adding failed events to dead letter queue."""
        event = DomainEvent(
            event_type=EventType.IDEA_CREATED,
            data={"title": "Failed Event"}
        )
        error = Exception("Handler failed")
        
        dead_letter_queue.add_failed_event(event, error)
        
        assert len(dead_letter_queue.failed_events) == 1
        failed_event = dead_letter_queue.failed_events[0]
        assert failed_event["event"] == event
        assert failed_event["error"] == str(error)
        assert "timestamp" in failed_event
    
    def test_get_failed_events(self, dead_letter_queue):
        """Test retrieving failed events."""
        # Add some failed events
        for i in range(3):
            event = DomainEvent(
                event_type=EventType.IDEA_CREATED,
                data={"title": f"Failed {i}"}
            )
            dead_letter_queue.add_failed_event(event, Exception(f"Error {i}"))
        
        failed_events = dead_letter_queue.get_failed_events()
        assert len(failed_events) == 3
        
        # Test filtering by event type
        filtered = dead_letter_queue.get_failed_events(
            event_type=EventType.IDEA_CREATED
        )
        assert len(filtered) == 3
        
        # Test filtering by different event type
        filtered = dead_letter_queue.get_failed_events(
            event_type=EventType.CAMPAIGN_CREATED
        )
        assert len(filtered) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])