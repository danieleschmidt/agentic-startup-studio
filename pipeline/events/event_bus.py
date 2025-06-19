"""
Event Bus System - Event-driven microservices communication backbone.

Implements asynchronous event publishing/subscription with domain event patterns,
ensuring loosely coupled service communication and reliable message delivery.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable, Type, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
import weakref

from pipeline.config.settings import get_settings

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Domain event types for the startup studio pipeline."""
    
    # Phase 1: Data Ingestion Events
    IDEA_CREATED = "idea.created"
    IDEA_VALIDATED = "idea.validated"
    IDEA_REJECTED = "idea.rejected"
    DUPLICATION_DETECTED = "idea.duplication_detected"
    
    # Phase 2: Data Processing Events
    EVIDENCE_COLLECTION_STARTED = "evidence.collection_started"
    EVIDENCE_COLLECTED = "evidence.collected"
    RESEARCH_COMPLETED = "research.completed"
    PROCESSING_COMPLETE = "processing.complete"
    
    # Phase 3: Data Transformation Events
    DECK_GENERATION_STARTED = "deck.generation_started"
    DECK_GENERATED = "deck.generated"
    TRANSFORMATION_COMPLETE = "transformation.complete"
    
    # Phase 4: Data Output Events
    CAMPAIGN_CREATED = "campaign.created"
    CAMPAIGN_DEPLOYED = "campaign.deployed"
    MVP_GENERATED = "mvp.generated"
    OUTPUT_COMPLETE = "output.complete"
    
    # Cross-cutting Events
    BUDGET_WARNING = "budget.warning"
    BUDGET_CRITICAL = "budget.critical"
    BUDGET_EMERGENCY = "budget.emergency"
    QUALITY_GATE_FAILED = "quality.gate_failed"
    CIRCUIT_BREAKER_OPENED = "circuit_breaker.opened"
    WORKFLOW_STATE_CHANGED = "workflow.state_changed"
    
    # Error Events
    SERVICE_ERROR = "service.error"
    PIPELINE_FAILED = "pipeline.failed"
    EXTERNAL_API_ERROR = "external_api.error"


@dataclass
class DomainEvent:
    """Base domain event with standard metadata."""
    
    event_type: EventType
    aggregate_id: str
    event_data: Dict[str, Any]
    
    # Metadata
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    
    # Event sourcing metadata
    version: int = 1
    sequence_number: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        event_dict = asdict(self)
        event_dict['event_type'] = self.event_type.value
        event_dict['timestamp'] = self.timestamp.isoformat()
        return event_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DomainEvent':
        """Create event from dictionary."""
        data = data.copy()
        data['event_type'] = EventType(data['event_type'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class EventHandler(ABC):
    """Abstract base class for event handlers."""
    
    @abstractmethod
    async def handle(self, event: DomainEvent) -> None:
        """Handle the domain event."""
        pass
    
    @property
    @abstractmethod
    def handled_events(self) -> List[EventType]:
        """List of event types this handler processes."""
        pass


@dataclass
class EventSubscription:
    """Event subscription configuration."""
    
    event_types: List[EventType]
    handler: EventHandler
    subscription_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Subscription options
    max_retries: int = 3
    retry_delay: float = 1.0
    dead_letter_enabled: bool = True
    
    # Filtering
    aggregate_id_filter: Optional[str] = None
    correlation_id_filter: Optional[str] = None


class EventStore:
    """Event store for event sourcing and replay capabilities."""
    
    def __init__(self):
        self.events: List[DomainEvent] = []
        self.sequence_counter = 0
        self._lock = asyncio.Lock()
        
    async def append_event(self, event: DomainEvent) -> None:
        """Append event to the store."""
        async with self._lock:
            self.sequence_counter += 1
            event.sequence_number = self.sequence_counter
            self.events.append(event)
            
            logger.debug(
                f"Event appended to store: {event.event_type.value} "
                f"(seq: {event.sequence_number}, id: {event.event_id})"
            )
    
    async def get_events(
        self,
        aggregate_id: Optional[str] = None,
        event_types: Optional[List[EventType]] = None,
        from_sequence: Optional[int] = None,
        to_sequence: Optional[int] = None
    ) -> List[DomainEvent]:
        """Get events with filtering."""
        async with self._lock:
            filtered_events = self.events.copy()
            
            if aggregate_id:
                filtered_events = [e for e in filtered_events if e.aggregate_id == aggregate_id]
            
            if event_types:
                filtered_events = [e for e in filtered_events if e.event_type in event_types]
            
            if from_sequence is not None:
                filtered_events = [e for e in filtered_events if e.sequence_number and e.sequence_number >= from_sequence]
            
            if to_sequence is not None:
                filtered_events = [e for e in filtered_events if e.sequence_number and e.sequence_number <= to_sequence]
            
            return filtered_events
    
    async def get_last_sequence_number(self) -> int:
        """Get the last sequence number."""
        async with self._lock:
            return self.sequence_counter


class EventBus:
    """
    Event bus implementation for publish/subscribe messaging.
    
    Provides asynchronous event publishing with reliable delivery,
    subscription management, and error handling.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)
        
        # Subscription management
        self._subscriptions: Dict[EventType, List[EventSubscription]] = {}
        self._subscription_lookup: Dict[str, EventSubscription] = {}
        
        # Event store for persistence and replay
        self.event_store = EventStore()
        
        # Delivery tracking
        self._delivery_queue: asyncio.Queue = asyncio.Queue()
        self._dead_letter_queue: List[tuple] = []  # (event, subscription, error)
        
        # Background tasks
        self._delivery_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Metrics
        self._published_count = 0
        self._delivered_count = 0
        self._failed_count = 0
        self._retry_count = 0
    
    async def start(self) -> None:
        """Start the event bus background processing."""
        if self._running:
            return
        
        self._running = True
        self._delivery_task = asyncio.create_task(self._process_delivery_queue())
        
        self.logger.info("Event bus started")
    
    async def stop(self) -> None:
        """Stop the event bus and cleanup resources."""
        if not self._running:
            return
        
        self._running = False
        
        if self._delivery_task:
            self._delivery_task.cancel()
            try:
                await self._delivery_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Event bus stopped")
    
    async def publish(
        self,
        event: DomainEvent,
        ensure_delivery: bool = True
    ) -> None:
        """
        Publish a domain event to all subscribers.
        
        Args:
            event: The domain event to publish
            ensure_delivery: Whether to ensure reliable delivery
        """
        if not self._running:
            await self.start()
        
        # Store event for event sourcing
        await self.event_store.append_event(event)
        
        # Track publishing metrics
        self._published_count += 1
        
        # Get subscribers for this event type
        subscriptions = self._subscriptions.get(event.event_type, [])
        
        self.logger.info(
            f"Publishing event {event.event_type.value} to {len(subscriptions)} subscribers",
            extra={
                'event_id': event.event_id,
                'event_type': event.event_type.value,
                'aggregate_id': event.aggregate_id,
                'correlation_id': event.correlation_id,
                'subscriber_count': len(subscriptions)
            }
        )
        
        # Queue for delivery
        for subscription in subscriptions:
            if self._should_deliver_to_subscription(event, subscription):
                await self._delivery_queue.put((event, subscription, 0))  # 0 = retry count
    
    def subscribe(
        self,
        event_types: Union[EventType, List[EventType]],
        handler: EventHandler,
        **subscription_options
    ) -> str:
        """
        Subscribe to event types with a handler.
        
        Args:
            event_types: Event type(s) to subscribe to
            handler: Handler function or class
            **subscription_options: Additional subscription configuration
            
        Returns:
            Subscription ID for management
        """
        if isinstance(event_types, EventType):
            event_types = [event_types]
        
        subscription = EventSubscription(
            event_types=event_types,
            handler=handler,
            **subscription_options
        )
        
        # Store subscription lookup
        self._subscription_lookup[subscription.subscription_id] = subscription
        
        # Add to event type mappings
        for event_type in event_types:
            if event_type not in self._subscriptions:
                self._subscriptions[event_type] = []
            self._subscriptions[event_type].append(subscription)
        
        self.logger.info(
            f"Subscription created for events {[et.value for et in event_types]}",
            extra={
                'subscription_id': subscription.subscription_id,
                'handler': handler.__class__.__name__,
                'event_types': [et.value for et in event_types]
            }
        )
        
        return subscription.subscription_id
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe using subscription ID.
        
        Args:
            subscription_id: The subscription to remove
            
        Returns:
            True if subscription was found and removed
        """
        subscription = self._subscription_lookup.get(subscription_id)
        if not subscription:
            return False
        
        # Remove from event type mappings
        for event_type in subscription.event_types:
            if event_type in self._subscriptions:
                self._subscriptions[event_type] = [
                    s for s in self._subscriptions[event_type]
                    if s.subscription_id != subscription_id
                ]
                
                # Clean up empty lists
                if not self._subscriptions[event_type]:
                    del self._subscriptions[event_type]
        
        # Remove from lookup
        del self._subscription_lookup[subscription_id]
        
        self.logger.info(f"Subscription removed: {subscription_id}")
        return True
    
    def _should_deliver_to_subscription(
        self,
        event: DomainEvent,
        subscription: EventSubscription
    ) -> bool:
        """Check if event should be delivered to subscription based on filters."""
        
        # Check aggregate ID filter
        if (subscription.aggregate_id_filter and 
            subscription.aggregate_id_filter != event.aggregate_id):
            return False
        
        # Check correlation ID filter
        if (subscription.correlation_id_filter and 
            subscription.correlation_id_filter != event.correlation_id):
            return False
        
        return True
    
    async def _process_delivery_queue(self) -> None:
        """Background task to process event delivery queue."""
        while self._running:
            try:
                # Get next delivery with timeout
                try:
                    event, subscription, retry_count = await asyncio.wait_for(
                        self._delivery_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Attempt delivery
                success = await self._deliver_event(event, subscription, retry_count)
                
                if success:
                    self._delivered_count += 1
                else:
                    self._failed_count += 1
                
            except Exception as e:
                self.logger.error(f"Error in delivery queue processing: {e}")
    
    async def _deliver_event(
        self,
        event: DomainEvent,
        subscription: EventSubscription,
        retry_count: int
    ) -> bool:
        """
        Deliver event to a specific subscription.
        
        Args:
            event: The event to deliver
            subscription: The subscription to deliver to
            retry_count: Current retry attempt count
            
        Returns:
            True if delivery successful, False otherwise
        """
        try:
            # Deliver to handler
            await subscription.handler.handle(event)
            
            self.logger.debug(
                f"Event delivered successfully: {event.event_type.value} -> {subscription.handler.__class__.__name__}",
                extra={
                    'event_id': event.event_id,
                    'subscription_id': subscription.subscription_id,
                    'retry_count': retry_count
                }
            )
            
            return True
            
        except Exception as e:
            self.logger.warning(
                f"Event delivery failed: {event.event_type.value} -> {subscription.handler.__class__.__name__}: {e}",
                extra={
                    'event_id': event.event_id,
                    'subscription_id': subscription.subscription_id,
                    'retry_count': retry_count,
                    'error': str(e)
                }
            )
            
            # Handle retries
            if retry_count < subscription.max_retries:
                self._retry_count += 1
                
                # Schedule retry with exponential backoff
                delay = subscription.retry_delay * (2 ** retry_count)
                await asyncio.sleep(delay)
                
                # Re-queue for retry
                await self._delivery_queue.put((event, subscription, retry_count + 1))
                
                return False
            
            # Max retries exceeded - send to dead letter queue
            if subscription.dead_letter_enabled:
                self._dead_letter_queue.append((event, subscription, e))
                
                self.logger.error(
                    f"Event sent to dead letter queue after {retry_count} retries: {event.event_type.value}",
                    extra={
                        'event_id': event.event_id,
                        'subscription_id': subscription.subscription_id,
                        'final_error': str(e)
                    }
                )
            
            return False
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get event bus metrics."""
        return {
            'published_count': self._published_count,
            'delivered_count': self._delivered_count,
            'failed_count': self._failed_count,
            'retry_count': self._retry_count,
            'active_subscriptions': len(self._subscription_lookup),
            'dead_letter_count': len(self._dead_letter_queue),
            'queue_size': self._delivery_queue.qsize(),
            'last_sequence_number': await self.event_store.get_last_sequence_number()
        }
    
    async def replay_events(
        self,
        handler: EventHandler,
        from_sequence: Optional[int] = None,
        event_types: Optional[List[EventType]] = None
    ) -> int:
        """
        Replay historical events to a handler.
        
        Args:
            handler: Handler to replay events to
            from_sequence: Start from this sequence number
            event_types: Only replay these event types
            
        Returns:
            Number of events replayed
        """
        events = await self.event_store.get_events(
            event_types=event_types,
            from_sequence=from_sequence
        )
        
        replayed_count = 0
        
        for event in events:
            try:
                await handler.handle(event)
                replayed_count += 1
            except Exception as e:
                self.logger.error(
                    f"Error replaying event {event.event_id}: {e}",
                    extra={
                        'event_id': event.event_id,
                        'event_type': event.event_type.value,
                        'handler': handler.__class__.__name__
                    }
                )
        
        self.logger.info(
            f"Replayed {replayed_count} events to {handler.__class__.__name__}",
            extra={
                'replayed_count': replayed_count,
                'total_events': len(events),
                'handler': handler.__class__.__name__
            }
        )
        
        return replayed_count


# Singleton instance
_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get singleton Event Bus instance."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus