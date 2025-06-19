"""
Events package - Event-driven communication components.

This package provides the event bus system and domain events for 
inter-service communication in the startup studio pipeline.
"""

from .event_bus import (
    EventType,
    DomainEvent,
    EventHandler,
    EventBus,
    get_event_bus
)

__all__ = [
    'EventType',
    'DomainEvent', 
    'EventHandler',
    'EventBus',
    'get_event_bus'
]