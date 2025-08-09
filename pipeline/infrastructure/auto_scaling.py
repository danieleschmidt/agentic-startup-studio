"""
Auto-Scaling Infrastructure for Agentic Startup Studio
Provides intelligent resource scaling based on load and performance metrics
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from enum import Enum


logger = logging.getLogger(__name__)


class ScalingTrigger(Enum):
    """Types of scaling triggers."""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    REQUEST_RATE = "request_rate"
    QUEUE_LENGTH = "queue_length"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"


@dataclass
class ScalingMetrics:
    """Current system metrics for scaling decisions."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    request_rate_per_second: float = 0.0
    avg_response_time_ms: float = 0.0
    error_rate_percent: float = 0.0
    active_connections: int = 0
    queue_length: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/monitoring."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_usage_percent': self.cpu_usage_percent,
            'memory_usage_percent': self.memory_usage_percent,
            'request_rate_per_second': self.request_rate_per_second,
            'avg_response_time_ms': self.avg_response_time_ms,
            'error_rate_percent': self.error_rate_percent,
            'active_connections': self.active_connections,
            'queue_length': self.queue_length,
        }


@dataclass
class ScalingRule:
    """Configuration for an auto-scaling rule."""
    name: str
    trigger: ScalingTrigger
    threshold_up: float  # Scale up when metric exceeds this
    threshold_down: float  # Scale down when metric falls below this
    scale_up_action: Callable[[], None]
    scale_down_action: Callable[[], None]
    cooldown_seconds: int = 300  # Minimum time between scaling actions
    enabled: bool = True
    last_action_time: Optional[datetime] = None


class AutoScaler:
    """
    Intelligent auto-scaling system that monitors metrics and triggers scaling actions.
    """
    
    def __init__(self):
        self.metrics_history: List[ScalingMetrics] = []
        self.scaling_rules: Dict[str, ScalingRule] = {}
        self.current_metrics = ScalingMetrics()
        self.monitoring_enabled = False
        self._monitoring_task: Optional[asyncio.Task] = None
        
    def add_scaling_rule(self, rule: ScalingRule) -> None:
        """Add a scaling rule to the system."""
        self.scaling_rules[rule.name] = rule
        logger.info(f"Added scaling rule: {rule.name} for {rule.trigger.value}")
        
    def remove_scaling_rule(self, rule_name: str) -> None:
        """Remove a scaling rule from the system."""
        if rule_name in self.scaling_rules:
            del self.scaling_rules[rule_name]
            logger.info(f"Removed scaling rule: {rule_name}")
            
    def update_metrics(self, metrics: ScalingMetrics) -> None:
        """Update current system metrics."""
        self.current_metrics = metrics
        self.metrics_history.append(metrics)
        
        # Keep only last 1000 metrics (roughly 1 hour at 5-second intervals)
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
            
    def get_metric_value(self, trigger: ScalingTrigger) -> float:
        """Get the current value for a specific metric."""
        metric_map = {
            ScalingTrigger.CPU_USAGE: self.current_metrics.cpu_usage_percent,
            ScalingTrigger.MEMORY_USAGE: self.current_metrics.memory_usage_percent,
            ScalingTrigger.REQUEST_RATE: self.current_metrics.request_rate_per_second,
            ScalingTrigger.QUEUE_LENGTH: self.current_metrics.queue_length,
            ScalingTrigger.RESPONSE_TIME: self.current_metrics.avg_response_time_ms,
            ScalingTrigger.ERROR_RATE: self.current_metrics.error_rate_percent,
        }
        return metric_map.get(trigger, 0.0)
        
    def should_scale_up(self, rule: ScalingRule) -> bool:
        """Check if we should scale up based on a rule."""
        if not rule.enabled:
            return False
            
        current_value = self.get_metric_value(rule.trigger)
        
        # Check cooldown period
        if rule.last_action_time:
            time_since_last_action = datetime.utcnow() - rule.last_action_time
            if time_since_last_action.total_seconds() < rule.cooldown_seconds:
                return False
                
        return current_value > rule.threshold_up
        
    def should_scale_down(self, rule: ScalingRule) -> bool:
        """Check if we should scale down based on a rule."""
        if not rule.enabled:
            return False
            
        current_value = self.get_metric_value(rule.trigger)
        
        # Check cooldown period
        if rule.last_action_time:
            time_since_last_action = datetime.utcnow() - rule.last_action_time
            if time_since_last_action.total_seconds() < rule.cooldown_seconds:
                return False
                
        return current_value < rule.threshold_down
        
    def check_scaling_rules(self) -> None:
        """Check all scaling rules and execute actions if needed."""
        for rule_name, rule in self.scaling_rules.items():
            try:
                if self.should_scale_up(rule):
                    logger.info(f"Scaling up: {rule_name} - {rule.trigger.value} = {self.get_metric_value(rule.trigger)}")
                    rule.scale_up_action()
                    rule.last_action_time = datetime.utcnow()
                    
                elif self.should_scale_down(rule):
                    logger.info(f"Scaling down: {rule_name} - {rule.trigger.value} = {self.get_metric_value(rule.trigger)}")
                    rule.scale_down_action()
                    rule.last_action_time = datetime.utcnow()
                    
            except Exception as e:
                logger.error(f"Error executing scaling rule {rule_name}: {e}")
                
    async def start_monitoring(self, check_interval_seconds: int = 30) -> None:
        """Start the auto-scaling monitoring loop."""
        if self.monitoring_enabled:
            logger.warning("Auto-scaling monitoring is already running")
            return
            
        self.monitoring_enabled = True
        self._monitoring_task = asyncio.create_task(
            self._monitoring_loop(check_interval_seconds)
        )
        logger.info(f"Started auto-scaling monitoring with {check_interval_seconds}s interval")
        
    async def stop_monitoring(self) -> None:
        """Stop the auto-scaling monitoring loop."""
        self.monitoring_enabled = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped auto-scaling monitoring")
        
    async def _monitoring_loop(self, check_interval_seconds: int) -> None:
        """Main monitoring loop that checks scaling rules periodically."""
        while self.monitoring_enabled:
            try:
                self.check_scaling_rules()
                await asyncio.sleep(check_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auto-scaling monitoring loop: {e}")
                await asyncio.sleep(check_interval_seconds)
                
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current auto-scaling status and metrics."""
        return {
            'monitoring_enabled': self.monitoring_enabled,
            'current_metrics': self.current_metrics.to_dict(),
            'active_rules': len([r for r in self.scaling_rules.values() if r.enabled]),
            'total_rules': len(self.scaling_rules),
            'rules': {
                name: {
                    'trigger': rule.trigger.value,
                    'threshold_up': rule.threshold_up,
                    'threshold_down': rule.threshold_down,
                    'enabled': rule.enabled,
                    'last_action': rule.last_action_time.isoformat() if rule.last_action_time else None
                }
                for name, rule in self.scaling_rules.items()
            }
        }


# Global auto-scaler instance
_auto_scaler = None


def get_auto_scaler() -> AutoScaler:
    """Get the global auto-scaler instance."""
    global _auto_scaler
    if _auto_scaler is None:
        _auto_scaler = AutoScaler()
    return _auto_scaler


# Example scaling actions
def scale_up_connection_pool() -> None:
    """Example scale-up action for connection pool."""
    logger.info("Scaling up connection pool size")


def scale_down_connection_pool() -> None:
    """Example scale-down action for connection pool."""
    logger.info("Scaling down connection pool size")


def setup_default_scaling_rules() -> None:
    """Setup default auto-scaling rules for the system."""
    scaler = get_auto_scaler()
    
    # CPU-based scaling
    scaler.add_scaling_rule(ScalingRule(
        name="cpu_scaling",
        trigger=ScalingTrigger.CPU_USAGE,
        threshold_up=80.0,
        threshold_down=30.0,
        scale_up_action=scale_up_connection_pool,
        scale_down_action=scale_down_connection_pool,
        cooldown_seconds=300
    ))
    
    # Request rate-based scaling
    scaler.add_scaling_rule(ScalingRule(
        name="request_rate_scaling", 
        trigger=ScalingTrigger.REQUEST_RATE,
        threshold_up=100.0,  # requests per second
        threshold_down=20.0,
        scale_up_action=scale_up_connection_pool,
        scale_down_action=scale_down_connection_pool,
        cooldown_seconds=180
    ))
    
    logger.info("Default auto-scaling rules configured")


# Initialize default rules on import
setup_default_scaling_rules()