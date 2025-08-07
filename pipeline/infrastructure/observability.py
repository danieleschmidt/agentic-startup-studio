"""
Observability Stack - Comprehensive monitoring, logging, and alerting.

Provides structured logging, metrics collection, health monitoring,
and performance tracking for the agentic startup studio pipeline.
"""

import asyncio
import json
import logging
import statistics
import time
import traceback
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any

from pipeline.events.event_bus import DomainEvent, EventType, get_event_bus

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Structured logging levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics for collection."""
    COUNTER = "counter"           # Incrementing values
    GAUGE = "gauge"              # Point-in-time values
    HISTOGRAM = "histogram"      # Distribution of values
    TIMER = "timer"              # Duration measurements


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class LogEntry:
    """Structured log entry with context."""

    timestamp: datetime
    level: LogLevel
    message: str

    # Context information
    service: str | None = None
    operation: str | None = None
    correlation_id: str | None = None
    user_id: str | None = None
    session_id: str | None = None

    # Technical context
    module: str | None = None
    function: str | None = None
    line_number: int | None = None

    # Additional structured data
    extra_data: dict[str, Any] = field(default_factory=dict)

    # Error information
    error_type: str | None = None
    error_message: str | None = None
    stack_trace: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert log entry to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'level': self.level.value,
            'message': self.message,
            'service': self.service,
            'operation': self.operation,
            'correlation_id': self.correlation_id,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'module': self.module,
            'function': self.function,
            'line_number': self.line_number,
            'extra_data': self.extra_data,
            'error_type': self.error_type,
            'error_message': self.error_message,
            'stack_trace': self.stack_trace
        }

    def to_json(self) -> str:
        """Convert log entry to JSON string."""
        return json.dumps(self.to_dict(), default=str)


@dataclass
class MetricValue:
    """Metric value with metadata."""

    name: str
    value: int | float
    metric_type: MetricType
    timestamp: datetime

    # Labels for metric dimensions
    labels: dict[str, str] = field(default_factory=dict)

    # Additional metadata
    unit: str | None = None
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert metric to dictionary."""
        return {
            'name': self.name,
            'value': self.value,
            'type': self.metric_type.value,
            'timestamp': self.timestamp.isoformat(),
            'labels': self.labels,
            'unit': self.unit,
            'description': self.description
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations."""

    operation_name: str
    start_time: float
    end_time: float
    duration_ms: float

    # Resource usage
    memory_usage_mb: float | None = None
    cpu_usage_percent: float | None = None

    # Operation context
    success: bool = True
    error_message: str | None = None
    items_processed: int | None = None

    # Labels
    labels: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert performance metrics to dictionary."""
        return {
            'operation_name': self.operation_name,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration_ms': self.duration_ms,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'success': self.success,
            'error_message': self.error_message,
            'items_processed': self.items_processed,
            'labels': self.labels
        }


@dataclass
class Alert:
    """Alert definition and state."""

    name: str
    severity: AlertSeverity
    message: str
    timestamp: datetime

    # Alert metadata
    source: str
    metric_name: str | None = None
    threshold_value: float | None = None
    actual_value: float | None = None

    # Alert lifecycle
    is_active: bool = True
    acknowledged: bool = False
    resolved_at: datetime | None = None

    # Context
    labels: dict[str, str] = field(default_factory=dict)
    additional_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'name': self.name,
            'severity': self.severity.value,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'metric_name': self.metric_name,
            'threshold_value': self.threshold_value,
            'actual_value': self.actual_value,
            'is_active': self.is_active,
            'acknowledged': self.acknowledged,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'labels': self.labels,
            'additional_info': self.additional_info
        }


class StructuredLogger:
    """
    Structured logger with context and correlation support.
    
    Provides consistent logging with structured data, context propagation,
    and integration with the observability stack.
    """

    def __init__(self, name: str, service: str | None = None):
        self.name = name
        self.service = service or "agentic-startup-studio"
        self.logger = logging.getLogger(name)
        self.default_context: dict[str, Any] = {}

        # Configure JSON formatter if not already configured
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def set_context(self, **kwargs) -> None:
        """Set default context for all log entries."""
        self.default_context.update(kwargs)

    def clear_context(self) -> None:
        """Clear default context."""
        self.default_context.clear()

    def _create_log_entry(
        self,
        level: LogLevel,
        message: str,
        **kwargs
    ) -> LogEntry:
        """Create structured log entry."""
        # Merge default context with provided context
        context = {**self.default_context, **kwargs}

        # Extract error information if present
        error_info = {}
        if 'error' in context:
            error = context.pop('error')
            if isinstance(error, Exception):
                error_info = {
                    'error_type': type(error).__name__,
                    'error_message': str(error),
                    'stack_trace': ''.join(traceback.format_exception(
                        type(error), error, error.__traceback__
                    ))
                }

        return LogEntry(
            timestamp=datetime.now(UTC),
            level=level,
            message=message,
            service=self.service,
            operation=context.pop('operation', None),
            correlation_id=context.pop('correlation_id', None),
            user_id=context.pop('user_id', None),
            session_id=context.pop('session_id', None),
            module=context.pop('module', None),
            function=context.pop('function', None),
            line_number=context.pop('line_number', None),
            extra_data=context,
            **error_info
        )

    def _log(self, level: LogLevel, message: str, **kwargs) -> None:
        """Internal logging method."""
        entry = self._create_log_entry(level, message, **kwargs)

        # Log using standard logger
        log_level = getattr(logging, level.value.upper())
        self.logger.log(log_level, entry.to_json())

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self._log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self._log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self._log(LogLevel.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self._log(LogLevel.CRITICAL, message, **kwargs)


class MetricsCollector:
    """
    Metrics collection and aggregation system.
    
    Collects, aggregates, and exposes metrics for monitoring
    and alerting purposes.
    """

    def __init__(self):
        self.metrics: dict[str, list[MetricValue]] = {}
        self.aggregated_metrics: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

        # Aggregation settings
        self.aggregation_window_seconds = 60
        self.max_metric_history = 1000

        self.logger = StructuredLogger(__name__ + ".MetricsCollector")

    async def record_metric(
        self,
        name: str,
        value: int | float,
        metric_type: MetricType,
        labels: dict[str, str] | None = None,
        unit: str | None = None,
        description: str | None = None
    ) -> None:
        """Record a metric value."""
        async with self._lock:
            metric = MetricValue(
                name=name,
                value=value,
                metric_type=metric_type,
                timestamp=datetime.now(UTC),
                labels=labels or {},
                unit=unit,
                description=description
            )

            if name not in self.metrics:
                self.metrics[name] = []

            self.metrics[name].append(metric)

            # Limit history size
            if len(self.metrics[name]) > self.max_metric_history:
                self.metrics[name] = self.metrics[name][-self.max_metric_history:]

            self.logger.debug(
                f"Recorded metric: {name} = {value}",
                metric_name=name,
                metric_value=value,
                metric_type=metric_type.value,
                labels=labels
            )

    async def increment_counter(
        self,
        name: str,
        value: int | float = 1,
        labels: dict[str, str] | None = None
    ) -> None:
        """Increment a counter metric."""
        await self.record_metric(name, value, MetricType.COUNTER, labels)

    async def set_gauge(
        self,
        name: str,
        value: int | float,
        labels: dict[str, str] | None = None
    ) -> None:
        """Set a gauge metric value."""
        await self.record_metric(name, value, MetricType.GAUGE, labels)

    async def record_histogram(
        self,
        name: str,
        value: int | float,
        labels: dict[str, str] | None = None
    ) -> None:
        """Record a histogram metric."""
        await self.record_metric(name, value, MetricType.HISTOGRAM, labels)

    async def record_timer(
        self,
        name: str,
        duration_ms: float,
        labels: dict[str, str] | None = None
    ) -> None:
        """Record a timer metric."""
        await self.record_metric(name, duration_ms, MetricType.TIMER, labels)

    async def get_metric_values(
        self,
        name: str,
        since: datetime | None = None
    ) -> list[MetricValue]:
        """Get metric values for a given metric name."""
        async with self._lock:
            if name not in self.metrics:
                return []

            values = self.metrics[name]

            if since:
                values = [v for v in values if v.timestamp >= since]

            return values

    async def get_aggregated_metrics(
        self,
        window_seconds: int | None = None
    ) -> dict[str, dict[str, Any]]:
        """Get aggregated metrics for the specified time window."""
        window_seconds = window_seconds or self.aggregation_window_seconds
        since = datetime.now(UTC) - timedelta(seconds=window_seconds)

        aggregated = {}

        async with self._lock:
            for name, values in self.metrics.items():
                recent_values = [v for v in values if v.timestamp >= since]

                if not recent_values:
                    continue

                numeric_values = [v.value for v in recent_values]

                # Aggregate based on metric type
                metric_type = recent_values[0].metric_type

                if metric_type == MetricType.COUNTER:
                    aggregated[name] = {
                        'type': 'counter',
                        'total': sum(numeric_values),
                        'rate_per_second': sum(numeric_values) / window_seconds,
                        'count': len(numeric_values)
                    }
                elif metric_type == MetricType.GAUGE:
                    aggregated[name] = {
                        'type': 'gauge',
                        'current': numeric_values[-1],
                        'min': min(numeric_values),
                        'max': max(numeric_values),
                        'avg': statistics.mean(numeric_values)
                    }
                elif metric_type in (MetricType.HISTOGRAM, MetricType.TIMER):
                    aggregated[name] = {
                        'type': metric_type.value,
                        'count': len(numeric_values),
                        'sum': sum(numeric_values),
                        'min': min(numeric_values),
                        'max': max(numeric_values),
                        'avg': statistics.mean(numeric_values),
                        'p50': statistics.median(numeric_values),
                        'p95': self._percentile(numeric_values, 0.95),
                        'p99': self._percentile(numeric_values, 0.99)
                    }

        return aggregated

    def _percentile(self, values: list[float], percentile: float) -> float:
        """Calculate percentile value."""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile)

        if index >= len(sorted_values):
            index = len(sorted_values) - 1

        return sorted_values[index]

    async def get_metrics_summary(self) -> dict[str, Any]:
        """Get comprehensive metrics summary."""
        aggregated = await self.get_aggregated_metrics()

        return {
            'total_metrics': len(self.metrics),
            'aggregation_window_seconds': self.aggregation_window_seconds,
            'timestamp': datetime.now(UTC).isoformat(),
            'metrics': aggregated
        }


class PerformanceMonitor:
    """
    Performance monitoring and profiling system.
    
    Tracks operation performance, resource usage, and provides
    performance analytics.
    """

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.performance_history: list[PerformanceMetrics] = []
        self.active_operations: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

        self.logger = StructuredLogger(__name__ + ".PerformanceMonitor")

    @asynccontextmanager
    async def track_operation(
        self,
        operation_name: str,
        labels: dict[str, str] | None = None
    ):
        """Context manager for tracking operation performance."""
        operation_id = f"{operation_name}_{time.time()}"
        start_time = time.time()

        # Record operation start
        async with self._lock:
            self.active_operations[operation_id] = {
                'name': operation_name,
                'start_time': start_time,
                'labels': labels or {}
            }

        self.logger.debug(
            f"Started tracking operation: {operation_name}",
            operation_name=operation_name,
            operation_id=operation_id
        )

        try:
            yield operation_id
        except Exception as e:
            # Record operation failure
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000

            await self._record_performance(
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                success=False,
                error_message=str(e),
                labels=labels or {}
            )

            self.logger.error(
                f"Operation failed: {operation_name}",
                operation_name=operation_name,
                operation_id=operation_id,
                duration_ms=duration_ms,
                error=e
            )

            raise
        else:
            # Record successful operation
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000

            await self._record_performance(
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                success=True,
                labels=labels or {}
            )

            self.logger.debug(
                f"Completed operation: {operation_name}",
                operation_name=operation_name,
                operation_id=operation_id,
                duration_ms=duration_ms
            )
        finally:
            # Clean up active operation
            async with self._lock:
                self.active_operations.pop(operation_id, None)

    async def _record_performance(
        self,
        operation_name: str,
        start_time: float,
        end_time: float,
        duration_ms: float,
        success: bool,
        labels: dict[str, str],
        error_message: str | None = None,
        items_processed: int | None = None
    ) -> None:
        """Record performance metrics for an operation."""
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            success=success,
            error_message=error_message,
            items_processed=items_processed,
            labels=labels
        )

        # Add to history
        async with self._lock:
            self.performance_history.append(metrics)

            # Limit history size
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]

        # Record metrics
        await self.metrics_collector.record_timer(
            f"operation_duration_{operation_name}",
            duration_ms,
            labels
        )

        await self.metrics_collector.increment_counter(
            f"operation_total_{operation_name}",
            1,
            {**labels, 'success': str(success).lower()}
        )

    async def get_performance_summary(
        self,
        operation_name: str | None = None,
        time_window_minutes: int = 60
    ) -> dict[str, Any]:
        """Get performance summary for operations."""
        since = datetime.now(UTC) - timedelta(minutes=time_window_minutes)

        async with self._lock:
            # Filter metrics by operation and time
            metrics = self.performance_history

            if operation_name:
                metrics = [m for m in metrics if m.operation_name == operation_name]

            # Filter by time (convert to datetime for comparison)
            metrics = [
                m for m in metrics
                if datetime.fromtimestamp(m.start_time, UTC) >= since
            ]

        if not metrics:
            return {
                'operation_name': operation_name,
                'time_window_minutes': time_window_minutes,
                'total_operations': 0,
                'success_rate': 0.0,
                'performance': {}
            }

        # Calculate statistics
        durations = [m.duration_ms for m in metrics]
        successful_ops = [m for m in metrics if m.success]

        return {
            'operation_name': operation_name,
            'time_window_minutes': time_window_minutes,
            'total_operations': len(metrics),
            'successful_operations': len(successful_ops),
            'failed_operations': len(metrics) - len(successful_ops),
            'success_rate': len(successful_ops) / len(metrics),
            'performance': {
                'avg_duration_ms': statistics.mean(durations),
                'min_duration_ms': min(durations),
                'max_duration_ms': max(durations),
                'p50_duration_ms': statistics.median(durations),
                'p95_duration_ms': self._percentile(durations, 0.95),
                'p99_duration_ms': self._percentile(durations, 0.99),
                'operations_per_minute': len(metrics) / time_window_minutes
            }
        }

    def _percentile(self, values: list[float], percentile: float) -> float:
        """Calculate percentile value."""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile)

        if index >= len(sorted_values):
            index = len(sorted_values) - 1

        return sorted_values[index]


def performance_monitor(operation_name: str, labels: dict[str, str] | None = None):
    """Decorator for monitoring function performance."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            monitor = get_observability_stack().performance_monitor
            async with monitor.track_operation(operation_name, labels):
                return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we can't use async context manager
            # This is a simplified sync version
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000

                # Log performance (sync version)
                logger.info(
                    f"Operation completed: {operation_name}",
                    extra={
                        'operation_name': operation_name,
                        'duration_ms': duration_ms,
                        'success': True
                    }
                )

                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000

                logger.error(
                    f"Operation failed: {operation_name}",
                    extra={
                        'operation_name': operation_name,
                        'duration_ms': duration_ms,
                        'success': False,
                        'error': str(e)
                    }
                )

                raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


class ObservabilityStack:
    """
    Comprehensive observability stack combining logging, metrics,
    and performance monitoring.
    """

    def __init__(self):
        self.logger = StructuredLogger(__name__ + ".ObservabilityStack")
        self.metrics_collector = MetricsCollector()
        self.performance_monitor = PerformanceMonitor(self.metrics_collector)
        self.alerts: list[Alert] = []
        self.event_bus = get_event_bus()

        # Initialize default metrics
        self._initialize_system_metrics()

    def _initialize_system_metrics(self) -> None:
        """Initialize system-level metrics."""
        # These would be expanded with actual system metrics collection
        pass

    def get_logger(self, name: str, service: str | None = None) -> StructuredLogger:
        """Get a structured logger instance."""
        return StructuredLogger(name, service)

    async def create_alert(
        self,
        name: str,
        severity: AlertSeverity,
        message: str,
        source: str,
        metric_name: str | None = None,
        threshold_value: float | None = None,
        actual_value: float | None = None,
        labels: dict[str, str] | None = None,
        additional_info: dict[str, Any] | None = None
    ) -> Alert:
        """Create and record an alert."""
        alert = Alert(
            name=name,
            severity=severity,
            message=message,
            timestamp=datetime.now(UTC),
            source=source,
            metric_name=metric_name,
            threshold_value=threshold_value,
            actual_value=actual_value,
            labels=labels or {},
            additional_info=additional_info or {}
        )

        self.alerts.append(alert)

        # Publish alert event
        await self.event_bus.publish(DomainEvent(
            event_type=EventType.WORKFLOW_STATE_CHANGED,  # Using generic event type
            aggregate_id=f"alert_{name}",
            event_data={
                'alert_type': 'created',
                'alert_name': name,
                'severity': severity.value,
                'message': message,
                'source': source,
                'timestamp': alert.timestamp.isoformat()
            }
        ))

        self.logger.warning(
            f"Alert created: {name}",
            alert_name=name,
            severity=severity.value,
            message=message,
            source=source
        )

        return alert

    async def get_health_status(self) -> dict[str, Any]:
        """Get comprehensive health status."""
        metrics_summary = await self.metrics_collector.get_metrics_summary()

        # Check for active critical alerts
        critical_alerts = [
            a for a in self.alerts
            if a.is_active and a.severity == AlertSeverity.CRITICAL
        ]

        # Determine overall health
        if critical_alerts:
            overall_status = "critical"
        elif any(a.is_active and a.severity == AlertSeverity.HIGH for a in self.alerts):
            overall_status = "degraded"
        else:
            overall_status = "healthy"

        return {
            'overall_status': overall_status,
            'timestamp': datetime.now(UTC).isoformat(),
            'components': {
                'metrics_collector': {
                    'status': 'healthy',
                    'total_metrics': metrics_summary['total_metrics']
                },
                'performance_monitor': {
                    'status': 'healthy',
                    'active_operations': len(self.performance_monitor.active_operations)
                },
                'alerting': {
                    'status': 'healthy',
                    'active_alerts': len([a for a in self.alerts if a.is_active]),
                    'critical_alerts': len(critical_alerts)
                }
            },
            'active_alerts': [a.to_dict() for a in self.alerts if a.is_active]
        }


# Singleton instance
_observability_stack: ObservabilityStack | None = None


def get_observability_stack() -> ObservabilityStack:
    """Get singleton observability stack instance."""
    global _observability_stack
    if _observability_stack is None:
        _observability_stack = ObservabilityStack()
    return _observability_stack


def get_logger(name: str, service: str | None = None) -> StructuredLogger:
    """Get a structured logger instance."""
    return get_observability_stack().get_logger(name, service)
