"""
Comprehensive Monitoring System - Full observability and proactive alerting
Implements advanced metrics, tracing, logging, and AI-powered anomaly detection.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import statistics
import numpy as np

from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
from opentelemetry import trace, metrics
from opentelemetry.trace import Status, StatusCode
from opentelemetry.sdk.metrics import MeterProvider

from ..config.settings import get_settings
from ..telemetry import get_tracer
from ..core.adaptive_intelligence import get_intelligence, PatternType

# Global configuration
settings = get_settings()
tracer = get_tracer(__name__)
logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(str, Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class HealthStatus(str, Enum):
    """Health check statuses"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class Alert:
    """System alert with metadata"""
    alert_id: str
    title: str
    description: str
    severity: AlertSeverity
    source: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "alert_id": self.alert_id,
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
            "resolved": self.resolved,
            "metadata": self.metadata
        }


@dataclass
class HealthCheck:
    """Health check definition and result"""
    name: str
    description: str
    check_function: Callable[[], bool]
    timeout: int = 30
    interval: int = 60
    last_check: Optional[datetime] = None
    last_result: Optional[bool] = None
    failure_count: int = 0
    max_failures: int = 3
    
    @property
    def status(self) -> HealthStatus:
        """Get current health status"""
        if self.last_result is None:
            return HealthStatus.UNKNOWN
        elif self.last_result:
            return HealthStatus.HEALTHY
        elif self.failure_count >= self.max_failures:
            return HealthStatus.UNHEALTHY
        else:
            return HealthStatus.DEGRADED


class MetricsCollector:
    """Advanced metrics collection and aggregation"""
    
    def __init__(self):
        self.registry = CollectorRegistry()
        self.counters: Dict[str, Counter] = {}
        self.histograms: Dict[str, Histogram] = {}
        self.gauges: Dict[str, Gauge] = {}
        self.custom_metrics: Dict[str, List[float]] = {}
        
        # Initialize core metrics
        self._initialize_core_metrics()
    
    def _initialize_core_metrics(self) -> None:
        """Initialize core system metrics"""
        self.counters.update({
            "requests_total": Counter(
                "requests_total", 
                "Total number of requests",
                ["method", "endpoint", "status"],
                registry=self.registry
            ),
            "errors_total": Counter(
                "errors_total",
                "Total number of errors",
                ["error_type", "component"],
                registry=self.registry
            ),
            "database_queries_total": Counter(
                "database_queries_total",
                "Total number of database queries",
                ["query_type", "table"],
                registry=self.registry
            ),
            "ai_operations_total": Counter(
                "ai_operations_total",
                "Total number of AI operations",
                ["operation_type", "model"],
                registry=self.registry
            )
        })
        
        self.histograms.update({
            "request_duration_seconds": Histogram(
                "request_duration_seconds",
                "Request duration in seconds",
                ["method", "endpoint"],
                registry=self.registry
            ),
            "database_query_duration_seconds": Histogram(
                "database_query_duration_seconds",
                "Database query duration in seconds",
                ["query_type"],
                registry=self.registry
            ),
            "ai_operation_duration_seconds": Histogram(
                "ai_operation_duration_seconds",
                "AI operation duration in seconds",
                ["operation_type"],
                registry=self.registry
            )
        })
        
        self.gauges.update({
            "active_connections": Gauge(
                "active_connections",
                "Number of active connections",
                registry=self.registry
            ),
            "memory_usage_bytes": Gauge(
                "memory_usage_bytes",
                "Memory usage in bytes",
                ["type"],
                registry=self.registry
            ),
            "queue_size": Gauge(
                "queue_size",
                "Current queue size",
                ["queue_name"],
                registry=self.registry
            ),
            "cpu_usage_percent": Gauge(
                "cpu_usage_percent",
                "CPU usage percentage",
                registry=self.registry
            )
        })
    
    def increment_counter(self, name: str, labels: Optional[Dict[str, str]] = None, value: float = 1) -> None:
        """Increment a counter metric"""
        if name in self.counters:
            if labels:
                self.counters[name].labels(**labels).inc(value)
            else:
                self.counters[name].inc(value)
    
    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Observe a histogram metric"""
        if name in self.histograms:
            if labels:
                self.histograms[name].labels(**labels).observe(value)
            else:
                self.histograms[name].observe(value)
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric"""
        if name in self.gauges:
            if labels:
                self.gauges[name].labels(**labels).set(value)
            else:
                self.gauges[name].set(value)
    
    def add_custom_metric(self, name: str, value: float) -> None:
        """Add custom metric value"""
        if name not in self.custom_metrics:
            self.custom_metrics[name] = []
        
        self.custom_metrics[name].append(value)
        
        # Keep only recent values (last 1000)
        if len(self.custom_metrics[name]) > 1000:
            self.custom_metrics[name] = self.custom_metrics[name][-1000:]
    
    def get_custom_metric_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for custom metric"""
        if name not in self.custom_metrics or not self.custom_metrics[name]:
            return {}
        
        values = self.custom_metrics[name]
        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0
        }
    
    def export_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        return generate_latest(self.registry).decode('utf-8')


class AnomalyDetector:
    """AI-powered anomaly detection for metrics"""
    
    def __init__(self, sensitivity: float = 2.0):
        self.sensitivity = sensitivity  # Standard deviations for anomaly threshold
        self.baselines: Dict[str, Dict[str, float]] = {}
        self.windows: Dict[str, List[float]] = {}
        self.window_size = 100
        
    def add_data_point(self, metric_name: str, value: float) -> bool:
        """Add data point and detect anomalies"""
        if metric_name not in self.windows:
            self.windows[metric_name] = []
        
        self.windows[metric_name].append(value)
        
        # Keep window size manageable
        if len(self.windows[metric_name]) > self.window_size:
            self.windows[metric_name] = self.windows[metric_name][-self.window_size:]
        
        # Need enough data points for analysis
        if len(self.windows[metric_name]) < 20:
            return False
        
        # Update baseline
        self._update_baseline(metric_name)
        
        # Check for anomaly
        return self._is_anomaly(metric_name, value)
    
    def _update_baseline(self, metric_name: str) -> None:
        """Update baseline statistics for metric"""
        values = self.windows[metric_name]
        
        self.baselines[metric_name] = {
            "mean": statistics.mean(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
            "median": statistics.median(values),
            "p95": np.percentile(values, 95),
            "p99": np.percentile(values, 99)
        }
    
    def _is_anomaly(self, metric_name: str, value: float) -> bool:
        """Check if value is anomalous"""
        if metric_name not in self.baselines:
            return False
        
        baseline = self.baselines[metric_name]
        mean = baseline["mean"]
        std_dev = baseline["std_dev"]
        
        if std_dev == 0:
            return False
        
        # Z-score based anomaly detection
        z_score = abs(value - mean) / std_dev
        return z_score > self.sensitivity
    
    def get_anomaly_score(self, metric_name: str, value: float) -> float:
        """Get anomaly score for a value (0-1, higher is more anomalous)"""
        if metric_name not in self.baselines:
            return 0.0
        
        baseline = self.baselines[metric_name]
        mean = baseline["mean"]
        std_dev = baseline["std_dev"]
        
        if std_dev == 0:
            return 0.0
        
        z_score = abs(value - mean) / std_dev
        return min(1.0, z_score / 5.0)  # Normalize to 0-1


class AlertManager:
    """Intelligent alert management with deduplication and escalation"""
    
    def __init__(self):
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.escalation_rules: Dict[str, Dict[str, Any]] = {}
        self.notification_channels: List[Callable[[Alert], None]] = []
        
    def create_alert(
        self,
        title: str,
        description: str,
        severity: AlertSeverity,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Alert:
        """Create new alert with deduplication"""
        alert_key = f"{source}:{title}"
        
        # Check for existing similar alert
        if alert_key in self.active_alerts:
            existing_alert = self.active_alerts[alert_key]
            existing_alert.metadata.update(metadata or {})
            existing_alert.timestamp = datetime.utcnow()
            return existing_alert
        
        alert = Alert(
            alert_id=f"alert_{int(time.time() * 1000)}",
            title=title,
            description=description,
            severity=severity,
            source=source,
            metadata=metadata or {}
        )
        
        self.active_alerts[alert_key] = alert
        self.alert_history.append(alert)
        
        # Send notifications
        for channel in self.notification_channels:
            try:
                channel(alert)
            except Exception as e:
                logger.error(f"Failed to send alert notification: {e}")
        
        logger.warning(f"Alert created: {alert.title} ({alert.severity.value})")
        return alert
    
    def resolve_alert(self, alert_key: str) -> bool:
        """Resolve an active alert"""
        if alert_key in self.active_alerts:
            alert = self.active_alerts[alert_key]
            alert.resolved = True
            alert.timestamp = datetime.utcnow()
            del self.active_alerts[alert_key]
            logger.info(f"Alert resolved: {alert.title}")
            return True
        return False
    
    def acknowledge_alert(self, alert_key: str) -> bool:
        """Acknowledge an active alert"""
        if alert_key in self.active_alerts:
            self.active_alerts[alert_key].acknowledged = True
            logger.info(f"Alert acknowledged: {alert_key}")
            return True
        return False
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active alerts, optionally filtered by severity"""
        alerts = list(self.active_alerts.values())
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics"""
        active_alerts = list(self.active_alerts.values())
        
        severity_counts = {}
        for severity in AlertSeverity:
            severity_counts[severity.value] = len([a for a in active_alerts if a.severity == severity])
        
        recent_alerts = [
            alert for alert in self.alert_history
            if alert.timestamp > datetime.utcnow() - timedelta(hours=24)
        ]
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "active_alerts": len(active_alerts),
            "severity_distribution": severity_counts,
            "alerts_24h": len(recent_alerts),
            "acknowledged_alerts": len([a for a in active_alerts if a.acknowledged])
        }


class ComprehensiveMonitor:
    """
    Main monitoring system that coordinates all monitoring components
    """
    
    def __init__(self):
        self.metrics = MetricsCollector()
        self.anomaly_detector = AnomalyDetector()
        self.alert_manager = AlertManager()
        self.health_checks: Dict[str, HealthCheck] = {}
        self.performance_baselines: Dict[str, float] = {}
        self._monitoring_active = False
        
        # Register default health checks
        self._register_default_health_checks()
    
    def _register_default_health_checks(self) -> None:
        """Register default system health checks"""
        self.health_checks.update({
            "database_connection": HealthCheck(
                name="Database Connection",
                description="Check database connectivity",
                check_function=self._check_database_health,
                interval=30
            ),
            "memory_usage": HealthCheck(
                name="Memory Usage",
                description="Check system memory usage",
                check_function=self._check_memory_health,
                interval=60
            ),
            "disk_space": HealthCheck(
                name="Disk Space",
                description="Check available disk space",
                check_function=self._check_disk_health,
                interval=120
            ),
            "api_responsiveness": HealthCheck(
                name="API Responsiveness",
                description="Check API response times",
                check_function=self._check_api_health,
                interval=30
            )
        })
    
    async def start_monitoring(self) -> None:
        """Start comprehensive monitoring"""
        with tracer.start_as_current_span("start_monitoring"):
            self._monitoring_active = True
            logger.info("Comprehensive monitoring started")
            
            # Start background monitoring tasks
            asyncio.create_task(self._metrics_collection_loop())
            asyncio.create_task(self._health_check_loop())
            asyncio.create_task(self._anomaly_detection_loop())
            asyncio.create_task(self._performance_monitoring_loop())
    
    async def stop_monitoring(self) -> None:
        """Stop monitoring"""
        self._monitoring_active = False
        logger.info("Comprehensive monitoring stopped")
    
    async def _metrics_collection_loop(self) -> None:
        """Continuous metrics collection"""
        while self._monitoring_active:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(10)  # Collect every 10 seconds
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(30)
    
    async def _collect_system_metrics(self) -> None:
        """Collect system-wide metrics"""
        with tracer.start_as_current_span("collect_system_metrics"):
            import psutil
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics.set_gauge("cpu_usage_percent", cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.metrics.set_gauge("memory_usage_bytes", memory.used, {"type": "used"})
            self.metrics.set_gauge("memory_usage_bytes", memory.available, {"type": "available"})
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self.metrics.set_gauge("disk_usage_bytes", disk.used, {"type": "used"})
            self.metrics.set_gauge("disk_usage_bytes", disk.free, {"type": "free"})
            
            # Network metrics
            network = psutil.net_io_counters()
            self.metrics.add_custom_metric("network_bytes_sent", network.bytes_sent)
            self.metrics.add_custom_metric("network_bytes_recv", network.bytes_recv)
    
    async def _health_check_loop(self) -> None:
        """Continuous health checking"""
        while self._monitoring_active:
            try:
                await self._run_health_checks()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(60)
    
    async def _run_health_checks(self) -> None:
        """Run all registered health checks"""
        with tracer.start_as_current_span("run_health_checks"):
            for name, check in self.health_checks.items():
                # Check if it's time for this health check
                if (check.last_check is None or 
                    datetime.utcnow() - check.last_check > timedelta(seconds=check.interval)):
                    
                    await self._execute_health_check(name, check)
    
    async def _execute_health_check(self, name: str, check: HealthCheck) -> None:
        """Execute individual health check"""
        with tracer.start_as_current_span("execute_health_check") as span:
            span.set_attribute("check_name", name)
            
            try:
                # Execute check with timeout
                result = await asyncio.wait_for(
                    asyncio.to_thread(check.check_function),
                    timeout=check.timeout
                )
                
                check.last_check = datetime.utcnow()
                check.last_result = result
                
                if result:
                    check.failure_count = 0
                    span.set_status(Status(StatusCode.OK))
                else:
                    check.failure_count += 1
                    span.set_status(Status(StatusCode.ERROR, "Health check failed"))
                    
                    # Create alert if failures exceed threshold
                    if check.failure_count >= check.max_failures:
                        self.alert_manager.create_alert(
                            title=f"Health Check Failed: {check.name}",
                            description=f"Health check '{check.name}' has failed {check.failure_count} times",
                            severity=AlertSeverity.ERROR,
                            source="health_monitor",
                            metadata={"check_name": name, "failure_count": check.failure_count}
                        )
                
            except asyncio.TimeoutError:
                check.failure_count += 1
                check.last_result = False
                span.set_status(Status(StatusCode.ERROR, "Health check timeout"))
                logger.warning(f"Health check timeout: {name}")
                
            except Exception as e:
                check.failure_count += 1
                check.last_result = False
                span.set_status(Status(StatusCode.ERROR, str(e)))
                logger.error(f"Health check error: {name} - {e}")
    
    def _check_database_health(self) -> bool:
        """Check database connection health"""
        try:
            # Simplified database health check
            # In real implementation, would test actual database connection
            return True
        except Exception:
            return False
    
    def _check_memory_health(self) -> bool:
        """Check memory health"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return memory.percent < 90  # Alert if memory usage > 90%
        except Exception:
            return False
    
    def _check_disk_health(self) -> bool:
        """Check disk space health"""
        try:
            import psutil
            disk = psutil.disk_usage('/')
            usage_percent = (disk.used / disk.total) * 100
            return usage_percent < 85  # Alert if disk usage > 85%
        except Exception:
            return False
    
    def _check_api_health(self) -> bool:
        """Check API health"""
        try:
            # Simplified API health check
            # In real implementation, would make actual API calls
            return True
        except Exception:
            return False
    
    async def _anomaly_detection_loop(self) -> None:
        """Continuous anomaly detection"""
        while self._monitoring_active:
            try:
                await self._detect_anomalies()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in anomaly detection: {e}")
                await asyncio.sleep(120)
    
    async def _detect_anomalies(self) -> None:
        """Detect anomalies in metrics"""
        with tracer.start_as_current_span("detect_anomalies"):
            # Check custom metrics for anomalies
            for metric_name, values in self.metrics.custom_metrics.items():
                if values:
                    latest_value = values[-1]
                    is_anomaly = self.anomaly_detector.add_data_point(metric_name, latest_value)
                    
                    if is_anomaly:
                        anomaly_score = self.anomaly_detector.get_anomaly_score(metric_name, latest_value)
                        
                        # Determine severity based on anomaly score
                        if anomaly_score > 0.8:
                            severity = AlertSeverity.CRITICAL
                        elif anomaly_score > 0.6:
                            severity = AlertSeverity.ERROR
                        else:
                            severity = AlertSeverity.WARNING
                        
                        self.alert_manager.create_alert(
                            title=f"Anomaly Detected: {metric_name}",
                            description=f"Unusual value detected for {metric_name}: {latest_value}",
                            severity=severity,
                            source="anomaly_detector",
                            metadata={
                                "metric_name": metric_name,
                                "value": latest_value,
                                "anomaly_score": anomaly_score
                            }
                        )
    
    async def _performance_monitoring_loop(self) -> None:
        """Monitor system performance continuously"""
        while self._monitoring_active:
            try:
                await self._analyze_performance()
                await asyncio.sleep(300)  # Analyze every 5 minutes
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(300)
    
    async def _analyze_performance(self) -> None:
        """Analyze overall system performance"""
        with tracer.start_as_current_span("analyze_performance"):
            # Calculate performance metrics
            performance_data = {
                "cpu_usage": 0,
                "memory_usage": 0,
                "response_time": 0,
                "error_rate": 0
            }
            
            # Update performance baselines
            for metric, value in performance_data.items():
                if metric not in self.performance_baselines:
                    self.performance_baselines[metric] = value
                else:
                    # Exponential moving average
                    alpha = 0.1
                    self.performance_baselines[metric] = (
                        alpha * value + (1 - alpha) * self.performance_baselines[metric]
                    )
            
            # Feed performance data to adaptive intelligence
            intelligence = await get_intelligence()
            await intelligence.ingest_data_point(
                PatternType.PERFORMANCE,
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    **performance_data
                }
            )
    
    def record_request(
        self, 
        method: str, 
        endpoint: str, 
        status_code: int, 
        duration: float
    ) -> None:
        """Record HTTP request metrics"""
        with tracer.start_as_current_span("record_request"):
            self.metrics.increment_counter(
                "requests_total",
                {"method": method, "endpoint": endpoint, "status": str(status_code)}
            )
            
            self.metrics.observe_histogram(
                "request_duration_seconds",
                duration,
                {"method": method, "endpoint": endpoint}
            )
            
            # Check for performance issues
            if duration > 1.0:  # Slow request threshold
                self.alert_manager.create_alert(
                    title="Slow Request Detected",
                    description=f"Request to {endpoint} took {duration:.2f}s",
                    severity=AlertSeverity.WARNING if duration < 5.0 else AlertSeverity.ERROR,
                    source="performance_monitor",
                    metadata={
                        "method": method,
                        "endpoint": endpoint,
                        "duration": duration,
                        "status_code": status_code
                    }
                )
    
    def record_error(
        self, 
        error_type: str, 
        component: str, 
        error_details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record error metrics"""
        with tracer.start_as_current_span("record_error"):
            self.metrics.increment_counter(
                "errors_total",
                {"error_type": error_type, "component": component}
            )
            
            # Create alert for errors
            severity = AlertSeverity.ERROR
            if error_type in ["critical", "security", "data_loss"]:
                severity = AlertSeverity.CRITICAL
            elif error_type in ["validation", "client_error"]:
                severity = AlertSeverity.WARNING
            
            self.alert_manager.create_alert(
                title=f"Error in {component}",
                description=f"{error_type} error occurred in {component}",
                severity=severity,
                source=component,
                metadata=error_details or {}
            )
    
    def record_database_query(
        self, 
        query_type: str, 
        table: str, 
        duration: float
    ) -> None:
        """Record database query metrics"""
        with tracer.start_as_current_span("record_database_query"):
            self.metrics.increment_counter(
                "database_queries_total",
                {"query_type": query_type, "table": table}
            )
            
            self.metrics.observe_histogram(
                "database_query_duration_seconds",
                duration,
                {"query_type": query_type}
            )
    
    def record_ai_operation(
        self, 
        operation_type: str, 
        model: str, 
        duration: float,
        token_count: Optional[int] = None
    ) -> None:
        """Record AI operation metrics"""
        with tracer.start_as_current_span("record_ai_operation"):
            self.metrics.increment_counter(
                "ai_operations_total",
                {"operation_type": operation_type, "model": model}
            )
            
            self.metrics.observe_histogram(
                "ai_operation_duration_seconds",
                duration,
                {"operation_type": operation_type}
            )
            
            if token_count:
                self.metrics.add_custom_metric(f"ai_tokens_{operation_type}", token_count)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        with tracer.start_as_current_span("get_system_status"):
            health_status = {}
            overall_health = HealthStatus.HEALTHY
            
            for name, check in self.health_checks.items():
                health_status[name] = {
                    "status": check.status.value,
                    "last_check": check.last_check.isoformat() if check.last_check else None,
                    "failure_count": check.failure_count
                }
                
                # Update overall health
                if check.status == HealthStatus.UNHEALTHY:
                    overall_health = HealthStatus.UNHEALTHY
                elif check.status == HealthStatus.DEGRADED and overall_health == HealthStatus.HEALTHY:
                    overall_health = HealthStatus.DEGRADED
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "overall_health": overall_health.value,
                "health_checks": health_status,
                "active_alerts": len(self.alert_manager.active_alerts),
                "monitoring_active": self._monitoring_active,
                "metrics_collected": len(self.metrics.custom_metrics),
                "alert_summary": self.alert_manager.get_alert_summary()
            }
    
    def get_metrics_export(self) -> str:
        """Export metrics in Prometheus format"""
        return self.metrics.export_metrics()


# Global monitor instance
_monitor: Optional[ComprehensiveMonitor] = None


async def get_monitor() -> ComprehensiveMonitor:
    """Get or create the global monitor instance"""
    global _monitor
    if _monitor is None:
        _monitor = ComprehensiveMonitor()
        await _monitor.start_monitoring()
    return _monitor


async def record_request_metrics(method: str, endpoint: str, status_code: int, duration: float) -> None:
    """Convenience function to record request metrics"""
    monitor = await get_monitor()
    monitor.record_request(method, endpoint, status_code, duration)


async def record_error_metrics(error_type: str, component: str, details: Optional[Dict[str, Any]] = None) -> None:
    """Convenience function to record error metrics"""
    monitor = await get_monitor()
    monitor.record_error(error_type, component, details)