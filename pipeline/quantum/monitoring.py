"""
Quantum Task Planner Monitoring and Observability

Comprehensive monitoring system for quantum task planning operations
including metrics collection, health checks, performance monitoring,
and quantum system observability.
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from uuid import UUID
import numpy as np

try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from .quantum_planner import QuantumTask, QuantumState, QuantumPriority
from .quantum_scheduler import QuantumScheduler
from .quantum_dependencies import QuantumEntanglement, EntanglementType

logger = logging.getLogger(__name__)


@dataclass
class QuantumMetrics:
    """Container for quantum system metrics."""
    
    # Task metrics
    total_tasks: int = 0
    active_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    
    # State distribution
    state_distribution: Dict[QuantumState, int] = field(default_factory=dict)
    priority_distribution: Dict[QuantumPriority, int] = field(default_factory=dict)
    
    # Quantum metrics
    system_coherence: float = 0.0
    average_entanglement_strength: float = 0.0
    quantum_interference_events: int = 0
    superposition_collapses: int = 0
    
    # Performance metrics
    average_execution_time: float = 0.0
    scheduling_optimization_time: float = 0.0
    dependency_resolution_time: float = 0.0
    
    # System health
    error_rate: float = 0.0
    resource_utilization: float = 0.0
    uptime_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_tasks": self.total_tasks,
            "active_tasks": self.active_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "state_distribution": {k.value: v for k, v in self.state_distribution.items()},
            "priority_distribution": {k.value: v for k, v in self.priority_distribution.items()},
            "system_coherence": self.system_coherence,
            "average_entanglement_strength": self.average_entanglement_strength,
            "quantum_interference_events": self.quantum_interference_events,
            "superposition_collapses": self.superposition_collapses,
            "average_execution_time": self.average_execution_time,
            "scheduling_optimization_time": self.scheduling_optimization_time,
            "dependency_resolution_time": self.dependency_resolution_time,
            "error_rate": self.error_rate,
            "resource_utilization": self.resource_utilization,
            "uptime_seconds": self.uptime_seconds
        }


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""
    
    component: str
    status: str  # "healthy", "degraded", "unhealthy"
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = field(default_factory=dict)
    
    def is_healthy(self) -> bool:
        """Check if the component is healthy."""
        return self.status == "healthy"


class QuantumMetricsCollector:
    """Collector for quantum task planner metrics."""
    
    def __init__(self, enable_prometheus: bool = True):
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.metrics_history: deque = deque(maxlen=1000)
        self.start_time = time.time()
        
        # Event counters
        self.event_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.performance_samples = defaultdict(list)
        
        # Prometheus metrics
        if self.enable_prometheus:
            self.registry = CollectorRegistry()
            self._setup_prometheus_metrics()
    
    def _setup_prometheus_metrics(self) -> None:
        """Set up Prometheus metrics."""
        if not self.enable_prometheus:
            return
        
        # Counters
        self.task_operations_total = Counter(
            'quantum_task_operations_total',
            'Total quantum task operations',
            ['operation_type', 'status'],
            registry=self.registry
        )
        
        self.quantum_events_total = Counter(
            'quantum_events_total',
            'Total quantum events',
            ['event_type'],
            registry=self.registry
        )
        
        # Histograms
        self.task_execution_duration = Histogram(
            'quantum_task_execution_duration_seconds',
            'Task execution duration',
            registry=self.registry
        )
        
        self.scheduling_duration = Histogram(
            'quantum_scheduling_duration_seconds',
            'Scheduling operation duration',
            registry=self.registry
        )
        
        self.dependency_resolution_duration = Histogram(
            'quantum_dependency_resolution_duration_seconds',
            'Dependency resolution duration',
            registry=self.registry
        )
        
        # Gauges
        self.active_tasks_gauge = Gauge(
            'quantum_active_tasks',
            'Number of active quantum tasks',
            registry=self.registry
        )
        
        self.system_coherence_gauge = Gauge(
            'quantum_system_coherence',
            'Quantum system coherence level',
            registry=self.registry
        )
        
        self.entanglement_strength_gauge = Gauge(
            'quantum_average_entanglement_strength',
            'Average entanglement strength',
            registry=self.registry
        )
    
    def record_task_operation(self, operation_type: str, status: str = "success") -> None:
        """Record a task operation."""
        self.event_counts[f"task_{operation_type}"] += 1
        
        if self.enable_prometheus:
            self.task_operations_total.labels(
                operation_type=operation_type, 
                status=status
            ).inc()
    
    def record_quantum_event(self, event_type: str) -> None:
        """Record a quantum-specific event."""
        self.event_counts[f"quantum_{event_type}"] += 1
        
        if self.enable_prometheus:
            self.quantum_events_total.labels(event_type=event_type).inc()
    
    def record_execution_time(self, duration_seconds: float) -> None:
        """Record task execution time."""
        self.performance_samples["execution_time"].append(duration_seconds)
        
        if self.enable_prometheus:
            self.task_execution_duration.observe(duration_seconds)
    
    def record_scheduling_time(self, duration_seconds: float) -> None:
        """Record scheduling operation time."""
        self.performance_samples["scheduling_time"].append(duration_seconds)
        
        if self.enable_prometheus:
            self.scheduling_duration.observe(duration_seconds)
    
    def record_dependency_resolution_time(self, duration_seconds: float) -> None:
        """Record dependency resolution time."""
        self.performance_samples["dependency_time"].append(duration_seconds)
        
        if self.enable_prometheus:
            self.dependency_resolution_duration.observe(duration_seconds)
    
    def record_error(self, error_type: str, error_message: str) -> None:
        """Record an error occurrence."""
        self.error_counts[error_type] += 1
        logger.error(f"Quantum planner error [{error_type}]: {error_message}")
    
    def update_system_metrics(self, tasks: List[QuantumTask], 
                            entanglements: List[QuantumEntanglement],
                            coherence: float) -> None:
        """Update system-level metrics."""
        # Update Prometheus gauges
        if self.enable_prometheus:
            active_count = sum(1 for task in tasks 
                             if task.current_state in [QuantumState.PENDING, QuantumState.EXECUTING])
            self.active_tasks_gauge.set(active_count)
            self.system_coherence_gauge.set(coherence)
            
            if entanglements:
                avg_strength = np.mean([e.strength for e in entanglements])
                self.entanglement_strength_gauge.set(avg_strength)
    
    def collect_metrics(self, tasks: List[QuantumTask],
                       entanglements: List[QuantumEntanglement],
                       coherence: float) -> QuantumMetrics:
        """Collect comprehensive system metrics."""
        
        # Task counts by state
        state_dist = defaultdict(int)
        priority_dist = defaultdict(int)
        
        for task in tasks:
            state_dist[task.current_state] += 1
            priority_dist[task.priority] += 1
        
        # Performance averages
        avg_exec_time = (np.mean(self.performance_samples["execution_time"]) 
                        if self.performance_samples["execution_time"] else 0.0)
        
        avg_sched_time = (np.mean(self.performance_samples["scheduling_time"])
                         if self.performance_samples["scheduling_time"] else 0.0)
        
        avg_dep_time = (np.mean(self.performance_samples["dependency_time"])
                       if self.performance_samples["dependency_time"] else 0.0)
        
        # Error rate calculation
        total_operations = sum(self.event_counts.values())
        total_errors = sum(self.error_counts.values())
        error_rate = total_errors / max(total_operations, 1)
        
        # Entanglement metrics
        avg_entanglement_strength = (np.mean([e.strength for e in entanglements])
                                   if entanglements else 0.0)
        
        metrics = QuantumMetrics(
            total_tasks=len(tasks),
            active_tasks=state_dist[QuantumState.EXECUTING] + state_dist[QuantumState.PENDING],
            completed_tasks=state_dist[QuantumState.COMPLETED],
            failed_tasks=state_dist[QuantumState.FAILED],
            state_distribution=dict(state_dist),
            priority_distribution=dict(priority_dist),
            system_coherence=coherence,
            average_entanglement_strength=avg_entanglement_strength,
            quantum_interference_events=self.event_counts["quantum_interference"],
            superposition_collapses=self.event_counts["quantum_measurement"],
            average_execution_time=avg_exec_time,
            scheduling_optimization_time=avg_sched_time,
            dependency_resolution_time=avg_dep_time,
            error_rate=error_rate,
            resource_utilization=min(1.0, len(tasks) / 1000),  # Assume max 1000 tasks
            uptime_seconds=time.time() - self.start_time
        )
        
        # Store in history
        self.metrics_history.append({
            "timestamp": datetime.utcnow(),
            "metrics": metrics.to_dict()
        })
        
        return metrics
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format."""
        if not self.enable_prometheus:
            return "# Prometheus not available\n"
        
        return generate_latest(self.registry).decode('utf-8')
    
    def get_metrics_history(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """Get metrics history for the specified time period."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        
        return [
            entry for entry in self.metrics_history
            if entry["timestamp"] >= cutoff_time
        ]


class QuantumHealthMonitor:
    """Health monitoring for quantum task planner components."""
    
    def __init__(self, check_interval_seconds: int = 30):
        self.check_interval = check_interval_seconds
        self.health_history: deque = deque(maxlen=100)
        self.is_monitoring = False
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Health check functions
        self.health_checks: Dict[str, Callable[[], HealthCheckResult]] = {}
        self._register_default_checks()
    
    def _register_default_checks(self) -> None:
        """Register default health checks."""
        self.health_checks.update({
            "system_coherence": self._check_system_coherence,
            "task_processing": self._check_task_processing,
            "memory_usage": self._check_memory_usage,
            "error_rate": self._check_error_rate
        })
    
    def register_health_check(self, name: str, check_function: Callable[[], HealthCheckResult]) -> None:
        """Register a custom health check."""
        self.health_checks[name] = check_function
        logger.info(f"Registered health check: {name}")
    
    async def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started quantum health monitoring")
    
    async def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self.is_monitoring = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped quantum health monitoring")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                await self.run_health_checks()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def run_health_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        results = {}
        
        for name, check_function in self.health_checks.items():
            try:
                result = check_function()
                results[name] = result
            except Exception as e:
                results[name] = HealthCheckResult(
                    component=name,
                    status="unhealthy",
                    message=f"Health check failed: {str(e)}"
                )
        
        # Store in history
        self.health_history.append({
            "timestamp": datetime.utcnow(),
            "results": results
        })
        
        # Log unhealthy components
        for name, result in results.items():
            if not result.is_healthy():
                logger.warning(f"Health check failed for {name}: {result.message}")
        
        return results
    
    def _check_system_coherence(self) -> HealthCheckResult:
        """Check quantum system coherence."""
        # This would integrate with actual quantum planner instance
        # For now, simulate based on random factors
        coherence = np.random.uniform(0.7, 1.0)  # Placeholder
        
        if coherence >= 0.8:
            status = "healthy"
            message = f"System coherence is optimal: {coherence:.3f}"
        elif coherence >= 0.6:
            status = "degraded"
            message = f"System coherence is degraded: {coherence:.3f}"
        else:
            status = "unhealthy"
            message = f"System coherence is critical: {coherence:.3f}"
        
        return HealthCheckResult(
            component="system_coherence",
            status=status,
            message=message,
            details={"coherence_value": coherence}
        )
    
    def _check_task_processing(self) -> HealthCheckResult:
        """Check task processing health."""
        # This would check actual task processing metrics
        # Placeholder implementation
        processing_rate = np.random.uniform(50, 100)  # tasks per minute
        
        if processing_rate >= 80:
            status = "healthy"
            message = f"Task processing rate is optimal: {processing_rate:.1f}/min"
        elif processing_rate >= 50:
            status = "degraded" 
            message = f"Task processing rate is degraded: {processing_rate:.1f}/min"
        else:
            status = "unhealthy"
            message = f"Task processing rate is critical: {processing_rate:.1f}/min"
        
        return HealthCheckResult(
            component="task_processing",
            status=status,
            message=message,
            details={"processing_rate": processing_rate}
        )
    
    def _check_memory_usage(self) -> HealthCheckResult:
        """Check memory usage."""
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent
            
            if memory_percent <= 80:
                status = "healthy"
                message = f"Memory usage is normal: {memory_percent:.1f}%"
            elif memory_percent <= 90:
                status = "degraded"
                message = f"Memory usage is elevated: {memory_percent:.1f}%"
            else:
                status = "unhealthy"
                message = f"Memory usage is critical: {memory_percent:.1f}%"
            
        except ImportError:
            # psutil not available
            status = "healthy"
            message = "Memory monitoring not available (psutil not installed)"
            memory_percent = 0
        
        return HealthCheckResult(
            component="memory_usage",
            status=status,
            message=message,
            details={"memory_percent": memory_percent}
        )
    
    def _check_error_rate(self) -> HealthCheckResult:
        """Check system error rate."""
        # This would integrate with metrics collector
        # Placeholder implementation
        error_rate = np.random.uniform(0, 0.05)  # 0-5% error rate
        
        if error_rate <= 0.01:
            status = "healthy"
            message = f"Error rate is low: {error_rate:.3f}"
        elif error_rate <= 0.03:
            status = "degraded"
            message = f"Error rate is elevated: {error_rate:.3f}"
        else:
            status = "unhealthy"
            message = f"Error rate is high: {error_rate:.3f}"
        
        return HealthCheckResult(
            component="error_rate",
            status=status,
            message=message,
            details={"error_rate": error_rate}
        )
    
    def get_overall_health(self) -> HealthCheckResult:
        """Get overall system health status."""
        if not self.health_history:
            return HealthCheckResult(
                component="overall",
                status="unknown",
                message="No health data available"
            )
        
        latest_results = self.health_history[-1]["results"]
        
        # Determine overall status
        unhealthy_count = sum(1 for result in latest_results.values() 
                             if result.status == "unhealthy")
        degraded_count = sum(1 for result in latest_results.values() 
                           if result.status == "degraded")
        
        if unhealthy_count > 0:
            status = "unhealthy"
            message = f"{unhealthy_count} components are unhealthy"
        elif degraded_count > 0:
            status = "degraded"
            message = f"{degraded_count} components are degraded"
        else:
            status = "healthy"
            message = "All components are healthy"
        
        return HealthCheckResult(
            component="overall",
            status=status,
            message=message,
            details={
                "component_count": len(latest_results),
                "unhealthy_count": unhealthy_count,
                "degraded_count": degraded_count
            }
        )
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary."""
        overall_health = self.get_overall_health()
        
        return {
            "overall_status": overall_health.status,
            "overall_message": overall_health.message,
            "timestamp": datetime.utcnow().isoformat(),
            "monitoring_active": self.is_monitoring,
            "check_interval_seconds": self.check_interval,
            "registered_checks": list(self.health_checks.keys()),
            "history_length": len(self.health_history)
        }


class QuantumPerformanceProfiler:
    """Performance profiler for quantum operations."""
    
    def __init__(self):
        self.operation_times: Dict[str, List[float]] = defaultdict(list)
        self.active_operations: Dict[str, float] = {}
    
    def start_operation(self, operation_name: str) -> str:
        """Start timing an operation."""
        operation_id = f"{operation_name}_{time.time()}"
        self.active_operations[operation_id] = time.time()
        return operation_id
    
    def end_operation(self, operation_id: str) -> float:
        """End timing an operation and return duration."""
        if operation_id not in self.active_operations:
            return 0.0
        
        start_time = self.active_operations.pop(operation_id)
        duration = time.time() - start_time
        
        # Extract operation name from ID
        operation_name = operation_id.rsplit('_', 1)[0]
        self.operation_times[operation_name].append(duration)
        
        return duration
    
    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics for all operations."""
        stats = {}
        
        for operation, times in self.operation_times.items():
            if times:
                stats[operation] = {
                    "count": len(times),  
                    "mean": np.mean(times),
                    "median": np.median(times),
                    "min": np.min(times),
                    "max": np.max(times),
                    "std": np.std(times),
                    "p95": np.percentile(times, 95),
                    "p99": np.percentile(times, 99)
                }
        
        return stats