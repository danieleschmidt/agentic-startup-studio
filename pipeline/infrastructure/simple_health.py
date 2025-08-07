"""
Simple Health Monitor for Agentic Startup Studio.

Provides basic health checks without complex async dependencies
to ensure reliable monitoring data collection.
"""

import logging
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class HealthStatus:
    """Simple health status data class."""
    component: str
    status: str  # "healthy", "degraded", "unhealthy"
    message: str
    timestamp: str
    metrics: dict[str, Any] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class SimpleHealthMonitor:
    """
    Simple health monitoring system that works reliably.
    
    Avoids complex async operations that were causing intermittent failures.
    """

    def __init__(self):
        """Initialize the health monitor."""
        self.checks: dict[str, Any] = {}
        self.last_check_time = None

    def add_check(self, name: str, check_function: callable, timeout: float = 5.0):
        """
        Add a health check function.
        
        Args:
            name: Name of the health check
            check_function: Function that returns True if healthy, False otherwise
            timeout: Timeout for the check in seconds
        """
        self.checks[name] = {
            'function': check_function,
            'timeout': timeout,
            'last_result': None,
            'last_check': None
        }

    def check_component(self, name: str) -> HealthStatus:
        """
        Check a single component's health.
        
        Args:
            name: Name of the component to check
            
        Returns:
            HealthStatus object with check results
        """
        if name not in self.checks:
            return HealthStatus(
                component=name,
                status="unhealthy",
                message=f"Component '{name}' not registered",
                timestamp=datetime.now(UTC).isoformat()
            )

        check_info = self.checks[name]
        start_time = time.time()

        try:
            # Simple timeout implementation
            result = check_info['function']()
            elapsed_time = time.time() - start_time

            if elapsed_time > check_info['timeout']:
                return HealthStatus(
                    component=name,
                    status="degraded",
                    message=f"Check completed but took {elapsed_time:.2f}s (timeout: {check_info['timeout']}s)",
                    timestamp=datetime.now(UTC).isoformat(),
                    metrics={"elapsed_time": elapsed_time}
                )

            status = "healthy" if result else "unhealthy"
            message = "Component is functioning normally" if result else "Component check failed"

            # Update check info
            check_info['last_result'] = result
            check_info['last_check'] = datetime.now(UTC).isoformat()

            return HealthStatus(
                component=name,
                status=status,
                message=message,
                timestamp=datetime.now(UTC).isoformat(),
                metrics={"elapsed_time": elapsed_time}
            )

        except Exception as e:
            return HealthStatus(
                component=name,
                status="unhealthy",
                message=f"Health check failed with error: {str(e)}",
                timestamp=datetime.now(UTC).isoformat(),
                metrics={"error": str(e)}
            )

    def check_all(self) -> dict[str, HealthStatus]:
        """
        Check all registered components.
        
        Returns:
            Dictionary mapping component names to HealthStatus objects
        """
        results = {}
        self.last_check_time = datetime.now(UTC).isoformat()

        for name in self.checks:
            results[name] = self.check_component(name)

        return results

    def get_overall_status(self) -> dict[str, Any]:
        """
        Get overall system health status.
        
        Returns:
            Dictionary with overall status and component details
        """
        component_results = self.check_all()

        # Determine overall status
        statuses = [result.status for result in component_results.values()]

        if any(status == "unhealthy" for status in statuses):
            overall_status = "unhealthy"
        elif any(status == "degraded" for status in statuses):
            overall_status = "degraded"
        else:
            overall_status = "healthy"

        return {
            "overall_status": overall_status,
            "timestamp": self.last_check_time,
            "components": {name: result.to_dict() for name, result in component_results.items()},
            "summary": {
                "total_components": len(component_results),
                "healthy": sum(1 for s in statuses if s == "healthy"),
                "degraded": sum(1 for s in statuses if s == "degraded"),
                "unhealthy": sum(1 for s in statuses if s == "unhealthy")
            }
        }


# Global instance
_health_monitor = SimpleHealthMonitor()


def get_health_monitor() -> SimpleHealthMonitor:
    """Get the global health monitor instance."""
    return _health_monitor


# Basic health check functions
def check_python_imports() -> bool:
    """Check if basic Python imports work."""
    try:
        import json
        import os
        import sys
        return True
    except ImportError:
        return False


def check_pipeline_imports() -> bool:
    """Check if pipeline modules can be imported."""
    try:
        import pipeline
        return True
    except ImportError:
        return False


def check_adapter_registry() -> bool:
    """Check if adapter registry is accessible."""
    try:
        from pipeline.adapters import ADAPTER_REGISTRY
        return len(ADAPTER_REGISTRY) > 0
    except Exception:
        return False


def check_circuit_breaker_basic() -> bool:
    """Check if circuit breaker can be imported and created."""
    try:
        from pipeline.infrastructure.circuit_breaker import CircuitBreakerRegistry
        registry = CircuitBreakerRegistry()
        return True
    except Exception:
        return False


def initialize_basic_health_checks():
    """Initialize basic health checks for the system."""
    monitor = get_health_monitor()

    # Add basic checks
    monitor.add_check("python_imports", check_python_imports, timeout=2.0)
    monitor.add_check("pipeline_imports", check_pipeline_imports, timeout=3.0)
    monitor.add_check("adapter_registry", check_adapter_registry, timeout=3.0)
    monitor.add_check("circuit_breaker_basic", check_circuit_breaker_basic, timeout=3.0)

    logger.info("Basic health checks initialized")


# Initialize on import
initialize_basic_health_checks()
