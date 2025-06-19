"""
Circuit Breaker Pattern Implementation for Agentic Startup Studio.

Provides fault tolerance and resilience for external service calls
with automatic failure detection and recovery mechanisms.
"""

import asyncio
import time
from enum import Enum
from typing import Dict, Any, Optional, Callable, TypeVar, Generic
from dataclasses import dataclass
from contextlib import asynccontextmanager
import logging

# Type for generic callable return values
T = TypeVar('T')

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states following the three-state pattern."""
    CLOSED = "closed"      # Normal operation, calls pass through
    OPEN = "open"          # Failing fast, calls rejected immediately
    HALF_OPEN = "half_open"  # Testing if service has recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5          # Number of failures before opening
    recovery_timeout: float = 60.0      # Seconds before trying half-open
    success_threshold: int = 3          # Successes needed to close from half-open
    timeout: float = 30.0               # Request timeout in seconds
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if self.recovery_timeout < 0:
            raise ValueError("recovery_timeout must be >= 0")
        if self.success_threshold < 1:
            raise ValueError("success_threshold must be >= 1")
        if self.timeout <= 0:
            raise ValueError("timeout must be > 0")


@dataclass
class CircuitBreakerMetrics:
    """Metrics and state tracking for circuit breaker."""
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    total_requests: int = 0
    total_failures: int = 0
    total_successes: int = 0
    
    def reset(self):
        """Reset failure and success counters."""
        self.failure_count = 0
        self.success_count = 0
    
    def record_success(self):
        """Record a successful operation."""
        self.success_count += 1
        self.total_successes += 1
        self.total_requests += 1
    
    def record_failure(self):
        """Record a failed operation."""
        self.failure_count += 1
        self.total_failures += 1
        self.total_requests += 1
        self.last_failure_time = time.time()


class CircuitBreakerError(Exception):
    """Base exception for circuit breaker related errors."""
    pass


class CircuitOpenError(CircuitBreakerError):
    """Raised when circuit breaker is open and rejecting calls."""
    def __init__(self, message: str = "Circuit breaker is open"):
        super().__init__(message)


class CircuitTimeoutError(CircuitBreakerError):
    """Raised when operation times out."""
    def __init__(self, message: str = "Operation timed out"):
        super().__init__(message)


class CircuitBreaker(Generic[T]):
    """
    Circuit breaker implementation for fault tolerance.
    
    Implements the three-state pattern (Closed, Open, Half-Open)
    with configurable thresholds and timeouts.
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker.
        
        Args:
            name: Unique identifier for this circuit breaker
            config: Configuration for behavior thresholds
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        self._lock = asyncio.Lock()
        
        logger.info(f"Circuit breaker '{name}' initialized in {self.state.value} state")
    
    @property
    def is_closed(self) -> bool:
        """Check if circuit breaker is closed (normal operation)."""
        return self.state == CircuitBreakerState.CLOSED
    
    @property
    def is_open(self) -> bool:
        """Check if circuit breaker is open (failing fast)."""
        return self.state == CircuitBreakerState.OPEN
    
    @property
    def is_half_open(self) -> bool:
        """Check if circuit breaker is half-open (testing recovery)."""
        return self.state == CircuitBreakerState.HALF_OPEN
    
    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function through circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            Result of function execution
            
        Raises:
            CircuitOpenError: If circuit is open
            CircuitTimeoutError: If operation times out
            Any exception raised by the function
        """
        async with self._lock:
            await self._check_state()
            
            if self.state == CircuitBreakerState.OPEN:
                raise CircuitOpenError(f"Circuit breaker '{self.name}' is open")
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self._execute_function(func, *args, **kwargs),
                timeout=self.config.timeout
            )
            
            await self._handle_success()
            return result
            
        except asyncio.TimeoutError:
            await self._handle_failure()
            raise CircuitTimeoutError(f"Operation timed out after {self.config.timeout}s")
        except Exception as e:
            await self._handle_failure()
            raise e
    
    async def _execute_function(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function, handling both sync and async callables."""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            # Run sync function in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
    
    async def _check_state(self):
        """Check and update circuit breaker state based on current conditions."""
        if self.state == CircuitBreakerState.OPEN:
            # Check if recovery timeout has passed
            if (self.metrics.last_failure_time and 
                time.time() - self.metrics.last_failure_time >= self.config.recovery_timeout):
                await self._transition_to_half_open()
    
    async def _handle_success(self):
        """Handle successful operation."""
        async with self._lock:
            self.metrics.record_success()
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.metrics.success_count >= self.config.success_threshold:
                    await self._transition_to_closed()
            elif self.state == CircuitBreakerState.CLOSED:
                # Reset failure count on success
                self.metrics.failure_count = 0
    
    async def _handle_failure(self):
        """Handle failed operation."""
        async with self._lock:
            self.metrics.record_failure()
            
            if (self.state in [CircuitBreakerState.CLOSED, CircuitBreakerState.HALF_OPEN] and
                self.metrics.failure_count >= self.config.failure_threshold):
                await self._transition_to_open()
    
    async def _transition_to_closed(self):
        """Transition to closed state."""
        self.state = CircuitBreakerState.CLOSED
        self.metrics.reset()
        logger.info(f"Circuit breaker '{self.name}' transitioned to CLOSED")
    
    async def _transition_to_open(self):
        """Transition to open state."""
        self.state = CircuitBreakerState.OPEN
        logger.warning(f"Circuit breaker '{self.name}' transitioned to OPEN after {self.metrics.failure_count} failures")
    
    async def _transition_to_half_open(self):
        """Transition to half-open state."""
        self.state = CircuitBreakerState.HALF_OPEN
        self.metrics.reset()
        logger.info(f"Circuit breaker '{self.name}' transitioned to HALF_OPEN for recovery testing")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current circuit breaker metrics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.metrics.failure_count,
            "success_count": self.metrics.success_count,
            "total_requests": self.metrics.total_requests,
            "total_failures": self.metrics.total_failures,
            "total_successes": self.metrics.total_successes,
            "failure_rate": (
                self.metrics.total_failures / self.metrics.total_requests 
                if self.metrics.total_requests > 0 else 0
            ),
            "last_failure_time": self.metrics.last_failure_time
        }
    
    @asynccontextmanager
    async def context(self):
        """Context manager for circuit breaker operations."""
        try:
            yield self
        except Exception:
            # Let the exception propagate, metrics already handled in call()
            raise


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""
    
    def __init__(self):
        """Initialize empty registry."""
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._configs: Dict[str, CircuitBreakerConfig] = {}
    
    def register(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """
        Register a new circuit breaker.
        
        Args:
            name: Unique name for the circuit breaker
            config: Configuration for the circuit breaker
            
        Returns:
            Circuit breaker instance
        """
        if name in self._breakers:
            return self._breakers[name]
        
        breaker_config = config or CircuitBreakerConfig()
        breaker = CircuitBreaker(name, breaker_config)
        
        self._breakers[name] = breaker
        self._configs[name] = breaker_config
        
        logger.info(f"Registered circuit breaker: {name}")
        return breaker
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self._breakers.get(name)
    
    def get_or_create(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get existing circuit breaker or create new one."""
        if name in self._breakers:
            return self._breakers[name]
        return self.register(name, config)
    
    def remove(self, name: str) -> bool:
        """Remove circuit breaker from registry."""
        if name in self._breakers:
            del self._breakers[name]
            del self._configs[name]
            logger.info(f"Removed circuit breaker: {name}")
            return True
        return False
    
    def list_breakers(self) -> Dict[str, str]:
        """List all registered circuit breakers with their states."""
        return {name: breaker.state.value for name, breaker in self._breakers.items()}
    
    async def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all circuit breakers."""
        return {name: breaker.get_metrics() for name, breaker in self._breakers.items()}
    
    def reset_all(self):
        """Reset all circuit breakers to closed state."""
        for breaker in self._breakers.values():
            asyncio.create_task(breaker._transition_to_closed())
        logger.info("Reset all circuit breakers to CLOSED state")
    
    def __len__(self) -> int:
        """Get number of registered circuit breakers."""
        return len(self._breakers)
    
    def __contains__(self, name: str) -> bool:
        """Check if circuit breaker is registered."""
        return name in self._breakers
    
    async def get_health_dashboard(self) -> Dict[str, Any]:
        """
        Get comprehensive health dashboard for all circuit breakers.
        
        Returns:
            Dictionary containing health status and metrics for all breakers
        """
        dashboard = {
            "status": "healthy",
            "total_breakers": len(self._breakers),
            "breakers": {},
            "summary": {
                "healthy": 0,
                "degraded": 0,
                "unhealthy": 0
            }
        }
        
        for name, breaker in self._breakers.items():
            breaker_health = {
                "name": name,
                "state": breaker.state.value,
                "metrics": breaker.get_metrics(),
                "status": "healthy" if breaker.is_closed else "degraded" if breaker.is_half_open else "unhealthy"
            }
            
            dashboard["breakers"][name] = breaker_health
            dashboard["summary"][breaker_health["status"]] += 1
        
        # Overall status is unhealthy if any breaker is unhealthy, degraded if any are degraded
        if dashboard["summary"]["unhealthy"] > 0:
            dashboard["status"] = "unhealthy"
        elif dashboard["summary"]["degraded"] > 0:
            dashboard["status"] = "degraded"
        
        return dashboard


# Global registry instance
_global_registry = CircuitBreakerRegistry()


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """Get the global circuit breaker registry."""
    return _global_registry


def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """
    Get or create a circuit breaker.
    
    Args:
        name: Circuit breaker name
        config: Optional configuration
        
    Returns:
        Circuit breaker instance
    """
    return _global_registry.get_or_create(name, config)


def create_circuit_breaker_config(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    success_threshold: int = 3,
    timeout: float = 30.0
) -> CircuitBreakerConfig:
    """
    Create circuit breaker configuration with validation.
    
    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before testing recovery
        success_threshold: Successes needed to close from half-open
        timeout: Request timeout in seconds
        
    Returns:
        Validated configuration
    """
    return CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        success_threshold=success_threshold,
        timeout=timeout
    )


# Utility decorators for common circuit breaker patterns
def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """
    Decorator to wrap functions with circuit breaker protection.
    
    Args:
        name: Circuit breaker name
        config: Optional configuration
    """
    def decorator(func):
        breaker = get_circuit_breaker(name, config)
        
        async def wrapper(*args, **kwargs):
            return await breaker.call(func, *args, **kwargs)
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    
    return decorator


# Alias for backward compatibility
circuit_breaker_decorator = circuit_breaker


# Pre-configured circuit breaker factory functions
def create_api_circuit_breaker(name: str = "api_calls") -> CircuitBreaker:
    """
    Create a circuit breaker optimized for API calls.
    
    Args:
        name: Circuit breaker name
        
    Returns:
        Configured circuit breaker for API operations
    """
    config = CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=60.0,
        success_threshold=2,
        timeout=30.0
    )
    return CircuitBreaker(name, config)


def create_database_circuit_breaker(name: str = "database") -> CircuitBreaker:
    """
    Create a circuit breaker optimized for database operations.
    
    Args:
        name: Circuit breaker name
        
    Returns:
        Configured circuit breaker for database operations
    """
    config = CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout=30.0,
        success_threshold=3,
        timeout=10.0
    )
    return CircuitBreaker(name, config)


def create_llm_circuit_breaker(name: str = "llm_service") -> CircuitBreaker:
    """
    Create a circuit breaker optimized for LLM service calls.
    
    Args:
        name: Circuit breaker name
        
    Returns:
        Configured circuit breaker for LLM operations
    """
    config = CircuitBreakerConfig(
        failure_threshold=2,
        recovery_timeout=180.0,  # LLM recovery may take longer
        success_threshold=1,     # Single success to close
        timeout=120.0           # Allow longer timeouts for LLM calls
    )
    return CircuitBreaker(name, config)