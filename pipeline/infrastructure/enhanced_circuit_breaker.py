"""
Enhanced Circuit Breaker with Advanced Resilience Patterns
Implements comprehensive failure handling, adaptive thresholds, and recovery strategies.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
import threading
import statistics
from collections import deque
import json

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states with enhanced monitoring"""
    CLOSED = "closed"         # Normal operation
    OPEN = "open"             # Failing fast
    HALF_OPEN = "half_open"   # Testing recovery
    ADAPTIVE = "adaptive"     # Smart threshold adjustment
    QUARANTINE = "quarantine" # Extended recovery period


class FailureType(str, Enum):
    """Types of failures for categorized handling"""
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION_ERROR = "authentication_error"
    SERVICE_ERROR = "service_error"
    UNKNOWN = "unknown"


@dataclass
class FailureRecord:
    """Record of a failure with context"""
    timestamp: datetime
    failure_type: FailureType
    error_message: str
    duration_ms: float
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitMetrics:
    """Comprehensive circuit breaker metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeouts: int = 0
    circuit_opens: int = 0
    recovery_attempts: int = 0
    average_response_time: float = 0.0
    error_rate: float = 0.0
    uptime_percentage: float = 100.0
    last_failure: Optional[datetime] = None
    last_success: Optional[datetime] = None


class EnhancedCircuitBreaker:
    """
    Advanced circuit breaker with adaptive thresholds and comprehensive monitoring.
    
    Features:
    - Adaptive failure thresholds based on historical performance
    - Categorized failure handling with different recovery strategies
    - Exponential backoff with jitter
    - Health scoring and predictive failure detection
    - Comprehensive metrics and monitoring
    - Self-healing capabilities
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        timeout: float = 10.0,
        adaptive_threshold: bool = True,
        max_failures_window: int = 100,
        health_check_interval: float = 30.0
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.timeout = timeout
        self.adaptive_threshold = adaptive_threshold
        self.max_failures_window = max_failures_window
        self.health_check_interval = health_check_interval
        
        # State management
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_success_time: Optional[datetime] = None
        
        # Advanced tracking
        self.failure_history: deque[FailureRecord] = deque(maxlen=max_failures_window)
        self.success_history: deque[datetime] = deque(maxlen=max_failures_window)
        self.response_times: deque[float] = deque(maxlen=100)
        
        # Metrics
        self.metrics = CircuitMetrics()
        
        # Adaptive threshold calculation
        self.base_failure_threshold = failure_threshold
        self.current_threshold = failure_threshold
        
        # Threading
        self._lock = threading.RLock()
        
        # Health monitoring
        self._health_score = 1.0
        self._last_health_check = datetime.utcnow()
        
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
    async def call(
        self, 
        func: Callable[..., Awaitable[Any]], 
        *args, 
        **kwargs
    ) -> Any:
        """
        Execute a function call with circuit breaker protection.
        
        Args:
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: When circuit is open
            TimeoutError: When function times out
            Original exception: When function fails
        """
        
        # Check circuit state before execution
        if not await self._can_execute():
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is {self.state.value}"
            )
            
        start_time = time.time()
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.timeout
            )
            
            # Record success
            execution_time = (time.time() - start_time) * 1000  # ms
            await self._record_success(execution_time)
            
            return result
            
        except asyncio.TimeoutError:
            execution_time = (time.time() - start_time) * 1000
            await self._record_failure(FailureType.TIMEOUT, "Function timeout", execution_time)
            raise
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            failure_type = self._classify_failure(e)
            await self._record_failure(failure_type, str(e), execution_time)
            raise
    
    async def _can_execute(self) -> bool:
        """Check if the circuit allows execution"""
        
        with self._lock:
            now = datetime.utcnow()
            
            # Update health score
            await self._update_health_score()
            
            if self.state == CircuitState.CLOSED:
                return True
                
            elif self.state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if (self.last_failure_time and 
                    now - self.last_failure_time >= timedelta(seconds=self.recovery_timeout)):
                    
                    self.state = CircuitState.HALF_OPEN
                    self.logger.info(f"Circuit breaker '{self.name}' transitioning to HALF_OPEN")
                    return True
                    
                return False
                
            elif self.state == CircuitState.HALF_OPEN:
                # Allow limited requests to test recovery
                return True
                
            elif self.state == CircuitState.ADAPTIVE:
                # Use health score for smart decisions
                return self._health_score > 0.5
                
            elif self.state == CircuitState.QUARANTINE:
                # Extended recovery period
                quarantine_time = self.recovery_timeout * 3
                if (self.last_failure_time and 
                    now - self.last_failure_time >= timedelta(seconds=quarantine_time)):
                    
                    self.state = CircuitState.HALF_OPEN
                    self.logger.info(f"Circuit breaker '{self.name}' exiting quarantine")
                    return True
                    
                return False
                
            return False
    
    async def _record_success(self, execution_time: float):
        """Record a successful execution"""
        
        with self._lock:
            now = datetime.utcnow()
            
            # Update metrics
            self.metrics.total_requests += 1
            self.metrics.successful_requests += 1
            self.metrics.last_success = now
            self.last_success_time = now
            
            # Track response time
            self.response_times.append(execution_time)
            if self.response_times:
                self.metrics.average_response_time = statistics.mean(self.response_times)
            
            # Update success history
            self.success_history.append(now)
            
            # State transitions
            if self.state == CircuitState.HALF_OPEN:
                # Successful recovery test
                self.failure_count = 0
                self.state = CircuitState.CLOSED
                self.logger.info(f"Circuit breaker '{self.name}' recovered to CLOSED")
                
            elif self.state in [CircuitState.ADAPTIVE, CircuitState.QUARANTINE]:
                # Check if we can return to normal operation
                if self._calculate_recent_success_rate() > 0.8:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.logger.info(f"Circuit breaker '{self.name}' recovered to CLOSED")
            
            # Update health score positively
            self._health_score = min(self._health_score + 0.1, 1.0)
            
            # Update error rate
            self._update_error_rate()
    
    async def _record_failure(self, failure_type: FailureType, error_message: str, execution_time: float):
        """Record a failed execution"""
        
        with self._lock:
            now = datetime.utcnow()
            
            # Create failure record
            failure_record = FailureRecord(
                timestamp=now,
                failure_type=failure_type,
                error_message=error_message,
                duration_ms=execution_time
            )
            
            self.failure_history.append(failure_record)
            
            # Update metrics
            self.metrics.total_requests += 1
            self.metrics.failed_requests += 1
            self.metrics.last_failure = now
            self.last_failure_time = now
            
            if failure_type == FailureType.TIMEOUT:
                self.metrics.timeouts += 1
            
            self.failure_count += 1
            
            # Adaptive threshold calculation
            if self.adaptive_threshold:
                self._update_adaptive_threshold()
            
            # State transitions based on failure patterns
            await self._evaluate_state_transition(failure_type)
            
            # Update health score negatively
            self._health_score = max(self._health_score - 0.2, 0.0)
            
            # Update error rate
            self._update_error_rate()
            
            self.logger.warning(
                f"Circuit breaker '{self.name}' recorded failure: {failure_type.value} - {error_message}"
            )
    
    def _classify_failure(self, exception: Exception) -> FailureType:
        """Classify the type of failure based on exception"""
        
        exception_name = type(exception).__name__.lower()
        error_message = str(exception).lower()
        
        if "timeout" in exception_name or "timeout" in error_message:
            return FailureType.TIMEOUT
            
        elif "connection" in exception_name or "connection" in error_message:
            return FailureType.CONNECTION_ERROR
            
        elif "rate" in error_message or "limit" in error_message:
            return FailureType.RATE_LIMIT
            
        elif "auth" in error_message or "unauthorized" in error_message:
            return FailureType.AUTHENTICATION_ERROR
            
        elif "service" in error_message or "server" in error_message:
            return FailureType.SERVICE_ERROR
            
        else:
            return FailureType.UNKNOWN
    
    async def _evaluate_state_transition(self, failure_type: FailureType):
        """Evaluate if circuit state should change based on failure patterns"""
        
        if self.state == CircuitState.CLOSED:
            
            # Check if we've exceeded the failure threshold
            if self.failure_count >= self.current_threshold:
                
                # Determine next state based on failure patterns
                recent_failures = self._get_recent_failures(minutes=5)
                failure_types = [f.failure_type for f in recent_failures]
                
                # If all recent failures are timeouts or connection errors, go to quarantine
                if len(recent_failures) >= 3 and all(
                    ft in [FailureType.TIMEOUT, FailureType.CONNECTION_ERROR] 
                    for ft in failure_types
                ):
                    self.state = CircuitState.QUARANTINE
                    self.metrics.circuit_opens += 1
                    self.logger.error(f"Circuit breaker '{self.name}' entered QUARANTINE due to persistent {failure_type.value}")
                    
                # If health score is very low, use adaptive mode
                elif self._health_score < 0.3:
                    self.state = CircuitState.ADAPTIVE
                    self.logger.warning(f"Circuit breaker '{self.name}' entered ADAPTIVE mode")
                    
                # Otherwise, standard open state
                else:
                    self.state = CircuitState.OPEN
                    self.metrics.circuit_opens += 1
                    self.logger.error(f"Circuit breaker '{self.name}' OPENED due to failures")
                    
        elif self.state == CircuitState.HALF_OPEN:
            # Failed during recovery test
            self.state = CircuitState.OPEN
            self.metrics.recovery_attempts += 1
            self.recovery_timeout = min(self.recovery_timeout * 1.5, 300.0)  # Exponential backoff
            self.logger.error(f"Circuit breaker '{self.name}' recovery failed, extending timeout")
    
    def _update_adaptive_threshold(self):
        """Update failure threshold based on historical performance"""
        
        if len(self.failure_history) < 10:
            return
            
        # Calculate recent error rate
        recent_error_rate = self._calculate_recent_error_rate()
        
        # Adjust threshold based on error rate trends
        if recent_error_rate > 0.2:  # High error rate
            self.current_threshold = max(self.base_failure_threshold - 2, 2)
        elif recent_error_rate < 0.05:  # Low error rate
            self.current_threshold = min(self.base_failure_threshold + 3, 15)
        else:
            self.current_threshold = self.base_failure_threshold
            
        self.logger.debug(f"Adaptive threshold updated to {self.current_threshold} (base: {self.base_failure_threshold})")
    
    def _calculate_recent_error_rate(self, minutes: int = 10) -> float:
        """Calculate error rate for recent requests"""
        
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        
        recent_failures = [f for f in self.failure_history if f.timestamp >= cutoff]
        recent_successes = [s for s in self.success_history if s >= cutoff]
        
        total_recent = len(recent_failures) + len(recent_successes)
        
        if total_recent == 0:
            return 0.0
            
        return len(recent_failures) / total_recent
    
    def _calculate_recent_success_rate(self, minutes: int = 10) -> float:
        """Calculate success rate for recent requests"""
        return 1.0 - self._calculate_recent_error_rate(minutes)
    
    def _get_recent_failures(self, minutes: int = 5) -> List[FailureRecord]:
        """Get failures from the last N minutes"""
        
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        return [f for f in self.failure_history if f.timestamp >= cutoff]
    
    async def _update_health_score(self):
        """Update the health score based on various factors"""
        
        now = datetime.utcnow()
        
        # Only update if enough time has passed
        if now - self._last_health_check < timedelta(seconds=10):
            return
            
        self._last_health_check = now
        
        # Base health calculation
        recent_success_rate = self._calculate_recent_success_rate(minutes=5)
        
        # Response time factor
        response_time_factor = 1.0
        if self.response_times:
            avg_response = statistics.mean(self.response_times)
            if avg_response > self.timeout * 1000 * 0.5:  # More than 50% of timeout
                response_time_factor = 0.7
            elif avg_response > self.timeout * 1000 * 0.8:  # More than 80% of timeout
                response_time_factor = 0.5
                
        # Failure pattern factor
        pattern_factor = 1.0
        recent_failures = self._get_recent_failures(minutes=5)
        if len(recent_failures) > 0:
            # Check for escalating failures
            failure_times = [f.timestamp for f in recent_failures]
            if len(failure_times) >= 3:
                time_diffs = [(failure_times[i] - failure_times[i-1]).total_seconds() 
                             for i in range(1, len(failure_times))]
                if all(diff < 30 for diff in time_diffs):  # Rapid failures
                    pattern_factor = 0.3
                    
        # Calculate composite health score
        self._health_score = recent_success_rate * response_time_factor * pattern_factor
        self._health_score = max(min(self._health_score, 1.0), 0.0)
    
    def _update_error_rate(self):
        """Update the current error rate metric"""
        
        if self.metrics.total_requests > 0:
            self.metrics.error_rate = self.metrics.failed_requests / self.metrics.total_requests
            self.metrics.uptime_percentage = (self.metrics.successful_requests / self.metrics.total_requests) * 100
    
    async def force_open(self, reason: str = "Manual override"):
        """Manually open the circuit breaker"""
        
        with self._lock:
            previous_state = self.state
            self.state = CircuitState.OPEN
            self.last_failure_time = datetime.utcnow()
            
            self.logger.warning(
                f"Circuit breaker '{self.name}' manually opened: {reason} "
                f"(previous state: {previous_state.value})"
            )
    
    async def force_close(self, reason: str = "Manual override"):
        """Manually close the circuit breaker"""
        
        with self._lock:
            previous_state = self.state
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.last_success_time = datetime.utcnow()
            
            self.logger.info(
                f"Circuit breaker '{self.name}' manually closed: {reason} "
                f"(previous state: {previous_state.value})"
            )
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive circuit breaker status"""
        
        with self._lock:
            recent_failures = self._get_recent_failures(minutes=10)
            
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "current_threshold": self.current_threshold,
                "base_threshold": self.base_failure_threshold,
                "health_score": round(self._health_score, 3),
                "recent_error_rate": round(self._calculate_recent_error_rate(), 3),
                "recent_failures": len(recent_failures),
                "recovery_timeout": self.recovery_timeout,
                "metrics": {
                    "total_requests": self.metrics.total_requests,
                    "successful_requests": self.metrics.successful_requests,
                    "failed_requests": self.metrics.failed_requests,
                    "error_rate": round(self.metrics.error_rate, 3),
                    "uptime_percentage": round(self.metrics.uptime_percentage, 2),
                    "average_response_time": round(self.metrics.average_response_time, 2),
                    "circuit_opens": self.metrics.circuit_opens,
                    "recovery_attempts": self.metrics.recovery_attempts,
                    "timeouts": self.metrics.timeouts
                },
                "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
                "last_success_time": self.last_success_time.isoformat() if self.last_success_time else None,
                "failure_types": {
                    failure_type.value: len([f for f in recent_failures if f.failure_type == failure_type])
                    for failure_type in FailureType
                }
            }
    
    def get_failure_patterns(self) -> Dict[str, Any]:
        """Analyze failure patterns for insights"""
        
        if not self.failure_history:
            return {"message": "No failure data available"}
            
        # Group failures by type
        failure_by_type = {}
        for failure in self.failure_history:
            if failure.failure_type not in failure_by_type:
                failure_by_type[failure.failure_type] = []
            failure_by_type[failure.failure_type].append(failure)
            
        # Analyze patterns
        patterns = {}
        for failure_type, failures in failure_by_type.items():
            if len(failures) >= 2:
                timestamps = [f.timestamp for f in failures]
                durations = [f.duration_ms for f in failures]
                
                # Calculate time between failures
                time_gaps = [(timestamps[i] - timestamps[i-1]).total_seconds() 
                           for i in range(1, len(timestamps))]
                
                patterns[failure_type.value] = {
                    "count": len(failures),
                    "average_duration": statistics.mean(durations),
                    "average_gap_seconds": statistics.mean(time_gaps) if time_gaps else 0,
                    "recent_count": len([f for f in failures 
                                       if f.timestamp >= datetime.utcnow() - timedelta(hours=1)])
                }
                
        return {
            "analysis_period": f"Last {len(self.failure_history)} failures",
            "total_failures": len(self.failure_history),
            "patterns": patterns,
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on failure patterns"""
        
        recommendations = []
        
        # High timeout rate
        timeout_rate = len([f for f in self.failure_history if f.failure_type == FailureType.TIMEOUT]) / max(len(self.failure_history), 1)
        if timeout_rate > 0.3:
            recommendations.append("Consider increasing timeout values - high timeout rate detected")
            
        # Rapid failures
        recent_failures = self._get_recent_failures(minutes=5)
        if len(recent_failures) >= 5:
            recommendations.append("Investigate system stability - rapid failure pattern detected")
            
        # Low health score
        if self._health_score < 0.5:
            recommendations.append("System health is degraded - consider maintenance or scaling")
            
        # High error rate
        if self.metrics.error_rate > 0.2:
            recommendations.append("Error rate is high - review system dependencies and configuration")
            
        return recommendations if recommendations else ["System appears to be operating normally"]


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""
    pass


# Global circuit breaker registry
_circuit_breakers: Dict[str, EnhancedCircuitBreaker] = {}
_registry_lock = threading.Lock()


def get_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    timeout: float = 10.0,
    adaptive_threshold: bool = True
) -> EnhancedCircuitBreaker:
    """
    Get or create a circuit breaker with the given configuration.
    
    Args:
        name: Circuit breaker name
        failure_threshold: Number of failures before opening
        recovery_timeout: Seconds to wait before attempting recovery
        timeout: Function execution timeout
        adaptive_threshold: Enable adaptive threshold adjustment
        
    Returns:
        EnhancedCircuitBreaker instance
    """
    
    with _registry_lock:
        if name not in _circuit_breakers:
            _circuit_breakers[name] = EnhancedCircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                timeout=timeout,
                adaptive_threshold=adaptive_threshold
            )
            
        return _circuit_breakers[name]


def get_all_circuit_breakers() -> Dict[str, Dict[str, Any]]:
    """Get status of all circuit breakers"""
    
    with _registry_lock:
        return {
            name: breaker.get_status()
            for name, breaker in _circuit_breakers.items()
        }


async def circuit_breaker_health_check() -> Dict[str, Any]:
    """Perform health check on all circuit breakers"""
    
    with _registry_lock:
        circuit_statuses = {}
        overall_health = "healthy"
        
        for name, breaker in _circuit_breakers.items():
            status = breaker.get_status()
            circuit_statuses[name] = status
            
            # Check if any circuit is in concerning state
            if status["state"] in ["open", "quarantine"]:
                overall_health = "degraded"
            elif status["state"] in ["adaptive"] and status["health_score"] < 0.5:
                overall_health = "degraded"
                
        return {
            "overall_health": overall_health,
            "total_circuits": len(_circuit_breakers),
            "circuits": circuit_statuses,
            "timestamp": datetime.utcnow().isoformat()
        }