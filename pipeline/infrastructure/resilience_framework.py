"""
Enterprise Resilience Framework - Production-Grade Reliability

Implements comprehensive resilience patterns:
- Circuit breakers with intelligent recovery
- Bulkhead isolation for fault containment
- Adaptive retry with exponential backoff
- Chaos engineering and failure injection
- Self-healing capabilities
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class FailureType(str, Enum):
    """Types of failures that can be injected."""
    TIMEOUT = "timeout"
    EXCEPTION = "exception"
    SLOW_RESPONSE = "slow_response"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_PARTITION = "network_partition"


class BulkheadType(str, Enum):
    """Types of bulkhead isolation."""
    THREAD_POOL = "thread_pool"
    SEMAPHORE = "semaphore"
    CIRCUIT_BREAKER = "circuit_breaker"
    RATE_LIMITER = "rate_limiter"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_seconds: float = 60.0
    half_open_max_calls: int = 3
    
    # Adaptive parameters
    adaptive_threshold: bool = True
    min_failure_threshold: int = 3
    max_failure_threshold: int = 20


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True
    
    # Adaptive parameters
    adaptive_delay: bool = True
    success_rate_threshold: float = 0.8


@dataclass
class BulkheadConfig:
    """Configuration for bulkhead isolation."""
    bulkhead_type: BulkheadType = BulkheadType.SEMAPHORE
    max_concurrent: int = 10
    queue_size: int = 100
    timeout_seconds: float = 30.0


@dataclass
class FailureInjectionConfig:
    """Configuration for chaos engineering."""
    enabled: bool = False
    failure_rate: float = 0.01  # 1% failure rate
    failure_types: List[FailureType] = field(default_factory=lambda: [FailureType.TIMEOUT])
    min_interval: float = 10.0  # Minimum seconds between failures
    max_duration: float = 5.0   # Maximum failure duration


@dataclass
class HealthMetrics:
    """Health and performance metrics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    
    total_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    
    circuit_breaker_trips: int = 0
    retry_attempts: int = 0
    bulkhead_rejections: int = 0
    
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    
    def update_success(self, response_time: float):
        """Update metrics for successful request."""
        self.total_requests += 1
        self.successful_requests += 1
        self.total_response_time += response_time
        self.min_response_time = min(self.min_response_time, response_time)
        self.max_response_time = max(self.max_response_time, response_time)
        self.last_success_time = datetime.now()
    
    def update_failure(self, response_time: float = 0.0):
        """Update metrics for failed request."""
        self.total_requests += 1
        self.failed_requests += 1
        if response_time > 0:
            self.total_response_time += response_time
        self.last_failure_time = datetime.now()
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests
    
    @property
    def average_response_time(self) -> float:
        """Calculate average response time."""
        if self.total_requests == 0:
            return 0.0
        return self.total_response_time / self.total_requests


class CircuitBreaker:
    """Intelligent circuit breaker with adaptive capabilities."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        
        # State tracking
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_calls = 0
        
        # Adaptive threshold management
        self.recent_failures: List[datetime] = []
        self.recent_successes: List[datetime] = []
        
        self.metrics = HealthMetrics()
        
        logger.info(f"Circuit breaker '{name}' initialized in {self.state.value} state")
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if not self._can_execute():
            self.metrics.circuit_breaker_trips += 1
            raise Exception(f"Circuit breaker '{self.name}' is {self.state.value}")
        
        start_time = time.time()
        
        try:
            # Execute the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            response_time = time.time() - start_time
            self._record_success(response_time)
            
            return result
            
        except Exception as e:
            response_time = time.time() - start_time
            self._record_failure(response_time)
            raise e
    
    def _can_execute(self) -> bool:
        """Check if circuit breaker allows execution."""
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            # Check if timeout period has elapsed
            if (self.last_failure_time and 
                datetime.now() - self.last_failure_time >= timedelta(seconds=self.config.timeout_seconds)):
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                logger.info(f"Circuit breaker '{self.name}' transitioned to HALF_OPEN")
                return True
            return False
        
        if self.state == CircuitState.HALF_OPEN:
            return self.half_open_calls < self.config.half_open_max_calls
        
        return False
    
    def _record_success(self, response_time: float):
        """Record successful execution."""
        self.metrics.update_success(response_time)
        self.recent_successes.append(datetime.now())
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            self.half_open_calls += 1
            
            if self.success_count >= self.config.success_threshold:
                self._transition_to_closed()
        else:
            self.failure_count = 0  # Reset failure count on success
        
        # Maintain recent success history
        cutoff_time = datetime.now() - timedelta(minutes=5)
        self.recent_successes = [t for t in self.recent_successes if t > cutoff_time]
    
    def _record_failure(self, response_time: float):
        """Record failed execution."""
        self.metrics.update_failure(response_time)
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        self.recent_failures.append(self.last_failure_time)
        
        if self.state == CircuitState.HALF_OPEN:
            self._transition_to_open()
        elif self.state == CircuitState.CLOSED:
            current_threshold = self._get_adaptive_threshold()
            if self.failure_count >= current_threshold:
                self._transition_to_open()
        
        # Maintain recent failure history
        cutoff_time = datetime.now() - timedelta(minutes=5)
        self.recent_failures = [t for t in self.recent_failures if t > cutoff_time]
    
    def _get_adaptive_threshold(self) -> int:
        """Calculate adaptive failure threshold."""
        if not self.config.adaptive_threshold:
            return self.config.failure_threshold
        
        # Adjust threshold based on recent performance
        recent_failure_rate = len(self.recent_failures) / max(1, len(self.recent_failures) + len(self.recent_successes))
        
        if recent_failure_rate > 0.2:  # High failure rate
            return max(self.config.min_failure_threshold, self.config.failure_threshold - 1)
        elif recent_failure_rate < 0.05:  # Low failure rate
            return min(self.config.max_failure_threshold, self.config.failure_threshold + 2)
        
        return self.config.failure_threshold
    
    def _transition_to_open(self):
        """Transition circuit breaker to OPEN state."""
        self.state = CircuitState.OPEN
        self.failure_count = 0
        self.success_count = 0
        logger.warning(f"Circuit breaker '{self.name}' transitioned to OPEN")
    
    def _transition_to_closed(self):
        """Transition circuit breaker to CLOSED state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        logger.info(f"Circuit breaker '{self.name}' transitioned to CLOSED")
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get current state information."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "success_rate": self.metrics.success_rate,
                "average_response_time": self.metrics.average_response_time,
                "circuit_breaker_trips": self.metrics.circuit_breaker_trips
            }
        }


class AdaptiveRetry:
    """Intelligent retry mechanism with exponential backoff and jitter."""
    
    def __init__(self, name: str, config: RetryConfig):
        self.name = name
        self.config = config
        self.metrics = HealthMetrics()
        
        # Adaptive delay tracking
        self.recent_success_rates: List[float] = []
        self.current_base_delay = config.base_delay
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with adaptive retry."""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            start_time = time.time()
            
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                response_time = time.time() - start_time
                self.metrics.update_success(response_time)
                
                # Update adaptive parameters on success
                if attempt > 0:
                    self._update_adaptive_parameters(success=True)
                
                return result
                
            except Exception as e:
                response_time = time.time() - start_time
                self.metrics.update_failure(response_time)
                self.metrics.retry_attempts += 1
                
                last_exception = e
                
                # Don't retry on final attempt
                if attempt == self.config.max_attempts - 1:
                    break
                
                # Calculate delay for next attempt
                delay = self._calculate_delay(attempt)
                
                logger.debug(f"Retry {attempt + 1}/{self.config.max_attempts} for '{self.name}' "
                           f"after {delay:.2f}s delay: {e}")
                
                await asyncio.sleep(delay)
        
        # All attempts failed
        self._update_adaptive_parameters(success=False)
        if last_exception:
            raise last_exception
        else:
            raise Exception(f"All {self.config.max_attempts} retry attempts failed")
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter."""
        # Exponential backoff
        delay = self.current_base_delay * (self.config.exponential_base ** attempt)
        
        # Apply maximum delay limit
        delay = min(delay, self.config.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.config.jitter:
            jitter_amount = delay * 0.1  # 10% jitter
            delay += (time.time() % 1) * jitter_amount - (jitter_amount / 2)
        
        return max(0.1, delay)
    
    def _update_adaptive_parameters(self, success: bool):
        """Update adaptive parameters based on success/failure."""
        if not self.config.adaptive_delay:
            return
        
        current_success_rate = self.metrics.success_rate
        self.recent_success_rates.append(current_success_rate)
        
        # Keep only recent history
        if len(self.recent_success_rates) > 20:
            self.recent_success_rates = self.recent_success_rates[-20:]
        
        # Adjust base delay based on performance
        avg_success_rate = sum(self.recent_success_rates) / len(self.recent_success_rates)
        
        if avg_success_rate < self.config.success_rate_threshold:
            # Poor success rate - increase delay
            self.current_base_delay = min(self.config.max_delay / 4, self.current_base_delay * 1.2)
        elif avg_success_rate > 0.95:
            # Excellent success rate - decrease delay
            self.current_base_delay = max(self.config.base_delay, self.current_base_delay * 0.9)


class BulkheadIsolation:
    """Bulkhead pattern for fault isolation."""
    
    def __init__(self, name: str, config: BulkheadConfig):
        self.name = name
        self.config = config
        self.metrics = HealthMetrics()
        
        # Initialize based on bulkhead type
        if config.bulkhead_type == BulkheadType.SEMAPHORE:
            self.semaphore = asyncio.Semaphore(config.max_concurrent)
        elif config.bulkhead_type == BulkheadType.THREAD_POOL:
            # For thread pool, we'll use asyncio executor with limited threads
            import concurrent.futures
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=config.max_concurrent)
        
        self.active_requests: Set[str] = set()
        self.queue_size = 0
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with bulkhead isolation."""
        request_id = str(uuid4())
        
        try:
            if self.config.bulkhead_type == BulkheadType.SEMAPHORE:
                return await self._execute_with_semaphore(request_id, func, *args, **kwargs)
            elif self.config.bulkhead_type == BulkheadType.THREAD_POOL:
                return await self._execute_with_thread_pool(request_id, func, *args, **kwargs)
            else:
                # Direct execution for other types
                return await self._execute_direct(request_id, func, *args, **kwargs)
                
        except Exception as e:
            self.metrics.bulkhead_rejections += 1
            raise e
    
    async def _execute_with_semaphore(self, request_id: str, func: Callable, *args, **kwargs) -> Any:
        """Execute with semaphore-based bulkhead."""
        try:
            # Acquire semaphore with timeout
            await asyncio.wait_for(
                self.semaphore.acquire(),
                timeout=self.config.timeout_seconds
            )
            
            self.active_requests.add(request_id)
            start_time = time.time()
            
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                response_time = time.time() - start_time
                self.metrics.update_success(response_time)
                return result
                
            finally:
                self.active_requests.discard(request_id)
                self.semaphore.release()
                
        except asyncio.TimeoutError:
            raise Exception(f"Bulkhead '{self.name}' rejected request due to capacity limit")
    
    async def _execute_with_thread_pool(self, request_id: str, func: Callable, *args, **kwargs) -> Any:
        """Execute with thread pool bulkhead."""
        if len(self.active_requests) >= self.config.max_concurrent:
            raise Exception(f"Bulkhead '{self.name}' at capacity")
        
        self.active_requests.add(request_id)
        start_time = time.time()
        
        try:
            # Execute in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.executor, func, *args, **kwargs)
            
            response_time = time.time() - start_time
            self.metrics.update_success(response_time)
            return result
            
        finally:
            self.active_requests.discard(request_id)
    
    async def _execute_direct(self, request_id: str, func: Callable, *args, **kwargs) -> Any:
        """Direct execution without specific bulkhead type."""
        self.active_requests.add(request_id)
        start_time = time.time()
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            response_time = time.time() - start_time
            self.metrics.update_success(response_time)
            return result
            
        finally:
            self.active_requests.discard(request_id)
    
    def get_status(self) -> Dict[str, Any]:
        """Get bulkhead status."""
        return {
            "name": self.name,
            "type": self.config.bulkhead_type.value,
            "active_requests": len(self.active_requests),
            "max_concurrent": self.config.max_concurrent,
            "utilization": len(self.active_requests) / self.config.max_concurrent,
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "success_rate": self.metrics.success_rate,
                "rejections": self.metrics.bulkhead_rejections
            }
        }


class ChaosEngineer:
    """Chaos engineering for failure injection and testing."""
    
    def __init__(self, config: FailureInjectionConfig):
        self.config = config
        self.last_failure_time: Optional[datetime] = None
        self.active_failures: Dict[str, datetime] = {}
    
    async def maybe_inject_failure(self, operation_name: str):
        """Maybe inject a failure based on configuration."""
        if not self.config.enabled:
            return
        
        # Check if enough time has passed since last failure
        now = datetime.now()
        if (self.last_failure_time and 
            (now - self.last_failure_time).total_seconds() < self.config.min_interval):
            return
        
        # Check if failure should be injected
        if time.time() % 1 < self.config.failure_rate:
            failure_type = random.choice(self.config.failure_types)
            await self._inject_failure(operation_name, failure_type)
            self.last_failure_time = now
    
    async def _inject_failure(self, operation_name: str, failure_type: FailureType):
        """Inject specific type of failure."""
        logger.warning(f"Chaos engineering: Injecting {failure_type.value} failure for {operation_name}")
        
        if failure_type == FailureType.TIMEOUT:
            await asyncio.sleep(self.config.max_duration)
            raise asyncio.TimeoutError(f"Chaos timeout injected for {operation_name}")
        
        elif failure_type == FailureType.EXCEPTION:
            raise Exception(f"Chaos exception injected for {operation_name}")
        
        elif failure_type == FailureType.SLOW_RESPONSE:
            delay = self.config.max_duration * time.time() % 1
            await asyncio.sleep(delay)
        
        elif failure_type == FailureType.RESOURCE_EXHAUSTION:
            # Simulate resource exhaustion
            large_data = [0] * 1000000  # Allocate large memory
            await asyncio.sleep(0.1)
            del large_data
            raise Exception(f"Resource exhaustion injected for {operation_name}")


class ResilienceFramework:
    """Main resilience framework orchestrating all patterns."""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_handlers: Dict[str, AdaptiveRetry] = {}
        self.bulkheads: Dict[str, BulkheadIsolation] = {}
        self.chaos_engineer: Optional[ChaosEngineer] = None
        
        # Global metrics
        self.global_metrics = HealthMetrics()
        self.start_time = datetime.now()
    
    def register_circuit_breaker(self, name: str, config: CircuitBreakerConfig) -> CircuitBreaker:
        """Register a new circuit breaker."""
        circuit_breaker = CircuitBreaker(name, config)
        self.circuit_breakers[name] = circuit_breaker
        return circuit_breaker
    
    def register_retry_handler(self, name: str, config: RetryConfig) -> AdaptiveRetry:
        """Register a new retry handler."""
        retry_handler = AdaptiveRetry(name, config)
        self.retry_handlers[name] = retry_handler
        return retry_handler
    
    def register_bulkhead(self, name: str, config: BulkheadConfig) -> BulkheadIsolation:
        """Register a new bulkhead."""
        bulkhead = BulkheadIsolation(name, config)
        self.bulkheads[name] = bulkhead
        return bulkhead
    
    def enable_chaos_engineering(self, config: FailureInjectionConfig):
        """Enable chaos engineering."""
        self.chaos_engineer = ChaosEngineer(config)
        logger.info("Chaos engineering enabled")
    
    async def execute_with_resilience(self, 
                                    operation_name: str,
                                    func: Callable,
                                    *args,
                                    use_circuit_breaker: bool = True,
                                    use_retry: bool = True,
                                    use_bulkhead: bool = True,
                                    **kwargs) -> Any:
        """Execute function with all resilience patterns."""
        start_time = time.time()
        
        try:
            # Chaos engineering injection
            if self.chaos_engineer:
                await self.chaos_engineer.maybe_inject_failure(operation_name)
            
            # Wrap function with resilience patterns
            protected_func = func
            
            # Apply bulkhead isolation
            if use_bulkhead and operation_name in self.bulkheads:
                bulkhead = self.bulkheads[operation_name]
                protected_func = lambda: bulkhead.execute(protected_func, *args, **kwargs)
                args, kwargs = (), {}
            
            # Apply circuit breaker
            if use_circuit_breaker and operation_name in self.circuit_breakers:
                circuit_breaker = self.circuit_breakers[operation_name]
                protected_func = lambda: circuit_breaker.call(protected_func, *args, **kwargs)
                args, kwargs = (), {}
            
            # Apply retry logic
            if use_retry and operation_name in self.retry_handlers:
                retry_handler = self.retry_handlers[operation_name]
                result = await retry_handler.execute(protected_func, *args, **kwargs)
            else:
                if asyncio.iscoroutinefunction(protected_func):
                    result = await protected_func(*args, **kwargs)
                else:
                    result = protected_func(*args, **kwargs)
            
            # Record success
            response_time = time.time() - start_time
            self.global_metrics.update_success(response_time)
            
            return result
            
        except Exception as e:
            # Record failure
            response_time = time.time() - start_time
            self.global_metrics.update_failure(response_time)
            raise e
    
    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        report = {
            "framework_status": {
                "uptime_seconds": uptime,
                "global_metrics": {
                    "total_requests": self.global_metrics.total_requests,
                    "success_rate": self.global_metrics.success_rate,
                    "average_response_time": self.global_metrics.average_response_time
                }
            },
            "circuit_breakers": {
                name: cb.get_state_info() for name, cb in self.circuit_breakers.items()
            },
            "bulkheads": {
                name: bh.get_status() for name, bh in self.bulkheads.items()
            },
            "retry_handlers": {
                name: {
                    "name": rh.name,
                    "total_requests": rh.metrics.total_requests,
                    "success_rate": rh.metrics.success_rate,
                    "retry_attempts": rh.metrics.retry_attempts,
                    "current_base_delay": rh.current_base_delay
                } for name, rh in self.retry_handlers.items()
            },
            "chaos_engineering": {
                "enabled": self.chaos_engineer is not None,
                "config": {
                    "failure_rate": self.chaos_engineer.config.failure_rate if self.chaos_engineer else 0,
                    "failure_types": [ft.value for ft in self.chaos_engineer.config.failure_types] if self.chaos_engineer else []
                } if self.chaos_engineer else None
            }
        }
        
        return report


# Factory functions
def create_resilience_framework() -> ResilienceFramework:
    """Create and return a configured resilience framework."""
    return ResilienceFramework()


# Example usage
async def resilience_demo():
    """Demonstrate resilience framework capabilities."""
    import random
    
    def unreliable_service(service_name: str, failure_rate: float = 0.3):
        """Mock unreliable service for testing."""
        import time
        time.sleep(random.uniform(0.1, 0.5))  # Simulate processing time
        
        if time.time() % 1 < failure_rate:
            raise Exception(f"Service {service_name} failed randomly")
        
        return f"Success from {service_name}"
    
    # Create resilience framework
    framework = create_resilience_framework()
    
    # Configure resilience patterns
    framework.register_circuit_breaker("test_service", CircuitBreakerConfig(
        failure_threshold=3,
        timeout_seconds=5.0
    ))
    
    framework.register_retry_handler("test_service", RetryConfig(
        max_attempts=3,
        base_delay=0.5
    ))
    
    framework.register_bulkhead("test_service", BulkheadConfig(
        bulkhead_type=BulkheadType.SEMAPHORE,
        max_concurrent=5
    ))
    
    # Enable chaos engineering
    framework.enable_chaos_engineering(FailureInjectionConfig(
        enabled=True,
        failure_rate=0.05,
        failure_types=[FailureType.TIMEOUT, FailureType.EXCEPTION]
    ))
    
    # Test resilience patterns
    successful_calls = 0
    total_calls = 20
    
    for i in range(total_calls):
        try:
            result = await framework.execute_with_resilience(
                "test_service",
                unreliable_service,
                f"call_{i}",
                0.2  # 20% failure rate
            )
            successful_calls += 1
            logger.info(f"Call {i+1}: {result}")
        except Exception as e:
            logger.error(f"Call {i+1} failed: {e}")
    
    # Generate health report
    health_report = framework.get_health_report()
    
    print(f"\nResilience Framework Demo Results:")
    print(f"Successful calls: {successful_calls}/{total_calls}")
    print(f"Global success rate: {health_report['framework_status']['global_metrics']['success_rate']:.2%}")
    print(f"Average response time: {health_report['framework_status']['global_metrics']['average_response_time']:.3f}s")
    
    return health_report


if __name__ == "__main__":
    import random
    # Run demonstration
    asyncio.run(resilience_demo())