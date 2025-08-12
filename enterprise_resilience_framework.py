#!/usr/bin/env python3
"""
Enterprise Resilience Framework - Generation 2 (MAKE IT ROBUST)
===============================================================

Advanced error handling, recovery, and resilience patterns for enterprise-grade systems.
Implements comprehensive failure detection, circuit breakers, retry mechanisms,
and self-healing capabilities with quantum-inspired recovery strategies.

Features:
- Multi-layered error handling
- Intelligent circuit breakers  
- Adaptive retry mechanisms
- Self-healing systems
- Chaos engineering integration
- Comprehensive monitoring
"""

import asyncio
import json
import logging
import math
import random
import time
import traceback
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
import uuid

logger = logging.getLogger(__name__)


class FailureType(str, Enum):
    """Types of failures the system can handle"""
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    AUTHENTICATION_ERROR = "authentication_error"
    RATE_LIMIT = "rate_limit"
    SERVICE_UNAVAILABLE = "service_unavailable"
    DATA_CORRUPTION = "data_corruption"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    LOGIC_ERROR = "logic_error"
    CONFIGURATION_ERROR = "configuration_error"
    UNKNOWN = "unknown"


class RecoveryStrategy(str, Enum):
    """Recovery strategies for different failure types"""
    IMMEDIATE_RETRY = "immediate_retry"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    CIRCUIT_BREAKER = "circuit_breaker"
    FALLBACK_OPERATION = "fallback_operation"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    QUANTUM_RECOVERY = "quantum_recovery"
    SYSTEM_RESTART = "system_restart"


class CircuitState(str, Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, failing fast
    HALF_OPEN = "half_open"  # Testing if service has recovered


class ResilienceLevel(str, Enum):
    """Resilience level configuration"""
    BASIC = "basic"
    ENHANCED = "enhanced"
    ENTERPRISE = "enterprise"
    QUANTUM = "quantum"


@dataclass
class FailureContext:
    """Context information about a failure"""
    failure_id: str
    failure_type: FailureType
    error_message: str
    stack_trace: Optional[str]
    timestamp: datetime
    component: str
    operation: str
    input_data: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    user_context: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    recovery_attempts: List[str] = field(default_factory=list)


@dataclass
class RecoveryResult:
    """Result of a recovery attempt"""
    success: bool
    strategy_used: RecoveryStrategy
    recovery_time_ms: float
    error_resolved: bool
    fallback_used: bool
    additional_info: Dict[str, Any] = field(default_factory=dict)
    next_strategy: Optional[RecoveryStrategy] = None


@dataclass  
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_seconds: int = 60
    half_open_max_calls: int = 10
    failure_rate_threshold: float = 0.5
    min_calls_for_evaluation: int = 10


class QuantumCircuitBreaker:
    """Quantum-inspired circuit breaker with adaptive behavior"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.half_open_call_count = 0
        
        # Quantum-inspired properties
        self.quantum_state_probability = [0.7, 0.2, 0.1]  # [closed, half_open, open]
        self.entanglement_partners = []
        self.coherence_time = 30  # seconds
        self.measurement_history = deque(maxlen=100)
        
        # Statistics
        self.call_history = deque(maxlen=1000)
        self.state_transitions = []
        self.total_calls = 0
        self.total_failures = 0
        
    async def call(self, operation: Callable[..., Any], *args, **kwargs) -> Any:
        """Execute operation through circuit breaker"""
        self.total_calls += 1
        
        # Check if circuit allows call
        if not await self._can_execute():
            raise CircuitBreakerOpenException(f"Circuit breaker {self.name} is OPEN")
        
        try:
            # Execute operation
            start_time = time.time()
            result = await operation(*args, **kwargs)
            execution_time = (time.time() - start_time) * 1000
            
            # Record success
            await self._record_success(execution_time)
            return result
            
        except Exception as e:
            # Record failure
            await self._record_failure(str(e))
            raise
    
    async def _can_execute(self) -> bool:
        """Check if circuit breaker allows execution"""
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            # Check if timeout has passed
            if (self.last_failure_time and 
                datetime.now() - self.last_failure_time > timedelta(seconds=self.config.timeout_seconds)):
                await self._transition_to_half_open()
                return True
            
            # Quantum tunneling - small probability of allowing calls even when open
            quantum_tunnel_probability = 0.1 * math.exp(-self.failure_count / 10.0)
            if random.random() < quantum_tunnel_probability:
                logger.debug(f"Quantum tunneling allowed call in {self.name}")
                return True
            
            return False
        
        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_call_count < self.config.half_open_max_calls:
                self.half_open_call_count += 1
                return True
            return False
        
        return False
    
    async def _record_success(self, execution_time_ms: float) -> None:
        """Record successful operation"""
        self.call_history.append({"success": True, "time": execution_time_ms, "timestamp": datetime.now()})
        
        if self.state == CircuitState.CLOSED:
            self.success_count += 1
        
        elif self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                await self._transition_to_closed()
        
        # Update quantum probabilities based on success
        self._update_quantum_probabilities(success=True)
    
    async def _record_failure(self, error_message: str) -> None:
        """Record failed operation"""
        self.total_failures += 1
        self.call_history.append({"success": False, "error": error_message, "timestamp": datetime.now()})
        
        if self.state == CircuitState.CLOSED:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.failure_count >= self.config.failure_threshold:
                await self._transition_to_open()
        
        elif self.state == CircuitState.HALF_OPEN:
            await self._transition_to_open()
        
        # Update quantum probabilities based on failure
        self._update_quantum_probabilities(success=False)
    
    async def _transition_to_open(self) -> None:
        """Transition circuit breaker to OPEN state"""
        old_state = self.state
        self.state = CircuitState.OPEN
        self.last_failure_time = datetime.now()
        self.state_transitions.append({
            "from": old_state.value,
            "to": self.state.value,
            "timestamp": datetime.now(),
            "reason": f"failure_threshold_exceeded({self.failure_count})"
        })
        
        # Notify entangled circuit breakers
        await self._notify_entangled_breakers("state_change", {"new_state": self.state.value})
        
        logger.warning(f"Circuit breaker {self.name} transitioned to OPEN")
    
    async def _transition_to_half_open(self) -> None:
        """Transition circuit breaker to HALF_OPEN state"""
        old_state = self.state
        self.state = CircuitState.HALF_OPEN
        self.half_open_call_count = 0
        self.success_count = 0
        
        self.state_transitions.append({
            "from": old_state.value,
            "to": self.state.value,
            "timestamp": datetime.now(),
            "reason": "timeout_expired"
        })
        
        logger.info(f"Circuit breaker {self.name} transitioned to HALF_OPEN")
    
    async def _transition_to_closed(self) -> None:
        """Transition circuit breaker to CLOSED state"""
        old_state = self.state
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_call_count = 0
        
        self.state_transitions.append({
            "from": old_state.value,
            "to": self.state.value,
            "timestamp": datetime.now(),
            "reason": f"success_threshold_met({self.success_count})"
        })
        
        logger.info(f"Circuit breaker {self.name} transitioned to CLOSED")
    
    def _update_quantum_probabilities(self, success: bool) -> None:
        """Update quantum state probabilities based on results"""
        if success:
            # Increase probability of closed state
            self.quantum_state_probability[0] = min(0.9, self.quantum_state_probability[0] + 0.05)
            self.quantum_state_probability[2] = max(0.05, self.quantum_state_probability[2] - 0.03)
        else:
            # Increase probability of open state
            self.quantum_state_probability[2] = min(0.8, self.quantum_state_probability[2] + 0.1)
            self.quantum_state_probability[0] = max(0.1, self.quantum_state_probability[0] - 0.05)
        
        # Normalize probabilities
        total = sum(self.quantum_state_probability)
        self.quantum_state_probability = [p / total for p in self.quantum_state_probability]
    
    async def _notify_entangled_breakers(self, event_type: str, data: Dict[str, Any]) -> None:
        """Notify entangled circuit breakers of events"""
        for partner in self.entanglement_partners:
            try:
                await partner.handle_entanglement_event(self, event_type, data)
            except Exception as e:
                logger.warning(f"Failed to notify entangled breaker {partner.name}: {e}")
    
    async def handle_entanglement_event(self, source: 'QuantumCircuitBreaker', event_type: str, data: Dict[str, Any]) -> None:
        """Handle events from entangled circuit breakers"""
        if event_type == "state_change" and data.get("new_state") == "open":
            # If entangled breaker opens, slightly increase our failure probability
            self.quantum_state_probability[2] = min(0.6, self.quantum_state_probability[2] + 0.1)
        
        logger.debug(f"Circuit breaker {self.name} received entanglement event from {source.name}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics"""
        recent_calls = [call for call in self.call_history if 
                       datetime.now() - call["timestamp"] < timedelta(minutes=5)]
        
        success_rate = 0.0
        if recent_calls:
            successes = sum(1 for call in recent_calls if call["success"])
            success_rate = successes / len(recent_calls)
        
        return {
            "name": self.name,
            "state": self.state.value,
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "success_rate": success_rate,
            "recent_calls": len(recent_calls),
            "state_transitions": len(self.state_transitions),
            "quantum_probabilities": {
                "closed": self.quantum_state_probability[0],
                "half_open": self.quantum_state_probability[1], 
                "open": self.quantum_state_probability[2]
            },
            "entanglement_partners": len(self.entanglement_partners)
        }


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open"""
    pass


class AdaptiveRetryManager:
    """Intelligent retry manager with adaptive strategies"""
    
    def __init__(self):
        self.retry_strategies = {}
        self.failure_patterns = defaultdict(list)
        self.success_rates = defaultdict(float)
        self.adaptive_parameters = {}
    
    async def execute_with_retry(
        self,
        operation: Callable[..., Any],
        failure_types: List[FailureType],
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: bool = True,
        *args,
        **kwargs
    ) -> Any:
        """Execute operation with intelligent retry logic"""
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    # Calculate delay with adaptive strategy
                    delay = await self._calculate_retry_delay(
                        operation.__name__, attempt, base_delay, max_delay, jitter, last_exception
                    )
                    await asyncio.sleep(delay)
                
                result = await operation(*args, **kwargs)
                
                # Record success for adaptive learning
                await self._record_success(operation.__name__, attempt)
                return result
                
            except Exception as e:
                last_exception = e
                failure_type = self._classify_failure(e)
                
                # Record failure for adaptive learning  
                await self._record_failure(operation.__name__, attempt, failure_type, str(e))
                
                if attempt == max_retries or failure_type not in failure_types:
                    # No more retries or unrecoverable error
                    break
                
                logger.warning(f"Retry {attempt + 1}/{max_retries} for {operation.__name__}: {e}")
        
        # All retries exhausted
        raise last_exception
    
    def _classify_failure(self, exception: Exception) -> FailureType:
        """Classify exception into failure type"""
        error_message = str(exception).lower()
        exception_type = type(exception).__name__.lower()
        
        if "timeout" in error_message or "timeout" in exception_type:
            return FailureType.TIMEOUT
        elif "connection" in error_message or "connection" in exception_type:
            return FailureType.CONNECTION_ERROR
        elif "auth" in error_message or "unauthorized" in error_message:
            return FailureType.AUTHENTICATION_ERROR
        elif "rate" in error_message and "limit" in error_message:
            return FailureType.RATE_LIMIT
        elif "unavailable" in error_message or "503" in error_message:
            return FailureType.SERVICE_UNAVAILABLE
        elif "corrupt" in error_message or "invalid" in error_message:
            return FailureType.DATA_CORRUPTION
        elif "memory" in error_message or "resource" in error_message:
            return FailureType.RESOURCE_EXHAUSTION
        elif isinstance(exception, (ValueError, TypeError, AttributeError)):
            return FailureType.LOGIC_ERROR
        else:
            return FailureType.UNKNOWN
    
    async def _calculate_retry_delay(
        self,
        operation_name: str,
        attempt: int,
        base_delay: float,
        max_delay: float,
        jitter: bool,
        last_exception: Exception
    ) -> float:
        """Calculate adaptive retry delay"""
        
        failure_type = self._classify_failure(last_exception)
        
        # Base exponential backoff
        delay = min(base_delay * (2 ** attempt), max_delay)
        
        # Adaptive adjustments based on failure type
        if failure_type == FailureType.RATE_LIMIT:
            delay *= 2.0  # Longer delays for rate limits
        elif failure_type == FailureType.CONNECTION_ERROR:
            delay *= 1.5  # Moderate delays for connection errors
        elif failure_type == FailureType.TIMEOUT:
            delay *= 0.8  # Shorter delays for timeouts
        
        # Historical success rate adjustment
        success_rate = self.success_rates.get(operation_name, 0.5)
        if success_rate < 0.3:
            delay *= 1.5  # Longer delays for historically unreliable operations
        elif success_rate > 0.8:
            delay *= 0.7  # Shorter delays for reliable operations
        
        # Add jitter to prevent thundering herd
        if jitter:
            jitter_amount = delay * 0.1 * random.uniform(0, 1)
            delay += jitter_amount
        
        return delay
    
    async def _record_success(self, operation_name: str, attempt: int) -> None:
        """Record successful operation for adaptive learning"""
        if operation_name not in self.success_rates:
            self.success_rates[operation_name] = 0.5
        
        # Update success rate with exponential moving average
        self.success_rates[operation_name] = (
            self.success_rates[operation_name] * 0.9 + 1.0 * 0.1
        )
    
    async def _record_failure(
        self, 
        operation_name: str, 
        attempt: int, 
        failure_type: FailureType, 
        error_message: str
    ) -> None:
        """Record failed operation for adaptive learning"""
        
        self.failure_patterns[operation_name].append({
            "attempt": attempt,
            "failure_type": failure_type.value,
            "error": error_message,
            "timestamp": datetime.now()
        })
        
        # Update success rate
        if operation_name not in self.success_rates:
            self.success_rates[operation_name] = 0.5
        
        self.success_rates[operation_name] = (
            self.success_rates[operation_name] * 0.9 + 0.0 * 0.1
        )


class SelfHealingSystem:
    """Self-healing system with autonomous recovery capabilities"""
    
    def __init__(self):
        self.healing_strategies = {}
        self.health_monitors = {}
        self.recovery_history = []
        self.healing_enabled = True
        self.quantum_healing_factor = 0.1
    
    async def register_healing_strategy(
        self,
        component: str,
        failure_type: FailureType,
        strategy: Callable[..., Any]
    ) -> None:
        """Register a healing strategy for a component and failure type"""
        key = f"{component}:{failure_type.value}"
        self.healing_strategies[key] = strategy
        logger.info(f"Registered healing strategy for {key}")
    
    async def attempt_healing(self, failure_context: FailureContext) -> RecoveryResult:
        """Attempt to heal a failure automatically"""
        if not self.healing_enabled:
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.FALLBACK_OPERATION,
                recovery_time_ms=0,
                error_resolved=False,
                fallback_used=True
            )
        
        start_time = time.time()
        
        # Find appropriate healing strategy
        strategy_key = f"{failure_context.component}:{failure_context.failure_type.value}"
        generic_key = f"*:{failure_context.failure_type.value}"
        
        healing_strategy = None
        if strategy_key in self.healing_strategies:
            healing_strategy = self.healing_strategies[strategy_key]
        elif generic_key in self.healing_strategies:
            healing_strategy = self.healing_strategies[generic_key]
        
        if not healing_strategy:
            return await self._fallback_healing(failure_context)
        
        try:
            # Attempt healing
            logger.info(f"Attempting healing for {failure_context.component}:{failure_context.failure_type.value}")
            
            result = await healing_strategy(failure_context)
            recovery_time = (time.time() - start_time) * 1000
            
            recovery_result = RecoveryResult(
                success=True,
                strategy_used=RecoveryStrategy.QUANTUM_RECOVERY,
                recovery_time_ms=recovery_time,
                error_resolved=True,
                fallback_used=False,
                additional_info=result if isinstance(result, dict) else {}
            )
            
            # Record successful healing
            self.recovery_history.append({
                "failure_context": failure_context,
                "recovery_result": recovery_result,
                "timestamp": datetime.now()
            })
            
            logger.info(f"Healing successful for {failure_context.component} in {recovery_time:.2f}ms")
            return recovery_result
            
        except Exception as e:
            logger.error(f"Healing strategy failed: {e}")
            return await self._fallback_healing(failure_context)
    
    async def _fallback_healing(self, failure_context: FailureContext) -> RecoveryResult:
        """Fallback healing when no specific strategy exists"""
        
        # Quantum-inspired healing attempt
        if random.random() < self.quantum_healing_factor:
            # Simulate quantum healing effect
            await asyncio.sleep(0.1)  # Brief pause
            
            return RecoveryResult(
                success=True,
                strategy_used=RecoveryStrategy.QUANTUM_RECOVERY,
                recovery_time_ms=100,
                error_resolved=True,
                fallback_used=True,
                additional_info={"quantum_healing": True}
            )
        
        return RecoveryResult(
            success=False,
            strategy_used=RecoveryStrategy.FALLBACK_OPERATION,
            recovery_time_ms=0,
            error_resolved=False,
            fallback_used=True
        )
    
    def get_healing_metrics(self) -> Dict[str, Any]:
        """Get self-healing system metrics"""
        recent_recoveries = [r for r in self.recovery_history 
                           if datetime.now() - r["timestamp"] < timedelta(hours=24)]
        
        success_rate = 0.0
        if recent_recoveries:
            successes = sum(1 for r in recent_recoveries if r["recovery_result"].success)
            success_rate = successes / len(recent_recoveries)
        
        return {
            "healing_enabled": self.healing_enabled,
            "registered_strategies": len(self.healing_strategies),
            "total_recoveries": len(self.recovery_history),
            "recent_recoveries": len(recent_recoveries),
            "success_rate": success_rate,
            "quantum_healing_factor": self.quantum_healing_factor
        }


class EnterpriseResilienceFramework:
    """Main enterprise resilience framework"""
    
    def __init__(self, resilience_level: ResilienceLevel = ResilienceLevel.ENTERPRISE):
        self.framework_id = str(uuid.uuid4())[:8]
        self.resilience_level = resilience_level
        
        # Core components
        self.circuit_breakers = {}
        self.retry_manager = AdaptiveRetryManager()
        self.healing_system = SelfHealingSystem()
        
        # Configuration
        self.global_timeout = 300  # seconds
        self.max_concurrent_failures = 10
        self.failure_cascade_threshold = 0.7
        
        # Monitoring
        self.failure_events = deque(maxlen=10000)
        self.recovery_events = deque(maxlen=10000)
        self.system_health_score = 1.0
        self.resilience_metrics = {}
        
        logger.info(f"Enterprise Resilience Framework initialized [ID: {self.framework_id}] - Level: {resilience_level.value}")
    
    def create_circuit_breaker(
        self, 
        name: str, 
        config: CircuitBreakerConfig = None
    ) -> QuantumCircuitBreaker:
        """Create a new quantum circuit breaker"""
        
        if config is None:
            config = self._get_default_circuit_config()
        
        circuit_breaker = QuantumCircuitBreaker(name, config)
        self.circuit_breakers[name] = circuit_breaker
        
        logger.info(f"Created circuit breaker: {name}")
        return circuit_breaker
    
    def _get_default_circuit_config(self) -> CircuitBreakerConfig:
        """Get default circuit breaker configuration based on resilience level"""
        
        if self.resilience_level == ResilienceLevel.BASIC:
            return CircuitBreakerConfig(
                failure_threshold=3,
                success_threshold=2,
                timeout_seconds=30
            )
        elif self.resilience_level == ResilienceLevel.ENHANCED:
            return CircuitBreakerConfig(
                failure_threshold=5,
                success_threshold=3,
                timeout_seconds=60
            )
        elif self.resilience_level == ResilienceLevel.ENTERPRISE:
            return CircuitBreakerConfig(
                failure_threshold=5,
                success_threshold=3,
                timeout_seconds=60,
                half_open_max_calls=10
            )
        else:  # QUANTUM
            return CircuitBreakerConfig(
                failure_threshold=7,
                success_threshold=5,
                timeout_seconds=90,
                half_open_max_calls=15
            )
    
    async def execute_resilient_operation(
        self,
        operation: Callable[..., Any],
        operation_name: str,
        circuit_breaker_name: str = None,
        retry_config: Dict[str, Any] = None,
        enable_healing: bool = True,
        *args,
        **kwargs
    ) -> Any:
        """Execute an operation with full resilience features"""
        
        start_time = time.time()
        failure_context = None
        
        try:
            # Set up circuit breaker if specified
            if circuit_breaker_name:
                if circuit_breaker_name not in self.circuit_breakers:
                    self.create_circuit_breaker(circuit_breaker_name)
                
                circuit_breaker = self.circuit_breakers[circuit_breaker_name]
                
                # Execute through circuit breaker and retry manager
                result = await circuit_breaker.call(
                    self._execute_with_retry_wrapper,
                    operation,
                    operation_name,
                    retry_config or {},
                    *args,
                    **kwargs
                )
            else:
                # Execute with retry only
                result = await self._execute_with_retry_wrapper(
                    operation,
                    operation_name,
                    retry_config or {},
                    *args,
                    **kwargs
                )
            
            execution_time = (time.time() - start_time) * 1000
            
            # Record success
            await self._record_success(operation_name, execution_time)
            
            return result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            # Create failure context
            failure_context = FailureContext(
                failure_id=str(uuid.uuid4())[:8],
                failure_type=self._classify_exception(e),
                error_message=str(e),
                stack_trace=traceback.format_exc(),
                timestamp=datetime.now(),
                component=circuit_breaker_name or "unknown",
                operation=operation_name,
                input_data={"args": str(args)[:200], "kwargs": str(kwargs)[:200]}
            )
            
            # Record failure
            await self._record_failure(failure_context, execution_time)
            
            # Attempt healing if enabled
            if enable_healing:
                recovery_result = await self.healing_system.attempt_healing(failure_context)
                if recovery_result.success and recovery_result.error_resolved:
                    # Healing successful, retry operation once
                    try:
                        result = await operation(*args, **kwargs)
                        await self._record_recovery(failure_context, recovery_result)
                        return result
                    except Exception:
                        pass  # Healing didn't work, continue with original error
            
            # Update system health
            await self._update_system_health()
            
            raise
    
    async def _execute_with_retry_wrapper(
        self,
        operation: Callable[..., Any],
        operation_name: str,
        retry_config: Dict[str, Any],
        *args,
        **kwargs
    ) -> Any:
        """Wrapper to execute operation with retry logic"""
        
        return await self.retry_manager.execute_with_retry(
            operation,
            failure_types=[FailureType.TIMEOUT, FailureType.CONNECTION_ERROR, 
                          FailureType.SERVICE_UNAVAILABLE, FailureType.RATE_LIMIT],
            max_retries=retry_config.get("max_retries", 3),
            base_delay=retry_config.get("base_delay", 1.0),
            max_delay=retry_config.get("max_delay", 60.0),
            *args,
            **kwargs
        )
    
    def _classify_exception(self, exception: Exception) -> FailureType:
        """Classify exception into failure type"""
        error_msg = str(exception).lower()
        exc_type = type(exception).__name__.lower()
        
        classification_rules = {
            FailureType.TIMEOUT: ["timeout", "time out", "deadline", "expired"],
            FailureType.CONNECTION_ERROR: ["connection", "network", "socket", "refused", "unreachable"],
            FailureType.AUTHENTICATION_ERROR: ["auth", "unauthorized", "forbidden", "credential", "permission"],
            FailureType.RATE_LIMIT: ["rate limit", "throttle", "quota", "too many requests"],
            FailureType.SERVICE_UNAVAILABLE: ["unavailable", "503", "502", "504", "maintenance"],
            FailureType.DATA_CORRUPTION: ["corrupt", "invalid", "malformed", "parse error"],
            FailureType.RESOURCE_EXHAUSTION: ["memory", "disk", "cpu", "resource", "capacity", "full"],
            FailureType.CONFIGURATION_ERROR: ["config", "setting", "parameter", "environment"]
        }
        
        for failure_type, keywords in classification_rules.items():
            if any(keyword in error_msg or keyword in exc_type for keyword in keywords):
                return failure_type
        
        # Check exception types
        if isinstance(exception, (ValueError, TypeError, AttributeError)):
            return FailureType.LOGIC_ERROR
        
        return FailureType.UNKNOWN
    
    async def _record_success(self, operation_name: str, execution_time: float) -> None:
        """Record successful operation"""
        # Update system health positively
        self.system_health_score = min(1.0, self.system_health_score + 0.001)
    
    async def _record_failure(self, failure_context: FailureContext, execution_time: float) -> None:
        """Record failed operation"""
        self.failure_events.append({
            "failure_context": failure_context,
            "execution_time": execution_time,
            "timestamp": datetime.now()
        })
        
        # Update system health negatively
        self.system_health_score = max(0.0, self.system_health_score - 0.01)
        
        logger.warning(f"Operation failure recorded: {failure_context.operation} - {failure_context.error_message}")
    
    async def _record_recovery(self, failure_context: FailureContext, recovery_result: RecoveryResult) -> None:
        """Record successful recovery"""
        self.recovery_events.append({
            "failure_context": failure_context,
            "recovery_result": recovery_result,
            "timestamp": datetime.now()
        })
        
        # Update system health positively for successful recovery
        self.system_health_score = min(1.0, self.system_health_score + 0.005)
        
        logger.info(f"Recovery recorded: {failure_context.operation} - {recovery_result.strategy_used.value}")
    
    async def _update_system_health(self) -> None:
        """Update overall system health score"""
        
        # Calculate recent failure rate
        recent_failures = [f for f in self.failure_events 
                          if datetime.now() - f["timestamp"] < timedelta(minutes=10)]
        
        if len(recent_failures) > 5:
            self.system_health_score = max(0.1, self.system_health_score - 0.05)
        
        # Gradual recovery if no recent failures
        if len(recent_failures) == 0:
            self.system_health_score = min(1.0, self.system_health_score + 0.002)
    
    async def setup_healing_strategies(self) -> None:
        """Setup default healing strategies"""
        
        # Strategy for connection errors
        async def heal_connection_error(failure_context: FailureContext) -> Dict[str, Any]:
            await asyncio.sleep(1)  # Brief pause
            return {"strategy": "connection_reset", "result": "success"}
        
        # Strategy for resource exhaustion
        async def heal_resource_exhaustion(failure_context: FailureContext) -> Dict[str, Any]:
            await asyncio.sleep(2)  # Simulate cleanup
            return {"strategy": "resource_cleanup", "result": "success"}
        
        # Strategy for service unavailable
        async def heal_service_unavailable(failure_context: FailureContext) -> Dict[str, Any]:
            await asyncio.sleep(5)  # Wait for service recovery
            return {"strategy": "service_wait", "result": "success"}
        
        # Register strategies
        await self.healing_system.register_healing_strategy(
            "*", FailureType.CONNECTION_ERROR, heal_connection_error
        )
        await self.healing_system.register_healing_strategy(
            "*", FailureType.RESOURCE_EXHAUSTION, heal_resource_exhaustion
        )
        await self.healing_system.register_healing_strategy(
            "*", FailureType.SERVICE_UNAVAILABLE, heal_service_unavailable
        )
        
        logger.info("Default healing strategies registered")
    
    async def entangle_circuit_breakers(self, breaker1_name: str, breaker2_name: str) -> None:
        """Create quantum entanglement between circuit breakers"""
        if breaker1_name in self.circuit_breakers and breaker2_name in self.circuit_breakers:
            breaker1 = self.circuit_breakers[breaker1_name]
            breaker2 = self.circuit_breakers[breaker2_name]
            
            breaker1.entanglement_partners.append(breaker2)
            breaker2.entanglement_partners.append(breaker1)
            
            logger.info(f"Entangled circuit breakers: {breaker1_name} <-> {breaker2_name}")
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive resilience metrics"""
        
        # Circuit breaker metrics
        circuit_metrics = {}
        for name, breaker in self.circuit_breakers.items():
            circuit_metrics[name] = breaker.get_metrics()
        
        # Recent events
        recent_failures = [f for f in self.failure_events 
                          if datetime.now() - f["timestamp"] < timedelta(hours=1)]
        recent_recoveries = [r for r in self.recovery_events 
                            if datetime.now() - r["timestamp"] < timedelta(hours=1)]
        
        # Calculate availability
        total_recent_events = len(recent_failures) + len(recent_recoveries)
        availability = 1.0 if total_recent_events == 0 else len(recent_recoveries) / total_recent_events
        
        return {
            "framework_id": self.framework_id,
            "resilience_level": self.resilience_level.value,
            "system_health_score": self.system_health_score,
            "availability": availability,
            "circuit_breakers": circuit_metrics,
            "recent_metrics": {
                "failures_last_hour": len(recent_failures),
                "recoveries_last_hour": len(recent_recoveries),
                "total_failure_events": len(self.failure_events),
                "total_recovery_events": len(self.recovery_events)
            },
            "healing_system": self.healing_system.get_healing_metrics()
        }


# Global resilience framework instance
_resilience_framework: Optional[EnterpriseResilienceFramework] = None


def get_resilience_framework() -> EnterpriseResilienceFramework:
    """Get or create global resilience framework"""
    global _resilience_framework
    if _resilience_framework is None:
        _resilience_framework = EnterpriseResilienceFramework()
    return _resilience_framework


def resilient(
    circuit_breaker: str = None,
    max_retries: int = 3,
    base_delay: float = 1.0,
    enable_healing: bool = True
):
    """Decorator to make functions resilient"""
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            framework = get_resilience_framework()
            
            retry_config = {
                "max_retries": max_retries,
                "base_delay": base_delay
            }
            
            return await framework.execute_resilient_operation(
                func,
                func.__name__,
                circuit_breaker,
                retry_config,
                enable_healing,
                *args,
                **kwargs
            )
        
        return wrapper
    return decorator


async def demo_resilience_framework():
    """Demonstrate enterprise resilience framework"""
    print("🛡️ Enterprise Resilience Framework Demo")
    print("=" * 60)
    
    framework = get_resilience_framework()
    await framework.setup_healing_strategies()
    
    # Create circuit breakers
    api_breaker = framework.create_circuit_breaker("api_service")
    db_breaker = framework.create_circuit_breaker("database") 
    
    # Entangle circuit breakers
    await framework.entangle_circuit_breakers("api_service", "database")
    
    # Demo operations
    async def reliable_operation():
        await asyncio.sleep(0.1)
        return "success"
    
    async def unreliable_operation():
        if random.random() < 0.7:  # 70% failure rate
            raise Exception("Service temporarily unavailable")
        return "success"
    
    # Demo 1: Resilient execution
    print("\n1. Resilient Operation Execution:")
    try:
        result = await framework.execute_resilient_operation(
            reliable_operation,
            "reliable_test",
            "api_service"
        )
        print(f"   Reliable operation result: {result}")
    except Exception as e:
        print(f"   Reliable operation failed: {e}")
    
    # Demo 2: Circuit breaker with failures
    print("\n2. Circuit Breaker with Failures:")
    for i in range(8):
        try:
            result = await framework.execute_resilient_operation(
                unreliable_operation,
                "unreliable_test",
                "api_service"
            )
            print(f"   Attempt {i+1}: Success")
        except Exception as e:
            print(f"   Attempt {i+1}: Failed - {str(e)[:50]}...")
    
    # Demo 3: Using decorator
    print("\n3. Resilient Decorator:")
    
    @resilient(circuit_breaker="database", max_retries=2)
    async def database_operation():
        if random.random() < 0.5:
            raise Exception("Database connection timeout")
        return "Data retrieved successfully"
    
    for i in range(3):
        try:
            result = await database_operation()
            print(f"   Database operation {i+1}: {result}")
        except Exception as e:
            print(f"   Database operation {i+1}: Failed")
    
    # Demo 4: System metrics
    print(f"\n4. System Resilience Metrics:")
    metrics = framework.get_comprehensive_metrics()
    print(f"   System Health Score: {metrics['system_health_score']:.3f}")
    print(f"   Availability: {metrics['availability']:.3f}")
    print(f"   Recent Failures: {metrics['recent_metrics']['failures_last_hour']}")
    print(f"   Recent Recoveries: {metrics['recent_metrics']['recoveries_last_hour']}")
    
    # Circuit breaker details
    print(f"\n   Circuit Breaker States:")
    for name, cb_metrics in metrics['circuit_breakers'].items():
        print(f"   - {name}: {cb_metrics['state']} "
              f"(success_rate: {cb_metrics['success_rate']:.2%})")
    
    return metrics


if __name__ == "__main__":
    asyncio.run(demo_resilience_framework())