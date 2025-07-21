"""
Comprehensive tests for Circuit Breaker implementation.

Tests critical fault tolerance infrastructure for external service reliability,
following TDD principles with extensive coverage of state transitions and edge cases.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from typing import Any

# Test imports
from pipeline.infrastructure.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerState,
    CircuitBreakerConfig,
    CircuitBreakerMetrics,
    CircuitBreakerError,
    CircuitOpenError,
    CircuitTimeoutError
)


class TestCircuitBreakerConfig:
    """Test CircuitBreakerConfig validation and behavior."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        
        assert config.failure_threshold == 5
        assert config.recovery_timeout == 60.0
        assert config.success_threshold == 3
        assert config.timeout == 30.0
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            recovery_timeout=120.0,
            success_threshold=5,
            timeout=45.0
        )
        
        assert config.failure_threshold == 10
        assert config.recovery_timeout == 120.0
        assert config.success_threshold == 5
        assert config.timeout == 45.0
    
    def test_invalid_failure_threshold(self):
        """Test validation of failure_threshold."""
        with pytest.raises(ValueError, match="failure_threshold must be >= 1"):
            CircuitBreakerConfig(failure_threshold=0)
        
        with pytest.raises(ValueError, match="failure_threshold must be >= 1"):
            CircuitBreakerConfig(failure_threshold=-1)
    
    def test_invalid_recovery_timeout(self):
        """Test validation of recovery_timeout."""
        with pytest.raises(ValueError, match="recovery_timeout must be >= 0"):
            CircuitBreakerConfig(recovery_timeout=-1)
    
    def test_invalid_success_threshold(self):
        """Test validation of success_threshold."""
        with pytest.raises(ValueError, match="success_threshold must be >= 1"):
            CircuitBreakerConfig(success_threshold=0)
        
        with pytest.raises(ValueError, match="success_threshold must be >= 1"):
            CircuitBreakerConfig(success_threshold=-1)
    
    def test_invalid_timeout(self):
        """Test validation of timeout."""
        with pytest.raises(ValueError, match="timeout must be > 0"):
            CircuitBreakerConfig(timeout=0)
        
        with pytest.raises(ValueError, match="timeout must be > 0"):
            CircuitBreakerConfig(timeout=-1)


class TestCircuitBreakerMetrics:
    """Test CircuitBreakerMetrics functionality."""
    
    def test_initial_state(self):
        """Test initial metrics state."""
        metrics = CircuitBreakerMetrics()
        
        assert metrics.failure_count == 0
        assert metrics.success_count == 0
        assert metrics.last_failure_time is None
        assert metrics.total_requests == 0
        assert metrics.total_failures == 0
        assert metrics.total_successes == 0
    
    def test_record_success(self):
        """Test recording successful operations."""
        metrics = CircuitBreakerMetrics()
        
        metrics.record_success()
        
        assert metrics.success_count == 1
        assert metrics.total_successes == 1
        assert metrics.total_requests == 1
        assert metrics.failure_count == 0
        assert metrics.total_failures == 0
        
        # Record another success
        metrics.record_success()
        
        assert metrics.success_count == 2
        assert metrics.total_successes == 2
        assert metrics.total_requests == 2
    
    def test_record_failure(self):
        """Test recording failed operations."""
        metrics = CircuitBreakerMetrics()
        
        before_time = time.time()
        metrics.record_failure()
        after_time = time.time()
        
        assert metrics.failure_count == 1
        assert metrics.total_failures == 1
        assert metrics.total_requests == 1
        assert metrics.success_count == 0
        assert metrics.total_successes == 0
        assert before_time <= metrics.last_failure_time <= after_time
        
        # Record another failure
        metrics.record_failure()
        
        assert metrics.failure_count == 2
        assert metrics.total_failures == 2
        assert metrics.total_requests == 2
    
    def test_reset(self):
        """Test resetting metrics."""
        metrics = CircuitBreakerMetrics()
        
        # Record some activity
        metrics.record_success()
        metrics.record_failure()
        
        assert metrics.success_count == 1
        assert metrics.failure_count == 1
        
        # Reset counters
        metrics.reset()
        
        assert metrics.success_count == 0
        assert metrics.failure_count == 0
        # Total counters should not be reset
        assert metrics.total_successes == 1
        assert metrics.total_failures == 1
        assert metrics.total_requests == 2
    
    def test_mixed_operations(self):
        """Test tracking mixed successful and failed operations."""
        metrics = CircuitBreakerMetrics()
        
        # Mix of operations
        metrics.record_success()
        metrics.record_failure()
        metrics.record_success()
        metrics.record_failure()
        metrics.record_failure()
        
        assert metrics.success_count == 2
        assert metrics.failure_count == 3
        assert metrics.total_successes == 2
        assert metrics.total_failures == 3
        assert metrics.total_requests == 5


class TestCircuitBreakerExceptions:
    """Test circuit breaker exception classes."""
    
    def test_circuit_breaker_error(self):
        """Test base CircuitBreakerError."""
        error = CircuitBreakerError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)
    
    def test_circuit_open_error(self):
        """Test CircuitOpenError."""
        # Default message
        error = CircuitOpenError()
        assert str(error) == "Circuit breaker is open"
        assert isinstance(error, CircuitBreakerError)
        
        # Custom message
        error = CircuitOpenError("Custom open message")
        assert str(error) == "Custom open message"
    
    def test_circuit_timeout_error(self):
        """Test CircuitTimeoutError."""
        # Default message
        error = CircuitTimeoutError()
        assert str(error) == "Operation timed out"
        assert isinstance(error, CircuitBreakerError)
        
        # Custom message
        error = CircuitTimeoutError("Custom timeout message")
        assert str(error) == "Custom timeout message"


class TestCircuitBreakerBasicFunctionality:
    """Test basic CircuitBreaker functionality."""
    
    @pytest.fixture
    def circuit_breaker(self):
        """Create a basic circuit breaker for testing."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=1.0,  # Short timeout for testing
            success_threshold=2,
            timeout=0.5
        )
        return CircuitBreaker("test_service", config)
    
    def test_initial_state(self, circuit_breaker):
        """Test circuit breaker initial state."""
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.name == "test_service"
        assert isinstance(circuit_breaker.config, CircuitBreakerConfig)
        assert isinstance(circuit_breaker.metrics, CircuitBreakerMetrics)
    
    @pytest.mark.asyncio
    async def test_successful_call_closed_state(self, circuit_breaker):
        """Test successful call in closed state."""
        async def successful_operation():
            return "success"
        
        result = await circuit_breaker.call(successful_operation)
        
        assert result == "success"
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.metrics.success_count == 1
        assert circuit_breaker.metrics.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_failed_call_closed_state(self, circuit_breaker):
        """Test failed call in closed state."""
        async def failing_operation():
            raise Exception("Service error")
        
        with pytest.raises(Exception, match="Service error"):
            await circuit_breaker.call(failing_operation)
        
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.metrics.failure_count == 1
        assert circuit_breaker.metrics.success_count == 0
    
    @pytest.mark.asyncio
    async def test_multiple_failures_trigger_open(self, circuit_breaker):
        """Test that multiple failures trigger open state."""
        async def failing_operation():
            raise Exception("Service error")
        
        # Fail below threshold - should stay closed
        for _ in range(2):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_operation)
            assert circuit_breaker.state == CircuitBreakerState.CLOSED
        
        # One more failure should open the circuit
        with pytest.raises(Exception):
            await circuit_breaker.call(failing_operation)
        
        assert circuit_breaker.state == CircuitBreakerState.OPEN
        assert circuit_breaker.metrics.failure_count == 3
    
    @pytest.mark.asyncio
    async def test_open_state_rejects_calls(self, circuit_breaker):
        """Test that open state rejects calls immediately."""
        # Force circuit to open state
        circuit_breaker._open_circuit()
        
        async def any_operation():
            return "should not execute"
        
        with pytest.raises(CircuitOpenError):
            await circuit_breaker.call(any_operation)
        
        assert circuit_breaker.state == CircuitBreakerState.OPEN
    
    @pytest.mark.asyncio
    async def test_half_open_transition(self, circuit_breaker):
        """Test transition from open to half-open state."""
        # Force circuit to open
        circuit_breaker._open_circuit()
        
        # Wait for recovery timeout
        await asyncio.sleep(1.1)  # Slightly longer than recovery_timeout
        
        async def test_operation():
            return "test"
        
        # Next call should transition to half-open
        result = await circuit_breaker.call(test_operation)
        
        assert result == "test"
        assert circuit_breaker.state == CircuitBreakerState.HALF_OPEN


class TestCircuitBreakerStateTransitions:
    """Test circuit breaker state transition logic."""
    
    @pytest.fixture
    def circuit_breaker(self):
        """Create a circuit breaker with test-friendly configuration."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.1,
            success_threshold=2,
            timeout=0.5
        )
        return CircuitBreaker("test_transitions", config)
    
    @pytest.mark.asyncio
    async def test_closed_to_open_transition(self, circuit_breaker):
        """Test transition from closed to open state."""
        async def failing_operation():
            raise Exception("Failure")
        
        # Initial state
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        
        # Fail enough times to trigger open
        for _ in range(2):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_operation)
        
        assert circuit_breaker.state == CircuitBreakerState.OPEN
    
    @pytest.mark.asyncio
    async def test_open_to_half_open_transition(self, circuit_breaker):
        """Test transition from open to half-open state."""
        # Force to open state
        circuit_breaker._open_circuit()
        assert circuit_breaker.state == CircuitBreakerState.OPEN
        
        # Wait for recovery timeout
        await asyncio.sleep(0.2)
        
        async def test_operation():
            return "recovery test"
        
        result = await circuit_breaker.call(test_operation)
        
        assert result == "recovery test"
        assert circuit_breaker.state == CircuitBreakerState.HALF_OPEN
    
    @pytest.mark.asyncio
    async def test_half_open_to_closed_transition(self, circuit_breaker):
        """Test transition from half-open to closed state."""
        # Force to half-open state
        circuit_breaker._half_open_circuit()
        
        async def successful_operation():
            return "success"
        
        # Need success_threshold successes to close
        for i in range(2):
            result = await circuit_breaker.call(successful_operation)
            assert result == "success"
        
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
    
    @pytest.mark.asyncio
    async def test_half_open_to_open_on_failure(self, circuit_breaker):
        """Test transition from half-open back to open on failure."""
        # Force to half-open state
        circuit_breaker._half_open_circuit()
        
        async def failing_operation():
            raise Exception("Still failing")
        
        with pytest.raises(Exception):
            await circuit_breaker.call(failing_operation)
        
        assert circuit_breaker.state == CircuitBreakerState.OPEN


class TestCircuitBreakerTimeout:
    """Test circuit breaker timeout functionality."""
    
    @pytest.fixture
    def circuit_breaker(self):
        """Create a circuit breaker with short timeout for testing."""
        config = CircuitBreakerConfig(timeout=0.1)  # Very short timeout
        return CircuitBreaker("test_timeout", config)
    
    @pytest.mark.asyncio
    async def test_operation_timeout(self, circuit_breaker):
        """Test that slow operations timeout correctly."""
        async def slow_operation():
            await asyncio.sleep(0.5)  # Longer than timeout
            return "should not return"
        
        with pytest.raises(CircuitTimeoutError):
            await circuit_breaker.call(slow_operation)
        
        # Timeout should count as failure
        assert circuit_breaker.metrics.failure_count == 1
    
    @pytest.mark.asyncio
    async def test_fast_operation_no_timeout(self, circuit_breaker):
        """Test that fast operations complete without timeout."""
        async def fast_operation():
            await asyncio.sleep(0.01)  # Much shorter than timeout
            return "success"
        
        result = await circuit_breaker.call(fast_operation)
        
        assert result == "success"
        assert circuit_breaker.metrics.success_count == 1


class TestCircuitBreakerDecorator:
    """Test circuit breaker decorator functionality."""
    
    @pytest.fixture
    def circuit_breaker(self):
        """Create a circuit breaker for decorator testing."""
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0.1)
        return CircuitBreaker("test_decorator", config)
    
    @pytest.mark.asyncio
    async def test_decorator_usage(self, circuit_breaker):
        """Test using circuit breaker as decorator."""
        
        @circuit_breaker
        async def decorated_function(value):
            if value == "fail":
                raise Exception("Decorator test failure")
            return f"decorated_{value}"
        
        # Test successful call
        result = await decorated_function("success")
        assert result == "decorated_success"
        
        # Test failed call
        with pytest.raises(Exception, match="Decorator test failure"):
            await decorated_function("fail")
        
        assert circuit_breaker.metrics.success_count == 1
        assert circuit_breaker.metrics.failure_count == 1


class TestCircuitBreakerEdgeCases:
    """Test circuit breaker edge cases and error conditions."""
    
    @pytest.fixture
    def circuit_breaker(self):
        """Create a circuit breaker for edge case testing."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=0.1,
            success_threshold=1
        )
        return CircuitBreaker("test_edge_cases", config)
    
    @pytest.mark.asyncio
    async def test_none_operation(self, circuit_breaker):
        """Test calling circuit breaker with None operation."""
        with pytest.raises(TypeError):
            await circuit_breaker.call(None)
    
    @pytest.mark.asyncio
    async def test_non_async_function(self, circuit_breaker):
        """Test calling circuit breaker with non-async function."""
        def sync_function():
            return "sync result"
        
        # Should handle sync functions by wrapping them
        # Implementation dependent - may raise TypeError or handle gracefully
        try:
            result = await circuit_breaker.call(sync_function)
            # If it works, verify the result
            assert result == "sync result"
        except TypeError:
            # Expected behavior for some implementations
            pass
    
    @pytest.mark.asyncio
    async def test_rapid_state_changes(self, circuit_breaker):
        """Test rapid state changes under load."""
        async def flaky_operation(should_fail=False):
            if should_fail:
                raise Exception("Flaky failure")
            return "success"
        
        # Cause rapid open
        with pytest.raises(Exception):
            await circuit_breaker.call(lambda: flaky_operation(True))
        
        assert circuit_breaker.state == CircuitBreakerState.OPEN
        
        # Wait for recovery
        await asyncio.sleep(0.2)
        
        # Should transition to half-open on next call
        result = await circuit_breaker.call(lambda: flaky_operation(False))
        assert result == "success"
        assert circuit_breaker.state == CircuitBreakerState.HALF_OPEN
    
    def test_concurrent_access(self, circuit_breaker):
        """Test circuit breaker behavior under concurrent access."""
        # This would require more complex async testing
        # Testing that state transitions are thread-safe
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        
        # Basic state safety check
        original_state = circuit_breaker.state
        circuit_breaker._open_circuit()
        assert circuit_breaker.state != original_state
        
        circuit_breaker._close_circuit()
        assert circuit_breaker.state == CircuitBreakerState.CLOSED


class TestCircuitBreakerMetricsAndMonitoring:
    """Test circuit breaker metrics and monitoring capabilities."""
    
    @pytest.fixture
    def circuit_breaker(self):
        """Create a circuit breaker for metrics testing."""
        config = CircuitBreakerConfig(failure_threshold=3)
        return CircuitBreaker("test_metrics", config)
    
    @pytest.mark.asyncio
    async def test_metrics_tracking(self, circuit_breaker):
        """Test comprehensive metrics tracking."""
        async def success_op():
            return "success"
        
        async def fail_op():
            raise Exception("failure")
        
        # Execute mixed operations
        await circuit_breaker.call(success_op)
        
        with pytest.raises(Exception):
            await circuit_breaker.call(fail_op)
        
        await circuit_breaker.call(success_op)
        
        # Verify metrics
        metrics = circuit_breaker.metrics
        assert metrics.total_requests == 3
        assert metrics.total_successes == 2
        assert metrics.total_failures == 1
        assert metrics.success_count == 2
        assert metrics.failure_count == 1
    
    def test_get_stats(self, circuit_breaker):
        """Test getting circuit breaker statistics."""
        # Add some metrics
        circuit_breaker.metrics.record_success()
        circuit_breaker.metrics.record_failure()
        circuit_breaker.metrics.record_success()
        
        stats = circuit_breaker.get_stats()
        
        assert "name" in stats
        assert "state" in stats
        assert "metrics" in stats
        assert stats["name"] == "test_metrics"
        assert stats["state"] == CircuitBreakerState.CLOSED.value
        assert stats["metrics"]["total_requests"] == 3
        assert stats["metrics"]["success_rate"] > 0
    
    def test_reset_metrics(self, circuit_breaker):
        """Test resetting circuit breaker metrics."""
        # Add some activity
        circuit_breaker.metrics.record_success()
        circuit_breaker.metrics.record_failure()
        
        assert circuit_breaker.metrics.success_count == 1
        assert circuit_breaker.metrics.failure_count == 1
        
        # Reset
        circuit_breaker.reset()
        
        assert circuit_breaker.metrics.success_count == 0
        assert circuit_breaker.metrics.failure_count == 0
        # Should preserve total counters
        assert circuit_breaker.metrics.total_requests == 2


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_multiple_circuit_breakers(self):
        """Test multiple independent circuit breakers."""
        cb1 = CircuitBreaker("service1", CircuitBreakerConfig(failure_threshold=2))
        cb2 = CircuitBreaker("service2", CircuitBreakerConfig(failure_threshold=2))
        
        async def service1_call():
            raise Exception("Service 1 down")
        
        async def service2_call():
            return "Service 2 ok"
        
        # Fail service 1
        for _ in range(2):
            with pytest.raises(Exception):
                await cb1.call(service1_call)
        
        # Service 1 should be open
        assert cb1.state == CircuitBreakerState.OPEN
        
        # Service 2 should still work
        result = await cb2.call(service2_call)
        assert result == "Service 2 ok"
        assert cb2.state == CircuitBreakerState.CLOSED
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_with_retries(self):
        """Test circuit breaker behavior with retry logic."""
        config = CircuitBreakerConfig(failure_threshold=3, timeout=0.1)
        cb = CircuitBreaker("retry_service", config)
        
        call_count = 0
        
        async def flaky_service():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception(f"Failure {call_count}")
            return "success after retries"
        
        # First two calls should fail
        for _ in range(2):
            with pytest.raises(Exception):
                await cb.call(flaky_service)
        
        # Third call should succeed
        result = await cb.call(flaky_service)
        assert result == "success after retries"
        assert cb.state == CircuitBreakerState.CLOSED


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])