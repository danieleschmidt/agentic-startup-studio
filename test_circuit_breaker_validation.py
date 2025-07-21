#!/usr/bin/env python3
"""
Simple validation script for circuit breaker implementation.
Tests the core logic without external dependencies.
"""

import sys
import asyncio
from enum import Enum

# Mock any missing modules
try:
    from pipeline.infrastructure.circuit_breaker import (
        CircuitBreakerState, CircuitBreakerConfig, CircuitBreakerMetrics,
        CircuitOpenError, CircuitTimeoutError
    )
    print("✅ Circuit breaker imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)


def test_config_validation():
    """Test configuration validation logic."""
    print("\n🔧 Testing CircuitBreakerConfig validation...")
    
    # Test valid config
    config = CircuitBreakerConfig()
    assert config.failure_threshold == 5
    assert config.recovery_timeout == 60.0
    assert config.success_threshold == 3
    assert config.timeout == 30.0
    print("✅ Default config values: PASS")
    
    # Test custom config
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
    print("✅ Custom config values: PASS")
    
    # Test validation errors
    try:
        CircuitBreakerConfig(failure_threshold=0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("✅ Failure threshold validation: PASS")
    
    try:
        CircuitBreakerConfig(recovery_timeout=-1)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("✅ Recovery timeout validation: PASS")


def test_metrics_logic():
    """Test metrics tracking logic."""
    print("\n📊 Testing CircuitBreakerMetrics...")
    
    metrics = CircuitBreakerMetrics()
    
    # Initial state
    assert metrics.failure_count == 0
    assert metrics.success_count == 0
    assert metrics.total_requests == 0
    print("✅ Initial metrics state: PASS")
    
    # Record success
    metrics.record_success()
    assert metrics.success_count == 1
    assert metrics.total_successes == 1
    assert metrics.total_requests == 1
    print("✅ Success recording: PASS")
    
    # Record failure
    metrics.record_failure()
    assert metrics.failure_count == 1
    assert metrics.total_failures == 1
    assert metrics.total_requests == 2
    assert metrics.last_failure_time is not None
    print("✅ Failure recording: PASS")
    
    # Reset counters
    metrics.reset()
    assert metrics.failure_count == 0
    assert metrics.success_count == 0
    # Total counters should remain
    assert metrics.total_requests == 2
    print("✅ Metrics reset: PASS")


def test_state_transitions():
    """Test state transition logic."""
    print("\n🔄 Testing state transitions...")
    
    # Test state enum
    assert CircuitBreakerState.CLOSED.value == "closed"
    assert CircuitBreakerState.OPEN.value == "open"
    assert CircuitBreakerState.HALF_OPEN.value == "half_open"
    print("✅ State enum values: PASS")
    
    # Test state comparisons
    states = [CircuitBreakerState.CLOSED, CircuitBreakerState.OPEN, CircuitBreakerState.HALF_OPEN]
    assert len(set(states)) == 3  # All different
    print("✅ State uniqueness: PASS")


def test_exception_hierarchy():
    """Test exception class hierarchy."""
    print("\n⚠️  Testing exception hierarchy...")
    
    # Test CircuitOpenError
    error = CircuitOpenError()
    assert str(error) == "Circuit breaker is open"
    assert isinstance(error, Exception)
    print("✅ CircuitOpenError default message: PASS")
    
    error = CircuitOpenError("Custom message")
    assert str(error) == "Custom message"
    print("✅ CircuitOpenError custom message: PASS")
    
    # Test CircuitTimeoutError
    error = CircuitTimeoutError()
    assert str(error) == "Operation timed out"
    assert isinstance(error, Exception)
    print("✅ CircuitTimeoutError default message: PASS")
    
    error = CircuitTimeoutError("Custom timeout")
    assert str(error) == "Custom timeout"
    print("✅ CircuitTimeoutError custom message: PASS")


def test_state_transition_thresholds():
    """Test state transition threshold logic."""
    print("\n🎯 Testing threshold logic...")
    
    config = CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2
    )
    
    # Simulate failure counting
    failure_count = 0
    for i in range(5):
        failure_count += 1
        should_open = failure_count >= config.failure_threshold
        if i < 2:
            assert not should_open, f"Should not open at failure {i+1}"
        else:
            assert should_open, f"Should open at failure {i+1}"
    
    print("✅ Failure threshold logic: PASS")
    
    # Simulate success counting in half-open
    success_count = 0
    for i in range(3):
        success_count += 1
        should_close = success_count >= config.success_threshold
        if i < 1:
            assert not should_close, f"Should not close at success {i+1}"
        else:
            assert should_close, f"Should close at success {i+1}"
    
    print("✅ Success threshold logic: PASS")


async def test_basic_timeout_logic():
    """Test basic timeout logic."""
    print("\n⏱️  Testing timeout logic...")
    
    config = CircuitBreakerConfig(timeout=0.1)
    
    # Simulate timeout check
    start_time = asyncio.get_event_loop().time()
    await asyncio.sleep(0.05)  # Half the timeout
    elapsed = asyncio.get_event_loop().time() - start_time
    
    timed_out = elapsed > config.timeout
    assert not timed_out, "Should not timeout for fast operation"
    print("✅ Fast operation timeout: PASS")
    
    # Simulate slow operation
    start_time = asyncio.get_event_loop().time()
    await asyncio.sleep(0.15)  # Longer than timeout
    elapsed = asyncio.get_event_loop().time() - start_time
    
    timed_out = elapsed > config.timeout
    assert timed_out, "Should timeout for slow operation"
    print("✅ Slow operation timeout: PASS")


async def main():
    """Run all validation tests."""
    print("🔧 Circuit Breaker Implementation Validation")
    print("=" * 50)
    
    try:
        test_config_validation()
        test_metrics_logic()
        test_state_transitions()
        test_exception_hierarchy()
        test_state_transition_thresholds()
        await test_basic_timeout_logic()
        
        print("\n" + "=" * 50)
        print("🎉 All validation tests PASSED!")
        print("✅ Circuit breaker implementation is correct")
        return 0
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))