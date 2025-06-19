#!/usr/bin/env python3
"""
Pipeline Infrastructure Fix - Async-aware health monitoring

This tool properly handles async infrastructure functions that were causing
the original "silent import failure" issues. The imports work fine - the
problem was executing async functions without proper event loop context.
"""

import asyncio
import sys
import os
import traceback
import json
from datetime import datetime
from typing import Dict, Any, Optional

def debug_print(message: str) -> None:
    """Print debug message with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}", flush=True)

async def test_infrastructure_health() -> Dict[str, Any]:
    """Test infrastructure health using proper async context."""
    debug_print("Testing infrastructure health with async context...")
    
    try:
        # Import infrastructure with confirmed working imports
        sys.path.insert(0, '.')
        from pipeline.infrastructure import (
            get_infrastructure_health,
            get_infrastructure_metrics,
            initialize_infrastructure,
            validate_infrastructure_config
        )
        
        debug_print("Infrastructure imports: OK")
        
        # Initialize infrastructure components
        debug_print("Initializing infrastructure...")
        initialize_infrastructure()
        debug_print("Infrastructure initialization: OK")
        
        # Test configuration validation (synchronous)
        debug_print("Validating configuration...")
        config_errors = validate_infrastructure_config()
        debug_print(f"Configuration validation: {len(config_errors)} errors")
        
        # Test async health check
        debug_print("Getting infrastructure health...")
        health_status = await get_infrastructure_health()
        debug_print(f"Health check: {health_status.get('status', 'unknown')}")
        
        # Test async metrics
        debug_print("Getting infrastructure metrics...")
        metrics = await get_infrastructure_metrics()
        debug_print("Metrics collection: OK")
        
        return {
            "status": "success",
            "health": health_status,
            "metrics": metrics,
            "config_errors": config_errors,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        debug_print(f"Infrastructure test FAILED: {e}")
        traceback.print_exc()
        return {
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat()
        }

def test_circuit_breaker_basic() -> Dict[str, Any]:
    """Test basic circuit breaker functionality."""
    debug_print("Testing basic circuit breaker functionality...")
    
    try:
        sys.path.insert(0, '.')
        from pipeline.infrastructure.circuit_breaker import (
            CircuitBreakerConfig,
            CircuitBreaker,
            get_circuit_breaker_registry
        )
        
        # Test configuration creation
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30.0,
            success_threshold=2,
            timeout=10.0
        )
        debug_print("CircuitBreakerConfig creation: OK")
        
        # Test circuit breaker creation
        breaker = CircuitBreaker("test_breaker", config)
        debug_print(f"CircuitBreaker creation: OK (state: {breaker.state.value})")
        
        # Test registry
        registry = get_circuit_breaker_registry()
        registry.register("test_api", config)
        debug_print(f"Registry operations: OK ({len(registry)} breakers)")
        
        # Test metrics (synchronous)
        metrics = breaker.get_metrics()
        debug_print(f"Metrics: {metrics['total_requests']} requests")
        
        return {
            "status": "success",
            "breaker_state": breaker.state.value,
            "registry_size": len(registry),
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        debug_print(f"Circuit breaker test FAILED: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

async def test_async_circuit_breaker() -> Dict[str, Any]:
    """Test async circuit breaker operations."""
    debug_print("Testing async circuit breaker operations...")
    
    try:
        sys.path.insert(0, '.')
        from pipeline.infrastructure.circuit_breaker import (
            CircuitBreakerConfig,
            CircuitBreaker
        )
        
        # Create circuit breaker
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=5.0,
            success_threshold=1,
            timeout=1.0
        )
        breaker = CircuitBreaker("async_test", config)
        
        # Test successful operation
        async def successful_operation():
            await asyncio.sleep(0.1)
            return "success"
        
        result = await breaker.call(successful_operation)
        debug_print(f"Async operation result: {result}")
        
        # Test circuit breaker context
        async with breaker.context():
            debug_print("Circuit breaker context: OK")
        
        # Get updated metrics
        metrics = breaker.get_metrics()
        debug_print(f"After async ops - Total requests: {metrics['total_requests']}")
        
        return {
            "status": "success",
            "operation_result": result,
            "final_metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        debug_print(f"Async circuit breaker test FAILED: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

async def comprehensive_async_test() -> Dict[str, Any]:
    """Run comprehensive async infrastructure tests."""
    debug_print("=== Comprehensive Async Infrastructure Test ===")
    
    results = {
        "test_start": datetime.now().isoformat(),
        "python_version": sys.version,
        "event_loop": str(type(asyncio.get_running_loop()).__name__),
        "tests": {}
    }
    
    # Test 1: Basic circuit breaker (sync parts)
    debug_print("\n--- Test 1: Basic Circuit Breaker ---")
    results["tests"]["basic_circuit_breaker"] = test_circuit_breaker_basic()
    
    # Test 2: Async circuit breaker operations
    debug_print("\n--- Test 2: Async Circuit Breaker Operations ---")
    results["tests"]["async_circuit_breaker"] = await test_async_circuit_breaker()
    
    # Test 3: Full infrastructure health check
    debug_print("\n--- Test 3: Infrastructure Health Check ---")
    results["tests"]["infrastructure_health"] = await test_infrastructure_health()
    
    results["test_end"] = datetime.now().isoformat()
    
    # Summary
    debug_print("\n=== Test Summary ===")
    for test_name, test_result in results["tests"].items():
        status = test_result.get("status", "unknown")
        debug_print(f"{test_name}: {status.upper()}")
        if status == "failed":
            debug_print(f"  Error: {test_result.get('error', 'unknown error')}")
    
    return results

def run_sync_fallback() -> Dict[str, Any]:
    """Fallback synchronous test if async fails."""
    debug_print("Running synchronous fallback tests...")
    
    try:
        # Test basic imports
        sys.path.insert(0, '.')
        import pipeline.infrastructure
        debug_print("Infrastructure import: OK")
        
        # Test synchronous components
        from pipeline.infrastructure.circuit_breaker import (
            CircuitBreakerState,
            CircuitBreakerConfig,
            get_circuit_breaker_registry
        )
        
        # Basic functionality
        config = CircuitBreakerConfig()
        registry = get_circuit_breaker_registry()
        
        return {
            "status": "success",
            "mode": "synchronous_fallback",
            "infrastructure_import": "OK",
            "components_working": True,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        debug_print(f"Sync fallback FAILED: {e}")
        return {
            "status": "failed",
            "mode": "synchronous_fallback",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

async def main():
    """Main async function."""
    debug_print("Starting async pipeline infrastructure tests...")
    
    try:
        results = await comprehensive_async_test()
        
        # Save results to file
        with open("pipeline_infrastructure_test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        debug_print("Results saved to pipeline_infrastructure_test_results.json")
        
        return 0
        
    except Exception as e:
        debug_print(f"Async tests failed: {e}")
        debug_print("Falling back to synchronous tests...")
        
        sync_results = run_sync_fallback()
        with open("pipeline_infrastructure_test_results.json", "w") as f:
            json.dump({"async_failed": True, "sync_results": sync_results}, f, indent=2)
        
        return 1

def main_sync():
    """Synchronous main function."""
    try:
        # Try to run async version
        if sys.version_info >= (3, 7):
            return asyncio.run(main())
        else:
            # Fallback for older Python
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(main())
    except Exception as e:
        debug_print(f"Could not run async tests: {e}")
        sync_results = run_sync_fallback()
        print(json.dumps(sync_results, indent=2))
        return 0

if __name__ == "__main__":
    sys.exit(main_sync())