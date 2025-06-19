#!/usr/bin/env python3
"""
Pipeline Debug Tool - Synchronous infrastructure analysis

This tool bypasses async dependencies to provide immediate debugging
feedback for the pipeline infrastructure issues.
"""

import sys
import os
import traceback
from datetime import datetime
from typing import Dict, Any

def debug_print(message: str) -> None:
    """Print debug message with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}", flush=True)

def test_basic_imports() -> Dict[str, Any]:
    """Test basic Python imports without async components."""
    debug_print("Testing basic imports...")
    results = {}
    
    try:
        import asyncio
        results["asyncio"] = "OK"
        debug_print("asyncio: OK")
    except Exception as e:
        results["asyncio"] = f"FAILED: {e}"
        debug_print(f"asyncio FAILED: {e}")
    
    try:
        sys.path.insert(0, '.')
        import pipeline
        results["pipeline"] = "OK"
        debug_print("pipeline: OK")
    except Exception as e:
        results["pipeline"] = f"FAILED: {e}"
        debug_print(f"pipeline FAILED: {e}")
    
    return results

def test_infrastructure_components() -> Dict[str, Any]:
    """Test individual infrastructure components."""
    debug_print("Testing infrastructure components...")
    results = {}
    
    # Test circuit breaker enum (safe import)
    try:
        debug_print("Importing CircuitBreakerState...")
        from pipeline.infrastructure.circuit_breaker import CircuitBreakerState
        results["circuit_breaker_state"] = "OK"
        debug_print("CircuitBreakerState: OK")
    except Exception as e:
        results["circuit_breaker_state"] = f"FAILED: {e}"
        debug_print(f"CircuitBreakerState FAILED: {e}")
    
    # Test circuit breaker config (should be safe)
    try:
        debug_print("Importing CircuitBreakerConfig...")
        from pipeline.infrastructure.circuit_breaker import CircuitBreakerConfig
        results["circuit_breaker_config"] = "OK"
        debug_print("CircuitBreakerConfig: OK")
    except Exception as e:
        results["circuit_breaker_config"] = f"FAILED: {e}"
        debug_print(f"CircuitBreakerConfig FAILED: {e}")
    
    # Test the problematic async circuit breaker class
    try:
        debug_print("Importing CircuitBreaker class (async)...")
        from pipeline.infrastructure.circuit_breaker import CircuitBreaker
        results["circuit_breaker_class"] = "OK"
        debug_print("CircuitBreaker class: OK")
    except Exception as e:
        results["circuit_breaker_class"] = f"FAILED: {e}"
        debug_print(f"CircuitBreaker class FAILED: {e}")
    
    # Test registry (likely problematic)
    try:
        debug_print("Importing CircuitBreakerRegistry...")
        from pipeline.infrastructure.circuit_breaker import CircuitBreakerRegistry
        results["circuit_breaker_registry"] = "OK"
        debug_print("CircuitBreakerRegistry: OK")
    except Exception as e:
        results["circuit_breaker_registry"] = f"FAILED: {e}"
        debug_print(f"CircuitBreakerRegistry FAILED: {e}")
    
    # Test the global registry function
    try:
        debug_print("Importing get_circuit_breaker_registry...")
        from pipeline.infrastructure.circuit_breaker import get_circuit_breaker_registry
        results["get_circuit_breaker_registry"] = "OK"
        debug_print("get_circuit_breaker_registry: OK")
    except Exception as e:
        results["get_circuit_breaker_registry"] = f"FAILED: {e}"
        debug_print(f"get_circuit_breaker_registry FAILED: {e}")
    
    return results

def test_infrastructure_init() -> Dict[str, Any]:
    """Test the problematic infrastructure __init__.py"""
    debug_print("Testing infrastructure __init__.py...")
    results = {}
    
    try:
        debug_print("Attempting full infrastructure import...")
        import pipeline.infrastructure
        results["infrastructure_import"] = "OK"
        debug_print("Infrastructure import: OK")
    except Exception as e:
        results["infrastructure_import"] = f"FAILED: {e}"
        debug_print(f"Infrastructure import FAILED: {e}")
        traceback.print_exc()
    
    try:
        debug_print("Attempting specific function import...")
        from pipeline.infrastructure import get_circuit_breaker_registry
        results["infrastructure_function"] = "OK"
        debug_print("Infrastructure function import: OK")
    except Exception as e:
        results["infrastructure_function"] = f"FAILED: {e}"
        debug_print(f"Infrastructure function import FAILED: {e}")
    
    return results

def analyze_async_issues() -> Dict[str, Any]:
    """Analyze potential async-related issues."""
    debug_print("Analyzing async environment...")
    results = {}
    
    try:
        import asyncio
        
        # Check if we're in an event loop
        try:
            loop = asyncio.get_running_loop()
            results["event_loop"] = f"RUNNING: {type(loop).__name__}"
            debug_print(f"Event loop running: {type(loop).__name__}")
        except RuntimeError:
            results["event_loop"] = "NOT_RUNNING"
            debug_print("No event loop running")
        
        # Check event loop policy
        policy = asyncio.get_event_loop_policy()
        results["loop_policy"] = f"{type(policy).__name__}"
        debug_print(f"Event loop policy: {type(policy).__name__}")
        
        # Check available loop implementation
        try:
            new_loop = asyncio.new_event_loop()
            results["loop_creation"] = f"OK: {type(new_loop).__name__}"
            new_loop.close()
            debug_print(f"Loop creation: OK ({type(new_loop).__name__})")
        except Exception as e:
            results["loop_creation"] = f"FAILED: {e}"
            debug_print(f"Loop creation FAILED: {e}")
            
    except Exception as e:
        results["asyncio_analysis"] = f"FAILED: {e}"
        debug_print(f"Asyncio analysis FAILED: {e}")
    
    return results

def main():
    """Main debugging function."""
    debug_print("=== Pipeline Infrastructure Debug Tool ===")
    debug_print(f"Python version: {sys.version}")
    debug_print(f"Working directory: {os.getcwd()}")
    
    all_results = {}
    
    # Test basic imports
    all_results["basic_imports"] = test_basic_imports()
    
    # Analyze async environment
    all_results["async_analysis"] = analyze_async_issues()
    
    # Test infrastructure components step by step
    all_results["infrastructure_components"] = test_infrastructure_components()
    
    # Test the problematic infrastructure init
    all_results["infrastructure_init"] = test_infrastructure_init()
    
    # Summary
    debug_print("\n=== Debug Summary ===")
    for category, tests in all_results.items():
        debug_print(f"\n{category.upper()}:")
        for test_name, result in tests.items():
            status = "PASS" if result == "OK" else "FAIL"
            debug_print(f"  {test_name}: {status}")
            if result != "OK":
                debug_print(f"    -> {result}")
    
    # Check for specific patterns
    debug_print("\n=== Issue Analysis ===")
    
    failed_tests = []
    for category, tests in all_results.items():
        for test_name, result in tests.items():
            if result != "OK":
                failed_tests.append(f"{category}.{test_name}: {result}")
    
    if failed_tests:
        debug_print("IDENTIFIED ISSUES:")
        for issue in failed_tests:
            debug_print(f"  - {issue}")
    else:
        debug_print("No obvious import failures detected.")
        debug_print("The issue may be related to:")
        debug_print("  - Async context initialization hanging")
        debug_print("  - Event loop conflicts")
        debug_print("  - Module-level async code execution")
    
    debug_print("\n=== Debug Complete ===")
    return 0

if __name__ == "__main__":
    sys.exit(main())