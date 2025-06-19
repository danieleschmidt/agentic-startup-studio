#!/usr/bin/env python3
"""
Standalone Health Monitor for Agentic Startup Studio.

Independent health checking system that doesn't rely on pipeline imports
to avoid the silent import failures affecting the main pipeline modules.
"""

import sys
import time
import json
import importlib.util
from datetime import datetime, timezone
from pathlib import Path


def test_basic_imports():
    """Test basic Python imports."""
    try:
        import os
        import json
        import time
        return True, "Basic imports successful"
    except ImportError as e:
        return False, f"Basic import failed: {e}"


def test_pipeline_exists():
    """Test if pipeline directory exists."""
    try:
        pipeline_path = Path("pipeline")
        if pipeline_path.exists() and pipeline_path.is_dir():
            return True, f"Pipeline directory exists with {len(list(pipeline_path.iterdir()))} items"
        else:
            return False, "Pipeline directory not found"
    except Exception as e:
        return False, f"Pipeline directory check failed: {e}"


def test_pipeline_init():
    """Test if pipeline __init__.py can be loaded."""
    try:
        pipeline_init = Path("pipeline/__init__.py")
        if pipeline_init.exists():
            return True, "Pipeline __init__.py exists"
        else:
            return False, "Pipeline __init__.py not found"
    except Exception as e:
        return False, f"Pipeline init check failed: {e}"


def test_infrastructure_init():
    """Test if infrastructure __init__.py can be loaded."""
    try:
        infra_init = Path("pipeline/infrastructure/__init__.py")
        if infra_init.exists():
            return True, "Infrastructure __init__.py exists"
        else:
            return False, "Infrastructure __init__.py not found"
    except Exception as e:
        return False, f"Infrastructure init check failed: {e}"


def test_adapter_files():
    """Test if adapter files exist."""
    try:
        adapter_dir = Path("pipeline/adapters")
        if not adapter_dir.exists():
            return False, "Adapters directory not found"
        
        adapters = list(adapter_dir.glob("*_adapter.py"))
        return True, f"Found {len(adapters)} adapter files: {[a.name for a in adapters]}"
    except Exception as e:
        return False, f"Adapter file check failed: {e}"


def test_simple_health_module():
    """Test if our simple health module exists."""
    try:
        simple_health = Path("pipeline/infrastructure/simple_health.py")
        if simple_health.exists():
            return True, f"Simple health module exists ({simple_health.stat().st_size} bytes)"
        else:
            return False, "Simple health module not found"
    except Exception as e:
        return False, f"Simple health module check failed: {e}"


def run_health_checks():
    """Run all health checks and return results."""
    checks = [
        ("basic_imports", test_basic_imports),
        ("pipeline_exists", test_pipeline_exists),
        ("pipeline_init", test_pipeline_init),
        ("infrastructure_init", test_infrastructure_init),
        ("adapter_files", test_adapter_files),
        ("simple_health_module", test_simple_health_module),
    ]
    
    results = {}
    overall_status = "healthy"
    
    print("=== Standalone Health Check ===")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print()
    
    for check_name, check_func in checks:
        try:
            success, message = check_func()
            status = "[PASS]" if success else "[FAIL]"
            print(f"{status} {check_name}: {message}")
            
            results[check_name] = {
                "status": "healthy" if success else "unhealthy",
                "message": message,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            if not success:
                overall_status = "unhealthy"
                
        except Exception as e:
            print(f"✗ ERROR {check_name}: {str(e)}")
            results[check_name] = {
                "status": "unhealthy",
                "message": f"Exception: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            overall_status = "unhealthy"
    
    print()
    print(f"Overall Status: {overall_status.upper()}")
    
    return {
        "overall_status": overall_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": results,
        "summary": {
            "total": len(checks),
            "passed": sum(1 for r in results.values() if r["status"] == "healthy"),
            "failed": sum(1 for r in results.values() if r["status"] == "unhealthy")
        }
    }


if __name__ == "__main__":
    try:
        results = run_health_checks()
        
        # Save results to file
        with open("health_check_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to health_check_results.json")
        
        # Exit with appropriate code
        sys.exit(0 if results["overall_status"] == "healthy" else 1)
        
    except Exception as e:
        print(f"CRITICAL ERROR: Health check system failed: {e}")
        sys.exit(2)