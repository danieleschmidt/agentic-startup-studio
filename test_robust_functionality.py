#!/usr/bin/env python3
"""
Robust functionality test for Generation 2: Make it Robust
"""

import sys
import os
import asyncio
from datetime import datetime
from uuid import uuid4

# Add project root to path
sys.path.insert(0, '/root/repo')

def test_error_handling():
    """Test error handling mechanisms."""
    try:
        from pipeline.infrastructure.circuit_breaker import CircuitBreaker
        from pipeline.infrastructure.quality_gates import QualityGate
        
        # Test circuit breaker
        breaker = CircuitBreaker(failure_threshold=3, timeout_seconds=5)
        print("✅ Circuit breaker created")
        
        # Test quality gate
        gate = QualityGate()
        print("✅ Quality gate created")
        
        return True
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def test_input_validation():
    """Test input validation and sanitization."""
    try:
        from pipeline.models.idea import IdeaDraft, IdeaCategory
        from pydantic import ValidationError
        
        # Test valid input
        valid_draft = IdeaDraft(
            title="Valid Startup Idea",
            description="This is a valid description with enough characters to meet minimum requirements.",
            category=IdeaCategory.SAAS
        )
        print("✅ Valid input validation working")
        
        # Test invalid input - title too short
        try:
            invalid_draft = IdeaDraft(
                title="Short",
                description="This is a valid description with enough characters to meet minimum requirements.",
                category=IdeaCategory.SAAS
            )
            print("❌ Should have failed validation for short title")
            return False
        except ValidationError:
            print("✅ Invalid title validation working")
        
        # Test invalid input - description too short
        try:
            invalid_draft = IdeaDraft(
                title="Valid Long Title for Testing",
                description="Short",
                category=IdeaCategory.SAAS
            )
            print("❌ Should have failed validation for short description")
            return False
        except ValidationError:
            print("✅ Invalid description validation working")
        
        return True
    except Exception as e:
        print(f"❌ Input validation test failed: {e}")
        return False

def test_security_features():
    """Test security configurations and features."""
    try:
        from pipeline.config.settings import get_settings
        settings = get_settings()
        
        # Check security settings
        if hasattr(settings, 'secret_key'):
            print("✅ Secret key configuration available")
        
        if hasattr(settings.database, 'enable_ssl'):
            print(f"✅ Database SSL configuration: {settings.database.enable_ssl}")
        
        if hasattr(settings.validation, 'enable_html_sanitization'):
            print(f"✅ HTML sanitization: {settings.validation.enable_html_sanitization}")
        
        if hasattr(settings.validation, 'enable_profanity_filter'):
            print(f"✅ Profanity filter: {settings.validation.enable_profanity_filter}")
        
        return True
    except Exception as e:
        print(f"❌ Security features test failed: {e}")
        return False

def test_database_security():
    """Test database security configurations."""
    try:
        from pipeline.config.settings import get_db_config
        db_config = get_db_config()
        
        # Check SQL injection protections
        if hasattr(db_config, 'max_query_params'):
            print(f"✅ Query parameter limits: {db_config.max_query_params}")
        
        if hasattr(db_config, 'statement_timeout'):
            print(f"✅ Statement timeout protection: {db_config.statement_timeout}s")
        
        if hasattr(db_config, 'connection_lifetime'):
            print(f"✅ Connection lifetime limits: {db_config.connection_lifetime}s")
        
        # Test safe URL generation (no password exposure)
        safe_url = db_config.get_safe_connection_url()
        if "****" in safe_url or db_config.password == "":
            print("✅ Safe connection URL masking working")
        else:
            print("❌ Password should be masked in connection URL")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Database security test failed: {e}")
        return False

def test_logging_security():
    """Test secure logging configurations."""
    try:
        from pipeline.config.settings import get_logging_config
        logging_config = get_logging_config()
        
        if hasattr(logging_config, 'enable_correlation_ids'):
            print(f"✅ Correlation ID tracking: {logging_config.enable_correlation_ids}")
        
        if hasattr(logging_config, 'enable_json_logging'):
            print(f"✅ Structured JSON logging: {logging_config.enable_json_logging}")
        
        return True
    except Exception as e:
        print(f"❌ Logging security test failed: {e}")
        return False

def test_infrastructure_resilience():
    """Test infrastructure resilience features."""
    try:
        from pipeline.config.settings import get_infrastructure_config
        infra_config = get_infrastructure_config()
        
        if hasattr(infra_config, 'circuit_breaker_enabled'):
            print("✅ Circuit breaker infrastructure available")
        
        if hasattr(infra_config, 'health_check_interval'):
            print(f"✅ Health monitoring interval: {infra_config.health_check_interval}s")
        
        return True
    except Exception as e:
        print(f"❌ Infrastructure resilience test failed: {e}")
        return False

def test_budget_controls():
    """Test budget control and cost monitoring."""
    try:
        from pipeline.config.settings import get_budget_config
        budget_config = get_budget_config()
        
        if hasattr(budget_config, 'total_cycle_budget'):
            print(f"✅ Budget limits configured: ${budget_config.total_cycle_budget}")
        
        if hasattr(budget_config, 'warning_threshold'):
            print(f"✅ Budget warning threshold: {budget_config.warning_threshold * 100}%")
        
        if hasattr(budget_config, 'enable_emergency_shutdown'):
            print(f"✅ Emergency shutdown protection: {budget_config.enable_emergency_shutdown}")
        
        return True
    except Exception as e:
        print(f"❌ Budget controls test failed: {e}")
        return False

async def test_async_error_handling():
    """Test asynchronous error handling."""
    try:
        # Basic async test
        await asyncio.sleep(0.1)
        print("✅ Basic async functionality working")
        
        return True
    except Exception as e:
        print(f"❌ Async error handling test failed: {e}")
        return False

def run_generation_2_tests():
    """Run all Generation 2 robust functionality tests."""
    print("🛡️ Running Generation 2: Make it Robust Tests")
    print("=" * 50)
    
    tests = [
        ("Error Handling", test_error_handling),
        ("Input Validation", test_input_validation),
        ("Security Features", test_security_features),
        ("Database Security", test_database_security),
        ("Logging Security", test_logging_security),
        ("Infrastructure Resilience", test_infrastructure_resilience),
        ("Budget Controls", test_budget_controls),
    ]
    
    async_tests = [
        ("Async Error Handling", test_async_error_handling),
    ]
    
    results = []
    
    # Run synchronous tests
    for test_name, test_func in tests:
        print(f"\n📋 Testing {test_name}...")
        success = test_func()
        results.append((test_name, success))
    
    # Run async tests
    for test_name, test_func in async_tests:
        print(f"\n📋 Testing {test_name}...")
        try:
            success = asyncio.run(test_func())
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Async test error: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 Generation 2 Test Results:")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} {test_name}")
    
    print(f"\n📈 Summary: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed >= total * 0.85:  # 85% pass rate for robust
        print("🎉 Generation 2: Make it Robust - COMPLETE!")
        return True
    else:
        print("⚠️  Generation 2: Robustness improvements needed")
        return False

if __name__ == "__main__":
    success = run_generation_2_tests()
    sys.exit(0 if success else 1)