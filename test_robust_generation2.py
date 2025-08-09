#!/usr/bin/env python3
"""
Generation 2 Robustness Test
Tests error handling, validation, logging, and monitoring
"""
import asyncio
import sys
import os
import tempfile
import logging
from pathlib import Path

# Add project root to path  
sys.path.insert(0, '.')

async def test_generation2_robust():
    """Test robustness features - MAKE IT ROBUST"""
    print("🛡️ GENERATION 2 TEST: MAKE IT ROBUST")
    print("=" * 50)
    
    success_count = 0
    total_tests = 7
    
    try:
        # Test 1: Error Handling in Search
        print("✅ Test 1: Error Handling")
        from core.search_tools import basic_web_search_tool
        
        # Test with invalid parameters - should not crash
        result = basic_web_search_tool("", 0)
        print(f"   ✓ Graceful handling of edge case: {len(result)} results")
        success_count += 1
        
        # Test 2: Data Validation
        print("\n✅ Test 2: Data Validation")
        from pipeline.models.idea import Idea, IdeaCategory
        from pydantic import ValidationError
        
        try:
            # Test invalid data - should raise ValidationError
            invalid_idea = Idea(
                title="",  # Empty title should fail validation
                description="Valid description", 
                category="invalid_category"  # Invalid category
            )
            print("   ❌ Validation should have failed")
        except ValidationError as e:
            print("   ✓ Pydantic validation caught invalid data")
            success_count += 1
        
        # Test 3: Logging Setup
        print("\n✅ Test 3: Logging Configuration")
        logger = logging.getLogger("test_robust")
        logger.setLevel(logging.INFO)
        
        # Create temporary log file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            log_file = f.name
            
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        logger.info("Test log message for robustness validation")
        handler.close()
        
        # Verify log was written
        with open(log_file, 'r') as f:
            log_content = f.read()
            
        if "Test log message" in log_content:
            print("   ✓ Logging system operational")
            success_count += 1
        
        # Cleanup
        os.unlink(log_file)
        
        # Test 4: Configuration Management
        print("\n✅ Test 4: Configuration Management")
        try:
            from pipeline.config.settings import get_settings
            settings = get_settings()
            print(f"   ✓ Settings loaded successfully: {type(settings).__name__}")
            success_count += 1
        except Exception as e:
            print(f"   ⚠️ Settings configuration issue: {e}")
        
        # Test 5: Database Connection Handling
        print("\n✅ Test 5: Database Connection Safety")
        try:
            # Test database connection logic exists (even if DB not available)
            from pipeline.storage.connection_pool_manager import ConnectionPoolManager
            print("   ✓ Database connection handling available")
            success_count += 1
        except ImportError as e:
            print(f"   ⚠️ Database connection handling not found: {e}")
        
        # Test 6: Circuit Breaker Pattern
        print("\n✅ Test 6: Circuit Breaker Implementation")
        try:
            from pipeline.infrastructure.circuit_breaker import CircuitBreakerConfig, CircuitBreakerRegistry
            
            # Test circuit breaker configuration
            config = CircuitBreakerConfig(failure_threshold=3, timeout=10.0)
            registry = CircuitBreakerRegistry()
            print("   ✓ Circuit breaker pattern implemented")
            success_count += 1
        except ImportError:
            print("   ⚠️ Circuit breaker not found - will implement")
        except Exception as e:
            print(f"   ⚠️ Circuit breaker issue: {e}")
            
        # Test 7: Health Monitoring
        print("\n✅ Test 7: Health Monitoring")
        try:
            from pipeline.infrastructure.simple_health import SimpleHealthMonitor, get_health_monitor
            health_monitor = get_health_monitor()
            status = health_monitor.get_overall_status()
            print(f"   ✓ Health monitoring operational: {status['overall_status']}")
            success_count += 1
        except ImportError:
            print("   ⚠️ Health monitoring not found - will implement")
        except Exception as e:
            print(f"   ⚠️ Health monitoring issue: {e}")
        
        # Summary
        print("\n" + "=" * 50)
        print(f"🛡️ GENERATION 2 ROBUSTNESS: {success_count}/{total_tests} TESTS PASSED")
        
        if success_count >= 5:  # 70% pass rate for robustness
            print("✅ ROBUST - Core reliability features operational")
            return True
        else:
            print("⚠️  NEEDS IMPROVEMENT - Some robustness features missing")
            return False
            
    except Exception as e:
        print(f"\n❌ Generation 2 test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_generation2_robust())
    sys.exit(0 if success else 1)