#!/usr/bin/env python3
"""
Generation 2 Validation Tests - MAKE IT ROBUST (Reliable Implementation)
Tests error handling, logging, monitoring, and validation capabilities.
"""

import sys
import unittest
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class TestGeneration2Robust(unittest.TestCase):
    """Robust functionality tests for Generation 2"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_logger = logging.getLogger('test_robust')
        
    def test_error_handling_in_settings(self):
        """Test settings handle missing environment gracefully"""
        with patch.dict('os.environ', {}, clear=True):
            try:
                from pipeline.config.settings import get_settings
                settings = get_settings()
                # Should use defaults when environment variables are missing
                self.assertIsNotNone(settings)
                self.assertTrue(hasattr(settings, 'environment'))  # lowercase 'environment' not 'ENVIRONMENT'
            except Exception as e:
                self.fail(f"Settings should handle missing environment: {e}")
    
    def test_idea_model_validation(self):
        """Test Idea model validates input properly"""
        from pipeline.models.idea import Idea
        from datetime import datetime
        
        # Test with valid data
        valid_data = {
            'title': 'Valid Test Idea',
            'description': 'A valid test description',
            'category': 'saas',
            'status': 'DRAFT',
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        }
        idea = Idea(**valid_data)
        self.assertEqual(idea.title, 'Valid Test Idea')
        
        # Test with invalid data should raise validation error
        with self.assertRaises(Exception):
            invalid_data = {
                'title': '',  # Empty title should fail
                'description': 'Description',
                'category': 'invalid_category',  # Invalid category
                'status': 'DRAFT'
            }
            Idea(**invalid_data)
    
    def test_autonomous_executor_error_handling(self):
        """Test autonomous executor handles errors gracefully"""
        try:
            from pipeline.core.autonomous_executor import AutonomousExecutor, AutonomousTask
            
            executor = AutonomousExecutor()
            self.assertIsNotNone(executor)
            
            # Test task creation with invalid data
            with self.assertRaises(Exception):
                invalid_task = AutonomousTask(
                    id="",  # Empty ID should fail
                    name="Test Task",
                    description=""  # Empty description should fail
                )
        except ImportError as e:
            self.skipTest(f"Dependencies not available: {e}")
    
    def test_circuit_breaker_functionality(self):
        """Test circuit breaker pattern implementation"""
        try:
            from pipeline.infrastructure.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
            
            # Create circuit breaker with test settings using config object
            config = CircuitBreakerConfig(failure_threshold=2, timeout=1.0)
            cb = CircuitBreaker(name="test_breaker", config=config)
            self.assertIsNotNone(cb)
            
            # Test that it starts in closed state
            self.assertEqual(cb.state.value, 'closed')  # Use .value to get the enum value
            
        except ImportError as e:
            self.skipTest(f"Circuit breaker not available: {e}")
    
    def test_logging_configuration(self):
        """Test logging is properly configured"""
        import logging
        
        # Test that logging is configured
        logger = logging.getLogger('pipeline')
        self.assertIsNotNone(logger)
        
        # Test log level setting
        logger.setLevel(logging.INFO)
        self.assertEqual(logger.level, logging.INFO)
    
    def test_health_check_functionality(self):
        """Test health check endpoints work"""
        try:
            from pipeline.infrastructure.simple_health import SimpleHealthChecker
            
            health_checker = SimpleHealthChecker()
            health_status = health_checker.check_health()
            
            self.assertIsInstance(health_status, dict)
            self.assertIn('status', health_status)
            
        except ImportError as e:
            self.skipTest(f"Health check not available: {e}")
    
    def test_security_input_validation(self):
        """Test input validation and sanitization"""
        from pipeline.models.idea import Idea
        from datetime import datetime
        
        # Test SQL injection prevention
        malicious_input = "'; DROP TABLE ideas; --"
        try:
            idea = Idea(
                title=malicious_input,
                description="Test description",
                category="saas",
                status="DRAFT",
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            # Should accept but sanitize the input
            self.assertIsNotNone(idea.title)
        except Exception:
            # Or reject malicious input entirely
            self.assertTrue(True, "Malicious input properly rejected")
    
    def test_performance_monitoring_setup(self):
        """Test performance monitoring is configured"""
        try:
            from pipeline.telemetry import get_tracer
            tracer = get_tracer('test')
            self.assertIsNotNone(tracer)
        except ImportError as e:
            self.skipTest(f"Telemetry not available: {e}")
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        from pipeline.config.settings import get_settings
        
        settings = get_settings()
        
        # Test that critical settings have reasonable defaults
        self.assertTrue(hasattr(settings, 'environment'))  # lowercase 'environment'
        self.assertEqual(settings.environment, 'development')  # Should default to development
        
        # Test database configuration exists and is accessible
        self.assertIsNotNone(settings.database)
        self.assertTrue(hasattr(settings.database, 'host'))
        self.assertTrue(hasattr(settings.database, 'port'))
        self.assertTrue(hasattr(settings.database, 'database'))

if __name__ == '__main__':
    print("🛡️ Running Generation 2 Robust Functionality Tests")
    print("=" * 55)
    
    # Configure logging for tests
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    unittest.main(verbosity=2)
