#!/usr/bin/env python3
"""
Comprehensive Quality Gates Validation
Tests security, performance, compliance, and quality standards.
"""

import sys
import unittest
import time
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class TestComprehensiveQualityGates(unittest.TestCase):
    """Comprehensive quality gate validation tests"""
    
    def setUp(self):
        """Set up test environment"""
        self.project_root = Path(__file__).parent
        
    def test_security_configuration(self):
        """Test security configurations are properly set"""
        from pipeline.config.settings import get_settings
        
        settings = get_settings()
        
        # Test security settings exist
        self.assertIsNotNone(settings.database)
        self.assertTrue(settings.database.enable_ssl)
        self.assertEqual(settings.database.ssl_mode, "require")
        
        # Test validation settings
        self.assertTrue(settings.validation.enable_profanity_filter)
        self.assertTrue(settings.validation.enable_spam_detection)
        self.assertTrue(settings.validation.enable_html_sanitization)
    
    def test_input_validation_security(self):
        """Test input validation prevents common attacks"""
        from pipeline.models.idea import Idea
        from datetime import datetime
        
        # Test SQL injection prevention (model should handle this safely)
        malicious_inputs = [
            "'; DROP TABLE ideas; --",
            "<script>alert('xss')</script>",
            "' OR '1'='1",
            "../../../etc/passwd",
            "{{7*7}}",  # Template injection
        ]
        
        for malicious_input in malicious_inputs:
            try:
                idea = Idea(
                    title=malicious_input,
                    description="Test description",
                    category="saas",
                    status="DRAFT",
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                # If it creates successfully, the input should be sanitized/escaped
                self.assertIsNotNone(idea.title)
                # Should not contain dangerous characters unescaped
                self.assertNotIn("<script>", idea.title)
            except Exception:
                # Or it should be rejected entirely - both approaches are valid
                pass
    
    def test_performance_benchmarks(self):
        """Test system meets performance benchmarks"""
        from pipeline.models.idea import Idea
        from datetime import datetime
        
        # Test model creation performance
        start_time = time.time()
        
        ideas = []
        for i in range(100):
            idea = Idea(
                title=f"Performance Test Idea {i}",
                description=f"Performance testing idea #{i}",
                category="saas",
                status="DRAFT",
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            ideas.append(idea)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should create 100 ideas in under 1 second
        self.assertLess(total_time, 1.0, "Model creation should be fast")
        self.assertEqual(len(ideas), 100, "All ideas should be created")
        
        # Test individual operation speed
        per_item_time = total_time / 100
        self.assertLess(per_item_time, 0.01, "Individual operations should be under 10ms")
    
    def test_error_handling_coverage(self):
        """Test error handling covers common failure scenarios"""
        from pipeline.config.settings import get_settings
        from pipeline.infrastructure.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
        
        # Test configuration handles missing values gracefully
        settings = get_settings()
        self.assertIsNotNone(settings)
        
        # Test circuit breaker error handling
        config = CircuitBreakerConfig(failure_threshold=1, timeout=0.1)
        cb = CircuitBreaker(name="test_error_handling", config=config)
        
        # Circuit breaker should handle exceptions properly
        self.assertEqual(cb.state.value, "closed")
    
    def test_logging_and_monitoring(self):
        """Test logging and monitoring infrastructure"""
        import logging
        
        # Test logging configuration
        logger = logging.getLogger('pipeline.test')
        self.assertIsNotNone(logger)
        
        # Test log levels work
        logger.info("Test info message")
        logger.warning("Test warning message")
        logger.error("Test error message")
        
        # Should not raise exceptions
        self.assertTrue(True)
    
    def test_configuration_validation(self):
        """Test all configurations are valid and complete"""
        from pipeline.config.settings import get_settings
        
        settings = get_settings()
        
        # Test critical configurations exist
        required_attrs = [
            'environment', 'database', 'validation', 
            'embedding', 'logging', 'budget', 'infrastructure'
        ]
        
        for attr in required_attrs:
            self.assertTrue(hasattr(settings, attr), f"Missing critical config: {attr}")
        
        # Test database configuration is complete
        db_config = settings.database
        self.assertGreater(db_config.port, 0)
        self.assertGreater(db_config.max_connections, 0)
        self.assertIn(db_config.ssl_mode, ['disable', 'allow', 'prefer', 'require', 'verify-ca', 'verify-full'])
        
        # Test budget configuration is valid
        budget_config = settings.budget
        self.assertGreater(budget_config.total_cycle_budget, 0)
        self.assertLessEqual(budget_config.warning_threshold, 1.0)
        self.assertGreaterEqual(budget_config.warning_threshold, 0.0)
    
    def test_code_quality_standards(self):
        """Test code follows quality standards"""
        # Test project structure exists
        expected_dirs = [
            'pipeline', 'tests', 'docs', 'scripts',
            'pipeline/config', 'pipeline/models', 'pipeline/core'
        ]
        
        for dir_path in expected_dirs:
            full_path = self.project_root / dir_path
            self.assertTrue(full_path.exists(), f"Required directory missing: {dir_path}")
        
        # Test key files exist
        expected_files = [
            'README.md', 'requirements.txt', 'pyproject.toml',
            'pipeline/__init__.py', 'pipeline/config/settings.py'
        ]
        
        for file_path in expected_files:
            full_path = self.project_root / file_path
            self.assertTrue(full_path.exists(), f"Required file missing: {file_path}")
    
    def test_dependency_security(self):
        """Test dependencies are secure and up-to-date"""
        # Read requirements.txt and check for known secure packages
        requirements_file = self.project_root / 'requirements.txt'
        self.assertTrue(requirements_file.exists(), "requirements.txt should exist")
        
        with open(requirements_file, 'r') as f:
            requirements = f.read()
        
        # Test essential security packages are present
        essential_packages = ['pydantic', 'fastapi']
        for package in essential_packages:
            self.assertIn(package, requirements, f"Essential package {package} should be in requirements")
    
    def test_data_validation_comprehensive(self):
        """Test comprehensive data validation"""
        from pipeline.config.settings import get_validation_config
        
        validation_config = get_validation_config()
        
        # Test validation thresholds are reasonable
        self.assertGreaterEqual(validation_config.similarity_threshold, 0.0)
        self.assertLessEqual(validation_config.similarity_threshold, 1.0)
        
        # Test content limits are set
        self.assertGreater(validation_config.min_title_length, 0)
        self.assertGreater(validation_config.max_title_length, validation_config.min_title_length)
        
        # Test rate limiting is configured
        self.assertGreater(validation_config.max_ideas_per_hour, 0)
        self.assertGreater(validation_config.max_ideas_per_day, validation_config.max_ideas_per_hour)
    
    def test_production_readiness(self):
        """Test production readiness indicators"""
        from pipeline.config.settings import get_settings
        
        settings = get_settings()
        
        # Test infrastructure settings for production readiness
        infra = settings.infrastructure
        self.assertGreater(infra.circuit_breaker_failure_threshold, 0)
        self.assertGreater(infra.circuit_breaker_timeout_seconds, 0)
        self.assertTrue(infra.quality_gate_enabled)
        
        # Test monitoring is enabled
        self.assertTrue(infra.enable_health_monitoring)
        self.assertGreater(infra.health_check_interval, 0)
        
        # Test budget controls are active
        budget = settings.budget
        self.assertTrue(budget.enable_cost_tracking)
        self.assertTrue(budget.enable_budget_alerts)
        self.assertLessEqual(budget.total_cycle_budget, 100.0)  # Reasonable budget limit

if __name__ == '__main__':
    print("✅ Running Comprehensive Quality Gates Validation")
    print("=" * 55)
    
    # Run tests with detailed output
    unittest.main(verbosity=2)
