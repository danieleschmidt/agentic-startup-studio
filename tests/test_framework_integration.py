"""
Integration tests for the testing framework components.

This module validates the integration between testing framework components,
pipeline configuration, and end-to-end validation workflows.
"""

import pytest
import asyncio
from unittest.mock import patch, Mock
from pathlib import Path

from tests.framework.config import (
    get_framework_settings,
    TestingFrameworkSettings,
    ValidationArea,
    ExecutionMode,
    PIPELINE_INTEGRATION_AVAILABLE
)
from tests.framework.data_manager import DataManager
from tests.framework.validation_engine import ValidationEngine, ValidationContext
from tests.framework.test_runner import TestRunner, TestRunConfiguration


class TestFrameworkIntegration:
    """Test framework component integration."""
    
    def test_framework_settings_initialization(self):
        """Test framework settings load correctly."""
        settings = get_framework_settings()
        
        assert settings is not None
        assert settings.framework is not None
        assert settings.data_manager is not None
        assert settings.validation_engine is not None
        assert settings.reporting is not None
        assert settings.security is not None
        
    def test_pipeline_integration_availability(self):
        """Test pipeline integration detection."""
        settings = get_framework_settings()
        
        # Pipeline integration should be detected
        if PIPELINE_INTEGRATION_AVAILABLE:
            assert settings.is_pipeline_integrated() or not settings.pipeline_integration
        else:
            assert not settings.is_pipeline_integrated()
    
    def test_shared_environment_configuration(self):
        """Test shared environment configuration."""
        settings = get_framework_settings()
        environment = settings.get_shared_environment()
        
        assert environment in ['test', 'development', 'staging', 'production']
    
    def test_validation_areas_configuration(self):
        """Test validation areas are properly configured."""
        settings = get_framework_settings()
        areas = settings.framework.enabled_validation_areas
        
        assert len(areas) > 0
        assert ValidationArea.COMPONENT in areas
        assert ValidationArea.INTEGRATION in areas
    
    def test_data_manager_initialization(self):
        """Test data manager initializes correctly."""
        settings = get_framework_settings()
        data_manager = DataManager(settings.data_manager)
        
        assert data_manager is not None
        assert data_manager.config is not None
        assert data_manager.synthetic_generator is not None
    
    def test_validation_engine_initialization(self):
        """Test validation engine initializes correctly."""
        settings = get_framework_settings()
        validation_engine = ValidationEngine(settings.validation_engine)
        
        assert validation_engine is not None
        registered_validators = validation_engine.get_registered_validators()
        
        # Should have validators for each area
        assert len(registered_validators) > 0
        assert ValidationArea.COMPONENT in registered_validators
        assert ValidationArea.INTEGRATION in registered_validators
    
    def test_test_runner_initialization(self):
        """Test runner initializes correctly."""
        settings = get_framework_settings()
        test_runner = TestRunner(settings)
        
        assert test_runner is not None
        assert test_runner.settings is not None
        assert test_runner.validation_engine is not None
        assert test_runner.data_manager is not None
        assert test_runner.dependency_manager is not None
    
    @pytest.mark.asyncio
    async def test_end_to_end_validation_workflow(self):
        """Test complete validation workflow."""
        # Create test configuration
        config = TestRunConfiguration(
            areas=[ValidationArea.COMPONENT],
            execution_mode=ExecutionMode.SEQUENTIAL,
            timeout=30.0,
            fail_fast=True
        )
        
        # Initialize test runner
        settings = get_framework_settings()
        test_runner = TestRunner(settings)
        
        # Run validation (with minimal setup)
        result = await test_runner.run_tests(config)
        
        assert result is not None
        assert result.run_id == config.run_id
        assert result.configuration == config
        # Status should be completed, failed, or error (not pending)
        assert result.status.value in ['completed', 'failed', 'error']


class TestPipelineIntegration:
    """Test pipeline integration functionality."""
    
    def test_pipeline_config_detection(self):
        """Test pipeline configuration detection."""
        settings = get_framework_settings()
        
        if settings.is_pipeline_integrated():
            pipeline_config = settings.get_pipeline_config()
            assert pipeline_config is not None
            assert hasattr(pipeline_config, 'environment')
            assert hasattr(pipeline_config, 'database')
    
    def test_shared_database_config(self):
        """Test shared database configuration."""
        settings = get_framework_settings()
        db_config = settings.get_shared_database_config()
        
        if settings.is_pipeline_integrated():
            assert db_config is not None
            assert hasattr(db_config, 'host')
            assert hasattr(db_config, 'port')
        else:
            assert db_config is None
    
    def test_graceful_fallback_when_pipeline_unavailable(self):
        """Test graceful fallback when pipeline is unavailable."""
        with patch('tests.framework.config.PIPELINE_INTEGRATION_AVAILABLE', False):
            settings = TestingFrameworkSettings()
            
            assert not settings.is_pipeline_integrated()
            assert settings.get_pipeline_config() is None
            assert settings.get_shared_database_config() is None
            # Should still return framework environment
            assert settings.get_shared_environment() == settings.framework.environment


class TestDataManagerIntegration:
    """Test data manager integration."""
    
    def test_synthetic_data_generation(self):
        """Test synthetic data generation."""
        settings = get_framework_settings()
        data_manager = DataManager(settings.data_manager)
        
        # Generate startup ideas
        dataset = data_manager.generate_synthetic_data('startup_ideas', count=5)
        
        assert dataset is not None
        assert dataset.name == 'synthetic_startup_ideas_5'
        assert 'records' in dataset.data
        assert len(dataset.data['records']) == 5
        
        # Verify startup idea structure
        idea = dataset.data['records'][0]
        assert 'title' in idea
        assert 'description' in idea
        assert 'category' in idea
    
    def test_user_data_anonymization(self):
        """Test user data anonymization."""
        settings = get_framework_settings()
        data_manager = DataManager(settings.data_manager)
        
        # Generate anonymized user data
        dataset = data_manager.generate_synthetic_data('users', count=3)
        
        assert dataset is not None
        user = dataset.data['records'][0]
        
        # Check anonymization
        assert '@example.com' in user['email']
        assert user['first_name'].startswith('User')
        assert user['last_name'] == 'Test'
        assert user['phone'] == '555-0000'


class TestValidationEngineIntegration:
    """Test validation engine integration."""
    
    @pytest.mark.asyncio
    async def test_component_validation_execution(self):
        """Test component validation execution."""
        settings = get_framework_settings()
        validation_engine = ValidationEngine(settings.validation_engine)
        
        # Create validation context
        context = ValidationContext(
            test_data={'test_modules': []},
            timeout=10.0
        )
        
        # Execute component validation
        results = await validation_engine.execute_validation(
            ValidationArea.COMPONENT, 
            context
        )
        
        assert results is not None
        assert len(results) > 0
        
        result = results[0]
        assert result.area == ValidationArea.COMPONENT
        assert result.validation_id is not None
        assert result.status.value in ['passed', 'failed', 'skipped', 'error']
    
    @pytest.mark.asyncio
    async def test_integration_validation_execution(self):
        """Test integration validation execution."""
        settings = get_framework_settings()
        validation_engine = ValidationEngine(settings.validation_engine)
        
        # Create validation context
        context = ValidationContext(
            test_data={'integration_tests': []},
            dependencies=[],
            timeout=10.0
        )
        
        # Execute integration validation
        results = await validation_engine.execute_validation(
            ValidationArea.INTEGRATION, 
            context
        )
        
        assert results is not None
        assert len(results) > 0
        
        result = results[0]
        assert result.area == ValidationArea.INTEGRATION
        assert result.validation_id is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])