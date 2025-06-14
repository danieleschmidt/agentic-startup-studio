"""
Tests for the testing framework configuration system.

Following TDD principles with comprehensive coverage of:
- ValidationConfig validation and defaults
- Environment variable loading and fallbacks
- SecurityConfig validation (encryption, audit logging)
- ReportingConfig validation (output formats, destinations)
- Factory function testing
- Invalid configuration handling
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from tests.framework.config import (
    TestFrameworkConfig,
    DataManagerConfig, 
    ValidationEngineConfig,
    ReportingConfig,
    SecurityConfig,
    TestingFrameworkSettings,
    ValidationArea,
    ExecutionMode,
    ReportFormat,
    get_framework_settings,
    get_framework_config,
    get_data_manager_config,
    get_validation_engine_config,
    get_reporting_config,
    get_security_config,
    validate_framework_environment,
    get_framework_summary
)


class TestTestFrameworkConfig:
    """Test TestFrameworkConfig validation and defaults."""
    
    def test_should_use_default_values_when_no_environment_variables_set(self):
        """Given no environment variables are set
        When creating TestFrameworkConfig
        Then should use default values"""
        config = TestFrameworkConfig()
        
        assert config.framework_name == "agentic-testing-framework"
        assert config.version == "1.0.0"
        assert config.environment == "test"
        assert config.debug_mode is False
        assert config.execution_mode == ExecutionMode.SEQUENTIAL
        assert config.max_parallel_workers == 4
        assert config.test_timeout == 300
        assert config.test_root_path == "tests"
        assert config.test_pattern == "test_*.py"
        assert config.enabled_validation_areas == list(ValidationArea)
    
    def test_should_load_values_from_environment_variables(self):
        """Given environment variables are set
        When creating TestFrameworkConfig
        Then should use environment values"""
        with patch.dict(os.environ, {
            'TEST_FRAMEWORK_NAME': 'custom-framework',
            'TEST_FRAMEWORK_VERSION': '2.0.0',
            'TEST_ENVIRONMENT': 'staging',
            'TEST_DEBUG_MODE': 'true',
            'TEST_EXECUTION_MODE': 'parallel',
            'TEST_MAX_PARALLEL_WORKERS': '8',
            'TEST_TIMEOUT': '600'
        }):
            config = TestFrameworkConfig()
            
            assert config.framework_name == 'custom-framework'
            assert config.version == '2.0.0'
            assert config.environment == 'staging'
            assert config.debug_mode is True
            assert config.execution_mode == ExecutionMode.PARALLEL
            assert config.max_parallel_workers == 8
            assert config.test_timeout == 600
    
    def test_should_validate_environment_values(self):
        """Given invalid environment value
        When creating TestFrameworkConfig
        Then should raise ValueError"""
        with patch.dict(os.environ, {'TEST_ENVIRONMENT': 'invalid'}):
            with pytest.raises(ValueError, match="Test environment must be one of"):
                TestFrameworkConfig()
    
    def test_should_validate_parallel_workers_range(self):
        """Given invalid parallel workers count
        When creating TestFrameworkConfig
        Then should raise ValueError"""
        with patch.dict(os.environ, {'TEST_MAX_PARALLEL_WORKERS': '0'}):
            with pytest.raises(ValueError, match="Parallel workers must be between 1 and 16"):
                TestFrameworkConfig()
                
        with patch.dict(os.environ, {'TEST_MAX_PARALLEL_WORKERS': '20'}):
            with pytest.raises(ValueError, match="Parallel workers must be between 1 and 16"):
                TestFrameworkConfig()
    
    def test_should_validate_timeout_range(self):
        """Given invalid timeout value
        When creating TestFrameworkConfig
        Then should raise ValueError"""
        with patch.dict(os.environ, {'TEST_TIMEOUT': '10'}):
            with pytest.raises(ValueError, match="Test timeout must be between 30 and 3600 seconds"):
                TestFrameworkConfig()
    
    def test_should_parse_validation_areas_from_environment(self):
        """Given comma-separated validation areas in environment
        When creating TestFrameworkConfig
        Then should parse into list of ValidationArea enums"""
        with patch.dict(os.environ, {
            'TEST_ENABLED_VALIDATION_AREAS': 'component,integration,performance'
        }):
            config = TestFrameworkConfig()
            
            expected_areas = [
                ValidationArea.COMPONENT,
                ValidationArea.INTEGRATION, 
                ValidationArea.PERFORMANCE
            ]
            assert config.enabled_validation_areas == expected_areas


class TestDataManagerConfig:
    """Test DataManagerConfig validation and defaults."""
    
    def test_should_use_default_values_when_no_environment_variables_set(self):
        """Given no environment variables are set
        When creating DataManagerConfig
        Then should use default values"""
        config = DataManagerConfig()
        
        assert config.test_data_root == "tests/data"
        assert config.synthetic_data_cache == "tests/cache/synthetic"
        assert config.mock_data_path == "tests/mocks"
        assert config.enable_synthetic_data is True
        assert config.synthetic_data_seed == 42
        assert config.max_synthetic_records == 1000
        assert config.enable_mock_services is True
        assert config.mock_service_port_range == "8000-8100"
        assert config.auto_cleanup is True
        assert config.preserve_on_failure is True
    
    def test_should_validate_synthetic_data_seed(self):
        """Given negative seed value
        When creating DataManagerConfig
        Then should raise ValueError"""
        with patch.dict(os.environ, {'SYNTHETIC_DATA_SEED': '-1'}):
            with pytest.raises(ValueError, match="Synthetic data seed must be non-negative"):
                DataManagerConfig()
    
    def test_should_validate_max_records_range(self):
        """Given invalid max records value
        When creating DataManagerConfig
        Then should raise ValueError"""
        with patch.dict(os.environ, {'MAX_SYNTHETIC_RECORDS': '0'}):
            with pytest.raises(ValueError, match="Max synthetic records must be between 1 and 100,000"):
                DataManagerConfig()
                
        with patch.dict(os.environ, {'MAX_SYNTHETIC_RECORDS': '200000'}):
            with pytest.raises(ValueError, match="Max synthetic records must be between 1 and 100,000"):
                DataManagerConfig()
    
    def test_should_validate_port_range_format(self):
        """Given invalid port range format
        When creating DataManagerConfig
        Then should raise ValueError"""
        with patch.dict(os.environ, {'MOCK_SERVICE_PORT_RANGE': 'invalid'}):
            with pytest.raises(ValueError, match="Port range must be in format 'start-end'"):
                DataManagerConfig()
        
        with patch.dict(os.environ, {'MOCK_SERVICE_PORT_RANGE': '9000-8000'}):
            with pytest.raises(ValueError, match="Invalid port range"):
                DataManagerConfig()


class TestValidationEngineConfig:
    """Test ValidationEngineConfig validation and defaults."""
    
    def test_should_use_default_values_when_no_environment_variables_set(self):
        """Given no environment variables are set
        When creating ValidationEngineConfig
        Then should use default values"""
        config = ValidationEngineConfig()
        
        assert config.plugin_directories == ["tests/framework/plugins"]
        assert config.auto_load_plugins is True
        assert config.strict_validation is True
        assert config.fail_fast is False
        assert config.retry_attempts == 2
        assert config.retry_delay == 1.0
        assert config.performance_threshold_cpu == 80.0
        assert config.performance_threshold_memory == 512
        assert config.performance_threshold_latency == 1000.0
        assert config.security_scan_timeout == 300
        assert config.enable_vulnerability_scanning is True
        assert config.security_severity_threshold == "medium"
    
    def test_should_validate_retry_attempts_range(self):
        """Given invalid retry attempts value
        When creating ValidationEngineConfig
        Then should raise ValueError"""
        with patch.dict(os.environ, {'VALIDATION_RETRY_ATTEMPTS': '-1'}):
            with pytest.raises(ValueError, match="Retry attempts must be between 0 and 10"):
                ValidationEngineConfig()
    
    def test_should_validate_retry_delay_range(self):
        """Given invalid retry delay value
        When creating ValidationEngineConfig
        Then should raise ValueError"""
        with patch.dict(os.environ, {'VALIDATION_RETRY_DELAY': '0.05'}):
            with pytest.raises(ValueError, match="Retry delay must be between 0.1 and 30.0 seconds"):
                ValidationEngineConfig()
    
    def test_should_validate_cpu_threshold_range(self):
        """Given invalid CPU threshold value
        When creating ValidationEngineConfig
        Then should raise ValueError"""
        with patch.dict(os.environ, {'VALIDATION_PERFORMANCE_THRESHOLD_CPU': '150.0'}):
            with pytest.raises(ValueError, match="CPU threshold must be between 0.0 and 100.0"):
                ValidationEngineConfig()
    
    def test_should_validate_security_severity_threshold(self):
        """Given invalid security severity threshold
        When creating ValidationEngineConfig
        Then should raise ValueError"""
        with patch.dict(os.environ, {'VALIDATION_SECURITY_SEVERITY_THRESHOLD': 'invalid'}):
            with pytest.raises(ValueError, match="Security threshold must be one of"):
                ValidationEngineConfig()


class TestReportingConfig:
    """Test ReportingConfig validation and defaults."""
    
    def test_should_use_default_values_when_no_environment_variables_set(self):
        """Given no environment variables are set
        When creating ReportingConfig
        Then should use default values"""
        config = ReportingConfig()
        
        assert config.report_output_dir == "tests/reports"
        assert config.report_formats == [ReportFormat.HTML, ReportFormat.JSON]
        assert config.include_performance_metrics is True
        assert config.include_coverage_report is True
        assert config.enable_real_time_monitoring is True
        assert config.monitoring_interval == 5
        assert config.grafana_enabled is False
        assert config.prometheus_enabled is False
        assert config.enable_alerts is True
    
    def test_should_validate_monitoring_interval_range(self):
        """Given invalid monitoring interval
        When creating ReportingConfig
        Then should raise ValueError"""
        with patch.dict(os.environ, {'TEST_MONITORING_INTERVAL': '0'}):
            with pytest.raises(ValueError, match="Monitoring interval must be between 1 and 60 seconds"):
                ReportingConfig()
    
    def test_should_parse_report_formats_from_environment(self):
        """Given comma-separated report formats in environment
        When creating ReportingConfig
        Then should parse into list of ReportFormat enums"""
        with patch.dict(os.environ, {'TEST_REPORT_FORMATS': 'json,markdown'}):
            config = ReportingConfig()
            
            expected_formats = [ReportFormat.JSON, ReportFormat.MARKDOWN]
            assert config.report_formats == expected_formats


class TestSecurityConfig:
    """Test SecurityConfig validation and defaults."""
    
    def test_should_use_default_values_when_no_environment_variables_set(self):
        """Given no environment variables are set
        When creating SecurityConfig
        Then should use default values"""
        config = SecurityConfig()
        
        assert config.enable_container_isolation is True
        assert config.container_runtime == "docker"
        assert config.enable_data_anonymization is True
        assert len(config.sensitive_data_patterns) == 2  # SSN, credit card
        assert config.require_authentication is False
        assert config.allowed_users == []
        assert config.enable_audit_logging is True
        assert config.audit_log_path == "tests/audit/audit.log"
        assert config.audit_retention_days == 90
    
    def test_should_validate_container_runtime(self):
        """Given invalid container runtime
        When creating SecurityConfig
        Then should raise ValueError"""
        with patch.dict(os.environ, {'CONTAINER_RUNTIME': 'invalid'}):
            with pytest.raises(ValueError, match="Container runtime must be one of"):
                SecurityConfig()
    
    def test_should_validate_audit_retention_days_range(self):
        """Given invalid audit retention days
        When creating SecurityConfig
        Then should raise ValueError"""
        with patch.dict(os.environ, {'AUDIT_RETENTION_DAYS': '0'}):
            with pytest.raises(ValueError, match="Audit retention must be between 1 and 365 days"):
                SecurityConfig()


class TestTestingFrameworkSettings:
    """Test main TestingFrameworkSettings integration."""
    
    def test_should_create_all_sub_configurations(self):
        """Given TestingFrameworkSettings creation
        When initializing
        Then should create all sub-configuration objects"""
        settings = TestingFrameworkSettings()
        
        assert isinstance(settings.framework, TestFrameworkConfig)
        assert isinstance(settings.data_manager, DataManagerConfig)
        assert isinstance(settings.validation_engine, ValidationEngineConfig)
        assert isinstance(settings.reporting, ReportingConfig)
        assert isinstance(settings.security, SecurityConfig)
        assert settings.pipeline_integration is True
    
    @patch('tests.framework.config.Path')
    def test_should_warn_when_pipeline_config_path_missing(self, mock_path):
        """Given pipeline config path does not exist
        When creating TestingFrameworkSettings
        Then should log warning but continue"""
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = False
        mock_path.return_value = mock_path_instance
        
        with patch('tests.framework.config.logger') as mock_logger:
            settings = TestingFrameworkSettings()
            mock_logger.warning.assert_called_once()


class TestFactoryFunctions:
    """Test factory functions for accessing configurations."""
    
    @patch('tests.framework.config.TestingFrameworkSettings')
    def test_get_framework_settings_should_be_cached(self, mock_settings_class):
        """Given multiple calls to get_framework_settings
        When called multiple times
        Then should return same cached instance"""
        # Clear cache first
        get_framework_settings.cache_clear()
        
        mock_instance = Mock()
        # Set up mock structure for logging calls
        mock_instance.framework.environment = "test"
        mock_instance.framework.execution_mode = "sequential"
        mock_instance.framework.enabled_validation_areas = []  # Empty list for len() call
        mock_instance.framework.debug_mode = False
        mock_settings_class.return_value = mock_instance
        
        result1 = get_framework_settings()
        result2 = get_framework_settings()
        
        assert result1 is result2
        mock_settings_class.assert_called_once()
    
    @patch('tests.framework.config.get_framework_settings')
    def test_get_framework_config_should_return_framework_attribute(self, mock_get_settings):
        """Given call to get_framework_config
        When called
        Then should return framework attribute from settings"""
        mock_settings = Mock()
        mock_framework_config = Mock()
        mock_settings.framework = mock_framework_config
        mock_get_settings.return_value = mock_settings
        
        result = get_framework_config()
        
        assert result is mock_framework_config
    
    @patch('tests.framework.config.get_framework_settings')
    def test_get_data_manager_config_should_return_data_manager_attribute(self, mock_get_settings):
        """Given call to get_data_manager_config
        When called
        Then should return data_manager attribute from settings"""
        mock_settings = Mock()
        mock_data_manager_config = Mock()
        mock_settings.data_manager = mock_data_manager_config
        mock_get_settings.return_value = mock_settings
        
        result = get_data_manager_config()
        
        assert result is mock_data_manager_config


class TestEnvironmentValidation:
    """Test environment validation functions."""
    
    @patch('tests.framework.config.Path')
    @patch('tests.framework.config.get_framework_settings')
    def test_validate_framework_environment_should_create_required_directories(
        self, mock_get_settings, mock_path_class
    ):
        """Given validate_framework_environment call
        When called
        Then should create all required directories"""
        # Setup mocks
        mock_settings = Mock()
        mock_settings.data_manager.test_data_root = "tests/data"
        mock_settings.data_manager.synthetic_data_cache = "tests/cache"
        mock_settings.data_manager.mock_data_path = "tests/mocks"
        mock_settings.reporting.report_output_dir = "tests/reports"
        mock_settings.security.audit_log_path = "tests/audit/audit.log"
        mock_settings.reporting.grafana_enabled = False
        mock_settings.reporting.prometheus_enabled = False
        mock_get_settings.return_value = mock_settings
        
        mock_path_instances = []
        def create_mock_path(*args, **kwargs):
            mock_instance = Mock()
            mock_path_instances.append(mock_instance)
            return mock_instance
        mock_path_class.side_effect = create_mock_path
        
        validate_framework_environment()
        
        # Verify directories were created
        for mock_instance in mock_path_instances:
            mock_instance.mkdir.assert_called_with(parents=True, exist_ok=True)
    
    @patch('tests.framework.config.get_framework_settings')
    def test_validate_framework_environment_should_raise_error_for_missing_grafana_url(
        self, mock_get_settings
    ):
        """Given Grafana enabled but no URL provided
        When validating environment
        Then should raise ValueError"""
        mock_settings = Mock()
        mock_settings.data_manager.test_data_root = "tests/data"
        mock_settings.data_manager.synthetic_data_cache = "tests/cache"
        mock_settings.data_manager.mock_data_path = "tests/mocks"
        mock_settings.reporting.report_output_dir = "tests/reports"
        mock_settings.security.audit_log_path = "tests/audit/audit.log"
        mock_settings.reporting.grafana_enabled = True
        mock_settings.reporting.grafana_url = None
        mock_settings.reporting.prometheus_enabled = False
        mock_get_settings.return_value = mock_settings
        
        with patch('tests.framework.config.Path'):
            with pytest.raises(ValueError, match="Grafana URL required"):
                validate_framework_environment()


class TestFrameworkSummary:
    """Test framework summary function."""
    
    @patch('tests.framework.config.get_framework_settings')
    def test_get_framework_summary_should_return_comprehensive_summary(self, mock_get_settings):
        """Given call to get_framework_summary
        When called
        Then should return comprehensive configuration summary"""
        # Setup detailed mock settings
        mock_settings = Mock()
        
        # Framework config
        mock_settings.framework.framework_name = "test-framework"
        mock_settings.framework.version = "1.0.0"
        mock_settings.framework.environment = "test"
        mock_settings.framework.execution_mode = ExecutionMode.PARALLEL
        mock_settings.framework.enabled_validation_areas = [
            ValidationArea.COMPONENT, ValidationArea.INTEGRATION
        ]
        
        # Data manager config
        mock_settings.data_manager.enable_synthetic_data = True
        mock_settings.data_manager.enable_mock_services = True
        mock_settings.data_manager.auto_cleanup = True
        
        # Validation engine config
        mock_settings.validation_engine.strict_validation = True
        mock_settings.validation_engine.fail_fast = False
        mock_settings.validation_engine.retry_attempts = 2
        
        # Reporting config
        mock_settings.reporting.report_formats = [ReportFormat.HTML, ReportFormat.JSON]
        mock_settings.reporting.enable_real_time_monitoring = True
        mock_settings.reporting.grafana_enabled = False
        mock_settings.reporting.prometheus_enabled = False
        
        # Security config
        mock_settings.security.enable_container_isolation = True
        mock_settings.security.enable_data_anonymization = True
        mock_settings.security.enable_audit_logging = True
        
        mock_get_settings.return_value = mock_settings
        
        summary = get_framework_summary()
        
        # Verify summary structure and content
        assert "framework" in summary
        assert summary["framework"]["name"] == "test-framework"
        assert summary["framework"]["execution_mode"] == ExecutionMode.PARALLEL
        assert summary["framework"]["enabled_areas"] == ["component", "integration"]
        
        assert "data_management" in summary
        assert summary["data_management"]["synthetic_data_enabled"] is True
        
        assert "validation" in summary
        assert summary["validation"]["strict_mode"] is True
        
        assert "reporting" in summary
        assert summary["reporting"]["formats"] == ["html", "json"]
        
        assert "security" in summary
        assert summary["security"]["audit_logging"] is True