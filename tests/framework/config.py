"""
Testing framework configuration management.

This module provides environment-driven configuration for the comprehensive testing
framework, supporting validation areas, execution modes, and integration settings.
"""

import os
from functools import lru_cache
from typing import Optional, List, Dict, Any, Union, Literal
from pathlib import Path
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict
import logging

logger = logging.getLogger(__name__)

# Pipeline integration imports
try:
    from pipeline.config.settings import get_settings as get_pipeline_settings
    PIPELINE_INTEGRATION_AVAILABLE = True
except ImportError:
    PIPELINE_INTEGRATION_AVAILABLE = False
    logger.warning("Pipeline configuration not available - running in standalone mode")


class ExecutionMode(str, Enum):
    """Test execution modes."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    SELECTIVE = "selective"


class ValidationArea(str, Enum):
    """Supported validation areas."""
    COMPONENT = "component"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    SECURITY = "security"
    DATA_FLOW = "data_flow"
    ERROR_HANDLING = "error_handling"
    MONITORING = "monitoring"
    OUTPUT_VERIFICATION = "output_verification"
    DATA_INTEGRITY = "data_integrity"
    BUSINESS_LOGIC = "business_logic"
    API_VALIDATION = "api_validation"
    END_TO_END = "end_to_end"


class ReportFormat(str, Enum):
    """Supported report formats."""
    HTML = "html"
    JSON = "json"
    MARKDOWN = "markdown"
    XML = "xml"


class ValidationStatus(str, Enum):
    """Status of validation execution."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    RUNNING = "running"


class ValidationSeverity(str, Enum):
    """Severity levels for validation results."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class TestFrameworkConfig(BaseSettings):
    """Core testing framework configuration."""
    
    model_config = SettingsConfigDict(
        env_prefix="TEST_",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        validate_assignment=True,
        use_enum_values=True
    )
    
    # Framework metadata
    framework_name: str = Field(default="agentic-testing-framework", alias="TEST_FRAMEWORK_NAME")
    version: str = Field(default="1.0.0", alias="TEST_FRAMEWORK_VERSION")
    
    # Environment settings
    environment: str = Field(default="test", alias="TEST_ENVIRONMENT")
    debug_mode: bool = Field(default=False, alias="TEST_DEBUG_MODE")
    
    # Execution configuration
    execution_mode: ExecutionMode = Field(default=ExecutionMode.SEQUENTIAL, alias="TEST_EXECUTION_MODE")
    max_parallel_workers: int = Field(default=4, alias="TEST_MAX_PARALLEL_WORKERS")
    test_timeout: int = Field(default=300, alias="TEST_TIMEOUT")  # seconds
    
    # Test discovery
    test_root_path: str = Field(default="tests", alias="TEST_ROOT_PATH")
    test_pattern: str = Field(default="test_*.py", alias="TEST_PATTERN")
    
    # Enabled validation areas
    enabled_validation_areas: Union[str, List[ValidationArea]] = Field(
        default_factory=lambda: list(ValidationArea),
        description="Areas to validate during testing"
    )
    
    @field_validator('environment')
    @classmethod
    def validate_environment(cls, v):
        """Validate test environment."""
        valid_envs = ['test', 'development', 'staging', 'ci']
        if isinstance(v, str) and v.lower() not in valid_envs:
            raise ValueError(f"Test environment must be one of: {valid_envs}")
        return v.lower() if isinstance(v, str) else v
    
    @field_validator('max_parallel_workers')
    @classmethod
    def validate_parallel_workers(cls, v):
        """Validate parallel worker count."""
        if not 1 <= v <= 16:
            raise ValueError("Parallel workers must be between 1 and 16")
        return v
    
    @field_validator('test_timeout')
    @classmethod
    def validate_timeout(cls, v):
        """Validate test timeout."""
        if not 30 <= v <= 3600:
            raise ValueError("Test timeout must be between 30 and 3600 seconds")
        return v
    
    @field_validator('enabled_validation_areas', mode='before')
    @classmethod
    def parse_validation_areas(cls, v):
        """Parse validation areas from environment."""
        if isinstance(v, str):
            # Handle comma-separated string from environment
            areas = [area.strip() for area in v.split(',') if area.strip()]
            return [ValidationArea(area) for area in areas]
        return v or list(ValidationArea)


class DataManagerConfig(BaseSettings):
    """Data management configuration."""
    
    # Test data paths
    test_data_root: str = Field(default="tests/data", env="TEST_DATA_ROOT")
    synthetic_data_cache: str = Field(default="tests/cache/synthetic", env="SYNTHETIC_DATA_CACHE")
    mock_data_path: str = Field(default="tests/mocks", env="MOCK_DATA_PATH")
    
    # Data generation
    enable_synthetic_data: bool = Field(default=True, env="ENABLE_SYNTHETIC_DATA")
    synthetic_data_seed: int = Field(default=42, env="SYNTHETIC_DATA_SEED")
    max_synthetic_records: int = Field(default=1000, env="MAX_SYNTHETIC_RECORDS")
    
    # Mock services
    enable_mock_services: bool = Field(default=True, env="ENABLE_MOCK_SERVICES")
    mock_service_port_range: str = Field(default="8000-8100", env="MOCK_SERVICE_PORT_RANGE")
    mock_service_timeout: int = Field(default=30, env="MOCK_SERVICE_TIMEOUT")
    
    # Data cleanup
    auto_cleanup: bool = Field(default=True, env="TEST_AUTO_CLEANUP")
    cleanup_timeout: int = Field(default=60, env="TEST_CLEANUP_TIMEOUT")
    preserve_on_failure: bool = Field(default=True, env="PRESERVE_DATA_ON_FAILURE")
    
    @field_validator('synthetic_data_seed')
    @classmethod
    def validate_seed(cls, v):
        """Validate synthetic data seed."""
        if v < 0:
            raise ValueError("Synthetic data seed must be non-negative")
        return v
    
    @field_validator('max_synthetic_records')
    @classmethod
    def validate_max_records(cls, v):
        """Validate maximum synthetic records."""
        if not 1 <= v <= 100000:
            raise ValueError("Max synthetic records must be between 1 and 100,000")
        return v
    
    @field_validator('mock_service_port_range')
    @classmethod
    def validate_port_range(cls, v):
        """Validate mock service port range."""
        try:
            start, end = v.split('-')
        except (ValueError, AttributeError):
            raise ValueError("Port range must be in format 'start-end' (e.g., '8000-8100')")
        
        try:
            start_port, end_port = int(start), int(end)
        except ValueError:
            raise ValueError("Port range must be in format 'start-end' (e.g., '8000-8100')")
        
        if not (1024 <= start_port <= end_port <= 65535):
            raise ValueError("Invalid port range")
        
        return v

    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        validate_assignment=True
    )


class ValidationEngineConfig(BaseSettings):
    """Validation engine configuration."""
    
    # Plugin system
    plugin_directories: List[str] = Field(
        default_factory=lambda: ["tests/framework/plugins"],
        env="VALIDATION_PLUGIN_DIRECTORIES"
    )
    auto_load_plugins: bool = Field(default=True, env="AUTO_LOAD_VALIDATION_PLUGINS")
    
    # Validation execution
    strict_validation: bool = Field(default=True, env="STRICT_VALIDATION")
    fail_fast: bool = Field(default=False, env="VALIDATION_FAIL_FAST")
    retry_attempts: int = Field(default=2, env="VALIDATION_RETRY_ATTEMPTS")
    retry_delay: float = Field(default=1.0, env="VALIDATION_RETRY_DELAY")
    
    # Performance validation
    performance_threshold_cpu: float = Field(default=80.0)  # percentage
    performance_threshold_memory: int = Field(default=512, env="PERFORMANCE_THRESHOLD_MEMORY")  # MB
    performance_threshold_latency: float = Field(default=1000.0, env="PERFORMANCE_THRESHOLD_LATENCY")  # ms
    
    # Security validation
    security_scan_timeout: int = Field(default=300, env="SECURITY_SCAN_TIMEOUT")
    enable_vulnerability_scanning: bool = Field(default=True, env="ENABLE_VULNERABILITY_SCANNING")
    security_severity_threshold: str = Field(default="medium", env="SECURITY_SEVERITY_THRESHOLD")
    
    @field_validator('retry_attempts')
    @classmethod
    def validate_retry_attempts(cls, v):
        """Validate retry attempts."""
        if not 0 <= v <= 10:
            raise ValueError("Retry attempts must be between 0 and 10")
        return v
    
    @field_validator('retry_delay')
    @classmethod
    def validate_retry_delay(cls, v):
        """Validate retry delay."""
        if not 0.1 <= v <= 30.0:
            raise ValueError("Retry delay must be between 0.1 and 30.0 seconds")
        return v
    
    @field_validator('performance_threshold_cpu')
    @classmethod
    def validate_cpu_threshold(cls, v):
        """Validate CPU threshold."""
        if not 0.0 <= v <= 100.0:
            raise ValueError("CPU threshold must be between 0.0 and 100.0")
        return v
    
    @field_validator('security_severity_threshold')
    @classmethod
    def validate_security_threshold(cls, v):
        """Validate security severity threshold."""
        valid_levels = ['low', 'medium', 'high', 'critical']
        if isinstance(v, str) and v.lower() not in valid_levels:
            raise ValueError(f"Security threshold must be one of: {valid_levels}")
        return v.lower() if isinstance(v, str) else v

    model_config = SettingsConfigDict(
        env_prefix="VALIDATION_",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        validate_assignment=True
    )


class ReportingConfig(BaseSettings):
    """Test reporting and monitoring configuration."""
    
    # Report generation
    report_output_dir: str = Field(default="tests/reports", env="TEST_REPORT_OUTPUT_DIR")
    report_formats: Union[str, List[ReportFormat]] = Field(
        default=[ReportFormat.HTML, ReportFormat.JSON],
        description="Report formats to generate"
    )
    include_performance_metrics: bool = Field(default=True, env="INCLUDE_PERFORMANCE_METRICS")
    include_coverage_report: bool = Field(default=True, env="INCLUDE_COVERAGE_REPORT")
    
    # Real-time monitoring
    enable_real_time_monitoring: bool = Field(default=True, env="ENABLE_REAL_TIME_MONITORING")
    monitoring_interval: int = Field(default=5, env="MONITORING_INTERVAL")  # seconds
    
    # External integrations
    grafana_enabled: bool = Field(default=False, env="GRAFANA_INTEGRATION_ENABLED")
    grafana_url: Optional[str] = Field(default=None, env="GRAFANA_URL")
    grafana_api_key: Optional[str] = Field(default=None, env="GRAFANA_API_KEY")
    
    prometheus_enabled: bool = Field(default=False, env="PROMETHEUS_INTEGRATION_ENABLED")
    prometheus_gateway_url: Optional[str] = Field(default=None, env="PROMETHEUS_GATEWAY_URL")
    
    # Alerting
    enable_alerts: bool = Field(default=True, env="ENABLE_TEST_ALERTS")
    alert_on_failure: bool = Field(default=True, env="ALERT_ON_FAILURE")
    alert_on_performance_regression: bool = Field(default=True, env="ALERT_ON_PERFORMANCE_REGRESSION")
    
    @field_validator('monitoring_interval')
    @classmethod
    def validate_monitoring_interval(cls, v):
        """Validate monitoring interval."""
        if not 1 <= v <= 60:
            raise ValueError("Monitoring interval must be between 1 and 60 seconds")
        return v
    
    @field_validator('report_formats', mode='before')
    @classmethod
    def parse_report_formats(cls, v):
        """Parse report formats from environment."""
        if isinstance(v, str):
            return [ReportFormat(fmt.strip()) for fmt in v.split(',') if fmt.strip()]
        return v or [ReportFormat.HTML, ReportFormat.JSON]

    model_config = SettingsConfigDict(
        env_prefix="TEST_",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        validate_assignment=True,
        env_parse_none_str=None  # Prevent automatic JSON parsing of env vars
    )


class SecurityConfig(BaseSettings):
    """Security and compliance configuration."""
    
    # Environment isolation
    enable_container_isolation: bool = Field(default=True, env="ENABLE_CONTAINER_ISOLATION")
    container_runtime: str = Field(default="docker", env="CONTAINER_RUNTIME")
    
    # Data security
    enable_data_anonymization: bool = Field(default=True, env="ENABLE_DATA_ANONYMIZATION")
    sensitive_data_patterns: List[str] = Field(
        default_factory=lambda: [r'\b\d{3}-\d{2}-\d{4}\b', r'\b\d{16}\b'],  # SSN, credit card
        env="SENSITIVE_DATA_PATTERNS"
    )
    
    # Access control
    require_authentication: bool = Field(default=False, env="TEST_REQUIRE_AUTHENTICATION")
    allowed_users: List[str] = Field(default_factory=list, env="TEST_ALLOWED_USERS")
    
    # Audit trail
    enable_audit_logging: bool = Field(default=True, env="ENABLE_AUDIT_LOGGING")
    audit_log_path: str = Field(default="tests/audit/audit.log", env="AUDIT_LOG_PATH")
    audit_retention_days: int = Field(default=90, env="AUDIT_RETENTION_DAYS")
    
    @field_validator('container_runtime')
    @classmethod
    def validate_container_runtime(cls, v):
        """Validate container runtime."""
        valid_runtimes = ['docker', 'podman', 'containerd']
        if isinstance(v, str) and v.lower() not in valid_runtimes:
            raise ValueError(f"Container runtime must be one of: {valid_runtimes}")
        return v.lower() if isinstance(v, str) else v
    
    @field_validator('audit_retention_days')
    @classmethod
    def validate_retention_days(cls, v):
        """Validate audit retention period."""
        if not 1 <= v <= 365:
            raise ValueError("Audit retention must be between 1 and 365 days")
        return v

    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        validate_assignment=True
    )


class TestingFrameworkSettings(BaseSettings):
    """Main testing framework configuration combining all sub-configurations."""
    
    # Core framework settings
    framework: TestFrameworkConfig = Field(default_factory=TestFrameworkConfig)
    data_manager: DataManagerConfig = Field(default_factory=DataManagerConfig)
    validation_engine: ValidationEngineConfig = Field(default_factory=ValidationEngineConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    
    # Integration with existing pipeline
    pipeline_integration: bool = Field(default=True, env="ENABLE_PIPELINE_INTEGRATION")
    pipeline_config_path: str = Field(default="pipeline/config/settings.py", env="PIPELINE_CONFIG_PATH")
    
    def __init__(self, **kwargs):
        """Initialize with nested configurations that read environment variables."""
        super().__init__(**kwargs)
        
        # Initialize pipeline integration if available
        self._pipeline_config = None
        self._initialize_pipeline_integration()
    
    def _initialize_pipeline_integration(self):
        """Initialize pipeline configuration integration."""
        if not self.pipeline_integration or not PIPELINE_INTEGRATION_AVAILABLE:
            logger.info("Pipeline integration disabled or unavailable")
            return
            
        try:
            self._pipeline_config = get_pipeline_settings()
            logger.info("Pipeline configuration integrated successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize pipeline integration: {e}")
            self._pipeline_config = None
    
    def get_pipeline_config(self):
        """Get pipeline configuration if available."""
        return self._pipeline_config
    
    def get_shared_environment(self) -> str:
        """Get shared environment setting from pipeline or framework."""
        if self._pipeline_config:
            return self._pipeline_config.environment
        return self.framework.environment
    
    def get_shared_database_config(self):
        """Get database configuration, preferring pipeline if available."""
        if self._pipeline_config:
            return self._pipeline_config.database
        return None
    
    def is_pipeline_integrated(self) -> bool:
        """Check if pipeline integration is active."""
        return self._pipeline_config is not None
    
    @field_validator('pipeline_config_path')
    @classmethod
    def validate_pipeline_config_path(cls, v):
        """Validate pipeline configuration path exists."""
        if not Path(v).exists():
            logger.warning(f"Pipeline config path does not exist: {v}")
        return v

    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False,
        env_nested_delimiter="__",  # Support nested config like FRAMEWORK__DEBUG_MODE
        env_file=".env",
        env_file_encoding="utf-8",
        validate_assignment=True
    )


@lru_cache()
def get_framework_settings() -> TestingFrameworkSettings:
    """
    Get cached testing framework settings.
    
    This function is cached to ensure settings are loaded only once
    and reused throughout the framework lifecycle.
    """
    try:
        settings = TestingFrameworkSettings()
        logger.info(
            f"Loaded testing framework configuration for environment: {settings.framework.environment}",
            extra={
                "environment": settings.framework.environment,
                "execution_mode": settings.framework.execution_mode,
                "enabled_areas": len(settings.framework.enabled_validation_areas),
                "debug_mode": settings.framework.debug_mode
            }
        )
        return settings
    except Exception as e:
        logger.error(f"Failed to load testing framework configuration: {e}")
        raise


def validate_framework_environment() -> None:
    """
    Validate testing framework environment setup.
    
    Ensures all required directories exist and permissions are correct.
    """
    settings = get_framework_settings()
    
    # Create required directories
    required_dirs = [
        settings.data_manager.test_data_root,
        settings.data_manager.synthetic_data_cache,
        settings.data_manager.mock_data_path,
        settings.reporting.report_output_dir,
        Path(settings.security.audit_log_path).parent
    ]
    
    for dir_path in required_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {dir_path}")
    
    # Validate integrations
    if settings.reporting.grafana_enabled and not settings.reporting.grafana_url:
        raise ValueError("Grafana URL required when Grafana integration is enabled")
    
    if settings.reporting.prometheus_enabled and not settings.reporting.prometheus_gateway_url:
        raise ValueError("Prometheus gateway URL required when Prometheus integration is enabled")
# Factory functions for accessing individual configurations
@lru_cache()
def get_framework_config() -> TestFrameworkConfig:
    """Get cached framework configuration."""
    return get_framework_settings().framework


@lru_cache()
def get_data_manager_config() -> DataManagerConfig:
    """Get cached data manager configuration."""
    return get_framework_settings().data_manager


@lru_cache()
def get_validation_engine_config() -> ValidationEngineConfig:
    """Get cached validation engine configuration."""
    return get_framework_settings().validation_engine


@lru_cache()
def get_reporting_config() -> ReportingConfig:
    """Get cached reporting configuration."""
    return get_framework_settings().reporting


@lru_cache()
def get_security_config() -> SecurityConfig:
    """Get cached security configuration."""
    return get_framework_settings().security


def get_framework_summary() -> Dict[str, Any]:
    """
    Get a summary of current framework configuration for logging/debugging.
    
    Sensitive values are masked for security.
    """
    settings = get_framework_settings()
    
    return {
        "framework": {
            "name": settings.framework.framework_name,
            "version": settings.framework.version,
            "environment": settings.framework.environment,
            "execution_mode": settings.framework.execution_mode,
            "enabled_areas": [area.value for area in settings.framework.enabled_validation_areas]
        },
        "data_management": {
            "synthetic_data_enabled": settings.data_manager.enable_synthetic_data,
            "mock_services_enabled": settings.data_manager.enable_mock_services,
            "auto_cleanup": settings.data_manager.auto_cleanup
        },
        "validation": {
            "strict_mode": settings.validation_engine.strict_validation,
            "fail_fast": settings.validation_engine.fail_fast,
            "retry_attempts": settings.validation_engine.retry_attempts
        },
        "reporting": {
            "formats": [fmt.value for fmt in settings.reporting.report_formats],
            "real_time_monitoring": settings.reporting.enable_real_time_monitoring,
            "grafana_enabled": settings.reporting.grafana_enabled,
            "prometheus_enabled": settings.reporting.prometheus_enabled
        },
        "security": {
            "container_isolation": settings.security.enable_container_isolation,
            "data_anonymization": settings.security.enable_data_anonymization,
            "audit_logging": settings.security.enable_audit_logging
        }
    }


# Export convenience functions for accessing specific configurations
def get_framework_config() -> TestFrameworkConfig:
    """Get framework configuration."""
    return get_framework_settings().framework


def get_data_manager_config() -> DataManagerConfig:
    """Get data manager configuration."""
    return get_framework_settings().data_manager


def get_validation_engine_config() -> ValidationEngineConfig:
    """Get validation engine configuration."""
    return get_framework_settings().validation_engine


def get_reporting_config() -> ReportingConfig:
    """Get reporting configuration."""
    return get_framework_settings().reporting


def get_security_config() -> SecurityConfig:
    """Get security configuration."""
    return get_framework_settings().security