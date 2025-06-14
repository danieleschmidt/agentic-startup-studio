"""
Comprehensive Testing Framework for End-to-End Pipeline Validation.

This framework provides modular, extensible testing capabilities across 8 validation areas:
- Component Validation
- Integration Testing  
- Performance Testing
- Security Testing
- Data Integrity Testing
- Business Logic Testing
- API Validation Testing
- End-to-End Testing

Key Features:
- Plugin architecture for extensible validation types
- Environment-driven configuration with no hardcoded values
- Support for sequential, parallel, and selective execution modes
- Comprehensive data management with synthetic data generation
- Integration with existing pytest infrastructure
- Docker containerization support for environment isolation
- Real-time monitoring and detailed reporting
- Graceful cleanup and resource management

Usage:
    from tests.framework import TestRunner, create_full_validation_config
    
    async def main():
        runner = TestRunner()
        config = create_full_validation_config()
        result = await runner.run_tests(config)
        print(f"Test run completed: {result.get_summary()}")

For more information, see: docs/testing-framework-architecture.md
"""

import logging
from typing import Dict, Any, List, Optional

# Core components
from .config import (
    TestingFrameworkSettings,
    TestFrameworkConfig,
    DataManagerConfig,
    ValidationEngineConfig,
    ReportingConfig,
    SecurityConfig,
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

from .data_manager import (
    DataManager,
    TestDataSet,
    MockServiceConfig,
    SyntheticDataGenerator,
    MockServiceManager,
    create_startup_test_data,
    create_performance_test_data,
    create_mock_api_service
)

from .validation_engine import ValidationEngine
from .base import BaseValidator, ValidationResult

from .test_runner import (
    TestRunner,
    TestRunConfiguration,
    TestRunResult,
    TestRunStatus,
    TestRunMetrics,
    DependencyManager,
    create_full_validation_config,
    create_smoke_test_config,
    create_performance_test_config,
    run_comprehensive_validation
)

# Configure logging for the framework
logger = logging.getLogger(__name__)

# Framework version
__version__ = "1.0.0"

# Export main classes and functions
__all__ = [
    # Core classes
    'TestRunner',
    'ValidationEngine', 
    'DataManager',
    
    # Configuration
    'ValidationEngineConfig',
    'ValidationArea',
    'ExecutionMode',
    
    # Results and data
    'ValidationResult',
    
    # Validators
    'BaseValidator',
    
    # Utilities
    'SyntheticDataGenerator',
    'MockServiceManager',
    'DependencyManager',
    
    # Factory functions
    'create_full_validation_config',
    'create_smoke_test_config',
    'create_performance_test_config',
    'create_startup_test_data',
    'create_performance_test_data',
    'create_mock_api_service',
    'run_comprehensive_validation',
    
    # Configuration functions
    'get_framework_settings',
    'get_framework_config',
    'get_data_manager_config',
    'get_validation_engine_config',
    'get_reporting_config',
    'get_security_config',
    'validate_framework_environment',
    'get_framework_summary'
]


def initialize_framework(
    validate_environment: bool = True,
    setup_logging: bool = True,
    log_level: str = "INFO"
) -> TestingFrameworkSettings:
    """
    Initialize the testing framework with optional environment validation.
    
    Args:
        validate_environment: Whether to validate the framework environment
        setup_logging: Whether to configure framework logging
        log_level: Logging level for the framework
    
    Returns:
        TestingFrameworkSettings: The loaded framework configuration
        
    Raises:
        ValueError: If environment validation fails
        RuntimeError: If framework initialization fails
    """
    try:
        # Setup logging if requested
        if setup_logging:
            logging.basicConfig(
                level=getattr(logging, log_level.upper()),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # Load framework configuration
        settings = get_framework_settings()
        
        # Validate environment if requested
        if validate_environment:
            validate_framework_environment()
            logger.info("Framework environment validation completed successfully")
        
        # Log framework summary
        summary = get_framework_summary()
        logger.info(f"Testing framework initialized: {summary['framework']['name']} v{summary['framework']['version']}")
        logger.debug(f"Framework configuration: {summary}")
        
        return settings
        
    except Exception as e:
        logger.error(f"Failed to initialize testing framework: {e}")
        raise RuntimeError(f"Framework initialization failed: {e}") from e


async def run_quick_validation(
    areas: Optional[List[ValidationArea]] = None,
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL,
    timeout: Optional[float] = None
) -> TestRunResult:
    """
    Run a quick validation with default configuration.
    
    Args:
        areas: List of validation areas to test (defaults to component and integration)
        execution_mode: How to execute validations
        timeout: Maximum time to allow for validation
        
    Returns:
        TestRunResult: Results of the validation run
    """
    # Default to basic validation areas for quick testing
    if areas is None:
        areas = [ValidationArea.COMPONENT, ValidationArea.INTEGRATION]
    
    # Create configuration
    config = TestRunConfiguration(
        areas=areas,
        execution_mode=execution_mode,
        timeout=timeout or 300,  # 5 minutes default
        fail_fast=True,
        cleanup_on_completion=True,
        generate_reports=False
    )
    
    # Run validation
    runner = TestRunner()
    return await runner.run_tests(config)


def get_available_validators() -> Dict[ValidationArea, List[str]]:
    """
    Get list of available validators by validation area.
    
    Returns:
        Dict mapping validation areas to lists of validator names
    """
    engine = ValidationEngine()
    return engine.get_registered_validators()


def create_custom_validator_config(
    areas: List[ValidationArea],
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL,
    **kwargs
) -> TestRunConfiguration:
    """
    Create a custom test run configuration.
    
    Args:
        areas: Validation areas to include
        execution_mode: How to execute validations
        **kwargs: Additional configuration options
        
    Returns:
        TestRunConfiguration: Custom configuration
    """
    return TestRunConfiguration(
        areas=areas,
        execution_mode=execution_mode,
        **kwargs
    )


# Framework status and health check
def get_framework_status() -> Dict[str, Any]:
    """
    Get current framework status and health information.
    
    Returns:
        Dict containing framework status information
    """
    try:
        settings = get_framework_settings()
        summary = get_framework_summary()
        
        return {
            'status': 'healthy',
            'version': __version__,
            'configuration': summary,
            'environment': settings.framework.environment,
            'available_areas': [area.value for area in ValidationArea],
            'available_modes': [mode.value for mode in ExecutionMode]
        }
    except Exception as e:
        return {
            'status': 'error',
            'version': __version__,
            'error': str(e)
        }


# Integration helpers for existing test infrastructure
def integrate_with_pytest(
    test_discovery_paths: Optional[List[str]] = None,
    pytest_markers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Configure integration with existing pytest infrastructure.
    
    Args:
        test_discovery_paths: Paths for test discovery
        pytest_markers: Custom pytest markers to add
        
    Returns:
        Dict containing integration configuration
    """
    # Default test paths based on project structure
    if test_discovery_paths is None:
        test_discovery_paths = [
            'tests/pipeline/',
            'tests/integration/',
            'tests/framework/'
        ]
    
    # Default markers for validation areas
    if pytest_markers is None:
        pytest_markers = {area.value: f"Tests for {area.value} validation" for area in ValidationArea}
    
    integration_config = {
        'test_paths': test_discovery_paths,
        'markers': pytest_markers,
        'pytest_args': [
            '--strict-markers',
            '--strict-config',
            '--tb=short',
            '-v'
        ]
    }
    
    logger.info(f"Configured pytest integration: {integration_config}")
    return integration_config


# Main entry point for the framework
if __name__ == "__main__":
    import asyncio
    
    async def main():
        """Main entry point for running the framework."""
        print(f"Testing Framework v{__version__}")
        print("=" * 50)
        
        # Initialize framework
        settings = initialize_framework()
        
        # Show framework status
        status = get_framework_status()
        print(f"Status: {status['status']}")
        print(f"Environment: {status.get('environment', 'unknown')}")
        print(f"Available validation areas: {len(status.get('available_areas', []))}")
        
        # Run quick validation as demo
        print("\nRunning quick validation...")
        result = await run_quick_validation()
        
        print(f"\nValidation completed:")
        print(f"Status: {result.status}")
        print(f"Success rate: {result.metrics.success_rate:.1f}%")
        print(f"Execution time: {result.execution_time:.2f}s")
        
    # Run the demo
    asyncio.run(main())