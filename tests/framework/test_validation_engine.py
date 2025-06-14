"""
Tests for ValidationEngine orchestrator.

Following TDD approach to test core validation orchestration functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List

from tests.framework.validation_engine import ValidationEngine
from tests.framework.base import BaseValidator, ValidationResult, ValidationContext
from tests.framework.config import ValidationArea, ValidationEngineConfig, ValidationStatus, ValidationSeverity


class MockValidator(BaseValidator):
    """Mock validator for testing."""
    
    def __init__(self, config, area: ValidationArea, name: str, should_pass: bool = True):
        super().__init__(config)
        self._area = area
        self._name = name
        self._should_pass = should_pass
    
    @property
    def area(self) -> ValidationArea:
        """Implementation of abstract area property."""
        return self._area
    
    @property
    def validation_area(self) -> ValidationArea:
        return self._area
    
    @property
    def name(self) -> str:
        return self._name
    
    def validate(self, context: ValidationContext = None, **kwargs) -> ValidationResult:
        """Sync validate method implementation."""
        validation_id = self.get_validation_id(context)
        status = ValidationStatus.PASSED if self._should_pass else ValidationStatus.FAILED
        message = f"Mock validation {'passed' if self._should_pass else 'failed'}"
        
        return ValidationResult(
            validation_id=validation_id,
            area=self.area,
            status=status,
            message=message
        )
    
    async def validate(self, context: ValidationContext) -> ValidationResult:
        """Async validate method."""
        validation_id = self.get_validation_id(context)
        status = ValidationStatus.PASSED if self._should_pass else ValidationStatus.FAILED
        message = f"Mock validation {'passed' if self._should_pass else 'failed'}"
        
        return ValidationResult(
            validation_id=validation_id,
            area=self.area,
            status=status,
            message=message
        )
    
    def get_validation_id(self, context: ValidationContext) -> str:
        """Generate validation ID from context."""
        return f"{self._name}_{id(context) if context else 'unknown'}"


class TestValidationEngine:
    """Test cases for ValidationEngine orchestrator."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock validation engine config."""
        config = Mock(spec=ValidationEngineConfig)
        config.plugin_directories = ["tests/framework/validators"]
        config.auto_load_plugins = True
        config.strict_validation = True
        config.fail_fast = False
        config.retry_attempts = 2
        config.retry_delay = 1.0
        return config
    
    @pytest.fixture
    def validation_engine(self, mock_config):
        """Create ValidationEngine instance for testing."""
        return ValidationEngine(mock_config)
    
    @pytest.fixture
    def sample_context(self):
        """Create sample validation context."""
        return ValidationContext(
            test_data={"sample": "data"},
            configuration={"env": "test"},
            environment={"TEST_ENV": "true"},
            dependencies=["mock:8080"],
            timeout=30.0
        )

    def test_validation_engine_initialization(self, mock_config):
        """Test ValidationEngine initializes correctly."""
        # Given a valid configuration
        # When creating ValidationEngine
        engine = ValidationEngine(mock_config)
        
        # Then it should initialize properly
        assert engine.config == mock_config
        assert engine._validators == {}
        assert engine._plugin_loader is not None

    def test_register_validator_success(self, validation_engine, mock_config):
        """Test successful validator registration."""
        # Given a validation engine and mock validator
        mock_validator_class = MockValidator
        
        # When registering the validator
        validation_engine.register_validator(mock_validator_class)
        
        # Then it should be registered in the correct area
        assert ValidationArea.DATA_FLOW in validation_engine._validators
        # Note: This test will fail until we implement register_validator method

    @pytest.mark.asyncio
    async def test_execute_validation_single_area(self, validation_engine, sample_context, mock_config):
        """Test executing validation for single area."""
        # Given a validator registered for specific area
        mock_validator = MockValidator(mock_config, ValidationArea.DATA_FLOW, "test_validator")
        validation_engine._validators[ValidationArea.DATA_FLOW] = [mock_validator]
        
        # When executing validation for that area
        result = await validation_engine.execute_validation(ValidationArea.DATA_FLOW, sample_context)
        
        # Then it should return successful result
        assert result.status == ValidationStatus.PASSED
        assert result.area == ValidationArea.DATA_FLOW

    @pytest.mark.asyncio
    async def test_execute_validation_no_validators(self, validation_engine, sample_context):
        """Test executing validation with no registered validators."""
        # Given no validators registered for area
        # When executing validation
        result = await validation_engine.execute_validation(ValidationArea.SECURITY, sample_context)
        
        # Then it should return skipped status
        assert result.status == ValidationStatus.SKIPPED
        assert "No validators registered" in result.message

    @pytest.mark.asyncio
    async def test_execute_all_validations(self, validation_engine, sample_context, mock_config):
        """Test executing all registered validations."""
        # Given multiple validators registered
        validator1 = MockValidator(mock_config, ValidationArea.DATA_FLOW, "validator1", True)
        validator2 = MockValidator(mock_config, ValidationArea.SECURITY, "validator2", True)
        validation_engine._validators[ValidationArea.DATA_FLOW] = [validator1]
        validation_engine._validators[ValidationArea.SECURITY] = [validator2]
        
        # When executing all validations
        results = await validation_engine.execute_all_validations(sample_context)
        
        # Then it should return results for all areas
        assert len(results) == 2
        assert ValidationArea.DATA_FLOW in results
        assert ValidationArea.SECURITY in results
        assert all(r.status == ValidationStatus.PASSED for area_results in results.values() for r in area_results)

    def test_get_registered_validators(self, validation_engine, mock_config):
        """Test getting list of registered validators."""
        # Given validators registered for multiple areas
        validator1 = MockValidator(mock_config, ValidationArea.DATA_FLOW, "validator1")
        validator2 = MockValidator(mock_config, ValidationArea.SECURITY, "validator2")
        validation_engine._validators[ValidationArea.DATA_FLOW] = [validator1]
        validation_engine._validators[ValidationArea.SECURITY] = [validator2]
        
        # When getting registered validators
        registered = validation_engine.get_registered_validators()
        
        # Then it should return all registered areas and names
        assert ValidationArea.DATA_FLOW in registered
        assert ValidationArea.SECURITY in registered
        assert "validator1" in registered[ValidationArea.DATA_FLOW]
        assert "validator2" in registered[ValidationArea.SECURITY]

    def test_get_validation_summary(self, validation_engine):
        """Test generating validation summary."""
        # Given validation results
        results = {
            ValidationArea.DATA_FLOW: [
                ValidationResult("test1", ValidationArea.DATA_FLOW, ValidationStatus.PASSED, "Test passed"),
                ValidationResult("test2", ValidationArea.DATA_FLOW, ValidationStatus.FAILED, "Test failed")
            ],
            ValidationArea.SECURITY: [
                ValidationResult("test3", ValidationArea.SECURITY, ValidationStatus.PASSED, "Security passed")
            ]
        }
        
        # When generating summary
        summary = validation_engine.get_validation_summary(results)
        
        # Then it should provide comprehensive summary
        assert summary["total_validations"] == 3
        assert summary["passed_validations"] == 2
        assert summary["failed_validations"] == 1
        assert summary["success_rate"] == pytest.approx(66.67, rel=1e-2)
        assert len(summary["areas"]) == 2


class TestValidationEngineRetryLogic:
    """Test retry and error handling logic."""
    
    @pytest.fixture
    def mock_config(self):
        """Create config with retry settings."""
        config = Mock(spec=ValidationEngineConfig)
        config.retry_attempts = 3
        config.retry_delay = 0.1
        config.fail_fast = False
        return config
    
    @pytest.fixture
    def validation_engine(self, mock_config):
        """Create ValidationEngine with retry config."""
        return ValidationEngine(mock_config)

    @pytest.mark.asyncio
    async def test_execute_with_retry_success_on_first_attempt(self, validation_engine, mock_config):
        """Test successful execution on first attempt."""
        # Given a validator that passes on first try
        mock_validator = MockValidator(mock_config, ValidationArea.DATA_FLOW, "test_validator", True)
        context = ValidationContext(test_data={})
        
        # When executing with retry
        result = await validation_engine._execute_with_retry(mock_validator, context)
        
        # Then it should succeed without retries
        assert result.status == ValidationStatus.PASSED

    @pytest.mark.asyncio
    async def test_execute_with_retry_eventual_success(self, validation_engine, mock_config):
        """Test successful execution after retries."""
        # Given a validator that fails then succeeds
        call_count = 0
        
        class FlakeyValidator(MockValidator):
            async def validate(self, context):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise Exception("Temporary failure")
                return await super().validate(context)
        
        validator = FlakeyValidator(mock_config, ValidationArea.DATA_FLOW, "flakey", True)
        context = ValidationContext(test_data={})
        
        # When executing with retry
        result = await validation_engine._execute_with_retry(validator, context)
        
        # Then it should eventually succeed
        assert result.status == ValidationStatus.PASSED
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_fail_fast_behavior(self, mock_config):
        """Test fail-fast behavior stops on first failure."""
        # Given fail-fast configuration
        mock_config.fail_fast = True
        engine = ValidationEngine(mock_config)
        
        # Setup failing validator
        failing_validator = MockValidator(mock_config, ValidationArea.DATA_FLOW, "failing", False)
        passing_validator = MockValidator(mock_config, ValidationArea.SECURITY, "passing", True)
        engine._validators[ValidationArea.DATA_FLOW] = [failing_validator]
        engine._validators[ValidationArea.SECURITY] = [passing_validator]
        
        context = ValidationContext(test_data={})
        
        # When executing all validations with fail-fast
        results = await engine.execute_all_validations(context)
        
        # Then it should stop after first failure
        assert len(results) == 1
        assert ValidationArea.DATA_FLOW in results
        assert ValidationArea.SECURITY not in results


class TestValidationEnginePluginLoading:
    """Test plugin loading functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create config with plugin settings."""
        config = Mock(spec=ValidationEngineConfig)
        config.plugin_directories = ["tests/framework/validators"]
        config.auto_load_plugins = True
        return config
    
    @pytest.fixture
    def validation_engine(self, mock_config):
        """Create ValidationEngine with plugin config."""
        return ValidationEngine(mock_config)

    def test_load_external_plugins(self, validation_engine):
        """Test loading external plugin validators."""
        # Given plugin directories configured
        # When loading external plugins
        validation_engine._load_external_plugins()
        
        # Then it should discover and register validators
        registered = validation_engine.get_registered_validators()
        assert len(registered) > 0

    def test_discover_validator_classes(self, validation_engine):
        """Test discovering validator classes from modules."""
        # Given a plugin directory
        plugin_dir = "tests/framework/validators"
        
        # When discovering validator classes
        discovered = validation_engine._discover_validator_classes(plugin_dir)
        
        # Then it should find available validators
        assert len(discovered) > 0
        assert all(hasattr(cls, 'validation_area') for cls in discovered)


class TestValidationResultAggregation:
    """Test result aggregation and reporting."""
    
    def test_aggregate_results_by_area(self):
        """Test aggregating validation results by area."""
        # Given validation results from multiple validators
        results = [
            ValidationResult("test1", ValidationArea.DATA_FLOW, ValidationStatus.PASSED, "Passed"),
            ValidationResult("test2", ValidationArea.DATA_FLOW, ValidationStatus.FAILED, "Failed"),
            ValidationResult("test3", ValidationArea.SECURITY, ValidationStatus.PASSED, "Security OK")
        ]
        
        # When aggregating by area
        engine = ValidationEngine(Mock())
        aggregated = engine._aggregate_results_by_area(results)
        
        # Then results should be grouped by validation area
        assert ValidationArea.DATA_FLOW in aggregated
        assert ValidationArea.SECURITY in aggregated
        assert len(aggregated[ValidationArea.DATA_FLOW]) == 2
        assert len(aggregated[ValidationArea.SECURITY]) == 1

    def test_calculate_success_metrics(self):
        """Test calculating validation success metrics."""
        # Given mixed validation results
        results = {
            ValidationArea.DATA_FLOW: [
                ValidationResult("test1", ValidationArea.DATA_FLOW, ValidationStatus.PASSED, "Passed"),
                ValidationResult("test2", ValidationArea.DATA_FLOW, ValidationStatus.FAILED, "Failed")
            ],
            ValidationArea.SECURITY: [
                ValidationResult("test3", ValidationArea.SECURITY, ValidationStatus.PASSED, "Passed")
            ]
        }
        
        # When calculating metrics
        engine = ValidationEngine(Mock())
        metrics = engine._calculate_success_metrics(results)
        
        # Then metrics should be accurate
        assert metrics["total_validations"] == 3
        assert metrics["passed_validations"] == 2
        assert metrics["failed_validations"] == 1
        assert metrics["success_rate"] == pytest.approx(66.67, rel=1e-2)


class TestValidationResult:
    """Test ValidationResult functionality."""
    
    def test_validation_result_creation(self):
        """Test ValidationResult creation and methods."""
        # Given validation parameters
        result = ValidationResult(
            validation_id="test_123",
            area=ValidationArea.DATA_FLOW,
            status=ValidationStatus.PASSED,
            message="Test validation passed"
        )
        
        # Then it should be created correctly
        assert result.validation_id == "test_123"
        assert result.area == ValidationArea.DATA_FLOW
        assert result.status == ValidationStatus.PASSED
        assert result.is_success() is True
        assert result.timestamp is not None

    def test_validation_result_add_error(self):
        """Test adding errors to validation result."""
        # Given a validation result
        result = ValidationResult("test", ValidationArea.DATA_FLOW, ValidationStatus.RUNNING, "Running")
        
        # When adding an error
        result.add_error("Test error occurred")
        
        # Then status should change to failed and error should be recorded
        assert result.status == ValidationStatus.FAILED
        assert "Test error occurred" in result.errors
        assert not result.is_success()

    def test_validation_result_add_metric(self):
        """Test adding metrics to validation result."""
        # Given a validation result
        result = ValidationResult("test", ValidationArea.DATA_FLOW, ValidationStatus.PASSED, "Passed")
        
        # When adding a metric
        result.add_metric("execution_time", 1.23)
        
        # Then metric should be recorded
        assert result.metrics["execution_time"] == 1.23