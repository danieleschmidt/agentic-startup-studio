"""
Test runner for the comprehensive testing framework.

This module orchestrates validation execution across 8 validation areas,
manages dependencies, and integrates with existing pytest infrastructure.
"""

import asyncio
import json
import time
import uuid
from typing import Dict, Any, List, Optional, Set, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging
import signal
import sys
from contextlib import asynccontextmanager

import pytest
from pytest import ExitCode

from tests.framework.config import (
    get_framework_settings,
    ValidationArea,
    ExecutionMode,
    TestingFrameworkSettings
)
from tests.framework.validation_engine import ValidationEngine
from tests.framework.base import ValidationResult, ValidationContext
from tests.framework.data_manager import DataManager

logger = logging.getLogger(__name__)


class TestRunStatus(str, Enum):
    """Test run execution status."""
    PENDING = "pending"
    INITIALIZING = "initializing"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass
class TestRunMetrics:
    """Metrics for test run execution."""
    total_validations: int = 0
    passed_validations: int = 0
    failed_validations: int = 0
    error_validations: int = 0
    skipped_validations: int = 0
    total_execution_time: float = 0.0
    validation_times: Dict[str, float] = field(default_factory=dict)
    area_results: Dict[ValidationArea, int] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_validations == 0:
            return 0.0
        return (self.passed_validations / self.total_validations) * 100
    
    @property
    def is_success(self) -> bool:
        """Check if test run was successful."""
        return self.failed_validations == 0 and self.error_validations == 0


@dataclass
class TestRunConfiguration:
    """Configuration for test run execution."""
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    areas: List[ValidationArea] = field(default_factory=list)
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    test_data_sources: List[str] = field(default_factory=list)
    environment_setup: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[float] = None
    fail_fast: bool = False
    cleanup_on_completion: bool = True
    generate_reports: bool = True
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize default areas if none provided."""
        if not self.areas:
            self.areas = list(ValidationArea)


@dataclass
class TestRunResult:
    """Complete result of test run execution."""
    run_id: str
    configuration: TestRunConfiguration
    status: TestRunStatus
    metrics: TestRunMetrics
    validation_results: Dict[ValidationArea, List[ValidationResult]]
    start_time: float
    end_time: Optional[float] = None
    error_message: Optional[str] = None
    reports_generated: List[str] = field(default_factory=list)
    
    @property
    def execution_time(self) -> float:
        """Calculate total execution time."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of test run results."""
        return {
            'run_id': self.run_id,
            'status': self.status.value,
            'execution_time': self.execution_time,
            'areas_tested': len(self.validation_results),
            'success_rate': self.metrics.success_rate,
            'total_validations': self.metrics.total_validations,
            'passed': self.metrics.passed_validations,
            'failed': self.metrics.failed_validations,
            'errors': self.metrics.error_validations,
            'overall_success': self.metrics.is_success
        }


class DependencyManager:
    """Manages dependencies between validation areas."""
    
    def __init__(self):
        """Initialize dependency manager."""
        self.dependencies: Dict[ValidationArea, Set[ValidationArea]] = {
            ValidationArea.COMPONENT: set(),
            ValidationArea.INTEGRATION: {ValidationArea.COMPONENT},
            ValidationArea.PERFORMANCE: {ValidationArea.COMPONENT, ValidationArea.INTEGRATION},
            ValidationArea.SECURITY: {ValidationArea.COMPONENT},
            ValidationArea.DATA_INTEGRITY: {ValidationArea.COMPONENT},
            ValidationArea.BUSINESS_LOGIC: {ValidationArea.COMPONENT, ValidationArea.DATA_INTEGRITY},
            ValidationArea.API_VALIDATION: {ValidationArea.COMPONENT, ValidationArea.INTEGRATION},
            ValidationArea.END_TO_END: {
                ValidationArea.COMPONENT, 
                ValidationArea.INTEGRATION,
                ValidationArea.API_VALIDATION
            }
        }
    
    def get_execution_order(self, areas: List[ValidationArea]) -> List[ValidationArea]:
        """Calculate optimal execution order based on dependencies."""
        ordered_areas = []
        remaining_areas = set(areas)
        
        while remaining_areas:
            # Find areas with no pending dependencies
            ready_areas = []
            for area in remaining_areas:
                dependencies = self.dependencies.get(area, set())
                if dependencies.issubset(set(ordered_areas)):
                    ready_areas.append(area)
            
            if not ready_areas:
                # Circular dependency or missing dependency
                logger.warning(f"Potential circular dependency detected. Remaining areas: {remaining_areas}")
                ready_areas = list(remaining_areas)  # Force execution
            
            # Sort ready areas for consistent ordering
            ready_areas.sort(key=lambda x: x.value)
            ordered_areas.extend(ready_areas)
            remaining_areas -= set(ready_areas)
        
        return ordered_areas
    
    def validate_dependencies(self, areas: List[ValidationArea]) -> bool:
        """Validate that all dependencies are satisfied."""
        area_set = set(areas)
        for area in areas:
            dependencies = self.dependencies.get(area, set())
            if not dependencies.issubset(area_set):
                missing = dependencies - area_set
                logger.error(f"Missing dependencies for {area}: {missing}")
                return False
        return True


class TestRunner:
    """Main test runner orchestrating validation execution."""
    
    def __init__(self, settings: Optional[TestingFrameworkSettings] = None):
        """Initialize test runner."""
        self.settings = settings or get_framework_settings()
        self.validation_engine = ValidationEngine(self.settings.validation_engine)
        self.data_manager = DataManager(self.settings.data_manager)
        self.dependency_manager = DependencyManager()
        
        self.current_run: Optional[TestRunResult] = None
        self.cancellation_event = asyncio.Event()
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.cancellation_event.set()
        
        if hasattr(signal, 'SIGINT'):
            signal.signal(signal.SIGINT, signal_handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, signal_handler)
    
    async def run_tests(self, config: TestRunConfiguration) -> TestRunResult:
        """Execute comprehensive test run."""
        logger.info(f"Starting test run {config.run_id} with {len(config.areas)} validation areas")
        
        # Initialize test run result
        result = TestRunResult(
            run_id=config.run_id,
            configuration=config,
            status=TestRunStatus.INITIALIZING,
            metrics=TestRunMetrics(),
            validation_results={},
            start_time=time.time()
        )
        
        self.current_run = result
        
        try:
            # Validate configuration
            await self._validate_configuration(config)
            
            # Setup test environment
            result.status = TestRunStatus.INITIALIZING
            await self._setup_test_environment(config)
            
            # Execute validations
            result.status = TestRunStatus.RUNNING
            await self._execute_validations(config, result)
            
            # Generate reports if requested
            if config.generate_reports:
                await self._generate_reports(result)
            
            # Determine final status
            if result.metrics.is_success:
                result.status = TestRunStatus.COMPLETED
            else:
                result.status = TestRunStatus.FAILED
            
            logger.info(f"Test run {config.run_id} completed with status: {result.status}")
            
        except asyncio.CancelledError:
            result.status = TestRunStatus.CANCELLED
            logger.info(f"Test run {config.run_id} was cancelled")
        except Exception as e:
            result.status = TestRunStatus.ERROR
            result.error_message = str(e)
            logger.error(f"Test run {config.run_id} failed with error: {e}", exc_info=True)
        finally:
            result.end_time = time.time()
            
            # Cleanup if requested
            if config.cleanup_on_completion:
                await self._cleanup_test_environment()
            
            self.current_run = None
        
        return result
    
    async def _validate_configuration(self, config: TestRunConfiguration):
        """Validate test run configuration."""
        if not config.areas:
            raise ValueError("No validation areas specified")
        
        # Validate dependencies
        if not self.dependency_manager.validate_dependencies(config.areas):
            raise ValueError("Invalid dependencies in validation areas")
        
        # Validate timeout
        if config.timeout and config.timeout <= 0:
            raise ValueError("Timeout must be positive")
        
        logger.debug(f"Configuration validated for run {config.run_id}")
    
    async def _setup_test_environment(self, config: TestRunConfiguration):
        """Setup test environment and resources."""
        try:
            # Initialize data manager
            async with self.data_manager as dm:
                # Load test data sources
                for data_source in config.test_data_sources:
                    try:
                        dataset_name = Path(data_source).stem
                        dm.load_test_data(data_source, dataset_name)
                        logger.debug(f"Loaded test data: {dataset_name}")
                    except Exception as e:
                        logger.warning(f"Failed to load test data {data_source}: {e}")
                
                # Setup environment configuration
                if config.environment_setup:
                    logger.debug("Applied environment configuration")
                
        except Exception as e:
            logger.error(f"Failed to setup test environment: {e}")
            raise
    
    async def _execute_validations(self, config: TestRunConfiguration, result: TestRunResult):
        """Execute validations based on configuration."""
        # Get execution order based on dependencies
        ordered_areas = self.dependency_manager.get_execution_order(config.areas)
        
        if config.execution_mode == ExecutionMode.PARALLEL:
            await self._execute_parallel_validations(ordered_areas, config, result)
        elif config.execution_mode == ExecutionMode.SELECTIVE:
            await self._execute_selective_validations(ordered_areas, config, result)
        else:  # Sequential
            await self._execute_sequential_validations(ordered_areas, config, result)
    
    async def _execute_sequential_validations(
        self, 
        areas: List[ValidationArea], 
        config: TestRunConfiguration, 
        result: TestRunResult
    ):
        """Execute validations sequentially."""
        for area in areas:
            if self.cancellation_event.is_set():
                break
            
            area_start_time = time.time()
            
            try:
                context = self._create_validation_context(config, area)
                validation_results = await self.validation_engine.execute_validation(area, context)
                
                result.validation_results[area] = validation_results
                self._update_metrics(result.metrics, validation_results)
                
                area_execution_time = time.time() - area_start_time
                result.metrics.validation_times[area.value] = area_execution_time
                
                logger.info(f"Completed {area} validation in {area_execution_time:.2f}s")
                
                # Check fail fast condition
                if config.fail_fast and any(vr.status == ValidationStatus.FAILED for vr in validation_results):
                    logger.info(f"Failing fast due to failures in {area}")
                    break
                    
            except Exception as e:
                logger.error(f"Error executing {area} validation: {e}")
                # Create error result
                error_result = ValidationResult(
                    validation_id=f"error_{area}_{int(time.time())}",
                    area=area,
                    status=ValidationStatus.ERROR,
                    message=f"Execution error: {str(e)}"
                )
                result.validation_results[area] = [error_result]
                result.metrics.error_validations += 1
    
    async def _execute_parallel_validations(
        self, 
        areas: List[ValidationArea], 
        config: TestRunConfiguration, 
        result: TestRunResult
    ):
        """Execute validations in parallel respecting dependencies."""
        completed_areas = set()
        running_tasks = {}
        
        while len(completed_areas) < len(areas) and not self.cancellation_event.is_set():
            # Find areas ready to execute
            ready_areas = []
            for area in areas:
                if area not in completed_areas and area not in running_tasks:
                    dependencies = self.dependency_manager.dependencies.get(area, set())
                    if dependencies.issubset(completed_areas):
                        ready_areas.append(area)
            
            # Start tasks for ready areas
            for area in ready_areas:
                context = self._create_validation_context(config, area)
                task = asyncio.create_task(
                    self.validation_engine.execute_validation(area, context)
                )
                running_tasks[area] = task
                logger.debug(f"Started parallel validation for {area}")
            
            # Wait for at least one task to complete
            if running_tasks:
                done, pending = await asyncio.wait(
                    running_tasks.values(), 
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Process completed tasks
                for task in done:
                    for area, area_task in running_tasks.items():
                        if area_task == task:
                            try:
                                validation_results = await task
                                result.validation_results[area] = validation_results
                                self._update_metrics(result.metrics, validation_results)
                                completed_areas.add(area)
                                
                                logger.info(f"Completed parallel validation for {area}")
                                
                                # Check fail fast condition
                                if (config.fail_fast and 
                                    any(vr.status == ValidationStatus.FAILED for vr in validation_results)):
                                    logger.info(f"Failing fast due to failures in {area}")
                                    # Cancel remaining tasks
                                    for pending_task in pending:
                                        pending_task.cancel()
                                    return
                                    
                            except Exception as e:
                                logger.error(f"Error in parallel validation for {area}: {e}")
                                completed_areas.add(area)  # Mark as completed even if failed
                            
                            del running_tasks[area]
                            break
            else:
                # No tasks running and no ready areas - potential deadlock
                break
    
    async def _execute_selective_validations(
        self, 
        areas: List[ValidationArea], 
        config: TestRunConfiguration, 
        result: TestRunResult
    ):
        """Execute validations selectively based on tags or criteria."""
        # For selective mode, filter areas based on tags
        selected_areas = []
        for area in areas:
            if not config.tags or any(tag in area.value for tag in config.tags):
                selected_areas.append(area)
        
        logger.info(f"Selective mode: executing {len(selected_areas)} of {len(areas)} areas")
        
        # Execute selected areas sequentially
        await self._execute_sequential_validations(selected_areas, config, result)
    
    def _create_validation_context(
        self, 
        config: TestRunConfiguration, 
        area: ValidationArea
    ) -> ValidationContext:
        """Create validation context for specific area."""
        # Get test data from data manager
        test_data = {}
        if hasattr(self.data_manager, 'active_datasets'):
            for dataset_name, dataset in self.data_manager.active_datasets.items():
                test_data[dataset_name] = dataset.data
        
        # Add area-specific test configuration
        test_data.update({
            'area': area.value,
            'run_id': config.run_id,
            'test_modules': self._get_test_modules_for_area(area),
            'integration_tests': self._get_integration_tests_for_area(area),
            'performance_tests': self._get_performance_tests_for_area(area),
            'security_tests': self._get_security_tests_for_area(area)
        })
        
        return ValidationContext(
            test_data=test_data,
            configuration=config.environment_setup,
            timeout=config.timeout,
            max_retries=self.settings.validation_engine.retry_attempts
        )
    
    def _get_test_modules_for_area(self, area: ValidationArea) -> List[str]:
        """Get test modules for specific validation area."""
        area_mapping = {
            ValidationArea.COMPONENT: ['tests/pipeline/models/', 'tests/pipeline/config/'],
            ValidationArea.INTEGRATION: ['tests/pipeline/ingestion/', 'tests/pipeline/storage/'],
            ValidationArea.PERFORMANCE: ['tests/performance/'],
            ValidationArea.SECURITY: ['tests/security/'],
            ValidationArea.DATA_INTEGRITY: ['tests/data/'],
            ValidationArea.BUSINESS_LOGIC: ['tests/pipeline/services/'],
            ValidationArea.API_VALIDATION: ['tests/api/'],
            ValidationArea.END_TO_END: ['tests/integration/']
        }
        
        return area_mapping.get(area, [])
    
    def _get_integration_tests_for_area(self, area: ValidationArea) -> List[str]:
        """Get integration tests for specific area."""
        if area == ValidationArea.INTEGRATION:
            return ['tests/integration/test_full_pipeline_integration.py']
        return []
    
    def _get_performance_tests_for_area(self, area: ValidationArea) -> List[str]:
        """Get performance tests for specific area."""
        if area == ValidationArea.PERFORMANCE:
            return ['tests/performance/']
        return []
    
    def _get_security_tests_for_area(self, area: ValidationArea) -> List[str]:
        """Get security tests for specific area."""
        if area == ValidationArea.SECURITY:
            return ['tests/security/']
        return []
    
    def _update_metrics(self, metrics: TestRunMetrics, validation_results: List[ValidationResult]):
        """Update metrics with validation results."""
        for result in validation_results:
            metrics.total_validations += 1
            metrics.total_execution_time += result.execution_time
            
            if result.status == ValidationStatus.PASSED:
                metrics.passed_validations += 1
            elif result.status == ValidationStatus.FAILED:
                metrics.failed_validations += 1
            elif result.status == ValidationStatus.ERROR:
                metrics.error_validations += 1
            elif result.status == ValidationStatus.SKIPPED:
                metrics.skipped_validations += 1
            
            # Update area results
            area = result.area
            if area not in metrics.area_results:
                metrics.area_results[area] = 0
            metrics.area_results[area] += 1
    
    async def _generate_reports(self, result: TestRunResult):
        """Generate test reports."""
        try:
            reports_dir = Path(self.settings.reporting.report_output_dir)
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate summary report
            summary_file = reports_dir / f"test_run_{result.run_id}_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(result.get_summary(), f, indent=2)
            
            result.reports_generated.append(str(summary_file))
            logger.info(f"Generated test report: {summary_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate reports: {e}")
    
    async def _cleanup_test_environment(self):
        """Cleanup test environment and resources."""
        try:
            # Cleanup data manager resources
            if hasattr(self.data_manager, 'cleanup_test_data'):
                self.data_manager.cleanup_test_data()
            
            logger.debug("Test environment cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during test environment cleanup: {e}")
    
    def run_with_pytest_integration(
        self, 
        config: TestRunConfiguration,
        pytest_args: Optional[List[str]] = None
    ) -> TestRunResult:
        """Run tests with pytest integration."""
        pytest_args = pytest_args or []
        
        # Add framework-specific pytest arguments
        framework_args = [
            '--tb=short',
            '-v',
            '--strict-markers',
            '--strict-config'
        ]
        
        # Add area-specific markers
        for area in config.areas:
            framework_args.extend(['-m', area.value])
        
        all_args = framework_args + pytest_args
        
        logger.info(f"Running pytest with args: {all_args}")
        
        # Run pytest
        exit_code = pytest.main(all_args)
        
        # Create simplified result for pytest integration
        result = TestRunResult(
            run_id=config.run_id,
            configuration=config,
            status=TestRunStatus.COMPLETED if exit_code == ExitCode.OK else TestRunStatus.FAILED,
            metrics=TestRunMetrics(),
            validation_results={},
            start_time=time.time(),
            end_time=time.time()
        )
        
        return result
    
    async def cancel_current_run(self):
        """Cancel currently running test."""
        if self.current_run:
            logger.info(f"Cancelling test run {self.current_run.run_id}")
            self.cancellation_event.set()
    
    def get_run_status(self) -> Optional[Dict[str, Any]]:
        """Get current run status."""
        if self.current_run:
            return {
                'run_id': self.current_run.run_id,
                'status': self.current_run.status.value,
                'execution_time': self.current_run.execution_time,
                'completed_areas': len(self.current_run.validation_results),
                'total_areas': len(self.current_run.configuration.areas)
            }
        return None


# Factory functions for common test configurations
def create_full_validation_config(execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL) -> TestRunConfiguration:
    """Create configuration for full validation across all areas."""
    return TestRunConfiguration(
        areas=list(ValidationArea),
        execution_mode=execution_mode,
        timeout=1800,  # 30 minutes
        fail_fast=False,
        cleanup_on_completion=True,
        generate_reports=True
    )


def create_smoke_test_config() -> TestRunConfiguration:
    """Create configuration for smoke testing."""
    return TestRunConfiguration(
        areas=[ValidationArea.COMPONENT, ValidationArea.INTEGRATION],
        execution_mode=ExecutionMode.SEQUENTIAL,
        timeout=300,  # 5 minutes
        fail_fast=True,
        cleanup_on_completion=True,
        generate_reports=False,
        tags=['smoke']
    )


def create_performance_test_config() -> TestRunConfiguration:
    """Create configuration for performance testing."""
    return TestRunConfiguration(
        areas=[ValidationArea.PERFORMANCE],
        execution_mode=ExecutionMode.SEQUENTIAL,
        timeout=900,  # 15 minutes
        fail_fast=False,
        cleanup_on_completion=True,
        generate_reports=True,
        tags=['performance']
    )


async def run_comprehensive_validation() -> TestRunResult:
    """Run comprehensive validation with default configuration."""
    runner = TestRunner()
    config = create_full_validation_config(ExecutionMode.PARALLEL)
    return await runner.run_tests(config)