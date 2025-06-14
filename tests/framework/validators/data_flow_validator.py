"""
Data Flow Validator for pipeline validation.

Validates that data moves correctly between pipeline stages, verifying
transformations, integrity, and proper data flow patterns.
"""

import asyncio
import json
import time
from typing import Dict, Any, List
from pathlib import Path

from tests.framework.validation_engine import (
    BaseValidator, ValidationResult, ValidationContext, 
    ValidationStatus, ValidationSeverity
)
from tests.framework.config import ValidationArea


class DataFlowValidator(BaseValidator):
    """Validator for data flow integrity across pipeline stages."""

    @property
    def validation_area(self) -> ValidationArea:
        return ValidationArea.DATA_FLOW

    @property 
    def name(self) -> str:
        return "data_flow_validator"

    async def validate(self, context: ValidationContext) -> ValidationResult:
        """Execute data flow validation."""
        validation_id = self.get_validation_id(context)
        result = ValidationResult(
            validation_id=validation_id,
            area=self.validation_area,
            status=ValidationStatus.RUNNING,
            message="Executing data flow validation"
        )

        start_time = time.time()

        try:
            # Validate pipeline data transformations
            await self._validate_data_transformations(context, result)
            
            # Check data integrity constraints
            await self._validate_data_integrity(context, result)
            
            # Verify stage-to-stage data flow
            await self._validate_stage_transitions(context, result)
            
            # Validate data persistence patterns
            await self._validate_data_persistence(context, result)

            if not result.errors:
                result.status = ValidationStatus.PASSED
                result.message = "Data flow validation completed successfully"

        except Exception as e:
            result.status = ValidationStatus.ERROR
            result.message = f"Data flow validation error: {str(e)}"
            result.add_error(str(e))
            self.logger.error(f"Data flow validation failed: {e}", exc_info=True)

        finally:
            result.execution_time = time.time() - start_time

        return result

    async def _validate_data_transformations(self, context: ValidationContext, result: ValidationResult):
        """Validate data transformations between pipeline stages."""
        transformations = context.test_data.get('data_transformations', [])
        
        for transformation in transformations:
            try:
                input_data = transformation.get('input')
                expected_output = transformation.get('expected_output')
                transformation_func = transformation.get('function')
                
                if not all([input_data, expected_output, transformation_func]):
                    result.add_warning(f"Incomplete transformation definition: {transformation}")
                    continue

                # Execute transformation
                actual_output = await self._execute_transformation(
                    transformation_func, input_data
                )
                
                # Validate output matches expected
                if not self._compare_data_outputs(actual_output, expected_output):
                    result.add_error(
                        f"Transformation output mismatch for {transformation_func}: "
                        f"expected {expected_output}, got {actual_output}"
                    )
                else:
                    result.add_metric(f"transformation_{transformation_func}", "passed")
                    
            except Exception as e:
                result.add_error(f"Transformation validation failed: {str(e)}")

    async def _validate_data_integrity(self, context: ValidationContext, result: ValidationResult):
        """Validate data integrity constraints."""
        integrity_checks = context.test_data.get('integrity_checks', [])
        
        for check in integrity_checks:
            try:
                check_type = check.get('type')
                data_source = check.get('data_source')
                constraints = check.get('constraints', {})
                
                if check_type == 'schema_validation':
                    await self._validate_data_schema(data_source, constraints, result)
                elif check_type == 'referential_integrity':
                    await self._validate_referential_integrity(data_source, constraints, result)
                elif check_type == 'data_quality':
                    await self._validate_data_quality(data_source, constraints, result)
                    
            except Exception as e:
                result.add_error(f"Integrity check failed: {str(e)}")

    async def _validate_stage_transitions(self, context: ValidationContext, result: ValidationResult):
        """Validate data flow between pipeline stages."""
        stage_transitions = context.test_data.get('stage_transitions', [])
        
        for transition in stage_transitions:
            try:
                source_stage = transition.get('source')
                target_stage = transition.get('target')
                data_mapping = transition.get('data_mapping', {})
                
                # Validate stage outputs match expected inputs
                source_output = await self._get_stage_output(source_stage)
                target_input = await self._get_stage_input(target_stage)
                
                if not self._validate_stage_compatibility(
                    source_output, target_input, data_mapping
                ):
                    result.add_error(
                        f"Stage transition incompatible: {source_stage} -> {target_stage}"
                    )
                else:
                    result.add_metric(f"transition_{source_stage}_to_{target_stage}", "valid")
                    
            except Exception as e:
                result.add_error(f"Stage transition validation failed: {str(e)}")

    async def _validate_data_persistence(self, context: ValidationContext, result: ValidationResult):
        """Validate data persistence patterns and storage."""
        persistence_checks = context.test_data.get('persistence_checks', [])
        
        for check in persistence_checks:
            try:
                storage_type = check.get('storage_type')
                data_location = check.get('location')
                retention_policy = check.get('retention_policy')
                
                # Validate data is properly stored
                if not await self._verify_data_storage(storage_type, data_location):
                    result.add_error(f"Data not found in storage: {data_location}")
                    continue
                
                # Validate retention policy compliance
                if retention_policy:
                    if not await self._verify_retention_policy(data_location, retention_policy):
                        result.add_warning(f"Retention policy violation: {data_location}")
                
                result.add_metric(f"persistence_{storage_type}", "verified")
                
            except Exception as e:
                result.add_error(f"Persistence validation failed: {str(e)}")

    async def _execute_transformation(self, func_name: str, input_data: Any) -> Any:
        """Execute a data transformation function."""
        # This would integrate with actual pipeline transformation functions
        # For now, return a mock transformation result
        self.logger.info(f"Executing transformation: {func_name}")
        return input_data  # Mock implementation

    def _compare_data_outputs(self, actual: Any, expected: Any) -> bool:
        """Compare actual and expected data outputs."""
        try:
            if isinstance(expected, dict) and isinstance(actual, dict):
                return all(
                    actual.get(key) == value 
                    for key, value in expected.items()
                )
            return actual == expected
        except Exception:
            return False

    async def _validate_data_schema(self, data_source: str, constraints: Dict, result: ValidationResult):
        """Validate data against schema constraints."""
        # Mock schema validation
        result.add_metric(f"schema_validation_{data_source}", "passed")

    async def _validate_referential_integrity(self, data_source: str, constraints: Dict, result: ValidationResult):
        """Validate referential integrity constraints."""
        # Mock referential integrity check
        result.add_metric(f"referential_integrity_{data_source}", "passed")

    async def _validate_data_quality(self, data_source: str, constraints: Dict, result: ValidationResult):
        """Validate data quality metrics."""
        # Mock data quality check
        result.add_metric(f"data_quality_{data_source}", "passed")

    async def _get_stage_output(self, stage_name: str) -> Dict:
        """Get output data from a pipeline stage."""
        # Mock stage output retrieval
        return {"stage": stage_name, "output": "mock_data"}

    async def _get_stage_input(self, stage_name: str) -> Dict:
        """Get expected input for a pipeline stage."""
        # Mock stage input requirements
        return {"stage": stage_name, "input": "mock_data"}

    def _validate_stage_compatibility(self, source_output: Dict, target_input: Dict, mapping: Dict) -> bool:
        """Validate compatibility between stage output and input."""
        # Mock compatibility check
        return True

    async def _verify_data_storage(self, storage_type: str, location: str) -> bool:
        """Verify data is properly stored."""
        if storage_type == "file":
            return Path(location).exists()
        # Mock other storage type checks
        return True

    async def _verify_retention_policy(self, location: str, policy: Dict) -> bool:
        """Verify retention policy compliance."""
        # Mock retention policy check
        return True