"""
Quality Gates System - Automated quality control and validation.

Implements quality gates for pipeline stages with configurable criteria,
automated validation, and quality-based workflow control.
"""

import asyncio
import logging
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pipeline.config.settings import get_settings
from pipeline.events.event_bus import DomainEvent, EventType, get_event_bus

logger = logging.getLogger(__name__)


class QualityGateStatus(Enum):
    """Quality gate evaluation status."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    PENDING = "pending"
    SKIPPED = "skipped"


class QualityCriteriaType(Enum):
    """Types of quality criteria."""
    THRESHOLD = "threshold"          # Numeric threshold check
    BOOLEAN = "boolean"              # True/false validation
    RANGE = "range"                  # Value within range
    PATTERN = "pattern"              # String pattern matching
    CUSTOM = "custom"                # Custom validation function
    COMPOSITE = "composite"          # Multiple criteria combination


class QualityLevel(Enum):
    """Quality requirement levels."""
    CRITICAL = "critical"            # Must pass - blocks pipeline
    HIGH = "high"                    # Should pass - creates warnings
    MEDIUM = "medium"                # Nice to pass - informational
    LOW = "low"                      # Optional - monitoring only


@dataclass
class QualityCriteria:
    """Definition of quality criteria for validation."""

    name: str
    description: str
    criteria_type: QualityCriteriaType
    level: QualityLevel

    # Threshold criteria
    threshold_value: float | None = None
    comparison_operator: str = "gte"  # gte, lte, gt, lt, eq, ne

    # Range criteria
    min_value: float | None = None
    max_value: float | None = None

    # Pattern criteria
    pattern: str | None = None

    # Boolean criteria
    expected_value: bool | None = None

    # Custom validation
    custom_validator: Callable | None = None

    # Metadata
    tags: list[str] = field(default_factory=list)
    owner: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def validate(self, value: Any) -> bool:
        """Validate value against criteria."""
        try:
            if self.criteria_type == QualityCriteriaType.THRESHOLD:
                return self._validate_threshold(value)
            if self.criteria_type == QualityCriteriaType.BOOLEAN:
                return self._validate_boolean(value)
            if self.criteria_type == QualityCriteriaType.RANGE:
                return self._validate_range(value)
            if self.criteria_type == QualityCriteriaType.PATTERN:
                return self._validate_pattern(value)
            if self.criteria_type == QualityCriteriaType.CUSTOM:
                return self._validate_custom(value)
            logger.warning(f"Unknown criteria type: {self.criteria_type}")
            return False
        except Exception as e:
            logger.error(f"Validation error for criteria '{self.name}': {e}")
            return False

    def _validate_threshold(self, value: int | float) -> bool:
        """Validate threshold criteria."""
        if self.threshold_value is None:
            return False

        numeric_value = float(value)
        threshold = self.threshold_value

        if self.comparison_operator == "gte":
            return numeric_value >= threshold
        if self.comparison_operator == "lte":
            return numeric_value <= threshold
        if self.comparison_operator == "gt":
            return numeric_value > threshold
        if self.comparison_operator == "lt":
            return numeric_value < threshold
        if self.comparison_operator == "eq":
            return abs(numeric_value - threshold) < 1e-9
        if self.comparison_operator == "ne":
            return abs(numeric_value - threshold) >= 1e-9
        return False

    def _validate_boolean(self, value: Any) -> bool:
        """Validate boolean criteria."""
        if self.expected_value is None:
            return False
        return bool(value) == self.expected_value

    def _validate_range(self, value: int | float) -> bool:
        """Validate range criteria."""
        numeric_value = float(value)

        if self.min_value is not None and numeric_value < self.min_value:
            return False
        if self.max_value is not None and numeric_value > self.max_value:
            return False

        return True

    def _validate_pattern(self, value: str) -> bool:
        """Validate pattern criteria."""
        if self.pattern is None:
            return False

        import re
        return bool(re.match(self.pattern, str(value)))

    def _validate_custom(self, value: Any) -> bool:
        """Validate using custom function."""
        if self.custom_validator is None:
            return False

        return self.custom_validator(value)


@dataclass
class QualityResult:
    """Result of quality criteria evaluation."""

    criteria_name: str
    status: QualityGateStatus
    actual_value: Any
    expected_value: Any
    score: float  # 0.0 to 1.0
    message: str

    # Metadata
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    evaluation_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'criteria_name': self.criteria_name,
            'status': self.status.value,
            'actual_value': self.actual_value,
            'expected_value': self.expected_value,
            'score': self.score,
            'message': self.message,
            'evaluated_at': self.evaluated_at.isoformat(),
            'evaluation_time_ms': self.evaluation_time_ms
        }


@dataclass
class QualityGateConfig:
    """Configuration for quality gate."""

    name: str
    description: str
    criteria: list[QualityCriteria]

    # Gate behavior
    require_all_critical: bool = True    # All critical criteria must pass
    warning_threshold: float = 0.8       # Overall score threshold for warnings
    failure_threshold: float = 0.6       # Overall score threshold for failure

    # Execution settings
    timeout_seconds: int = 60
    retry_attempts: int = 3
    retry_delay: float = 1.0

    # Reporting
    generate_report: bool = True
    store_history: bool = True
    alert_on_failure: bool = True


@dataclass
class QualityGateResult:
    """Complete quality gate evaluation result."""

    gate_name: str
    overall_status: QualityGateStatus
    overall_score: float
    criteria_results: list[QualityResult]

    # Statistics
    total_criteria: int
    passed_criteria: int
    failed_criteria: int
    warning_criteria: int

    # Execution metadata
    execution_id: str
    started_at: datetime
    completed_at: datetime
    execution_time_seconds: float

    # Context
    pipeline_stage: str | None = None
    aggregate_id: str | None = None

    def get_critical_failures(self) -> list[QualityResult]:
        """Get results for failed critical criteria."""
        return [
            result for result in self.criteria_results
            if result.status == QualityGateStatus.FAILED
        ]

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics."""
        return {
            'gate_name': self.gate_name,
            'overall_status': self.overall_status.value,
            'overall_score': self.overall_score,
            'execution_time_seconds': self.execution_time_seconds,
            'statistics': {
                'total_criteria': self.total_criteria,
                'passed_criteria': self.passed_criteria,
                'failed_criteria': self.failed_criteria,
                'warning_criteria': self.warning_criteria,
                'pass_rate': self.passed_criteria / self.total_criteria if self.total_criteria > 0 else 0.0
            },
            'critical_failures': len(self.get_critical_failures())
        }


class QualityGate:
    """
    Quality gate implementation for automated validation.
    
    Evaluates multiple quality criteria and determines if pipeline
    stage meets quality requirements for progression.
    """

    def __init__(self, config: QualityGateConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{config.name}")
        self.event_bus = get_event_bus()

        # Execution tracking
        self.execution_history: list[QualityGateResult] = []
        self._lock = asyncio.Lock()

        # Metrics
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0

        self.logger.info(f"Quality gate '{config.name}' initialized with {len(config.criteria)} criteria")

    async def evaluate(
        self,
        data: dict[str, Any],
        context: dict[str, Any] | None = None
    ) -> QualityGateResult:
        """
        Evaluate quality gate against provided data.
        
        Args:
            data: Data to evaluate against criteria
            context: Additional context information
            
        Returns:
            Complete quality gate evaluation result
        """
        async with self._lock:
            execution_id = f"{self.config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            started_at = datetime.now(UTC)

            context = context or {}

            self.logger.info(
                f"Starting quality gate evaluation [{execution_id}]",
                extra={
                    'gate_name': self.config.name,
                    'execution_id': execution_id,
                    'criteria_count': len(self.config.criteria)
                }
            )

            try:
                # Evaluate all criteria
                criteria_results = await self._evaluate_criteria(data, context)

                # Calculate overall result
                completed_at = datetime.now(UTC)
                execution_time = (completed_at - started_at).total_seconds()

                result = await self._calculate_overall_result(
                    execution_id=execution_id,
                    started_at=started_at,
                    completed_at=completed_at,
                    execution_time=execution_time,
                    criteria_results=criteria_results,
                    context=context
                )

                # Update metrics
                self.total_executions += 1
                if result.overall_status == QualityGateStatus.PASSED:
                    self.successful_executions += 1
                else:
                    self.failed_executions += 1

                # Store history
                if self.config.store_history:
                    self.execution_history.append(result)
                    # Keep only last 100 executions
                    if len(self.execution_history) > 100:
                        self.execution_history = self.execution_history[-100:]

                # Publish events
                await self._publish_result_events(result)

                self.logger.info(
                    f"Quality gate evaluation completed [{execution_id}]: {result.overall_status.value}",
                    extra={
                        'gate_name': self.config.name,
                        'execution_id': execution_id,
                        'overall_status': result.overall_status.value,
                        'overall_score': result.overall_score,
                        'execution_time_seconds': execution_time
                    }
                )

                return result

            except Exception as e:
                self.failed_executions += 1
                self.logger.error(
                    f"Quality gate evaluation failed [{execution_id}]: {e}",
                    extra={
                        'gate_name': self.config.name,
                        'execution_id': execution_id,
                        'error': str(e)
                    }
                )
                raise

    async def _evaluate_criteria(
        self,
        data: dict[str, Any],
        context: dict[str, Any]
    ) -> list[QualityResult]:
        """Evaluate all criteria against data."""
        results = []

        for criteria in self.config.criteria:
            start_time = asyncio.get_event_loop().time()

            try:
                # Extract value for criteria
                value = self._extract_value(data, criteria.name, context)

                # Validate criteria
                is_valid = criteria.validate(value)

                # Calculate score
                score = 1.0 if is_valid else 0.0

                # Determine status
                if is_valid:
                    status = QualityGateStatus.PASSED
                    message = f"Criteria '{criteria.name}' passed validation"
                elif criteria.level == QualityLevel.CRITICAL:
                    status = QualityGateStatus.FAILED
                    message = f"Critical criteria '{criteria.name}' failed validation"
                elif criteria.level == QualityLevel.HIGH:
                    status = QualityGateStatus.WARNING
                    message = f"High priority criteria '{criteria.name}' failed validation"
                else:
                    status = QualityGateStatus.WARNING
                    message = f"Criteria '{criteria.name}' failed validation"

                evaluation_time = (asyncio.get_event_loop().time() - start_time) * 1000

                result = QualityResult(
                    criteria_name=criteria.name,
                    status=status,
                    actual_value=value,
                    expected_value=self._get_expected_value(criteria),
                    score=score,
                    message=message,
                    evaluation_time_ms=evaluation_time
                )

                results.append(result)

            except Exception as e:
                evaluation_time = (asyncio.get_event_loop().time() - start_time) * 1000

                result = QualityResult(
                    criteria_name=criteria.name,
                    status=QualityGateStatus.FAILED,
                    actual_value=None,
                    expected_value=self._get_expected_value(criteria),
                    score=0.0,
                    message=f"Evaluation error: {str(e)}",
                    evaluation_time_ms=evaluation_time
                )

                results.append(result)

                self.logger.error(
                    f"Error evaluating criteria '{criteria.name}': {e}",
                    extra={
                        'gate_name': self.config.name,
                        'criteria_name': criteria.name
                    }
                )

        return results

    def _extract_value(self, data: dict[str, Any], criteria_name: str, context: dict[str, Any]) -> Any:
        """Extract value for criteria from data."""
        # Try direct lookup first
        if criteria_name in data:
            return data[criteria_name]

        # Try nested lookup with dot notation
        keys = criteria_name.split('.')
        value = data
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                # Try context if not found in data
                if criteria_name in context:
                    return context[criteria_name]
                raise ValueError(f"Value not found for criteria '{criteria_name}'")

        return value

    def _get_expected_value(self, criteria: QualityCriteria) -> Any:
        """Get expected value for criteria."""
        if criteria.criteria_type == QualityCriteriaType.THRESHOLD:
            return f"{criteria.comparison_operator} {criteria.threshold_value}"
        if criteria.criteria_type == QualityCriteriaType.BOOLEAN:
            return criteria.expected_value
        if criteria.criteria_type == QualityCriteriaType.RANGE:
            return f"[{criteria.min_value}, {criteria.max_value}]"
        if criteria.criteria_type == QualityCriteriaType.PATTERN:
            return criteria.pattern
        return "Custom validation"

    async def _calculate_overall_result(
        self,
        execution_id: str,
        started_at: datetime,
        completed_at: datetime,
        execution_time: float,
        criteria_results: list[QualityResult],
        context: dict[str, Any]
    ) -> QualityGateResult:
        """Calculate overall quality gate result."""

        # Count results by status
        total_criteria = len(criteria_results)
        passed_criteria = sum(1 for r in criteria_results if r.status == QualityGateStatus.PASSED)
        failed_criteria = sum(1 for r in criteria_results if r.status == QualityGateStatus.FAILED)
        warning_criteria = sum(1 for r in criteria_results if r.status == QualityGateStatus.WARNING)

        # Calculate overall score (weighted by criteria level)
        total_weight = 0.0
        weighted_score = 0.0

        for result in criteria_results:
            # Find corresponding criteria to get level
            criteria = next((c for c in self.config.criteria if c.name == result.criteria_name), None)
            if criteria:
                weight = self._get_criteria_weight(criteria.level)
                total_weight += weight
                weighted_score += result.score * weight

        overall_score = weighted_score / total_weight if total_weight > 0 else 0.0

        # Determine overall status
        critical_failures = [
            r for r in criteria_results
            if r.status == QualityGateStatus.FAILED
        ]

        if critical_failures and self.config.require_all_critical or overall_score < self.config.failure_threshold:
            overall_status = QualityGateStatus.FAILED
        elif overall_score < self.config.warning_threshold:
            overall_status = QualityGateStatus.WARNING
        else:
            overall_status = QualityGateStatus.PASSED

        return QualityGateResult(
            gate_name=self.config.name,
            overall_status=overall_status,
            overall_score=overall_score,
            criteria_results=criteria_results,
            total_criteria=total_criteria,
            passed_criteria=passed_criteria,
            failed_criteria=failed_criteria,
            warning_criteria=warning_criteria,
            execution_id=execution_id,
            started_at=started_at,
            completed_at=completed_at,
            execution_time_seconds=execution_time,
            pipeline_stage=context.get('pipeline_stage'),
            aggregate_id=context.get('aggregate_id')
        )

    def _get_criteria_weight(self, level: QualityLevel) -> float:
        """Get weight for criteria level."""
        weights = {
            QualityLevel.CRITICAL: 4.0,
            QualityLevel.HIGH: 3.0,
            QualityLevel.MEDIUM: 2.0,
            QualityLevel.LOW: 1.0
        }
        return weights.get(level, 1.0)

    async def _publish_result_events(self, result: QualityGateResult) -> None:
        """Publish quality gate result events."""

        # Publish quality gate result event
        event_data = {
            'gate_name': self.config.name,
            'execution_id': result.execution_id,
            'overall_status': result.overall_status.value,
            'overall_score': result.overall_score,
            'execution_time_seconds': result.execution_time_seconds,
            'pipeline_stage': result.pipeline_stage,
            'aggregate_id': result.aggregate_id,
            'summary': result.get_summary()
        }

        # Choose event type based on result
        if result.overall_status == QualityGateStatus.FAILED:
            event_type = EventType.QUALITY_GATE_FAILED
        else:
            event_type = EventType.WORKFLOW_STATE_CHANGED  # Use generic event for success

        await self.event_bus.publish(DomainEvent(
            event_type=event_type,
            aggregate_id=result.aggregate_id or self.config.name,
            event_data=event_data
        ))

    async def get_metrics(self) -> dict[str, Any]:
        """Get quality gate metrics."""
        success_rate = (
            self.successful_executions / self.total_executions
            if self.total_executions > 0 else 0.0
        )

        # Calculate average scores from recent history
        recent_scores = [r.overall_score for r in self.execution_history[-20:]]
        avg_score = statistics.mean(recent_scores) if recent_scores else 0.0

        return {
            'gate_name': self.config.name,
            'total_executions': self.total_executions,
            'successful_executions': self.successful_executions,
            'failed_executions': self.failed_executions,
            'success_rate': success_rate,
            'average_score': avg_score,
            'criteria_count': len(self.config.criteria),
            'last_execution': (
                self.execution_history[-1].completed_at.isoformat()
                if self.execution_history else None
            )
        }

    async def get_health_status(self) -> dict[str, Any]:
        """Get quality gate health status."""
        recent_results = self.execution_history[-10:]

        health_score = 1.0
        if recent_results:
            # Calculate health based on recent success rate
            recent_successes = sum(
                1 for r in recent_results
                if r.overall_status == QualityGateStatus.PASSED
            )
            health_score = recent_successes / len(recent_results)

        status = "healthy" if health_score >= 0.8 else "degraded" if health_score >= 0.5 else "unhealthy"

        return {
            'gate_name': self.config.name,
            'status': status,
            'health_score': health_score,
            'recent_executions': len(recent_results),
            'metrics': await self.get_metrics()
        }


class QualityGateManager:
    """Manager for multiple quality gates."""

    def __init__(self):
        self.gates: dict[str, QualityGate] = {}
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)

    def register_gate(self, config: QualityGateConfig) -> QualityGate:
        """Register a new quality gate."""
        gate = QualityGate(config)
        self.gates[config.name] = gate

        self.logger.info(f"Registered quality gate: {config.name}")
        return gate

    def get_gate(self, name: str) -> QualityGate | None:
        """Get quality gate by name."""
        return self.gates.get(name)

    async def evaluate_gate(
        self,
        gate_name: str,
        data: dict[str, Any],
        context: dict[str, Any] | None = None
    ) -> QualityGateResult:
        """Evaluate specific quality gate."""
        gate = self.get_gate(gate_name)
        if not gate:
            raise ValueError(f"Quality gate '{gate_name}' not found")

        return await gate.evaluate(data, context)

    async def evaluate_pipeline_stage(
        self,
        stage_name: str,
        data: dict[str, Any],
        context: dict[str, Any] | None = None
    ) -> dict[str, QualityGateResult]:
        """Evaluate all quality gates for a pipeline stage."""
        context = context or {}
        context['pipeline_stage'] = stage_name

        results = {}
        stage_gates = [
            gate for gate in self.gates.values()
            if stage_name in gate.config.tags or not gate.config.tags
        ]

        for gate in stage_gates:
            try:
                result = await gate.evaluate(data, context)
                results[gate.config.name] = result
            except Exception as e:
                self.logger.error(f"Failed to evaluate gate {gate.config.name}: {e}")

        return results

    async def get_all_metrics(self) -> dict[str, Any]:
        """Get metrics for all quality gates."""
        metrics = {}
        for name, gate in self.gates.items():
            metrics[name] = await gate.get_metrics()

        return {
            'gates': metrics,
            'summary': {
                'total_gates': len(self.gates),
                'total_executions': sum(m['total_executions'] for m in metrics.values()),
                'average_success_rate': (
                    sum(m['success_rate'] for m in metrics.values()) / len(metrics)
                    if metrics else 0.0
                )
            }
        }

    async def get_health_dashboard(self) -> dict[str, Any]:
        """Get comprehensive health dashboard."""
        health_data = {}

        for name, gate in self.gates.items():
            health_data[name] = await gate.get_health_status()

        # Calculate overall system health
        health_scores = [data['health_score'] for data in health_data.values()]
        overall_health = statistics.mean(health_scores) if health_scores else 1.0

        healthy_gates = sum(1 for data in health_data.values() if data['status'] == 'healthy')

        return {
            'overall_health_score': overall_health,
            'overall_status': (
                'healthy' if overall_health >= 0.8 else
                'degraded' if overall_health >= 0.5 else
                'unhealthy'
            ),
            'gates': health_data,
            'summary': {
                'total_gates': len(self.gates),
                'healthy_gates': healthy_gates,
                'degraded_gates': len(self.gates) - healthy_gates
            }
        }


# Singleton manager
_quality_gate_manager: QualityGateManager | None = None


def get_quality_gate_manager() -> QualityGateManager:
    """Get singleton quality gate manager."""
    global _quality_gate_manager
    if _quality_gate_manager is None:
        _quality_gate_manager = QualityGateManager()
    return _quality_gate_manager


# Pre-defined quality gate configurations for common pipeline stages
def create_idea_validation_gate() -> QualityGateConfig:
    """Create quality gate for idea validation stage."""
    return QualityGateConfig(
        name="idea_validation",
        description="Quality gate for startup idea validation",
        criteria=[
            QualityCriteria(
                name="overall_score",
                description="Overall validation score must be >= 0.7",
                criteria_type=QualityCriteriaType.THRESHOLD,
                level=QualityLevel.CRITICAL,
                threshold_value=0.7,
                comparison_operator="gte"
            ),
            QualityCriteria(
                name="market_score",
                description="Market opportunity score",
                criteria_type=QualityCriteriaType.THRESHOLD,
                level=QualityLevel.HIGH,
                threshold_value=0.6,
                comparison_operator="gte"
            ),
            QualityCriteria(
                name="technical_feasibility",
                description="Technical feasibility assessment",
                criteria_type=QualityCriteriaType.BOOLEAN,
                level=QualityLevel.CRITICAL,
                expected_value=True
            )
        ],
        require_all_critical=True,
        warning_threshold=0.8,
        failure_threshold=0.6
    )


def create_evidence_collection_gate() -> QualityGateConfig:
    """Create quality gate for evidence collection stage."""
    return QualityGateConfig(
        name="evidence_collection",
        description="Quality gate for evidence collection validation",
        criteria=[
            QualityCriteria(
                name="evidence_count",
                description="Minimum number of evidence items",
                criteria_type=QualityCriteriaType.THRESHOLD,
                level=QualityLevel.CRITICAL,
                threshold_value=5,
                comparison_operator="gte"
            ),
            QualityCriteria(
                name="average_quality_score",
                description="Average evidence quality score",
                criteria_type=QualityCriteriaType.THRESHOLD,
                level=QualityLevel.HIGH,
                threshold_value=0.7,
                comparison_operator="gte"
            ),
            QualityCriteria(
                name="domain_coverage",
                description="Number of research domains covered",
                criteria_type=QualityCriteriaType.THRESHOLD,
                level=QualityLevel.MEDIUM,
                threshold_value=3,
                comparison_operator="gte"
            )
        ]
    )


def create_pitch_deck_gate() -> QualityGateConfig:
    """Create quality gate for pitch deck generation."""
    return QualityGateConfig(
        name="pitch_deck_generation",
        description="Quality gate for pitch deck quality validation",
        criteria=[
            QualityCriteria(
                name="overall_quality_score",
                description="Overall pitch deck quality",
                criteria_type=QualityCriteriaType.THRESHOLD,
                level=QualityLevel.CRITICAL,
                threshold_value=0.8,
                comparison_operator="gte"
            ),
            QualityCriteria(
                name="slide_count",
                description="Number of slides in acceptable range",
                criteria_type=QualityCriteriaType.RANGE,
                level=QualityLevel.HIGH,
                min_value=8,
                max_value=15
            ),
            QualityCriteria(
                name="evidence_strength_score",
                description="Evidence integration strength",
                criteria_type=QualityCriteriaType.THRESHOLD,
                level=QualityLevel.HIGH,
                threshold_value=0.7,
                comparison_operator="gte"
            )
        ]
    )
