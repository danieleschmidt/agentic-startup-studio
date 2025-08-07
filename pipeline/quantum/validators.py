"""
Quantum Task Planner Validation System

Comprehensive validation framework for quantum tasks, entanglements,
and scheduling operations with security controls and data integrity checks.
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

import numpy as np

from .exceptions import (
    EntanglementCreationError,
    QuantumAmplitudeValidationError,
    QuantumTaskValidationError,
    ValidationError,
)
from .quantum_dependencies import EntanglementType
from .quantum_planner import (
    QuantumAmplitude,
    QuantumPriority,
    QuantumState,
    QuantumTask,
)

logger = logging.getLogger(__name__)


class QuantumTaskValidator:
    """Validator for quantum task integrity and security."""

    # Security constraints
    MAX_TITLE_LENGTH = 500
    MAX_DESCRIPTION_LENGTH = 2000
    MAX_TAGS_COUNT = 20
    MAX_TAG_LENGTH = 50
    MAX_DEPENDENCIES = 100
    MAX_ENTANGLED_TASKS = 50

    # Validation patterns
    SAFE_TEXT_PATTERN = re.compile(r'^[a-zA-Z0-9\s\-_.,!?()[\]{}:;@#$%&*+=<>/\\|~`\'"]+$')
    UUID_PATTERN = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)

    @classmethod
    def validate_task(cls, task: QuantumTask) -> None:
        """
        Comprehensive validation of a quantum task.
        
        Args:
            task: Quantum task to validate
            
        Raises:
            QuantumTaskValidationError: If validation fails
        """
        validation_errors = []

        try:
            # Basic field validation
            cls._validate_task_fields(task, validation_errors)

            # Quantum-specific validation
            cls._validate_quantum_properties(task, validation_errors)

            # Security validation
            cls._validate_security_constraints(task, validation_errors)

            # Business logic validation
            cls._validate_business_rules(task, validation_errors)

            if validation_errors:
                raise QuantumTaskValidationError(task.id, validation_errors)

        except Exception as e:
            if isinstance(e, QuantumTaskValidationError):
                raise
            logger.error(f"Unexpected error validating task {task.id}: {e}")
            raise QuantumTaskValidationError(task.id, [f"Validation error: {str(e)}"])

    @classmethod
    def _validate_task_fields(cls, task: QuantumTask, errors: list[str]) -> None:
        """Validate basic task fields."""

        # ID validation
        if not isinstance(task.id, UUID):
            errors.append("Task ID must be a valid UUID")

        # Title validation
        if not task.title or not task.title.strip():
            errors.append("Task title cannot be empty")
        elif len(task.title) > cls.MAX_TITLE_LENGTH:
            errors.append(f"Task title exceeds maximum length of {cls.MAX_TITLE_LENGTH}")
        elif not cls._is_safe_text(task.title):
            errors.append("Task title contains unsafe characters")

        # Description validation
        if len(task.description) > cls.MAX_DESCRIPTION_LENGTH:
            errors.append(f"Task description exceeds maximum length of {cls.MAX_DESCRIPTION_LENGTH}")
        elif task.description and not cls._is_safe_text(task.description):
            errors.append("Task description contains unsafe characters")

        # Priority validation
        if not isinstance(task.priority, QuantumPriority):
            errors.append("Task priority must be a valid QuantumPriority")

        # Date validation
        if task.due_date and task.due_date < datetime.utcnow() - timedelta(days=365):
            errors.append("Due date cannot be more than 1 year in the past")

        if task.created_at > datetime.utcnow() + timedelta(minutes=5):
            errors.append("Created timestamp cannot be in the future")

        # Duration validation
        if task.estimated_duration <= timedelta(0):
            errors.append("Estimated duration must be positive")
        elif task.estimated_duration > timedelta(days=365):
            errors.append("Estimated duration cannot exceed 1 year")

        # Tags validation
        if len(task.tags) > cls.MAX_TAGS_COUNT:
            errors.append(f"Too many tags (max {cls.MAX_TAGS_COUNT})")

        for tag in task.tags:
            if not isinstance(tag, str):
                errors.append("All tags must be strings")
            elif len(tag) > cls.MAX_TAG_LENGTH:
                errors.append(f"Tag '{tag}' exceeds maximum length of {cls.MAX_TAG_LENGTH}")
            elif not cls._is_safe_text(tag):
                errors.append(f"Tag '{tag}' contains unsafe characters")

        # Dependencies validation
        if len(task.dependencies) > cls.MAX_DEPENDENCIES:
            errors.append(f"Too many dependencies (max {cls.MAX_DEPENDENCIES})")

        for dep_id in task.dependencies:
            if not isinstance(dep_id, UUID):
                errors.append(f"Dependency ID {dep_id} must be a valid UUID")
            elif dep_id == task.id:
                errors.append("Task cannot depend on itself")

        # Entangled tasks validation
        if len(task.entangled_tasks) > cls.MAX_ENTANGLED_TASKS:
            errors.append(f"Too many entangled tasks (max {cls.MAX_ENTANGLED_TASKS})")

        for entangled_id in task.entangled_tasks:
            if not isinstance(entangled_id, UUID):
                errors.append(f"Entangled task ID {entangled_id} must be a valid UUID")
            elif entangled_id == task.id:
                errors.append("Task cannot be entangled with itself")

    @classmethod
    def _validate_quantum_properties(cls, task: QuantumTask, errors: list[str]) -> None:
        """Validate quantum-specific properties."""

        # Current state validation
        if not isinstance(task.current_state, QuantumState):
            errors.append("Current state must be a valid QuantumState")

        # Amplitudes validation
        if task.amplitudes:
            try:
                cls._validate_quantum_amplitudes(task.amplitudes)
            except QuantumAmplitudeValidationError as e:
                errors.extend(e.context.get('validation_failures', [str(e)]))

    @classmethod
    def _validate_security_constraints(cls, task: QuantumTask, errors: list[str]) -> None:
        """Validate security constraints."""

        # Check for potential injection patterns
        dangerous_patterns = [
            r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>',  # Script tags
            r'javascript:',  # JavaScript URLs
            r'on\w+\s*=',  # Event handlers
            r'eval\s*\(',  # Eval function
            r'exec\s*\(',  # Exec function
            r'\$\{.*\}',  # Template injection
            r'<%.*%>',  # Server-side includes
        ]

        text_fields = [task.title, task.description] + task.tags

        for field_value in text_fields:
            if field_value:
                for pattern in dangerous_patterns:
                    if re.search(pattern, field_value, re.IGNORECASE):
                        errors.append("Potentially dangerous content detected in task field")
                        break

        # Metadata validation
        if task.metadata:
            cls._validate_metadata_security(task.metadata, errors)

    @classmethod
    def _validate_business_rules(cls, task: QuantumTask, errors: list[str]) -> None:
        """Validate business logic rules."""

        # State consistency checks
        if (task.current_state == QuantumState.COMPLETED and
            task.updated_at < task.created_at):
            errors.append("Completed task cannot have update time before creation time")

        # Priority-duration correlation check
        if (task.priority == QuantumPriority.IONIZED and
            task.estimated_duration > timedelta(days=30)):
            errors.append("Highest priority tasks should not have duration > 30 days")

        # Dependency cycle prevention (basic check)
        if task.id in task.dependencies:
            errors.append("Task cannot depend on itself (direct cycle)")

    @classmethod
    def _validate_metadata_security(cls, metadata: dict[str, Any], errors: list[str]) -> None:
        """Validate metadata for security issues."""

        MAX_METADATA_SIZE = 10000  # bytes
        MAX_METADATA_DEPTH = 5

        try:
            import json
            metadata_str = json.dumps(metadata)

            if len(metadata_str) > MAX_METADATA_SIZE:
                errors.append(f"Metadata size exceeds limit ({MAX_METADATA_SIZE} bytes)")

            # Check nesting depth
            if cls._get_dict_depth(metadata) > MAX_METADATA_DEPTH:
                errors.append(f"Metadata nesting depth exceeds limit ({MAX_METADATA_DEPTH})")

        except (TypeError, ValueError) as e:
            errors.append(f"Invalid metadata format: {e}")

    @classmethod
    def _get_dict_depth(cls, d: dict[str, Any], depth: int = 0) -> int:
        """Calculate nesting depth of dictionary."""
        if not isinstance(d, dict):
            return depth

        max_depth = depth
        for value in d.values():
            if isinstance(value, dict):
                max_depth = max(max_depth, cls._get_dict_depth(value, depth + 1))

        return max_depth

    @classmethod
    def _is_safe_text(cls, text: str) -> bool:
        """Check if text contains only safe characters."""
        return bool(cls.SAFE_TEXT_PATTERN.match(text))

    @classmethod
    def _validate_quantum_amplitudes(cls, amplitudes: dict[QuantumState, QuantumAmplitude]) -> None:
        """
        Validate quantum amplitudes for mathematical consistency.
        
        Args:
            amplitudes: Dictionary of quantum amplitudes
            
        Raises:
            QuantumAmplitudeValidationError: If validation fails
        """
        if not amplitudes:
            return

        validation_errors = []

        # Check probability normalization
        total_probability = sum(amp.probability for amp in amplitudes.values())

        if abs(total_probability - 1.0) > 0.001:  # Allow small floating point errors
            validation_errors.append(f"Probabilities not normalized: sum = {total_probability}")

        # Check individual amplitude constraints
        for state, amplitude in amplitudes.items():
            if not isinstance(state, QuantumState):
                validation_errors.append(f"Invalid quantum state: {state}")

            if not isinstance(amplitude, QuantumAmplitude):
                validation_errors.append(f"Invalid amplitude object for state {state}")
                continue

            # Probability bounds
            if amplitude.probability < 0 or amplitude.probability > 1:
                validation_errors.append(f"Probability for {state} out of bounds: {amplitude.probability}")

            # Phase bounds
            if amplitude.phase < 0 or amplitude.phase >= 2 * np.pi:
                validation_errors.append(f"Phase for {state} out of bounds: {amplitude.phase}")

        if validation_errors:
            raise QuantumAmplitudeValidationError(
                {str(k): v.probability for k, v in amplitudes.items()},
                "; ".join(validation_errors)
            )


class QuantumEntanglementValidator:
    """Validator for quantum entanglement operations."""

    @classmethod
    def validate_entanglement_creation(cls, task_ids: set[UUID],
                                     entanglement_type: EntanglementType,
                                     strength: float) -> None:
        """
        Validate entanglement creation parameters.
        
        Args:
            task_ids: Set of task IDs to entangle
            entanglement_type: Type of entanglement
            strength: Entanglement strength
            
        Raises:
            EntanglementCreationError: If validation fails
        """
        # Minimum tasks check
        if len(task_ids) < 2:
            raise EntanglementCreationError(
                list(task_ids),
                "At least 2 tasks required for entanglement"
            )

        # Maximum tasks check (prevent system overload)
        MAX_ENTANGLED_TASKS = 10
        if len(task_ids) > MAX_ENTANGLED_TASKS:
            raise EntanglementCreationError(
                list(task_ids),
                f"Too many tasks for entanglement (max {MAX_ENTANGLED_TASKS})"
            )

        # Strength validation
        if not (0.0 <= strength <= 1.0):
            raise EntanglementCreationError(
                list(task_ids),
                f"Entanglement strength must be between 0 and 1, got {strength}"
            )

        # Type validation
        if not isinstance(entanglement_type, EntanglementType):
            raise EntanglementCreationError(
                list(task_ids),
                f"Invalid entanglement type: {entanglement_type}"
            )

        # UUID validation
        for task_id in task_ids:
            if not isinstance(task_id, UUID):
                raise EntanglementCreationError(
                    list(task_ids),
                    f"Invalid task ID format: {task_id}"
                )

    @classmethod
    def validate_entanglement_compatibility(cls, tasks: list[QuantumTask],
                                          entanglement_type: EntanglementType) -> None:
        """
        Validate that tasks are compatible for specific entanglement type.
        
        Args:
            tasks: List of tasks to entangle
            entanglement_type: Type of entanglement
            
        Raises:
            EntanglementCreationError: If tasks are incompatible
        """
        if not tasks:
            return

        task_ids = [task.id for task in tasks]

        # Type-specific compatibility checks
        if entanglement_type == EntanglementType.SYNC_COMPLETION:
            # All tasks should have similar estimated durations for sync completion
            durations = [task.estimated_duration.total_seconds() for task in tasks]
            max_duration = max(durations)
            min_duration = min(durations)

            if max_duration > 0 and (max_duration / min_duration) > 10:
                raise EntanglementCreationError(
                    task_ids,
                    "Tasks with very different durations are not suitable for sync completion"
                )

        elif entanglement_type == EntanglementType.ANTI_CORRELATION:
            # Anti-correlated tasks should not have dependencies between them
            for i, task1 in enumerate(tasks):
                for j, task2 in enumerate(tasks[i+1:], i+1):
                    if (task2.id in task1.dependencies or
                        task1.id in task2.dependencies):
                        raise EntanglementCreationError(
                            task_ids,
                            "Tasks with direct dependencies cannot be anti-correlated"
                        )

        elif entanglement_type == EntanglementType.RESOURCE_SHARE:
            # Resource sharing tasks should have similar priorities
            priorities = [task.priority for task in tasks]
            unique_priorities = set(priorities)

            if len(unique_priorities) > 2:
                raise EntanglementCreationError(
                    task_ids,
                    "Resource sharing works best with tasks of similar priority"
                )


class QuantumSchedulerValidator:
    """Validator for quantum scheduling operations."""

    @classmethod
    def validate_scheduling_parameters(cls, tasks: list[QuantumTask],
                                     max_concurrent: int) -> None:
        """
        Validate scheduling parameters.
        
        Args:
            tasks: List of tasks to schedule
            max_concurrent: Maximum concurrent tasks
            
        Raises:
            ValidationError: If parameters are invalid
        """
        if not tasks:
            raise ValidationError(
                "Cannot schedule empty task list",
                "tasks", tasks, "non_empty_list"
            )

        if max_concurrent <= 0:
            raise ValidationError(
                "Maximum concurrent tasks must be positive",
                "max_concurrent", max_concurrent, "positive_integer"
            )

        # Check for duplicate task IDs
        task_ids = [task.id for task in tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValidationError(
                "Duplicate task IDs found in scheduling list",
                "tasks", len(task_ids), "unique_task_ids"
            )

        # Validate individual tasks
        for task in tasks:
            try:
                QuantumTaskValidator.validate_task(task)
            except QuantumTaskValidationError as e:
                raise ValidationError(
                    f"Invalid task in scheduling list: {e}",
                    "tasks", task.id, "valid_quantum_task"
                )

    @classmethod
    def validate_dependency_graph(cls, tasks: list[QuantumTask]) -> None:
        """
        Validate that task dependencies form a valid DAG.
        
        Args:
            tasks: List of tasks to validate
            
        Raises:
            ValidationError: If dependency graph is invalid
        """
        # Build dependency graph
        task_map = {task.id: task for task in tasks}

        # Check for missing dependencies
        for task in tasks:
            for dep_id in task.dependencies:
                if dep_id not in task_map:
                    raise ValidationError(
                        f"Task {task.id} depends on non-existent task {dep_id}",
                        "dependencies", dep_id, "existing_task_id"
                    )

        # Check for cycles using DFS
        visited = set()
        rec_stack = set()

        def has_cycle(task_id: UUID) -> bool:
            if task_id in rec_stack:
                return True
            if task_id in visited:
                return False

            visited.add(task_id)
            rec_stack.add(task_id)

            if task_id in task_map:
                for dep_id in task_map[task_id].dependencies:
                    if has_cycle(dep_id):
                        return True

            rec_stack.remove(task_id)
            return False

        for task in tasks:
            if task.id not in visited:
                if has_cycle(task.id):
                    raise ValidationError(
                        f"Circular dependency detected involving task {task.id}",
                        "dependency_graph", task.id, "acyclic_graph"
                    )


class SecurityValidator:
    """Security-focused validation for quantum task planner."""

    # Rate limiting constants
    MAX_TASKS_PER_MINUTE = 100
    MAX_ENTANGLEMENTS_PER_MINUTE = 20
    MAX_MEASUREMENTS_PER_MINUTE = 1000

    @classmethod
    def validate_rate_limits(cls, operation_type: str, count: int,
                           time_window_minutes: int = 1) -> None:
        """
        Validate operation rate limits.
        
        Args:
            operation_type: Type of operation
            count: Number of operations
            time_window_minutes: Time window in minutes
            
        Raises:
            ValidationError: If rate limit exceeded
        """
        limits = {
            "task_creation": cls.MAX_TASKS_PER_MINUTE,
            "entanglement_creation": cls.MAX_ENTANGLEMENTS_PER_MINUTE,
            "quantum_measurement": cls.MAX_MEASUREMENTS_PER_MINUTE
        }

        limit = limits.get(operation_type)
        if limit is None:
            return  # No limit defined

        adjusted_limit = limit * time_window_minutes

        if count > adjusted_limit:
            raise ValidationError(
                f"Rate limit exceeded for {operation_type}: {count} > {adjusted_limit}",
                "rate_limit", count, f"max_{adjusted_limit}_per_{time_window_minutes}min"
            )

    @classmethod
    def validate_resource_constraints(cls, current_tasks: int, new_tasks: int,
                                    max_total_tasks: int) -> None:
        """
        Validate resource constraints.
        
        Args:
            current_tasks: Current number of tasks
            new_tasks: Number of new tasks to add
            max_total_tasks: Maximum allowed total tasks
            
        Raises:
            ValidationError: If resource constraints violated
        """
        total_after_addition = current_tasks + new_tasks

        if total_after_addition > max_total_tasks:
            raise ValidationError(
                f"Would exceed maximum tasks: {total_after_addition} > {max_total_tasks}",
                "resource_constraint", total_after_addition, f"max_{max_total_tasks}"
            )

    @classmethod
    def sanitize_user_input(cls, text: str) -> str:
        """
        Sanitize user input to prevent injection attacks.
        
        Args:
            text: Input text to sanitize
            
        Returns:
            Sanitized text
        """
        if not isinstance(text, str):
            return str(text)

        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '"', "'", '&', '\x00', '\r', '\n']
        sanitized = text

        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')

        # Limit length
        MAX_INPUT_LENGTH = 10000
        if len(sanitized) > MAX_INPUT_LENGTH:
            sanitized = sanitized[:MAX_INPUT_LENGTH]

        return sanitized.strip()
