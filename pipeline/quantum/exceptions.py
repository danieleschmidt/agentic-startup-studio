"""
Quantum Task Planner Exception Hierarchy

Comprehensive exception handling for quantum-inspired task planning system
with specific error types for quantum operations, entanglement failures,
and scheduling conflicts.
"""

from typing import Any, Optional, List, Dict
from uuid import UUID


class QuantumPlannerError(Exception):
    """Base exception for all quantum planner errors."""
    
    def __init__(self, message: str, error_code: str = "QP_GENERIC", 
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}
        self.message = message
    
    def __str__(self) -> str:
        context_str = f" Context: {self.context}" if self.context else ""
        return f"[{self.error_code}] {self.message}{context_str}"


class QuantumStateError(QuantumPlannerError):
    """Errors related to quantum state operations."""
    
    def __init__(self, message: str, task_id: Optional[UUID] = None, 
                 current_state: Optional[str] = None):
        context = {}
        if task_id:
            context["task_id"] = str(task_id)
        if current_state:
            context["current_state"] = current_state
        
        super().__init__(message, "QP_STATE_ERROR", context)


class SuperpositionCollapseError(QuantumStateError):
    """Error during quantum superposition collapse."""
    
    def __init__(self, task_id: UUID, amplitudes: Optional[Dict[str, float]] = None):
        message = f"Failed to collapse superposition for task {task_id}"
        context = {"task_id": str(task_id)}
        if amplitudes:
            context["amplitudes"] = amplitudes
        
        super().__init__(message, task_id)
        self.error_code = "QP_SUPERPOSITION_COLLAPSE"


class QuantumMeasurementError(QuantumStateError):
    """Error during quantum measurement operations."""
    
    def __init__(self, message: str, measurement_type: str, task_id: Optional[UUID] = None):
        context = {"measurement_type": measurement_type}
        if task_id:
            context["task_id"] = str(task_id)
        
        super().__init__(message, task_id)
        self.error_code = "QP_MEASUREMENT_ERROR"


class EntanglementError(QuantumPlannerError):
    """Errors related to quantum entanglement operations."""
    
    def __init__(self, message: str, entanglement_id: Optional[UUID] = None,
                 task_ids: Optional[List[UUID]] = None):
        context = {}
        if entanglement_id:
            context["entanglement_id"] = str(entanglement_id)
        if task_ids:
            context["task_ids"] = [str(tid) for tid in task_ids]
        
        super().__init__(message, "QP_ENTANGLEMENT_ERROR", context)


class EntanglementCreationError(EntanglementError):
    """Error during entanglement creation."""
    
    def __init__(self, task_ids: List[UUID], reason: str):
        message = f"Cannot create entanglement between tasks {task_ids}: {reason}"
        super().__init__(message, task_ids=task_ids)
        self.error_code = "QP_ENTANGLEMENT_CREATE"


class EntanglementCorrelationError(EntanglementError):
    """Error in quantum correlation calculations."""
    
    def __init__(self, task1_id: UUID, task2_id: UUID, correlation_type: str):
        message = f"Failed to calculate {correlation_type} correlation between {task1_id} and {task2_id}"
        super().__init__(message, task_ids=[task1_id, task2_id])
        self.error_code = "QP_CORRELATION_ERROR"


class SchedulingError(QuantumPlannerError):
    """Errors related to quantum scheduling operations."""
    
    def __init__(self, message: str, task_count: Optional[int] = None,
                 strategy: Optional[str] = None):
        context = {}
        if task_count is not None:
            context["task_count"] = task_count
        if strategy:
            context["strategy"] = strategy
        
        super().__init__(message, "QP_SCHEDULING_ERROR", context)


class QuantumSchedulingOptimizationError(SchedulingError):
    """Error during quantum scheduling optimization."""
    
    def __init__(self, strategy: str, error_details: str):
        message = f"Quantum scheduling optimization failed for strategy '{strategy}': {error_details}"
        super().__init__(message, strategy=strategy)
        self.error_code = "QP_SCHEDULING_OPTIMIZATION"


class QuantumAnnealingError(SchedulingError):
    """Error during quantum annealing process."""
    
    def __init__(self, iteration: int, temperature: float, energy_diff: float):
        message = f"Quantum annealing failed at iteration {iteration}"
        context = {
            "iteration": iteration,
            "temperature": temperature,
            "energy_diff": energy_diff
        }
        super().__init__(message)
        self.error_code = "QP_ANNEALING_ERROR"
        self.context.update(context)


class DependencyError(QuantumPlannerError):
    """Errors related to dependency management."""
    
    def __init__(self, message: str, task_id: Optional[UUID] = None,
                 dependency_chain: Optional[List[UUID]] = None):
        context = {}
        if task_id:
            context["task_id"] = str(task_id)
        if dependency_chain:
            context["dependency_chain"] = [str(tid) for tid in dependency_chain]
        
        super().__init__(message, "QP_DEPENDENCY_ERROR", context)


class CircularDependencyError(DependencyError):
    """Circular dependency detected in task graph."""
    
    def __init__(self, cycle_tasks: List[UUID]):
        message = f"Circular dependency detected in task chain: {' -> '.join(str(t) for t in cycle_tasks)}"
        super().__init__(message, dependency_chain=cycle_tasks)
        self.error_code = "QP_CIRCULAR_DEPENDENCY"


class DependencyResolutionError(DependencyError):
    """Error during quantum dependency resolution."""
    
    def __init__(self, task_id: UUID, unresolved_dependencies: List[UUID]):
        message = f"Cannot resolve dependencies for task {task_id}"
        context = {
            "task_id": str(task_id),
            "unresolved_dependencies": [str(tid) for tid in unresolved_dependencies]
        }
        super().__init__(message, task_id)
        self.error_code = "QP_DEPENDENCY_RESOLUTION"
        self.context.update(context)


class QuantumInterferenceError(QuantumPlannerError):
    """Error during quantum interference calculations."""
    
    def __init__(self, operation: str, phase_info: Optional[Dict[str, float]] = None):
        message = f"Quantum interference error during {operation}"
        context = {"operation": operation}
        if phase_info:
            context["phase_info"] = phase_info
        
        super().__init__(message, "QP_INTERFERENCE_ERROR", context)


class QuantumCoherenceError(QuantumPlannerError):
    """Error related to quantum coherence calculations."""
    
    def __init__(self, coherence_value: float, threshold: float):
        message = f"Quantum coherence {coherence_value} below threshold {threshold}"
        context = {
            "coherence_value": coherence_value,
            "threshold": threshold
        }
        super().__init__(message, "QP_COHERENCE_ERROR", context)


class TaskExecutionError(QuantumPlannerError):
    """Errors during task execution."""
    
    def __init__(self, message: str, task_id: UUID, execution_phase: str):
        context = {
            "task_id": str(task_id),
            "execution_phase": execution_phase
        }
        super().__init__(message, "QP_EXECUTION_ERROR", context)


class QuantumTunnelingError(TaskExecutionError):
    """Error during quantum tunneling operations."""
    
    def __init__(self, task_id: UUID, tunneling_probability: float):
        message = f"Quantum tunneling failed for task {task_id}"
        super().__init__(message, task_id, "quantum_tunneling")
        self.error_code = "QP_TUNNELING_ERROR"
        self.context["tunneling_probability"] = tunneling_probability


class ValidationError(QuantumPlannerError):
    """Validation errors for quantum task planner."""
    
    def __init__(self, message: str, field: str, value: Any, 
                 validation_rule: str):
        context = {
            "field": field,
            "value": str(value),
            "validation_rule": validation_rule
        }
        super().__init__(message, "QP_VALIDATION_ERROR", context)


class QuantumTaskValidationError(ValidationError):
    """Validation error specific to quantum tasks."""
    
    def __init__(self, task_id: UUID, validation_failures: List[str]):
        message = f"Quantum task {task_id} failed validation"
        context = {
            "task_id": str(task_id),
            "validation_failures": validation_failures
        }
        super().__init__(message, "task", task_id, "quantum_task_validation")
        self.error_code = "QP_TASK_VALIDATION"


class QuantumAmplitudeValidationError(ValidationError):
    """Validation error for quantum amplitudes."""
    
    def __init__(self, amplitudes: Dict[str, float], issue: str):
        message = f"Quantum amplitude validation failed: {issue}"
        super().__init__(message, "amplitudes", amplitudes, "amplitude_normalization")
        self.error_code = "QP_AMPLITUDE_VALIDATION"


class ResourceError(QuantumPlannerError):
    """Resource-related errors in quantum planning."""
    
    def __init__(self, message: str, resource_type: str, 
                 requested: Optional[int] = None, available: Optional[int] = None):
        context = {"resource_type": resource_type}
        if requested is not None:
            context["requested"] = requested
        if available is not None:
            context["available"] = available
        
        super().__init__(message, "QP_RESOURCE_ERROR", context)


class QuantumResourceExhaustionError(ResourceError):
    """Quantum computational resources exhausted."""
    
    def __init__(self, resource_type: str, required: int, available: int):
        message = f"Quantum {resource_type} resources exhausted: need {required}, have {available}"
        super().__init__(message, resource_type, required, available)
        self.error_code = "QP_RESOURCE_EXHAUSTION"


class ConcurrencyError(QuantumPlannerError):
    """Concurrency-related errors."""
    
    def __init__(self, message: str, concurrent_tasks: int, max_allowed: int):
        context = {
            "concurrent_tasks": concurrent_tasks,
            "max_allowed": max_allowed
        }
        super().__init__(message, "QP_CONCURRENCY_ERROR", context)


class QuantumDecoherenceError(QuantumPlannerError):
    """Error due to quantum decoherence."""
    
    def __init__(self, decoherence_rate: float, time_elapsed: float):
        message = f"Quantum decoherence detected: rate {decoherence_rate}, time {time_elapsed}"
        context = {
            "decoherence_rate": decoherence_rate,
            "time_elapsed": time_elapsed
        }
        super().__init__(message, "QP_DECOHERENCE_ERROR", context)


class ConfigurationError(QuantumPlannerError):
    """Configuration-related errors."""
    
    def __init__(self, message: str, config_key: str, config_value: Any):
        context = {
            "config_key": config_key,
            "config_value": str(config_value)
        }
        super().__init__(message, "QP_CONFIG_ERROR", context)