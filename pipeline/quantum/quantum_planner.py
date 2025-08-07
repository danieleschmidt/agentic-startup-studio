"""
Quantum-Inspired Task Planner

Implements quantum computing principles for advanced task scheduling:
- Quantum superposition: Tasks exist in multiple states simultaneously
- Quantum entanglement: Task dependencies are correlated across the system
- Quantum interference: Optimization through constructive/destructive interference
- Quantum measurement: Collapsing superposition to deterministic execution
"""

import logging
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class QuantumState(str, Enum):
    """Quantum states for tasks following superposition principle."""
    SUPERPOSITION = "superposition"  # Task exists in multiple potential states
    PENDING = "pending"              # Task ready for execution
    EXECUTING = "executing"          # Task currently running
    COMPLETED = "completed"          # Task finished successfully
    FAILED = "failed"               # Task execution failed
    CANCELLED = "cancelled"         # Task was cancelled
    BLOCKED = "blocked"             # Task blocked by dependencies


class QuantumPriority(str, Enum):
    """Priority levels with quantum-inspired naming."""
    GROUND_STATE = "ground"      # Lowest energy/priority
    EXCITED_1 = "excited_1"      # Medium-low priority
    EXCITED_2 = "excited_2"      # Medium priority
    EXCITED_3 = "excited_3"      # Medium-high priority
    IONIZED = "ionized"          # Highest energy/priority


@dataclass
class QuantumAmplitude:
    """Quantum amplitude representing probability of task execution path."""
    state: QuantumState
    probability: float = 0.0
    phase: float = 0.0  # Quantum phase for interference calculations

    def __post_init__(self):
        """Ensure probability is normalized."""
        if self.probability < 0 or self.probability > 1:
            self.probability = max(0, min(1, self.probability))


class QuantumTask(BaseModel):
    """
    Quantum-inspired task representation with superposition capabilities.
    
    Tasks can exist in multiple states simultaneously until measurement
    (execution) collapses them to a definitive state.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: UUID = Field(default_factory=uuid4)
    title: str = Field(..., min_length=1, max_length=500)
    description: str = Field(default="", max_length=2000)
    priority: QuantumPriority = Field(default=QuantumPriority.EXCITED_2)

    # Quantum properties
    amplitudes: dict[QuantumState, QuantumAmplitude] = Field(default_factory=dict)
    current_state: QuantumState = Field(default=QuantumState.SUPERPOSITION)
    entangled_tasks: set[UUID] = Field(default_factory=set)

    # Traditional properties
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    due_date: datetime | None = None
    estimated_duration: timedelta = Field(default=timedelta(hours=1))
    dependencies: set[UUID] = Field(default_factory=set)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __post_init__(self):
        """Initialize quantum amplitudes if not provided."""
        if not self.amplitudes:
            self._initialize_superposition()

    def _initialize_superposition(self) -> None:
        """Initialize task in quantum superposition across multiple states."""
        # Create superposition with higher probability for pending state
        states_probabilities = {
            QuantumState.PENDING: 0.7,
            QuantumState.EXECUTING: 0.15,
            QuantumState.COMPLETED: 0.1,
            QuantumState.BLOCKED: 0.05
        }

        for state, prob in states_probabilities.items():
            phase = random.uniform(0, 2 * np.pi)  # Random quantum phase
            self.amplitudes[state] = QuantumAmplitude(
                state=state,
                probability=prob,
                phase=phase
            )

    def measure(self) -> QuantumState:
        """
        Quantum measurement: collapse superposition to definite state.
        
        Returns:
            The measured quantum state based on probability amplitudes.
        """
        if not self.amplitudes:
            return QuantumState.PENDING

        # Calculate probabilities considering quantum interference
        total_prob = 0.0
        state_probs = {}

        for state, amplitude in self.amplitudes.items():
            # Apply quantum interference (simplified)
            interference_factor = np.cos(amplitude.phase)
            adjusted_prob = amplitude.probability * (1 + 0.1 * interference_factor)
            adjusted_prob = max(0, adjusted_prob)  # Ensure non-negative

            state_probs[state] = adjusted_prob
            total_prob += adjusted_prob

        # Normalize probabilities
        if total_prob > 0:
            for state in state_probs:
                state_probs[state] /= total_prob

        # Quantum measurement using weighted random selection
        rand_val = random.random()
        cumulative_prob = 0.0

        for state, prob in state_probs.items():
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                self.current_state = state
                self.updated_at = datetime.utcnow()
                logger.info(f"Task {self.id} collapsed to state: {state}")
                return state

        # Fallback
        self.current_state = QuantumState.PENDING
        return QuantumState.PENDING

    def evolve_state(self, time_delta: float = 1.0) -> None:
        """
        Quantum state evolution over time using Schrödinger-like equation.
        
        Args:
            time_delta: Time step for evolution
        """
        for state, amplitude in self.amplitudes.items():
            # Simple quantum evolution: rotate phase
            amplitude.phase += time_delta * 0.1 * hash(str(self.id)) % (2 * np.pi)
            amplitude.phase = amplitude.phase % (2 * np.pi)

    def entangle_with(self, other_task: 'QuantumTask') -> None:
        """
        Create quantum entanglement between tasks.
        
        Args:
            other_task: Task to entangle with
        """
        self.entangled_tasks.add(other_task.id)
        other_task.entangled_tasks.add(self.id)

        # Synchronize some quantum properties
        avg_phase = (
            np.mean([amp.phase for amp in self.amplitudes.values()]) +
            np.mean([amp.phase for amp in other_task.amplitudes.values()])
        ) / 2

        for amplitude in self.amplitudes.values():
            amplitude.phase = avg_phase + random.uniform(-0.1, 0.1)

        for amplitude in other_task.amplitudes.values():
            amplitude.phase = avg_phase + random.uniform(-0.1, 0.1)

        logger.info(f"Tasks {self.id} and {other_task.id} are now entangled")

    def calculate_execution_probability(self) -> float:
        """Calculate probability that task will execute successfully."""
        if QuantumState.PENDING in self.amplitudes:
            base_prob = self.amplitudes[QuantumState.PENDING].probability
        else:
            base_prob = 0.5

        # Adjust based on priority (higher energy = higher probability)
        priority_multiplier = {
            QuantumPriority.GROUND_STATE: 0.8,
            QuantumPriority.EXCITED_1: 0.9,
            QuantumPriority.EXCITED_2: 1.0,
            QuantumPriority.EXCITED_3: 1.1,
            QuantumPriority.IONIZED: 1.2
        }

        return min(1.0, base_prob * priority_multiplier.get(self.priority, 1.0))


class QuantumTaskPlanner:
    """
    Main quantum-inspired task planning system.
    
    Implements quantum algorithms for optimal task scheduling, resource allocation,
    and dependency management using principles from quantum computing.
    """

    def __init__(self, max_parallel_tasks: int = 5):
        self.tasks: dict[UUID, QuantumTask] = {}
        self.max_parallel_tasks = max_parallel_tasks
        self.quantum_clock = 0.0
        self.execution_history: list[dict[str, Any]] = []

    async def add_task(self, task: QuantumTask) -> UUID:
        """
        Add a new task to the quantum planning system.
        
        Args:
            task: QuantumTask to add
            
        Returns:
            Task UUID
        """
        self.tasks[task.id] = task
        logger.info(f"Added quantum task: {task.title} (ID: {task.id})")
        return task.id

    async def remove_task(self, task_id: UUID) -> bool:
        """Remove task from the system."""
        if task_id in self.tasks:
            # Clean up entanglements
            task = self.tasks[task_id]
            for entangled_id in task.entangled_tasks:
                if entangled_id in self.tasks:
                    self.tasks[entangled_id].entangled_tasks.discard(task_id)

            del self.tasks[task_id]
            logger.info(f"Removed quantum task: {task_id}")
            return True
        return False

    async def quantum_evolve(self, time_step: float = 1.0) -> None:
        """
        Evolve all quantum tasks according to quantum mechanics principles.
        
        Args:
            time_step: Time evolution step
        """
        self.quantum_clock += time_step

        for task in self.tasks.values():
            task.evolve_state(time_step)

        logger.debug(f"Quantum system evolved to time: {self.quantum_clock}")

    async def measure_system(self) -> dict[UUID, QuantumState]:
        """
        Perform quantum measurement on all tasks, collapsing superpositions.
        
        Returns:
            Dictionary mapping task IDs to measured states
        """
        measurements = {}

        for task_id, task in self.tasks.items():
            measured_state = task.measure()
            measurements[task_id] = measured_state

        logger.info(f"Quantum measurement completed for {len(measurements)} tasks")
        return measurements

    async def optimize_schedule(self) -> list[QuantumTask]:
        """
        Use quantum-inspired optimization to find optimal task execution order.
        
        This implements a simplified quantum annealing approach for scheduling.
        
        Returns:
            Optimized list of tasks ready for execution
        """
        # Get all tasks in pending or superposition state
        available_tasks = [
            task for task in self.tasks.values()
            if task.current_state in [QuantumState.PENDING, QuantumState.SUPERPOSITION]
            and self._dependencies_satisfied(task)
        ]

        if not available_tasks:
            return []

        # Quantum annealing simulation for optimization
        best_schedule = await self._quantum_annealing_schedule(available_tasks)

        logger.info(f"Optimized schedule contains {len(best_schedule)} tasks")
        return best_schedule

    async def _quantum_annealing_schedule(self, tasks: list[QuantumTask]) -> list[QuantumTask]:
        """
        Simplified quantum annealing for task scheduling optimization.
        
        Args:
            tasks: List of available tasks
            
        Returns:
            Optimized task order
        """
        if not tasks:
            return []

        # Initialize with random schedule
        current_schedule = tasks.copy()
        random.shuffle(current_schedule)
        current_energy = self._calculate_schedule_energy(current_schedule)

        # Quantum annealing parameters
        initial_temperature = 10.0
        final_temperature = 0.01
        cooling_rate = 0.95
        iterations = 100

        temperature = initial_temperature

        for _ in range(iterations):
            # Generate neighbor by swapping two random tasks
            neighbor_schedule = current_schedule.copy()
            if len(neighbor_schedule) > 1:
                i, j = random.sample(range(len(neighbor_schedule)), 2)
                neighbor_schedule[i], neighbor_schedule[j] = neighbor_schedule[j], neighbor_schedule[i]

            neighbor_energy = self._calculate_schedule_energy(neighbor_schedule)
            energy_diff = neighbor_energy - current_energy

            # Accept or reject based on quantum probability
            if energy_diff < 0 or random.random() < np.exp(-energy_diff / temperature):
                current_schedule = neighbor_schedule
                current_energy = neighbor_energy

            # Cool down
            temperature *= cooling_rate

            if temperature < final_temperature:
                break

        return current_schedule[:self.max_parallel_tasks]

    def _calculate_schedule_energy(self, schedule: list[QuantumTask]) -> float:
        """
        Calculate energy (cost) of a task schedule.
        Lower energy = better schedule.
        
        Args:
            schedule: Task execution order
            
        Returns:
            Energy value (lower is better)
        """
        if not schedule:
            return 0.0

        energy = 0.0

        # Energy components:
        # 1. Priority-based cost (higher priority = lower energy)
        priority_weights = {
            QuantumPriority.GROUND_STATE: 5.0,
            QuantumPriority.EXCITED_1: 4.0,
            QuantumPriority.EXCITED_2: 3.0,
            QuantumPriority.EXCITED_3: 2.0,
            QuantumPriority.IONIZED: 1.0
        }

        for task in schedule:
            energy += priority_weights.get(task.priority, 3.0)

        # 2. Due date penalty
        current_time = datetime.utcnow()
        for i, task in enumerate(schedule):
            if task.due_date:
                expected_start = current_time + timedelta(hours=i)
                if expected_start > task.due_date:
                    overdue_hours = (expected_start - task.due_date).total_seconds() / 3600
                    energy += overdue_hours * 2.0

        # 3. Dependency violations (should be handled by filtering)
        for task in schedule:
            unresolved_deps = len([
                dep_id for dep_id in task.dependencies
                if dep_id in self.tasks and
                self.tasks[dep_id].current_state != QuantumState.COMPLETED
            ])
            energy += unresolved_deps * 10.0  # Heavy penalty

        return energy

    def _dependencies_satisfied(self, task: QuantumTask) -> bool:
        """Check if all task dependencies are satisfied."""
        for dep_id in task.dependencies:
            if dep_id in self.tasks:
                dep_task = self.tasks[dep_id]
                if dep_task.current_state != QuantumState.COMPLETED:
                    return False
        return True

    async def get_system_coherence(self) -> float:
        """
        Calculate quantum coherence of the task system.
        
        Returns:
            Coherence value between 0 and 1 (1 = fully coherent)
        """
        if not self.tasks:
            return 1.0

        total_coherence = 0.0

        for task in self.tasks.values():
            # Calculate coherence based on superposition entropy
            if task.amplitudes:
                probabilities = [amp.probability for amp in task.amplitudes.values()]
                # Normalize
                total_prob = sum(probabilities)
                if total_prob > 0:
                    probabilities = [p / total_prob for p in probabilities]

                # Shannon entropy as measure of coherence
                entropy = -sum(p * np.log2(p + 1e-10) for p in probabilities if p > 0)
                max_entropy = np.log2(len(probabilities))

                if max_entropy > 0:
                    coherence = 1.0 - (entropy / max_entropy)
                else:
                    coherence = 1.0

                total_coherence += coherence

        return total_coherence / len(self.tasks)

    async def get_system_stats(self) -> dict[str, Any]:
        """Get comprehensive system statistics."""
        state_counts = {}
        priority_counts = {}

        for task in self.tasks.values():
            # Count states
            state_counts[task.current_state] = state_counts.get(task.current_state, 0) + 1

            # Count priorities
            priority_counts[task.priority] = priority_counts.get(task.priority, 0) + 1

        return {
            "total_tasks": len(self.tasks),
            "quantum_clock": self.quantum_clock,
            "state_distribution": state_counts,
            "priority_distribution": priority_counts,
            "system_coherence": await self.get_system_coherence(),
            "max_parallel_tasks": self.max_parallel_tasks
        }
