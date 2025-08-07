"""
Quantum-Inspired Scheduler Implementation

Advanced scheduling algorithms using quantum computing principles:
- Superposition scheduling: Execute multiple scheduling strategies simultaneously
- Quantum parallelism: Leverage quantum-like parallelism for optimization
- Quantum tunneling: Overcome local optima in scheduling decisions
- Quantum interference: Optimize through constructive interference patterns
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

import numpy as np

from .quantum_planner import QuantumPriority, QuantumState, QuantumTask

logger = logging.getLogger(__name__)


@dataclass
class SchedulingStrategy:
    """Represents a quantum superposition of scheduling strategies."""
    name: str
    weight: float
    execute_func: Callable[[list[QuantumTask]], list[QuantumTask]]
    quantum_phase: float = 0.0


class SuperpositionScheduler:
    """
    Scheduler that maintains multiple scheduling strategies in superposition
    and collapses to the optimal strategy through measurement.
    """

    def __init__(self):
        self.strategies: list[SchedulingStrategy] = []
        self.measurement_history: list[dict[str, Any]] = []
        self._initialize_strategies()

    def _initialize_strategies(self) -> None:
        """Initialize the superposition of scheduling strategies."""
        self.strategies = [
            SchedulingStrategy(
                name="priority_first",
                weight=0.3,
                execute_func=self._priority_first_schedule,
                quantum_phase=0.0
            ),
            SchedulingStrategy(
                name="shortest_first",
                weight=0.25,
                execute_func=self._shortest_job_first_schedule,
                quantum_phase=np.pi/4
            ),
            SchedulingStrategy(
                name="deadline_aware",
                weight=0.25,
                execute_func=self._deadline_aware_schedule,
                quantum_phase=np.pi/2
            ),
            SchedulingStrategy(
                name="load_balanced",
                weight=0.2,
                execute_func=self._load_balanced_schedule,
                quantum_phase=3*np.pi/4
            )
        ]

    async def quantum_schedule(self, tasks: list[QuantumTask]) -> list[QuantumTask]:
        """
        Execute scheduling in quantum superposition and measure optimal result.
        
        Args:
            tasks: Tasks to schedule
            
        Returns:
            Optimally scheduled task list
        """
        if not tasks:
            return []

        # Execute all strategies in superposition
        strategy_results = {}
        strategy_energies = {}

        for strategy in self.strategies:
            try:
                scheduled_tasks = strategy.execute_func(tasks.copy())
                energy = self._calculate_schedule_energy(scheduled_tasks)

                strategy_results[strategy.name] = scheduled_tasks
                strategy_energies[strategy.name] = energy

                logger.debug(f"Strategy '{strategy.name}' energy: {energy}")

            except Exception as e:
                logger.warning(f"Strategy '{strategy.name}' failed: {e}")
                strategy_energies[strategy.name] = float('inf')

        # Quantum measurement - collapse to best strategy
        if not strategy_energies:
            return tasks

        # Apply quantum interference to modify strategy weights
        self._apply_quantum_interference()

        # Select strategy using quantum probability
        selected_strategy = self._quantum_measurement(strategy_energies)

        result = strategy_results.get(selected_strategy, tasks)

        # Record measurement
        self.measurement_history.append({
            "timestamp": datetime.utcnow(),
            "selected_strategy": selected_strategy,
            "strategy_energies": strategy_energies.copy(),
            "num_tasks": len(tasks)
        })

        logger.info(f"Quantum scheduler selected strategy: {selected_strategy}")
        return result

    def _apply_quantum_interference(self) -> None:
        """Apply quantum interference effects to strategy phases and weights."""
        for i, strategy in enumerate(self.strategies):
            # Phase evolution
            strategy.quantum_phase += 0.1 * np.sin(time.time() + i)
            strategy.quantum_phase = strategy.quantum_phase % (2 * np.pi)

            # Interference with other strategies
            interference = 0.0
            for j, other_strategy in enumerate(self.strategies):
                if i != j:
                    phase_diff = strategy.quantum_phase - other_strategy.quantum_phase
                    interference += 0.05 * np.cos(phase_diff) * other_strategy.weight

            # Adjust weight based on interference (bounded)
            strategy.weight = max(0.1, min(0.5, strategy.weight + interference))

        # Normalize weights
        total_weight = sum(s.weight for s in self.strategies)
        if total_weight > 0:
            for strategy in self.strategies:
                strategy.weight /= total_weight

    def _quantum_measurement(self, strategy_energies: dict[str, float]) -> str:
        """
        Perform quantum measurement to select optimal strategy.
        
        Args:
            strategy_energies: Energy values for each strategy
            
        Returns:
            Selected strategy name
        """
        # Convert energies to probabilities (lower energy = higher probability)
        min_energy = min(strategy_energies.values())
        max_energy = max(strategy_energies.values())

        if max_energy == min_energy:
            # All strategies have same energy, use quantum weights
            strategy_probs = {s.name: s.weight for s in self.strategies}
        else:
            strategy_probs = {}
            for strategy in self.strategies:
                if strategy.name in strategy_energies:
                    # Invert energy to probability (lower energy = higher prob)
                    energy = strategy_energies[strategy.name]
                    if energy == float('inf'):
                        prob = 0.0
                    else:
                        # Normalize energy to [0,1] and invert
                        normalized_energy = (energy - min_energy) / (max_energy - min_energy)
                        prob = (1.0 - normalized_energy) * strategy.weight

                    strategy_probs[strategy.name] = prob

        # Quantum measurement using weighted random selection
        total_prob = sum(strategy_probs.values())
        if total_prob <= 0:
            return self.strategies[0].name  # Fallback

        # Normalize probabilities
        for name in strategy_probs:
            strategy_probs[name] /= total_prob

        # Select strategy
        rand_val = np.random.random()
        cumulative_prob = 0.0

        for name, prob in strategy_probs.items():
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                return name

        return list(strategy_probs.keys())[0]  # Fallback

    def _priority_first_schedule(self, tasks: list[QuantumTask]) -> list[QuantumTask]:
        """Schedule tasks by priority (highest first)."""
        priority_order = {
            QuantumPriority.IONIZED: 5,
            QuantumPriority.EXCITED_3: 4,
            QuantumPriority.EXCITED_2: 3,
            QuantumPriority.EXCITED_1: 2,
            QuantumPriority.GROUND_STATE: 1
        }

        return sorted(tasks, key=lambda t: priority_order.get(t.priority, 0), reverse=True)

    def _shortest_job_first_schedule(self, tasks: list[QuantumTask]) -> list[QuantumTask]:
        """Schedule tasks by estimated duration (shortest first)."""
        return sorted(tasks, key=lambda t: t.estimated_duration.total_seconds())

    def _deadline_aware_schedule(self, tasks: list[QuantumTask]) -> list[QuantumTask]:
        """Schedule tasks by deadline urgency."""
        current_time = datetime.utcnow()

        def urgency_score(task: QuantumTask) -> float:
            if not task.due_date:
                return float('inf')  # No deadline = lowest urgency

            time_left = (task.due_date - current_time).total_seconds()
            duration = task.estimated_duration.total_seconds()

            if time_left <= 0:
                return -1  # Overdue = highest urgency

            return time_left / duration  # Ratio of time left to duration needed

        return sorted(tasks, key=urgency_score)

    def _load_balanced_schedule(self, tasks: list[QuantumTask]) -> list[QuantumTask]:
        """Schedule tasks for balanced resource utilization."""
        # Simple load balancing: alternate between different task types
        categorized = {}

        for task in tasks:
            category = task.priority  # Use priority as category proxy
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(task)

        # Interleave tasks from different categories
        result = []
        category_iterators = {cat: iter(task_list) for cat, task_list in categorized.items()}

        while category_iterators:
            for category in list(category_iterators.keys()):
                try:
                    task = next(category_iterators[category])
                    result.append(task)
                except StopIteration:
                    del category_iterators[category]

        return result

    def _calculate_schedule_energy(self, scheduled_tasks: list[QuantumTask]) -> float:
        """
        Calculate energy (cost) of a schedule.
        
        Args:
            scheduled_tasks: Tasks in execution order
            
        Returns:
            Energy value (lower is better)
        """
        if not scheduled_tasks:
            return 0.0

        energy = 0.0
        current_time = datetime.utcnow()

        for i, task in enumerate(scheduled_tasks):
            # Priority cost (higher priority should execute sooner)
            priority_weights = {
                QuantumPriority.IONIZED: 1.0,
                QuantumPriority.EXCITED_3: 2.0,
                QuantumPriority.EXCITED_2: 3.0,
                QuantumPriority.EXCITED_1: 4.0,
                QuantumPriority.GROUND_STATE: 5.0
            }

            priority_cost = priority_weights.get(task.priority, 3.0) * i
            energy += priority_cost

            # Deadline penalty
            if task.due_date:
                estimated_completion = current_time + timedelta(hours=i + 1)
                if estimated_completion > task.due_date:
                    overdue_hours = (estimated_completion - task.due_date).total_seconds() / 3600
                    energy += overdue_hours * 10.0  # Heavy penalty for being late

            # Duration impact
            duration_hours = task.estimated_duration.total_seconds() / 3600
            energy += duration_hours * 0.1  # Small penalty for longer tasks

        return energy


class QuantumScheduler:
    """
    Main quantum scheduler that orchestrates task execution using quantum principles.
    """

    def __init__(self, max_concurrent_tasks: int = 5):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.superposition_scheduler = SuperpositionScheduler()
        self.executing_tasks: dict[UUID, asyncio.Task] = {}
        self.execution_stats = {
            "total_scheduled": 0,
            "total_completed": 0,
            "total_failed": 0,
            "average_execution_time": 0.0
        }

    async def schedule_and_execute(self, tasks: list[QuantumTask]) -> dict[str, Any]:
        """
        Schedule and execute tasks using quantum algorithms.
        
        Args:
            tasks: Tasks to schedule and execute
            
        Returns:
            Execution results and statistics
        """
        if not tasks:
            return {"scheduled": 0, "completed": 0, "failed": 0}

        logger.info(f"Quantum scheduling {len(tasks)} tasks")

        # Filter executable tasks
        executable_tasks = [
            task for task in tasks
            if task.current_state in [QuantumState.PENDING, QuantumState.SUPERPOSITION]
        ]

        if not executable_tasks:
            logger.info("No executable tasks found")
            return {"scheduled": 0, "completed": 0, "failed": 0}

        # Quantum scheduling optimization
        optimal_schedule = await self.superposition_scheduler.quantum_schedule(executable_tasks)

        # Execute tasks respecting concurrency limits
        results = await self._execute_scheduled_tasks(optimal_schedule)

        # Update statistics
        self.execution_stats["total_scheduled"] += len(optimal_schedule)
        self.execution_stats["total_completed"] += results["completed"]
        self.execution_stats["total_failed"] += results["failed"]

        logger.info(f"Quantum execution completed: {results}")
        return results

    async def _execute_scheduled_tasks(self, scheduled_tasks: list[QuantumTask]) -> dict[str, Any]:
        """
        Execute scheduled tasks with concurrency control.
        
        Args:
            scheduled_tasks: Tasks in optimal execution order
            
        Returns:
            Execution results
        """
        completed = 0
        failed = 0
        start_time = time.time()

        # Execute tasks in batches respecting concurrency limit
        for i in range(0, len(scheduled_tasks), self.max_concurrent_tasks):
            batch = scheduled_tasks[i:i + self.max_concurrent_tasks]

            # Start batch execution
            batch_tasks = []
            for task in batch:
                execution_task = asyncio.create_task(self._execute_single_task(task))
                batch_tasks.append(execution_task)
                self.executing_tasks[task.id] = execution_task

            # Wait for batch completion
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Process results
            for task, result in zip(batch, batch_results, strict=False):
                if task.id in self.executing_tasks:
                    del self.executing_tasks[task.id]

                if isinstance(result, Exception):
                    task.current_state = QuantumState.FAILED
                    failed += 1
                    logger.error(f"Task {task.id} failed: {result}")
                else:
                    task.current_state = QuantumState.COMPLETED
                    completed += 1
                    logger.info(f"Task {task.id} completed successfully")

        execution_time = time.time() - start_time

        return {
            "scheduled": len(scheduled_tasks),
            "completed": completed,
            "failed": failed,
            "execution_time": execution_time
        }

    async def _execute_single_task(self, task: QuantumTask) -> Any:
        """
        Execute a single quantum task.
        
        Args:
            task: Task to execute
            
        Returns:
            Execution result
        """
        logger.info(f"Executing task: {task.title} (ID: {task.id})")

        # Quantum measurement to collapse task state
        measured_state = task.measure()

        if measured_state == QuantumState.BLOCKED:
            raise RuntimeError(f"Task {task.id} is blocked by dependencies")

        # Mark as executing
        task.current_state = QuantumState.EXECUTING
        task.updated_at = datetime.utcnow()

        # Simulate task execution
        execution_time = task.estimated_duration.total_seconds()

        # Add some quantum uncertainty to execution time
        uncertainty_factor = np.random.uniform(0.8, 1.2)
        actual_execution_time = execution_time * uncertainty_factor

        await asyncio.sleep(min(actual_execution_time, 5.0))  # Cap simulation time

        # Random chance of quantum tunneling (overcoming obstacles)
        if np.random.random() < 0.1:  # 10% chance
            logger.info(f"Quantum tunneling activated for task {task.id}")
            # Task completes despite potential obstacles

        # Simulate execution probability
        success_probability = task.calculate_execution_probability()

        if np.random.random() < success_probability:
            return {"status": "success", "execution_time": actual_execution_time}
        raise RuntimeError(f"Task execution failed (probability: {success_probability})")

    async def get_execution_status(self) -> dict[str, Any]:
        """Get current execution status."""
        currently_executing = len(self.executing_tasks)

        return {
            "currently_executing": currently_executing,
            "max_concurrent": self.max_concurrent_tasks,
            "execution_stats": self.execution_stats.copy(),
            "strategy_history": len(self.superposition_scheduler.measurement_history)
        }

    async def cancel_task(self, task_id: UUID) -> bool:
        """Cancel a currently executing task."""
        if task_id in self.executing_tasks:
            execution_task = self.executing_tasks[task_id]
            execution_task.cancel()
            del self.executing_tasks[task_id]
            logger.info(f"Cancelled task execution: {task_id}")
            return True
        return False

    async def quantum_tunnel_task(self, task_id: UUID) -> bool:
        """
        Apply quantum tunneling to help a blocked task overcome obstacles.
        
        Args:
            task_id: ID of task to tunnel
            
        Returns:
            True if tunneling was applied
        """
        # This would implement quantum tunneling logic
        # For now, it's a placeholder for the concept
        logger.info(f"Applying quantum tunneling to task: {task_id}")
        return True
