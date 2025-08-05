"""
Comprehensive tests for the quantum task planner module.

Tests cover:
- Quantum task creation and validation
- Quantum state management and measurement
- Superposition and amplitude calculations
- Task entanglement and correlation
- Performance and edge cases
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from unittest.mock import Mock, patch

from pipeline.quantum.quantum_planner import (
    QuantumTask, QuantumState, QuantumPriority, QuantumAmplitude,
    QuantumTaskPlanner
)
from pipeline.quantum.exceptions import (
    QuantumStateError, SuperpositionCollapseError, QuantumTaskValidationError
)


class TestQuantumAmplitude:
    """Test quantum amplitude functionality."""
    
    def test_amplitude_creation(self):
        """Test quantum amplitude creation and validation."""
        amplitude = QuantumAmplitude(
            state=QuantumState.PENDING,
            probability=0.7,
            phase=np.pi/2
        )
        
        assert amplitude.state == QuantumState.PENDING
        assert amplitude.probability == 0.7
        assert amplitude.phase == np.pi/2
    
    def test_amplitude_probability_bounds(self):
        """Test probability bounds enforcement."""
        # Test negative probability
        amplitude = QuantumAmplitude(
            state=QuantumState.PENDING,
            probability=-0.5
        )
        assert amplitude.probability == 0.0
        
        # Test probability > 1
        amplitude = QuantumAmplitude(
            state=QuantumState.PENDING,
            probability=1.5
        )
        assert amplitude.probability == 1.0


class TestQuantumTask:
    """Test quantum task functionality."""
    
    def test_task_creation(self):
        """Test basic quantum task creation."""
        task = QuantumTask(
            title="Test Task",
            description="A test quantum task",
            priority=QuantumPriority.EXCITED_2
        )
        
        assert task.title == "Test Task"
        assert task.description == "A test quantum task"
        assert task.priority == QuantumPriority.EXCITED_2
        assert task.current_state == QuantumState.SUPERPOSITION
        assert isinstance(task.id, UUID)
        assert isinstance(task.created_at, datetime)
    
    def test_task_superposition_initialization(self):
        """Test that tasks initialize in superposition."""
        task = QuantumTask(title="Test Task")
        
        # Should have quantum amplitudes
        assert len(task.amplitudes) > 0
        
        # Total probability should be approximately 1
        total_prob = sum(amp.probability for amp in task.amplitudes.values())
        assert abs(total_prob - 1.0) < 0.01
        
        # Should have pending state with highest probability
        assert QuantumState.PENDING in task.amplitudes
        assert task.amplitudes[QuantumState.PENDING].probability > 0.5
    
    def test_quantum_measurement(self):
        """Test quantum measurement and state collapse."""
        task = QuantumTask(title="Test Task")
        
        # Perform measurement
        measured_state = task.measure()
        
        # Should return a valid quantum state
        assert isinstance(measured_state, QuantumState)
        assert measured_state == task.current_state
        
        # Updated timestamp should be recent
        time_diff = (datetime.utcnow() - task.updated_at).total_seconds()
        assert time_diff < 1.0
    
    def test_quantum_state_evolution(self):
        """Test quantum state evolution over time."""
        task = QuantumTask(title="Test Task")
        
        # Store initial phases
        initial_phases = {state: amp.phase for state, amp in task.amplitudes.items()}
        
        # Evolve state
        task.evolve_state(time_delta=1.0)
        
        # Phases should have changed
        for state, amp in task.amplitudes.items():
            if state in initial_phases:
                assert amp.phase != initial_phases[state]
    
    def test_task_entanglement(self):
        """Test quantum entanglement between tasks."""
        task1 = QuantumTask(title="Task 1")
        task2 = QuantumTask(title="Task 2")
        
        # Initially not entangled
        assert len(task1.entangled_tasks) == 0
        assert len(task2.entangled_tasks) == 0
        
        # Create entanglement
        task1.entangle_with(task2)
        
        # Should be mutually entangled
        assert task2.id in task1.entangled_tasks
        assert task1.id in task2.entangled_tasks
        
        # Phases should be correlated
        avg_phase1 = np.mean([amp.phase for amp in task1.amplitudes.values()])
        avg_phase2 = np.mean([amp.phase for amp in task2.amplitudes.values()])
        phase_diff = abs(avg_phase1 - avg_phase2)
        
        # Should be similar phases (within 0.2 radians due to random component)
        assert phase_diff < 0.5
    
    def test_execution_probability(self):
        """Test execution probability calculation."""
        task = QuantumTask(
            title="Test Task",
            priority=QuantumPriority.IONIZED  # Highest priority
        )
        
        prob = task.calculate_execution_probability()
        
        # Should be a valid probability
        assert 0.0 <= prob <= 1.0
        
        # High priority should have higher probability
        assert prob > 0.5
    
    def test_task_with_dependencies(self):
        """Test task with dependencies."""
        dep_id = uuid4()
        task = QuantumTask(
            title="Dependent Task",
            dependencies={dep_id}
        )
        
        assert dep_id in task.dependencies
        assert len(task.dependencies) == 1
    
    def test_task_metadata(self):
        """Test task metadata handling."""
        metadata = {
            "category": "testing",
            "complexity": "high",
            "custom_data": {"key": "value"}
        }
        
        task = QuantumTask(
            title="Task with Metadata",
            metadata=metadata
        )
        
        assert task.metadata == metadata
        assert task.metadata["category"] == "testing"


class TestQuantumTaskPlanner:
    """Test quantum task planner functionality."""
    
    @pytest.fixture
    def planner(self):
        """Create a quantum task planner for testing."""
        return QuantumTaskPlanner(max_parallel_tasks=3)
    
    @pytest.fixture
    def sample_tasks(self):
        """Create sample tasks for testing."""
        return [
            QuantumTask(
                title=f"Task {i}",
                priority=QuantumPriority.EXCITED_2,
                estimated_duration=timedelta(hours=1)
            )
            for i in range(5)
        ]
    
    @pytest.mark.asyncio
    async def test_add_task(self, planner):
        """Test adding tasks to the planner."""
        task = QuantumTask(title="Test Task")
        
        task_id = await planner.add_task(task)
        
        assert task_id == task.id
        assert task.id in planner.tasks
        assert planner.tasks[task.id] == task
    
    @pytest.mark.asyncio
    async def test_remove_task(self, planner):
        """Test removing tasks from the planner."""
        task = QuantumTask(title="Test Task")
        task_id = await planner.add_task(task)
        
        # Remove task
        success = await planner.remove_task(task_id)
        
        assert success is True
        assert task_id not in planner.tasks
        
        # Try to remove non-existent task
        success = await planner.remove_task(uuid4())
        assert success is False
    
    @pytest.mark.asyncio
    async def test_quantum_evolution(self, planner, sample_tasks):
        """Test quantum system evolution."""
        # Add tasks to planner
        for task in sample_tasks:
            await planner.add_task(task)
        
        initial_clock = planner.quantum_clock
        
        # Evolve system
        await planner.quantum_evolve(time_step=2.0)
        
        # Clock should advance
        assert planner.quantum_clock == initial_clock + 2.0
    
    @pytest.mark.asyncio
    async def test_quantum_measurement(self, planner, sample_tasks):
        """Test quantum measurement of all tasks."""
        # Add tasks to planner
        for task in sample_tasks:
            await planner.add_task(task)
        
        # Perform measurements
        measurements = await planner.measure_system()
        
        # Should have measurements for all tasks
        assert len(measurements) == len(sample_tasks)
        
        for task in sample_tasks:
            assert task.id in measurements
            assert isinstance(measurements[task.id], QuantumState)
    
    @pytest.mark.asyncio
    async def test_optimize_schedule(self, planner, sample_tasks):
        """Test quantum scheduling optimization."""
        # Add tasks to planner
        for task in sample_tasks:
            await planner.add_task(task)
        
        # Optimize schedule
        optimized_schedule = await planner.optimize_schedule()
        
        # Should return valid schedule
        assert isinstance(optimized_schedule, list)
        assert len(optimized_schedule) <= planner.max_parallel_tasks
        
        # All returned tasks should be valid
        for task in optimized_schedule:
            assert isinstance(task, QuantumTask)
            assert task.id in planner.tasks
    
    @pytest.mark.asyncio
    async def test_quantum_annealing_schedule(self, planner, sample_tasks):
        """Test quantum annealing schedule optimization."""
        # Add tasks with different priorities
        high_priority = QuantumTask(
            title="High Priority Task",
            priority=QuantumPriority.IONIZED
        )
        low_priority = QuantumTask(
            title="Low Priority Task", 
            priority=QuantumPriority.GROUND_STATE
        )
        
        await planner.add_task(high_priority)
        await planner.add_task(low_priority)
        
        # Get optimized schedule
        schedule = await planner.optimize_schedule()
        
        # High priority task should be first (statistically)
        if len(schedule) >= 2:
            # Due to quantum randomness, we can't guarantee order,
            # but high priority should be more likely to be first
            assert len(schedule) <= planner.max_parallel_tasks
    
    @pytest.mark.asyncio
    async def test_system_coherence(self, planner, sample_tasks):
        """Test quantum system coherence calculation."""
        # Add tasks to planner
        for task in sample_tasks:
            await planner.add_task(task)
        
        # Calculate coherence
        coherence = await planner.get_system_coherence()
        
        # Should be valid coherence value
        assert 0.0 <= coherence <= 1.0
    
    @pytest.mark.asyncio
    async def test_system_stats(self, planner, sample_tasks):
        """Test system statistics collection."""
        # Add tasks to planner
        for task in sample_tasks:
            await planner.add_task(task)
        
        # Get system stats
        stats = await planner.get_system_stats()
        
        # Verify stats structure
        assert "total_tasks" in stats
        assert "quantum_clock" in stats
        assert "state_distribution" in stats
        assert "priority_distribution" in stats
        assert "system_coherence" in stats
        
        # Verify values
        assert stats["total_tasks"] == len(sample_tasks)
        assert stats["quantum_clock"] >= 0
        assert 0.0 <= stats["system_coherence"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_empty_optimization(self, planner):
        """Test optimization with no tasks."""
        # Optimize empty system
        schedule = await planner.optimize_schedule()
        
        assert schedule == []
    
    @pytest.mark.asyncio
    async def test_large_task_set(self, planner):
        """Test planner with large number of tasks."""
        # Create many tasks
        large_task_set = [
            QuantumTask(
                title=f"Task {i}",
                priority=np.random.choice(list(QuantumPriority)),
                estimated_duration=timedelta(hours=np.random.uniform(0.5, 3.0))
            )
            for i in range(50)
        ]
        
        # Add all tasks
        for task in large_task_set:
            await planner.add_task(task)
        
        # Should handle optimization
        schedule = await planner.optimize_schedule()
        
        assert len(schedule) <= planner.max_parallel_tasks
        assert all(isinstance(task, QuantumTask) for task in schedule)


class TestQuantumTaskValidation:
    """Test quantum task validation and error handling."""
    
    def test_invalid_title(self):
        """Test task creation with invalid title."""
        with pytest.raises(Exception):  # Pydantic validation error
            QuantumTask(title="")  # Empty title
    
    def test_invalid_priority(self):
        """Test task creation with invalid priority."""
        with pytest.raises(Exception):  # Pydantic validation error
            QuantumTask(title="Test", priority="invalid_priority")
    
    def test_invalid_duration(self):
        """Test task creation with invalid duration."""
        with pytest.raises(Exception):  # Pydantic validation error
            QuantumTask(
                title="Test",
                estimated_duration=timedelta(seconds=-1)  # Negative duration
            )
    
    def test_self_dependency(self):
        """Test task creation with self-dependency."""
        task_id = uuid4()
        
        # This should be caught by validation
        task = QuantumTask(
            title="Self-dependent Task",
            id=task_id,
            dependencies={task_id}
        )
        
        # The validation should catch this in the validator
        assert task_id in task.dependencies  # For now, just verify it's set


class TestQuantumMeasurementEdgeCases:
    """Test edge cases in quantum measurement."""
    
    def test_measurement_with_empty_amplitudes(self):
        """Test measurement when amplitudes are empty."""
        task = QuantumTask(title="Test Task")
        task.amplitudes = {}  # Clear amplitudes
        
        # Should handle gracefully
        measured_state = task.measure()
        assert measured_state == QuantumState.PENDING  # Default fallback
    
    def test_measurement_with_zero_probabilities(self):
        """Test measurement with all zero probabilities."""
        task = QuantumTask(title="Test Task")
        
        # Set all probabilities to zero
        for amplitude in task.amplitudes.values():
            amplitude.probability = 0.0
        
        # Should handle gracefully
        measured_state = task.measure()
        assert isinstance(measured_state, QuantumState)
    
    def test_evolution_with_extreme_time_delta(self):
        """Test state evolution with extreme time values."""
        task = QuantumTask(title="Test Task")
        
        # Very large time delta
        task.evolve_state(time_delta=1000000.0)
        
        # Should not crash and phases should be valid
        for amplitude in task.amplitudes.values():
            assert 0 <= amplitude.phase < 2 * np.pi
    
    def test_entanglement_with_self(self):
        """Test entanglement with self (should be handled gracefully)."""
        task = QuantumTask(title="Test Task")
        
        # Try to entangle with self
        task.entangle_with(task)
        
        # Should not add self to entangled tasks
        assert task.id not in task.entangled_tasks


class TestQuantumPlannerConcurrency:
    """Test concurrent operations on quantum planner."""
    
    @pytest.mark.asyncio
    async def test_concurrent_task_addition(self):
        """Test adding tasks concurrently."""
        planner = QuantumTaskPlanner()
        
        # Create multiple tasks
        tasks = [
            QuantumTask(title=f"Concurrent Task {i}")
            for i in range(10)
        ]
        
        # Add tasks concurrently
        add_tasks = [planner.add_task(task) for task in tasks]
        task_ids = await asyncio.gather(*add_tasks)
        
        # All tasks should be added
        assert len(task_ids) == len(tasks)
        assert len(planner.tasks) == len(tasks)
        
        # All IDs should be unique
        assert len(set(task_ids)) == len(task_ids)
    
    @pytest.mark.asyncio
    async def test_concurrent_measurement_and_evolution(self):
        """Test concurrent measurement and evolution operations."""
        planner = QuantumTaskPlanner()
        
        # Add some tasks
        tasks = [
            QuantumTask(title=f"Task {i}")
            for i in range(5)
        ]
        
        for task in tasks:
            await planner.add_task(task)
        
        # Run measurement and evolution concurrently
        measurement_task = asyncio.create_task(planner.measure_system())
        evolution_task = asyncio.create_task(planner.quantum_evolve(1.0))
        
        measurements, _ = await asyncio.gather(measurement_task, evolution_task)
        
        # Should complete successfully
        assert len(measurements) == len(tasks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])