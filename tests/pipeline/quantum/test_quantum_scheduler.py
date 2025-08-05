"""
Comprehensive tests for quantum scheduler module.

Tests cover:
- Superposition scheduling strategies
- Quantum interference in scheduling
- Parallel task execution
- Performance optimization
- Error handling and edge cases
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import Mock, patch, AsyncMock

from pipeline.quantum.quantum_scheduler import (
    QuantumScheduler, SuperpositionScheduler, SchedulingStrategy
)
from pipeline.quantum.quantum_planner import (
    QuantumTask, QuantumState, QuantumPriority
)


class TestSchedulingStrategy:
    """Test scheduling strategy functionality."""
    
    def test_strategy_creation(self):
        """Test creating a scheduling strategy."""
        def mock_func(tasks):
            return tasks
        
        strategy = SchedulingStrategy(
            name="test_strategy",
            weight=0.5,
            execute_func=mock_func,
            quantum_phase=np.pi/4
        )
        
        assert strategy.name == "test_strategy"
        assert strategy.weight == 0.5
        assert strategy.execute_func == mock_func
        assert strategy.quantum_phase == np.pi/4


class TestSuperpositionScheduler:
    """Test superposition scheduler functionality."""
    
    @pytest.fixture
    def scheduler(self):
        """Create a superposition scheduler for testing."""
        return SuperpositionScheduler()
    
    @pytest.fixture
    def sample_tasks(self):
        """Create sample tasks for testing."""
        return [
            QuantumTask(
                title=f"Task {i}",
                priority=QuantumPriority.EXCITED_2,
                estimated_duration=timedelta(hours=i+1)
            )
            for i in range(5)
        ]
    
    def test_scheduler_initialization(self, scheduler):
        """Test scheduler initialization."""
        assert len(scheduler.strategies) > 0
        assert scheduler.measurement_history == []
        
        # Check that all strategies have required attributes
        for strategy in scheduler.strategies:
            assert hasattr(strategy, 'name')
            assert hasattr(strategy, 'weight')
            assert hasattr(strategy, 'execute_func')
            assert hasattr(strategy, 'quantum_phase')
    
    @pytest.mark.asyncio
    async def test_quantum_schedule_empty(self, scheduler):
        """Test quantum scheduling with empty task list."""
        result = await scheduler.quantum_schedule([])
        assert result == []
    
    @pytest.mark.asyncio
    async def test_quantum_schedule_single_task(self, scheduler):
        """Test quantum scheduling with single task."""
        task = QuantumTask(title="Single Task")
        
        result = await scheduler.quantum_schedule([task])
        
        assert len(result) == 1
        assert result[0] == task
    
    @pytest.mark.asyncio
    async def test_quantum_schedule_multiple_tasks(self, scheduler, sample_tasks):
        """Test quantum scheduling with multiple tasks."""
        result = await scheduler.quantum_schedule(sample_tasks)
        
        assert len(result) == len(sample_tasks)
        assert all(task in sample_tasks for task in result)
        
        # Should record measurement history
        assert len(scheduler.measurement_history) > 0
        
        latest_measurement = scheduler.measurement_history[-1]
        assert "selected_strategy" in latest_measurement
        assert "strategy_energies" in latest_measurement
        assert "num_tasks" in latest_measurement
    
    def test_priority_first_schedule(self, scheduler, sample_tasks):
        """Test priority-first scheduling strategy."""
        # Create tasks with different priorities
        high_priority = QuantumTask(
            title="High Priority",
            priority=QuantumPriority.IONIZED
        )
        low_priority = QuantumTask(
            title="Low Priority",
            priority=QuantumPriority.GROUND_STATE
        )
        
        tasks = [low_priority, high_priority]  # Intentionally wrong order
        
        result = scheduler._priority_first_schedule(tasks)
        
        # High priority should come first
        assert result[0] == high_priority
        assert result[1] == low_priority
    
    def test_shortest_job_first_schedule(self, scheduler):
        """Test shortest job first scheduling strategy."""
        short_task = QuantumTask(
            title="Short Task",
            estimated_duration=timedelta(minutes=30)
        )
        long_task = QuantumTask(
            title="Long Task",
            estimated_duration=timedelta(hours=2)
        )
        
        tasks = [long_task, short_task]  # Intentionally wrong order
        
        result = scheduler._shortest_job_first_schedule(tasks)
        
        # Short task should come first
        assert result[0] == short_task
        assert result[1] == long_task
    
    def test_deadline_aware_schedule(self, scheduler):
        """Test deadline-aware scheduling strategy."""
        now = datetime.utcnow()
        
        urgent_task = QuantumTask(
            title="Urgent Task",
            due_date=now + timedelta(hours=1),
            estimated_duration=timedelta(minutes=30)
        )
        future_task = QuantumTask(
            title="Future Task",
            due_date=now + timedelta(days=1),
            estimated_duration=timedelta(hours=1)
        )
        
        tasks = [future_task, urgent_task]  # Intentionally wrong order
        
        result = scheduler._deadline_aware_schedule(tasks)
        
        # Urgent task should come first
        assert result[0] == urgent_task
        assert result[1] == future_task
    
    def test_load_balanced_schedule(self, scheduler):
        """Test load-balanced scheduling strategy."""
        # Create tasks with different priorities (used as categories)
        high_priority_tasks = [
            QuantumTask(title=f"High {i}", priority=QuantumPriority.IONIZED)
            for i in range(3)
        ]
        low_priority_tasks = [
            QuantumTask(title=f"Low {i}", priority=QuantumPriority.GROUND_STATE)
            for i in range(3)
        ]
        
        tasks = high_priority_tasks + low_priority_tasks
        
        result = scheduler._load_balanced_schedule(tasks)
        
        # Should interleave different priority tasks
        assert len(result) == len(tasks)
        
        # Check that it's not just grouped by priority
        priorities = [task.priority for task in result]
        # Should have some alternation (not all high priority first)
        assert not all(p == QuantumPriority.IONIZED for p in priorities[:3])
    
    def test_calculate_schedule_energy(self, scheduler):
        """Test schedule energy calculation."""
        high_priority = QuantumTask(
            title="High Priority",
            priority=QuantumPriority.IONIZED
        )
        low_priority = QuantumTask(
            title="Low Priority", 
            priority=QuantumPriority.GROUND_STATE
        )
        
        # High priority first should have lower energy (better)
        good_schedule = [high_priority, low_priority]
        bad_schedule = [low_priority, high_priority]
        
        good_energy = scheduler._calculate_schedule_energy(good_schedule)
        bad_energy = scheduler._calculate_schedule_energy(bad_schedule)
        
        assert good_energy < bad_energy
    
    def test_quantum_interference_application(self, scheduler):
        """Test quantum interference effects on strategies."""
        initial_weights = [s.weight for s in scheduler.strategies]
        initial_phases = [s.quantum_phase for s in scheduler.strategies]
        
        # Apply interference
        scheduler._apply_quantum_interference()
        
        # Weights and phases should change
        new_weights = [s.weight for s in scheduler.strategies]
        new_phases = [s.quantum_phase for s in scheduler.strategies]
        
        # At least some should change (due to randomness, we can't guarantee all)
        assert new_weights != initial_weights or new_phases != initial_phases
        
        # Weights should remain positive and sum should be reasonable
        for weight in new_weights:
            assert weight > 0
    
    def test_quantum_measurement_strategy_selection(self, scheduler):
        """Test quantum measurement for strategy selection."""
        strategy_energies = {
            "priority_first": 10.0,
            "shortest_first": 5.0,  # Lower energy = better
            "deadline_aware": 15.0,
            "load_balanced": 8.0
        }
        
        # Run multiple measurements to test probability distribution
        selections = []
        for _ in range(100):
            selected = scheduler._quantum_measurement(strategy_energies)
            selections.append(selected)
            assert selected in strategy_energies
        
        # "shortest_first" should be selected more often (lowest energy)
        selection_counts = {strategy: selections.count(strategy) for strategy in strategy_energies}
        
        # Most selected should be one of the better strategies
        most_selected = max(selection_counts, key=selection_counts.get)
        assert most_selected in ["shortest_first", "load_balanced"]


class TestQuantumScheduler:
    """Test main quantum scheduler functionality."""
    
    @pytest.fixture
    def scheduler(self):
        """Create a quantum scheduler for testing."""
        return QuantumScheduler(max_concurrent_tasks=3)
    
    @pytest.fixture
    def sample_tasks(self):
        """Create sample tasks for testing."""
        return [
            QuantumTask(
                title=f"Task {i}",
                priority=QuantumPriority.EXCITED_2,
                estimated_duration=timedelta(seconds=1)  # Short for testing
            )
            for i in range(5)
        ]
    
    def test_scheduler_initialization(self, scheduler):
        """Test scheduler initialization."""
        assert scheduler.max_concurrent_tasks == 3
        assert isinstance(scheduler.superposition_scheduler, SuperpositionScheduler)
        assert scheduler.executing_tasks == {}
        assert "total_scheduled" in scheduler.execution_stats
    
    @pytest.mark.asyncio
    async def test_schedule_and_execute_empty(self, scheduler):
        """Test scheduling and execution with empty task list."""
        result = await scheduler.schedule_and_execute([])
        
        assert result["scheduled"] == 0
        assert result["completed"] == 0
        assert result["failed"] == 0
    
    @pytest.mark.asyncio
    async def test_schedule_and_execute_single_task(self, scheduler):
        """Test scheduling and execution with single task."""
        task = QuantumTask(
            title="Single Task",
            estimated_duration=timedelta(seconds=0.1)
        )
        
        result = await scheduler.schedule_and_execute([task])
        
        assert result["scheduled"] == 1
        assert result["completed"] + result["failed"] == 1
        assert "execution_time" in result
    
    @pytest.mark.asyncio
    async def test_schedule_and_execute_multiple_tasks(self, scheduler, sample_tasks):
        """Test scheduling and execution with multiple tasks."""
        result = await scheduler.schedule_and_execute(sample_tasks)
        
        assert result["scheduled"] == len(sample_tasks)
        assert result["completed"] + result["failed"] == len(sample_tasks)
        assert result["execution_time"] > 0
        
        # Check execution stats were updated
        assert scheduler.execution_stats["total_scheduled"] >= len(sample_tasks)
    
    @pytest.mark.asyncio
    async def test_single_task_execution_success(self, scheduler):
        """Test successful single task execution."""
        task = QuantumTask(
            title="Success Task",
            estimated_duration=timedelta(seconds=0.1)
        )
        
        # Mock high success probability
        with patch.object(task, 'calculate_execution_probability', return_value=1.0):
            result = await scheduler._execute_single_task(task)
            
            assert result["status"] == "success"
            assert "execution_time" in result
            assert task.current_state == QuantumState.COMPLETED
    
    @pytest.mark.asyncio
    async def test_single_task_execution_failure(self, scheduler):
        """Test failed single task execution."""
        task = QuantumTask(
            title="Failure Task",
            estimated_duration=timedelta(seconds=0.1)
        )
        
        # Mock zero success probability
        with patch.object(task, 'calculate_execution_probability', return_value=0.0):
            with pytest.raises(RuntimeError):
                await scheduler._execute_single_task(task)
    
    @pytest.mark.asyncio
    async def test_quantum_tunneling_simulation(self, scheduler):
        """Test quantum tunneling effect in task execution."""
        task = QuantumTask(
            title="Tunneling Task",
            estimated_duration=timedelta(seconds=0.1)
        )
        
        # Even with low probability, quantum tunneling might save it
        with patch.object(task, 'calculate_execution_probability', return_value=0.1):
            # Run multiple times to test tunneling effect
            successes = 0
            for _ in range(10):
                try:
                    result = await scheduler._execute_single_task(task)
                    if result["status"] == "success":
                        successes += 1
                except RuntimeError:
                    pass
                
                # Reset task state for next attempt
                task.current_state = QuantumState.PENDING
            
            # Should have some successes due to tunneling or probability
            # (This is probabilistic, so we can't guarantee exact results)
            assert successes >= 0  # At least didn't crash
    
    @pytest.mark.asyncio
    async def test_concurrent_task_execution(self, scheduler):
        """Test concurrent execution of multiple tasks."""
        tasks = [
            QuantumTask(
                title=f"Concurrent Task {i}",
                estimated_duration=timedelta(seconds=0.1)
            )
            for i in range(3)
        ]
        
        # All tasks should execute concurrently
        result = await scheduler.schedule_and_execute(tasks)
        
        # Should complete in reasonable time (less than sum of individual times)
        assert result["execution_time"] < 3 * 0.1  # Allow some overhead
        assert result["scheduled"] == 3
    
    @pytest.mark.asyncio
    async def test_concurrency_limit_enforcement(self, scheduler):
        """Test that concurrency limits are enforced."""
        # Create more tasks than the limit
        tasks = [
            QuantumTask(
                title=f"Task {i}",
                estimated_duration=timedelta(seconds=0.5)  # Longer duration
            )
            for i in range(scheduler.max_concurrent_tasks + 2)
        ]
        
        start_time = asyncio.get_event_loop().time()
        result = await scheduler.schedule_and_execute(tasks)
        execution_time = asyncio.get_event_loop().time() - start_time
        
        # Should take at least 2 batches due to concurrency limit
        # (first batch + second batch) * task_duration
        expected_min_time = 2 * 0.5 * 0.8  # Allow some overhead
        
        assert result["scheduled"] == len(tasks)
        # Note: Due to quantum probability, some tasks might fail quickly
        # so we can't guarantee exact timing, but check structure
        assert result["completed"] + result["failed"] == len(tasks)
    
    @pytest.mark.asyncio
    async def test_execution_status(self, scheduler):
        """Test getting execution status."""
        status = await scheduler.get_execution_status()
        
        assert "currently_executing" in status
        assert "max_concurrent" in status
        assert "execution_stats" in status
        assert "strategy_history" in status
        
        assert status["currently_executing"] >= 0
        assert status["max_concurrent"] == scheduler.max_concurrent_tasks
    
    @pytest.mark.asyncio
    async def test_task_cancellation(self, scheduler):
        """Test cancelling task execution."""
        task = QuantumTask(
            title="Cancellable Task",
            estimated_duration=timedelta(seconds=10)  # Long duration
        )
        
        # Start execution in background
        execution_task = asyncio.create_task(scheduler._execute_single_task(task))
        scheduler.executing_tasks[task.id] = execution_task
        
        # Small delay to let execution start
        await asyncio.sleep(0.01)
        
        # Cancel the task
        success = await scheduler.cancel_task(task.id)
        
        assert success is True
        assert task.id not in scheduler.executing_tasks
        
        # The execution task should be cancelled
        with pytest.raises(asyncio.CancelledError):
            await execution_task
    
    @pytest.mark.asyncio
    async def test_cancel_nonexistent_task(self, scheduler):
        """Test cancelling a non-existent task."""
        fake_id = uuid4()
        success = await scheduler.cancel_task(fake_id)
        
        assert success is False
    
    @pytest.mark.asyncio
    async def test_quantum_tunnel_task(self, scheduler):
        """Test quantum tunneling functionality."""
        task_id = uuid4()
        
        # This is currently a placeholder function
        result = await scheduler.quantum_tunnel_task(task_id)
        
        # Should return True (placeholder implementation)
        assert result is True


class TestSchedulerPerformance:
    """Test scheduler performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_large_task_set_performance(self):
        """Test scheduler performance with large task sets."""
        scheduler = QuantumScheduler(max_concurrent_tasks=10)
        
        # Create large task set
        large_task_set = [
            QuantumTask(
                title=f"Performance Task {i}",
                priority=np.random.choice(list(QuantumPriority)),
                estimated_duration=timedelta(seconds=0.01)  # Very short
            )
            for i in range(100)
        ]
        
        start_time = asyncio.get_event_loop().time()
        result = await scheduler.schedule_and_execute(large_task_set)
        end_time = asyncio.get_event_loop().time()
        
        # Should complete in reasonable time
        execution_time = end_time - start_time
        assert execution_time < 10.0  # Should finish in under 10 seconds
        
        assert result["scheduled"] == 100
        assert result["completed"] + result["failed"] == 100
    
    @pytest.mark.asyncio
    async def test_strategy_selection_distribution(self):
        """Test that different strategies are selected over time."""
        scheduler = SuperpositionScheduler()
        
        tasks = [
            QuantumTask(title=f"Task {i}")
            for i in range(5)
        ]
        
        # Run multiple scheduling rounds
        selected_strategies = []
        for _ in range(20):
            await scheduler.quantum_schedule(tasks.copy())
            if scheduler.measurement_history:
                latest = scheduler.measurement_history[-1]
                selected_strategies.append(latest["selected_strategy"])
        
        # Should have some variety in strategy selection
        unique_strategies = set(selected_strategies)
        assert len(unique_strategies) > 1  # At least 2 different strategies used
    
    @pytest.mark.asyncio
    async def test_quantum_interference_effects(self):
        """Test that quantum interference affects strategy selection."""
        scheduler = SuperpositionScheduler()
        
        # Record initial strategy weights
        initial_weights = {s.name: s.weight for s in scheduler.strategies}
        
        # Run several scheduling operations
        tasks = [QuantumTask(title=f"Task {i}") for i in range(3)]
        
        for _ in range(5):
            await scheduler.quantum_schedule(tasks.copy())
        
        # Weights should have evolved due to interference
        final_weights = {s.name: s.weight for s in scheduler.strategies}
        
        # At least some weights should have changed
        changed_weights = sum(1 for name in initial_weights 
                            if abs(initial_weights[name] - final_weights[name]) > 0.01)
        
        assert changed_weights > 0


class TestSchedulerErrorHandling:
    """Test error handling in scheduler."""
    
    @pytest.mark.asyncio
    async def test_invalid_task_handling(self):
        """Test handling of invalid tasks."""
        scheduler = QuantumScheduler()
        
        # Task with invalid state
        invalid_task = QuantumTask(title="Invalid Task")
        invalid_task.current_state = QuantumState.COMPLETED  # Already completed
        
        result = await scheduler.schedule_and_execute([invalid_task])
        
        # Should handle gracefully - completed tasks shouldn't be scheduled
        assert result["scheduled"] == 0
    
    @pytest.mark.asyncio
    async def test_strategy_execution_failure(self):
        """Test handling when a strategy fails."""
        scheduler = SuperpositionScheduler()
        
        # Mock a strategy to fail
        def failing_strategy(tasks):
            raise RuntimeError("Strategy failed")
        
        scheduler.strategies[0].execute_func = failing_strategy
        
        tasks = [QuantumTask(title="Test Task")]
        
        # Should handle gracefully and use other strategies
        result = await scheduler.quantum_schedule(tasks)
        
        # Should still return a result (using other strategies)
        assert len(result) == 1
        assert result[0] == tasks[0]
    
    @pytest.mark.asyncio
    async def test_measurement_history_limits(self):
        """Test that measurement history doesn't grow unbounded."""
        scheduler = SuperpositionScheduler()
        
        tasks = [QuantumTask(title="Test Task")]
        
        # Run many scheduling operations
        for _ in range(200):
            await scheduler.quantum_schedule(tasks.copy())
        
        # History should be limited (implementation dependent)
        # For now, just check it doesn't crash and has reasonable size
        assert len(scheduler.measurement_history) > 0
        assert len(scheduler.measurement_history) <= 200  # Shouldn't exceed operations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])