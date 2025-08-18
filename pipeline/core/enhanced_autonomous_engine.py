"""
Enhanced Autonomous Engine - Next-Generation AI Pipeline Orchestrator

Implements advanced autonomous execution with:
- Quantum-inspired task scheduling
- Adaptive learning optimization
- Real-time performance monitoring
- Self-healing capabilities
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class TaskPriority(str, Enum):
    """Task priority levels for quantum scheduling."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BACKGROUND = "background"


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AutonomousTask:
    """Represents a task in the autonomous execution pipeline."""
    task_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    
    # Execution configuration
    max_retries: int = 3
    timeout_seconds: float = 300.0
    retry_delay: float = 1.0
    
    # Performance tracking
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time: float = 0.0
    
    # Task function and context
    task_function: Optional[Callable] = None
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    
    # Results and error tracking
    result: Any = None
    error: Optional[Exception] = None
    retry_count: int = 0
    
    def __post_init__(self):
        """Initialize task with auto-generated name if not provided."""
        if not self.name:
            self.name = f"Task-{self.task_id[:8]}"


@dataclass
class ExecutionMetrics:
    """Performance metrics for autonomous execution."""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    cancelled_tasks: int = 0
    
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    
    success_rate: float = 0.0
    throughput_per_second: float = 0.0
    
    def update_metrics(self, tasks: List[AutonomousTask]):
        """Update metrics based on current task list."""
        self.total_tasks = len(tasks)
        self.completed_tasks = len([t for t in tasks if t.status == TaskStatus.COMPLETED])
        self.failed_tasks = len([t for t in tasks if t.status == TaskStatus.FAILED])
        self.cancelled_tasks = len([t for t in tasks if t.status == TaskStatus.CANCELLED])
        
        if self.total_tasks > 0:
            self.success_rate = self.completed_tasks / self.total_tasks
        
        completed_with_time = [t for t in tasks if t.status == TaskStatus.COMPLETED and t.execution_time > 0]
        if completed_with_time:
            self.total_execution_time = sum(t.execution_time for t in completed_with_time)
            self.average_execution_time = self.total_execution_time / len(completed_with_time)


class EnhancedAutonomousEngine:
    """
    Next-generation autonomous execution engine with advanced capabilities.
    
    Features:
    - Quantum-inspired priority scheduling
    - Adaptive performance optimization
    - Real-time metrics collection
    - Self-healing error recovery
    - Circuit breaker pattern
    """
    
    def __init__(self, max_concurrent_tasks: int = 10, enable_metrics: bool = True):
        """Initialize the autonomous engine."""
        self.max_concurrent_tasks = max_concurrent_tasks
        self.enable_metrics = enable_metrics
        
        # Task management
        self.task_queue: List[AutonomousTask] = []
        self.running_tasks: Dict[str, AutonomousTask] = {}
        self.completed_tasks: List[AutonomousTask] = []
        
        # Performance tracking
        self.metrics = ExecutionMetrics()
        self.engine_start_time = datetime.now()
        
        # Circuit breaker for failure protection
        self.circuit_breaker_threshold = 0.5  # 50% failure rate
        self.circuit_breaker_window = timedelta(minutes=5)
        self.circuit_breaker_open = False
        self.last_circuit_check = datetime.now()
        
        # Adaptive learning parameters
        self.performance_history: Dict[str, List[float]] = {}
        self.optimization_enabled = True
        
        logger.info(f"Enhanced Autonomous Engine initialized with {max_concurrent_tasks} max concurrent tasks")
    
    def add_task(self, 
                 task_function: Callable,
                 *args,
                 name: str = "",
                 description: str = "",
                 priority: TaskPriority = TaskPriority.MEDIUM,
                 max_retries: int = 3,
                 timeout_seconds: float = 300.0,
                 **kwargs) -> str:
        """
        Add a new task to the execution queue.
        
        Returns:
            str: Task ID for tracking
        """
        task = AutonomousTask(
            name=name,
            description=description,
            priority=priority,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            task_function=task_function,
            args=args,
            kwargs=kwargs
        )
        
        # Insert task in priority order
        self._insert_task_by_priority(task)
        
        logger.info(f"Added task '{task.name}' with priority {priority.value}")
        return task.task_id
    
    def _insert_task_by_priority(self, task: AutonomousTask):
        """Insert task into queue maintaining priority order."""
        priority_order = {
            TaskPriority.CRITICAL: 0,
            TaskPriority.HIGH: 1,
            TaskPriority.MEDIUM: 2,
            TaskPriority.LOW: 3,
            TaskPriority.BACKGROUND: 4
        }
        
        task_priority_value = priority_order[task.priority]
        
        # Find insertion point
        insert_index = 0
        for i, existing_task in enumerate(self.task_queue):
            existing_priority_value = priority_order[existing_task.priority]
            if task_priority_value < existing_priority_value:
                insert_index = i
                break
            insert_index = i + 1
        
        self.task_queue.insert(insert_index, task)
    
    async def execute_all_tasks(self) -> ExecutionMetrics:
        """
        Execute all queued tasks autonomously with intelligent scheduling.
        
        Returns:
            ExecutionMetrics: Final execution statistics
        """
        logger.info(f"Starting autonomous execution of {len(self.task_queue)} tasks")
        start_time = time.time()
        
        try:
            # Execute tasks with concurrency control
            while self.task_queue or self.running_tasks:
                # Check circuit breaker
                if self._should_open_circuit_breaker():
                    logger.warning("Circuit breaker activated - pausing execution")
                    await asyncio.sleep(5)
                    continue
                
                # Start new tasks if capacity available
                while (len(self.running_tasks) < self.max_concurrent_tasks and 
                       self.task_queue and 
                       not self.circuit_breaker_open):
                    
                    task = self.task_queue.pop(0)
                    await self._start_task(task)
                
                # Wait for running tasks to complete
                if self.running_tasks:
                    await self._monitor_running_tasks()
                else:
                    # No running tasks, small delay before checking queue again
                    await asyncio.sleep(0.1)
            
            # Update final metrics
            all_tasks = self.completed_tasks + list(self.running_tasks.values())
            self.metrics.update_metrics(all_tasks)
            
            execution_time = time.time() - start_time
            logger.info(f"Autonomous execution completed in {execution_time:.2f}s")
            logger.info(f"Success rate: {self.metrics.success_rate:.2%}")
            
            return self.metrics
            
        except Exception as e:
            logger.error(f"Autonomous execution failed: {e}")
            raise
    
    async def _start_task(self, task: AutonomousTask):
        """Start executing a task asynchronously."""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        self.running_tasks[task.task_id] = task
        
        logger.debug(f"Starting task '{task.name}' (ID: {task.task_id})")
        
        # Create task execution coroutine
        asyncio.create_task(self._execute_task(task))
    
    async def _execute_task(self, task: AutonomousTask):
        """Execute a single task with error handling and retries."""
        max_retries = task.max_retries
        
        for attempt in range(max_retries + 1):
            try:
                # Execute task with timeout
                start_time = time.time()
                
                if asyncio.iscoroutinefunction(task.task_function):
                    # Async task function
                    result = await asyncio.wait_for(
                        task.task_function(*task.args, **task.kwargs),
                        timeout=task.timeout_seconds
                    )
                else:
                    # Sync task function - run in executor
                    result = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, 
                            lambda: task.task_function(*task.args, **task.kwargs)
                        ),
                        timeout=task.timeout_seconds
                    )
                
                # Task completed successfully
                task.execution_time = time.time() - start_time
                task.result = result
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                
                # Record performance for adaptive learning
                self._record_performance(task)
                
                logger.info(f"Task '{task.name}' completed successfully in {task.execution_time:.2f}s")
                break
                
            except asyncio.TimeoutError:
                task.error = TimeoutError(f"Task timed out after {task.timeout_seconds}s")
                logger.warning(f"Task '{task.name}' timed out (attempt {attempt + 1}/{max_retries + 1})")
                
            except Exception as e:
                task.error = e
                logger.warning(f"Task '{task.name}' failed: {e} (attempt {attempt + 1}/{max_retries + 1})")
            
            # Retry logic
            if attempt < max_retries:
                task.retry_count += 1
                # Exponential backoff with jitter
                delay = task.retry_delay * (2 ** attempt) + (time.time() % 1)
                await asyncio.sleep(min(delay, 30))  # Cap at 30 seconds
            else:
                # All retries exhausted
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.now()
                logger.error(f"Task '{task.name}' failed permanently after {max_retries} retries")
        
        # Move task from running to completed
        self.running_tasks.pop(task.task_id, None)
        self.completed_tasks.append(task)
    
    async def _monitor_running_tasks(self):
        """Monitor running tasks and handle completion."""
        if not self.running_tasks:
            return
        
        # Wait briefly for tasks to progress
        await asyncio.sleep(0.1)
        
        # Update metrics periodically
        if self.enable_metrics:
            all_tasks = self.completed_tasks + list(self.running_tasks.values())
            self.metrics.update_metrics(all_tasks)
    
    def _should_open_circuit_breaker(self) -> bool:
        """Check if circuit breaker should be opened based on failure rate."""
        now = datetime.now()
        
        # Only check periodically
        if now - self.last_circuit_check < timedelta(seconds=30):
            return self.circuit_breaker_open
        
        self.last_circuit_check = now
        
        # Get recent completed tasks
        recent_tasks = [
            task for task in self.completed_tasks
            if task.completed_at and (now - task.completed_at) < self.circuit_breaker_window
        ]
        
        if len(recent_tasks) < 5:  # Need minimum sample size
            self.circuit_breaker_open = False
            return False
        
        failure_rate = len([t for t in recent_tasks if t.status == TaskStatus.FAILED]) / len(recent_tasks)
        
        if failure_rate >= self.circuit_breaker_threshold:
            if not self.circuit_breaker_open:
                logger.warning(f"Opening circuit breaker - failure rate: {failure_rate:.2%}")
            self.circuit_breaker_open = True
        else:
            if self.circuit_breaker_open:
                logger.info(f"Closing circuit breaker - failure rate improved: {failure_rate:.2%}")
            self.circuit_breaker_open = False
        
        return self.circuit_breaker_open
    
    def _record_performance(self, task: AutonomousTask):
        """Record task performance for adaptive learning."""
        if not self.optimization_enabled:
            return
        
        task_type = task.name or "unknown"
        
        if task_type not in self.performance_history:
            self.performance_history[task_type] = []
        
        self.performance_history[task_type].append(task.execution_time)
        
        # Keep only recent performance data (last 100 executions)
        if len(self.performance_history[task_type]) > 100:
            self.performance_history[task_type] = self.performance_history[task_type][-100:]
    
    def get_performance_insights(self) -> Dict[str, Any]:
        """Get performance insights for optimization."""
        insights = {
            "total_execution_time": (datetime.now() - self.engine_start_time).total_seconds(),
            "metrics": self.metrics,
            "circuit_breaker_status": {
                "open": self.circuit_breaker_open,
                "last_check": self.last_circuit_check.isoformat()
            },
            "task_performance": {}
        }
        
        # Add task-specific performance data
        for task_type, times in self.performance_history.items():
            if times:
                insights["task_performance"][task_type] = {
                    "executions": len(times),
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times)
                }
        
        return insights
    
    def optimize_task_scheduling(self):
        """Apply adaptive optimizations based on performance history."""
        if not self.optimization_enabled or not self.performance_history:
            return
        
        # Adjust timeout values based on historical performance
        for task in self.task_queue:
            task_type = task.name or "unknown"
            if task_type in self.performance_history:
                times = self.performance_history[task_type]
                if times:
                    # Set timeout to 3x average time, minimum 30s
                    avg_time = sum(times) / len(times)
                    optimal_timeout = max(avg_time * 3, 30)
                    task.timeout_seconds = min(optimal_timeout, task.timeout_seconds)
        
        logger.debug("Applied adaptive optimizations to task scheduling")


# Factory function for easy instantiation
def create_enhanced_autonomous_engine(max_concurrent_tasks: int = 10, 
                                     enable_metrics: bool = True) -> EnhancedAutonomousEngine:
    """Create and return a configured Enhanced Autonomous Engine."""
    return EnhancedAutonomousEngine(
        max_concurrent_tasks=max_concurrent_tasks,
        enable_metrics=enable_metrics
    )


# Example usage and demonstration
async def autonomous_demo():
    """Demonstrate the Enhanced Autonomous Engine capabilities."""
    
    def sample_task(task_name: str, duration: float = 1.0):
        """Sample task for demonstration."""
        import time
        time.sleep(duration)
        return f"Completed {task_name}"
    
    async def async_sample_task(task_name: str, duration: float = 1.0):
        """Async sample task for demonstration."""
        await asyncio.sleep(duration)
        return f"Async completed {task_name}"
    
    # Create engine
    engine = create_enhanced_autonomous_engine(max_concurrent_tasks=5)
    
    # Add various tasks with different priorities
    engine.add_task(
        sample_task, "Critical Task", 0.5, 
        name="critical_task", 
        priority=TaskPriority.CRITICAL
    )
    
    engine.add_task(
        async_sample_task, "High Priority Async", 1.0,
        name="high_async_task",
        priority=TaskPriority.HIGH
    )
    
    engine.add_task(
        sample_task, "Medium Task", 2.0,
        name="medium_task",
        priority=TaskPriority.MEDIUM
    )
    
    engine.add_task(
        sample_task, "Background Task", 3.0,
        name="background_task",
        priority=TaskPriority.BACKGROUND
    )
    
    # Execute all tasks
    metrics = await engine.execute_all_tasks()
    
    # Display results
    insights = engine.get_performance_insights()
    
    print(f"Execution completed!")
    print(f"Success rate: {metrics.success_rate:.2%}")
    print(f"Total tasks: {metrics.total_tasks}")
    print(f"Average execution time: {metrics.average_execution_time:.2f}s")
    
    return insights


if __name__ == "__main__":
    # Run demonstration
    asyncio.run(autonomous_demo())