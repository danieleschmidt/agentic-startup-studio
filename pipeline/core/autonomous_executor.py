"""
Autonomous Execution Engine - Core orchestration for self-improving systems
Implements adaptive learning and autonomous decision-making capabilities.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from opentelemetry import trace

from ..config.settings import get_settings
from ..infrastructure.circuit_breaker import CircuitBreaker
from ..telemetry import get_tracer

# Global configuration
settings = get_settings()
tracer = get_tracer(__name__)
logger = logging.getLogger(__name__)


class ExecutionStatus(str, Enum):
    """Execution status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Priority(str, Enum):
    """Task priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ExecutionMetrics:
    """Metrics for tracking execution performance"""
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    success_rate: float = 0.0
    error_count: int = 0
    retries: int = 0
    
    def complete(self) -> None:
        """Mark execution as complete and calculate metrics"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time


class AutonomousTask(BaseModel):
    """Self-executing task with learning capabilities"""
    id: str = Field(..., description="Unique task identifier")
    name: str = Field(..., description="Human-readable task name")
    description: str = Field(..., description="Detailed task description")
    priority: Priority = Field(default=Priority.MEDIUM)
    status: ExecutionStatus = Field(default=ExecutionStatus.PENDING)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    scheduled_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    max_retries: int = Field(default=3)
    retry_delay: int = Field(default=60)  # seconds
    dependencies: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    execution_context: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class AutonomousExecutor:
    """
    Core autonomous execution engine that manages self-improving workflows
    """
    
    def __init__(self):
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
            timeout_seconds=settings.CIRCUIT_BREAKER_TIMEOUT_SECONDS
        )
        self.tasks: Dict[str, AutonomousTask] = {}
        self.metrics: Dict[str, ExecutionMetrics] = {}
        self.learning_data: Dict[str, Any] = {}
        self._running = False
        
    async def start(self) -> None:
        """Start the autonomous executor"""
        with tracer.start_as_current_span("autonomous_executor_start"):
            self._running = True
            logger.info("Autonomous executor started")
            
            # Start background tasks
            asyncio.create_task(self._execution_loop())
            asyncio.create_task(self._metrics_collector())
            asyncio.create_task(self._learning_loop())
    
    async def stop(self) -> None:
        """Stop the autonomous executor"""
        self._running = False
        logger.info("Autonomous executor stopped")
    
    async def submit_task(self, task: AutonomousTask) -> str:
        """Submit a task for autonomous execution"""
        with tracer.start_as_current_span("submit_task") as span:
            span.set_attributes({
                "task.id": task.id,
                "task.name": task.name,
                "task.priority": task.priority.value
            })
            
            self.tasks[task.id] = task
            self.metrics[task.id] = ExecutionMetrics(start_time=time.time())
            
            logger.info(f"Task submitted: {task.name} ({task.id})")
            return task.id
    
    async def _execution_loop(self) -> None:
        """Main execution loop for processing tasks"""
        while self._running:
            try:
                await self._process_pending_tasks()
                await asyncio.sleep(5)  # Check every 5 seconds
            except Exception as e:
                logger.error(f"Error in execution loop: {e}")
                await asyncio.sleep(10)  # Back off on errors
    
    async def _process_pending_tasks(self) -> None:
        """Process all pending tasks based on priority and dependencies"""
        with tracer.start_as_current_span("process_pending_tasks"):
            # Get ready tasks (pending, dependencies met, scheduled time passed)
            ready_tasks = self._get_ready_tasks()
            
            # Sort by priority
            ready_tasks.sort(key=lambda t: self._get_priority_weight(t.priority), reverse=True)
            
            for task in ready_tasks[:5]:  # Process up to 5 tasks concurrently
                asyncio.create_task(self._execute_task(task))
    
    def _get_ready_tasks(self) -> List[AutonomousTask]:
        """Get tasks that are ready for execution"""
        ready_tasks = []
        now = datetime.utcnow()
        
        for task in self.tasks.values():
            if task.status != ExecutionStatus.PENDING:
                continue
                
            # Check if scheduled time has passed
            if task.scheduled_at and task.scheduled_at > now:
                continue
                
            # Check if dependencies are met
            if not self._dependencies_met(task):
                continue
                
            ready_tasks.append(task)
        
        return ready_tasks
    
    def _dependencies_met(self, task: AutonomousTask) -> bool:
        """Check if all task dependencies are completed"""
        for dep_id in task.dependencies:
            if dep_id not in self.tasks:
                return False
            if self.tasks[dep_id].status != ExecutionStatus.COMPLETED:
                return False
        return True
    
    def _get_priority_weight(self, priority: Priority) -> int:
        """Get numeric weight for priority comparison"""
        weights = {
            Priority.LOW: 1,
            Priority.MEDIUM: 2,
            Priority.HIGH: 3,
            Priority.CRITICAL: 4
        }
        return weights.get(priority, 2)
    
    async def _execute_task(self, task: AutonomousTask) -> None:
        """Execute a single task with error handling and retries"""
        with tracer.start_as_current_span("execute_task") as span:
            span.set_attributes({
                "task.id": task.id,
                "task.name": task.name
            })
            
            task.status = ExecutionStatus.IN_PROGRESS
            logger.info(f"Executing task: {task.name}")
            
            try:
                # Execute with circuit breaker protection
                result = await self.circuit_breaker.call(
                    self._execute_task_logic, task
                )
                
                task.status = ExecutionStatus.COMPLETED
                task.completed_at = datetime.utcnow()
                self.metrics[task.id].complete()
                
                logger.info(f"Task completed: {task.name}")
                
            except Exception as e:
                logger.error(f"Task failed: {task.name} - {e}")
                await self._handle_task_failure(task, e)
    
    async def _execute_task_logic(self, task: AutonomousTask) -> Any:
        """Core task execution logic - override in subclasses"""
        # Simulate work - in real implementation, this would route to appropriate handlers
        await asyncio.sleep(1)
        
        # Apply learning from previous executions
        learning = self.learning_data.get(task.name, {})
        if learning.get('optimization_factor'):
            # Apply learned optimizations
            await asyncio.sleep(0.1)  # Optimized execution
        
        return {"status": "success", "optimized": bool(learning)}
    
    async def _handle_task_failure(self, task: AutonomousTask, error: Exception) -> None:
        """Handle task failures with retry logic"""
        self.metrics[task.id].error_count += 1
        
        if self.metrics[task.id].retries < task.max_retries:
            self.metrics[task.id].retries += 1
            
            # Schedule retry with exponential backoff
            delay = task.retry_delay * (2 ** self.metrics[task.id].retries)
            task.scheduled_at = datetime.utcnow() + timedelta(seconds=delay)
            task.status = ExecutionStatus.PENDING
            
            logger.info(f"Task retry scheduled: {task.name} (attempt {self.metrics[task.id].retries + 1})")
        else:
            task.status = ExecutionStatus.FAILED
            logger.error(f"Task failed permanently: {task.name}")
    
    async def _metrics_collector(self) -> None:
        """Collect and analyze execution metrics"""
        while self._running:
            try:
                await self._analyze_performance_metrics()
                await asyncio.sleep(60)  # Collect every minute
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(60)
    
    async def _analyze_performance_metrics(self) -> None:
        """Analyze performance metrics and identify optimization opportunities"""
        with tracer.start_as_current_span("analyze_metrics"):
            total_tasks = len(self.tasks)
            completed_tasks = len([t for t in self.tasks.values() if t.status == ExecutionStatus.COMPLETED])
            failed_tasks = len([t for t in self.tasks.values() if t.status == ExecutionStatus.FAILED])
            
            success_rate = completed_tasks / total_tasks if total_tasks > 0 else 0
            
            logger.info(f"Metrics - Total: {total_tasks}, Completed: {completed_tasks}, Failed: {failed_tasks}, Success Rate: {success_rate:.2%}")
    
    async def _learning_loop(self) -> None:
        """Continuous learning loop for optimization"""
        while self._running:
            try:
                await self._update_learning_data()
                await asyncio.sleep(300)  # Learn every 5 minutes
            except Exception as e:
                logger.error(f"Error in learning loop: {e}")
                await asyncio.sleep(300)
    
    async def _update_learning_data(self) -> None:
        """Update learning data based on execution patterns"""
        with tracer.start_as_current_span("update_learning"):
            # Analyze successful task patterns
            successful_tasks = [t for t in self.tasks.values() if t.status == ExecutionStatus.COMPLETED]
            
            for task in successful_tasks:
                task_name = task.name
                if task_name not in self.learning_data:
                    self.learning_data[task_name] = {}
                
                # Calculate optimization factor based on execution metrics
                if task.id in self.metrics:
                    metrics = self.metrics[task.id]
                    if metrics.duration and metrics.duration < 1.0:  # Fast execution
                        self.learning_data[task_name]['optimization_factor'] = 1.2
                    
            logger.debug(f"Updated learning data for {len(self.learning_data)} task types")
    
    def get_status_report(self) -> Dict[str, Any]:
        """Generate comprehensive status report"""
        with tracer.start_as_current_span("status_report"):
            total_tasks = len(self.tasks)
            status_counts = {}
            
            for status in ExecutionStatus:
                status_counts[status.value] = len([t for t in self.tasks.values() if t.status == status])
            
            avg_execution_time = 0
            completed_metrics = [m for m in self.metrics.values() if m.duration]
            if completed_metrics:
                avg_execution_time = sum(m.duration for m in completed_metrics) / len(completed_metrics)
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "total_tasks": total_tasks,
                "status_distribution": status_counts,
                "success_rate": status_counts.get("completed", 0) / total_tasks if total_tasks > 0 else 0,
                "average_execution_time": avg_execution_time,
                "learning_patterns": len(self.learning_data),
                "circuit_breaker_status": self.circuit_breaker.state.value if hasattr(self.circuit_breaker.state, 'value') else str(self.circuit_breaker.state),
                "running": self._running
            }


# Global executor instance
_executor: Optional[AutonomousExecutor] = None


async def get_executor() -> AutonomousExecutor:
    """Get or create the global autonomous executor instance"""
    global _executor
    if _executor is None:
        _executor = AutonomousExecutor()
        await _executor.start()
    return _executor


async def submit_autonomous_task(
    name: str,
    description: str,
    priority: Priority = Priority.MEDIUM,
    dependencies: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Convenience function to submit autonomous tasks"""
    executor = await get_executor()
    
    task = AutonomousTask(
        id=f"{name}_{int(time.time())}",
        name=name,
        description=description,
        priority=priority,
        dependencies=dependencies or [],
        metadata=metadata or {}
    )
    
    return await executor.submit_task(task)


async def get_execution_status() -> Dict[str, Any]:
    """Get current execution status"""
    executor = await get_executor()
    return executor.get_status_report()