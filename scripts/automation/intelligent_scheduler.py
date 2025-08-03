#!/usr/bin/env python3
"""
Intelligent Scheduler for SDLC Automation

This module provides intelligent scheduling capabilities for automation tasks,
including dynamic scheduling based on system load, priority queues, and
adaptive execution strategies.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Callable, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import heapq
import psutil
import aiofiles
from croniter import croniter


class ScheduleType(Enum):
    """Types of scheduling strategies."""
    IMMEDIATE = "immediate"
    CRON = "cron"
    INTERVAL = "interval"
    CONDITIONAL = "conditional"
    ADAPTIVE = "adaptive"


class ResourceConstraint(Enum):
    """System resource constraints."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"


@dataclass
class ScheduleRule:
    """Scheduling rule definition."""
    rule_id: str
    name: str
    schedule_type: ScheduleType
    expression: str  # cron expression, interval seconds, or condition
    enabled: bool = True
    priority: int = 0  # Higher number = higher priority
    resource_requirements: Dict[ResourceConstraint, float] = None
    conditions: Dict[str, Any] = None
    max_concurrent: int = 1
    timeout_seconds: int = 3600


@dataclass
class ScheduledTask:
    """Scheduled task instance."""
    task_id: str
    schedule_rule_id: str
    scheduled_time: datetime
    priority: int
    resource_requirements: Dict[ResourceConstraint, float]
    timeout_seconds: int
    attempts: int = 0
    max_attempts: int = 3
    
    def __lt__(self, other):
        # For priority queue - higher priority first, then earlier time
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.scheduled_time < other.scheduled_time


class SystemResourceMonitor:
    """Monitors system resources for intelligent scheduling."""
    
    def __init__(self):
        self.thresholds = {
            ResourceConstraint.CPU: 80.0,
            ResourceConstraint.MEMORY: 85.0,
            ResourceConstraint.DISK: 90.0,
            ResourceConstraint.NETWORK: 75.0
        }
    
    async def get_current_usage(self) -> Dict[ResourceConstraint, float]:
        """Get current system resource usage."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage (root partition)
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Network usage (simplified - could be enhanced)
            network_stats = psutil.net_io_counters()
            # For simplicity, we'll use a basic calculation
            network_percent = min(50.0, (network_stats.bytes_sent + network_stats.bytes_recv) / (1024 * 1024 * 1024) * 10)
            
            return {
                ResourceConstraint.CPU: cpu_percent,
                ResourceConstraint.MEMORY: memory_percent,
                ResourceConstraint.DISK: disk_percent,
                ResourceConstraint.NETWORK: network_percent
            }
        except Exception as e:
            logging.warning(f"Failed to get system resource usage: {e}")
            return {constraint: 0.0 for constraint in ResourceConstraint}
    
    async def can_accommodate(self, requirements: Dict[ResourceConstraint, float]) -> bool:
        """Check if system can accommodate resource requirements."""
        current_usage = await self.get_current_usage()
        
        for constraint, required in requirements.items():
            current = current_usage.get(constraint, 0.0)
            if current + required > self.thresholds.get(constraint, 100.0):
                return False
        
        return True
    
    def get_system_load_score(self, current_usage: Dict[ResourceConstraint, float]) -> float:
        """Calculate system load score (0.0 to 1.0, where 1.0 is fully loaded)."""
        total_score = 0.0
        for constraint, usage in current_usage.items():
            threshold = self.thresholds.get(constraint, 100.0)
            normalized_usage = min(usage / threshold, 1.0)
            total_score += normalized_usage
        
        return total_score / len(current_usage)


class IntelligentScheduler:
    """Intelligent scheduler for SDLC automation tasks."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.resource_monitor = SystemResourceMonitor()
        
        # Scheduling state
        self.schedule_rules: Dict[str, ScheduleRule] = {}
        self.task_queue: List[ScheduledTask] = []  # Priority queue
        self.running_tasks: Dict[str, ScheduledTask] = {}
        self.completed_tasks: List[Dict[str, Any]] = []
        
        # Statistics
        self.execution_stats = {
            "total_scheduled": 0,
            "total_executed": 0,
            "total_failed": 0,
            "total_skipped": 0,
            "avg_execution_time": 0.0,
            "last_execution": None
        }
        
        self.shutdown_requested = False
        
        # Load default scheduling rules
        self._load_default_rules()
    
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load scheduler configuration."""
        default_config = {
            "max_concurrent_tasks": 4,
            "check_interval_seconds": 30,
            "resource_check_enabled": True,
            "adaptive_scheduling": True,
            "priority_boost_threshold": 3600,  # Boost priority after 1 hour
            "system_load_threshold": 0.8,
            "retry_delay_base": 60,
            "retry_delay_multiplier": 2.0,
            "max_retry_delay": 1800,
            "cleanup_interval_hours": 24
        }
        
        if config_path and config_path.exists():
            try:
                with open(config_path) as f:
                    file_config = json.load(f)
                    default_config.update(file_config)
            except Exception as e:
                logging.warning(f"Failed to load scheduler config: {e}")
        
        return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the scheduler."""
        logger = logging.getLogger("intelligent_scheduler")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_default_rules(self):
        """Load default scheduling rules."""
        default_rules = [
            ScheduleRule(
                rule_id="daily_health_check",
                name="Daily System Health Check",
                schedule_type=ScheduleType.CRON,
                expression="0 8 * * *",  # Daily at 8 AM
                priority=100,
                resource_requirements={
                    ResourceConstraint.CPU: 10.0,
                    ResourceConstraint.MEMORY: 5.0
                },
                timeout_seconds=300
            ),
            ScheduleRule(
                rule_id="security_scan_weekly",
                name="Weekly Security Scan",
                schedule_type=ScheduleType.CRON,
                expression="0 2 * * 1",  # Monday at 2 AM
                priority=200,
                resource_requirements={
                    ResourceConstraint.CPU: 30.0,
                    ResourceConstraint.MEMORY: 15.0
                },
                timeout_seconds=1800
            ),
            ScheduleRule(
                rule_id="dependency_check_weekly",
                name="Weekly Dependency Check",
                schedule_type=ScheduleType.CRON,
                expression="0 3 * * 1",  # Monday at 3 AM
                priority=150,
                resource_requirements={
                    ResourceConstraint.CPU: 20.0,
                    ResourceConstraint.MEMORY: 10.0
                },
                timeout_seconds=900
            ),
            ScheduleRule(
                rule_id="metrics_collection",
                name="Metrics Collection",
                schedule_type=ScheduleType.INTERVAL,
                expression="300",  # Every 5 minutes
                priority=50,
                resource_requirements={
                    ResourceConstraint.CPU: 5.0,
                    ResourceConstraint.MEMORY: 2.0
                },
                timeout_seconds=120
            ),
            ScheduleRule(
                rule_id="log_cleanup",
                name="Log Cleanup",
                schedule_type=ScheduleType.CRON,
                expression="0 1 * * 0",  # Sunday at 1 AM
                priority=25,
                resource_requirements={
                    ResourceConstraint.CPU: 10.0,
                    ResourceConstraint.DISK: 5.0
                },
                timeout_seconds=600
            )
        ]
        
        for rule in default_rules:
            self.schedule_rules[rule.rule_id] = rule
    
    def add_schedule_rule(self, rule: ScheduleRule):
        """Add a new scheduling rule."""
        self.schedule_rules[rule.rule_id] = rule
        self.logger.info(f"Added scheduling rule: {rule.name}")
    
    def remove_schedule_rule(self, rule_id: str):
        """Remove a scheduling rule."""
        if rule_id in self.schedule_rules:
            del self.schedule_rules[rule_id]
            self.logger.info(f"Removed scheduling rule: {rule_id}")
    
    def _calculate_next_execution_time(self, rule: ScheduleRule, base_time: Optional[datetime] = None) -> Optional[datetime]:
        """Calculate next execution time for a rule."""
        if base_time is None:
            base_time = datetime.now(timezone.utc)
        
        try:
            if rule.schedule_type == ScheduleType.IMMEDIATE:
                return base_time
            
            elif rule.schedule_type == ScheduleType.CRON:
                cron = croniter(rule.expression, base_time)
                return cron.get_next(datetime)
            
            elif rule.schedule_type == ScheduleType.INTERVAL:
                interval_seconds = int(rule.expression)
                return base_time + timedelta(seconds=interval_seconds)
            
            elif rule.schedule_type == ScheduleType.CONDITIONAL:
                # Conditional scheduling would require evaluation of conditions
                # For now, return None to indicate no scheduling
                return None
            
            elif rule.schedule_type == ScheduleType.ADAPTIVE:
                # Adaptive scheduling based on system load and priority
                system_load = asyncio.create_task(self._get_adaptive_delay(rule))
                # This is a simplified approach - in practice, you'd want to handle this asynchronously
                return base_time + timedelta(seconds=300)  # Default 5 minutes
            
        except Exception as e:
            self.logger.error(f"Failed to calculate next execution time for rule {rule.rule_id}: {e}")
            return None
    
    async def _get_adaptive_delay(self, rule: ScheduleRule) -> int:
        """Calculate adaptive delay based on system conditions."""
        current_usage = await self.resource_monitor.get_current_usage()
        load_score = self.resource_monitor.get_system_load_score(current_usage)
        
        # Base delay increases with system load
        base_delay = 60  # 1 minute
        if load_score > 0.8:
            delay_multiplier = 5.0
        elif load_score > 0.6:
            delay_multiplier = 3.0
        elif load_score > 0.4:
            delay_multiplier = 2.0
        else:
            delay_multiplier = 1.0
        
        # Priority affects delay (higher priority = shorter delay)
        priority_factor = max(0.1, (200 - rule.priority) / 200)
        
        return int(base_delay * delay_multiplier * priority_factor)
    
    async def schedule_tasks(self):
        """Generate scheduled tasks based on rules."""
        current_time = datetime.now(timezone.utc)
        new_tasks = []
        
        for rule_id, rule in self.schedule_rules.items():
            if not rule.enabled:
                continue
            
            # Check if task is already running
            running_count = sum(1 for task in self.running_tasks.values() 
                              if task.schedule_rule_id == rule_id)
            
            if running_count >= rule.max_concurrent:
                continue
            
            # Calculate next execution time
            next_time = self._calculate_next_execution_time(rule, current_time)
            if next_time is None:
                continue
            
            # Check if we should schedule this task
            should_schedule = False
            
            if rule.schedule_type == ScheduleType.IMMEDIATE:
                should_schedule = True
            elif next_time <= current_time + timedelta(seconds=self.config["check_interval_seconds"]):
                should_schedule = True
            
            if should_schedule:
                # Check for resource availability if enabled
                if self.config["resource_check_enabled"] and rule.resource_requirements:
                    if not await self.resource_monitor.can_accommodate(rule.resource_requirements):
                        self.logger.debug(f"Delaying task {rule.name} due to resource constraints")
                        continue
                
                # Create scheduled task
                task = ScheduledTask(
                    task_id=f"{rule_id}_{int(current_time.timestamp())}",
                    schedule_rule_id=rule_id,
                    scheduled_time=next_time,
                    priority=rule.priority,
                    resource_requirements=rule.resource_requirements or {},
                    timeout_seconds=rule.timeout_seconds
                )
                
                new_tasks.append(task)
                self.logger.info(f"Scheduled task: {rule.name} at {next_time}")
        
        # Add new tasks to queue
        for task in new_tasks:
            heapq.heappush(self.task_queue, task)
            self.execution_stats["total_scheduled"] += 1
    
    async def execute_ready_tasks(self) -> List[Dict[str, Any]]:
        """Execute tasks that are ready to run."""
        current_time = datetime.now(timezone.utc)
        executed_tasks = []
        
        # Check how many tasks we can run concurrently
        max_concurrent = self.config["max_concurrent_tasks"]
        current_running = len(self.running_tasks)
        
        if current_running >= max_concurrent:
            return executed_tasks
        
        # Get ready tasks from queue
        ready_tasks = []
        remaining_tasks = []
        
        while self.task_queue:
            task = heapq.heappop(self.task_queue)
            
            if task.scheduled_time <= current_time:
                ready_tasks.append(task)
            else:
                remaining_tasks.append(task)
        
        # Put non-ready tasks back in queue
        for task in remaining_tasks:
            heapq.heappush(self.task_queue, task)
        
        # Sort ready tasks by priority
        ready_tasks.sort(key=lambda t: (-t.priority, t.scheduled_time))
        
        # Execute tasks up to concurrent limit
        for task in ready_tasks[:max_concurrent - current_running]:
            try:
                # Check resource requirements again
                if self.config["resource_check_enabled"] and task.resource_requirements:
                    if not await self.resource_monitor.can_accommodate(task.resource_requirements):
                        # Reschedule for later
                        task.scheduled_time = current_time + timedelta(seconds=300)
                        heapq.heappush(self.task_queue, task)
                        continue
                
                # Start task execution
                execution_result = await self._execute_scheduled_task(task)
                executed_tasks.append(execution_result)
                
            except Exception as e:
                self.logger.error(f"Failed to execute task {task.task_id}: {e}")
                execution_result = {
                    "task_id": task.task_id,
                    "schedule_rule_id": task.schedule_rule_id,
                    "status": "failed",
                    "error": str(e),
                    "execution_time": current_time.isoformat()
                }
                executed_tasks.append(execution_result)
        
        return executed_tasks
    
    async def _execute_scheduled_task(self, task: ScheduledTask) -> Dict[str, Any]:
        """Execute a single scheduled task."""
        start_time = datetime.now(timezone.utc)
        self.running_tasks[task.task_id] = task
        
        try:
            rule = self.schedule_rules[task.schedule_rule_id]
            
            # Get the actual automation task and execute it
            # This would integrate with the SDLC automation orchestrator
            
            # For now, simulate task execution
            execution_time = min(30, task.timeout_seconds / 10)  # Simulated execution
            await asyncio.sleep(execution_time)
            
            execution_result = {
                "task_id": task.task_id,
                "schedule_rule_id": task.schedule_rule_id,
                "rule_name": rule.name,
                "status": "completed",
                "start_time": start_time.isoformat(),
                "end_time": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": execution_time,
                "priority": task.priority,
                "attempts": task.attempts + 1
            }
            
            self.execution_stats["total_executed"] += 1
            self.execution_stats["last_execution"] = datetime.now(timezone.utc).isoformat()
            
            # Update average execution time
            if self.execution_stats["avg_execution_time"] == 0:
                self.execution_stats["avg_execution_time"] = execution_time
            else:
                total_executions = self.execution_stats["total_executed"]
                current_avg = self.execution_stats["avg_execution_time"]
                self.execution_stats["avg_execution_time"] = (
                    (current_avg * (total_executions - 1) + execution_time) / total_executions
                )
            
            self.logger.info(f"Completed scheduled task: {rule.name}")
            
        except Exception as e:
            execution_result = {
                "task_id": task.task_id,
                "schedule_rule_id": task.schedule_rule_id,
                "status": "failed",
                "error": str(e),
                "start_time": start_time.isoformat(),
                "end_time": datetime.now(timezone.utc).isoformat(),
                "attempts": task.attempts + 1
            }
            
            self.execution_stats["total_failed"] += 1
            self.logger.error(f"Failed to execute scheduled task {task.task_id}: {e}")
            
            # Handle retry logic
            if task.attempts < task.max_attempts - 1:
                retry_delay = min(
                    self.config["max_retry_delay"],
                    self.config["retry_delay_base"] * (self.config["retry_delay_multiplier"] ** task.attempts)
                )
                
                task.attempts += 1
                task.scheduled_time = datetime.now(timezone.utc) + timedelta(seconds=retry_delay)
                heapq.heappush(self.task_queue, task)
                
                self.logger.info(f"Rescheduled failed task {task.task_id} for retry in {retry_delay} seconds")
        
        finally:
            # Remove from running tasks
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]
            
            # Store completed task
            self.completed_tasks.append(execution_result)
        
        return execution_result
    
    async def cleanup_old_data(self):
        """Clean up old execution data."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.config["cleanup_interval_hours"])
        
        # Remove old completed tasks
        initial_count = len(self.completed_tasks)
        self.completed_tasks = [
            task for task in self.completed_tasks
            if datetime.fromisoformat(task.get("end_time", "").replace('Z', '+00:00')) > cutoff_time
        ]
        
        removed_count = initial_count - len(self.completed_tasks)
        if removed_count > 0:
            self.logger.info(f"Cleaned up {removed_count} old task execution records")
    
    async def get_scheduler_status(self) -> Dict[str, Any]:
        """Get current scheduler status."""
        current_usage = await self.resource_monitor.get_current_usage()
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "statistics": self.execution_stats,
            "queued_tasks": len(self.task_queue),
            "running_tasks": len(self.running_tasks),
            "total_rules": len(self.schedule_rules),
            "enabled_rules": sum(1 for rule in self.schedule_rules.values() if rule.enabled),
            "system_usage": current_usage,
            "system_load_score": self.resource_monitor.get_system_load_score(current_usage),
            "next_tasks": [
                {
                    "task_id": task.task_id,
                    "rule_id": task.schedule_rule_id,
                    "scheduled_time": task.scheduled_time.isoformat(),
                    "priority": task.priority
                }
                for task in sorted(self.task_queue)[:5]
            ],
            "running_task_details": [
                {
                    "task_id": task.task_id,
                    "rule_id": task.schedule_rule_id,
                    "priority": task.priority
                }
                for task in self.running_tasks.values()
            ]
        }
    
    async def run_scheduler_loop(self):
        """Main scheduler loop."""
        self.logger.info("Starting intelligent scheduler loop...")
        
        try:
            while not self.shutdown_requested:
                loop_start = datetime.now(timezone.utc)
                
                # Schedule new tasks
                await self.schedule_tasks()
                
                # Execute ready tasks
                executed_tasks = await self.execute_ready_tasks()
                
                if executed_tasks:
                    self.logger.info(f"Executed {len(executed_tasks)} tasks in this cycle")
                
                # Periodic cleanup
                if datetime.now(timezone.utc).hour == 0:  # Cleanup at midnight
                    await self.cleanup_old_data()
                
                # Calculate sleep time
                loop_duration = (datetime.now(timezone.utc) - loop_start).total_seconds()
                sleep_time = max(0, self.config["check_interval_seconds"] - loop_duration)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
        except KeyboardInterrupt:
            self.logger.info("Scheduler interrupted by user")
        except Exception as e:
            self.logger.error(f"Scheduler loop error: {e}")
            raise
        finally:
            self.logger.info("Scheduler loop stopped")
    
    async def shutdown(self):
        """Graceful shutdown of the scheduler."""
        self.logger.info("Shutting down intelligent scheduler...")
        self.shutdown_requested = True
        
        # Wait for running tasks to complete (with timeout)
        if self.running_tasks:
            self.logger.info(f"Waiting for {len(self.running_tasks)} running tasks to complete...")
            timeout = 60  # 1 minute timeout
            start_time = datetime.now()
            
            while self.running_tasks and (datetime.now() - start_time).seconds < timeout:
                await asyncio.sleep(1)
        
        self.logger.info("Scheduler shutdown completed")


async def main():
    """Main entry point for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Intelligent Scheduler for SDLC Automation")
    parser.add_argument("--config", type=Path, help="Configuration file path")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    parser.add_argument("--status", action="store_true", help="Show scheduler status")
    parser.add_argument("--schedule", help="Schedule a specific rule by ID")
    
    args = parser.parse_args()
    
    scheduler = IntelligentScheduler(args.config)
    
    try:
        if args.status:
            status = await scheduler.get_scheduler_status()
            print(json.dumps(status, indent=2))
        
        elif args.schedule:
            if args.schedule in scheduler.schedule_rules:
                await scheduler.schedule_tasks()
                print(f"Scheduled tasks for rule: {args.schedule}")
            else:
                print(f"Rule not found: {args.schedule}")
        
        elif args.daemon:
            await scheduler.run_scheduler_loop()
        
        else:
            parser.print_help()
    
    finally:
        await scheduler.shutdown()


if __name__ == "__main__":
    asyncio.run(main())