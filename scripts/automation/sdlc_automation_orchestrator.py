#!/usr/bin/env python3
"""
SDLC Automation Orchestrator for Enterprise Environments

This module provides comprehensive automation orchestration for all SDLC phases,
including intelligent scheduling, dependency management, and failure recovery.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import subprocess
import sys
import aiofiles
import yaml
from concurrent.futures import ThreadPoolExecutor
import signal


class AutomationPhase(Enum):
    """SDLC automation phases."""
    PLANNING = "planning"
    DEVELOPMENT = "development"
    TESTING = "testing"
    BUILD = "build"
    SECURITY = "security"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    MAINTENANCE = "maintenance"


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


@dataclass
class AutomationTask:
    """Individual automation task definition."""
    task_id: str
    name: str
    phase: AutomationPhase
    command: str
    description: str
    priority: TaskPriority
    dependencies: List[str]
    timeout_seconds: int
    retry_count: int
    retry_delay: int
    environment: Dict[str, str]
    working_directory: Optional[str] = None
    success_criteria: Optional[str] = None
    failure_action: Optional[str] = None
    schedule: Optional[str] = None  # cron-like schedule
    enabled: bool = True


@dataclass
class TaskExecution:
    """Task execution tracking."""
    task_id: str
    execution_id: str
    status: TaskStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    retry_attempt: int = 0
    error_message: Optional[str] = None


class SDLCAutomationOrchestrator:
    """Orchestrates comprehensive SDLC automation workflows."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.tasks: Dict[str, AutomationTask] = {}
        self.executions: Dict[str, TaskExecution] = {}
        self.running_tasks: Set[str] = set()
        self.executor = ThreadPoolExecutor(max_workers=self.config.get("max_concurrent_tasks", 4))
        self.shutdown_requested = False
        
        # Load automation tasks
        self._load_automation_tasks()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load orchestrator configuration."""
        default_config = {
            "max_concurrent_tasks": 4,
            "default_timeout": 3600,
            "default_retry_count": 3,
            "default_retry_delay": 60,
            "log_level": "INFO",
            "execution_history_days": 30,
            "metrics_enabled": True,
            "notification_enabled": True,
            "notification_webhook": None,
            "phases_enabled": [phase.value for phase in AutomationPhase],
            "environment_variables": {},
            "working_directory": str(Path.cwd())
        }
        
        if config_path and config_path.exists():
            try:
                with open(config_path) as f:
                    if config_path.suffix.lower() in ['.yml', '.yaml']:
                        file_config = yaml.safe_load(f)
                    else:
                        file_config = json.load(f)
                    default_config.update(file_config)
            except Exception as e:
                logging.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging for automation orchestration."""
        logger = logging.getLogger("sdlc_automation")
        logger.setLevel(getattr(logging, self.config.get("log_level", "INFO")))
        
        if not logger.handlers:
            # File handler for audit trail
            log_file = Path("automation_orchestrator.log")
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
        self.shutdown_requested = True
    
    def _load_automation_tasks(self):
        """Load automation task definitions."""
        # Development phase tasks
        dev_tasks = [
            AutomationTask(
                task_id="code_quality_check",
                name="Code Quality Analysis",
                phase=AutomationPhase.DEVELOPMENT,
                command="make lint && make type-check",
                description="Run comprehensive code quality analysis",
                priority=TaskPriority.HIGH,
                dependencies=[],
                timeout_seconds=300,
                retry_count=2,
                retry_delay=30,
                environment={"PYTHONPATH": "."},
                success_criteria="exit_code == 0"
            ),
            AutomationTask(
                task_id="dependency_update",
                name="Dependency Updates",
                phase=AutomationPhase.DEVELOPMENT,
                command="python scripts/automation/dependency_updater.py --check-updates",
                description="Check for and apply safe dependency updates",
                priority=TaskPriority.NORMAL,
                dependencies=[],
                timeout_seconds=600,
                retry_count=3,
                retry_delay=60,
                environment={},
                schedule="0 9 * * 1"  # Monday 9 AM
            )
        ]
        
        # Testing phase tasks
        test_tasks = [
            AutomationTask(
                task_id="unit_tests",
                name="Unit Test Suite",
                phase=AutomationPhase.TESTING,
                command="make test-unit",
                description="Execute comprehensive unit test suite",
                priority=TaskPriority.CRITICAL,
                dependencies=["code_quality_check"],
                timeout_seconds=1200,
                retry_count=2,
                retry_delay=60,
                environment={"ENVIRONMENT": "testing"}
            ),
            AutomationTask(
                task_id="integration_tests",
                name="Integration Test Suite",
                phase=AutomationPhase.TESTING,
                command="make test-integration",
                description="Execute integration test suite",
                priority=TaskPriority.HIGH,
                dependencies=["unit_tests"],
                timeout_seconds=1800,
                retry_count=2,
                retry_delay=120,
                environment={"ENVIRONMENT": "testing"}
            ),
            AutomationTask(
                task_id="performance_tests",
                name="Performance Testing",
                phase=AutomationPhase.TESTING,
                command="python scripts/performance_benchmark.py --comprehensive",
                description="Execute performance benchmarks",
                priority=TaskPriority.HIGH,
                dependencies=["integration_tests"],
                timeout_seconds=2400,
                retry_count=1,
                retry_delay=300,
                environment={"ENVIRONMENT": "testing"}
            )
        ]
        
        # Security phase tasks
        security_tasks = [
            AutomationTask(
                task_id="security_scan",
                name="Security Vulnerability Scan",
                phase=AutomationPhase.SECURITY,
                command="make security",
                description="Run comprehensive security scans",
                priority=TaskPriority.CRITICAL,
                dependencies=[],
                timeout_seconds=900,
                retry_count=2,
                retry_delay=60,
                environment={}
            ),
            AutomationTask(
                task_id="compliance_check",
                name="Compliance Validation",
                phase=AutomationPhase.SECURITY,
                command="python monitoring/compliance_monitor.py --assessment",
                description="Validate compliance with security standards",
                priority=TaskPriority.HIGH,
                dependencies=["security_scan"],
                timeout_seconds=600,
                retry_count=2,
                retry_delay=120,
                environment={}
            ),
            AutomationTask(
                task_id="sbom_generation",
                name="SBOM Generation",
                phase=AutomationPhase.SECURITY,
                command="make generate-sbom",
                description="Generate Software Bill of Materials",
                priority=TaskPriority.NORMAL,
                dependencies=["security_scan"],
                timeout_seconds=300,
                retry_count=2,
                retry_delay=60,
                environment={}
            )
        ]
        
        # Build phase tasks
        build_tasks = [
            AutomationTask(
                task_id="container_build",
                name="Container Image Build",
                phase=AutomationPhase.BUILD,
                command="make build-docker",
                description="Build and tag container image",
                priority=TaskPriority.HIGH,
                dependencies=["unit_tests", "security_scan"],
                timeout_seconds=1800,
                retry_count=2,
                retry_delay=120,
                environment={}
            ),
            AutomationTask(
                task_id="container_security_scan",
                name="Container Security Scan",
                phase=AutomationPhase.BUILD,
                command="make scan-security",
                description="Scan container for vulnerabilities",
                priority=TaskPriority.HIGH,
                dependencies=["container_build"],
                timeout_seconds=600,
                retry_count=2,
                retry_delay=60,
                environment={}
            )
        ]
        
        # Monitoring phase tasks
        monitoring_tasks = [
            AutomationTask(
                task_id="metrics_collection",
                name="Metrics Collection",
                phase=AutomationPhase.MONITORING,
                command="python monitoring/enterprise_metrics_collector.py --once",
                description="Collect and analyze system metrics",
                priority=TaskPriority.NORMAL,
                dependencies=[],
                timeout_seconds=300,
                retry_count=3,
                retry_delay=60,
                environment={},
                schedule="*/15 * * * *"  # Every 15 minutes
            ),
            AutomationTask(
                task_id="health_check",
                name="System Health Check",
                phase=AutomationPhase.MONITORING,
                command="make health-check",
                description="Comprehensive system health validation",
                priority=TaskPriority.HIGH,
                dependencies=[],
                timeout_seconds=180,
                retry_count=2,
                retry_delay=30,
                environment={},
                schedule="*/5 * * * *"  # Every 5 minutes
            )
        ]
        
        # Maintenance phase tasks
        maintenance_tasks = [
            AutomationTask(
                task_id="log_cleanup",
                name="Log File Cleanup",
                phase=AutomationPhase.MAINTENANCE,
                command="find logs/ -name '*.log' -mtime +30 -delete",
                description="Clean up old log files",
                priority=TaskPriority.LOW,
                dependencies=[],
                timeout_seconds=120,
                retry_count=1,
                retry_delay=60,
                environment={},
                schedule="0 2 * * 0"  # Sunday 2 AM
            ),
            AutomationTask(
                task_id="cache_cleanup",
                name="Cache Cleanup",
                phase=AutomationPhase.MAINTENANCE,
                command="make clean",
                description="Clean up build artifacts and caches",
                priority=TaskPriority.LOW,
                dependencies=[],
                timeout_seconds=300,
                retry_count=1,
                retry_delay=60,
                environment={},
                schedule="0 3 * * 0"  # Sunday 3 AM
            ),
            AutomationTask(
                task_id="backup_validation",
                name="Backup Validation",
                phase=AutomationPhase.MAINTENANCE,
                command="python scripts/validate_backups.py",
                description="Validate system backups",
                priority=TaskPriority.HIGH,
                dependencies=[],
                timeout_seconds=600,
                retry_count=2,
                retry_delay=300,
                environment={},
                schedule="0 1 * * *"  # Daily 1 AM
            )
        ]
        
        # Register all tasks
        all_tasks = dev_tasks + test_tasks + security_tasks + build_tasks + monitoring_tasks + maintenance_tasks
        for task in all_tasks:
            if task.phase.value in self.config["phases_enabled"]:
                self.tasks[task.task_id] = task
    
    def _create_execution_id(self) -> str:
        """Create unique execution ID."""
        import uuid
        return str(uuid.uuid4())[:8]
    
    async def _execute_task(self, task: AutomationTask) -> TaskExecution:
        """Execute a single automation task."""
        execution_id = self._create_execution_id()
        execution = TaskExecution(
            task_id=task.task_id,
            execution_id=execution_id,
            status=TaskStatus.PENDING
        )
        
        self.executions[execution_id] = execution
        self.logger.info(f"Starting task execution: {task.name} ({execution_id})")
        
        try:
            execution.status = TaskStatus.RUNNING
            execution.start_time = datetime.now(timezone.utc)
            
            # Prepare environment
            env = {**self.config.get("environment_variables", {}), **task.environment}
            working_dir = task.working_directory or self.config["working_directory"]
            
            # Execute command with timeout
            process = await asyncio.create_subprocess_shell(
                task.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=working_dir
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=task.timeout_seconds
                )
                
                execution.exit_code = process.returncode
                execution.stdout = stdout.decode('utf-8', errors='replace')
                execution.stderr = stderr.decode('utf-8', errors='replace')
                
                if process.returncode == 0:
                    execution.status = TaskStatus.COMPLETED
                    self.logger.info(f"Task completed successfully: {task.name} ({execution_id})")
                else:
                    execution.status = TaskStatus.FAILED
                    execution.error_message = f"Command failed with exit code {process.returncode}"
                    self.logger.error(f"Task failed: {task.name} ({execution_id}) - {execution.error_message}")
                
            except asyncio.TimeoutError:
                process.kill()
                execution.status = TaskStatus.FAILED
                execution.error_message = f"Task timed out after {task.timeout_seconds} seconds"
                self.logger.error(f"Task timed out: {task.name} ({execution_id})")
                
        except Exception as e:
            execution.status = TaskStatus.FAILED
            execution.error_message = str(e)
            self.logger.error(f"Task execution error: {task.name} ({execution_id}) - {e}")
        
        finally:
            execution.end_time = datetime.now(timezone.utc)
        
        return execution
    
    async def _execute_task_with_retry(self, task: AutomationTask) -> TaskExecution:
        """Execute task with retry logic."""
        last_execution = None
        
        for attempt in range(task.retry_count + 1):
            if self.shutdown_requested:
                break
                
            execution = await self._execute_task(task)
            execution.retry_attempt = attempt
            last_execution = execution
            
            if execution.status == TaskStatus.COMPLETED:
                break
            
            if attempt < task.retry_count:
                self.logger.info(f"Retrying task {task.name} in {task.retry_delay} seconds (attempt {attempt + 1}/{task.retry_count})")
                await asyncio.sleep(task.retry_delay)
        
        return last_execution
    
    def _resolve_dependencies(self, task_ids: List[str]) -> List[str]:
        """Resolve task dependencies to get execution order."""
        resolved = []
        visiting = set()
        visited = set()
        
        def visit(task_id: str):
            if task_id in visiting:
                raise ValueError(f"Circular dependency detected involving task: {task_id}")
            if task_id in visited:
                return
            
            visiting.add(task_id)
            
            if task_id in self.tasks:
                for dep in self.tasks[task_id].dependencies:
                    if dep in task_ids:  # Only consider dependencies in the current set
                        visit(dep)
            
            visiting.remove(task_id)
            visited.add(task_id)
            resolved.append(task_id)
        
        for task_id in task_ids:
            if task_id not in visited:
                visit(task_id)
        
        return resolved
    
    async def execute_phase(self, phase: AutomationPhase) -> Dict[str, TaskExecution]:
        """Execute all tasks in a specific phase."""
        phase_tasks = [task_id for task_id, task in self.tasks.items() 
                      if task.phase == phase and task.enabled]
        
        if not phase_tasks:
            self.logger.info(f"No tasks found for phase: {phase.value}")
            return {}
        
        self.logger.info(f"Starting phase execution: {phase.value} ({len(phase_tasks)} tasks)")
        
        try:
            # Resolve dependencies
            execution_order = self._resolve_dependencies(phase_tasks)
            executions = {}
            
            for task_id in execution_order:
                if self.shutdown_requested:
                    break
                    
                task = self.tasks[task_id]
                
                # Check if dependencies completed successfully
                dependency_failed = False
                for dep_id in task.dependencies:
                    if dep_id in executions and executions[dep_id].status != TaskStatus.COMPLETED:
                        dependency_failed = True
                        break
                
                if dependency_failed:
                    execution = TaskExecution(
                        task_id=task_id,
                        execution_id=self._create_execution_id(),
                        status=TaskStatus.SKIPPED,
                        start_time=datetime.now(timezone.utc),
                        end_time=datetime.now(timezone.utc),
                        error_message="Dependency failed"
                    )
                    self.logger.warning(f"Skipping task {task.name} due to failed dependency")
                else:
                    self.running_tasks.add(task_id)
                    execution = await self._execute_task_with_retry(task)
                    self.running_tasks.discard(task_id)
                
                executions[task_id] = execution
                self.executions[execution.execution_id] = execution
            
            self.logger.info(f"Phase execution completed: {phase.value}")
            return executions
            
        except Exception as e:
            self.logger.error(f"Phase execution failed: {phase.value} - {e}")
            raise
    
    async def execute_workflow(self, phases: Optional[List[AutomationPhase]] = None) -> Dict[str, Any]:
        """Execute complete automation workflow."""
        if phases is None:
            phases = [AutomationPhase(phase) for phase in self.config["phases_enabled"]]
        
        self.logger.info(f"Starting workflow execution with phases: {[p.value for p in phases]}")
        
        workflow_start = datetime.now(timezone.utc)
        workflow_results = {
            "workflow_id": self._create_execution_id(),
            "start_time": workflow_start.isoformat(),
            "phases": {},
            "overall_status": "running",
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "skipped_tasks": 0
        }
        
        try:
            for phase in phases:
                if self.shutdown_requested:
                    break
                    
                self.logger.info(f"Executing phase: {phase.value}")
                phase_executions = await self.execute_phase(phase)
                
                workflow_results["phases"][phase.value] = {
                    "executions": {k: asdict(v) for k, v in phase_executions.items()},
                    "total_tasks": len(phase_executions),
                    "successful_tasks": sum(1 for e in phase_executions.values() if e.status == TaskStatus.COMPLETED),
                    "failed_tasks": sum(1 for e in phase_executions.values() if e.status == TaskStatus.FAILED),
                    "skipped_tasks": sum(1 for e in phase_executions.values() if e.status == TaskStatus.SKIPPED)
                }
                
                # Update overall statistics
                workflow_results["total_tasks"] += len(phase_executions)
                workflow_results["successful_tasks"] += workflow_results["phases"][phase.value]["successful_tasks"]
                workflow_results["failed_tasks"] += workflow_results["phases"][phase.value]["failed_tasks"]
                workflow_results["skipped_tasks"] += workflow_results["phases"][phase.value]["skipped_tasks"]
                
                # Check if critical tasks failed
                critical_failed = any(
                    e.status == TaskStatus.FAILED and self.tasks[task_id].priority == TaskPriority.CRITICAL
                    for task_id, e in phase_executions.items()
                )
                
                if critical_failed:
                    self.logger.error(f"Critical task failed in phase {phase.value}. Stopping workflow.")
                    workflow_results["overall_status"] = "failed"
                    break
            
            # Determine overall status
            if workflow_results["overall_status"] != "failed":
                if workflow_results["failed_tasks"] == 0:
                    workflow_results["overall_status"] = "completed"
                else:
                    workflow_results["overall_status"] = "completed_with_failures"
            
            workflow_end = datetime.now(timezone.utc)
            workflow_results["end_time"] = workflow_end.isoformat()
            workflow_results["duration_seconds"] = (workflow_end - workflow_start).total_seconds()
            
            self.logger.info(f"Workflow execution completed: {workflow_results['overall_status']}")
            
            # Store workflow results
            await self._store_workflow_results(workflow_results)
            
            return workflow_results
            
        except Exception as e:
            workflow_results["overall_status"] = "error"
            workflow_results["error"] = str(e)
            workflow_results["end_time"] = datetime.now(timezone.utc).isoformat()
            self.logger.error(f"Workflow execution error: {e}")
            raise
    
    async def _store_workflow_results(self, results: Dict[str, Any]):
        """Store workflow execution results."""
        try:
            results_dir = Path("automation_results")
            results_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            results_file = results_dir / f"workflow_{results['workflow_id']}_{timestamp}.json"
            
            async with aiofiles.open(results_file, "w") as f:
                await f.write(json.dumps(results, indent=2))
            
            self.logger.info(f"Workflow results stored: {results_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to store workflow results: {e}")
    
    async def execute_task_by_id(self, task_id: str) -> TaskExecution:
        """Execute a specific task by ID."""
        if task_id not in self.tasks:
            raise ValueError(f"Task not found: {task_id}")
        
        task = self.tasks[task_id]
        if not task.enabled:
            raise ValueError(f"Task is disabled: {task_id}")
        
        self.logger.info(f"Executing individual task: {task.name}")
        return await self._execute_task_with_retry(task)
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get current status of a task."""
        if task_id in self.running_tasks:
            return TaskStatus.RUNNING
        
        # Find most recent execution
        recent_execution = None
        for execution in self.executions.values():
            if execution.task_id == task_id:
                if recent_execution is None or execution.start_time > recent_execution.start_time:
                    recent_execution = execution
        
        return recent_execution.status if recent_execution else None
    
    def get_workflow_summary(self) -> Dict[str, Any]:
        """Get summary of all workflow executions."""
        summary = {
            "total_tasks": len(self.tasks),
            "enabled_tasks": sum(1 for task in self.tasks.values() if task.enabled),
            "phases": {},
            "recent_executions": [],
            "running_tasks": list(self.running_tasks)
        }
        
        # Summarize by phase
        for phase in AutomationPhase:
            phase_tasks = [task for task in self.tasks.values() if task.phase == phase]
            summary["phases"][phase.value] = {
                "total_tasks": len(phase_tasks),
                "enabled_tasks": sum(1 for task in phase_tasks if task.enabled)
            }
        
        # Recent executions
        recent_executions = sorted(
            self.executions.values(),
            key=lambda e: e.start_time or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True
        )[:10]
        
        summary["recent_executions"] = [asdict(execution) for execution in recent_executions]
        
        return summary
    
    async def shutdown(self):
        """Graceful shutdown of the orchestrator."""
        self.logger.info("Shutting down SDLC Automation Orchestrator...")
        self.shutdown_requested = True
        
        # Wait for running tasks to complete (with timeout)
        if self.running_tasks:
            self.logger.info(f"Waiting for {len(self.running_tasks)} running tasks to complete...")
            timeout = 60  # 1 minute timeout
            start_time = datetime.now()
            
            while self.running_tasks and (datetime.now() - start_time).seconds < timeout:
                await asyncio.sleep(1)
        
        self.executor.shutdown(wait=True)
        self.logger.info("Shutdown completed")


async def main():
    """Main entry point for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SDLC Automation Orchestrator")
    parser.add_argument("--config", type=Path, help="Configuration file path")
    parser.add_argument("--phase", choices=[p.value for p in AutomationPhase], 
                       help="Execute specific phase")
    parser.add_argument("--task", help="Execute specific task by ID")
    parser.add_argument("--workflow", action="store_true", help="Execute complete workflow")
    parser.add_argument("--status", action="store_true", help="Show workflow status")
    parser.add_argument("--list-tasks", action="store_true", help="List all tasks")
    
    args = parser.parse_args()
    
    orchestrator = SDLCAutomationOrchestrator(args.config)
    
    try:
        if args.list_tasks:
            print("Available Tasks:")
            for task_id, task in orchestrator.tasks.items():
                status = "enabled" if task.enabled else "disabled"
                print(f"  {task_id} - {task.name} ({task.phase.value}) [{status}]")
        
        elif args.status:
            summary = orchestrator.get_workflow_summary()
            print(json.dumps(summary, indent=2))
        
        elif args.task:
            execution = await orchestrator.execute_task_by_id(args.task)
            print(json.dumps(asdict(execution), indent=2, default=str))
        
        elif args.phase:
            phase = AutomationPhase(args.phase)
            executions = await orchestrator.execute_phase(phase)
            print(json.dumps({k: asdict(v) for k, v in executions.items()}, indent=2, default=str))
        
        elif args.workflow:
            results = await orchestrator.execute_workflow()
            print(json.dumps(results, indent=2, default=str))
        
        else:
            parser.print_help()
    
    finally:
        await orchestrator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())