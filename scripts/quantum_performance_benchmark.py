#!/usr/bin/env python3
"""
Quantum Task Planner Performance Benchmark

Comprehensive performance testing for the quantum task planning system:
- Task creation and management benchmarks
- Quantum state measurement performance
- Scheduling optimization benchmarks
- Concurrency and scalability tests
- Memory usage and resource efficiency
- End-to-end workflow performance
"""

import os
import sys
import json
import time
import asyncio
import statistics
import psutil
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipeline.quantum.quantum_planner import (
    QuantumTask, QuantumTaskPlanner, QuantumState, QuantumPriority
)
from pipeline.quantum.quantum_scheduler import QuantumScheduler
from pipeline.quantum.quantum_dependencies import DependencyGraph
from pipeline.quantum.performance import QuantumOptimizedTaskPlanner

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    test_name: str
    duration_seconds: float
    operations_per_second: float
    memory_usage_mb: float
    cpu_usage_percent: float
    success_rate: float
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_name": self.test_name,
            "duration_seconds": self.duration_seconds,
            "operations_per_second": self.operations_per_second,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "success_rate": self.success_rate,
            "additional_metrics": self.additional_metrics
        }


@dataclass 
class BenchmarkSuite:
    """Complete benchmark suite results."""
    suite_name: str
    start_time: datetime
    end_time: datetime
    total_duration: float
    results: List[BenchmarkResult] = field(default_factory=list)
    system_info: Dict[str, Any] = field(default_factory=dict)
    
    def add_result(self, result: BenchmarkResult):
        """Add a benchmark result."""
        self.results.append(result)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get benchmark summary statistics."""
        if not self.results:
            return {}
        
        durations = [r.duration_seconds for r in self.results]
        ops_per_sec = [r.operations_per_second for r in self.results]
        memory_usage = [r.memory_usage_mb for r in self.results]
        success_rates = [r.success_rate for r in self.results]
        
        return {
            "total_tests": len(self.results),
            "total_duration": self.total_duration,
            "average_duration": statistics.mean(durations),
            "median_duration": statistics.median(durations),
            "max_duration": max(durations),
            "min_duration": min(durations),
            "average_ops_per_second": statistics.mean(ops_per_sec),
            "peak_ops_per_second": max(ops_per_sec),
            "average_memory_mb": statistics.mean(memory_usage),
            "peak_memory_mb": max(memory_usage),
            "overall_success_rate": statistics.mean(success_rates),
            "failed_tests": sum(1 for r in self.results if r.success_rate < 1.0)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "suite_name": self.suite_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "total_duration": self.total_duration,
            "system_info": self.system_info,
            "summary": self.get_summary(),
            "results": [r.to_dict() for r in self.results]
        }


class PerformanceMonitor:
    """Monitor system performance during benchmarks."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.monitoring = False
        self.samples = []
        self._monitor_task = None
    
    async def start_monitoring(self, interval: float = 0.1):
        """Start performance monitoring."""
        self.monitoring = True
        self.samples = []
        self._monitor_task = asyncio.create_task(self._monitor_loop(interval))
    
    async def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return statistics."""
        self.monitoring = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        if not self.samples:
            return {"memory_mb": 0, "cpu_percent": 0}
        
        memory_samples = [s["memory_mb"] for s in self.samples]
        cpu_samples = [s["cpu_percent"] for s in self.samples]
        
        return {
            "memory_mb": statistics.mean(memory_samples),
            "peak_memory_mb": max(memory_samples),
            "cpu_percent": statistics.mean(cpu_samples),
            "peak_cpu_percent": max(cpu_samples)
        }
    
    async def _monitor_loop(self, interval: float):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                memory_info = self.process.memory_info()
                cpu_percent = self.process.cpu_percent()
                
                self.samples.append({
                    "timestamp": time.time(),
                    "memory_mb": memory_info.rss / 1024 / 1024,
                    "cpu_percent": cpu_percent
                })
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Monitoring error: {e}")
                await asyncio.sleep(interval)


class QuantumPerformanceBenchmark:
    """Main performance benchmark suite for quantum task planner."""
    
    def __init__(self):
        self.monitor = PerformanceMonitor()
        self.suite = BenchmarkSuite(
            suite_name="Quantum Task Planner Performance Benchmark",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            total_duration=0.0
        )
        
        # Collect system information
        self.suite.system_info = {
            "cpu_count": psutil.cpu_count(),
            "cpu_freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            "memory_total_gb": psutil.virtual_memory().total / 1024**3,
            "python_version": sys.version,
            "platform": sys.platform
        }
    
    async def run_all_benchmarks(self) -> BenchmarkSuite:
        """Run all performance benchmarks."""
        logger.info("Starting quantum task planner performance benchmarks")
        
        # Task management benchmarks
        await self.benchmark_task_creation()
        await self.benchmark_task_measurement()
        await self.benchmark_quantum_evolution()
        
        # Scheduling benchmarks
        await self.benchmark_scheduling_optimization()
        await self.benchmark_concurrent_scheduling()
        
        # Scalability benchmarks
        await self.benchmark_large_task_sets()
        await self.benchmark_memory_efficiency()
        
        # End-to-end benchmarks
        await self.benchmark_full_workflow()
        await self.benchmark_optimized_planner()
        
        # Dependency management benchmarks
        await self.benchmark_dependency_resolution()
        
        self.suite.end_time = datetime.utcnow()
        self.suite.total_duration = (self.suite.end_time - self.suite.start_time).total_seconds()
        
        logger.info(f"Benchmark suite completed in {self.suite.total_duration:.2f} seconds")
        return self.suite
    
    async def benchmark_task_creation(self):
        """Benchmark quantum task creation performance."""
        logger.info("Benchmarking task creation")
        
        await self.monitor.start_monitoring()
        
        num_tasks = 1000
        start_time = time.time()
        
        tasks = []
        for i in range(num_tasks):
            task = QuantumTask(
                title=f"Benchmark Task {i}",
                description=f"Performance test task number {i}",
                priority=np.random.choice(list(QuantumPriority)),
                estimated_duration=timedelta(hours=np.random.uniform(0.5, 4.0))
            )
            tasks.append(task)
        
        end_time = time.time()
        duration = end_time - start_time
        
        perf_stats = await self.monitor.stop_monitoring()
        
        # Verify tasks were created correctly
        success_rate = sum(1 for task in tasks if task.title.startswith("Benchmark Task")) / num_tasks
        
        result = BenchmarkResult(
            test_name="task_creation",
            duration_seconds=duration,
            operations_per_second=num_tasks / duration,
            memory_usage_mb=perf_stats["memory_mb"],
            cpu_usage_percent=perf_stats["cpu_percent"],
            success_rate=success_rate,
            additional_metrics={
                "tasks_created": len(tasks),
                "average_task_creation_time_ms": (duration / num_tasks) * 1000
            }
        )
        
        self.suite.add_result(result)
        logger.info(f"Task creation: {result.operations_per_second:.2f} tasks/sec")
    
    async def benchmark_task_measurement(self):
        """Benchmark quantum measurement performance."""
        logger.info("Benchmarking quantum measurement")
        
        # Create tasks for measurement
        tasks = [
            QuantumTask(title=f"Measurement Task {i}")
            for i in range(500)
        ]
        
        await self.monitor.start_monitoring()
        
        start_time = time.time()
        
        measurements = []
        for task in tasks:
            measured_state = task.measure()
            measurements.append(measured_state)
        
        end_time = time.time()
        duration = end_time - start_time
        
        perf_stats = await self.monitor.stop_monitoring()
        
        # Verify measurements
        success_rate = sum(1 for m in measurements if isinstance(m, QuantumState)) / len(tasks)
        
        result = BenchmarkResult(
            test_name="quantum_measurement",
            duration_seconds=duration,
            operations_per_second=len(tasks) / duration,
            memory_usage_mb=perf_stats["memory_mb"],
            cpu_usage_percent=perf_stats["cpu_percent"],
            success_rate=success_rate,
            additional_metrics={
                "measurements_performed": len(measurements),
                "average_measurement_time_ms": (duration / len(tasks)) * 1000,
                "state_distribution": {state.value: measurements.count(state) for state in QuantumState}
            }
        )
        
        self.suite.add_result(result)
        logger.info(f"Quantum measurement: {result.operations_per_second:.2f} measurements/sec")
    
    async def benchmark_quantum_evolution(self):
        """Benchmark quantum state evolution performance."""
        logger.info("Benchmarking quantum evolution")
        
        # Create tasks for evolution
        tasks = [
            QuantumTask(title=f"Evolution Task {i}")
            for i in range(200)
        ]
        
        await self.monitor.start_monitoring()
        
        start_time = time.time()
        
        # Perform multiple evolution steps
        evolution_steps = 10
        for step in range(evolution_steps):
            for task in tasks:
                task.evolve_state(time_delta=1.0)
        
        end_time = time.time()
        duration = end_time - start_time
        
        perf_stats = await self.monitor.stop_monitoring()
        
        total_operations = len(tasks) * evolution_steps
        
        result = BenchmarkResult(
            test_name="quantum_evolution",
            duration_seconds=duration,
            operations_per_second=total_operations / duration,
            memory_usage_mb=perf_stats["memory_mb"],
            cpu_usage_percent=perf_stats["cpu_percent"],
            success_rate=1.0,  # Evolution should always succeed
            additional_metrics={
                "evolution_steps": evolution_steps,
                "tasks_evolved": len(tasks),
                "total_evolution_operations": total_operations
            }
        )
        
        self.suite.add_result(result)
        logger.info(f"Quantum evolution: {result.operations_per_second:.2f} evolutions/sec")
    
    async def benchmark_scheduling_optimization(self):
        """Benchmark scheduling optimization performance."""
        logger.info("Benchmarking scheduling optimization")
        
        planner = QuantumTaskPlanner(max_parallel_tasks=10)
        
        # Create tasks with different priorities and constraints
        tasks = []
        for i in range(100):
            task = QuantumTask(
                title=f"Schedule Task {i}",
                priority=np.random.choice(list(QuantumPriority)),
                estimated_duration=timedelta(hours=np.random.uniform(0.5, 3.0)),
                due_date=datetime.utcnow() + timedelta(days=np.random.uniform(1, 30))
            )
            tasks.append(task)
            await planner.add_task(task)
        
        await self.monitor.start_monitoring()
        
        start_time = time.time()
        
        # Run multiple optimization cycles
        optimization_cycles = 20
        schedules = []
        
        for cycle in range(optimization_cycles):
            schedule = await planner.optimize_schedule()
            schedules.append(schedule)
        
        end_time = time.time()
        duration = end_time - start_time
        
        perf_stats = await self.monitor.stop_monitoring()
        
        # Calculate optimization quality metrics
        avg_schedule_length = statistics.mean(len(s) for s in schedules)
        
        result = BenchmarkResult(
            test_name="scheduling_optimization",
            duration_seconds=duration,
            operations_per_second=optimization_cycles / duration,
            memory_usage_mb=perf_stats["memory_mb"],
            cpu_usage_percent=perf_stats["cpu_percent"],
            success_rate=sum(1 for s in schedules if len(s) > 0) / optimization_cycles,
            additional_metrics={
                "optimization_cycles": optimization_cycles,
                "total_tasks": len(tasks),
                "average_schedule_length": avg_schedule_length,
                "max_parallel_tasks": planner.max_parallel_tasks
            }
        )
        
        self.suite.add_result(result)
        logger.info(f"Scheduling optimization: {result.operations_per_second:.2f} optimizations/sec")
    
    async def benchmark_concurrent_scheduling(self):
        """Benchmark concurrent scheduling performance."""
        logger.info("Benchmarking concurrent scheduling")
        
        scheduler = QuantumScheduler(max_concurrent_tasks=5)
        
        # Create tasks for concurrent execution
        tasks = [
            QuantumTask(
                title=f"Concurrent Task {i}",
                estimated_duration=timedelta(seconds=0.1)  # Short duration for testing
            )
            for i in range(50)
        ]
        
        await self.monitor.start_monitoring()
        
        start_time = time.time()
        
        # Run concurrent scheduling
        result_data = await scheduler.schedule_and_execute(tasks)
        
        end_time = time.time()
        duration = end_time - start_time
        
        perf_stats = await self.monitor.stop_monitoring()
        
        success_rate = result_data["completed"] / result_data["scheduled"] if result_data["scheduled"] > 0 else 0
        
        result = BenchmarkResult(
            test_name="concurrent_scheduling",
            duration_seconds=duration,
            operations_per_second=result_data["scheduled"] / duration,
            memory_usage_mb=perf_stats["memory_mb"],
            cpu_usage_percent=perf_stats["cpu_percent"],
            success_rate=success_rate,
            additional_metrics={
                "scheduled_tasks": result_data["scheduled"],
                "completed_tasks": result_data["completed"],
                "failed_tasks": result_data["failed"],
                "max_concurrent": scheduler.max_concurrent_tasks
            }
        )
        
        self.suite.add_result(result)
        logger.info(f"Concurrent scheduling: {result.operations_per_second:.2f} tasks/sec")
    
    async def benchmark_large_task_sets(self):
        """Benchmark performance with large task sets."""
        logger.info("Benchmarking large task sets")
        
        planner = QuantumTaskPlanner(max_parallel_tasks=20)
        
        # Create large task set
        large_task_count = 2000
        tasks = []
        
        await self.monitor.start_monitoring()
        
        start_time = time.time()
        
        # Create and add tasks
        for i in range(large_task_count):
            task = QuantumTask(
                title=f"Large Set Task {i}",
                priority=np.random.choice(list(QuantumPriority)),
                estimated_duration=timedelta(hours=np.random.uniform(0.1, 2.0))
            )
            tasks.append(task)
            await planner.add_task(task)
        
        # Perform system operations
        await planner.quantum_evolve(1.0)
        measurements = await planner.measure_system()
        schedule = await planner.optimize_schedule()
        system_stats = await planner.get_system_stats()
        
        end_time = time.time()
        duration = end_time - start_time
        
        perf_stats = await self.monitor.stop_monitoring()
        
        result = BenchmarkResult(
            test_name="large_task_sets",
            duration_seconds=duration,
            operations_per_second=large_task_count / duration,
            memory_usage_mb=perf_stats["memory_mb"],
            cpu_usage_percent=perf_stats["cpu_percent"],
            success_rate=1.0 if len(measurements) == large_task_count else 0.0,
            additional_metrics={
                "task_count": large_task_count,
                "measurements_count": len(measurements),
                "schedule_length": len(schedule),
                "system_coherence": system_stats["system_coherence"],
                "memory_per_task_kb": (perf_stats["memory_mb"] * 1024) / large_task_count
            }
        )
        
        self.suite.add_result(result)
        logger.info(f"Large task sets: {result.operations_per_second:.2f} tasks/sec")
    
    async def benchmark_memory_efficiency(self):
        """Benchmark memory usage efficiency."""
        logger.info("Benchmarking memory efficiency")
        
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        planner = QuantumTaskPlanner()
        tasks = []
        
        # Gradually add tasks and measure memory growth
        memory_samples = []
        task_counts = []
        
        for batch_size in [100, 500, 1000, 1500, 2000]:
            # Add batch of tasks
            for i in range(len(tasks), batch_size):
                task = QuantumTask(
                    title=f"Memory Test Task {i}",
                    description="A" * 100,  # Some content to consume memory
                    estimated_duration=timedelta(hours=1)
                )
                tasks.append(task)
                await planner.add_task(task)
            
            # Measure memory
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_samples.append(current_memory - initial_memory)
            task_counts.append(len(tasks))
        
        # Calculate memory efficiency
        memory_per_task = []
        for i in range(1, len(memory_samples)):
            if task_counts[i] > task_counts[i-1]:
                memory_diff = memory_samples[i] - memory_samples[i-1]
                task_diff = task_counts[i] - task_counts[i-1]
                memory_per_task.append(memory_diff / task_diff)
        
        avg_memory_per_task = statistics.mean(memory_per_task) if memory_per_task else 0
        
        result = BenchmarkResult(
            test_name="memory_efficiency",
            duration_seconds=0.0,  # Not time-based
            operations_per_second=0.0,  # Not applicable
            memory_usage_mb=memory_samples[-1],
            cpu_usage_percent=0.0,  # Not measured for this test
            success_rate=1.0,
            additional_metrics={
                "total_tasks": len(tasks),
                "total_memory_mb": memory_samples[-1],
                "memory_per_task_kb": avg_memory_per_task * 1024,
                "memory_growth_samples": list(zip(task_counts, memory_samples))
            }
        )
        
        self.suite.add_result(result)
        logger.info(f"Memory efficiency: {avg_memory_per_task*1024:.2f} KB per task")
    
    async def benchmark_full_workflow(self):
        """Benchmark complete end-to-end workflow."""
        logger.info("Benchmarking full workflow")
        
        await self.monitor.start_monitoring()
        
        start_time = time.time()
        
        # Full workflow simulation
        planner = QuantumTaskPlanner(max_parallel_tasks=8)
        scheduler = QuantumScheduler(max_concurrent_tasks=8)
        
        # Create workflow tasks
        workflow_tasks = []
        for i in range(100):
            task = QuantumTask(
                title=f"Workflow Task {i}",
                priority=np.random.choice(list(QuantumPriority)),
                estimated_duration=timedelta(seconds=0.05),  # Short for testing
            )
            workflow_tasks.append(task)
            await planner.add_task(task)
        
        # Execute full workflow
        await planner.quantum_evolve(1.0)
        measurements = await planner.measure_system()
        schedule = await planner.optimize_schedule()
        
        # Execute scheduled tasks
        execution_result = await scheduler.schedule_and_execute(schedule)
        
        # Final system analysis
        final_stats = await planner.get_system_stats()
        coherence = await planner.get_system_coherence()
        
        end_time = time.time()
        duration = end_time - start_time
        
        perf_stats = await self.monitor.stop_monitoring()
        
        success_rate = execution_result["completed"] / len(workflow_tasks)
        
        result = BenchmarkResult(
            test_name="full_workflow",
            duration_seconds=duration,
            operations_per_second=len(workflow_tasks) / duration,
            memory_usage_mb=perf_stats["memory_mb"],
            cpu_usage_percent=perf_stats["cpu_percent"],
            success_rate=success_rate,
            additional_metrics={
                "workflow_tasks": len(workflow_tasks),
                "measurements": len(measurements),
                "scheduled_tasks": len(schedule),
                "completed_tasks": execution_result["completed"],
                "failed_tasks": execution_result["failed"],
                "final_coherence": coherence,
                "end_to_end_throughput": len(workflow_tasks) / duration
            }
        )
        
        self.suite.add_result(result)
        logger.info(f"Full workflow: {result.operations_per_second:.2f} tasks/sec end-to-end")
    
    async def benchmark_optimized_planner(self):
        """Benchmark the optimized quantum planner with caching."""
        logger.info("Benchmarking optimized planner")
        
        config = {
            "cache_size_mb": 50,
            "max_workers": 4,
            "enable_caching": True,
            "enable_adaptive_resources": True
        }
        
        optimized_planner = QuantumOptimizedTaskPlanner(config)
        
        # Create tasks for optimization testing
        tasks = [
            QuantumTask(
                title=f"Optimized Task {i}",
                priority=np.random.choice(list(QuantumPriority)),
                estimated_duration=timedelta(seconds=0.01)
            )
            for i in range(200)
        ]
        
        await self.monitor.start_monitoring()
        
        start_time = time.time()
        
        # Test optimized scheduling (should use caching)
        results = []
        for i in range(10):  # Run multiple times to test caching
            result = await optimized_planner.optimized_schedule_tasks(tasks)
            results.append(result)
        
        # Get optimization statistics
        opt_stats = await optimized_planner.get_optimization_statistics()
        
        end_time = time.time()
        duration = end_time - start_time
        
        perf_stats = await self.monitor.stop_monitoring()
        
        # Cleanup
        await optimized_planner.shutdown()
        
        cache_hit_rate = opt_stats["cache_stats"].get("hit_rate", 0)
        
        result = BenchmarkResult(
            test_name="optimized_planner",
            duration_seconds=duration,
            operations_per_second=(len(tasks) * len(results)) / duration,
            memory_usage_mb=perf_stats["memory_mb"],
            cpu_usage_percent=perf_stats["cpu_percent"],
            success_rate=sum(1 for r in results if r.get("optimization_applied")) / len(results),
            additional_metrics={
                "scheduling_runs": len(results),
                "tasks_per_run": len(tasks),
                "cache_hit_rate": cache_hit_rate,
                "cache_stats": opt_stats["cache_stats"],
                "parallel_processing_stats": opt_stats["parallel_processing_stats"]
            }
        )
        
        self.suite.add_result(result)
        logger.info(f"Optimized planner: {result.operations_per_second:.2f} tasks/sec (cache hit rate: {cache_hit_rate:.2%})")
    
    async def benchmark_dependency_resolution(self):
        """Benchmark dependency resolution performance."""
        logger.info("Benchmarking dependency resolution")
        
        dependency_graph = DependencyGraph()
        
        # Create tasks with complex dependency chains
        tasks = []
        for i in range(100):
            task = QuantumTask(title=f"Dependency Task {i}")
            tasks.append(task)
            await dependency_graph.register_task(task)
        
        # Create dependency relationships
        for i in range(1, len(tasks)):
            # Each task depends on 1-3 previous tasks
            num_deps = min(i, np.random.randint(1, 4))
            for j in range(num_deps):
                dep_idx = np.random.randint(0, i)
                await dependency_graph.add_dependency(tasks[i].id, tasks[dep_idx].id)
        
        await self.monitor.start_monitoring()
        
        start_time = time.time()
        
        # Resolve dependencies for all tasks
        resolution_results = []
        for task in tasks:
            dependencies = await dependency_graph.get_dependency_chain(task.id)
            resolution_results.append(dependencies)
        
        # Get ready tasks
        ready_tasks = await dependency_graph.get_ready_tasks()
        
        # Analyze system
        system_analysis = await dependency_graph.analyze_system()
        
        end_time = time.time()
        duration = end_time - start_time
        
        perf_stats = await self.monitor.stop_monitoring()
        
        avg_dependencies = statistics.mean(len(deps) for deps in resolution_results)
        
        result = BenchmarkResult(
            test_name="dependency_resolution",
            duration_seconds=duration,
            operations_per_second=len(tasks) / duration,
            memory_usage_mb=perf_stats["memory_mb"],
            cpu_usage_percent=perf_stats["cpu_percent"],
            success_rate=1.0,  # All resolutions should succeed
            additional_metrics={
                "total_tasks": len(tasks),
                "dependency_resolutions": len(resolution_results),
                "average_dependencies_per_task": avg_dependencies,
                "ready_tasks": len(ready_tasks),
                "system_analysis": system_analysis
            }
        )
        
        self.suite.add_result(result)
        logger.info(f"Dependency resolution: {result.operations_per_second:.2f} resolutions/sec")


async def main():
    """Main function to run performance benchmarks."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quantum Task Planner Performance Benchmark")
    parser.add_argument("--output", "-o", help="Output file for benchmark results",
                       default="quantum_performance_benchmark.json")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run benchmarks
    benchmark = QuantumPerformanceBenchmark()
    
    logger.info("Starting quantum task planner performance benchmarks")
    suite = await benchmark.run_all_benchmarks()
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(suite.to_dict(), f, indent=2, default=str)
    
    # Print summary
    summary = suite.get_summary()
    
    print(f"\n{'='*60}")
    print("QUANTUM TASK PLANNER PERFORMANCE BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"Total Duration: {summary['total_duration']:.2f} seconds")
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Failed Tests: {summary['failed_tests']}")
    print(f"\nPerformance Summary:")
    print(f"  Peak Operations/Second: {summary['peak_ops_per_second']:.2f}")
    print(f"  Average Operations/Second: {summary['average_ops_per_second']:.2f}")
    print(f"  Peak Memory Usage: {summary['peak_memory_mb']:.2f} MB")
    print(f"  Average Memory Usage: {summary['average_memory_mb']:.2f} MB")
    print(f"  Overall Success Rate: {summary['overall_success_rate']:.2%}")
    
    print(f"\nTest Results:")
    for result in suite.results:
        status = "✅" if result.success_rate >= 1.0 else "⚠️"
        print(f"  {status} {result.test_name}: {result.operations_per_second:.2f} ops/sec, "
              f"{result.memory_usage_mb:.1f} MB, {result.success_rate:.1%} success")
    
    print(f"\nDetailed results saved to: {args.output}")
    
    # Performance scoring
    if summary["overall_success_rate"] < 0.95:
        print(f"\n⚠️  LOW SUCCESS RATE: {summary['overall_success_rate']:.1%}")
        sys.exit(1)
    elif summary["failed_tests"] > 0:
        print(f"\n⚠️  {summary['failed_tests']} TESTS FAILED")
        sys.exit(1)
    else:
        print(f"\n✅ All performance benchmarks passed")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())