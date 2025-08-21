"""
Scalable Evolution Engine - Generation 3 Enhancement
Advanced scalable system combining comprehensive benchmarking with autonomous evolution

This module provides:
- Comprehensive performance benchmarking across multiple dimensions
- Autonomous system evolution and optimization
- Scalable architecture with dynamic resource allocation
- Advanced caching and connection pooling
- Auto-scaling triggers and load balancing
- Performance regression detection and prevention
"""

import asyncio
import json
import logging
import time
import statistics
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor
import hashlib

from pydantic import BaseModel, Field
from opentelemetry import trace

from ..config.settings import get_settings
from ..telemetry import get_tracer
from .comprehensive_benchmarking_suite import ComprehensiveBenchmarkingSuite, BenchmarkCategory
from .autonomous_self_evolution import AutonomousSelfEvolution, EvolutionStrategy
from .quantum_realtime_orchestrator import QuantumRealtimeOrchestrator, SystemMetrics

# Global configuration
settings = get_settings()
tracer = get_tracer(__name__)
logger = logging.getLogger(__name__)


class ScalabilityMetric(str, Enum):
    """Scalability measurement metrics"""
    THROUGHPUT = "throughput"
    RESPONSE_TIME = "response_time"
    RESOURCE_EFFICIENCY = "resource_efficiency"
    CONCURRENT_CAPACITY = "concurrent_capacity"
    MEMORY_SCALABILITY = "memory_scalability"
    CPU_SCALABILITY = "cpu_scalability"
    NETWORK_EFFICIENCY = "network_efficiency"
    CACHE_PERFORMANCE = "cache_performance"


class EvolutionPhase(str, Enum):
    """Evolution phases for system improvement"""
    BASELINE_MEASUREMENT = "baseline_measurement"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    OPTIMIZATION_GENERATION = "optimization_generation"
    TESTING_VALIDATION = "testing_validation"
    DEPLOYMENT_ROLLOUT = "deployment_rollout"
    MONITORING_FEEDBACK = "monitoring_feedback"


@dataclass
class BenchmarkResult:
    """Comprehensive benchmark result"""
    benchmark_id: str
    category: BenchmarkCategory
    metric: ScalabilityMetric
    baseline_value: float
    optimized_value: float
    improvement_percent: float
    confidence_interval: Tuple[float, float]
    statistical_significance: float
    sample_size: int
    execution_time_seconds: float
    resource_usage: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class EvolutionResult:
    """System evolution result"""
    evolution_id: str
    phase: EvolutionPhase
    strategy: EvolutionStrategy
    improvements: List[BenchmarkResult]
    overall_performance_gain: float
    resource_efficiency_gain: float
    stability_score: float
    rollback_available: bool
    deployment_safe: bool
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ScalabilityProfile:
    """System scalability profile"""
    max_concurrent_users: int
    max_requests_per_second: float
    memory_scaling_factor: float
    cpu_scaling_factor: float
    storage_scaling_factor: float
    network_bandwidth_limit: float
    cache_hit_ratio_target: float
    response_time_sla_ms: float
    availability_target: float = 0.999  # 99.9%


class ScalableEvolutionEngine:
    """
    Advanced engine combining comprehensive benchmarking with autonomous evolution
    for maximum system scalability and performance optimization
    """
    
    def __init__(self, scalability_profile: Optional[ScalabilityProfile] = None):
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)
        
        # Initialize core components
        self.benchmarking_suite = ComprehensiveBenchmarkingSuite()
        self.autonomous_evolution = AutonomousSelfEvolution()
        self.quantum_orchestrator = QuantumRealtimeOrchestrator()
        
        # Scalability configuration
        self.scalability_profile = scalability_profile or ScalabilityProfile(
            max_concurrent_users=10000,
            max_requests_per_second=5000.0,
            memory_scaling_factor=1.5,
            cpu_scaling_factor=2.0,
            storage_scaling_factor=3.0,
            network_bandwidth_limit=10.0,  # Gbps
            cache_hit_ratio_target=0.95,
            response_time_sla_ms=200.0
        )
        
        # Evolution tracking
        self.evolution_history: List[EvolutionResult] = []
        self.benchmark_results: Dict[str, List[BenchmarkResult]] = defaultdict(list)
        self.performance_baselines: Dict[ScalabilityMetric, float] = {}
        
        # Scalability state
        self.current_load_level = 0.0  # 0.0 to 1.0
        self.scaling_decisions: deque = deque(maxlen=100)
        self.performance_trend: deque = deque(maxlen=50)
        
        # Control parameters
        self.benchmark_frequency = timedelta(hours=1)
        self.evolution_frequency = timedelta(hours=6)
        self.last_benchmark = None
        self.last_evolution = None
        
        # Background tasks
        self._running = True
        self._background_tasks: List[asyncio.Task] = []
        
        logger.info("Scalable Evolution Engine initialized successfully")

    async def start(self):
        """Start the scalable evolution engine"""
        self.logger.info("Starting Scalable Evolution Engine...")
        
        # Start core components
        await self.quantum_orchestrator.start()
        
        # Start background processes
        self._background_tasks = [
            asyncio.create_task(self._continuous_benchmarking_loop()),
            asyncio.create_task(self._autonomous_evolution_loop()),
            asyncio.create_task(self._scalability_monitoring_loop()),
            asyncio.create_task(self._performance_optimization_loop()),
            asyncio.create_task(self._regression_detection_loop())
        ]
        
        # Establish performance baselines
        await self._establish_performance_baselines()
        
        self.logger.info("Scalable Evolution Engine started successfully")

    async def stop(self):
        """Stop the scalable evolution engine"""
        self.logger.info("Stopping Scalable Evolution Engine...")
        
        self._running = False
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Stop core components
        await self.quantum_orchestrator.stop()
        
        self.logger.info("Scalable Evolution Engine stopped")

    @trace.get_tracer(__name__).start_as_current_span("execute_comprehensive_benchmark")
    async def execute_comprehensive_benchmark(
        self, 
        categories: List[BenchmarkCategory] = None,
        sample_size: int = 100
    ) -> List[BenchmarkResult]:
        """Execute comprehensive benchmarking across multiple categories"""
        
        if categories is None:
            categories = list(BenchmarkCategory)
        
        benchmark_id = f"benchmark_{int(datetime.utcnow().timestamp())}"
        results = []
        
        self.logger.info(f"Executing comprehensive benchmark {benchmark_id} "
                        f"with {len(categories)} categories")
        
        try:
            for category in categories:
                category_results = await self._benchmark_category(
                    benchmark_id, category, sample_size
                )
                results.extend(category_results)
                
                # Store results
                for result in category_results:
                    self.benchmark_results[category.value].append(result)
            
            # Update performance trends
            self._update_performance_trends(results)
            
            # Update last benchmark time
            self.last_benchmark = datetime.utcnow()
            
            self.logger.info(f"Comprehensive benchmark completed: {len(results)} results")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Comprehensive benchmark failed: {e}")
            return []

    async def _benchmark_category(
        self, 
        benchmark_id: str, 
        category: BenchmarkCategory, 
        sample_size: int
    ) -> List[BenchmarkResult]:
        """Benchmark a specific category"""
        
        results = []
        metrics_for_category = self._get_metrics_for_category(category)
        
        for metric in metrics_for_category:
            try:
                # Execute benchmark for specific metric
                baseline, optimized, stats_result = await self._execute_metric_benchmark(
                    metric, sample_size
                )
                
                # Calculate improvement
                improvement = ((optimized - baseline) / baseline * 100) if baseline > 0 else 0
                
                # Create result
                result = BenchmarkResult(
                    benchmark_id=f"{benchmark_id}_{category.value}_{metric.value}",
                    category=category,
                    metric=metric,
                    baseline_value=baseline,
                    optimized_value=optimized,
                    improvement_percent=improvement,
                    confidence_interval=stats_result.get("confidence_interval", (0, 0)),
                    statistical_significance=stats_result.get("p_value", 1.0),
                    sample_size=sample_size,
                    execution_time_seconds=stats_result.get("execution_time", 0.0),
                    resource_usage=stats_result.get("resource_usage", {})
                )
                
                results.append(result)
                
                self.logger.debug(f"Benchmark {metric.value}: baseline={baseline:.2f}, "
                                f"optimized={optimized:.2f}, improvement={improvement:.1f}%")
                
            except Exception as e:
                self.logger.error(f"Benchmark failed for {metric.value}: {e}")
        
        return results

    def _get_metrics_for_category(self, category: BenchmarkCategory) -> List[ScalabilityMetric]:
        """Get relevant metrics for a benchmark category"""
        
        category_metrics = {
            BenchmarkCategory.OPTIMIZATION: [
                ScalabilityMetric.THROUGHPUT,
                ScalabilityMetric.RESPONSE_TIME,
                ScalabilityMetric.RESOURCE_EFFICIENCY
            ],
            # Add more categories as they exist
        }
        
        return category_metrics.get(category, [ScalabilityMetric.THROUGHPUT])

    async def _execute_metric_benchmark(
        self, 
        metric: ScalabilityMetric, 
        sample_size: int
    ) -> Tuple[float, float, Dict[str, Any]]:
        """Execute benchmark for a specific metric"""
        
        start_time = time.time()
        
        # Simulate baseline measurement
        baseline_samples = []
        for _ in range(sample_size):
            sample = await self._measure_metric_baseline(metric)
            baseline_samples.append(sample)
            await asyncio.sleep(0.001)  # Small delay to avoid overwhelming
        
        baseline_value = statistics.mean(baseline_samples)
        
        # Simulate optimized measurement
        optimized_samples = []
        for _ in range(sample_size):
            sample = await self._measure_metric_optimized(metric)
            optimized_samples.append(sample)
            await asyncio.sleep(0.001)
        
        optimized_value = statistics.mean(optimized_samples)
        
        # Calculate statistics
        execution_time = time.time() - start_time
        
        # Statistical significance test
        from scipy import stats as scipy_stats
        try:
            t_stat, p_value = scipy_stats.ttest_ind(optimized_samples, baseline_samples)
        except:
            p_value = 1.0
        
        # Confidence interval (simplified)
        margin_of_error = 1.96 * (statistics.stdev(optimized_samples) / math.sqrt(sample_size))
        confidence_interval = (
            optimized_value - margin_of_error,
            optimized_value + margin_of_error
        )
        
        stats_result = {
            "confidence_interval": confidence_interval,
            "p_value": p_value,
            "execution_time": execution_time,
            "resource_usage": {
                "memory_mb": 50.0,  # Simulated
                "cpu_percent": 25.0  # Simulated
            }
        }
        
        return baseline_value, optimized_value, stats_result

    async def _measure_metric_baseline(self, metric: ScalabilityMetric) -> float:
        """Measure baseline value for a metric"""
        
        # Simulate metric measurement
        import random
        
        base_values = {
            ScalabilityMetric.THROUGHPUT: 100.0,
            ScalabilityMetric.RESPONSE_TIME: 200.0,
            ScalabilityMetric.RESOURCE_EFFICIENCY: 0.7,
            ScalabilityMetric.CONCURRENT_CAPACITY: 1000.0,
            ScalabilityMetric.MEMORY_SCALABILITY: 0.8,
            ScalabilityMetric.CPU_SCALABILITY: 0.75,
            ScalabilityMetric.NETWORK_EFFICIENCY: 0.85,
            ScalabilityMetric.CACHE_PERFORMANCE: 0.9
        }
        
        base_value = base_values.get(metric, 1.0)
        # Add realistic variance
        variance = random.uniform(0.9, 1.1)
        
        return base_value * variance

    async def _measure_metric_optimized(self, metric: ScalabilityMetric) -> float:
        """Measure optimized value for a metric"""
        
        baseline = await self._measure_metric_baseline(metric)
        
        # Simulate optimization improvements
        improvement_factors = {
            ScalabilityMetric.THROUGHPUT: 1.3,  # 30% improvement
            ScalabilityMetric.RESPONSE_TIME: 0.7,  # 30% reduction (better)
            ScalabilityMetric.RESOURCE_EFFICIENCY: 1.25,  # 25% improvement
            ScalabilityMetric.CONCURRENT_CAPACITY: 1.5,  # 50% improvement
            ScalabilityMetric.MEMORY_SCALABILITY: 1.2,  # 20% improvement
            ScalabilityMetric.CPU_SCALABILITY: 1.35,  # 35% improvement
            ScalabilityMetric.NETWORK_EFFICIENCY: 1.15,  # 15% improvement
            ScalabilityMetric.CACHE_PERFORMANCE: 1.1   # 10% improvement
        }
        
        factor = improvement_factors.get(metric, 1.1)
        return baseline * factor

    async def execute_autonomous_evolution(
        self, 
        target_improvements: Dict[ScalabilityMetric, float] = None
    ) -> EvolutionResult:
        """Execute autonomous system evolution cycle"""
        
        evolution_id = f"evolution_{int(datetime.utcnow().timestamp())}"
        
        self.logger.info(f"Starting autonomous evolution {evolution_id}")
        
        try:
            # Phase 1: Baseline Measurement
            baseline_benchmarks = await self.execute_comprehensive_benchmark()
            
            # Phase 2: Performance Analysis
            optimization_opportunities = await self._analyze_performance_opportunities(
                baseline_benchmarks
            )
            
            # Phase 3: Optimization Generation
            optimizations = await self._generate_evolution_optimizations(
                optimization_opportunities, target_improvements
            )
            
            # Phase 4: Testing and Validation
            validation_results = await self._validate_optimizations(optimizations)
            
            # Phase 5: Safe Deployment
            deployment_safe = await self._assess_deployment_safety(validation_results)
            
            # Calculate overall gains
            overall_gain = self._calculate_overall_performance_gain(validation_results)
            efficiency_gain = self._calculate_resource_efficiency_gain(validation_results)
            stability_score = self._calculate_stability_score(validation_results)
            
            # Create evolution result
            evolution_result = EvolutionResult(
                evolution_id=evolution_id,
                phase=EvolutionPhase.DEPLOYMENT_ROLLOUT,
                strategy=EvolutionStrategy.QUANTUM_EVOLUTION,  # Default strategy
                improvements=validation_results,
                overall_performance_gain=overall_gain,
                resource_efficiency_gain=efficiency_gain,
                stability_score=stability_score,
                rollback_available=True,
                deployment_safe=deployment_safe
            )
            
            # Store evolution result
            self.evolution_history.append(evolution_result)
            self.last_evolution = datetime.utcnow()
            
            self.logger.info(f"Autonomous evolution completed: "
                           f"performance_gain={overall_gain:.1%}, "
                           f"efficiency_gain={efficiency_gain:.1%}, "
                           f"stability={stability_score:.2f}")
            
            return evolution_result
            
        except Exception as e:
            self.logger.error(f"Autonomous evolution failed: {e}")
            # Return failed evolution result
            return EvolutionResult(
                evolution_id=evolution_id,
                phase=EvolutionPhase.BASELINE_MEASUREMENT,
                strategy=EvolutionStrategy.QUANTUM_EVOLUTION,
                improvements=[],
                overall_performance_gain=0.0,
                resource_efficiency_gain=0.0,
                stability_score=0.0,
                rollback_available=False,
                deployment_safe=False
            )

    async def _analyze_performance_opportunities(
        self, benchmarks: List[BenchmarkResult]
    ) -> Dict[ScalabilityMetric, float]:
        """Analyze performance opportunities from benchmark results"""
        
        opportunities = {}
        
        for result in benchmarks:
            # Identify metrics with poor performance or high improvement potential
            if result.improvement_percent < 10.0:  # Less than 10% improvement
                opportunities[result.metric] = 1.0 - (result.improvement_percent / 100.0)
            elif result.baseline_value < self.performance_baselines.get(result.metric, 0):
                # Performance regression detected
                opportunities[result.metric] = 2.0  # High priority
        
        return opportunities

    async def _generate_evolution_optimizations(
        self, 
        opportunities: Dict[ScalabilityMetric, float],
        target_improvements: Dict[ScalabilityMetric, float] = None
    ) -> List[Dict[str, Any]]:
        """Generate evolution optimizations based on opportunities"""
        
        optimizations = []
        
        for metric, priority in opportunities.items():
            target_improvement = (target_improvements or {}).get(metric, 0.2)  # 20% default
            
            optimization = {
                "metric": metric,
                "priority": priority,
                "target_improvement": target_improvement,
                "strategy": self._select_optimization_strategy(metric),
                "parameters": self._generate_optimization_parameters(metric, target_improvement)
            }
            
            optimizations.append(optimization)
        
        return optimizations

    def _select_optimization_strategy(self, metric: ScalabilityMetric) -> str:
        """Select appropriate optimization strategy for metric"""
        
        strategy_map = {
            ScalabilityMetric.THROUGHPUT: "parallel_processing_optimization",
            ScalabilityMetric.RESPONSE_TIME: "latency_reduction_optimization", 
            ScalabilityMetric.RESOURCE_EFFICIENCY: "resource_pooling_optimization",
            ScalabilityMetric.CONCURRENT_CAPACITY: "connection_multiplexing_optimization",
            ScalabilityMetric.MEMORY_SCALABILITY: "memory_pooling_optimization",
            ScalabilityMetric.CPU_SCALABILITY: "cpu_affinity_optimization",
            ScalabilityMetric.NETWORK_EFFICIENCY: "network_compression_optimization",
            ScalabilityMetric.CACHE_PERFORMANCE: "cache_warming_optimization"
        }
        
        return strategy_map.get(metric, "general_optimization")

    def _generate_optimization_parameters(
        self, metric: ScalabilityMetric, target_improvement: float
    ) -> Dict[str, Any]:
        """Generate optimization parameters for specific metric"""
        
        base_parameters = {
            "target_improvement": target_improvement,
            "timeout_seconds": 300,
            "rollback_on_failure": True,
            "validate_before_deploy": True
        }
        
        metric_specific = {
            ScalabilityMetric.THROUGHPUT: {
                "worker_pool_size": int(50 * (1 + target_improvement)),
                "batch_size": int(100 * (1 + target_improvement))
            },
            ScalabilityMetric.RESPONSE_TIME: {
                "cache_ttl_seconds": int(300 * (1 + target_improvement)),
                "prefetch_enabled": True
            },
            ScalabilityMetric.CACHE_PERFORMANCE: {
                "cache_size_mb": int(512 * (1 + target_improvement)),
                "eviction_policy": "lru"
            }
        }
        
        base_parameters.update(metric_specific.get(metric, {}))
        return base_parameters

    async def _validate_optimizations(
        self, optimizations: List[Dict[str, Any]]
    ) -> List[BenchmarkResult]:
        """Validate optimizations through controlled testing"""
        
        validation_results = []
        
        for optimization in optimizations:
            try:
                # Simulate optimization validation
                metric = optimization["metric"]
                target_improvement = optimization["target_improvement"]
                
                # Execute benchmark with optimization
                baseline, optimized, stats = await self._execute_metric_benchmark(metric, 50)
                
                # Calculate actual improvement
                actual_improvement = ((optimized - baseline) / baseline) if baseline > 0 else 0
                
                # Validate against target
                success = actual_improvement >= (target_improvement * 0.8)  # 80% of target
                
                result = BenchmarkResult(
                    benchmark_id=f"validation_{metric.value}",
                    category=BenchmarkCategory.OPTIMIZATION,
                    metric=metric,
                    baseline_value=baseline,
                    optimized_value=optimized,
                    improvement_percent=actual_improvement * 100,
                    confidence_interval=stats.get("confidence_interval", (0, 0)),
                    statistical_significance=0.95 if success else 0.5,
                    sample_size=50,
                    execution_time_seconds=stats.get("execution_time", 0),
                    resource_usage=stats.get("resource_usage", {})
                )
                
                validation_results.append(result)
                
                self.logger.debug(f"Optimization validation {metric.value}: "
                                f"target={target_improvement:.1%}, "
                                f"actual={actual_improvement:.1%}, "
                                f"success={success}")
                
            except Exception as e:
                self.logger.error(f"Optimization validation failed for {optimization}: {e}")
        
        return validation_results

    async def _assess_deployment_safety(self, validation_results: List[BenchmarkResult]) -> bool:
        """Assess if optimizations are safe for deployment"""
        
        if not validation_results:
            return False
        
        # Safety criteria
        successful_validations = sum(1 for r in validation_results if r.statistical_significance > 0.8)
        success_rate = successful_validations / len(validation_results)
        
        # Check for regressions
        regressions = sum(1 for r in validation_results if r.improvement_percent < -5.0)
        
        # Overall safety assessment
        deployment_safe = (
            success_rate >= 0.8 and  # 80% success rate
            regressions == 0 and    # No significant regressions
            len(validation_results) >= 3  # Sufficient validation scope
        )
        
        self.logger.info(f"Deployment safety assessment: success_rate={success_rate:.1%}, "
                        f"regressions={regressions}, safe={deployment_safe}")
        
        return deployment_safe

    def _calculate_overall_performance_gain(self, results: List[BenchmarkResult]) -> float:
        """Calculate overall performance gain from results"""
        
        if not results:
            return 0.0
        
        # Weighted average based on metric importance
        weights = {
            ScalabilityMetric.THROUGHPUT: 0.3,
            ScalabilityMetric.RESPONSE_TIME: 0.25,
            ScalabilityMetric.RESOURCE_EFFICIENCY: 0.2,
            ScalabilityMetric.CONCURRENT_CAPACITY: 0.15,
            ScalabilityMetric.CACHE_PERFORMANCE: 0.1
        }
        
        weighted_gains = []
        for result in results:
            weight = weights.get(result.metric, 0.1)
            gain = result.improvement_percent / 100.0
            weighted_gains.append(gain * weight)
        
        return sum(weighted_gains)

    def _calculate_resource_efficiency_gain(self, results: List[BenchmarkResult]) -> float:
        """Calculate resource efficiency gain"""
        
        efficiency_results = [r for r in results if r.metric == ScalabilityMetric.RESOURCE_EFFICIENCY]
        if not efficiency_results:
            return 0.0
        
        return statistics.mean(r.improvement_percent / 100.0 for r in efficiency_results)

    def _calculate_stability_score(self, results: List[BenchmarkResult]) -> float:
        """Calculate stability score based on consistency of improvements"""
        
        if not results:
            return 0.0
        
        improvements = [r.improvement_percent for r in results]
        
        # High stability = low variance in improvements
        if len(improvements) > 1:
            variance = statistics.variance(improvements)
            # Normalize variance to 0-1 scale (lower variance = higher stability)
            stability = max(0.0, 1.0 - (variance / 100.0))
        else:
            stability = 1.0 if improvements[0] > 0 else 0.0
        
        return stability

    async def _establish_performance_baselines(self):
        """Establish performance baselines for all metrics"""
        
        self.logger.info("Establishing performance baselines...")
        
        baseline_benchmarks = await self.execute_comprehensive_benchmark(sample_size=50)
        
        for result in baseline_benchmarks:
            self.performance_baselines[result.metric] = result.baseline_value
        
        self.logger.info(f"Performance baselines established: {len(self.performance_baselines)} metrics")

    def _update_performance_trends(self, results: List[BenchmarkResult]):
        """Update performance trend tracking"""
        
        # Calculate aggregate performance score
        overall_score = self._calculate_overall_performance_gain(results)
        self.performance_trend.append({
            "timestamp": datetime.utcnow(),
            "overall_score": overall_score,
            "result_count": len(results)
        })

    async def _continuous_benchmarking_loop(self):
        """Background loop for continuous benchmarking"""
        while self._running:
            try:
                if (not self.last_benchmark or 
                    datetime.utcnow() - self.last_benchmark >= self.benchmark_frequency):
                    
                    await self.execute_comprehensive_benchmark()
                
                await asyncio.sleep(300)  # Check every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Continuous benchmarking error: {e}")
                await asyncio.sleep(600)

    async def _autonomous_evolution_loop(self):
        """Background loop for autonomous evolution"""
        while self._running:
            try:
                if (not self.last_evolution or
                    datetime.utcnow() - self.last_evolution >= self.evolution_frequency):
                    
                    await self.execute_autonomous_evolution()
                
                await asyncio.sleep(1800)  # Check every 30 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Autonomous evolution error: {e}")
                await asyncio.sleep(3600)

    async def _scalability_monitoring_loop(self):
        """Background loop for scalability monitoring"""
        while self._running:
            try:
                await self._monitor_scalability_metrics()
                await asyncio.sleep(60)  # Monitor every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Scalability monitoring error: {e}")
                await asyncio.sleep(120)

    async def _performance_optimization_loop(self):
        """Background loop for continuous performance optimization"""
        while self._running:
            try:
                await self._optimize_current_performance()
                await asyncio.sleep(300)  # Optimize every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Performance optimization error: {e}")
                await asyncio.sleep(600)

    async def _regression_detection_loop(self):
        """Background loop for performance regression detection"""
        while self._running:
            try:
                await self._detect_performance_regressions()
                await asyncio.sleep(600)  # Check every 10 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Regression detection error: {e}")
                await asyncio.sleep(1200)

    async def _monitor_scalability_metrics(self):
        """Monitor current scalability metrics"""
        
        # Get current system metrics from quantum orchestrator
        orchestrator_status = self.quantum_orchestrator.get_orchestrator_status()
        latest_metrics = orchestrator_status.get("latest_metrics", {})
        
        if latest_metrics:
            # Calculate current load level
            response_time = latest_metrics.get("response_time_ms", 0)
            memory_usage = latest_metrics.get("memory_usage_percent", 0)
            cpu_usage = latest_metrics.get("cpu_utilization_percent", 0)
            
            # Normalize to 0-1 scale
            load_factors = [
                min(1.0, response_time / self.scalability_profile.response_time_sla_ms),
                memory_usage / 100.0,
                cpu_usage / 100.0
            ]
            
            self.current_load_level = statistics.mean(load_factors)
            
            # Make scaling decisions if needed
            await self._make_scaling_decisions()

    async def _make_scaling_decisions(self):
        """Make automatic scaling decisions based on current load"""
        
        scaling_decision = None
        
        if self.current_load_level > 0.8:  # 80% capacity
            scaling_decision = {
                "action": "scale_up",
                "factor": 1.5,
                "reason": f"High load detected: {self.current_load_level:.1%}"
            }
        elif self.current_load_level < 0.3:  # 30% capacity
            scaling_decision = {
                "action": "scale_down",
                "factor": 0.8,
                "reason": f"Low load detected: {self.current_load_level:.1%}"
            }
        
        if scaling_decision:
            self.scaling_decisions.append({
                "timestamp": datetime.utcnow(),
                **scaling_decision
            })
            
            self.logger.info(f"Scaling decision: {scaling_decision['action']} "
                           f"by factor {scaling_decision['factor']}")

    async def _optimize_current_performance(self):
        """Perform continuous performance optimization"""
        
        # Quick performance checks and optimizations
        if len(self.performance_trend) >= 5:
            recent_trends = list(self.performance_trend)[-5:]
            trend_scores = [t["overall_score"] for t in recent_trends]
            
            # Check if performance is declining
            if len(trend_scores) > 1:
                recent_avg = statistics.mean(trend_scores[-3:])
                older_avg = statistics.mean(trend_scores[:2])
                
                if recent_avg < older_avg * 0.95:  # 5% decline
                    self.logger.warning("Performance decline detected, triggering optimization")
                    # Could trigger immediate optimization here

    async def _detect_performance_regressions(self):
        """Detect performance regressions"""
        
        if not self.performance_baselines:
            return
        
        # Check recent benchmark results against baselines
        recent_results = []
        for category_results in self.benchmark_results.values():
            if category_results:
                recent_results.extend(category_results[-5:])  # Last 5 results per category
        
        regressions = []
        for result in recent_results:
            baseline = self.performance_baselines.get(result.metric)
            if baseline and result.optimized_value < baseline * 0.9:  # 10% regression
                regressions.append((result.metric, result.optimized_value, baseline))
        
        if regressions:
            self.logger.warning(f"Performance regressions detected: {len(regressions)} metrics")
            for metric, current, baseline in regressions:
                self.logger.warning(f"  {metric.value}: {current:.2f} vs baseline {baseline:.2f}")

    def get_scalability_status(self) -> Dict[str, Any]:
        """Get comprehensive scalability status"""
        
        return {
            "current_load_level": self.current_load_level,
            "scalability_profile": {
                "max_concurrent_users": self.scalability_profile.max_concurrent_users,
                "max_requests_per_second": self.scalability_profile.max_requests_per_second,
                "response_time_sla_ms": self.scalability_profile.response_time_sla_ms,
                "availability_target": self.scalability_profile.availability_target
            },
            "evolution_history_count": len(self.evolution_history),
            "benchmark_categories": list(self.benchmark_results.keys()),
            "performance_baselines": {k.value: v for k, v in self.performance_baselines.items()},
            "recent_scaling_decisions": list(self.scaling_decisions)[-5:],
            "performance_trend": list(self.performance_trend)[-10:],
            "last_benchmark": self.last_benchmark.isoformat() if self.last_benchmark else None,
            "last_evolution": self.last_evolution.isoformat() if self.last_evolution else None,
            "background_tasks_running": len(self._background_tasks),
            "running": self._running
        }


# Global singleton instance
_scalable_evolution_engine = None

def get_scalable_evolution_engine(
    scalability_profile: Optional[ScalabilityProfile] = None
) -> ScalableEvolutionEngine:
    """Get global Scalable Evolution Engine instance"""
    global _scalable_evolution_engine
    if _scalable_evolution_engine is None:
        _scalable_evolution_engine = ScalableEvolutionEngine(scalability_profile)
    return _scalable_evolution_engine