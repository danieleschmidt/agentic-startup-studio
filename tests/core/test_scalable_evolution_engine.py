"""
Tests for Scalable Evolution Engine
Comprehensive testing for Generation 3 scalable optimization enhancements
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from pipeline.core.scalable_evolution_engine import (
    ScalableEvolutionEngine,
    BenchmarkResult,
    EvolutionResult,
    ScalabilityProfile,
    ScalabilityMetric,
    EvolutionPhase,
    BenchmarkCategory,
    EvolutionStrategy,
    get_scalable_evolution_engine
)


class TestScalableEvolutionEngine:
    """Test suite for Scalable Evolution Engine"""

    @pytest.fixture
    def scalability_profile(self):
        """Create test scalability profile"""
        return ScalabilityProfile(
            max_concurrent_users=5000,
            max_requests_per_second=2500.0,
            memory_scaling_factor=1.5,
            cpu_scaling_factor=2.0,
            storage_scaling_factor=3.0,
            network_bandwidth_limit=5.0,
            cache_hit_ratio_target=0.9,
            response_time_sla_ms=150.0,
            availability_target=0.995
        )

    @pytest.fixture
    def evolution_engine(self, scalability_profile):
        """Create test instance of evolution engine"""
        return ScalableEvolutionEngine(scalability_profile)

    @pytest.fixture
    def sample_benchmark_result(self):
        """Sample benchmark result for testing"""
        return BenchmarkResult(
            benchmark_id="test_benchmark_001",
            category=BenchmarkCategory.OPTIMIZATION,
            metric=ScalabilityMetric.THROUGHPUT,
            baseline_value=100.0,
            optimized_value=130.0,
            improvement_percent=30.0,
            confidence_interval=(125.0, 135.0),
            statistical_significance=0.95,
            sample_size=100,
            execution_time_seconds=45.0,
            resource_usage={"memory_mb": 50.0, "cpu_percent": 25.0}
        )

    def test_evolution_engine_initialization(self, evolution_engine, scalability_profile):
        """Test proper initialization of evolution engine"""
        assert evolution_engine is not None
        assert evolution_engine.scalability_profile == scalability_profile
        assert len(evolution_engine.evolution_history) == 0
        assert len(evolution_engine.benchmark_results) == 0
        assert len(evolution_engine.performance_baselines) == 0
        assert evolution_engine.current_load_level == 0.0
        assert evolution_engine._running is True
        assert evolution_engine.benchmark_frequency == timedelta(hours=1)
        assert evolution_engine.evolution_frequency == timedelta(hours=6)

    @pytest.mark.asyncio
    async def test_metric_baseline_measurement(self, evolution_engine):
        """Test baseline metric measurement"""
        
        metrics_to_test = [
            ScalabilityMetric.THROUGHPUT,
            ScalabilityMetric.RESPONSE_TIME,
            ScalabilityMetric.RESOURCE_EFFICIENCY,
            ScalabilityMetric.CONCURRENT_CAPACITY
        ]
        
        for metric in metrics_to_test:
            baseline = await evolution_engine._measure_metric_baseline(metric)
            assert isinstance(baseline, (int, float))
            assert baseline > 0
            
            # Test multiple measurements for consistency
            baselines = []
            for _ in range(5):
                baseline = await evolution_engine._measure_metric_baseline(metric)
                baselines.append(baseline)
            
            # Should have some variance but be in reasonable range
            assert min(baselines) > 0
            assert max(baselines) / min(baselines) < 2.0  # Not more than 2x variance

    @pytest.mark.asyncio
    async def test_metric_optimized_measurement(self, evolution_engine):
        """Test optimized metric measurement"""
        
        # Test that optimized values show improvement over baseline
        for metric in [ScalabilityMetric.THROUGHPUT, ScalabilityMetric.RESOURCE_EFFICIENCY]:
            baseline = await evolution_engine._measure_metric_baseline(metric)
            optimized = await evolution_engine._measure_metric_optimized(metric)
            
            # For these metrics, optimized should be better (higher)
            assert optimized > baseline
            
        # Test response time (lower is better)
        baseline_rt = await evolution_engine._measure_metric_baseline(ScalabilityMetric.RESPONSE_TIME)
        optimized_rt = await evolution_engine._measure_metric_optimized(ScalabilityMetric.RESPONSE_TIME)
        assert optimized_rt < baseline_rt  # Lower response time is better

    @pytest.mark.asyncio
    async def test_execute_metric_benchmark(self, evolution_engine):
        """Test metric benchmark execution"""
        
        metric = ScalabilityMetric.THROUGHPUT
        sample_size = 10  # Small sample for testing
        
        baseline, optimized, stats = await evolution_engine._execute_metric_benchmark(
            metric, sample_size
        )
        
        assert isinstance(baseline, (int, float))
        assert isinstance(optimized, (int, float))
        assert isinstance(stats, dict)
        
        # Check stats structure
        assert "confidence_interval" in stats
        assert "p_value" in stats
        assert "execution_time" in stats
        assert "resource_usage" in stats
        
        # Validate confidence interval
        ci = stats["confidence_interval"]
        assert len(ci) == 2
        assert ci[0] <= optimized <= ci[1]
        
        # Validate p_value
        assert 0.0 <= stats["p_value"] <= 1.0
        
        # Validate execution time
        assert stats["execution_time"] > 0

    @pytest.mark.asyncio
    async def test_benchmark_category_execution(self, evolution_engine):
        """Test benchmark execution for a category"""
        
        benchmark_id = "test_benchmark"
        category = BenchmarkCategory.OPTIMIZATION
        sample_size = 10
        
        results = await evolution_engine._benchmark_category(
            benchmark_id, category, sample_size
        )
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        # Check each result
        for result in results:
            assert isinstance(result, BenchmarkResult)
            assert result.benchmark_id.startswith(benchmark_id)
            assert result.category == category
            assert result.sample_size == sample_size
            assert result.baseline_value > 0
            assert result.optimized_value > 0
            assert result.execution_time_seconds > 0

    @pytest.mark.asyncio
    async def test_comprehensive_benchmark_execution(self, evolution_engine):
        """Test comprehensive benchmark execution"""
        
        # Mock the core components to avoid dependencies
        evolution_engine.benchmarking_suite = Mock()
        evolution_engine.autonomous_evolution = Mock()
        evolution_engine.quantum_orchestrator = Mock()
        
        categories = [BenchmarkCategory.OPTIMIZATION]
        sample_size = 5
        
        results = await evolution_engine.execute_comprehensive_benchmark(
            categories, sample_size
        )
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        # Check that results are stored
        assert len(evolution_engine.benchmark_results) > 0
        assert evolution_engine.last_benchmark is not None

    @pytest.mark.asyncio
    async def test_performance_opportunity_analysis(self, evolution_engine):
        """Test performance opportunity analysis"""
        
        # Create test benchmark results
        test_results = [
            BenchmarkResult(
                benchmark_id="test1",
                category=BenchmarkCategory.OPTIMIZATION,
                metric=ScalabilityMetric.THROUGHPUT,
                baseline_value=100.0,
                optimized_value=105.0,  # Only 5% improvement - opportunity
                improvement_percent=5.0,
                confidence_interval=(100.0, 110.0),
                statistical_significance=0.8,
                sample_size=50,
                execution_time_seconds=30.0,
                resource_usage={}
            ),
            BenchmarkResult(
                benchmark_id="test2",
                category=BenchmarkCategory.OPTIMIZATION,
                metric=ScalabilityMetric.RESPONSE_TIME,
                baseline_value=200.0,
                optimized_value=220.0,  # Performance regression
                improvement_percent=-10.0,
                confidence_interval=(210.0, 230.0),
                statistical_significance=0.9,
                sample_size=50,
                execution_time_seconds=25.0,
                resource_usage={}
            )
        ]
        
        opportunities = await evolution_engine._analyze_performance_opportunities(test_results)
        
        assert isinstance(opportunities, dict)
        assert ScalabilityMetric.THROUGHPUT in opportunities
        assert ScalabilityMetric.RESPONSE_TIME in opportunities
        
        # Response time should have higher priority due to regression
        assert opportunities[ScalabilityMetric.RESPONSE_TIME] > opportunities[ScalabilityMetric.THROUGHPUT]

    @pytest.mark.asyncio
    async def test_optimization_generation(self, evolution_engine):
        """Test optimization generation"""
        
        opportunities = {
            ScalabilityMetric.THROUGHPUT: 0.8,
            ScalabilityMetric.RESPONSE_TIME: 1.5
        }
        
        target_improvements = {
            ScalabilityMetric.THROUGHPUT: 0.25,
            ScalabilityMetric.RESPONSE_TIME: 0.3
        }
        
        optimizations = await evolution_engine._generate_evolution_optimizations(
            opportunities, target_improvements
        )
        
        assert isinstance(optimizations, list)
        assert len(optimizations) == 2
        
        for optimization in optimizations:
            assert "metric" in optimization
            assert "priority" in optimization
            assert "target_improvement" in optimization
            assert "strategy" in optimization
            assert "parameters" in optimization
            
            assert optimization["metric"] in opportunities
            assert optimization["priority"] > 0
            assert optimization["target_improvement"] > 0

    def test_optimization_strategy_selection(self, evolution_engine):
        """Test optimization strategy selection for different metrics"""
        
        strategies = {}
        for metric in ScalabilityMetric:
            strategy = evolution_engine._select_optimization_strategy(metric)
            strategies[metric] = strategy
            assert isinstance(strategy, str)
            assert len(strategy) > 0
        
        # Check that different metrics get different strategies
        unique_strategies = set(strategies.values())
        assert len(unique_strategies) > 1  # Should have variety

    def test_optimization_parameter_generation(self, evolution_engine):
        """Test optimization parameter generation"""
        
        metric = ScalabilityMetric.THROUGHPUT
        target_improvement = 0.3
        
        parameters = evolution_engine._generate_optimization_parameters(
            metric, target_improvement
        )
        
        assert isinstance(parameters, dict)
        assert "target_improvement" in parameters
        assert "timeout_seconds" in parameters
        assert "rollback_on_failure" in parameters
        assert "validate_before_deploy" in parameters
        
        assert parameters["target_improvement"] == target_improvement
        assert parameters["timeout_seconds"] > 0
        assert isinstance(parameters["rollback_on_failure"], bool)

    @pytest.mark.asyncio
    async def test_optimization_validation(self, evolution_engine):
        """Test optimization validation"""
        
        optimizations = [
            {
                "metric": ScalabilityMetric.THROUGHPUT,
                "target_improvement": 0.2,
                "strategy": "test_strategy",
                "parameters": {"test": "value"}
            }
        ]
        
        results = await evolution_engine._validate_optimizations(optimizations)
        
        assert isinstance(results, list)
        assert len(results) == 1
        
        result = results[0]
        assert isinstance(result, BenchmarkResult)
        assert result.metric == ScalabilityMetric.THROUGHPUT
        assert result.sample_size == 50

    @pytest.mark.asyncio
    async def test_deployment_safety_assessment(self, evolution_engine):
        """Test deployment safety assessment"""
        
        # Test safe deployment scenario
        safe_results = [
            BenchmarkResult(
                benchmark_id=f"safe_{i}",
                category=BenchmarkCategory.OPTIMIZATION,
                metric=ScalabilityMetric.THROUGHPUT,
                baseline_value=100.0,
                optimized_value=120.0,
                improvement_percent=20.0,
                confidence_interval=(115.0, 125.0),
                statistical_significance=0.95,
                sample_size=50,
                execution_time_seconds=30.0,
                resource_usage={}
            ) for i in range(3)
        ]
        
        safe = await evolution_engine._assess_deployment_safety(safe_results)
        assert safe is True
        
        # Test unsafe deployment scenario (with regressions)
        unsafe_results = [
            BenchmarkResult(
                benchmark_id="unsafe_1",
                category=BenchmarkCategory.OPTIMIZATION,
                metric=ScalabilityMetric.THROUGHPUT,
                baseline_value=100.0,
                optimized_value=80.0,  # Regression
                improvement_percent=-20.0,
                confidence_interval=(75.0, 85.0),
                statistical_significance=0.95,
                sample_size=50,
                execution_time_seconds=30.0,
                resource_usage={}
            )
        ]
        
        unsafe = await evolution_engine._assess_deployment_safety(unsafe_results)
        assert unsafe is False

    def test_performance_gain_calculations(self, evolution_engine):
        """Test performance gain calculations"""
        
        test_results = [
            BenchmarkResult(
                benchmark_id="test1",
                category=BenchmarkCategory.OPTIMIZATION,
                metric=ScalabilityMetric.THROUGHPUT,
                baseline_value=100.0,
                optimized_value=130.0,
                improvement_percent=30.0,
                confidence_interval=(125.0, 135.0),
                statistical_significance=0.95,
                sample_size=50,
                execution_time_seconds=30.0,
                resource_usage={}
            ),
            BenchmarkResult(
                benchmark_id="test2",
                category=BenchmarkCategory.OPTIMIZATION,
                metric=ScalabilityMetric.RESPONSE_TIME,
                baseline_value=200.0,
                optimized_value=160.0,
                improvement_percent=20.0,
                confidence_interval=(155.0, 165.0),
                statistical_significance=0.9,
                sample_size=50,
                execution_time_seconds=25.0,
                resource_usage={}
            )
        ]
        
        # Test overall performance gain
        overall_gain = evolution_engine._calculate_overall_performance_gain(test_results)
        assert isinstance(overall_gain, float)
        assert overall_gain > 0
        
        # Test resource efficiency gain
        efficiency_gain = evolution_engine._calculate_resource_efficiency_gain(test_results)
        assert isinstance(efficiency_gain, float)
        
        # Test stability score
        stability = evolution_engine._calculate_stability_score(test_results)
        assert isinstance(stability, float)
        assert 0.0 <= stability <= 1.0

    @pytest.mark.asyncio
    async def test_autonomous_evolution_execution(self, evolution_engine):
        """Test autonomous evolution execution"""
        
        # Mock dependencies
        evolution_engine.benchmarking_suite = Mock()
        evolution_engine.autonomous_evolution = Mock()
        evolution_engine.quantum_orchestrator = Mock()
        
        target_improvements = {
            ScalabilityMetric.THROUGHPUT: 0.2,
            ScalabilityMetric.RESPONSE_TIME: 0.15
        }
        
        result = await evolution_engine.execute_autonomous_evolution(target_improvements)
        
        assert isinstance(result, EvolutionResult)
        assert result.evolution_id.startswith("evolution_")
        assert result.phase == EvolutionPhase.DEPLOYMENT_ROLLOUT
        assert isinstance(result.overall_performance_gain, float)
        assert isinstance(result.resource_efficiency_gain, float)
        assert isinstance(result.stability_score, float)
        assert isinstance(result.rollback_available, bool)
        assert isinstance(result.deployment_safe, bool)
        
        # Check that result is stored
        assert len(evolution_engine.evolution_history) == 1
        assert evolution_engine.last_evolution is not None

    def test_scalability_metrics_monitoring(self, evolution_engine):
        """Test scalability metrics monitoring"""
        
        # Mock quantum orchestrator status
        evolution_engine.quantum_orchestrator.get_orchestrator_status = Mock(return_value={
            "latest_metrics": {
                "response_time_ms": 100.0,
                "memory_usage_percent": 60.0,
                "cpu_utilization_percent": 40.0
            }
        })
        
        # Run monitoring
        asyncio.run(evolution_engine._monitor_scalability_metrics())
        
        # Check load level calculation
        assert 0.0 <= evolution_engine.current_load_level <= 1.0

    @pytest.mark.asyncio
    async def test_scaling_decisions(self, evolution_engine):
        """Test automatic scaling decisions"""
        
        # Test high load scenario
        evolution_engine.current_load_level = 0.85  # 85% load
        await evolution_engine._make_scaling_decisions()
        
        assert len(evolution_engine.scaling_decisions) > 0
        decision = evolution_engine.scaling_decisions[-1]
        assert decision["action"] == "scale_up"
        assert decision["factor"] > 1.0
        
        # Test low load scenario
        evolution_engine.current_load_level = 0.25  # 25% load
        await evolution_engine._make_scaling_decisions()
        
        decision = evolution_engine.scaling_decisions[-1]
        assert decision["action"] == "scale_down"
        assert decision["factor"] < 1.0

    @pytest.mark.asyncio
    async def test_performance_baseline_establishment(self, evolution_engine):
        """Test performance baseline establishment"""
        
        # Mock dependencies
        evolution_engine.benchmarking_suite = Mock()
        evolution_engine.autonomous_evolution = Mock()
        evolution_engine.quantum_orchestrator = Mock()
        
        await evolution_engine._establish_performance_baselines()
        
        # Should have established baselines
        assert len(evolution_engine.performance_baselines) > 0
        
        # All baselines should be positive numbers
        for metric, baseline in evolution_engine.performance_baselines.items():
            assert isinstance(baseline, (int, float))
            assert baseline > 0

    def test_scalability_status_reporting(self, evolution_engine):
        """Test scalability status reporting"""
        
        # Add some test data
        evolution_engine.current_load_level = 0.6
        evolution_engine.performance_baselines = {
            ScalabilityMetric.THROUGHPUT: 100.0,
            ScalabilityMetric.RESPONSE_TIME: 150.0
        }
        
        status = evolution_engine.get_scalability_status()
        
        assert isinstance(status, dict)
        assert "current_load_level" in status
        assert "scalability_profile" in status
        assert "evolution_history_count" in status
        assert "benchmark_categories" in status
        assert "performance_baselines" in status
        assert "recent_scaling_decisions" in status
        assert "performance_trend" in status
        assert "running" in status
        
        assert status["current_load_level"] == 0.6
        assert status["running"] is True

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self, evolution_engine):
        """Test engine start/stop lifecycle"""
        
        # Mock dependencies
        evolution_engine.benchmarking_suite = Mock()
        evolution_engine.autonomous_evolution = Mock()
        evolution_engine.quantum_orchestrator = Mock()
        evolution_engine.quantum_orchestrator.start = AsyncMock()
        evolution_engine.quantum_orchestrator.stop = AsyncMock()
        
        # Mock background methods to avoid long-running loops
        with patch.object(evolution_engine, '_continuous_benchmarking_loop', new_callable=AsyncMock):
            with patch.object(evolution_engine, '_autonomous_evolution_loop', new_callable=AsyncMock):
                with patch.object(evolution_engine, '_scalability_monitoring_loop', new_callable=AsyncMock):
                    with patch.object(evolution_engine, '_performance_optimization_loop', new_callable=AsyncMock):
                        with patch.object(evolution_engine, '_regression_detection_loop', new_callable=AsyncMock):
                            with patch.object(evolution_engine, '_establish_performance_baselines', new_callable=AsyncMock):
                                
                                # Test start
                                await evolution_engine.start()
                                assert len(evolution_engine._background_tasks) == 5
                                
                                # Test stop
                                await evolution_engine.stop()
                                assert evolution_engine._running is False

    def test_singleton_pattern(self):
        """Test singleton pattern for global instance"""
        
        instance1 = get_scalable_evolution_engine()
        instance2 = get_scalable_evolution_engine()
        
        assert instance1 is instance2
        assert isinstance(instance1, ScalableEvolutionEngine)


class TestScalabilityProfile:
    """Test ScalabilityProfile data structure"""

    def test_scalability_profile_creation(self):
        """Test ScalabilityProfile instantiation"""
        
        profile = ScalabilityProfile(
            max_concurrent_users=1000,
            max_requests_per_second=500.0,
            memory_scaling_factor=1.2,
            cpu_scaling_factor=1.8,
            storage_scaling_factor=2.5,
            network_bandwidth_limit=5.0,
            cache_hit_ratio_target=0.85,
            response_time_sla_ms=300.0,
            availability_target=0.995
        )
        
        assert profile.max_concurrent_users == 1000
        assert profile.max_requests_per_second == 500.0
        assert profile.memory_scaling_factor == 1.2
        assert profile.cpu_scaling_factor == 1.8
        assert profile.storage_scaling_factor == 2.5
        assert profile.network_bandwidth_limit == 5.0
        assert profile.cache_hit_ratio_target == 0.85
        assert profile.response_time_sla_ms == 300.0
        assert profile.availability_target == 0.995


class TestBenchmarkResult:
    """Test BenchmarkResult data structure"""

    def test_benchmark_result_creation(self):
        """Test BenchmarkResult instantiation"""
        
        result = BenchmarkResult(
            benchmark_id="test_001",
            category=BenchmarkCategory.OPTIMIZATION,
            metric=ScalabilityMetric.THROUGHPUT,
            baseline_value=100.0,
            optimized_value=125.0,
            improvement_percent=25.0,
            confidence_interval=(120.0, 130.0),
            statistical_significance=0.95,
            sample_size=100,
            execution_time_seconds=60.0,
            resource_usage={"memory_mb": 75.0, "cpu_percent": 30.0}
        )
        
        assert result.benchmark_id == "test_001"
        assert result.category == BenchmarkCategory.OPTIMIZATION
        assert result.metric == ScalabilityMetric.THROUGHPUT
        assert result.baseline_value == 100.0
        assert result.optimized_value == 125.0
        assert result.improvement_percent == 25.0
        assert result.confidence_interval == (120.0, 130.0)
        assert result.statistical_significance == 0.95
        assert result.sample_size == 100
        assert result.execution_time_seconds == 60.0
        assert result.resource_usage["memory_mb"] == 75.0
        assert isinstance(result.timestamp, datetime)