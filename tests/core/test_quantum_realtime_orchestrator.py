"""
Tests for Quantum Real-time Orchestrator
Comprehensive testing for Generation 2 robust enhancements
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from pipeline.core.quantum_realtime_orchestrator import (
    QuantumRealtimeOrchestrator,
    SystemMetrics,
    OptimizationAction,
    OrchestratorState,
    SystemState,
    OrchestratorMode,
    OptimizationTarget,
    OptimizationStrategy,
    Priority,
    get_quantum_realtime_orchestrator
)


class TestQuantumRealtimeOrchestrator:
    """Test suite for Quantum Real-time Orchestrator"""

    @pytest.fixture
    def orchestrator(self):
        """Create test instance of orchestrator"""
        return QuantumRealtimeOrchestrator(mode=OrchestratorMode.ADAPTIVE)

    @pytest.fixture
    def sample_metrics(self):
        """Sample system metrics for testing"""
        return SystemMetrics(
            timestamp=datetime.utcnow(),
            response_time_ms=150.0,
            throughput_rps=75.0,
            memory_usage_percent=65.0,
            cpu_utilization_percent=45.0,
            error_rate_percent=0.5,
            active_connections=50,
            cache_hit_rate_percent=88.0,
            queue_depth=5
        )

    @pytest.fixture
    def critical_metrics(self):
        """Critical system metrics for testing"""
        return SystemMetrics(
            timestamp=datetime.utcnow(),
            response_time_ms=1200.0,  # Very high response time
            throughput_rps=10.0,      # Very low throughput
            memory_usage_percent=95.0, # Very high memory usage
            cpu_utilization_percent=98.0, # Very high CPU
            error_rate_percent=8.0,   # High error rate
            active_connections=200,
            cache_hit_rate_percent=45.0, # Poor cache performance
            queue_depth=50
        )

    def test_orchestrator_initialization(self, orchestrator):
        """Test proper initialization of orchestrator"""
        assert orchestrator is not None
        assert orchestrator.state.mode == OrchestratorMode.ADAPTIVE
        assert orchestrator.state.system_state == SystemState.OPTIMAL
        assert len(orchestrator.state.active_optimizations) == 0
        assert len(orchestrator.optimization_history) == 0
        assert orchestrator._running is True
        assert orchestrator.max_concurrent_optimizations == 3
        assert orchestrator.anomaly_threshold == 2.0

    @pytest.mark.asyncio
    async def test_metrics_collection(self, orchestrator):
        """Test system metrics collection"""
        
        # Mock quantum optimizer and realtime intelligence
        orchestrator.quantum_optimizer = Mock()
        orchestrator.realtime_intelligence = Mock()
        
        metrics = await orchestrator.collect_metrics()
        
        assert isinstance(metrics, SystemMetrics)
        assert metrics.timestamp is not None
        assert metrics.response_time_ms >= 0
        assert metrics.throughput_rps >= 0
        assert 0 <= metrics.memory_usage_percent <= 100
        assert 0 <= metrics.cpu_utilization_percent <= 100
        assert metrics.error_rate_percent >= 0
        assert metrics.active_connections >= 0
        assert 0 <= metrics.cache_hit_rate_percent <= 100
        
        # Check metrics are stored in history
        assert len(orchestrator.state.metrics_history) == 1
        assert orchestrator.state.metrics_history[0] == metrics

    @pytest.mark.asyncio
    async def test_system_state_analysis_optimal(self, orchestrator, sample_metrics):
        """Test system state analysis with optimal metrics"""
        
        # Add sample metrics to history
        orchestrator.state.metrics_history.append(sample_metrics)
        
        await orchestrator._analyze_system_state()
        
        # Should remain optimal with good metrics
        assert orchestrator.state.system_state == SystemState.OPTIMAL

    @pytest.mark.asyncio
    async def test_system_state_analysis_critical(self, orchestrator, critical_metrics):
        """Test system state analysis with critical metrics"""
        
        # Add critical metrics to history
        orchestrator.state.metrics_history.append(critical_metrics)
        
        await orchestrator._analyze_system_state()
        
        # Should detect critical state
        assert orchestrator.state.system_state == SystemState.CRITICAL

    @pytest.mark.asyncio
    async def test_anomaly_detection(self, orchestrator):
        """Test anomaly detection capabilities"""
        
        # Generate stable baseline metrics
        baseline_response = 100.0
        for i in range(15):
            metrics = SystemMetrics(
                timestamp=datetime.utcnow(),
                response_time_ms=baseline_response + (i % 3 - 1) * 5,  # Small variance
                throughput_rps=50.0,
                memory_usage_percent=60.0,
                cpu_utilization_percent=40.0,
                error_rate_percent=0.1,
                active_connections=30,
                cache_hit_rate_percent=90.0
            )
            orchestrator.state.metrics_history.append(metrics)
        
        # Add anomalous metric
        anomaly_metrics = SystemMetrics(
            timestamp=datetime.utcnow(),
            response_time_ms=500.0,  # Significant spike
            throughput_rps=50.0,
            memory_usage_percent=60.0,
            cpu_utilization_percent=40.0,
            error_rate_percent=0.1,
            active_connections=30,
            cache_hit_rate_percent=90.0
        )
        orchestrator.state.metrics_history.append(anomaly_metrics)
        
        # Mock anomaly handler
        with patch.object(orchestrator, '_handle_anomaly', new_callable=AsyncMock) as mock_handler:
            await orchestrator._detect_anomalies()
            
            # Should detect and handle anomaly
            mock_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_optimization_recommendations(self, orchestrator):
        """Test optimization recommendation generation"""
        
        # Set degraded system state
        orchestrator.state.system_state = SystemState.DEGRADED
        orchestrator.state.last_optimization = None  # No cooldown
        
        # Add metrics with issues
        degraded_metrics = SystemMetrics(
            timestamp=datetime.utcnow(),
            response_time_ms=300.0,  # High response time
            throughput_rps=20.0,
            memory_usage_percent=85.0,  # High memory usage
            cpu_utilization_percent=70.0,
            error_rate_percent=2.0,
            active_connections=80,
            cache_hit_rate_percent=65.0  # Poor cache performance
        )
        orchestrator.state.metrics_history.append(degraded_metrics)
        
        await orchestrator._generate_optimization_recommendations()
        
        # Should generate optimization recommendations
        assert len(orchestrator.state.active_optimizations) > 0
        
        # Check recommendation types
        targets = [action.target for action in orchestrator.state.active_optimizations]
        assert OptimizationTarget.RESPONSE_TIME in targets or \
               OptimizationTarget.MEMORY_USAGE in targets or \
               OptimizationTarget.CACHE_HIT_RATE in targets

    @pytest.mark.asyncio
    async def test_quantum_optimization_analysis(self, orchestrator):
        """Test quantum-enhanced optimization analysis"""
        
        # Add metrics requiring optimization
        metrics = SystemMetrics(
            timestamp=datetime.utcnow(),
            response_time_ms=250.0,  # Above 200ms threshold
            throughput_rps=30.0,
            memory_usage_percent=80.0,  # Above 75% threshold
            cpu_utilization_percent=60.0,
            error_rate_percent=1.0,
            active_connections=60,
            cache_hit_rate_percent=75.0  # Below 80% threshold
        )
        orchestrator.state.metrics_history.append(metrics)
        
        recommendations = await orchestrator._quantum_optimization_analysis()
        
        assert len(recommendations) > 0
        
        # Check that recommendations target the right issues
        targets = [rec.target for rec in recommendations]
        expected_targets = [
            OptimizationTarget.RESPONSE_TIME,
            OptimizationTarget.MEMORY_USAGE,
            OptimizationTarget.CACHE_HIT_RATE
        ]
        
        for target in expected_targets:
            assert target in targets
        
        # Check recommendation properties
        for rec in recommendations:
            assert isinstance(rec, OptimizationAction)
            assert 0.0 <= rec.expected_improvement <= 1.0
            assert 0.0 <= rec.confidence <= 1.0
            assert rec.priority in [Priority.HIGH, Priority.MEDIUM, Priority.LOW, Priority.CRITICAL]

    @pytest.mark.asyncio
    async def test_optimization_execution(self, orchestrator):
        """Test optimization action execution"""
        
        # Create mock optimization action
        action = OptimizationAction(
            target=OptimizationTarget.RESPONSE_TIME,
            strategy=OptimizationStrategy.QUANTUM_ANNEALING,
            parameters={"test": "parameters"},
            expected_improvement=0.3,
            confidence=0.8,
            priority=Priority.HIGH
        )
        
        orchestrator.state.active_optimizations.append(action)
        
        # Mock quantum optimizer
        mock_result = {"improvement": 0.25, "success": True}
        orchestrator.quantum_optimizer = Mock()
        orchestrator.quantum_optimizer.optimize = AsyncMock(return_value=mock_result)
        
        await orchestrator._execute_pending_optimizations()
        
        # Check execution
        assert action.executed_at is not None
        assert action.result == mock_result
        assert len(orchestrator.optimization_history) == 1
        assert orchestrator.state.last_optimization is not None

    @pytest.mark.asyncio
    async def test_prediction_generation(self, orchestrator):
        """Test predictive analytics generation"""
        
        # Add historical metrics with a trend
        base_response = 100.0
        for i in range(25):
            metrics = SystemMetrics(
                timestamp=datetime.utcnow(),
                response_time_ms=base_response + (i * 2),  # Increasing trend
                throughput_rps=50.0,
                memory_usage_percent=60.0,
                cpu_utilization_percent=40.0,
                error_rate_percent=0.1,
                active_connections=30,
                cache_hit_rate_percent=90.0
            )
            orchestrator.state.metrics_history.append(metrics)
        
        await orchestrator._generate_predictions()
        
        # Should generate response time prediction
        assert "response_time_ms" in orchestrator.state.predictions
        assert orchestrator.state.predictions["response_time_ms"] > base_response

    @pytest.mark.asyncio
    async def test_self_healing_critical_state(self, orchestrator):
        """Test self-healing capabilities in critical state"""
        
        # Set critical system state
        orchestrator.state.system_state = SystemState.CRITICAL
        
        initial_optimizations = len(orchestrator.state.active_optimizations)
        
        await orchestrator._attempt_self_healing()
        
        # Should add emergency optimization and change state to recovering
        assert len(orchestrator.state.active_optimizations) > initial_optimizations
        assert orchestrator.state.system_state == SystemState.RECOVERING
        
        # Check emergency action properties
        emergency_actions = [a for a in orchestrator.state.active_optimizations 
                           if a.priority == Priority.CRITICAL]
        assert len(emergency_actions) > 0
        
        emergency_action = emergency_actions[0]
        assert emergency_action.parameters.get("emergency_mode") is True
        assert emergency_action.confidence >= 0.9

    @pytest.mark.asyncio
    async def test_performance_baseline_establishment(self, orchestrator):
        """Test performance baseline establishment"""
        
        # Mock collect_metrics to return consistent values
        mock_metrics = [
            SystemMetrics(
                timestamp=datetime.utcnow(),
                response_time_ms=100.0 + i,
                throughput_rps=50.0,
                memory_usage_percent=60.0,
                cpu_utilization_percent=40.0,
                error_rate_percent=0.1,
                active_connections=30,
                cache_hit_rate_percent=90.0
            )
            for i in range(10)
        ]
        
        with patch.object(orchestrator, 'collect_metrics', side_effect=mock_metrics):
            await orchestrator._establish_performance_baseline()
        
        # Check baseline is established
        assert len(orchestrator.performance_baseline) > 0
        assert "response_time_ms" in orchestrator.performance_baseline
        assert "throughput_rps" in orchestrator.performance_baseline
        assert orchestrator.performance_baseline["response_time_ms"] > 100.0
        assert orchestrator.performance_baseline["response_time_ms"] < 110.0

    def test_orchestrator_status(self, orchestrator, sample_metrics):
        """Test orchestrator status reporting"""
        
        # Add some test data
        orchestrator.state.metrics_history.append(sample_metrics)
        orchestrator.performance_baseline = {"response_time_ms": 120.0}
        orchestrator.state.predictions = {"response_time_ms": 140.0}
        
        status = orchestrator.get_orchestrator_status()
        
        assert "mode" in status
        assert "system_state" in status
        assert "active_optimizations" in status
        assert "completed_optimizations" in status
        assert "metrics_collected" in status
        assert "predictions" in status
        assert "latest_metrics" in status
        assert "performance_baseline" in status
        assert "uptime_seconds" in status
        
        assert status["mode"] == OrchestratorMode.ADAPTIVE.value
        assert status["system_state"] == SystemState.OPTIMAL.value
        assert status["metrics_collected"] == 1
        assert status["predictions"]["response_time_ms"] == 140.0

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self, orchestrator):
        """Test orchestrator start/stop lifecycle"""
        
        # Mock dependencies
        orchestrator.quantum_optimizer = Mock()
        orchestrator.realtime_intelligence = Mock()
        
        # Mock background task methods
        with patch.object(orchestrator, '_metrics_collection_loop', new_callable=AsyncMock):
            with patch.object(orchestrator, '_real_time_analysis_loop', new_callable=AsyncMock):
                with patch.object(orchestrator, '_optimization_execution_loop', new_callable=AsyncMock):
                    with patch.object(orchestrator, '_prediction_engine_loop', new_callable=AsyncMock):
                        with patch.object(orchestrator, '_self_healing_loop', new_callable=AsyncMock):
                            with patch.object(orchestrator, '_establish_performance_baseline', new_callable=AsyncMock):
                                
                                # Test start
                                await orchestrator.start()
                                assert len(orchestrator._background_tasks) == 5
                                
                                # Test stop
                                await orchestrator.stop()
                                assert orchestrator._running is False

    def test_singleton_pattern(self):
        """Test singleton pattern for global instance"""
        
        instance1 = get_quantum_realtime_orchestrator()
        instance2 = get_quantum_realtime_orchestrator()
        
        assert instance1 is instance2
        assert isinstance(instance1, QuantumRealtimeOrchestrator)


class TestSystemMetrics:
    """Test SystemMetrics data structure"""

    def test_system_metrics_creation(self):
        """Test SystemMetrics instantiation"""
        
        timestamp = datetime.utcnow()
        metrics = SystemMetrics(
            timestamp=timestamp,
            response_time_ms=150.0,
            throughput_rps=75.0,
            memory_usage_percent=65.0,
            cpu_utilization_percent=45.0,
            error_rate_percent=0.5,
            active_connections=50,
            cache_hit_rate_percent=88.0,
            queue_depth=5
        )
        
        assert metrics.timestamp == timestamp
        assert metrics.response_time_ms == 150.0
        assert metrics.throughput_rps == 75.0
        assert metrics.memory_usage_percent == 65.0
        assert metrics.cpu_utilization_percent == 45.0
        assert metrics.error_rate_percent == 0.5
        assert metrics.active_connections == 50
        assert metrics.cache_hit_rate_percent == 88.0
        assert metrics.queue_depth == 5


class TestOptimizationAction:
    """Test OptimizationAction data structure"""

    def test_optimization_action_creation(self):
        """Test OptimizationAction instantiation"""
        
        action = OptimizationAction(
            target=OptimizationTarget.RESPONSE_TIME,
            strategy=OptimizationStrategy.QUANTUM_ANNEALING,
            parameters={"test": "value"},
            expected_improvement=0.3,
            confidence=0.8,
            priority=Priority.HIGH
        )
        
        assert action.target == OptimizationTarget.RESPONSE_TIME
        assert action.strategy == OptimizationStrategy.QUANTUM_ANNEALING
        assert action.parameters == {"test": "value"}
        assert action.expected_improvement == 0.3
        assert action.confidence == 0.8
        assert action.priority == Priority.HIGH
        assert action.executed_at is None
        assert action.result is None
        assert isinstance(action.created_at, datetime)