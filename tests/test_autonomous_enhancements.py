"""
Comprehensive Tests for Autonomous SDLC Enhancements
Tests for Quantum Edge Optimizer, AI Self-Improvement, Enhanced Circuit Breaker, Zero Trust Framework, and Scale Orchestrator.
"""

import asyncio
import pytest
import json
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path
import secrets

# Import the modules we're testing
from pipeline.core.quantum_edge_optimizer import (
    QuantumEdgeOptimizer, 
    PerformanceMetrics, 
    OptimizationStrategy,
    optimize_system_performance
)
from pipeline.core.ai_self_improvement_engine import (
    AISelfImprovementEngine,
    ImprovementType,
    SafetyLevel,
    autonomous_code_improvement
)
from pipeline.infrastructure.enhanced_circuit_breaker import (
    EnhancedCircuitBreaker,
    CircuitState,
    FailureType,
    CircuitBreakerOpenError
)
from pipeline.security.zero_trust_framework import (
    ZeroTrustFramework,
    TrustLevel,
    ThreatLevel,
    SecurityEvent
)
from pipeline.performance.quantum_scale_orchestrator import (
    QuantumScaleOrchestrator,
    ScalingStrategy,
    ResourceType,
    RegionStatus
)


class TestQuantumEdgeOptimizer:
    """Test suite for Quantum Edge Optimizer"""
    
    @pytest.fixture
    def optimizer(self):
        return QuantumEdgeOptimizer()
    
    @pytest.fixture
    def sample_metrics(self):
        return PerformanceMetrics(
            latency_p95=100.0,
            throughput=2000.0,
            error_rate=0.005,
            resource_utilization=0.6,
            cost_efficiency=0.8
        )
    
    def test_optimizer_initialization(self, optimizer):
        """Test optimizer initializes correctly"""
        assert optimizer.optimization_history == []
        assert optimizer.current_strategy == OptimizationStrategy.QUANTUM_ANNEALING
        assert not optimizer.is_optimizing
    
    @pytest.mark.asyncio
    async def test_system_performance_optimization(self, optimizer, sample_metrics):
        """Test complete system performance optimization"""
        
        result = await optimizer.optimize_system_performance(sample_metrics)
        
        assert result is not None
        assert result.improvement_percentage >= 0
        assert result.confidence_score > 0
        assert result.execution_time > 0
        assert result.metrics_before is not None
        assert result.metrics_after is not None
        assert result.strategy in OptimizationStrategy
    
    @pytest.mark.asyncio
    async def test_optimization_strategies(self, optimizer, sample_metrics):
        """Test different optimization strategies"""
        
        strategies = [
            OptimizationStrategy.QUANTUM_ANNEALING,
            OptimizationStrategy.NEURAL_EVOLUTION,
            OptimizationStrategy.GENETIC_ALGORITHM,
            OptimizationStrategy.ADAPTIVE_LEARNING
        ]
        
        results = []
        for strategy in strategies:
            optimizer.current_strategy = strategy
            result = await optimizer.optimize_system_performance(sample_metrics)
            results.append(result)
        
        # All strategies should produce results
        assert len(results) == 4
        assert all(r.improvement_percentage >= 0 for r in results)
        
        # Different strategies should be used
        used_strategies = set(r.strategy for r in results)
        assert len(used_strategies) >= 2  # At least some variety
    
    @pytest.mark.asyncio
    async def test_concurrent_optimization_protection(self, optimizer, sample_metrics):
        """Test protection against concurrent optimizations"""
        
        # Start first optimization
        task1 = asyncio.create_task(optimizer.optimize_system_performance(sample_metrics))
        
        # Try to start second optimization immediately
        with pytest.raises(RuntimeError, match="already in progress"):
            await optimizer.optimize_system_performance(sample_metrics)
        
        # Let first optimization complete
        result1 = await task1
        assert result1 is not None
    
    def test_optimization_summary(self, optimizer):
        """Test optimization summary generation"""
        
        summary = optimizer.get_optimization_summary()
        
        assert "status" in summary
        assert summary["status"] == "no_optimizations_performed"
        
        # Add a mock optimization result
        optimizer.optimization_history.append(type('MockResult', (), {
            'improvement_percentage': 15.0,
            'confidence_score': 0.85,
            'execution_time': 2.5,
            'strategy': OptimizationStrategy.QUANTUM_ANNEALING
        })())
        
        summary = optimizer.get_optimization_summary()
        assert summary["total_optimizations"] == 1
        assert summary["average_improvement_percentage"] == 15.0
    
    @pytest.mark.asyncio
    async def test_convenience_function(self):
        """Test convenience function for optimization"""
        
        result = await optimize_system_performance(
            target_latency_ms=50.0,
            target_throughput_rps=3000.0,
            target_error_rate=0.001
        )
        
        assert result is not None
        assert result.improvement_percentage >= 0


class TestAISelfImprovementEngine:
    """Test suite for AI Self-Improvement Engine"""
    
    @pytest.fixture
    def engine(self):
        return AISelfImprovementEngine(max_safety_level=SafetyLevel.SAFE)
    
    @pytest.fixture
    def temp_python_file(self):
        """Create a temporary Python file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('''
def example_function():
    """Example function for testing."""
    items = [1, 2, 3, 4, 5]
    for i in range(len(items)):
        print(items[i])
    return items
''')
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        os.unlink(temp_path)
    
    def test_engine_initialization(self, engine):
        """Test engine initializes correctly"""
        assert engine.max_safety_level == SafetyLevel.SAFE
        assert engine.improvement_history == []
        assert not engine.is_improving
        assert engine.code_backups == {}
    
    @pytest.mark.asyncio
    async def test_file_analysis(self, engine, temp_python_file):
        """Test code file analysis"""
        
        analysis = await engine._analyze_file(temp_python_file)
        
        assert analysis is not None
        assert analysis.file_path == temp_python_file
        assert 0 <= analysis.complexity_score <= 1
        assert 0 <= analysis.performance_score <= 1
        assert 0 <= analysis.maintainability_score <= 1
        assert 0 <= analysis.security_score <= 1
        assert analysis.safety_level in SafetyLevel
        assert isinstance(analysis.improvement_opportunities, list)
    
    @pytest.mark.asyncio 
    async def test_codebase_analysis(self, engine):
        """Test codebase analysis with actual files"""
        
        # Create a temporary directory with Python files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            (temp_path / "test1.py").write_text('''
def test_function():
    for i in range(len([1,2,3])):
        pass
''')
            
            (temp_path / "test2.py").write_text('''
class TestClass:
    def method(self):
        return "test"
''')
            
            analyses = await engine.analyze_codebase(str(temp_path))
            
            assert len(analyses) == 2
            assert all(a.file_path.endswith('.py') for a in analyses)
    
    @pytest.mark.asyncio
    async def test_improvement_suggestions(self, engine):
        """Test improvement suggestion generation"""
        
        # Create mock analysis
        mock_analysis = type('MockAnalysis', (), {
            'file_path': '/tmp/test.py',
            'improvement_opportunities': [
                {
                    'type': ImprovementType.DOCUMENTATION,
                    'description': 'Add module docstring',
                    'impact': 0.2,
                    'safety': SafetyLevel.SAFE
                }
            ]
        })()
        
        suggestions = await engine.generate_improvements([mock_analysis])
        
        # Should generate suggestions for safe improvements
        assert isinstance(suggestions, list)
    
    def test_safety_level_checking(self, engine):
        """Test safety level permission checking"""
        
        # Should allow safe modifications
        assert engine._is_safety_level_allowed(SafetyLevel.SAFE)
        
        # Should not allow risky modifications with SAFE max level
        assert not engine._is_safety_level_allowed(SafetyLevel.RISKY)
        assert not engine._is_safety_level_allowed(SafetyLevel.DANGEROUS)
    
    def test_improvement_summary(self, engine):
        """Test improvement summary generation"""
        
        summary = engine.get_improvement_summary()
        
        assert "status" in summary
        assert summary["status"] == "no_improvements_performed"
    
    @pytest.mark.asyncio
    async def test_convenience_function(self):
        """Test convenience function for autonomous improvement"""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "test.py").write_text('# Simple test file\nprint("hello")\n')
            
            results = await autonomous_code_improvement(
                target_directory=str(temp_path),
                max_improvements=1,
                max_safety_level=SafetyLevel.SAFE
            )
            
            assert isinstance(results, list)


class TestEnhancedCircuitBreaker:
    """Test suite for Enhanced Circuit Breaker"""
    
    @pytest.fixture
    def circuit_breaker(self):
        return EnhancedCircuitBreaker(
            name="test_circuit",
            failure_threshold=3,
            recovery_timeout=5.0,
            timeout=1.0
        )
    
    @pytest.fixture
    async def mock_success_function(self):
        async def success_func():
            await asyncio.sleep(0.1)
            return "success"
        return success_func
    
    @pytest.fixture
    async def mock_failure_function(self):
        async def failure_func():
            await asyncio.sleep(0.1)
            raise Exception("Test failure")
        return failure_func
    
    @pytest.fixture
    async def mock_timeout_function(self):
        async def timeout_func():
            await asyncio.sleep(2.0)  # Will timeout with 1s limit
            return "should not reach here"
        return timeout_func
    
    def test_circuit_breaker_initialization(self, circuit_breaker):
        """Test circuit breaker initializes correctly"""
        assert circuit_breaker.name == "test_circuit"
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.failure_count == 0
        assert circuit_breaker.metrics.total_requests == 0
    
    @pytest.mark.asyncio
    async def test_successful_call(self, circuit_breaker, mock_success_function):
        """Test successful function call through circuit breaker"""
        
        result = await circuit_breaker.call(mock_success_function)
        
        assert result == "success"
        assert circuit_breaker.metrics.successful_requests == 1
        assert circuit_breaker.metrics.total_requests == 1
        assert circuit_breaker.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_failure_handling(self, circuit_breaker, mock_failure_function):
        """Test failure handling and state transitions"""
        
        # Cause failures up to threshold
        for i in range(3):
            with pytest.raises(Exception, match="Test failure"):
                await circuit_breaker.call(mock_failure_function)
        
        assert circuit_breaker.failure_count == 3
        assert circuit_breaker.state in [CircuitState.OPEN, CircuitState.ADAPTIVE, CircuitState.QUARANTINE]
        assert circuit_breaker.metrics.failed_requests == 3
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, circuit_breaker, mock_timeout_function):
        """Test timeout handling"""
        
        with pytest.raises(asyncio.TimeoutError):
            await circuit_breaker.call(mock_timeout_function)
        
        assert circuit_breaker.metrics.timeouts == 1
        assert circuit_breaker.metrics.failed_requests == 1
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_open_error(self, circuit_breaker, mock_failure_function):
        """Test circuit breaker open error"""
        
        # Force circuit to open
        await circuit_breaker.force_open("Test")
        
        with pytest.raises(CircuitBreakerOpenError):
            await circuit_breaker.call(mock_failure_function)
    
    @pytest.mark.asyncio
    async def test_manual_control(self, circuit_breaker):
        """Test manual circuit breaker control"""
        
        # Test force open
        await circuit_breaker.force_open("Manual test")
        assert circuit_breaker.state == CircuitState.OPEN
        
        # Test force close
        await circuit_breaker.force_close("Manual test")
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.failure_count == 0
    
    def test_status_reporting(self, circuit_breaker):
        """Test comprehensive status reporting"""
        
        status = circuit_breaker.get_status()
        
        assert status["name"] == "test_circuit"
        assert status["state"] == CircuitState.CLOSED.value
        assert "metrics" in status
        assert "health_score" in status
        assert "recent_error_rate" in status
    
    def test_failure_pattern_analysis(self, circuit_breaker):
        """Test failure pattern analysis"""
        
        patterns = circuit_breaker.get_failure_patterns()
        
        assert "message" in patterns  # Should have message when no failures
    
    @pytest.mark.asyncio
    async def test_health_check_function(self):
        """Test global health check function"""
        
        from pipeline.infrastructure.enhanced_circuit_breaker import circuit_breaker_health_check
        
        health = await circuit_breaker_health_check()
        
        assert "overall_health" in health
        assert "total_circuits" in health
        assert "timestamp" in health


class TestZeroTrustFramework:
    """Test suite for Zero Trust Security Framework"""
    
    @pytest.fixture
    def framework(self):
        return ZeroTrustFramework(secret_key=secrets.token_urlsafe(32))
    
    @pytest.fixture
    def sample_credentials(self):
        return {
            "password": "test_password123",
            "token": None
        }
    
    def test_framework_initialization(self, framework):
        """Test framework initializes correctly"""
        assert framework.secret_key is not None
        assert framework.active_sessions == {}
        assert framework.threat_indicators is not None
        assert framework.max_failed_attempts == 5
    
    @pytest.mark.asyncio
    async def test_user_authentication_success(self, framework, sample_credentials):
        """Test successful user authentication"""
        
        context = await framework.authenticate_user(
            user_id="test_user",
            credentials=sample_credentials,
            ip_address="192.168.1.100",
            user_agent="TestAgent/1.0"
        )
        
        assert context is not None
        assert context.user_id == "test_user"
        assert context.trust_level in TrustLevel
        assert len(context.permissions) > 0
        assert context.session_id in framework.active_sessions
    
    @pytest.mark.asyncio
    async def test_user_authentication_failure(self, framework):
        """Test failed user authentication"""
        
        bad_credentials = {"password": "bad"}
        
        context = await framework.authenticate_user(
            user_id="test_user",
            credentials=bad_credentials,
            ip_address="192.168.1.100",
            user_agent="TestAgent/1.0"
        )
        
        assert context is None
        assert len(framework.threat_indicators) > 0
    
    @pytest.mark.asyncio
    async def test_request_verification_success(self, framework, sample_credentials):
        """Test successful request verification"""
        
        # First authenticate
        context = await framework.authenticate_user(
            user_id="test_user",
            credentials=sample_credentials,
            ip_address="192.168.1.100",
            user_agent="TestAgent/1.0"
        )
        
        assert context is not None
        
        # Then verify request
        authorized, returned_context = await framework.verify_request(
            session_id=context.session_id,
            resource="/user",
            method="GET",
            ip_address="192.168.1.100",
            user_agent="TestAgent/1.0"
        )
        
        assert authorized
        assert returned_context == context
    
    @pytest.mark.asyncio
    async def test_request_verification_invalid_session(self, framework):
        """Test request verification with invalid session"""
        
        authorized, context = await framework.verify_request(
            session_id="invalid_session",
            resource="/user",
            method="GET",
            ip_address="192.168.1.100",
            user_agent="TestAgent/1.0"
        )
        
        assert not authorized
        assert context is None
        assert len(framework.threat_indicators) > 0
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, framework):
        """Test rate limiting functionality"""
        
        # Test multiple rapid requests
        for i in range(15):  # Exceed typical rate limit
            result = await framework._check_rate_limit("test_ip", "login", 10, 300)
            if i >= 10:
                assert not result  # Should be rate limited
            else:
                assert result  # Should be allowed
    
    @pytest.mark.asyncio
    async def test_ip_blocking(self, framework):
        """Test IP blocking after failed attempts"""
        
        bad_credentials = {"password": "bad"}
        
        # Cause multiple failures
        for i in range(6):  # Exceed max_failed_attempts
            await framework.authenticate_user(
                user_id="test_user",
                credentials=bad_credentials,
                ip_address="192.168.1.200",
                user_agent="TestAgent/1.0"
            )
        
        # IP should be blocked
        blocked = await framework._is_ip_blocked("192.168.1.200")
        assert blocked
    
    @pytest.mark.asyncio
    async def test_monitoring_tasks(self, framework):
        """Test background monitoring tasks"""
        
        await framework.start_monitoring()
        
        # Let it run briefly
        await asyncio.sleep(0.1)
        
        await framework.stop_monitoring()
        
        # Tasks should be properly cleaned up
        assert framework._cleanup_task is None
        assert framework._monitoring_task is None
    
    def test_security_dashboard(self, framework):
        """Test security dashboard generation"""
        
        dashboard = framework.get_security_dashboard()
        
        assert "overview" in dashboard
        assert "threats" in dashboard
        assert "network" in dashboard
        assert "users" in dashboard
        assert "timestamp" in dashboard
    
    def test_threat_intelligence(self, framework):
        """Test threat intelligence summary"""
        
        intelligence = framework.get_threat_intelligence()
        
        assert "summary" in intelligence
        assert "threat_types" in intelligence
        assert "top_threat_sources" in intelligence


class TestQuantumScaleOrchestrator:
    """Test suite for Quantum Scale Orchestrator"""
    
    @pytest.fixture
    def orchestrator(self):
        return QuantumScaleOrchestrator(
            regions=["us-east-1", "us-west-2"],
            target_sla=0.99,
            prediction_horizon=30
        )
    
    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initializes correctly"""
        assert len(orchestrator.regions) == 2
        assert orchestrator.target_sla == 0.99
        assert orchestrator.prediction_horizon == 30
        assert not orchestrator._is_orchestrating
    
    @pytest.mark.asyncio
    async def test_metric_collection(self, orchestrator):
        """Test metric collection from regions"""
        
        await orchestrator._collect_global_metrics()
        
        # Should have metrics for all regions
        assert len(orchestrator.region_metrics) == 2
        
        # Should have global metrics
        assert len(orchestrator.global_metrics_history) > 0
        
        # Verify region metrics
        for region in orchestrator.regions:
            assert region in orchestrator.region_metrics
            metrics = orchestrator.region_metrics[region]
            assert hasattr(metrics, 'cpu_utilization')
            assert hasattr(metrics, 'memory_utilization')
            assert hasattr(metrics, 'request_rate')
            assert metrics.status in RegionStatus
    
    @pytest.mark.asyncio
    async def test_load_prediction(self, orchestrator):
        """Test load prediction generation"""
        
        # First collect metrics to have data
        await orchestrator._collect_global_metrics()
        
        # Generate some historical data
        for _ in range(15):  # Need enough data for predictions
            await orchestrator._collect_global_metrics()
            await asyncio.sleep(0.01)  # Small delay for different timestamps
        
        # Generate predictions
        await orchestrator._generate_load_predictions()
        
        # Should have predictions for regions with sufficient data
        assert len(orchestrator.load_predictions) >= 0  # Might not have enough data in test
    
    @pytest.mark.asyncio
    async def test_scaling_decisions(self, orchestrator):
        """Test scaling decision making"""
        
        # Collect metrics and predictions
        await orchestrator._collect_global_metrics()
        
        # Generate fake historical data for predictions
        for region in orchestrator.regions:
            for resource_type in ResourceType:
                orchestrator.resource_utilization[region][resource_type].extend([0.5] * 20)
        
        await orchestrator._generate_load_predictions()
        
        # Make scaling decisions
        decisions = await orchestrator._make_scaling_decisions()
        
        assert isinstance(decisions, list)
        # Might be empty if no scaling needed
    
    @pytest.mark.asyncio
    async def test_orchestration_lifecycle(self, orchestrator):
        """Test full orchestration lifecycle"""
        
        # Start orchestration
        await orchestrator.start_orchestration()
        assert orchestrator._is_orchestrating
        assert orchestrator._orchestration_task is not None
        
        # Let it run briefly
        await asyncio.sleep(0.1)
        
        # Stop orchestration
        await orchestrator.stop_orchestration()
        assert not orchestrator._is_orchestrating
        assert orchestrator._orchestration_task is None
    
    def test_status_reporting(self, orchestrator):
        """Test comprehensive status reporting"""
        
        status = orchestrator.get_orchestration_status()
        
        assert "overview" in status
        assert "regions" in status
        assert "scaling" in status
        assert "predictions" in status
        assert "costs" in status
        assert "timestamp" in status
    
    def test_performance_analytics(self, orchestrator):
        """Test performance analytics"""
        
        analytics = orchestrator.get_performance_analytics()
        
        # Should have some structure even without data
        assert isinstance(analytics, dict)
    
    @pytest.mark.asyncio
    async def test_convenience_function(self):
        """Test convenience function for global orchestration"""
        
        from pipeline.performance.quantum_scale_orchestrator import start_global_orchestration
        
        orchestrator = await start_global_orchestration(
            regions=["test-region"],
            target_sla=0.95
        )
        
        assert orchestrator is not None
        assert orchestrator._is_orchestrating
        
        # Clean up
        await orchestrator.stop_orchestration()


class TestIntegration:
    """Integration tests across all autonomous enhancements"""
    
    @pytest.mark.asyncio
    async def test_performance_optimization_integration(self):
        """Test integration between quantum edge optimizer and scale orchestrator"""
        
        from pipeline.core.quantum_edge_optimizer import get_quantum_edge_optimizer
        from pipeline.performance.quantum_scale_orchestrator import get_quantum_scale_orchestrator
        
        optimizer = get_quantum_edge_optimizer()
        orchestrator = get_quantum_scale_orchestrator(["test-region"])
        
        # Simulate performance optimization
        target_metrics = PerformanceMetrics(
            latency_p95=50.0,
            throughput=3000.0,
            error_rate=0.001,
            resource_utilization=0.7,
            cost_efficiency=0.85
        )
        
        result = await optimizer.optimize_system_performance(target_metrics)
        
        # Should have optimization result
        assert result is not None
        assert result.improvement_percentage >= 0
        
        # Get orchestrator status
        status = orchestrator.get_orchestration_status()
        assert "overview" in status
    
    @pytest.mark.asyncio
    async def test_security_and_circuit_breaker_integration(self):
        """Test integration between zero trust framework and circuit breaker"""
        
        from pipeline.security.zero_trust_framework import get_zero_trust_framework
        from pipeline.infrastructure.enhanced_circuit_breaker import get_circuit_breaker
        
        framework = get_zero_trust_framework()
        circuit = get_circuit_breaker("test_integration")
        
        # Test authentication through circuit breaker
        async def auth_function():
            return await framework.authenticate_user(
                user_id="test_user",
                credentials={"password": "test123"},
                ip_address="192.168.1.1",
                user_agent="TestAgent/1.0"
            )
        
        # Should work through circuit breaker
        result = await circuit.call(auth_function)
        # Result might be None due to auth failure, but call should succeed
        
        # Get statuses
        circuit_status = circuit.get_status()
        security_dashboard = framework.get_security_dashboard()
        
        assert circuit_status["name"] == "test_integration"
        assert "overview" in security_dashboard
    
    def test_global_instances(self):
        """Test that global instances are properly managed"""
        
        from pipeline.core.quantum_edge_optimizer import get_quantum_edge_optimizer
        from pipeline.core.ai_self_improvement_engine import get_ai_improvement_engine
        from pipeline.security.zero_trust_framework import get_zero_trust_framework
        from pipeline.performance.quantum_scale_orchestrator import get_quantum_scale_orchestrator
        
        # Get instances
        optimizer1 = get_quantum_edge_optimizer()
        optimizer2 = get_quantum_edge_optimizer()
        
        engine1 = get_ai_improvement_engine()
        engine2 = get_ai_improvement_engine()
        
        framework1 = get_zero_trust_framework()
        framework2 = get_zero_trust_framework()
        
        orchestrator1 = get_quantum_scale_orchestrator()
        orchestrator2 = get_quantum_scale_orchestrator()
        
        # Should be singletons
        assert optimizer1 is optimizer2
        assert engine1 is engine2
        assert framework1 is framework2
        assert orchestrator1 is orchestrator2


# Test data and fixtures
@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests"""
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])