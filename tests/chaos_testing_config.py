"""
Chaos testing configuration for enterprise resilience validation.

This module provides chaos engineering capabilities to test system
resilience under various failure conditions and ensure graceful
degradation and recovery.
"""

import pytest
import asyncio
import random
import time
from typing import Dict, Any, List, Callable, Optional
from unittest.mock import patch, MagicMock
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum


class ChaosType(Enum):
    """Types of chaos experiments."""
    NETWORK_LATENCY = "network_latency"
    NETWORK_FAILURE = "network_failure"
    DATABASE_SLOWDOWN = "database_slowdown"
    DATABASE_FAILURE = "database_failure"
    MEMORY_PRESSURE = "memory_pressure"
    CPU_SPIKE = "cpu_spike"
    API_RATE_LIMIT = "api_rate_limit"
    TIMEOUT_FAILURE = "timeout_failure"
    PARTIAL_SERVICE_FAILURE = "partial_service_failure"


@dataclass
class ChaosExperiment:
    """Configuration for a chaos experiment."""
    name: str
    chaos_type: ChaosType
    duration_seconds: float
    intensity: float  # 0.0 to 1.0
    target_component: str
    failure_rate: float = 0.1  # Probability of failure
    recovery_time: float = 1.0  # Time to recover
    metadata: Dict[str, Any] = None


class ChaosEngine:
    """Engine for executing chaos experiments."""
    
    def __init__(self):
        self.active_experiments: List[ChaosExperiment] = []
        self.experiment_results: Dict[str, Dict[str, Any]] = {}
    
    async def inject_network_latency(
        self, 
        target_function: Callable,
        delay_ms: int = 500,
        jitter_ms: int = 100
    ):
        """Inject network latency into target function."""
        original_function = target_function
        
        async def delayed_function(*args, **kwargs):
            # Add random delay
            delay = delay_ms + random.randint(-jitter_ms, jitter_ms)
            await asyncio.sleep(delay / 1000.0)
            return await original_function(*args, **kwargs)
        
        return delayed_function
    
    async def inject_intermittent_failures(
        self,
        target_function: Callable,
        failure_rate: float = 0.1,
        exception_type: type = ConnectionError
    ):
        """Inject intermittent failures into target function."""
        original_function = target_function
        
        async def failing_function(*args, **kwargs):
            if random.random() < failure_rate:
                raise exception_type("Chaos-induced failure")
            return await original_function(*args, **kwargs)
        
        return failing_function
    
    async def inject_resource_pressure(
        self,
        memory_mb: int = 100,
        duration_seconds: float = 5.0
    ):
        """Inject memory pressure to simulate resource constraints."""
        # Allocate memory to simulate pressure
        pressure_data = []
        
        try:
            # Gradually increase memory usage
            for _ in range(memory_mb):
                pressure_data.append(b"X" * (1024 * 1024))  # 1MB chunks
                await asyncio.sleep(0.01)
            
            # Hold memory for duration
            await asyncio.sleep(duration_seconds)
            
        finally:
            # Clean up
            del pressure_data
    
    @asynccontextmanager
    async def chaos_context(self, experiment: ChaosExperiment):
        """Context manager for running chaos experiments."""
        start_time = time.time()
        
        try:
            # Record experiment start
            self.active_experiments.append(experiment)
            self.experiment_results[experiment.name] = {
                "start_time": start_time,
                "status": "running",
                "errors": [],
                "metrics": {}
            }
            
            # Apply chaos based on type
            if experiment.chaos_type == ChaosType.NETWORK_LATENCY:
                # Network latency simulation would be applied here
                pass
            elif experiment.chaos_type == ChaosType.MEMORY_PRESSURE:
                # Start memory pressure in background
                asyncio.create_task(
                    self.inject_resource_pressure(
                        memory_mb=int(experiment.intensity * 200),
                        duration_seconds=experiment.duration_seconds
                    )
                )
            
            yield self
            
        except Exception as e:
            # Record experiment error
            self.experiment_results[experiment.name]["errors"].append(str(e))
            self.experiment_results[experiment.name]["status"] = "error"
            raise
            
        finally:
            # Record experiment completion
            end_time = time.time()
            self.experiment_results[experiment.name].update({
                "end_time": end_time,
                "duration": end_time - start_time,
                "status": "completed" if self.experiment_results[experiment.name]["status"] != "error" else "error"
            })
            
            # Remove from active experiments
            if experiment in self.active_experiments:
                self.active_experiments.remove(experiment)
    
    def get_experiment_results(self, experiment_name: str) -> Optional[Dict[str, Any]]:
        """Get results for a specific experiment."""
        return self.experiment_results.get(experiment_name)
    
    def get_all_results(self) -> Dict[str, Dict[str, Any]]:
        """Get all experiment results."""
        return self.experiment_results.copy()


class ResilienceValidator:
    """Validates system resilience under chaos conditions."""
    
    def __init__(self):
        self.validation_results: Dict[str, Any] = {}
    
    async def validate_graceful_degradation(
        self,
        service_function: Callable,
        chaos_experiment: ChaosExperiment,
        expected_fallback_behavior: Callable = None
    ) -> Dict[str, Any]:
        """Validate that service degrades gracefully under chaos."""
        results = {
            "graceful_degradation": False,
            "fallback_triggered": False,
            "error_rate": 0.0,
            "response_times": [],
            "errors": []
        }
        
        chaos_engine = ChaosEngine()
        
        # Run experiment
        async with chaos_engine.chaos_context(chaos_experiment):
            # Test service under chaos
            attempts = 10
            errors = 0
            
            for i in range(attempts):
                start_time = time.time()
                
                try:
                    result = await service_function()
                    
                    # Check if fallback was used
                    if expected_fallback_behavior and callable(expected_fallback_behavior):
                        if await expected_fallback_behavior(result):
                            results["fallback_triggered"] = True
                    
                except Exception as e:
                    errors += 1
                    results["errors"].append(str(e))
                
                end_time = time.time()
                results["response_times"].append(end_time - start_time)
                
                await asyncio.sleep(0.1)  # Brief pause between attempts
        
        # Calculate metrics
        results["error_rate"] = errors / attempts
        results["graceful_degradation"] = results["error_rate"] < 0.5  # Less than 50% errors
        
        return results
    
    async def validate_recovery_behavior(
        self,
        service_function: Callable,
        chaos_duration: float = 5.0,
        recovery_check_interval: float = 1.0,
        max_recovery_time: float = 30.0
    ) -> Dict[str, Any]:
        """Validate that service recovers after chaos ends."""
        results = {
            "recovered": False,
            "recovery_time": 0.0,
            "pre_chaos_baseline": None,
            "post_chaos_performance": None,
            "errors": []
        }
        
        # Establish baseline
        try:
            baseline_start = time.time()
            await service_function()
            results["pre_chaos_baseline"] = time.time() - baseline_start
        except Exception as e:
            results["errors"].append(f"Baseline error: {str(e)}")
            return results
        
        # Apply chaos
        chaos_experiment = ChaosExperiment(
            name="recovery_test",
            chaos_type=ChaosType.NETWORK_FAILURE,
            duration_seconds=chaos_duration,
            intensity=0.8,
            target_component="test_service"
        )
        
        chaos_engine = ChaosEngine()
        
        async with chaos_engine.chaos_context(chaos_experiment):
            await asyncio.sleep(chaos_duration)
        
        # Test recovery
        recovery_start = time.time()
        
        while time.time() - recovery_start < max_recovery_time:
            try:
                test_start = time.time()
                await service_function()
                test_duration = time.time() - test_start
                
                # Service is considered recovered if it responds within
                # 2x the baseline time
                if test_duration <= results["pre_chaos_baseline"] * 2:
                    results["recovered"] = True
                    results["recovery_time"] = time.time() - recovery_start
                    results["post_chaos_performance"] = test_duration
                    break
                    
            except Exception as e:
                results["errors"].append(f"Recovery test error: {str(e)}")
            
            await asyncio.sleep(recovery_check_interval)
        
        return results


# Pytest fixtures for chaos testing
@pytest.fixture
def chaos_engine():
    """Provide chaos engineering engine."""
    return ChaosEngine()


@pytest.fixture
def resilience_validator():
    """Provide resilience validation utilities."""
    return ResilienceValidator()


@pytest.fixture
def standard_chaos_experiments():
    """Provide standard set of chaos experiments."""
    return [
        ChaosExperiment(
            name="network_latency_test",
            chaos_type=ChaosType.NETWORK_LATENCY,
            duration_seconds=10.0,
            intensity=0.5,
            target_component="api_client"
        ),
        ChaosExperiment(
            name="database_slowdown_test",
            chaos_type=ChaosType.DATABASE_SLOWDOWN,
            duration_seconds=15.0,
            intensity=0.7,
            target_component="database"
        ),
        ChaosExperiment(
            name="memory_pressure_test",
            chaos_type=ChaosType.MEMORY_PRESSURE,
            duration_seconds=20.0,
            intensity=0.6,
            target_component="application"
        ),
        ChaosExperiment(
            name="partial_failure_test",
            chaos_type=ChaosType.PARTIAL_SERVICE_FAILURE,
            duration_seconds=12.0,
            intensity=0.3,
            target_component="external_apis",
            failure_rate=0.2
        )
    ]


@pytest.fixture
def mock_resilient_service():
    """Provide mock service with resilience features."""
    class MockResilientService:
        def __init__(self):
            self.fallback_count = 0
            self.primary_failures = 0
            self.circuit_breaker_open = False
        
        async def primary_operation(self):
            """Primary operation that can fail."""
            if self.circuit_breaker_open:
                return await self.fallback_operation()
            
            # Simulate potential failure
            if random.random() < 0.1:  # 10% failure rate
                self.primary_failures += 1
                if self.primary_failures >= 3:
                    self.circuit_breaker_open = True
                raise ConnectionError("Primary service failure")
            
            return {"status": "success", "data": "primary_result"}
        
        async def fallback_operation(self):
            """Fallback operation for resilience."""
            self.fallback_count += 1
            await asyncio.sleep(0.1)  # Simulate fallback processing
            return {"status": "fallback", "data": "cached_result"}
        
        async def health_check(self):
            """Health check for recovery testing."""
            if self.circuit_breaker_open and self.fallback_count > 5:
                # Simulate recovery after enough fallback operations
                self.circuit_breaker_open = False
                self.primary_failures = 0
            
            return not self.circuit_breaker_open
    
    return MockResilientService()


# Custom pytest markers for chaos testing
pytest_chaos_markers = [
    "chaos: Chaos engineering tests",
    "resilience: Resilience validation tests",
    "recovery: Recovery behavior tests",
    "stress: Stress testing under adverse conditions",
    "fault_tolerance: Fault tolerance validation"
]


# Export for easy importing
__all__ = [
    'ChaosType',
    'ChaosExperiment', 
    'ChaosEngine',
    'ResilienceValidator',
    'chaos_engine',
    'resilience_validator',
    'standard_chaos_experiments',
    'mock_resilient_service'
]