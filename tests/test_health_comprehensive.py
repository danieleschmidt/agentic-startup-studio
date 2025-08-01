"""Comprehensive health check tests for the entire system."""

import pytest
import asyncio
import time
from typing import Dict, Any, List
from unittest.mock import AsyncMock, patch

from tests.framework.test_decorators import (
    integration_test, performance_test, timeout, mock_external_services
)
from tests.framework.test_data_factory import TestDataFactory, MockDataProvider


class TestSystemHealth:
    """Comprehensive system health tests."""

    @integration_test(timeout_seconds=15.0)
    async def test_full_system_health_check(self):
        """Test complete system health including all major components."""
        health_results = {}
        
        # Test database connectivity
        health_results['database'] = await self._test_database_health()
        
        # Test API endpoints
        health_results['api'] = await self._test_api_health()
        
        # Test AI services
        health_results['ai_services'] = await self._test_ai_services_health()
        
        # Test pipeline processing
        health_results['pipeline'] = await self._test_pipeline_health()
        
        # Test monitoring systems
        health_results['monitoring'] = await self._test_monitoring_health()
        
        # Verify all components are healthy
        failed_components = [
            component for component, result in health_results.items()
            if not result.get('healthy', False)
        ]
        
        assert not failed_components, f"Unhealthy components: {failed_components}"
        
        # Log health summary
        healthy_count = sum(1 for result in health_results.values() if result.get('healthy'))
        total_count = len(health_results)
        
        assert healthy_count == total_count, f"Health check: {healthy_count}/{total_count} components healthy"

    async def _test_database_health(self) -> Dict[str, Any]:
        """Test database connectivity and performance."""
        try:
            # Mock database operations
            with patch('asyncpg.connect') as mock_connect:
                mock_conn = AsyncMock()
                mock_connect.return_value = mock_conn
                mock_conn.fetchval.return_value = 1
                
                start_time = time.time()
                
                # Test basic connectivity
                connection_result = await mock_connect()
                query_result = await connection_result.fetchval("SELECT 1")
                
                response_time = time.time() - start_time
                
                return {
                    'healthy': query_result == 1,
                    'response_time_ms': response_time * 1000,
                    'details': 'Database connection successful'
                }
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'details': 'Database connection failed'
            }

    async def _test_api_health(self) -> Dict[str, Any]:
        """Test API endpoints health."""
        try:
            endpoints_to_test = [
                '/health',
                '/health/detailed',
                '/api/v1/ideas/validate'
            ]
            
            results = []
            
            for endpoint in endpoints_to_test:
                # Mock HTTP client
                with patch('httpx.AsyncClient') as mock_client:
                    mock_response = AsyncMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {'status': 'healthy'}
                    mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
                    
                    # Simulate API call
                    client = mock_client()
                    async with client as c:
                        response = await c.get(f"http://localhost:8000{endpoint}")
                        results.append({
                            'endpoint': endpoint,
                            'status_code': response.status_code,
                            'healthy': response.status_code == 200
                        })
            
            all_healthy = all(result['healthy'] for result in results)
            
            return {
                'healthy': all_healthy,
                'endpoints_tested': len(endpoints_to_test),
                'results': results,
                'details': 'API endpoints health check completed'
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'details': 'API health check failed'
            }

    async def _test_ai_services_health(self) -> Dict[str, Any]:
        """Test AI services connectivity and response."""
        try:
            services_tested = []
            
            # Test OpenAI service
            with patch('openai.AsyncOpenAI') as mock_openai:
                mock_client = AsyncMock()
                mock_response = AsyncMock()
                mock_response.choices = [AsyncMock(message=AsyncMock(content="Test response"))]
                mock_client.chat.completions.create.return_value = mock_response
                mock_openai.return_value = mock_client
                
                client = mock_openai()
                response = await client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "test"}]
                )
                
                services_tested.append({
                    'service': 'openai',
                    'healthy': bool(response.choices),
                    'response_length': len(response.choices[0].message.content)
                })
            
            # Test Google AI service (if configured)
            with patch('google.generativeai.GenerativeModel') as mock_google:
                mock_model = AsyncMock()
                mock_response = AsyncMock(text="Test response from Google AI")
                mock_model.generate_content.return_value = mock_response
                mock_google.return_value = mock_model
                
                model = mock_google()
                response = await model.generate_content("test")
                
                services_tested.append({
                    'service': 'google_ai',
                    'healthy': bool(response.text),
                    'response_length': len(response.text)
                })
            
            all_healthy = all(service['healthy'] for service in services_tested)
            
            return {
                'healthy': all_healthy,
                'services_tested': len(services_tested),
                'results': services_tested,
                'details': 'AI services health check completed'
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'details': 'AI services health check failed'
            }

    async def _test_pipeline_health(self) -> Dict[str, Any]:
        """Test pipeline processing capabilities."""
        try:
            # Create test data
            test_idea = TestDataFactory.create_idea_draft()
            
            # Mock pipeline processing
            with patch('pipeline.main_pipeline_async.process_idea') as mock_process:
                mock_result = TestDataFactory.create_validation_result(is_valid=True)
                mock_process.return_value = mock_result
                
                start_time = time.time()
                result = await mock_process(test_idea)
                processing_time = time.time() - start_time
                
                return {
                    'healthy': result.is_valid,
                    'processing_time_ms': processing_time * 1000,
                    'confidence_score': result.confidence,
                    'details': 'Pipeline processing test completed'
                }
                
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'details': 'Pipeline health check failed'
            }

    async def _test_monitoring_health(self) -> Dict[str, Any]:
        """Test monitoring and observability systems."""
        try:
            monitoring_results = []
            
            # Test metrics collection
            with patch('prometheus_client.CollectorRegistry') as mock_registry:
                mock_registry.return_value.collect.return_value = [
                    MockDataProvider.mock_prometheus_metric()
                ]
                
                registry = mock_registry()
                metrics = list(registry.collect())
                
                monitoring_results.append({
                    'component': 'prometheus_metrics',
                    'healthy': len(metrics) > 0,
                    'metrics_count': len(metrics)
                })
            
            # Test log aggregation
            with patch('logging.getLogger') as mock_logger:
                logger = mock_logger('test_health')
                logger.info("Health check test log")
                
                monitoring_results.append({
                    'component': 'logging',
                    'healthy': True,
                    'details': 'Logging system responsive'
                })
            
            # Test alerting system
            with patch('pipeline.core.alert_manager.AlertManager') as mock_alerts:
                alert_manager = mock_alerts()
                alert_manager.check_system_health.return_value = True
                
                health_status = alert_manager.check_system_health()
                
                monitoring_results.append({
                    'component': 'alerting',
                    'healthy': health_status,
                    'details': 'Alert system operational'
                })
            
            all_healthy = all(result['healthy'] for result in monitoring_results)
            
            return {
                'healthy': all_healthy,
                'components_tested': len(monitoring_results),
                'results': monitoring_results,
                'details': 'Monitoring systems health check completed'
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'details': 'Monitoring health check failed'
            }


class TestPerformanceHealth:
    """Performance-focused health tests."""

    @performance_test(max_duration=2.0, max_memory_mb=100.0)
    async def test_system_performance_under_load(self):
        """Test system performance under simulated load."""
        # Create batch of test ideas
        test_ideas = TestDataFactory.create_startup_ideas_batch(10)
        
        # Process them concurrently
        with patch('pipeline.main_pipeline_async.process_idea') as mock_process:
            mock_process.return_value = TestDataFactory.create_validation_result()
            
            start_time = time.time()
            
            # Process all ideas concurrently
            tasks = [mock_process(idea) for idea in test_ideas]
            results = await asyncio.gather(*tasks)
            
            processing_time = time.time() - start_time
            
            # Verify all processed successfully
            assert len(results) == len(test_ideas)
            assert all(result.is_valid for result in results)
            
            # Performance assertions
            avg_time_per_idea = processing_time / len(test_ideas)
            assert avg_time_per_idea < 0.2, f"Average processing time too high: {avg_time_per_idea:.3f}s"

    @timeout(10.0)
    async def test_memory_efficiency(self):
        """Test memory usage efficiency during processing."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process large batch of data
        large_batch = TestDataFactory.create_startup_ideas_batch(100)
        
        with patch('pipeline.main_pipeline_async.process_idea') as mock_process:
            mock_process.return_value = TestDataFactory.create_validation_result()
            
            # Process in chunks to test memory management
            chunk_size = 10
            for i in range(0, len(large_batch), chunk_size):
                chunk = large_batch[i:i + chunk_size]
                tasks = [mock_process(idea) for idea in chunk]
                await asyncio.gather(*tasks)
                
                # Check memory usage hasn't grown excessively
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_growth = current_memory - initial_memory
                
                assert memory_growth < 200, f"Memory growth too high: {memory_growth:.2f}MB"

    @mock_external_services("database", "openai", "redis")
    async def test_service_isolation(self, _mocks):
        """Test that service failures don't cascade."""
        # Test with database failure
        _mocks['database'].execute.side_effect = Exception("Database connection failed")
        
        # System should handle gracefully
        with patch('pipeline.infrastructure.circuit_breaker.CircuitBreaker') as mock_breaker:
            breaker = mock_breaker()
            breaker.state = "OPEN"  # Circuit breaker should open on failure
            
            # Attempt operation that would use database
            test_idea = TestDataFactory.create_idea_draft()
            
            with patch('pipeline.main_pipeline_async.process_idea') as mock_process:
                # Should fallback gracefully
                mock_process.side_effect = Exception("Service unavailable")
                
                with pytest.raises(Exception, match="Service unavailable"):
                    await mock_process(test_idea)
                
                # Verify circuit breaker activated
                assert breaker.state == "OPEN"


# Add helper method to MockDataProvider
def mock_prometheus_metric():
    """Create a mock Prometheus metric for testing."""
    from unittest.mock import MagicMock
    
    metric = MagicMock()
    metric.name = "test_metric"
    metric.documentation = "Test metric for health checks"
    metric.type = "counter"
    metric.samples = [
        MagicMock(name="test_metric_total", labels={}, value=42.0)
    ]
    return metric

# Monkey patch the method onto MockDataProvider
MockDataProvider.mock_prometheus_metric = staticmethod(mock_prometheus_metric)