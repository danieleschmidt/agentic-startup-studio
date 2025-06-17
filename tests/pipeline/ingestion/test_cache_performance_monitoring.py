"""
Acceptance tests for cache layer with integrated performance monitoring.

Tests the high-level behavior of the cache system including:
- Multi-level caching (L1 memory, L2 Redis)
- Performance metrics collection and monitoring
- Cache hit rate tracking and alerting
- Feedback loop integration for optimization
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4
from typing import Dict, Any, List, Tuple, Optional

from pipeline.models.idea import IdeaDraft, DuplicateCheckResult
from pipeline.config.settings import ValidationConfig


class TestCacheLayerPerformanceMonitoring:
    """Acceptance tests for cache layer with performance monitoring."""

    @pytest.fixture
    def mock_metrics_collector(self):
        """Mock metrics collector for performance monitoring."""
        collector = Mock()
        collector.record_cache_hit = Mock()
        collector.record_cache_miss = Mock()
        collector.record_operation_duration = Mock()
        collector.get_cache_hit_rate = Mock(return_value=0.85)
        return collector

    @pytest.fixture
    def mock_redis_client(self):
        """Mock Redis client for L2 cache."""
        client = AsyncMock()
        client.get = AsyncMock(return_value=None)
        client.set = AsyncMock()
        client.delete = AsyncMock()
        return client

    @pytest.fixture
    def sample_idea_draft(self):
        """Sample idea draft for testing."""
        return IdeaDraft(
            title="AI-powered startup accelerator",
            description="Platform that uses AI to help startups validate and scale their ideas",
            category="ai_ml",
            problem_statement="Startups fail due to lack of validation",
            solution_description="AI-driven validation platform",
            target_market="Early-stage startups",
            evidence_links=["https://example.com/research"]
        )

    @pytest.fixture
    def expected_duplicate_result(self):
        """Expected duplicate check result for caching."""
        return DuplicateCheckResult(
            found_similar=True,
            similar_ideas=[uuid4(), uuid4()],
            similarity_scores={"id1": 0.85, "id2": 0.72},
            exact_matches=[]
        )

    @pytest.mark.asyncio
    async def test_cache_enabled_duplicate_detection_performance_monitoring(
        self, sample_idea_draft, expected_duplicate_result, mock_metrics_collector, mock_redis_client
    ):
        """
        GIVEN a cache-enabled duplicate detector with performance monitoring
        WHEN checking for duplicates with caching enabled
        THEN it should use multi-level cache and record performance metrics
        """
        # This test will fail initially - we need to implement the cache layer
        from pipeline.ingestion.cache.cache_manager import CacheManager
        from pipeline.ingestion.monitoring.metrics_collector import MetricsCollector
        from pipeline.ingestion.duplicate_detector import CacheableDuplicateDetector
        
        # Setup cache manager with performance monitoring
        cache_manager = CacheManager(
            redis_client=mock_redis_client,
            metrics_collector=mock_metrics_collector,
            lru_size=1000
        )
        
        # Setup duplicate detector with cache integration
        mock_repository = AsyncMock()
        config = ValidationConfig()
        
        detector = CacheableDuplicateDetector(
            repository=mock_repository,
            cache_manager=cache_manager,
            metrics_collector=mock_metrics_collector,
            config=config
        )
        
        # First call - should miss cache and record metrics
        result1 = await detector.check_for_duplicates(sample_idea_draft, use_cache=True)
        
        # Verify cache miss was recorded
        mock_metrics_collector.record_cache_miss.assert_called_once_with("duplicate_check")
        
        # Verify operation duration was recorded
        mock_metrics_collector.record_operation_duration.assert_called()
        
        # Second call with same draft - should hit L1 cache
        result2 = await detector.check_for_duplicates(sample_idea_draft, use_cache=True)
        
        # Verify cache hit was recorded
        mock_metrics_collector.record_cache_hit.assert_called_with("duplicate_check")

    @pytest.mark.asyncio
    async def test_cache_hit_rate_monitoring_and_alerting(
        self, mock_metrics_collector, mock_redis_client
    ):
        """
        GIVEN a cache manager with hit rate monitoring
        WHEN cache hit rate falls below threshold
        THEN it should trigger performance alerts and optimization feedback
        """
        from pipeline.ingestion.cache.cache_manager import CacheManager
        from pipeline.ingestion.monitoring.performance_monitor import PerformanceMonitor
        
        # Setup performance monitor with alerting
        mock_alert_manager = Mock()
        mock_alert_manager.trigger_cache_performance_alert = Mock()
        
        performance_monitor = PerformanceMonitor(
            metrics_collector=mock_metrics_collector,
            alert_manager=mock_alert_manager,
            cache_hit_rate_threshold=0.8
        )
        
        cache_manager = CacheManager(
            redis_client=mock_redis_client,
            metrics_collector=mock_metrics_collector,
            performance_monitor=performance_monitor
        )
        
        # Simulate low cache hit rate (below 80% threshold)
        mock_metrics_collector.get_cache_hit_rate.return_value = 0.65
        
        # Trigger performance check
        await performance_monitor.check_cache_performance()
        
        # Verify alert was triggered
        mock_alert_manager.trigger_cache_performance_alert.assert_called_once()

    @pytest.mark.asyncio
    async def test_multi_level_cache_performance_optimization(
        self, sample_idea_draft, mock_metrics_collector, mock_redis_client
    ):
        """
        GIVEN a multi-level cache system (L1 memory + L2 Redis)
        WHEN retrieving cached data
        THEN it should optimize performance using cache hierarchy
        """
        from pipeline.ingestion.cache.vector_cache import VectorCacheManager
        
        # Setup vector cache manager with L1 and L2 levels
        vector_cache = VectorCacheManager(
            redis_client=mock_redis_client,
            lru_size=1000,
            metrics_collector=mock_metrics_collector
        )
        
        vector_hash = "test_vector_hash_123"
        expected_similarities = [(uuid4(), 0.85), (uuid4(), 0.72)]
        
        # First call - should miss both L1 and L2, then cache result
        result1 = await vector_cache.get_vector_similarity(vector_hash)
        assert result1 is None
        
        # Store in cache
        await vector_cache.set_vector_similarity(vector_hash, expected_similarities)
        
        # Second call - should hit L1 cache (fastest)
        result2 = await vector_cache.get_vector_similarity(vector_hash)
        assert result2 == expected_similarities
        
        # Verify L1 cache hit was recorded
        mock_metrics_collector.record_hit.assert_called_with("l1_cache")

    @pytest.mark.asyncio
    async def test_cache_invalidation_with_performance_tracking(
        self, mock_metrics_collector, mock_redis_client
    ):
        """
        GIVEN a cache system with invalidation capabilities
        WHEN invalidating cache patterns
        THEN it should track invalidation performance and update metrics
        """
        from pipeline.ingestion.cache.cache_manager import CacheManager
        
        cache_manager = CacheManager(
            redis_client=mock_redis_client,
            metrics_collector=mock_metrics_collector
        )
        
        # Mock Redis pattern deletion
        mock_redis_client.scan_iter = AsyncMock(return_value=["key1", "key2", "key3"])
        mock_redis_client.delete = AsyncMock(return_value=3)
        
        # Invalidate pattern and track performance
        start_time = asyncio.get_event_loop().time()
        await cache_manager.invalidate_pattern("duplicate_check:*")
        
        # Verify invalidation was tracked
        mock_metrics_collector.record_cache_invalidation.assert_called()
        
        # Verify Redis operations were called
        mock_redis_client.scan_iter.assert_called_with(match="duplicate_check:*")
        mock_redis_client.delete.assert_called_with("key1", "key2", "key3")

    @pytest.mark.asyncio 
    async def test_performance_feedback_loop_integration(
        self, mock_metrics_collector, mock_redis_client
    ):
        """
        GIVEN a cache system with feedback loop integration
        WHEN performance metrics indicate optimization opportunities
        THEN it should trigger automated cache optimization adjustments
        """
        from pipeline.ingestion.monitoring.feedback_loop import CacheFeedbackLoop
        from pipeline.ingestion.cache.cache_manager import CacheManager
        
        # Setup feedback loop with optimization triggers
        feedback_loop = CacheFeedbackLoop(
            metrics_collector=mock_metrics_collector,
            optimization_threshold=0.7
        )
        
        cache_manager = CacheManager(
            redis_client=mock_redis_client,
            metrics_collector=mock_metrics_collector,
            feedback_loop=feedback_loop
        )
        
        # Simulate performance data indicating need for optimization
        performance_data = {
            "cache_hit_rate": 0.65,  # Below 70% threshold
            "avg_response_time": 250,  # ms
            "memory_usage": 0.85  # 85% of allocated memory
        }
        
        mock_metrics_collector.get_performance_summary.return_value = performance_data
        
        # Trigger feedback loop analysis
        optimization_actions = await feedback_loop.analyze_and_optimize(cache_manager)
        
        # Verify optimization actions were recommended
        assert "increase_cache_size" in optimization_actions
        assert "adjust_ttl_values" in optimization_actions
        
        # Verify cache configuration was updated
        assert cache_manager.is_optimization_applied() is True