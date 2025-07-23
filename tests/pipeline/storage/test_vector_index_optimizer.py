"""
Comprehensive tests for VectorIndexOptimizer - Advanced pgvector index optimization.

Tests cover:
- HNSW and IVFFlat index creation and management
- Query optimization and execution plan generation
- Performance benchmarking and statistics
- Index maintenance and recommendations
- Error handling and edge cases
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from uuid import uuid4, UUID
from typing import List, Dict, Any

# Import the module under test
from pipeline.storage.vector_index_optimizer import (
    VectorIndexOptimizer,
    IndexType,
    DistanceMetric,
    IndexConfig,
    IndexStats,
    QueryPlan
)


class TestVectorIndexOptimizerInitialization:
    """Test VectorIndexOptimizer initialization and setup."""

    def test_init_with_default_config(self):
        """Test initialization with default configuration."""
        optimizer = VectorIndexOptimizer()
        
        assert optimizer.config.index_type == IndexType.HNSW
        assert optimizer.config.distance_metric == DistanceMetric.COSINE
        assert optimizer.config.hnsw_m == 16
        assert optimizer.config.hnsw_ef_construction == 64
        assert optimizer.config.hnsw_ef_search == 40
        assert optimizer.stats.index_type == "hnsw"
        assert optimizer.stats.total_vectors == 0

    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        config = IndexConfig(
            index_type=IndexType.IVFFLAT,
            distance_metric=DistanceMetric.L2,
            ivfflat_lists=200,
            maintenance_threshold=500
        )
        
        optimizer = VectorIndexOptimizer(config)
        
        assert optimizer.config.index_type == IndexType.IVFFLAT
        assert optimizer.config.distance_metric == DistanceMetric.L2
        assert optimizer.config.ivfflat_lists == 200
        assert optimizer.config.maintenance_threshold == 500

    @pytest.mark.asyncio
    async def test_initialize_sets_up_extensions(self):
        """Test that initialize properly sets up PostgreSQL extensions."""
        optimizer = VectorIndexOptimizer()
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        # Mock existing indexes analysis
        mock_conn.fetch.return_value = []
        mock_conn.fetchval.return_value = 1000
        
        await optimizer.initialize(mock_pool)
        
        # Verify extension setup calls
        expected_calls = [
            "CREATE EXTENSION IF NOT EXISTS vector",
            "SET maintenance_work_mem = '1GB'",
            "SET max_parallel_maintenance_workers = 4",
            "SET max_parallel_workers_per_gather = 2"
        ]
        
        for expected_call in expected_calls:
            mock_conn.execute.assert_any_call(expected_call)
        
        assert optimizer._connection_pool == mock_pool
        assert optimizer.stats.total_vectors == 1000


class TestIndexAnalysisAndCreation:
    """Test index analysis and creation functionality."""

    @pytest.fixture
    def mock_optimizer(self):
        """Create optimizer with mocked connection pool."""
        optimizer = VectorIndexOptimizer()
        mock_pool = AsyncMock()
        optimizer._connection_pool = mock_pool
        return optimizer, mock_pool

    @pytest.mark.asyncio
    async def test_analyze_current_indexes_with_existing_index(self, mock_optimizer):
        """Test analysis of existing vector indexes."""
        optimizer, mock_pool = mock_optimizer
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        # Mock existing index data
        mock_conn.fetch.return_value = [
            {
                'indexname': 'idx_embeddings_hnsw',
                'indexdef': 'CREATE INDEX ... USING hnsw',
                'size': '150 MB'
            }
        ]
        mock_conn.fetchval.return_value = 5000
        
        await optimizer._analyze_current_indexes()
        
        assert optimizer.stats.total_vectors == 5000
        assert optimizer.stats.index_size_mb == 150.0

    @pytest.mark.asyncio
    async def test_create_hnsw_index_success(self, mock_optimizer):
        """Test successful HNSW index creation."""
        optimizer, mock_pool = mock_optimizer
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        # Mock no existing optimal index
        mock_conn.fetchrow.return_value = None
        mock_conn.fetch.return_value = []  # No existing indexes to drop
        
        result = await optimizer.create_optimized_index()
        
        assert result is True
        
        # Verify HNSW index creation
        expected_index_sql = (
            "CREATE INDEX idx_idea_embeddings_hnsw_optimized ON idea_embeddings "
            "USING hnsw (description_embedding vector_cosine_ops) "
            "WITH (m = 16, ef_construction = 64)"
        )
        mock_conn.execute.assert_any_call(expected_index_sql)
        
        # Verify search parameter setting
        mock_conn.execute.assert_any_call("SET hnsw.ef_search = 40")

    @pytest.mark.asyncio
    async def test_create_ivfflat_index_success(self, mock_optimizer):
        """Test successful IVFFlat index creation."""
        config = IndexConfig(index_type=IndexType.IVFFLAT, ivfflat_lists=150)
        optimizer = VectorIndexOptimizer(config)
        mock_pool = AsyncMock()
        optimizer._connection_pool = mock_pool
        
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        # Mock no existing optimal index
        mock_conn.fetchrow.return_value = None
        mock_conn.fetch.return_value = []
        
        result = await optimizer.create_optimized_index()
        
        assert result is True
        
        # Verify IVFFlat index creation
        expected_index_sql = (
            "CREATE INDEX idx_idea_embeddings_ivfflat_optimized ON idea_embeddings "
            "USING ivfflat (description_embedding vector_cosine_ops) "
            "WITH (lists = 150)"
        )
        mock_conn.execute.assert_any_call(expected_index_sql)

    @pytest.mark.asyncio
    async def test_skip_index_creation_if_optimal_exists(self, mock_optimizer):
        """Test skipping index creation if optimal index already exists."""
        optimizer, mock_pool = mock_optimizer
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        # Mock existing optimal index
        mock_conn.fetchrow.return_value = {
            'indexname': 'idx_idea_embeddings_hnsw_optimized',
            'indexdef': 'CREATE INDEX ... USING hnsw'
        }
        
        result = await optimizer.create_optimized_index()
        
        assert result is True
        # Should not attempt to create new index
        assert not any("CREATE INDEX" in str(call) for call in mock_conn.execute.call_args_list)

    @pytest.mark.asyncio
    async def test_drop_suboptimal_indexes(self, mock_optimizer):
        """Test dropping of suboptimal indexes before creating optimal one."""
        optimizer, mock_pool = mock_optimizer
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        # Mock existing suboptimal indexes
        mock_conn.fetchrow.return_value = None  # No optimal index exists
        mock_conn.fetch.return_value = [
            {'indexname': 'old_embedding_index'},
            {'indexname': 'idx_embeddings_btree'}
        ]
        
        await optimizer.create_optimized_index()
        
        # Verify old indexes were dropped
        mock_conn.execute.assert_any_call("DROP INDEX IF EXISTS old_embedding_index")
        mock_conn.execute.assert_any_call("DROP INDEX IF EXISTS idx_embeddings_btree")


class TestQueryOptimization:
    """Test query optimization and execution plan generation."""

    @pytest.fixture
    def optimizer_with_stats(self):
        """Create optimizer with sample statistics."""
        optimizer = VectorIndexOptimizer()
        optimizer.stats.total_vectors = 10000
        optimizer.stats.avg_query_time_ms = 25.0
        return optimizer

    @pytest.mark.asyncio
    async def test_optimize_query_with_high_selectivity(self, optimizer_with_stats):
        """Test query optimization for high selectivity queries."""
        optimizer = optimizer_with_stats
        test_embedding = np.random.random(1536).astype(np.float32)
        
        query, params, plan = await optimizer.optimize_query(
            embedding=test_embedding,
            threshold=0.95,  # High selectivity
            limit=5
        )
        
        assert "SELECT" in query
        assert "similarity_score" in query
        assert "ORDER BY similarity_score DESC" in query
        assert "LIMIT" in query
        
        assert len(params) == 3  # embedding, threshold, limit
        assert params[0] == test_embedding.tolist()
        assert params[1] == 0.95
        assert params[2] == 5
        
        # High selectivity should favor index usage
        assert plan.use_index is True
        assert plan.estimated_rows < 1000  # Small result set expected
        assert plan.parallel_workers == 1  # Index scans don't parallelize well

    @pytest.mark.asyncio
    async def test_optimize_query_with_low_selectivity(self, optimizer_with_stats):
        """Test query optimization for low selectivity queries."""
        optimizer = optimizer_with_stats
        test_embedding = np.random.random(1536).astype(np.float32)
        
        query, params, plan = await optimizer.optimize_query(
            embedding=test_embedding,
            threshold=0.3,  # Low selectivity
            limit=100
        )
        
        # Low selectivity should favor sequential scan
        assert plan.use_index is False
        assert plan.estimated_rows > 5000  # Large result set expected 
        assert plan.parallel_workers >= 1  # May use parallel workers

    @pytest.mark.asyncio
    async def test_optimize_query_with_exclusions(self, optimizer_with_stats):
        """Test query optimization with ID exclusions."""
        optimizer = optimizer_with_stats
        test_embedding = np.random.random(1536).astype(np.float32)
        exclude_ids = [uuid4(), uuid4(), uuid4()]
        
        query, params, plan = await optimizer.optimize_query(
            embedding=test_embedding,
            threshold=0.8,
            limit=10,
            exclude_ids=exclude_ids
        )
        
        assert "NOT IN" in query
        assert len(params) == 3 + len(exclude_ids) + 1  # embedding, threshold, exclusions, limit
        
        # Verify exclusion IDs are in parameters
        for exclude_id in exclude_ids:
            assert exclude_id in params

    @pytest.mark.asyncio
    async def test_query_plan_cost_estimation(self, optimizer_with_stats):
        """Test query execution plan cost estimation."""
        optimizer = optimizer_with_stats
        
        # Test high selectivity (should prefer index)
        plan = await optimizer._analyze_query_requirements(0.9, 5, None)
        high_sel_cost = plan.estimated_cost
        
        # Test low selectivity (should prefer seq scan)
        plan = await optimizer._analyze_query_requirements(0.3, 100, None)
        low_sel_cost = plan.estimated_cost
        
        # High selectivity should have lower cost with index
        assert high_sel_cost < low_sel_cost

    @pytest.mark.asyncio
    async def test_parallel_execution_planning(self, optimizer_with_stats):
        """Test parallel execution planning for large result sets."""
        optimizer = optimizer_with_stats
        
        # Large result set should use parallel workers
        plan = await optimizer._analyze_query_requirements(0.2, 1000, None)
        assert plan.parallel_workers >= 1
        
        # Small result set should use single worker
        plan = await optimizer._analyze_query_requirements(0.95, 5, None)
        assert plan.parallel_workers == 1


class TestPerformanceBenchmarking:
    """Test performance benchmarking functionality."""

    @pytest.fixture
    def optimizer_with_pool(self):
        """Create optimizer with mocked connection pool."""
        optimizer = VectorIndexOptimizer()
        mock_pool = AsyncMock()
        optimizer._connection_pool = mock_pool
        return optimizer, mock_pool

    @pytest.mark.asyncio
    async def test_benchmark_performance_success(self, optimizer_with_pool):
        """Test successful performance benchmarking."""
        optimizer, mock_pool = optimizer_with_pool
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        # Mock query execution with varying response times
        mock_conn.fetch.side_effect = [
            [],  # Fast query result
            [],  # Medium query result  
            []   # Slow query result
        ] * 100  # Repeat for all test queries
        
        with patch('time.time', side_effect=[0, 0.01, 0.02, 0.03] * 300):  # Mock timing
            results = await optimizer.benchmark_performance(test_queries=12)
        
        assert 'overall_avg_ms' in results
        assert 'overall_p95_ms' in results
        assert 'queries_per_second' in results
        assert 'high_selectivity_avg_ms' in results
        assert 'medium_selectivity_avg_ms' in results
        assert 'low_selectivity_avg_ms' in results
        
        # Verify reasonable performance metrics
        assert results['queries_per_second'] > 0
        assert results['overall_avg_ms'] >= 0

    @pytest.mark.asyncio
    async def test_benchmark_batch_query_timing(self, optimizer_with_pool):
        """Test benchmark batch query timing accuracy."""
        optimizer, mock_pool = optimizer_with_pool
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        # Mock consistent query execution
        mock_conn.fetch.return_value = []
        
        test_vectors = [np.random.random(1536).astype(np.float32) for _ in range(3)]
        
        # Mock time progression: start, query1_end, query2_end, query3_end
        with patch('time.time', side_effect=[0, 0.025, 0.05, 0.075, 0.1]):
            times = await optimizer._benchmark_query_batch(test_vectors, 0.8, 10)
        
        assert len(times) == 3
        # Each query should take ~25ms based on mocked timings
        for query_time in times:
            assert 20 <= query_time <= 30

    @pytest.mark.asyncio
    async def test_benchmark_handles_query_failures(self, optimizer_with_pool):
        """Test benchmark handling of query failures."""
        optimizer, mock_pool = optimizer_with_pool
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        # Mock query failures
        mock_conn.fetch.side_effect = [
            [],  # Successful query
            Exception("Database error"),  # Failed query
            []   # Successful query
        ]
        
        test_vectors = [np.random.random(1536).astype(np.float32) for _ in range(3)]
        
        with patch('time.time', side_effect=[0, 0.01, 0.02, 0.03, 0.04]):
            times = await optimizer._benchmark_query_batch(test_vectors, 0.8, 10)
        
        # Should filter out failed queries
        assert len(times) == 2  # Only successful queries
        assert all(t != float('inf') for t in times)


class TestIndexMaintenance:
    """Test index maintenance functionality."""

    @pytest.fixture
    def optimizer_with_maintenance_needed(self):
        """Create optimizer that needs maintenance."""
        optimizer = VectorIndexOptimizer()
        optimizer.stats.total_vectors = 1000
        optimizer.config.maintenance_threshold = 500
        mock_pool = AsyncMock()
        optimizer._connection_pool = mock_pool
        return optimizer, mock_pool

    @pytest.mark.asyncio
    async def test_maintain_index_triggers_reindex(self, optimizer_with_maintenance_needed):
        """Test index maintenance triggers reindex when threshold exceeded."""
        optimizer, mock_pool = optimizer_with_maintenance_needed
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        # Mock vector count that exceeds maintenance threshold
        current_vectors = optimizer.stats.total_vectors + optimizer.config.maintenance_threshold + 100
        mock_conn.fetchval.return_value = current_vectors
        
        result = await optimizer.maintain_index()
        
        assert result is True
        
        # Verify maintenance operations
        mock_conn.execute.assert_any_call("ANALYZE idea_embeddings")
        mock_conn.execute.assert_any_call("REINDEX INDEX CONCURRENTLY idx_idea_embeddings_hnsw_optimized")
        
        # Verify stats updated
        assert optimizer.stats.total_vectors == current_vectors
        assert optimizer.stats.last_maintenance is not None
        assert optimizer.stats.queries_since_maintenance == 0

    @pytest.mark.asyncio
    async def test_maintain_index_skips_reindex_when_not_needed(self, optimizer_with_maintenance_needed):
        """Test index maintenance skips reindex when threshold not exceeded."""
        optimizer, mock_pool = optimizer_with_maintenance_needed
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        # Mock vector count below maintenance threshold
        current_vectors = optimizer.stats.total_vectors + 100  # Below threshold
        mock_conn.fetchval.return_value = current_vectors
        
        result = await optimizer.maintain_index()
        
        assert result is True
        
        # Should only run ANALYZE, not REINDEX
        mock_conn.execute.assert_any_call("ANALYZE idea_embeddings")
        assert not any("REINDEX" in str(call) for call in mock_conn.execute.call_args_list)

    @pytest.mark.asyncio
    async def test_maintain_index_handles_errors(self, optimizer_with_maintenance_needed):
        """Test index maintenance error handling."""
        optimizer, mock_pool = optimizer_with_maintenance_needed
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        # Mock database error
        mock_conn.execute.side_effect = Exception("Database connection lost")
        
        result = await optimizer.maintain_index()
        
        assert result is False


class TestRecommendationsAndStats:
    """Test optimization recommendations and statistics."""

    def test_get_recommendations_for_small_dataset(self):
        """Test recommendations for small vector datasets."""
        optimizer = VectorIndexOptimizer()
        optimizer.stats.total_vectors = 500  # Small dataset
        
        recommendations = optimizer.get_optimization_recommendations()
        
        assert any("Vector count is low" in rec for rec in recommendations)

    def test_get_recommendations_for_slow_queries(self):
        """Test recommendations for slow query performance."""
        optimizer = VectorIndexOptimizer()
        optimizer.stats.total_vectors = 10000
        optimizer.stats.avg_query_time_ms = 100  # Slow queries
        
        recommendations = optimizer.get_optimization_recommendations()
        
        assert any("Query performance is slow" in rec for rec in recommendations)

    def test_get_recommendations_for_large_index(self):
        """Test recommendations for oversized indexes."""
        optimizer = VectorIndexOptimizer()
        optimizer.stats.total_vectors = 1000
        optimizer.stats.index_size_mb = 500  # Very large relative to data
        
        recommendations = optimizer.get_optimization_recommendations()
        
        assert any("Index size is large" in rec for rec in recommendations)

    def test_get_recommendations_for_no_index(self):
        """Test recommendations when no index is configured."""
        config = IndexConfig(index_type=IndexType.NONE)
        optimizer = VectorIndexOptimizer(config)
        
        recommendations = optimizer.get_optimization_recommendations()
        
        assert any("No vector index configured" in rec for rec in recommendations)

    def test_get_stats_returns_current_state(self):
        """Test that get_stats returns accurate current state."""
        optimizer = VectorIndexOptimizer()
        optimizer.stats.total_vectors = 5000
        optimizer.stats.avg_query_time_ms = 15.5
        optimizer.stats.index_size_mb = 75.2
        
        stats = optimizer.get_stats()
        
        assert stats.total_vectors == 5000
        assert stats.avg_query_time_ms == 15.5
        assert stats.index_size_mb == 75.2
        assert stats.index_type == "hnsw"


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios."""

    @pytest.mark.asyncio
    async def test_optimize_query_with_uninitialized_optimizer(self):
        """Test query optimization with uninitialized optimizer."""
        optimizer = VectorIndexOptimizer()
        # Don't initialize connection pool
        
        test_embedding = np.random.random(1536).astype(np.float32)
        
        # Should still generate query (doesn't require database connection)
        query, params, plan = await optimizer.optimize_query(test_embedding, 0.8, 10)
        
        assert query is not None
        assert len(params) == 3
        assert plan is not None

    @pytest.mark.asyncio
    async def test_benchmark_with_uninitialized_optimizer(self):
        """Test benchmark with uninitialized optimizer raises error."""
        optimizer = VectorIndexOptimizer()
        # Don't initialize connection pool
        
        with pytest.raises(RuntimeError, match="Optimizer not initialized"):
            await optimizer.benchmark_performance()

    @pytest.mark.asyncio 
    async def test_create_index_with_invalid_parameters(self):
        """Test index creation with invalid parameters."""
        config = IndexConfig(
            hnsw_m=-1,  # Invalid parameter
            hnsw_ef_construction=0  # Invalid parameter
        )
        optimizer = VectorIndexOptimizer(config)
        mock_pool = AsyncMock()
        optimizer._connection_pool = mock_pool
        
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.fetchrow.return_value = None
        mock_conn.fetch.return_value = []
        
        # Mock database error for invalid index parameters
        mock_conn.execute.side_effect = Exception("Invalid index parameter")
        
        result = await optimizer.create_optimized_index()
        
        assert result is False

    def test_query_plan_with_extreme_parameters(self):
        """Test query plan generation with extreme parameters."""
        optimizer = VectorIndexOptimizer()
        optimizer.stats.total_vectors = 1000000  # Very large dataset
        
        # Test with extreme threshold values
        loop = asyncio.get_event_loop()
        
        # Very high threshold (should be very selective)
        plan_high = loop.run_until_complete(
            optimizer._analyze_query_requirements(0.999, 1, None)
        )
        assert plan_high.use_index is True
        assert plan_high.estimated_rows < 1000
        
        # Very low threshold (should be very unselective)
        plan_low = loop.run_until_complete(
            optimizer._analyze_query_requirements(0.001, 10000, None)
        )
        assert plan_low.use_index is False
        assert plan_low.estimated_rows > 50000

    @pytest.mark.asyncio
    async def test_initialization_with_database_errors(self):
        """Test initialization handling of database errors."""
        optimizer = VectorIndexOptimizer()
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        # Mock extension setup failure
        mock_conn.execute.side_effect = Exception("Permission denied")
        
        with pytest.raises(Exception):
            await optimizer.initialize(mock_pool)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])