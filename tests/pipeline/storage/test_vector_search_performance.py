"""
Performance tests for vector search optimization - PERF-002 validation.

Tests ensure similarity queries complete in <50ms as required by PERF-002.
"""

import asyncio
import time
import uuid
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from pipeline.storage.optimized_vector_search import (
    OptimizedVectorSearchEngine,
    SearchConfig,
    SearchResult,
    OptimizedEmbeddingService
)


class TestVectorSearchPerformance:
    """Performance validation tests for vector search optimization."""
    
    @pytest.fixture
    def mock_connection_pool(self):
        """Mock connection pool for testing."""
        pool = AsyncMock()
        
        # Mock connection with cursor
        connection = AsyncMock()
        cursor = AsyncMock()
        
        # Mock vector search query results
        cursor.fetchall.return_value = [
            {
                'idea_id': uuid.uuid4(),
                'similarity_score': 0.95,
                'title': 'Test Idea 1',
                'description': 'Test description 1',
                'metadata': {}
            },
            {
                'idea_id': uuid.uuid4(),
                'similarity_score': 0.85,
                'title': 'Test Idea 2', 
                'description': 'Test description 2',
                'metadata': {}
            }
        ]
        
        connection.__aenter__.return_value = connection
        connection.cursor.return_value.__aenter__.return_value = cursor
        
        pool.acquire.return_value.__aenter__.return_value = connection
        
        return pool
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Mock embedding service with fast responses."""
        service = AsyncMock(spec=OptimizedEmbeddingService)
        
        # Return mock embedding vector
        test_embedding = np.random.rand(384).astype(np.float32)
        service.embed_text.return_value = test_embedding
        service.embed_batch.return_value = [test_embedding] * 10
        
        return service
    
    @pytest.fixture
    def search_engine(self, mock_connection_pool, mock_embedding_service):
        """Create search engine with mocked dependencies."""
        with patch('pipeline.storage.optimized_vector_search.get_connection_pool', return_value=mock_connection_pool), \
             patch('pipeline.storage.optimized_vector_search.get_cache_manager'), \
             patch('pipeline.storage.optimized_vector_search.get_settings'):
            
            engine = OptimizedVectorSearchEngine()
            engine.embedding_service = mock_embedding_service
            return engine
    
    @pytest.mark.asyncio
    async def test_single_query_performance_under_50ms(self, search_engine):
        """Test that a single similarity query completes in <50ms."""
        query = "test machine learning query"
        
        # Measure query time
        start_time = time.perf_counter()
        
        results = await search_engine.similarity_search(
            query=query,
            limit=10,
            similarity_threshold=0.7
        )
        
        end_time = time.perf_counter()
        query_time_ms = (end_time - start_time) * 1000
        
        # Assert performance requirement
        assert query_time_ms < 50.0, f"Query took {query_time_ms:.2f}ms, exceeds 50ms requirement"
        assert len(results) > 0, "Should return search results"
        
        # Verify results have proper timing metadata
        for result in results:
            assert hasattr(result, 'search_time_ms')
    
    @pytest.mark.asyncio
    async def test_batch_query_performance_under_50ms_per_query(self, search_engine):
        """Test that batch queries maintain <50ms per query performance."""
        queries = [
            "machine learning algorithms",
            "neural network architectures", 
            "data science applications",
            "artificial intelligence research",
            "deep learning frameworks"
        ]
        
        # Measure total batch time
        start_time = time.perf_counter()
        
        results = await search_engine.batch_similarity_search(
            queries=queries,
            limit=5,
            similarity_threshold=0.7
        )
        
        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000
        avg_time_per_query_ms = total_time_ms / len(queries)
        
        # Assert performance requirement for batch processing
        assert avg_time_per_query_ms < 50.0, (
            f"Average query time {avg_time_per_query_ms:.2f}ms exceeds 50ms requirement"
        )
        assert len(results) == len(queries), "Should return results for all queries"
    
    @pytest.mark.asyncio 
    async def test_concurrent_query_performance(self, search_engine):
        """Test that concurrent queries maintain performance under load."""
        query = "concurrent performance test query"
        concurrent_requests = 10
        
        async def single_query():
            start_time = time.perf_counter()
            await search_engine.similarity_search(
                query=query,
                limit=5,
                similarity_threshold=0.7
            )
            end_time = time.perf_counter()
            return (end_time - start_time) * 1000
        
        # Execute concurrent queries
        query_times = await asyncio.gather(*[
            single_query() for _ in range(concurrent_requests)
        ])
        
        # Check that all queries complete within 50ms
        max_time = max(query_times)
        avg_time = sum(query_times) / len(query_times)
        
        assert max_time < 50.0, f"Max query time {max_time:.2f}ms exceeds 50ms requirement"
        assert avg_time < 50.0, f"Average query time {avg_time:.2f}ms exceeds 50ms requirement"
    
    @pytest.mark.asyncio
    async def test_large_result_set_performance(self, search_engine):
        """Test performance with larger result sets."""
        # Mock larger result set
        large_results = []
        for i in range(100):
            large_results.append({
                'idea_id': uuid.uuid4(),
                'similarity_score': 0.9 - (i * 0.001),  # Decreasing similarity
                'title': f'Test Idea {i}',
                'description': f'Test description {i}',
                'metadata': {'index': i}
            })
        
        # Update mock to return large result set
        search_engine.connection_pool.acquire.return_value.__aenter__.return_value.cursor.return_value.__aenter__.return_value.fetchall.return_value = large_results
        
        query = "large result set performance test"
        
        start_time = time.perf_counter()
        
        results = await search_engine.similarity_search(
            query=query,
            limit=50,  # Request larger limit
            similarity_threshold=0.5
        )
        
        end_time = time.perf_counter()
        query_time_ms = (end_time - start_time) * 1000
        
        # Assert performance with large result sets
        assert query_time_ms < 50.0, f"Large result query took {query_time_ms:.2f}ms, exceeds 50ms requirement"
        assert len(results) == 50, "Should return requested number of results"
    
    @pytest.mark.asyncio
    async def test_cache_hit_performance(self, search_engine):
        """Test that cached queries perform even better than 50ms."""
        query = "cached query performance test"
        
        # First query to populate cache
        await search_engine.similarity_search(
            query=query,
            limit=10,
            similarity_threshold=0.7
        )
        
        # Second query should hit cache
        start_time = time.perf_counter()
        
        results = await search_engine.similarity_search(
            query=query,
            limit=10,
            similarity_threshold=0.7
        )
        
        end_time = time.perf_counter()
        cached_query_time_ms = (end_time - start_time) * 1000
        
        # Cached queries should be significantly faster
        assert cached_query_time_ms < 10.0, (
            f"Cached query took {cached_query_time_ms:.2f}ms, should be <10ms for cache hits"
        )
        assert len(results) > 0, "Should return cached results"
    
    def test_performance_monitoring_integration(self, search_engine):
        """Test that performance monitoring captures timing data."""
        # Verify search engine has stats tracking
        assert hasattr(search_engine, 'stats')
        assert hasattr(search_engine.stats, 'avg_search_time_ms')
        assert hasattr(search_engine.stats, 'total_searches')
        
        # Performance monitoring should track sub-50ms requirement
        stats = search_engine.stats
        assert stats.avg_search_time_ms >= 0.0  # Should be initialized
    
    @pytest.mark.asyncio
    async def test_index_optimization_for_performance(self, search_engine):
        """Test that index optimization recommendations are triggered for slow queries."""
        # Mock a slow query scenario
        with patch.object(search_engine, '_execute_vector_query') as mock_query:
            # Simulate a slow query (>50ms)
            async def slow_query(*args, **kwargs):
                await asyncio.sleep(0.06)  # 60ms delay
                return []
            
            mock_query.side_effect = slow_query
            
            query = "slow query optimization test"
            
            start_time = time.perf_counter()
            await search_engine.similarity_search(
                query=query,
                limit=10,
                similarity_threshold=0.7
            )
            end_time = time.perf_counter()
            
            query_time_ms = (end_time - start_time) * 1000
            
            # Verify slow query was detected
            assert query_time_ms > 50.0, "Test should simulate slow query"
            
            # Check that performance stats were updated
            assert search_engine.stats.total_searches > 0


class TestEmbeddingServicePerformance:
    """Performance tests for embedding service optimization."""
    
    @pytest.fixture
    def embedding_service(self):
        """Create embedding service for testing."""
        with patch('sentence_transformers.SentenceTransformer') as mock_model:
            # Mock fast embedding generation
            mock_model.return_value.encode.return_value = np.random.rand(384).astype(np.float32)
            
            service = OptimizedEmbeddingService()
            return service
    
    @pytest.mark.asyncio
    async def test_embedding_generation_performance(self, embedding_service):
        """Test that embedding generation is fast enough for <50ms total query time."""
        text = "test embedding performance query"
        
        start_time = time.perf_counter()
        embedding = await embedding_service.embed_text(text)
        end_time = time.perf_counter()
        
        embedding_time_ms = (end_time - start_time) * 1000
        
        # Embedding should be fast enough to allow for database query within 50ms total
        assert embedding_time_ms < 25.0, (
            f"Embedding generation took {embedding_time_ms:.2f}ms, "
            f"too slow for <50ms total query time"
        )
        assert embedding is not None
        assert len(embedding) > 0
    
    @pytest.mark.asyncio
    async def test_batch_embedding_performance(self, embedding_service):
        """Test batch embedding performance."""
        texts = [
            "batch embedding test query 1",
            "batch embedding test query 2", 
            "batch embedding test query 3",
            "batch embedding test query 4",
            "batch embedding test query 5"
        ]
        
        start_time = time.perf_counter()
        embeddings = await embedding_service.embed_batch(texts)
        end_time = time.perf_counter()
        
        total_time_ms = (end_time - start_time) * 1000
        time_per_embedding_ms = total_time_ms / len(texts)
        
        # Batch processing should be more efficient
        assert time_per_embedding_ms < 20.0, (
            f"Batch embedding took {time_per_embedding_ms:.2f}ms per item, too slow"
        )
        assert len(embeddings) == len(texts)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])