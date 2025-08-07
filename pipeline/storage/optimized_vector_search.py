"""
Optimized Vector Search Engine - High-performance pgvector search implementation.

Provides optimized vector similarity search with advanced indexing,
caching, batch processing, and query optimization features.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import numpy as np
from sentence_transformers import SentenceTransformer

from pipeline.config.cache_manager import get_cache_manager
from pipeline.config.connection_pool import get_connection_pool
from pipeline.config.settings import get_settings
from pipeline.storage.vector_index_optimizer import (
    IndexConfig,
    VectorIndexOptimizer,
)


@dataclass
class SearchResult:
    """Vector search result with metadata."""
    idea_id: UUID
    similarity_score: float
    title: str
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)
    search_time_ms: float = 0.0


@dataclass
class SearchStats:
    """Vector search performance statistics."""
    total_searches: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_search_time_ms: float = 0.0
    total_search_time_ms: float = 0.0
    index_usage_count: int = 0
    batch_searches: int = 0


class OptimizedEmbeddingService:
    """Optimized embedding service with caching and batch processing."""

    def __init__(self):
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)

        # Model initialization with optimizations
        model_name = getattr(self.settings, 'embedding_model', 'all-MiniLM-L6-v2')
        self.model = SentenceTransformer(model_name)

        # Set to CPU or GPU based on availability
        device = getattr(self.settings, 'embedding_device', 'cpu')
        self.model = self.model.to(device)

        # Embedding cache (separate from main cache for performance)
        self._embedding_cache: dict[str, np.ndarray] = {}
        self._max_cache_size = 10000

        # Batch processing settings
        self.batch_size = getattr(self.settings, 'embedding_batch_size', 32)

        # Statistics
        self.stats = {
            'embeddings_generated': 0,
            'cache_hits': 0,
            'batch_operations': 0,
            'total_time_ms': 0.0
        }

    async def generate_embedding(self, text: str) -> np.ndarray:
        """Generate optimized embedding for single text."""
        # Check cache first
        cache_key = self._get_cache_key(text)
        if cache_key in self._embedding_cache:
            self.stats['cache_hits'] += 1
            return self._embedding_cache[cache_key]

        start_time = time.time()

        # Generate embedding
        embedding = await asyncio.get_event_loop().run_in_executor(
            None, self._generate_embedding_sync, text
        )

        # Cache the result
        await self._cache_embedding(cache_key, embedding)

        # Update stats
        elapsed_ms = (time.time() - start_time) * 1000
        self.stats['embeddings_generated'] += 1
        self.stats['total_time_ms'] += elapsed_ms

        return embedding

    async def generate_embeddings_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for multiple texts in batches."""
        if not texts:
            return []

        embeddings = []
        cached_embeddings = {}
        texts_to_generate = []
        text_indices = {}

        # Check cache for all texts
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self._embedding_cache:
                cached_embeddings[i] = self._embedding_cache[cache_key]
                self.stats['cache_hits'] += 1
            else:
                texts_to_generate.append(text)
                text_indices[len(texts_to_generate) - 1] = i

        # Generate embeddings for uncached texts in batches
        if texts_to_generate:
            start_time = time.time()

            # Process in batches
            generated_embeddings = []
            for i in range(0, len(texts_to_generate), self.batch_size):
                batch = texts_to_generate[i:i + self.batch_size]
                batch_embeddings = await asyncio.get_event_loop().run_in_executor(
                    None, self._generate_embeddings_batch_sync, batch
                )
                generated_embeddings.extend(batch_embeddings)

            # Cache generated embeddings
            for i, embedding in enumerate(generated_embeddings):
                text = texts_to_generate[i]
                cache_key = self._get_cache_key(text)
                await self._cache_embedding(cache_key, embedding)

            # Update stats
            elapsed_ms = (time.time() - start_time) * 1000
            self.stats['batch_operations'] += 1
            self.stats['embeddings_generated'] += len(generated_embeddings)
            self.stats['total_time_ms'] += elapsed_ms

        # Combine cached and generated embeddings in correct order
        result_embeddings = [None] * len(texts)

        # Add cached embeddings
        for i, embedding in cached_embeddings.items():
            result_embeddings[i] = embedding

        # Add generated embeddings
        generated_idx = 0
        for original_idx in text_indices.values():
            result_embeddings[original_idx] = generated_embeddings[generated_idx]
            generated_idx += 1

        return result_embeddings

    def _generate_embedding_sync(self, text: str) -> np.ndarray:
        """Synchronous embedding generation."""
        return self.model.encode(text, convert_to_numpy=True)

    def _generate_embeddings_batch_sync(self, texts: list[str]) -> list[np.ndarray]:
        """Synchronous batch embedding generation."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return [embedding for embedding in embeddings]

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return f"emb:{hash(text)}"

    async def _cache_embedding(self, key: str, embedding: np.ndarray):
        """Cache embedding with size management."""
        # Evict oldest entries if cache is full
        if len(self._embedding_cache) >= self._max_cache_size:
            # Remove 10% of oldest entries (simple FIFO)
            keys_to_remove = list(self._embedding_cache.keys())[:self._max_cache_size // 10]
            for key_to_remove in keys_to_remove:
                del self._embedding_cache[key_to_remove]

        self._embedding_cache[key] = embedding


class OptimizedVectorSearch:
    """High-performance vector search engine with advanced optimizations."""

    def __init__(self, index_config: IndexConfig | None = None):
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)

        # Services
        self.embedding_service = OptimizedEmbeddingService()
        self.pool_manager = None
        self.cache_manager = None

        # Advanced vector index optimization
        self.index_optimizer = VectorIndexOptimizer(index_config)
        self._index_initialized = False

        # Search configuration
        self.default_threshold = getattr(self.settings, 'vector_search_threshold', 0.7)
        self.max_results = getattr(self.settings, 'vector_search_max_results', 50)
        self.enable_caching = getattr(self.settings, 'vector_search_cache', True)
        self.cache_ttl = getattr(self.settings, 'vector_search_cache_ttl', 1800)  # 30 minutes

        # Performance optimizations
        self.use_parallel_search = getattr(self.settings, 'vector_parallel_search', True)
        self.enable_index_hints = getattr(self.settings, 'vector_index_hints', True)
        self.prefilter_enabled = getattr(self.settings, 'vector_prefilter', True)
        self.enable_query_optimization = getattr(self.settings, 'vector_query_optimization', True)

        # Statistics
        self.stats = SearchStats()

    async def initialize(self):
        """Initialize search engine components."""
        self.pool_manager = await get_connection_pool()
        if self.enable_caching:
            self.cache_manager = await get_cache_manager()

        # Initialize advanced vector index optimizer
        await self.index_optimizer.initialize(self.pool_manager._pool)
        self._index_initialized = True

        # Create optimized indexes if they don't exist
        index_created = await self.index_optimizer.create_optimized_index()
        if index_created:
            self.logger.info("Vector indexes optimized successfully")

        # Warm up embedding model
        await self.embedding_service.generate_embedding("test warmup text")

        # Get optimization recommendations
        recommendations = self.index_optimizer.get_optimization_recommendations()
        if recommendations:
            self.logger.info(f"Vector search optimization recommendations: {recommendations}")

        self.logger.info("Optimized vector search engine initialized")

    async def search_similar(
        self,
        query_text: str,
        threshold: float | None = None,
        limit: int | None = None,
        exclude_ids: list[UUID] | None = None,
        filters: dict[str, Any] | None = None,
        include_metadata: bool = False
    ) -> list[SearchResult]:
        """
        Search for similar ideas with optimized performance.
        
        Args:
            query_text: Text to search for similar items
            threshold: Minimum similarity threshold (default: configured threshold)
            limit: Maximum results to return (default: configured max)
            exclude_ids: Idea IDs to exclude from results
            filters: Additional filters (status, category, etc.)
            include_metadata: Whether to include full metadata in results
            
        Returns:
            List of search results ordered by similarity score
        """
        start_time = time.time()
        threshold = threshold or self.default_threshold
        limit = limit or self.max_results

        try:
            # Check cache first
            cache_key = None
            if self.enable_caching:
                cache_key = self._build_cache_key(query_text, threshold, limit, exclude_ids, filters)
                cached_results = await self.cache_manager.get(cache_key)
                if cached_results:
                    self.stats.cache_hits += 1
                    self.logger.debug(f"Cache hit for vector search: {len(cached_results)} results")
                    return self._deserialize_results(cached_results)

            # Generate query embedding
            query_embedding = await self.embedding_service.generate_embedding(query_text)

            # Use advanced query optimization if available
            if self._index_initialized and self.enable_query_optimization:
                # Get optimized query with execution plan
                search_query, query_params, query_plan = await self.index_optimizer.optimize_query(
                    embedding=query_embedding,
                    threshold=threshold,
                    limit=limit,
                    exclude_ids=exclude_ids
                )

                # Log query plan for debugging
                self.logger.debug(
                    f"Using optimized query plan: index={query_plan.use_index}, "
                    f"cost={query_plan.estimated_cost:.2f}, workers={query_plan.parallel_workers}"
                )
            else:
                # Fallback to basic query building
                search_query, query_params = self._build_search_query(
                    query_embedding, threshold, limit, exclude_ids, filters, include_metadata
                )

            # Execute search
            results = await self._execute_search(search_query, query_params)

            # Process results
            search_results = self._process_results(results, include_metadata)

            # Cache results
            if self.enable_caching and cache_key:
                serialized_results = self._serialize_results(search_results)
                await self.cache_manager.set(cache_key, serialized_results, self.cache_ttl)

            # Update stats
            search_time_ms = (time.time() - start_time) * 1000
            self.stats.total_searches += 1
            self.stats.total_search_time_ms += search_time_ms
            self.stats.avg_search_time_ms = self.stats.total_search_time_ms / self.stats.total_searches
            if not cached_results:
                self.stats.cache_misses += 1

            # Performance monitoring for PERF-002 compliance
            if search_time_ms > 50.0:
                self.logger.warning(
                    f"PERF-002 VIOLATION: Vector search took {search_time_ms:.2f}ms, "
                    f"exceeds 50ms requirement. Query: {query[:100]}..."
                )
                # Consider triggering index optimization
                if hasattr(self, 'index_optimizer') and self.index_optimizer:
                    await self._schedule_index_optimization(search_time_ms)

            # Add search time to results
            for result in search_results:
                result.search_time_ms = search_time_ms

            self.logger.debug(
                f"Vector search completed: {len(search_results)} results in {search_time_ms:.2f}ms "
                f"(Target: <50ms)"
            )

            return search_results

        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            raise

    async def search_batch(
        self,
        queries: list[dict[str, Any]]
    ) -> list[list[SearchResult]]:
        """
        Search for multiple queries in batch for better performance.
        
        Args:
            queries: List of query dictionaries with search parameters
            
        Returns:
            List of search results for each query
        """
        if not queries:
            return []

        start_time = time.time()

        try:
            # Extract query texts for batch embedding generation
            query_texts = [q.get('query_text', '') for q in queries]

            # Generate embeddings in batch
            query_embeddings = await self.embedding_service.generate_embeddings_batch(query_texts)

            # Prepare batch search tasks
            search_tasks = []
            for i, (query, embedding) in enumerate(zip(queries, query_embeddings, strict=False)):
                task = self._search_with_embedding(
                    embedding,
                    query.get('threshold', self.default_threshold),
                    query.get('limit', self.max_results),
                    query.get('exclude_ids'),
                    query.get('filters'),
                    query.get('include_metadata', False)
                )
                search_tasks.append(task)

            # Execute searches in parallel
            if self.use_parallel_search:
                batch_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            else:
                batch_results = []
                for task in search_tasks:
                    result = await task
                    batch_results.append(result)

            # Process results and handle exceptions
            processed_results = []
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    self.logger.error(f"Batch search query {i} failed: {result}")
                    processed_results.append([])
                else:
                    processed_results.append(result)

            # Update stats
            search_time_ms = (time.time() - start_time) * 1000
            self.stats.batch_searches += 1

            self.logger.debug(
                f"Batch vector search completed: {len(queries)} queries in {search_time_ms:.2f}ms"
            )

            return processed_results

        except Exception as e:
            self.logger.error(f"Batch vector search failed: {e}")
            raise

    async def _search_with_embedding(
        self,
        query_embedding: np.ndarray,
        threshold: float,
        limit: int,
        exclude_ids: list[UUID] | None,
        filters: dict[str, Any] | None,
        include_metadata: bool
    ) -> list[SearchResult]:
        """Search with pre-computed embedding."""
        search_query, query_params = self._build_search_query(
            query_embedding, threshold, limit, exclude_ids, filters, include_metadata
        )

        results = await self._execute_search(search_query, query_params)
        return self._process_results(results, include_metadata)

    def _build_search_query(
        self,
        query_embedding: np.ndarray,
        threshold: float,
        limit: int,
        exclude_ids: list[UUID] | None,
        filters: dict[str, Any] | None,
        include_metadata: bool
    ) -> tuple[str, list[Any]]:
        """Build optimized PostgreSQL search query with sub-50ms performance focus."""
        # Minimal field selection for maximum performance
        select_fields = [
            "i.idea_id",
            "1 - (e.description_embedding <=> $1) as similarity_score",
            "i.title",
            "i.description"
        ]

        # Only add metadata fields if explicitly requested to minimize data transfer
        if include_metadata:
            select_fields.extend([
                "i.category",
                "i.status",
                "i.current_stage",
                "i.created_at",
                "i.updated_at"
            ])

        query_parts = [
            f"SELECT {', '.join(select_fields)}",
            "FROM idea_embeddings e",
            "JOIN ideas i ON e.idea_id = i.idea_id"
        ]

        # Build WHERE clause
        where_conditions = ["1 - (e.description_embedding <=> $1) >= $2"]
        query_params = [query_embedding.tolist(), threshold]
        param_count = 2

        # Add exclusions
        if exclude_ids:
            param_count += 1
            placeholders = ", ".join([f"${param_count + i}" for i in range(len(exclude_ids))])
            where_conditions.append(f"i.idea_id NOT IN ({placeholders})")
            query_params.extend(exclude_ids)
            param_count += len(exclude_ids)

        # Add filters
        if filters:
            for key, value in filters.items():
                if key in ['status', 'category', 'current_stage'] and value:
                    param_count += 1
                    where_conditions.append(f"i.{key} = ${param_count}")
                    query_params.append(value)

        # Combine WHERE clause
        if where_conditions:
            query_parts.append(f"WHERE {' AND '.join(where_conditions)}")

        # Add performance hints
        if self.enable_index_hints:
            # PostgreSQL doesn't support index hints like MySQL, but we can optimize with query structure
            query_parts.append("ORDER BY e.description_embedding <=> $1")  # Use index operator
        else:
            query_parts.append("ORDER BY similarity_score DESC")

        # Add limit
        param_count += 1
        query_parts.append(f"LIMIT ${param_count}")
        query_params.append(limit)

        return " ".join(query_parts), query_params

    async def _execute_search(self, query: str, params: list[Any]) -> list[dict[str, Any]]:
        """Execute search query with connection pooling."""
        return await self.pool_manager.execute_query(query, params)

    def _process_results(self, raw_results: list[dict[str, Any]], include_metadata: bool) -> list[SearchResult]:
        """Process raw database results into SearchResult objects."""
        results = []

        for row in raw_results:
            metadata = {}
            if include_metadata:
                metadata = {
                    'category': row.get('category'),
                    'status': row.get('status'),
                    'current_stage': row.get('current_stage'),
                    'created_at': row.get('created_at'),
                    'updated_at': row.get('updated_at')
                }

            result = SearchResult(
                idea_id=row['idea_id'],
                similarity_score=float(row['similarity_score']),
                title=row['title'],
                description=row['description'],
                metadata=metadata
            )
            results.append(result)

        return results

    def _build_cache_key(
        self,
        query_text: str,
        threshold: float,
        limit: int,
        exclude_ids: list[UUID] | None,
        filters: dict[str, Any] | None
    ) -> str:
        """Build cache key for search parameters."""
        key_parts = [
            f"vsearch:{hash(query_text)}",
            f"t:{threshold}",
            f"l:{limit}"
        ]

        if exclude_ids:
            sorted_ids = sorted(str(id) for id in exclude_ids)
            key_parts.append(f"ex:{hash(':'.join(sorted_ids))}")

        if filters:
            filter_str = ":".join(f"{k}={v}" for k, v in sorted(filters.items()) if v)
            if filter_str:
                key_parts.append(f"f:{hash(filter_str)}")

        return ":".join(key_parts)

    def _serialize_results(self, results: list[SearchResult]) -> dict[str, Any]:
        """Serialize search results for caching."""
        return {
            'results': [
                {
                    'idea_id': str(result.idea_id),
                    'similarity_score': result.similarity_score,
                    'title': result.title,
                    'description': result.description,
                    'metadata': result.metadata
                }
                for result in results
            ]
        }

    def _deserialize_results(self, data: dict[str, Any]) -> list[SearchResult]:
        """Deserialize cached search results."""
        results = []
        for item in data.get('results', []):
            result = SearchResult(
                idea_id=UUID(item['idea_id']),
                similarity_score=item['similarity_score'],
                title=item['title'],
                description=item['description'],
                metadata=item.get('metadata', {})
            )
            results.append(result)
        return results

    async def get_search_stats(self) -> dict[str, Any]:
        """Get search engine statistics."""
        embedding_stats = self.embedding_service.stats.copy()

        return {
            'search_stats': {
                'total_searches': self.stats.total_searches,
                'cache_hits': self.stats.cache_hits,
                'cache_misses': self.stats.cache_misses,
                'cache_hit_rate': (
                    self.stats.cache_hits / (self.stats.cache_hits + self.stats.cache_misses)
                    if (self.stats.cache_hits + self.stats.cache_misses) > 0 else 0.0
                ),
                'avg_search_time_ms': self.stats.avg_search_time_ms,
                'batch_searches': self.stats.batch_searches
            },
            'embedding_stats': embedding_stats,
            'configuration': {
                'default_threshold': self.default_threshold,
                'max_results': self.max_results,
                'caching_enabled': self.enable_caching,
                'parallel_search': self.use_parallel_search,
                'cache_ttl_seconds': self.cache_ttl
            }
        }

    async def optimize_indexes(self, force_rebuild: bool = False):
        """Optimize vector search indexes for better performance."""
        try:
            if self._index_initialized:
                # Use advanced index optimizer
                success = await self.index_optimizer.create_optimized_index(force_rebuild)
                if success:
                    self.logger.info("Advanced vector indexes optimized successfully")
                    return True

            # Fallback to basic optimization
            analyze_query = """
                ANALYZE idea_embeddings;
                REINDEX INDEX CONCURRENTLY idx_embeddings_vector;  -- Use concurrent reindex
            """

            await self.pool_manager.execute_query(analyze_query)

            # Update index statistics
            self.stats.index_usage_count += 1

            self.logger.info("Basic vector search indexes optimized")
            return True

        except Exception as e:
            self.logger.error(f"Index optimization failed: {e}")
            return False

    async def maintain_indexes(self) -> bool:
        """Perform routine index maintenance."""
        if self._index_initialized:
            return await self.index_optimizer.maintain_index()
        self.logger.warning("Advanced index optimizer not initialized - skipping maintenance")
        return False

    async def benchmark_performance(self, test_queries: int = 50) -> dict[str, float]:
        """Benchmark vector search performance."""
        if self._index_initialized:
            return await self.index_optimizer.benchmark_performance(test_queries)
        self.logger.warning("Advanced index optimizer not initialized - cannot benchmark")
        return {}

    def get_optimization_recommendations(self) -> list[str]:
        """Get recommendations for improving vector search performance."""
        if self._index_initialized:
            return self.index_optimizer.get_optimization_recommendations()
        return ["Initialize advanced index optimizer for performance recommendations"]

    def get_index_stats(self) -> dict[str, Any]:
        """Get current index statistics and performance metrics."""
        base_stats = {
            'search_stats': {
                'total_searches': self.stats.total_searches,
                'cache_hits': self.stats.cache_hits,
                'cache_misses': self.stats.cache_misses,
                'avg_search_time_ms': self.stats.avg_search_time_ms,
                'batch_searches': self.stats.batch_searches
            },
            'configuration': {
                'default_threshold': self.default_threshold,
                'max_results': self.max_results,
                'caching_enabled': self.enable_caching,
                'parallel_search': self.use_parallel_search,
                'query_optimization_enabled': self.enable_query_optimization,
                'cache_ttl_seconds': self.cache_ttl
            }
        }

        if self._index_initialized:
            index_stats = self.index_optimizer.get_stats()
            base_stats['index_stats'] = {
                'index_type': index_stats.index_type,
                'index_size_mb': index_stats.index_size_mb,
                'total_vectors': index_stats.total_vectors,
                'avg_query_time_ms': index_stats.avg_query_time_ms,
                'index_selectivity': index_stats.index_selectivity,
                'last_maintenance': index_stats.last_maintenance,
                'queries_since_maintenance': index_stats.queries_since_maintenance
            }

        return base_stats

    async def _schedule_index_optimization(self, query_time_ms: float) -> None:
        """
        Schedule index optimization when queries exceed 50ms threshold.
        
        Args:
            query_time_ms: The slow query time that triggered this call
        """
        try:
            if hasattr(self, 'index_optimizer') and self.index_optimizer:
                # Check if we should reindex based on performance degradation
                current_avg = self.stats.avg_search_time_ms

                # Trigger reindexing if average is consistently over 50ms
                if current_avg > 50.0 and self.stats.total_searches > 100:
                    self.logger.info(
                        f"Scheduling index optimization due to performance degradation. "
                        f"Average: {current_avg:.2f}ms, Current: {query_time_ms:.2f}ms"
                    )

                    # Schedule async index maintenance
                    asyncio.create_task(self._optimize_indexes_async())

        except Exception as e:
            self.logger.error(f"Failed to schedule index optimization: {e}")

    async def _optimize_indexes_async(self) -> None:
        """Perform asynchronous index optimization for better performance."""
        try:
            self.logger.info("Starting index optimization for PERF-002 compliance")

            if hasattr(self, 'index_optimizer') and self.index_optimizer:
                # Get current index statistics
                stats = await self.index_optimizer.get_index_stats()

                # Optimize if needed
                if stats and stats.get('fragmentation', 0) > 0.3:
                    await self.index_optimizer.optimize_indexes()
                    self.logger.info("Index optimization completed")
                else:
                    self.logger.debug("Indexes are already optimized")

        except Exception as e:
            self.logger.error(f"Index optimization failed: {e}")

    def get_performance_report(self) -> dict[str, Any]:
        """
        Generate performance report for PERF-002 compliance monitoring.
        
        Returns:
            Dictionary with performance metrics and compliance status
        """
        compliance_status = "COMPLIANT" if self.stats.avg_search_time_ms < 50.0 else "NON_COMPLIANT"

        return {
            "perf_002_compliance": {
                "status": compliance_status,
                "target_ms": 50.0,
                "current_avg_ms": self.stats.avg_search_time_ms,
                "total_searches": self.stats.total_searches,
                "cache_hit_rate": (self.stats.cache_hits / max(self.stats.total_searches, 1)) * 100,
                "performance_violations": max(0, self.stats.total_searches - sum(1 for _ in range(int(self.stats.total_searches)) if self.stats.avg_search_time_ms < 50.0))
            },
            "recommendations": self._get_performance_recommendations()
        }

    def _get_performance_recommendations(self) -> list[str]:
        """Get performance optimization recommendations."""
        recommendations = []

        if self.stats.avg_search_time_ms > 50.0:
            recommendations.append("Consider index optimization or reindexing")

        cache_hit_rate = (self.stats.cache_hits / max(self.stats.total_searches, 1)) * 100
        if cache_hit_rate < 70.0:
            recommendations.append("Increase cache TTL or cache size for better hit rates")

        if self.stats.total_searches > 1000 and not hasattr(self, 'index_optimizer'):
            recommendations.append("Enable index optimizer for automatic performance tuning")

        if not recommendations:
            recommendations.append("Performance is optimal - no changes needed")

        return recommendations


# Singleton instance
_vector_search = None


async def get_vector_search() -> OptimizedVectorSearch:
    """Get singleton optimized vector search instance."""
    global _vector_search
    if _vector_search is None:
        _vector_search = OptimizedVectorSearch()
        await _vector_search.initialize()
    return _vector_search
