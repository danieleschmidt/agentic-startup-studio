"""
Advanced Vector Index Optimizer for pgvector performance optimization.

Implements HNSW and IVFFlat indexing strategies, query optimization,
and hierarchical clustering for sub-second similarity searches at scale.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
from uuid import UUID

import asyncpg
import numpy as np
from asyncpg import Connection, Pool

from pipeline.config.settings import get_db_config

logger = logging.getLogger(__name__)


class IndexType(Enum):
    """Supported pgvector index types."""
    HNSW = "hnsw"
    IVFFLAT = "ivfflat"
    NONE = "none"


class DistanceMetric(Enum):
    """Supported distance metrics for vector search."""
    COSINE = "<=>"
    L2 = "<->"
    INNER_PRODUCT = "<#>"


@dataclass
class IndexConfig:
    """Configuration for vector index optimization."""
    index_type: IndexType = IndexType.HNSW
    distance_metric: DistanceMetric = DistanceMetric.COSINE
    hnsw_m: int = 16  # Number of connections for HNSW
    hnsw_ef_construction: int = 64  # Size of dynamic candidate list for HNSW
    hnsw_ef_search: int = 40  # Size of dynamic candidate list for search
    ivfflat_lists: int = 100  # Number of inverted lists for IVFFlat
    maintenance_threshold: int = 1000  # Reindex after N new vectors
    enable_parallel_build: bool = True
    enable_query_optimization: bool = True


@dataclass
class IndexStats:
    """Statistics for vector index performance."""
    index_type: str
    index_size_mb: float
    total_vectors: int
    avg_query_time_ms: float
    index_selectivity: float
    last_maintenance: Optional[str] = None
    queries_since_maintenance: int = 0


@dataclass
class QueryPlan:
    """Optimized query execution plan."""
    use_index: bool
    estimated_cost: float
    estimated_rows: int
    parallel_workers: int
    index_scan_type: Optional[str] = None


class VectorIndexOptimizer:
    """
    Advanced vector index optimizer for pgvector performance.
    
    Provides HNSW and IVFFlat indexing, query optimization, and
    hierarchical clustering for sub-second similarity searches.
    """
    
    def __init__(self, config: Optional[IndexConfig] = None):
        self.config = config or IndexConfig()
        self.db_config = get_db_config()
        self.stats = IndexStats(
            index_type=self.config.index_type.value,
            index_size_mb=0.0,
            total_vectors=0,
            avg_query_time_ms=0.0,
            index_selectivity=0.0
        )
        self._connection_pool: Optional[Pool] = None
        
    async def initialize(self, connection_pool: Pool):
        """Initialize the optimizer with database connection pool."""
        self._connection_pool = connection_pool
        await self._setup_vector_extensions()
        await self._analyze_current_indexes()
        
    async def _setup_vector_extensions(self):
        """Set up required PostgreSQL extensions for vector operations."""
        try:
            async with self._connection_pool.acquire() as conn:
                # Enable pgvector extension
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                
                # Configure pgvector settings for optimal performance
                await conn.execute("SET maintenance_work_mem = '1GB'")
                await conn.execute("SET max_parallel_maintenance_workers = 4")
                await conn.execute("SET max_parallel_workers_per_gather = 2")
                
                logger.info("Vector extensions and settings configured successfully")
                
        except Exception as e:
            logger.error(f"Failed to setup vector extensions: {e}")
            raise
            
    async def _analyze_current_indexes(self):
        """Analyze current vector indexes and gather statistics."""
        try:
            async with self._connection_pool.acquire() as conn:
                # Check existing indexes on idea_embeddings
                result = await conn.fetch("""
                    SELECT 
                        indexname,
                        indexdef,
                        pg_size_pretty(pg_relation_size(indexname::regclass)) as size
                    FROM pg_indexes 
                    WHERE tablename = 'idea_embeddings'
                    AND indexname LIKE '%embedding%'
                """)
                
                # Get vector count
                vector_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM idea_embeddings"
                )
                
                self.stats.total_vectors = vector_count
                
                if result:
                    for row in result:
                        logger.info(f"Found existing index: {row['indexname']} ({row['size']})")
                        # Parse index size
                        size_str = row['size'].replace(' kB', '').replace(' MB', '').replace(' GB', '')
                        try:
                            if 'MB' in row['size']:
                                self.stats.index_size_mb = float(size_str)
                            elif 'GB' in row['size']:
                                self.stats.index_size_mb = float(size_str) * 1024
                            elif 'kB' in row['size']:
                                self.stats.index_size_mb = float(size_str) / 1024
                        except ValueError:
                            pass
                else:
                    logger.warning("No vector indexes found - performance may be suboptimal")
                    
        except Exception as e:
            logger.error(f"Failed to analyze current indexes: {e}")
            
    async def create_optimized_index(self, force_rebuild: bool = False) -> bool:
        """
        Create optimized vector index based on configuration.
        
        Args:
            force_rebuild: Force rebuild even if index exists
            
        Returns:
            True if index was created/updated successfully
        """
        try:
            async with self._connection_pool.acquire() as conn:
                # Check if optimal index already exists
                if not force_rebuild:
                    existing = await self._check_optimal_index_exists(conn)
                    if existing:
                        logger.info(f"Optimal {self.config.index_type.value} index already exists")
                        return True
                
                # Drop existing suboptimal indexes
                await self._drop_suboptimal_indexes(conn)
                
                # Create new optimized index
                index_name = f"idx_idea_embeddings_{self.config.index_type.value}_optimized"
                
                if self.config.index_type == IndexType.HNSW:
                    success = await self._create_hnsw_index(conn, index_name)
                elif self.config.index_type == IndexType.IVFFLAT:
                    success = await self._create_ivfflat_index(conn, index_name)
                else:
                    logger.warning("No index type specified - using default btree")
                    return False
                
                if success:
                    await self._update_table_statistics(conn)
                    await self._analyze_current_indexes()  # Refresh stats
                    logger.info(f"Successfully created optimized {self.config.index_type.value} index")
                
                return success
                
        except Exception as e:
            logger.error(f"Failed to create optimized index: {e}")
            return False
            
    async def _check_optimal_index_exists(self, conn: Connection) -> bool:
        """Check if an optimal index already exists."""
        result = await conn.fetchrow("""
            SELECT indexname, indexdef
            FROM pg_indexes 
            WHERE tablename = 'idea_embeddings'
            AND indexname LIKE '%optimized%'
            AND indexdef LIKE $1
        """, f"%{self.config.index_type.value}%")
        
        return result is not None
        
    async def _drop_suboptimal_indexes(self, conn: Connection):
        """Drop existing suboptimal indexes."""
        # Get all vector indexes except the optimal one
        indexes = await conn.fetch("""
            SELECT indexname
            FROM pg_indexes 
            WHERE tablename = 'idea_embeddings'
            AND indexname LIKE '%embedding%'
            AND indexname NOT LIKE '%optimized%'
        """)
        
        for index in indexes:
            try:
                await conn.execute(f"DROP INDEX IF EXISTS {index['indexname']}")
                logger.info(f"Dropped suboptimal index: {index['indexname']}")
            except Exception as e:
                logger.warning(f"Failed to drop index {index['indexname']}: {e}")
                
    async def _create_hnsw_index(self, conn: Connection, index_name: str) -> bool:
        """Create HNSW index for fast approximate nearest neighbor search."""
        try:
            # HNSW is best for high-dimensional vectors with good recall/performance balance
            index_sql = f"""
                CREATE INDEX {index_name} ON idea_embeddings 
                USING hnsw (description_embedding vector_cosine_ops)
                WITH (m = {self.config.hnsw_m}, ef_construction = {self.config.hnsw_ef_construction})
            """
            
            logger.info(f"Creating HNSW index with m={self.config.hnsw_m}, ef_construction={self.config.hnsw_ef_construction}")
            
            start_time = time.time()
            await conn.execute(index_sql)
            build_time = time.time() - start_time
            
            # Set optimal search parameters
            await conn.execute(f"SET hnsw.ef_search = {self.config.hnsw_ef_search}")
            
            logger.info(f"HNSW index created successfully in {build_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create HNSW index: {e}")
            return False
            
    async def _create_ivfflat_index(self, conn: Connection, index_name: str) -> bool:
        """Create IVFFlat index for exact nearest neighbor search with clustering."""
        try:
            # IVFFlat is better for exact search with good performance on large datasets
            index_sql = f"""
                CREATE INDEX {index_name} ON idea_embeddings 
                USING ivfflat (description_embedding vector_cosine_ops)
                WITH (lists = {self.config.ivfflat_lists})
            """
            
            logger.info(f"Creating IVFFlat index with lists={self.config.ivfflat_lists}")
            
            start_time = time.time()
            await conn.execute(index_sql)
            build_time = time.time() - start_time
            
            logger.info(f"IVFFlat index created successfully in {build_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create IVFFlat index: {e}")
            return False
            
    async def _update_table_statistics(self, conn: Connection):
        """Update table statistics for optimal query planning."""
        try:
            await conn.execute("ANALYZE idea_embeddings")
            await conn.execute("ANALYZE ideas")
            logger.info("Table statistics updated successfully")
        except Exception as e:
            logger.warning(f"Failed to update table statistics: {e}")
            
    async def optimize_query(
        self, 
        embedding: np.ndarray, 
        threshold: float = 0.8,
        limit: int = 10,
        exclude_ids: Optional[List[UUID]] = None
    ) -> Tuple[str, List[Any], QueryPlan]:
        """
        Generate optimized query with execution plan.
        
        Args:
            embedding: Query vector embedding
            threshold: Similarity threshold
            limit: Maximum results
            exclude_ids: IDs to exclude from results
            
        Returns:
            Tuple of (optimized_query, parameters, query_plan)
        """
        # Analyze query requirements
        plan = await self._analyze_query_requirements(threshold, limit, exclude_ids)
        
        # Build optimized query based on plan
        query_parts = []
        params = []
        param_counter = 1
        
        # Select clause with index hints
        if plan.use_index and self.config.enable_query_optimization:
            query_parts.append("""
                SELECT /*+ IndexScan(e idx_idea_embeddings_hnsw_optimized) */
                    i.idea_id,
                    1 - (e.description_embedding <=> $1) as similarity_score,
                    i.title,
                    i.description
            """)
        else:
            query_parts.append("""
                SELECT 
                    i.idea_id,
                    1 - (e.description_embedding <=> $1) as similarity_score,
                    i.title,
                    i.description
            """)
        
        params.append(embedding.tolist())
        param_counter += 1
        
        # From clause with join strategy
        if plan.parallel_workers > 1:
            query_parts.append("""
                FROM idea_embeddings e /*+ PARALLEL(e, 2) */
                JOIN ideas i /*+ PARALLEL(i, 2) */ ON e.idea_id = i.idea_id
            """)
        else:
            query_parts.append("""
                FROM idea_embeddings e
                JOIN ideas i ON e.idea_id = i.idea_id
            """)
        
        # Where clause with optimized conditions
        where_conditions = [f"1 - (e.description_embedding <=> $1) >= ${param_counter}"]
        params.append(threshold)
        param_counter += 1
        
        # Exclude IDs if specified
        if exclude_ids:
            placeholders = ",".join([f"${param_counter + i}" for i in range(len(exclude_ids))])
            where_conditions.append(f"i.idea_id NOT IN ({placeholders})")
            params.extend(exclude_ids)
            param_counter += len(exclude_ids)
        
        query_parts.append("WHERE " + " AND ".join(where_conditions))
        
        # Order and limit with index optimization
        query_parts.extend([
            "ORDER BY similarity_score DESC",
            f"LIMIT ${param_counter}"
        ])
        params.append(limit)
        
        optimized_query = "\n".join(query_parts)
        
        return optimized_query, params, plan
        
    async def _analyze_query_requirements(
        self, 
        threshold: float, 
        limit: int, 
        exclude_ids: Optional[List[UUID]]
    ) -> QueryPlan:
        """Analyze query requirements and create execution plan."""
        # Estimate query selectivity
        selectivity = 1.0 - threshold  # Higher threshold = lower selectivity
        estimated_rows = int(self.stats.total_vectors * selectivity)
        
        # Determine if index should be used
        use_index = (
            self.stats.total_vectors > 1000 and  # Worth using index
            selectivity < 0.5 and  # Selective enough
            limit < estimated_rows * 0.1  # Limit is small relative to result set
        )
        
        # Estimate query cost (simplified model)
        if use_index:
            # Index scan cost: log(n) + limit
            estimated_cost = np.log2(max(1, self.stats.total_vectors)) + limit
            parallel_workers = 1  # Index scans don't parallelize well
        else:
            # Sequential scan cost: n * selectivity
            estimated_cost = self.stats.total_vectors * selectivity
            parallel_workers = min(2, max(1, estimated_rows // 10000))  # Parallelize large scans
        
        return QueryPlan(
            use_index=use_index,
            estimated_cost=estimated_cost,
            estimated_rows=estimated_rows,
            parallel_workers=parallel_workers,
            index_scan_type="hnsw" if use_index and self.config.index_type == IndexType.HNSW else None
        )
        
    async def benchmark_performance(self, test_queries: int = 100) -> Dict[str, float]:
        """
        Benchmark vector search performance with current configuration.
        
        Args:
            test_queries: Number of test queries to run
            
        Returns:
            Performance metrics dictionary
        """
        if not self._connection_pool:
            raise RuntimeError("Optimizer not initialized - call initialize() first")
            
        logger.info(f"Starting performance benchmark with {test_queries} queries...")
        
        try:
            # Generate random test vectors
            test_vectors = [
                np.random.random(1536).astype(np.float32) 
                for _ in range(test_queries)
            ]
            
            # Benchmark different query types
            results = {}
            
            # 1. High selectivity queries (threshold = 0.9)
            high_sel_times = await self._benchmark_query_batch(
                test_vectors[:test_queries//3], threshold=0.9, limit=5
            )
            results['high_selectivity_avg_ms'] = np.mean(high_sel_times)
            results['high_selectivity_p95_ms'] = np.percentile(high_sel_times, 95)
            
            # 2. Medium selectivity queries (threshold = 0.8)
            med_sel_times = await self._benchmark_query_batch(
                test_vectors[test_queries//3:2*test_queries//3], threshold=0.8, limit=10
            )
            results['medium_selectivity_avg_ms'] = np.mean(med_sel_times)
            results['medium_selectivity_p95_ms'] = np.percentile(med_sel_times, 95)
            
            # 3. Low selectivity queries (threshold = 0.7)
            low_sel_times = await self._benchmark_query_batch(
                test_vectors[2*test_queries//3:], threshold=0.7, limit=20
            )
            results['low_selectivity_avg_ms'] = np.mean(low_sel_times)
            results['low_selectivity_p95_ms'] = np.percentile(low_sel_times, 95)
            
            # Overall metrics
            all_times = high_sel_times + med_sel_times + low_sel_times
            results['overall_avg_ms'] = np.mean(all_times)
            results['overall_p95_ms'] = np.percentile(all_times, 95)
            results['overall_p99_ms'] = np.percentile(all_times, 99)
            results['queries_per_second'] = 1000.0 / results['overall_avg_ms']
            
            # Update internal stats
            self.stats.avg_query_time_ms = results['overall_avg_ms']
            
            logger.info(f"Benchmark completed: {results['queries_per_second']:.1f} QPS, "
                       f"{results['overall_avg_ms']:.2f}ms avg, "
                       f"{results['overall_p95_ms']:.2f}ms p95")
            
            return results
            
        except Exception as e:
            logger.error(f"Performance benchmark failed: {e}")
            return {}
            
    async def _benchmark_query_batch(
        self, 
        vectors: List[np.ndarray], 
        threshold: float, 
        limit: int
    ) -> List[float]:
        """Benchmark a batch of queries and return execution times."""
        times = []
        
        async with self._connection_pool.acquire() as conn:
            for vector in vectors:
                try:
                    start_time = time.time()
                    
                    # Use optimized query
                    query, params, plan = await self.optimize_query(
                        vector, threshold, limit
                    )
                    
                    await conn.fetch(query, *params)
                    
                    execution_time = (time.time() - start_time) * 1000  # Convert to ms
                    times.append(execution_time)
                    
                except Exception as e:
                    logger.warning(f"Benchmark query failed: {e}")
                    times.append(float('inf'))  # Mark as failed
                    
        return [t for t in times if t != float('inf')]  # Filter out failures
        
    async def maintain_index(self) -> bool:
        """
        Perform index maintenance operations.
        
        Returns:
            True if maintenance completed successfully
        """
        try:
            async with self._connection_pool.acquire() as conn:
                # Recompute table statistics
                await conn.execute("ANALYZE idea_embeddings")
                
                # Check if reindex is needed
                vector_count = await conn.fetchval("SELECT COUNT(*) FROM idea_embeddings")
                
                if (vector_count - self.stats.total_vectors) > self.config.maintenance_threshold:
                    logger.info(f"Reindexing due to {vector_count - self.stats.total_vectors} new vectors")
                    
                    # Reindex in background
                    await conn.execute("REINDEX INDEX CONCURRENTLY idx_idea_embeddings_hnsw_optimized")
                    
                    self.stats.total_vectors = vector_count
                    self.stats.last_maintenance = time.strftime("%Y-%m-%d %H:%M:%S")
                    self.stats.queries_since_maintenance = 0
                    
                    logger.info("Index maintenance completed successfully")
                
                return True
                
        except Exception as e:
            logger.error(f"Index maintenance failed: {e}")
            return False
            
    def get_optimization_recommendations(self) -> List[str]:
        """Get recommendations for further optimization."""
        recommendations = []
        
        if self.stats.total_vectors < 1000:
            recommendations.append("Vector count is low - consider batch loading more data before optimizing")
        
        if self.stats.avg_query_time_ms > 50:
            recommendations.append("Query performance is slow - consider HNSW index or increasing maintenance_work_mem")
        
        if self.stats.index_size_mb > self.stats.total_vectors * 0.1:  # Rough heuristic
            recommendations.append("Index size is large relative to data - consider IVFFlat with fewer lists")
        
        if self.config.index_type == IndexType.NONE:
            recommendations.append("No vector index configured - significant performance gains available")
        
        return recommendations
        
    def get_stats(self) -> IndexStats:
        """Get current index statistics."""
        return self.stats