"""
Connection Pool Manager - Optimized database connection management.

Provides connection pooling, batch processing, and async database operations
for high-performance pipeline execution.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    asyncpg = None

try:
    import psycopg2
    from psycopg2 import pool
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    psycopg2 = None
    pool = None

from pipeline.config.settings import get_settings


@dataclass
class PoolStats:
    """Connection pool statistics."""
    total_connections: int
    active_connections: int
    idle_connections: int
    created_at: datetime
    total_queries: int
    failed_queries: int
    avg_query_time: float


class ConnectionPoolManager:
    """High-performance database connection pool manager."""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)
        
        # Async connection pool (preferred)
        self.async_pool: Optional[asyncpg.Pool] = None
        self.async_pool_available = False
        
        # Sync connection pool (fallback)
        self.sync_pool: Optional[pool.ThreadedConnectionPool] = None
        self.sync_pool_available = False
        
        # Thread pool for sync operations
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        
        # Statistics
        self.stats = {
            'total_queries': 0,
            'failed_queries': 0,
            'query_times': [],
            'created_at': datetime.utcnow()
        }
        
        # Configuration
        self.min_connections = getattr(self.settings, 'db_min_connections', 5)
        self.max_connections = getattr(self.settings, 'db_max_connections', 20)
        self.command_timeout = getattr(self.settings, 'db_command_timeout', 30)
    
    async def initialize(self):
        """Initialize connection pools."""
        # Try async pool first (best performance)
        if ASYNCPG_AVAILABLE:
            await self._init_async_pool()
        
        # Fallback to sync pool
        if not self.async_pool_available and PSYCOPG2_AVAILABLE:
            await self._init_sync_pool()
        
        if not self.async_pool_available and not self.sync_pool_available:
            raise RuntimeError("No database connection libraries available")
    
    async def _init_async_pool(self):
        """Initialize asyncpg connection pool."""
        try:
            database_url = self._get_database_url()
            
            self.async_pool = await asyncpg.create_pool(
                database_url,
                min_size=self.min_connections,
                max_size=self.max_connections,
                command_timeout=self.command_timeout,
                server_settings={
                    'application_name': 'agentic_startup_studio',
                    'timezone': 'UTC'
                }
            )
            
            # Test the pool
            async with self.async_pool.acquire() as conn:
                await conn.execute('SELECT 1')
            
            self.async_pool_available = True
            self.logger.info(f"AsyncPG pool initialized: {self.min_connections}-{self.max_connections} connections")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize AsyncPG pool: {e}")
            if self.async_pool:
                await self.async_pool.close()
                self.async_pool = None
    
    async def _init_sync_pool(self):
        """Initialize psycopg2 connection pool."""
        try:
            database_url = self._get_database_url()
            
            # Create threaded connection pool
            self.sync_pool = pool.ThreadedConnectionPool(
                self.min_connections,
                self.max_connections,
                database_url
            )
            
            # Test the pool
            conn = self.sync_pool.getconn()
            try:
                cursor = conn.cursor()
                cursor.execute('SELECT 1')
                cursor.close()
            finally:
                self.sync_pool.putconn(conn)
            
            self.sync_pool_available = True
            self.logger.info(f"Psycopg2 pool initialized: {self.min_connections}-{self.max_connections} connections")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize psycopg2 pool: {e}")
            if self.sync_pool:
                self.sync_pool.closeall()
                self.sync_pool = None
    
    def _get_database_url(self) -> str:
        """Get database URL from settings."""
        if hasattr(self.settings, 'database_url') and self.settings.database_url:
            return self.settings.database_url
        
        # Build URL from components
        host = getattr(self.settings, 'db_host', 'localhost')
        port = getattr(self.settings, 'db_port', 5432)
        database = getattr(self.settings, 'db_name', 'startup_studio')
        username = getattr(self.settings, 'db_username', 'postgres')
        password = getattr(self.settings, 'db_password', '')
        
        return f"postgresql://{username}:{password}@{host}:{port}/{database}"
    
    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool."""
        if self.async_pool_available:
            async with self.async_pool.acquire() as conn:
                yield conn
        elif self.sync_pool_available:
            # Use thread pool for sync operations
            conn = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool, self.sync_pool.getconn
            )
            try:
                yield conn
            finally:
                await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool, self.sync_pool.putconn, conn
                )
        else:
            raise RuntimeError("No database connection pools available")
    
    async def execute_query(self, query: str, params: Optional[List] = None) -> List[Dict[str, Any]]:
        """Execute a single query and return results."""
        start_time = datetime.utcnow()
        
        try:
            async with self.get_connection() as conn:
                if self.async_pool_available:
                    # AsyncPG query
                    if params:
                        rows = await conn.fetch(query, *params)
                    else:
                        rows = await conn.fetch(query)
                    
                    # Convert to dict format
                    results = [dict(row) for row in rows]
                else:
                    # Psycopg2 query
                    cursor = conn.cursor()
                    try:
                        if params:
                            cursor.execute(query, params)
                        else:
                            cursor.execute(query)
                        
                        if cursor.description:
                            columns = [desc[0] for desc in cursor.description]
                            rows = cursor.fetchall()
                            results = [dict(zip(columns, row)) for row in rows]
                        else:
                            results = []
                    finally:
                        cursor.close()
            
            # Update statistics
            query_time = (datetime.utcnow() - start_time).total_seconds()
            self.stats['total_queries'] += 1
            self.stats['query_times'].append(query_time)
            
            # Keep only last 1000 query times for average calculation
            if len(self.stats['query_times']) > 1000:
                self.stats['query_times'] = self.stats['query_times'][-1000:]
            
            return results
            
        except Exception as e:
            self.stats['failed_queries'] += 1
            self.logger.error(f"Query execution failed: {e}")
            raise
    
    async def execute_batch(self, queries: List[tuple]) -> List[List[Dict[str, Any]]]:
        """Execute multiple queries in batch for better performance."""
        if not queries:
            return []
        
        results = []
        
        async with self.get_connection() as conn:
            if self.async_pool_available:
                # Use asyncpg batch execution
                async with conn.transaction():
                    for query, params in queries:
                        if params:
                            rows = await conn.fetch(query, *params)
                        else:
                            rows = await conn.fetch(query)
                        results.append([dict(row) for row in rows])
            else:
                # Use psycopg2 with cursor reuse
                cursor = conn.cursor()
                try:
                    for query, params in queries:
                        if params:
                            cursor.execute(query, params)
                        else:
                            cursor.execute(query)
                        
                        if cursor.description:
                            columns = [desc[0] for desc in cursor.description]
                            rows = cursor.fetchall()
                            results.append([dict(zip(columns, row)) for row in rows])
                        else:
                            results.append([])
                    
                    conn.commit()
                finally:
                    cursor.close()
        
        # Update statistics
        self.stats['total_queries'] += len(queries)
        
        return results
    
    async def execute_transaction(self, operations: List[tuple]) -> bool:
        """Execute multiple operations in a single transaction."""
        try:
            async with self.get_connection() as conn:
                if self.async_pool_available:
                    async with conn.transaction():
                        for query, params in operations:
                            if params:
                                await conn.execute(query, *params)
                            else:
                                await conn.execute(query)
                else:
                    cursor = conn.cursor()
                    try:
                        for query, params in operations:
                            if params:
                                cursor.execute(query, params)
                            else:
                                cursor.execute(query)
                        conn.commit()
                    except Exception:
                        conn.rollback()
                        raise
                    finally:
                        cursor.close()
            
            self.stats['total_queries'] += len(operations)
            return True
            
        except Exception as e:
            self.stats['failed_queries'] += len(operations)
            self.logger.error(f"Transaction failed: {e}")
            raise
    
    async def get_pool_stats(self) -> PoolStats:
        """Get connection pool statistics."""
        if self.async_pool_available and self.async_pool:
            return PoolStats(
                total_connections=self.async_pool.get_size(),
                active_connections=self.async_pool.get_size() - self.async_pool.get_idle_size(),
                idle_connections=self.async_pool.get_idle_size(),
                created_at=self.stats['created_at'],
                total_queries=self.stats['total_queries'],
                failed_queries=self.stats['failed_queries'],
                avg_query_time=sum(self.stats['query_times']) / len(self.stats['query_times']) if self.stats['query_times'] else 0.0
            )
        elif self.sync_pool_available and self.sync_pool:
            # psycopg2 doesn't provide as detailed stats
            return PoolStats(
                total_connections=self.max_connections,
                active_connections=0,  # Not easily available
                idle_connections=0,    # Not easily available
                created_at=self.stats['created_at'],
                total_queries=self.stats['total_queries'],
                failed_queries=self.stats['failed_queries'],
                avg_query_time=sum(self.stats['query_times']) / len(self.stats['query_times']) if self.stats['query_times'] else 0.0
            )
        else:
            return PoolStats(
                total_connections=0,
                active_connections=0,
                idle_connections=0,
                created_at=self.stats['created_at'],
                total_queries=self.stats['total_queries'],
                failed_queries=self.stats['failed_queries'],
                avg_query_time=0.0
            )
    
    async def close(self):
        """Close all connections and cleanup."""
        if self.async_pool:
            await self.async_pool.close()
            self.async_pool = None
            self.async_pool_available = False
        
        if self.sync_pool:
            self.sync_pool.closeall()
            self.sync_pool = None
            self.sync_pool_available = False
        
        self.thread_pool.shutdown(wait=True)
        
        self.logger.info("Connection pools closed")


# Batch processing utilities
class BatchProcessor:
    """Utility for batching database operations."""
    
    def __init__(self, pool_manager: ConnectionPoolManager, batch_size: int = 100):
        self.pool_manager = pool_manager
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
    
    async def batch_insert(self, table: str, records: List[Dict[str, Any]]) -> int:
        """Batch insert records into table."""
        if not records:
            return 0
        
        # Split into batches
        batches = [records[i:i + self.batch_size] for i in range(0, len(records), self.batch_size)]
        total_inserted = 0
        
        for batch in batches:
            # Build insert query
            columns = list(batch[0].keys())
            placeholders = ', '.join(['%s'] * len(columns))
            query = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
            
            # Prepare batch operations
            operations = []
            for record in batch:
                values = [record[col] for col in columns]
                operations.append((query, values))
            
            # Execute batch
            await self.pool_manager.execute_transaction(operations)
            total_inserted += len(batch)
            
            self.logger.debug(f"Inserted batch of {len(batch)} records into {table}")
        
        return total_inserted
    
    async def batch_update(self, table: str, updates: List[Dict[str, Any]], key_column: str) -> int:
        """Batch update records in table."""
        if not updates:
            return 0
        
        batches = [updates[i:i + self.batch_size] for i in range(0, len(updates), self.batch_size)]
        total_updated = 0
        
        for batch in batches:
            operations = []
            
            for record in batch:
                # Build update query
                key_value = record[key_column]
                update_columns = [col for col in record.keys() if col != key_column]
                
                set_clause = ', '.join([f"{col} = %s" for col in update_columns])
                query = f"UPDATE {table} SET {set_clause} WHERE {key_column} = %s"
                
                values = [record[col] for col in update_columns] + [key_value]
                operations.append((query, values))
            
            # Execute batch
            await self.pool_manager.execute_transaction(operations)
            total_updated += len(batch)
            
            self.logger.debug(f"Updated batch of {len(batch)} records in {table}")
        
        return total_updated


# Singleton instance
_pool_manager = None


async def get_connection_pool() -> ConnectionPoolManager:
    """Get singleton connection pool manager."""
    global _pool_manager
    if _pool_manager is None:
        _pool_manager = ConnectionPoolManager()
        await _pool_manager.initialize()
    return _pool_manager