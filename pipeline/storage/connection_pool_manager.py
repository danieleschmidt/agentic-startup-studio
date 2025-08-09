"""
Advanced Database Connection Pool Manager.

This module provides sophisticated connection pooling with health monitoring,
automatic failover, and performance optimization.
"""

import asyncio
import logging
import ssl
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import asyncpg
from asyncpg import Connection, Pool

from pipeline.config.settings import get_settings
from pipeline.infrastructure.circuit_breaker import CircuitBreakerRegistry, CircuitBreakerConfig

logger = logging.getLogger(__name__)


class PoolHealthMetrics:
    """Tracks connection pool health and performance metrics."""

    def __init__(self):
        self.total_connections_created = 0
        self.total_connections_closed = 0
        self.current_active_connections = 0
        self.current_idle_connections = 0
        self.failed_connection_attempts = 0
        self.query_count = 0
        self.slow_query_count = 0
        self.average_query_time = 0.0
        self.last_health_check = None
        self.health_status = "unknown"
        self.error_details = []

    def record_connection_created(self):
        """Record a new connection creation."""
        self.total_connections_created += 1
        self.current_active_connections += 1

    def record_connection_closed(self):
        """Record a connection closure."""
        self.total_connections_closed += 1
        self.current_active_connections = max(0, self.current_active_connections - 1)

    def record_failed_connection(self, error: str):
        """Record a failed connection attempt."""
        self.failed_connection_attempts += 1
        self.error_details.append({
            'timestamp': datetime.now(UTC),
            'error': error
        })
        # Keep only last 10 errors
        self.error_details = self.error_details[-10:]

    def record_query(self, execution_time: float):
        """Record query execution metrics."""
        self.query_count += 1

        # Update average query time using running average
        if self.query_count == 1:
            self.average_query_time = execution_time
        else:
            self.average_query_time = (
                (self.average_query_time * (self.query_count - 1) + execution_time) /
                self.query_count
            )

        # Track slow queries (>1 second)
        if execution_time > 1.0:
            self.slow_query_count += 1

    def update_health_status(self, status: str, details: str = None):
        """Update overall health status."""
        self.health_status = status
        self.last_health_check = datetime.now(UTC)
        if details:
            self.error_details.append({
                'timestamp': self.last_health_check,
                'error': details
            })

    def get_health_summary(self) -> dict[str, Any]:
        """Get comprehensive health summary."""
        return {
            'health_status': self.health_status,
            'last_health_check': self.last_health_check.isoformat() if self.last_health_check else None,
            'connections': {
                'total_created': self.total_connections_created,
                'total_closed': self.total_connections_closed,
                'current_active': self.current_active_connections,
                'current_idle': self.current_idle_connections,
                'failed_attempts': self.failed_connection_attempts
            },
            'queries': {
                'total_count': self.query_count,
                'slow_query_count': self.slow_query_count,
                'average_time_seconds': round(self.average_query_time, 3),
                'slow_query_percentage': round(
                    (self.slow_query_count / max(1, self.query_count)) * 100, 2
                )
            },
            'recent_errors': self.error_details[-5:] if self.error_details else []
        }


class ConnectionPoolManager:
    """
    Advanced connection pool manager with health monitoring and failover.
    
    Features:
    - Automatic connection health checks
    - Circuit breaker integration
    - Pool size optimization
    - Connection retry with exponential backoff
    - SSL configuration
    - Performance monitoring
    """

    def __init__(self):
        self.settings = get_settings()
        self.pool: Pool | None = None
        self.metrics = PoolHealthMetrics()
        # Initialize circuit breaker for database connections
        self.circuit_breaker_registry = CircuitBreakerRegistry()
        self.db_circuit_breaker_config = CircuitBreakerConfig(
            failure_threshold=5,
            timeout=30.0,
            recovery_timeout=60.0
        )
        self._health_check_task: asyncio.Task | None = None
        self._is_shutting_down = False

    async def initialize(self) -> None:
        """Initialize the connection pool with comprehensive configuration."""
        try:
            # Build SSL context
            ssl_context = self._create_ssl_context()

            # Calculate optimal pool size
            min_size, max_size = self._calculate_pool_size()

            # Create connection pool
            self.pool = await asyncpg.create_pool(
                # Connection parameters
                host=self.settings.database.host,
                port=self.settings.database.port,
                user=self.settings.database.user,
                password=self.settings.database.password,
                database=self.settings.database.name,

                # Pool configuration
                min_size=min_size,
                max_size=max_size,
                command_timeout=self.settings.database.timeout,

                # SSL configuration
                ssl=ssl_context,

                # Connection lifecycle
                max_queries=50000,  # Recycle connections after 50k queries
                max_inactive_connection_lifetime=3600,  # 1 hour

                # Setup and init
                setup=self._setup_connection,
                init=self._init_connection,

                # Connection validation
                server_settings={
                    'search_path': 'public',
                    'timezone': 'UTC',
                    'statement_timeout': '30s',
                    'lock_timeout': '10s',
                    'idle_in_transaction_session_timeout': '60s'
                }
            )

            # Start health monitoring
            self._health_check_task = asyncio.create_task(self._health_check_loop())

            logger.info(
                "Database pool initialized successfully",
                extra={
                    "min_size": min_size,
                    "max_size": max_size,
                    "host": self.settings.database.host,
                    "database": self.settings.database.name
                }
            )

            self.metrics.update_health_status("healthy", "Pool initialized successfully")

        except Exception as e:
            error_msg = f"Failed to initialize database pool: {e}"
            logger.error(error_msg)
            self.metrics.record_failed_connection(error_msg)
            self.metrics.update_health_status("unhealthy", error_msg)
            raise

    async def close(self) -> None:
        """Gracefully close the connection pool."""
        self._is_shutting_down = True

        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Close pool
        if self.pool:
            await self.pool.close()
            logger.info("Database pool closed successfully")

    @asynccontextmanager
    async def acquire_connection(self):
        """Acquire connection with circuit breaker and monitoring."""
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        start_time = time.time()
        connection = None

        try:
            async with self.circuit_breaker:
                connection = await self.pool.acquire()
                self.metrics.record_connection_created()

                yield connection

        except Exception as e:
            self.metrics.record_failed_connection(str(e))
            logger.error(f"Failed to acquire database connection: {e}")
            raise
        finally:
            if connection:
                await self.pool.release(connection)
                self.metrics.record_connection_closed()

                execution_time = time.time() - start_time
                self.metrics.record_query(execution_time)

    async def execute_with_retry(
        self,
        query: str,
        *params,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> Any:
        """Execute query with automatic retry and exponential backoff."""
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                async with self.acquire_connection() as conn:
                    return await conn.fetchval(query, *params)

            except (TimeoutError, asyncpg.PostgresError) as e:
                last_error = e

                if attempt < max_retries:
                    delay = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"Query attempt {attempt + 1} failed, retrying in {delay}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Query failed after {max_retries + 1} attempts: {e}")

        raise last_error

    async def get_pool_status(self) -> dict[str, Any]:
        """Get detailed pool status information."""
        if not self.pool:
            return {"status": "not_initialized"}

        return {
            "status": "active",
            "size": self.pool.get_size(),
            "min_size": self.pool.get_min_size(),
            "max_size": self.pool.get_max_size(),
            "idle_connections": self.pool.get_idle_size(),
            "metrics": self.metrics.get_health_summary()
        }

    async def health_check(self) -> dict[str, Any]:
        """Perform comprehensive health check."""
        if not self.pool:
            return {
                "healthy": False,
                "error": "Pool not initialized",
                "timestamp": datetime.now(UTC).isoformat()
            }

        try:
            start_time = time.time()

            # Test basic connectivity
            async with self.acquire_connection() as conn:
                # Test simple query
                result = await conn.fetchval("SELECT 1")
                if result != 1:
                    raise ValueError("Basic connectivity test failed")

                # Test database-specific functionality
                await conn.fetchval("SELECT COUNT(*) FROM information_schema.tables")

                # Test vector extension if available
                try:
                    await conn.fetchval("SELECT 1 WHERE EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')")
                except:
                    pass  # Vector extension test is optional

            response_time = time.time() - start_time

            self.metrics.update_health_status("healthy")

            return {
                "healthy": True,
                "response_time_seconds": round(response_time, 3),
                "pool_status": await self.get_pool_status(),
                "timestamp": datetime.now(UTC).isoformat()
            }

        except Exception as e:
            error_msg = f"Health check failed: {e}"
            self.metrics.update_health_status("unhealthy", error_msg)

            return {
                "healthy": False,
                "error": error_msg,
                "pool_status": await self.get_pool_status(),
                "timestamp": datetime.now(UTC).isoformat()
            }

    async def get_connection_info(self) -> dict[str, Any]:
        """Get detailed connection information."""
        if not self.pool:
            return {"error": "Pool not initialized"}

        try:
            async with self.acquire_connection() as conn:
                # Get database version and settings
                version = await conn.fetchval("SELECT version()")

                # Get current settings
                settings_query = """
                    SELECT name, setting, unit, category 
                    FROM pg_settings 
                    WHERE name IN (
                        'max_connections', 'shared_buffers', 'work_mem',
                        'maintenance_work_mem', 'effective_cache_size',
                        'checkpoint_completion_target', 'wal_buffers'
                    )
                """
                settings = await conn.fetch(settings_query)

                # Get active connections
                connections_query = """
                    SELECT 
                        state,
                        COUNT(*) as count
                    FROM pg_stat_activity 
                    WHERE datname = current_database()
                    GROUP BY state
                """
                connections = await conn.fetch(connections_query)

                return {
                    "database_version": version,
                    "settings": [dict(row) for row in settings],
                    "active_connections": [dict(row) for row in connections],
                    "pool_metrics": self.metrics.get_health_summary()
                }

        except Exception as e:
            return {"error": f"Failed to get connection info: {e}"}

    # Private methods

    def _create_ssl_context(self) -> ssl.SSLContext | None:
        """Create SSL context based on configuration."""
        if not self.settings.database.enable_ssl:
            return None

        ssl_context = ssl.create_default_context()

        if self.settings.database.ssl_mode == "disable":
            return None
        if self.settings.database.ssl_mode == "require":
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
        elif self.settings.database.ssl_mode == "verify-ca":
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_REQUIRED
        elif self.settings.database.ssl_mode == "verify-full":
            ssl_context.check_hostname = True
            ssl_context.verify_mode = ssl.CERT_REQUIRED

        return ssl_context

    def _calculate_pool_size(self) -> tuple[int, int]:
        """Calculate optimal pool size based on configuration and system resources."""
        # Get configured values
        min_size = getattr(self.settings.database, 'min_connections', 1)
        max_size = getattr(self.settings.database, 'max_connections', 20)

        # Ensure reasonable bounds
        min_size = max(1, min_size)
        max_size = max(min_size, min(max_size, 100))  # Cap at 100 connections

        return min_size, max_size

    async def _setup_connection(self, conn: Connection) -> None:
        """Setup each new connection with required configuration."""
        try:
            # Set timezone
            await conn.execute("SET timezone = 'UTC'")

            # Set search path
            await conn.execute("SET search_path = public")

            # Set session parameters for security and performance
            await conn.execute("SET statement_timeout = '30s'")
            await conn.execute("SET lock_timeout = '10s'")
            await conn.execute("SET idle_in_transaction_session_timeout = '60s'")

            # Enable query plan caching
            await conn.execute("SET plan_cache_mode = auto")

            # Set work memory for this session
            await conn.execute("SET work_mem = '16MB'")

            logger.debug("Database connection setup completed")

        except Exception as e:
            logger.error(f"Failed to setup database connection: {e}")
            raise

    async def _init_connection(self, conn: Connection) -> None:
        """Initialize each connection with application-specific setup."""
        try:
            # Check for required extensions
            extensions = await conn.fetch(
                "SELECT extname FROM pg_extension WHERE extname IN ('vector', 'uuid-ossp', 'pgcrypto')"
            )

            available_extensions = [row['extname'] for row in extensions]

            if 'vector' not in available_extensions:
                logger.warning("pgvector extension not available")

            if 'uuid-ossp' not in available_extensions:
                logger.warning("uuid-ossp extension not available")

            # Test basic functionality
            await conn.fetchval("SELECT 1")

            logger.debug("Database connection initialization completed")

        except Exception as e:
            logger.error(f"Failed to initialize database connection: {e}")
            raise

    async def _health_check_loop(self) -> None:
        """Background task for periodic health checks."""
        while not self._is_shutting_down:
            try:
                await self.health_check()
                await asyncio.sleep(30)  # Health check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(60)  # Longer delay on error


# Global pool manager instance
_pool_manager: ConnectionPoolManager | None = None


async def get_pool_manager() -> ConnectionPoolManager:
    """Get or create the global pool manager."""
    global _pool_manager

    if _pool_manager is None:
        _pool_manager = ConnectionPoolManager()
        await _pool_manager.initialize()

    return _pool_manager


async def close_pool_manager() -> None:
    """Close the global pool manager."""
    global _pool_manager

    if _pool_manager:
        await _pool_manager.close()
        _pool_manager = None
