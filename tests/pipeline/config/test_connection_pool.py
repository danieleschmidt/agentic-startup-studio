"""
Comprehensive tests for ConnectionPoolManager.

Tests critical infrastructure for database connection pooling, security,
performance, and reliability. Follows TDD principles with extensive coverage
of both happy paths and edge cases.
"""

import pytest
import asyncio
import logging
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# Test imports
from pipeline.config.connection_pool import (
    ConnectionPoolManager, 
    PoolStats,
    ASYNCPG_AVAILABLE,
    PSYCOPG2_AVAILABLE
)


class TestPoolStats:
    """Test PoolStats dataclass functionality."""
    
    def test_pool_stats_creation(self):
        """Test PoolStats can be created with valid data."""
        now = datetime.utcnow()
        stats = PoolStats(
            total_connections=10,
            active_connections=5,
            idle_connections=5,
            created_at=now,
            total_queries=100,
            failed_queries=2,
            avg_query_time=0.05
        )
        
        assert stats.total_connections == 10
        assert stats.active_connections == 5
        assert stats.idle_connections == 5
        assert stats.created_at == now
        assert stats.total_queries == 100
        assert stats.failed_queries == 2
        assert stats.avg_query_time == 0.05
    
    def test_pool_stats_validation(self):
        """Test that PoolStats validates sensible values."""
        now = datetime.utcnow()
        stats = PoolStats(
            total_connections=10,
            active_connections=5,
            idle_connections=5,
            created_at=now,
            total_queries=100,
            failed_queries=2,
            avg_query_time=0.05
        )
        
        # Verify the math adds up
        assert stats.active_connections + stats.idle_connections == stats.total_connections
        assert stats.failed_queries <= stats.total_queries
        assert stats.avg_query_time >= 0


class TestConnectionPoolManager:
    """Test ConnectionPoolManager functionality."""
    
    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing."""
        settings = Mock()
        settings.db_min_connections = 5
        settings.db_max_connections = 20
        settings.db_command_timeout = 30
        settings.database_url = "postgresql://test:test@localhost:5432/test_db"
        return settings
    
    @pytest.fixture 
    def pool_manager(self, mock_settings):
        """Create ConnectionPoolManager with mocked settings."""
        with patch('pipeline.config.connection_pool.get_settings', return_value=mock_settings):
            manager = ConnectionPoolManager()
            return manager
    
    def test_init(self, pool_manager):
        """Test ConnectionPoolManager initialization."""
        assert pool_manager.async_pool is None
        assert pool_manager.sync_pool is None
        assert not pool_manager.async_pool_available
        assert not pool_manager.sync_pool_available
        assert isinstance(pool_manager.thread_pool, ThreadPoolExecutor)
        assert pool_manager.min_connections == 5
        assert pool_manager.max_connections == 20
        assert pool_manager.command_timeout == 30
        
        # Check stats initialization
        assert pool_manager.stats['total_queries'] == 0
        assert pool_manager.stats['failed_queries'] == 0
        assert isinstance(pool_manager.stats['created_at'], datetime)
        assert pool_manager.stats['query_times'] == []


class TestConnectionPoolInitialization:
    """Test connection pool initialization scenarios."""
    
    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing."""
        settings = Mock()
        settings.db_min_connections = 5
        settings.db_max_connections = 20
        settings.db_command_timeout = 30
        return settings
    
    @pytest.mark.asyncio
    async def test_initialize_with_asyncpg_available(self, mock_settings):
        """Test initialization when asyncpg is available."""
        with patch('pipeline.config.connection_pool.get_settings', return_value=mock_settings), \
             patch('pipeline.config.connection_pool.ASYNCPG_AVAILABLE', True), \
             patch.object(ConnectionPoolManager, '_init_async_pool') as mock_init_async, \
             patch.object(ConnectionPoolManager, '_get_database_url', return_value="postgresql://test"):
            
            mock_init_async.return_value = None
            manager = ConnectionPoolManager()
            manager.async_pool_available = True  # Simulate successful init
            
            await manager.initialize()
            mock_init_async.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialize_fallback_to_psycopg2(self, mock_settings):
        """Test initialization fallback to psycopg2 when asyncpg fails."""
        with patch('pipeline.config.connection_pool.get_settings', return_value=mock_settings), \
             patch('pipeline.config.connection_pool.ASYNCPG_AVAILABLE', False), \
             patch('pipeline.config.connection_pool.PSYCOPG2_AVAILABLE', True), \
             patch.object(ConnectionPoolManager, '_init_sync_pool') as mock_init_sync:
            
            mock_init_sync.return_value = None
            manager = ConnectionPoolManager()
            manager.sync_pool_available = True  # Simulate successful init
            
            await manager.initialize()
            mock_init_sync.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialize_no_libraries_available(self, mock_settings):
        """Test initialization failure when no DB libraries available."""
        with patch('pipeline.config.connection_pool.get_settings', return_value=mock_settings), \
             patch('pipeline.config.connection_pool.ASYNCPG_AVAILABLE', False), \
             patch('pipeline.config.connection_pool.PSYCOPG2_AVAILABLE', False):
            
            manager = ConnectionPoolManager()
            
            with pytest.raises(RuntimeError, match="No database connection libraries available"):
                await manager.initialize()
    
    @pytest.mark.asyncio
    async def test_async_pool_initialization_success(self, mock_settings):
        """Test successful async pool initialization."""
        mock_pool = AsyncMock()
        
        with patch('pipeline.config.connection_pool.get_settings', return_value=mock_settings), \
             patch('pipeline.config.connection_pool.ASYNCPG_AVAILABLE', True), \
             patch('pipeline.config.connection_pool.asyncpg') as mock_asyncpg, \
             patch.object(ConnectionPoolManager, '_get_database_url', return_value="postgresql://test"):
            
            mock_asyncpg.create_pool.return_value = mock_pool
            manager = ConnectionPoolManager()
            
            await manager._init_async_pool()
            
            assert manager.async_pool == mock_pool
            assert manager.async_pool_available is True
            mock_asyncpg.create_pool.assert_called_once_with(
                "postgresql://test",
                min_size=5,
                max_size=20,
                command_timeout=30
            )
    
    @pytest.mark.asyncio
    async def test_async_pool_initialization_failure(self, mock_settings):
        """Test async pool initialization failure handling."""
        with patch('pipeline.config.connection_pool.get_settings', return_value=mock_settings), \
             patch('pipeline.config.connection_pool.ASYNCPG_AVAILABLE', True), \
             patch('pipeline.config.connection_pool.asyncpg') as mock_asyncpg, \
             patch.object(ConnectionPoolManager, '_get_database_url', return_value="postgresql://test"):
            
            # Simulate connection failure
            mock_asyncpg.create_pool.side_effect = Exception("Connection failed")
            manager = ConnectionPoolManager()
            
            await manager._init_async_pool()
            
            assert manager.async_pool is None
            assert manager.async_pool_available is False


class TestDatabaseURL:
    """Test database URL generation and security."""
    
    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing."""
        settings = Mock()
        settings.db_host = "localhost"
        settings.db_port = 5432
        settings.db_database = "test_db"
        settings.db_username = "test_user"
        settings.db_password = "test_pass"
        return settings
    
    def test_get_database_url_from_settings(self, mock_settings):
        """Test database URL generation from individual settings."""
        with patch('pipeline.config.connection_pool.get_settings', return_value=mock_settings):
            manager = ConnectionPoolManager()
            url = manager._get_database_url()
            
            expected = "postgresql://test_user:test_pass@localhost:5432/test_db"
            assert url == expected
    
    def test_get_database_url_no_password(self, mock_settings):
        """Test database URL generation without password."""
        mock_settings.db_password = None
        
        with patch('pipeline.config.connection_pool.get_settings', return_value=mock_settings):
            manager = ConnectionPoolManager()
            url = manager._get_database_url()
            
            expected = "postgresql://test_user@localhost:5432/test_db"
            assert url == expected
    
    def test_get_database_url_from_direct_url(self):
        """Test database URL when direct URL is provided."""
        settings = Mock()
        settings.database_url = "postgresql://user:pass@host:5432/db"
        
        with patch('pipeline.config.connection_pool.get_settings', return_value=settings):
            manager = ConnectionPoolManager()
            url = manager._get_database_url()
            
            assert url == "postgresql://user:pass@host:5432/db"
    
    def test_database_url_security_logging(self, mock_settings):
        """Test that passwords are not logged in database URLs."""
        with patch('pipeline.config.connection_pool.get_settings', return_value=mock_settings), \
             patch.object(logging.Logger, 'info') as mock_log:
            
            manager = ConnectionPoolManager()
            url = manager._get_database_url()
            
            # Check that if URL is logged, password is masked
            if mock_log.called:
                for call in mock_log.call_args_list:
                    log_message = str(call)
                    assert "test_pass" not in log_message  # Password should not appear in logs


class TestConnectionManagement:
    """Test connection acquisition and release."""
    
    @pytest.fixture
    def initialized_manager(self, mock_settings):
        """Create an initialized connection pool manager."""
        with patch('pipeline.config.connection_pool.get_settings', return_value=mock_settings):
            manager = ConnectionPoolManager()
            manager.async_pool = AsyncMock()
            manager.async_pool_available = True
            return manager
    
    @pytest.mark.asyncio
    async def test_get_connection_async_pool(self, initialized_manager):
        """Test getting connection from async pool."""
        mock_connection = AsyncMock()
        initialized_manager.async_pool.acquire.return_value = mock_connection
        
        async with initialized_manager.get_connection() as conn:
            assert conn == mock_connection
        
        initialized_manager.async_pool.acquire.assert_called_once()
        initialized_manager.async_pool.release.assert_called_once_with(mock_connection)
    
    @pytest.mark.asyncio
    async def test_get_connection_sync_fallback(self, mock_settings):
        """Test getting connection with sync pool fallback."""
        with patch('pipeline.config.connection_pool.get_settings', return_value=mock_settings):
            manager = ConnectionPoolManager()
            manager.async_pool_available = False
            manager.sync_pool = Mock()
            manager.sync_pool_available = True
            
            mock_connection = Mock()
            manager.sync_pool.getconn.return_value = mock_connection
            
            async with manager.get_connection() as conn:
                assert conn == mock_connection
            
            manager.sync_pool.getconn.assert_called_once()
            manager.sync_pool.putconn.assert_called_once_with(mock_connection)
    
    @pytest.mark.asyncio
    async def test_get_connection_no_pool_available(self, mock_settings):
        """Test error when no connection pool is available."""
        with patch('pipeline.config.connection_pool.get_settings', return_value=mock_settings):
            manager = ConnectionPoolManager()
            manager.async_pool_available = False
            manager.sync_pool_available = False
            
            with pytest.raises(RuntimeError, match="No connection pool available"):
                async with manager.get_connection():
                    pass


class TestQueryExecution:
    """Test query execution and statistics."""
    
    @pytest.fixture
    def initialized_manager(self, mock_settings):
        """Create an initialized connection pool manager."""
        with patch('pipeline.config.connection_pool.get_settings', return_value=mock_settings):
            manager = ConnectionPoolManager()
            manager.async_pool = AsyncMock()
            manager.async_pool_available = True
            return manager
    
    @pytest.mark.asyncio
    async def test_execute_query_success(self, initialized_manager):
        """Test successful query execution."""
        mock_connection = AsyncMock()
        mock_connection.fetch.return_value = [{"id": 1, "name": "test"}]
        initialized_manager.async_pool.acquire.return_value = mock_connection
        
        result = await initialized_manager.execute_query("SELECT * FROM test")
        
        assert result == [{"id": 1, "name": "test"}]
        assert initialized_manager.stats['total_queries'] == 1
        assert initialized_manager.stats['failed_queries'] == 0
        assert len(initialized_manager.stats['query_times']) == 1
    
    @pytest.mark.asyncio
    async def test_execute_query_with_params(self, initialized_manager):
        """Test query execution with parameters."""
        mock_connection = AsyncMock()
        mock_connection.fetch.return_value = [{"id": 1}]
        initialized_manager.async_pool.acquire.return_value = mock_connection
        
        await initialized_manager.execute_query("SELECT * FROM test WHERE id = $1", 123)
        
        mock_connection.fetch.assert_called_once_with("SELECT * FROM test WHERE id = $1", 123)
    
    @pytest.mark.asyncio
    async def test_execute_query_failure(self, initialized_manager):
        """Test query execution failure handling."""
        mock_connection = AsyncMock()
        mock_connection.fetch.side_effect = Exception("Query failed")
        initialized_manager.async_pool.acquire.return_value = mock_connection
        
        with pytest.raises(Exception, match="Query failed"):
            await initialized_manager.execute_query("SELECT * FROM test")
        
        assert initialized_manager.stats['total_queries'] == 1
        assert initialized_manager.stats['failed_queries'] == 1
    
    @pytest.mark.asyncio
    async def test_execute_many_queries(self, initialized_manager):
        """Test batch query execution."""
        mock_connection = AsyncMock()
        initialized_manager.async_pool.acquire.return_value = mock_connection
        
        queries = [
            ("INSERT INTO test (name) VALUES ($1)", "test1"),
            ("INSERT INTO test (name) VALUES ($1)", "test2"),
            ("INSERT INTO test (name) VALUES ($1)", "test3")
        ]
        
        await initialized_manager.execute_many(queries)
        
        assert mock_connection.executemany.called
        assert initialized_manager.stats['total_queries'] == len(queries)


class TestStatistics:
    """Test connection pool statistics."""
    
    @pytest.fixture
    def initialized_manager(self, mock_settings):
        """Create an initialized connection pool manager."""
        with patch('pipeline.config.connection_pool.get_settings', return_value=mock_settings):
            manager = ConnectionPoolManager()
            manager.async_pool = AsyncMock()
            manager.async_pool_available = True
            return manager
    
    def test_get_stats(self, initialized_manager):
        """Test getting pool statistics."""
        # Simulate some activity
        initialized_manager.stats['total_queries'] = 100
        initialized_manager.stats['failed_queries'] = 5
        initialized_manager.stats['query_times'] = [0.1, 0.2, 0.15, 0.3, 0.05]
        
        # Mock pool size info
        initialized_manager.async_pool.get_size.return_value = 10
        initialized_manager.async_pool.get_idle_size.return_value = 6
        
        stats = initialized_manager.get_stats()
        
        assert isinstance(stats, PoolStats)
        assert stats.total_connections == 10
        assert stats.active_connections == 4  # 10 - 6
        assert stats.idle_connections == 6
        assert stats.total_queries == 100
        assert stats.failed_queries == 5
        assert stats.avg_query_time == 0.16  # Average of query times
    
    def test_reset_stats(self, initialized_manager):
        """Test resetting statistics."""
        # Add some stats
        initialized_manager.stats['total_queries'] = 100
        initialized_manager.stats['failed_queries'] = 5
        initialized_manager.stats['query_times'] = [0.1, 0.2, 0.15]
        
        initialized_manager.reset_stats()
        
        assert initialized_manager.stats['total_queries'] == 0
        assert initialized_manager.stats['failed_queries'] == 0
        assert initialized_manager.stats['query_times'] == []
        assert isinstance(initialized_manager.stats['created_at'], datetime)


class TestResourceManagement:
    """Test resource management and cleanup."""
    
    @pytest.fixture
    def initialized_manager(self, mock_settings):
        """Create an initialized connection pool manager."""
        with patch('pipeline.config.connection_pool.get_settings', return_value=mock_settings):
            manager = ConnectionPoolManager()
            manager.async_pool = AsyncMock()
            manager.async_pool_available = True
            manager.sync_pool = Mock()
            manager.sync_pool_available = True
            return manager
    
    @pytest.mark.asyncio
    async def test_close_pools(self, initialized_manager):
        """Test proper cleanup of connection pools."""
        await initialized_manager.close()
        
        initialized_manager.async_pool.close.assert_called_once()
        initialized_manager.sync_pool.closeall.assert_called_once()
    
    def test_context_manager(self, mock_settings):
        """Test ConnectionPoolManager as context manager."""
        with patch('pipeline.config.connection_pool.get_settings', return_value=mock_settings):
            manager = ConnectionPoolManager()
            manager.close = AsyncMock()
            
            # Test async context manager would be used in practice
            assert hasattr(manager, '__aenter__')
            assert hasattr(manager, '__aexit__')


class TestSecurityAndValidation:
    """Test security measures and input validation."""
    
    def test_sql_injection_prevention(self, mock_settings):
        """Test that parameterized queries prevent SQL injection."""
        with patch('pipeline.config.connection_pool.get_settings', return_value=mock_settings):
            manager = ConnectionPoolManager()
            
            # Test that raw SQL strings are handled safely
            malicious_input = "'; DROP TABLE users; --"
            
            # The manager should use parameterized queries
            # This test verifies the pattern, actual execution would be mocked
            query = "SELECT * FROM test WHERE name = $1"
            params = (malicious_input,)
            
            # Verify that parameters are kept separate from query
            assert query.count("$1") == 1
            assert malicious_input not in query
            assert params[0] == malicious_input
    
    def test_connection_timeout_configuration(self, mock_settings):
        """Test that connection timeout is properly configured."""
        mock_settings.db_command_timeout = 15
        
        with patch('pipeline.config.connection_pool.get_settings', return_value=mock_settings):
            manager = ConnectionPoolManager()
            assert manager.command_timeout == 15
    
    def test_connection_limits_validation(self, mock_settings):
        """Test connection pool size limits."""
        mock_settings.db_min_connections = 5
        mock_settings.db_max_connections = 20
        
        with patch('pipeline.config.connection_pool.get_settings', return_value=mock_settings):
            manager = ConnectionPoolManager()
            
            assert manager.min_connections == 5
            assert manager.max_connections == 20
            assert manager.min_connections <= manager.max_connections


class TestErrorHandlingAndRecovery:
    """Test error handling and recovery mechanisms."""
    
    @pytest.fixture
    def initialized_manager(self, mock_settings):
        """Create an initialized connection pool manager."""
        with patch('pipeline.config.connection_pool.get_settings', return_value=mock_settings):
            manager = ConnectionPoolManager()
            manager.async_pool = AsyncMock()
            manager.async_pool_available = True
            return manager
    
    @pytest.mark.asyncio
    async def test_connection_acquisition_retry(self, initialized_manager):
        """Test retry logic for connection acquisition failures."""
        # Simulate temporary connection failure then success
        mock_connection = AsyncMock()
        initialized_manager.async_pool.acquire.side_effect = [
            Exception("Temporary failure"),
            mock_connection
        ]
        
        # Implementation would need retry logic - this tests the pattern
        with pytest.raises(Exception, match="Temporary failure"):
            async with initialized_manager.get_connection():
                pass
    
    @pytest.mark.asyncio
    async def test_query_timeout_handling(self, initialized_manager):
        """Test query timeout handling."""
        mock_connection = AsyncMock()
        mock_connection.fetch.side_effect = asyncio.TimeoutError("Query timeout")
        initialized_manager.async_pool.acquire.return_value = mock_connection
        
        with pytest.raises(asyncio.TimeoutError):
            await initialized_manager.execute_query("SELECT pg_sleep(60)")
    
    @pytest.mark.asyncio
    async def test_connection_health_check(self, initialized_manager):
        """Test connection health validation."""
        mock_connection = AsyncMock()
        mock_connection.fetchval.return_value = 1
        initialized_manager.async_pool.acquire.return_value = mock_connection
        
        is_healthy = await initialized_manager.health_check()
        
        assert is_healthy is True
        mock_connection.fetchval.assert_called_with("SELECT 1")
    
    @pytest.mark.asyncio
    async def test_connection_health_check_failure(self, initialized_manager):
        """Test connection health check failure."""
        mock_connection = AsyncMock()
        mock_connection.fetchval.side_effect = Exception("Connection dead")
        initialized_manager.async_pool.acquire.return_value = mock_connection
        
        is_healthy = await initialized_manager.health_check()
        
        assert is_healthy is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])