"""
Comprehensive test suite for storage layer components.

Tests database operations (CRUD), pgvector similarity search, connection
handling, transaction management, error scenarios and data consistency.
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock, patch, MagicMock, call
from uuid import UUID, uuid4
from datetime import datetime, timezone
from typing import List, Tuple, Optional, Any

from pipeline.storage.idea_repository import (
    EmbeddingService, DatabaseManager, IdeaRepository,
    DatabaseError, ConnectionError, QueryError,
    create_idea_repository
)
from pipeline.models.idea import (
    Idea, IdeaDraft, IdeaStatus, PipelineStage, IdeaCategory,
    QueryParams, AuditEntry
)
from pipeline.config.settings import DatabaseConfig


class TestEmbeddingService:
    """Test embedding generation and caching functionality."""
    
    @pytest.fixture
    def config(self) -> DatabaseConfig:
        """Provide test database configuration."""
        return DatabaseConfig(vector_dimensions=1536)
    
    @pytest.fixture
    def embedding_service(self, config) -> EmbeddingService:
        """Provide EmbeddingService instance."""
        return EmbeddingService(config)
    
    @pytest.mark.asyncio
    async def test_when_generate_embedding_then_returns_normalized_vector(self, embedding_service):
        """Given text input, when generating embedding, then returns normalized vector."""
        text = "Test text for embedding generation"
        
        embedding = await embedding_service.generate_embedding(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1536,)  # Default vector dimensions
        # Check if normalized (magnitude should be approximately 1)
        magnitude = np.linalg.norm(embedding)
        assert abs(magnitude - 1.0) < 1e-6
    
    @pytest.mark.asyncio
    async def test_when_same_text_then_returns_cached_embedding(self, embedding_service):
        """Given same text multiple times, when generating embedding, then returns cached result."""
        text = "Consistent text for caching test"
        
        # Generate embedding twice
        embedding1 = await embedding_service.generate_embedding(text)
        embedding2 = await embedding_service.generate_embedding(text)
        
        # Should be identical (cached)
        np.testing.assert_array_equal(embedding1, embedding2)
        assert len(embedding_service._embedding_cache) == 1
    
    @pytest.mark.asyncio
    async def test_when_different_text_then_different_embeddings(self, embedding_service):
        """Given different text, when generating embeddings, then returns different vectors."""
        text1 = "First unique text"
        text2 = "Second unique text"
        
        embedding1 = await embedding_service.generate_embedding(text1)
        embedding2 = await embedding_service.generate_embedding(text2)
        
        # Should be different vectors
        assert not np.array_equal(embedding1, embedding2)
        assert len(embedding_service._embedding_cache) == 2
    
    def test_when_calculate_cosine_similarity_then_returns_correct_value(self, embedding_service):
        """Given two vectors, when calculating cosine similarity, then returns correct value."""
        # Create two similar normalized vectors
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.8, 0.6, 0.0])
        
        # Normalize vectors
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        
        similarity = embedding_service.calculate_cosine_similarity(vec1, vec2)
        
        # Should be positive similarity
        assert 0.0 <= similarity <= 1.0
        assert isinstance(similarity, float)
    
    def test_when_identical_vectors_then_similarity_is_one(self, embedding_service):
        """Given identical vectors, when calculating similarity, then returns 1.0."""
        vec = np.array([1.0, 0.0, 0.0])
        
        similarity = embedding_service.calculate_cosine_similarity(vec, vec)
        
        assert abs(similarity - 1.0) < 1e-6


class TestDatabaseManager:
    """Test database connection and schema management."""
    
    @pytest.fixture
    def config(self) -> DatabaseConfig:
        """Provide test database configuration."""
        return DatabaseConfig(
            host="test-host",
            port=5432,
            database="test_db",
            username="test_user",
            password="test_pass"
        )
    
    @pytest.fixture
    def db_manager(self, config) -> DatabaseManager:
        """Provide DatabaseManager instance."""
        return DatabaseManager(config)
    
    @pytest.mark.asyncio
    async def test_when_initialize_then_creates_pool_and_schema(self, db_manager):
        """Given database manager, when initializing, then creates connection pool and schema."""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        
        with patch('pipeline.storage.idea_repository.asyncpg.create_pool', new_callable=AsyncMock, return_value=mock_pool) as mock_create_pool, \
             patch.object(db_manager, '_ensure_schema_exists') as mock_schema:
            
            await db_manager.initialize()
            
            # Verify pool creation
            mock_create_pool.assert_called_once_with(
                host="test-host",
                port=5432,
                user="test_user",
                password="test_pass",
                database="test_db",
                min_size=1,
                max_size=20,
                command_timeout=30
            )
            
            # Verify schema creation
            mock_schema.assert_called_once()
            assert db_manager.pool is mock_pool
    
    @pytest.mark.asyncio
    async def test_when_initialize_fails_then_raises_connection_error(self, db_manager):
        """Given initialization failure, when initializing, then raises ConnectionError."""
        with patch('pipeline.storage.idea_repository.asyncpg.create_pool', side_effect=Exception("Connection failed")):
            
            with pytest.raises(ConnectionError, match="Database initialization failed"):
                await db_manager.initialize()
    
    @pytest.mark.asyncio
    async def test_when_close_then_closes_pool(self, db_manager):
        """Given initialized manager, when closing, then closes connection pool."""
        mock_pool = AsyncMock()
        db_manager.pool = mock_pool
        
        await db_manager.close()
        
        mock_pool.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_when_get_connection_with_no_pool_then_raises_error(self, db_manager):
        """Given uninitialized manager, when getting connection, then raises ConnectionError."""
        with pytest.raises(ConnectionError, match="Database pool not initialized"):
            async with db_manager.get_connection():
                pass
    
    @pytest.mark.asyncio
    async def test_when_get_connection_then_yields_connection(self, db_manager):
        """Given initialized manager, when getting connection, then yields connection from pool."""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        db_manager.pool = mock_pool
        
        async with db_manager.get_connection() as conn:
            assert conn is mock_conn
        
        mock_pool.acquire.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_when_ensure_schema_exists_then_executes_schema_sql(self, db_manager):
        """Given database manager, when ensuring schema exists, then executes schema SQL."""
        mock_conn = AsyncMock()
        
        with patch.object(db_manager, 'get_connection') as mock_get_conn:
            mock_get_conn.return_value.__aenter__.return_value = mock_conn
            
            await db_manager._ensure_schema_exists()
            
            # Verify schema SQL was executed
            mock_conn.execute.assert_called_once()
            call_args = mock_conn.execute.call_args[0]
            assert "CREATE EXTENSION IF NOT EXISTS vector" in call_args[0]
            assert "CREATE TABLE IF NOT EXISTS ideas" in call_args[0]
            assert call_args[1] == db_manager.config.vector_dimensions


class TestIdeaRepository:
    """Test idea repository CRUD operations and similarity search."""
    
    @pytest.fixture
    def mock_db_manager(self):
        """Provide mock database manager."""
        manager = Mock(spec=DatabaseManager)
        manager.config = DatabaseConfig(enable_vector_search=True, vector_dimensions=1536)
        manager.embedding_service = Mock(spec=EmbeddingService)
        return manager
    
    @pytest.fixture
    def repository(self, mock_db_manager) -> IdeaRepository:
        """Provide IdeaRepository instance with mocked dependencies."""
        return IdeaRepository(mock_db_manager)
    
    @pytest.fixture
    def sample_idea(self) -> Idea:
        """Provide sample idea for testing."""
        return Idea(
            idea_id=uuid4(),
            title="Test AI productivity tool",
            description="Revolutionary AI solution for workplace productivity",
            category=IdeaCategory.AI_ML,
            created_by="test_user"
        )
    
    @pytest.mark.asyncio
    async def test_when_save_idea_then_inserts_with_embedding(self, repository, mock_db_manager, sample_idea):
        """Given new idea, when saving, then inserts idea and generates embedding."""
        mock_conn = AsyncMock()
        mock_db_manager.embedding_service.generate_embedding.return_value = np.random.random(1536)
        
        # Mock async context manager for get_connection
        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__.return_value = mock_conn
        mock_context_manager.__aexit__.return_value = None
        mock_db_manager.get_connection.return_value = mock_context_manager
        
        # Mock transaction context - transaction() should return context manager directly
        mock_transaction = AsyncMock()
        mock_transaction.__aenter__.return_value = mock_transaction
        mock_transaction.__aexit__.return_value = None
        # Make transaction a regular Mock that returns the async context manager
        from unittest.mock import Mock
        mock_conn.transaction = Mock(return_value=mock_transaction)
        
        result_id = await repository.save_idea(sample_idea, "test-correlation-id")
        
        # Verify idea insertion
        assert mock_conn.execute.call_count == 3  # Idea insert, embedding insert, audit insert
        
        # Check idea insertion call
        idea_insert_call = mock_conn.execute.call_args_list[0]
        assert "INSERT INTO ideas" in idea_insert_call[0][0]
        assert idea_insert_call[0][1] == sample_idea.idea_id
        
        # Check embedding insertion call
        embedding_insert_call = mock_conn.execute.call_args_list[1]
        assert "INSERT INTO idea_embeddings" in embedding_insert_call[0][0]
        
        # Verify embedding service was called
        mock_db_manager.embedding_service.generate_embedding.assert_called_once_with(sample_idea.description)
        
        assert result_id == sample_idea.idea_id
    
    @pytest.mark.asyncio
    async def test_when_save_idea_with_vector_disabled_then_skips_embedding(self, repository, mock_db_manager, sample_idea):
        """Given vector search disabled, when saving idea, then skips embedding generation."""
        mock_db_manager.config.enable_vector_search = False
        mock_conn = AsyncMock()
        
        # Mock async context manager for get_connection
        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__.return_value = mock_conn
        mock_context_manager.__aexit__.return_value = None
        mock_db_manager.get_connection.return_value = mock_context_manager
        
        # Mock transaction context manager
        mock_transaction = AsyncMock()
        mock_transaction.__aenter__ = AsyncMock(return_value=mock_transaction)
        mock_transaction.__aexit__ = AsyncMock(return_value=None)
        from unittest.mock import Mock
        mock_conn.transaction = Mock(return_value=mock_transaction)
        
        await repository.save_idea(sample_idea)
        
        # Should only have idea insert and audit insert (no embedding)
        assert mock_conn.execute.call_count == 2
        mock_db_manager.embedding_service.generate_embedding.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_when_save_idea_fails_then_raises_database_error(self, repository, mock_db_manager, sample_idea):
        """Given database error during save, when saving idea, then raises DatabaseError."""
        mock_conn = AsyncMock()
        mock_conn.execute.side_effect = Exception("Database error")
        
        # Mock async context manager for get_connection
        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__.return_value = mock_conn
        mock_context_manager.__aexit__.return_value = None
        mock_db_manager.get_connection.return_value = mock_context_manager
        
        # Mock transaction context manager
        mock_transaction = AsyncMock()
        mock_transaction.__aenter__.return_value = mock_transaction
        mock_transaction.__aexit__.return_value = None
        from unittest.mock import Mock
        mock_conn.transaction = Mock(return_value=mock_transaction)
        
        with pytest.raises(DatabaseError, match="Failed to save idea"):
            await repository.save_idea(sample_idea)
    
    @pytest.mark.asyncio
    async def test_when_find_by_id_exists_then_returns_idea(self, repository, mock_db_manager, sample_idea):
        """Given existing idea ID, when finding by ID, then returns idea."""
        mock_conn = AsyncMock()
        
        # Mock database row data
        mock_row = {
            'idea_id': sample_idea.idea_id,
            'title': sample_idea.title,
            'description': sample_idea.description,
            'category': sample_idea.category.value,
            'status': sample_idea.status.value,
            'current_stage': sample_idea.current_stage.value,
            'stage_progress': sample_idea.stage_progress,
            'problem_statement': sample_idea.problem_statement,
            'solution_description': sample_idea.solution_description,
            'target_market': sample_idea.target_market,
            'evidence_links': sample_idea.evidence_links,
            'created_at': sample_idea.created_at,
            'updated_at': sample_idea.updated_at,
            'created_by': sample_idea.created_by,
            'deck_path': sample_idea.deck_path,
            'research_data': sample_idea.research_data,
            'investor_scores': sample_idea.investor_scores
        }
        
        mock_conn.fetchrow.return_value = mock_row
        
        # Mock async context manager for get_connection
        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_context_manager.__aexit__ = AsyncMock(return_value=None)
        mock_db_manager.get_connection.return_value = mock_context_manager
        
        # Mock the _row_to_idea method
        with patch.object(repository, '_row_to_idea', return_value=sample_idea) as mock_row_to_idea:
            result = await repository.find_by_id(sample_idea.idea_id)
            
            assert result == sample_idea
            mock_conn.fetchrow.assert_called_once()
            mock_row_to_idea.assert_called_once_with(mock_row)
    
    @pytest.mark.asyncio
    async def test_when_find_by_id_not_exists_then_returns_none(self, repository, mock_db_manager):
        """Given non-existent idea ID, when finding by ID, then returns None."""
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = None
        
        # Mock async context manager for get_connection
        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__.return_value = mock_conn
        mock_context_manager.__aexit__.return_value = None
        mock_db_manager.get_connection.return_value = mock_context_manager
        
        result = await repository.find_by_id(uuid4())
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_when_find_by_id_fails_then_raises_query_error(self, repository, mock_db_manager):
        """Given database error during find, when finding by ID, then raises QueryError."""
        mock_conn = AsyncMock()
        mock_conn.fetchrow.side_effect = Exception("Query error")
        
        # Mock async context manager for get_connection
        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_context_manager.__aexit__ = AsyncMock(return_value=None)
        mock_db_manager.get_connection.return_value = mock_context_manager
        
        with pytest.raises(QueryError, match="Failed to find idea"):
            await repository.find_by_id(uuid4())
    
    @pytest.mark.asyncio
    async def test_when_find_with_filters_then_builds_correct_query(self, repository, mock_db_manager):
        """Given query parameters, when finding with filters, then builds correct SQL query."""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        
        # Mock async context manager for get_connection
        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_context_manager.__aexit__ = AsyncMock(return_value=None)
        mock_db_manager.get_connection.return_value = mock_context_manager
        
        # Create query parameters with filters
        params = QueryParams(
            status_filter=[IdeaStatus.DRAFT, IdeaStatus.VALIDATED],
            stage_filter=[PipelineStage.IDEATE],
            category_filter=[IdeaCategory.AI_ML],
            search_text="AI productivity",
            limit=10,
            offset=5,
            sort_by="created_at",
            sort_desc=True
        )
        
        await repository.find_with_filters(params)
        
        # Verify query was executed
        mock_conn.fetch.assert_called_once()
        query_call = mock_conn.fetch.call_args[0]
        
        # Check query structure
        assert "WHERE" in query_call[0]
        assert "status IN" in query_call[0]
        assert "current_stage IN" in query_call[0]
        assert "category IN" in query_call[0]
        assert "ILIKE" in query_call[0]
        assert "ORDER BY created_at DESC" in query_call[0]
        assert "LIMIT" in query_call[0]
        assert "OFFSET" in query_call[0]
    
    @pytest.mark.asyncio
    async def test_when_find_similar_by_embedding_then_performs_vector_search(self, repository, mock_db_manager):
        """Given description text, when finding similar ideas, then performs vector similarity search."""
        mock_conn = AsyncMock()
        mock_embedding = np.random.random(1536)
        
        # Mock embedding generation
        mock_db_manager.embedding_service.generate_embedding.return_value = mock_embedding
        
        # Mock database results
        mock_rows = [
            {'idea_id': uuid4(), 'similarity_score': 0.95},
            {'idea_id': uuid4(), 'similarity_score': 0.87}
        ]
        mock_conn.fetch.return_value = mock_rows
        
        # Mock async context manager for get_connection
        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_context_manager.__aexit__ = AsyncMock(return_value=None)
        mock_db_manager.get_connection.return_value = mock_context_manager
        
        results = await repository.find_similar_by_embedding(
            description="AI productivity tool",
            threshold=0.8,
            limit=10
        )
        
        # Verify embedding was generated
        mock_db_manager.embedding_service.generate_embedding.assert_called_once_with("AI productivity tool")
        
        # Verify vector search query
        mock_conn.fetch.assert_called_once()
        query_call = mock_conn.fetch.call_args[0]
        assert "description_embedding <=>" in query_call[0]  # pgvector distance operator
        assert "ORDER BY similarity_score DESC" in query_call[0]
        
        # Verify results
        assert len(results) == 2
        assert all(isinstance(result[0], UUID) and isinstance(result[1], float) for result in results)
    
    @pytest.mark.asyncio
    async def test_when_vector_search_disabled_then_returns_empty_list(self, repository, mock_db_manager):
        """Given vector search disabled, when finding similar ideas, then returns empty list."""
        mock_db_manager.config.enable_vector_search = False
        
        results = await repository.find_similar_by_embedding("test description")
        
        assert results == []
        mock_db_manager.embedding_service.generate_embedding.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_when_find_by_title_exact_then_returns_matching_ids(self, repository, mock_db_manager):
        """Given exact title, when finding by title, then returns matching idea IDs."""
        mock_conn = AsyncMock()
        test_ids = [uuid4(), uuid4()]
        mock_rows = [{'idea_id': id} for id in test_ids]
        mock_conn.fetch.return_value = mock_rows
        
        # Mock async context manager for get_connection
        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_context_manager.__aexit__ = AsyncMock(return_value=None)
        mock_db_manager.get_connection.return_value = mock_context_manager
        
        results = await repository.find_by_title_exact("Exact Title Match")
        
        # Verify exact match query
        mock_conn.fetch.assert_called_once()
        query_call = mock_conn.fetch.call_args[0]
        assert "title = $1" in query_call[0]
        assert query_call[1] == "Exact Title Match"
        
        assert results == test_ids
    
    @pytest.mark.asyncio
    async def test_when_update_idea_then_updates_with_new_embedding(self, repository, mock_db_manager, sample_idea):
        """Given updated idea, when updating, then updates database and regenerates embedding."""
        mock_conn = AsyncMock()
        mock_conn.execute.return_value = "UPDATE 1"  # Successful update
        
        # Mock async context manager for get_connection
        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_context_manager.__aexit__ = AsyncMock(return_value=None)
        mock_db_manager.get_connection.return_value = mock_context_manager
        
        mock_db_manager.embedding_service.generate_embedding.return_value = np.random.random(1536)
        
        # Mock transaction context manager
        mock_transaction = AsyncMock()
        mock_transaction.__aenter__ = AsyncMock(return_value=mock_transaction)
        mock_transaction.__aexit__ = AsyncMock(return_value=None)
        from unittest.mock import Mock
        mock_conn.transaction = Mock(return_value=mock_transaction)
        
        await repository.update_idea(sample_idea, "test_user", "test-correlation")
        
        # Verify update call
        assert mock_conn.execute.call_count == 3  # Update idea, update embedding, audit
        
        update_call = mock_conn.execute.call_args_list[0]
        assert "UPDATE ideas SET" in update_call[0][0]
        
        # Verify embedding update
        embedding_call = mock_conn.execute.call_args_list[1]
        assert "INSERT INTO idea_embeddings" in embedding_call[0][0]
        assert "ON CONFLICT" in embedding_call[0][0]  # Upsert
    
    @pytest.mark.asyncio
    async def test_when_update_nonexistent_idea_then_raises_error(self, repository, mock_db_manager, sample_idea):
        """Given non-existent idea, when updating, then raises DatabaseError."""
        mock_conn = AsyncMock()
        mock_conn.execute.return_value = "UPDATE 0"  # No rows updated
        mock_db_manager.get_connection.return_value.__aenter__.return_value = mock_conn
        mock_transaction = AsyncMock()
        mock_conn.transaction.return_value.__aenter__.return_value = mock_transaction
        
        with pytest.raises(DatabaseError, match="not found for update"):
            await repository.update_idea(sample_idea)


class TestCreateIdeaRepository:
    """Test repository factory function."""
    
    @pytest.mark.asyncio
    async def test_when_no_config_then_creates_with_default_config(self):
        """Given no config, when creating repository, then uses default database config."""
        with patch('pipeline.storage.idea_repository.get_db_config') as mock_get_config, \
             patch('pipeline.storage.idea_repository.DatabaseManager') as mock_db_manager_class, \
             patch('pipeline.storage.idea_repository.IdeaRepository') as mock_repo_class:
            
            mock_config = Mock()
            mock_get_config.return_value = mock_config
            mock_db_manager = AsyncMock()
            mock_db_manager_class.return_value = mock_db_manager
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo
            
            result = await create_idea_repository()
            
            # Verify default config was fetched
            mock_get_config.assert_called_once()
            
            # Verify database manager was created and initialized
            mock_db_manager_class.assert_called_once_with(mock_config)
            mock_db_manager.initialize.assert_called_once()
            
            # Verify repository was created
            mock_repo_class.assert_called_once_with(mock_db_manager)
            assert result == mock_repo
    
    @pytest.mark.asyncio
    async def test_when_config_provided_then_uses_provided_config(self):
        """Given config, when creating repository, then uses provided config."""
        test_config = DatabaseConfig()
        
        with patch('pipeline.storage.idea_repository.DatabaseManager') as mock_db_manager_class, \
             patch('pipeline.storage.idea_repository.IdeaRepository') as mock_repo_class:
            
            mock_db_manager = AsyncMock()
            mock_db_manager_class.return_value = mock_db_manager
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo
            
            result = await create_idea_repository(test_config)
            
            # Verify provided config was used
            mock_db_manager_class.assert_called_once_with(test_config)
            mock_db_manager.initialize.assert_called_once()
            mock_repo_class.assert_called_once_with(mock_db_manager)
            assert result == mock_repo


class TestRowConversionMethods:
    """Test database row to model conversion methods."""
    
    @pytest.fixture
    def repository(self):
        """Provide repository for testing row conversion."""
        mock_db_manager = Mock()
        return IdeaRepository(mock_db_manager)
    
    def test_when_row_to_idea_then_converts_correctly(self, repository):
        """Given database row, when converting to Idea, then maps all fields correctly."""
        test_id = uuid4()
        test_time = datetime.now(timezone.utc)
        
        mock_row = {
            'idea_id': test_id,
            'title': "Test Title",
            'description': "Test Description",
            'category': 'ai_ml',
            'status': 'DRAFT',
            'current_stage': 'IDEATE',
            'stage_progress': 0.5,
            'problem_statement': "Test Problem",
            'solution_description': "Test Solution",
            'target_market': "Test Market",
            'evidence_links': ["https://example.com"],
            'created_at': test_time,
            'updated_at': test_time,
            'created_by': "test_user",
            'deck_path': "/path/to/deck",
            'research_data': {"key": "value"},
            'investor_scores': {"score": 0.8}
        }
        
        idea = repository._row_to_idea(mock_row)
        
        assert isinstance(idea, Idea)
        assert idea.idea_id == test_id
        assert idea.title == "Test Title"
        assert idea.description == "Test Description"
        assert idea.category == IdeaCategory.AI_ML
        assert idea.status == IdeaStatus.DRAFT
        assert idea.current_stage == PipelineStage.IDEATE
        assert idea.stage_progress == 0.5
        assert idea.problem_statement == "Test Problem"
        assert idea.solution_description == "Test Solution"
        assert idea.target_market == "Test Market"
        assert idea.evidence_links == ["https://example.com"]
        assert idea.created_at == test_time
        assert idea.updated_at == test_time
        assert idea.created_by == "test_user"
        assert idea.deck_path == "/path/to/deck"
        assert idea.research_data == {"key": "value"}
        assert idea.investor_scores == {"score": 0.8}
    
    def test_when_row_to_audit_entry_then_converts_correctly(self, repository):
        """Given database row, when converting to AuditEntry, then maps all fields correctly."""
        test_entry_id = uuid4()
        test_idea_id = uuid4()
        test_time = datetime.now(timezone.utc)
        
        mock_row = {
            'entry_id': test_entry_id,
            'idea_id': test_idea_id,
            'action': 'idea_created',
            'changes': {"field": "value"},
            'user_id': "test_user",
            'timestamp': test_time,
            'correlation_id': "test-correlation"
        }
        
        audit_entry = repository._row_to_audit_entry(mock_row)
        
        assert isinstance(audit_entry, AuditEntry)
        assert audit_entry.entry_id == test_entry_id
        assert audit_entry.idea_id == test_idea_id
        assert audit_entry.action == 'idea_created'
        assert audit_entry.changes == {"field": "value"}
        assert audit_entry.user_id == "test_user"
        assert audit_entry.timestamp == test_time
        assert audit_entry.correlation_id == "test-correlation"