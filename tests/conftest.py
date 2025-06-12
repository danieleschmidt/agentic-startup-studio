"""
Shared test fixtures and configuration for Phase 1 Data Ingestion test suite.

This module provides common fixtures, test utilities, and configuration
that can be used across all test modules to ensure consistency and reduce duplication.
"""

import pytest
import asyncio
import os
import tempfile
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from uuid import UUID, uuid4
from unittest.mock import Mock, AsyncMock, patch

from pipeline.models.idea import (
    Idea, IdeaDraft, IdeaStatus, PipelineStage, IdeaCategory,
    ValidationResult, DuplicateCheckResult, QueryParams, AuditEntry, IdeaSummary
)
from pipeline.config.settings import (
    DatabaseConfig, ValidationConfig, EmbeddingConfig, LoggingConfig, IngestionConfig
)


# Test Database Configuration
@pytest.fixture(scope="session")
def test_db_config() -> DatabaseConfig:
    """Provide test database configuration."""
    return DatabaseConfig(
        host="localhost",
        port=5432,
        database="test_startup_studio",
        user="test_user",
        password="test_password",
        pool_size=5,
        max_overflow=10,
        enable_vector_extension=True
    )


@pytest.fixture(scope="session")
def test_validation_config() -> ValidationConfig:
    """Provide test validation configuration."""
    return ValidationConfig(
        min_title_length=5,
        max_title_length=200,
        min_description_length=20,
        max_description_length=5000,
        similarity_threshold=0.8,
        title_fuzzy_threshold=0.7,
        content_quality_threshold=0.6,
        max_tags=10,
        require_evidence=False,  # More lenient for testing
        enable_duplicate_detection=True,
        enable_content_analysis=True
    )


@pytest.fixture(scope="session")
def test_embedding_config() -> EmbeddingConfig:
    """Provide test embedding configuration."""
    return EmbeddingConfig(
        model_name="test-embedding-model",
        api_key="test-api-key",
        max_tokens=8192,
        batch_size=10,
        cache_embeddings=True,
        vector_dimensions=1536
    )


@pytest.fixture(scope="session")
def test_logging_config() -> LoggingConfig:
    """Provide test logging configuration."""
    return LoggingConfig(
        level="DEBUG",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        file_path=None,  # No file logging in tests
        max_file_size=10485760,
        backup_count=3,
        enable_structured_logging=False
    )


@pytest.fixture(scope="session")
def test_ingestion_config(
    test_db_config, test_validation_config, test_embedding_config, test_logging_config
) -> IngestionConfig:
    """Provide complete test ingestion configuration."""
    return IngestionConfig(
        database=test_db_config,
        validation=test_validation_config,
        embedding=test_embedding_config,
        logging=test_logging_config,
        environment="test",
        debug_mode=True,
        async_timeout=30.0
    )


# Sample Data Fixtures
@pytest.fixture
def sample_idea_draft() -> IdeaDraft:
    """Provide sample idea draft for testing."""
    return IdeaDraft(
        title="AI-powered productivity tool",
        description="Revolutionary solution using artificial intelligence to boost workplace productivity with automated task management and intelligent recommendations",
        category=IdeaCategory.AI_ML,
        tags=["ai", "productivity", "automation"],
        evidence="Market research shows 40% productivity increase in pilot studies"
    )


@pytest.fixture
def sample_idea() -> Idea:
    """Provide sample idea for testing."""
    return Idea(
        idea_id=uuid4(),
        title="AI-powered productivity tool",
        description="Revolutionary solution using artificial intelligence to boost workplace productivity",
        category=IdeaCategory.AI_ML,
        tags=["ai", "productivity", "automation"],
        evidence="Market research shows 40% productivity increase",
        status=IdeaStatus.DRAFT,
        current_stage=PipelineStage.IDEATE,
        stage_progress=0.3,
        created_by="test_user",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        embedding_vector=None
    )


@pytest.fixture
def sample_raw_idea_data() -> Dict[str, Any]:
    """Provide sample raw idea data for testing."""
    return {
        "title": "AI-powered productivity tool",
        "description": "Revolutionary solution using artificial intelligence to boost workplace productivity with automated task management",
        "category": "ai_ml",
        "tags": ["ai", "productivity", "automation"],
        "evidence": "Market research shows 40% productivity increase"
    }


@pytest.fixture
def sample_validation_result_valid() -> ValidationResult:
    """Provide valid validation result for testing."""
    result = ValidationResult(is_valid=True)
    result.add_warning("Consider adding more specific evidence")
    return result


@pytest.fixture
def sample_validation_result_invalid() -> ValidationResult:
    """Provide invalid validation result for testing."""
    result = ValidationResult(is_valid=False)
    result.add_error("Title too short")
    result.add_error("Description missing evidence")
    return result


@pytest.fixture
def sample_duplicate_result_clean() -> DuplicateCheckResult:
    """Provide clean duplicate check result for testing."""
    return DuplicateCheckResult(found_similar=False)


@pytest.fixture
def sample_duplicate_result_found() -> DuplicateCheckResult:
    """Provide duplicate check result with matches for testing."""
    result = DuplicateCheckResult(found_similar=True)
    result.exact_matches = [uuid4()]
    result.similar_ideas = [uuid4(), uuid4()]
    result.similarity_scores = {
        str(uuid4()): 0.85,
        str(uuid4()): 0.82
    }
    return result


@pytest.fixture
def sample_idea_summary() -> IdeaSummary:
    """Provide sample idea summary for testing."""
    return IdeaSummary(
        id=uuid4(),
        title="AI-powered productivity tool",
        status=IdeaStatus.DRAFT,
        stage=PipelineStage.IDEATE,
        progress=0.3,
        created_at=datetime.now(timezone.utc)
    )


@pytest.fixture
def sample_query_params() -> QueryParams:
    """Provide sample query parameters for testing."""
    return QueryParams(
        status=IdeaStatus.DRAFT,
        category=IdeaCategory.AI_ML,
        stage=PipelineStage.IDEATE,
        limit=10,
        offset=0,
        sort_by="created_at",
        sort_order="desc"
    )


@pytest.fixture
def sample_audit_entry() -> AuditEntry:
    """Provide sample audit entry for testing."""
    return AuditEntry(
        action="created",
        user_id="test_user",
        timestamp=datetime.now(timezone.utc),
        details={"field": "status", "old_value": None, "new_value": "draft"},
        correlation_id="test-correlation-123"
    )


# Mock Fixtures
@pytest.fixture
def mock_database_connection():
    """Provide mock database connection."""
    mock_conn = Mock()
    mock_conn.execute = AsyncMock()
    mock_conn.fetch = AsyncMock()
    mock_conn.fetchrow = AsyncMock()
    mock_conn.fetchval = AsyncMock()
    mock_conn.close = AsyncMock()
    return mock_conn


@pytest.fixture
def mock_connection_pool():
    """Provide mock connection pool."""
    mock_pool = Mock()
    mock_pool.acquire = AsyncMock()
    mock_pool.close = AsyncMock()
    return mock_pool


@pytest.fixture
def mock_embedding_client():
    """Provide mock embedding client."""
    mock_client = Mock()
    mock_client.generate_embedding = AsyncMock(return_value=[0.1] * 1536)
    mock_client.generate_embeddings = AsyncMock(return_value=[[0.1] * 1536])
    return mock_client


# Test Utilities
@pytest.fixture
def temp_directory():
    """Provide temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_environment_variables():
    """Provide mock environment variables for testing."""
    test_env = {
        "DB_HOST": "localhost",
        "DB_PORT": "5432",
        "DB_NAME": "test_db",
        "DB_USER": "test_user",
        "DB_PASSWORD": "test_pass",
        "EMBEDDING_API_KEY": "test-key",
        "ENVIRONMENT": "test"
    }
    
    with patch.dict(os.environ, test_env, clear=False):
        yield test_env


# Async Test Configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Test ID Generation
@pytest.fixture
def test_uuid() -> UUID:
    """Provide deterministic UUID for testing."""
    return UUID("12345678-1234-5678-9abc-123456789012")


@pytest.fixture
def multiple_test_uuids() -> list[UUID]:
    """Provide multiple deterministic UUIDs for testing."""
    return [
        UUID("12345678-1234-5678-9abc-123456789012"),
        UUID("12345678-1234-5678-9abc-123456789013"),
        UUID("12345678-1234-5678-9abc-123456789014"),
        UUID("12345678-1234-5678-9abc-123456789015")
    ]


# Test Data Cleanup
@pytest.fixture(autouse=True)
def cleanup_test_data():
    """Automatically cleanup test data after each test."""
    yield
    # Cleanup logic would go here if needed
    # For now, using mocks so no actual cleanup required


# Performance Test Helpers
@pytest.fixture
def performance_timer():
    """Provide performance timing utilities for tests."""
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = datetime.now()
        
        def stop(self):
            self.end_time = datetime.now()
        
        @property
        def elapsed_seconds(self) -> float:
            if self.start_time and self.end_time:
                return (self.end_time - self.start_time).total_seconds()
            return 0.0
    
    return Timer()


# Custom Assertion Helpers
def assert_valid_uuid(value: str) -> UUID:
    """Assert that a string is a valid UUID and return the UUID object."""
    try:
        return UUID(value)
    except ValueError:
        pytest.fail(f"Expected valid UUID, got: {value}")


def assert_datetime_recent(dt: datetime, max_age_seconds: int = 60):
    """Assert that a datetime is recent (within max_age_seconds)."""
    now = datetime.now(timezone.utc)
    age = (now - dt).total_seconds()
    assert age <= max_age_seconds, f"Datetime {dt} is too old ({age}s > {max_age_seconds}s)"


def assert_dict_contains(actual: dict, expected_subset: dict):
    """Assert that actual dict contains all key-value pairs from expected_subset."""
    for key, expected_value in expected_subset.items():
        assert key in actual, f"Missing key: {key}"
        assert actual[key] == expected_value, f"Key {key}: expected {expected_value}, got {actual[key]}"


# Export test utilities for easy import
__all__ = [
    'assert_valid_uuid',
    'assert_datetime_recent', 
    'assert_dict_contains'
]