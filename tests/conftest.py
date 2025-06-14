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
from dataclasses import dataclass

from pipeline.models.idea import (
    Idea, IdeaDraft, IdeaStatus, PipelineStage, IdeaCategory,
    ValidationResult, DuplicateCheckResult, QueryParams, AuditEntry, IdeaSummary
)
from pipeline.config.settings import (
    DatabaseConfig, ValidationConfig, EmbeddingConfig, LoggingConfig, IngestionConfig
)
from pipeline.services.budget_sentinel import BudgetCategory
from pipeline.services.pitch_deck_generator import InvestorType
from pipeline.services.campaign_generator import MVPType, MVPRequest
from pipeline.services.evidence_collector import ResearchDomain
@dataclass
class StartupTestCase:
    """Test case for startup idea validation."""
    name: str
    description: str
    expected_quality_threshold: float
    expected_validation_score: float
    expected_market_potential: str
    target_investor: InvestorType
    should_generate_mvp: bool = True


# Comprehensive test scenarios with diverse startup ideas
STARTUP_TEST_CASES = [
    StartupTestCase(
        name="ai_productivity_assistant",
        description="AI-powered productivity assistant that learns individual work patterns and automatically prioritizes tasks, schedules meetings, and manages email responses for remote teams",
        expected_quality_threshold=0.75,
        expected_validation_score=0.8,
        expected_market_potential="high",
        target_investor=InvestorType.SEED
    ),
    StartupTestCase(
        name="sustainable_food_delivery",
        description="Zero-waste food delivery platform using reusable containers and electric bikes, partnering with local restaurants to reduce packaging waste and carbon footprint",
        expected_quality_threshold=0.70,
        expected_validation_score=0.75,
        expected_market_potential="medium",
        target_investor=InvestorType.SERIES_A
    ),
    StartupTestCase(
        name="blockchain_supply_chain",
        description="Blockchain-based supply chain transparency platform for luxury goods authentication, tracking products from manufacture to consumer with immutable records",
        expected_quality_threshold=0.65,
        expected_validation_score=0.70,
        expected_market_potential="medium",
        target_investor=InvestorType.SERIES_A
    ),
    StartupTestCase(
        name="mental_health_vr",
        description="Virtual reality therapy platform for anxiety and PTSD treatment, providing immersive therapeutic environments with AI-guided sessions and real-time biometric monitoring",
        expected_quality_threshold=0.80,
        expected_validation_score=0.85,
        expected_market_potential="high",
        target_investor=InvestorType.SEED
    ),
    StartupTestCase(
        name="elderly_companion_robot",
        description="AI-powered companion robot for elderly care providing medication reminders, emergency detection, social interaction, and health monitoring in home environments",
        expected_quality_threshold=0.75,
        expected_validation_score=0.82,
        expected_market_potential="high",
        target_investor=InvestorType.SERIES_A,
        should_generate_mvp=False  # Complex hardware project
    )
]


class MockExternalServices:
    """Comprehensive mocking for all external services."""
    
    def __init__(self):
        self.setup_mocks()
    
    def setup_mocks(self):
        """Setup all external service mocks."""
        # OpenAI/LLM Service Mocks
        self.openai_mock = Mock()
        self.openai_mock.chat.completions.create = AsyncMock()
        
        # Google Ads API Mock
        self.google_ads_mock = Mock()
        self.google_ads_mock.create_campaign = AsyncMock(return_value={
            'campaign_id': 'gads_12345',
            'status': 'ENABLED',
            'budget': 25.0
        })
        
        # PostHog Analytics Mock
        self.posthog_mock = Mock()
        self.posthog_mock.create_project = AsyncMock(return_value={
            'project_id': 'ph_67890',
            'api_key': 'phc_test_key'
        })
        
        # GPT-Engineer Mock
        self.gpt_engineer_mock = Mock()
        self.gpt_engineer_mock.generate_mvp = AsyncMock(return_value={
            'status': 'completed',
            'files': ['app.py', 'index.html', 'style.css'],
            'deployment_url': 'https://mvp-demo.fly.dev'
        })
        
        # Fly.io Deployment Mock
        self.flyio_mock = Mock()
        self.flyio_mock.deploy = AsyncMock(return_value={
            'app_name': 'startup-mvp-test',
            'url': 'https://startup-mvp-test.fly.dev',
            'status': 'deployed'
        })
        
        # Search Engine Mocks (for RAG evidence collection)
        self.search_mock = Mock()
        self.search_mock.search = AsyncMock(return_value=[
            {
                'title': 'Market Research Report 2024',
                'url': 'https://research.example.com/report-2024',
                'snippet': 'Market size projected to reach $50B by 2027',
                'quality_score': 0.85
            },
            {
                'title': 'Technology Trends Analysis',
                'url': 'https://tech.example.com/trends',
                'snippet': 'AI adoption accelerating across industries',
                'quality_score': 0.78
            }
        ])


@pytest.fixture
def mock_external_services():
    """Fixture providing comprehensive external service mocking."""
    return MockExternalServices()


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
@pytest.fixture
def enhanced_pipeline_mocks(mock_external_services):
    """Enhanced pipeline mocks with realistic data and error scenarios."""
    
    # Budget Sentinel Mock
    budget_sentinel = Mock()
    budget_context = AsyncMock()
    budget_context.__aenter__ = AsyncMock(return_value=budget_context)
    budget_context.__aexit__ = AsyncMock(return_value=None)
    budget_sentinel.track_operation = Mock(return_value=budget_context)
    budget_sentinel.get_budget_status = AsyncMock(return_value={
        'total_budget': 62.0,
        'total_spent': 35.0,
        'category_budgets': {
            'INFRASTRUCTURE': {'limit': 10.0, 'spent': 2.0},
            'AI_SERVICES': {'limit': 30.0, 'spent': 18.0},
            'EXTERNAL_APIS': {'limit': 22.0, 'spent': 15.0}
        }
    })
    
    # Workflow Orchestrator Mock
    workflow_orchestrator = Mock()
    workflow_orchestrator.start_pipeline = AsyncMock()
    workflow_orchestrator.update_stage = AsyncMock()
    workflow_orchestrator.complete_pipeline = AsyncMock()
    
    # Enhanced Evidence Collector Mock with realistic research domains
    evidence_collector = Mock()
    evidence_collector.collect_evidence = AsyncMock(side_effect=lambda claim, research_domains, **kwargs: {
        domain.name: [
            Mock(
                claim_text=f"Evidence for {claim[:50]}... in {domain.name}",
                citation_url=f"https://research.example.com/{domain.name}/report-{i}",
                citation_title=f"{domain.name.title()} Research Report {i}",
                composite_score=min(0.9, domain.quality_threshold + 0.1 + i * 0.05)
            )
            for i in range(domain.min_evidence_count)
        ]
        for domain in research_domains
    })
    
    # Enhanced Pitch Deck Generator Mock
    pitch_deck_generator = Mock()
    def create_pitch_deck_mock(startup_idea, evidence_by_domain, target_investor, **kwargs):
        quality_base = 0.75 if "AI" in startup_idea else 0.65
        mock_deck = Mock()
        mock_deck.startup_name = startup_idea.split()[0].title() + "Corp"
        mock_deck.investor_type = target_investor
        mock_deck.slides = [Mock() for _ in range(12)]  # Standard pitch deck length
        mock_deck.overall_quality_score = quality_base + 0.1
        mock_deck.completeness_score = 0.9
        mock_deck.evidence_strength_score = quality_base
        
        # Add realistic slide details
        slide_types = ["title", "problem", "solution", "market", "product", "business_model", 
                      "competition", "team", "financials", "funding", "timeline", "appendix"]
        for i, slide in enumerate(mock_deck.slides):
            slide.slide_type = Mock()
            slide.slide_type.value = slide_types[i] if i < len(slide_types) else f"slide_{i}"
            slide.title = slide_types[i].title().replace("_", " ") if i < len(slide_types) else f"Slide {i+1}"
            slide.quality_score = quality_base + (i % 3) * 0.05
            slide.supporting_evidence = [Mock() for _ in range(2 + i % 3)]
        
        return mock_deck
    
    pitch_deck_generator.generate_pitch_deck = AsyncMock(side_effect=create_pitch_deck_mock)
    
    # Enhanced Campaign Generator Mock
    campaign_generator = Mock()
    
    def create_campaign_mock(pitch_deck, budget_limit, duration_days, **kwargs):
        mock_campaign = Mock()
        mock_campaign.name = f"{pitch_deck.startup_name} Smoke Test Campaign"
        mock_campaign.campaign_type = Mock()
        mock_campaign.campaign_type.value = "smoke_test"
        mock_campaign.status = Mock()
        mock_campaign.status.value = "active"
        mock_campaign.budget_limit = budget_limit
        mock_campaign.assets = [Mock() for _ in range(5)]  # Ad assets
        mock_campaign.relevance_score = 0.75 + (hash(pitch_deck.startup_name) % 10) * 0.02
        mock_campaign.engagement_prediction = 0.60 + (hash(pitch_deck.startup_name) % 15) * 0.02
        mock_campaign.google_ads_campaign_id = f"gads_{hash(pitch_deck.startup_name) % 100000}"
        mock_campaign.posthog_project_id = f"ph_{hash(pitch_deck.startup_name) % 100000}"
        mock_campaign.landing_page_url = f"https://{pitch_deck.startup_name.lower().replace(' ', '')}.test.com"
        return mock_campaign
    
    campaign_generator.generate_smoke_test_campaign = AsyncMock(side_effect=create_campaign_mock)
    campaign_generator.execute_campaign = AsyncMock(side_effect=lambda campaign: campaign)
    
    def create_mvp_mock(mvp_request, max_cost, **kwargs):
        mock_mvp = Mock()
        mock_mvp.mvp_type = mvp_request.mvp_type
        mock_mvp.generated_files = mvp_request.tech_stack + ["README.md", "requirements.txt"]
        mock_mvp.deployment_url = f"https://{mvp_request.startup_name.lower().replace(' ', '')}-mvp.fly.dev"
        mock_mvp.generation_status = "completed"
        mock_mvp.deployment_status = "deployed"
        mock_mvp.code_quality_score = 0.80 + (hash(mvp_request.startup_name) % 10) * 0.02
        mock_mvp.deployment_success = True
        mock_mvp.generation_cost = min(max_cost, 3.5 + (hash(mvp_request.startup_name) % 10) * 0.1)
        return mock_mvp
    
    campaign_generator.generate_mvp = AsyncMock(side_effect=create_mvp_mock)
    
    # Idea Manager Mock
    idea_manager = Mock()
    idea_manager.create_idea = AsyncMock(side_effect=lambda idea_data: {
        'id': f"idea_{hash(idea_data.get('idea', ''))[:8]}",
        'created_at': datetime.utcnow().isoformat(),
        'validation_score': idea_data.get('validation_score', 0.0)
    })
    
    # Startup Validator Mock with realistic scoring
    startup_validator = Mock()
    def validate_startup_mock(idea_data):
        idea = idea_data.get('idea', '')
        
        # Realistic scoring based on content
        market_score = 0.8 if any(word in idea.lower() for word in ['ai', 'platform', 'saas']) else 0.6
        market_score += 0.1 if 'remote' in idea.lower() else 0
        
        technical_score = 0.9 if 'ai' in idea.lower() else 0.7
        technical_score += 0.1 if any(word in idea.lower() for word in ['blockchain', 'vr', 'robot']) else 0
        
        overall_score = (market_score + technical_score) / 2
        
        return {
            'is_valid': overall_score >= 0.6,
            'overall_score': overall_score,
            'market_score': market_score,
            'technical_score': technical_score,
            'validation_details': {
                'market_potential': 'high' if market_score >= 0.8 else 'medium' if market_score >= 0.6 else 'low',
                'technical_feasibility': 'feasible' if technical_score >= 0.7 else 'challenging',
                'competitive_advantage': 'strong' if overall_score >= 0.8 else 'moderate'
            }
        }
    
    startup_validator.validate_startup_idea = AsyncMock(side_effect=validate_startup_mock)
    
    return {
        'budget_sentinel': budget_sentinel,
        'workflow_orchestrator': workflow_orchestrator,
        'evidence_collector': evidence_collector,
        'pitch_deck_generator': pitch_deck_generator,
        'campaign_generator': campaign_generator,
        'idea_manager': idea_manager,
        'startup_validator': startup_validator,
        'external_services': mock_external_services
    }