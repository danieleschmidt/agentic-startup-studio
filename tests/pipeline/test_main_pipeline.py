"""
Test suite for Main Pipeline - End-to-end pipeline orchestration.

Tests comprehensive pipeline execution with proper mocking of all external services.
Covers budget enforcement, error recovery, quality gates, and performance requirements.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import dataclass

from pipeline.main_pipeline import MainPipeline, PipelineResult, get_main_pipeline
from pipeline.services.budget_sentinel import BudgetCategory
from pipeline.services.pitch_deck_generator import InvestorType
from pipeline.services.campaign_generator import MVPType


@pytest.fixture
def mock_budget_sentinel():
    """Mock budget sentinel for cost tracking tests."""
    mock_sentinel = Mock()
    
    # Create proper async context manager for track_operation
    class AsyncContextManager:
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None
    
    mock_sentinel.track_operation = Mock(return_value=AsyncContextManager())
    mock_sentinel.get_budget_status = AsyncMock(return_value={
        'total_budget': 62.0,
        'total_spent': 30.0,
        'category_budgets': {
            'TOTAL': {'limit': 62.0, 'spent': 30.0}
        }
    })
    return mock_sentinel


@pytest.fixture
def mock_workflow_orchestrator():
    """Mock workflow orchestrator for state management tests."""
    mock_orchestrator = Mock()
    mock_orchestrator.start_pipeline = AsyncMock()
    mock_orchestrator.update_stage = AsyncMock()
    mock_orchestrator.complete_pipeline = AsyncMock()
    return mock_orchestrator


@pytest.fixture  
def mock_evidence_collector():
    """Mock evidence collector for RAG research tests."""
    mock_collector = Mock()
    mock_collector.collect_evidence = AsyncMock(return_value={
        'market_research': [
            Mock(
                claim_text="Market size is $10B",
                citation_url="https://example.com/research",
                citation_title="Market Research Report",
                composite_score=0.85
            )
        ],
        'technology_trends': [
            Mock(
                claim_text="AI adoption growing 40% YoY",
                citation_url="https://example.com/trends", 
                citation_title="Tech Trends 2024",
                composite_score=0.78
            )
        ]
    })
    return mock_collector


@pytest.fixture
def mock_pitch_deck_generator():
    """Mock pitch deck generator for investment materials tests."""
    mock_generator = Mock()
    mock_pitch_deck = Mock()
    mock_pitch_deck.startup_name = "TestStartup"
    mock_pitch_deck.investor_type = InvestorType.SEED
    mock_pitch_deck.slides = [Mock(), Mock(), Mock()]  # 3 slides
    mock_pitch_deck.overall_quality_score = 0.82
    mock_pitch_deck.completeness_score = 0.9
    mock_pitch_deck.evidence_strength_score = 0.75
    
    # Add slide details
    for i, slide in enumerate(mock_pitch_deck.slides):
        slide.slide_type = Mock()
        slide.slide_type.value = f"slide_type_{i}"
        slide.title = f"Slide {i+1}"
        slide.quality_score = 0.8
        slide.supporting_evidence = [Mock(), Mock()]
    
    mock_generator.generate_pitch_deck = AsyncMock(return_value=mock_pitch_deck)
    return mock_generator


@pytest.fixture
def mock_campaign_generator():
    """Mock campaign generator for marketing automation tests."""
    mock_generator = Mock()
    
    # Mock smoke test campaign
    mock_campaign = Mock()
    mock_campaign.name = "TestStartup Smoke Test"
    mock_campaign.campaign_type = Mock()
    mock_campaign.campaign_type.value = "smoke_test"
    mock_campaign.status = Mock()
    mock_campaign.status.value = "active"
    mock_campaign.budget_limit = 25.0
    mock_campaign.assets = [Mock(), Mock()]
    mock_campaign.relevance_score = 0.78
    mock_campaign.engagement_prediction = 0.65
    mock_campaign.google_ads_campaign_id = "ads_123"
    mock_campaign.posthog_project_id = "ph_456"
    mock_campaign.landing_page_url = "https://teststartup.com"
    
    # Mock MVP result
    mock_mvp_result = Mock()
    mock_mvp_result.mvp_type = Mock()
    mock_mvp_result.mvp_type.value = "landing_page"
    mock_mvp_result.generated_files = ["index.html", "app.py", "style.css"]
    mock_mvp_result.deployment_url = "https://mvp.teststartup.com"
    mock_mvp_result.generation_status = "completed"
    mock_mvp_result.deployment_status = "deployed"
    mock_mvp_result.code_quality_score = 0.88
    mock_mvp_result.deployment_success = True
    mock_mvp_result.generation_cost = 3.5
    
    mock_generator.generate_smoke_test_campaign = AsyncMock(return_value=mock_campaign)
    mock_generator.execute_campaign = AsyncMock(return_value=mock_campaign)
    mock_generator.generate_mvp = AsyncMock(return_value=mock_mvp_result)
    return mock_generator


@pytest.fixture
def mock_idea_manager():
    """Mock idea manager for data ingestion tests."""
    mock_manager = Mock()
    mock_manager.create_idea = AsyncMock(return_value={'id': 'idea_123'})
    return mock_manager


@pytest.fixture
def mock_startup_validator():
    """Mock startup validator for validation tests.""" 
    mock_validator = Mock()
    mock_validator.validate_startup_idea = AsyncMock(return_value={
        'is_valid': True,
        'overall_score': 0.85,
        'market_score': 0.8,
        'technical_score': 0.9,
        'validation_details': {
            'market_potential': 'high',
            'technical_feasibility': 'feasible',
            'competitive_advantage': 'strong'
        }
    })
    return mock_validator


@pytest.fixture
def mock_pipeline_dependencies(
    mock_budget_sentinel,
    mock_workflow_orchestrator, 
    mock_evidence_collector,
    mock_pitch_deck_generator,
    mock_campaign_generator,
    mock_idea_manager,
    mock_startup_validator
):
    """Setup all pipeline dependencies with mocks."""
    with patch('pipeline.main_pipeline.get_budget_sentinel', return_value=mock_budget_sentinel), \
         patch('pipeline.main_pipeline.get_workflow_orchestrator', return_value=mock_workflow_orchestrator), \
         patch('pipeline.main_pipeline.get_evidence_collector', return_value=mock_evidence_collector), \
         patch('pipeline.main_pipeline.get_pitch_deck_generator', return_value=mock_pitch_deck_generator), \
         patch('pipeline.main_pipeline.get_campaign_generator', return_value=mock_campaign_generator), \
         patch('pipeline.main_pipeline.create_idea_manager', new=AsyncMock(return_value=mock_idea_manager)), \
         patch('pipeline.main_pipeline.create_validator', new=AsyncMock(return_value=mock_startup_validator)), \
         patch('pipeline.main_pipeline.get_settings', return_value=Mock()):
        
        yield {
            'budget_sentinel': mock_budget_sentinel,
            'workflow_orchestrator': mock_workflow_orchestrator,
            'evidence_collector': mock_evidence_collector,
            'pitch_deck_generator': mock_pitch_deck_generator,
            'campaign_generator': mock_campaign_generator,
            'idea_manager': mock_idea_manager,
            'startup_validator': mock_startup_validator
        }


class TestMainPipelineAcceptance:
    """Acceptance tests for complete pipeline execution behavior."""
    
    @pytest.mark.asyncio
    async def test_given_valid_startup_idea_when_execute_full_pipeline_then_completes_all_phases_successfully(
        self, mock_pipeline_dependencies
    ):
        """
        GIVEN a valid startup idea
        WHEN executing the full pipeline 
        THEN all 4 phases complete successfully with quality metrics
        """
        # Arrange
        pipeline = MainPipeline()
        startup_idea = "AI-powered productivity assistant for remote teams"
        
        # Act
        result = await pipeline.execute_full_pipeline(
            startup_idea=startup_idea,
            target_investor=InvestorType.SEED,
            generate_mvp=True,
            max_total_budget=60.0
        )
        
        # Assert
        assert result.startup_idea == startup_idea
        assert len(result.phases_completed) == 4
        assert len(result.phases_failed) == 0
        assert len(result.errors) == 0
        assert result.overall_quality_score > 0.0
        assert result.budget_utilization > 0.0
        assert result.execution_time_seconds > 0.0
        assert result.completed_at is not None
        
        # Verify phase results are populated
        assert result.validation_result is not None
        assert result.evidence_collection_result is not None  
        assert result.pitch_deck_result is not None
        assert result.campaign_result is not None
        assert result.mvp_result is not None


    @pytest.mark.asyncio
    async def test_given_budget_limit_when_execute_pipeline_then_tracks_operation_within_budget(
        self, mock_pipeline_dependencies
    ):
        """
        GIVEN a budget limit for pipeline execution
        WHEN executing the pipeline
        THEN budget tracking is enforced and utilization calculated
        """
        # Arrange
        pipeline = MainPipeline()
        max_budget = 50.0
        deps = mock_pipeline_dependencies
        
        # Act
        result = await pipeline.execute_full_pipeline(
            startup_idea="Test idea",
            max_total_budget=max_budget
        )
        
        # Assert - Budget tracking was called
        deps['budget_sentinel'].track_operation.assert_called_once_with(
            "main_pipeline",
            "execute_full_pipeline",
            BudgetCategory.INFRASTRUCTURE,
            max_budget
        )
        
        # Budget utilization calculated
        deps['budget_sentinel'].get_budget_status.assert_called_once()
        assert result.budget_utilization == 30.0 / 62.0  # Based on mock return


    @pytest.mark.asyncio
    async def test_given_pipeline_execution_when_phase_fails_then_continues_gracefully_with_error_tracking(
        self, mock_pipeline_dependencies
    ):
        """
        GIVEN pipeline execution in progress
        WHEN a phase fails with an exception
        THEN pipeline continues and tracks the failure gracefully
        """
        # Arrange
        pipeline = MainPipeline()
        deps = mock_pipeline_dependencies
        
        # Make Phase 2 fail
        deps['evidence_collector'].collect_evidence.side_effect = RuntimeError("Evidence collection failed")
        
        # Act
        result = await pipeline.execute_full_pipeline(
            startup_idea="Test idea",
            generate_mvp=False
        )
        
        # Assert
        assert len(result.phases_failed) >= 1
        assert "phase_2_processing" in result.phases_failed
        assert any("Phase 2 failed" in error for error in result.errors)
        assert result.completed_at is not None
        
        # Other phases should still attempt to execute
        assert "phase_1_ingestion" in result.phases_completed


class TestMainPipelineBudgetEnforcement:
    """Tests for budget tracking and cost enforcement."""
    
    @pytest.mark.asyncio
    async def test_given_budget_sentinel_context_when_pipeline_executes_then_budget_operation_tracked(
        self, mock_pipeline_dependencies
    ):
        """
        GIVEN budget sentinel context manager
        WHEN pipeline executes
        THEN budget operation is properly tracked
        """
        # Arrange
        pipeline = MainPipeline()
        deps = mock_pipeline_dependencies
        
        # Setup budget sentinel context manager
        budget_context = AsyncMock()
        deps['budget_sentinel'].track_operation.return_value.__aenter__ = AsyncMock(return_value=budget_context)
        deps['budget_sentinel'].track_operation.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Act
        result = await pipeline.execute_full_pipeline(startup_idea="Test idea")
        
        # Assert
        deps['budget_sentinel'].track_operation.assert_called_once()
        assert result.budget_utilization >= 0.0


class TestMainPipelineServiceCollaboration:
    """Tests for service interaction and collaboration patterns."""
    
    @pytest.mark.asyncio
    async def test_given_pipeline_phases_when_executed_then_services_collaborate_in_correct_sequence(
        self, mock_pipeline_dependencies
    ):
        """
        GIVEN pipeline execution
        WHEN phases are executed
        THEN services are called in the correct sequence
        """
        # Arrange
        pipeline = MainPipeline()  
        deps = mock_pipeline_dependencies
        
        # Act
        await pipeline.execute_full_pipeline(startup_idea="Test idea")
        
        # Assert service collaboration sequence
        # Phase 1: Validation and storage
        deps['startup_validator'].validate_startup_idea.assert_called_once()
        deps['idea_manager'].create_idea.assert_called_once()
        
        # Phase 2: Evidence collection  
        deps['evidence_collector'].collect_evidence.assert_called_once()
        
        # Phase 3: Pitch deck generation
        deps['pitch_deck_generator'].generate_pitch_deck.assert_called_once()
        
        # Phase 4: Campaign and MVP generation
        deps['campaign_generator'].generate_smoke_test_campaign.assert_called_once()
        deps['campaign_generator'].execute_campaign.assert_called_once()
        deps['campaign_generator'].generate_mvp.assert_called_once()


class TestMainPipelineErrorRecovery:
    """Tests for error handling and recovery patterns."""
    
    @pytest.mark.asyncio
    async def test_given_multiple_phase_failures_when_pipeline_executes_then_tracks_all_failures(
        self, mock_pipeline_dependencies
    ):
        """
        GIVEN multiple service failures
        WHEN pipeline executes  
        THEN all failures are tracked and pipeline completes
        """
        # Arrange
        pipeline = MainPipeline()
        deps = mock_pipeline_dependencies
        
        # Make multiple phases fail
        deps['startup_validator'].validate_startup_idea.side_effect = RuntimeError("Validation failed")
        deps['evidence_collector'].collect_evidence.side_effect = RuntimeError("Evidence failed")
        
        # Act
        result = await pipeline.execute_full_pipeline(startup_idea="Test idea")
        
        # Assert 
        assert len(result.phases_failed) >= 2
        assert len(result.errors) >= 2
        assert result.completed_at is not None


class TestMainPipelinePerformance:
    """Tests for performance requirements and timing constraints."""
    
    @pytest.mark.asyncio
    async def test_given_pipeline_execution_when_completed_then_execution_time_recorded(
        self, mock_pipeline_dependencies
    ):
        """
        GIVEN pipeline execution
        WHEN pipeline completes
        THEN execution time is properly recorded
        """
        # Arrange
        pipeline = MainPipeline()
        
        # Act
        result = await pipeline.execute_full_pipeline(startup_idea="Test idea")
        
        # Assert
        assert result.execution_time_seconds > 0.0
        assert result.started_at is not None  
        assert result.completed_at is not None
        assert result.completed_at > result.started_at


class TestMainPipelineQualityGates:
    """Tests for quality scoring and validation gates."""
    
    @pytest.mark.asyncio  
    async def test_given_pipeline_results_when_calculate_metrics_then_quality_score_computed(
        self, mock_pipeline_dependencies
    ):
        """
        GIVEN pipeline execution results
        WHEN final metrics are calculated
        THEN overall quality score is computed from phase scores
        """
        # Arrange
        pipeline = MainPipeline()
        
        # Act
        result = await pipeline.execute_full_pipeline(startup_idea="Test idea")
        
        # Assert
        assert result.overall_quality_score > 0.0
        # Should average quality scores from validation, pitch deck, campaign, MVP
        expected_avg = (0.85 + 0.82 + 0.78 + 0.88) / 4  # Based on mock scores
        assert abs(result.overall_quality_score - expected_avg) < 0.01


class TestMainPipelineReportGeneration:
    """Tests for pipeline reporting and recommendations."""
    
    @pytest.mark.asyncio
    async def test_given_pipeline_result_when_generate_report_then_comprehensive_report_created(
        self, mock_pipeline_dependencies
    ):
        """
        GIVEN a completed pipeline result
        WHEN generating a pipeline report
        THEN comprehensive report with all metrics is created
        """
        # Arrange
        pipeline = MainPipeline()
        result = await pipeline.execute_full_pipeline(startup_idea="Test idea")
        
        # Act  
        report = await pipeline.generate_pipeline_report(result)
        
        # Assert
        assert 'execution_summary' in report
        assert 'quality_metrics' in report
        assert 'budget_tracking' in report
        assert 'phase_results' in report
        assert 'recommendations' in report
        
        # Verify report content
        exec_summary = report['execution_summary']
        assert exec_summary['execution_id'] == result.execution_id
        assert exec_summary['phases_completed'] == len(result.phases_completed)
        assert exec_summary['overall_success'] == (len(result.phases_failed) == 0)


class TestMainPipelineSingleton:
    """Tests for singleton pattern implementation."""
    
    def test_given_multiple_calls_when_get_main_pipeline_then_returns_same_instance(self):
        """
        GIVEN multiple calls to get_main_pipeline
        WHEN getting pipeline instances
        THEN same singleton instance is returned
        """
        # Act
        pipeline1 = get_main_pipeline()
        pipeline2 = get_main_pipeline()
        
        # Assert
        assert pipeline1 is pipeline2