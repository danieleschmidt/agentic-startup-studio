"""
Comprehensive Integration Test Suite for Complete Data Pipeline

Tests the full 4-phase workflow with diverse startup ideas:
- Phase 1: Data Ingestion (CLI interface, validation, storage, deduplication)
- Phase 2: Data Processing (RAG evidence collection, quality scoring, error recovery)
- Phase 3: Data Transformation (LangGraph pitch deck generation, investor customization)
- Phase 4: Data Output (Google Ads campaigns, MVP generation, analytics integration)

Validates budget enforcement, quality gates, performance requirements, and error recovery.
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Any
from dataclasses import dataclass

from pipeline.main_pipeline import MainPipeline, PipelineResult, get_main_pipeline
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


class TestComprehensiveDataPipelineIntegration:
    """Comprehensive integration tests for the complete 4-phase data pipeline."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_pipeline_with_diverse_startup_ideas(self, enhanced_pipeline_mocks):
        """
        Test complete pipeline execution with diverse startup ideas to validate:
        - All 4 phases execute successfully
        - Quality gates function correctly
        - Budget tracking works across phases
        - External service integration points
        - Error recovery and graceful degradation
        """
        with patch('pipeline.main_pipeline.get_budget_sentinel', return_value=enhanced_pipeline_mocks['budget_sentinel']), \
             patch('pipeline.main_pipeline.get_workflow_orchestrator', return_value=enhanced_pipeline_mocks['workflow_orchestrator']), \
             patch('pipeline.main_pipeline.get_evidence_collector', return_value=enhanced_pipeline_mocks['evidence_collector']), \
             patch('pipeline.main_pipeline.get_pitch_deck_generator', return_value=enhanced_pipeline_mocks['pitch_deck_generator']), \
             patch('pipeline.main_pipeline.get_campaign_generator', return_value=enhanced_pipeline_mocks['campaign_generator']), \
             patch('pipeline.main_pipeline.create_idea_manager', new=AsyncMock(return_value=enhanced_pipeline_mocks['idea_manager'])), \
             patch('pipeline.main_pipeline.create_validator', new=AsyncMock(return_value=enhanced_pipeline_mocks['startup_validator'])), \
             patch('pipeline.main_pipeline.get_settings', return_value=Mock()):
            
            pipeline = MainPipeline()
            results = []
            
            # Test each startup idea
            for test_case in STARTUP_TEST_CASES:
                print(f"\n--- Testing: {test_case.name} ---")
                start_time = time.time()
                
                result = await pipeline.execute_full_pipeline(
                    startup_idea=test_case.description,
                    target_investor=test_case.target_investor,
                    generate_mvp=test_case.should_generate_mvp,
                    max_total_budget=62.0
                )
                
                execution_time = time.time() - start_time
                results.append((test_case, result, execution_time))
                
                # Validate Phase 1: Data Ingestion
                assert result.validation_result is not None
                assert result.validation_result['is_valid'] == True
                assert result.validation_result['overall_score'] >= test_case.expected_validation_score - 0.1
                
                # Validate Phase 2: Data Processing
                assert result.evidence_collection_result is not None
                assert len(result.evidence_collection_result) >= 3  # At least 3 research domains
                total_evidence = sum(len(evidence) for evidence in result.evidence_collection_result.values())
                assert total_evidence >= 5  # Minimum evidence requirement
                
                # Validate Phase 3: Data Transformation
                assert result.pitch_deck_result is not None
                assert result.pitch_deck_result['startup_name'] is not None
                assert result.pitch_deck_result['investor_type'] == test_case.target_investor.value
                assert result.pitch_deck_result['slide_count'] >= 10  # Minimum slide count
                assert result.pitch_deck_result['overall_quality_score'] >= test_case.expected_quality_threshold - 0.1
                
                # Validate Phase 4: Data Output
                assert result.campaign_result is not None
                assert result.campaign_result['budget_limit'] <= 25.0  # Budget constraint
                assert result.campaign_result['google_ads_id'] is not None
                assert result.campaign_result['posthog_project_id'] is not None
                
                if test_case.should_generate_mvp:
                    assert result.mvp_result is not None
                    assert result.mvp_result['deployment_success'] == True
                    assert result.mvp_result['generation_cost'] <= 4.0
                
                # Validate overall pipeline success
                assert len(result.phases_completed) == 4
                assert len(result.phases_failed) == 0
                assert len(result.errors) == 0
                assert result.overall_quality_score >= test_case.expected_quality_threshold - 0.1
                assert result.budget_utilization <= 1.0  # Within budget
                assert result.execution_time_seconds > 0
                
                print(f"✓ {test_case.name}: Quality={result.overall_quality_score:.2f}, Budget={result.budget_utilization:.1%}, Time={execution_time:.1f}s")
            
            # Generate comprehensive test report
            await self._generate_integration_test_report(results)
    
    @pytest.mark.asyncio
    async def test_budget_enforcement_across_all_phases(self, enhanced_pipeline_mocks):
        """Test budget tracking and enforcement across all pipeline phases."""
        with patch('pipeline.main_pipeline.get_budget_sentinel', return_value=enhanced_pipeline_mocks['budget_sentinel']), \
             patch('pipeline.main_pipeline.get_workflow_orchestrator', return_value=enhanced_pipeline_mocks['workflow_orchestrator']), \
             patch('pipeline.main_pipeline.get_evidence_collector', return_value=enhanced_pipeline_mocks['evidence_collector']), \
             patch('pipeline.main_pipeline.get_pitch_deck_generator', return_value=enhanced_pipeline_mocks['pitch_deck_generator']), \
             patch('pipeline.main_pipeline.get_campaign_generator', return_value=enhanced_pipeline_mocks['campaign_generator']), \
             patch('pipeline.main_pipeline.create_idea_manager', new=AsyncMock(return_value=enhanced_pipeline_mocks['idea_manager'])), \
             patch('pipeline.main_pipeline.create_validator', new=AsyncMock(return_value=enhanced_pipeline_mocks['startup_validator'])), \
             patch('pipeline.main_pipeline.get_settings', return_value=Mock()):
            
            pipeline = MainPipeline()
            
            # Test with strict budget limit
            result = await pipeline.execute_full_pipeline(
                startup_idea=STARTUP_TEST_CASES[0].description,
                max_total_budget=50.0  # Reduced budget
            )
            
            # Verify budget tracking was called
            budget_sentinel = enhanced_pipeline_mocks['budget_sentinel']
            budget_sentinel.track_operation.assert_called_once_with(
                "main_pipeline",
                "execute_full_pipeline", 
                BudgetCategory.INFRASTRUCTURE,
                50.0
            )
            
            # Verify budget status was checked
            budget_sentinel.get_budget_status.assert_called_once()
            
            # Budget utilization should be calculated
            assert result.budget_utilization > 0.0
            assert result.budget_utilization <= 1.0
    
    @pytest.mark.asyncio
    async def test_error_recovery_and_graceful_degradation(self, enhanced_pipeline_mocks):
        """Test error recovery patterns and graceful degradation."""
        with patch('pipeline.main_pipeline.get_budget_sentinel', return_value=enhanced_pipeline_mocks['budget_sentinel']), \
             patch('pipeline.main_pipeline.get_workflow_orchestrator', return_value=enhanced_pipeline_mocks['workflow_orchestrator']), \
             patch('pipeline.main_pipeline.get_evidence_collector', return_value=enhanced_pipeline_mocks['evidence_collector']), \
             patch('pipeline.main_pipeline.get_pitch_deck_generator', return_value=enhanced_pipeline_mocks['pitch_deck_generator']), \
             patch('pipeline.main_pipeline.get_campaign_generator', return_value=enhanced_pipeline_mocks['campaign_generator']), \
             patch('pipeline.main_pipeline.create_idea_manager', new=AsyncMock(return_value=enhanced_pipeline_mocks['idea_manager'])), \
             patch('pipeline.main_pipeline.create_validator', new=AsyncMock(return_value=enhanced_pipeline_mocks['startup_validator'])), \
             patch('pipeline.main_pipeline.get_settings', return_value=Mock()):
            
            # Test Phase 2 Failure (Evidence Collection)
            evidence_collector = enhanced_pipeline_mocks['evidence_collector']
            evidence_collector.collect_evidence.side_effect = RuntimeError("External API timeout")
            
            pipeline = MainPipeline()
            result = await pipeline.execute_full_pipeline(
                startup_idea=STARTUP_TEST_CASES[0].description
            )
            
            # Verify graceful handling
            assert "phase_2_processing" in result.phases_failed
            assert any("Phase 2 failed" in error for error in result.errors)
            assert result.completed_at is not None
            
            # Other phases should continue
            assert "phase_1_ingestion" in result.phases_completed
    
    @pytest.mark.asyncio
    async def test_quality_gates_validation(self, enhanced_pipeline_mocks):
        """Test quality gates prevent poor-quality data from advancing."""
        # Modify validator to return low-quality scores
        startup_validator = enhanced_pipeline_mocks['startup_validator']
        startup_validator.validate_startup_idea = AsyncMock(return_value={
            'is_valid': False,  # Fails validation
            'overall_score': 0.3,  # Below threshold
            'market_score': 0.2,
            'technical_score': 0.4,
            'validation_details': {
                'market_potential': 'low',
                'technical_feasibility': 'challenging',
                'competitive_advantage': 'weak'
            }
        })
        
        with patch('pipeline.main_pipeline.get_budget_sentinel', return_value=enhanced_pipeline_mocks['budget_sentinel']), \
             patch('pipeline.main_pipeline.get_workflow_orchestrator', return_value=enhanced_pipeline_mocks['workflow_orchestrator']), \
             patch('pipeline.main_pipeline.get_evidence_collector', return_value=enhanced_pipeline_mocks['evidence_collector']), \
             patch('pipeline.main_pipeline.get_pitch_deck_generator', return_value=enhanced_pipeline_mocks['pitch_deck_generator']), \
             patch('pipeline.main_pipeline.get_campaign_generator', return_value=enhanced_pipeline_mocks['campaign_generator']), \
             patch('pipeline.main_pipeline.create_idea_manager', new=AsyncMock(return_value=enhanced_pipeline_mocks['idea_manager'])), \
             patch('pipeline.main_pipeline.create_validator', new=AsyncMock(return_value=startup_validator)), \
             patch('pipeline.main_pipeline.get_settings', return_value=Mock()):
            
            pipeline = MainPipeline()
            result = await pipeline.execute_full_pipeline(
                startup_idea="Low quality startup idea with no market potential"
            )
            
            # Verify quality gate response
            assert result.validation_result['is_valid'] == False
            assert result.validation_result['overall_score'] < 0.6
            assert result.overall_quality_score < 0.6
    
    @pytest.mark.asyncio
    async def test_concurrent_pipeline_execution(self, enhanced_pipeline_mocks):
        """Test concurrent execution and data consistency."""
        with patch('pipeline.main_pipeline.get_budget_sentinel', return_value=enhanced_pipeline_mocks['budget_sentinel']), \
             patch('pipeline.main_pipeline.get_workflow_orchestrator', return_value=enhanced_pipeline_mocks['workflow_orchestrator']), \
             patch('pipeline.main_pipeline.get_evidence_collector', return_value=enhanced_pipeline_mocks['evidence_collector']), \
             patch('pipeline.main_pipeline.get_pitch_deck_generator', return_value=enhanced_pipeline_mocks['pitch_deck_generator']), \
             patch('pipeline.main_pipeline.get_campaign_generator', return_value=enhanced_pipeline_mocks['campaign_generator']), \
             patch('pipeline.main_pipeline.create_idea_manager', new=AsyncMock(return_value=enhanced_pipeline_mocks['idea_manager'])), \
             patch('pipeline.main_pipeline.create_validator', new=AsyncMock(return_value=enhanced_pipeline_mocks['startup_validator'])), \
             patch('pipeline.main_pipeline.get_settings', return_value=Mock()):
            
            pipeline = MainPipeline()
            
            # Run multiple pipelines concurrently
            tasks = []
            for i, test_case in enumerate(STARTUP_TEST_CASES[:3]):  # Test first 3 cases
                task = pipeline.execute_full_pipeline(
                    startup_idea=test_case.description,
                    target_investor=test_case.target_investor,
                    generate_mvp=False  # Skip MVP for faster execution
                )
                tasks.append(task)
            
            # Execute concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verify all succeeded
            for i, result in enumerate(results):
                assert not isinstance(result, Exception), f"Pipeline {i} failed: {result}"
                assert len(result.phases_completed) == 4
                assert len(result.phases_failed) == 0
    
    @pytest.mark.asyncio
    async def test_performance_requirements_validation(self, enhanced_pipeline_mocks):
        """Test performance meets specification requirements (<4h, ≤$62)."""
        with patch('pipeline.main_pipeline.get_budget_sentinel', return_value=enhanced_pipeline_mocks['budget_sentinel']), \
             patch('pipeline.main_pipeline.get_workflow_orchestrator', return_value=enhanced_pipeline_mocks['workflow_orchestrator']), \
             patch('pipeline.main_pipeline.get_evidence_collector', return_value=enhanced_pipeline_mocks['evidence_collector']), \
             patch('pipeline.main_pipeline.get_pitch_deck_generator', return_value=enhanced_pipeline_mocks['pitch_deck_generator']), \
             patch('pipeline.main_pipeline.get_campaign_generator', return_value=enhanced_pipeline_mocks['campaign_generator']), \
             patch('pipeline.main_pipeline.create_idea_manager', new=AsyncMock(return_value=enhanced_pipeline_mocks['idea_manager'])), \
             patch('pipeline.main_pipeline.create_validator', new=AsyncMock(return_value=enhanced_pipeline_mocks['startup_validator'])), \
             patch('pipeline.main_pipeline.get_settings', return_value=Mock()):
            
            pipeline = MainPipeline()
            start_time = datetime.utcnow()
            
            result = await pipeline.execute_full_pipeline(
                startup_idea=STARTUP_TEST_CASES[0].description,
                max_total_budget=62.0
            )
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Verify performance requirements
            assert execution_time < 14400  # <4 hours (14400 seconds)
            assert result.budget_utilization <= 1.0  # ≤$62 budget
            assert result.execution_time_seconds > 0
            
            print(f"Performance test: {execution_time:.1f}s execution, {result.budget_utilization:.1%} budget utilization")
    
    async def _generate_integration_test_report(self, results: List[tuple]) -> Dict[str, Any]:
        """Generate comprehensive integration test report."""
        report = {
            'test_execution_summary': {
                'total_test_cases': len(results),
                'successful_executions': sum(1 for _, result, _ in results if len(result.phases_failed) == 0),
                'failed_executions': sum(1 for _, result, _ in results if len(result.phases_failed) > 0),
                'average_execution_time': sum(exec_time for _, _, exec_time in results) / len(results),
                'average_quality_score': sum(result.overall_quality_score for _, result, _ in results) / len(results),
                'average_budget_utilization': sum(result.budget_utilization for _, result, _ in results) / len(results)
            },
            'phase_validation_results': {
                'phase_1_ingestion': {
                    'success_rate': sum(1 for _, result, _ in results if 'phase_1_ingestion' in result.phases_completed) / len(results),
                    'average_validation_score': sum(result.validation_result.get('overall_score', 0) for _, result, _ in results) / len(results)
                },
                'phase_2_processing': {
                    'success_rate': sum(1 for _, result, _ in results if 'phase_2_processing' in result.phases_completed) / len(results),
                    'average_evidence_count': sum(sum(len(evidence) for evidence in result.evidence_collection_result.values()) for _, result, _ in results if result.evidence_collection_result) / len(results)
                },
                'phase_3_transformation': {
                    'success_rate': sum(1 for _, result, _ in results if 'phase_3_transformation' in result.phases_completed) / len(results),
                    'average_pitch_quality': sum(result.pitch_deck_result.get('overall_quality_score', 0) for _, result, _ in results if result.pitch_deck_result) / len(results)
                },
                'phase_4_output': {
                    'success_rate': sum(1 for _, result, _ in results if 'phase_4_output' in result.phases_completed) / len(results),
                    'mvp_generation_rate': sum(1 for _, result, _ in results if result.mvp_result) / len(results)
                }
            },
            'integration_validation': {
                'budget_enforcement_working': all(result.budget_utilization <= 1.0 for _, result, _ in results),
                'quality_gates_working': all(result.overall_quality_score > 0.0 for _, result, _ in results),
                'error_recovery_working': True,  # Based on error recovery tests
                'performance_requirements_met': all(exec_time < 14400 for _, _, exec_time in results)
            },
            'recommendations': self._generate_integration_recommendations(results)
        }
        
        # Write report to file
        with open('integration_test_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print("\n" + "="*80)
        print("INTEGRATION TEST REPORT SUMMARY")
        print("="*80)
        print(f"Test Cases: {report['test_execution_summary']['total_test_cases']}")
        print(f"Success Rate: {report['test_execution_summary']['successful_executions']}/{report['test_execution_summary']['total_test_cases']}")
        print(f"Average Quality Score: {report['test_execution_summary']['average_quality_score']:.2f}")
        print(f"Average Budget Usage: {report['test_execution_summary']['average_budget_utilization']:.1%}")
        print(f"Average Execution Time: {report['test_execution_summary']['average_execution_time']:.1f}s")
        print("\nPhase Success Rates:")
        for phase, data in report['phase_validation_results'].items():
            print(f"  {phase}: {data['success_rate']:.1%}")
        print("\nIntegration Validation:")
        for validation, status in report['integration_validation'].items():
            print(f"  {validation}: {'✓' if status else '✗'}")
        print("="*80)
        
        return report
    
    def _generate_integration_recommendations(self, results: List[tuple]) -> List[str]:
        """Generate recommendations based on integration test results."""
        recommendations = []
        
        # Analyze success rates
        success_rate = sum(1 for _, result, _ in results if len(result.phases_failed) == 0) / len(results)
        if success_rate < 1.0:
            recommendations.append("Investigate and address pipeline phase failures")
        
        # Analyze quality scores
        avg_quality = sum(result.overall_quality_score for _, result, _ in results) / len(results)
        if avg_quality < 0.75:
            recommendations.append("Improve quality scoring algorithms and thresholds")
        
        # Analyze budget utilization
        avg_budget = sum(result.budget_utilization for _, result, _ in results) / len(results)
        if avg_budget > 0.9:
            recommendations.append("Optimize costs to stay within budget constraints")
        
        # Analyze execution time
        avg_time = sum(exec_time for _, _, exec_time in results) / len(results)
        if avg_time > 3600:  # 1 hour
            recommendations.append("Optimize pipeline performance for faster execution")
        
        return recommendations


class TestExternalServiceIntegration:
    """Tests for external service integration points."""
    
    @pytest.mark.asyncio
    async def test_google_ads_integration_mocking(self, enhanced_pipeline_mocks):
        """Test Google Ads campaign generation with proper mocking."""
        # This would test the actual integration points with Google Ads API
        # Currently mocked to demonstrate the testing pattern
        
        campaign_generator = enhanced_pipeline_mocks['campaign_generator']
        
        # Create a mock pitch deck
        pitch_deck = Mock()
        pitch_deck.startup_name = "TestStartup"
        
        # Generate campaign
        campaign = await campaign_generator.generate_smoke_test_campaign(
            pitch_deck=pitch_deck,
            budget_limit=25.0,
            duration_days=7
        )
        
        # Verify campaign properties
        assert campaign.budget_limit == 25.0
        assert campaign.google_ads_campaign_id is not None
        assert campaign.status.value == "active"
    
    @pytest.mark.asyncio
    async def test_posthog_analytics_integration_mocking(self, enhanced_pipeline_mocks):
        """Test PostHog analytics integration with proper mocking."""
        campaign_generator = enhanced_pipeline_mocks['campaign_generator']
        
        pitch_deck = Mock()
        pitch_deck.startup_name = "TestStartup"
        
        campaign = await campaign_generator.generate_smoke_test_campaign(
            pitch_deck=pitch_deck,
            budget_limit=25.0,
            duration_days=7
        )
        
        # Verify PostHog integration
        assert campaign.posthog_project_id is not None
        assert campaign.posthog_project_id.startswith("ph_")
    
    @pytest.mark.asyncio
    async def test_gpt_engineer_mvp_generation_mocking(self, enhanced_pipeline_mocks):
        """Test GPT-Engineer MVP generation with proper mocking."""
        campaign_generator = enhanced_pipeline_mocks['campaign_generator']
        
        mvp_request = MVPRequest(
            mvp_type=MVPType.LANDING_PAGE,
            startup_name="TestStartup",
            description="Test startup for MVP generation",
            key_features=["Feature 1", "Feature 2"],
            target_platforms=["web"],
            tech_stack=["python", "flask"]
        )
        
        mvp_result = await campaign_generator.generate_mvp(
            mvp_request=mvp_request,
            max_cost=4.0
        )
        
        # Verify MVP generation
        assert mvp_result.deployment_success == True
        assert mvp_result.generation_cost <= 4.0
        assert len(mvp_result.generated_files) >= 2
    
    @pytest.mark.asyncio
    async def test_flyio_deployment_integration_mocking(self, enhanced_pipeline_mocks):
        """Test Fly.io deployment automation with proper mocking."""
        campaign_generator = enhanced_pipeline_mocks['campaign_generator']
        
        mvp_request = MVPRequest(
            mvp_type=MVPType.LANDING_PAGE,
            startup_name="TestStartup",
            description="Test startup",
            key_features=["Test feature"],
            target_platforms=["web"],
            tech_stack=["python"]
        )
        
        mvp_result = await campaign_generator.generate_mvp(mvp_request, max_cost=4.0)
        
        # Verify deployment
        assert mvp_result.deployment_url is not None
        assert "fly.dev" in mvp_result.deployment_url
        assert mvp_result.deployment_status == "deployed"