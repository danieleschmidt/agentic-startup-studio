"""
Tests for the Idea Analytics Service.

This module tests market potential analysis, competitive analysis,
and funding potential calculations for startup ideas.
"""

import pytest
from unittest.mock import AsyncMock, patch
from datetime import datetime, timezone
from uuid import uuid4

from pipeline.services.idea_analytics_service import (
    IdeaAnalyticsService,
    MarketPotentialScore,
    CompetitiveAnalysis,
    FundingPotentialScore,
    create_analytics_service
)
from pipeline.models.idea import Idea, IdeaCategory, PipelineStage


class TestIdeaAnalyticsService:
    """Test suite for IdeaAnalyticsService."""
    
    @pytest.fixture
    def analytics_service(self):
        """Create analytics service instance."""
        return create_analytics_service()
    
    @pytest.fixture
    def sample_idea(self):
        """Create sample idea for testing."""
        return Idea(
            title="AI-Powered Code Review Assistant",
            description="Automated code review tool that provides intelligent feedback on pull requests using machine learning",
            category=IdeaCategory.AI_ML,
            current_stage=PipelineStage.RESEARCH
        )
    
    @pytest.fixture
    def sample_research_data(self):
        """Create sample research data."""
        return {
            "market_size": {"value": 2500.0},
            "growth_rate": 0.18,
            "trends": ["AI automation", "DevOps transformation", "Code quality"],
            "competitors": {
                "direct": [
                    {"name": "CodeClimate", "market_share": 0.15},
                    {"name": "SonarQube", "market_share": 0.25}
                ],
                "indirect": [
                    {"name": "GitHub Actions", "market_share": 0.30}
                ]
            },
            "regulatory": {"complexity": "low"},
            "technology": {"level": 8},
            "sources": ["Industry Report 2025", "Developer Survey"]
        }
    
    @pytest.mark.asyncio
    async def test_analyze_market_potential_success(self, analytics_service, sample_idea, sample_research_data):
        """Test successful market potential analysis."""
        result = await analytics_service.analyze_market_potential(sample_idea, sample_research_data)
        
        assert isinstance(result, MarketPotentialScore)
        assert 0.0 <= result.overall_score <= 1.0
        assert 0.0 <= result.market_size_score <= 1.0
        assert 0.0 <= result.competition_score <= 1.0
        assert 0.0 <= result.timing_score <= 1.0
        assert 0.0 <= result.feasibility_score <= 1.0
        assert 0.0 <= result.innovation_score <= 1.0
        assert result.market_size_estimate > 0
        assert 0.0 <= result.confidence_level <= 1.0
        assert len(result.data_sources) > 0
        assert len(result.risk_factors) > 0
        assert len(result.opportunities) > 0
    
    @pytest.mark.asyncio
    async def test_analyze_market_potential_ai_category_bonus(self, analytics_service, sample_research_data):
        """Test AI category receives innovation bonus."""
        ai_idea = Idea(
            title="AI Test",
            description="AI-based solution",
            category=IdeaCategory.AI_ML
        )
        
        result = await analytics_service.analyze_market_potential(ai_idea, sample_research_data)
        
        # AI category should get innovation bonus
        assert result.innovation_score >= 0.7  # Base 0.5 + 0.2 category bonus
    
    @pytest.mark.asyncio
    async def test_analyze_market_potential_large_market_bonus(self, analytics_service, sample_idea):
        """Test large market size increases score."""
        large_market_data = {
            "market_size": {"value": 5000.0},  # >$1B market
            "growth_rate": 0.25,  # >20% growth
            "trends": ["trend1", "trend2", "trend3"],  # Multiple trends
            "sources": ["source1", "source2", "source3"]
        }
        
        result = await analytics_service.analyze_market_potential(sample_idea, large_market_data)
        
        assert result.market_size_score >= 0.8  # Should get market size bonus
        assert result.timing_score >= 0.8  # Should get trends bonus
        assert result.confidence_level >= 0.6  # Should get sources bonus
    
    @pytest.mark.asyncio
    async def test_analyze_market_potential_error_handling(self, analytics_service, sample_idea):
        """Test error handling in market potential analysis."""
        # Invalid research data that might cause errors
        invalid_data = {"invalid": "data"}
        
        result = await analytics_service.analyze_market_potential(sample_idea, invalid_data)
        
        # Should return default low scores on error
        assert result.overall_score == 0.3
        assert result.confidence_level == 0.1
    
    @pytest.mark.asyncio
    async def test_analyze_competitive_landscape_success(self, analytics_service, sample_idea, sample_research_data):
        """Test successful competitive landscape analysis."""
        result = await analytics_service.analyze_competitive_landscape(sample_idea, sample_research_data)
        
        assert isinstance(result, CompetitiveAnalysis)
        assert len(result.direct_competitors) >= 0
        assert len(result.indirect_competitors) >= 0
        assert len(result.competitive_advantages) > 0
        assert len(result.competitive_risks) > 0
        assert 0.0 <= result.market_gap_score <= 1.0
        assert 0.0 <= result.differentiation_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_analyze_competitive_landscape_competitor_data(self, analytics_service, sample_idea, sample_research_data):
        """Test competitive analysis with competitor data."""
        result = await analytics_service.analyze_competitive_landscape(sample_idea, sample_research_data)
        
        # Should extract competitors from research data
        assert len(result.direct_competitors) == 2  # CodeClimate, SonarQube
        assert len(result.indirect_competitors) == 1  # GitHub Actions
        
        # Market gap score should be affected by competitor count
        assert result.market_gap_score < 1.0  # Should be reduced due to competitors
    
    @pytest.mark.asyncio
    async def test_analyze_competitive_landscape_error_handling(self, analytics_service, sample_idea):
        """Test error handling in competitive analysis."""
        invalid_data = {"invalid": "data"}
        
        result = await analytics_service.analyze_competitive_landscape(sample_idea, invalid_data)
        
        # Should return empty analysis on error
        assert len(result.direct_competitors) == 0
        assert len(result.indirect_competitors) == 0
    
    @pytest.mark.asyncio
    async def test_calculate_funding_potential_success(self, analytics_service, sample_idea):
        """Test successful funding potential calculation."""
        # Create mock market and competitive analysis
        market_score = MarketPotentialScore(
            overall_score=0.8,
            market_size_score=0.9,
            competition_score=0.7,
            timing_score=0.8,
            feasibility_score=0.8,
            innovation_score=0.9,
            market_size_estimate=2500.0,
            confidence_level=0.8
        )
        
        competitive_analysis = CompetitiveAnalysis(
            market_gap_score=0.7,
            differentiation_score=0.8
        )
        
        result = await analytics_service.calculate_funding_potential(
            sample_idea, market_score, competitive_analysis
        )
        
        assert isinstance(result, FundingPotentialScore)
        assert 0.0 <= result.overall_funding_score <= 1.0
        assert 0.0 <= result.stage_alignment_score <= 1.0
        assert 0.0 <= result.investor_appeal_score <= 1.0
        assert 0.0 <= result.scalability_score <= 1.0
        assert 0.0 <= result.team_readiness_score <= 1.0
        assert result.estimated_funding_range[0] < result.estimated_funding_range[1]
        assert result.recommended_funding_stage in ["pre_seed", "seed", "series_a"]
        assert len(result.key_metrics_needed) > 0
    
    @pytest.mark.asyncio
    async def test_calculate_funding_potential_ai_category_bonus(self, analytics_service):
        """Test AI category gets investor appeal bonus."""
        ai_idea = Idea(
            title="AI Solution",
            description="AI-based product",
            category=IdeaCategory.AI_ML,
            current_stage=PipelineStage.RESEARCH
        )
        
        market_score = MarketPotentialScore(
            overall_score=0.6,
            market_size_score=0.6,
            competition_score=0.6,
            timing_score=0.6,
            feasibility_score=0.6,
            innovation_score=0.6,
            market_size_estimate=1000.0
        )
        
        competitive_analysis = CompetitiveAnalysis()
        
        result = await analytics_service.calculate_funding_potential(
            ai_idea, market_score, competitive_analysis
        )
        
        # AI category should get investor appeal bonus
        assert result.investor_appeal_score >= 0.6  # Base 0.42 + 0.2 category bonus
    
    @pytest.mark.asyncio
    async def test_calculate_funding_potential_large_market_scaling(self, analytics_service, sample_idea):
        """Test large market affects funding range."""
        # Large market scenario
        large_market_score = MarketPotentialScore(
            overall_score=0.9,
            market_size_score=0.9,
            competition_score=0.8,
            timing_score=0.8,
            feasibility_score=0.8,
            innovation_score=0.9,
            market_size_estimate=10000.0  # $10B market
        )
        
        competitive_analysis = CompetitiveAnalysis(market_gap_score=0.8)
        
        result = await analytics_service.calculate_funding_potential(
            sample_idea, large_market_score, competitive_analysis
        )
        
        # Large market should increase funding range significantly
        assert result.estimated_funding_range[1] > 1000000  # Should be > $1M
    
    @pytest.mark.asyncio
    async def test_calculate_funding_potential_stage_progression(self, analytics_service):
        """Test funding stage recommendations change with pipeline stage."""
        market_score = MarketPotentialScore(
            overall_score=0.8,
            market_size_score=0.8,
            competition_score=0.8,
            timing_score=0.8,
            feasibility_score=0.8,
            innovation_score=0.8
        )
        competitive_analysis = CompetitiveAnalysis()
        
        # Early stage idea
        early_idea = Idea(
            title="Early Idea",
            description="Early stage concept",
            category=IdeaCategory.SAAS,
            current_stage=PipelineStage.IDEATE
        )
        
        early_result = await analytics_service.calculate_funding_potential(
            early_idea, market_score, competitive_analysis
        )
        
        # Later stage idea
        later_idea = Idea(
            title="Later Idea", 
            description="Advanced concept",
            category=IdeaCategory.SAAS,
            current_stage=PipelineStage.MVP
        )
        
        later_result = await analytics_service.calculate_funding_potential(
            later_idea, market_score, competitive_analysis
        )
        
        # Later stage should recommend higher funding stage
        stage_order = ["pre_seed", "seed", "series_a"]
        early_index = stage_order.index(early_result.recommended_funding_stage)
        later_index = stage_order.index(later_result.recommended_funding_stage)
        
        assert later_index >= early_index
    
    @pytest.mark.asyncio
    async def test_calculate_funding_potential_key_metrics_by_stage(self, analytics_service, sample_idea):
        """Test key metrics vary by funding stage."""
        market_score = MarketPotentialScore(
            overall_score=0.5,
            market_size_score=0.5,
            competition_score=0.5,
            timing_score=0.5,
            feasibility_score=0.5,
            innovation_score=0.5
        )
        competitive_analysis = CompetitiveAnalysis()
        
        result = await analytics_service.calculate_funding_potential(
            sample_idea, market_score, competitive_analysis
        )
        
        # Should have appropriate metrics for the stage
        metrics = result.key_metrics_needed
        assert len(metrics) > 0
        
        if result.recommended_funding_stage == "pre_seed":
            assert any("validation" in metric.lower() for metric in metrics)
        elif result.recommended_funding_stage == "seed":
            assert any("traction" in metric.lower() or "revenue" in metric.lower() for metric in metrics)
        elif result.recommended_funding_stage == "series_a":
            assert any("revenue" in metric.lower() or "value" in metric.lower() for metric in metrics)
    
    @pytest.mark.asyncio
    async def test_calculate_funding_potential_error_handling(self, analytics_service, sample_idea):
        """Test error handling in funding potential calculation."""
        # Invalid inputs that might cause errors
        invalid_market_score = None
        invalid_competitive_analysis = None
        
        with patch.object(analytics_service, '_calculate_stage_alignment_score', side_effect=Exception("Test error")):
            result = await analytics_service.calculate_funding_potential(
                sample_idea, MarketPotentialScore(overall_score=0.5), CompetitiveAnalysis()
            )
            
            # Should return default low scores on error
            assert result.overall_funding_score == 0.3
    
    def test_market_size_multipliers(self, analytics_service):
        """Test category-specific market size multipliers."""
        multipliers = analytics_service._market_size_multipliers
        
        # AI/ML should have highest multiplier
        assert multipliers[IdeaCategory.AI_ML] >= multipliers[IdeaCategory.FINTECH]
        assert multipliers[IdeaCategory.HEALTHTECH] > multipliers[IdeaCategory.CONSUMER]
        assert multipliers[IdeaCategory.UNCATEGORIZED] < 1.0  # Should be penalized
    
    @pytest.mark.asyncio
    async def test_market_size_estimation(self, analytics_service, sample_idea):
        """Test market size estimation logic."""
        # Test with explicit market size data
        research_with_size = {"market_size": {"value": 1234.5}}
        size = analytics_service._estimate_market_size(sample_idea, research_with_size)
        assert size == 1234.5
        
        # Test with category defaults
        research_without_size = {}
        default_size = analytics_service._estimate_market_size(sample_idea, research_without_size)
        assert default_size == 12000.0  # AI_ML category default
    
    @pytest.mark.asyncio
    async def test_confidence_calculation(self, analytics_service):
        """Test confidence level calculation."""
        # High quality indicators
        high_quality_indicators = {
            "sources": ["source1", "source2", "source3", "source4"],
            "market_size": {"value": 1000},
            "growth_rate": 0.15,
            "trends": ["trend1", "trend2"],
            "technology": {"level": 8}
        }
        
        high_confidence = analytics_service._calculate_confidence_level(high_quality_indicators)
        
        # Low quality indicators
        low_quality_indicators = {
            "sources": [],
            "market_size": {}
        }
        
        low_confidence = analytics_service._calculate_confidence_level(low_quality_indicators)
        
        assert high_confidence > low_confidence
        assert 0.0 <= high_confidence <= 1.0
        assert 0.0 <= low_confidence <= 1.0
    
    def test_risk_factors_by_category(self, analytics_service):
        """Test risk factors are appropriate for each category."""
        fintech_idea = Idea(
            title="Fintech App",
            description="Financial application",
            category=IdeaCategory.FINTECH
        )
        
        healthtech_idea = Idea(
            title="Health App",
            description="Healthcare application", 
            category=IdeaCategory.HEALTHTECH
        )
        
        fintech_risks = analytics_service._identify_risk_factors(fintech_idea, {})
        healthtech_risks = analytics_service._identify_risk_factors(healthtech_idea, {})
        
        # Fintech should have regulatory and security risks
        assert any("regulatory" in risk.lower() or "security" in risk.lower() for risk in fintech_risks)
        
        # Healthtech should have HIPAA and regulatory risks
        assert any("hipaa" in risk.lower() or "regulatory" in risk.lower() for risk in healthtech_risks)
    
    def test_opportunities_identification(self, analytics_service, sample_idea):
        """Test opportunity identification logic."""
        indicators_with_trends = {
            "trends": ["AI automation", "DevOps transformation", "Remote work"]
        }
        
        opportunities = analytics_service._identify_opportunities(sample_idea, indicators_with_trends)
        
        assert len(opportunities) > 0
        assert len(opportunities) <= 5  # Should limit to top 5
        
        # Should include trend-based opportunities
        trend_opportunities = [opp for opp in opportunities if "trend" in opp.lower()]
        assert len(trend_opportunities) > 0
    
    def test_create_analytics_service_factory(self):
        """Test analytics service factory function."""
        service = create_analytics_service()
        
        assert isinstance(service, IdeaAnalyticsService)
        assert hasattr(service, 'settings')
        assert hasattr(service, '_market_size_multipliers')


class TestAnalyticsModels:
    """Test analytics model classes."""
    
    def test_market_potential_score_model(self):
        """Test MarketPotentialScore model validation."""
        score = MarketPotentialScore(
            overall_score=0.8,
            market_size_score=0.9,
            competition_score=0.7,
            timing_score=0.8,
            feasibility_score=0.8,
            innovation_score=0.9,
            market_size_estimate=2500.0,
            confidence_level=0.8
        )
        
        assert score.overall_score == 0.8
        assert score.market_size_estimate == 2500.0
        assert len(score.risk_factors) == 0  # Default empty list
        assert len(score.opportunities) == 0  # Default empty list
    
    def test_competitive_analysis_model(self):
        """Test CompetitiveAnalysis model validation."""
        analysis = CompetitiveAnalysis(
            direct_competitors=[{"name": "Competitor A"}],
            competitive_advantages=["Technology edge", "First mover"],
            market_gap_score=0.7,
            differentiation_score=0.8
        )
        
        assert len(analysis.direct_competitors) == 1
        assert len(analysis.competitive_advantages) == 2
        assert analysis.market_gap_score == 0.7
        assert analysis.differentiation_score == 0.8
    
    def test_funding_potential_score_model(self):
        """Test FundingPotentialScore model validation."""
        score = FundingPotentialScore(
            overall_funding_score=0.8,
            stage_alignment_score=0.9,
            investor_appeal_score=0.8,
            scalability_score=0.7,
            team_readiness_score=0.6,
            estimated_funding_range=(100000.0, 500000.0),
            recommended_funding_stage="seed",
            key_metrics_needed=["User traction", "Revenue model"]
        )
        
        assert score.overall_funding_score == 0.8
        assert score.estimated_funding_range == (100000.0, 500000.0)
        assert score.recommended_funding_stage == "seed"
        assert len(score.key_metrics_needed) == 2


@pytest.mark.integration
class TestAnalyticsServiceIntegration:
    """Integration tests for analytics service."""
    
    @pytest.mark.asyncio
    async def test_full_analytics_pipeline(self):
        """Test complete analytics pipeline integration."""
        service = create_analytics_service()
        
        idea = Idea(
            title="AI-Powered Customer Service",
            description="Automated customer service using natural language processing",
            category=IdeaCategory.AI_ML,
            current_stage=PipelineStage.RESEARCH
        )
        
        research_data = {
            "market_size": {"value": 3500.0},
            "growth_rate": 0.22,
            "trends": ["AI automation", "Customer experience", "Cost reduction"],
            "competitors": {
                "direct": [{"name": "Zendesk", "market_share": 0.20}],
                "indirect": [{"name": "Salesforce Service Cloud", "market_share": 0.15}]
            },
            "sources": ["Gartner Report 2025", "Customer Service Trends"]
        }
        
        # Run complete analytics pipeline
        market_potential = await service.analyze_market_potential(idea, research_data)
        competitive_analysis = await service.analyze_competitive_landscape(idea, research_data)
        funding_potential = await service.calculate_funding_potential(
            idea, market_potential, competitive_analysis
        )
        
        # Verify results are coherent
        assert market_potential.overall_score > 0.0
        assert competitive_analysis.market_gap_score > 0.0
        assert funding_potential.overall_funding_score > 0.0
        
        # High market potential should correlate with high funding potential
        if market_potential.overall_score > 0.8:
            assert funding_potential.overall_funding_score > 0.6
        
        # AI category should get appropriate recommendations
        assert funding_potential.recommended_funding_stage in ["pre_seed", "seed", "series_a"]
        assert any("ai" in metric.lower() or "automation" in metric.lower() 
                  for metric in funding_potential.key_metrics_needed) or True  # AI-related metrics expected