"""
Tests for the Enhanced Evidence Collector Service.

This module tests comprehensive evidence collection, source validation,
and evidence quality assessment for startup ideas.
"""

import pytest
from unittest.mock import AsyncMock, patch, Mock
from datetime import datetime, timezone
from uuid import uuid4

from pipeline.services.enhanced_evidence_collector import (
    EnhancedEvidenceCollector,
    EvidenceSource,
    MarketEvidence,
    TechnicalEvidence,
    BusinessEvidence,
    ComprehensiveEvidence,
    create_enhanced_evidence_collector
)
from pipeline.models.idea import Idea, IdeaCategory, PipelineStage


class TestEnhancedEvidenceCollector:
    """Test suite for EnhancedEvidenceCollector."""
    
    @pytest.fixture
    def evidence_collector(self):
        """Create evidence collector instance."""
        return create_enhanced_evidence_collector()
    
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
    def mock_search_results(self):
        """Create mock search results."""
        return [
            {
                "url": "https://arxiv.org/paper/ai-code-review",
                "title": "AI-Based Code Review Systems: A Survey",
                "author": "Dr. Smith",
                "publish_date": "2024-01-15T00:00:00Z",
                "citation_count": 25
            },
            {
                "url": "https://techcrunch.com/ai-code-review-market",
                "title": "AI Code Review Market Expected to Grow 25% by 2025",
                "author": "Tech Reporter",
                "publish_date": "2024-06-01T00:00:00Z"
            },
            {
                "url": "https://company.com/blog/code-review",
                "title": "Our Experience with Automated Code Review",
                "author": "Company Blog"
            }
        ]
    
    @pytest.mark.asyncio
    async def test_collect_comprehensive_evidence_success(self, evidence_collector, sample_idea):
        """Test successful comprehensive evidence collection."""
        with patch.object(evidence_collector, '_collect_market_evidence') as mock_market, \
             patch.object(evidence_collector, '_collect_technical_evidence') as mock_technical, \
             patch.object(evidence_collector, '_collect_business_evidence') as mock_business:
            
            # Mock evidence returns
            mock_market.return_value = MarketEvidence(confidence_score=0.8)
            mock_technical.return_value = TechnicalEvidence(confidence_score=0.7)
            mock_business.return_value = BusinessEvidence(confidence_score=0.6)
            
            result = await evidence_collector.collect_comprehensive_evidence(sample_idea)
            
            assert isinstance(result, ComprehensiveEvidence)
            assert result.idea_id == sample_idea.idea_id
            assert 0.0 <= result.overall_confidence <= 1.0
            assert 0.0 <= result.evidence_quality_score <= 1.0
            assert result.collection_timestamp is not None
            assert len(result.summary) > 0
            assert len(result.key_insights) > 0
            assert len(result.risk_factors) > 0
            assert len(result.opportunities) > 0
            
            # Verify all collection methods were called
            mock_market.assert_called_once_with(sample_idea, "standard")
            mock_technical.assert_called_once_with(sample_idea, "standard")
            mock_business.assert_called_once_with(sample_idea, "standard")
    
    @pytest.mark.asyncio
    async def test_collect_comprehensive_evidence_depth_levels(self, evidence_collector, sample_idea):
        """Test different depth levels affect collection."""
        with patch.object(evidence_collector, '_collect_market_evidence') as mock_market, \
             patch.object(evidence_collector, '_collect_technical_evidence') as mock_technical, \
             patch.object(evidence_collector, '_collect_business_evidence') as mock_business:
            
            mock_market.return_value = MarketEvidence()
            mock_technical.return_value = TechnicalEvidence()
            mock_business.return_value = BusinessEvidence()
            
            # Test basic depth
            await evidence_collector.collect_comprehensive_evidence(sample_idea, depth="basic")
            mock_market.assert_called_with(sample_idea, "basic")
            
            # Test comprehensive depth
            await evidence_collector.collect_comprehensive_evidence(sample_idea, depth="comprehensive")
            mock_market.assert_called_with(sample_idea, "comprehensive")
    
    @pytest.mark.asyncio
    async def test_collect_comprehensive_evidence_error_handling(self, evidence_collector, sample_idea):
        """Test error handling in comprehensive evidence collection."""
        with patch.object(evidence_collector, '_collect_market_evidence', side_effect=Exception("Market error")), \
             patch.object(evidence_collector, '_collect_technical_evidence', side_effect=Exception("Technical error")), \
             patch.object(evidence_collector, '_collect_business_evidence', side_effect=Exception("Business error")):
            
            result = await evidence_collector.collect_comprehensive_evidence(sample_idea)
            
            # Should return minimal evidence on error
            assert result.overall_confidence == 0.1
            assert result.evidence_quality_score == 0.1
            assert "Evidence collection failed" in result.summary
    
    @pytest.mark.asyncio
    async def test_collect_market_evidence(self, evidence_collector, sample_idea, mock_search_results):
        """Test market evidence collection."""
        with patch.object(evidence_collector, '_search_web_evidence', return_value=[
            EvidenceSource(
                url="https://example.com",
                title="Market Research",
                source_type="industry_report",
                credibility_score=0.8,
                relevance_score=0.9
            )
        ]):
            result = await evidence_collector._collect_market_evidence(sample_idea, "standard")
            
            assert isinstance(result, MarketEvidence)
            assert len(result.sources) > 0
            assert result.market_size_data is not None
            assert len(result.growth_projections) >= 0
            assert result.competitive_landscape is not None
            assert len(result.technology_trends) > 0
            assert 0.0 <= result.confidence_score <= 1.0
            assert result.last_updated is not None
    
    @pytest.mark.asyncio
    async def test_collect_technical_evidence(self, evidence_collector, sample_idea):
        """Test technical evidence collection."""
        with patch.object(evidence_collector, '_search_web_evidence', return_value=[
            EvidenceSource(
                url="https://technical.com",
                title="Technical Implementation Guide",
                source_type="academic",
                credibility_score=0.9,
                relevance_score=0.8
            )
        ]):
            result = await evidence_collector._collect_technical_evidence(sample_idea, "standard")
            
            assert isinstance(result, TechnicalEvidence)
            assert result.technology_readiness is not None
            assert result.implementation_complexity in ["low", "medium", "high"]
            assert len(result.required_resources) > 0
            assert len(result.technical_risks) > 0
            assert result.development_timeline is not None
            assert len(result.sources) > 0
            assert 0.0 <= result.confidence_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_collect_business_evidence(self, evidence_collector, sample_idea):
        """Test business evidence collection."""
        with patch.object(evidence_collector, '_search_web_evidence', return_value=[
            EvidenceSource(
                url="https://business.com",
                title="Business Model Analysis",
                source_type="industry_report",
                credibility_score=0.8,
                relevance_score=0.7
            )
        ]):
            result = await evidence_collector._collect_business_evidence(sample_idea, "standard")
            
            assert isinstance(result, BusinessEvidence)
            assert len(result.revenue_models) > 0
            assert result.cost_structure is not None
            assert result.funding_landscape is not None
            assert len(result.success_stories) >= 0
            assert len(result.sources) > 0
            assert 0.0 <= result.confidence_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_search_web_evidence_success(self, evidence_collector, mock_search_results):
        """Test successful web evidence search."""
        with patch('pipeline.services.enhanced_evidence_collector.search_for_evidence', 
                   return_value=mock_search_results):
            
            sources = await evidence_collector._search_web_evidence("AI code review", "technical")
            
            assert len(sources) == 3
            assert all(isinstance(source, EvidenceSource) for source in sources)
            
            # Check academic source classification
            academic_source = sources[0]
            assert academic_source.source_type == "academic"
            assert academic_source.credibility_score > 0.8  # Academic sources should have high credibility
            
            # Check news source classification
            news_source = sources[1]
            assert news_source.source_type == "news"
            
            # Check company source classification
            company_source = sources[2]
            assert company_source.source_type == "company"
    
    @pytest.mark.asyncio
    async def test_search_web_evidence_error_handling(self, evidence_collector):
        """Test error handling in web evidence search."""
        with patch('pipeline.services.enhanced_evidence_collector.search_for_evidence', 
                   side_effect=Exception("Search failed")):
            
            sources = await evidence_collector._search_web_evidence("test query", "market")
            
            # Should return empty list on error
            assert sources == []
    
    def test_classify_source_type(self, evidence_collector):
        """Test source type classification."""
        # Academic sources
        assert evidence_collector._classify_source_type("https://arxiv.org/paper") == "academic"
        assert evidence_collector._classify_source_type("https://scholar.google.com/paper") == "academic"
        assert evidence_collector._classify_source_type("https://university.edu/research") == "academic"
        
        # Industry reports
        assert evidence_collector._classify_source_type("https://mckinsey.com/report") == "industry_report"
        assert evidence_collector._classify_source_type("https://bcg.com/insights") == "industry_report"
        
        # Government sources
        assert evidence_collector._classify_source_type("https://fda.gov/guidance") == "government"
        assert evidence_collector._classify_source_type("https://europa.eu/report") == "government"
        
        # News sources
        assert evidence_collector._classify_source_type("https://techcrunch.com/article") == "news"
        assert evidence_collector._classify_source_type("https://bloomberg.com/news") == "news"
        
        # Patent sources
        assert evidence_collector._classify_source_type("https://patents.google.com/patent") == "patent"
        
        # Blog sources
        assert evidence_collector._classify_source_type("https://medium.com/post") == "blog"
        assert evidence_collector._classify_source_type("https://blog.wordpress.com") == "blog"
        
        # Company sources (default)
        assert evidence_collector._classify_source_type("https://company.com/about") == "company"
    
    def test_calculate_credibility_score(self, evidence_collector):
        """Test credibility score calculation."""
        # Academic source with citations
        academic_result = {
            "url": "https://arxiv.org/paper",
            "citation_count": 15,
            "publish_date": "2024-01-01T00:00:00Z"
        }
        score = evidence_collector._calculate_credibility_score(academic_result)
        assert score >= 0.9  # Academic + citations + recent
        
        # News source without citations
        news_result = {
            "url": "https://techcrunch.com/article",
            "citation_count": 0
        }
        score = evidence_collector._calculate_credibility_score(news_result)
        assert 0.5 <= score <= 0.7  # News source range
        
        # Blog source
        blog_result = {
            "url": "https://medium.com/post"
        }
        score = evidence_collector._calculate_credibility_score(blog_result)
        assert score <= 0.5  # Blog sources have lower credibility
    
    def test_calculate_relevance_score(self, evidence_collector):
        """Test relevance score calculation."""
        # High relevance - all query terms in title
        high_relevance = {
            "title": "AI Code Review Machine Learning Analysis"
        }
        score = evidence_collector._calculate_relevance_score(high_relevance, "AI code review")
        assert score >= 0.66  # 2/3 terms match
        
        # Low relevance - no matching terms
        low_relevance = {
            "title": "Unrelated Technology Article"
        }
        score = evidence_collector._calculate_relevance_score(low_relevance, "AI code review")
        assert score == 0.0  # No matches
        
        # Medium relevance - some matching terms
        medium_relevance = {
            "title": "AI Technology Review and Analysis"
        }
        score = evidence_collector._calculate_relevance_score(medium_relevance, "AI code review")
        assert 0.0 < score < 1.0  # Partial match
    
    def test_extract_publish_date(self, evidence_collector):
        """Test publish date extraction."""
        # Valid ISO date
        result_with_date = {"publish_date": "2024-01-01T00:00:00Z"}
        date = evidence_collector._extract_publish_date(result_with_date)
        assert date is not None
        assert date.year == 2024
        
        # Invalid date
        result_with_invalid_date = {"publish_date": "invalid-date"}
        date = evidence_collector._extract_publish_date(result_with_invalid_date)
        assert date is None
        
        # No date
        result_without_date = {}
        date = evidence_collector._extract_publish_date(result_without_date)
        assert date is None
    
    @pytest.mark.asyncio
    async def test_extract_market_size_data(self, evidence_collector, sample_idea):
        """Test market size data extraction."""
        sources = [
            EvidenceSource(
                url="https://example.com",
                title="Market Research",
                source_type="industry_report",
                credibility_score=0.8,
                relevance_score=0.9
            )
        ]
        
        result = await evidence_collector._extract_market_size_data(sources, sample_idea)
        
        assert "value" in result
        assert "currency" in result
        assert "unit" in result
        assert result["currency"] == "USD"
        assert result["unit"] == "millions"
        assert result["value"] > 0
        assert result["sources_analyzed"] == len(sources)
        
        # AI_ML category should have appropriate market size
        assert result["value"] == 12000.0  # AI_ML default
    
    @pytest.mark.asyncio
    async def test_extract_growth_projections(self, evidence_collector, sample_idea):
        """Test growth projections extraction."""
        sources = []
        
        result = await evidence_collector._extract_growth_projections(sources, sample_idea)
        
        assert len(result) > 0
        projection = result[0]
        assert "period" in projection
        assert "cagr" in projection
        assert "projected_value" in projection
        assert "confidence" in projection
        
        # AI_ML category should have high growth rate
        assert projection["cagr"] == 0.25  # AI_ML growth rate
    
    def test_extract_technology_trends(self, evidence_collector):
        """Test technology trends extraction by category."""
        # AI_ML trends
        ai_trends = evidence_collector._extract_technology_trends(IdeaCategory.AI_ML)
        assert "Large Language Models" in ai_trends
        assert "Computer Vision" in ai_trends
        
        # Fintech trends
        fintech_trends = evidence_collector._extract_technology_trends(IdeaCategory.FINTECH)
        assert "Open Banking" in fintech_trends
        assert "DeFi" in fintech_trends
        
        # Unknown category gets defaults
        unknown_trends = evidence_collector._extract_technology_trends(IdeaCategory.UNCATEGORIZED)
        assert "Digital Transformation" in unknown_trends
    
    def test_calculate_source_confidence(self, evidence_collector):
        """Test source confidence calculation."""
        # High quality sources
        high_quality_sources = [
            EvidenceSource(
                url="https://arxiv.org/paper",
                title="Research Paper",
                source_type="academic",
                credibility_score=0.9,
                relevance_score=0.8
            ),
            EvidenceSource(
                url="https://mckinsey.com/report",
                title="Industry Report",
                source_type="industry_report",
                credibility_score=0.8,
                relevance_score=0.9
            )
        ]
        
        high_confidence = evidence_collector._calculate_source_confidence(high_quality_sources)
        
        # Low quality sources
        low_quality_sources = [
            EvidenceSource(
                url="https://blog.com/post",
                title="Blog Post",
                source_type="blog",
                credibility_score=0.4,
                relevance_score=0.3
            )
        ]
        
        low_confidence = evidence_collector._calculate_source_confidence(low_quality_sources)
        
        # No sources
        no_confidence = evidence_collector._calculate_source_confidence([])
        
        assert high_confidence > low_confidence
        assert no_confidence == 0.1  # Minimum confidence
        assert 0.0 <= high_confidence <= 1.0
        assert 0.0 <= low_confidence <= 1.0
    
    def test_assess_technology_readiness(self, evidence_collector, sample_idea):
        """Test technology readiness assessment."""
        sources = []
        
        result = evidence_collector._assess_technology_readiness(sample_idea, sources)
        
        assert "trl_level" in result
        assert "readiness_description" in result
        assert "key_technologies" in result
        assert "technology_gaps" in result
        assert "source_count" in result
        
        # AI_ML should have high TRL
        assert result["trl_level"] == 7  # AI_ML TRL
        assert "Machine Learning" in result["key_technologies"]
        assert result["source_count"] == 0
    
    def test_identify_key_technologies(self, evidence_collector):
        """Test key technologies identification by category."""
        # AI_ML technologies
        ai_tech = evidence_collector._identify_key_technologies(IdeaCategory.AI_ML)
        assert "Machine Learning" in ai_tech
        assert "Deep Learning" in ai_tech
        
        # Fintech technologies
        fintech_tech = evidence_collector._identify_key_technologies(IdeaCategory.FINTECH)
        assert "Blockchain" in fintech_tech
        assert "API Integration" in fintech_tech
        
        # Unknown category gets defaults
        unknown_tech = evidence_collector._identify_key_technologies(IdeaCategory.UNCATEGORIZED)
        assert "Web Development" in unknown_tech
    
    def test_identify_technology_gaps(self, evidence_collector):
        """Test technology gaps identification."""
        # Low TRL has more gaps
        low_trl_gaps = evidence_collector._identify_technology_gaps(IdeaCategory.AI_ML, 5)
        assert len(low_trl_gaps) > 2
        assert "Operational environment testing needed" in low_trl_gaps
        
        # High TRL has fewer gaps
        high_trl_gaps = evidence_collector._identify_technology_gaps(IdeaCategory.AI_ML, 9)
        assert len(high_trl_gaps) == 0
    
    def test_assess_implementation_complexity(self, evidence_collector, sample_idea):
        """Test implementation complexity assessment."""
        sources = []
        
        complexity = evidence_collector._assess_implementation_complexity(sample_idea, sources)
        
        assert complexity in ["low", "medium", "high"]
        # AI_ML should be high complexity
        assert complexity == "high"
    
    def test_identify_required_resources(self, evidence_collector, sample_idea):
        """Test required resources identification."""
        resources = evidence_collector._identify_required_resources(sample_idea)
        
        assert len(resources) > 0
        # AI_ML should require ML Engineers
        assert "ML Engineers" in resources
        assert "Data Scientists" in resources
    
    def test_identify_technical_risks(self, evidence_collector, sample_idea):
        """Test technical risks identification."""
        sources = []
        
        risks = evidence_collector._identify_technical_risks(sample_idea, sources)
        
        assert len(risks) > 0
        # AI_ML should have model accuracy risks
        assert "Model accuracy" in risks
        assert "Data bias" in risks
    
    def test_estimate_development_timeline(self, evidence_collector, sample_idea):
        """Test development timeline estimation."""
        # High complexity timeline
        high_timeline = evidence_collector._estimate_development_timeline(sample_idea, "high")
        assert "mvp" in high_timeline
        assert "production" in high_timeline
        assert "scale" in high_timeline
        assert "6-12 months" in high_timeline["mvp"]  # High complexity MVP
        
        # Low complexity timeline
        low_timeline = evidence_collector._estimate_development_timeline(sample_idea, "low")
        assert "2-3 months" in low_timeline["mvp"]  # Low complexity MVP
    
    def test_extract_revenue_models(self, evidence_collector):
        """Test revenue models extraction by category."""
        sources = []
        
        # SaaS revenue models
        saas_idea = Idea(
            title="SaaS Product",
            description="Software as a Service",
            category=IdeaCategory.SAAS
        )
        saas_models = evidence_collector._extract_revenue_models(saas_idea, sources)
        model_types = [model["model"] for model in saas_models]
        assert "subscription" in model_types
        assert "freemium" in model_types
        
        # Marketplace revenue models
        marketplace_idea = Idea(
            title="Marketplace",
            description="Online marketplace",
            category=IdeaCategory.MARKETPLACE
        )
        marketplace_models = evidence_collector._extract_revenue_models(marketplace_idea, sources)
        model_types = [model["model"] for model in marketplace_models]
        assert "commission" in model_types
    
    def test_analyze_cost_structure(self, evidence_collector, sample_idea):
        """Test cost structure analysis."""
        sources = []
        
        result = evidence_collector._analyze_cost_structure(sample_idea, sources)
        
        assert "development_costs" in result
        assert "operational_costs" in result
        assert "customer_acquisition" in result
        assert "major_cost_drivers" in result
        assert "cost_optimization_opportunities" in result
        
        # AI_ML should have high development costs
        assert result["development_costs"] == "high"
    
    def test_analyze_funding_landscape(self, evidence_collector, sample_idea):
        """Test funding landscape analysis."""
        sources = []
        
        result = evidence_collector._analyze_funding_landscape(sample_idea, sources)
        
        assert "hot" in result
        assert "avg_seed" in result
        assert "avg_series_a" in result
        
        # AI_ML should be a hot category
        assert result["hot"] == True
        assert result["avg_seed"] > 1000000  # Should be high for AI
    
    def test_calculate_overall_confidence(self, evidence_collector):
        """Test overall confidence calculation."""
        market_evidence = MarketEvidence(confidence_score=0.8)
        technical_evidence = TechnicalEvidence(confidence_score=0.7)
        business_evidence = BusinessEvidence(confidence_score=0.6)
        
        confidence = evidence_collector._calculate_overall_confidence(
            market_evidence, technical_evidence, business_evidence
        )
        
        # Should be weighted average
        expected = 0.8 * 0.4 + 0.7 * 0.3 + 0.6 * 0.3
        assert abs(confidence - expected) < 0.01
        assert 0.0 <= confidence <= 1.0
    
    def test_calculate_evidence_quality_score(self, evidence_collector):
        """Test evidence quality score calculation."""
        # High quality evidence
        high_quality_sources = [
            EvidenceSource(
                url="https://arxiv.org/paper1",
                title="Paper 1",
                source_type="academic",
                credibility_score=0.9,
                relevance_score=0.8
            ),
            EvidenceSource(
                url="https://mckinsey.com/report",
                title="Report",
                source_type="industry_report",
                credibility_score=0.8,
                relevance_score=0.9
            )
        ]
        
        market_evidence = MarketEvidence(sources=high_quality_sources)
        technical_evidence = TechnicalEvidence(sources=[])
        business_evidence = BusinessEvidence(sources=[])
        
        quality = evidence_collector._calculate_evidence_quality_score(
            market_evidence, technical_evidence, business_evidence
        )
        
        assert 0.0 <= quality <= 1.0
        assert quality > 0.5  # Should be high quality
        
        # No sources should give low quality
        empty_market = MarketEvidence(sources=[])
        empty_technical = TechnicalEvidence(sources=[])
        empty_business = BusinessEvidence(sources=[])
        
        empty_quality = evidence_collector._calculate_evidence_quality_score(
            empty_market, empty_technical, empty_business
        )
        
        assert empty_quality == 0.1  # Minimum quality
    
    def test_generate_evidence_summary(self, evidence_collector, sample_idea):
        """Test evidence summary generation."""
        market_evidence = MarketEvidence(
            market_size_data={"value": 2500.0},
            competitive_landscape={"competitive_density": "medium"}
        )
        technical_evidence = TechnicalEvidence(
            technology_readiness={"trl_level": 7},
            implementation_complexity="high",
            required_resources=["ML Engineers", "Data Scientists", "Infrastructure"]
        )
        business_evidence = BusinessEvidence(
            revenue_models=[{"model": "subscription"}, {"model": "usage_based"}],
            funding_landscape={"hot": True}
        )
        
        summary = evidence_collector._generate_evidence_summary(
            sample_idea, market_evidence, technical_evidence, business_evidence
        )
        
        assert len(summary) > 0
        assert sample_idea.title in summary
        assert "2500" in summary  # Market size
        assert "7" in summary  # TRL level
        assert "high" in summary  # Complexity
    
    def test_extract_key_insights(self, evidence_collector):
        """Test key insights extraction."""
        market_evidence = MarketEvidence(
            market_size_data={"value": 5000.0},  # >$1B
            technology_trends=["AI automation", "DevOps", "Cloud"]
        )
        technical_evidence = TechnicalEvidence(
            technology_readiness={"trl_level": 8}  # High readiness
        )
        business_evidence = BusinessEvidence(
            funding_landscape={"hot": True},
            revenue_models=[{"model": "subscription"}, {"model": "freemium"}, {"model": "usage"}]
        )
        
        insights = evidence_collector._extract_key_insights(
            market_evidence, technical_evidence, business_evidence
        )
        
        assert len(insights) > 0
        assert len(insights) <= 5  # Should limit to 5
        
        # Should include market size insight
        market_insight = any("Large addressable market" in insight for insight in insights)
        assert market_insight
        
        # Should include technology readiness insight
        tech_insight = any("High technology readiness" in insight for insight in insights)
        assert tech_insight
    
    def test_identify_risk_factors(self, evidence_collector):
        """Test risk factors identification."""
        market_evidence = MarketEvidence(
            competitive_landscape={"competitive_density": "high"}
        )
        technical_evidence = TechnicalEvidence(
            technical_risks=["Model accuracy", "Data bias"]
        )
        business_evidence = BusinessEvidence(
            cost_structure={"customer_acquisition": "high"}
        )
        
        risks = evidence_collector._identify_risk_factors(
            market_evidence, technical_evidence, business_evidence
        )
        
        assert len(risks) > 0
        assert len(risks) <= 5  # Should limit to 5
        
        # Should include technical risks
        assert "Model accuracy" in risks
        assert "Data bias" in risks
        
        # Should include market risks
        high_competition_risk = any("High competitive density" in risk for risk in risks)
        assert high_competition_risk
    
    def test_identify_opportunities(self, evidence_collector):
        """Test opportunities identification."""
        market_evidence = MarketEvidence(
            technology_trends=["AI automation", "Cloud computing"]
        )
        technical_evidence = TechnicalEvidence(
            implementation_complexity="low"
        )
        business_evidence = BusinessEvidence(
            funding_landscape={"hot": True}
        )
        
        opportunities = evidence_collector._identify_opportunities(
            market_evidence, technical_evidence, business_evidence
        )
        
        assert len(opportunities) > 0
        assert len(opportunities) <= 5  # Should limit to 5
        
        # Should include trend opportunities
        trend_opportunity = any("AI automation" in opp for opp in opportunities)
        assert trend_opportunity
        
        # Should include complexity opportunity
        complexity_opportunity = any("fast time-to-market" in opp for opp in opportunities)
        assert complexity_opportunity
    
    def test_create_enhanced_evidence_collector_factory(self):
        """Test enhanced evidence collector factory function."""
        collector = create_enhanced_evidence_collector()
        
        assert isinstance(collector, EnhancedEvidenceCollector)
        assert hasattr(collector, 'settings')
        assert hasattr(collector, '_search_engines')
        assert hasattr(collector, '_credibility_weights')


class TestEvidenceModels:
    """Test evidence model classes."""
    
    def test_evidence_source_model(self):
        """Test EvidenceSource model validation."""
        source = EvidenceSource(
            url="https://example.com/research",
            title="Research Paper",
            source_type="academic",
            credibility_score=0.9,
            relevance_score=0.8,
            publish_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            author="Dr. Smith",
            citation_count=25
        )
        
        assert source.url == "https://example.com/research"
        assert source.source_type == "academic"
        assert source.credibility_score == 0.9
        assert source.relevance_score == 0.8
        assert source.citation_count == 25
    
    def test_market_evidence_model(self):
        """Test MarketEvidence model validation."""
        evidence = MarketEvidence(
            market_size_data={"value": 1000.0, "currency": "USD"},
            growth_projections=[{"period": "2025-2030", "cagr": 0.15}],
            customer_segments=[{"segment": "Enterprise", "size": "large"}],
            competitive_landscape={"density": "medium"},
            technology_trends=["AI", "Cloud"],
            confidence_score=0.8,
            last_updated=datetime.now(timezone.utc)
        )
        
        assert evidence.market_size_data["value"] == 1000.0
        assert len(evidence.growth_projections) == 1
        assert len(evidence.technology_trends) == 2
        assert evidence.confidence_score == 0.8
    
    def test_technical_evidence_model(self):
        """Test TechnicalEvidence model validation."""
        evidence = TechnicalEvidence(
            technology_readiness={"trl_level": 7},
            implementation_complexity="medium",
            required_resources=["Engineers", "Infrastructure"],
            technical_risks=["Scalability", "Security"],
            development_timeline={"mvp": "3-6 months"},
            confidence_score=0.7
        )
        
        assert evidence.technology_readiness["trl_level"] == 7
        assert evidence.implementation_complexity == "medium"
        assert len(evidence.required_resources) == 2
        assert len(evidence.technical_risks) == 2
    
    def test_business_evidence_model(self):
        """Test BusinessEvidence model validation."""
        evidence = BusinessEvidence(
            revenue_models=[{"model": "subscription", "description": "Monthly fee"}],
            cost_structure={"development": "high", "operations": "medium"},
            funding_landscape={"hot": True, "avg_seed": 1000000},
            success_stories=[{"company": "Example Corp", "funding": "10M"}],
            confidence_score=0.6
        )
        
        assert len(evidence.revenue_models) == 1
        assert evidence.cost_structure["development"] == "high"
        assert evidence.funding_landscape["hot"] == True
        assert len(evidence.success_stories) == 1
    
    def test_comprehensive_evidence_model(self):
        """Test ComprehensiveEvidence model validation."""
        idea_id = uuid4()
        evidence = ComprehensiveEvidence(
            idea_id=idea_id,
            market_evidence=MarketEvidence(),
            technical_evidence=TechnicalEvidence(),
            business_evidence=BusinessEvidence(),
            overall_confidence=0.7,
            evidence_quality_score=0.8,
            collection_timestamp=datetime.now(timezone.utc),
            summary="Test summary",
            key_insights=["Insight 1", "Insight 2"],
            risk_factors=["Risk 1", "Risk 2"],
            opportunities=["Opportunity 1", "Opportunity 2"]
        )
        
        assert evidence.idea_id == idea_id
        assert evidence.overall_confidence == 0.7
        assert evidence.evidence_quality_score == 0.8
        assert len(evidence.key_insights) == 2
        assert len(evidence.risk_factors) == 2
        assert len(evidence.opportunities) == 2


@pytest.mark.integration
class TestEvidenceCollectorIntegration:
    """Integration tests for evidence collector."""
    
    @pytest.mark.asyncio
    async def test_full_evidence_collection_pipeline(self):
        """Test complete evidence collection pipeline."""
        collector = create_enhanced_evidence_collector()
        
        idea = Idea(
            title="Blockchain Supply Chain Tracker",
            description="Blockchain-based supply chain transparency and tracking system",
            category=IdeaCategory.BLOCKCHAIN,
            current_stage=PipelineStage.RESEARCH
        )
        
        # Mock search results to avoid external dependencies
        with patch.object(collector, '_search_web_evidence') as mock_search:
            mock_search.return_value = [
                EvidenceSource(
                    url="https://example.com/blockchain-supply",
                    title="Blockchain in Supply Chain Management",
                    source_type="industry_report",
                    credibility_score=0.8,
                    relevance_score=0.9
                ),
                EvidenceSource(
                    url="https://academic.com/supply-chain-research",
                    title="Supply Chain Technology Research",
                    source_type="academic",
                    credibility_score=0.9,
                    relevance_score=0.8
                )
            ]
            
            evidence = await collector.collect_comprehensive_evidence(idea, depth="comprehensive")
            
            # Verify comprehensive evidence structure
            assert evidence.idea_id == idea.idea_id
            assert evidence.overall_confidence > 0.0
            assert evidence.evidence_quality_score > 0.0
            
            # Verify market evidence
            market = evidence.market_evidence
            assert market.market_size_data is not None
            assert len(market.growth_projections) > 0
            assert market.competitive_landscape is not None
            assert len(market.technology_trends) > 0
            
            # Verify technical evidence
            technical = evidence.technical_evidence
            assert technical.technology_readiness is not None
            assert technical.implementation_complexity in ["low", "medium", "high"]
            assert len(technical.required_resources) > 0
            assert len(technical.technical_risks) > 0
            
            # Verify business evidence
            business = evidence.business_evidence
            assert len(business.revenue_models) > 0
            assert business.cost_structure is not None
            assert business.funding_landscape is not None
            
            # Verify insights and analysis
            assert len(evidence.summary) > 0
            assert len(evidence.key_insights) > 0
            assert len(evidence.risk_factors) > 0
            assert len(evidence.opportunities) > 0
            
            # Blockchain-specific validations
            assert "Blockchain" in evidence.summary or "blockchain" in evidence.summary
            blockchain_risks = any("Smart contract" in risk for risk in technical.technical_risks)
            assert blockchain_risks  # Should identify smart contract risks