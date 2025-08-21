"""
Tests for Research-Neural Integration Engine
Comprehensive testing for autonomous research execution with neural enhancement
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from pipeline.core.research_neural_integration import (
    ResearchNeuralIntegration,
    ResearchResult,
    ResearchPhase,
    get_research_neural_integration
)


class TestResearchNeuralIntegration:
    """Test suite for Research-Neural Integration Engine"""

    @pytest.fixture
    def integration_engine(self):
        """Create test instance of integration engine"""
        return ResearchNeuralIntegration()

    @pytest.fixture
    def sample_research_question(self):
        """Sample research question for testing"""
        return "Does neural evolution optimization improve startup idea validation accuracy?"

    @pytest.fixture
    def sample_data_sources(self):
        """Sample data sources for testing"""
        return ["validation_metrics.json", "user_feedback.csv", "market_data.json"]

    @pytest.fixture
    def sample_success_metrics(self):
        """Sample success metrics for testing"""
        return {
            "accuracy_improvement": 0.15,
            "statistical_significance": 0.05,
            "effect_size_minimum": 0.3
        }

    @pytest.mark.asyncio
    async def test_integration_initialization(self, integration_engine):
        """Test proper initialization of integration engine"""
        assert integration_engine is not None
        assert hasattr(integration_engine, 'research_framework')
        assert hasattr(integration_engine, 'neural_engine')
        assert hasattr(integration_engine, 'adaptive_intelligence')
        assert integration_engine.learning_iterations == 0
        assert len(integration_engine.active_experiments) == 0
        assert len(integration_engine.research_history) == 0

    @pytest.mark.asyncio
    async def test_hypothesis_generation(self, integration_engine, sample_research_question, sample_data_sources):
        """Test enhanced hypothesis generation with neural optimization"""
        
        # Mock neural engine response
        mock_pattern_analysis = {
            "insights": "Neural patterns suggest strong correlation between optimization and accuracy",
            "confidence": 0.85
        }
        
        with patch.object(integration_engine.neural_engine, 'evolve_pattern_recognition', 
                         new_callable=AsyncMock, return_value=mock_pattern_analysis):
            
            hypothesis = await integration_engine._generate_enhanced_hypothesis(
                sample_research_question, sample_data_sources
            )
            
            assert "Enhanced Research Hypothesis" in hypothesis
            assert "Neural Pattern Insights" in hypothesis
            assert "Testable Hypothesis" in hypothesis
            assert sample_research_question.lower() in hypothesis.lower()
            assert "statistical significance" in hypothesis
            assert "p < 0.05" in hypothesis

    @pytest.mark.asyncio
    async def test_experimental_design_optimization(self, integration_engine, sample_success_metrics):
        """Test neural-optimized experimental design"""
        
        hypothesis = "Neural optimization improves validation accuracy"
        
        # Mock optimization result
        mock_optimization = {
            "optimal_sample_size": 150,
            "control_vars": ["baseline_accuracy", "processing_time"],
            "frequency": "continuous",
            "power": 0.85,
            "effect_size": 0.4,
            "score": 0.92
        }
        
        with patch.object(integration_engine.neural_engine, 'optimize_experimental_design',
                         new_callable=AsyncMock, return_value=mock_optimization):
            
            protocol = await integration_engine._design_neural_optimized_experiment(
                hypothesis, sample_success_metrics
            )
            
            assert protocol["methodology"] == "neural_enhanced_controlled_experiment"
            assert protocol["sample_size"] == 150
            assert protocol["statistical_power"] == 0.85
            assert protocol["alpha_level"] == 0.05
            assert "baseline_accuracy" in protocol["control_variables"]
            assert protocol["neural_optimization_score"] == 0.92

    @pytest.mark.asyncio
    async def test_adaptive_data_collection(self, integration_engine, sample_data_sources):
        """Test adaptive data collection with neural feedback"""
        
        protocol = {
            "sample_size": 50,
            "expected_effect_size": 0.3,
            "methodology": "neural_enhanced"
        }
        
        # Mock sampling strategy responses
        mock_strategies = [
            {"batch_size": 10, "iteration": i, "strategy": "adaptive"}
            for i in range(5)
        ]
        
        with patch.object(integration_engine.neural_engine, 'adapt_sampling_strategy',
                         new_callable=AsyncMock, side_effect=mock_strategies):
            
            data = await integration_engine._collect_adaptive_data(
                protocol, sample_data_sources, max_iterations=5
            )
            
            assert "observations" in data
            assert "metadata" in data
            assert len(data["observations"]) >= 40  # Should collect substantial data
            assert data["metadata"]["collection_method"] == "adaptive_neural"
            assert data["metadata"]["sources_used"] == sample_data_sources

    @pytest.mark.asyncio
    async def test_statistical_analysis_with_neural_validation(self, integration_engine, sample_success_metrics):
        """Test statistical analysis enhanced with neural validation"""
        
        # Create realistic test data
        test_data = {
            "observations": [
                {"control": 0.1, "experimental": 0.4, "effect": 0.3},
                {"control": -0.2, "experimental": 0.2, "effect": 0.4},
                {"control": 0.0, "experimental": 0.5, "effect": 0.5},
                {"control": 0.3, "experimental": 0.6, "effect": 0.3},
                {"control": -0.1, "experimental": 0.3, "effect": 0.4}
            ] * 20  # 100 observations total
        }
        
        # Mock neural validation
        mock_validation = {
            "score": 0.88,
            "assumptions_valid": True,
            "normality_check": True,
            "variance_equality": True
        }
        
        with patch.object(integration_engine.neural_engine, 'validate_statistical_assumptions',
                         new_callable=AsyncMock, return_value=mock_validation):
            
            analysis = await integration_engine._perform_neural_statistical_analysis(
                test_data, sample_success_metrics
            )
            
            assert "significance" in analysis
            assert "p_value" in analysis
            assert "effect_size" in analysis
            assert "confidence_interval" in analysis
            assert analysis["neural_validation_score"] == 0.88
            assert analysis["assumptions_met"] is True
            assert len(analysis["recommendations"]) > 0
            assert "Statistical analysis revealed" in analysis["summary"]

    @pytest.mark.asyncio
    async def test_reproducibility_validation(self, integration_engine):
        """Test reproducibility validation scoring"""
        
        protocol = {
            "methodology": "neural_enhanced",
            "sample_size": 100,
            "statistical_power": 0.8
        }
        
        analysis = {
            "significance": 1.0,
            "effect_size": 0.45,
            "neural_validation_score": 0.9
        }
        
        reproducibility = await integration_engine._validate_reproducibility(protocol, analysis)
        
        assert 0.0 <= reproducibility <= 1.0
        assert reproducibility > 0.7  # Should be high for good protocol and analysis

    @pytest.mark.asyncio
    async def test_publication_readiness_assessment(self, integration_engine):
        """Test publication artifact preparation and readiness scoring"""
        
        hypothesis = "Neural optimization significantly improves validation accuracy"
        analysis = {
            "significance": 1.0,
            "effect_size": 0.55,
            "summary": "Significant improvement observed with large effect size"
        }
        reproducibility = 0.85
        
        readiness = await integration_engine._prepare_publication_artifacts(
            hypothesis, analysis, reproducibility
        )
        
        assert 0.0 <= readiness <= 1.0
        assert readiness > 0.7  # Should be publication-ready with these metrics

    @pytest.mark.asyncio
    async def test_autonomous_learning_trigger(self, integration_engine):
        """Test autonomous learning from research results"""
        
        result = ResearchResult(
            hypothesis="Test hypothesis",
            experiment_id="test_001",
            phase=ResearchPhase.PUBLICATION_PREP,
            statistical_significance=1.0,
            effect_size=0.4,
            confidence_interval=(0.2, 0.6),
            p_value=0.02,
            data_points=100,
            methodology="neural_enhanced",
            reproducibility_score=0.85,
            publication_readiness=0.8,
            execution_time=120.0
        )
        
        initial_iterations = integration_engine.learning_iterations
        
        # Mock neural engine learning methods
        with patch.object(integration_engine.neural_engine, 'learn_from_research_outcome',
                         new_callable=AsyncMock) as mock_learn:
            with patch.object(integration_engine.adaptive_intelligence, 'incorporate_research_insights',
                             new_callable=AsyncMock) as mock_incorporate:
                
                await integration_engine._trigger_autonomous_learning(result)
                
                assert integration_engine.learning_iterations == initial_iterations + 1
                mock_learn.assert_called_once()
                mock_incorporate.assert_called_once()

    @pytest.mark.asyncio 
    async def test_full_autonomous_research_execution(self, integration_engine, sample_research_question, 
                                                    sample_data_sources, sample_success_metrics):
        """Test complete autonomous research execution cycle"""
        
        # Mock all neural engine methods
        with patch.object(integration_engine.neural_engine, 'evolve_pattern_recognition',
                         new_callable=AsyncMock, return_value={"insights": "test insights"}):
            with patch.object(integration_engine.neural_engine, 'optimize_experimental_design',
                             new_callable=AsyncMock, return_value={"optimal_sample_size": 100, "score": 0.9}):
                with patch.object(integration_engine.neural_engine, 'adapt_sampling_strategy',
                                 new_callable=AsyncMock, return_value={"batch_size": 10, "iteration": 0}):
                    with patch.object(integration_engine.neural_engine, 'validate_statistical_assumptions',
                                     new_callable=AsyncMock, return_value={"score": 0.85, "assumptions_valid": True}):
                        with patch.object(integration_engine.neural_engine, 'learn_from_research_outcome',
                                         new_callable=AsyncMock):
                            with patch.object(integration_engine.adaptive_intelligence, 'incorporate_research_insights',
                                             new_callable=AsyncMock):
                                
                                result = await integration_engine.execute_autonomous_research(
                                    sample_research_question,
                                    sample_data_sources,
                                    sample_success_metrics,
                                    max_iterations=3
                                )
                                
                                # Verify result structure
                                assert isinstance(result, ResearchResult)
                                assert result.experiment_id.startswith("exp_")
                                assert result.phase == ResearchPhase.PUBLICATION_PREP
                                assert result.hypothesis != ""
                                assert 0.0 <= result.statistical_significance <= 1.0
                                assert isinstance(result.p_value, float)
                                assert isinstance(result.effect_size, float)
                                assert len(result.confidence_interval) == 2
                                assert result.data_points > 0
                                assert result.methodology != ""
                                assert 0.0 <= result.reproducibility_score <= 1.0
                                assert 0.0 <= result.publication_readiness <= 1.0
                                assert result.execution_time > 0
                                
                                # Verify research is tracked
                                assert result.experiment_id in integration_engine.active_experiments
                                assert len(integration_engine.research_history) == 1
                                assert integration_engine.learning_iterations == 1

    def test_research_summary_generation(self, integration_engine):
        """Test research summary generation"""
        
        # Test empty history
        summary = integration_engine.get_research_summary()
        assert summary["status"] == "no_research_completed"
        
        # Add mock research results
        integration_engine.research_history = [
            ResearchResult(
                hypothesis=f"Test hypothesis {i}",
                experiment_id=f"exp_{i}",
                phase=ResearchPhase.PUBLICATION_PREP,
                statistical_significance=1.0 if i % 2 == 0 else 0.0,
                effect_size=0.3 + (i * 0.1),
                confidence_interval=(0.1, 0.5),
                p_value=0.01 if i % 2 == 0 else 0.1,
                data_points=100,
                methodology="neural_enhanced",
                reproducibility_score=0.8,
                publication_readiness=0.7 + (i * 0.05),
                execution_time=60.0
            )
            for i in range(5)
        ]
        
        integration_engine.learning_iterations = 5
        
        summary = integration_engine.get_research_summary()
        
        assert summary["total_experiments"] == 5
        assert summary["success_rate"] == 0.6  # 3 out of 5 successful
        assert summary["average_effect_size"] > 0.3
        assert summary["average_reproducibility"] == 0.8
        assert summary["learning_iterations"] == 5
        assert len(summary["research_timeline"]) == 5

    def test_singleton_pattern(self):
        """Test singleton pattern for global instance"""
        
        instance1 = get_research_neural_integration()
        instance2 = get_research_neural_integration()
        
        assert instance1 is instance2
        assert isinstance(instance1, ResearchNeuralIntegration)


class TestResearchResult:
    """Test ResearchResult data structure"""

    def test_research_result_creation(self):
        """Test ResearchResult instantiation and defaults"""
        
        result = ResearchResult(
            hypothesis="Test hypothesis",
            experiment_id="exp_001",
            phase=ResearchPhase.ANALYSIS,
            statistical_significance=1.0,
            effect_size=0.5,
            confidence_interval=(0.2, 0.8),
            p_value=0.01,
            data_points=150,
            methodology="neural_enhanced",
            reproducibility_score=0.85,
            publication_readiness=0.9
        )
        
        assert result.hypothesis == "Test hypothesis"
        assert result.experiment_id == "exp_001"
        assert result.phase == ResearchPhase.ANALYSIS
        assert result.statistical_significance == 1.0
        assert result.effect_size == 0.5
        assert result.confidence_interval == (0.2, 0.8)
        assert result.p_value == 0.01
        assert result.data_points == 150
        assert result.methodology == "neural_enhanced"
        assert result.reproducibility_score == 0.85
        assert result.publication_readiness == 0.9
        assert isinstance(result.raw_data, dict)
        assert isinstance(result.recommendations, list)
        assert isinstance(result.created_at, datetime)

    def test_research_phases_enum(self):
        """Test ResearchPhase enumeration"""
        
        phases = list(ResearchPhase)
        expected_phases = [
            ResearchPhase.HYPOTHESIS_FORMATION,
            ResearchPhase.EXPERIMENTAL_DESIGN,
            ResearchPhase.DATA_COLLECTION,
            ResearchPhase.ANALYSIS,
            ResearchPhase.VALIDATION,
            ResearchPhase.PUBLICATION_PREP
        ]
        
        assert len(phases) == len(expected_phases)
        for phase in expected_phases:
            assert phase in phases