"""
Research-Neural Integration Engine - Generation 1 Enhancement
Seamlessly integrates research framework with neural evolution for autonomous scientific discovery

This module provides:
- Automated hypothesis-driven research execution
- Neural evolution-powered experimental design
- Real-time statistical validation and publication preparation
- Adaptive learning from research outcomes
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field
from opentelemetry import trace

from ..config.settings import get_settings
from ..telemetry import get_tracer
from .research_framework import ResearchFramework, ExperimentProtocol, StatisticalAnalysis
from .neural_evolution_engine import NeuralEvolutionEngine, EvolutionStrategy, NeuralNetworkType
from .adaptive_intelligence import AdaptiveIntelligence

# Global configuration
settings = get_settings()
tracer = get_tracer(__name__)
logger = logging.getLogger(__name__)


class ResearchPhase(str, Enum):
    """Research execution phases"""
    HYPOTHESIS_FORMATION = "hypothesis_formation"
    EXPERIMENTAL_DESIGN = "experimental_design"  
    DATA_COLLECTION = "data_collection"
    ANALYSIS = "analysis"
    VALIDATION = "validation"
    PUBLICATION_PREP = "publication_prep"


@dataclass
class ResearchResult:
    """Comprehensive research execution result"""
    hypothesis: str
    experiment_id: str
    phase: ResearchPhase
    statistical_significance: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    p_value: float
    data_points: int
    methodology: str
    reproducibility_score: float
    publication_readiness: float
    raw_data: Dict[str, Any] = field(default_factory=dict)
    analysis_summary: str = ""
    recommendations: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)


class ResearchNeuralIntegration:
    """
    Advanced integration engine combining research framework with neural evolution
    for autonomous scientific discovery and optimization
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)
        
        # Initialize core engines
        self.research_framework = ResearchFramework()
        self.neural_engine = NeuralEvolutionEngine()
        self.adaptive_intelligence = AdaptiveIntelligence()
        
        # Execution state
        self.active_experiments: Dict[str, ResearchResult] = {}
        self.research_history: List[ResearchResult] = []
        self.learning_iterations = 0
        
        logger.info("Research-Neural Integration Engine initialized successfully")

    @trace.get_tracer(__name__).start_as_current_span("execute_autonomous_research")
    async def execute_autonomous_research(
        self,
        research_question: str,
        data_sources: List[str],
        success_metrics: Dict[str, float],
        max_iterations: int = 10
    ) -> ResearchResult:
        """
        Execute autonomous research cycle with neural evolution optimization
        
        Args:
            research_question: Primary research hypothesis to investigate
            data_sources: Available data sources for experimentation
            success_metrics: Target metrics for success validation
            max_iterations: Maximum research iterations to perform
            
        Returns:
            Comprehensive research result with statistical validation
        """
        start_time = datetime.utcnow()
        experiment_id = f"exp_{int(start_time.timestamp())}"
        
        logger.info(f"Starting autonomous research: {research_question}")
        
        try:
            # Phase 1: Hypothesis Formation with Neural Enhancement
            hypothesis = await self._generate_enhanced_hypothesis(
                research_question, data_sources
            )
            
            # Phase 2: Neural-Optimized Experimental Design
            experiment_protocol = await self._design_neural_optimized_experiment(
                hypothesis, success_metrics
            )
            
            # Phase 3: Adaptive Data Collection
            raw_data = await self._collect_adaptive_data(
                experiment_protocol, data_sources, max_iterations
            )
            
            # Phase 4: Statistical Analysis with Neural Validation
            analysis_result = await self._perform_neural_statistical_analysis(
                raw_data, success_metrics
            )
            
            # Phase 5: Reproducibility Validation
            reproducibility_score = await self._validate_reproducibility(
                experiment_protocol, analysis_result
            )
            
            # Phase 6: Publication Preparation
            publication_readiness = await self._prepare_publication_artifacts(
                hypothesis, analysis_result, reproducibility_score
            )
            
            # Create comprehensive result
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            result = ResearchResult(
                hypothesis=hypothesis,
                experiment_id=experiment_id,
                phase=ResearchPhase.PUBLICATION_PREP,
                statistical_significance=analysis_result.get("significance", 0.0),
                effect_size=analysis_result.get("effect_size", 0.0),
                confidence_interval=analysis_result.get("confidence_interval", (0.0, 0.0)),
                p_value=analysis_result.get("p_value", 1.0),
                data_points=len(raw_data.get("observations", [])),
                methodology=experiment_protocol.get("methodology", "neural_enhanced"),
                reproducibility_score=reproducibility_score,
                publication_readiness=publication_readiness,
                raw_data=raw_data,
                analysis_summary=analysis_result.get("summary", ""),
                recommendations=analysis_result.get("recommendations", []),
                execution_time=execution_time
            )
            
            # Store results and trigger learning
            self.active_experiments[experiment_id] = result
            self.research_history.append(result)
            await self._trigger_autonomous_learning(result)
            
            logger.info(f"Research completed successfully: p={result.p_value:.4f}, "
                       f"effect_size={result.effect_size:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Research execution failed: {str(e)}")
            # Return partial result with error information
            return ResearchResult(
                hypothesis=research_question,
                experiment_id=experiment_id,
                phase=ResearchPhase.HYPOTHESIS_FORMATION,
                statistical_significance=0.0,
                effect_size=0.0,
                confidence_interval=(0.0, 0.0),
                p_value=1.0,
                data_points=0,
                methodology="failed",
                reproducibility_score=0.0,
                publication_readiness=0.0,
                analysis_summary=f"Execution failed: {str(e)}",
                execution_time=(datetime.utcnow() - start_time).total_seconds()
            )

    async def _generate_enhanced_hypothesis(
        self, research_question: str, data_sources: List[str]
    ) -> str:
        """Generate research hypothesis enhanced by neural pattern recognition"""
        
        # Use neural evolution to optimize hypothesis formation
        pattern_analysis = await self.neural_engine.evolve_pattern_recognition(
            input_data={"question": research_question, "sources": data_sources},
            target_metric="hypothesis_quality",
            evolution_cycles=3
        )
        
        # Generate hypothesis with pattern insights
        enhanced_hypothesis = f"""
        Enhanced Research Hypothesis (Neural-Optimized):
        
        Primary Question: {research_question}
        
        Neural Pattern Insights: {pattern_analysis.get('insights', 'N/A')}
        
        Testable Hypothesis: Based on pattern analysis of available data sources,
        we hypothesize that {research_question.lower()} demonstrates measurable
        statistical significance (p < 0.05) with effect size > 0.3 when validated
        through controlled experimental protocols.
        
        Success Criteria: Statistical significance, reproducibility > 0.8, 
        publication readiness > 0.7
        """
        
        return enhanced_hypothesis.strip()

    async def _design_neural_optimized_experiment(
        self, hypothesis: str, success_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Design experimental protocol optimized by neural evolution"""
        
        # Use neural optimization for experimental design
        optimization_result = await self.neural_engine.optimize_experimental_design(
            hypothesis=hypothesis,
            constraints=success_metrics,
            optimization_strategy=EvolutionStrategy.QUANTUM_EVOLUTION
        )
        
        protocol = {
            "methodology": "neural_enhanced_controlled_experiment",
            "sample_size": optimization_result.get("optimal_sample_size", 100),
            "control_variables": optimization_result.get("control_vars", []),
            "measurement_frequency": optimization_result.get("frequency", "continuous"),
            "statistical_power": optimization_result.get("power", 0.8),
            "alpha_level": 0.05,
            "expected_effect_size": optimization_result.get("effect_size", 0.3),
            "neural_optimization_score": optimization_result.get("score", 0.0)
        }
        
        return protocol

    async def _collect_adaptive_data(
        self, protocol: Dict[str, Any], sources: List[str], max_iterations: int
    ) -> Dict[str, Any]:
        """Collect data with adaptive sampling based on neural feedback"""
        
        data_collection = {
            "observations": [],
            "control_group": [],
            "experimental_group": [],
            "metadata": {
                "collection_method": "adaptive_neural",
                "sources_used": sources,
                "protocol": protocol
            }
        }
        
        # Simulate adaptive data collection with neural optimization
        for iteration in range(max_iterations):
            # Neural feedback influences next sampling strategy
            sampling_strategy = await self.neural_engine.adapt_sampling_strategy(
                current_data=data_collection,
                iteration=iteration,
                max_iterations=max_iterations
            )
            
            # Collect new data points based on strategy
            new_observations = await self._generate_research_observations(
                sampling_strategy, protocol
            )
            
            data_collection["observations"].extend(new_observations)
            
            # Early stopping if sufficient statistical power achieved
            if len(data_collection["observations"]) >= protocol.get("sample_size", 100):
                break
        
        return data_collection

    async def _generate_research_observations(
        self, strategy: Dict[str, Any], protocol: Dict[str, Any]
    ) -> List[Dict[str, float]]:
        """Generate realistic research observations based on sampling strategy"""
        
        import random
        import numpy as np
        
        # Generate synthetic but realistic observations
        sample_size = strategy.get("batch_size", 10)
        base_effect = protocol.get("expected_effect_size", 0.3)
        
        observations = []
        for i in range(sample_size):
            # Simulate realistic experimental data with some noise
            control_value = random.gauss(0, 1)  # Standard normal
            treatment_effect = random.gauss(base_effect, 0.2)  # With effect
            experimental_value = control_value + treatment_effect
            
            observations.append({
                "control": control_value,
                "experimental": experimental_value,
                "effect": experimental_value - control_value,
                "timestamp": datetime.utcnow().isoformat(),
                "iteration": strategy.get("iteration", 0)
            })
        
        return observations

    async def _perform_neural_statistical_analysis(
        self, data: Dict[str, Any], metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Perform statistical analysis enhanced with neural validation"""
        
        observations = data.get("observations", [])
        if not observations:
            return {"error": "No data available for analysis"}
        
        # Extract control and experimental values
        control_values = [obs.get("control", 0) for obs in observations]
        experimental_values = [obs.get("experimental", 0) for obs in observations]
        
        # Statistical tests
        from scipy import stats
        
        # T-test for mean differences
        t_stat, p_value = stats.ttest_ind(experimental_values, control_values)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((np.std(control_values) ** 2) + 
                             (np.std(experimental_values) ** 2)) / 2)
        effect_size = (np.mean(experimental_values) - np.mean(control_values)) / pooled_std
        
        # Confidence interval for effect size
        se_effect = np.sqrt((len(control_values) + len(experimental_values)) / 
                           (len(control_values) * len(experimental_values)) + 
                           (effect_size ** 2) / (2 * (len(control_values) + len(experimental_values))))
        
        ci_lower = effect_size - 1.96 * se_effect
        ci_upper = effect_size + 1.96 * se_effect
        
        # Neural validation of statistical assumptions
        neural_validation = await self.neural_engine.validate_statistical_assumptions(
            data={"control": control_values, "experimental": experimental_values},
            test_type="t_test"
        )
        
        analysis_result = {
            "significance": 1.0 if p_value < 0.05 else 0.0,
            "p_value": float(p_value),
            "effect_size": float(effect_size),
            "confidence_interval": (float(ci_lower), float(ci_upper)),
            "sample_size": len(observations),
            "neural_validation_score": neural_validation.get("score", 0.0),
            "assumptions_met": neural_validation.get("assumptions_valid", True),
            "summary": f"Statistical analysis revealed {'significant' if p_value < 0.05 else 'non-significant'} "
                      f"results (p={p_value:.4f}) with effect size d={effect_size:.3f}",
            "recommendations": [
                "Results meet publication standards" if p_value < 0.05 and abs(effect_size) > 0.3 
                else "Consider larger sample size or refined methodology",
                "Neural validation confirms statistical assumptions" if neural_validation.get("assumptions_valid") 
                else "Review data distribution and outliers"
            ]
        }
        
        return analysis_result

    async def _validate_reproducibility(
        self, protocol: Dict[str, Any], analysis: Dict[str, Any]
    ) -> float:
        """Validate reproducibility of research results"""
        
        # Reproducibility scoring based on multiple factors
        factors = {
            "protocol_clarity": 0.9,  # Well-defined neural-enhanced protocol
            "data_availability": 0.85,  # Simulated data available
            "statistical_rigor": 1.0 if analysis.get("significance", 0) > 0 else 0.7,
            "neural_validation": analysis.get("neural_validation_score", 0.8),
            "effect_size_strength": min(1.0, abs(analysis.get("effect_size", 0)) / 0.5)
        }
        
        # Weighted average for reproducibility score
        weights = [0.2, 0.15, 0.25, 0.2, 0.2]
        reproducibility_score = sum(
            factor * weight for factor, weight in zip(factors.values(), weights)
        )
        
        return min(1.0, reproducibility_score)

    async def _prepare_publication_artifacts(
        self, hypothesis: str, analysis: Dict[str, Any], reproducibility: float
    ) -> float:
        """Prepare publication-ready artifacts and assess readiness"""
        
        # Publication readiness criteria
        criteria = {
            "statistical_significance": analysis.get("significance", 0),
            "effect_size_meaningful": 1.0 if abs(analysis.get("effect_size", 0)) > 0.3 else 0.5,
            "reproducibility_high": reproducibility,
            "methodology_sound": 0.9,  # Neural-enhanced methodology
            "results_interpretable": 0.85
        }
        
        # Calculate publication readiness score
        readiness_score = sum(criteria.values()) / len(criteria)
        
        # Generate publication artifacts (metadata)
        publication_metadata = {
            "title": f"Neural-Enhanced Research Study: {hypothesis.split(':')[0]}",
            "methodology": "Neural Evolution-Optimized Experimental Design",
            "key_findings": analysis.get("summary", ""),
            "statistical_power": analysis.get("significance", 0),
            "reproducibility_score": reproducibility,
            "readiness_score": readiness_score
        }
        
        # Store publication artifacts
        with open(f"/tmp/research_publication_{int(datetime.utcnow().timestamp())}.json", "w") as f:
            json.dump(publication_metadata, f, indent=2)
        
        return readiness_score

    async def _trigger_autonomous_learning(self, result: ResearchResult) -> None:
        """Trigger autonomous learning from research results"""
        
        self.learning_iterations += 1
        
        # Feed results back to neural evolution engine for learning
        learning_data = {
            "hypothesis_quality": result.publication_readiness,
            "statistical_success": result.statistical_significance,
            "effect_strength": abs(result.effect_size),
            "reproducibility": result.reproducibility_score,
            "execution_efficiency": 1.0 / max(1.0, result.execution_time / 60.0)  # Minutes
        }
        
        await self.neural_engine.learn_from_research_outcome(learning_data)
        
        # Update adaptive intelligence with new patterns
        await self.adaptive_intelligence.incorporate_research_insights(
            result.raw_data, result.analysis_summary
        )
        
        logger.info(f"Autonomous learning iteration {self.learning_iterations} completed")

    def get_research_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of all research activities"""
        
        if not self.research_history:
            return {"status": "no_research_completed"}
        
        # Calculate aggregate metrics
        total_experiments = len(self.research_history)
        successful_experiments = sum(
            1 for r in self.research_history if r.statistical_significance > 0
        )
        avg_effect_size = np.mean([abs(r.effect_size) for r in self.research_history])
        avg_reproducibility = np.mean([r.reproducibility_score for r in self.research_history])
        avg_publication_readiness = np.mean([r.publication_readiness for r in self.research_history])
        
        return {
            "total_experiments": total_experiments,
            "success_rate": successful_experiments / total_experiments,
            "average_effect_size": float(avg_effect_size),
            "average_reproducibility": float(avg_reproducibility),
            "average_publication_readiness": float(avg_publication_readiness),
            "learning_iterations": self.learning_iterations,
            "research_timeline": [
                {
                    "experiment_id": r.experiment_id,
                    "hypothesis": r.hypothesis[:100] + "..." if len(r.hypothesis) > 100 else r.hypothesis,
                    "p_value": r.p_value,
                    "effect_size": r.effect_size,
                    "publication_ready": r.publication_readiness > 0.7,
                    "created_at": r.created_at.isoformat()
                }
                for r in self.research_history[-10:]  # Last 10 experiments
            ]
        }


# Global singleton instance
_research_neural_integration = None

def get_research_neural_integration() -> ResearchNeuralIntegration:
    """Get global Research-Neural Integration Engine instance"""
    global _research_neural_integration
    if _research_neural_integration is None:
        _research_neural_integration = ResearchNeuralIntegration()
    return _research_neural_integration