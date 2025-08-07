"""
Advanced analytics service for startup idea evaluation and scoring.

This service implements sophisticated algorithms for idea analysis,
market potential assessment, and competitive landscape evaluation.
"""

import logging
import random
from typing import Any

from pydantic import BaseModel

from pipeline.config.settings import get_settings
from pipeline.models.idea import Idea, IdeaCategory, PipelineStage

logger = logging.getLogger(__name__)


class MarketPotentialScore(BaseModel):
    """Market potential assessment result."""

    overall_score: float  # 0.0 to 1.0
    market_size_score: float
    competition_score: float
    timing_score: float
    feasibility_score: float
    innovation_score: float

    market_size_estimate: float | None = None  # USD millions
    competition_density: float | None = None  # 0.0 to 1.0
    risk_factors: list[str] = []
    opportunities: list[str] = []

    confidence_level: float = 0.0  # 0.0 to 1.0
    data_sources: list[str] = []


class CompetitiveAnalysis(BaseModel):
    """Competitive landscape analysis."""

    direct_competitors: list[dict[str, Any]] = []
    indirect_competitors: list[dict[str, Any]] = []
    competitive_advantages: list[str] = []
    competitive_risks: list[str] = []
    market_gap_score: float = 0.0  # 0.0 to 1.0
    differentiation_score: float = 0.0  # 0.0 to 1.0


class FundingPotentialScore(BaseModel):
    """Funding potential assessment."""

    overall_funding_score: float  # 0.0 to 1.0
    stage_alignment_score: float
    investor_appeal_score: float
    scalability_score: float
    team_readiness_score: float

    estimated_funding_range: tuple[float, float] = (0.0, 0.0)  # Min, Max USD
    recommended_funding_stage: str = "pre_seed"
    key_metrics_needed: list[str] = []


class IdeaAnalyticsService:
    """Advanced analytics service for comprehensive idea evaluation."""

    def __init__(self):
        self.settings = get_settings()
        self._market_size_multipliers = {
            IdeaCategory.FINTECH: 1.2,
            IdeaCategory.HEALTHTECH: 1.3,
            IdeaCategory.AI_ML: 1.4,
            IdeaCategory.ENTERPRISE: 1.1,
            IdeaCategory.SAAS: 1.15,
            IdeaCategory.ECOMMERCE: 1.0,
            IdeaCategory.EDTECH: 1.05,
            IdeaCategory.BLOCKCHAIN: 0.9,
            IdeaCategory.CONSUMER: 0.95,
            IdeaCategory.MARKETPLACE: 1.1,
            IdeaCategory.UNCATEGORIZED: 0.8
        }

    async def analyze_market_potential(
        self,
        idea: Idea,
        research_data: dict[str, Any]
    ) -> MarketPotentialScore:
        """
        Perform comprehensive market potential analysis.
        
        Args:
            idea: The startup idea to analyze
            research_data: Additional research context
            
        Returns:
            MarketPotentialScore with detailed assessment
        """
        try:
            # Extract market indicators from research data
            market_indicators = self._extract_market_indicators(research_data)

            # Calculate component scores
            market_size_score = await self._calculate_market_size_score(
                idea, market_indicators
            )
            competition_score = await self._calculate_competition_score(
                idea, market_indicators
            )
            timing_score = await self._calculate_timing_score(
                idea, market_indicators
            )
            feasibility_score = await self._calculate_feasibility_score(
                idea, market_indicators
            )
            innovation_score = await self._calculate_innovation_score(
                idea, market_indicators
            )

            # Weighted overall score
            weights = {
                'market_size': 0.25,
                'competition': 0.20,
                'timing': 0.20,
                'feasibility': 0.25,
                'innovation': 0.10
            }

            overall_score = (
                market_size_score * weights['market_size'] +
                competition_score * weights['competition'] +
                timing_score * weights['timing'] +
                feasibility_score * weights['feasibility'] +
                innovation_score * weights['innovation']
            )

            # Extract additional insights
            market_size_estimate = self._estimate_market_size(idea, market_indicators)
            competition_density = self._calculate_competition_density(market_indicators)
            risk_factors = self._identify_risk_factors(idea, market_indicators)
            opportunities = self._identify_opportunities(idea, market_indicators)

            # Calculate confidence based on data quality
            confidence_level = self._calculate_confidence_level(market_indicators)

            result = MarketPotentialScore(
                overall_score=min(1.0, max(0.0, overall_score)),
                market_size_score=market_size_score,
                competition_score=competition_score,
                timing_score=timing_score,
                feasibility_score=feasibility_score,
                innovation_score=innovation_score,
                market_size_estimate=market_size_estimate,
                competition_density=competition_density,
                risk_factors=risk_factors,
                opportunities=opportunities,
                confidence_level=confidence_level,
                data_sources=market_indicators.get('sources', [])
            )

            logger.info(
                f"Market potential analysis completed for idea {idea.idea_id}",
                extra={
                    "idea_id": str(idea.idea_id),
                    "overall_score": overall_score,
                    "confidence": confidence_level
                }
            )

            return result

        except Exception as e:
            logger.error(
                f"Market potential analysis failed for idea {idea.idea_id}: {e}",
                extra={"idea_id": str(idea.idea_id), "error": str(e)}
            )
            # Return default low scores on error
            return MarketPotentialScore(
                overall_score=0.3,
                market_size_score=0.3,
                competition_score=0.3,
                timing_score=0.3,
                feasibility_score=0.3,
                innovation_score=0.3,
                confidence_level=0.1
            )

    async def analyze_competitive_landscape(
        self,
        idea: Idea,
        research_data: dict[str, Any]
    ) -> CompetitiveAnalysis:
        """
        Analyze competitive landscape and positioning opportunities.
        
        Args:
            idea: The startup idea to analyze
            research_data: Research context with competitor data
            
        Returns:
            CompetitiveAnalysis with detailed competitive insights
        """
        try:
            # Extract competitor information
            competitors_data = research_data.get('competitors', {})

            # Classify competitors
            direct_competitors = self._identify_direct_competitors(
                idea, competitors_data
            )
            indirect_competitors = self._identify_indirect_competitors(
                idea, competitors_data
            )

            # Analyze competitive advantages
            competitive_advantages = self._analyze_competitive_advantages(
                idea, direct_competitors, indirect_competitors
            )

            # Identify competitive risks
            competitive_risks = self._identify_competitive_risks(
                idea, direct_competitors
            )

            # Calculate positioning scores
            market_gap_score = self._calculate_market_gap_score(
                idea, direct_competitors, indirect_competitors
            )
            differentiation_score = self._calculate_differentiation_score(
                idea, competitive_advantages
            )

            result = CompetitiveAnalysis(
                direct_competitors=direct_competitors,
                indirect_competitors=indirect_competitors,
                competitive_advantages=competitive_advantages,
                competitive_risks=competitive_risks,
                market_gap_score=market_gap_score,
                differentiation_score=differentiation_score
            )

            logger.info(
                f"Competitive analysis completed for idea {idea.idea_id}",
                extra={
                    "idea_id": str(idea.idea_id),
                    "direct_competitors": len(direct_competitors),
                    "market_gap_score": market_gap_score
                }
            )

            return result

        except Exception as e:
            logger.error(
                f"Competitive analysis failed for idea {idea.idea_id}: {e}",
                extra={"idea_id": str(idea.idea_id), "error": str(e)}
            )
            return CompetitiveAnalysis()

    async def calculate_funding_potential(
        self,
        idea: Idea,
        market_score: MarketPotentialScore,
        competitive_analysis: CompetitiveAnalysis
    ) -> FundingPotentialScore:
        """
        Calculate funding potential based on multiple factors.
        
        Args:
            idea: The startup idea
            market_score: Market potential assessment
            competitive_analysis: Competitive landscape analysis
            
        Returns:
            FundingPotentialScore with funding recommendations
        """
        try:
            # Calculate component scores
            stage_alignment_score = self._calculate_stage_alignment_score(idea)
            investor_appeal_score = self._calculate_investor_appeal_score(
                idea, market_score
            )
            scalability_score = self._calculate_scalability_score(
                idea, market_score, competitive_analysis
            )
            team_readiness_score = self._calculate_team_readiness_score(idea)

            # Weighted overall funding score
            weights = {
                'stage_alignment': 0.15,
                'investor_appeal': 0.35,
                'scalability': 0.30,
                'team_readiness': 0.20
            }

            overall_funding_score = (
                stage_alignment_score * weights['stage_alignment'] +
                investor_appeal_score * weights['investor_appeal'] +
                scalability_score * weights['scalability'] +
                team_readiness_score * weights['team_readiness']
            )

            # Estimate funding range
            funding_range = self._estimate_funding_range(
                idea, overall_funding_score, market_score
            )

            # Recommend funding stage
            recommended_stage = self._recommend_funding_stage(
                idea, overall_funding_score
            )

            # Identify key metrics needed
            key_metrics = self._identify_key_metrics_needed(idea, recommended_stage)

            result = FundingPotentialScore(
                overall_funding_score=min(1.0, max(0.0, overall_funding_score)),
                stage_alignment_score=stage_alignment_score,
                investor_appeal_score=investor_appeal_score,
                scalability_score=scalability_score,
                team_readiness_score=team_readiness_score,
                estimated_funding_range=funding_range,
                recommended_funding_stage=recommended_stage,
                key_metrics_needed=key_metrics
            )

            logger.info(
                f"Funding potential calculated for idea {idea.idea_id}",
                extra={
                    "idea_id": str(idea.idea_id),
                    "funding_score": overall_funding_score,
                    "recommended_stage": recommended_stage
                }
            )

            return result

        except Exception as e:
            logger.error(
                f"Funding potential calculation failed for idea {idea.idea_id}: {e}",
                extra={"idea_id": str(idea.idea_id), "error": str(e)}
            )
            return FundingPotentialScore(
                overall_funding_score=0.3,
                stage_alignment_score=0.3,
                investor_appeal_score=0.3,
                scalability_score=0.3,
                team_readiness_score=0.3
            )

    # Private helper methods
    def _extract_market_indicators(self, research_data: dict[str, Any]) -> dict[str, Any]:
        """Extract market indicators from research data."""
        return {
            'market_size': research_data.get('market_size', {}),
            'growth_rate': research_data.get('growth_rate', 0.0),
            'trends': research_data.get('trends', []),
            'regulatory_environment': research_data.get('regulatory', {}),
            'technology_readiness': research_data.get('technology', {}),
            'sources': research_data.get('sources', [])
        }

    async def _calculate_market_size_score(
        self,
        idea: Idea,
        indicators: dict[str, Any]
    ) -> float:
        """Calculate market size score."""
        base_score = 0.5  # Default medium score

        # Apply category multiplier
        category_multiplier = self._market_size_multipliers.get(
            idea.category, 1.0
        )

        # Adjust based on market size data
        market_size = indicators.get('market_size', {})
        if market_size.get('value', 0) > 1000:  # > $1B market
            base_score += 0.3
        elif market_size.get('value', 0) > 100:  # > $100M market
            base_score += 0.2

        # Adjust based on growth rate
        growth_rate = indicators.get('growth_rate', 0.0)
        if growth_rate > 0.20:  # >20% growth
            base_score += 0.2
        elif growth_rate > 0.10:  # >10% growth
            base_score += 0.1

        return min(1.0, base_score * category_multiplier)

    async def _calculate_competition_score(
        self,
        idea: Idea,
        indicators: dict[str, Any]
    ) -> float:
        """Calculate competition score (higher = less competition)."""
        # Start with moderate competition assumption
        base_score = 0.6

        # This would integrate with actual competitive intelligence
        # For now, return base score with some randomization for demo
        return min(1.0, max(0.1, base_score + random.uniform(-0.1, 0.1)))

    async def _calculate_timing_score(
        self,
        idea: Idea,
        indicators: dict[str, Any]
    ) -> float:
        """Calculate market timing score."""
        base_score = 0.5

        # Analyze trends alignment
        trends = indicators.get('trends', [])
        if len(trends) > 2:  # Multiple supporting trends
            base_score += 0.3
        elif len(trends) > 0:
            base_score += 0.1

        return min(1.0, base_score)

    async def _calculate_feasibility_score(
        self,
        idea: Idea,
        indicators: dict[str, Any]
    ) -> float:
        """Calculate technical and business feasibility score."""
        base_score = 0.6

        # Technology readiness assessment
        tech_readiness = indicators.get('technology_readiness', {})
        if tech_readiness.get('level', 0) >= 7:  # High readiness
            base_score += 0.2
        elif tech_readiness.get('level', 0) >= 5:  # Medium readiness
            base_score += 0.1

        return min(1.0, base_score)

    async def _calculate_innovation_score(
        self,
        idea: Idea,
        indicators: dict[str, Any]
    ) -> float:
        """Calculate innovation and differentiation score."""
        # This would use NLP analysis of the idea description
        # For now, return a base score with category adjustments
        base_score = 0.5

        if idea.category in [IdeaCategory.AI_ML, IdeaCategory.BLOCKCHAIN]:
            base_score += 0.2
        elif idea.category in [IdeaCategory.FINTECH, IdeaCategory.HEALTHTECH]:
            base_score += 0.1

        return min(1.0, base_score)

    def _estimate_market_size(
        self,
        idea: Idea,
        indicators: dict[str, Any]
    ) -> float | None:
        """Estimate total addressable market size."""
        market_data = indicators.get('market_size', {})
        if 'value' in market_data:
            return float(market_data['value'])

        # Default estimates by category (in millions USD)
        category_estimates = {
            IdeaCategory.FINTECH: 5000.0,
            IdeaCategory.HEALTHTECH: 8000.0,
            IdeaCategory.AI_ML: 12000.0,
            IdeaCategory.ENTERPRISE: 3000.0,
            IdeaCategory.SAAS: 4000.0,
            IdeaCategory.ECOMMERCE: 6000.0,
            IdeaCategory.EDTECH: 2000.0,
            IdeaCategory.BLOCKCHAIN: 1000.0,
            IdeaCategory.CONSUMER: 2500.0,
            IdeaCategory.MARKETPLACE: 1500.0,
            IdeaCategory.UNCATEGORIZED: 1000.0
        }

        return category_estimates.get(idea.category, 1000.0)

    def _calculate_competition_density(self, indicators: dict[str, Any]) -> float:
        """Calculate competition density (0.0 = low, 1.0 = high)."""
        # This would analyze competitor count and market saturation
        return 0.5  # Default medium density

    def _identify_risk_factors(
        self,
        idea: Idea,
        indicators: dict[str, Any]
    ) -> list[str]:
        """Identify key risk factors for the idea."""
        risks = []

        # Common risk factors by category
        category_risks = {
            IdeaCategory.FINTECH: [
                "Regulatory compliance complexity",
                "Security and fraud risks",
                "Customer trust and adoption"
            ],
            IdeaCategory.HEALTHTECH: [
                "Regulatory approval timelines",
                "Data privacy and HIPAA compliance",
                "Clinical validation requirements"
            ],
            IdeaCategory.AI_ML: [
                "Model accuracy and bias",
                "Data quality and availability",
                "Ethical AI considerations"
            ],
            IdeaCategory.ENTERPRISE: [
                "Long sales cycles",
                "Integration complexity",
                "Change management resistance"
            ]
        }

        risks.extend(category_risks.get(idea.category, [
            "Market timing risk",
            "Competitive pressure",
            "Technology adoption barriers"
        ]))

        return risks[:5]  # Return top 5 risks

    def _identify_opportunities(
        self,
        idea: Idea,
        indicators: dict[str, Any]
    ) -> list[str]:
        """Identify key opportunities for the idea."""
        opportunities = []

        # Growth trends create opportunities
        trends = indicators.get('trends', [])
        for trend in trends[:3]:
            opportunities.append(f"Leverage {trend} trend")

        # Category-specific opportunities
        if idea.category == IdeaCategory.AI_ML:
            opportunities.append("AI automation demand growth")
        elif idea.category == IdeaCategory.FINTECH:
            opportunities.append("Digital banking transformation")
        elif idea.category == IdeaCategory.HEALTHTECH:
            opportunities.append("Telehealth market expansion")

        return opportunities[:5]  # Return top 5 opportunities

    def _calculate_confidence_level(self, indicators: dict[str, Any]) -> float:
        """Calculate confidence level based on data quality."""
        sources = indicators.get('sources', [])
        data_points = len([k for k, v in indicators.items() if v])

        base_confidence = 0.3
        base_confidence += min(0.4, len(sources) * 0.1)  # More sources = higher confidence
        base_confidence += min(0.3, data_points * 0.05)  # More data = higher confidence

        return min(1.0, base_confidence)

    def _identify_direct_competitors(
        self,
        idea: Idea,
        competitors_data: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Identify direct competitors."""
        # This would use actual competitor research data
        return competitors_data.get('direct', [])

    def _identify_indirect_competitors(
        self,
        idea: Idea,
        competitors_data: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Identify indirect competitors."""
        return competitors_data.get('indirect', [])

    def _analyze_competitive_advantages(
        self,
        idea: Idea,
        direct_competitors: list[dict[str, Any]],
        indirect_competitors: list[dict[str, Any]]
    ) -> list[str]:
        """Analyze potential competitive advantages."""
        advantages = []

        if len(direct_competitors) < 3:
            advantages.append("Early market entry opportunity")

        if idea.category in [IdeaCategory.AI_ML, IdeaCategory.BLOCKCHAIN]:
            advantages.append("Technology differentiation potential")

        advantages.extend([
            "Focus on underserved market segment",
            "Potential for strong network effects",
            "Data-driven competitive moat potential"
        ])

        return advantages[:5]

    def _identify_competitive_risks(
        self,
        idea: Idea,
        direct_competitors: list[dict[str, Any]]
    ) -> list[str]:
        """Identify competitive risks."""
        risks = []

        if len(direct_competitors) > 5:
            risks.append("High competitive density")

        risks.extend([
            "Large incumbent advantage",
            "Price competition pressure",
            "Customer switching costs",
            "Technology commoditization risk"
        ])

        return risks[:5]

    def _calculate_market_gap_score(
        self,
        idea: Idea,
        direct_competitors: list[dict[str, Any]],
        indirect_competitors: list[dict[str, Any]]
    ) -> float:
        """Calculate market gap score."""
        base_score = 0.7

        # Adjust based on competitor density
        total_competitors = len(direct_competitors) + len(indirect_competitors)
        if total_competitors > 10:
            base_score -= 0.3
        elif total_competitors > 5:
            base_score -= 0.2
        elif total_competitors < 2:
            base_score += 0.2

        return min(1.0, max(0.0, base_score))

    def _calculate_differentiation_score(
        self,
        idea: Idea,
        advantages: list[str]
    ) -> float:
        """Calculate differentiation score."""
        base_score = 0.5
        base_score += min(0.5, len(advantages) * 0.1)
        return min(1.0, base_score)

    def _calculate_stage_alignment_score(self, idea: Idea) -> float:
        """Calculate how well idea aligns with current stage."""
        stage_scores = {
            PipelineStage.IDEATE: 0.8,
            PipelineStage.RESEARCH: 0.9,
            PipelineStage.DECK: 0.85,
            PipelineStage.INVESTORS: 0.7,
            PipelineStage.MVP: 0.6,
            PipelineStage.BUILDING: 0.5,
            PipelineStage.SMOKE_TEST: 0.4,
            PipelineStage.COMPLETE: 0.3
        }
        return stage_scores.get(idea.current_stage, 0.5)

    def _calculate_investor_appeal_score(
        self,
        idea: Idea,
        market_score: MarketPotentialScore
    ) -> float:
        """Calculate investor appeal score."""
        base_score = market_score.overall_score * 0.7

        # Category appeal to investors
        if idea.category in [IdeaCategory.AI_ML, IdeaCategory.FINTECH]:
            base_score += 0.2
        elif idea.category in [IdeaCategory.HEALTHTECH, IdeaCategory.ENTERPRISE]:
            base_score += 0.1

        return min(1.0, base_score)

    def _calculate_scalability_score(
        self,
        idea: Idea,
        market_score: MarketPotentialScore,
        competitive_analysis: CompetitiveAnalysis
    ) -> float:
        """Calculate scalability potential score."""
        base_score = 0.6

        # Large market = better scalability
        if market_score.market_size_estimate and market_score.market_size_estimate > 1000:
            base_score += 0.2

        # Low competition = better scalability
        if competitive_analysis.market_gap_score > 0.7:
            base_score += 0.2

        return min(1.0, base_score)

    def _calculate_team_readiness_score(self, idea: Idea) -> float:
        """Calculate team readiness score."""
        # This would analyze team composition, experience, etc.
        # For now, return moderate score
        return 0.6

    def _estimate_funding_range(
        self,
        idea: Idea,
        funding_score: float,
        market_score: MarketPotentialScore
    ) -> tuple[float, float]:
        """Estimate funding range in USD."""
        base_min = 25000  # $25K minimum
        base_max = 100000  # $100K base maximum

        # Scale based on funding score
        multiplier = 1 + (funding_score * 10)
        max_funding = base_max * multiplier

        # Scale based on market size
        if market_score.market_size_estimate:
            if market_score.market_size_estimate > 5000:  # $5B+ market
                max_funding *= 5
            elif market_score.market_size_estimate > 1000:  # $1B+ market
                max_funding *= 3
            elif market_score.market_size_estimate > 100:  # $100M+ market
                max_funding *= 2

        min_funding = min(base_min * multiplier, max_funding * 0.2)

        return (min_funding, max_funding)

    def _recommend_funding_stage(self, idea: Idea, funding_score: float) -> str:
        """Recommend appropriate funding stage."""
        if idea.current_stage in [PipelineStage.IDEATE, PipelineStage.RESEARCH]:
            return "pre_seed"
        if idea.current_stage in [PipelineStage.DECK, PipelineStage.INVESTORS]:
            return "seed" if funding_score > 0.6 else "pre_seed"
        if idea.current_stage in [PipelineStage.MVP, PipelineStage.BUILDING]:
            return "series_a" if funding_score > 0.8 else "seed"
        return "series_a"

    def _identify_key_metrics_needed(
        self,
        idea: Idea,
        funding_stage: str
    ) -> list[str]:
        """Identify key metrics needed for funding stage."""
        stage_metrics = {
            "pre_seed": [
                "Problem validation",
                "Market size research",
                "Competitive analysis",
                "MVP wireframes",
                "Go-to-market strategy"
            ],
            "seed": [
                "User traction metrics",
                "Revenue model validation",
                "Customer acquisition cost",
                "Product-market fit indicators",
                "Team composition and equity"
            ],
            "series_a": [
                "Monthly recurring revenue",
                "Customer lifetime value",
                "Churn rate and retention",
                "Unit economics",
                "Scalable growth channels"
            ]
        }

        return stage_metrics.get(funding_stage, stage_metrics["pre_seed"])


def create_analytics_service() -> IdeaAnalyticsService:
    """Factory function to create analytics service."""
    return IdeaAnalyticsService()
