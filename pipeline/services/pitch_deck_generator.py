"""
Pitch Deck Generation Service - LangGraph-based transformation of evidence into investment materials.

Implements automated pitch deck generation using:
- LangGraph state machine for structured content generation
- Multi-agent coordination for different slide types
- Quality gates and validation between generation steps
- Template-based customization for different investor types
- Cost tracking and budget enforcement integration
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

# START is not re-exported from langgraph.graph in some versions
# so import it from the submodule where it is defined.
from langgraph.graph import Graph
from langgraph.graph.graph import END, START

from pipeline.config.settings import get_settings
from pipeline.services.budget_sentinel import (
    BudgetCategory,
    BudgetExceededException,
    get_budget_sentinel,
)
from pipeline.services.evidence_collector import Evidence


class SlideType(Enum):
    """Types of pitch deck slides."""

    TITLE = "title"
    PROBLEM = "problem"
    SOLUTION = "solution"
    MARKET_SIZE = "market_size"
    BUSINESS_MODEL = "business_model"
    TRACTION = "traction"
    TEAM = "team"
    FINANCIAL_PROJECTIONS = "financial_projections"
    FUNDING_ASK = "funding_ask"
    USE_OF_FUNDS = "use_of_funds"
    APPENDIX = "appendix"


class InvestorType(Enum):
    """Types of investors for customization."""

    SEED = "seed"
    SERIES_A = "series_a"
    GROWTH = "growth"
    STRATEGIC = "strategic"


@dataclass
class SlideContent:
    """Content for a single slide."""

    slide_type: SlideType
    title: str
    content: str
    bullet_points: list[str] = field(default_factory=list)
    visual_suggestions: list[str] = field(default_factory=list)
    supporting_evidence: list[Evidence] = field(default_factory=list)
    quality_score: float = 0.0
    generated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PitchDeck:
    """Complete pitch deck structure."""

    startup_name: str
    investor_type: InvestorType
    slides: list[SlideContent] = field(default_factory=list)
    executive_summary: str = ""
    appendix_materials: list[str] = field(default_factory=list)

    # Quality metrics
    overall_quality_score: float = 0.0
    completeness_score: float = 0.0
    evidence_strength_score: float = 0.0

    # Metadata
    generated_at: datetime = field(default_factory=datetime.utcnow)
    generated_by: str = "pitch_deck_generator"
    version: str = "1.0"


@dataclass
class GenerationState:
    """State for pitch deck generation workflow."""

    startup_idea: str
    target_investor: InvestorType
    evidence_by_domain: dict[str, list[Evidence]]

    # Generation progress
    current_slide_index: int = 0
    generated_slides: list[SlideContent] = field(default_factory=list)
    failed_slides: list[tuple[SlideType, str]] = field(default_factory=list)

    # Quality tracking
    quality_gates_passed: list[str] = field(default_factory=list)
    quality_issues: list[str] = field(default_factory=list)

    # Cost tracking
    total_cost: float = 0.0
    generation_start_time: datetime = field(default_factory=datetime.utcnow)


class PitchDeckGenerator:
    """LangGraph-based pitch deck generation service."""

    def __init__(self):
        self.settings = get_settings()
        self.budget_sentinel = get_budget_sentinel()
        self.logger = logging.getLogger(__name__)

        # Initialize LLM with proper API key handling
        api_key = self.settings.embedding.api_key
        if not api_key:
            self.logger.warning(
                "OpenAI API key not configured - pitch deck generation will fail"
            )
            self.llm = None
        else:
            self.llm = ChatOpenAI(
                api_key=api_key, model="gpt-4", temperature=0.7, max_tokens=2000
            )

        # Slide templates by investor type
        self.slide_templates = self._load_slide_templates()

        # Quality thresholds
        self.quality_thresholds = {
            "min_slide_quality": 0.7,
            "min_deck_completeness": 0.8,
            "min_evidence_strength": 0.6,
        }

        # Build LangGraph workflow
        self.workflow = self._build_workflow()

    async def generate_pitch_deck(
        self,
        startup_idea: str,
        evidence_by_domain: dict[str, list[Evidence]],
        target_investor: InvestorType = InvestorType.SEED,
        max_cost: float = 10.0,
    ) -> PitchDeck:
        """
        Generate complete pitch deck from startup idea and evidence.

        Args:
            startup_idea: Description of the startup concept
            evidence_by_domain: Evidence collected by research domains
            target_investor: Type of investor to target
            max_cost: Maximum cost allowed for generation

        Returns:
            Complete pitch deck with all slides and quality metrics
        """
        self.logger.info(f"Starting pitch deck generation for: {startup_idea[:100]}...")

        try:
            async with self.budget_sentinel.track_operation(
                "pitch_deck_generator",
                "generate_pitch_deck",
                BudgetCategory.OPENAI,
                max_cost,
            ):
                # Initialize generation state
                state = GenerationState(
                    startup_idea=startup_idea,
                    target_investor=target_investor,
                    evidence_by_domain=evidence_by_domain,
                )

                # Execute workflow
                final_state = await self._execute_workflow(state)

                # Build final pitch deck
                pitch_deck = await self._build_pitch_deck(final_state)

                # Validate quality gates
                await self._validate_final_quality(pitch_deck)

                self.logger.info(
                    f"Pitch deck generation completed: {len(pitch_deck.slides)} slides, "
                    f"quality score: {pitch_deck.overall_quality_score:.2f}"
                )

                return pitch_deck

        except BudgetExceededException as e:
            self.logger.error(f"Pitch deck generation blocked by budget: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Pitch deck generation failed: {e}")
            raise

    def _build_workflow(self) -> Graph:
        """Build LangGraph workflow for pitch deck generation."""
        workflow = Graph()

        # Define workflow nodes
        workflow.add_node("prepare_generation", self._prepare_generation)
        workflow.add_node("generate_title_slide", self._generate_title_slide)
        workflow.add_node("generate_problem_slide", self._generate_problem_slide)
        workflow.add_node("generate_solution_slide", self._generate_solution_slide)
        workflow.add_node("generate_market_slide", self._generate_market_slide)
        workflow.add_node(
            "generate_business_model_slide", self._generate_business_model_slide
        )
        workflow.add_node("generate_traction_slide", self._generate_traction_slide)
        workflow.add_node("generate_team_slide", self._generate_team_slide)
        workflow.add_node("generate_financials_slide", self._generate_financials_slide)
        workflow.add_node("generate_funding_slide", self._generate_funding_slide)
        workflow.add_node("validate_quality", self._validate_quality_gates)
        workflow.add_node("finalize_deck", self._finalize_deck)

        # Define workflow edges
        workflow.add_edge(START, "prepare_generation")
        workflow.add_edge("prepare_generation", "generate_title_slide")
        workflow.add_edge("generate_title_slide", "generate_problem_slide")
        workflow.add_edge("generate_problem_slide", "generate_solution_slide")
        workflow.add_edge("generate_solution_slide", "generate_market_slide")
        workflow.add_edge("generate_market_slide", "generate_business_model_slide")
        workflow.add_edge("generate_business_model_slide", "generate_traction_slide")
        workflow.add_edge("generate_traction_slide", "generate_team_slide")
        workflow.add_edge("generate_team_slide", "generate_financials_slide")
        workflow.add_edge("generate_financials_slide", "generate_funding_slide")
        workflow.add_edge("generate_funding_slide", "validate_quality")
        workflow.add_edge("validate_quality", "finalize_deck")
        workflow.add_edge("finalize_deck", END)

        return workflow.compile()

    async def _execute_workflow(self, state: GenerationState) -> GenerationState:
        """Execute the pitch deck generation workflow."""
        try:
            # Run the workflow
            result = await self.workflow.ainvoke({"state": state})
            return result["state"]

        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            raise

    async def _prepare_generation(self, inputs: dict) -> dict:
        """Prepare for pitch deck generation."""
        state = inputs["state"]

        self.logger.info("Preparing pitch deck generation workflow")

        # Extract startup name from idea
        startup_name = await self._extract_startup_name(state.startup_idea)

        # Analyze evidence for key insights
        key_insights = await self._analyze_evidence_insights(state.evidence_by_domain)

        # Store preparation results
        state.startup_name = startup_name
        state.key_insights = key_insights
        state.quality_gates_passed.append("preparation_complete")

        return {"state": state}

    async def _generate_title_slide(self, inputs: dict) -> dict:
        """Generate title slide."""
        state = inputs["state"]

        slide_content = await self._generate_slide_content(
            state,
            SlideType.TITLE,
            f"Create a compelling title slide for {state.startup_name}",
            [],  # No specific evidence needed for title
        )

        state.generated_slides.append(slide_content)
        state.current_slide_index += 1

        return {"state": state}

    async def _generate_problem_slide(self, inputs: dict) -> dict:
        """Generate problem slide."""
        state = inputs["state"]

        # Find relevant evidence for problem statement
        problem_evidence = self._find_relevant_evidence(
            state.evidence_by_domain,
            ["market research", "user research", "pain points", "problems"],
        )

        slide_content = await self._generate_slide_content(
            state,
            SlideType.PROBLEM,
            "Define the core problem this startup solves",
            problem_evidence,
        )

        state.generated_slides.append(slide_content)
        state.current_slide_index += 1

        return {"state": state}

    async def _generate_solution_slide(self, inputs: dict) -> dict:
        """Generate solution slide."""
        state = inputs["state"]

        solution_evidence = self._find_relevant_evidence(
            state.evidence_by_domain,
            ["solution", "technology", "approach", "methodology"],
        )

        slide_content = await self._generate_slide_content(
            state,
            SlideType.SOLUTION,
            "Present the innovative solution and unique value proposition",
            solution_evidence,
        )

        state.generated_slides.append(slide_content)
        state.current_slide_index += 1

        return {"state": state}

    async def _generate_market_slide(self, inputs: dict) -> dict:
        """Generate market size slide."""
        state = inputs["state"]

        market_evidence = self._find_relevant_evidence(
            state.evidence_by_domain,
            ["market size", "TAM", "SAM", "market analysis", "industry"],
        )

        slide_content = await self._generate_slide_content(
            state,
            SlideType.MARKET_SIZE,
            "Analyze the market opportunity and size",
            market_evidence,
        )

        state.generated_slides.append(slide_content)
        state.current_slide_index += 1

        return {"state": state}

    async def _generate_business_model_slide(self, inputs: dict) -> dict:
        """Generate business model slide."""
        state = inputs["state"]

        business_evidence = self._find_relevant_evidence(
            state.evidence_by_domain,
            ["business model", "revenue", "pricing", "monetization"],
        )

        slide_content = await self._generate_slide_content(
            state,
            SlideType.BUSINESS_MODEL,
            "Outline the business model and revenue strategy",
            business_evidence,
        )

        state.generated_slides.append(slide_content)
        state.current_slide_index += 1

        return {"state": state}

    async def _generate_traction_slide(self, inputs: dict) -> dict:
        """Generate traction slide."""
        state = inputs["state"]

        traction_evidence = self._find_relevant_evidence(
            state.evidence_by_domain,
            ["traction", "growth", "customers", "metrics", "KPIs"],
        )

        slide_content = await self._generate_slide_content(
            state,
            SlideType.TRACTION,
            "Demonstrate early traction and growth metrics",
            traction_evidence,
        )

        state.generated_slides.append(slide_content)
        state.current_slide_index += 1

        return {"state": state}

    async def _generate_team_slide(self, inputs: dict) -> dict:
        """Generate team slide."""
        state = inputs["state"]

        team_evidence = self._find_relevant_evidence(
            state.evidence_by_domain,
            ["team", "founders", "leadership", "experience", "background"],
        )

        slide_content = await self._generate_slide_content(
            state,
            SlideType.TEAM,
            "Present the founding team and key personnel",
            team_evidence,
        )

        state.generated_slides.append(slide_content)
        state.current_slide_index += 1

        return {"state": state}

    async def _generate_financials_slide(self, inputs: dict) -> dict:
        """Generate financial projections slide."""
        state = inputs["state"]

        financial_evidence = self._find_relevant_evidence(
            state.evidence_by_domain,
            ["financials", "projections", "revenue", "costs", "budget"],
        )

        slide_content = await self._generate_slide_content(
            state,
            SlideType.FINANCIAL_PROJECTIONS,
            "Present financial projections and key metrics",
            financial_evidence,
        )

        state.generated_slides.append(slide_content)
        state.current_slide_index += 1

        return {"state": state}

    async def _generate_funding_slide(self, inputs: dict) -> dict:
        """Generate funding ask slide."""
        state = inputs["state"]

        funding_evidence = self._find_relevant_evidence(
            state.evidence_by_domain,
            ["funding", "investment", "capital", "use of funds"],
        )

        slide_content = await self._generate_slide_content(
            state,
            SlideType.FUNDING_ASK,
            "Present funding requirements and use of funds",
            funding_evidence,
        )

        state.generated_slides.append(slide_content)
        state.current_slide_index += 1

        return {"state": state}

    async def _validate_quality_gates(self, inputs: dict) -> dict:
        """Validate quality gates for generated content."""
        state = inputs["state"]

        # Validate slide quality
        low_quality_slides = [
            slide
            for slide in state.generated_slides
            if slide.quality_score < self.quality_thresholds["min_slide_quality"]
        ]

        if low_quality_slides:
            state.quality_issues.append(
                f"Found {len(low_quality_slides)} slides below quality threshold"
            )

        # Validate completeness
        expected_slides = len(SlideType) - 1  # Exclude appendix
        completeness = len(state.generated_slides) / expected_slides

        if completeness < self.quality_thresholds["min_deck_completeness"]:
            state.quality_issues.append(
                f"Deck completeness {completeness:.2f} below threshold"
            )

        # Validate evidence strength
        total_evidence = sum(
            len(slide.supporting_evidence) for slide in state.generated_slides
        )
        evidence_strength = min(total_evidence / len(state.generated_slides), 1.0)

        if evidence_strength < self.quality_thresholds["min_evidence_strength"]:
            state.quality_issues.append(
                f"Evidence strength {evidence_strength:.2f} below threshold"
            )

        if not state.quality_issues:
            state.quality_gates_passed.append("quality_validation_passed")

        return {"state": state}

    async def _finalize_deck(self, inputs: dict) -> dict:
        """Finalize the pitch deck."""
        state = inputs["state"]

        # Generate executive summary
        state.executive_summary = await self._generate_executive_summary(state)

        # Calculate final quality scores
        state.overall_quality_score = sum(
            slide.quality_score for slide in state.generated_slides
        ) / len(state.generated_slides)

        state.quality_gates_passed.append("finalization_complete")

        return {"state": state}

    async def _generate_slide_content(
        self,
        state: GenerationState,
        slide_type: SlideType,
        prompt: str,
        evidence: list[Evidence],
    ) -> SlideContent:
        """Generate content for a specific slide."""
        try:
            # Build context from evidence
            evidence_context = self._build_evidence_context(evidence)

            # Get template for investor type
            template = self.slide_templates[state.target_investor][slide_type]

            # Create generation prompt
            full_prompt = (
                f"""
            {template}

            Startup Idea: {state.startup_idea}

            Task: {prompt}

            Supporting Evidence:
            {evidence_context}

            Generate engaging, investor-focused content for this slide.
            Include specific bullet points and visual suggestions.
            """
            )

            # Generate content using LLM
            messages = [SystemMessage(content=full_prompt)]
            # Use exponential backoff for transient LLM failures
            from pipeline.utils.backoff import async_retry

            response = await async_retry(self.llm.ainvoke, messages)

            # Parse response into structured content
            slide_content = self._parse_slide_response(
                slide_type, response.content, evidence
            )

            # Score quality
            slide_content.quality_score = await self._score_slide_quality(slide_content)

            return slide_content

        except Exception as e:
            self.logger.error(f"Failed to generate {slide_type.value} slide: {e}")
            # Return minimal slide content on failure
            return SlideContent(
                slide_type=slide_type,
                title=f"{slide_type.value.replace('_', ' ').title()}",
                content="Content generation failed",
                quality_score=0.0,
            )

    def _find_relevant_evidence(
        self, evidence_by_domain: dict[str, list[Evidence]], keywords: list[str]
    ) -> list[Evidence]:
        """Find evidence relevant to specific keywords."""
        relevant_evidence = []

        for domain_evidence in evidence_by_domain.values():
            for evidence in domain_evidence:
                # Check if evidence contains any of the keywords
                content_lower = (evidence.claim_text + " " + evidence.snippet).lower()
                if any(keyword.lower() in content_lower for keyword in keywords):
                    relevant_evidence.append(evidence)

        # Sort by composite score and return top 3
        relevant_evidence.sort(key=lambda x: x.composite_score, reverse=True)
        return relevant_evidence[:3]

    def _build_evidence_context(self, evidence: list[Evidence]) -> str:
        """Build context string from evidence."""
        if not evidence:
            return "No specific evidence available."

        context_parts = []
        for i, item in enumerate(evidence, 1):
            context_parts.append(
                f"{i}. {item.claim_text}\n"
                f"   Source: {item.citation_title} ({item.citation_source})\n"
                f"   Quality: {item.composite_score:.2f}"
            )

        return "\n\n".join(context_parts)

    def _parse_slide_response(
        self, slide_type: SlideType, response_content: str, evidence: list[Evidence]
    ) -> SlideContent:
        """Parse LLM response into structured slide content."""
        # Simple parsing (in production, would use more sophisticated parsing)
        lines = response_content.strip().split("\n")

        title = slide_type.value.replace("_", " ").title()
        content = response_content
        bullet_points = [
            line.strip("- ") for line in lines if line.strip().startswith("- ")
        ]

        return SlideContent(
            slide_type=slide_type,
            title=title,
            content=content,
            bullet_points=bullet_points,
            supporting_evidence=evidence,
            visual_suggestions=["Chart", "Graph", "Diagram"],  # Placeholder
        )

    async def _score_slide_quality(self, slide_content: SlideContent) -> float:
        """Score the quality of a slide (0.0 to 1.0)."""
        score = 0.0

        # Content length check
        if len(slide_content.content) > 100:
            score += 0.2

        # Bullet points check
        if len(slide_content.bullet_points) >= 3:
            score += 0.3

        # Evidence support check
        if len(slide_content.supporting_evidence) > 0:
            score += 0.3

        # Visual suggestions check
        if len(slide_content.visual_suggestions) > 0:
            score += 0.2

        return min(score, 1.0)

    async def _build_pitch_deck(self, state: GenerationState) -> PitchDeck:
        """Build final pitch deck from generation state."""
        return PitchDeck(
            startup_name=getattr(state, "startup_name", "Startup"),
            investor_type=state.target_investor,
            slides=state.generated_slides,
            executive_summary=getattr(state, "executive_summary", ""),
            overall_quality_score=state.overall_quality_score,
            completeness_score=len(state.generated_slides) / 9,  # 9 core slides
            evidence_strength_score=sum(
                len(slide.supporting_evidence) for slide in state.generated_slides
            )
            / max(len(state.generated_slides), 1),
        )

    async def _validate_final_quality(self, pitch_deck: PitchDeck) -> None:
        """Validate final pitch deck quality."""
        if pitch_deck.overall_quality_score < 0.6:
            self.logger.warning(
                f"Pitch deck quality score {pitch_deck.overall_quality_score:.2f} "
                "below recommended threshold"
            )

    # Placeholder methods for complex operations
    async def _extract_startup_name(self, startup_idea: str) -> str:
        """Extract startup name from idea description."""
        # Simplified extraction
        words = startup_idea.split()[:3]
        return " ".join(word.capitalize() for word in words)

    async def _analyze_evidence_insights(
        self, evidence_by_domain: dict[str, list[Evidence]]
    ) -> dict:
        """Analyze evidence for key insights."""
        return {"total_evidence": sum(len(ev) for ev in evidence_by_domain.values())}

    async def _generate_executive_summary(self, state: GenerationState) -> str:
        """Generate executive summary."""
        return f"Executive summary for {getattr(state, 'startup_name', 'startup')}"

    def _load_slide_templates(self) -> dict[InvestorType, dict[SlideType, str]]:
        """Load slide templates by investor type."""
        # Simplified templates (in production, would load from files)
        base_template = "Create a professional slide focusing on the key points."

        return {
            investor_type: dict.fromkeys(SlideType, base_template)
            for investor_type in InvestorType
        }


# Singleton instance
_pitch_deck_generator = None


def get_pitch_deck_generator() -> PitchDeckGenerator:
    """Get singleton Pitch Deck Generator instance."""
    global _pitch_deck_generator
    if _pitch_deck_generator is None:
        _pitch_deck_generator = PitchDeckGenerator()
    return _pitch_deck_generator
