"""
Main Pipeline Integration - Complete end-to-end data pipeline demonstration.

Orchestrates the full 4-phase workflow:
- Phase 1: Data Ingestion (idea validation and storage)
- Phase 2: Data Processing (RAG evidence collection)
- Phase 3: Data Transformation (pitch deck generation)
- Phase 4: Data Output (campaign creation and MVP generation)

Integrates all core services with budget tracking and quality gates.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pipeline.config.cache_manager import get_cache_manager

# Configuration and utilities
from pipeline.config.settings import get_settings
from pipeline.ingestion.idea_manager import create_idea_manager
from pipeline.ingestion.validators import create_validator

# Core service imports
from pipeline.services.budget_sentinel import BudgetCategory, get_budget_sentinel
from pipeline.services.campaign_generator import (
    MVPRequest,
    MVPType,
    get_campaign_generator,
)
from pipeline.services.evidence_collector import ResearchDomain, get_evidence_collector
from pipeline.services.pitch_deck_generator import (
    InvestorType,
    get_pitch_deck_generator,
)
from pipeline.services.workflow_orchestrator import (
    get_workflow_orchestrator,
)
from pipeline.core.research_neural_integration import (
    get_research_neural_integration,
)
from pipeline.core.quantum_realtime_orchestrator import (
    get_quantum_realtime_orchestrator,
    OrchestratorMode,
)
from pipeline.core.scalable_evolution_engine import (
    get_scalable_evolution_engine,
    ScalabilityProfile,
)
from pipeline.core.global_compliance_engine import (
    get_global_compliance_engine,
    ComplianceRegion,
    DataClassification,
)


@dataclass
class PipelineResult:
    """Complete pipeline execution result."""
    startup_idea: str

    # Phase results
    validation_result: dict[str, Any] = field(default_factory=dict)
    evidence_collection_result: dict[str, list] = field(default_factory=dict)
    research_result: dict[str, Any] = field(default_factory=dict)  # Research-Neural Integration result
    quantum_orchestrator_metrics: dict[str, Any] = field(default_factory=dict)  # Quantum Real-time Orchestrator metrics
    scalable_evolution_metrics: dict[str, Any] = field(default_factory=dict)  # Scalable Evolution Engine metrics
    global_compliance_metrics: dict[str, Any] = field(default_factory=dict)  # Global Compliance Engine metrics
    global_compliance_assessment: dict[str, Any] = field(default_factory=dict)  # Global compliance assessment results
    data_residency_validations: dict[str, Any] = field(default_factory=dict)  # Data residency validation results
    pitch_deck_result: Any = None  # PitchDeck object
    campaign_result: Any = None    # Campaign object
    mvp_result: Any = None         # MVPResult object

    # Quality metrics
    overall_quality_score: float = 0.0
    budget_utilization: float = 0.0
    execution_time_seconds: float = 0.0

    # Status tracking
    phases_completed: list[str] = field(default_factory=list)
    phases_failed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # Metadata
    execution_id: str = ""
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None


class MainPipeline:
    """Main pipeline orchestrator for complete workflow execution."""

    def __init__(self):
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)

        # Initialize all services
        self.budget_sentinel = get_budget_sentinel()
        self.workflow_orchestrator = get_workflow_orchestrator()
        self.evidence_collector = get_evidence_collector()
        self.pitch_deck_generator = get_pitch_deck_generator()
        self.campaign_generator = get_campaign_generator()
        self.research_neural_integration = get_research_neural_integration()
        self.quantum_orchestrator = get_quantum_realtime_orchestrator(OrchestratorMode.ADAPTIVE)
        
        # Initialize scalability profile for maximum performance
        scalability_profile = ScalabilityProfile(
            max_concurrent_users=10000,
            max_requests_per_second=5000.0,
            memory_scaling_factor=2.0,
            cpu_scaling_factor=3.0,
            storage_scaling_factor=5.0,
            network_bandwidth_limit=20.0,  # 20 Gbps
            cache_hit_ratio_target=0.98,
            response_time_sla_ms=100.0,
            availability_target=0.9999  # 99.99% uptime
        )
        self.scalable_evolution_engine = get_scalable_evolution_engine(scalability_profile)
        self.global_compliance_engine = get_global_compliance_engine()

        # These will be initialized async in execute_full_pipeline
        self.idea_manager = None
        self.startup_validator = None
        self.cache_manager = None

        # Execution tracking
        self.current_execution_id = ""
        self.execution_start_time = None

    async def _initialize_async_dependencies(self):
        """Initialize async dependencies in parallel."""
        # Initialize dependencies concurrently for better performance
        tasks = []

        if self.idea_manager is None:
            tasks.append(self._init_idea_manager())
        if self.startup_validator is None:
            tasks.append(self._init_validator())
        if self.cache_manager is None:
            tasks.append(self._init_cache_manager())

        if tasks:
            await asyncio.gather(*tasks)

    async def _init_idea_manager(self):
        """Initialize idea manager."""
        self.idea_manager = await create_idea_manager()

    async def _init_validator(self):
        """Initialize startup validator."""
        self.startup_validator = await create_validator()

    async def _init_cache_manager(self):
        """Initialize cache manager."""
        self.cache_manager = await get_cache_manager()

    async def execute_full_pipeline(
        self,
        startup_idea: str,
        target_investor: InvestorType = InvestorType.SEED,
        generate_mvp: bool = True,
        max_total_budget: float = 60.0
    ) -> PipelineResult:
        """
        Execute the complete 4-phase data pipeline.
        
        Args:
            startup_idea: The startup concept to process
            target_investor: Type of investor to target for pitch deck
            generate_mvp: Whether to generate MVP alongside campaign
            max_total_budget: Maximum budget for entire pipeline execution
            
        Returns:
            Complete pipeline execution result with all phase outputs
        """
        execution_id = f"pipeline_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.current_execution_id = execution_id
        self.execution_start_time = datetime.utcnow()

        self.logger.info(f"Starting full pipeline execution [{execution_id}]: {startup_idea[:100]}...")

        # Initialize result tracking
        result = PipelineResult(
            startup_idea=startup_idea,
            execution_id=execution_id
        )

        try:
            # Initialize async dependencies
            await self._initialize_async_dependencies()

            # Track total pipeline cost
            async with self.budget_sentinel.track_operation(
                "main_pipeline",
                "execute_full_pipeline",
                BudgetCategory.INFRASTRUCTURE,
                max_total_budget
            ):
                # Initialize and start advanced orchestration engines
                await self.quantum_orchestrator.start()
                await self.scalable_evolution_engine.start()
                
                # Execute global compliance assessment
                await self._execute_global_compliance_assessment(startup_idea, result)
                
                # Phase 1: Data Ingestion and Validation
                await self._execute_phase_1(startup_idea, result)

                # Phase 2: Data Processing (Evidence Collection)
                await self._execute_phase_2(startup_idea, result)

                # Phase 2.5: Research-Neural Integration (Enhanced Analysis)
                await self._execute_phase_2_5_research(startup_idea, result)

                # Phase 3: Data Transformation (Pitch Deck Generation)
                await self._execute_phase_3(startup_idea, target_investor, result)

                # Phase 4: Data Output (Campaign and MVP Generation)
                # Campaign and MVP can be generated in parallel if MVP is requested
                if generate_mvp and result.pitch_deck_result:
                    await self._execute_phase_4_parallel(startup_idea, result)
                else:
                    await self._execute_phase_4(startup_idea, generate_mvp, result)

                # Calculate final metrics
                await self._calculate_final_metrics(result)
                
                # Collect metrics from all advanced engines
                orchestrator_status = self.quantum_orchestrator.get_orchestrator_status()
                result.quantum_orchestrator_metrics = orchestrator_status
                
                evolution_status = self.scalable_evolution_engine.get_scalability_status()
                result.scalable_evolution_metrics = evolution_status
                
                compliance_status = self.global_compliance_engine.get_compliance_summary()
                result.global_compliance_metrics = compliance_status
                
                # Stop advanced engines
                await self.quantum_orchestrator.stop()
                await self.scalable_evolution_engine.stop()

                result.completed_at = datetime.utcnow()
                result.execution_time_seconds = (
                    result.completed_at - result.started_at
                ).total_seconds()

                self.logger.info(
                    f"Pipeline execution completed [{execution_id}]: "
                    f"{len(result.phases_completed)}/4 phases successful, "
                    f"quality score: {result.overall_quality_score:.2f}, "
                    f"budget utilization: {result.budget_utilization:.1%}, "
                    f"execution time: {result.execution_time_seconds:.1f}s"
                )

                return result

        except Exception as e:
            result.errors.append(f"Pipeline execution failed: {str(e)}")
            result.completed_at = datetime.utcnow()
            self.logger.error(f"Pipeline execution failed [{execution_id}]: {e}")
            return result

    async def _execute_phase_1(self, startup_idea: str, result: PipelineResult) -> None:
        """Execute Phase 1: Data Ingestion and Validation."""
        self.logger.info("Executing Phase 1: Data Ingestion and Validation")

        try:
            # Validate startup idea
            validation_result = await self.startup_validator.validate_startup_idea({
                'idea': startup_idea,
                'target_market': 'general',
                'business_model': 'tbd'
            })

            # Store idea if validation passes
            if validation_result.get('is_valid', False):
                idea_record = await self.idea_manager.create_idea({
                    'idea': startup_idea,
                    'validation_score': validation_result.get('overall_score', 0.0),
                    'market_potential': validation_result.get('market_score', 0.0)
                })
                validation_result['idea_id'] = idea_record.get('id')

            result.validation_result = validation_result
            result.phases_completed.append("phase_1_ingestion")

            self.logger.info(
                f"Phase 1 completed: validation score {validation_result.get('overall_score', 0.0):.2f}"
            )

        except Exception as e:
            result.phases_failed.append("phase_1_ingestion")
            result.errors.append(f"Phase 1 failed: {str(e)}")
            self.logger.error(f"Phase 1 execution failed: {e}")

    async def _execute_phase_2(self, startup_idea: str, result: PipelineResult) -> None:
        """Execute Phase 2: Data Processing (Evidence Collection)."""
        self.logger.info("Executing Phase 2: Data Processing (Evidence Collection)")

        try:
            # Define research domains for evidence collection
            research_domains = [
                ResearchDomain(
                    name="market_research",
                    keywords=["market size", "TAM", "industry analysis", "competition"],
                    min_evidence_count=3,
                    quality_threshold=0.6
                ),
                ResearchDomain(
                    name="technology_trends",
                    keywords=["technology", "innovation", "trends", "disruption"],
                    min_evidence_count=2,
                    quality_threshold=0.7
                ),
                ResearchDomain(
                    name="business_validation",
                    keywords=["business model", "revenue", "validation", "success"],
                    min_evidence_count=2,
                    quality_threshold=0.6
                )
            ]

            # Check cache for evidence collection results
            domain_names = [domain.name for domain in research_domains]
            cache_key = f"evidence:{hash(startup_idea)}:{':'.join(sorted(domain_names))}"

            evidence_by_domain = None
            if self.cache_manager:
                evidence_by_domain = await self.cache_manager.get(cache_key)
                if evidence_by_domain:
                    self.logger.info("Using cached evidence collection results")

            # Collect evidence if not cached
            if evidence_by_domain is None:
                evidence_by_domain = await self.evidence_collector.collect_evidence(
                    claim=startup_idea,
                    research_domains=research_domains,
                    min_total_evidence=5,
                    timeout=120
                )

                # Cache the results for 30 minutes
                if self.cache_manager:
                    await self.cache_manager.set(cache_key, evidence_by_domain, ttl_seconds=1800)

            result.evidence_collection_result = {
                domain: [
                    {
                        'claim': evidence.claim_text,
                        'url': evidence.citation_url,
                        'title': evidence.citation_title,
                        'quality_score': evidence.composite_score
                    }
                    for evidence in evidence_list
                ]
                for domain, evidence_list in evidence_by_domain.items()
            }

            result.phases_completed.append("phase_2_processing")

            total_evidence = sum(len(evidence) for evidence in evidence_by_domain.values())
            self.logger.info(f"Phase 2 completed: collected {total_evidence} evidence items")

        except Exception as e:
            result.phases_failed.append("phase_2_processing")
            result.errors.append(f"Phase 2 failed: {str(e)}")
            self.logger.error(f"Phase 2 execution failed: {e}")

    async def _execute_phase_2_5_research(self, startup_idea: str, result: PipelineResult) -> None:
        """Execute Phase 2.5: Research-Neural Integration (Enhanced Analysis)."""
        self.logger.info("Executing Phase 2.5: Research-Neural Integration (Enhanced Analysis)")

        try:
            # Define research question based on startup idea
            research_question = f"Can neural evolution optimization improve the validation accuracy and market potential assessment for the startup idea: {startup_idea}?"
            
            # Define data sources from previous evidence collection
            data_sources = [
                "evidence_collection_results.json",
                "market_validation_data.json", 
                "competitive_analysis.json"
            ]
            
            # Define success metrics for research validation
            success_metrics = {
                "validation_accuracy_improvement": 0.15,  # 15% improvement target
                "statistical_significance": 0.05,        # p < 0.05
                "effect_size_minimum": 0.3,              # Cohen's d > 0.3
                "reproducibility_threshold": 0.8         # 80% reproducibility
            }
            
            # Execute autonomous research with neural enhancement
            research_result = await self.research_neural_integration.execute_autonomous_research(
                research_question=research_question,
                data_sources=data_sources,
                success_metrics=success_metrics,
                max_iterations=5
            )
            
            # Integrate research findings into pipeline result
            result.research_result = {
                'experiment_id': research_result.experiment_id,
                'hypothesis': research_result.hypothesis,
                'statistical_significance': research_result.statistical_significance,
                'p_value': research_result.p_value,
                'effect_size': research_result.effect_size,
                'confidence_interval': research_result.confidence_interval,
                'data_points': research_result.data_points,
                'reproducibility_score': research_result.reproducibility_score,
                'publication_readiness': research_result.publication_readiness,
                'methodology': research_result.methodology,
                'execution_time': research_result.execution_time,
                'recommendations': research_result.recommendations,
                'neural_enhanced': True
            }
            
            # Update validation scores based on research findings if significant
            if research_result.statistical_significance > 0 and research_result.effect_size > 0.3:
                enhancement_factor = min(1.5, 1.0 + (research_result.effect_size * 0.3))
                
                # Enhance existing validation scores
                if 'validation_result' in result.__dict__ and result.validation_result:
                    original_score = result.validation_result.get('overall_score', 0.0)
                    enhanced_score = min(1.0, original_score * enhancement_factor)
                    result.validation_result['neural_enhanced_score'] = enhanced_score
                    result.validation_result['enhancement_factor'] = enhancement_factor
                    result.validation_result['research_validated'] = True
                    
                    self.logger.info(f"Neural research enhanced validation score: "
                                   f"{original_score:.3f} → {enhanced_score:.3f} "
                                   f"(factor: {enhancement_factor:.2f})")

            result.phases_completed.append("phase_2_5_research")
            
            self.logger.info(f"Phase 2.5 completed: research p-value={research_result.p_value:.4f}, "
                           f"effect_size={research_result.effect_size:.3f}, "
                           f"publication_ready={research_result.publication_readiness > 0.7}")

        except Exception as e:
            result.phases_failed.append("phase_2_5_research")
            result.errors.append(f"Phase 2.5 failed: {str(e)}")
            self.logger.error(f"Phase 2.5 execution failed: {e}")
            
            # Add placeholder research result for downstream phases
            result.research_result = {
                'experiment_id': 'failed',
                'statistical_significance': 0.0,
                'neural_enhanced': False,
                'error': str(e)
            }

    async def _execute_global_compliance_assessment(self, startup_idea: str, result: PipelineResult) -> None:
        """Execute global compliance assessment for international markets"""
        self.logger.info("Executing Global Compliance Assessment")
        
        try:
            # Define data types involved in startup operations
            data_types = [
                DataClassification.PERSONAL,           # User data
                DataClassification.SENSITIVE_PERSONAL, # Payment/health data
                DataClassification.CONFIDENTIAL        # Business data
            ]
            
            # Define target markets (global deployment)
            target_regions = [
                ComplianceRegion.EU,        # European market (GDPR)
                ComplianceRegion.US,        # US market (CCPA, HIPAA)
                ComplianceRegion.SINGAPORE, # APAC market (PDPA)
                ComplianceRegion.CANADA     # Canadian market (PIPEDA)
            ]
            
            # Perform comprehensive compliance assessment
            compliance_assessments = await self.global_compliance_engine.assess_compliance(
                data_types=data_types,
                regions=target_regions
            )
            
            # Calculate compliance metrics
            total_assessments = len(compliance_assessments)
            compliant_count = sum(1 for assessment in compliance_assessments if assessment.compliant)
            compliance_rate = compliant_count / total_assessments if total_assessments > 0 else 0.0
            avg_compliance_score = sum(a.score for a in compliance_assessments) / total_assessments if total_assessments > 0 else 0.0
            
            # Store compliance results
            result.global_compliance_assessment = {
                'total_assessments': total_assessments,
                'compliant_assessments': compliant_count,
                'compliance_rate': compliance_rate,
                'average_score': avg_compliance_score,
                'target_regions': [region.value for region in target_regions],
                'assessments': [
                    {
                        'framework': assessment.framework.value,
                        'region': assessment.region.value,
                        'compliant': assessment.compliant,
                        'score': assessment.score,
                        'violations': assessment.violations,
                        'recommendations': assessment.recommendations
                    }
                    for assessment in compliance_assessments
                ],
                'global_ready': compliance_rate >= 0.8 and avg_compliance_score >= 0.9
            }
            
            # Validate data residency requirements
            residency_validations = {}
            for source_region in target_regions:
                for target_region in target_regions:
                    if source_region != target_region:
                        residency_key = f"{source_region.value}_to_{target_region.value}"
                        residency_result = await self.global_compliance_engine.ensure_data_residency(
                            data_types=data_types,
                            source_region=source_region,
                            target_region=target_region
                        )
                        residency_validations[residency_key] = residency_result
            
            result.data_residency_validations = residency_validations
            
            self.logger.info(f"Global compliance assessment completed: "
                           f"compliance_rate={compliance_rate:.1%}, "
                           f"avg_score={avg_compliance_score:.3f}, "
                           f"global_ready={result.global_compliance_assessment['global_ready']}")
                           
        except Exception as e:
            result.phases_failed.append("global_compliance_assessment")
            result.errors.append(f"Global compliance assessment failed: {str(e)}")
            self.logger.error(f"Global compliance assessment failed: {e}")
            
            # Add placeholder compliance result
            result.global_compliance_assessment = {
                'compliance_rate': 0.0,
                'global_ready': False,
                'error': str(e)
            }

    async def _execute_phase_3(
        self,
        startup_idea: str,
        target_investor: InvestorType,
        result: PipelineResult
    ) -> None:
        """Execute Phase 3: Data Transformation (Pitch Deck Generation)."""
        self.logger.info("Executing Phase 3: Data Transformation (Pitch Deck Generation)")

        try:
            # Convert evidence collection result back to evidence objects for pitch deck generation
            # In a real implementation, we'd maintain object references or use proper serialization
            evidence_by_domain = {}

            # Generate pitch deck from startup idea and evidence
            pitch_deck = await self.pitch_deck_generator.generate_pitch_deck(
                startup_idea=startup_idea,
                evidence_by_domain=evidence_by_domain,  # Would use actual evidence objects
                target_investor=target_investor,
                max_cost=8.0
            )

            result.pitch_deck_result = {
                'startup_name': pitch_deck.startup_name,
                'investor_type': pitch_deck.investor_type.value,
                'slide_count': len(pitch_deck.slides),
                'overall_quality_score': pitch_deck.overall_quality_score,
                'completeness_score': pitch_deck.completeness_score,
                'evidence_strength_score': pitch_deck.evidence_strength_score,
                'slides': [
                    {
                        'type': slide.slide_type.value,
                        'title': slide.title,
                        'quality_score': slide.quality_score,
                        'evidence_count': len(slide.supporting_evidence)
                    }
                    for slide in pitch_deck.slides
                ]
            }

            # Store reference to actual pitch deck for next phase
            result._pitch_deck_object = pitch_deck

            result.phases_completed.append("phase_3_transformation")

            self.logger.info(
                f"Phase 3 completed: generated {len(pitch_deck.slides)} slides, "
                f"quality score {pitch_deck.overall_quality_score:.2f}"
            )

        except Exception as e:
            result.phases_failed.append("phase_3_transformation")
            result.errors.append(f"Phase 3 failed: {str(e)}")
            self.logger.error(f"Phase 3 execution failed: {e}")

    async def _execute_phase_4(
        self,
        startup_idea: str,
        generate_mvp: bool,
        result: PipelineResult
    ) -> None:
        """Execute Phase 4: Data Output (Campaign and MVP Generation)."""
        self.logger.info("Executing Phase 4: Data Output (Campaign and MVP Generation)")

        try:
            # Generate smoke test campaign from pitch deck
            if hasattr(result, '_pitch_deck_object'):
                campaign = await self.campaign_generator.generate_smoke_test_campaign(
                    pitch_deck=result._pitch_deck_object,
                    budget_limit=25.0,
                    duration_days=7
                )

                # Execute the campaign
                executed_campaign = await self.campaign_generator.execute_campaign(campaign)

                result.campaign_result = {
                    'name': executed_campaign.name,
                    'type': executed_campaign.campaign_type.value,
                    'status': executed_campaign.status.value,
                    'budget_limit': executed_campaign.budget_limit,
                    'asset_count': len(executed_campaign.assets),
                    'relevance_score': executed_campaign.relevance_score,
                    'engagement_prediction': executed_campaign.engagement_prediction,
                    'google_ads_id': executed_campaign.google_ads_campaign_id,
                    'posthog_project_id': executed_campaign.posthog_project_id,
                    'landing_page_url': executed_campaign.landing_page_url
                }

            # Generate MVP if requested
            if generate_mvp:
                startup_name = result.pitch_deck_result.get('startup_name', 'Startup') if result.pitch_deck_result else 'Startup'

                mvp_request = MVPRequest(
                    mvp_type=MVPType.LANDING_PAGE,
                    startup_name=startup_name,
                    description=startup_idea,
                    key_features=[
                        "User registration and authentication",
                        "Core product demonstration",
                        "Contact and feedback collection"
                    ],
                    target_platforms=["web"],
                    tech_stack=["python", "flask", "html", "css", "javascript"]
                )

                mvp_result = await self.campaign_generator.generate_mvp(
                    mvp_request=mvp_request,
                    max_cost=4.0
                )

                result.mvp_result = {
                    'type': mvp_result.mvp_type.value,
                    'file_count': len(mvp_result.generated_files),
                    'deployment_url': mvp_result.deployment_url,
                    'generation_status': mvp_result.generation_status,
                    'deployment_status': mvp_result.deployment_status,
                    'code_quality_score': mvp_result.code_quality_score,
                    'deployment_success': mvp_result.deployment_success,
                    'generation_cost': mvp_result.generation_cost
                }

            result.phases_completed.append("phase_4_output")

            self.logger.info("Phase 4 completed: campaign and MVP generated successfully")

        except Exception as e:
            result.phases_failed.append("phase_4_output")
            result.errors.append(f"Phase 4 failed: {str(e)}")
            self.logger.error(f"Phase 4 execution failed: {e}")

    async def _execute_phase_4_parallel(self, startup_idea: str, result: PipelineResult) -> None:
        """Execute Phase 4 with parallel campaign and MVP generation."""
        self.logger.info("Executing Phase 4: Parallel Campaign and MVP Generation")

        try:
            # Prepare tasks for parallel execution
            tasks = []

            # Campaign generation task
            if hasattr(result, '_pitch_deck_object'):
                tasks.append(self._generate_campaign(result))

            # MVP generation task
            tasks.append(self._generate_mvp(startup_idea, result))

            # Execute both tasks in parallel
            if tasks:
                campaign_result, mvp_result = await asyncio.gather(*tasks, return_exceptions=True)

                # Handle campaign result
                if not isinstance(campaign_result, Exception):
                    result.campaign_result = campaign_result
                else:
                    result.errors.append(f"Campaign generation failed: {campaign_result}")

                # Handle MVP result
                if not isinstance(mvp_result, Exception):
                    result.mvp_result = mvp_result
                else:
                    result.errors.append(f"MVP generation failed: {mvp_result}")

            result.phases_completed.append("phase_4_output")
            self.logger.info("Phase 4 parallel execution completed successfully")

        except Exception as e:
            result.phases_failed.append("phase_4_output")
            result.errors.append(f"Phase 4 parallel execution failed: {str(e)}")
            self.logger.error(f"Phase 4 parallel execution failed: {e}")

    async def _generate_campaign(self, result: PipelineResult) -> dict[str, Any]:
        """Generate smoke test campaign."""
        campaign = await self.campaign_generator.generate_smoke_test_campaign(
            pitch_deck=result._pitch_deck_object,
            budget_limit=25.0,
            duration_days=7
        )

        executed_campaign = await self.campaign_generator.execute_campaign(campaign)

        return {
            'name': executed_campaign.name,
            'type': executed_campaign.campaign_type.value,
            'status': executed_campaign.status.value,
            'budget_limit': executed_campaign.budget_limit,
            'asset_count': len(executed_campaign.assets),
            'relevance_score': executed_campaign.relevance_score,
            'engagement_prediction': executed_campaign.engagement_prediction,
            'google_ads_id': executed_campaign.google_ads_campaign_id,
            'posthog_project_id': executed_campaign.posthog_project_id,
            'landing_page_url': executed_campaign.landing_page_url
        }

    async def _generate_mvp(self, startup_idea: str, result: PipelineResult) -> dict[str, Any]:
        """Generate MVP."""
        startup_name = result.pitch_deck_result.get('startup_name', 'Startup') if result.pitch_deck_result else 'Startup'

        mvp_request = MVPRequest(
            mvp_type=MVPType.LANDING_PAGE,
            startup_name=startup_name,
            description=startup_idea,
            key_features=[
                "User registration and authentication",
                "Core product demonstration",
                "Contact and feedback collection"
            ],
            target_platforms=["web"],
            tech_stack=["python", "flask", "html", "css", "javascript"]
        )

        mvp_result = await self.campaign_generator.generate_mvp(
            mvp_request=mvp_request,
            max_cost=4.0
        )

        return {
            'type': mvp_result.mvp_type.value,
            'file_count': len(mvp_result.generated_files),
            'deployment_url': mvp_result.deployment_url,
            'generation_status': mvp_result.generation_status,
            'deployment_status': mvp_result.deployment_status,
            'code_quality_score': mvp_result.code_quality_score,
            'deployment_success': mvp_result.deployment_success,
            'generation_cost': mvp_result.generation_cost
        }

    async def _calculate_final_metrics(self, result: PipelineResult) -> None:
        """Calculate final pipeline quality and budget metrics."""
        try:
            # Calculate overall quality score
            quality_scores = []

            if result.validation_result:
                quality_scores.append(result.validation_result.get('overall_score', 0.0))

            if result.pitch_deck_result:
                quality_scores.append(result.pitch_deck_result.get('overall_quality_score', 0.0))

            if result.campaign_result:
                quality_scores.append(result.campaign_result.get('relevance_score', 0.0))

            if result.mvp_result:
                quality_scores.append(result.mvp_result.get('code_quality_score', 0.0))

            result.overall_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

            # Calculate budget utilization
            budget_info = await self.budget_sentinel.get_budget_status()
            total_budget = budget_info.get('total_budget', 62.0)
            spent_budget = budget_info.get('total_spent', 0.0)
            result.budget_utilization = spent_budget / total_budget if total_budget > 0 else 0.0

        except Exception as e:
            self.logger.warning(f"Failed to calculate final metrics: {e}")

    async def generate_pipeline_report(self, result: PipelineResult) -> dict[str, Any]:
        """Generate comprehensive pipeline execution report."""
        return {
            'execution_summary': {
                'execution_id': result.execution_id,
                'startup_idea': result.startup_idea[:200] + "..." if len(result.startup_idea) > 200 else result.startup_idea,
                'phases_completed': len(result.phases_completed),
                'phases_failed': len(result.phases_failed),
                'overall_success': len(result.phases_failed) == 0,
                'execution_time_seconds': result.execution_time_seconds,
                'started_at': result.started_at.isoformat(),
                'completed_at': result.completed_at.isoformat() if result.completed_at else None
            },
            'quality_metrics': {
                'overall_quality_score': result.overall_quality_score,
                'validation_score': result.validation_result.get('overall_score', 0.0) if result.validation_result else 0.0,
                'pitch_deck_quality': result.pitch_deck_result.get('overall_quality_score', 0.0) if result.pitch_deck_result else 0.0,
                'campaign_relevance': result.campaign_result.get('relevance_score', 0.0) if result.campaign_result else 0.0,
                'mvp_code_quality': result.mvp_result.get('code_quality_score', 0.0) if result.mvp_result else 0.0
            },
            'budget_tracking': {
                'budget_utilization': result.budget_utilization,
                'within_budget': result.budget_utilization <= 1.0
            },
            'phase_results': {
                'validation': result.validation_result,
                'evidence_collection': {
                    'domains_researched': len(result.evidence_collection_result),
                    'total_evidence_items': sum(len(evidence) for evidence in result.evidence_collection_result.values())
                },
                'pitch_deck': result.pitch_deck_result,
                'campaign': result.campaign_result,
                'mvp': result.mvp_result
            },
            'errors': result.errors,
            'recommendations': self._generate_recommendations(result)
        }

    def _generate_recommendations(self, result: PipelineResult) -> list[str]:
        """Generate recommendations based on pipeline results."""
        recommendations = []

        if result.overall_quality_score < 0.6:
            recommendations.append("Consider refining the startup concept for better market fit")

        if result.budget_utilization > 0.9:
            recommendations.append("Budget utilization is high; consider optimizing costs")

        if len(result.phases_failed) > 0:
            recommendations.append(f"Address failed phases: {', '.join(result.phases_failed)}")

        if result.validation_result and result.validation_result.get('overall_score', 0.0) < 0.7:
            recommendations.append("Improve idea validation before proceeding to market")

        if result.campaign_result and result.campaign_result.get('relevance_score', 0.0) < 0.7:
            recommendations.append("Optimize campaign messaging and targeting")

        return recommendations


# Singleton instance
_main_pipeline = None


def get_main_pipeline() -> MainPipeline:
    """Get singleton Main Pipeline instance."""
    global _main_pipeline
    if _main_pipeline is None:
        _main_pipeline = MainPipeline()
    return _main_pipeline


# For usage examples and demonstrations, see pipeline/demo_pipeline.py
