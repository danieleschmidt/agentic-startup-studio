"""
Main Pipeline Integration - Complete end-to-end data pipeline demonstration.

Orchestrates the full 4-phase workflow:
- Phase 1: Data Ingestion (idea validation and storage)
- Phase 2: Data Processing (RAG evidence collection)
- Phase 3: Data Transformation (pitch deck generation)
- Phase 4: Data Output (campaign creation and MVP generation)

Integrates all core services with budget tracking and quality gates.
"""

import logging
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

# Core service imports
from pipeline.services.budget_sentinel import get_budget_sentinel, BudgetCategory
from pipeline.services.workflow_orchestrator import get_workflow_orchestrator, WorkflowState
from pipeline.services.evidence_collector import get_evidence_collector, ResearchDomain
from pipeline.services.pitch_deck_generator import get_pitch_deck_generator, InvestorType
from pipeline.services.campaign_generator import get_campaign_generator, MVPType, MVPRequest

# Configuration and utilities
from pipeline.config.settings import get_settings
from pipeline.ingestion.idea_manager import create_idea_manager
from pipeline.ingestion.validators import create_validator


@dataclass
class PipelineResult:
    """Complete pipeline execution result."""
    startup_idea: str
    
    # Phase results
    validation_result: Dict[str, Any] = field(default_factory=dict)
    evidence_collection_result: Dict[str, List] = field(default_factory=dict)
    pitch_deck_result: Any = None  # PitchDeck object
    campaign_result: Any = None    # Campaign object
    mvp_result: Any = None         # MVPResult object
    
    # Quality metrics
    overall_quality_score: float = 0.0
    budget_utilization: float = 0.0
    execution_time_seconds: float = 0.0
    
    # Status tracking
    phases_completed: List[str] = field(default_factory=list)
    phases_failed: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    # Metadata
    execution_id: str = ""
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


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
        
        # These will be initialized async in execute_full_pipeline
        self.idea_manager = None
        self.startup_validator = None
        
        # Execution tracking
        self.current_execution_id = ""
        self.execution_start_time = None
    
    async def _initialize_async_dependencies(self):
        """Initialize async dependencies."""
        if self.idea_manager is None:
            self.idea_manager = await create_idea_manager()
        if self.startup_validator is None:
            self.startup_validator = await create_validator()
    
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
                # Phase 1: Data Ingestion and Validation
                await self._execute_phase_1(startup_idea, result)
                
                # Phase 2: Data Processing (Evidence Collection)
                await self._execute_phase_2(startup_idea, result)
                
                # Phase 3: Data Transformation (Pitch Deck Generation)
                await self._execute_phase_3(startup_idea, target_investor, result)
                
                # Phase 4: Data Output (Campaign and MVP Generation)
                await self._execute_phase_4(startup_idea, generate_mvp, result)
                
                # Calculate final metrics
                await self._calculate_final_metrics(result)
                
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
            
            # Collect evidence using RAG methodology
            evidence_by_domain = await self.evidence_collector.collect_evidence(
                claim=startup_idea,
                research_domains=research_domains,
                min_total_evidence=5,
                timeout=120
            )
            
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
    
    async def generate_pipeline_report(self, result: PipelineResult) -> Dict[str, Any]:
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
    
    def _generate_recommendations(self, result: PipelineResult) -> List[str]:
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