"""
Service Factory - Factory pattern for service creation and configuration.

Provides centralized service creation with dependency injection,
configuration management, and lifecycle control.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Type, TypeVar

from pipeline.config.settings import get_settings
from pipeline.core.interfaces import *
from pipeline.core.service_registry import ServiceRegistry, get_service_registry

T = TypeVar('T')


class ServiceFactory:
    """Factory for creating and configuring pipeline services."""
    
    def __init__(self, registry: Optional[ServiceRegistry] = None):
        self.registry = registry or get_service_registry()
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the service factory and register all services."""
        if self._initialized:
            return
        
        # Register core services in dependency order
        await self._register_core_services()
        await self._register_infrastructure_services()
        await self._register_business_services()
        await self._register_pipeline_services()
        
        # Initialize all services
        await self.registry.initialize_all()
        
        self._initialized = True
        self.logger.info("Service factory initialized successfully")
    
    async def _register_core_services(self) -> None:
        """Register core infrastructure services."""
        from pipeline.config.cache_manager import CacheManager
        from pipeline.config.connection_pool import ConnectionPoolManager
        from pipeline.config.secrets_manager import SecretsManager
        
        # Cache Manager (highest priority - many services depend on it)
        self.registry.register_service(
            service_type=CacheManager,
            service_id="cache_manager",
            dependencies=[],
            startup_priority=10
        )
        
        # Connection Pool Manager
        self.registry.register_service(
            service_type=ConnectionPoolManager,
            service_id="connection_pool",
            dependencies=[],
            startup_priority=20
        )
        
        # Secrets Manager
        self.registry.register_service(
            service_type=SecretsManager,
            service_id="secrets_manager",
            dependencies=[],
            startup_priority=15
        )
    
    async def _register_infrastructure_services(self) -> None:
        """Register infrastructure and data services."""
        from pipeline.storage.optimized_vector_search import OptimizedVectorSearch
        from pipeline.storage.idea_repository import DatabaseManager, IdeaRepository
        
        # Vector Search Service
        self.registry.register_service(
            service_type=OptimizedVectorSearch,
            service_id="vector_search",
            dependencies=["cache_manager", "connection_pool"],
            startup_priority=30
        )
        
        # Database Manager
        self.registry.register_service(
            service_type=DatabaseManager,
            service_id="database_manager",
            dependencies=["connection_pool", "secrets_manager"],
            startup_priority=25,
            initialization_args={"config": self.settings.database}
        )
        
        # Idea Repository
        self.registry.register_service(
            service_type=IdeaRepository,
            service_id="idea_repository",
            dependencies=["database_manager"],
            startup_priority=35
        )
    
    async def _register_business_services(self) -> None:
        """Register business logic services."""
        from pipeline.services.budget_sentinel import BudgetSentinel
        from pipeline.services.evidence_collector import EvidenceCollector
        from pipeline.services.pitch_deck_generator import PitchDeckGenerator
        from pipeline.services.campaign_generator import CampaignGenerator
        from pipeline.services.claude_code_service import ClaudeCodeService
        
        # Claude Code Service (high priority if enabled)
        if self.settings.claude_code.enabled:
            self.registry.register_service(
                service_type=ClaudeCodeService,
                service_id="claude_code_service",
                dependencies=["cache_manager"],
                startup_priority=35
            )
        
        # Budget Sentinel (high priority - controls spending)
        self.registry.register_service(
            service_type=BudgetSentinel,
            service_id="budget_sentinel",
            dependencies=["cache_manager"],
            startup_priority=40
        )
        
        # Evidence Collector
        self.registry.register_service(
            service_type=EvidenceCollector,
            service_id="evidence_collector",
            dependencies=["budget_sentinel", "cache_manager"],
            startup_priority=50
        )
        
        # Pitch Deck Generator
        self.registry.register_service(
            service_type=PitchDeckGenerator,
            service_id="pitch_deck_generator",
            dependencies=["budget_sentinel", "cache_manager"],
            startup_priority=60
        )
        
        # Campaign Generator
        self.registry.register_service(
            service_type=CampaignGenerator,
            service_id="campaign_generator",
            dependencies=["budget_sentinel", "cache_manager"],
            startup_priority=70
        )
    
    async def _register_pipeline_services(self) -> None:
        """Register pipeline orchestration services."""
        from pipeline.services.workflow_orchestrator import WorkflowOrchestrator
        from pipeline.main_pipeline import MainPipeline
        from pipeline.ingestion.idea_manager import IdeaManager
        from pipeline.ingestion.validators import StartupValidator
        
        # Workflow Orchestrator
        self.registry.register_service(
            service_type=WorkflowOrchestrator,
            service_id="workflow_orchestrator",
            dependencies=["budget_sentinel"],
            startup_priority=80
        )
        
        # Idea Manager
        self.registry.register_service(
            service_type=IdeaManager,
            service_id="idea_manager",
            dependencies=["idea_repository", "vector_search"],
            startup_priority=90
        )
        
        # Startup Validator
        self.registry.register_service(
            service_type=StartupValidator,
            service_id="startup_validator",
            dependencies=["cache_manager"],
            startup_priority=95
        )
        
        # Main Pipeline (lowest priority - depends on everything)
        self.registry.register_service(
            service_type=MainPipeline,
            service_id="main_pipeline",
            dependencies=[
                "budget_sentinel",
                "workflow_orchestrator", 
                "evidence_collector",
                "pitch_deck_generator",
                "campaign_generator",
                "idea_manager",
                "startup_validator",
                "cache_manager"
            ],
            startup_priority=100
        )
    
    async def get_budget_tracker(self) -> IBudgetTracker:
        """Get budget tracking service."""
        return await self.registry.get_service("budget_sentinel")
    
    async def get_evidence_collector(self) -> IEvidenceCollector:
        """Get evidence collection service."""
        return await self.registry.get_service("evidence_collector")
    
    async def get_pitch_deck_generator(self) -> IPitchDeckGenerator:
        """Get pitch deck generation service."""
        return await self.registry.get_service("pitch_deck_generator")
    
    async def get_campaign_generator(self) -> ICampaignGenerator:
        """Get campaign generation service."""
        return await self.registry.get_service("campaign_generator")
    
    async def get_workflow_orchestrator(self) -> IWorkflowOrchestrator:
        """Get workflow orchestration service."""
        return await self.registry.get_service("workflow_orchestrator")
    
    async def get_idea_repository(self) -> IIdeaRepository:
        """Get idea repository service."""
        return await self.registry.get_service("idea_repository")
    
    async def get_cache_manager(self) -> ICacheManager:
        """Get cache management service."""
        return await self.registry.get_service("cache_manager")
    
    async def get_vector_search(self) -> IVectorSearch:
        """Get vector search service."""
        return await self.registry.get_service("vector_search")
    
    async def get_pipeline_executor(self) -> IPipelineExecutor:
        """Get main pipeline executor."""
        return await self.registry.get_service("main_pipeline")
    
    async def get_claude_code_service(self):
        """Get Claude Code service if enabled."""
        if self.settings.claude_code.enabled:
            from pipeline.services.claude_code_service import ClaudeCodeService
            return await self.registry.get_service("claude_code_service")
        return None
    
    async def get_service_by_interface(self, interface_type: Type[T]) -> T:
        """Get service by interface type."""
        return await self.registry.get_service_by_type(interface_type)
    
    async def shutdown(self) -> None:
        """Shutdown all services."""
        await self.registry.shutdown_all()
        self._initialized = False
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all services."""
        service_health = await self.registry.health_check_all()
        service_info = self.registry.get_service_info()
        
        overall_health = all(service_health.values())
        
        return {
            'overall_health': overall_health,
            'service_count': len(service_health),
            'healthy_services': sum(service_health.values()),
            'unhealthy_services': len(service_health) - sum(service_health.values()),
            'services': {
                service_id: {
                    'healthy': service_health[service_id],
                    'info': service_info[service_id]
                }
                for service_id in service_health
            }
        }


class ServiceContainer:
    """Dependency injection container for pipeline services."""
    
    def __init__(self):
        self.factory = ServiceFactory()
        self._context_manager = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.factory.initialize()
        self._context_manager = self.factory.registry.service_context()
        await self._context_manager.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._context_manager:
            await self._context_manager.__aexit__(exc_type, exc_val, exc_tb)
    
    async def get_service(self, service_id: str) -> Any:
        """Get service by ID."""
        return await self.factory.registry.get_service(service_id)
    
    async def get_service_by_type(self, service_type: Type[T]) -> T:
        """Get service by type."""
        return await self.factory.registry.get_service_by_type(service_type)
    
    # Convenience methods for common services
    async def budget_tracker(self) -> IBudgetTracker:
        return await self.factory.get_budget_tracker()
    
    async def evidence_collector(self) -> IEvidenceCollector:
        return await self.factory.get_evidence_collector()
    
    async def pitch_deck_generator(self) -> IPitchDeckGenerator:
        return await self.factory.get_pitch_deck_generator()
    
    async def campaign_generator(self) -> ICampaignGenerator:
        return await self.factory.get_campaign_generator()
    
    async def workflow_orchestrator(self) -> IWorkflowOrchestrator:
        return await self.factory.get_workflow_orchestrator()
    
    async def idea_repository(self) -> IIdeaRepository:
        return await self.factory.get_idea_repository()
    
    async def cache_manager(self) -> ICacheManager:
        return await self.factory.get_cache_manager()
    
    async def vector_search(self) -> IVectorSearch:
        return await self.factory.get_vector_search()
    
    async def pipeline_executor(self) -> IPipelineExecutor:
        return await self.factory.get_pipeline_executor()
    
    async def claude_code_service(self):
        return await self.factory.get_claude_code_service()


# Singleton factory instance
_service_factory = None


async def get_service_factory() -> ServiceFactory:
    """Get the global service factory."""
    global _service_factory
    if _service_factory is None:
        _service_factory = ServiceFactory()
        await _service_factory.initialize()
    return _service_factory


def create_service_container() -> ServiceContainer:
    """Create a new service container."""
    return ServiceContainer()


# Convenience functions for common operations
async def execute_pipeline(
    startup_idea: str,
    target_investor: InvestorType = InvestorType.SEED,
    generate_mvp: bool = True,
    max_total_budget: float = 60.0
) -> Dict[str, Any]:
    """Execute the complete pipeline with service dependency injection."""
    async with create_service_container() as container:
        pipeline = await container.pipeline_executor()
        return await pipeline.execute_full_pipeline(
            startup_idea=startup_idea,
            target_investor=target_investor,
            generate_mvp=generate_mvp,
            max_total_budget=max_total_budget
        )


async def search_similar_ideas(
    query_text: str,
    threshold: float = 0.7,
    limit: int = 10
) -> List[Any]:
    """Search for similar ideas using the vector search service."""
    async with create_service_container() as container:
        vector_search = await container.vector_search()
        return await vector_search.search_similar(
            query_text=query_text,
            threshold=threshold,
            limit=limit
        )


async def get_pipeline_health() -> Dict[str, Any]:
    """Get health status of all pipeline services."""
    factory = await get_service_factory()
    return await factory.health_check()