"""
Service Interfaces - Abstract interfaces for pipeline services.

Defines contracts and interfaces for all pipeline services to ensure
proper separation of concerns and testability.
"""

from abc import abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pipeline.core.service_registry import ServiceInterface


class PipelineStage(Enum):
    """Pipeline execution stages."""
    IDEATE = "ideate"
    RESEARCH = "research"
    DECK_GENERATION = "deck_generation"
    INVESTOR_EVALUATION = "investor_evaluation"
    SMOKE_TEST = "smoke_test"
    MVP_GENERATION = "mvp_generation"
    COMPLETED = "completed"
    FAILED = "failed"


class BudgetCategory(Enum):
    """Budget allocation categories."""
    OPENAI_TOKENS = "openai_tokens"
    GOOGLE_ADS = "google_ads"
    INFRASTRUCTURE = "infrastructure"
    EXTERNAL_APIS = "external_apis"


class InvestorType(Enum):
    """Types of investors for pitch deck targeting."""
    SEED = "seed"
    SERIES_A = "series_a"
    SERIES_B = "series_b"
    VENTURE = "venture"
    ANGEL = "angel"


# Budget and Cost Management Interfaces

class IBudgetTracker(ServiceInterface):
    """Interface for budget tracking and cost management."""

    @abstractmethod
    async def track_cost(
        self,
        category: BudgetCategory,
        amount: float,
        operation: str,
        metadata: dict[str, Any] | None = None
    ) -> bool:
        """Track cost for a specific category."""
        pass

    @abstractmethod
    async def get_budget_status(self) -> dict[str, Any]:
        """Get current budget utilization across all categories."""
        pass

    @abstractmethod
    async def check_budget_availability(
        self,
        category: BudgetCategory,
        amount: float
    ) -> bool:
        """Check if budget is available for a specific amount."""
        pass

    @abstractmethod
    async def get_spending_history(
        self,
        category: BudgetCategory | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None
    ) -> list[dict[str, Any]]:
        """Get spending history with optional filters."""
        pass


# Research and Evidence Collection Interfaces

class IEvidenceCollector(ServiceInterface):
    """Interface for evidence collection and research."""

    @abstractmethod
    async def collect_evidence(
        self,
        claim: str,
        research_domains: list[Any],
        min_total_evidence: int = 5,
        timeout: int = 120
    ) -> dict[str, list[Any]]:
        """Collect evidence for a given claim across research domains."""
        pass

    @abstractmethod
    async def verify_citation(self, url: str) -> dict[str, Any]:
        """Verify if a citation URL is accessible and valid."""
        pass

    @abstractmethod
    async def score_evidence_quality(
        self,
        evidence: Any,
        criteria: dict[str, float]
    ) -> float:
        """Score evidence quality based on criteria."""
        pass


# Content Generation Interfaces

class IPitchDeckGenerator(ServiceInterface):
    """Interface for pitch deck generation."""

    @abstractmethod
    async def generate_pitch_deck(
        self,
        startup_idea: str,
        evidence_by_domain: dict[str, list[Any]],
        target_investor: InvestorType,
        max_cost: float = 8.0
    ) -> Any:
        """Generate a complete pitch deck."""
        pass

    @abstractmethod
    async def generate_slide(
        self,
        slide_type: str,
        content_data: dict[str, Any],
        template_config: dict[str, Any] | None = None
    ) -> Any:
        """Generate a single slide."""
        pass

    @abstractmethod
    async def validate_pitch_deck(self, pitch_deck: Any) -> dict[str, Any]:
        """Validate generated pitch deck quality."""
        pass


class ICampaignGenerator(ServiceInterface):
    """Interface for marketing campaign generation."""

    @abstractmethod
    async def generate_smoke_test_campaign(
        self,
        pitch_deck: Any,
        budget_limit: float,
        duration_days: int
    ) -> Any:
        """Generate a smoke test marketing campaign."""
        pass

    @abstractmethod
    async def execute_campaign(self, campaign: Any) -> Any:
        """Execute a marketing campaign."""
        pass

    @abstractmethod
    async def generate_mvp(
        self,
        mvp_request: Any,
        max_cost: float = 4.0
    ) -> Any:
        """Generate an MVP based on requirements."""
        pass

    @abstractmethod
    async def get_campaign_metrics(self, campaign_id: str) -> dict[str, Any]:
        """Get campaign performance metrics."""
        pass


# Workflow and Orchestration Interfaces

class IWorkflowOrchestrator(ServiceInterface):
    """Interface for workflow orchestration."""

    @abstractmethod
    async def execute_workflow(
        self,
        idea_id: str,
        idea_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute complete workflow for an idea."""
        pass

    @abstractmethod
    async def resume_workflow(self, idea_id: str) -> dict[str, Any] | None:
        """Resume workflow from last checkpoint."""
        pass

    @abstractmethod
    async def get_workflow_status(self, idea_id: str) -> dict[str, Any]:
        """Get current workflow status."""
        pass

    @abstractmethod
    async def cancel_workflow(self, idea_id: str) -> bool:
        """Cancel a running workflow."""
        pass


# Data Management Interfaces

class IIdeaRepository(ServiceInterface):
    """Interface for idea data persistence."""

    @abstractmethod
    async def save_idea(self, idea: Any, correlation_id: str | None = None) -> UUID:
        """Save a new idea."""
        pass

    @abstractmethod
    async def get_idea(self, idea_id: UUID) -> Any | None:
        """Get idea by ID."""
        pass

    @abstractmethod
    async def update_idea(
        self,
        idea_id: UUID,
        updates: dict[str, Any],
        correlation_id: str | None = None
    ) -> bool:
        """Update an existing idea."""
        pass

    @abstractmethod
    async def find_similar_ideas(
        self,
        description: str,
        threshold: float = 0.8,
        limit: int = 10
    ) -> list[Any]:
        """Find similar ideas using vector search."""
        pass

    @abstractmethod
    async def get_ideas_by_status(self, status: str) -> list[Any]:
        """Get ideas filtered by status."""
        pass


class ICacheManager(ServiceInterface):
    """Interface for caching operations."""

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """Get value from cache."""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> bool:
        """Set value in cache with TTL."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass

    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries."""
        pass


class IVectorSearch(ServiceInterface):
    """Interface for vector similarity search."""

    @abstractmethod
    async def search_similar(
        self,
        query_text: str,
        threshold: float | None = None,
        limit: int | None = None,
        exclude_ids: list[UUID] | None = None,
        filters: dict[str, Any] | None = None,
        include_metadata: bool = False
    ) -> list[Any]:
        """Search for similar items."""
        pass

    @abstractmethod
    async def search_batch(self, queries: list[dict[str, Any]]) -> list[list[Any]]:
        """Search for multiple queries in batch."""
        pass

    @abstractmethod
    async def index_document(
        self,
        document_id: str,
        text: str,
        metadata: dict[str, Any] | None = None
    ) -> bool:
        """Index a document for search."""
        pass


# Validation and Quality Control Interfaces

class IValidator(ServiceInterface):
    """Interface for data validation."""

    @abstractmethod
    async def validate_startup_idea(self, idea_data: dict[str, Any]) -> dict[str, Any]:
        """Validate startup idea data."""
        pass

    @abstractmethod
    async def validate_pitch_deck_content(self, content: dict[str, Any]) -> dict[str, Any]:
        """Validate pitch deck content."""
        pass

    @abstractmethod
    async def validate_campaign_data(self, campaign_data: dict[str, Any]) -> dict[str, Any]:
        """Validate campaign data."""
        pass


class IQualityGate(ServiceInterface):
    """Interface for quality gate validation."""

    @abstractmethod
    async def evaluate_quality(
        self,
        stage: PipelineStage,
        data: dict[str, Any],
        criteria: dict[str, Any]
    ) -> dict[str, Any]:
        """Evaluate quality for a pipeline stage."""
        pass

    @abstractmethod
    async def get_quality_thresholds(self, stage: PipelineStage) -> dict[str, float]:
        """Get quality thresholds for a stage."""
        pass


# Monitoring and Observability Interfaces

class IMetricsCollector(ServiceInterface):
    """Interface for metrics collection."""

    @abstractmethod
    async def record_metric(
        self,
        metric_name: str,
        value: int | float,
        tags: dict[str, str] | None = None,
        timestamp: datetime | None = None
    ) -> None:
        """Record a metric value."""
        pass

    @abstractmethod
    async def record_event(
        self,
        event_name: str,
        properties: dict[str, Any] | None = None,
        timestamp: datetime | None = None
    ) -> None:
        """Record an event."""
        pass

    @abstractmethod
    async def get_metrics(
        self,
        metric_name: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        tags: dict[str, str] | None = None
    ) -> list[dict[str, Any]]:
        """Get metrics data."""
        pass


class IHealthChecker(ServiceInterface):
    """Interface for health checking services."""

    @abstractmethod
    async def check_service_health(self, service_name: str) -> dict[str, Any]:
        """Check health of a specific service."""
        pass

    @abstractmethod
    async def check_all_services(self) -> dict[str, dict[str, Any]]:
        """Check health of all services."""
        pass

    @abstractmethod
    async def register_health_check(
        self,
        service_name: str,
        check_function: Any
    ) -> None:
        """Register a custom health check."""
        pass


# Configuration and Settings Interfaces

class IConfigurationManager(ServiceInterface):
    """Interface for configuration management."""

    @abstractmethod
    async def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        pass

    @abstractmethod
    async def set_config(self, key: str, value: Any) -> bool:
        """Set configuration value."""
        pass

    @abstractmethod
    async def reload_config(self) -> None:
        """Reload configuration from source."""
        pass

    @abstractmethod
    async def get_all_config(self) -> dict[str, Any]:
        """Get all configuration values."""
        pass


# Security Interfaces

class ISecurityManager(ServiceInterface):
    """Interface for security operations."""

    @abstractmethod
    async def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        pass

    @abstractmethod
    async def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        pass

    @abstractmethod
    async def validate_api_key(self, api_key: str, service: str) -> bool:
        """Validate API key for a service."""
        pass

    @abstractmethod
    async def get_secret(self, secret_name: str, required: bool = False) -> str | None:
        """Get secret value from secure storage."""
        pass


# Pipeline Execution Interface

class IPipelineExecutor(ServiceInterface):
    """Interface for complete pipeline execution."""

    @abstractmethod
    async def execute_full_pipeline(
        self,
        startup_idea: str,
        target_investor: InvestorType = InvestorType.SEED,
        generate_mvp: bool = True,
        max_total_budget: float = 60.0
    ) -> dict[str, Any]:
        """Execute the complete pipeline."""
        pass

    @abstractmethod
    async def execute_pipeline_stage(
        self,
        stage: PipelineStage,
        context: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a specific pipeline stage."""
        pass

    @abstractmethod
    async def get_pipeline_status(self, execution_id: str) -> dict[str, Any]:
        """Get pipeline execution status."""
        pass
