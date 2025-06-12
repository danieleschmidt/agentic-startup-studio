"""
Domain models for startup ideas with comprehensive validation.

This module defines Pydantic models for idea representation, validation,
and status management following the data pipeline requirements.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
import logging

logger = logging.getLogger(__name__)


class IdeaStatus(str, Enum):
    """Enumeration of possible idea statuses."""
    DRAFT = "DRAFT"
    VALIDATING = "VALIDATING"
    VALIDATED = "VALIDATED"
    REJECTED = "REJECTED"
    RESEARCHING = "RESEARCHING"
    BUILDING = "BUILDING"
    TESTING = "TESTING"
    DEPLOYED = "DEPLOYED"
    ARCHIVED = "ARCHIVED"


class PipelineStage(str, Enum):
    """Enumeration of pipeline processing stages."""
    IDEATE = "IDEATE"
    RESEARCH = "RESEARCH"
    DECK = "DECK"
    INVESTORS = "INVESTORS"
    MVP = "MVP"
    BUILDING = "BUILDING"
    SMOKE_TEST = "SMOKE_TEST"
    COMPLETE = "COMPLETE"


class IdeaCategory(str, Enum):
    """Predefined categories for startup ideas."""
    FINTECH = "fintech"
    HEALTHTECH = "healthtech"
    EDTECH = "edtech"
    SAAS = "saas"
    ECOMMERCE = "ecommerce"
    AI_ML = "ai_ml"
    BLOCKCHAIN = "blockchain"
    CONSUMER = "consumer"
    ENTERPRISE = "enterprise"
    MARKETPLACE = "marketplace"
    UNCATEGORIZED = "uncategorized"


class IdeaDraft(BaseModel):
    """Input model for creating new ideas with validation."""
    
    title: str = Field(
        ..., 
        min_length=10, 
        max_length=200,
        description="Concise idea title"
    )
    description: str = Field(
        ..., 
        min_length=10, 
        max_length=5000,
        description="Detailed idea description"
    )
    category: Optional[IdeaCategory] = Field(
        default=IdeaCategory.UNCATEGORIZED,
        description="Business category classification"
    )
    problem_statement: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Problem this idea solves"
    )
    solution_description: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="How the solution works"
    )
    target_market: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Target customer segment"
    )
    evidence_links: List[str] = Field(
        default_factory=list,
        description="Supporting evidence URLs"
    )

    @field_validator('title', 'description')
    @classmethod
    def validate_text_content(cls, v):
        """Validate text fields for basic security and quality."""
        if not v or not v.strip():
            raise ValueError("Text content cannot be empty or whitespace only")
        
        # Basic HTML/script injection prevention
        dangerous_patterns = ['<script', 'javascript:', 'data:', 'vbscript:']
        v_lower = v.lower()
        for pattern in dangerous_patterns:
            if pattern in v_lower:
                raise ValueError(f"Potentially dangerous content detected: {pattern}")
        
        return v.strip()

    @field_validator('evidence_links')
    @classmethod
    def validate_evidence_links(cls, v):
        """Validate evidence links are proper URLs."""
        if not v:
            return v
        
        validated_links = []
        for link in v:
            if not isinstance(link, str):
                continue
            link = link.strip()
            if link and (link.startswith('http://') or link.startswith('https://')):
                validated_links.append(link)
        
        return validated_links

    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True
    )


class Idea(BaseModel):
    """Complete idea entity with metadata and tracking."""
    
    idea_id: UUID = Field(default_factory=uuid4, description="Unique identifier")
    title: str = Field(..., min_length=10, max_length=200)
    description: str = Field(..., min_length=10, max_length=5000)
    category: IdeaCategory = Field(default=IdeaCategory.UNCATEGORIZED)
    status: IdeaStatus = Field(default=IdeaStatus.DRAFT)
    current_stage: PipelineStage = Field(default=PipelineStage.IDEATE)
    stage_progress: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Optional detailed fields
    problem_statement: Optional[str] = Field(default=None, max_length=1000)
    solution_description: Optional[str] = Field(default=None, max_length=1000)
    target_market: Optional[str] = Field(default=None, max_length=500)
    evidence_links: List[str] = Field(default_factory=list)
    
    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = Field(default=None, max_length=100)
    
    # Pipeline artifacts
    deck_path: Optional[str] = Field(default=None, max_length=500)
    research_data: Dict[str, Any] = Field(default_factory=dict)
    investor_scores: Dict[str, float] = Field(default_factory=dict)
    
    @field_validator('stage_progress')
    @classmethod
    def validate_progress(cls, v):
        """Ensure progress is consistent with stage."""
        if v < 0.0 or v > 1.0:
            raise ValueError("Progress must be between 0.0 and 1.0")
        return v

    @model_validator(mode='after')
    def validate_idea_consistency(self):
        """Validate cross-field consistency."""
        # Ensure status and stage are consistent
        if self.status == IdeaStatus.DRAFT and self.current_stage != PipelineStage.IDEATE:
            raise ValueError("Draft ideas must be in IDEATE stage")
        
        return self

    def update_progress(self, new_progress: float) -> None:
        """Update stage progress with validation."""
        if not 0.0 <= new_progress <= 1.0:
            raise ValueError("Progress must be between 0.0 and 1.0")
        
        self.stage_progress = new_progress
        self.updated_at = datetime.now(timezone.utc)

    def advance_stage(self, next_stage: PipelineStage) -> None:
        """Advance to next pipeline stage."""
        # Update status first to avoid validation conflicts
        if next_stage == PipelineStage.RESEARCH:
            self.status = IdeaStatus.RESEARCHING
        elif next_stage == PipelineStage.MVP:
            self.status = IdeaStatus.BUILDING
        elif next_stage == PipelineStage.SMOKE_TEST:
            self.status = IdeaStatus.TESTING
        elif next_stage == PipelineStage.COMPLETE:
            self.status = IdeaStatus.DEPLOYED
        
        # Then update stage and progress
        self.current_stage = next_stage
        self.stage_progress = 0.0
        self.updated_at = datetime.now(timezone.utc)
        
        logger.info(
            f"Idea {self.idea_id} advanced to stage {next_stage.value}",
            extra={
                "idea_id": str(self.idea_id),
                "stage": next_stage.value,
                "title": self.title
            }
        )

    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }
    )


class IdeaSummary(BaseModel):
    """Lightweight idea summary for listing operations."""
    
    id: UUID
    title: str
    status: IdeaStatus
    stage: PipelineStage
    created_at: datetime
    progress: float = Field(ge=0.0, le=1.0)
    
    model_config = ConfigDict(
        use_enum_values=True
    )


class ValidationResult(BaseModel):
    """Result of idea validation with errors and warnings."""
    
    is_valid: bool = Field(default=True)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    def has_errors(self) -> bool:
        """Check if validation found errors."""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if validation found warnings."""
        return len(self.warnings) > 0
    
    def add_error(self, error: str) -> None:
        """Add validation error."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str) -> None:
        """Add validation warning."""
        self.warnings.append(warning)


class DuplicateCheckResult(BaseModel):
    """Result of duplicate detection analysis."""
    
    found_similar: bool = Field(default=False)
    exact_matches: List[UUID] = Field(default_factory=list)
    similar_ideas: List[UUID] = Field(default_factory=list)
    similarity_scores: Dict[str, float] = Field(default_factory=dict)
    
    def get_top_matches(self, limit: int = 5) -> List[tuple[UUID, float]]:
        """Get top similar ideas by score."""
        sorted_matches = sorted(
            self.similarity_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [(UUID(idea_id), score) for idea_id, score in sorted_matches[:limit]]


class QueryParams(BaseModel):
    """Parameters for querying ideas with filters."""
    
    status_filter: Optional[List[IdeaStatus]] = Field(default=None)
    stage_filter: Optional[List[PipelineStage]] = Field(default=None)
    category_filter: Optional[List[IdeaCategory]] = Field(default=None)
    created_after: Optional[datetime] = Field(default=None)
    created_before: Optional[datetime] = Field(default=None)
    search_text: Optional[str] = Field(default=None, max_length=200)
    
    # Pagination
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
    
    # Sorting
    sort_by: str = Field(default="created_at")
    sort_desc: bool = Field(default=True)
    
    def has_similarity_filter(self) -> bool:
        """Check if query includes similarity-based filtering."""
        return self.search_text is not None

    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class AuditEntry(BaseModel):
    """Audit trail entry for idea changes."""
    
    entry_id: UUID = Field(default_factory=uuid4)
    idea_id: UUID
    action: str
    changes: Dict[str, Any] = Field(default_factory=dict)
    user_id: Optional[str] = Field(default=None)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: Optional[str] = Field(default=None)
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }
    )