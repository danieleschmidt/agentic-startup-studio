"""
Comprehensive test suite for idea domain models.

Tests Pydantic model validation, serialization/deserialization,
lifecycle state transitions, and audit trail functionality.
"""

import pytest
from datetime import datetime, timezone
from uuid import UUID, uuid4
from typing import Dict, Any
from pydantic import ValidationError

from pipeline.models.idea import (
    IdeaDraft, Idea, IdeaStatus, PipelineStage, IdeaCategory,
    ValidationResult, DuplicateCheckResult, QueryParams, AuditEntry, IdeaSummary
)


class TestIdeaCategory:
    """Test IdeaCategory enum validation and usage."""
    
    def test_when_valid_category_then_creates_enum(self):
        """Given valid category string, when creating enum, then succeeds."""
        category = IdeaCategory.FINTECH
        assert category.value == "fintech"
    
    def test_when_invalid_category_then_raises_error(self):
        """Given invalid category, when creating enum, then raises ValueError."""
        with pytest.raises(ValueError):
            IdeaCategory("invalid_category")
    
    def test_when_comparing_categories_then_equality_works(self):
        """Given two category instances, when comparing, then equality works."""
        assert IdeaCategory.FINTECH == IdeaCategory.FINTECH
        assert IdeaCategory.FINTECH != IdeaCategory.HEALTHTECH


class TestIdeaStatus:
    """Test IdeaStatus enum validation and transitions."""
    
    def test_when_valid_status_then_creates_enum(self):
        """Given valid status string, when creating enum, then succeeds."""
        status = IdeaStatus.DRAFT
        assert status.value == "DRAFT"
    
    def test_when_all_statuses_defined_then_complete_lifecycle(self):
        """Given all status values, when checking coverage, then complete lifecycle exists."""
        expected_statuses = {
            "DRAFT", "VALIDATING", "VALIDATED", "REJECTED", 
            "RESEARCHING", "BUILDING", "TESTING", "DEPLOYED", "ARCHIVED"
        }
        actual_statuses = {status.value for status in IdeaStatus}
        assert actual_statuses == expected_statuses


class TestPipelineStage:
    """Test PipelineStage enum validation and progression."""
    
    def test_when_valid_stage_then_creates_enum(self):
        """Given valid stage string, when creating enum, then succeeds."""
        stage = PipelineStage.IDEATE
        assert stage.value == "IDEATE"
    
    def test_when_all_stages_defined_then_complete_pipeline(self):
        """Given all stage values, when checking coverage, then complete pipeline exists."""
        expected_stages = {
            "IDEATE", "RESEARCH", "DECK", "INVESTORS", "MVP", "BUILDING", "SMOKE_TEST", "COMPLETE"
        }
        actual_stages = {stage.value for stage in PipelineStage}
        assert actual_stages == expected_stages


class TestIdeaDraft:
    """Test IdeaDraft model validation and sanitization."""
    
    @pytest.fixture
    def valid_draft_data(self) -> Dict[str, Any]:
        """Provide valid draft data for testing."""
        return {
            "title": "Revolutionary AI-powered productivity tool",
            "description": "An innovative solution that leverages artificial intelligence to boost workplace productivity by 50%",
            "category": IdeaCategory.AI_ML,
            "problem_statement": "Current productivity tools lack intelligent automation",
            "solution_description": "AI agent that learns user patterns and automates routine tasks",
            "target_market": "Knowledge workers in medium to large enterprises",
            "evidence_links": ["https://example.com/research1", "https://example.com/research2"]
        }
    
    def test_when_valid_data_then_creates_draft(self, valid_draft_data):
        """Given valid draft data, when creating IdeaDraft, then succeeds."""
        draft = IdeaDraft(**valid_draft_data)
        
        assert draft.title == valid_draft_data["title"]
        assert draft.description == valid_draft_data["description"]
        assert draft.category == valid_draft_data["category"]
        assert draft.problem_statement == valid_draft_data["problem_statement"]
        assert draft.solution_description == valid_draft_data["solution_description"]
        assert draft.target_market == valid_draft_data["target_market"]
        assert draft.evidence_links == valid_draft_data["evidence_links"]
    
    def test_when_minimal_data_then_creates_draft_with_defaults(self):
        """Given minimal required data, when creating IdeaDraft, then uses defaults."""
        draft = IdeaDraft(
            title="Test idea title here",
            description="Test description here"
        )
        
        assert draft.title == "Test idea title here"
        assert draft.description == "Test description here"
        assert draft.category == IdeaCategory.UNCATEGORIZED
        assert draft.problem_statement is None
        assert draft.solution_description is None
        assert draft.target_market is None
        assert draft.evidence_links == []
    
    def test_when_title_too_short_then_validation_fails(self):
        """Given title shorter than 10 chars, when creating IdeaDraft, then raises ValidationError."""
        with pytest.raises(ValidationError, match="String should have at least 10 characters"):
            IdeaDraft(title="Short", description="Valid description here")
    
    def test_when_title_too_long_then_validation_fails(self):
        """Given title longer than 200 chars, when creating IdeaDraft, then raises ValidationError."""
        long_title = "x" * 201
        with pytest.raises(ValidationError, match="String should have at most"):
            IdeaDraft(title=long_title, description="Valid description here")
    
    def test_when_description_too_short_then_validation_fails(self):
        """Given description shorter than 10 chars, when creating IdeaDraft, then raises ValidationError."""
        with pytest.raises(ValidationError, match="String should have at least 10 characters"):
            IdeaDraft(title="Valid title here", description="Short")
    
    def test_when_description_too_long_then_validation_fails(self):
        """Given description longer than 5000 chars, when creating IdeaDraft, then raises ValidationError."""
        long_desc = "x" * 5001
        with pytest.raises(ValidationError, match="String should have at most"):
            IdeaDraft(title="Valid title here", description=long_desc)
    
    def test_when_dangerous_script_in_title_then_sanitizes(self):
        """Given title with script tag, when creating IdeaDraft, then removes dangerous content."""
        with pytest.raises(ValueError, match="dangerous content"):
            IdeaDraft(
                title="<script>alert('xss')</script>Valid title",
                description="Valid description here"
            )
    
    def test_when_dangerous_javascript_url_then_sanitizes(self):
        """Given title with javascript protocol, when creating IdeaDraft, then removes dangerous content."""
        with pytest.raises(ValueError, match="dangerous content"):
            IdeaDraft(
                title="javascript:alert('xss') Valid title here",
                description="Valid description here"
            )
    
    def test_when_invalid_urls_in_evidence_then_filters_out(self):
        """Given invalid URLs in evidence links, when creating IdeaDraft, then filters them."""
        draft = IdeaDraft(
            title="Valid title here",
            description="Valid description here",
            evidence_links=["https://valid.com", "invalid-url", "ftp://invalid.com"]
        )
        
        assert draft.evidence_links == ["https://valid.com"]
    
    def test_when_whitespace_only_title_then_validation_fails(self):
        """Given whitespace-only title, when creating IdeaDraft, then raises ValidationError."""
        with pytest.raises(ValidationError, match="String should have at least 10 characters"):
            IdeaDraft(title="   ", description="Valid description here")


class TestIdea:
    """Test complete Idea model with metadata and lifecycle management."""
    
    @pytest.fixture
    def valid_idea_data(self) -> Dict[str, Any]:
        """Provide valid idea data for testing."""
        return {
            "title": "Revolutionary AI-powered productivity tool",
            "description": "An innovative solution that leverages artificial intelligence to boost workplace productivity by 50%",
            "category": IdeaCategory.AI_ML,
            "status": IdeaStatus.DRAFT,
            "current_stage": PipelineStage.IDEATE,
            "stage_progress": 0.0,
            "problem_statement": "Current productivity tools lack intelligent automation",
            "solution_description": "AI agent that learns user patterns and automates routine tasks",
            "target_market": "Knowledge workers in medium to large enterprises",
            "evidence_links": ["https://example.com/research1"],
            "created_by": "test_user"
        }
    
    def test_when_valid_data_then_creates_idea(self, valid_idea_data):
        """Given valid idea data, when creating Idea, then succeeds with defaults."""
        idea = Idea(**valid_idea_data)
        
        assert isinstance(idea.idea_id, UUID)
        assert idea.title == valid_idea_data["title"]
        assert idea.description == valid_idea_data["description"]
        assert idea.category == valid_idea_data["category"]
        assert idea.status == valid_idea_data["status"]
        assert idea.current_stage == valid_idea_data["current_stage"]
        assert idea.stage_progress == 0.0
        assert isinstance(idea.created_at, datetime)
        assert isinstance(idea.updated_at, datetime)
        assert idea.research_data == {}
        assert idea.investor_scores == {}
    
    def test_when_minimal_data_then_creates_idea_with_defaults(self):
        """Given minimal required data, when creating Idea, then uses appropriate defaults."""
        idea = Idea(
            title="Test idea title here",
            description="Test description here"
        )
        
        assert idea.category == IdeaCategory.UNCATEGORIZED
        assert idea.status == IdeaStatus.DRAFT
        assert idea.current_stage == PipelineStage.IDEATE
        assert idea.stage_progress == 0.0
        assert idea.evidence_links == []
        assert idea.research_data == {}
        assert idea.investor_scores == {}
    
    def test_when_progress_negative_then_validation_fails(self):
        """Given negative progress value, when creating Idea, then raises ValidationError."""
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            Idea(
                title="Valid title here",
                description="Valid description here",
                stage_progress=-0.1
            )
    
    def test_when_progress_greater_than_one_then_validation_fails(self):
        """Given progress > 1.0, when creating Idea, then raises ValidationError."""
        with pytest.raises(ValueError, match="less than or equal to 1"):
            Idea(
                title="Valid title here",
                description="Valid description here",
                stage_progress=1.1
            )
    
    def test_when_draft_status_with_wrong_stage_then_validation_fails(self):
        """Given DRAFT status with non-IDEATE stage, when creating Idea, then raises ValidationError."""
        with pytest.raises(ValueError, match="Draft ideas must be in IDEATE stage"):
            Idea(
                title="Valid title here",
                description="Valid description here",
                status=IdeaStatus.DRAFT,
                current_stage=PipelineStage.RESEARCH
            )
    
    def test_when_update_progress_valid_then_succeeds(self):
        """Given valid progress value, when calling update_progress, then updates correctly."""
        idea = Idea(title="Test title", description="Test description")
        original_updated_at = idea.updated_at
        
        idea.update_progress(0.5)
        
        assert idea.stage_progress == 0.5
        assert idea.updated_at > original_updated_at
    
    def test_when_update_progress_invalid_then_raises_error(self):
        """Given invalid progress value, when calling update_progress, then raises ValueError."""
        idea = Idea(title="Test title", description="Test description")
        
        with pytest.raises(ValueError, match="Progress must be between 0.0 and 1.0"):
            idea.update_progress(1.5)
    
    def test_when_advance_stage_then_resets_progress_and_updates_timestamp(self):
        """Given new stage, when calling advance_stage, then resets progress and updates timestamp."""
        idea = Idea(
            title="Test title", 
            description="Test description",
            stage_progress=0.8
        )
        original_updated_at = idea.updated_at
        
        idea.advance_stage(PipelineStage.RESEARCH)
        
        assert idea.current_stage == PipelineStage.RESEARCH
        assert idea.stage_progress == 0.0
        assert idea.updated_at > original_updated_at
    
    def test_when_serialize_to_json_then_proper_format(self):
        """Given Idea instance, when serializing to JSON, then uses proper encoders."""
        idea = Idea(title="Test title", description="Test description")
        
        json_data = idea.json()
        parsed = idea.parse_raw(json_data)
        
        assert parsed.idea_id == idea.idea_id
        assert parsed.title == idea.title
        assert parsed.created_at == idea.created_at


class TestValidationResult:
    """Test ValidationResult for tracking validation errors and warnings."""
    
    def test_when_created_then_defaults_to_valid(self):
        """Given new ValidationResult, when checking state, then defaults to valid."""
        result = ValidationResult()
        
        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []
        assert not result.has_errors()
        assert not result.has_warnings()
    
    def test_when_add_error_then_becomes_invalid(self):
        """Given ValidationResult, when adding error, then becomes invalid."""
        result = ValidationResult()
        
        result.add_error("Test error message")
        
        assert result.is_valid is False
        assert result.errors == ["Test error message"]
        assert result.has_errors()
    
    def test_when_add_warning_then_remains_valid(self):
        """Given ValidationResult, when adding warning, then remains valid."""
        result = ValidationResult()
        
        result.add_warning("Test warning message")
        
        assert result.is_valid is True
        assert result.warnings == ["Test warning message"]
        assert result.has_warnings()
    
    def test_when_multiple_errors_then_accumulates(self):
        """Given ValidationResult, when adding multiple errors, then accumulates all."""
        result = ValidationResult()
        
        result.add_error("First error")
        result.add_error("Second error")
        
        assert len(result.errors) == 2
        assert "First error" in result.errors
        assert "Second error" in result.errors


class TestDuplicateCheckResult:
    """Test DuplicateCheckResult for tracking similarity detection."""
    
    def test_when_created_then_defaults_to_no_duplicates(self):
        """Given new DuplicateCheckResult, when checking state, then defaults to no duplicates."""
        result = DuplicateCheckResult()
        
        assert result.found_similar is False
        assert result.exact_matches == []
        assert result.similar_ideas == []
        assert result.similarity_scores == {}
    
    def test_when_get_top_matches_then_returns_sorted_by_score(self):
        """Given similarity scores, when calling get_top_matches, then returns sorted by score."""
        result = DuplicateCheckResult(
            found_similar=True,
            similarity_scores={
                str(uuid4()): 0.8,
                str(uuid4()): 0.9,
                str(uuid4()): 0.7
            }
        )
        
        top_matches = result.get_top_matches(limit=2)
        
        assert len(top_matches) == 2
        assert top_matches[0][1] == 0.9  # Highest score first
        assert top_matches[1][1] == 0.8  # Second highest
        assert all(isinstance(match[0], UUID) for match in top_matches)


class TestQueryParams:
    """Test QueryParams for filtering and pagination."""
    
    def test_when_created_with_defaults_then_proper_values(self):
        """Given QueryParams with defaults, when checking values, then uses proper defaults."""
        params = QueryParams()
        
        assert params.status_filter is None
        assert params.stage_filter is None
        assert params.category_filter is None
        assert params.created_after is None
        assert params.created_before is None
        assert params.search_text is None
        assert params.limit == 20
        assert params.offset == 0
        assert params.sort_by == "created_at"
        assert params.sort_desc is True
    
    def test_when_limit_too_high_then_validation_fails(self):
        """Given limit > 100, when creating QueryParams, then raises ValidationError."""
        with pytest.raises(ValueError, match="less than or equal to 100"):
            QueryParams(limit=150)
    
    def test_when_limit_zero_then_validation_fails(self):
        """Given limit = 0, when creating QueryParams, then raises ValidationError."""
        with pytest.raises(ValueError, match="greater than or equal to 1"):
            QueryParams(limit=0)
    
    def test_when_negative_offset_then_validation_fails(self):
        """Given negative offset, when creating QueryParams, then raises ValidationError."""
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            QueryParams(offset=-1)
    
    def test_when_has_search_text_then_similarity_filter_true(self):
        """Given search text, when checking has_similarity_filter, then returns True."""
        params = QueryParams(search_text="AI productivity")
        assert params.has_similarity_filter() is True
    
    def test_when_no_search_text_then_similarity_filter_false(self):
        """Given no search text, when checking has_similarity_filter, then returns False."""
        params = QueryParams()
        assert params.has_similarity_filter() is False


class TestAuditEntry:
    """Test AuditEntry for tracking changes and audit trail."""
    
    def test_when_created_then_generates_id_and_timestamp(self):
        """Given new AuditEntry, when checking metadata, then generates ID and timestamp."""
        entry = AuditEntry(
            idea_id=uuid4(),
            action="test_action",
            changes={"field": "value"}
        )
        
        assert isinstance(entry.entry_id, UUID)
        assert isinstance(entry.timestamp, datetime)
        assert entry.idea_id is not None
        assert entry.action == "test_action"
        assert entry.changes == {"field": "value"}
    
    def test_when_serialize_to_json_then_proper_format(self):
        """Given AuditEntry, when serializing to JSON, then uses proper encoders."""
        entry = AuditEntry(
            idea_id=uuid4(),
            action="test_action"
        )
        
        json_data = entry.json()
        parsed = entry.parse_raw(json_data)
        
        assert parsed.entry_id == entry.entry_id
        assert parsed.idea_id == entry.idea_id
        assert parsed.timestamp == entry.timestamp


class TestIdeaSummary:
    """Test IdeaSummary for lightweight idea listings."""
    
    def test_when_created_then_validates_progress_range(self):
        """Given IdeaSummary with valid progress, when creating, then succeeds."""
        summary = IdeaSummary(
            id=uuid4(),
            title="Test Idea",
            status=IdeaStatus.DRAFT,
            stage=PipelineStage.IDEATE,
            created_at=datetime.now(timezone.utc),
            progress=0.5
        )
        
        assert summary.progress == 0.5
    
    def test_when_progress_out_of_range_then_validation_fails(self):
        """Given IdeaSummary with invalid progress, when creating, then raises ValidationError."""
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            IdeaSummary(
                id=uuid4(),
                title="Test Idea",
                status=IdeaStatus.DRAFT,
                stage=PipelineStage.IDEATE,
                created_at=datetime.now(timezone.utc),
                progress=-0.1
            )