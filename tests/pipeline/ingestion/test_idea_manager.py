"""
Comprehensive test suite for idea management orchestration.

Tests end-to-end workflow, duplicate detection, lifecycle management,
error handling, and integration with all dependencies using proper mocking.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock, call
from uuid import UUID, uuid4
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Any, Optional

from pipeline.ingestion.idea_manager import (
    DuplicateDetector, IdeaLifecycleManager, IdeaManager,
    IdeaManagementError, DuplicateIdeaError, ValidationError, StorageError,
    create_idea_manager
)
from pipeline.models.idea import (
    Idea, IdeaDraft, IdeaStatus, PipelineStage, IdeaCategory,
    QueryParams, ValidationResult, DuplicateCheckResult, IdeaSummary
)
from pipeline.config.settings import ValidationConfig
from pipeline.ingestion.validators import IdeaValidator
from pipeline.storage.idea_repository import IdeaRepository


class TestDuplicateDetector:
    """Test duplicate detection and similarity analysis."""
    
    @pytest.fixture
    def config(self) -> ValidationConfig:
        """Provide test validation configuration."""
        return ValidationConfig(
            similarity_threshold=0.8,
            title_fuzzy_threshold=0.7
        )
    
    @pytest.fixture
    def mock_repository(self):
        """Provide mock idea repository."""
        return Mock(spec=IdeaRepository)
    
    @pytest.fixture
    def detector(self, mock_repository, config) -> DuplicateDetector:
        """Provide DuplicateDetector with mocked dependencies."""
        return DuplicateDetector(mock_repository, config)
    
    @pytest.fixture
    def sample_draft(self) -> IdeaDraft:
        """Provide sample idea draft for testing."""
        return IdeaDraft(
            title="AI-powered productivity tool",
            description="Revolutionary solution using artificial intelligence to boost workplace productivity"
        )
    
    @pytest.mark.asyncio
    async def test_when_no_duplicates_then_returns_clean_result(self, detector, mock_repository, sample_draft):
        """Given no duplicate ideas, when checking duplicates, then returns clean result."""
        # Setup mocks - no exact matches, no similar ideas
        mock_repository.find_by_title_exact.return_value = []
        mock_repository.find_similar_by_embedding.return_value = []
        mock_repository.find_with_filters.return_value = []  # For fuzzy matching
        
        result = await detector.check_for_duplicates(sample_draft)
        
        assert result.found_similar is False
        assert result.exact_matches == []
        assert result.similar_ideas == []
        assert result.similarity_scores == {}
        
        # Verify repository calls
        mock_repository.find_by_title_exact.assert_called_once_with(sample_draft.title)
        mock_repository.find_similar_by_embedding.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_when_exact_title_match_then_returns_early_with_matches(self, detector, mock_repository, sample_draft):
        """Given exact title matches, when checking duplicates, then returns early with exact matches."""
        # Setup mocks - exact title matches found
        exact_match_ids = [uuid4(), uuid4()]
        mock_repository.find_by_title_exact.return_value = exact_match_ids
        
        result = await detector.check_for_duplicates(sample_draft)
        
        assert result.found_similar is True
        assert result.exact_matches == exact_match_ids
        
        # Should not call vector search when exact matches found
        mock_repository.find_similar_by_embedding.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_when_similar_ideas_found_then_includes_in_result(self, detector, mock_repository, sample_draft):
        """Given similar ideas via vector search, when checking duplicates, then includes in result."""
        # Setup mocks - no exact matches, but similar ideas
        mock_repository.find_by_title_exact.return_value = []
        
        similar_id1, similar_id2 = uuid4(), uuid4()
        similar_results = [
            (similar_id1, 0.85),
            (similar_id2, 0.82)
        ]
        mock_repository.find_similar_by_embedding.return_value = similar_results
        mock_repository.find_with_filters.return_value = []  # No fuzzy matches
        
        result = await detector.check_for_duplicates(sample_draft)
        
        assert result.found_similar is True
        assert result.similar_ideas == [similar_id1, similar_id2]
        assert result.similarity_scores[str(similar_id1)] == 0.85
        assert result.similarity_scores[str(similar_id2)] == 0.82
        
        # Verify vector search parameters
        mock_repository.find_similar_by_embedding.assert_called_once_with(
            description=sample_draft.description,
            threshold=0.8,  # config.similarity_threshold
            exclude_statuses=[IdeaStatus.ARCHIVED, IdeaStatus.REJECTED],
            limit=10
        )
    
    @pytest.mark.asyncio
    async def test_when_fuzzy_title_matches_then_includes_in_result(self, detector, mock_repository, config, sample_draft):
        """Given fuzzy title matches, when checking duplicates, then includes fuzzy matches."""
        # Setup mocks - no exact matches, no vector similarity
        mock_repository.find_by_title_exact.return_value = []
        mock_repository.find_similar_by_embedding.return_value = []
        
        # Mock fuzzy matching ideas
        fuzzy_idea1_id = uuid4()
        fuzzy_idea1 = Mock()
        fuzzy_idea1.idea_id = fuzzy_idea1_id
        fuzzy_idea1.title = "AI powered productivity solution"  # Similar words
        
        mock_repository.find_with_filters.return_value = [fuzzy_idea1]
        
        result = await detector.check_for_duplicates(sample_draft)
        
        assert result.found_similar is True
        assert fuzzy_idea1_id in result.similar_ideas
        assert str(fuzzy_idea1_id) in result.similarity_scores
        # Fuzzy score should be above threshold
        assert result.similarity_scores[str(fuzzy_idea1_id)] >= config.title_fuzzy_threshold
    
    @pytest.mark.asyncio
    async def test_when_short_title_then_skips_fuzzy_matching(self, detector, mock_repository):
        """Given short title, when checking duplicates, then skips fuzzy matching."""
        short_title_draft = IdeaDraft(
            title="Short title for testing",  # At least 10 characters
            description="Valid description here"
        )
        
        mock_repository.find_by_title_exact.return_value = []
        mock_repository.find_similar_by_embedding.return_value = []
        
        result = await detector.check_for_duplicates(short_title_draft)
        
        # Should skip fuzzy matching due to short title
        mock_repository.find_with_filters.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_when_exception_occurs_then_returns_empty_result(self, detector, mock_repository, sample_draft):
        """Given exception during duplicate check, when checking duplicates, then returns empty result."""
        # Setup mock to raise exception
        mock_repository.find_by_title_exact.side_effect = Exception("Database error")
        
        result = await detector.check_for_duplicates(sample_draft)
        
        # Should return empty result on error to allow processing to continue
        assert result.found_similar is False
        assert result.exact_matches == []
        assert result.similar_ideas == []


class TestIdeaLifecycleManager:
    """Test idea lifecycle operations and state transitions."""
    
    @pytest.fixture
    def mock_repository(self):
        """Provide mock idea repository."""
        return Mock(spec=IdeaRepository)
    
    @pytest.fixture
    def lifecycle_manager(self, mock_repository) -> IdeaLifecycleManager:
        """Provide IdeaLifecycleManager with mocked dependencies."""
        return IdeaLifecycleManager(mock_repository)
    
    @pytest.fixture
    def sample_idea(self) -> Idea:
        """Provide sample idea in IDEATE stage."""
        return Idea(
            idea_id=uuid4(),
            title="Test idea",
            description="Test description",
            status=IdeaStatus.DRAFT,
            current_stage=PipelineStage.IDEATE
        )
    
    @pytest.mark.asyncio
    async def test_when_valid_stage_transition_then_advances_idea(self, lifecycle_manager, mock_repository, sample_idea):
        """Given valid stage transition, when advancing stage, then updates idea successfully."""
        mock_repository.find_by_id.return_value = sample_idea
        mock_repository.update_idea = AsyncMock()
        
        result = await lifecycle_manager.advance_idea_stage(
            idea_id=sample_idea.idea_id,
            next_stage=PipelineStage.RESEARCH,
            user_id="test_user",
            correlation_id="test-correlation"
        )
        
        assert result is True
        
        # Verify idea was updated
        assert sample_idea.current_stage == PipelineStage.RESEARCH
        assert sample_idea.stage_progress == 0.0  # Reset on stage change
        assert sample_idea.status == IdeaStatus.RESEARCHING  # Updated based on stage
        
        mock_repository.update_idea.assert_called_once_with(
            sample_idea, "test_user", "test-correlation"
        )
    
    @pytest.mark.asyncio
    async def test_when_idea_not_found_then_raises_error(self, lifecycle_manager, mock_repository):
        """Given non-existent idea, when advancing stage, then raises IdeaManagementError."""
        mock_repository.find_by_id.return_value = None
        
        with pytest.raises(IdeaManagementError, match="not found"):
            await lifecycle_manager.advance_idea_stage(
                idea_id=uuid4(),
                next_stage=PipelineStage.RESEARCH
            )
    
    @pytest.mark.asyncio
    async def test_when_invalid_stage_transition_then_raises_error(self, lifecycle_manager, mock_repository, sample_idea):
        """Given invalid stage transition, when advancing stage, then raises IdeaManagementError."""
        mock_repository.find_by_id.return_value = sample_idea
        
        # Try to go from IDEATE directly to MVP (invalid transition)
        with pytest.raises(IdeaManagementError, match="Invalid stage transition"):
            await lifecycle_manager.advance_idea_stage(
                idea_id=sample_idea.idea_id,
                next_stage=PipelineStage.MVP
            )
    
    @pytest.mark.asyncio
    async def test_when_update_stage_progress_then_updates_idea(self, lifecycle_manager, mock_repository, sample_idea):
        """Given valid progress value, when updating stage progress, then updates idea."""
        mock_repository.find_by_id.return_value = sample_idea
        mock_repository.update_idea = AsyncMock()
        
        result = await lifecycle_manager.update_stage_progress(
            idea_id=sample_idea.idea_id,
            progress=0.6,
            user_id="test_user"
        )
        
        assert result is True
        assert sample_idea.stage_progress == 0.6
        mock_repository.update_idea.assert_called_once_with(sample_idea, "test_user")
    
    @pytest.mark.asyncio
    async def test_when_repository_error_then_raises_management_error(self, lifecycle_manager, mock_repository, sample_idea):
        """Given repository error, when advancing stage, then raises IdeaManagementError."""
        mock_repository.find_by_id.return_value = sample_idea
        mock_repository.update_idea = AsyncMock(side_effect=Exception("Database error"))
        
        with pytest.raises(IdeaManagementError, match="Stage advancement failed"):
            await lifecycle_manager.advance_idea_stage(
                idea_id=sample_idea.idea_id,
                next_stage=PipelineStage.RESEARCH
            )
    
    def test_when_valid_stage_transitions_then_allows_progression(self, lifecycle_manager):
        """Given valid stage transitions, when checking validity, then returns True."""
        # Test forward progression
        assert lifecycle_manager._is_valid_stage_transition(
            PipelineStage.IDEATE, PipelineStage.RESEARCH
        ) is True
        
        # Test backward progression (allowed for flexibility)
        assert lifecycle_manager._is_valid_stage_transition(
            PipelineStage.RESEARCH, PipelineStage.IDEATE
        ) is True
        
        # Test invalid jump
        assert lifecycle_manager._is_valid_stage_transition(
            PipelineStage.IDEATE, PipelineStage.MVP
        ) is False
    
    def test_when_get_status_for_stage_then_returns_correct_mapping(self, lifecycle_manager):
        """Given pipeline stage, when getting status, then returns correct status mapping."""
        assert lifecycle_manager._get_status_for_stage(PipelineStage.IDEATE) == IdeaStatus.DRAFT
        assert lifecycle_manager._get_status_for_stage(PipelineStage.RESEARCH) == IdeaStatus.RESEARCHING
        assert lifecycle_manager._get_status_for_stage(PipelineStage.BUILDING) == IdeaStatus.BUILDING
        assert lifecycle_manager._get_status_for_stage(PipelineStage.TESTING) == IdeaStatus.TESTING
        assert lifecycle_manager._get_status_for_stage(PipelineStage.COMPLETE) == IdeaStatus.DEPLOYED


class TestIdeaManager:
    """Test main idea management orchestrator."""
    
    @pytest.fixture
    def config(self) -> ValidationConfig:
        """Provide test validation configuration."""
        return ValidationConfig()
    
    @pytest.fixture
    def mock_repository(self):
        """Provide mock idea repository."""
        return Mock(spec=IdeaRepository)
    
    @pytest.fixture
    def mock_validator(self):
        """Provide mock idea validator."""
        return Mock(spec=IdeaValidator)
    
    @pytest.fixture
    def mock_duplicate_detector(self):
        """Provide mock duplicate detector."""
        return Mock(spec=DuplicateDetector)
    
    @pytest.fixture
    def mock_lifecycle_manager(self):
        """Provide mock lifecycle manager."""
        return Mock(spec=IdeaLifecycleManager)
    
    @pytest.fixture
    def idea_manager(self, mock_repository, mock_validator, config, mock_duplicate_detector, mock_lifecycle_manager):
        """Provide IdeaManager with mocked dependencies."""
        manager = IdeaManager(mock_repository, mock_validator, config)
        manager.duplicate_detector = mock_duplicate_detector
        manager.lifecycle_manager = mock_lifecycle_manager
        return manager
    
    @pytest.fixture
    def sample_raw_data(self) -> Dict[str, Any]:
        """Provide sample raw data for idea creation."""
        return {
            "title": "AI-powered productivity tool",
            "description": "Revolutionary solution using AI to boost productivity",
            "category": "ai_ml"
        }
    
    @pytest.fixture
    def sample_draft(self) -> IdeaDraft:
        """Provide sample validated draft."""
        return IdeaDraft(
            title="AI-powered productivity tool",
            description="Revolutionary solution using AI to boost productivity",
            category=IdeaCategory.AI_ML
        )
    
    @pytest.mark.asyncio
    async def test_when_successful_creation_then_returns_idea_id_and_warnings(
        self, idea_manager, mock_validator, mock_duplicate_detector, 
        mock_repository, sample_raw_data, sample_draft
    ):
        """Given valid data without duplicates, when creating idea, then returns ID and warnings."""
        # Setup mocks
        validation_result = ValidationResult(is_valid=True)
        validation_result.add_warning("Minor formatting issue")
        mock_validator.validate_and_sanitize_draft.return_value = (sample_draft, validation_result)
        
        duplicate_result = DuplicateCheckResult(found_similar=False)
        mock_duplicate_detector.check_for_duplicates.return_value = duplicate_result
        
        test_idea_id = uuid4()
        mock_repository.save_idea.return_value = test_idea_id
        
        # Execute
        idea_id, warnings = await idea_manager.create_idea(
            raw_data=sample_raw_data,
            force_create=False,
            user_id="test_user",
            correlation_id="test-correlation"
        )
        
        # Verify
        assert idea_id == test_idea_id
        assert warnings == ["Minor formatting issue"]
        
        # Verify workflow steps
        mock_validator.validate_and_sanitize_draft.assert_called_once_with(sample_raw_data)
        mock_duplicate_detector.check_for_duplicates.assert_called_once_with(sample_draft)
        mock_repository.save_idea.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_when_validation_fails_then_raises_validation_error(
        self, idea_manager, mock_validator, sample_raw_data
    ):
        """Given validation failure, when creating idea, then raises ValidationError."""
        validation_result = ValidationResult(is_valid=False)
        validation_result.add_error("Title too short")
        mock_validator.validate_and_sanitize_draft.return_value = (None, validation_result)
        
        with pytest.raises(ValidationError, match="Validation failed"):
            await idea_manager.create_idea(sample_raw_data)
    
    @pytest.mark.asyncio
    async def test_when_duplicates_found_without_force_then_raises_duplicate_error(
        self, idea_manager, mock_validator, mock_duplicate_detector, 
        sample_raw_data, sample_draft
    ):
        """Given duplicates found without force flag, when creating idea, then raises DuplicateIdeaError."""
        # Setup mocks
        validation_result = ValidationResult(is_valid=True)
        mock_validator.validate_and_sanitize_draft.return_value = (sample_draft, validation_result)
        
        duplicate_result = DuplicateCheckResult(found_similar=True)
        duplicate_result.exact_matches = [uuid4()]
        duplicate_result.similar_ideas = [uuid4()]
        duplicate_result.similarity_scores = {str(uuid4()): 0.85}
        mock_duplicate_detector.check_for_duplicates.return_value = duplicate_result
        
        with pytest.raises(DuplicateIdeaError, match="Similar ideas found"):
            await idea_manager.create_idea(sample_raw_data, force_create=False)
    
    @pytest.mark.asyncio
    async def test_when_duplicates_found_with_force_then_creates_anyway(
        self, idea_manager, mock_validator, mock_duplicate_detector, 
        mock_repository, sample_raw_data, sample_draft
    ):
        """Given duplicates found with force flag, when creating idea, then creates anyway."""
        # Setup mocks
        validation_result = ValidationResult(is_valid=True)
        mock_validator.validate_and_sanitize_draft.return_value = (sample_draft, validation_result)
        
        duplicate_result = DuplicateCheckResult(found_similar=True)
        mock_duplicate_detector.check_for_duplicates.return_value = duplicate_result
        
        test_idea_id = uuid4()
        mock_repository.save_idea.return_value = test_idea_id
        
        # Execute with force flag
        idea_id, warnings = await idea_manager.create_idea(
            sample_raw_data, force_create=True
        )
        
        # Should succeed despite duplicates
        assert idea_id == test_idea_id
        mock_repository.save_idea.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_when_repository_error_then_raises_storage_error(
        self, idea_manager, mock_validator, mock_duplicate_detector, 
        mock_repository, sample_raw_data, sample_draft
    ):
        """Given repository error, when creating idea, then raises StorageError."""
        # Setup mocks
        validation_result = ValidationResult(is_valid=True)
        mock_validator.validate_and_sanitize_draft.return_value = (sample_draft, validation_result)
        
        duplicate_result = DuplicateCheckResult(found_similar=False)
        mock_duplicate_detector.check_for_duplicates.return_value = duplicate_result
        
        mock_repository.save_idea.side_effect = Exception("Database connection failed")
        
        with pytest.raises(StorageError, match="Failed to create idea"):
            await idea_manager.create_idea(sample_raw_data)
    
    @pytest.mark.asyncio
    async def test_when_get_idea_then_returns_from_repository(self, idea_manager, mock_repository):
        """Given idea ID, when getting idea, then returns from repository."""
        test_id = uuid4()
        test_idea = Mock()
        mock_repository.find_by_id.return_value = test_idea
        
        result = await idea_manager.get_idea(test_id)
        
        assert result == test_idea
        mock_repository.find_by_id.assert_called_once_with(test_id)
    
    @pytest.mark.asyncio
    async def test_when_list_ideas_then_returns_summaries(self, idea_manager, mock_repository):
        """Given filter parameters, when listing ideas, then returns idea summaries."""
        # Setup mock ideas
        test_idea = Mock()
        test_idea.idea_id = uuid4()
        test_idea.title = "Test Idea"
        test_idea.status = IdeaStatus.DRAFT
        test_idea.current_stage = PipelineStage.IDEATE
        test_idea.created_at = datetime.now(timezone.utc)
        test_idea.stage_progress = 0.3
        
        mock_repository.find_with_filters.return_value = [test_idea]
        
        # Mock progress calculation
        with patch.object(idea_manager, '_calculate_overall_progress', return_value=0.3):
            result = await idea_manager.list_ideas()
        
        assert len(result) == 1
        summary = result[0]
        assert isinstance(summary, IdeaSummary)
        assert summary.id == test_idea.idea_id
        assert summary.title == test_idea.title
        assert summary.progress == 0.3
    
    @pytest.mark.asyncio
    async def test_when_update_idea_in_locked_status_then_raises_error(self, idea_manager, mock_repository):
        """Given idea in locked status, when updating, then raises IdeaManagementError."""
        locked_idea = Mock()
        locked_idea.status = IdeaStatus.DEPLOYED  # Locked status
        mock_repository.find_by_id.return_value = locked_idea
        
        with pytest.raises(IdeaManagementError, match="Cannot modify idea in status"):
            await idea_manager.update_idea(uuid4(), {"title": "New title"})
    
    @pytest.mark.asyncio
    async def test_when_advance_stage_then_delegates_to_lifecycle_manager(
        self, idea_manager, mock_lifecycle_manager
    ):
        """Given stage advancement request, when advancing stage, then delegates to lifecycle manager."""
        test_id = uuid4()
        mock_lifecycle_manager.advance_idea_stage.return_value = True
        
        result = await idea_manager.advance_stage(
            idea_id=test_id,
            next_stage=PipelineStage.RESEARCH,
            user_id="test_user"
        )
        
        assert result is True
        mock_lifecycle_manager.advance_idea_stage.assert_called_once_with(
            test_id, PipelineStage.RESEARCH, "test_user", None
        )


class TestCreateIdeaManager:
    """Test idea manager factory function."""
    
    @pytest.mark.asyncio
    async def test_when_create_idea_manager_then_initializes_all_components(self):
        """Given factory call, when creating idea manager, then initializes all components."""
        with patch('pipeline.ingestion.idea_manager.create_idea_repository') as mock_create_repo, \
             patch('pipeline.ingestion.idea_manager.create_validator') as mock_create_validator, \
             patch('pipeline.ingestion.idea_manager.get_validation_config') as mock_get_config:
            
            mock_repo = Mock()
            mock_validator = Mock()
            mock_config = Mock()
            
            mock_create_repo.return_value = mock_repo
            mock_create_validator.return_value = mock_validator
            mock_get_config.return_value = mock_config
            
            result = await create_idea_manager()
            
            assert isinstance(result, IdeaManager)
            assert result.repository == mock_repo
            assert result.validator == mock_validator
            assert result.config == mock_config
            
            # Verify factory functions were called
            mock_create_repo.assert_called_once()
            mock_create_validator.assert_called_once()
            mock_get_config.assert_called_once()


class TestIdeaManagerIntegrationScenarios:
    """Test complex integration scenarios with multiple components."""
    
    @pytest.fixture
    def full_idea_manager(self):
        """Provide IdeaManager with real sub-components but mocked repository."""
        config = ValidationConfig()
        mock_repository = Mock(spec=IdeaRepository)
        mock_validator = Mock(spec=IdeaValidator)
        
        manager = IdeaManager(mock_repository, mock_validator, config)
        # Use real sub-components for integration testing
        manager.duplicate_detector = DuplicateDetector(mock_repository, config)
        manager.lifecycle_manager = IdeaLifecycleManager(mock_repository)
        return manager, mock_repository, mock_validator
    
    @pytest.mark.asyncio
    async def test_when_complete_workflow_then_all_steps_executed(self, full_idea_manager):
        """Given complete workflow, when creating idea, then executes all steps in order."""
        manager, mock_repository, mock_validator = full_idea_manager
        
        # Setup complex workflow
        sample_draft = IdeaDraft(
            title="Complex AI productivity tool",
            description="Advanced solution for enterprise productivity"
        )
        
        validation_result = ValidationResult(is_valid=True)
        validation_result.add_warning("Consider adding more evidence")
        mock_validator.validate_and_sanitize_draft.return_value = (sample_draft, validation_result)
        
        # No duplicates found
        mock_repository.find_by_title_exact.return_value = []
        mock_repository.find_similar_by_embedding.return_value = []
        mock_repository.find_with_filters.return_value = []
        
        test_idea_id = uuid4()
        mock_repository.save_idea.return_value = test_idea_id
        
        # Execute complete workflow
        idea_id, warnings = await manager.create_idea(
            raw_data={
                "title": "Complex AI productivity tool",
                "description": "Advanced solution for enterprise productivity",
                "category": "ai_ml"
            },
            user_id="integration_test_user",
            correlation_id="integration-test-123"
        )
        
        # Verify complete workflow
        assert idea_id == test_idea_id
        assert "Consider adding more evidence" in warnings
        
        # Verify all repository interactions
        mock_repository.find_by_title_exact.assert_called_once()
        mock_repository.find_similar_by_embedding.assert_called_once()
        mock_repository.save_idea.assert_called_once()
        
        # Verify idea was properly constructed
        save_call = mock_repository.save_idea.call_args[0][0]
        assert save_call.title == sample_draft.title
        assert save_call.description == sample_draft.description
        assert save_call.created_by == "integration_test_user"