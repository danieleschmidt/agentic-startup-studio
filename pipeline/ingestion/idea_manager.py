"""
Idea management orchestrator for startup idea ingestion.

This module coordinates validation, deduplication, storage, and business
logic for the complete idea lifecycle management.
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID, uuid4

from pipeline.models.idea import (
    Idea, IdeaDraft, IdeaStatus, PipelineStage, IdeaSummary,
    QueryParams, ValidationResult, DuplicateCheckResult
)
from pipeline.config.settings import get_validation_config, ValidationConfig
from pipeline.ingestion.validators import IdeaValidator, create_validator
from pipeline.storage.idea_repository import IdeaRepository, create_idea_repository

logger = logging.getLogger(__name__)


class IdeaManagementError(Exception):
    """Base exception for idea management operations."""
    pass


class DuplicateIdeaError(IdeaManagementError):
    """Raised when duplicate ideas are detected and not forced."""
    pass


class ValidationError(IdeaManagementError):
    """Raised when idea validation fails."""
    pass


class StorageError(IdeaManagementError):
    """Raised when storage operations fail."""
    pass


class DuplicateDetector:
    """Handles duplicate detection and similarity analysis."""
    
    def __init__(self, repository: IdeaRepository, config: ValidationConfig):
        self.repository = repository
        self.config = config
    
    async def check_for_duplicates(self, draft: IdeaDraft) -> DuplicateCheckResult:
        """
        Comprehensive duplicate detection using exact and similarity matching.
        
        Args:
            draft: Idea draft to check for duplicates
            
        Returns:
            DuplicateCheckResult with found matches and scores
        """
        try:
            result = DuplicateCheckResult(found_similar=False)
            
            # Step 1: Check for exact title matches
            exact_matches = await self.repository.find_by_title_exact(draft.title)
            if exact_matches:
                result.found_similar = True
                result.exact_matches = exact_matches
                logger.warning(
                    f"Exact title matches found for idea: {draft.title}",
                    extra={"title": draft.title, "matches": len(exact_matches)}
                )
                return result
            
            # Step 2: Vector similarity search using description
            similar_ideas = await self.repository.find_similar_by_embedding(
                description=draft.description,
                threshold=self.config.similarity_threshold,
                exclude_statuses=[IdeaStatus.ARCHIVED, IdeaStatus.REJECTED],
                limit=10
            )
            
            if similar_ideas:
                result.found_similar = True
                result.similar_ideas = [idea_id for idea_id, score in similar_ideas]
                result.similarity_scores = {
                    str(idea_id): score for idea_id, score in similar_ideas
                }
                
                logger.info(
                    f"Similar ideas found using vector search",
                    extra={
                        "title": draft.title,
                        "similar_count": len(similar_ideas),
                        "max_similarity": max([score for _, score in similar_ideas]) if similar_ideas else 0
                    }
                )
            
            # Step 3: Fuzzy title matching as additional check
            await self._add_fuzzy_title_matches(draft, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Duplicate detection failed: {e}")
            # Return empty result on error to allow processing to continue
            return DuplicateCheckResult(found_similar=False)
    
    async def _add_fuzzy_title_matches(self, draft: IdeaDraft, result: DuplicateCheckResult) -> None:
        """Add fuzzy title matching results to duplicate check."""
        try:
            # Normalize and tokenize draft title
            draft_words = self._normalize_title_words(draft.title)
            
            if len(draft_words) < 2:
                return  # Skip fuzzy matching for very short titles
            
            # Get recent ideas for fuzzy comparison
            recent_params = QueryParams(
                limit=100,
                sort_by="created_at",
                sort_desc=True
            )
            recent_ideas = await self.repository.find_with_filters(recent_params)
            
            fuzzy_matches = []
            for idea in recent_ideas:
                if str(idea.idea_id) in [str(uid) for uid in result.exact_matches]:
                    continue  # Skip exact matches
                
                idea_words = self._normalize_title_words(idea.title)
                if len(idea_words) < 2:
                    continue
                
                # Calculate improved word overlap ratio with partial matching
                similarity_score = self._calculate_word_similarity(draft_words, idea_words)
                
                if similarity_score >= self.config.title_fuzzy_threshold:
                    fuzzy_matches.append((idea.idea_id, similarity_score))
            
            # Add fuzzy matches to result
            if fuzzy_matches:
                fuzzy_matches.sort(key=lambda x: x[1], reverse=True)
                for idea_id, score in fuzzy_matches[:5]:  # Top 5 fuzzy matches
                    if str(idea_id) not in result.similarity_scores:
                        result.similar_ideas.append(idea_id)
                        result.similarity_scores[str(idea_id)] = score
                        result.found_similar = True
                
                logger.debug(
                    f"Added {len(fuzzy_matches)} fuzzy title matches",
                    extra={
                        "draft_title": draft.title,
                        "matches": [(str(mid), score) for mid, score in fuzzy_matches]
                    }
                )
                
        except Exception as e:
            logger.warning(f"Fuzzy title matching failed: {e}")
    
    def _normalize_title_words(self, title: str) -> set:
        """Normalize title into comparable word tokens."""
        # Convert to lowercase and split on various separators
        import re
        # Split on spaces, hyphens, underscores, and other common separators
        words = re.split(r'[\s\-_]+', title.lower())
        
        # Filter out empty strings and very short words
        normalized_words = set()
        for word in words:
            cleaned = re.sub(r'[^\w]', '', word)  # Remove punctuation
            if len(cleaned) >= 2:  # Only keep words with 2+ characters
                normalized_words.add(cleaned)
        
        return normalized_words
    
    def _calculate_word_similarity(self, words1: set, words2: set) -> float:
        """Calculate similarity between two sets of words using Jaccard coefficient."""
        if not words1 or not words2:
            return 0.0
        
        # Calculate Jaccard coefficient: intersection / union
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
        
        jaccard_score = intersection / union
        
        # Boost score for partial word matches (e.g., "ai" matches "artificial")
        partial_match_score = 0.0
        for word1 in words1:
            for word2 in words2:
                if word1 != word2:  # Don't double-count exact matches
                    # Check if one word is a substring of another (min 3 chars)
                    if len(word1) >= 3 and len(word2) >= 3:
                        if word1 in word2 or word2 in word1:
                            partial_match_score += 0.1  # Small boost for partial matches
        
        # Combine Jaccard and partial match score, capping at 1.0
        combined_score = min(1.0, jaccard_score + partial_match_score)
        
        logger.debug(
            f"Calculated similarity: Jaccard={jaccard_score:.2f}, Partial={partial_match_score:.2f}, Combined={combined_score:.2f}",
            extra={"words1": list(words1), "words2": list(words2)}
        )
        
        return combined_score
        
        # Normalize partial matches and add to Jaccard score
        partial_score = partial_matches / max(len(words1), len(words2))
        
        return min(1.0, jaccard_score + partial_score)


class IdeaLifecycleManager:
    """Manages idea lifecycle operations and state transitions."""
    
    def __init__(self, repository: IdeaRepository):
        self.repository = repository
    
    async def advance_idea_stage(
        self, 
        idea_id: UUID, 
        next_stage: PipelineStage,
        user_id: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> bool:
        """
        Advance idea to next pipeline stage.
        
        Args:
            idea_id: Idea to advance
            next_stage: Target pipeline stage
            user_id: User making the change
            correlation_id: Optional correlation ID
            
        Returns:
            True if advancement successful
            
        Raises:
            IdeaManagementError: If advancement fails
        """
        try:
            idea = await self.repository.find_by_id(idea_id)
            if not idea:
                raise IdeaManagementError(f"Idea {idea_id} not found")
            
            # Validate stage transition
            if not self._is_valid_stage_transition(idea.current_stage, next_stage):
                raise IdeaManagementError(
                    f"Invalid stage transition from {idea.current_stage} to {next_stage}"
                )
            
            # Update idea
            idea.advance_stage(next_stage)
            
            # Update status based on stage
            idea.status = self._get_status_for_stage(next_stage)
            
            await self.repository.update_idea(idea, user_id, correlation_id)
            
            logger.info(
                f"Idea {idea_id} advanced to stage {next_stage}",
                extra={
                    "idea_id": str(idea_id),
                    "stage": next_stage.value,
                    "status": idea.status.value,
                    "correlation_id": correlation_id
                }
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to advance idea {idea_id} to stage {next_stage}: {e}")
            raise IdeaManagementError(f"Stage advancement failed: {e}")
    
    async def update_stage_progress(
        self, 
        idea_id: UUID, 
        progress: float,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Update progress within current stage.
        
        Args:
            idea_id: Idea to update
            progress: Progress value (0.0 to 1.0)
            user_id: User making the change
            
        Returns:
            True if update successful
        """
        try:
            idea = await self.repository.find_by_id(idea_id)
            if not idea:
                raise IdeaManagementError(f"Idea {idea_id} not found")
            
            idea.update_progress(progress)
            await self.repository.update_idea(idea, user_id)
            
            logger.debug(f"Updated progress for idea {idea_id} to {progress}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update progress for idea {idea_id}: {e}")
            raise IdeaManagementError(f"Progress update failed: {e}")
    
    def _is_valid_stage_transition(self, current: PipelineStage, next_stage: PipelineStage) -> bool:
        """Validate if stage transition is allowed."""
        # Define valid stage progressions
        valid_transitions = {
            PipelineStage.IDEATE: [PipelineStage.RESEARCH],
            PipelineStage.RESEARCH: [PipelineStage.DECK, PipelineStage.IDEATE],
            PipelineStage.DECK: [PipelineStage.INVESTORS, PipelineStage.RESEARCH],
            PipelineStage.INVESTORS: [PipelineStage.MVP, PipelineStage.DECK],
            PipelineStage.MVP: [PipelineStage.SMOKE_TEST, PipelineStage.INVESTORS],
            PipelineStage.SMOKE_TEST: [PipelineStage.COMPLETE, PipelineStage.MVP],
            PipelineStage.COMPLETE: []  # Terminal stage
        }
        
        return next_stage in valid_transitions.get(current, [])
    
    def _get_status_for_stage(self, stage: PipelineStage) -> IdeaStatus:
        """Get appropriate status for pipeline stage."""
        status_mapping = {
            PipelineStage.IDEATE: IdeaStatus.DRAFT,
            PipelineStage.RESEARCH: IdeaStatus.RESEARCHING,
            PipelineStage.DECK: IdeaStatus.BUILDING,
            PipelineStage.INVESTORS: IdeaStatus.VALIDATING,
            PipelineStage.MVP: IdeaStatus.BUILDING,
            PipelineStage.SMOKE_TEST: IdeaStatus.TESTING,
            PipelineStage.COMPLETE: IdeaStatus.DEPLOYED
        }
        
        return status_mapping.get(stage, IdeaStatus.DRAFT)


class IdeaManager:
    """Main idea management orchestrator."""
    
    def __init__(
        self, 
        repository: IdeaRepository, 
        validator: IdeaValidator,
        config: ValidationConfig
    ):
        self.repository = repository
        self.validator = validator
        self.config = config
        self.duplicate_detector = DuplicateDetector(repository, config)
        self.lifecycle_manager = IdeaLifecycleManager(repository)
    
    async def create_idea(
        self, 
        raw_data: Dict[str, Any], 
        force_create: bool = False,
        user_id: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> Tuple[UUID, List[str]]:
        """
        Complete idea creation workflow with validation and duplicate detection.
        
        Args:
            raw_data: Raw idea data from user input
            force_create: Skip duplicate confirmation if True
            user_id: User creating the idea
            correlation_id: Optional correlation ID for tracking
            
        Returns:
            Tuple of (idea_id, warnings)
            
        Raises:
            ValidationError: If validation fails
            DuplicateIdeaError: If duplicates found and not forced
            StorageError: If storage fails
        """
        correlation_id = correlation_id or str(uuid4())
        
        try:
            logger.info(
                f"Starting idea creation workflow",
                extra={
                    "user_id": user_id,
                    "correlation_id": correlation_id,
                    "force_create": force_create
                }
            )
            
            # Step 1: Validate and sanitize input
            draft, validation_result = self.validator.validate_and_sanitize_draft(raw_data)
            
            if not validation_result.is_valid:
                raise ValidationError(f"Validation failed: {'; '.join(validation_result.errors)}")
            
            # Step 2: Check for duplicates
            if not force_create:
                duplicate_result = await self.duplicate_detector.check_for_duplicates(draft)
                
                if duplicate_result.found_similar:
                    similar_info = []
                    if duplicate_result.exact_matches:
                        similar_info.append(f"Exact matches: {len(duplicate_result.exact_matches)}")
                    if duplicate_result.similar_ideas:
                        max_score = max(duplicate_result.similarity_scores.values()) if duplicate_result.similarity_scores else 0
                        similar_info.append(f"Similar ideas: {len(duplicate_result.similar_ideas)} (max similarity: {max_score:.2f})")
                    
                    raise DuplicateIdeaError(
                        f"Similar ideas found: {'; '.join(similar_info)}. "
                        f"Use force_create=True to override."
                    )
            
            # Step 3: Create and save idea
            idea = Idea(
                title=draft.title,
                description=draft.description,
                category=draft.category,
                problem_statement=draft.problem_statement,
                solution_description=draft.solution_description,
                target_market=draft.target_market,
                evidence_links=draft.evidence_links,
                created_by=user_id
            )
            
            idea_id = await self.repository.save_idea(idea, correlation_id)
            
            logger.info(
                f"Idea created successfully",
                extra={
                    "idea_id": str(idea_id),
                    "title": idea.title,
                    "user_id": user_id,
                    "correlation_id": correlation_id,
                    "validation_warnings": len(validation_result.warnings)
                }
            )
            
            return idea_id, validation_result.warnings
            
        except (ValidationError, DuplicateIdeaError):
            raise
        except Exception as e:
            logger.error(f"Idea creation failed: {e}", extra={"correlation_id": correlation_id})
            raise StorageError(f"Failed to create idea: {e}")
    
    async def get_idea(self, idea_id: UUID) -> Optional[Idea]:
        """
        Retrieve idea by ID.
        
        Args:
            idea_id: Idea UUID
            
        Returns:
            Idea entity or None if not found
        """
        try:
            return await self.repository.find_by_id(idea_id)
        except Exception as e:
            logger.error(f"Failed to retrieve idea {idea_id}: {e}")
            raise IdeaManagementError(f"Failed to retrieve idea: {e}")
    
    async def list_ideas(
        self, 
        filters: Optional[QueryParams] = None,
        user_id: Optional[str] = None
    ) -> List[IdeaSummary]:
        """
        List ideas with filtering and pagination.
        
        Args:
            filters: Query parameters for filtering
            user_id: Optional user ID for access control
            
        Returns:
            List of idea summaries
        """
        try:
            filters = filters or QueryParams()
            
            ideas = await self.repository.find_with_filters(filters)
            
            # Convert to summaries
            summaries = []
            for idea in ideas:
                progress = self._calculate_overall_progress(idea)
                summary = IdeaSummary(
                    id=idea.idea_id,
                    title=idea.title,
                    status=idea.status,
                    stage=idea.current_stage,
                    created_at=idea.created_at,
                    progress=progress
                )
                summaries.append(summary)
            
            logger.debug(f"Listed {len(summaries)} ideas", extra={"user_id": user_id})
            return summaries
            
        except Exception as e:
            logger.error(f"Failed to list ideas: {e}")
            raise IdeaManagementError(f"Failed to list ideas: {e}")
    
    async def update_idea(
        self, 
        idea_id: UUID, 
        updates: Dict[str, Any],
        user_id: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> bool:
        """
        Update existing idea with validation.
        
        Args:
            idea_id: Idea to update
            updates: Fields to update
            user_id: User making the update
            correlation_id: Optional correlation ID
            
        Returns:
            True if update successful
            
        Raises:
            ValidationError: If validation fails
            IdeaManagementError: If update fails
        """
        try:
            # Get existing idea
            existing_idea = await self.repository.find_by_id(idea_id)
            if not existing_idea:
                raise IdeaManagementError(f"Idea {idea_id} not found")
            
            # Check if idea is in a modifiable state
            locked_statuses = [IdeaStatus.DEPLOYED, IdeaStatus.ARCHIVED]
            if existing_idea.status in locked_statuses:
                raise IdeaManagementError(
                    f"Cannot modify idea in status: {existing_idea.status}"
                )
            
            # Validate updates
            validation_result = self.validator.validate_partial_update(existing_idea, updates)
            if not validation_result.is_valid:
                raise ValidationError(f"Update validation failed: {'; '.join(validation_result.errors)}")
            
            # Apply updates
            for field, value in updates.items():
                if hasattr(existing_idea, field):
                    setattr(existing_idea, field, value)
            
            existing_idea.updated_at = datetime.utcnow()
            
            await self.repository.update_idea(existing_idea, user_id, correlation_id)
            
            logger.info(
                f"Idea {idea_id} updated successfully",
                extra={
                    "idea_id": str(idea_id),
                    "fields_updated": list(updates.keys()),
                    "user_id": user_id,
                    "correlation_id": correlation_id
                }
            )
            
            return True
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to update idea {idea_id}: {e}")
            raise IdeaManagementError(f"Failed to update idea: {e}")
    
    async def delete_idea(
        self, 
        idea_id: UUID,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Delete idea by ID.
        
        Args:
            idea_id: Idea to delete
            user_id: User performing deletion
            
        Returns:
            True if deletion successful
        """
        try:
            result = await self.repository.delete_idea(idea_id, user_id)
            
            if result:
                logger.info(f"Idea {idea_id} deleted", extra={"user_id": user_id})
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to delete idea {idea_id}: {e}")
            raise IdeaManagementError(f"Failed to delete idea: {e}")
    
    async def advance_stage(
        self, 
        idea_id: UUID, 
        next_stage: PipelineStage,
        user_id: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> bool:
        """
        Advance idea to next pipeline stage.
        
        Args:
            idea_id: Idea to advance
            next_stage: Target stage
            user_id: User making the change
            correlation_id: Optional correlation ID
            
        Returns:
            True if advancement successful
        """
        return await self.lifecycle_manager.advance_idea_stage(
            idea_id, next_stage, user_id, correlation_id
        )
    
    async def get_similar_ideas(
        self, 
        idea_id: UUID, 
        limit: int = 5
    ) -> List[Tuple[UUID, float]]:
        """
        Find ideas similar to the given idea.
        
        Args:
            idea_id: Reference idea
            limit: Maximum number of similar ideas
            
        Returns:
            List of (similar_idea_id, similarity_score) tuples
        """
        try:
            idea = await self.repository.find_by_id(idea_id)
            if not idea:
                return []
            
            similar = await self.repository.find_similar_by_embedding(
                description=idea.description,
                threshold=0.5,  # Lower threshold for similarity search
                limit=limit + 1  # +1 to exclude the idea itself
            )
            
            # Filter out the original idea
            filtered_similar = [
                (uid, score) for uid, score in similar 
                if uid != idea_id
            ]
            
            return filtered_similar[:limit]
            
        except Exception as e:
            logger.error(f"Failed to find similar ideas for {idea_id}: {e}")
            return []
    
    def _calculate_overall_progress(self, idea: Idea) -> float:
        """Calculate overall progress across all pipeline stages."""
        # Define stage weights (could be configurable)
        stage_weights = {
            PipelineStage.IDEATE: 0.1,
            PipelineStage.RESEARCH: 0.2,
            PipelineStage.DECK: 0.3,
            PipelineStage.INVESTORS: 0.4,
            PipelineStage.MVP: 0.7,
            PipelineStage.SMOKE_TEST: 0.9,
            PipelineStage.COMPLETE: 1.0
        }
        
        base_progress = stage_weights.get(idea.current_stage, 0.0)
        stage_contribution = idea.stage_progress * 0.1  # Current stage contributes 10%
        
        return min(base_progress + stage_contribution, 1.0)


# Factory function for easy instantiation
async def create_idea_manager(
    config: Optional[ValidationConfig] = None
) -> IdeaManager:
    """
    Create and initialize idea manager with dependencies.
    
    Args:
        config: Optional validation configuration
        
    Returns:
        Initialized IdeaManager instance
    """
    validation_config = config or get_validation_config()
    
    # Initialize dependencies
    validator = create_validator(validation_config)
    repository = await create_idea_repository()
    
    return IdeaManager(repository, validator, validation_config)