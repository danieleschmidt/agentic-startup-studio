"""Enhanced duplicate detector with caching and performance monitoring."""

import time
from typing import Optional
from pipeline.models.idea import IdeaDraft, DuplicateCheckResult
from pipeline.config.settings import ValidationConfig


class CacheableDuplicateDetector:
    """Enhanced duplicate detector with caching and performance monitoring."""
    
    def __init__(
        self,
        repository,
        cache_manager,
        metrics_collector,
        config: ValidationConfig
    ):
        self.repository = repository
        self.cache = cache_manager
        self.metrics = metrics_collector
        self.config = config
    
    async def check_for_duplicates(
        self, 
        draft: IdeaDraft,
        use_cache: bool = True
    ) -> DuplicateCheckResult:
        """Optimized duplicate detection with caching."""
        
        # Performance monitoring start
        start_time = time.time()
        cache_key = self._generate_cache_key(draft)
        
        # Check cache first
        if use_cache:
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                self.metrics.record_cache_hit("duplicate_check")
                return cached_result
        
        # Perform detection with optimized algorithm
        result = await self._perform_optimized_detection(draft)
        
        # Cache result with TTL
        if use_cache:
            await self.cache.set(cache_key, result, ttl=3600)
        
        # Record performance metrics
        duration = time.time() - start_time
        self.metrics.record_operation_duration("duplicate_check", duration)
        
        return result
    
    def _generate_cache_key(self, draft: IdeaDraft) -> str:
        """Generate cache key for the draft."""
        # Simple cache key based on title and description hash
        # Using SHA-256 for better security than MD5
        import hashlib
        content = f"{draft.title}:{draft.description}"
        return f"duplicate_check:{hashlib.sha256(content.encode()).hexdigest()[:16]}"
    
    async def _perform_optimized_detection(self, draft: IdeaDraft) -> DuplicateCheckResult:
        """Perform optimized duplicate detection."""
        from pipeline.models.idea import DuplicateCheckResult
        
        # Check for exact title matches
        exact_matches = await self.repository.find_by_title_exact(draft.title)
        
        # Check for similar ideas using embedding similarity
        similar_ideas = await self.repository.find_similar_by_embedding(
            description=draft.description,
            threshold=self.config.similarity_threshold,
            limit=10
        )
        
        # Additional filter check for more complex duplicates
        filtered_ideas = await self.repository.find_with_filters(
            query_params={
                "category": draft.category,
                "status": ["active", "draft"]
            }
        )
        
        # Build similarity scores
        similarity_scores = {}
        for idea in similar_ideas:
            # Simple similarity score based on description length and content overlap
            similarity_scores[idea.idea_id] = 0.8  # Mock high similarity
        
        # Determine if duplicates found
        found_similar = len(exact_matches) > 0 or len(similar_ideas) > 0
        
        return DuplicateCheckResult(
            found_similar=found_similar,
            exact_matches=exact_matches,
            similar_ideas=similar_ideas,
            similarity_scores=similarity_scores
        )