"""Vector cache manager for specialized vector operations."""

from typing import List, Tuple, Optional
from uuid import UUID


class VectorCacheManager:
    """Specialized cache manager for vector operations with L1/L2 caching."""
    
    def __init__(self, redis_client=None, lru_size: int = 1000, metrics_collector=None):
        self.redis_client = redis_client
        self.lru_size = lru_size
        self.metrics_collector = metrics_collector
        self._l1_cache = {}  # Simple in-memory L1 cache
    
    async def get_vector_similarity(self, vector_hash: str) -> Optional[List[Tuple[UUID, float]]]:
        """Get cached vector similarity results."""
        # Try L1 cache first
        if vector_hash in self._l1_cache:
            if self.metrics_collector:
                if hasattr(self.metrics_collector, 'record_hit'):
                    self.metrics_collector.record_hit("l1_cache")
            return self._l1_cache[vector_hash]
        
        # Try L2 cache (Redis) - mock implementation
        if self.redis_client:
            result = await self.redis_client.get(f"vector_sim:{vector_hash}")
            if result:
                # Promote to L1 cache
                self._l1_cache[vector_hash] = result
                if self.metrics_collector:
                    if hasattr(self.metrics_collector, 'record_hit'):
                        self.metrics_collector.record_hit("l2_cache")
                return result
        
        # Cache miss
        if self.metrics_collector:
            if hasattr(self.metrics_collector, 'record_miss'):
                self.metrics_collector.record_miss()
        return None
    
    async def set_vector_similarity(self, vector_hash: str, similarities: List[Tuple[UUID, float]]) -> None:
        """Store vector similarity results in cache."""
        self._l1_cache[vector_hash] = similarities
        
        # Also store in L2 cache if available
        if self.redis_client:
            await self.redis_client.set(f"vector_sim:{vector_hash}", similarities)