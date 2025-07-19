"""
Cache Manager - Redis-based caching for pipeline performance optimization.

Provides distributed caching for expensive operations including:
- Evidence collection results
- Pitch deck generation intermediate data
- Vector similarity search results
- API responses and processed data
"""

import asyncio
import json
import logging
import pickle
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

from pipeline.config.settings import get_settings


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    ttl_seconds: int
    created_at: datetime = field(default_factory=datetime.utcnow)
    hit_count: int = 0
    size_bytes: int = 0


class CacheManager:
    """Redis-based cache manager with fallback to in-memory cache."""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)
        
        # Redis connection
        self.redis_client: Optional[redis.Redis] = None
        self.redis_available = False
        
        # In-memory fallback cache
        self._memory_cache: Dict[str, CacheEntry] = {}
        self._memory_cache_size = 0
        self._max_memory_cache_size = 100 * 1024 * 1024  # 100MB
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'evictions': 0,
            'redis_errors': 0
        }
    
    async def initialize(self):
        """Initialize Redis connection with fallback to memory cache."""
        if not REDIS_AVAILABLE:
            self.logger.warning("Redis not available, using memory cache only")
            return
        
        try:
            # Try to connect to Redis
            redis_url = self.settings.redis_url if hasattr(self.settings, 'redis_url') else "redis://localhost:6379"
            
            self.redis_client = redis.from_url(
                redis_url,
                decode_responses=False,  # We handle encoding ourselves
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            await self.redis_client.ping()
            self.redis_available = True
            self.logger.info("Redis cache initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"Redis connection failed, using memory cache: {e}")
            self.redis_available = False
            if self.redis_client:
                await self.redis_client.close()
                self.redis_client = None
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            # Try Redis first
            if self.redis_available and self.redis_client:
                try:
                    data = await self.redis_client.get(self._make_key(key))
                    if data:
                        value = pickle.loads(data)
                        self.stats['hits'] += 1
                        return value
                except Exception as e:
                    self.logger.warning(f"Redis get error: {e}")
                    self.stats['redis_errors'] += 1
            
            # Fallback to memory cache
            if key in self._memory_cache:
                entry = self._memory_cache[key]
                
                # Check if expired
                if self._is_expired(entry):
                    del self._memory_cache[key]
                    self._memory_cache_size -= entry.size_bytes
                    self.stats['misses'] += 1
                    return None
                
                entry.hit_count += 1
                self.stats['hits'] += 1
                return entry.value
            
            self.stats['misses'] += 1
            return None
            
        except Exception as e:
            self.logger.error(f"Cache get error for key {key}: {e}")
            self.stats['misses'] += 1
            return None
    
    async def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> bool:
        """Set value in cache with TTL."""
        try:
            # Try Redis first
            if self.redis_available and self.redis_client:
                try:
                    data = pickle.dumps(value)
                    await self.redis_client.setex(self._make_key(key), ttl_seconds, data)
                    self.stats['sets'] += 1
                    return True
                except Exception as e:
                    self.logger.warning(f"Redis set error: {e}")
                    self.stats['redis_errors'] += 1
            
            # Fallback to memory cache
            data = pickle.dumps(value)
            size_bytes = len(data)
            
            # Check memory limit
            if self._memory_cache_size + size_bytes > self._max_memory_cache_size:
                await self._evict_memory_cache()
            
            # Remove existing entry if present
            if key in self._memory_cache:
                self._memory_cache_size -= self._memory_cache[key].size_bytes
            
            # Add new entry
            entry = CacheEntry(
                key=key,
                value=value,
                ttl_seconds=ttl_seconds,
                size_bytes=size_bytes
            )
            
            self._memory_cache[key] = entry
            self._memory_cache_size += size_bytes
            self.stats['sets'] += 1
            
            return True
            
        except Exception as e:
            self.logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            deleted = False
            
            # Delete from Redis
            if self.redis_available and self.redis_client:
                try:
                    result = await self.redis_client.delete(self._make_key(key))
                    deleted = result > 0
                except Exception as e:
                    self.logger.warning(f"Redis delete error: {e}")
                    self.stats['redis_errors'] += 1
            
            # Delete from memory cache
            if key in self._memory_cache:
                entry = self._memory_cache[key]
                self._memory_cache_size -= entry.size_bytes
                del self._memory_cache[key]
                deleted = True
            
            return deleted
            
        except Exception as e:
            self.logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            # Check Redis first
            if self.redis_available and self.redis_client:
                try:
                    exists = await self.redis_client.exists(self._make_key(key))
                    if exists:
                        return True
                except Exception as e:
                    self.logger.warning(f"Redis exists error: {e}")
                    self.stats['redis_errors'] += 1
            
            # Check memory cache
            if key in self._memory_cache:
                entry = self._memory_cache[key]
                if self._is_expired(entry):
                    del self._memory_cache[key]
                    self._memory_cache_size -= entry.size_bytes
                    return False
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Cache exists error for key {key}: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all cache entries."""
        try:
            cleared = False
            
            # Clear Redis
            if self.redis_available and self.redis_client:
                try:
                    # Delete all keys with our prefix
                    keys = await self.redis_client.keys(f"{self._get_prefix()}*")
                    if keys:
                        await self.redis_client.delete(*keys)
                    cleared = True
                except Exception as e:
                    self.logger.warning(f"Redis clear error: {e}")
                    self.stats['redis_errors'] += 1
            
            # Clear memory cache
            self._memory_cache.clear()
            self._memory_cache_size = 0
            cleared = True
            
            return cleared
            
        except Exception as e:
            self.logger.error(f"Cache clear error: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        memory_stats = {
            'entry_count': len(self._memory_cache),
            'size_bytes': self._memory_cache_size,
            'size_mb': round(self._memory_cache_size / 1024 / 1024, 2)
        }
        
        redis_stats = {}
        if self.redis_available and self.redis_client:
            try:
                info = await self.redis_client.info('memory')
                redis_stats = {
                    'used_memory': info.get('used_memory', 0),
                    'used_memory_human': info.get('used_memory_human', 'N/A'),
                    'connected_clients': info.get('connected_clients', 0)
                }
            except Exception as e:
                redis_stats = {'error': str(e)}
        
        return {
            'general': self.stats.copy(),
            'memory_cache': memory_stats,
            'redis_cache': redis_stats,
            'redis_available': self.redis_available
        }
    
    def _make_key(self, key: str) -> str:
        """Create prefixed cache key."""
        return f"{self._get_prefix()}{key}"
    
    def _get_prefix(self) -> str:
        """Get cache key prefix."""
        return "pipeline:cache:"
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired."""
        expiry_time = entry.created_at + timedelta(seconds=entry.ttl_seconds)
        return datetime.utcnow() > expiry_time
    
    async def _evict_memory_cache(self):
        """Evict least recently used entries from memory cache."""
        if not self._memory_cache:
            return
        
        # Sort by hit count (LRU approximation)
        sorted_entries = sorted(
            self._memory_cache.items(),
            key=lambda x: (x[1].hit_count, x[1].created_at)
        )
        
        # Remove 25% of entries
        entries_to_remove = max(1, len(sorted_entries) // 4)
        
        for i in range(entries_to_remove):
            key, entry = sorted_entries[i]
            self._memory_cache_size -= entry.size_bytes
            del self._memory_cache[key]
            self.stats['evictions'] += 1
    
    async def close(self):
        """Close cache connections."""
        if self.redis_client:
            await self.redis_client.close()


# Cache decorators for common use cases
def cache_result(ttl_seconds: int = 3600, key_prefix: str = ""):
    """Decorator to cache function results."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            cache = get_cache_manager()
            
            # Create cache key from function name and arguments
            key_parts = [key_prefix, func.__name__]
            if args:
                key_parts.extend(str(arg) for arg in args)
            if kwargs:
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            
            cache_key = ":".join(key_parts)
            
            # Try to get cached result
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache.set(cache_key, result, ttl_seconds)
            
            return result
        
        return wrapper
    return decorator


# Singleton instance
_cache_manager = None


async def get_cache_manager() -> CacheManager:
    """Get singleton cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
        await _cache_manager.initialize()
    return _cache_manager


# Specific cache functions for common operations
async def cache_evidence_collection(claim: str, domains: List[str], evidence_data: Any, ttl: int = 1800):
    """Cache evidence collection results."""
    cache = await get_cache_manager()
    key = f"evidence:{hash(claim)}:{':'.join(sorted(domains))}"
    await cache.set(key, evidence_data, ttl)


async def get_cached_evidence_collection(claim: str, domains: List[str]) -> Optional[Any]:
    """Get cached evidence collection results."""
    cache = await get_cache_manager()
    key = f"evidence:{hash(claim)}:{':'.join(sorted(domains))}"
    return await cache.get(key)


async def cache_pitch_deck(startup_idea: str, target_investor: str, deck_data: Any, ttl: int = 3600):
    """Cache pitch deck generation results."""
    cache = await get_cache_manager()
    key = f"pitch_deck:{hash(startup_idea)}:{target_investor}"
    await cache.set(key, deck_data, ttl)


async def get_cached_pitch_deck(startup_idea: str, target_investor: str) -> Optional[Any]:
    """Get cached pitch deck generation results."""
    cache = await get_cache_manager()
    key = f"pitch_deck:{hash(startup_idea)}:{target_investor}"
    return await cache.get(key)


async def cache_vector_search(query_vector: str, search_params: Dict, results: Any, ttl: int = 900):
    """Cache vector similarity search results."""
    cache = await get_cache_manager()
    params_key = ":".join(f"{k}={v}" for k, v in sorted(search_params.items()))
    key = f"vector_search:{hash(query_vector)}:{params_key}"
    await cache.set(key, results, ttl)


async def get_cached_vector_search(query_vector: str, search_params: Dict) -> Optional[Any]:
    """Get cached vector similarity search results."""
    cache = await get_cache_manager()
    params_key = ":".join(f"{k}={v}" for k, v in sorted(search_params.items()))
    key = f"vector_search:{hash(query_vector)}:{params_key}"
    return await cache.get(key)