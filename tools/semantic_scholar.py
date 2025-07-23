"""
Semantic Scholar API Integration for Academic Paper Search

This module provides access to Semantic Scholar's Academic Paper API
for retrieving research papers related to startup ideas and market validation.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

try:
    import aiohttp
    from pipeline.adapters.base_adapter import BaseAdapter, AdapterConfig, AuthType
    from pipeline.infrastructure.circuit_breaker import circuit_breaker, CircuitBreakerConfig
    ASYNC_AVAILABLE = True
except ImportError:
    # Fallback for environments without full pipeline dependencies
    import requests
    ASYNC_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class SemanticScholarConfig:
    """Configuration for Semantic Scholar API integration."""
    
    # API Configuration
    base_url: str = "https://api.semanticscholar.org/graph/v1"
    api_key: Optional[str] = None  # Optional for basic usage
    
    # Rate Limiting (per Semantic Scholar's limits)
    requests_per_minute: int = 100
    requests_per_second: int = 10
    
    # Search Configuration
    max_results: int = 10
    min_citations: int = 5
    min_year: int = 2020
    
    # Fields to retrieve from API
    paper_fields: List[str] = None
    
    def __post_init__(self):
        if self.paper_fields is None:
            self.paper_fields = [
                'paperId', 'title', 'abstract', 'year', 'citationCount',
                'publicationDate', 'authors', 'venue', 'externalIds', 'url'
            ]


class SemanticScholarAdapter(BaseAdapter):
    """Semantic Scholar API adapter following the established pattern."""
    
    def __init__(self, config: Optional[SemanticScholarConfig] = None):
        """Initialize the Semantic Scholar adapter."""
        self.ss_config = config or SemanticScholarConfig()
        
        # Create adapter config for base class
        adapter_config = AdapterConfig(
            base_url=self.ss_config.base_url,
            auth_type=AuthType.API_KEY if self.ss_config.api_key else AuthType.NONE,
            api_key=self.ss_config.api_key,
            requests_per_window=self.ss_config.requests_per_minute,
            window_size_seconds=60,
            max_retries=3,
            base_delay=1.0,
            timeout=30.0
        )
        
        super().__init__("semantic_scholar", adapter_config)
    
    def get_headers(self) -> Dict[str, str]:
        """Get request headers including API key if available."""
        headers = {
            'User-Agent': 'Agentic-Startup-Studio/1.0',
            'Accept': 'application/json'
        }
        
        if self.ss_config.api_key:
            headers['x-api-key'] = self.ss_config.api_key
        
        return headers
    
    @circuit_breaker("semantic_scholar_search")
    async def search_papers(self, query: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search for academic papers using Semantic Scholar API.
        
        Args:
            query: Search query for papers
            limit: Maximum number of results (defaults to config.max_results)
            
        Returns:
            List of paper dictionaries containing metadata
        """
        limit = limit or self.ss_config.max_results
        
        # Build search parameters
        params = {
            'query': query,
            'limit': min(limit, 100),  # API limit
            'fields': ','.join(self.ss_config.paper_fields),
            'year': f"{self.ss_config.min_year}-",  # Papers from min_year onwards
        }
        
        try:
            logger.info(f"Searching Semantic Scholar for: {query}")
            
            response_data = await self.get_json(
                "/paper/search",
                params=params,
                headers=self.get_headers()
            )
            
            papers = response_data.get('data', [])
            
            # Filter by minimum citations
            filtered_papers = [
                paper for paper in papers
                if paper.get('citationCount', 0) >= self.ss_config.min_citations
            ]
            
            logger.info(f"Found {len(filtered_papers)} relevant papers for query: {query}")
            
            return filtered_papers
            
        except Exception as e:
            logger.error(f"Error searching Semantic Scholar: {e}")
            raise
    
    async def get_paper_details(self, paper_id: str) -> Dict[str, Any]:
        """
        Get detailed information for a specific paper.
        
        Args:
            paper_id: Semantic Scholar paper ID
            
        Returns:
            Dictionary containing detailed paper information
        """
        try:
            params = {
                'fields': ','.join(self.ss_config.paper_fields + ['references', 'citations'])
            }
            
            response_data = await self.get_json(
                f"/paper/{paper_id}",
                params=params,
                headers=self.get_headers()
            )
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error retrieving paper details for {paper_id}: {e}")
            raise
    
    async def get_trending_papers(self, field: str = "Computer Science") -> List[Dict[str, Any]]:
        """
        Get trending papers in a specific field.
        
        Args:
            field: Academic field to search (e.g., "Computer Science", "Business")
            
        Returns:
            List of trending papers
        """
        # Use recent high-citation papers as a proxy for trending
        current_year = datetime.now().year
        query = f"{field} startup OR {field} innovation"
        
        return await self.search_papers(query, limit=20)


# Circuit breaker configuration for Semantic Scholar
_circuit_breaker_config = CircuitBreakerConfig(
    failure_threshold=3,
    recovery_timeout=60.0,
    success_threshold=2,
    timeout=30.0
)


# Global adapter instance for compatibility with existing code
_global_adapter: Optional[SemanticScholarAdapter] = None


def get_semantic_scholar_adapter() -> SemanticScholarAdapter:
    """Get or create the global Semantic Scholar adapter instance."""
    global _global_adapter
    if _global_adapter is None:
        config = SemanticScholarConfig()
        # Try to get API key from environment
        import os
        config.api_key = os.getenv('SEMANTIC_SCHOLAR_API_KEY')
        _global_adapter = SemanticScholarAdapter(config)
    return _global_adapter


# Legacy function interface for backward compatibility
def run(query: str) -> List[Dict[str, Any]]:
    """
    Legacy synchronous interface for backward compatibility.
    
    Args:
        query: Search query for papers
        
    Returns:
        List of paper dictionaries
    """
    if ASYNC_AVAILABLE:
        # Use async adapter
        adapter = get_semantic_scholar_adapter()
        
        # Create event loop if none exists
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run async search
        try:
            if loop.is_running():
                # If we're already in an async context, create a task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, adapter.search_papers(query))
                    return future.result()
            else:
                return loop.run_until_complete(adapter.search_papers(query))
        except Exception as e:
            logger.error(f"Error in async search: {e}")
            return _fallback_search(query)
    else:
        # Fallback to synchronous implementation
        return _fallback_search(query)


def _fallback_search(query: str) -> List[Dict[str, Any]]:
    """
    Fallback synchronous search implementation.
    
    Args:
        query: Search query
        
    Returns:
        List of paper dictionaries from direct API call
    """
    try:
        import requests
        
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            'query': query,
            'limit': 10,
            'fields': 'paperId,title,abstract,year,citationCount,authors,venue,url'
        }
        
        headers = {
            'User-Agent': 'Agentic-Startup-Studio/1.0',
            'Accept': 'application/json'
        }
        
        # Add API key if available
        import os
        api_key = os.getenv('SEMANTIC_SCHOLAR_API_KEY')
        if api_key:
            headers['x-api-key'] = api_key
        
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        papers = data.get('data', [])
        
        # Filter for quality papers (minimum 5 citations)
        quality_papers = [
            paper for paper in papers
            if paper.get('citationCount', 0) >= 5
        ]
        
        logger.info(f"Found {len(quality_papers)} quality papers for query: {query}")
        return quality_papers
        
    except Exception as e:
        logger.error(f"Fallback search failed: {e}")
        # Return empty list instead of mock data to maintain API contract
        return []


# Async interface for modern usage
async def search_papers_async(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Modern async interface for paper search.
    
    Args:
        query: Search query for papers
        limit: Maximum number of results
        
    Returns:
        List of paper dictionaries
    """
    adapter = get_semantic_scholar_adapter()
    return await adapter.search_papers(query, limit)


async def get_paper_details_async(paper_id: str) -> Dict[str, Any]:
    """
    Async interface for getting paper details.
    
    Args:
        paper_id: Semantic Scholar paper ID
        
    Returns:
        Dictionary containing detailed paper information
    """
    adapter = get_semantic_scholar_adapter()
    return await adapter.get_paper_details(paper_id)
