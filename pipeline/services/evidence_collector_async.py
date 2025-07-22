"""
Optimized Async Evidence Collector - High-performance evidence collection.

Key optimizations:
- Parallel search across multiple domains and queries
- Connection pooling for API requests
- Async DNS resolution
- Batch URL validation
- Smart caching with TTL
- Rate limiting with semaphores
- Circuit breaker for fault tolerance
"""

import asyncio
import aiohttp
import aiodns
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from urllib.parse import urlparse, quote_plus
import hashlib
from asyncio import Semaphore

from pipeline.services.budget_sentinel import get_budget_sentinel, BudgetCategory
from pipeline.infrastructure.circuit_breaker import CircuitBreaker
from pipeline.config.cache_manager import get_cache_manager


@dataclass
class Evidence:
    """Evidence item with metadata and quality scores."""
    claim_text: str
    citation_url: str
    citation_title: str
    source_type: str  # 'academic', 'news', 'blog', 'social'
    publication_date: Optional[datetime]
    relevance_score: float  # 0-1 score for query relevance
    credibility_score: float  # 0-1 score for source credibility
    freshness_score: float  # 0-1 score based on recency
    composite_score: float = 0.0  # Calculated weighted score
    
    def __post_init__(self):
        """Calculate composite score after initialization."""
        self.composite_score = (
            self.relevance_score * 0.4 +
            self.credibility_score * 0.4 +
            self.freshness_score * 0.2
        )


@dataclass
class ResearchDomain:
    """Research domain configuration."""
    name: str
    keywords: List[str]
    min_evidence_count: int = 3
    quality_threshold: float = 0.6
    search_depth: int = 2  # Number of result pages to fetch


@dataclass 
class SearchResult:
    """Raw search result from API."""
    title: str
    url: str
    snippet: str
    source: str
    published_date: Optional[datetime] = None


class AsyncEvidenceCollector:
    """Optimized async evidence collector with parallel operations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Services
        self.budget_sentinel = get_budget_sentinel()
        self.cache_manager = None  # Initialized async
        
        # Connection pooling
        self.session: Optional[aiohttp.ClientSession] = None
        self.dns_resolver = aiodns.DNSResolver()
        
        # Concurrency control
        self.search_semaphore = Semaphore(self.config.get('max_concurrent_searches', 5))
        self.url_semaphore = Semaphore(self.config.get('max_concurrent_url_checks', 10))
        self.api_rate_limiter = Semaphore(self.config.get('api_rate_limit', 10))
        
        # Circuit breakers for different services
        self.circuit_breakers = {
            'google_search': CircuitBreaker(failure_threshold=5, recovery_timeout=60),
            'bing_search': CircuitBreaker(failure_threshold=5, recovery_timeout=60),
            'semantic_scholar': CircuitBreaker(failure_threshold=3, recovery_timeout=120),
            'url_validation': CircuitBreaker(failure_threshold=10, recovery_timeout=30)
        }
        
        # Caching
        self.search_cache: Dict[str, List[SearchResult]] = {}
        self.url_validation_cache: Dict[str, bool] = {}
        self.dns_cache: Dict[str, str] = {}
        
        # Deduplication
        self.citation_cache: Set[str] = set()
        
        # Performance tracking
        self.stats = {
            'searches_performed': 0,
            'cache_hits': 0,
            'urls_validated': 0,
            'parallel_operations': 0
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._cleanup()
    
    async def _initialize(self):
        """Initialize async resources."""
        # Initialize cache manager
        self.cache_manager = await get_cache_manager()
        
        # Create connection pool
        connector = aiohttp.TCPConnector(
            limit=50,
            limit_per_host=10,
            ttl_dns_cache=300,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'User-Agent': 'AsyncEvidenceCollector/1.0'
            }
        )
        
        self.logger.info("Async evidence collector initialized")
    
    async def _cleanup(self):
        """Cleanup async resources."""
        if self.session:
            await self.session.close()
        
        self.logger.info(
            f"Evidence collector stats - "
            f"Searches: {self.stats['searches_performed']}, "
            f"Cache hits: {self.stats['cache_hits']}, "
            f"URLs validated: {self.stats['urls_validated']}, "
            f"Parallel ops: {self.stats['parallel_operations']}"
        )
    
    async def collect_evidence(
        self,
        claim: str,
        research_domains: List[ResearchDomain],
        min_total_evidence: int = 10,
        timeout: int = 120
    ) -> Dict[str, List[Evidence]]:
        """
        Collect evidence for a claim across multiple research domains in parallel.
        
        Key optimizations:
        - Parallel domain searches
        - Batch URL validation
        - Smart result caching
        - Async DNS resolution
        """
        self.logger.info(f"Collecting evidence for: {claim[:100]}...")
        
        try:
            # Track operation cost
            async with self.budget_sentinel.track_operation(
                "evidence_collector",
                "collect_evidence",
                BudgetCategory.RESEARCH,
                max_cost=10.0
            ):
                # Create tasks for parallel domain searches
                domain_tasks = []
                for domain in research_domains:
                    task = self._collect_domain_evidence_async(claim, domain)
                    domain_tasks.append(task)
                
                # Execute all domain searches in parallel with timeout
                self.stats['parallel_operations'] += 1
                results = await asyncio.wait_for(
                    asyncio.gather(*domain_tasks, return_exceptions=True),
                    timeout=timeout
                )
                
                # Process results
                evidence_by_domain = {}
                total_evidence = 0
                
                for i, result in enumerate(results):
                    domain = research_domains[i]
                    if isinstance(result, Exception):
                        self.logger.error(f"Domain {domain.name} search failed: {result}")
                        evidence_by_domain[domain.name] = []
                    else:
                        evidence_by_domain[domain.name] = result
                        total_evidence += len(result)
                
                self.logger.info(
                    f"Evidence collection complete: {total_evidence} items across "
                    f"{len(evidence_by_domain)} domains"
                )
                
                return evidence_by_domain
                
        except asyncio.TimeoutError:
            self.logger.error(f"Evidence collection timed out after {timeout}s")
            return {}
        except Exception as e:
            self.logger.error(f"Evidence collection failed: {e}")
            return {}
    
    async def _collect_domain_evidence_async(
        self,
        claim: str,
        domain: ResearchDomain
    ) -> List[Evidence]:
        """Collect evidence for a specific research domain with parallel queries."""
        self.logger.info(f"Collecting evidence for domain: {domain.name}")
        
        try:
            # Generate search queries for this domain
            queries = self._generate_search_queries(claim, domain.keywords)
            
            # Search across queries in parallel
            query_tasks = []
            for query in queries[:3]:  # Limit queries per domain
                task = self._search_with_circuit_breaker(query, domain)
                query_tasks.append(task)
            
            # Execute queries in parallel
            async with self.search_semaphore:
                search_results = await asyncio.gather(*query_tasks, return_exceptions=True)
            
            # Flatten and deduplicate results
            all_results = []
            for result in search_results:
                if isinstance(result, list):
                    all_results.extend(result)
            
            # Validate URLs in parallel batches
            valid_results = await self._batch_validate_urls(all_results)
            
            # Convert to evidence with quality scoring
            evidence_list = await self._create_evidence_from_results(
                valid_results, claim, domain
            )
            
            # Sort by quality and return top results
            evidence_list.sort(key=lambda e: e.composite_score, reverse=True)
            return evidence_list[:domain.min_evidence_count * 2]
            
        except Exception as e:
            self.logger.error(f"Domain evidence collection failed for {domain.name}: {e}")
            return []
    
    async def _search_with_circuit_breaker(
        self,
        query: str,
        domain: ResearchDomain
    ) -> List[SearchResult]:
        """Execute search with circuit breaker protection and caching."""
        # Check cache first
        cache_key = f"search:{hashlib.md5(query.encode()).hexdigest()}"
        
        if cache_key in self.search_cache:
            self.stats['cache_hits'] += 1
            return self.search_cache[cache_key]
        
        # Try different search providers with circuit breakers
        providers = [
            ('google_search', self._search_google),
            ('bing_search', self._search_bing),
            ('semantic_scholar', self._search_semantic_scholar)
        ]
        
        for provider_name, search_func in providers:
            if await self.circuit_breakers[provider_name].call_async():
                try:
                    results = await search_func(query, domain.search_depth)
                    self.search_cache[cache_key] = results
                    self.stats['searches_performed'] += 1
                    return results
                except Exception as e:
                    self.logger.warning(f"{provider_name} failed: {e}")
                    await self.circuit_breakers[provider_name].record_failure()
        
        # All providers failed
        return []
    
    async def _search_google(self, query: str, pages: int = 1) -> List[SearchResult]:
        """Search using Google Custom Search API (placeholder)."""
        # In production, this would use actual Google API
        async with self.api_rate_limiter:
            await asyncio.sleep(0.1)  # Simulate API call
            
            return [
                SearchResult(
                    title=f"Google result {i+1} for: {query[:50]}",
                    url=f"https://example{i+1}.com/{quote_plus(query[:20])}",
                    snippet=f"This is a relevant snippet about {query[:50]}...",
                    source="google",
                    published_date=datetime.utcnow() - timedelta(days=i*30)
                )
                for i in range(5 * pages)
            ]
    
    async def _search_bing(self, query: str, pages: int = 1) -> List[SearchResult]:
        """Search using Bing Search API (placeholder)."""
        async with self.api_rate_limiter:
            await asyncio.sleep(0.1)  # Simulate API call
            
            return [
                SearchResult(
                    title=f"Bing result {i+1} for: {query[:50]}",
                    url=f"https://sample{i+1}.org/{quote_plus(query[:20])}",
                    snippet=f"Bing found this about {query[:50]}...",
                    source="bing",
                    published_date=datetime.utcnow() - timedelta(days=i*15)
                )
                for i in range(5 * pages)
            ]
    
    async def _search_semantic_scholar(self, query: str, pages: int = 1) -> List[SearchResult]:
        """Search academic papers using Semantic Scholar API (placeholder)."""
        async with self.api_rate_limiter:
            await asyncio.sleep(0.1)  # Simulate API call
            
            return [
                SearchResult(
                    title=f"Academic paper {i+1}: {query[:40]}",
                    url=f"https://academic{i+1}.edu/paper/{quote_plus(query[:20])}",
                    snippet=f"Academic research shows that {query[:50]}...",
                    source="semantic_scholar",
                    published_date=datetime.utcnow() - timedelta(days=i*60)
                )
                for i in range(3 * pages)
            ]
    
    async def _batch_validate_urls(self, results: List[SearchResult]) -> List[SearchResult]:
        """Validate URLs in parallel batches."""
        # Group URLs for batch validation
        urls_to_validate = []
        for result in results:
            if result.url not in self.url_validation_cache:
                urls_to_validate.append(result.url)
        
        if urls_to_validate:
            # Validate in batches
            batch_size = 20
            for i in range(0, len(urls_to_validate), batch_size):
                batch = urls_to_validate[i:i + batch_size]
                validation_tasks = [
                    self._validate_url_async(url) for url in batch
                ]
                
                async with self.url_semaphore:
                    validation_results = await asyncio.gather(
                        *validation_tasks, return_exceptions=True
                    )
                
                # Update cache
                for url, is_valid in zip(batch, validation_results):
                    if not isinstance(is_valid, Exception):
                        self.url_validation_cache[url] = is_valid
                        self.stats['urls_validated'] += 1
        
        # Filter results based on validation
        valid_results = []
        for result in results:
            if self.url_validation_cache.get(result.url, False):
                valid_results.append(result)
        
        return valid_results
    
    async def _validate_url_async(self, url: str) -> bool:
        """Validate URL accessibility with async DNS and HTTP check."""
        if not await self.circuit_breakers['url_validation'].call_async():
            return False
        
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.hostname:
                return False
            
            # Async DNS resolution with caching
            if parsed.hostname not in self.dns_cache:
                try:
                    # Use aiodns for async DNS resolution
                    result = await self.dns_resolver.query(parsed.hostname, 'A')
                    if result:
                        self.dns_cache[parsed.hostname] = str(result[0].host)
                except Exception:
                    return False
            
            # Quick HTTP HEAD request to check accessibility
            async with self.session.head(url, allow_redirects=True) as response:
                return 200 <= response.status < 400
                
        except Exception as e:
            await self.circuit_breakers['url_validation'].record_failure()
            return False
    
    async def _create_evidence_from_results(
        self,
        results: List[SearchResult],
        claim: str,
        domain: ResearchDomain
    ) -> List[Evidence]:
        """Create evidence objects with quality scoring."""
        evidence_list = []
        
        for result in results:
            # Skip if already cited
            url_hash = hashlib.md5(result.url.encode()).hexdigest()
            if url_hash in self.citation_cache:
                continue
            
            # Calculate quality scores
            relevance_score = await self._calculate_relevance_score(
                claim, result.title, result.snippet
            )
            
            credibility_score = self._calculate_credibility_score(
                result.source, result.url
            )
            
            freshness_score = self._calculate_freshness_score(
                result.published_date
            )
            
            # Create evidence object
            evidence = Evidence(
                claim_text=result.snippet,
                citation_url=result.url,
                citation_title=result.title,
                source_type=self._classify_source_type(result.url),
                publication_date=result.published_date,
                relevance_score=relevance_score,
                credibility_score=credibility_score,
                freshness_score=freshness_score
            )
            
            # Add if meets quality threshold
            if evidence.composite_score >= domain.quality_threshold:
                evidence_list.append(evidence)
                self.citation_cache.add(url_hash)
        
        return evidence_list
    
    async def _calculate_relevance_score(
        self,
        claim: str,
        title: str,
        snippet: str
    ) -> float:
        """Calculate relevance score using async text analysis."""
        # Simple keyword matching (in production, use embeddings or LLM)
        claim_words = set(claim.lower().split())
        text_words = set((title + " " + snippet).lower().split())
        
        overlap = len(claim_words & text_words)
        return min(overlap / len(claim_words), 1.0) if claim_words else 0.0
    
    def _calculate_credibility_score(self, source: str, url: str) -> float:
        """Calculate source credibility score."""
        # Academic sources get higher credibility
        if source == "semantic_scholar" or ".edu" in url:
            return 0.9
        elif any(domain in url for domain in [".gov", ".org"]):
            return 0.8
        elif source in ["google", "bing"]:
            return 0.7
        else:
            return 0.5
    
    def _calculate_freshness_score(self, published_date: Optional[datetime]) -> float:
        """Calculate freshness score based on publication date."""
        if not published_date:
            return 0.5
        
        age_days = (datetime.utcnow() - published_date).days
        
        if age_days < 30:
            return 1.0
        elif age_days < 180:
            return 0.8
        elif age_days < 365:
            return 0.6
        else:
            return 0.4
    
    def _classify_source_type(self, url: str) -> str:
        """Classify the source type based on URL."""
        if any(domain in url for domain in [".edu", "scholar", "academic"]):
            return "academic"
        elif any(domain in url for domain in ["news", "times", "post", "journal"]):
            return "news"
        elif any(domain in url for domain in ["twitter", "facebook", "reddit"]):
            return "social"
        else:
            return "blog"
    
    def _generate_search_queries(self, claim: str, keywords: List[str]) -> List[str]:
        """Generate multiple search queries from claim and keywords."""
        queries = []
        
        # Base claim query
        queries.append(claim)
        
        # Claim + keyword combinations
        for keyword in keywords[:3]:
            queries.append(f"{claim} {keyword}")
        
        # Keyword combinations
        if len(keywords) >= 2:
            queries.append(" ".join(keywords[:2]))
        
        return queries


# Factory function
async def create_async_evidence_collector(
    config: Optional[Dict[str, Any]] = None
) -> AsyncEvidenceCollector:
    """Create and initialize an async evidence collector."""
    collector = AsyncEvidenceCollector(config)
    await collector._initialize()
    return collector