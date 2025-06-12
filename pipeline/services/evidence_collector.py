"""
Evidence Collection Service - RAG-based research with citation verification.

Implements multi-domain evidence collection using RAG methodology with:
- Multiple search engine integration with rate limiting
- Citation verification and accessibility checks
- Evidence quality scoring with configurable rubrics
- Retry logic and fallback strategies for resilience
"""

import logging
import asyncio
import aiohttp
import ipaddress
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urlparse
import re
import socket

from pipeline.config.settings import get_settings
from pipeline.services.budget_sentinel import get_budget_sentinel, BudgetCategory, BudgetExceededException


class SearchEngine(Enum):
    """Supported search engines."""
    DUCKDUCKGO = "duckduckgo"
    BING = "bing"
    SEARX = "searx"


class EvidenceType(Enum):
    """Types of evidence sources."""
    ACADEMIC = "academic"
    NEWS = "news"
    BLOG = "blog"
    COMPANY = "company"
    GOVERNMENT = "government"
    UNKNOWN = "unknown"


@dataclass
class SearchResult:
    """Individual search result."""
    title: str
    url: str
    snippet: str
    source: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Evidence:
    """Evidence item with quality metrics."""
    claim_text: str
    citation_url: str
    citation_title: str
    citation_source: str
    snippet: str
    
    # Quality scores (0.0 to 1.0)
    relevance_score: float = 0.0
    credibility_score: float = 0.0
    recency_score: float = 0.0
    accessibility_score: float = 0.0
    
    # Metadata
    evidence_type: EvidenceType = EvidenceType.UNKNOWN
    collected_at: datetime = field(default_factory=datetime.utcnow)
    collected_by_agent: str = "evidence_collector"
    verified: bool = False
    
    @property
    def composite_score(self) -> float:
        """Calculate weighted composite quality score."""
        weights = {
            'credibility': 0.40,
            'relevance': 0.30,
            'accessibility': 0.20,
            'recency': 0.10
        }
        
        return (
            self.credibility_score * weights['credibility'] +
            self.relevance_score * weights['relevance'] +
            self.accessibility_score * weights['accessibility'] +
            self.recency_score * weights['recency']
        )


@dataclass
class ResearchDomain:
    """Research domain configuration."""
    name: str
    keywords: List[str]
    min_evidence_count: int = 3
    quality_threshold: float = 0.7


class EvidenceCollector:
    """RAG-based evidence collection service."""
    
    def __init__(self):
        self.settings = get_settings()
        self.budget_sentinel = get_budget_sentinel()
        self.logger = logging.getLogger(__name__)
        
        # Search engine configurations
        self.search_engines = {
            SearchEngine.DUCKDUCKGO: self._search_duckduckgo,
            SearchEngine.BING: self._search_bing,
            SearchEngine.SEARX: self._search_searx,
        }
        
        # Evidence type patterns for classification
        self.evidence_patterns = {
            EvidenceType.ACADEMIC: [
                r'\.edu/', r'arxiv\.org', r'scholar\.google', r'pubmed',
                r'researchgate', r'academia\.edu', r'jstor'
            ],
            EvidenceType.NEWS: [
                r'bbc\.com', r'cnn\.com', r'reuters\.com', r'ap\.org',
                r'nytimes\.com', r'washingtonpost\.com', r'wsj\.com'
            ],
            EvidenceType.GOVERNMENT: [
                r'\.gov/', r'\.mil/', r'europa\.eu', r'un\.org'
            ],
            EvidenceType.COMPANY: [
                r'about\.', r'blog\.', r'press\.', r'investor\.',
                r'/about', r'/blog', r'/press', r'/news'
            ]
        }
        
        # Citation cache for duplicate detection
        self.citation_cache: Dict[str, Evidence] = {}
        
        # Rate limiting tracking
        self.request_counts: Dict[SearchEngine, List[datetime]] = {
            engine: [] for engine in SearchEngine
        }
    
    async def collect_evidence(
        self,
        claim: str,
        research_domains: List[ResearchDomain],
        min_total_evidence: int = 3,
        timeout: int = 300
    ) -> Dict[str, List[Evidence]]:
        """
        Collect evidence across multiple research domains.
        
        Args:
            claim: The claim to find evidence for
            research_domains: List of domains to research
            min_total_evidence: Minimum total evidence items required
            timeout: Timeout in seconds
            
        Returns:
            Dict mapping domain names to evidence lists
        """
        self.logger.info(f"Starting evidence collection for claim: {claim[:100]}...")
        
        try:
            # Track cost for evidence collection operation
            estimated_cost = len(research_domains) * 0.50  # Estimate per domain
            
            async with self.budget_sentinel.track_operation(
                "evidence_collector",
                "collect_evidence",
                BudgetCategory.EXTERNAL_APIS,
                estimated_cost
            ):
                # Collect evidence from all domains in parallel
                domain_tasks = [
                    self._collect_domain_evidence(claim, domain, timeout // len(research_domains))
                    for domain in research_domains
                ]
                
                domain_results = await asyncio.gather(*domain_tasks, return_exceptions=True)
                
                # Process results and handle exceptions
                evidence_by_domain = {}
                total_evidence_count = 0
                
                for domain, result in zip(research_domains, domain_results):
                    if isinstance(result, Exception):
                        self.logger.error(f"Domain {domain.name} collection failed: {result}")
                        evidence_by_domain[domain.name] = []
                    else:
                        evidence_by_domain[domain.name] = result
                        total_evidence_count += len(result)
                
                # Validate minimum evidence requirements
                if total_evidence_count < min_total_evidence:
                    self.logger.warning(
                        f"Collected {total_evidence_count} evidence items, "
                        f"minimum required: {min_total_evidence}"
                    )
                
                # Log collection summary
                self.logger.info(
                    f"Evidence collection completed: {total_evidence_count} total items "
                    f"across {len(research_domains)} domains"
                )
                
                return evidence_by_domain
                
        except BudgetExceededException as e:
            self.logger.error(f"Evidence collection blocked by budget: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Evidence collection failed: {e}")
            raise
    
    async def _collect_domain_evidence(
        self,
        claim: str,
        domain: ResearchDomain,
        timeout: int
    ) -> List[Evidence]:
        """Collect evidence for a specific research domain."""
        self.logger.info(f"Collecting evidence for domain: {domain.name}")
        
        evidence_items = []
        search_queries = self._generate_search_queries(claim, domain.keywords)
        
        # Search using multiple engines with fallback
        for query in search_queries[:3]:  # Limit to top 3 queries per domain
            try:
                # Try primary search engine first
                results = await self._search_with_fallback(query, max_results=5)
                
                # Convert search results to evidence
                for result in results:
                    evidence = await self._create_evidence_from_result(claim, result)
                    if evidence and evidence.composite_score >= domain.quality_threshold:
                        evidence_items.append(evidence)
                        
                        # Stop if we have enough quality evidence
                        if len(evidence_items) >= domain.min_evidence_count:
                            break
                            
            except Exception as e:
                self.logger.warning(f"Search query failed: {query[:50]}... Error: {e}")
                continue
            
            # Respect rate limits between queries
            await asyncio.sleep(1)
        
        # Sort by composite score (highest first)
        evidence_items.sort(key=lambda x: x.composite_score, reverse=True)
        
        self.logger.info(
            f"Domain {domain.name}: collected {len(evidence_items)} evidence items"
        )
        
        return evidence_items[:domain.min_evidence_count]  # Return top N items
    
    def _generate_search_queries(self, claim: str, keywords: List[str]) -> List[str]:
        """Generate targeted search queries for the claim and domain with input sanitization."""
        # SECURITY: Sanitize and validate inputs to prevent injection attacks
        sanitized_claim = self._sanitize_search_input(claim)
        sanitized_keywords = [self._sanitize_search_input(kw) for kw in keywords if kw]
        
        if not sanitized_claim.strip():
            self.logger.warning("Empty or invalid claim after sanitization")
            return []
        
        # Extract key terms from sanitized claim
        claim_terms = re.findall(r'\b\w{4,}\b', sanitized_claim.lower())
        claim_terms = [term for term in claim_terms if self._is_safe_search_term(term)]
        
        queries = []
        
        # Base query with sanitized claim
        queries.append(sanitized_claim)
        
        # Combine claim terms with sanitized domain keywords
        for keyword in sanitized_keywords[:3]:  # Limit to top 3 keywords
            if not keyword.strip():
                continue
            for term in claim_terms[:2]:  # Limit to top 2 claim terms
                queries.append(f"{term} {keyword}")
        
        # Add evidence-specific queries with sanitized claim
        evidence_queries = [
            f"{sanitized_claim} study research",
            f"{sanitized_claim} statistics data",
            f"{sanitized_claim} analysis report"
        ]
        queries.extend(evidence_queries)
        
        # Final validation of all queries
        safe_queries = [q for q in queries if self._validate_search_query(q)]
        
        return safe_queries[:8]  # Limit total queries
    
    def _sanitize_search_input(self, input_text: str) -> str:
        """Sanitize search input to prevent injection attacks."""
        if not input_text or not isinstance(input_text, str):
            return ""
        
        # Remove dangerous characters and sequences
        sanitized = input_text.strip()
        
        # Remove potential injection patterns
        dangerous_patterns = [
            r'<script[^>]*>.*?</script>',  # Script tags
            r'javascript:',               # JavaScript protocol
            r'data:',                    # Data protocol
            r'vbscript:',               # VBScript protocol
            r'on\w+\s*=',               # Event handlers
            r'[\x00-\x1f\x7f-\x9f]',   # Control characters
            r'[<>"\'\\\x00]',           # Dangerous chars
            r'\beval\b',                # eval function
            r'\bexec\b',                # exec function
            r'\bfunction\b',            # function keyword
            r'\bselect\b.*\bfrom\b',    # SQL injection
            r'\bunion\b.*\bselect\b',   # SQL union
            r'\bdrop\b.*\btable\b',     # SQL drop
            r'\binsert\b.*\binto\b',    # SQL insert
            r'\bupdate\b.*\bset\b',     # SQL update
            r'\bdelete\b.*\bfrom\b',    # SQL delete
        ]
        
        for pattern in dangerous_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
        
        # Limit length to prevent DoS
        max_length = 500
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
            self.logger.warning(f"Search input truncated to {max_length} characters")
        
        return sanitized
    
    def _is_safe_search_term(self, term: str) -> bool:
        """Validate that search term is safe to use."""
        if not term or len(term) < 2:
            return False
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'^[\d\W]+$',           # Only numbers/special chars
            r'^\s*$',               # Only whitespace
            r'[<>"\'\\\x00-\x1f]', # Dangerous characters
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, term):
                return False
        
        return True
    
    def _validate_search_query(self, query: str) -> bool:
        """Final validation of complete search query."""
        if not query or not isinstance(query, str):
            return False
        
        query = query.strip()
        
        # Basic length check
        if len(query) < 1 or len(query) > 1000:
            return False
        
        # Check for remaining dangerous patterns
        if re.search(r'[<>"\'\\\x00-\x1f\x7f-\x9f]', query):
            return False
        
        # Ensure query has actual content
        if not re.search(r'\w', query):
            return False
        
        return True
    
    async def _search_with_fallback(
        self,
        query: str,
        max_results: int = 10
    ) -> List[SearchResult]:
        """Search with fallback to alternative engines."""
        search_order = [SearchEngine.DUCKDUCKGO, SearchEngine.BING, SearchEngine.SEARX]
        
        for engine in search_order:
            try:
                # Check rate limits
                if not self._check_rate_limit(engine):
                    self.logger.warning(f"Rate limit exceeded for {engine.value}")
                    continue
                
                # Perform search
                results = await self.search_engines[engine](query, max_results)
                
                if results:
                    self.logger.debug(f"Search successful with {engine.value}: {len(results)} results")
                    return results
                    
            except Exception as e:
                self.logger.warning(f"Search engine {engine.value} failed: {e}")
                continue
        
        self.logger.error(f"All search engines failed for query: {query[:50]}...")
        return []
    
    def _check_rate_limit(self, engine: SearchEngine, limit: int = 60, window: int = 60) -> bool:
        """Check if rate limit allows request (60 requests per minute by default)."""
        now = datetime.utcnow()
        cutoff = now - timedelta(seconds=window)
        
        # Clean old requests
        self.request_counts[engine] = [
            req_time for req_time in self.request_counts[engine]
            if req_time > cutoff
        ]
        
        # Check if under limit
        if len(self.request_counts[engine]) < limit:
            self.request_counts[engine].append(now)
            return True
    
    def _validate_url_security(self, url: str) -> bool:
        """Validate URL against SSRF attacks by blocking internal networks."""
        try:
            parsed = urlparse(url)
            
            # Only allow HTTP/HTTPS protocols
            if parsed.scheme not in ['http', 'https']:
                self.logger.warning(f"Blocked non-HTTP protocol: {parsed.scheme}")
                return False
            
            # Block URLs without hostname
            if not parsed.hostname:
                self.logger.warning("Blocked URL without hostname")
                return False
            
            # Resolve hostname to IP address
            try:
                ip_str = socket.gethostbyname(parsed.hostname)
                ip = ipaddress.ip_address(ip_str)
            except (socket.gaierror, ValueError) as e:
                self.logger.warning(f"Failed to resolve hostname {parsed.hostname}: {e}")
                return False
            
            # Block private networks (RFC 1918)
            if ip.is_private:
                self.logger.warning(f"Blocked private IP address: {ip}")
                return False
            
            # Block loopback addresses
            if ip.is_loopback:
                self.logger.warning(f"Blocked loopback address: {ip}")
                return False
            
            # Block link-local addresses
            if ip.is_link_local:
                self.logger.warning(f"Blocked link-local address: {ip}")
                return False
            
            # Block multicast addresses
            if ip.is_multicast:
                self.logger.warning(f"Blocked multicast address: {ip}")
                return False
            
            # Block reserved addresses
            if ip.is_reserved:
                self.logger.warning(f"Blocked reserved address: {ip}")
                return False
            
            # Additional IPv4 specific checks
            if isinstance(ip, ipaddress.IPv4Address):
                # Block carrier-grade NAT (100.64.0.0/10)
                if ip in ipaddress.IPv4Network('100.64.0.0/10'):
                    self.logger.warning(f"Blocked carrier-grade NAT address: {ip}")
                    return False
                
                # Block IANA special-use addresses
                special_ranges = [
                    '0.0.0.0/8',      # "This" network
                    '127.0.0.0/8',    # Loopback
                    '169.254.0.0/16', # Link-local
                    '224.0.0.0/4',    # Multicast
                    '240.0.0.0/4',    # Reserved
                ]
                
                for range_str in special_ranges:
                    if ip in ipaddress.IPv4Network(range_str):
                        self.logger.warning(f"Blocked special-use address: {ip} in {range_str}")
                        return False
            
            # Validate port restrictions (if specified)
            if parsed.port:
                # Block common internal service ports
                restricted_ports = {
                    22,    # SSH
                    23,    # Telnet  
                    25,    # SMTP
                    53,    # DNS
                    110,   # POP3
                    135,   # RPC
                    139,   # NetBIOS
                    143,   # IMAP
                    445,   # SMB
                    993,   # IMAPS
                    995,   # POP3S
                    1433,  # SQL Server
                    1521,  # Oracle
                    3306,  # MySQL
                    3389,  # RDP
                    5432,  # PostgreSQL
                    5984,  # CouchDB
                    6379,  # Redis
                    8080,  # Common dev port
                    9200,  # Elasticsearch
                    27017, # MongoDB
                }
                
                if parsed.port in restricted_ports:
                    self.logger.warning(f"Blocked restricted port: {parsed.port}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"URL security validation failed for {url}: {e}")
            return False
        
        return False
    
    async def _create_evidence_from_result(
        self,
        claim: str,
        result: SearchResult
    ) -> Optional[Evidence]:
        """Create evidence item from search result with quality scoring."""
        try:
            # Check for duplicates
            if result.url in self.citation_cache:
                return None
            
            # Verify accessibility
            accessibility_score = await self._verify_accessibility(result.url)
            if accessibility_score < 0.5:  # Skip inaccessible sources
                return None
            
            # Create evidence item
            evidence = Evidence(
                claim_text=claim,
                citation_url=result.url,
                citation_title=result.title,
                citation_source=result.source,
                snippet=result.snippet,
                accessibility_score=accessibility_score
            )
            
            # Score evidence quality
            evidence.relevance_score = self._score_relevance(claim, result)
            evidence.credibility_score = self._score_credibility(result.url)
            evidence.recency_score = self._score_recency(result.timestamp)
            evidence.evidence_type = self._classify_evidence_type(result.url)
            
            # Verify citation (basic check)
            evidence.verified = evidence.accessibility_score > 0.7
            
            # Cache to prevent duplicates
            self.citation_cache[result.url] = evidence
            
            return evidence
            
        except Exception as e:
            self.logger.warning(f"Failed to create evidence from {result.url}: {e}")
            return None
    
    async def _verify_accessibility(self, url: str, timeout: int = 10) -> float:
        """Verify URL accessibility and return accessibility score with SSRF protection."""
        try:
            # CRITICAL SECURITY: Validate URL to prevent SSRF attacks
            if not self._validate_url_security(url):
                self.logger.warning(f"URL blocked by security validation: {url}")
                return 0.0
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.head(url) as response:
                    # Check status code
                    if response.status == 200:
                        return 1.0
                    elif response.status in [301, 302, 303, 307, 308]:  # Redirects
                        return 0.8
                    elif response.status == 403:  # Forbidden but exists
                        return 0.3
                    else:
                        return 0.0
                        
        except asyncio.TimeoutError:
            return 0.2  # Timeout suggests poor accessibility
        except Exception:
            return 0.0  # Connection failed
    
    def _score_relevance(self, claim: str, result: SearchResult) -> float:
        """Score relevance of result to claim (0.0 to 1.0)."""
        claim_terms = set(re.findall(r'\b\w{3,}\b', claim.lower()))
        
        # Check title relevance
        title_terms = set(re.findall(r'\b\w{3,}\b', result.title.lower()))
        title_overlap = len(claim_terms.intersection(title_terms)) / max(len(claim_terms), 1)
        
        # Check snippet relevance
        snippet_terms = set(re.findall(r'\b\w{3,}\b', result.snippet.lower()))
        snippet_overlap = len(claim_terms.intersection(snippet_terms)) / max(len(claim_terms), 1)
        
        # Weighted combination
        relevance = (title_overlap * 0.7) + (snippet_overlap * 0.3)
        return min(relevance, 1.0)
    
    def _score_credibility(self, url: str) -> float:
        """Score source credibility based on domain patterns."""
        domain = urlparse(url).netloc.lower()
        
        # High credibility domains
        if any(pattern in domain for pattern in ['.edu', '.gov', '.mil']):
            return 0.9
        
        # Academic and research platforms
        if any(platform in domain for platform in ['scholar.google', 'pubmed', 'arxiv', 'jstor']):
            return 0.85
        
        # Major news organizations
        if any(news in domain for news in ['bbc.com', 'reuters.com', 'ap.org', 'nytimes.com']):
            return 0.8
        
        # Professional organizations
        if any(prof in domain for prof in ['ieee.org', 'acm.org', 'nature.com', 'science.org']):
            return 0.85
        
        # Default credibility for other domains
        return 0.6
    
    def _score_recency(self, timestamp: datetime) -> float:
        """Score content recency (newer = higher score)."""
        now = datetime.utcnow()
        age_days = (now - timestamp).days
        
        if age_days <= 30:
            return 1.0
        elif age_days <= 180:
            return 0.8
        elif age_days <= 365:
            return 0.6
        elif age_days <= 730:
            return 0.4
        else:
            return 0.2
    
    def _classify_evidence_type(self, url: str) -> EvidenceType:
        """Classify evidence type based on URL patterns."""
        url_lower = url.lower()
        
        for evidence_type, patterns in self.evidence_patterns.items():
            if any(re.search(pattern, url_lower) for pattern in patterns):
                return evidence_type
        
        return EvidenceType.UNKNOWN
    
    # Search engine implementations (placeholder methods)
    async def _search_duckduckgo(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using DuckDuckGo (placeholder implementation)."""
        await asyncio.sleep(0.1)  # Simulate API call
        # In real implementation, would make actual API calls
        return [
            SearchResult(
                title=f"DuckDuckGo Result for {query[:20]}",
                url=f"https://example.com/ddg-{hash(query) % 1000}",
                snippet=f"Sample snippet for {query}",
                source="duckduckgo"
            )
        ]
    
    async def _search_bing(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using Bing (placeholder implementation)."""
        await asyncio.sleep(0.1)  # Simulate API call
        return [
            SearchResult(
                title=f"Bing Result for {query[:20]}",
                url=f"https://example.com/bing-{hash(query) % 1000}",
                snippet=f"Sample snippet for {query}",
                source="bing"
            )
        ]
    
    async def _search_searx(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using SearX (placeholder implementation)."""
        await asyncio.sleep(0.1)  # Simulate API call
        return [
            SearchResult(
                title=f"SearX Result for {query[:20]}",
                url=f"https://example.com/searx-{hash(query) % 1000}",
                snippet=f"Sample snippet for {query}",
                source="searx"
            )
        ]


# Singleton instance
_evidence_collector = None


def get_evidence_collector() -> EvidenceCollector:
    """Get singleton Evidence Collector instance."""
    global _evidence_collector
    if _evidence_collector is None:
        _evidence_collector = EvidenceCollector()
    return _evidence_collector