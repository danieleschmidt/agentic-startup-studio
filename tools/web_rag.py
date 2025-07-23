"""
Web content extraction and processing for RAG (Retrieval-Augmented Generation).

This module provides comprehensive web scraping functionality with proper
HTML parsing, content extraction, and text cleaning for use in RAG systems.
"""

import logging
import re
import asyncio
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
from datetime import datetime

try:
    import aiohttp
    import requests
    from bs4 import BeautifulSoup, NavigableString
    from pipeline.infrastructure.circuit_breaker import circuit_breaker, CircuitBreakerConfig
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    # Fallback imports for minimal functionality
    import requests
    import re
    DEPENDENCIES_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class WebRAGConfig:
    """Configuration for web content extraction."""
    
    # Request settings
    timeout: int = 30
    max_retries: int = 3
    user_agent: str = "Agentic-Startup-Studio/1.0 (Research Bot)"
    max_content_length: int = 1_000_000  # 1MB limit
    
    # Content extraction settings
    max_text_length: int = 10_000
    preserve_links: bool = True
    preserve_structure: bool = True
    extract_metadata: bool = True
    
    # Content filters
    remove_scripts: bool = True
    remove_styles: bool = True
    remove_comments: bool = True
    remove_navigation: bool = True
    
    # Text cleaning
    normalize_whitespace: bool = True
    remove_empty_lines: bool = True
    min_paragraph_length: int = 20


class WebRAGExtractor:
    """Advanced web content extractor for RAG systems."""
    
    def __init__(self, config: Optional[WebRAGConfig] = None):
        """Initialize the web RAG extractor."""
        self.config = config or WebRAGConfig()
        self._session = None
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers for web scraping."""
        return {
            'User-Agent': self.config.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format and scheme."""
        try:
            parsed = urlparse(url)
            return parsed.scheme in ('http', 'https') and bool(parsed.netloc)
        except Exception:
            return False
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""
        
        if self.config.normalize_whitespace:
            # Normalize whitespace and remove extra spaces
            text = re.sub(r'\s+', ' ', text.strip())
        
        if self.config.remove_empty_lines:
            # Remove empty lines and excessive line breaks
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            text = '\n'.join(lines)
        
        # Remove control characters except newlines and tabs
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        return text.strip()
    
    def _extract_metadata(self, soup: 'BeautifulSoup') -> Dict[str, Any]:
        """Extract metadata from HTML document."""
        metadata = {}
        
        try:
            # Title
            title_tag = soup.find('title')
            if title_tag:
                metadata['title'] = title_tag.get_text().strip()
            
            # Meta description
            desc_tag = soup.find('meta', attrs={'name': 'description'})
            if desc_tag:
                metadata['description'] = desc_tag.get('content', '').strip()
            
            # Meta keywords
            keywords_tag = soup.find('meta', attrs={'name': 'keywords'})
            if keywords_tag:
                metadata['keywords'] = keywords_tag.get('content', '').strip()
            
            # Open Graph metadata
            og_tags = soup.find_all('meta', attrs={'property': re.compile(r'^og:')})
            og_data = {}
            for tag in og_tags:
                prop = tag.get('property', '').replace('og:', '')
                content = tag.get('content', '').strip()
                if prop and content:
                    og_data[prop] = content
            if og_data:
                metadata['og'] = og_data
            
            # Language
            html_tag = soup.find('html')
            if html_tag:
                lang = html_tag.get('lang')
                if lang:
                    metadata['language'] = lang
            
            # Publication date (multiple possible selectors)
            date_selectors = [
                'meta[property="article:published_time"]',
                'meta[name="pubdate"]',
                'meta[name="date"]',
                'time[datetime]',
                '.published',
                '.date'
            ]
            
            for selector in date_selectors:
                date_elem = soup.select_one(selector)
                if date_elem:
                    date_text = date_elem.get('content') or date_elem.get('datetime') or date_elem.get_text()
                    if date_text:
                        metadata['published_date'] = date_text.strip()
                        break
            
        except Exception as e:
            logger.warning(f"Error extracting metadata: {e}")
        
        return metadata
    
    def _extract_main_content(self, soup: 'BeautifulSoup') -> str:
        """Extract main content from HTML, filtering out navigation and ads."""
        
        # Remove unwanted elements
        unwanted_selectors = []
        
        if self.config.remove_scripts:
            unwanted_selectors.extend(['script', 'noscript'])
        
        if self.config.remove_styles:
            unwanted_selectors.extend(['style', 'link[rel="stylesheet"]'])
        
        if self.config.remove_comments:
            # Remove HTML comments
            for comment in soup(text=lambda text: isinstance(text, str) and text.strip().startswith('<!--')):
                comment.extract()
        
        if self.config.remove_navigation:
            unwanted_selectors.extend([
                'nav', 'header', 'footer', 'aside',
                '.nav', '.navbar', '.navigation', '.menu',
                '.header', '.footer', '.sidebar', '.aside',
                '.advertisement', '.ad', '.ads', '.banner',
                '.social', '.share', '.comments', '.comment-form'
            ])
        
        # Remove unwanted elements
        for selector in unwanted_selectors:
            for element in soup.select(selector):
                element.decompose()
        
        # Try to find main content area with common selectors
        main_content_selectors = [
            'main', 'article', '.main-content', '.content',
            '.post-content', '.entry-content', '.article-content',
            '.body', '.main', '.primary', '.wrapper'
        ]
        
        main_content = None
        for selector in main_content_selectors:
            element = soup.select_one(selector)
            if element:
                main_content = element
                break
        
        # Fallback to body if no main content found
        if not main_content:
            main_content = soup.find('body') or soup
        
        # Extract text with structure preservation
        if self.config.preserve_structure:
            text_parts = []
            
            # Process headers
            for header in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                text = header.get_text().strip()
                if text:
                    level = int(header.name[1])
                    prefix = '#' * level + ' '
                    text_parts.append(f"\n{prefix}{text}\n")
            
            # Process paragraphs
            for para in main_content.find_all(['p', 'div'], recursive=True):
                text = para.get_text().strip()
                if text and len(text) >= self.config.min_paragraph_length:
                    text_parts.append(text + '\n')
            
            # Process lists
            for list_elem in main_content.find_all(['ul', 'ol']):
                for item in list_elem.find_all('li'):
                    text = item.get_text().strip()
                    if text:
                        text_parts.append(f"• {text}")
                text_parts.append('')  # Add blank line after list
            
            content_text = '\n'.join(text_parts)
        else:
            # Simple text extraction
            content_text = main_content.get_text(separator=' ', strip=True)
        
        return self._clean_text(content_text)
    
    def _extract_links(self, soup: 'BeautifulSoup', base_url: str) -> List[Dict[str, str]]:
        """Extract and normalize links from the page."""
        links = []
        
        try:
            for link in soup.find_all('a', href=True):
                href = link.get('href', '').strip()
                text = link.get_text().strip()
                
                if href and not href.startswith(('#', 'javascript:', 'mailto:')):
                    # Convert relative URLs to absolute
                    full_url = urljoin(base_url, href)
                    
                    if self._is_valid_url(full_url):
                        links.append({
                            'url': full_url,
                            'text': text,
                            'title': link.get('title', '').strip()
                        })
        except Exception as e:
            logger.warning(f"Error extracting links: {e}")
        
        return links
    
    @circuit_breaker("web_rag_extraction")
    async def extract_async(self, url: str) -> Dict[str, Any]:
        """Asynchronously extract content from a web page."""
        if not self._is_valid_url(url):
            raise ValueError(f"Invalid URL: {url}")
        
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                headers=self._get_headers()
            ) as session:
                
                async with session.get(url) as response:
                    # Check content length
                    content_length = response.headers.get('content-length')
                    if content_length and int(content_length) > self.config.max_content_length:
                        raise ValueError(f"Content too large: {content_length} bytes")
                    
                    response.raise_for_status()
                    html_content = await response.text()
                    
                    return self._process_html(html_content, url)
        
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            raise
    
    def extract_sync(self, url: str) -> Dict[str, Any]:
        """Synchronously extract content from a web page."""
        if not self._is_valid_url(url):
            raise ValueError(f"Invalid URL: {url}")
        
        try:
            response = requests.get(
                url,
                headers=self._get_headers(),
                timeout=self.config.timeout,
                stream=True
            )
            
            # Check content length
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > self.config.max_content_length:
                raise ValueError(f"Content too large: {content_length} bytes")
            
            response.raise_for_status()
            html_content = response.text
            
            return self._process_html(html_content, url)
        
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            raise
    
    def _process_html(self, html_content: str, url: str) -> Dict[str, Any]:
        """Process HTML content and extract structured data."""
        if not DEPENDENCIES_AVAILABLE:
            # Fallback to simple text extraction
            return self._fallback_extraction(html_content, url)
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        result = {
            'url': url,
            'timestamp': datetime.now().isoformat(),
            'content': self._extract_main_content(soup),
            'success': True
        }
        
        if self.config.extract_metadata:
            result['metadata'] = self._extract_metadata(soup)
        
        if self.config.preserve_links:
            result['links'] = self._extract_links(soup, url)
        
        # Truncate content if too long
        if len(result['content']) > self.config.max_text_length:
            result['content'] = result['content'][:self.config.max_text_length] + "..."
            result['truncated'] = True
        
        return result
    
    def _fallback_extraction(self, html_content: str, url: str) -> Dict[str, Any]:
        """Fallback extraction using only standard library."""
        # Simple HTML tag removal
        text = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Extract title
        title_match = re.search(r'<title[^>]*>(.*?)</title>', html_content, re.IGNORECASE | re.DOTALL)
        title = title_match.group(1).strip() if title_match else ""
        
        return {
            'url': url,
            'timestamp': datetime.now().isoformat(),
            'content': text[:self.config.max_text_length],
            'metadata': {'title': title} if title else {},
            'success': True,
            'fallback_mode': True
        }


# Global extractor instance
_global_extractor: Optional[WebRAGExtractor] = None


def get_web_rag_extractor() -> WebRAGExtractor:
    """Get or create the global web RAG extractor instance."""
    global _global_extractor
    if _global_extractor is None:
        _global_extractor = WebRAGExtractor()
    return _global_extractor


# Legacy function interface for backward compatibility
def run(url: str) -> str:
    """
    Legacy synchronous interface for backward compatibility.
    
    Args:
        url: URL to extract content from
        
    Returns:
        Extracted text content (for backward compatibility)
    """
    try:
        extractor = get_web_rag_extractor()
        result = extractor.extract_sync(url)
        return result.get('content', '')
    except Exception as e:
        logger.error(f"Error in web RAG extraction: {e}")
        return ""


# Modern async interface
async def extract_content_async(url: str, config: Optional[WebRAGConfig] = None) -> Dict[str, Any]:
    """
    Modern async interface for web content extraction.
    
    Args:
        url: URL to extract content from
        config: Optional configuration for extraction
        
    Returns:
        Dictionary containing extracted content and metadata
    """
    extractor = WebRAGExtractor(config)
    return await extractor.extract_async(url)


def extract_content_sync(url: str, config: Optional[WebRAGConfig] = None) -> Dict[str, Any]:
    """
    Modern sync interface for web content extraction.
    
    Args:
        url: URL to extract content from
        config: Optional configuration for extraction
        
    Returns:
        Dictionary containing extracted content and metadata
    """
    extractor = WebRAGExtractor(config)
    return extractor.extract_sync(url)


# Batch processing utilities
async def extract_multiple_async(urls: List[str], config: Optional[WebRAGConfig] = None) -> List[Dict[str, Any]]:
    """
    Extract content from multiple URLs concurrently.
    
    Args:
        urls: List of URLs to process
        config: Optional configuration for extraction
        
    Returns:
        List of extraction results
    """
    extractor = WebRAGExtractor(config)
    tasks = [extractor.extract_async(url) for url in urls]
    
    results = []
    for task in asyncio.as_completed(tasks):
        try:
            result = await task
            results.append(result)
        except Exception as e:
            logger.error(f"Error in batch extraction: {e}")
            results.append({
                'url': 'unknown',
                'content': '',
                'success': False,
                'error': str(e)
            })
    
    return results
