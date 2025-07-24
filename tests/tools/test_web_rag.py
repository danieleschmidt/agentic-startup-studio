"""
Tests for Web RAG content extraction.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio

# Import the module to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from tools.web_rag import (
    extract_content_async, extract_content_sync,
    WebRAGConfig, WebRAGExtractor, get_web_rag_extractor
)


class TestWebRAGConfig:
    """Test configuration class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = WebRAGConfig()
        
        assert config.timeout == 30
        assert config.max_retries == 3
        assert config.max_content_length == 1_000_000
        assert config.max_text_length == 10_000
        assert config.preserve_links is True
        assert config.preserve_structure is True
        assert config.extract_metadata is True
        assert config.remove_scripts is True
        assert config.normalize_whitespace is True
        assert config.min_paragraph_length == 20
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = WebRAGConfig(
            timeout=60,
            max_text_length=5000,
            preserve_links=False,
            extract_metadata=False
        )
        
        assert config.timeout == 60
        assert config.max_text_length == 5000
        assert config.preserve_links is False
        assert config.extract_metadata is False


class TestWebRAGExtractor:
    """Test the extractor class."""
    
    @pytest.fixture
    def config(self):
        """Test configuration."""
        return WebRAGConfig(timeout=10, max_text_length=1000)
    
    @pytest.fixture
    def extractor(self, config):
        """Test extractor instance."""
        return WebRAGExtractor(config)
    
    def test_extractor_initialization(self, extractor, config):
        """Test extractor initialization."""
        assert extractor.config == config
        assert extractor._session is None
    
    def test_url_validation(self, extractor):
        """Test URL validation."""
        # Valid URLs
        assert extractor._is_valid_url("https://example.com") is True
        assert extractor._is_valid_url("http://example.com/path") is True
        
        # Invalid URLs
        assert extractor._is_valid_url("not-a-url") is False
        assert extractor._is_valid_url("file:///local/path") is False
        assert extractor._is_valid_url("ftp://example.com") is False
        assert extractor._is_valid_url("") is False
    
    def test_get_headers(self, extractor):
        """Test header generation."""
        headers = extractor._get_headers()
        
        assert "User-Agent" in headers
        assert "Agentic-Startup-Studio" in headers["User-Agent"]
        assert headers["Accept"].startswith("text/html")
        assert headers["Accept-Language"] == "en-US,en;q=0.5"
        assert headers["Connection"] == "keep-alive"
    
    def test_text_cleaning(self, extractor):
        """Test text cleaning functionality."""
        # Test whitespace normalization
        dirty_text = "  Multiple   spaces\n\n\nand   lines  "
        clean_text = extractor._clean_text(dirty_text)
        assert clean_text == "Multiple spaces and lines"
        
        # Test empty string handling
        assert extractor._clean_text("") == ""
        assert extractor._clean_text(None) == ""
        
        # Test control character removal
        text_with_control = "Text\x00with\x08control\x1Fchars"
        clean_control = extractor._clean_text(text_with_control)
        assert "\x00" not in clean_control
        assert "\x08" not in clean_control
        assert "\x1F" not in clean_control
    
    @patch('tools.web_rag.requests.get')
    def test_sync_extraction_success(self, mock_get, extractor):
        """Test successful synchronous extraction."""
        # Mock HTML content
        html_content = """
        <html>
            <head>
                <title>Test Page</title>
                <meta name="description" content="Test description">
            </head>
            <body>
                <article>
                    <h1>Main Article</h1>
                    <p>This is the main content of the article with enough text to pass the minimum length requirement.</p>
                    <a href="https://example.com">External link</a>
                </article>
                <nav>Navigation content to be removed</nav>
                <script>alert('script to be removed');</script>
            </body>
        </html>
        """
        
        # Mock response
        mock_response = MagicMock()
        mock_response.text = html_content
        mock_response.headers = {'content-length': '1000'}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = extractor.extract_sync("https://example.com")
        
        assert result['success'] is True
        assert result['url'] == "https://example.com"
        assert 'timestamp' in result
        assert 'Main Article' in result['content']
        assert 'main content of the article' in result['content']
        assert 'Navigation content' not in result['content']  # Should be removed
        assert 'script to be removed' not in result['content']  # Should be removed
        
        # Check metadata extraction
        if extractor.config.extract_metadata:
            assert 'metadata' in result
            assert result['metadata']['title'] == 'Test Page'
            assert result['metadata']['description'] == 'Test description'
        
        # Check links extraction
        if extractor.config.preserve_links:
            assert 'links' in result
            assert len(result['links']) > 0
            assert any(link['url'] == 'https://example.com' for link in result['links'])
    
    @patch('tools.web_rag.requests.get')
    def test_sync_extraction_error_handling(self, mock_get, extractor):
        """Test error handling in synchronous extraction."""
        mock_get.side_effect = Exception("Network error")
        
        with pytest.raises(Exception):
            extractor.extract_sync("https://example.com")
    
    @patch('tools.web_rag.requests.get')
    def test_content_length_limit(self, mock_get, extractor):
        """Test content length limiting."""
        mock_response = MagicMock()
        mock_response.headers = {'content-length': str(extractor.config.max_content_length + 1)}
        mock_get.return_value = mock_response
        
        with pytest.raises(ValueError, match="Content too large"):
            extractor.extract_sync("https://example.com")
    
    def test_invalid_url_handling(self, extractor):
        """Test invalid URL handling."""
        with pytest.raises(ValueError, match="Invalid URL"):
            extractor.extract_sync("not-a-url")
    
    def test_fallback_extraction(self, extractor):
        """Test fallback extraction method."""
        html_content = """
        <html>
            <head><title>Fallback Test</title></head>
            <body>
                <script>alert('remove me');</script>
                <style>.hidden { display: none; }</style>
                <p>This is the main content.</p>
                <p>Another paragraph.</p>
            </body>
        </html>
        """
        
        result = extractor._fallback_extraction(html_content, "https://example.com")
        
        assert result['success'] is True
        assert result['fallback_mode'] is True
        assert 'This is the main content' in result['content']
        assert 'alert(' not in result['content']  # Script should be removed
        assert '.hidden' not in result['content']  # Style should be removed
        assert result['metadata']['title'] == 'Fallback Test'




class TestModernInterfaces:
    """Test modern async and sync interfaces."""
    
    @patch('tools.web_rag.WebRAGExtractor')
    def test_extract_content_sync(self, mock_extractor_class):
        """Test modern sync interface."""
        mock_extractor = MagicMock()
        mock_extractor.extract_sync.return_value = {'content': 'test'}
        mock_extractor_class.return_value = mock_extractor
        
        config = WebRAGConfig(timeout=10)
        result = extract_content_sync("https://example.com", config)
        
        assert result == {'content': 'test'}
        mock_extractor_class.assert_called_once_with(config)
        mock_extractor.extract_sync.assert_called_once_with("https://example.com")


class TestMetadataExtraction:
    """Test metadata extraction functionality."""
    
    @pytest.fixture
    def extractor(self):
        """Test extractor with metadata enabled."""
        config = WebRAGConfig(extract_metadata=True)
        return WebRAGExtractor(config)
    
    def test_metadata_extraction_comprehensive(self, extractor):
        """Test comprehensive metadata extraction."""
        # Mock BeautifulSoup for metadata testing
        try:
            from bs4 import BeautifulSoup
            
            html_content = """
            <html lang="en">
                <head>
                    <title>Test Article Title</title>
                    <meta name="description" content="Test article description">
                    <meta name="keywords" content="test, article, content">
                    <meta property="og:title" content="OG Title">
                    <meta property="og:description" content="OG Description">
                    <meta property="article:published_time" content="2023-01-01T12:00:00Z">
                </head>
                <body>Test content</body>
            </html>
            """
            
            soup = BeautifulSoup(html_content, 'html.parser')
            metadata = extractor._extract_metadata(soup)
            
            assert metadata['title'] == 'Test Article Title'
            assert metadata['description'] == 'Test article description'
            assert metadata['keywords'] == 'test, article, content'
            assert metadata['language'] == 'en'
            assert 'og' in metadata
            assert metadata['og']['title'] == 'OG Title'
            assert metadata['og']['description'] == 'OG Description'
            assert metadata['published_date'] == '2023-01-01T12:00:00Z'
            
        except ImportError:
            # Skip test if BeautifulSoup not available
            pytest.skip("BeautifulSoup not available for metadata testing")


class TestGlobalExtractor:
    """Test global extractor functionality."""
    
    def test_global_extractor_singleton(self):
        """Test that global extractor is a singleton."""
        extractor1 = get_web_rag_extractor()
        extractor2 = get_web_rag_extractor()
        
        assert extractor1 is extractor2  # Should be the same instance


class TestContentFiltering:
    """Test content filtering and cleaning."""
    
    @pytest.fixture
    def extractor(self):
        """Test extractor with all filters enabled."""
        config = WebRAGConfig(
            remove_scripts=True,
            remove_styles=True,
            remove_navigation=True,
            normalize_whitespace=True,
            min_paragraph_length=10
        )
        return WebRAGExtractor(config)
    
    def test_content_filtering(self, extractor):
        """Test that unwanted content is properly filtered."""
        try:
            from bs4 import BeautifulSoup
            
            html_content = """
            <html>
                <head>
                    <script>alert('unwanted');</script>
                    <style>.hidden { display: none; }</style>
                </head>
                <body>
                    <nav>Navigation menu</nav>
                    <header>Page header</header>
                    <main>
                        <article>
                            <h1>Main Content</h1>
                            <p>This is a substantial paragraph with enough content to pass the minimum length filter.</p>
                            <p>Short.</p>  <!-- Should be filtered due to length -->
                        </article>
                    </main>
                    <footer>Page footer</footer>
                    <aside>Sidebar content</aside>
                </body>
            </html>
            """
            
            soup = BeautifulSoup(html_content, 'html.parser')
            content = extractor._extract_main_content(soup)
            
            # Should contain main content
            assert 'Main Content' in content
            assert 'substantial paragraph' in content
            
            # Should not contain filtered elements
            assert 'alert(' not in content
            assert '.hidden' not in content
            assert 'Navigation menu' not in content
            assert 'Page header' not in content
            assert 'Page footer' not in content
            assert 'Sidebar content' not in content
            
        except ImportError:
            # Skip test if BeautifulSoup not available
            pytest.skip("BeautifulSoup not available for content filtering testing")


if __name__ == "__main__":
    pytest.main([__file__])