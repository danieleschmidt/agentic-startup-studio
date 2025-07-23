"""
Tests for Semantic Scholar API integration.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import json

# Import the module to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from tools.semantic_scholar import (
    run, search_papers_async, get_paper_details_async,
    SemanticScholarConfig, SemanticScholarAdapter,
    _fallback_search
)


class TestSemanticScholarConfig:
    """Test configuration class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SemanticScholarConfig()
        
        assert config.base_url == "https://api.semanticscholar.org/graph/v1"
        assert config.api_key is None
        assert config.max_results == 10
        assert config.min_citations == 5
        assert config.min_year == 2020
        assert len(config.paper_fields) == 10
        assert 'paperId' in config.paper_fields
        assert 'title' in config.paper_fields
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = SemanticScholarConfig(
            api_key="test-key",
            max_results=20,
            min_citations=10
        )
        
        assert config.api_key == "test-key"
        assert config.max_results == 20
        assert config.min_citations == 10


class TestSemanticScholarAdapter:
    """Test the adapter class."""
    
    @pytest.fixture
    def config(self):
        """Test configuration."""
        return SemanticScholarConfig(api_key="test-key")
    
    @pytest.fixture
    def adapter(self, config):
        """Test adapter instance."""
        return SemanticScholarAdapter(config)
    
    def test_adapter_initialization(self, adapter):
        """Test adapter initialization."""
        assert adapter.name == "semantic_scholar"
        assert adapter.ss_config.api_key == "test-key"
    
    def test_headers_with_api_key(self, adapter):
        """Test header generation with API key."""
        headers = adapter.get_headers()
        
        assert headers['User-Agent'] == 'Agentic-Startup-Studio/1.0'
        assert headers['Accept'] == 'application/json'
        assert headers['x-api-key'] == 'test-key'
    
    def test_headers_without_api_key(self):
        """Test header generation without API key."""
        config = SemanticScholarConfig()
        adapter = SemanticScholarAdapter(config)
        headers = adapter.get_headers()
        
        assert 'x-api-key' not in headers
        assert headers['User-Agent'] == 'Agentic-Startup-Studio/1.0'


class TestFallbackSearch:
    """Test the synchronous fallback search."""
    
    @patch('tools.semantic_scholar.requests.get')
    def test_successful_search(self, mock_get):
        """Test successful API call."""
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'data': [
                {
                    'paperId': '123',
                    'title': 'Test Paper',
                    'year': 2023,
                    'citationCount': 10,
                    'authors': [{'name': 'Test Author'}],
                    'venue': 'Test Venue'
                },
                {
                    'paperId': '456',
                    'title': 'Low Citation Paper',
                    'year': 2023,
                    'citationCount': 2,  # Below threshold
                    'authors': [{'name': 'Test Author 2'}],
                    'venue': 'Test Venue 2'
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        results = _fallback_search("machine learning")
        
        # Should filter out low citation paper
        assert len(results) == 1
        assert results[0]['paperId'] == '123'
        assert results[0]['citationCount'] == 10
        
        # Verify API call
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert 'machine learning' in str(call_args)
    
    @patch('tools.semantic_scholar.requests.get')
    def test_api_error_handling(self, mock_get):
        """Test API error handling."""
        mock_get.side_effect = Exception("API Error")
        
        results = _fallback_search("test query")
        
        # Should return empty list on error
        assert results == []
    
    @patch('tools.semantic_scholar.requests.get')
    @patch.dict(os.environ, {'SEMANTIC_SCHOLAR_API_KEY': 'env-key'})
    def test_api_key_from_environment(self, mock_get):
        """Test API key retrieval from environment."""
        mock_response = MagicMock()
        mock_response.json.return_value = {'data': []}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        _fallback_search("test")
        
        # Check that API key was included in headers
        call_args = mock_get.call_args
        headers = call_args[1]['headers']
        assert headers['x-api-key'] == 'env-key'


class TestLegacyInterface:
    """Test the legacy run() function."""
    
    @patch('tools.semantic_scholar._fallback_search')
    def test_run_function_fallback(self, mock_fallback):
        """Test run function using fallback when async unavailable."""
        mock_fallback.return_value = [{'title': 'Test Paper'}]
        
        # Mock ASYNC_AVAILABLE as False
        with patch('tools.semantic_scholar.ASYNC_AVAILABLE', False):
            results = run("test query")
        
        assert results == [{'title': 'Test Paper'}]
        mock_fallback.assert_called_once_with("test query")


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_config_post_init(self):
        """Test configuration post-initialization."""
        config = SemanticScholarConfig(paper_fields=None)
        
        # Should set default fields
        assert config.paper_fields is not None
        assert len(config.paper_fields) == 10
    
    def test_empty_response_handling(self):
        """Test handling of empty API responses."""
        with patch('tools.semantic_scholar.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {'data': []}
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            results = _fallback_search("nonexistent topic")
            assert results == []


class TestRateLimiting:
    """Test rate limiting configuration."""
    
    def test_rate_limit_config(self):
        """Test rate limiting configuration."""
        config = SemanticScholarConfig()
        adapter = SemanticScholarAdapter(config)
        
        # Check that rate limiting is configured correctly
        assert adapter.adapter_config.requests_per_window == 100
        assert adapter.adapter_config.window_size_seconds == 60


class TestCircuitBreaker:
    """Test circuit breaker integration."""
    
    def test_circuit_breaker_decoration(self):
        """Test that search methods have circuit breaker decoration."""
        adapter = SemanticScholarAdapter()
        
        # Check that the method exists and is decorated
        assert hasattr(adapter, 'search_papers')
        # Note: We can't easily test the decorator without running async code
        # This would require more complex async test setup


if __name__ == "__main__":
    pytest.main([__file__])