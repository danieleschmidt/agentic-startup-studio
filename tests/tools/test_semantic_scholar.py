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
    search_papers_async, get_paper_details_async,
    SemanticScholarConfig, SemanticScholarAdapter
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






class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_config_post_init(self):
        """Test configuration post-initialization."""
        config = SemanticScholarConfig(paper_fields=None)
        
        # Should set default fields
        assert config.paper_fields is not None
        assert len(config.paper_fields) == 10
    


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