"""
Tests for the secrets management system.

Covers both local development (environment variables) and 
production (Google Cloud Secret Manager) configurations.
"""

import os
import pytest
from unittest.mock import MagicMock, patch

from pipeline.config.secrets_manager import (
    EnvironmentSecretsProvider,
    GoogleCloudSecretsProvider,
    SecretsManager,
    get_secrets_manager,
    mask_secret,
    validate_secret_format
)


class TestEnvironmentSecretsProvider:
    """Test environment variables secrets provider."""
    
    def test_get_secret_exists(self):
        """Test getting an existing environment variable."""
        with patch.dict(os.environ, {"TEST_SECRET": "test_value"}):
            provider = EnvironmentSecretsProvider()
            assert provider.get_secret("TEST_SECRET") == "test_value"
    
    def test_get_secret_not_exists(self):
        """Test getting a non-existent environment variable."""
        provider = EnvironmentSecretsProvider()
        assert provider.get_secret("NON_EXISTENT_SECRET") is None
    
    def test_get_multiple_secrets(self):
        """Test getting multiple secrets at once."""
        with patch.dict(os.environ, {"SECRET1": "value1", "SECRET2": "value2"}):
            provider = EnvironmentSecretsProvider()
            result = provider.get_multiple_secrets(["SECRET1", "SECRET2", "SECRET3"])
            
            assert result == {
                "SECRET1": "value1",
                "SECRET2": "value2", 
                "SECRET3": None
            }


class TestGoogleCloudSecretsProvider:
    """Test Google Cloud Secret Manager provider."""
    
    def test_init_missing_project_id(self):
        """Test initialization without project ID fails."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="GOOGLE_CLOUD_PROJECT"):
                GoogleCloudSecretsProvider()
    
    @patch('google.cloud.secretmanager.SecretManagerServiceClient')
    def test_init_with_project_id(self, mock_client):
        """Test successful initialization with project ID."""
        with patch.dict(os.environ, {"GOOGLE_CLOUD_PROJECT": "test-project"}):
            provider = GoogleCloudSecretsProvider()
            assert provider.project_id == "test-project"
            mock_client.assert_called_once()
    
    @patch('google.cloud.secretmanager.SecretManagerServiceClient')
    def test_get_secret_success(self, mock_client_class):
        """Test successful secret retrieval."""
        # Mock the secret manager client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.payload.data.decode.return_value = "secret_value"
        mock_client.access_secret_version.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        with patch.dict(os.environ, {"GOOGLE_CLOUD_PROJECT": "test-project"}):
            provider = GoogleCloudSecretsProvider()
            result = provider.get_secret("test_secret")
            
            assert result == "secret_value"
            mock_client.access_secret_version.assert_called_once()
    
    @patch('google.cloud.secretmanager.SecretManagerServiceClient')
    def test_get_secret_fallback_to_env(self, mock_client_class):
        """Test fallback to environment variable when GCloud fails."""
        # Mock the secret manager to raise an exception
        mock_client = MagicMock()
        mock_client.access_secret_version.side_effect = Exception("API Error")
        mock_client_class.return_value = mock_client
        
        with patch.dict(os.environ, {"GOOGLE_CLOUD_PROJECT": "test-project", "TEST_SECRET": "env_value"}):
            provider = GoogleCloudSecretsProvider()
            result = provider.get_secret("TEST_SECRET")
            
            assert result == "env_value"


class TestSecretsManager:
    """Test the main SecretsManager class."""
    
    def test_development_uses_environment_provider(self):
        """Test development environment uses environment provider."""
        manager = SecretsManager("development")
        assert isinstance(manager.provider, EnvironmentSecretsProvider)
    
    @patch('pipeline.config.secrets_manager.GoogleCloudSecretsProvider')
    def test_production_uses_google_cloud_provider(self, mock_gcp_provider):
        """Test production environment uses Google Cloud provider."""
        manager = SecretsManager("production")
        mock_gcp_provider.assert_called_once()
    
    def test_get_secret_caching(self):
        """Test that secrets are cached after first retrieval."""
        with patch.dict(os.environ, {"TEST_SECRET": "cached_value"}):
            manager = SecretsManager("development")
            
            # First call should hit the provider
            result1 = manager.get_secret("TEST_SECRET")
            assert result1 == "cached_value"
            
            # Second call should come from cache
            with patch.object(manager.provider, 'get_secret') as mock_get:
                result2 = manager.get_secret("TEST_SECRET")
                assert result2 == "cached_value"
                mock_get.assert_not_called()
    
    def test_get_secret_required_missing(self):
        """Test that required missing secrets raise ValueError."""
        manager = SecretsManager("development")
        
        with pytest.raises(ValueError, match="Required secret 'MISSING_SECRET' not found"):
            manager.get_secret("MISSING_SECRET", required=True)
    
    def test_get_database_url(self):
        """Test database URL construction."""
        env_vars = {
            "DB_HOST": "testhost",
            "DB_PORT": "5433",
            "DB_NAME": "testdb",
            "DB_USER": "testuser",
            "DB_PASSWORD": "testpass"
        }
        
        with patch.dict(os.environ, env_vars):
            manager = SecretsManager("development")
            url = manager.get_database_url()
            
            assert url == "postgresql://testuser:testpass@testhost:5433/testdb"
    
    def test_get_database_url_no_password(self):
        """Test database URL construction without password."""
        env_vars = {
            "DB_HOST": "testhost",
            "DB_PORT": "5433", 
            "DB_NAME": "testdb",
            "DB_USER": "testuser"
        }
        
        with patch.dict(os.environ, env_vars):
            manager = SecretsManager("development")
            url = manager.get_database_url()
            
            assert url == "postgresql://testuser@testhost:5433/testdb"


class TestSecurityUtilities:
    """Test security utility functions."""
    
    def test_mask_secret(self):
        """Test secret masking functionality."""
        assert mask_secret("supersecretkey", 4) == "**********tkey"
        assert mask_secret("short") == "*hort"  # Default visible_chars=4, string has 5 chars
        assert mask_secret("") == ""
        assert mask_secret("abc", 4) == "***"  # String shorter than visible_chars, all masked
    
    def test_validate_secret_format(self):
        """Test secret format validation."""
        assert validate_secret_format("longenoughsecret") is True
        assert validate_secret_format("short") is False
        assert validate_secret_format("") is False
        assert validate_secret_format("verylongsecretkey", min_length=16) is True


class TestGetSecretsManager:
    """Test the get_secrets_manager factory function."""
    
    def test_get_secrets_manager_cached(self):
        """Test that get_secrets_manager returns cached instances."""
        manager1 = get_secrets_manager("development")
        manager2 = get_secrets_manager("development")
        
        assert manager1 is manager2
    
    def test_get_secrets_manager_different_environments(self):
        """Test that different environments get different managers."""
        dev_manager = get_secrets_manager("development")
        prod_manager = get_secrets_manager("production")
        
        assert dev_manager is not prod_manager
    
    def test_get_secrets_manager_default_environment(self):
        """Test default environment when none specified."""
        with patch.dict(os.environ, {"ENVIRONMENT": "staging"}):
            manager = get_secrets_manager()
            assert manager.environment == "staging"
    
    def test_get_secrets_manager_fallback_default(self):
        """Test fallback to development when ENVIRONMENT not set."""
        # Clear the cache first to ensure fresh instance
        get_secrets_manager.cache_clear()
        
        with patch.dict(os.environ, {}, clear=True):
            manager = get_secrets_manager()
            assert manager.environment == "development"