"""
Production secrets management with Google Cloud Secret Manager integration.

This module provides secure handling of sensitive configuration values:
- Local development: Uses .env files and environment variables
- Production: Integrates with Google Cloud Secret Manager
- Fallback: Environment variables as backup
- Security: Prevents secrets from being logged or exposed
"""

import logging
import os
import warnings
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, Dict, Optional, List

logger = logging.getLogger(__name__)


class SecretsProvider(ABC):
    """Abstract base class for secrets providers."""
    
    @abstractmethod
    def get_secret(self, secret_name: str) -> Optional[str]:
        """Retrieve a secret value."""
        pass
    
    @abstractmethod
    def get_multiple_secrets(self, secret_names: List[str]) -> Dict[str, Optional[str]]:
        """Retrieve multiple secret values."""
        pass


class EnvironmentSecretsProvider(SecretsProvider):
    """Local development secrets provider using environment variables."""
    
    def get_secret(self, secret_name: str) -> Optional[str]:
        """Get secret from environment variables."""
        return os.getenv(secret_name)
    
    def get_multiple_secrets(self, secret_names: List[str]) -> Dict[str, Optional[str]]:
        """Get multiple secrets from environment variables."""
        return {name: os.getenv(name) for name in secret_names}


class GoogleCloudSecretsProvider(SecretsProvider):
    """Production secrets provider using Google Cloud Secret Manager."""
    
    def __init__(self, project_id: Optional[str] = None):
        """Initialize Google Cloud Secret Manager client."""
        self.project_id = project_id or os.getenv('GOOGLE_CLOUD_PROJECT')
        if not self.project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT environment variable is required for Google Cloud secrets")
        
        try:
            from google.cloud import secretmanager
            self.client = secretmanager.SecretManagerServiceClient()
        except ImportError as e:
            raise ImportError(
                "google-cloud-secret-manager is required for production secrets management. "
                "Install with: pip install google-cloud-secret-manager"
            ) from e
    
    def get_secret(self, secret_name: str) -> Optional[str]:
        """Get secret from Google Cloud Secret Manager."""
        try:
            secret_path = f"projects/{self.project_id}/secrets/{secret_name}/versions/latest"
            response = self.client.access_secret_version(request={"name": secret_path})
            return response.payload.data.decode('UTF-8')
        except Exception as e:
            logger.warning(f"Failed to retrieve secret '{secret_name}' from Google Cloud: {e}")
            # Fallback to environment variable
            return os.getenv(secret_name)
    
    def get_multiple_secrets(self, secret_names: List[str]) -> Dict[str, Optional[str]]:
        """Get multiple secrets from Google Cloud Secret Manager."""
        results = {}
        for secret_name in secret_names:
            results[secret_name] = self.get_secret(secret_name)
        return results


class SecretsManager:
    """Centralized secrets management with multiple provider support."""
    
    def __init__(self, environment: str = "development"):
        """Initialize secrets manager with appropriate provider."""
        self.environment = environment
        self.provider = self._get_provider()
        
        # Cache for secrets to avoid repeated API calls
        self._secret_cache: Dict[str, str] = {}
        
        logger.info(f"Initialized secrets manager for environment: {environment}")
    
    def _get_provider(self) -> SecretsProvider:
        """Select appropriate secrets provider based on environment."""
        if self.environment == "production":
            try:
                return GoogleCloudSecretsProvider()
            except (ValueError, ImportError) as e:
                logger.warning(f"Failed to initialize Google Cloud secrets provider: {e}")
                logger.warning("Falling back to environment variables")
                return EnvironmentSecretsProvider()
        else:
            return EnvironmentSecretsProvider()
    
    def get_secret(self, secret_name: str, required: bool = False) -> Optional[str]:
        """
        Get a secret value with caching.
        
        Args:
            secret_name: Name of the secret to retrieve
            required: If True, raises ValueError if secret is not found
            
        Returns:
            Secret value or None if not found
            
        Raises:
            ValueError: If required=True and secret is not found
        """
        # Check cache first
        if secret_name in self._secret_cache:
            return self._secret_cache[secret_name]
        
        # Retrieve from provider
        secret_value = self.provider.get_secret(secret_name)
        
        if secret_value is None and required:
            raise ValueError(f"Required secret '{secret_name}' not found")
        
        # Cache the result (even if None to avoid repeated failed lookups)
        if secret_value is not None:
            self._secret_cache[secret_name] = secret_value
        
        return secret_value
    
    def get_database_url(self) -> str:
        """Get database connection URL with secrets."""
        db_host = self.get_secret("DB_HOST") or "localhost"
        db_port = self.get_secret("DB_PORT") or "5432"
        db_name = self.get_secret("DB_NAME") or "startup_studio"
        db_user = self.get_secret("DB_USER") or "postgres"
        db_password = self.get_secret("DB_PASSWORD", required=self.environment == "production")
        
        if db_password:
            return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        else:
            return f"postgresql://{db_user}@{db_host}:{db_port}/{db_name}"
    
    def get_openai_api_key(self) -> str:
        """Get OpenAI API key."""
        api_key = self.get_secret("OPENAI_API_KEY", required=True)
        return api_key
    
    def get_app_secret_key(self) -> str:
        """Get application secret key for sessions/JWT."""
        secret_key = self.get_secret("SECRET_KEY", required=self.environment == "production")
        if not secret_key:
            if self.environment == "production":
                raise ValueError("SECRET_KEY is required for production")
            else:
                # Generate a warning for development
                warnings.warn(
                    "SECRET_KEY not set - using insecure default for development only",
                    UserWarning
                )
                return "dev-secret-key-not-for-production"
        return secret_key
    
    def validate_required_secrets(self) -> List[str]:
        """
        Validate that all required secrets are available.
        
        Returns:
            List of missing required secrets
        """
        required_secrets = [
            "SECRET_KEY",
            "DB_PASSWORD",
            "OPENAI_API_KEY"
        ]
        
        missing_secrets = []
        for secret_name in required_secrets:
            if self.get_secret(secret_name) is None:
                missing_secrets.append(secret_name)
        
        return missing_secrets
    
    def clear_cache(self):
        """Clear the secrets cache (useful for testing or refreshing)."""
        self._secret_cache.clear()
        logger.debug("Secrets cache cleared")


@lru_cache()
def get_secrets_manager(environment: str = None) -> SecretsManager:
    """
    Get singleton secrets manager instance.
    
    Args:
        environment: Environment name (development, staging, production)
                    If None, uses ENVIRONMENT env var or defaults to 'development'
    """
    if environment is None:
        environment = os.getenv("ENVIRONMENT", "development")
    
    return SecretsManager(environment)


def setup_secrets_from_file(secrets_file: str = ".env.secrets"):
    """
    Load secrets from a file into environment variables for development.
    
    This is a helper function for local development only.
    Never use this in production.
    
    Args:
        secrets_file: Path to secrets file
    """
    if not os.path.exists(secrets_file):
        logger.warning(f"Secrets file {secrets_file} not found")
        return
    
    try:
        from dotenv import load_dotenv
        load_dotenv(secrets_file)
        logger.info(f"Loaded secrets from {secrets_file}")
    except ImportError:
        logger.warning("python-dotenv not available - install with: pip install python-dotenv")
    except Exception as e:
        logger.error(f"Failed to load secrets from {secrets_file}: {e}")


# Security utility functions
def mask_secret(secret: str, visible_chars: int = 4) -> str:
    """
    Mask a secret value for logging/display.
    
    Args:
        secret: Secret value to mask
        visible_chars: Number of characters to show at the end
        
    Returns:
        Masked secret string
    """
    if not secret or len(secret) <= visible_chars:
        return "*" * len(secret) if secret else ""
    
    return "*" * (len(secret) - visible_chars) + secret[-visible_chars:]


def validate_secret_format(secret: str, min_length: int = 8) -> bool:
    """
    Validate secret format for basic security requirements.
    
    Args:
        secret: Secret to validate
        min_length: Minimum required length
        
    Returns:
        True if secret meets basic requirements
    """
    if not secret or len(secret) < min_length:
        return False
    
    # Add more validation rules as needed
    return True