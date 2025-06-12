"""
Comprehensive test suite for configuration management.

Tests environment variable loading, validation, configuration hierarchies,
overrides, and missing/invalid configuration scenarios.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, Any

from pipeline.config.settings import (
    DatabaseConfig, ValidationConfig, EmbeddingConfig, LoggingConfig,
    IngestionConfig, get_settings, validate_required_env_vars,
    get_config_summary, get_db_config, get_validation_config,
    get_embedding_config, get_logging_config
)


class TestDatabaseConfig:
    """Test database configuration validation and URL generation."""
    
    def test_when_default_values_then_proper_defaults(self):
        """Given no environment variables, when creating DatabaseConfig, then uses proper defaults."""
        config = DatabaseConfig()
        
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "startup_studio"
        assert config.username == "postgres"
        assert config.password == ""
        assert config.min_connections == 1
        assert config.max_connections == 20
        assert config.timeout == 30
        assert config.vector_dimensions == 1536
        assert config.enable_vector_search is True
    
    @patch.dict(os.environ, {
        'DB_HOST': 'custom-host',
        'DB_PORT': '5433',
        'DB_NAME': 'custom_db',
        'DB_USER': 'custom_user',
        'DB_PASSWORD': 'secret123'
    })
    def test_when_env_vars_set_then_overrides_defaults(self):
        """Given environment variables, when creating DatabaseConfig, then overrides defaults."""
        config = DatabaseConfig()
        
        assert config.host == "custom-host"
        assert config.port == 5433
        assert config.database == "custom_db"
        assert config.username == "custom_user"
        assert config.password == "secret123"
    
    def test_when_invalid_port_then_validation_fails(self):
        """Given invalid port number, when creating DatabaseConfig, then raises ValidationError."""
        with pytest.raises(ValueError, match="Database port must be between 1 and 65535"):
            DatabaseConfig(port=0)
        
        with pytest.raises(ValueError, match="Database port must be between 1 and 65535"):
            DatabaseConfig(port=70000)
    
    def test_when_negative_timeout_then_validation_fails(self):
        """Given negative timeout, when creating DatabaseConfig, then raises ValidationError."""
        with pytest.raises(ValueError, match="Database timeout must be positive"):
            DatabaseConfig(timeout=-1)
    
    def test_when_invalid_vector_dimensions_then_validation_fails(self):
        """Given invalid vector dimensions, when creating DatabaseConfig, then raises ValidationError."""
        with pytest.raises(ValueError, match="Vector dimensions must be between 100 and 4096"):
            DatabaseConfig(vector_dimensions=50)
        
        with pytest.raises(ValueError, match="Vector dimensions must be between 100 and 4096"):
            DatabaseConfig(vector_dimensions=5000)
    
    def test_when_get_connection_url_with_password_then_includes_password(self):
        """Given config with password, when getting connection URL, then includes password."""
        config = DatabaseConfig(
            host="test-host",
            port=5432,
            username="testuser",
            password="testpass",
            database="testdb"
        )
        
        url = config.get_connection_url(include_password=True)
        
        assert url == "postgresql://testuser:testpass@test-host:5432/testdb"
    
    def test_when_get_connection_url_without_password_then_excludes_password(self):
        """Given config with password, when getting connection URL without password, then excludes password."""
        config = DatabaseConfig(
            host="test-host",
            port=5432,
            username="testuser",
            password="testpass",
            database="testdb"
        )
        
        url = config.get_connection_url(include_password=False)
        
        assert url == "postgresql://testuser@test-host:5432/testdb"
    
    def test_when_no_password_then_connection_url_excludes_password(self):
        """Given config without password, when getting connection URL, then excludes password."""
        config = DatabaseConfig(
            host="test-host",
            username="testuser",
            database="testdb"
        )
        
        url = config.get_connection_url(include_password=True)
        
        assert url == "postgresql://testuser@test-host:5432/testdb"


class TestValidationConfig:
    """Test validation configuration and business rules."""
    
    def test_when_default_values_then_proper_defaults(self):
        """Given no environment variables, when creating ValidationConfig, then uses proper defaults."""
        config = ValidationConfig()
        
        assert config.min_title_length == 10
        assert config.max_title_length == 200
        assert config.min_description_length == 10
        assert config.max_description_length == 5000
        assert config.enable_profanity_filter is True
        assert config.enable_spam_detection is True
        assert config.enable_html_sanitization is True
        assert config.similarity_threshold == 0.8
        assert config.title_fuzzy_threshold == 0.7
        assert config.max_ideas_per_hour == 10
        assert config.max_ideas_per_day == 50
    
    @patch.dict(os.environ, {
        'MIN_TITLE_LENGTH': '15',
        'MAX_TITLE_LENGTH': '150',
        'SIMILARITY_THRESHOLD': '0.9',
        'ENABLE_PROFANITY_FILTER': 'false'
    })
    def test_when_env_vars_set_then_overrides_defaults(self):
        """Given environment variables, when creating ValidationConfig, then overrides defaults."""
        config = ValidationConfig()
        
        assert config.min_title_length == 15
        assert config.max_title_length == 150
        assert config.similarity_threshold == 0.9
        assert config.enable_profanity_filter is False
    
    def test_when_negative_min_length_then_validation_fails(self):
        """Given negative minimum length, when creating ValidationConfig, then raises ValidationError."""
        with pytest.raises(ValueError, match="Minimum length must be positive"):
            ValidationConfig(min_title_length=-1)
    
    def test_when_invalid_threshold_then_validation_fails(self):
        """Given threshold outside 0-1 range, when creating ValidationConfig, then raises ValidationError."""
        with pytest.raises(ValueError, match="Threshold must be between 0.0 and 1.0"):
            ValidationConfig(similarity_threshold=1.5)
        
        with pytest.raises(ValueError, match="Threshold must be between 0.0 and 1.0"):
            ValidationConfig(title_fuzzy_threshold=-0.1)
    
    def test_when_max_less_than_min_then_validation_fails(self):
        """Given max length less than min length, when creating ValidationConfig, then raises ValidationError."""
        with pytest.raises(ValueError, match="max_title_length must be greater than min_title_length"):
            ValidationConfig(min_title_length=50, max_title_length=30)
        
        with pytest.raises(ValueError, match="max_description_length must be greater than min_description_length"):
            ValidationConfig(min_description_length=100, max_description_length=50)


class TestEmbeddingConfig:
    """Test embedding service configuration and validation."""
    
    def test_when_default_values_then_proper_defaults(self):
        """Given no environment variables, when creating EmbeddingConfig, then uses proper defaults."""
        config = EmbeddingConfig()
        
        assert config.provider == "openai"
        assert config.api_key == ""
        assert config.model_name == "text-embedding-ada-002"
        assert config.enable_cache is True
        assert config.cache_ttl == 86400
        assert config.cache_size == 1000
        assert config.batch_size == 10
        assert config.retry_attempts == 3
        assert config.retry_delay == 1.0
    
    @patch.dict(os.environ, {
        'EMBEDDING_PROVIDER': 'HUGGINGFACE',
        'EMBEDDING_API_KEY': 'test-key-123',
        'EMBEDDING_MODEL': 'custom-model',
        'EMBEDDING_BATCH_SIZE': '20'
    })
    def test_when_env_vars_set_then_overrides_defaults(self):
        """Given environment variables, when creating EmbeddingConfig, then overrides defaults."""
        config = EmbeddingConfig()
        
        assert config.provider == "huggingface"  # Lowercased
        assert config.api_key == "test-key-123"
        assert config.model_name == "custom-model"
        assert config.batch_size == 20
    
    def test_when_invalid_provider_then_validation_fails(self):
        """Given invalid provider, when creating EmbeddingConfig, then raises ValidationError."""
        with pytest.raises(ValueError, match="Provider must be one of"):
            EmbeddingConfig(provider="invalid_provider")
    
    def test_when_negative_values_then_validation_fails(self):
        """Given negative values for positive fields, when creating EmbeddingConfig, then raises ValidationError."""
        with pytest.raises(ValueError, match="Value must be positive"):
            EmbeddingConfig(cache_ttl=-1)
        
        with pytest.raises(ValueError, match="Value must be positive"):
            EmbeddingConfig(batch_size=0)
    
    def test_when_invalid_retry_delay_then_validation_fails(self):
        """Given invalid retry delay, when creating EmbeddingConfig, then raises ValidationError."""
        with pytest.raises(ValueError, match="Retry delay must be between 0.1 and 60.0 seconds"):
            EmbeddingConfig(retry_delay=0.05)
        
        with pytest.raises(ValueError, match="Retry delay must be between 0.1 and 60.0 seconds"):
            EmbeddingConfig(retry_delay=70.0)


class TestLoggingConfig:
    """Test logging configuration and validation."""
    
    def test_when_default_values_then_proper_defaults(self):
        """Given no environment variables, when creating LoggingConfig, then uses proper defaults."""
        config = LoggingConfig()
        
        assert config.log_level == "INFO"
        assert config.database_log_level == "WARNING"
        assert config.enable_json_logging is True
        assert config.enable_correlation_ids is True
        assert config.log_file is None
        assert config.enable_console_logging is True
        assert config.enable_metrics is True
        assert config.metrics_prefix == "idea_ingestion"
    
    @patch.dict(os.environ, {
        'LOG_LEVEL': 'debug',
        'DB_LOG_LEVEL': 'error',
        'ENABLE_JSON_LOGGING': 'false',
        'LOG_FILE': '/var/log/app.log'
    })
    def test_when_env_vars_set_then_overrides_defaults(self):
        """Given environment variables, when creating LoggingConfig, then overrides defaults."""
        config = LoggingConfig()
        
        assert config.log_level == "DEBUG"  # Uppercased
        assert config.database_log_level == "ERROR"  # Uppercased
        assert config.enable_json_logging is False
        assert config.log_file == "/var/log/app.log"
    
    def test_when_invalid_log_level_then_validation_fails(self):
        """Given invalid log level, when creating LoggingConfig, then raises ValidationError."""
        with pytest.raises(ValueError, match="Log level must be one of"):
            LoggingConfig(log_level="INVALID")


class TestIngestionConfig:
    """Test main ingestion configuration combining all sub-configurations."""
    
    def test_when_default_values_then_proper_defaults(self):
        """Given no environment variables, when creating IngestionConfig, then uses proper defaults."""
        config = IngestionConfig()
        
        assert config.environment == "development"
        assert config.debug_mode is False
        assert config.app_name == "agentic-startup-studio"
        assert config.version == "1.0.0"
        assert config.secret_key == ""
        assert config.allowed_origins == ["localhost", "127.0.0.1"]
        
        # Sub-configurations should be properly initialized
        assert isinstance(config.database, DatabaseConfig)
        assert isinstance(config.validation, ValidationConfig)
        assert isinstance(config.embedding, EmbeddingConfig)
        assert isinstance(config.logging, LoggingConfig)
    
    @patch.dict(os.environ, {
        'ENVIRONMENT': 'PRODUCTION',
        'DEBUG_MODE': 'true',
        'APP_NAME': 'custom-app',
        'SECRET_KEY': 'super-secret-key',
        'ALLOWED_ORIGINS': 'example.com,api.example.com'
    })
    def test_when_env_vars_set_then_overrides_defaults(self):
        """Given environment variables, when creating IngestionConfig, then overrides defaults."""
        config = IngestionConfig()
        
        assert config.environment == "production"  # Lowercased
        assert config.debug_mode is True
        assert config.app_name == "custom-app"
        assert config.secret_key == "super-secret-key"
        assert config.allowed_origins == ["example.com", "api.example.com"]
    
    def test_when_invalid_environment_then_validation_fails(self):
        """Given invalid environment, when creating IngestionConfig, then raises ValidationError."""
        with pytest.raises(ValueError, match="Environment must be one of"):
            IngestionConfig(environment="invalid_env")
    
    def test_when_production_without_secret_key_then_validation_fails(self):
        """Given production environment without secret key, when creating IngestionConfig, then raises ValidationError."""
        with pytest.raises(ValueError, match="SECRET_KEY is required for production environment"):
            IngestionConfig(environment="production", secret_key="")
    
    def test_when_is_production_then_returns_true(self):
        """Given production environment, when calling is_production, then returns True."""
        config = IngestionConfig(environment="production", secret_key="test-key")
        assert config.is_production() is True
    
    def test_when_is_development_then_returns_true(self):
        """Given development environment, when calling is_development, then returns True."""
        config = IngestionConfig(environment="development")
        assert config.is_development() is True
    
    def test_when_get_correlation_id_header_then_returns_standard_header(self):
        """Given config, when getting correlation ID header, then returns standard header name."""
        config = IngestionConfig()
        assert config.get_correlation_id_header() == "X-Correlation-ID"
    
    def test_when_allowed_origins_as_string_then_parses_correctly(self):
        """Given allowed origins as comma-separated string, when creating config, then parses to list."""
        config = IngestionConfig()
        
        # Test the validator directly
        result = config.parse_allowed_origins("host1.com, host2.com , host3.com")
        assert result == ["host1.com", "host2.com", "host3.com"]
    
    def test_when_nested_delimiter_support_then_works_correctly(self):
        """Given nested environment variables, when creating config, then supports nested delimiter."""
        # This tests the env_nested_delimiter = "__" configuration
        with patch.dict(os.environ, {'DB__HOST': 'nested-host', 'DB__PORT': '3306'}):
            config = IngestionConfig()
            assert config.database.host == 'nested-host'
            assert config.database.port == 3306


class TestSettingsFunction:
    """Test global settings functions and caching."""
    
    @patch('pipeline.config.settings.IngestionConfig')
    def test_when_get_settings_then_caches_result(self, mock_config_class):
        """Given multiple calls to get_settings, when called, then caches result."""
        mock_instance = MagicMock()
        mock_config_class.return_value = mock_instance
        
        # Clear any existing cache
        get_settings.cache_clear()
        
        # First call
        result1 = get_settings()
        # Second call
        result2 = get_settings()
        
        # Should be same instance (cached)
        assert result1 is result2
        # Constructor should only be called once
        assert mock_config_class.call_count == 1
    
    @patch('pipeline.config.settings.logger')
    def test_when_get_settings_succeeds_then_logs_info(self, mock_logger):
        """Given successful settings loading, when get_settings called, then logs configuration info."""
        get_settings.cache_clear()
        
        result = get_settings()
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert "Loaded configuration for environment" in call_args[0][0]
    
    @patch('pipeline.config.settings.IngestionConfig')
    @patch('pipeline.config.settings.logger')
    def test_when_get_settings_fails_then_logs_error_and_raises(self, mock_logger, mock_config_class):
        """Given settings loading failure, when get_settings called, then logs error and raises."""
        mock_config_class.side_effect = Exception("Config error")
        get_settings.cache_clear()
        
        with pytest.raises(Exception, match="Config error"):
            get_settings()
        
        mock_logger.error.assert_called_once()


class TestEnvironmentValidation:
    """Test environment variable validation functions."""
    
    @patch.dict(os.environ, {}, clear=True)
    def test_when_development_env_then_no_required_vars(self):
        """Given development environment, when validating required vars, then succeeds."""
        with patch.dict(os.environ, {'ENVIRONMENT': 'development'}):
            # Should not raise any exception
            validate_required_env_vars()
    
    @patch.dict(os.environ, {
        'ENVIRONMENT': 'production',
        'SECRET_KEY': 'test-secret',
        'DB_PASSWORD': 'test-password',
        'EMBEDDING_API_KEY': 'test-api-key'
    }, clear=True)
    def test_when_production_with_all_vars_then_succeeds(self):
        """Given production environment with all required vars, when validating, then succeeds."""
        # Should not raise any exception
        validate_required_env_vars()
    
    @patch.dict(os.environ, {'ENVIRONMENT': 'production'}, clear=True)
    def test_when_production_missing_vars_then_raises_error(self):
        """Given production environment missing required vars, when validating, then raises ValueError."""
        with pytest.raises(ValueError, match="Missing required environment variables"):
            validate_required_env_vars()


class TestConfigSummary:
    """Test configuration summary generation."""
    
    @patch('pipeline.config.settings.get_settings')
    def test_when_get_config_summary_then_returns_summary_dict(self, mock_get_settings):
        """Given configuration, when getting summary, then returns summary with masked sensitive data."""
        mock_config = MagicMock()
        mock_config.environment = "development"
        mock_config.debug_mode = True
        mock_config.app_name = "test-app"
        mock_config.version = "1.0.0"
        
        # Configure database mock
        mock_config.database.host = "localhost"
        mock_config.database.port = 5432
        mock_config.database.database = "testdb"
        mock_config.database.enable_vector_search = True
        mock_config.database.vector_dimensions = 1536
        
        # Configure other sub-configs
        mock_config.validation.similarity_threshold = 0.8
        mock_config.validation.enable_profanity_filter = True
        mock_config.validation.enable_spam_detection = True
        
        mock_config.embedding.provider = "openai"
        mock_config.embedding.model_name = "text-embedding-ada-002"
        mock_config.embedding.enable_cache = True
        
        mock_config.logging.log_level = "INFO"
        mock_config.logging.enable_json_logging = True
        mock_config.logging.enable_metrics = True
        
        mock_get_settings.return_value = mock_config
        
        summary = get_config_summary()
        
        # Check main structure
        assert summary["environment"] == "development"
        assert summary["debug_mode"] is True
        assert summary["app_info"]["name"] == "test-app"
        assert summary["app_info"]["version"] == "1.0.0"
        
        # Check database section
        assert summary["database"]["host"] == "localhost"
        assert summary["database"]["port"] == 5432
        assert summary["database"]["database"] == "testdb"
        assert summary["database"]["vector_enabled"] is True
        assert summary["database"]["vector_dimensions"] == 1536
        
        # Check other sections exist
        assert "validation" in summary
        assert "embedding" in summary
        assert "logging" in summary


class TestConfigAccessors:
    """Test configuration accessor functions."""
    
    @patch('pipeline.config.settings.get_settings')
    def test_when_get_db_config_then_returns_database_config(self, mock_get_settings):
        """Given settings, when getting db config, then returns database configuration."""
        mock_settings = MagicMock()
        mock_db_config = MagicMock()
        mock_settings.database = mock_db_config
        mock_get_settings.return_value = mock_settings
        
        result = get_db_config()
        
        assert result is mock_db_config
    
    @patch('pipeline.config.settings.get_settings')
    def test_when_get_validation_config_then_returns_validation_config(self, mock_get_settings):
        """Given settings, when getting validation config, then returns validation configuration."""
        mock_settings = MagicMock()
        mock_validation_config = MagicMock()
        mock_settings.validation = mock_validation_config
        mock_get_settings.return_value = mock_settings
        
        result = get_validation_config()
        
        assert result is mock_validation_config
    
    @patch('pipeline.config.settings.get_settings')
    def test_when_get_embedding_config_then_returns_embedding_config(self, mock_get_settings):
        """Given settings, when getting embedding config, then returns embedding configuration."""
        mock_settings = MagicMock()
        mock_embedding_config = MagicMock()
        mock_settings.embedding = mock_embedding_config
        mock_get_settings.return_value = mock_settings
        
        result = get_embedding_config()
        
        assert result is mock_embedding_config
    
    @patch('pipeline.config.settings.get_settings')
    def test_when_get_logging_config_then_returns_logging_config(self, mock_get_settings):
        """Given settings, when getting logging config, then returns logging configuration."""
        mock_settings = MagicMock()
        mock_logging_config = MagicMock()
        mock_settings.logging = mock_logging_config
        mock_get_settings.return_value = mock_settings
        
        result = get_logging_config()
        
        assert result is mock_logging_config