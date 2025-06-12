"""
Configuration management for data ingestion pipeline.

This module provides environment-driven configuration with validation,
defaults, and type safety for all pipeline components.
"""

import os
from functools import lru_cache
from typing import Optional, List, Dict, Any, Union
from pathlib import Path

from pydantic import BaseModel, Field, validator, model_validator
from pydantic_settings import BaseSettings
import logging

logger = logging.getLogger(__name__)


class DatabaseConfig(BaseSettings):
    """Database connection and behavior configuration."""
    
    host: str = Field(default="localhost", env="DB_HOST")
    port: int = Field(default=5432, env="DB_PORT")
    database: str = Field(default="startup_studio", env="DB_NAME")
    username: str = Field(default="postgres", env="DB_USER")
    password: str = Field(default="", env="DB_PASSWORD")
    
    # Connection pool settings
    min_connections: int = Field(default=1, env="DB_MIN_CONNECTIONS")
    max_connections: int = Field(default=20, env="DB_MAX_CONNECTIONS")
    timeout: int = Field(default=30, env="DB_TIMEOUT")
    
    # pgvector configuration
    vector_dimensions: int = Field(default=1536, env="VECTOR_DIMENSIONS")
    enable_vector_search: bool = Field(default=True, env="ENABLE_VECTOR_SEARCH")
    
    @validator('port')
    def validate_port(cls, v):
        """Validate database port is within valid range."""
        if not 1 <= v <= 65535:
            raise ValueError("Database port must be between 1 and 65535")
        return v
    
    @validator('timeout')
    def validate_timeout(cls, v):
        """Validate timeout is positive."""
        if v <= 0:
            raise ValueError("Database timeout must be positive")
        return v
    
    @validator('vector_dimensions')
    def validate_vector_dimensions(cls, v):
        """Validate vector dimensions are reasonable."""
        if not 100 <= v <= 4096:
            raise ValueError("Vector dimensions must be between 100 and 4096")
        return v
    
    def get_connection_url(self, include_password: bool = False) -> str:
        """
        Generate database connection URL with secure defaults.
        
        SECURITY: Password inclusion disabled by default to prevent credential exposure.
        Only enable when absolutely necessary for actual database connections.
        """
        if include_password and self.password:
            return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        return f"postgresql://{self.username}@{self.host}:{self.port}/{self.database}"
    
    def get_safe_connection_url(self) -> str:
        """Get connection URL with password masked for logging/display purposes."""
        if self.password:
            return f"postgresql://{self.username}:****@{self.host}:{self.port}/{self.database}"
        return f"postgresql://{self.username}@{self.host}:{self.port}/{self.database}"

    class Config:
        """Pydantic configuration."""
        env_prefix = ""
        case_sensitive = False


class ValidationConfig(BaseSettings):
    """Input validation and security configuration."""
    
    # Content length limits
    min_title_length: int = Field(default=10, env="MIN_TITLE_LENGTH")
    max_title_length: int = Field(default=200, env="MAX_TITLE_LENGTH")
    min_description_length: int = Field(default=10, env="MIN_DESCRIPTION_LENGTH")
    max_description_length: int = Field(default=5000, env="MAX_DESCRIPTION_LENGTH")
    
    # Security features
    enable_profanity_filter: bool = Field(default=True, env="ENABLE_PROFANITY_FILTER")
    enable_spam_detection: bool = Field(default=True, env="ENABLE_SPAM_DETECTION")
    enable_html_sanitization: bool = Field(default=True, env="ENABLE_HTML_SANITIZATION")
    
    # Similarity detection
    similarity_threshold: float = Field(default=0.8, env="SIMILARITY_THRESHOLD")
    title_fuzzy_threshold: float = Field(default=0.7, env="TITLE_FUZZY_THRESHOLD")
    
    # Rate limiting
    max_ideas_per_hour: int = Field(default=10, env="MAX_IDEAS_PER_HOUR")
    max_ideas_per_day: int = Field(default=50, env="MAX_IDEAS_PER_DAY")
    
    @validator('min_title_length', 'min_description_length')
    def validate_min_lengths(cls, v):
        """Validate minimum lengths are positive."""
        if v <= 0:
            raise ValueError("Minimum length must be positive")
        return v
    
    @validator('similarity_threshold', 'title_fuzzy_threshold')
    def validate_thresholds(cls, v):
        """Validate thresholds are between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        return v
    
    @model_validator(mode='after')
    def validate_length_consistency(self):
        """Ensure max lengths are greater than min lengths."""
        if self.max_title_length <= self.min_title_length:
            raise ValueError("max_title_length must be greater than min_title_length")
        
        if self.max_description_length <= self.min_description_length:
            raise ValueError("max_description_length must be greater than min_description_length")
        
        return self

    class Config:
        """Pydantic configuration."""
        env_prefix = ""
        case_sensitive = False


class EmbeddingConfig(BaseSettings):
    """Configuration for text embedding and vector operations."""
    
    # API configuration
    provider: str = Field(default="openai", env="EMBEDDING_PROVIDER")
    api_key: str = Field(default="", env="EMBEDDING_API_KEY")
    model_name: str = Field(default="text-embedding-ada-002", env="EMBEDDING_MODEL")
    
    # Caching
    enable_cache: bool = Field(default=True, env="ENABLE_EMBEDDING_CACHE")
    cache_ttl: int = Field(default=86400, env="EMBEDDING_CACHE_TTL")  # 24 hours
    cache_size: int = Field(default=1000, env="EMBEDDING_CACHE_SIZE")
    
    # Performance
    batch_size: int = Field(default=10, env="EMBEDDING_BATCH_SIZE")
    retry_attempts: int = Field(default=3, env="EMBEDDING_RETRY_ATTEMPTS")
    retry_delay: float = Field(default=1.0, env="EMBEDDING_RETRY_DELAY")
    
    @validator('provider')
    def validate_provider(cls, v):
        """Validate embedding provider is supported."""
        supported_providers = ['openai', 'huggingface', 'sentence-transformers']
        if v.lower() not in supported_providers:
            raise ValueError(f"Provider must be one of: {supported_providers}")
        return v.lower()
    
    @validator('cache_ttl', 'cache_size', 'batch_size', 'retry_attempts')
    def validate_positive_integers(cls, v):
        """Validate positive integer fields."""
        if v <= 0:
            raise ValueError("Value must be positive")
        return v
    
    @validator('retry_delay')
    def validate_retry_delay(cls, v):
        """Validate retry delay is reasonable."""
        if not 0.1 <= v <= 60.0:
            raise ValueError("Retry delay must be between 0.1 and 60.0 seconds")
        return v

    class Config:
        """Pydantic configuration."""
        env_prefix = ""
        case_sensitive = False


class LoggingConfig(BaseSettings):
    """Logging and monitoring configuration."""
    
    # Log levels
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    database_log_level: str = Field(default="WARNING", env="DB_LOG_LEVEL")
    
    # Log formats
    enable_json_logging: bool = Field(default=True, env="ENABLE_JSON_LOGGING")
    enable_correlation_ids: bool = Field(default=True, env="ENABLE_CORRELATION_IDS")
    
    # Log destinations
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    enable_console_logging: bool = Field(default=True, env="ENABLE_CONSOLE_LOGGING")
    
    # Metrics
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_prefix: str = Field(default="idea_ingestion", env="METRICS_PREFIX")
    
    @validator('log_level', 'database_log_level')
    def validate_log_level(cls, v):
        """Validate log level is supported."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()

    class Config:
        """Pydantic configuration."""
        env_prefix = ""
        case_sensitive = False


class BudgetConfig(BaseSettings):
    """Budget allocation and cost control configuration."""
    
    # Total cycle budget (≤$62 per specification)
    total_cycle_budget: float = Field(default=62.00, env="TOTAL_CYCLE_BUDGET")
    
    # Category allocations
    openai_budget: float = Field(default=10.00, env="OPENAI_BUDGET")
    google_ads_budget: float = Field(default=45.00, env="GOOGLE_ADS_BUDGET")
    infrastructure_budget: float = Field(default=5.00, env="INFRASTRUCTURE_BUDGET")
    
    # Alert thresholds (as percentages)
    warning_threshold: float = Field(default=0.80, env="BUDGET_WARNING_THRESHOLD")
    critical_threshold: float = Field(default=0.95, env="BUDGET_CRITICAL_THRESHOLD")
    emergency_threshold: float = Field(default=1.00, env="BUDGET_EMERGENCY_THRESHOLD")
    
    # Cost tracking settings
    enable_cost_tracking: bool = Field(default=True, env="ENABLE_COST_TRACKING")
    enable_budget_alerts: bool = Field(default=True, env="ENABLE_BUDGET_ALERTS")
    enable_emergency_shutdown: bool = Field(default=True, env="ENABLE_EMERGENCY_SHUTDOWN")
    
    # Monitoring intervals
    budget_check_interval: int = Field(default=60, env="BUDGET_CHECK_INTERVAL")  # seconds
    
    @validator('total_cycle_budget', 'openai_budget', 'google_ads_budget', 'infrastructure_budget')
    def validate_positive_budgets(cls, v):
        """Validate budget amounts are positive."""
        if v <= 0:
            raise ValueError("Budget amounts must be positive")
        return v
    
    @validator('warning_threshold', 'critical_threshold', 'emergency_threshold')
    def validate_thresholds(cls, v):
        """Validate thresholds are between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Budget thresholds must be between 0.0 and 1.0")
        return v
    
    @model_validator(mode='after')
    def validate_budget_consistency(self):
        """Ensure budget allocations don't exceed total budget."""
        allocated_total = (
            self.openai_budget +
            self.google_ads_budget +
            self.infrastructure_budget
        )
        
        if allocated_total > self.total_cycle_budget:
            raise ValueError(
                f"Total allocated budget (${allocated_total}) exceeds "
                f"cycle budget (${self.total_cycle_budget})"
            )
        
        # Validate threshold ordering
        if not (self.warning_threshold <= self.critical_threshold <= self.emergency_threshold):
            raise ValueError(
                "Budget thresholds must be in order: warning ≤ critical ≤ emergency"
            )
        
        return self
    
    @validator('budget_check_interval')
    def validate_check_interval(cls, v):
        """Validate budget check interval is reasonable."""
        if not 10 <= v <= 3600:  # 10 seconds to 1 hour
            raise ValueError("Budget check interval must be between 10 and 3600 seconds")
        return v

    class Config:
        """Pydantic configuration."""
        env_prefix = ""
        case_sensitive = False


class IngestionConfig(BaseSettings):
    """Main configuration combining all sub-configurations."""
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug_mode: bool = Field(default=False, env="DEBUG_MODE")
    
    # Component configurations
    database: DatabaseConfig = DatabaseConfig()
    validation: ValidationConfig = ValidationConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    logging: LoggingConfig = LoggingConfig()
    budget: BudgetConfig = BudgetConfig()
    
    # Application settings
    app_name: str = Field(default="agentic-startup-studio", env="APP_NAME")
    version: str = Field(default="1.0.0", env="APP_VERSION")
    
    # Security
    secret_key: str = Field(default="", env="SECRET_KEY")
    # Use Union to prevent automatic JSON parsing of environment variables
    allowed_origins: Union[str, List[str]] = Field(
        default_factory=lambda: ["localhost", "127.0.0.1"]
    )
    
    @validator('environment')
    def validate_environment(cls, v):
        """Validate environment is supported."""
        valid_envs = ['development', 'testing', 'staging', 'production']
        if v.lower() not in valid_envs:
            raise ValueError(f"Environment must be one of: {valid_envs}")
        return v.lower()
    
    @validator('secret_key')
    def validate_secret_key(cls, v, values):
        """Validate secret key is set for production."""
        environment = values.get('environment', 'development')
        if environment == 'production' and not v:
            raise ValueError("SECRET_KEY is required for production environment")
        return v
    
    @validator('allowed_origins', pre=True)
    def parse_allowed_origins(cls, v):
        """Parse allowed origins from string or list."""
        # Handle environment variable directly to prevent JSON parsing
        env_value = os.getenv('ALLOWED_ORIGINS')
        if env_value:
            # Parse comma-separated string from environment
            return [origin.strip() for origin in env_value.split(',') if origin.strip()]
        
        # Handle default or existing list values
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',') if origin.strip()]
        elif isinstance(v, list):
            return v
        else:
            return ["localhost", "127.0.0.1"]
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == 'production'
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == 'development'
    
    def get_correlation_id_header(self) -> str:
        """Get correlation ID header name."""
        return "X-Correlation-ID"

    class Config:
        """Pydantic configuration."""
        env_prefix = ""
        case_sensitive = False
        env_nested_delimiter = "__"  # Support nested config like DB__HOST


@lru_cache()
def get_settings() -> IngestionConfig:
    """
    Get cached application settings.
    
    This function is cached to ensure settings are loaded only once
    and reused throughout the application lifecycle.
    """
    try:
        settings = IngestionConfig()
        logger.info(
            f"Loaded configuration for environment: {settings.environment}",
            extra={
                "environment": settings.environment,
                "debug_mode": settings.debug_mode,
                "app_name": settings.app_name,
                "version": settings.version
            }
        )
        return settings
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise


def validate_required_env_vars() -> None:
    """
    Validate that all required environment variables are set.
    
    This function should be called at application startup to ensure
    all necessary configuration is available.
    """
    required_vars = []
    
    # Check environment-specific requirements
    env = os.getenv('ENVIRONMENT', 'development').lower()
    
    if env == 'production':
        required_vars.extend([
            'SECRET_KEY',
            'DB_PASSWORD',
            'EMBEDDING_API_KEY'
        ])
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )


def get_config_summary() -> Dict[str, Any]:
    """
    Get a summary of current configuration for logging/debugging.
    
    Sensitive values (passwords, keys) are masked for security.
    """
    settings = get_settings()
    
    return {
        "environment": settings.environment,
        "debug_mode": settings.debug_mode,
        "app_info": {
            "name": settings.app_name,
            "version": settings.version
        },
        "database": {
            "host": settings.database.host,
            "port": settings.database.port,
            "database": settings.database.database,
            "vector_enabled": settings.database.enable_vector_search,
            "vector_dimensions": settings.database.vector_dimensions
        },
        "validation": {
            "similarity_threshold": settings.validation.similarity_threshold,
            "profanity_filter": settings.validation.enable_profanity_filter,
            "spam_detection": settings.validation.enable_spam_detection
        },
        "embedding": {
            "provider": settings.embedding.provider,
            "model": settings.embedding.model_name,
            "cache_enabled": settings.embedding.enable_cache
        },
        "logging": {
            "level": settings.logging.log_level,
            "json_format": settings.logging.enable_json_logging,
            "metrics_enabled": settings.logging.enable_metrics
        }
    }


# Export commonly used configuration instances
def get_db_config() -> DatabaseConfig:
    """Get database configuration."""
    return get_settings().database


def get_validation_config() -> ValidationConfig:
    """Get validation configuration."""
    return get_settings().validation


def get_embedding_config() -> EmbeddingConfig:
    """Get embedding configuration."""
    return get_settings().embedding


def get_logging_config() -> LoggingConfig:
    """Get logging configuration."""
    return get_settings().logging


def get_budget_config() -> BudgetConfig:
    """Get budget configuration."""
    return get_settings().budget