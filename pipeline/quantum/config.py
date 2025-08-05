"""
Quantum Task Planner Configuration Management

Production-ready configuration system with:
- Environment-based configuration
- Secure secret management
- Runtime configuration validation
- Performance tuning parameters
- Multi-environment support (dev, staging, prod)
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import yaml

from pydantic import BaseSettings, Field, validator
from pydantic.env_settings import SettingsSourceCallable

logger = logging.getLogger(__name__)


class Environment(str, Enum):
    """Deployment environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class QuantumPerformanceConfig:
    """Performance-related configuration."""
    
    # Quantum computation settings
    max_parallel_tasks: int = 10
    max_concurrent_measurements: int = 100
    quantum_evolution_interval: float = 1.0
    superposition_collapse_timeout: float = 30.0
    
    # Caching configuration
    enable_caching: bool = True
    cache_size_mb: int = 100
    cache_ttl_seconds: int = 3600
    cache_cleanup_interval: int = 300
    
    # Scheduling optimization
    scheduling_algorithm: str = "quantum_annealing"
    annealing_iterations: int = 100
    annealing_temperature_initial: float = 10.0
    annealing_temperature_final: float = 0.01
    annealing_cooling_rate: float = 0.95
    
    # Resource limits
    max_task_count: int = 10000
    max_entanglements: int = 1000
    max_dependency_depth: int = 20
    memory_limit_mb: int = 2048
    
    # Monitoring intervals
    metrics_collection_interval: int = 60
    health_check_interval: int = 30
    performance_sampling_rate: float = 0.1


@dataclass  
class QuantumSecurityConfig:
    """Security-related configuration."""
    
    # Authentication
    enable_authentication: bool = True
    jwt_secret_key: str = ""
    jwt_expiration_hours: int = 24
    jwt_refresh_enabled: bool = True
    
    # Encryption
    enable_encryption: bool = True
    encryption_algorithm: str = "AES-256-GCM"
    key_rotation_days: int = 90
    
    # Rate limiting
    enable_rate_limiting: bool = True
    rate_limit_per_minute: int = 100
    rate_limit_burst: int = 200
    
    # Input validation
    max_input_length: int = 10000
    enable_input_sanitization: bool = True
    allowed_file_types: List[str] = field(default_factory=lambda: ['.json', '.yaml', '.txt'])
    
    # Audit logging
    enable_audit_logging: bool = True
    audit_log_retention_days: int = 2555  # 7 years
    sensitive_data_masking: bool = True


@dataclass
class QuantumDatabaseConfig:
    """Database configuration."""
    
    # Connection settings
    host: str = "localhost"
    port: int = 5432
    database: str = "quantum_planner"
    username: str = "quantum_user"
    password: str = ""
    
    # Connection pool
    min_connections: int = 5
    max_connections: int = 20
    connection_timeout: int = 30
    command_timeout: int = 60
    
    # Performance
    enable_ssl: bool = True
    ssl_mode: str = "require"
    statement_timeout: int = 300
    
    # Vector database (pgvector)
    vector_dimensions: int = 384
    vector_index_type: str = "hnsw"
    vector_index_m: int = 16
    vector_index_ef_construction: int = 64


@dataclass
class QuantumObservabilityConfig:
    """Observability and monitoring configuration."""
    
    # Metrics
    enable_prometheus: bool = True
    prometheus_port: int = 9090
    metrics_path: str = "/metrics"
    
    # Tracing
    enable_tracing: bool = True
    jaeger_agent_host: str = "localhost"
    jaeger_agent_port: int = 6831
    trace_sampling_rate: float = 0.1
    
    # Logging
    log_level: LogLevel = LogLevel.INFO
    log_format: str = "json"
    enable_structured_logging: bool = True
    log_rotation_size_mb: int = 100
    log_retention_days: int = 30
    
    # Health checks
    enable_health_endpoints: bool = True
    health_check_port: int = 8080
    health_check_path: str = "/health"
    readiness_check_path: str = "/ready"


class QuantumPlannerSettings(BaseSettings):
    """Main configuration settings for quantum task planner."""
    
    # Environment
    environment: Environment = Field(Environment.DEVELOPMENT, env="ENVIRONMENT")
    debug: bool = Field(False, env="DEBUG")
    
    # Application settings
    app_name: str = Field("Quantum Task Planner", env="APP_NAME")
    app_version: str = Field("1.0.0", env="APP_VERSION")
    api_prefix: str = Field("/api/v1", env="API_PREFIX")
    
    # Server configuration
    host: str = Field("localhost", env="HOST")
    port: int = Field(8000, env="PORT")
    workers: int = Field(1, env="WORKERS")
    
    # Performance configuration
    performance: QuantumPerformanceConfig = Field(default_factory=QuantumPerformanceConfig)
    
    # Security configuration
    security: QuantumSecurityConfig = Field(default_factory=QuantumSecurityConfig)
    
    # Database configuration
    database: QuantumDatabaseConfig = Field(default_factory=QuantumDatabaseConfig)
    
    # Observability configuration
    observability: QuantumObservabilityConfig = Field(default_factory=QuantumObservabilityConfig)
    
    # Internationalization
    default_locale: str = Field("en_US", env="DEFAULT_LOCALE")
    supported_locales: List[str] = Field(
        default_factory=lambda: ["en_US", "es_ES", "fr_FR", "de_DE", "ja_JP", "zh_CN"]
    )
    
    # Compliance
    enable_gdpr_compliance: bool = Field(True, env="ENABLE_GDPR_COMPLIANCE")
    enable_ccpa_compliance: bool = Field(False, env="ENABLE_CCPA_COMPLIANCE")
    data_retention_days: int = Field(2555, env="DATA_RETENTION_DAYS")  # 7 years
    
    # Feature flags
    enable_quantum_caching: bool = Field(True, env="ENABLE_QUANTUM_CACHING")
    enable_advanced_scheduling: bool = Field(True, env="ENABLE_ADVANCED_SCHEDULING")
    enable_quantum_entanglement: bool = Field(True, env="ENABLE_QUANTUM_ENTANGLEMENT")
    enable_parallel_processing: bool = Field(True, env="ENABLE_PARALLEL_PROCESSING")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        @classmethod
        def customise_sources(
            cls,
            init_settings: SettingsSourceCallable,
            env_settings: SettingsSourceCallable,
            file_secret_settings: SettingsSourceCallable,
        ) -> tuple[SettingsSourceCallable, ...]:
            """Customize settings sources priority."""
            return (
                init_settings,
                env_settings,
                file_secret_settings,
            )
    
    @validator("environment")
    def validate_environment(cls, v):
        """Validate environment setting."""
        if isinstance(v, str):
            try:
                return Environment(v.lower())
            except ValueError:
                raise ValueError(f"Invalid environment: {v}")
        return v
    
    @validator("port")
    def validate_port(cls, v):
        """Validate port number."""
        if not (1 <= v <= 65535):
            raise ValueError("Port must be between 1 and 65535")
        return v
    
    @validator("workers")
    def validate_workers(cls, v):
        """Validate worker count."""
        if v < 1:
            raise ValueError("Workers must be at least 1")
        return min(v, os.cpu_count() or 1)
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PRODUCTION
    
    def is_debug_enabled(self) -> bool:
        """Check if debug mode is enabled."""
        return self.debug and not self.is_production()
    
    def get_database_url(self) -> str:
        """Get database connection URL."""
        return (
            f"postgresql://{self.database.username}:{self.database.password}"
            f"@{self.database.host}:{self.database.port}/{self.database.database}"
        )
    
    def get_redis_url(self) -> str:
        """Get Redis connection URL for caching."""
        # This would be implemented if using Redis for distributed caching
        return "redis://localhost:6379/0"
    
    def validate_security_config(self) -> List[str]:
        """Validate security configuration and return warnings."""
        warnings = []
        
        if self.is_production():
            if not self.security.jwt_secret_key:
                warnings.append("JWT secret key is not set in production")
            
            if not self.security.enable_encryption:
                warnings.append("Encryption is disabled in production")
            
            if not self.security.enable_rate_limiting:
                warnings.append("Rate limiting is disabled in production")
            
            if not self.database.enable_ssl:
                warnings.append("Database SSL is disabled in production")
            
            if self.debug:
                warnings.append("Debug mode is enabled in production")
        
        return warnings
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary (excluding sensitive data)."""
        config_dict = self.dict()
        
        # Mask sensitive fields
        if "password" in config_dict.get("database", {}):
            config_dict["database"]["password"] = "***masked***"
        
        if "jwt_secret_key" in config_dict.get("security", {}):
            config_dict["security"]["jwt_secret_key"] = "***masked***"
        
        return config_dict


class ConfigurationLoader:
    """Configuration loader with multiple source support."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path("config")
        self.settings: Optional[QuantumPlannerSettings] = None
    
    def load_configuration(self, 
                          environment: Optional[Environment] = None,
                          config_file: Optional[Path] = None) -> QuantumPlannerSettings:
        """
        Load configuration from multiple sources.
        
        Args:
            environment: Target environment
            config_file: Specific configuration file
            
        Returns:
            Loaded configuration settings
        """
        # Determine environment
        environment = environment or Environment(os.getenv("ENVIRONMENT", "development"))
        
        # Load base configuration
        base_config = self._load_base_config()
        
        # Load environment-specific configuration
        env_config = self._load_environment_config(environment)
        
        # Load from specific file if provided
        file_config = {}
        if config_file and config_file.exists():
            file_config = self._load_config_file(config_file)
        
        # Merge configurations (environment variables take precedence)
        merged_config = {**base_config, **env_config, **file_config}
        
        # Create settings instance
        self.settings = QuantumPlannerSettings(**merged_config)
        
        # Validate configuration
        self._validate_configuration()
        
        logger.info(f"Configuration loaded for environment: {environment}")
        return self.settings
    
    def _load_base_config(self) -> Dict[str, Any]:
        """Load base configuration."""
        base_config_file = self.config_dir / "base.yaml"
        
        if base_config_file.exists():
            return self._load_config_file(base_config_file)
        
        return {}
    
    def _load_environment_config(self, environment: Environment) -> Dict[str, Any]:
        """Load environment-specific configuration."""
        env_config_file = self.config_dir / f"{environment.value}.yaml"
        
        if env_config_file.exists():
            return self._load_config_file(env_config_file)
        
        return {}
    
    def _load_config_file(self, config_file: Path) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                if config_file.suffix.lower() in ['.yaml', '.yml']:
                    return yaml.safe_load(f) or {}
                elif config_file.suffix.lower() == '.json':
                    return json.load(f) or {}
                else:
                    logger.warning(f"Unsupported config file format: {config_file}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Failed to load config file {config_file}: {e}")
            return {}
    
    def _validate_configuration(self):
        """Validate loaded configuration."""
        if not self.settings:
            raise ValueError("Configuration not loaded")
        
        # Check for security warnings
        security_warnings = self.settings.validate_security_config()
        for warning in security_warnings:
            logger.warning(f"Security configuration warning: {warning}")
        
        # Validate required settings for production
        if self.settings.is_production():
            self._validate_production_config()
    
    def _validate_production_config(self):
        """Validate production-specific configuration."""
        required_settings = [
            ("security.jwt_secret_key", self.settings.security.jwt_secret_key),
            ("database.password", self.settings.database.password),
        ]
        
        missing_settings = []
        for setting_name, setting_value in required_settings:
            if not setting_value:
                missing_settings.append(setting_name)
        
        if missing_settings:
            raise ValueError(f"Missing required production settings: {missing_settings}")
        
        logger.info("Production configuration validation passed")
    
    def save_config_template(self, output_file: Path, environment: Environment):
        """Save configuration template for specific environment."""
        template_config = self._get_config_template(environment)
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                yaml.dump(template_config, f, default_flow_style=False, sort_keys=True)
            
            logger.info(f"Configuration template saved: {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save config template: {e}")
    
    def _get_config_template(self, environment: Environment) -> Dict[str, Any]:
        """Get configuration template for environment."""
        base_template = {
            "environment": environment.value,
            "debug": environment != Environment.PRODUCTION,
            "app_name": "Quantum Task Planner",
            "host": "0.0.0.0" if environment == Environment.PRODUCTION else "localhost",
            "port": 8000,
            "workers": 4 if environment == Environment.PRODUCTION else 1,
            
            "performance": {
                "max_parallel_tasks": 20 if environment == Environment.PRODUCTION else 5,
                "enable_caching": True,
                "cache_size_mb": 500 if environment == Environment.PRODUCTION else 50,
                "max_task_count": 50000 if environment == Environment.PRODUCTION else 1000,
            },
            
            "security": {
                "enable_authentication": environment == Environment.PRODUCTION,
                "jwt_secret_key": "CHANGE_ME_IN_PRODUCTION" if environment == Environment.PRODUCTION else "dev_secret",
                "enable_encryption": environment == Environment.PRODUCTION,
                "enable_rate_limiting": environment == Environment.PRODUCTION,
                "enable_audit_logging": environment == Environment.PRODUCTION,
            },
            
            "database": {
                "host": "postgres" if environment == Environment.PRODUCTION else "localhost",
                "port": 5432,
                "database": f"quantum_planner_{environment.value}",
                "username": "quantum_user",
                "password": "CHANGE_ME_IN_PRODUCTION",
                "min_connections": 10 if environment == Environment.PRODUCTION else 2,
                "max_connections": 50 if environment == Environment.PRODUCTION else 10,
                "enable_ssl": environment == Environment.PRODUCTION,
            },
            
            "observability": {
                "enable_prometheus": True,
                "enable_tracing": environment == Environment.PRODUCTION,
                "log_level": "INFO" if environment == Environment.PRODUCTION else "DEBUG",
                "enable_structured_logging": environment == Environment.PRODUCTION,
                "log_retention_days": 90 if environment == Environment.PRODUCTION else 7,
            },
        }
        
        return base_template


# Global configuration instance
_config: Optional[QuantumPlannerSettings] = None
_config_loader: Optional[ConfigurationLoader] = None


def get_settings(reload: bool = False) -> QuantumPlannerSettings:
    """
    Get global configuration settings.
    
    Args:
        reload: Force reload configuration
        
    Returns:
        Configuration settings instance
    """
    global _config, _config_loader
    
    if _config is None or reload:
        if _config_loader is None:
            _config_loader = ConfigurationLoader()
        
        _config = _config_loader.load_configuration()
    
    return _config


def load_configuration(environment: Optional[Environment] = None,
                      config_file: Optional[Path] = None,
                      config_dir: Optional[Path] = None) -> QuantumPlannerSettings:
    """
    Load configuration with specific parameters.
    
    Args:
        environment: Target environment
        config_file: Specific configuration file
        config_dir: Configuration directory
        
    Returns:
        Loaded configuration settings
    """
    global _config, _config_loader
    
    _config_loader = ConfigurationLoader(config_dir)
    _config = _config_loader.load_configuration(environment, config_file)
    
    return _config