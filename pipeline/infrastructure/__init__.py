"""
Infrastructure layer components for the agentic startup studio.

This module provides resilience, quality control, and operational
infrastructure components for the pipeline system including:

- Circuit breaker pattern for fault tolerance
- Quality gates for automated validation
- Monitoring and observability utilities
- External service integration patterns
"""

from .circuit_breaker import (
    CircuitBreakerState,
    CircuitBreakerConfig,
    CircuitBreakerMetrics,
    CircuitBreaker,
    CircuitBreakerRegistry,
    get_circuit_breaker_registry,
    
    # Pre-configured circuit breakers
    create_api_circuit_breaker,
    create_database_circuit_breaker,
    create_llm_circuit_breaker
)

from .quality_gates import (
    QualityGateStatus,
    QualityCriteriaType,
    QualityLevel,
    QualityCriteria,
    QualityResult,
    QualityGateConfig,
    QualityGateResult,
    QualityGate,
    QualityGateManager,
    get_quality_gate_manager,
    
    # Pre-defined quality gate configurations
    create_idea_validation_gate,
    create_evidence_collection_gate,
    create_pitch_deck_gate
)

from .observability import (
    LogLevel,
    MetricType,
    AlertSeverity,
    LogEntry,
    MetricValue,
    PerformanceMetrics,
    Alert,
    StructuredLogger,
    MetricsCollector,
    PerformanceMonitor,
    ObservabilityStack,
    get_observability_stack,
    get_logger,
    performance_monitor
)

__all__ = [
    # Circuit Breaker Components
    "CircuitBreakerState",
    "CircuitBreakerConfig",
    "CircuitBreakerMetrics",
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "get_circuit_breaker_registry",
    
    # Pre-configured Circuit Breakers
    "create_api_circuit_breaker",
    "create_database_circuit_breaker",
    "create_llm_circuit_breaker",
    
    # Quality Gate Components
    "QualityGateStatus",
    "QualityCriteriaType",
    "QualityLevel",
    "QualityCriteria",
    "QualityResult",
    "QualityGateConfig",
    "QualityGateResult",
    "QualityGate",
    "QualityGateManager",
    "get_quality_gate_manager",
    
    # Pre-defined Quality Gates
    "create_idea_validation_gate",
    "create_evidence_collection_gate",
    "create_pitch_deck_gate",
    
    # Observability Components
    "LogLevel",
    "MetricType",
    "AlertSeverity",
    "LogEntry",
    "MetricValue",
    "PerformanceMetrics",
    "Alert",
    "StructuredLogger",
    "MetricsCollector",
    "PerformanceMonitor",
    "ObservabilityStack",
    "get_observability_stack",
    "get_logger",
    "performance_monitor",
    
    # Infrastructure Management
    "initialize_infrastructure",
    "get_infrastructure_health",
    "get_infrastructure_metrics",
    "validate_infrastructure_config"
]

# Initialize default infrastructure components
def initialize_infrastructure() -> None:
    """
    Initialize default infrastructure components.
    
    Sets up default circuit breakers, quality gates, and observability
    stack for common pipeline operations.
    """
    # Initialize circuit breaker registry with common patterns
    cb_registry = get_circuit_breaker_registry()
    
    # Register default circuit breakers
    cb_registry.register("api_calls", create_api_circuit_breaker())
    cb_registry.register("database", create_database_circuit_breaker())
    cb_registry.register("llm_service", create_llm_circuit_breaker())
    
    # Initialize quality gate manager with pipeline gates
    qg_manager = get_quality_gate_manager()
    
    # Register default quality gates
    qg_manager.register_gate(create_idea_validation_gate())
    qg_manager.register_gate(create_evidence_collection_gate())
    qg_manager.register_gate(create_pitch_deck_gate())
    
    # Initialize observability stack
    observability = get_observability_stack()
    
    # Set up default logger
    logger = observability.get_logger(__name__)
    logger.info("Infrastructure components initialized successfully")


# Health check utilities
async def get_infrastructure_health() -> dict:
    """
    Get comprehensive health status of infrastructure components.
    
    Returns:
        Dictionary containing health status of all infrastructure components
    """
    health_status = {
        "status": "healthy",
        "timestamp": None,
        "components": {}
    }
    
    from datetime import datetime, timezone
    health_status["timestamp"] = datetime.now(timezone.utc).isoformat()
    
    try:
        # Check circuit breaker health
        cb_registry = get_circuit_breaker_registry()
        cb_health = await cb_registry.get_health_dashboard()
        health_status["components"]["circuit_breakers"] = cb_health
        
        # Check quality gate health
        qg_manager = get_quality_gate_manager()
        qg_health = await qg_manager.get_health_dashboard()
        health_status["components"]["quality_gates"] = qg_health
        
        # Check observability stack health
        observability = get_observability_stack()
        obs_health = await observability.get_health_status()
        health_status["components"]["observability"] = obs_health
        
        # Determine overall status
        cb_status = cb_health.get("overall_status", "healthy")
        qg_status = qg_health.get("overall_status", "healthy")
        obs_status = obs_health.get("overall_status", "healthy")
        
        unhealthy_components = [
            status for status in [cb_status, qg_status, obs_status]
            if status == "unhealthy"
        ]
        degraded_components = [
            status for status in [cb_status, qg_status, obs_status]
            if status == "degraded"
        ]
        
        if unhealthy_components:
            health_status["status"] = "unhealthy"
        elif degraded_components:
            health_status["status"] = "degraded"
        else:
            health_status["status"] = "healthy"
            
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["error"] = str(e)
    
    return health_status


async def get_infrastructure_metrics() -> dict:
    """
    Get comprehensive metrics from all infrastructure components.
    
    Returns:
        Dictionary containing metrics from all components
    """
    metrics = {
        "timestamp": None,
        "circuit_breakers": {},
        "quality_gates": {}
    }
    
    from datetime import datetime, timezone
    metrics["timestamp"] = datetime.now(timezone.utc).isoformat()
    
    try:
        # Get circuit breaker metrics
        cb_registry = get_circuit_breaker_registry()
        metrics["circuit_breakers"] = await cb_registry.get_all_metrics()
        
        # Get quality gate metrics
        qg_manager = get_quality_gate_manager()
        metrics["quality_gates"] = await qg_manager.get_all_metrics()
        
    except Exception as e:
        metrics["error"] = str(e)
        
    return metrics


# Configuration validation
def validate_infrastructure_config() -> list:
    """
    Validate infrastructure configuration.
    
    Returns:
        List of validation errors, empty if configuration is valid
    """
    errors = []
    
    try:
        from pipeline.config.settings import get_settings
        settings = get_settings()
        
        # Validate circuit breaker settings
        if not hasattr(settings, 'circuit_breaker_failure_threshold'):
            errors.append("Missing circuit_breaker_failure_threshold setting")
        elif settings.circuit_breaker_failure_threshold <= 0:
            errors.append("circuit_breaker_failure_threshold must be positive")
            
        if not hasattr(settings, 'circuit_breaker_timeout_seconds'):
            errors.append("Missing circuit_breaker_timeout_seconds setting")
        elif settings.circuit_breaker_timeout_seconds <= 0:
            errors.append("circuit_breaker_timeout_seconds must be positive")
            
        # Validate quality gate settings
        if not hasattr(settings, 'quality_gate_enabled'):
            errors.append("Missing quality_gate_enabled setting")
            
        # Add more validation as needed
        
    except Exception as e:
        errors.append(f"Configuration validation error: {str(e)}")
        
    return errors