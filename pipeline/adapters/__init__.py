"""
External API Adapters for Agentic Startup Studio.

This module provides resilient adapters for external service integration
using circuit breaker patterns, retry logic, and comprehensive error handling.
"""

from .base_adapter import (
    AdapterConfig,
    AdapterError,
    APIError,
    AuthenticationError,
    BaseAdapter,
    ConnectionError,
    RateLimitError,
    TimeoutError,
)
from .flyio_adapter import (
    AppConfig,
    AppMetrics,
    AppStatus,
    DeploymentConfig,
    FlyioAdapter,
    FlyioConfig,
    MachineConfig,
    RegionCode,
    ScalingConfig,
    create_flyio_adapter,
)
from .google_ads_adapter import (
    BiddingStrategy,
    CampaignData,
    CampaignStatus,
    GoogleAdsAdapter,
    GoogleAdsConfig,
    KeywordData,
    PerformanceMetrics,
    create_google_ads_adapter,
)
from .posthog_adapter import (
    AnalyticsQuery,
    EventData,
    EventType,
    FeatureFlagData,
    FeatureFlagType,
    PostHogAdapter,
    PostHogConfig,
    UserData,
    create_posthog_adapter,
)

__all__ = [
    # Base Adapter
    "BaseAdapter",
    "AdapterConfig",
    "AdapterError",
    "ConnectionError",
    "TimeoutError",
    "RateLimitError",
    "AuthenticationError",
    "APIError",

    # Google Ads Adapter
    "GoogleAdsAdapter",
    "GoogleAdsConfig",
    "CampaignData",
    "KeywordData",
    "PerformanceMetrics",
    "CampaignStatus",
    "BiddingStrategy",
    "create_google_ads_adapter",

    # PostHog Adapter
    "PostHogAdapter",
    "PostHogConfig",
    "EventData",
    "UserData",
    "FeatureFlagData",
    "AnalyticsQuery",
    "EventType",
    "FeatureFlagType",
    "create_posthog_adapter",

    # Fly.io Adapter
    "FlyioAdapter",
    "FlyioConfig",
    "AppConfig",
    "DeploymentConfig",
    "ScalingConfig",
    "MachineConfig",
    "AppMetrics",
    "AppStatus",
    "RegionCode",
    "create_flyio_adapter",

    # Utility Functions
    "get_adapter",
    "create_adapters_from_config",
    "get_all_adapter_health",
    "ADAPTER_REGISTRY"
]

# Adapter registry for easy access
ADAPTER_REGISTRY = {
    "google_ads": GoogleAdsAdapter,
    "posthog": PostHogAdapter,
    "flyio": FlyioAdapter
}


def get_adapter(adapter_name: str, config: dict):
    """
    Get adapter instance by name.
    
    Args:
        adapter_name: Name of the adapter (google_ads, posthog, flyio)
        config: Configuration dictionary for the adapter
        
    Returns:
        Adapter instance
        
    Raises:
        ValueError: If adapter name is not found
    """
    if adapter_name not in ADAPTER_REGISTRY:
        raise ValueError(f"Unknown adapter: {adapter_name}. Available: {list(ADAPTER_REGISTRY.keys())}")

    adapter_class = ADAPTER_REGISTRY[adapter_name]
    return adapter_class(config)


async def get_all_adapter_health() -> dict:
    """
    Get health status for all registered adapters.
    
    Returns:
        Dictionary mapping adapter names to their health status
    """
    from pipeline.config.settings import get_settings

    health_status = {}
    settings = get_settings()

    # Check Google Ads adapter
    try:
        if hasattr(settings, 'GOOGLE_ADS_DEVELOPER_TOKEN') and settings.GOOGLE_ADS_DEVELOPER_TOKEN:
            adapter = create_google_ads_adapter()
            async with adapter:
                health_status['google_ads'] = await adapter.health_check()
        else:
            health_status['google_ads'] = {
                'status': 'not_configured',
                'service': 'Google Ads API',
                'error': 'API credentials not configured'
            }
    except Exception as e:
        health_status['google_ads'] = {
            'status': 'error',
            'service': 'Google Ads API',
            'error': str(e)
        }

    # Check PostHog adapter
    try:
        if hasattr(settings, 'POSTHOG_PROJECT_API_KEY') and settings.POSTHOG_PROJECT_API_KEY:
            adapter = create_posthog_adapter()
            async with adapter:
                health_status['posthog'] = await adapter.health_check()
        else:
            health_status['posthog'] = {
                'status': 'not_configured',
                'service': 'PostHog Analytics',
                'error': 'API credentials not configured'
            }
    except Exception as e:
        health_status['posthog'] = {
            'status': 'error',
            'service': 'PostHog Analytics',
            'error': str(e)
        }

    # Check Fly.io adapter
    try:
        if hasattr(settings, 'FLYIO_API_TOKEN') and settings.FLYIO_API_TOKEN:
            adapter = create_flyio_adapter()
            async with adapter:
                health_status['flyio'] = await adapter.health_check()
        else:
            health_status['flyio'] = {
                'status': 'not_configured',
                'service': 'Fly.io Platform',
                'error': 'API credentials not configured'
            }
    except Exception as e:
        health_status['flyio'] = {
            'status': 'error',
            'service': 'Fly.io Platform',
            'error': str(e)
        }

    return health_status


def get_available_adapters() -> list:
    """
    Get list of available adapter names.
    
    Returns:
        List of available adapter names
    """
    return list(ADAPTER_REGISTRY.keys())


def get_adapter_factory(adapter_name: str):
    """
    Get factory function for creating adapter with environment configuration.
    
    Args:
        adapter_name: Name of the adapter
        
    Returns:
        Factory function for the adapter
        
    Raises:
        ValueError: If adapter name is not found
    """
    factory_map = {
        'google_ads': create_google_ads_adapter,
        'posthog': create_posthog_adapter,
        'flyio': create_flyio_adapter
    }

    if adapter_name not in factory_map:
        raise ValueError(f"Unknown adapter: {adapter_name}. Available: {list(factory_map.keys())}")

    return factory_map[adapter_name]


def create_adapters_from_config(adapters_config: dict) -> dict:
    """
    Create multiple adapter instances from configuration.
    
    Args:
        adapters_config: Dictionary mapping adapter names to their configs
        
    Returns:
        Dictionary mapping adapter names to their instances
    """
    adapters = {}

    for adapter_name, config in adapters_config.items():
        if adapter_name in ADAPTER_REGISTRY:
            adapters[adapter_name] = get_adapter(adapter_name, config)

    return adapters
