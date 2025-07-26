# core/service_factory.py
"""
Service factory for creating and configuring core services with dependency injection.
Provides factory methods for common service configurations.
"""
from typing import Optional, Dict, Any
from core.interfaces import IAlertManager, IBudgetSentinel
from core.alert_service import AlertService
from core.token_budget_sentinel import TokenBudgetSentinel
from core.service_registry import ServiceRegistry


class ServiceFactory:
    """Factory for creating and configuring core services"""
    
    def __init__(self, registry: Optional[ServiceRegistry] = None):
        self.registry = registry or ServiceRegistry()
        
    def create_alert_service(self, log_file_path: Optional[str] = None) -> IAlertManager:
        """Create and configure an alert service"""
        return AlertService(log_file_path)
        
    def create_token_budget_sentinel(
        self, 
        max_tokens: int, 
        alert_service: Optional[IAlertManager] = None
    ) -> IBudgetSentinel:
        """Create and configure a token budget sentinel"""
        if alert_service is None:
            alert_service = self.registry.get_optional("alert_service")
        return TokenBudgetSentinel(max_tokens, alert_service)
        
    def setup_default_services(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Setup default services in the registry"""
        config = config or {}
        
        # Create alert service
        alert_log_path = config.get("alert_log_path", "logs/alerts.log")
        alert_service = self.create_alert_service(alert_log_path)
        self.registry.register("alert_service", alert_service)
        
        # Register factory functions for other services
        default_token_limit = config.get("default_token_limit", 10000)
        self.registry.register_factory(
            "token_budget_sentinel",
            lambda max_tokens=default_token_limit: self.create_token_budget_sentinel(
                max_tokens, self.registry.get("alert_service")
            )
        )
        
    def get_registry(self) -> ServiceRegistry:
        """Get the service registry"""
        return self.registry


def create_default_factory(config: Optional[Dict[str, Any]] = None) -> ServiceFactory:
    """Create a service factory with default configuration"""
    factory = ServiceFactory()
    factory.setup_default_services(config)
    return factory