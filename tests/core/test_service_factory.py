# tests/core/test_service_factory.py
import pytest
from core.service_factory import ServiceFactory, create_default_factory
from core.service_registry import ServiceRegistry
from core.interfaces import IAlertManager, IBudgetSentinel


class TestServiceFactory:
    
    def setup_method(self):
        """Setup test environment"""
        self.registry = ServiceRegistry()
        self.factory = ServiceFactory(self.registry)
        
    def test_create_alert_service(self):
        """Test alert service creation"""
        alert_service = self.factory.create_alert_service()
        
        assert isinstance(alert_service, IAlertManager)
        
        # Test functionality
        alert_service.record_alert("test message", "warning")
        alerts = alert_service.get_alerts()
        
        assert len(alerts) == 1
        assert alerts[0]["message"] == "test message"
        assert alerts[0]["severity"] == "warning"
        
    def test_create_alert_service_with_log_path(self):
        """Test alert service creation with custom log path"""
        alert_service = self.factory.create_alert_service("custom/alerts.log")
        
        assert isinstance(alert_service, IAlertManager)
        
        # Access underlying manager to check log path
        underlying = alert_service.get_underlying_manager()
        assert underlying.log_file_path == "custom/alerts.log"
        
    def test_create_token_budget_sentinel(self):
        """Test token budget sentinel creation"""
        alert_service = self.factory.create_alert_service()
        sentinel = self.factory.create_token_budget_sentinel(5000, alert_service)
        
        assert isinstance(sentinel, IBudgetSentinel)
        assert sentinel.max_budget == 5000
        assert sentinel.alert_manager is alert_service
        
    def test_create_token_budget_sentinel_without_alert_service(self):
        """Test token budget sentinel creation without alert service"""
        sentinel = self.factory.create_token_budget_sentinel(3000)
        
        assert isinstance(sentinel, IBudgetSentinel)
        assert sentinel.max_budget == 3000
        assert sentinel.alert_manager is None
        
    def test_setup_default_services(self):
        """Test default services setup"""
        config = {"alert_log_path": "test/alerts.log"}
        self.factory.setup_default_services(config)
        
        # Check alert service is registered
        assert self.registry.is_registered("alert_service")
        alert_service = self.registry.get("alert_service")
        assert isinstance(alert_service, IAlertManager)
        
        # Check token budget sentinel factory is registered
        assert self.registry.is_registered("token_budget_sentinel")
        sentinel = self.registry.get("token_budget_sentinel")
        assert isinstance(sentinel, IBudgetSentinel)
        
    def test_setup_default_services_no_config(self):
        """Test default services setup without config"""
        self.factory.setup_default_services()
        
        # Should still create services with defaults
        assert self.registry.is_registered("alert_service")
        assert self.registry.is_registered("token_budget_sentinel")
        
    def test_get_registry(self):
        """Test getting the service registry"""
        registry = self.factory.get_registry()
        assert registry is self.registry
        
    def test_token_budget_sentinel_uses_registered_alert_service(self):
        """Test that token budget sentinel uses registered alert service"""
        self.factory.setup_default_services()
        
        # Get the registered alert service
        alert_service = self.registry.get("alert_service")
        
        # Create sentinel - should use registered alert service automatically
        sentinel = self.registry.get("token_budget_sentinel")
        assert sentinel.alert_manager is alert_service


class TestCreateDefaultFactory:
    
    def test_create_default_factory(self):
        """Test creating default factory"""
        factory = create_default_factory()
        
        assert isinstance(factory, ServiceFactory)
        
        # Should have services already set up
        registry = factory.get_registry()
        assert registry.is_registered("alert_service")
        assert registry.is_registered("token_budget_sentinel")
        
    def test_create_default_factory_with_config(self):
        """Test creating default factory with custom config"""
        config = {"alert_log_path": "custom/path.log"}
        factory = create_default_factory(config)
        
        # Check that config was applied
        alert_service = factory.get_registry().get("alert_service")
        underlying = alert_service.get_underlying_manager()
        assert underlying.log_file_path == "custom/path.log"
        
    def test_factory_integration(self):
        """Test full integration of factory-created services"""
        factory = create_default_factory()
        registry = factory.get_registry()
        
        # Get services
        alert_service = registry.get("alert_service")
        sentinel = registry.get("token_budget_sentinel")
        
        # Test integration - token usage triggering alert
        # Use high usage to trigger alert
        result = sentinel.check_usage(12000, "Integration Test")  # Over 10000 default
        
        # Should return False (budget exceeded)
        assert not result
        
        # Alert should be recorded
        alerts = alert_service.get_alerts()
        assert len(alerts) >= 1
        
        # Find our alert
        integration_alerts = [a for a in alerts if "Integration Test" in a["message"]]
        assert len(integration_alerts) >= 1