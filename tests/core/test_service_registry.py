# tests/core/test_service_registry.py
import pytest
import threading
import time
from core.service_registry import ServiceRegistry, get_global_registry
from core.interfaces import IAlertManager


class MockService:
    """Mock service for testing"""
    def __init__(self, name: str = "mock"):
        self.name = name
        
    def get_name(self) -> str:
        return self.name


class MockAlertManager(IAlertManager):
    """Mock alert manager for testing"""
    def __init__(self):
        self.alerts = []
        
    def record_alert(self, message: str, severity: str = "warning") -> None:
        self.alerts.append({"message": message, "severity": severity})
        
    def get_alerts(self):
        return self.alerts.copy()


class TestServiceRegistry:
    
    def setup_method(self):
        """Setup test environment"""
        self.registry = ServiceRegistry()
        
    def test_register_and_get_service(self):
        """Test basic service registration and retrieval"""
        service = MockService("test")
        self.registry.register("test_service", service)
        
        retrieved = self.registry.get("test_service")
        assert retrieved is service
        assert retrieved.get_name() == "test"
        
    def test_get_nonexistent_service_raises_error(self):
        """Test that getting non-existent service raises KeyError"""
        with pytest.raises(KeyError, match="Service 'nonexistent' not registered"):
            self.registry.get("nonexistent")
            
    def test_get_optional_service(self):
        """Test optional service retrieval"""
        # Non-existent service returns None
        assert self.registry.get_optional("nonexistent") is None
        
        # Existing service returns the service
        service = MockService("test")
        self.registry.register("test_service", service)
        assert self.registry.get_optional("test_service") is service
        
    def test_register_singleton(self):
        """Test singleton service registration"""
        self.registry.register_singleton("singleton_service", MockService, "singleton")
        
        # First retrieval creates instance
        service1 = self.registry.get("singleton_service")
        assert service1.get_name() == "singleton"
        
        # Second retrieval returns same instance
        service2 = self.registry.get("singleton_service")
        assert service1 is service2
        
    def test_register_factory(self):
        """Test factory service registration"""
        call_count = 0
        
        def factory():
            nonlocal call_count
            call_count += 1
            return MockService(f"factory_{call_count}")
            
        self.registry.register_factory("factory_service", factory)
        
        # Each retrieval creates new instance
        service1 = self.registry.get("factory_service")
        service2 = self.registry.get("factory_service")
        
        assert service1.get_name() == "factory_1"
        assert service2.get_name() == "factory_2"
        assert service1 is not service2
        
    def test_is_registered(self):
        """Test service registration check"""
        assert not self.registry.is_registered("test_service")
        
        self.registry.register("test_service", MockService())
        assert self.registry.is_registered("test_service")
        
        self.registry.register_singleton("singleton_service", MockService)
        assert self.registry.is_registered("singleton_service")
        
    def test_unregister_service(self):
        """Test service unregistration"""
        service = MockService()
        self.registry.register("test_service", service)
        
        assert self.registry.is_registered("test_service")
        assert self.registry.unregister("test_service")
        assert not self.registry.is_registered("test_service")
        
        # Unregistering non-existent service returns False
        assert not self.registry.unregister("nonexistent")
        
    def test_list_services(self):
        """Test listing all registered services"""
        assert self.registry.list_services() == []
        
        self.registry.register("service1", MockService())
        self.registry.register_singleton("service2", MockService)
        self.registry.register_factory("service3", lambda: MockService())
        
        services = self.registry.list_services()
        assert sorted(services) == ["service1", "service2", "service3"]
        
    def test_clear_services(self):
        """Test clearing all services"""
        self.registry.register("service1", MockService())
        self.registry.register_singleton("service2", MockService)
        
        assert len(self.registry.list_services()) == 2
        
        self.registry.clear()
        assert len(self.registry.list_services()) == 0
        
    def test_empty_service_name_raises_error(self):
        """Test that empty service name raises ValueError"""
        with pytest.raises(ValueError, match="Service name cannot be empty"):
            self.registry.register("", MockService())
            
        with pytest.raises(ValueError, match="Service name cannot be empty"):
            self.registry.register_singleton("", MockService)
            
    def test_thread_safety(self):
        """Test thread safety of service registry"""
        services_created = []
        
        def register_service(index: int):
            service = MockService(f"service_{index}")
            self.registry.register(f"service_{index}", service)
            services_created.append(service)
            
        # Create multiple threads registering services
        threads = []
        for i in range(10):
            thread = threading.Thread(target=register_service, args=(i,))
            threads.append(thread)
            thread.start()
            
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
            
        # Verify all services were registered
        assert len(services_created) == 10
        assert len(self.registry.list_services()) == 10
        
        # Verify all services can be retrieved
        for i in range(10):
            service = self.registry.get(f"service_{i}")
            assert service.get_name() == f"service_{i}"


class TestGlobalRegistry:
    
    def test_global_registry_singleton(self):
        """Test that global registry is singleton"""
        registry1 = get_global_registry()
        registry2 = get_global_registry()
        
        assert registry1 is registry2
        
    def test_global_registry_thread_safety(self):
        """Test thread safety of global registry access"""
        registries = []
        
        def get_registry():
            registries.append(get_global_registry())
            
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=get_registry)
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
            
        # All threads should get the same registry instance
        assert all(registry is registries[0] for registry in registries)