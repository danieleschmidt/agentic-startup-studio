# core/service_registry.py
"""
Service registry for dependency injection and service discovery.
Implements the service locator pattern with thread-safe operations.
"""
import threading
from typing import Any, Dict, List, Optional, Type, TypeVar
from core.interfaces import IServiceRegistry


T = TypeVar('T')


class ServiceRegistry(IServiceRegistry):
    """
    Thread-safe service registry implementing service locator pattern.
    Allows registration and retrieval of service instances with lifecycle management.
    """
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._singletons: Dict[str, Any] = {}
        self._factories: Dict[str, Any] = {}
        self._lock = threading.RLock()
        
    def register(self, service_name: str, service_instance: Any) -> None:
        """Register a service instance"""
        if not service_name:
            raise ValueError("Service name cannot be empty")
            
        with self._lock:
            self._services[service_name] = service_instance
            
    def register_singleton(self, service_name: str, service_class: Type[T], *args, **kwargs) -> None:
        """Register a service as singleton (lazy instantiation)"""
        if not service_name:
            raise ValueError("Service name cannot be empty")
            
        with self._lock:
            self._singletons[service_name] = (service_class, args, kwargs)
            
    def register_factory(self, service_name: str, factory_func: Any) -> None:
        """Register a factory function for creating service instances"""
        if not service_name:
            raise ValueError("Service name cannot be empty")
            
        with self._lock:
            self._factories[service_name] = factory_func
            
    def get(self, service_name: str) -> Any:
        """Retrieve a registered service"""
        with self._lock:
            # Check for direct instance registration
            if service_name in self._services:
                return self._services[service_name]
                
            # Check for singleton registration
            if service_name in self._singletons:
                if service_name not in self._services:
                    service_class, args, kwargs = self._singletons[service_name]
                    instance = service_class(*args, **kwargs)
                    self._services[service_name] = instance
                return self._services[service_name]
                
            # Check for factory registration
            if service_name in self._factories:
                factory = self._factories[service_name]
                return factory()
                
            raise KeyError(f"Service '{service_name}' not registered")
            
    def get_optional(self, service_name: str) -> Optional[Any]:
        """Retrieve a service if registered, None otherwise"""
        try:
            return self.get(service_name)
        except KeyError:
            return None
            
    def is_registered(self, service_name: str) -> bool:
        """Check if a service is registered"""
        with self._lock:
            return (service_name in self._services or 
                   service_name in self._singletons or 
                   service_name in self._factories)
            
    def unregister(self, service_name: str) -> bool:
        """Unregister a service"""
        with self._lock:
            removed = False
            if service_name in self._services:
                del self._services[service_name]
                removed = True
            if service_name in self._singletons:
                del self._singletons[service_name]
                removed = True
            if service_name in self._factories:
                del self._factories[service_name]
                removed = True
            return removed
            
    def list_services(self) -> List[str]:
        """List all registered service names"""
        with self._lock:
            all_services = set()
            all_services.update(self._services.keys())
            all_services.update(self._singletons.keys())
            all_services.update(self._factories.keys())
            return sorted(list(all_services))
            
    def clear(self) -> None:
        """Clear all registered services (useful for testing)"""
        with self._lock:
            self._services.clear()
            self._singletons.clear()
            self._factories.clear()


# Global service registry instance
_global_registry: Optional[ServiceRegistry] = None
_registry_lock = threading.Lock()


def get_global_registry() -> ServiceRegistry:
    """Get the global service registry instance (singleton)"""
    global _global_registry
    
    if _global_registry is None:
        with _registry_lock:
            if _global_registry is None:
                _global_registry = ServiceRegistry()
                
    return _global_registry


def register_service(service_name: str, service_instance: Any) -> None:
    """Convenience function to register service in global registry"""
    get_global_registry().register(service_name, service_instance)


def get_service(service_name: str) -> Any:
    """Convenience function to get service from global registry"""
    return get_global_registry().get(service_name)


def get_service_optional(service_name: str) -> Optional[Any]:
    """Convenience function to get optional service from global registry"""
    return get_global_registry().get_optional(service_name)