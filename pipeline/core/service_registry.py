"""
Service Registry - Dependency injection and service lifecycle management.

Provides centralized service registration, dependency injection, and lifecycle management
for all pipeline services with proper separation of concerns.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar, Union
from uuid import uuid4

T = TypeVar('T')


class ServiceState(Enum):
    """Service lifecycle states."""
    REGISTERED = "registered"
    INITIALIZING = "initializing"
    READY = "ready"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class ServiceMetadata:
    """Service registration metadata."""
    service_id: str
    service_type: Type
    instance: Optional[Any] = None
    state: ServiceState = ServiceState.REGISTERED
    dependencies: List[str] = field(default_factory=list)
    startup_priority: int = 100  # Lower numbers start first
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_health_check: Optional[datetime] = None
    health_status: bool = True
    initialization_args: Dict[str, Any] = field(default_factory=dict)
    lifecycle_callbacks: Dict[str, List[Callable]] = field(default_factory=dict)


class ServiceInterface(ABC):
    """Base interface for all services."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the service."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the service gracefully."""
        pass
    
    async def health_check(self) -> bool:
        """Check service health. Override in subclasses."""
        return True
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get service information. Override in subclasses."""
        return {
            'name': self.__class__.__name__,
            'version': '1.0.0',
            'status': 'ready'
        }


class DependencyResolver:
    """Resolves service dependencies and determines startup order."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def resolve_dependencies(self, services: Dict[str, ServiceMetadata]) -> List[str]:
        """
        Resolve service dependencies and return startup order.
        
        Args:
            services: Dictionary of service metadata
            
        Returns:
            List of service IDs in dependency order
            
        Raises:
            ValueError: If circular dependencies are detected
        """
        # Build dependency graph
        dependency_graph = {}
        for service_id, metadata in services.items():
            dependency_graph[service_id] = metadata.dependencies.copy()
        
        # Detect circular dependencies
        self._detect_circular_dependencies(dependency_graph)
        
        # Topological sort with priority consideration
        return self._topological_sort_with_priority(services, dependency_graph)
    
    def _detect_circular_dependencies(self, graph: Dict[str, List[str]]):
        """Detect circular dependencies using DFS."""
        visited = set()
        rec_stack = set()
        
        def has_cycle(node: str) -> bool:
            if node in rec_stack:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if has_cycle(neighbor):
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in graph:
            if node not in visited:
                if has_cycle(node):
                    raise ValueError(f"Circular dependency detected involving service: {node}")
    
    def _topological_sort_with_priority(
        self, 
        services: Dict[str, ServiceMetadata], 
        graph: Dict[str, List[str]]
    ) -> List[str]:
        """Topological sort considering startup priorities."""
        in_degree = {node: 0 for node in graph}
        
        # Calculate in-degrees
        for node in graph:
            for neighbor in graph[node]:
                if neighbor in in_degree:
                    in_degree[neighbor] += 1
        
        # Priority queue of services with no dependencies
        ready_services = []
        for node, degree in in_degree.items():
            if degree == 0:
                priority = services[node].startup_priority
                ready_services.append((priority, node))
        
        ready_services.sort()  # Sort by priority
        
        result = []
        
        while ready_services:
            _, current = ready_services.pop(0)
            result.append(current)
            
            # Update dependencies
            for neighbor in graph.get(current, []):
                if neighbor in in_degree:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        priority = services[neighbor].startup_priority
                        ready_services.append((priority, neighbor))
                        ready_services.sort()
        
        # Check if all services were processed
        if len(result) != len(graph):
            remaining = set(graph.keys()) - set(result)
            raise ValueError(f"Could not resolve dependencies for services: {remaining}")
        
        return result


class ServiceRegistry:
    """Central service registry with dependency injection."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.services: Dict[str, ServiceMetadata] = {}
        self.dependency_resolver = DependencyResolver()
        self._startup_order: Optional[List[str]] = None
        self._shutdown_order: Optional[List[str]] = None
        self._startup_lock = asyncio.Lock()
        self._shutdown_lock = asyncio.Lock()
        self._registry_state = ServiceState.REGISTERED
    
    def register_service(
        self,
        service_type: Type[T],
        service_id: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
        startup_priority: int = 100,
        initialization_args: Optional[Dict[str, Any]] = None,
        singleton: bool = True
    ) -> str:
        """
        Register a service in the registry.
        
        Args:
            service_type: Service class type
            service_id: Unique service identifier (auto-generated if None)
            dependencies: List of service IDs this service depends on
            startup_priority: Startup priority (lower starts first)
            initialization_args: Arguments for service initialization
            singleton: Whether to create singleton instance
            
        Returns:
            Service ID
        """
        service_id = service_id or f"{service_type.__name__}_{uuid4().hex[:8]}"
        
        if service_id in self.services:
            raise ValueError(f"Service {service_id} is already registered")
        
        metadata = ServiceMetadata(
            service_id=service_id,
            service_type=service_type,
            dependencies=dependencies or [],
            startup_priority=startup_priority,
            initialization_args=initialization_args or {}
        )
        
        self.services[service_id] = metadata
        self._startup_order = None  # Reset cached order
        
        self.logger.info(f"Registered service: {service_id} ({service_type.__name__})")
        return service_id
    
    def register_instance(
        self,
        instance: T,
        service_id: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
        startup_priority: int = 100
    ) -> str:
        """
        Register a service instance directly.
        
        Args:
            instance: Service instance
            service_id: Unique service identifier
            dependencies: List of service IDs this service depends on
            startup_priority: Startup priority
            
        Returns:
            Service ID
        """
        service_type = type(instance)
        service_id = service_id or f"{service_type.__name__}_{uuid4().hex[:8]}"
        
        if service_id in self.services:
            raise ValueError(f"Service {service_id} is already registered")
        
        metadata = ServiceMetadata(
            service_id=service_id,
            service_type=service_type,
            instance=instance,
            state=ServiceState.READY,
            dependencies=dependencies or [],
            startup_priority=startup_priority
        )
        
        self.services[service_id] = metadata
        self._startup_order = None
        
        self.logger.info(f"Registered service instance: {service_id}")
        return service_id
    
    async def get_service(self, service_id: str) -> Any:
        """
        Get service instance by ID.
        
        Args:
            service_id: Service identifier
            
        Returns:
            Service instance
            
        Raises:
            ValueError: If service not found or not ready
        """
        if service_id not in self.services:
            raise ValueError(f"Service {service_id} not found")
        
        metadata = self.services[service_id]
        
        # Ensure service is initialized
        if metadata.state == ServiceState.REGISTERED:
            await self._initialize_service(service_id)
        elif metadata.state not in [ServiceState.READY]:
            raise ValueError(f"Service {service_id} is not ready (state: {metadata.state})")
        
        return metadata.instance
    
    async def get_service_by_type(self, service_type: Type[T]) -> T:
        """
        Get service instance by type.
        
        Args:
            service_type: Service class type
            
        Returns:
            Service instance
        """
        for metadata in self.services.values():
            if metadata.service_type == service_type:
                return await self.get_service(metadata.service_id)
        
        raise ValueError(f"No service of type {service_type.__name__} found")
    
    async def initialize_all(self) -> None:
        """Initialize all registered services in dependency order."""
        async with self._startup_lock:
            if self._registry_state != ServiceState.REGISTERED:
                return
            
            self._registry_state = ServiceState.INITIALIZING
            
            try:
                # Resolve startup order
                if not self._startup_order:
                    self._startup_order = self.dependency_resolver.resolve_dependencies(self.services)
                    self._shutdown_order = list(reversed(self._startup_order))
                
                # Initialize services in order
                for service_id in self._startup_order:
                    await self._initialize_service(service_id)
                
                self._registry_state = ServiceState.READY
                self.logger.info(f"All services initialized: {len(self.services)} services ready")
                
            except Exception as e:
                self._registry_state = ServiceState.FAILED
                self.logger.error(f"Service initialization failed: {e}")
                raise
    
    async def shutdown_all(self) -> None:
        """Shutdown all services in reverse dependency order."""
        async with self._shutdown_lock:
            if self._registry_state in [ServiceState.STOPPING, ServiceState.STOPPED]:
                return
            
            self._registry_state = ServiceState.STOPPING
            
            try:
                # Use cached shutdown order or compute it
                shutdown_order = self._shutdown_order or list(reversed(
                    self.dependency_resolver.resolve_dependencies(self.services)
                ))
                
                # Shutdown services in reverse order
                for service_id in shutdown_order:
                    await self._shutdown_service(service_id)
                
                self._registry_state = ServiceState.STOPPED
                self.logger.info("All services shutdown successfully")
                
            except Exception as e:
                self.logger.error(f"Service shutdown failed: {e}")
                raise
    
    async def _initialize_service(self, service_id: str) -> None:
        """Initialize a single service."""
        metadata = self.services[service_id]
        
        if metadata.state != ServiceState.REGISTERED:
            return
        
        try:
            metadata.state = ServiceState.INITIALIZING
            
            # Create instance if not already created
            if metadata.instance is None:
                if metadata.initialization_args:
                    metadata.instance = metadata.service_type(**metadata.initialization_args)
                else:
                    metadata.instance = metadata.service_type()
            
            # Initialize the service if it implements ServiceInterface
            if isinstance(metadata.instance, ServiceInterface):
                await metadata.instance.initialize()
            
            metadata.state = ServiceState.READY
            metadata.last_health_check = datetime.utcnow()
            
            self.logger.info(f"Service initialized: {service_id}")
            
            # Call lifecycle callbacks
            await self._call_lifecycle_callbacks(service_id, 'on_initialized')
            
        except Exception as e:
            metadata.state = ServiceState.FAILED
            self.logger.error(f"Failed to initialize service {service_id}: {e}")
            raise
    
    async def _shutdown_service(self, service_id: str) -> None:
        """Shutdown a single service."""
        metadata = self.services[service_id]
        
        if metadata.state in [ServiceState.STOPPING, ServiceState.STOPPED]:
            return
        
        try:
            metadata.state = ServiceState.STOPPING
            
            # Call lifecycle callbacks
            await self._call_lifecycle_callbacks(service_id, 'on_shutdown')
            
            # Shutdown the service if it implements ServiceInterface
            if isinstance(metadata.instance, ServiceInterface):
                await metadata.instance.shutdown()
            
            metadata.state = ServiceState.STOPPED
            self.logger.info(f"Service shutdown: {service_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to shutdown service {service_id}: {e}")
            raise
    
    async def _call_lifecycle_callbacks(self, service_id: str, event: str) -> None:
        """Call lifecycle callbacks for service events."""
        metadata = self.services[service_id]
        callbacks = metadata.lifecycle_callbacks.get(event, [])
        
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(metadata.instance)
                else:
                    callback(metadata.instance)
            except Exception as e:
                self.logger.warning(f"Lifecycle callback failed for {service_id}.{event}: {e}")
    
    def add_lifecycle_callback(
        self, 
        service_id: str, 
        event: str, 
        callback: Callable
    ) -> None:
        """Add lifecycle callback for service events."""
        if service_id not in self.services:
            raise ValueError(f"Service {service_id} not found")
        
        metadata = self.services[service_id]
        if event not in metadata.lifecycle_callbacks:
            metadata.lifecycle_callbacks[event] = []
        
        metadata.lifecycle_callbacks[event].append(callback)
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Perform health checks on all services."""
        results = {}
        
        for service_id, metadata in self.services.items():
            try:
                if metadata.instance and isinstance(metadata.instance, ServiceInterface):
                    metadata.health_status = await metadata.instance.health_check()
                    metadata.last_health_check = datetime.utcnow()
                else:
                    metadata.health_status = metadata.state == ServiceState.READY
                
                results[service_id] = metadata.health_status
                
            except Exception as e:
                self.logger.warning(f"Health check failed for {service_id}: {e}")
                metadata.health_status = False
                results[service_id] = False
        
        return results
    
    def get_service_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all registered services."""
        info = {}
        
        for service_id, metadata in self.services.items():
            service_info = {
                'service_id': service_id,
                'service_type': metadata.service_type.__name__,
                'state': metadata.state.value,
                'dependencies': metadata.dependencies,
                'startup_priority': metadata.startup_priority,
                'created_at': metadata.created_at.isoformat(),
                'last_health_check': (
                    metadata.last_health_check.isoformat() 
                    if metadata.last_health_check else None
                ),
                'health_status': metadata.health_status
            }
            
            # Add service-specific info if available
            if metadata.instance and isinstance(metadata.instance, ServiceInterface):
                try:
                    service_info.update(metadata.instance.get_service_info())
                except Exception as e:
                    self.logger.warning(f"Failed to get service info for {service_id}: {e}")
            
            info[service_id] = service_info
        
        return info
    
    @asynccontextmanager
    async def service_context(self):
        """Context manager for service lifecycle management."""
        try:
            await self.initialize_all()
            yield self
        finally:
            await self.shutdown_all()


# Singleton registry instance
_service_registry = None


def get_service_registry() -> ServiceRegistry:
    """Get the global service registry."""
    global _service_registry
    if _service_registry is None:
        _service_registry = ServiceRegistry()
    return _service_registry


# Convenience decorators and functions
def service(
    service_id: Optional[str] = None,
    dependencies: Optional[List[str]] = None,
    startup_priority: int = 100,
    singleton: bool = True
):
    """Decorator to register a service class."""
    def decorator(cls):
        registry = get_service_registry()
        registry.register_service(
            service_type=cls,
            service_id=service_id,
            dependencies=dependencies,
            startup_priority=startup_priority,
            singleton=singleton
        )
        return cls
    return decorator


async def inject(service_id_or_type: Union[str, Type[T]]) -> T:
    """Inject a service dependency."""
    registry = get_service_registry()
    
    if isinstance(service_id_or_type, str):
        return await registry.get_service(service_id_or_type)
    else:
        return await registry.get_service_by_type(service_id_or_type)