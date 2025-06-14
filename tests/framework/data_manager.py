"""
Test data management for the comprehensive testing framework.

This module provides test data generation, mock service orchestration,
environment isolation, and cleanup mechanisms for all validation areas.
"""

import os
import json
import asyncio
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Generator, AsyncGenerator
from dataclasses import dataclass
from contextlib import asynccontextmanager, contextmanager
from unittest.mock import Mock, AsyncMock
import logging

import faker
import docker
import aiohttp
from aioresponses import aioresponses
import responses
import httpx

from tests.framework.config import (
    get_data_manager_config, 
    get_security_config,
    get_framework_config,
    DataManagerConfig
)

logger = logging.getLogger(__name__)


@dataclass
class TestDataSet:
    """Test data set metadata and content."""
    name: str
    data: Dict[str, Any]
    schema: Optional[Dict[str, Any]] = None
    source: str = "synthetic"
    created_at: Optional[str] = None
    tags: Optional[List[str]] = None


@dataclass
class MockServiceConfig:
    """Configuration for mock service setup."""
    name: str
    port: int
    endpoints: List[Dict[str, Any]]
    response_delay: float = 0.0
    failure_rate: float = 0.0
    container_image: Optional[str] = None
    environment_vars: Optional[Dict[str, str]] = None


class SyntheticDataGenerator:
    """Generates synthetic test data using Faker and custom schemas."""
    
    def __init__(self, seed: int = 42, locale: str = 'en_US'):
        """Initialize synthetic data generator."""
        self.fake = faker.Faker(locale)
        self.fake.seed_instance(seed)
        self._seed = seed
        
    def generate_startup_idea(self) -> Dict[str, Any]:
        """Generate synthetic startup idea data."""
        categories = ['ai_ml', 'fintech', 'healthtech', 'edtech', 'e_commerce', 'saas']
        statuses = ['draft', 'validated', 'in_development', 'launched']
        
        return {
            'title': f"{self.fake.catch_phrase()} for {self.fake.word()}".title(),
            'description': self.fake.text(max_nb_chars=500),
            'category': self.fake.random_element(categories),
            'tags': [self.fake.word() for _ in range(self.fake.random_int(1, 5))],
            'evidence': self.fake.paragraph(),
            'status': self.fake.random_element(statuses),
            'created_by': self.fake.user_name(),
            'market_size': self.fake.random_int(1000000, 10000000000),
            'target_audience': self.fake.job(),
            'business_model': self.fake.random_element(['subscription', 'freemium', 'marketplace', 'saas']),
            'competition_level': self.fake.random_element(['low', 'medium', 'high']),
            'implementation_complexity': self.fake.random_int(1, 10)
        }
    
    def generate_user_data(self, anonymized: bool = True) -> Dict[str, Any]:
        """Generate synthetic user data with optional anonymization."""
        user_data = {
            'user_id': str(self.fake.uuid4()),
            'email': self.fake.email(),
            'first_name': self.fake.first_name(),
            'last_name': self.fake.last_name(),
            'phone': self.fake.phone_number(),
            'address': {
                'street': self.fake.street_address(),
                'city': self.fake.city(),
                'state': self.fake.state(),
                'zip_code': self.fake.zipcode(),
                'country': self.fake.country()
            },
            'preferences': {
                'newsletter': self.fake.boolean(),
                'notifications': self.fake.boolean(),
                'theme': self.fake.random_element(['light', 'dark'])
            },
            'created_at': self.fake.date_time_this_year().isoformat()
        }
        
        if anonymized:
            # Apply data anonymization
            user_data['email'] = f"user_{user_data['user_id'][:8]}@example.com"
            user_data['first_name'] = f"User{self.fake.random_int(1000, 9999)}"
            user_data['last_name'] = "Test"
            user_data['phone'] = "555-0000"
            
        return user_data
    
    def generate_api_response(self, endpoint: str, success: bool = True) -> Dict[str, Any]:
        """Generate synthetic API response data."""
        if success:
            return {
                'status': 'success',
                'data': self.fake.pydict(nb_elements=5),
                'timestamp': self.fake.date_time().isoformat(),
                'request_id': str(self.fake.uuid4()),
                'endpoint': endpoint
            }
        else:
            return {
                'status': 'error',
                'error': {
                    'code': self.fake.random_int(400, 500),
                    'message': self.fake.sentence(),
                    'details': self.fake.text(max_nb_chars=200)
                },
                'timestamp': self.fake.date_time().isoformat(),
                'request_id': str(self.fake.uuid4()),
                'endpoint': endpoint
            }
    
    def generate_performance_metrics(self) -> Dict[str, Any]:
        """Generate synthetic performance metrics."""
        return {
            'cpu_usage': self.fake.random.uniform(10.0, 90.0),
            'memory_usage': self.fake.random_int(100, 2048),  # MB
            'disk_usage': self.fake.random.uniform(20.0, 80.0),
            'network_io': {
                'bytes_sent': self.fake.random_int(1000, 1000000),
                'bytes_received': self.fake.random_int(1000, 1000000)
            },
            'response_time': self.fake.random.uniform(50.0, 2000.0),  # ms
            'throughput': self.fake.random_int(100, 10000),  # requests/second
            'error_rate': self.fake.random.uniform(0.0, 5.0),  # percentage
            'timestamp': self.fake.date_time().isoformat()
        }


class MockServiceManager:
    """Manages mock services for testing external dependencies."""
    
    def __init__(self, config: DataManagerConfig):
        """Initialize mock service manager."""
        self.config = config
        self.active_services: Dict[str, Dict[str, Any]] = {}
        self.docker_client = None
        self._port_allocator = self._create_port_allocator()
        
    def _create_port_allocator(self) -> Generator[int, None, None]:
        """Create port allocator for mock services."""
        start_port, end_port = map(int, self.config.mock_service_port_range.split('-'))
        for port in range(start_port, end_port + 1):
            if self._is_port_available(port):
                yield port
    
    def _is_port_available(self, port: int) -> bool:
        """Check if port is available for use."""
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(('localhost', port))
                return True
        except OSError:
            return False
    
    @asynccontextmanager
    async def mock_http_service(self, service_config: MockServiceConfig):
        """Create mock HTTP service using aioresponses."""
        try:
            with aioresponses() as mock_session:
                base_url = f"http://localhost:{service_config.port}"
                
                for endpoint_config in service_config.endpoints:
                    method = endpoint_config.get('method', 'GET').upper()
                    path = endpoint_config['path']
                    response_data = endpoint_config.get('response', {})
                    status_code = endpoint_config.get('status', 200)
                    
                    url = f"{base_url}{path}"
                    
                    if method == 'GET':
                        mock_session.get(url, payload=response_data, status=status_code)
                    elif method == 'POST':
                        mock_session.post(url, payload=response_data, status=status_code)
                    elif method == 'PUT':
                        mock_session.put(url, payload=response_data, status=status_code)
                    elif method == 'DELETE':
                        mock_session.delete(url, status=status_code)
                
                self.active_services[service_config.name] = {
                    'type': 'http_mock',
                    'config': service_config,
                    'base_url': base_url
                }
                
                logger.info(f"Started mock HTTP service: {service_config.name} on {base_url}")
                yield base_url
                
        finally:
            if service_config.name in self.active_services:
                del self.active_services[service_config.name]
                logger.info(f"Stopped mock HTTP service: {service_config.name}")
    
    @asynccontextmanager
    async def mock_database_service(self, service_config: MockServiceConfig):
        """Create mock database service using Docker."""
        if not self.config.enable_mock_services:
            raise RuntimeError("Mock services are disabled in configuration")
        
        container = None
        try:
            if not self.docker_client:
                self.docker_client = docker.from_env()
            
            port = next(self._port_allocator)
            container_name = f"test_db_{service_config.name}"
            
            # Use PostgreSQL as default test database
            image = service_config.container_image or "postgres:13-alpine"
            environment = {
                'POSTGRES_DB': 'testdb',
                'POSTGRES_USER': 'testuser',
                'POSTGRES_PASSWORD': 'testpass',
                **(service_config.environment_vars or {})
            }
            
            container = self.docker_client.containers.run(
                image,
                name=container_name,
                ports={'5432/tcp': port},
                environment=environment,
                detach=True,
                remove=True
            )
            
            # Wait for database to be ready
            await self._wait_for_service_ready(port, timeout=30)
            
            connection_string = f"postgresql://testuser:testpass@localhost:{port}/testdb"
            
            self.active_services[service_config.name] = {
                'type': 'database_mock',
                'config': service_config,
                'container': container,
                'connection_string': connection_string,
                'port': port
            }
            
            logger.info(f"Started mock database service: {service_config.name} on port {port}")
            yield connection_string
            
        except Exception as e:
            logger.error(f"Failed to start mock database service: {e}")
            if container:
                container.stop()
            raise
        finally:
            if container:
                container.stop()
            if service_config.name in self.active_services:
                del self.active_services[service_config.name]
                logger.info(f"Stopped mock database service: {service_config.name}")
    
    async def _wait_for_service_ready(self, port: int, timeout: int = 30):
        """Wait for service to become ready."""
        import socket
        import time
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(1)
                    result = sock.connect_ex(('localhost', port))
                    if result == 0:
                        return
            except Exception:
                pass
            await asyncio.sleep(1)
        
        raise TimeoutError(f"Service on port {port} did not become ready within {timeout} seconds")


class DataManager:
    """Comprehensive test data management system."""
    
    def __init__(self, config: Optional[DataManagerConfig] = None):
        """Initialize data manager."""
        self.config = config or get_data_manager_config()
        self.security_config = get_security_config()
        self.framework_config = get_framework_config()
        
        self.synthetic_generator = SyntheticDataGenerator(seed=self.config.synthetic_data_seed)
        self.mock_service_manager = MockServiceManager(self.config)
        
        self.active_datasets: Dict[str, TestDataSet] = {}
        self.temp_directories: List[Path] = []
        
        # Ensure required directories exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure required directories exist."""
        directories = [
            self.config.test_data_root,
            self.config.synthetic_data_cache,
            self.config.mock_data_path
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def load_test_data(self, data_path: Union[str, Path], dataset_name: str) -> TestDataSet:
        """Load test data from file system."""
        try:
            file_path = Path(data_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Test data file not found: {data_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() == '.json':
                    data = json.load(f)
                else:
                    data = {'content': f.read()}
            
            dataset = TestDataSet(
                name=dataset_name,
                data=data,
                source=str(file_path),
                created_at=file_path.stat().st_mtime
            )
            
            self.active_datasets[dataset_name] = dataset
            logger.info(f"Loaded test dataset: {dataset_name} from {data_path}")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load test data from {data_path}: {e}")
            raise
    
    def generate_synthetic_data(
        self, 
        schema: str, 
        count: int = 10, 
        dataset_name: Optional[str] = None
    ) -> TestDataSet:
        """Generate synthetic test data based on schema type."""
        if count > self.config.max_synthetic_records:
            raise ValueError(f"Requested count {count} exceeds maximum {self.config.max_synthetic_records}")
        
        dataset_name = dataset_name or f"synthetic_{schema}_{count}"
        
        try:
            if schema == 'startup_ideas':
                data = [self.synthetic_generator.generate_startup_idea() for _ in range(count)]
            elif schema == 'users':
                anonymized = self.security_config.enable_data_anonymization
                data = [self.synthetic_generator.generate_user_data(anonymized) for _ in range(count)]
            elif schema == 'api_responses':
                data = [
                    self.synthetic_generator.generate_api_response(f"/api/test/{i}", 
                    success=i % 10 != 0) for i in range(count)
                ]
            elif schema == 'performance_metrics':
                data = [self.synthetic_generator.generate_performance_metrics() for _ in range(count)]
            else:
                raise ValueError(f"Unknown schema type: {schema}")
            
            dataset = TestDataSet(
                name=dataset_name,
                data={'records': data, 'count': count},
                schema={'type': schema, 'count': count},
                source='synthetic'
            )
            
            self.active_datasets[dataset_name] = dataset
            
            # Cache if enabled
            if self.config.enable_synthetic_data:
                self._cache_dataset(dataset)
            
            logger.info(f"Generated synthetic dataset: {dataset_name} with {count} {schema} records")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to generate synthetic data for schema {schema}: {e}")
            raise
    
    def _cache_dataset(self, dataset: TestDataSet):
        """Cache dataset to file system."""
        try:
            cache_dir = Path(self.config.synthetic_data_cache)
            cache_file = cache_dir / f"{dataset.name}.json"
            
            cache_data = {
                'name': dataset.name,
                'data': dataset.data,
                'schema': dataset.schema,
                'source': dataset.source,
                'created_at': dataset.created_at,
                'tags': dataset.tags
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, default=str)
            
            logger.debug(f"Cached dataset {dataset.name} to {cache_file}")
            
        except Exception as e:
            logger.warning(f"Failed to cache dataset {dataset.name}: {e}")
    
    @contextmanager
    def isolated_environment(self, environment_name: str):
        """Create isolated test environment."""
        temp_dir = None
        try:
            # Create temporary directory for isolated environment
            temp_dir = Path(tempfile.mkdtemp(prefix=f"test_env_{environment_name}_"))
            self.temp_directories.append(temp_dir)
            
            # Set up environment-specific configuration
            old_env = os.environ.copy()
            os.environ.update({
                'TEST_ENVIRONMENT': 'isolated',
                'TEST_DATA_ROOT': str(temp_dir / 'data'),
                'TEST_CACHE_DIR': str(temp_dir / 'cache'),
                'TEST_LOGS_DIR': str(temp_dir / 'logs')
            })
            
            # Create environment directories
            for subdir in ['data', 'cache', 'logs']:
                (temp_dir / subdir).mkdir(exist_ok=True)
            
            logger.info(f"Created isolated environment: {environment_name} at {temp_dir}")
            yield temp_dir
            
        finally:
            # Restore environment
            os.environ.clear()
            os.environ.update(old_env)
            
            # Cleanup if auto-cleanup is enabled
            if self.config.auto_cleanup and temp_dir:
                self._cleanup_directory(temp_dir, preserve_on_failure=False)
    
    async def setup_mock_services(self, service_configs: List[MockServiceConfig]):
        """Set up multiple mock services."""
        active_services = []
        
        try:
            for config in service_configs:
                if config.container_image:
                    # Database or container-based service
                    service_context = self.mock_service_manager.mock_database_service(config)
                else:
                    # HTTP-based service
                    service_context = self.mock_service_manager.mock_http_service(config)
                
                active_services.append(service_context)
            
            # Start all services concurrently
            async with asyncio.gather(*[service.__aenter__() for service in active_services]):
                logger.info(f"Started {len(service_configs)} mock services")
                yield self.mock_service_manager.active_services
                
        except Exception as e:
            logger.error(f"Failed to setup mock services: {e}")
            raise
        finally:
            # Cleanup happens automatically via context managers
            pass
    
    def get_dataset(self, dataset_name: str) -> Optional[TestDataSet]:
        """Get active dataset by name."""
        return self.active_datasets.get(dataset_name)
    
    def list_datasets(self) -> List[str]:
        """List all active dataset names."""
        return list(self.active_datasets.keys())
    
    def cleanup_test_data(self, preserve_on_failure: Optional[bool] = None):
        """Clean up all test data and temporary resources."""
        preserve = preserve_on_failure if preserve_on_failure is not None else self.config.preserve_on_failure
        
        try:
            # Clean up datasets
            self.active_datasets.clear()
            
            # Clean up temporary directories
            for temp_dir in self.temp_directories:
                self._cleanup_directory(temp_dir, preserve)
            
            self.temp_directories.clear()
            
            # Clean up mock services
            if hasattr(self.mock_service_manager, 'docker_client') and self.mock_service_manager.docker_client:
                self.mock_service_manager.docker_client.close()
            
            logger.info("Completed test data cleanup")
            
        except Exception as e:
            logger.error(f"Error during test data cleanup: {e}")
            if not preserve:
                raise
    
    def _cleanup_directory(self, directory: Path, preserve_on_failure: bool):
        """Clean up a directory with failure preservation logic."""
        try:
            if preserve_on_failure and self._has_test_failures():
                logger.info(f"Preserving directory due to test failures: {directory}")
                return
            
            if directory.exists():
                shutil.rmtree(directory)
                logger.debug(f"Cleaned up directory: {directory}")
        except Exception as e:
            logger.warning(f"Failed to cleanup directory {directory}: {e}")
    
    def _has_test_failures(self) -> bool:
        """Check if there were test failures (simplified implementation)."""
        # This would integrate with test result tracking
        # For now, return False to allow cleanup
        return False
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        self.cleanup_test_data(preserve_on_failure=exc_type is not None)


# Factory functions for common use cases
def create_startup_test_data(count: int = 50) -> TestDataSet:
    """Factory function to create startup idea test data."""
    manager = DataManager()
    return manager.generate_synthetic_data('startup_ideas', count)


def create_performance_test_data(count: int = 100) -> TestDataSet:
    """Factory function to create performance metrics test data."""
    manager = DataManager()
    return manager.generate_synthetic_data('performance_metrics', count)


async def create_mock_api_service(endpoints: List[Dict[str, Any]], port: int = 8080) -> MockServiceConfig:
    """Factory function to create mock API service configuration."""
    return MockServiceConfig(
        name="mock_api",
        port=port,
        endpoints=endpoints,
        response_delay=0.1,
        failure_rate=0.05
    )