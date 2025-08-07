"""
Base Adapter for External Service Integration.

Provides common functionality for all external API adapters including
circuit breaker protection, retry logic, authentication, and error handling.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import aiohttp

from pipeline.infrastructure.circuit_breaker import get_circuit_breaker_registry
from pipeline.infrastructure.observability import get_logger, performance_monitor

logger = get_logger(__name__)


class AuthType(Enum):
    """Authentication types supported by adapters."""
    NONE = "none"
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    OAUTH2 = "oauth2"
    CUSTOM = "custom"


class RetryStrategy(Enum):
    """Retry strategies for failed requests."""
    NONE = "none"
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"


@dataclass
class AdapterConfig:
    """Base configuration for all adapters."""

    # Service connection
    base_url: str
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    retry_delay_seconds: float = 1.0

    # Authentication
    auth_type: AuthType = AuthType.NONE
    api_key: str | None = None
    bearer_token: str | None = None
    username: str | None = None
    password: str | None = None
    oauth_config: dict[str, Any] | None = None

    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60

    # Circuit breaker
    use_circuit_breaker: bool = True
    circuit_breaker_name: str | None = None

    # Monitoring
    enable_metrics: bool = True
    enable_logging: bool = True

    # Custom headers
    custom_headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.base_url:
            raise ValueError("base_url is required")

        if self.auth_type != AuthType.NONE and not self._has_auth_credentials():
            raise ValueError(f"Authentication credentials required for {self.auth_type.value}")

    def _has_auth_credentials(self) -> bool:
        """Check if required authentication credentials are provided."""
        if self.auth_type == AuthType.API_KEY:
            return bool(self.api_key)
        if self.auth_type == AuthType.BEARER_TOKEN:
            return bool(self.bearer_token)
        if self.auth_type == AuthType.BASIC_AUTH:
            return bool(self.username and self.password)
        if self.auth_type == AuthType.OAUTH2:
            return bool(self.oauth_config)
        return True


# Exception hierarchy for adapter errors
class AdapterError(Exception):
    """Base exception for all adapter errors."""

    def __init__(self, message: str, status_code: int | None = None, response_data: Any | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data
        self.timestamp = datetime.now(UTC)


class ConnectionError(AdapterError):
    """Raised when connection to external service fails."""
    pass


class TimeoutError(AdapterError):
    """Raised when request times out."""
    pass


class RateLimitError(AdapterError):
    """Raised when rate limit is exceeded."""
    pass


class AuthenticationError(AdapterError):
    """Raised when authentication fails."""
    pass


class APIError(AdapterError):
    """Raised when API returns an error response."""
    pass


class BaseAdapter(ABC):
    """
    Base class for all external service adapters.
    
    Provides common functionality including:
    - HTTP client management
    - Authentication handling
    - Circuit breaker protection
    - Retry logic with backoff
    - Rate limiting
    - Request/response logging
    - Error handling and transformation
    """

    def __init__(self, config: AdapterConfig):
        self.config = config
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self._session: aiohttp.ClientSession | None = None
        self._circuit_breaker = None

        # Rate limiting state
        self._request_times: list[float] = []
        self._rate_limit_lock = asyncio.Lock()

        # Initialize circuit breaker if enabled
        if self.config.use_circuit_breaker:
            self._setup_circuit_breaker()

        self.logger.info(f"Initialized {self.__class__.__name__} adapter")

    def _setup_circuit_breaker(self) -> None:
        """Setup circuit breaker for this adapter."""
        registry = get_circuit_breaker_registry()
        cb_name = self.config.circuit_breaker_name or f"{self.__class__.__name__.lower()}_adapter"

        if not registry.get_circuit_breaker(cb_name):
            from pipeline.infrastructure.circuit_breaker import (
                CircuitBreaker,
                CircuitBreakerConfig,
            )

            cb_config = CircuitBreakerConfig(
                failure_threshold=5,
                timeout_seconds=60,
                recovery_timeout_seconds=30
            )

            self._circuit_breaker = CircuitBreaker(cb_name, cb_config)
            registry.register(cb_name, self._circuit_breaker)
        else:
            self._circuit_breaker = registry.get_circuit_breaker(cb_name)

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def _ensure_session(self) -> None:
        """Ensure HTTP session is initialized."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)

            headers = {
                'User-Agent': f'AgenticStartupStudio/{self.__class__.__name__}',
                **self.config.custom_headers
            }

            # Add authentication headers
            auth_headers = self._get_auth_headers()
            headers.update(auth_headers)

            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers=headers,
                connector=aiohttp.TCPConnector(limit=10)
            )

    def _get_auth_headers(self) -> dict[str, str]:
        """Get authentication headers based on config."""
        headers = {}

        if self.config.auth_type == AuthType.API_KEY and self.config.api_key:
            headers['X-API-Key'] = self.config.api_key
        elif self.config.auth_type == AuthType.BEARER_TOKEN and self.config.bearer_token:
            headers['Authorization'] = f'Bearer {self.config.bearer_token}'
        elif self.config.auth_type == AuthType.BASIC_AUTH and self.config.username and self.config.password:
            import base64
            credentials = base64.b64encode(f"{self.config.username}:{self.config.password}".encode()).decode()
            headers['Authorization'] = f'Basic {credentials}'

        return headers

    async def _check_rate_limit(self) -> None:
        """Check and enforce rate limiting."""
        async with self._rate_limit_lock:
            now = time.time()

            # Remove requests outside the time window
            cutoff = now - self.config.rate_limit_window_seconds
            self._request_times = [t for t in self._request_times if t > cutoff]

            # Check if we're at the rate limit
            if len(self._request_times) >= self.config.rate_limit_requests:
                oldest_request = min(self._request_times)
                sleep_time = self.config.rate_limit_window_seconds - (now - oldest_request)

                if sleep_time > 0:
                    self.logger.warning(f"Rate limit exceeded, sleeping for {sleep_time:.2f} seconds")
                    await asyncio.sleep(sleep_time)

            # Record this request
            self._request_times.append(now)

    @performance_monitor("adapter_request")
    async def _make_request(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> aiohttp.ClientResponse:
        """
        Make HTTP request with retry logic and error handling.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: Full URL or path relative to base_url
            **kwargs: Additional arguments for aiohttp request
            
        Returns:
            HTTP response
            
        Raises:
            Various adapter exceptions based on error type
        """
        await self._ensure_session()
        await self._check_rate_limit()

        # Construct full URL if relative path provided
        if not url.startswith(('http://', 'https://')):
            url = f"{self.config.base_url.rstrip('/')}/{url.lstrip('/')}"

        # Apply circuit breaker if configured
        if self._circuit_breaker:
            return await self._circuit_breaker.call(self._execute_request, method, url, **kwargs)
        return await self._execute_request(method, url, **kwargs)

    async def _execute_request(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> aiohttp.ClientResponse:
        """Execute HTTP request with retry logic."""
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                self.logger.debug(f"Making {method} request to {url} (attempt {attempt + 1})")

                async with self._session.request(method, url, **kwargs) as response:
                    # Check for HTTP errors
                    if response.status >= 400:
                        error_text = await response.text()

                        if response.status == 401:
                            raise AuthenticationError(
                                f"Authentication failed: {error_text}",
                                status_code=response.status,
                                response_data=error_text
                            )
                        if response.status == 429:
                            raise RateLimitError(
                                f"Rate limit exceeded: {error_text}",
                                status_code=response.status,
                                response_data=error_text
                            )
                        raise APIError(
                            f"API error {response.status}: {error_text}",
                            status_code=response.status,
                            response_data=error_text
                        )

                    # Success - return response
                    self.logger.debug(f"Request successful: {response.status}")
                    return response

            except aiohttp.ClientConnectorError as e:
                last_exception = ConnectionError(f"Connection failed: {str(e)}")
            except aiohttp.ServerTimeoutError as e:
                last_exception = TimeoutError(f"Request timed out: {str(e)}")
            except (AuthenticationError, RateLimitError, APIError):
                # Don't retry these errors
                raise
            except Exception as e:
                last_exception = AdapterError(f"Unexpected error: {str(e)}")

            # Calculate retry delay
            if attempt < self.config.max_retries:
                delay = self._calculate_retry_delay(attempt)
                self.logger.warning(f"Request failed, retrying in {delay:.2f} seconds: {last_exception}")
                await asyncio.sleep(delay)

        # All retries exhausted
        self.logger.error(f"All retries exhausted for {method} {url}")
        raise last_exception or AdapterError("Request failed after all retries")

    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate delay before next retry attempt."""
        base_delay = self.config.retry_delay_seconds

        if self.config.retry_strategy == RetryStrategy.NONE:
            return 0.0
        if self.config.retry_strategy == RetryStrategy.FIXED:
            return base_delay
        if self.config.retry_strategy == RetryStrategy.LINEAR:
            return base_delay * (attempt + 1)
        if self.config.retry_strategy == RetryStrategy.EXPONENTIAL:
            return base_delay * (2 ** attempt)
        return base_delay

    async def close(self) -> None:
        """Close adapter and cleanup resources."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

        self.logger.info(f"Closed {self.__class__.__name__} adapter")

    # Abstract methods for subclasses to implement
    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """
        Check adapter health and connectivity.
        
        Returns:
            Dictionary with health status information
        """
        pass

    @abstractmethod
    async def get_service_info(self) -> dict[str, Any]:
        """
        Get information about the external service.
        
        Returns:
            Dictionary with service information
        """
        pass

    # Utility methods for common operations
    async def get_json(self, url: str, **kwargs) -> dict[str, Any]:
        """Make GET request and return JSON response."""
        response = await self._make_request('GET', url, **kwargs)
        return await response.json()

    async def post_json(self, url: str, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Make POST request with JSON data and return JSON response."""
        kwargs['json'] = data
        response = await self._make_request('POST', url, **kwargs)
        return await response.json()

    async def put_json(self, url: str, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Make PUT request with JSON data and return JSON response."""
        kwargs['json'] = data
        response = await self._make_request('PUT', url, **kwargs)
        return await response.json()

    async def delete(self, url: str, **kwargs) -> bool:
        """Make DELETE request and return success status."""
        response = await self._make_request('DELETE', url, **kwargs)
        return response.status < 400
