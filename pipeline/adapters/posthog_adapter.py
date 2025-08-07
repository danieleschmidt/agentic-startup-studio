"""
PostHog Analytics Adapter for Event Tracking and Feature Flags.

Provides functionality to interact with PostHog API including:
- Event tracking and analytics
- Feature flag management
- User identification and properties
- Funnel analysis
- A/B testing and experimentation
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pipeline.adapters.base_adapter import (
    AdapterConfig,
    AdapterError,
    AuthenticationError,
    AuthType,
    BaseAdapter,
    RetryStrategy,
)
from pipeline.config.settings import get_settings
from pipeline.infrastructure.observability import get_logger, performance_monitor

logger = get_logger(__name__)


class EventType(Enum):
    """Common event types for analytics tracking."""
    PAGE_VIEW = "page_view"
    CLICK = "click"
    FORM_SUBMIT = "form_submit"
    SIGNUP = "signup"
    LOGIN = "login"
    LOGOUT = "logout"
    PURCHASE = "purchase"
    FEATURE_USED = "feature_used"
    ERROR = "error"
    CUSTOM = "custom"


class FeatureFlagType(Enum):
    """Types of feature flags."""
    BOOLEAN = "boolean"
    MULTIVARIATE = "multivariate"
    ROLLOUT = "rollout"


class PropertyType(Enum):
    """Types of user/event properties."""
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    LIST = "list"
    OBJECT = "object"


@dataclass
class PostHogConfig(AdapterConfig):
    """Configuration for PostHog adapter."""

    # API configuration
    project_api_key: str | None = None
    personal_api_key: str | None = None
    host: str = "https://app.posthog.com"

    # Batch processing
    batch_size: int = 100
    flush_interval_seconds: int = 10

    # Feature flags
    enable_feature_flags: bool = True
    feature_flag_timeout_seconds: int = 5

    # Analytics
    enable_analytics: bool = True
    track_performance: bool = True

    def __post_init__(self):
        """Validate PostHog specific configuration."""
        super().__post_init__()

        if not self.project_api_key:
            raise ValueError("project_api_key is required for PostHog API")

        if self.enable_feature_flags and not self.personal_api_key:
            raise ValueError("personal_api_key is required for feature flag operations")


@dataclass
class EventData:
    """Data structure for PostHog events."""
    event: str
    distinct_id: str
    properties: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime | None = None
    uuid: str | None = None

    def __post_init__(self):
        """Set default values after initialization."""
        if self.timestamp is None:
            self.timestamp = datetime.now(UTC)
        if self.uuid is None:
            self.uuid = str(uuid.uuid4())


@dataclass
class UserData:
    """Data structure for PostHog user identification."""
    distinct_id: str
    properties: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime | None = None

    def __post_init__(self):
        """Set default timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now(UTC)


@dataclass
class FeatureFlagData:
    """Data structure for feature flags."""
    key: str
    name: str
    flag_type: FeatureFlagType
    active: bool = True
    rollout_percentage: float | None = None
    filters: dict[str, Any] = field(default_factory=dict)
    variants: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class FunnelData:
    """Data structure for funnel analysis."""
    name: str
    steps: list[dict[str, Any]]
    date_range: dict[str, str]
    breakdown_by: str | None = None
    filters: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalyticsQuery:
    """Data structure for analytics queries."""
    event_name: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    properties: list[dict[str, Any]] = field(default_factory=list)
    breakdown: str | None = None
    interval: str = "day"


class PostHogAdapter(BaseAdapter):
    """
    PostHog analytics adapter for event tracking and feature flags.
    
    Provides functionality to:
    - Track events and user interactions
    - Manage user identification and properties
    - Handle feature flags and A/B testing
    - Analyze funnels and user journeys
    - Perform custom analytics queries
    """

    def __init__(self, config: PostHogConfig):
        if not isinstance(config, PostHogConfig):
            raise ValueError("PostHogConfig required for PostHogAdapter")

        # Set base URL for PostHog API
        config.base_url = f"{config.host}/api"
        config.auth_type = AuthType.API_KEY
        config.api_key = config.project_api_key
        config.circuit_breaker_name = "posthog_adapter"

        super().__init__(config)
        self.config: PostHogConfig = config

        # Event batching
        self._event_batch: list[EventData] = []
        self._batch_lock = asyncio.Lock()
        self._flush_task: asyncio.Task | None = None

        # Start batch processing if analytics enabled
        if self.config.enable_analytics:
            self._start_batch_processing()

        self.logger.info(f"Initialized PostHog adapter for project {config.project_api_key[:8]}...")

    def _start_batch_processing(self) -> None:
        """Start background task for batch processing events."""
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(self._batch_flush_loop())

    async def _batch_flush_loop(self) -> None:
        """Background loop to flush event batches periodically."""
        while True:
            try:
                await asyncio.sleep(self.config.flush_interval_seconds)
                await self._flush_batch()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in batch flush loop: {e}")

    async def _flush_batch(self) -> None:
        """Flush current batch of events to PostHog."""
        async with self._batch_lock:
            if not self._event_batch:
                return

            batch_to_send = self._event_batch.copy()
            self._event_batch.clear()

        if batch_to_send:
            await self._send_batch(batch_to_send)

    async def _send_batch(self, events: list[EventData]) -> None:
        """Send batch of events to PostHog."""
        try:
            batch_data = {
                "api_key": self.config.project_api_key,
                "batch": [
                    {
                        "event": event.event,
                        "distinct_id": event.distinct_id,
                        "properties": {
                            **event.properties,
                            "$timestamp": event.timestamp.isoformat() if event.timestamp else None
                        },
                        "timestamp": event.timestamp.isoformat() if event.timestamp else None,
                        "uuid": event.uuid
                    }
                    for event in events
                ]
            }

            await self._ensure_session()
            response = await self.post_json("batch/", batch_data)

            self.logger.debug(f"Successfully sent batch of {len(events)} events")

        except Exception as e:
            self.logger.error(f"Failed to send event batch: {e}")
            # Re-add events to batch for retry
            async with self._batch_lock:
                self._event_batch.extend(events)

    @performance_monitor("posthog_health_check")
    async def health_check(self) -> dict[str, Any]:
        """Check PostHog API connectivity and project status."""
        try:
            # Test API connectivity with a simple request
            if self.config.personal_api_key:
                # Use personal API key for project info
                headers = {
                    'Authorization': f'Bearer {self.config.personal_api_key}',
                    'Content-Type': 'application/json'
                }
                self._session.headers.update(headers)

                response = await self.get_json("projects/")
                project_count = len(response.get('results', []))

                return {
                    'status': 'healthy',
                    'service': 'PostHog Analytics',
                    'host': self.config.host,
                    'project_count': project_count,
                    'feature_flags_enabled': self.config.enable_feature_flags,
                    'analytics_enabled': self.config.enable_analytics,
                    'timestamp': datetime.now(UTC).isoformat()
                }
            # Basic connectivity test
            test_event = EventData(
                event="health_check",
                distinct_id="health_check_user",
                properties={"source": "health_check"}
            )

            await self.track_event(test_event)

            return {
                'status': 'healthy',
                'service': 'PostHog Analytics',
                'host': self.config.host,
                'feature_flags_enabled': False,
                'analytics_enabled': self.config.enable_analytics,
                'timestamp': datetime.now(UTC).isoformat()
            }

        except Exception as e:
            self.logger.error(f"PostHog health check failed: {e}")
            return {
                'status': 'unhealthy',
                'service': 'PostHog Analytics',
                'error': str(e),
                'timestamp': datetime.now(UTC).isoformat()
            }

    async def get_service_info(self) -> dict[str, Any]:
        """Get PostHog service information."""
        return {
            'service_name': 'PostHog Analytics',
            'host': self.config.host,
            'project_api_key': self.config.project_api_key[:8] + '...' if self.config.project_api_key else None,
            'personal_api_key_configured': bool(self.config.personal_api_key),
            'batch_size': self.config.batch_size,
            'flush_interval_seconds': self.config.flush_interval_seconds,
            'supported_features': [
                'event_tracking',
                'user_identification',
                'feature_flags' if self.config.enable_feature_flags else None,
                'funnel_analysis',
                'custom_analytics'
            ]
        }

    @performance_monitor("posthog_track_event")
    async def track_event(self, event_data: EventData, immediate: bool = False) -> bool:
        """Track an event in PostHog."""
        try:
            if not self.config.enable_analytics:
                self.logger.warning("Analytics disabled, skipping event tracking")
                return False

            # Add performance tracking properties if enabled
            if self.config.track_performance:
                event_data.properties.update({
                    "$performance_timestamp": datetime.now(UTC).isoformat(),
                    "$lib": "agentic-startup-studio",
                    "$lib_version": "1.0.0"
                })

            if immediate:
                # Send immediately
                await self._send_batch([event_data])
                return True
            # Add to batch
            async with self._batch_lock:
                self._event_batch.append(event_data)

                # Auto-flush if batch is full
                if len(self._event_batch) >= self.config.batch_size:
                    await self._flush_batch()

            return True

        except Exception as e:
            self.logger.error(f"Failed to track event: {e}")
            return False

    @performance_monitor("posthog_identify_user")
    async def identify_user(self, user_data: UserData) -> bool:
        """Identify a user in PostHog."""
        try:
            if not self.config.enable_analytics:
                self.logger.warning("Analytics disabled, skipping user identification")
                return False

            identify_data = {
                "api_key": self.config.project_api_key,
                "event": "$identify",
                "distinct_id": user_data.distinct_id,
                "properties": {
                    "$set": user_data.properties,
                    "$timestamp": user_data.timestamp.isoformat() if user_data.timestamp else None
                }
            }

            await self._ensure_session()
            response = await self.post_json("capture/", identify_data)

            self.logger.debug(f"Successfully identified user: {user_data.distinct_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to identify user: {e}")
            return False

    @performance_monitor("posthog_get_feature_flag")
    async def get_feature_flag(self, flag_key: str, distinct_id: str, user_properties: dict[str, Any] | None = None) -> dict[str, Any]:
        """Get feature flag value for a user."""
        try:
            if not self.config.enable_feature_flags:
                return {"enabled": False, "reason": "feature_flags_disabled"}

            if not self.config.personal_api_key:
                return {"enabled": False, "reason": "personal_api_key_required"}

            # Prepare request data
            request_data = {
                "distinct_id": distinct_id,
                "groups": {},
                "person_properties": user_properties or {},
                "group_properties": {}
            }

            # Set authorization header
            headers = {
                'Authorization': f'Bearer {self.config.personal_api_key}',
                'Content-Type': 'application/json'
            }
            self._session.headers.update(headers)

            # Make request to feature flags endpoint
            response = await self.post_json("feature_flag/local_evaluation", {
                "token": self.config.project_api_key,
                "distinct_id": distinct_id,
                "groups": {},
                "person_properties": user_properties or {},
                "group_properties": {},
                "only_evaluate_locally": True
            })

            flag_value = response.get("featureFlags", {}).get(flag_key)

            result = {
                "flag_key": flag_key,
                "distinct_id": distinct_id,
                "enabled": flag_value is not None and flag_value is not False,
                "value": flag_value,
                "timestamp": datetime.now(UTC).isoformat()
            }

            self.logger.debug(f"Retrieved feature flag {flag_key} for user {distinct_id}: {flag_value}")
            return result

        except Exception as e:
            self.logger.error(f"Failed to get feature flag: {e}")
            return {
                "flag_key": flag_key,
                "distinct_id": distinct_id,
                "enabled": False,
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat()
            }

    @performance_monitor("posthog_create_feature_flag")
    async def create_feature_flag(self, flag_data: FeatureFlagData) -> dict[str, Any]:
        """Create a new feature flag."""
        try:
            if not self.config.personal_api_key:
                raise AuthenticationError("Personal API key required for feature flag management")

            # Set authorization header
            headers = {
                'Authorization': f'Bearer {self.config.personal_api_key}',
                'Content-Type': 'application/json'
            }
            self._session.headers.update(headers)

            # Prepare flag data
            create_data = {
                "key": flag_data.key,
                "name": flag_data.name,
                "active": flag_data.active,
                "filters": {
                    "groups": [
                        {
                            "properties": [],
                            "rollout_percentage": flag_data.rollout_percentage or 100 if flag_data.active else 0
                        }
                    ]
                }
            }

            # Add variants for multivariate flags
            if flag_data.flag_type == FeatureFlagType.MULTIVARIATE and flag_data.variants:
                create_data["filters"]["multivariate"] = {
                    "variants": flag_data.variants
                }

            response = await self.post_json("feature_flag/", create_data)

            self.logger.info(f"Successfully created feature flag: {flag_data.key}")
            return response

        except Exception as e:
            self.logger.error(f"Failed to create feature flag: {e}")
            raise AdapterError(f"Failed to create feature flag: {str(e)}")

    @performance_monitor("posthog_run_query")
    async def run_analytics_query(self, query: AnalyticsQuery) -> dict[str, Any]:
        """Run custom analytics query."""
        try:
            if not self.config.personal_api_key:
                raise AuthenticationError("Personal API key required for analytics queries")

            # Set authorization header
            headers = {
                'Authorization': f'Bearer {self.config.personal_api_key}',
                'Content-Type': 'application/json'
            }
            self._session.headers.update(headers)

            # Build query data
            query_data = {
                "kind": "EventsQuery",
                "select": ["*"],
                "orderBy": ["-timestamp"]
            }

            # Add event filter
            if query.event_name:
                query_data["event"] = query.event_name

            # Add date range
            if query.date_from:
                query_data["after"] = query.date_from
            if query.date_to:
                query_data["before"] = query.date_to

            # Add property filters
            if query.properties:
                query_data["properties"] = query.properties

            # Add breakdown
            if query.breakdown:
                query_data["breakdown"] = query.breakdown

            response = await self.post_json("query/", {"query": query_data})

            self.logger.debug("Successfully ran analytics query")
            return response

        except Exception as e:
            self.logger.error(f"Failed to run analytics query: {e}")
            raise AdapterError(f"Failed to run analytics query: {str(e)}")

    @performance_monitor("posthog_track_conversion")
    async def track_conversion(self, distinct_id: str, event_name: str, conversion_value: float | None = None, properties: dict[str, Any] | None = None) -> bool:
        """Track a conversion event."""
        conversion_properties = {
            "conversion": True,
            **(properties or {})
        }

        if conversion_value is not None:
            conversion_properties["conversion_value"] = conversion_value

        event_data = EventData(
            event=event_name,
            distinct_id=distinct_id,
            properties=conversion_properties
        )

        return await self.track_event(event_data, immediate=True)

    async def close(self) -> None:
        """Close adapter and cleanup resources."""
        # Cancel batch processing task
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Flush remaining events
        await self._flush_batch()

        # Close parent adapter
        await super().close()

        self.logger.info("Closed PostHog adapter")


def create_posthog_adapter() -> PostHogAdapter:
    """Factory function to create PostHog adapter with environment configuration."""
    settings = get_settings()

    config = PostHogConfig(
        base_url="",  # Will be set by adapter
        project_api_key=settings.POSTHOG_PROJECT_API_KEY,
        personal_api_key=settings.POSTHOG_PERSONAL_API_KEY,
        host=settings.POSTHOG_HOST,
        timeout_seconds=settings.POSTHOG_TIMEOUT_SECONDS,
        max_retries=settings.POSTHOG_MAX_RETRIES,
        retry_strategy=RetryStrategy.EXPONENTIAL,
        batch_size=settings.POSTHOG_BATCH_SIZE,
        flush_interval_seconds=settings.POSTHOG_FLUSH_INTERVAL_SECONDS,
        rate_limit_requests=settings.POSTHOG_RATE_LIMIT_REQUESTS,
        rate_limit_window_seconds=settings.POSTHOG_RATE_LIMIT_WINDOW_SECONDS,
        use_circuit_breaker=True,
        enable_metrics=True,
        enable_logging=True,
        enable_feature_flags=settings.POSTHOG_ENABLE_FEATURE_FLAGS,
        enable_analytics=settings.POSTHOG_ENABLE_ANALYTICS,
        track_performance=settings.POSTHOG_TRACK_PERFORMANCE
    )

    return PostHogAdapter(config)
