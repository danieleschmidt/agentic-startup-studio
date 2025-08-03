"""
External Service Integrations for Agentic Startup Studio.

This module provides integrations with external services including:
- GitHub API for code repository management
- Notification services (email, Slack, Discord)
- Authentication providers (OAuth, JWT, API keys)
- Third-party APIs (market data, competitive intelligence)
"""

from .github_integration import GitHubIntegration
from .notification_service import NotificationService
from .auth_provider import AuthProvider
from .market_data_service import MarketDataService

__all__ = [
    "GitHubIntegration",
    "NotificationService", 
    "AuthProvider",
    "MarketDataService"
]