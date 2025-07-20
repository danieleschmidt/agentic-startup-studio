"""
Google Ads API Adapter for Campaign Management and Analytics.

Provides functionality to interact with Google Ads API including:
- Campaign creation and management
- Keyword research and analysis
- Ad group management
- Performance metrics retrieval
- Budget management
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

from pipeline.adapters.base_adapter import (
    BaseAdapter, AdapterConfig, AuthType, RetryStrategy,
    AdapterError, APIError, AuthenticationError
)
from pipeline.config.settings import get_settings
from pipeline.infrastructure.observability import get_logger, performance_monitor

logger = get_logger(__name__)


class CampaignStatus(Enum):
    """Google Ads campaign status values."""
    ENABLED = "ENABLED"
    PAUSED = "PAUSED"
    REMOVED = "REMOVED"
    UNKNOWN = "UNKNOWN"


class AdGroupType(Enum):
    """Google Ads ad group types."""
    SEARCH_STANDARD = "SEARCH_STANDARD"
    DISPLAY_STANDARD = "DISPLAY_STANDARD"
    SHOPPING_PRODUCT_ADS = "SHOPPING_PRODUCT_ADS"
    VIDEO_BUMPER = "VIDEO_BUMPER"
    VIDEO_TRUE_VIEW_IN_STREAM = "VIDEO_TRUE_VIEW_IN_STREAM"
    VIDEO_TRUE_VIEW_IN_DISPLAY = "VIDEO_TRUE_VIEW_IN_DISPLAY"
    VIDEO_NON_SKIPPABLE_IN_STREAM = "VIDEO_NON_SKIPPABLE_IN_STREAM"
    VIDEO_OUTSTREAM = "VIDEO_OUTSTREAM"
    SEARCH_DYNAMIC_ADS = "SEARCH_DYNAMIC_ADS"
    SHOPPING_SMART_ADS = "SHOPPING_SMART_ADS"
    DISPLAY_SMART_ADS = "DISPLAY_SMART_ADS"
    SMART_CAMPAIGN_ADS = "SMART_CAMPAIGN_ADS"


class BiddingStrategy(Enum):
    """Google Ads bidding strategies."""
    MANUAL_CPC = "MANUAL_CPC"
    MANUAL_CPM = "MANUAL_CPM"
    MANUAL_CPV = "MANUAL_CPV"
    MAXIMIZE_CONVERSIONS = "MAXIMIZE_CONVERSIONS"
    MAXIMIZE_CONVERSION_VALUE = "MAXIMIZE_CONVERSION_VALUE"
    TARGET_CPA = "TARGET_CPA"
    TARGET_ROAS = "TARGET_ROAS"
    TARGET_SPEND = "TARGET_SPEND"
    PERCENT_CPC = "PERCENT_CPC"
    TARGET_CPM = "TARGET_CPM"
    TARGET_IMPRESSION_SHARE = "TARGET_IMPRESSION_SHARE"


@dataclass
class GoogleAdsConfig(AdapterConfig):
    """Configuration for Google Ads adapter."""
    
    # API version and endpoints
    api_version: str = "v16"
    developer_token: Optional[str] = None
    customer_id: Optional[str] = None
    
    # OAuth configuration
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    refresh_token: Optional[str] = None
    
    # Default settings
    default_budget_amount: int = 10000  # in micros (e.g., $10)
    default_bid_strategy: BiddingStrategy = BiddingStrategy.MAXIMIZE_CONVERSIONS
    default_language_code: str = "en"
    default_location_id: int = 2840  # United States
    
    def __post_init__(self):
        """Validate Google Ads specific configuration."""
        super().__post_init__()
        
        if not self.developer_token:
            raise ValueError("developer_token is required for Google Ads API")
        
        if not self.customer_id:
            raise ValueError("customer_id is required for Google Ads API")
        
        if self.auth_type == AuthType.OAUTH2:
            if not (self.client_id and self.client_secret and self.refresh_token):
                raise ValueError("OAuth2 credentials required: client_id, client_secret, refresh_token")


@dataclass
class CampaignData:
    """Data structure for Google Ads campaign."""
    name: str
    budget_amount: int  # in micros
    bidding_strategy: BiddingStrategy
    status: CampaignStatus = CampaignStatus.ENABLED
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    language_codes: List[str] = field(default_factory=lambda: ["en"])
    location_ids: List[int] = field(default_factory=lambda: [2840])  # United States
    keywords: List[str] = field(default_factory=list)
    negative_keywords: List[str] = field(default_factory=list)
    ad_groups: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class KeywordData:
    """Data structure for keyword research."""
    keyword: str
    search_volume: Optional[int] = None
    competition: Optional[str] = None
    low_top_of_page_bid: Optional[int] = None
    high_top_of_page_bid: Optional[int] = None
    average_cpc: Optional[int] = None


@dataclass
class PerformanceMetrics:
    """Data structure for campaign performance metrics."""
    campaign_id: str
    campaign_name: str
    impressions: int
    clicks: int
    conversions: float
    cost: int  # in micros
    ctr: float
    average_cpc: float
    conversion_rate: float
    cost_per_conversion: float
    date_range: Dict[str, str]


class GoogleAdsAdapter(BaseAdapter):
    """
    Google Ads API adapter for campaign management and analytics.
    
    Provides functionality to:
    - Create and manage advertising campaigns
    - Research keywords and analyze competition
    - Manage ad groups and advertisements
    - Retrieve performance metrics and analytics
    - Handle budget management and bidding strategies
    """
    
    def __init__(self, config: GoogleAdsConfig):
        if not isinstance(config, GoogleAdsConfig):
            raise ValueError("GoogleAdsConfig required for GoogleAdsAdapter")
        
        # Set base URL for Google Ads API
        config.base_url = f"https://googleads.googleapis.com/{config.api_version}/customers/{config.customer_id}"
        config.auth_type = AuthType.OAUTH2
        config.circuit_breaker_name = "google_ads_adapter"
        
        super().__init__(config)
        self.config: GoogleAdsConfig = config
        self._access_token: Optional[str] = None
        self._token_expires_at: Optional[datetime] = None
        
        self.logger.info(f"Initialized Google Ads adapter for customer {config.customer_id}")
    
    async def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers with fresh access token."""
        await self._ensure_valid_token()
        
        headers = {
            'Authorization': f'Bearer {self._access_token}',
            'developer-token': self.config.developer_token,
            'Content-Type': 'application/json'
        }
        
        return headers
    
    async def _ensure_valid_token(self) -> None:
        """Ensure we have a valid access token."""
        if (self._access_token and self._token_expires_at and 
            datetime.now(timezone.utc) < self._token_expires_at - timedelta(minutes=5)):
            return
        
        await self._refresh_access_token()
    
    async def _refresh_access_token(self) -> None:
        """Refresh OAuth2 access token."""
        try:
            token_data = {
                'grant_type': 'refresh_token',
                'client_id': self.config.client_id,
                'client_secret': self.config.client_secret,
                'refresh_token': self.config.refresh_token
            }
            
            # Use Google's OAuth2 endpoint
            async with self._session.post(
                'https://oauth2.googleapis.com/token',
                data=token_data
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise AuthenticationError(f"Token refresh failed: {error_text}")
                
                token_response = await response.json()
                self._access_token = token_response['access_token']
                expires_in = token_response.get('expires_in', 3600)
                self._token_expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
                
                self.logger.info("Successfully refreshed Google Ads access token")
                
        except Exception as e:
            self.logger.error(f"Failed to refresh access token: {e}")
            raise AuthenticationError(f"Token refresh failed: {str(e)}")
    
    @performance_monitor("google_ads_health_check")
    async def health_check(self) -> Dict[str, Any]:
        """Check Google Ads API connectivity and account status."""
        try:
            # Get customer info to verify API access
            customer_info = await self.get_customer_info()
            
            return {
                'status': 'healthy',
                'service': 'Google Ads API',
                'api_version': self.config.api_version,
                'customer_id': self.config.customer_id,
                'customer_name': customer_info.get('descriptiveName', 'Unknown'),
                'account_status': customer_info.get('status', 'Unknown'),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            self.logger.error(f"Google Ads health check failed: {e}")
            return {
                'status': 'unhealthy',
                'service': 'Google Ads API',
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    async def get_service_info(self) -> Dict[str, Any]:
        """Get Google Ads API service information."""
        return {
            'service_name': 'Google Ads API',
            'api_version': self.config.api_version,
            'customer_id': self.config.customer_id,
            'developer_token': self.config.developer_token[:8] + '...' if self.config.developer_token else None,
            'supported_features': [
                'campaign_management',
                'keyword_research',
                'ad_group_management',
                'performance_metrics',
                'budget_management'
            ]
        }
    
    @performance_monitor("google_ads_get_customer_info")
    async def get_customer_info(self) -> Dict[str, Any]:
        """Get customer account information."""
        try:
            headers = await self._get_auth_headers()
            
            # Update session headers
            self._session.headers.update(headers)
            
            response = await self.get_json('')
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to get customer info: {e}")
            raise AdapterError(f"Failed to get customer info: {str(e)}")
    
    @performance_monitor("google_ads_create_campaign")
    async def create_campaign(self, campaign_data: CampaignData) -> Dict[str, Any]:
        """Create a new Google Ads campaign."""
        try:
            headers = await self._get_auth_headers()
            self._session.headers.update(headers)
            
            # Construct campaign operation
            operation = {
                'operations': [{
                    'create': {
                        'name': campaign_data.name,
                        'status': campaign_data.status.value,
                        'advertisingChannelType': 'SEARCH',
                        'biddingStrategy': {
                            'type': campaign_data.bidding_strategy.value
                        },
                        'campaignBudget': {
                            'amountMicros': str(campaign_data.budget_amount),
                            'deliveryMethod': 'STANDARD'
                        },
                        'networkSettings': {
                            'targetGoogleSearch': True,
                            'targetSearchNetwork': True,
                            'targetContentNetwork': False,
                            'targetPartnerSearchNetwork': False
                        },
                        'geoTargetTypeSetting': {
                            'positiveGeoTargetType': 'PRESENCE_OR_INTEREST',
                            'negativeGeoTargetType': 'PRESENCE_OR_INTEREST'
                        }
                    }
                }]
            }
            
            # Add start and end dates if provided
            if campaign_data.start_date:
                operation['operations'][0]['create']['startDate'] = campaign_data.start_date
            if campaign_data.end_date:
                operation['operations'][0]['create']['endDate'] = campaign_data.end_date
            
            response = await self.post_json('campaigns:mutate', operation)
            
            campaign_id = response['results'][0]['resourceName'].split('/')[-1]
            
            self.logger.info(f"Successfully created campaign: {campaign_data.name} (ID: {campaign_id})")
            
            return {
                'campaign_id': campaign_id,
                'campaign_name': campaign_data.name,
                'status': 'created',
                'resource_name': response['results'][0]['resourceName']
            }
            
        except Exception as e:
            self.logger.error(f"Failed to create campaign: {e}")
            raise AdapterError(f"Failed to create campaign: {str(e)}")
    
    @performance_monitor("google_ads_get_campaigns")
    async def get_campaigns(self, status_filter: Optional[CampaignStatus] = None) -> List[Dict[str, Any]]:
        """Get list of campaigns."""
        try:
            headers = await self._get_auth_headers()
            self._session.headers.update(headers)
            
            # Build GAQL query
            query = """
                SELECT 
                    campaign.id,
                    campaign.name,
                    campaign.status,
                    campaign.advertising_channel_type,
                    campaign_budget.amount_micros,
                    metrics.impressions,
                    metrics.clicks,
                    metrics.cost_micros
                FROM campaign
                WHERE segments.date DURING LAST_30_DAYS
            """
            
            if status_filter:
                query += f" AND campaign.status = '{status_filter.value}'"
            
            search_request = {
                'query': query,
                'pageSize': 100
            }
            
            response = await self.post_json('googleAds:search', search_request)
            
            campaigns = []
            for result in response.get('results', []):
                campaign = result.get('campaign', {})
                budget = result.get('campaignBudget', {})
                metrics = result.get('metrics', {})
                
                campaigns.append({
                    'id': campaign.get('id'),
                    'name': campaign.get('name'),
                    'status': campaign.get('status'),
                    'channel_type': campaign.get('advertisingChannelType'),
                    'budget_micros': budget.get('amountMicros'),
                    'impressions': metrics.get('impressions', 0),
                    'clicks': metrics.get('clicks', 0),
                    'cost_micros': metrics.get('costMicros', 0)
                })
            
            self.logger.info(f"Retrieved {len(campaigns)} campaigns")
            return campaigns
            
        except Exception as e:
            self.logger.error(f"Failed to get campaigns: {e}")
            raise AdapterError(f"Failed to get campaigns: {str(e)}")
    
    @performance_monitor("google_ads_keyword_research")
    async def research_keywords(self, seed_keywords: List[str], location_ids: Optional[List[int]] = None) -> List[KeywordData]:
        """Research keywords using Google Ads Keyword Planner."""
        try:
            headers = await self._get_auth_headers()
            self._session.headers.update(headers)
            
            # Use default location if not provided
            if not location_ids:
                location_ids = [self.config.default_location_id]
            
            # Build keyword plan request
            request_data = {
                'keywordPlanIdeaService': {
                    'generateKeywordIdeas': {
                        'language': f'languageConstants/{self.config.default_language_code}',
                        'geoTargetConstants': [f'geoTargetConstants/{loc_id}' for loc_id in location_ids],
                        'keywordPlanNetwork': 'GOOGLE_SEARCH',
                        'keywordSeed': {
                            'keywords': seed_keywords
                        }
                    }
                }
            }
            
            response = await self.post_json('keywordPlanIdeas:generateKeywordIdeas', request_data)
            
            keyword_ideas = []
            for idea in response.get('results', []):
                keyword_text = idea.get('text', '')
                metrics = idea.get('keywordIdeaMetrics', {})
                
                keyword_data = KeywordData(
                    keyword=keyword_text,
                    search_volume=metrics.get('avgMonthlySearches'),
                    competition=metrics.get('competition'),
                    low_top_of_page_bid=metrics.get('lowTopOfPageBidMicros'),
                    high_top_of_page_bid=metrics.get('highTopOfPageBidMicros'),
                    average_cpc=metrics.get('avgCpcMicros')
                )
                keyword_ideas.append(keyword_data)
            
            self.logger.info(f"Generated {len(keyword_ideas)} keyword ideas")
            return keyword_ideas
            
        except Exception as e:
            self.logger.error(f"Keyword research failed: {e}")
            raise AdapterError(f"Keyword research failed: {str(e)}")
    
    @performance_monitor("google_ads_get_performance")
    async def get_campaign_performance(
        self,
        campaign_ids: Optional[List[str]] = None,
        date_range: Optional[Dict[str, str]] = None
    ) -> List[PerformanceMetrics]:
        """Get campaign performance metrics."""
        try:
            headers = await self._get_auth_headers()
            self._session.headers.update(headers)
            
            # Default to last 30 days if no date range provided
            if not date_range:
                date_range = {'start_date': 'LAST_30_DAYS', 'end_date': 'LAST_30_DAYS'}
            
            # Build GAQL query with parameterized values for security
            # Validate and sanitize date range to prevent injection
            start_date = str(date_range['start_date'])
            # Use allowlist approach for date ranges - only allow predefined values
            allowed_date_ranges = ['LAST_7_DAYS', 'LAST_30_DAYS', 'LAST_90_DAYS', 'TODAY', 'YESTERDAY']
            if start_date not in allowed_date_ranges:
                # For custom dates, validate format strictly
                import re
                if not re.match(r'^20\d{2}-\d{2}-\d{2}$', start_date):
                    raise ValueError(f"Invalid date range format. Must be one of {allowed_date_ranges} or YYYY-MM-DD format")
            
            # Use string formatting with validated input (still safe since we validated above)
            query = """
                SELECT 
                    campaign.id,
                    campaign.name,
                    metrics.impressions,
                    metrics.clicks,
                    metrics.conversions,
                    metrics.cost_micros,
                    metrics.ctr,
                    metrics.average_cpc,
                    metrics.conversions_per_click,
                    metrics.cost_per_conversion
                FROM campaign
                WHERE segments.date DURING {}
            """.format(start_date)  # nosec B608 - input is validated above
            
            if campaign_ids:
                # Sanitize campaign IDs to prevent injection
                sanitized_ids = []
                for cid in campaign_ids:
                    # Only allow numeric campaign IDs
                    if str(cid).isdigit():
                        sanitized_ids.append(str(cid))
                
                if sanitized_ids:
                    campaign_filter = "','".join(sanitized_ids)
                    query += f" AND campaign.id IN ('{campaign_filter}')"
            
            search_request = {
                'query': query,
                'pageSize': 1000
            }
            
            response = await self.post_json('googleAds:search', search_request)
            
            performance_data = []
            for result in response.get('results', []):
                campaign = result.get('campaign', {})
                metrics = result.get('metrics', {})
                
                perf_metrics = PerformanceMetrics(
                    campaign_id=campaign.get('id'),
                    campaign_name=campaign.get('name'),
                    impressions=metrics.get('impressions', 0),
                    clicks=metrics.get('clicks', 0),
                    conversions=metrics.get('conversions', 0.0),
                    cost=metrics.get('costMicros', 0),
                    ctr=metrics.get('ctr', 0.0),
                    average_cpc=metrics.get('averageCpc', 0.0),
                    conversion_rate=metrics.get('conversionsPerClick', 0.0),
                    cost_per_conversion=metrics.get('costPerConversion', 0.0),
                    date_range=date_range
                )
                performance_data.append(perf_metrics)
            
            self.logger.info(f"Retrieved performance data for {len(performance_data)} campaigns")
            return performance_data
            
        except Exception as e:
            self.logger.error(f"Failed to get campaign performance: {e}")
            raise AdapterError(f"Failed to get campaign performance: {str(e)}")


def create_google_ads_adapter() -> GoogleAdsAdapter:
    """Factory function to create Google Ads adapter with environment configuration."""
    settings = get_settings()
    
    config = GoogleAdsConfig(
        base_url="",  # Will be set by adapter
        developer_token=settings.GOOGLE_ADS_DEVELOPER_TOKEN,
        customer_id=settings.GOOGLE_ADS_CUSTOMER_ID,
        client_id=settings.GOOGLE_ADS_CLIENT_ID,
        client_secret=settings.GOOGLE_ADS_CLIENT_SECRET,
        refresh_token=settings.GOOGLE_ADS_REFRESH_TOKEN,
        timeout_seconds=settings.GOOGLE_ADS_TIMEOUT_SECONDS,
        max_retries=settings.GOOGLE_ADS_MAX_RETRIES,
        retry_strategy=RetryStrategy.EXPONENTIAL,
        rate_limit_requests=settings.GOOGLE_ADS_RATE_LIMIT_REQUESTS,
        rate_limit_window_seconds=settings.GOOGLE_ADS_RATE_LIMIT_WINDOW_SECONDS,
        use_circuit_breaker=True,
        enable_metrics=True,
        enable_logging=True
    )
    
    return GoogleAdsAdapter(config)