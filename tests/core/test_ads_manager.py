import pytest
from core.ads_manager import (
    deploy_landing_page_to_unbounce,
    create_google_ads_campaign,
    get_campaign_metrics,
)
from typing import Dict, Any


def test_deploy_landing_page_to_unbounce():
    page_config: Dict[str, Any] = {
        "name": "Test Landing Page",
        "content": "<h1>Hello</h1>",
    }
    result = deploy_landing_page_to_unbounce(page_config)
    assert isinstance(result, str)
    assert "http://unbouncepages.com/mock-test-landing-page" == result

    # Test with a name that needs slugification
    page_config_spaces: Dict[str, Any] = {"name": "My Awesome Page"}
    result_spaces = deploy_landing_page_to_unbounce(page_config_spaces)
    assert "http://unbouncepages.com/mock-my-awesome-page" == result_spaces

    # Test with default name if 'name' key is missing
    page_config_no_name: Dict[str, Any] = {"content": "Some content"}
    result_no_name = deploy_landing_page_to_unbounce(page_config_no_name)
    assert "http://unbouncepages.com/mock-page" == result_no_name

    # Test with non-string name (should be converted to string)
    page_config_int_name: Dict[str, Any] = {"name": 12345}
    result_int_name = deploy_landing_page_to_unbounce(page_config_int_name)
    assert "http://unbouncepages.com/mock-12345" == result_int_name


def test_create_google_ads_campaign():
    campaign_config: Dict[str, Any] = {
        "name": "Test Ad Campaign",
        "keywords": ["test", "ads"],
    }
    budget: float = 50.0
    result = create_google_ads_campaign(campaign_config, budget)
    assert isinstance(result, str)
    assert result.startswith("mock-gads-campaign-test-ad-campaign-")

    # Test with a name that needs slugification
    campaign_config_spaces: Dict[str, Any] = {"name": "My Ad Campaign"}
    result_spaces = create_google_ads_campaign(campaign_config_spaces, budget)
    assert result_spaces.startswith("mock-gads-campaign-my-ad-campaign-")

    # Test with default name if 'name' key is missing
    campaign_config_no_name: Dict[str, Any] = {"keywords": ["generic"]}
    result_no_name = create_google_ads_campaign(campaign_config_no_name, budget)
    assert result_no_name.startswith("mock-gads-campaign-campaign-")

    # Test with non-string name
    campaign_config_int_name: Dict[str, Any] = {"name": 987}
    result_int_name = create_google_ads_campaign(campaign_config_int_name, budget)
    assert result_int_name.startswith("mock-gads-campaign-987-")


def test_get_campaign_metrics():
    campaign_id = "mock-gads-campaign-12345"
    result = get_campaign_metrics(campaign_id)
    assert isinstance(result, dict)
    assert result["campaign_id"] == campaign_id
    assert "impressions" in result
    assert "clicks" in result
    assert "ctr" in result
    assert result["ctr"] == 0.10
    assert result["conversions"] == 5
    assert result["conversion_rate"] == 0.05
    assert result["cost_per_click"] == 1.50
    assert result["total_cost"] == 150.00

    # Test with a different campaign ID
    another_campaign_id = "another-mock-id"
    result_another = get_campaign_metrics(another_campaign_id)
    assert result_another["campaign_id"] == another_campaign_id
    # Other metrics are static in the mock, so they'll be the same
    assert result_another["impressions"] == 1000
