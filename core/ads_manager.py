from typing import (
    Dict,
    Any,
)  # List was missing in the prompt, but good for type hinting


def deploy_landing_page_to_unbounce(page_config: Dict[str, Any]) -> str:
    """
    Simulates deploying a landing page configuration to Unbounce.
    In a real scenario, this would involve API calls to Unbounce.

    Args:
        page_config: A dictionary containing landing page details
                     (e.g., name, content, target URL).

    Returns:
        A mock deployment URL for the landing page.
    """
    page_name = page_config.get("name", "page")
    # Ensure page_name is a string before calling lower() and replace()
    if not isinstance(page_name, str):
        page_name = str(page_name)  # Or handle error appropriately

    print(
        f"Simulating deployment of landing page: "
        f"{page_config.get('name', 'Unknown Page')}"
    )
    # In a real implementation, you'd make an API call here.
    mock_url = f"http://unbouncepages.com/mock-{page_name.lower().replace(' ', '-')}"
    print(f"Mock deployment URL: {mock_url}")
    return mock_url


def create_google_ads_campaign(campaign_config: Dict[str, Any], budget: float) -> str:
    """
    Simulates creating a Google Ads campaign.
    In a real scenario, this would involve API calls to Google Ads.

    Args:
        campaign_config: A dictionary containing campaign details
                         (e.g., name, target keywords, ad copy).
        budget: The budget for the campaign.

    Returns:
        A mock campaign ID.
    """
    campaign_name = campaign_config.get("name", "campaign")
    # Ensure campaign_name is a string
    if not isinstance(campaign_name, str):
        campaign_name = str(campaign_name)

    print(
        f"Simulating creation of Google Ads campaign: "
        f"{campaign_config.get('name', 'Unknown Campaign')} with budget ${budget}"
    )
    # In a real implementation, you'd make an API call here.
    # Using hash for some variability in mock ID, ensuring it's positive with abs()
    campaign_name_slug = campaign_name.lower().replace(" ", "-")
    unique_hash_part = abs(hash(str(campaign_config) + str(budget))) % 10000
    mock_campaign_id = f"mock-gads-campaign-{campaign_name_slug}-{unique_hash_part}"
    print(f"Mock campaign ID: {mock_campaign_id}")
    return mock_campaign_id


def get_campaign_metrics(campaign_id: str) -> Dict[str, Any]:
    """
    Simulates fetching metrics for a given Google Ads campaign ID.
    In a real scenario, this would involve API calls to Google Ads.

    Args:
        campaign_id: The ID of the campaign to fetch metrics for.

    Returns:
        A dictionary containing mock campaign metrics.
    """
    print(f"Simulating fetching metrics for campaign ID: {campaign_id}")
    # In a real implementation, you'd make an API call here.
    mock_metrics = {
        "campaign_id": campaign_id,
        "impressions": 1000,
        "clicks": 100,
        "ctr": 0.10,  # Click-Through Rate
        "conversions": 5,
        "conversion_rate": 0.05,  # (conversions / clicks)
        "cost_per_click": 1.50,  # Mock CPC
        "total_cost": 150.00,  # (clicks * CPC)
    }
    print(f"Mock metrics: {mock_metrics}")
    return mock_metrics


def adjust_campaign_budget(
    campaign_id: str, new_budget: float, reason: str = "Performance based adjustment"
) -> bool:
    """
    Simulates adjusting the budget for an existing Google Ads campaign.
    In a real scenario, this would involve API calls to Google Ads.

    Args:
        campaign_id: The ID of the campaign to adjust.
        new_budget: The new budget to set for the campaign.
        reason: The reason for the budget adjustment.

    Returns:
        True if the budget adjustment was "successful", False otherwise.
    """
    if new_budget <= 0:
        print(
            f"Error: New budget for campaign {campaign_id} must be positive. "
            f"Attempted: ${new_budget:.2f}"
        )
        return False

    print(f"Simulating budget adjustment for campaign ID: {campaign_id}")
    print(f"Setting new budget to: ${new_budget:.2f}. Reason: {reason}")
    # In a real implementation, you'd make an API call here.
    print(f"Mock budget adjustment successful for campaign {campaign_id}.")
    return True


if __name__ == "__main__":
    print("--- Testing Ads Manager Functions ---")

    # Test deploy_landing_page_to_unbounce
    print("\nTesting deploy_landing_page_to_unbounce...")
    page_conf = {"name": "My Test Page", "content": "Some HTML"}
    page_url = deploy_landing_page_to_unbounce(page_conf)
    assert page_url == "http://unbouncepages.com/mock-my-test-page"

    page_conf_no_name = {"content": "Some HTML"}
    page_url_no_name = deploy_landing_page_to_unbounce(page_conf_no_name)
    assert page_url_no_name == "http://unbouncepages.com/mock-page"
    print("deploy_landing_page_to_unbounce tests passed.")

    # Test create_google_ads_campaign
    print("\nTesting create_google_ads_campaign...")
    campaign_conf = {"name": "Summer Sale Campaign", "keywords": ["summer", "sale"]}
    budget_val = 100.00
    campaign_id_val = create_google_ads_campaign(campaign_conf, budget_val)
    assert campaign_id_val.startswith("mock-gads-campaign-summer-sale-campaign-")

    campaign_conf_no_name = {"keywords": ["generic"]}
    campaign_id_no_name = create_google_ads_campaign(campaign_conf_no_name, budget_val)
    assert campaign_id_no_name.startswith("mock-gads-campaign-campaign-")
    print("create_google_ads_campaign tests passed.")

    # Test get_campaign_metrics
    print("\nTesting get_campaign_metrics...")
    metrics = get_campaign_metrics(campaign_id_val)
    assert metrics["campaign_id"] == campaign_id_val
    assert metrics["impressions"] == 1000
    assert metrics["clicks"] == 100
    assert metrics["ctr"] == 0.10
    print("get_campaign_metrics tests passed.")

    # Test adjust_campaign_budget
    print("\nTesting adjust_campaign_budget...")
    adj_success = adjust_campaign_budget(campaign_id_val, 150.0, "Initial boost")
    assert adj_success is True
    adj_fail_zero = adjust_campaign_budget(campaign_id_val, 0.0)
    assert adj_fail_zero is False
    adj_fail_neg = adjust_campaign_budget(campaign_id_val, -50.0)
    assert adj_fail_neg is False
    print("adjust_campaign_budget tests passed.")

    print("\n--- Ads Manager Functions Test Finished ---")
