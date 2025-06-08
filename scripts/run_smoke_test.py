import click
import json
import os
import time

# Assuming core.ads_manager and core.idea_ledger are accessible
# This script is intended to be run with PYTHONPATH including the project root.
from core.ads_manager import (
    deploy_landing_page_to_unbounce,
    create_google_ads_campaign,
    get_campaign_metrics,
)

# from core.idea_ledger import get_idea_by_id # Placeholder for future use
from core.ad_budget_sentinel import AdBudgetSentinel  # Import AdBudgetSentinel

SMOKE_TEST_RESULTS_DIR = "smoke_tests"  # Default results directory


@click.command()
@click.option(
    "--idea-id", "idea_id", required=True, help="The ID of the idea to smoke test."
)
@click.option(
    "--budget", type=float, default=50.0, help="Advertising budget for the smoke test."
)
@click.option(
    "--results-dir",
    default=SMOKE_TEST_RESULTS_DIR,
    help="Directory to store smoke test results.",
    type=click.Path(file_okay=False, dir_okay=True, writable=True, resolve_path=True),
)
def run_smoke_test(idea_id: str, budget: float, results_dir: str):
    """
    Orchestrates the smoke test process for a given idea.
    This involves:
    1. Fetching idea details (placeholder).
    2. Preparing landing page configuration (mocked).
    3. Deploying a mock landing page (using ads_manager).
    4. Creating a mock ad campaign (using ads_manager).
    5. Simulating campaign run and fetching mock metrics (using ads_manager).
    6. Storing mock metrics.
    7. Simulating pushing metrics to PostHog (placeholder).
    """
    click.echo(f"Starting smoke test for Idea ID: {idea_id} with budget: ${budget}")
    click.echo(f"Results will be stored in: {results_dir}")

    # 1. Fetch idea details (placeholder)
    click.echo(f"Step 1: Fetching details for Idea ID: {idea_id} (Placeholder)")
    # Example of how UUID validation could be done if idea_id is expected to be UUID.
    # try:
    #     idea_uuid = UUID(idea_id)
    #     # idea_details = get_idea_by_id(idea_uuid) # Actual call
    #     # if not idea_details:
    #     #    click.echo(f"Error: Idea with ID {idea_id} not found.")
    #     #    return
    #     # click.echo(f"Successfully fetched details for: {idea_details.name}")
    # except ValueError:
    #     # Allowing non-UUIDs for idea_id as per current test case (test-idea-xyz)
    #     # click.echo(f"Warning: Idea ID {idea_id} is not a valid UUID.")
    #     pass # Proceed with non-UUID idea_id for this version

    # Using a simple mock based on idea_id string directly
    mock_idea_details = {"name": f"Idea {idea_id}", "description": "A fantastic idea."}
    click.echo(f"Mock idea details: {mock_idea_details['name']}")

    # 2. Prepare landing page configuration (mocked)
    click.echo("Step 2: Preparing landing page configuration (Mocked)")
    landing_page_config = {
        "name": f"{mock_idea_details['name']} Landing Page",
        "content": (
            f"<h1>Welcome to {mock_idea_details['name']}</h1>"
            f"<p>{mock_idea_details['description']}</p>"
            f"<p>Sign up now!</p>"
        ),
        "target_url_for_ads": (
            f"http://example.com/landing/{idea_id.replace(' ', '_').lower()}"
        ),  # Mock target
    }
    click.echo(f"Landing page config name: {landing_page_config['name']}")

    # 3. Deploy mock landing page
    click.echo("Step 3: Deploying mock landing page...")
    deployment_url = deploy_landing_page_to_unbounce(landing_page_config)
    click.echo(f"Mock landing page deployed to: {deployment_url}")

    # 4. Create mock ad campaign
    click.echo("Step 4: Creating mock ad campaign...")
    campaign_config = {
        "name": f"{mock_idea_details['name']} Ad Campaign",
        "target_keywords": ["startup", "innovation", idea_id],
        "ad_copy": f"Discover {mock_idea_details['name']} - Sign up today!",
        "landing_page_url": deployment_url,
    }
    campaign_id = create_google_ads_campaign(campaign_config, budget)
    click.echo(f"Mock ad campaign created with ID: {campaign_id}")

    # Instantiate AdBudgetSentinel
    ad_sentinel = AdBudgetSentinel(
        max_budget=budget,
        campaign_id=campaign_id,
        halt_callback=lambda camp_id, reason: click.echo(
            f"HALT_CALLBACK: Campaign {camp_id} ordered to halt. Reason: {reason}",
            err=True,
        ),
        alert_callback=lambda msg: click.echo(f"ALERT_CALLBACK: {msg}", err=True),
    )
    click.echo(
        f"AdBudgetSentinel initialized for campaign {campaign_id} "
        f"with max budget ${budget:.2f}"
    )

    # 5. Simulate campaign run and fetch mock metrics
    click.echo("Step 5: Simulating campaign run (waiting 3 seconds)...")
    time.sleep(3)  # Simulate time delay
    click.echo("Fetching mock campaign metrics...")
    metrics = get_campaign_metrics(campaign_id)
    click.echo(f"Retrieved metrics: {metrics}")

    # Check ad spend
    current_spend = metrics.get("total_cost", 0.0)
    click.echo(
        f"Checking ad spend: ${current_spend:.2f} against budget "
        f"using AdBudgetSentinel."
    )
    if not ad_sentinel.check_spend(current_spend):
        click.echo(
            f"Ad budget exceeded for campaign {campaign_id}. "
            f"Further actions would be halted.",
            err=True,
        )
    else:
        click.echo(f"Ad spend for campaign {campaign_id} is within budget.")

    # 6. Store mock metrics
    click.echo("Step 6: Storing mock metrics...")
    # Use the resolved 'results_dir' from Click option
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)  # Ensure base results_dir exists

    # Sanitize idea_id for use as a directory name if it contains problematic chars
    # For this example, we assume idea_id is simple enough or paths are handled.
    # A more robust slugification might be needed for arbitrary idea_id strings.
    idea_id_slug = "".join(
        c if c.isalnum() or c in ("-", "_") else "_" for c in idea_id
    )
    idea_results_path = os.path.join(results_dir, idea_id_slug)

    if not os.path.exists(idea_results_path):
        os.makedirs(idea_results_path)

    analytics_file_path = os.path.join(idea_results_path, "analytics.json")
    try:
        with open(analytics_file_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        click.echo(f"Mock metrics saved to: {analytics_file_path}")
    except IOError as e:
        click.echo(f"Error saving metrics: {e}", err=True)

    # 7. Simulate pushing metrics to PostHog (placeholder)
    click.echo("Step 7: Simulating push of metrics to PostHog (Placeholder)")
    click.echo(f"Metrics for PostHog: {metrics}")

    click.echo(f"Smoke test for Idea ID: {idea_id} completed.")


if __name__ == "__main__":
    # This allows running the script directly for testing,
    # e.g., python scripts/run_smoke_test.py --idea-id test-idea-123
    run_smoke_test()
