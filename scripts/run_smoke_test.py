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
from core.alert_manager import AlertManager  # Import AlertManager

# from core.idea_ledger import get_idea_by_id # Placeholder for future use
from core.ad_budget_sentinel import AdBudgetSentinel  # Import AdBudgetSentinel

SMOKE_TEST_RESULTS_DIR = "smoke_tests"  # Default results directory
TARGET_CTR_FOR_BUDGET_INCREASE = 0.05  # 5% CTR target for budget increase


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
    # Sanitize idea_id for use in paths (simple version)
    idea_id_slug = "".join(
        c if c.isalnum() or c in ("-", "_") else "_" for c in idea_id
    )

    # Ensure base results_dir and specific idea_results_path exist for logs
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    idea_log_path_base = os.path.join(results_dir, idea_id_slug)
    if not os.path.exists(idea_log_path_base):
        os.makedirs(idea_log_path_base)  # Ensure idea-specific log/results directory

    # Instantiate AlertManager for this smoke test run
    alert_log_fname = f"alerts_smoke_test_{idea_id_slug}.log"
    alert_manager_log_path = os.path.join(idea_log_path_base, alert_log_fname)
    alert_manager = AlertManager(log_file_path=alert_manager_log_path)
    am_init_msg = (
        f"AlertManager initialized. Alerts will be logged to: {alert_manager_log_path}"
    )
    click.echo(am_init_msg)

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
        alert_manager=alert_manager,  # Pass the AlertManager instance
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

    # Compare CTR and suggest budget increase if applicable
    actual_ctr = metrics.get("ctr", 0.0)
    click.echo(
        f"Comparing actual CTR ({actual_ctr:.4f}) against target CTR "
        f"({TARGET_CTR_FOR_BUDGET_INCREASE:.4f})."
    )
    if actual_ctr >= TARGET_CTR_FOR_BUDGET_INCREASE:
        suggested_next_budget = budget * 1.5  # Example: Suggest 50% increase
        click.echo(
            f"SUCCESS: Actual CTR ({actual_ctr:.4f}) met or exceeded target "
            f"({TARGET_CTR_FOR_BUDGET_INCREASE:.4f}). "
            f"Consider increasing budget for next run. "
            f"Suggested next budget: ${suggested_next_budget:.2f}"
        )
    else:
        click.echo(
            f"INFO: Actual CTR ({actual_ctr:.4f}) did not meet target "
            f"({TARGET_CTR_FOR_BUDGET_INCREASE:.4f}). "
            "Budget increase not suggested based on CTR."
        )

    # 6. Store mock metrics
    click.echo("Step 6: Storing mock metrics...")
    # Use the resolved 'results_dir' from Click option
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)  # Ensure base results_dir exists

    idea_results_path = idea_log_path_base  # Use the same path for analytics.json

    # Ensure the directory exists (it should from alert manager setup,
    # but good practice)
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

    # Print alerts from this run
    logged_alerts = alert_manager.get_logged_alerts()
    if logged_alerts:
        click.echo("\n--- Alerts Recorded During This Smoke Test ---")
        for alert_idx, alert_msg_dict in enumerate(logged_alerts):
            click.echo(
                f"- Alert {alert_idx + 1}: Level: {alert_msg_dict['level']}, "
                f"Source: {alert_msg_dict['source']}, "
                f"Message: {alert_msg_dict['message']}"
            )
        # alert_manager.clear_logged_alerts() # Optional: clear after showing
    else:
        no_alerts_msg = (
            "\nNo critical alerts recorded by AlertManager during this smoke test run."
        )
        click.echo(no_alerts_msg)


if __name__ == "__main__":
    # This allows running the script directly for testing,
    # e.g., python scripts/run_smoke_test.py --idea-id test-idea-123
    run_smoke_test()
