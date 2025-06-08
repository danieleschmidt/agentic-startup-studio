import pytest
from click.testing import CliRunner
from scripts.run_smoke_test import run_smoke_test

# SMOKE_TEST_RESULTS_DIR can be imported for assertion,
# but patching it in the script's namespace is more direct for tests.
# from scripts.run_smoke_test import SMOKE_TEST_RESULTS_DIR as SCRIPT_DEFAULT_DIR
import os
import json
import shutil  # For cleaning up directories
from unittest import mock  # Use unittest.mock for mocking
from pathlib import Path

# Define a directory for test outputs to avoid cluttering the main smoke_tests dir
# Using Path for robust path construction
TEST_OUTPUT_BASE_DIR = Path("tests/temp_smoke_test_outputs")


@pytest.fixture(scope="function", autouse=True)
def manage_test_output_dir():
    """Creates and cleans up the test output directory for each test function."""
    if TEST_OUTPUT_BASE_DIR.exists():
        shutil.rmtree(TEST_OUTPUT_BASE_DIR)
    TEST_OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    yield  # Test runs here

    # Clean up after tests by removing the temp directory
    if TEST_OUTPUT_BASE_DIR.exists():
        shutil.rmtree(TEST_OUTPUT_BASE_DIR)


@mock.patch("scripts.run_smoke_test.deploy_landing_page_to_unbounce")
@mock.patch("scripts.run_smoke_test.create_google_ads_campaign")
@mock.patch("scripts.run_smoke_test.get_campaign_metrics")
@mock.patch("scripts.run_smoke_test.AdBudgetSentinel")  # New mock
# Patch SMOKE_TEST_RESULTS_DIR in the script's context
@mock.patch("scripts.run_smoke_test.SMOKE_TEST_RESULTS_DIR", str(TEST_OUTPUT_BASE_DIR))
def test_run_smoke_test_successful_flow(
    mock_ad_sentinel_class,  # New mock argument, adjust order based on decorators
    mock_get_metrics,
    mock_create_campaign,
    mock_deploy_page,
):
    """Tests the successful execution flow of the run_smoke_test CLI command."""
    # Setup mock return values
    mock_deploy_page.return_value = "http://mockpages.com/test-idea-xyz"
    mock_campaign_id_val = "mock-campaign-id-xyz-123"
    mock_create_campaign.return_value = mock_campaign_id_val
    mock_metrics_data = {
        "clicks": 150,
        "conversions": 15,
        "ctr": 0.15,
        "campaign_id": mock_campaign_id_val,
        "total_cost": 40.0,  # Budget for test is 75.0, so this is within budget
    }
    mock_get_metrics.return_value = mock_metrics_data

    # Configure AdBudgetSentinel mock
    mock_sentinel_instance = mock_ad_sentinel_class.return_value
    mock_sentinel_instance.check_spend.return_value = True  # Simulate within budget

    runner = CliRunner()
    idea_id_to_test = "test-idea-xyz"
    budget_to_test = "75.0"

    # Invoke the CLI command
    result = runner.invoke(
        run_smoke_test,
        [
            "--idea-id",
            idea_id_to_test,
            "--budget",
            budget_to_test,
            "--results-dir",
            str(TEST_OUTPUT_BASE_DIR),
        ],
    )

    assert result.exit_code == 0, f"CLI command failed: {result.output}"
    budget_float = float(budget_to_test)
    start_msg = (
        f"Starting smoke test for Idea ID: {idea_id_to_test} "
        f"with budget: ${budget_float:.1f}"
    )
    assert start_msg in result.output

    resolved_path_str = str(TEST_OUTPUT_BASE_DIR.resolve())
    assert f"Results will be stored in: {resolved_path_str}" in result.output

    step1_msg = f"Step 1: Fetching details for Idea ID: {idea_id_to_test} (Placeholder)"
    assert step1_msg in result.output
    assert "Step 2: Preparing landing page configuration (Mocked)" in result.output

    mock_deploy_page.assert_called_once()
    assert mock_deploy_page.return_value in result.output

    mock_create_campaign.assert_called_once()
    assert mock_create_campaign.call_args[0][1] == budget_float
    assert mock_create_campaign.return_value in result.output

    assert "Fetching mock campaign metrics..." in result.output
    # Ensure get_campaign_metrics is called with ID from create_google_ads_campaign
    mock_get_metrics.assert_called_once_with(mock_create_campaign.return_value)

    # Slugify idea_id as done in the script
    idea_id_slug = "".join(
        c if c.isalnum() or c in ("-", "_") else "_" for c in idea_id_to_test
    )
    expected_analytics_path = TEST_OUTPUT_BASE_DIR / idea_id_slug / "analytics.json"

    assert expected_analytics_path.exists(), (
        f"Analytics file not found at {expected_analytics_path}"
    )
    with open(expected_analytics_path, "r") as f:
        saved_metrics = json.load(f)
    assert saved_metrics == mock_metrics_data

    expected_save_msg = (
        f"Mock metrics saved to: {str(expected_analytics_path.resolve())}"
    )
    assert expected_save_msg in result.output

    step7_msg = "Step 7: Simulating push of metrics to PostHog (Placeholder)"
    assert step7_msg in result.output
    completed_msg = f"Smoke test for Idea ID: {idea_id_to_test} completed."
    assert completed_msg in result.output

    # Assert AdBudgetSentinel calls
    mock_ad_sentinel_class.assert_called_once()
    assert mock_ad_sentinel_class.call_args[1]["max_budget"] == float(budget_to_test)
    assert mock_ad_sentinel_class.call_args[1]["campaign_id"] == mock_campaign_id_val
    mock_sentinel_instance.check_spend.assert_called_once_with(
        mock_metrics_data["total_cost"]
    )
    within_budget_msg = (
        f"Ad spend for campaign {mock_campaign_id_val} is within budget."
    )
    assert within_budget_msg in result.output


def test_run_smoke_test_missing_idea_id():
    """Tests CLI behavior when a required option is missing."""
    runner = CliRunner()
    result = runner.invoke(run_smoke_test, ["--budget", "10.0"])
    assert result.exit_code != 0
    assert "Missing option '--idea-id'" in result.output


def test_run_smoke_test_invalid_budget_type():
    """Tests CLI behavior with invalid type for an option."""
    runner = CliRunner()
    result = runner.invoke(
        run_smoke_test, ["--idea-id", "my-idea", "--budget", "not-a-float"]
    )
    assert result.exit_code != 0
    expected_msg = "Invalid value for '--budget': 'not-a-float' is not a valid float."
    assert expected_msg in result.output


@mock.patch("scripts.run_smoke_test.deploy_landing_page_to_unbounce")
@mock.patch("scripts.run_smoke_test.create_google_ads_campaign")
@mock.patch("scripts.run_smoke_test.get_campaign_metrics")
# Mock json.dump to simulate IO error
@mock.patch("scripts.run_smoke_test.json.dump")
@mock.patch("scripts.run_smoke_test.SMOKE_TEST_RESULTS_DIR", str(TEST_OUTPUT_BASE_DIR))
@mock.patch(
    "scripts.run_smoke_test.AdBudgetSentinel"
)  # Also mock AdBudgetSentinel here
def test_run_smoke_test_metrics_saving_error(
    mock_ad_sentinel_class,  # Add mock argument
    mock_json_dump,
    mock_get_metrics,
    mock_create_campaign,
    mock_deploy_page,
):
    """Tests if an error during metrics saving is handled."""
    # Configure AdBudgetSentinel mock to behave normally (within budget)
    mock_sentinel_instance = mock_ad_sentinel_class.return_value
    mock_sentinel_instance.check_spend.return_value = True

    mock_deploy_page.return_value = "http://mockpages.com/test-io-error"
    mock_create_campaign.return_value = "mock-campaign-id-io-error"
    mock_get_metrics.return_value = {"data": "test"}
    mock_json_dump.side_effect = IOError("Simulated disk full error")

    runner = CliRunner()
    result = runner.invoke(
        run_smoke_test,
        ["--idea-id", "test-io-error", "--results-dir", str(TEST_OUTPUT_BASE_DIR)],
    )

    assert result.exit_code == 0  # Script should still complete
    assert "Error saving metrics: Simulated disk full error" in result.output
    assert "Smoke test for Idea ID: test-io-error completed." in result.output


# New test case for budget exceeded scenario
@mock.patch("scripts.run_smoke_test.deploy_landing_page_to_unbounce")
@mock.patch("scripts.run_smoke_test.create_google_ads_campaign")
@mock.patch("scripts.run_smoke_test.get_campaign_metrics")
@mock.patch("scripts.run_smoke_test.AdBudgetSentinel")
@mock.patch("scripts.run_smoke_test.SMOKE_TEST_RESULTS_DIR", str(TEST_OUTPUT_BASE_DIR))
def test_run_smoke_test_ad_budget_exceeded(
    mock_ad_sentinel_class, mock_get_metrics, mock_create_campaign, mock_deploy_page
):
    # Setup mocks for ads_manager functions
    mock_deploy_page.return_value = "http://mockpages.com/test-idea-budget"
    mock_campaign_id = "mock-campaign-budget-exceeded"
    mock_create_campaign.return_value = mock_campaign_id
    mock_metrics_data = {
        "clicks": 200,
        "conversions": 10,
        "ctr": 0.1,
        "campaign_id": mock_campaign_id,
        "total_cost": 150.0,  # Over budget
    }
    mock_get_metrics.return_value = mock_metrics_data

    # Configure AdBudgetSentinel mock
    mock_sentinel_instance = mock_ad_sentinel_class.return_value
    mock_sentinel_instance.check_spend.return_value = False  # Simulate budget exceeded

    runner = CliRunner()
    budget = 50.0  # Campaign budget from CLI
    result = runner.invoke(
        run_smoke_test, ["--idea-id", "test-idea-budget", "--budget", str(budget)]
    )

    assert result.exit_code == 0  # Script should still complete
    mock_ad_sentinel_class.assert_called_once()
    # Check if AdBudgetSentinel was called with the correct budget
    assert mock_ad_sentinel_class.call_args[1]["max_budget"] == budget
    assert mock_ad_sentinel_class.call_args[1]["campaign_id"] == mock_campaign_id

    mock_sentinel_instance.check_spend.assert_called_once_with(150.0)  # total_cost
    assert "AdBudgetSentinel initialized" in result.output
    assert "Checking ad spend: $150.00 against budget" in result.output
    msg = (
        f"Ad budget exceeded for campaign {mock_campaign_id}. "
        f"Further actions would be halted."
    )
    assert msg in result.output
