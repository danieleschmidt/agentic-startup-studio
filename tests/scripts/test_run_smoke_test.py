import pytest
from click.testing import CliRunner
from unittest import mock
import os

# Assuming scripts.run_smoke_test and core.ads_manager are accessible
# This script is intended to be run with PYTHONPATH including the project root.
from scripts.run_smoke_test import run_smoke_test

# Mock core components to isolate the script's logic
@pytest.fixture(autouse=True)
def mock_ads_manager_functions():
    with mock.patch("scripts.run_smoke_test.deploy_landing_page_to_unbounce") as mock_deploy:
        mock_deploy.return_value = "http://mock-landing-page.com/test-idea"
        with mock.patch("scripts.run_smoke_test.create_google_ads_campaign") as mock_create_campaign:
            mock_create_campaign.return_value = "campaign_123"
            with mock.patch("scripts.run_smoke_test.get_campaign_metrics") as mock_get_metrics:
                mock_get_metrics.return_value = {
                    "impressions": 1000,
                    "clicks": 100,
                    "ctr": 0.1,
                    "total_cost": 20.0,
                    "average_cpc": 0.20,
                }
                yield mock_deploy, mock_create_campaign, mock_get_metrics

@pytest.fixture
def runner():
    return CliRunner()

@pytest.fixture
def mock_posthog_capture():
    # This fixture ensures that posthog.capture is always mocked for tests in this file.
    # We access posthog via the run_smoke_test module's namespace.
    with mock.patch("scripts.run_smoke_test.posthog.capture") as mock_capture:
        yield mock_capture

@pytest.fixture
def manage_posthog_state():
    """
    A fixture to manage PostHog state (API key, host, and disabled status) for tests.
    It allows tests to specify whether PostHog should be 'enabled' or 'disabled'.
    """
    original_api_key = os.environ.get("POSTHOG_API_KEY")
    original_host = os.environ.get("POSTHOG_HOST")
    original_pytest_current_test = os.environ.get("PYTEST_CURRENT_TEST")

    def _manage(enabled: bool, clear_pytest_env_var: bool = False):
        if enabled:
            # Ensure PostHog is "enabled" for the test
            os.environ["POSTHOG_API_KEY"] = "test_key"
            os.environ["POSTHOG_HOST"] = "test_host"
            if clear_pytest_env_var: # If we explicitly want to test non-pytest scenario
                 if "PYTEST_CURRENT_TEST" in os.environ:
                    del os.environ["PYTEST_CURRENT_TEST"]
            # Patch the module-level constants and posthog.disabled
            # This directly influences the behavior of send_metrics_to_posthog
            patches = [
                mock.patch("scripts.run_smoke_test.POSTHOG_API_KEY", "test_key"),
                mock.patch("scripts.run_smoke_test.POSTHOG_HOST", "test_host"),
                mock.patch("scripts.run_smoke_test.posthog.disabled", False),
            ]
        else:
            # Ensure PostHog is "disabled" for the test
            # This can be due to missing API key/host or PYTEST_CURRENT_TEST being set
            if "POSTHOG_API_KEY" in os.environ: del os.environ["POSTHOG_API_KEY"]
            if "POSTHOG_HOST" in os.environ: del os.environ["POSTHOG_HOST"]
            # Ensure PYTEST_CURRENT_TEST is set to also trigger disabled state if that's desired for the test
            # os.environ["PYTEST_CURRENT_TEST"] = "any_test_name" # This is one way to disable
            patches = [
                mock.patch("scripts.run_smoke_test.POSTHOG_API_KEY", None),
                mock.patch("scripts.run_smoke_test.POSTHOG_HOST", None),
                mock.patch("scripts.run_smoke_test.posthog.disabled", True), # Primary way to disable
            ]

        for p in patches:
            p.start()

        return patches

    yield _manage

    # Teardown: Stop all patches and restore original environment variables
    mock.patch.stopall() # Stops all active patches started by mock.patch()

    if original_api_key is None:
        if "POSTHOG_API_KEY" in os.environ: del os.environ["POSTHOG_API_KEY"]
    else:
        os.environ["POSTHOG_API_KEY"] = original_api_key

    if original_host is None:
        if "POSTHOG_HOST" in os.environ: del os.environ["POSTHOG_HOST"]
    else:
        os.environ["POSTHOG_HOST"] = original_host

    if original_pytest_current_test is None:
        if "PYTEST_CURRENT_TEST" in os.environ: del os.environ["PYTEST_CURRENT_TEST"]
    else:
        os.environ["PYTEST_CURRENT_TEST"] = original_pytest_current_test


def test_run_smoke_test_posthog_enabled_and_called(runner, mock_posthog_capture, manage_posthog_state, tmp_path):
    """
    Test that posthog.capture is called with correct arguments when PostHog is enabled.
    """
    manage_posthog_state(enabled=True, clear_pytest_env_var=True) # Ensure it's not disabled by PYTEST_CURRENT_TEST

    idea_id = "test-idea-ph-enabled"
    results_dir = tmp_path / "smoke_results"

    result = runner.invoke(run_smoke_test, ["--idea-id", idea_id, "--results-dir", str(results_dir)])

    assert result.exit_code == 0, f"CLI Errored: {result.output}"
    mock_posthog_capture.assert_called_once()

    args, kwargs = mock_posthog_capture.call_args
    expected_distinct_id = f"smoke_test_system_{idea_id}"
    expected_event_name = "smoke_test_campaign_metrics"

    assert args[0] == expected_distinct_id
    assert kwargs["event"] == expected_event_name

    properties = kwargs["properties"]
    assert properties["idea_id"] == idea_id
    assert properties["campaign_name"] == f"Idea {idea_id} Ad Campaign"
    assert properties["deployment_url"] == "http://mock-landing-page.com/test-idea"
    assert properties["impressions"] == 1000
    assert "Successfully sent metrics to PostHog" in result.output

def test_run_smoke_test_posthog_disabled_by_pytest_env(runner, mock_posthog_capture, manage_posthog_state, tmp_path):
    """
    Test that posthog.capture is NOT called when PYTEST_CURRENT_TEST is set (simulating test environment).
    """
    # Explicitly set PYTEST_CURRENT_TEST and ensure posthog.disabled reflects this
    os.environ["PYTEST_CURRENT_TEST"] = "sometest"
    # The manage_posthog_state will by default set posthog.disabled to True if enabled=False
    # We also need to ensure the module-level check for PYTEST_CURRENT_TEST in run_smoke_test.py
    # has its intended effect. Patching posthog.disabled directly is the most robust.
    with mock.patch("scripts.run_smoke_test.posthog.disabled", True):
        idea_id = "test-idea-ph-disabled-pytest"
        results_dir = tmp_path / "smoke_results"

        result = runner.invoke(run_smoke_test, ["--idea-id", idea_id, "--results-dir", str(results_dir)])

        assert result.exit_code == 0, f"CLI Errored: {result.output}"
        mock_posthog_capture.assert_not_called()
        assert "PostHog is disabled. Skipping sending metrics." in result.output

    if "PYTEST_CURRENT_TEST" in os.environ: # Clean up env var
        del os.environ["PYTEST_CURRENT_TEST"]


def test_run_smoke_test_posthog_disabled_by_missing_config(runner, mock_posthog_capture, manage_posthog_state, tmp_path):
    """
    Test that posthog.capture is NOT called when PostHog API key/host are not set.
    """
    manage_posthog_state(enabled=False) # This sets API keys to None and posthog.disabled to True

    idea_id = "test-idea-ph-disabled-config"
    results_dir = tmp_path / "smoke_results"

    result = runner.invoke(run_smoke_test, ["--idea-id", idea_id, "--results-dir", str(results_dir)])

    assert result.exit_code == 0, f"CLI Errored: {result.output}"
    mock_posthog_capture.assert_not_called()
    assert "PostHog is disabled. Skipping sending metrics." in result.output
    # The warning about missing config is printed at module load time.
    # If other tests ran first and loaded the module with config, that warning might not show here.
    # The "Skipping sending metrics" is the key operational check.

def test_run_smoke_test_successful_execution_structure(runner, manage_posthog_state, tmp_path):
    """
    Basic test to ensure the command runs without error and creates output.
    PostHog calls are not the focus here.
    """
    manage_posthog_state(enabled=False) # Disable PostHog for this structural test

    idea_id = "test-idea-basic"
    results_dir = tmp_path / "smoke_results"

    result = runner.invoke(run_smoke_test, ["--idea-id", idea_id, "--results-dir", str(results_dir)])

    assert result.exit_code == 0, f"CLI Errored: {result.output}"
    assert f"Smoke test for Idea ID: {idea_id} completed." in result.output

    idea_results_path = results_dir / "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in idea_id)
    analytics_file = idea_results_path / "analytics.json"
    assert analytics_file.exists()

# General notes on these tests:
# - `mock_posthog_capture` is autoused to always mock `posthog.capture`.
# - `manage_posthog_state` is a more robust fixture to control the perceived state of PostHog's configuration
#   and `posthog.disabled` status within the `run_smoke_test` module. It directly patches
#   the necessary attributes within the `scripts.run_smoke_test` module's namespace.
# - This approach avoids issues with Python's module import caching, where module-level
#   configurations (like PostHog API keys or the `posthog.disabled` flag based on initial
#   `os.environ` checks) might not be re-evaluated if `os.environ` is changed after import.
# - The tests cover:
#   1. PostHog enabled: `posthog.capture` is called with correct data.
#   2. PostHog disabled due to `PYTEST_CURRENT_TEST`: `posthog.capture` is not called.
#   3. PostHog disabled due to missing API key/host: `posthog.capture` is not called.
#   4. Basic script execution: ensures the script runs and produces expected output files.
# - The `autouse=True` for `mock_ads_manager_functions` simplifies test setup by always mocking
#   external dependencies of the script.
# - `tmp_path` pytest fixture is used for creating temporary directories for test results,
#   ensuring tests don't leave behind artifacts.
# - Slugification of idea_id for results path is duplicated here to match script's behavior.
#   This could be refactored into a shared utility if used in more places.
# - The CLI output checks (`assert "message" in result.output`) are useful for verifying
#   user feedback and debugging.
# - `mock.patch.stopall()` in `manage_posthog_state` teardown is important to prevent patches
#   from leaking between tests.
# - `clear_pytest_env_var=True` in `manage_posthog_state` for the "enabled" test case ensures that
#   even if `PYTEST_CURRENT_TEST` was set globally (e.g., by a test runner), we can override
#   its effect for that specific test to check the PostHog enabled path.
# - The test for `PYTEST_CURRENT_TEST` disabling PostHog (`test_run_smoke_test_posthog_disabled_by_pytest_env`)
#   now explicitly sets the env var and then relies on patching `posthog.disabled` to `True`
#   to ensure the intended behavior is tested regardless of when the module was imported.
#   This is because the module-level check in `run_smoke_test.py` for `PYTEST_CURRENT_TEST`
#   only runs once at import time.
#
# To run:
# Ensure posthog is in requirements and installed.
# PYTHONPATH=. pytest tests/scripts/test_run_smoke_test.py
# (or however you normally run pytest for your project structure)
