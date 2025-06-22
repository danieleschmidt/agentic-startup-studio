import importlib
import shutil
from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner

from scripts.run_build_cycle import GENERATED_MVPS_DIR, run_build_cycle

# Define a directory for test outputs, distinct from build_tools_manager tests
TEST_BUILD_CYCLE_BASE_OUTPUT_DIR = Path("tests/temp_build_cycle_outputs")


@pytest.fixture(scope="function", autouse=True)
def manage_test_output_dir_for_script():
    """Creates and cleans up the test output directory for each test function."""
    # The script uses GENERATED_MVPS_DIR as a subdir for idea outputs.
    # So, our test output base will contain this GENERATED_MVPS_DIR.
    # Patched GENERATED_MVPS_DIR for script will be:
    # TEST_BUILD_CYCLE_BASE_OUTPUT_DIR / GENERATED_MVPS_DIR

    if TEST_BUILD_CYCLE_BASE_OUTPUT_DIR.exists():
        shutil.rmtree(TEST_BUILD_CYCLE_BASE_OUTPUT_DIR)
    # Test setup should handle dir creation if script doesn't,
    # or test script's ability to create it.
    yield
    if TEST_BUILD_CYCLE_BASE_OUTPUT_DIR.exists():
        shutil.rmtree(TEST_BUILD_CYCLE_BASE_OUTPUT_DIR)


# Path used by the SCRIPT when its GENERATED_MVPS_DIR is patched.
PATCHED_SCRIPT_MVPS_DIR = str(TEST_BUILD_CYCLE_BASE_OUTPUT_DIR / GENERATED_MVPS_DIR)


@mock.patch("scripts.run_build_cycle.check_fly_io_health")  # Patch where looked up
@mock.patch("scripts.run_build_cycle.deploy_to_fly_io")  # Patch where looked up
@mock.patch("scripts.run_build_cycle.run_gpt_engineer")
@mock.patch("scripts.run_build_cycle.run_opendevin_debug_loop")
@mock.patch("scripts.run_build_cycle.GENERATED_MVPS_DIR", PATCHED_SCRIPT_MVPS_DIR)
def test_run_build_cycle_successful_flow(
    mock_opendevin,  # Innermost, from scripts.run_build_cycle
    mock_gpt_engineer,  # from scripts.run_build_cycle
    mock_deploy_fly,  # from scripts.run_build_cycle
    mock_check_health,  # Outermost, from scripts.run_build_cycle
):
    mock_gpt_engineer.return_value = True
    mock_opendevin.return_value = True
    mock_deploy_fly.return_value = "https://mock-app-test-build-idea.fly.dev"
    mock_check_health.return_value = True

    runner = CliRunner()
    idea_id_to_test = "test-build-idea"
    result = runner.invoke(run_build_cycle, ["--idea-id", idea_id_to_test])

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert f"Starting build cycle for Idea ID: {idea_id_to_test}" in result.output
    assert (
        f"Step 1: Fetching details for Idea ID: {idea_id_to_test} (Mocked)"
    ) in result.output

    expected_output_path_for_idea = Path(PATCHED_SCRIPT_MVPS_DIR) / idea_id_to_test

    mock_gpt_engineer.assert_called_once()
    call_args_gpt = mock_gpt_engineer.call_args[0]
    assert call_args_gpt[1] == str(expected_output_path_for_idea)  # output_path
    assert "Mock GPT-Engineer scaffolding successful." in result.output

    mock_opendevin.assert_called_once_with(str(expected_output_path_for_idea))
    assert "Mock OpenDevin debug loop successful" in result.output

    mock_deploy_fly.assert_called_once()
    # Example: assert call_args for mock_deploy_fly if app_name generation is complex
    # call_args_deploy = mock_deploy_fly.call_args[0]
    # assert call_args_deploy[0] == str(expected_output_path_for_idea)
    assert "Step 4: Deploying project" in result.output
    deploy_success_msg = (
        "Mock deployment successful. App URL: https://mock-app-test-build-idea.fly.dev"
    )
    assert deploy_success_msg in result.output

    mock_check_health.assert_called_once()
    # Example: assert app name passed to check_fly_io_health
    # app_name_fly_gen = f"app-{idea_id_slug[:30]}-{abs(hash(idea_id_to_test))%10000}"
    # assert mock_check_health.call_args[0][0] == app_name_fly_gen
    assert "Step 5: Checking health" in result.output
    assert "Mock health check successful." in result.output

    completed_msg = (
        f"Build cycle for Idea ID: {idea_id_to_test} "
        f"conceptually completed with deployment."
    )
    assert completed_msg in result.output


@mock.patch("scripts.run_build_cycle.run_gpt_engineer")
@mock.patch("scripts.run_build_cycle.GENERATED_MVPS_DIR", PATCHED_SCRIPT_MVPS_DIR)
def test_run_build_cycle_gpt_engineer_fails(mock_gpt_engineer):
    mock_gpt_engineer.return_value = False  # Simulate failure

    runner = CliRunner()
    result = runner.invoke(run_build_cycle, ["--idea-id", "fail-gpt"])

    assert result.exit_code is None or result.exit_code == 0
    assert "Mock GPT-Engineer scaffolding failed. Exiting build cycle." in result.output
    assert "Running mock OpenDevin debug loop" not in result.output


@mock.patch("scripts.run_build_cycle.run_gpt_engineer", return_value=True)
@mock.patch("scripts.run_build_cycle.run_opendevin_debug_loop")
@mock.patch("scripts.run_build_cycle.GENERATED_MVPS_DIR", PATCHED_SCRIPT_MVPS_DIR)
def test_run_build_cycle_opendevin_fails(mock_opendevin, mock_gpt_engineer_unused):
    mock_opendevin.return_value = False  # Simulate OpenDevin failure

    runner = CliRunner()
    result = runner.invoke(run_build_cycle, ["--idea-id", "fail-opendevin"])

    assert result.exit_code is None or result.exit_code == 0
    assert "Mock OpenDevin debug loop failed. Build cycle incomplete." in result.output
    # Conceptual completion message will now include "with deployment"
    completed_msg = (
        "Build cycle for Idea ID: fail-opendevin "
        "conceptually completed with deployment."
    )
    assert completed_msg in result.output


@mock.patch("scripts.run_build_cycle.shutil.rmtree")
@mock.patch("scripts.run_build_cycle.os.path.exists")
@mock.patch("scripts.run_build_cycle.os.makedirs")  # Mock makedirs as well
@mock.patch("scripts.run_build_cycle.GENERATED_MVPS_DIR", PATCHED_SCRIPT_MVPS_DIR)
def test_run_build_cycle_with_clean_output(
    mock_makedirs, mock_path_exists, mock_rmtree
):
    # --- Test case 1: Directory exists and is cleaned ---
    mock_path_exists.return_value = True  # Simulate output_path exists for cleaning

    runner = CliRunner()
    with (
        mock.patch("scripts.run_build_cycle.run_gpt_engineer", return_value=True),
        mock.patch(
            "scripts.run_build_cycle.run_opendevin_debug_loop", return_value=True
        ),
    ):
        result = runner.invoke(
            run_build_cycle, ["--idea-id", "clean-test-1", "--clean-output"]
        )

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    expected_clean_path = Path(PATCHED_SCRIPT_MVPS_DIR) / "clean-test-1"
    mock_rmtree.assert_called_once_with(str(expected_clean_path))
    clean_msg = f"Cleaning existing output directory: {str(expected_clean_path)}"
    assert clean_msg in result.output

    # Reset mocks for next scenario
    mock_path_exists.reset_mock()
    mock_rmtree.reset_mock()
    mock_makedirs.reset_mock()

    # --- Test case 2: Directory does not exist (clean should not trigger rmtree) ---
    mock_path_exists.return_value = False  # Simulate output_path does NOT exist

    with (
        mock.patch("scripts.run_build_cycle.run_gpt_engineer", return_value=True),
        mock.patch(
            "scripts.run_build_cycle.run_opendevin_debug_loop", return_value=True
        ),
    ):
        result = runner.invoke(
            run_build_cycle, ["--idea-id", "clean-test-2", "--clean-output"]
        )

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    mock_rmtree.assert_not_called()  # rmtree should not be called
    # Script logic: if not os.path.exists(GENERATED_MVPS_DIR): os.makedirs(...)
    # This test doesn't deeply verify os.makedirs calls on the base dir,
    # focuses on rmtree not being called.


def test_run_build_cycle_missing_idea_id():
    runner = CliRunner()
    result = runner.invoke(run_build_cycle, [])  # No idea-id
    assert result.exit_code != 0
    assert "Missing option '--idea-id'" in result.output


def test_generated_mvps_dir_env(monkeypatch):
    monkeypatch.setenv("GENERATED_MVPS_DIR", "env_mvp")
    module = importlib.reload(importlib.import_module("scripts.run_build_cycle"))
    assert module.GENERATED_MVPS_DIR.endswith("env_mvp")
