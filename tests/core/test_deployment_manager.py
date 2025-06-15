# tests/core/test_deployment_manager.py
import pytest
from core.deployment_manager import deploy_to_fly_io, check_fly_io_health
import re  # For testing slugification
from unittest import mock  # Moved import to top


def test_deploy_to_fly_io_success():
    project_path = "/path/to/project"
    app_name = "MyTestApp"
    expected_slug = "mytestapp"  # Based on current slugify logic
    url = deploy_to_fly_io(project_path, app_name)
    assert url == f"https://{expected_slug}.fly.dev"


def test_deploy_to_fly_io_with_special_chars_in_name():
    project_path = "/path/to/project"
    app_name = "My Test App!@#123"
    # Slugify logic: re.sub(r'[^a-z0-9-]+', '-', app_name.lower()).strip('-')
    expected_slug = "my-test-app-123"
    url = deploy_to_fly_io(project_path, app_name)
    assert url == f"https://{expected_slug}.fly.dev"


def test_deploy_to_fly_io_with_leading_trailing_hyphens_in_name():
    project_path = "/path/to/project"
    app_name = "-My-Test-App-"
    expected_slug = "my-test-app"  # strip('-') should handle these
    url = deploy_to_fly_io(project_path, app_name)
    assert url == f"https://{expected_slug}.fly.dev"


def test_deploy_to_fly_io_name_with_only_special_chars_generates_fallback():
    project_path = "/path/to/project"
    app_name = "!@#$%^"
    url = deploy_to_fly_io(project_path, app_name)
    # Expects fallback like "app-1234" where 1234 is from hash
    assert ".fly.dev" in url
    assert url.startswith("https://app-")
    # Validate that the part after "app-" is numeric (from hash % 10000)
    match = re.search(r"https://app-(\d+)\.fly\.dev", url)
    assert match is not None
    assert match.group(1).isdigit()


def test_deploy_to_fly_io_empty_inputs():
    assert deploy_to_fly_io("", "app") == ""
    assert deploy_to_fly_io("path", "") == ""
    assert deploy_to_fly_io("", "") == ""


def test_check_fly_io_health_success():
    app_name = "MyTestApp"
    assert check_fly_io_health(app_name) is True


def test_check_fly_io_health_empty_app_name():
    assert check_fly_io_health("") is False


def test_check_fly_io_health_simulates_delay(monkeypatch):
    # Simple test to ensure time.sleep was called, indicating a simulated delay
    mock_sleep = mock.Mock()
    monkeypatch.setattr("time.sleep", mock_sleep)

    check_fly_io_health("some-app")
    mock_sleep.assert_called_once_with(0.5)


def test_deploy_to_fly_io_simulates_delay(monkeypatch):
    # Simple test to ensure time.sleep was called
    mock_sleep = mock.Mock()
    monkeypatch.setattr("time.sleep", mock_sleep)

    deploy_to_fly_io("/some/path", "some-app")
    mock_sleep.assert_called_once_with(1)
