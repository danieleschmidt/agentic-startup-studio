# tests/core/test_alert_manager.py
import pytest
import os
import datetime
from pathlib import Path
from core.alert_manager import AlertManager
from unittest.mock import patch, mock_open  # Import mock_open for file write simulation


def test_alert_manager_init_default_logfile():
    # Test default log file path setting (does not create file/dir here)
    # To prevent state leakage from other tests or __main__ block if run in same session
    default_log_path = "logs/alerts.log"
    default_log_dir = "logs"
    if os.path.exists(default_log_path):
        os.remove(default_log_path)
    if os.path.exists(default_log_dir) and not os.listdir(default_log_dir):
        os.rmdir(default_log_dir)

    manager = AlertManager()
    assert manager.log_file_path == default_log_path

    # Clean up again in case AlertManager created it (it might if path was None before)
    if os.path.exists(default_log_path):
        os.remove(default_log_path)
    if os.path.exists(default_log_dir) and not os.listdir(default_log_dir):
        os.rmdir(default_log_dir)


def test_alert_manager_init_no_logfile():
    manager = AlertManager(log_file_path=None)
    assert manager.log_file_path is None


def test_record_alert_console_output(capsys):
    manager = AlertManager(log_file_path=None)  # No file logging for this test

    # Mock datetime to control timestamp for assertion
    fixed_datetime = datetime.datetime(
        2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc
    )
    expected_timestamp = fixed_datetime.isoformat()

    with patch("core.alert_manager.datetime.datetime") as mock_dt:
        mock_dt.now.return_value = fixed_datetime
        # Ensure utc is available on the mocked datetime object
        mock_dt.timezone.utc = datetime.timezone.utc

        manager.record_alert("Test console alert", level="INFO", source="ConsoleTest")

    captured = capsys.readouterr()
    expected_output = f"{expected_timestamp} [INFO] [ConsoleTest] Test console alert"
    assert expected_output in captured.out


def test_record_alert_in_memory_list():
    manager = AlertManager(log_file_path=None)

    fixed_dt_1 = datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
    fixed_dt_2 = datetime.datetime(2024, 1, 1, 12, 0, 5, tzinfo=datetime.timezone.utc)

    with patch("core.alert_manager.datetime.datetime") as mock_dt:
        mock_dt.now.side_effect = [fixed_dt_1, fixed_dt_2]
        mock_dt.timezone.utc = datetime.timezone.utc

        manager.record_alert("Memory Test 1", level="DEBUG", source="MemTest")
        manager.record_alert("Memory Test 2")  # Default level WARNING, no source

    alerts = manager.get_logged_alerts()
    assert len(alerts) == 2
    assert f"{fixed_dt_1.isoformat()} [DEBUG] [MemTest] Memory Test 1" == alerts[0]
    assert f"{fixed_dt_2.isoformat()} [WARNING] Memory Test 2" == alerts[1]

    manager.clear_logged_alerts()
    assert len(manager.get_logged_alerts()) == 0


def test_record_alert_file_logging(tmp_path: Path):
    log_file = tmp_path / "test_alerts.log"
    manager = AlertManager(log_file_path=str(log_file))

    fixed_dt_1 = datetime.datetime(2024, 1, 1, 12, 1, 0, tzinfo=datetime.timezone.utc)
    fixed_dt_2 = datetime.datetime(2024, 1, 1, 12, 1, 5, tzinfo=datetime.timezone.utc)

    with patch("core.alert_manager.datetime.datetime") as mock_dt:
        mock_dt.now.side_effect = [fixed_dt_1, fixed_dt_2]
        mock_dt.timezone.utc = datetime.timezone.utc

        manager.record_alert("File log test 1", level="ERROR", source="FileTest")
        manager.record_alert("File log test 2", level="INFO")

    assert log_file.exists()
    content = log_file.read_text()

    assert f"{fixed_dt_1.isoformat()} [ERROR] [FileTest] File log test 1" in content
    assert f"{fixed_dt_2.isoformat()} [INFO] File log test 2" in content

    # Also check in-memory list
    assert len(manager.get_logged_alerts()) == 2


def test_log_directory_creation(tmp_path: Path):
    log_dir = tmp_path / "new_log_dir"
    log_file = log_dir / "alerts.log"
    # log_dir does not exist yet

    manager = AlertManager(log_file_path=str(log_file))
    assert log_dir.exists()  # AlertManager should create it
    manager.record_alert("Test dir creation")  # This will attempt to write to the file
    assert log_file.exists()  # File should be created after first record_alert


@patch("core.alert_manager.os.makedirs")
def test_log_directory_creation_failure_disables_file_logging(mock_makedirs, capsys):
    mock_makedirs.side_effect = OSError("Permission denied")

    # Path that would require directory creation
    manager = AlertManager(log_file_path="non_existent_dir/alerts.log")

    assert manager.log_file_path is None  # File logging should be disabled
    mock_makedirs.assert_called_once()  # Attempted to create dir

    captured = capsys.readouterr()
    assert "Warning: Could not create log directory non_existent_dir" in captured.out
    assert "Permission denied" in captured.out
    assert "File logging will be disabled" in captured.out

    # Ensure record_alert doesn't try to write to file if path is None
    manager.record_alert("This should not cause file write error")
    # No IOError should be raised or printed from file writing attempt by this call


# Mock open to simulate IOError on file write
@patch("builtins.open", new_callable=mock_open)
def test_file_logging_io_error_on_write(mock_file_open, capsys):
    mock_file_open.side_effect = IOError("Disk full")

    # For this test, we assume the directory exists, but writing fails.
    # The AlertManager's __init__ might still try to create "logs" if it doesn't exist.
    # To isolate, ensure "logs" dir exists or use a path that doesn't trigger makedirs.
    test_log_path = "logs/problematic_alerts.log"
    if not os.path.exists("logs"):
        os.makedirs("logs")  # Ensure "logs" dir exists for this specific test scenario

    manager = AlertManager(log_file_path=test_log_path)
    manager.record_alert("Trying to write to problematic file", source="IOErrorTest")

    captured = capsys.readouterr()
    assert f"Warning: Could not write to log file {test_log_path}" in captured.out
    assert "Disk full" in captured.out
    assert len(manager.get_logged_alerts()) == 1  # Still logged in memory and console

    # Clean up the "logs" directory if it was created just for this test
    if os.path.exists(test_log_path):
        os.remove(test_log_path)
    if os.path.exists("logs") and not os.listdir("logs"):
        os.rmdir("logs")


def test_alert_formatting_timestamp_and_level():
    manager = AlertManager(log_file_path=None)

    fixed_datetime = datetime.datetime(
        2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc
    )
    with patch("core.alert_manager.datetime.datetime") as mock_dt:
        mock_dt.now.return_value = fixed_datetime
        mock_dt.timezone.utc = datetime.timezone.utc  # Ensure utc is available on mock

        manager.record_alert("Timestamp test", "MYLEVEL", "MySource")

    expected_timestamp = fixed_datetime.isoformat()
    expected_alert = f"{expected_timestamp} [MYLEVEL] [MySource] Timestamp test"

    logged_alerts = manager.get_logged_alerts()
    assert len(logged_alerts) == 1
    assert logged_alerts[0] == expected_alert
