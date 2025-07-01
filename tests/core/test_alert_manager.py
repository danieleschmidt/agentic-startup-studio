import pytest
import os
import datetime
from unittest.mock import patch, mock_open

from core.alert_manager import AlertManager

@pytest.fixture
def mock_log_file_path():
    return "/tmp/test_alerts.log"

@pytest.fixture
def alert_manager_with_file(mock_log_file_path):
    # Ensure the directory exists for the mock log file
    os.makedirs(os.path.dirname(mock_log_file_path), exist_ok=True)
    manager = AlertManager(log_file_path=mock_log_file_path)
    yield manager
    # Clean up after test
    if os.path.exists(mock_log_file_path):
        os.remove(mock_log_file_path)
    if os.path.exists(os.path.dirname(mock_log_file_path)) and not os.listdir(os.path.dirname(mock_log_file_path)):
        os.rmdir(os.path.dirname(mock_log_file_path))

@pytest.fixture
def alert_manager_no_file():
    return AlertManager(log_file_path=None)

def test_alert_manager_initialization_with_file(alert_manager_with_file, mock_log_file_path):
    assert alert_manager_with_file.log_file_path == mock_log_file_path
    assert alert_manager_with_file.logged_alerts == []
    assert os.path.exists(os.path.dirname(mock_log_file_path))

def test_alert_manager_initialization_no_file(alert_manager_no_file):
    assert alert_manager_no_file.log_file_path is None
    assert alert_manager_no_file.logged_alerts == []

def test_record_alert_console_output(alert_manager_no_file, capsys):
    message = "Test message"
    level = "INFO"
    source = "TestSource"
    extra_data = {"key": "value"}
    alert_manager_no_file.record_alert(message, level, source, extra_data=extra_data)
    captured = capsys.readouterr()
    assert f"[{level.upper()}] [{source}] {message} | Data: {{'key': 'value'}}" in captured.out
    assert len(alert_manager_no_file.get_logged_alerts()) == 1
    assert f"[{level.upper()}] [{source}] {message} | Data: {{'key': 'value'}}" in alert_manager_no_file.get_logged_alerts()[0]

def test_record_alert_file_logging(alert_manager_with_file, mock_log_file_path):
    message = "File log test"
    level = "WARNING"
    source = "FileSource"
    extra_data = {"file_key": "file_value"}
    alert_manager_with_file.record_alert(message, level, source, extra_data=extra_data)
    
    with open(mock_log_file_path, "r", encoding="utf-8") as f:
        content = f.read()
        assert f"[{level.upper()}] [{source}] {message} | Data: {{'file_key': 'file_value'}}" in content

def test_get_logged_alerts(alert_manager_no_file):
    alert_manager_no_file.record_alert("Alert 1")
    alert_manager_no_file.record_alert("Alert 2")
    alerts = alert_manager_no_file.get_logged_alerts()
    assert len(alerts) == 2
    assert "Alert 1" in alerts[0]
    assert "Alert 2" in alerts[1]

def test_clear_logged_alerts(alert_manager_no_file):
    alert_manager_no_file.record_alert("Alert to clear")
    assert len(alert_manager_no_file.get_logged_alerts()) == 1
    alert_manager_no_file.clear_logged_alerts()
    assert len(alert_manager_no_file.get_logged_alerts()) == 0

def test_record_alert_no_source(alert_manager_no_file, capsys):
    message = "No source test"
    level = "ERROR"
    extra_data = {"no_source_key": "no_source_value"}
    alert_manager_no_file.record_alert(message, level=level, extra_data=extra_data)
    captured = capsys.readouterr()
    assert f"[{level.upper()}] {message} | Data: {{'no_source_key': 'no_source_value'}}" in captured.out
    assert len(alert_manager_no_file.get_logged_alerts()) == 1
    assert f"[{level.upper()}] {message} | Data: {{'no_source_key': 'no_source_value'}}" in alert_manager_no_file.get_logged_alerts()[0]

def test_log_file_creation_failure(capsys):
    with patch('os.makedirs') as mock_makedirs:
        mock_makedirs.side_effect = OSError("Permission denied")
        manager = AlertManager(log_file_path="/forbidden/alerts.log")
        captured = capsys.readouterr()
        assert "Warning: Could not create log directory" in captured.out
        assert manager.log_file_path is None
        manager.record_alert("Should not log to file")
        assert len(manager.get_logged_alerts()) == 1