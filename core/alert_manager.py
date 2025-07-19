# core/alert_manager.py
import datetime
import os
from typing import Optional, List, Dict, Any


class AlertManager:
    """
    Handles recording and logging of alerts from various system components.
    """

    def __init__(self, log_file_path: Optional[str] = "logs/alerts.log"):
        """
        Initializes the AlertManager.

        Args:
            log_file_path: Optional path for alert log file. If None, alerts
                           are only printed to console and stored in memory.
        """
        self.log_file_path = log_file_path
        self.logged_alerts: List[str] = []

        if self.log_file_path:
            try:
                # Ensure the directory for the log file exists
                log_dir = os.path.dirname(self.log_file_path)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                # Touch file to ensure it exists, or clear if specified (not done here)
                # For now, just ensuring directory exists is enough.
            except OSError as e:
                print(
                    f"Warning: Could not create log directory {log_dir}. "
                    f"Error: {e}. File logging will be disabled."
                )
                self.log_file_path = None  # Disable file logging

    def record_alert(
        self, message: str, level: str = "WARNING", source: Optional[str] = None, extra_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Records an alert: prints, logs to file (if configured), stores in memory.

        Args:
            message: The core alert message.
            level: Alert severity (e.g., "INFO", "WARNING", "CRITICAL").
            source: Optional identifier of the alert's source component.
        """
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        source_prefix = f"[{source}] " if source else ""
        formatted_alert = f"{timestamp} [{level.upper()}] {source_prefix}{message}"
        if extra_data:
            formatted_alert += f" | Data: {extra_data}"

        print(formatted_alert)  # Always print to console
        self.logged_alerts.append(formatted_alert)

        if self.log_file_path:
            try:
                with open(self.log_file_path, "a", encoding="utf-8") as f:
                    f.write(formatted_alert + "\n")
            except IOError as e:
                print(f"Warning: Could not write to log file {self.log_file_path}: {e}")

    def get_logged_alerts(self) -> List[str]:
        """Returns the list of alerts recorded in memory during this session."""
        return self.logged_alerts

    def clear_logged_alerts(self) -> None:
        """Clears the in-memory list of recorded alerts."""
        self.logged_alerts = []


if __name__ == "__main__":
    print("--- Testing AlertManager ---")

    # Test with file logging
    print("\nScenario 1: With file logging (to logs/demo_alerts.log)")
    demo_log_file = "logs/demo_alerts.log"
    # Clean up previous demo log if it exists
    if os.path.exists(demo_log_file):
        os.remove(demo_log_file)
    if os.path.exists("logs") and not os.listdir("logs"):  # Clean dir if empty
        os.rmdir("logs")

    manager_with_file = AlertManager(log_file_path=demo_log_file)

    manager_with_file.record_alert(
        "This is a warning.", level="WARNING", source="TestSystem", extra_data={"code": 101, "severity": "medium"}
    )
    manager_with_file.record_alert(
        "Critical issue detected!", level="CRITICAL", source="CoreModule", extra_data={"component": "database", "error_code": 500}
    )
    manager_with_file.record_alert("Just an FYI.", level="INFO", extra_data={"event": "startup"})

    print(f"In-memory alerts: {manager_with_file.get_logged_alerts()}")
    log_path = manager_with_file.log_file_path
    if log_path and os.path.exists(log_path):
        print(f"Check log file: {log_path}")
        with open(log_path, "r", encoding="utf-8") as f:
            print("Log file content:")
            print(f.read())
    else:
        print("Log file was not created or logging is disabled.")

    # Test without file logging
    print("\nScenario 2: Without file logging (log_file_path=None)")
    manager_no_file = AlertManager(log_file_path=None)
    manager_no_file.record_alert(
        "This alert goes to console and memory only.", source="NoFileTest"
    )
    print(f"In-memory alerts (no file): {manager_no_file.get_logged_alerts()}")

    # Test log directory creation failure (conceptual)
    print("\nScenario 3: Log directory creation failure (conceptual)")
    # To truly test this, make os.makedirs fail (e.g., permissions).
    # manager_bad_path = AlertManager(log_file_path="/proc/forbidden/alerts.log")
    # manager_bad_path.record_alert("This should not log to file.")
    print("If log dir creation fails, file logging is disabled (see warning).")

    print("\n--- AlertManager Test Finished ---")

    # Clean up logs directory if it was created by this demo and is empty
    if os.path.exists("logs") and not os.listdir("logs"):
        os.rmdir("logs")
    elif os.path.exists(demo_log_file):  # If only demo_alerts.log is there
        if len(os.listdir("logs")) == 1 and os.path.exists(demo_log_file):
            os.remove(demo_log_file)
            os.rmdir("logs")
