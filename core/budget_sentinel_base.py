from typing import Optional, Callable, List

class BaseBudgetSentinel:
    """
    Base class for budget monitoring sentinels.
    Provides common functionality for checking budget, triggering alerts,
    and managing alert messages.
    """

    def __init__(
        self,
        max_budget: float,
        alert_callback: Optional[Callable[[str], None]] = None,
    ):
        if max_budget <= 0:
            raise ValueError("max_budget must be positive.")
        self.max_budget = max_budget
        self.alert_callback = alert_callback
        self.alerts_triggered: List[str] = []

    def _trigger_alert(self, message: str, context: str = "") -> None:
        full_message = f"ALERT: {message}"
        if context:
            full_message = f"ALERT: {message} in context '{context}'."

        print(f"Warning: {full_message}")  # Default logging
        self.alerts_triggered.append(full_message)

        if self.alert_callback:
            try:
                self.alert_callback(full_message)
            except Exception as e:
                print(f"Error in alert_callback: {e}")

    def get_alerts(self) -> List[str]:
        """Returns a list of alerts triggered."""
        return self.alerts_triggered

    def clear_alerts(self) -> None:
        """Clears the internal list of triggered alerts."""
        self.alerts_triggered = []
