# core/token_budget_sentinel.py
from typing import Optional, Callable, List


class TokenBudgetSentinel:
    """
    Monitors token usage against a predefined budget and triggers alerts.
    """

    def __init__(
        self, max_tokens: int, alert_callback: Optional[Callable[[str], None]] = None
    ):
        """
        Initializes the sentinel with a max token budget and an optional alert callback.

        Args:
            max_tokens: The maximum number of tokens allowed.
            alert_callback: Optional function called when budget is exceeded.
                            Accepts a single string argument (alert message).
        """
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")
        self.max_tokens = max_tokens
        self.alert_callback = alert_callback
        self.alerts_triggered: List[str] = []  # To store triggered alert messages

    def check_usage(
        self, current_tokens_used: int, context: str = "General Usage"
    ) -> bool:
        """
        Checks the current token usage against the budget.
        If exceeded, logs a warning and triggers the alert callback.

        Args:
            current_tokens_used: The number of tokens used so far.
            context: A string describing the context of the token usage
                     (e.g., "Pitch Loop Generation").

        Returns:
            True if usage is within budget, False if budget is exceeded.
        """
        if current_tokens_used < 0:
            raise ValueError("current_tokens_used cannot be negative.")

        if current_tokens_used > self.max_tokens:
            alert_message = (
                f"ALERT: Token budget exceeded in context '{context}'. "
                f"Usage: {current_tokens_used}, Budget: {self.max_tokens}."
            )
            print(f"Warning: {alert_message}")  # Default logging
            self.alerts_triggered.append(alert_message)
            if self.alert_callback:
                try:
                    self.alert_callback(alert_message)
                except Exception as e:
                    # Print error from callback but don't let it crash the sentinel
                    print(f"Error in alert_callback: {e}")
            return False

        # Optional: Log normal usage
        # print(
        #    f"Token usage in '{context}' is within budget: "
        #    f"{current_tokens_used}/{self.max_tokens}"
        # )
        return True

    def get_alerts(self) -> List[str]:
        """Returns a list of alerts triggered during check_usage calls."""
        return self.alerts_triggered

    def clear_alerts(self) -> None:
        """Clears the internal list of triggered alerts."""
        self.alerts_triggered = []


if __name__ == "__main__":

    def simple_alert_handler(message: str):
        print(f"SIMPLE ALERT HANDLER RECEIVED: {message}")

    # Example usage
    sentinel = TokenBudgetSentinel(max_tokens=1000, alert_callback=simple_alert_handler)

    print("\nTesting under budget:")
    sentinel.check_usage(500, "Component A")
    print(f"Alerts: {sentinel.get_alerts()}")
    sentinel.clear_alerts()

    print("\nTesting over budget:")
    sentinel.check_usage(1200, "Component B")
    print(f"Alerts: {sentinel.get_alerts()}")
    sentinel.clear_alerts()

    print("\nTesting exactly at budget (should be fine):")
    sentinel.check_usage(1000, "Component C")  # Should be fine
    print(f"Alerts: {sentinel.get_alerts()}")
    sentinel.clear_alerts()

    print("\nTesting with no callback:")
    sentinel_no_cb = TokenBudgetSentinel(max_tokens=100)
    sentinel_no_cb.check_usage(150, "Component D")
    print(f"Alerts (no_cb): {sentinel_no_cb.get_alerts()}")
    sentinel_no_cb.clear_alerts()

    try:
        print("\nTesting invalid max_tokens:")
        TokenBudgetSentinel(max_tokens=0)
    except ValueError as e:
        print(f"Caught expected error: {e}")

    try:
        print("\nTesting invalid current_tokens_used:")
        sentinel.check_usage(-10, "Invalid Usage")
    except ValueError as e:
        print(f"Caught expected error: {e}")
