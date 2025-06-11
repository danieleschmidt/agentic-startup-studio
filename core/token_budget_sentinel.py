# core/token_budget_sentinel.py
from typing import Optional, List  # Callable removed if not used in constructor
from core.alert_manager import AlertManager  # Added import


class TokenBudgetSentinel:
    """
    Monitors token usage against a predefined budget and triggers alerts
    via an AlertManager.
    """

    def __init__(self, max_tokens: int, alert_manager: Optional[AlertManager] = None):
        """
        Initializes the sentinel with a maximum token budget and an AlertManager.

        Args:
            max_tokens: The maximum number of tokens allowed.
            alert_manager: An optional AlertManager instance to record alerts.
        """
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")
        self.max_tokens = max_tokens
        self.alert_manager = alert_manager
        self.alerts_triggered_messages_for_test: List[str] = []  # For direct testing

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
            # For direct testing of this class, store the message
            self.alerts_triggered_messages_for_test.append(alert_message)

            if self.alert_manager:
                try:
                    self.alert_manager.record_alert(
                        message=alert_message,
                        level="CRITICAL",  # More specific level
                        source=f"TokenBudgetSentinel({context})",
                    )
                except Exception as e:
                    # Fallback print if alert_manager itself fails
                    print(f"Error using AlertManager from TokenBudgetSentinel: {e}")
                    # Still print the core warning if AlertManager fails
                    print(f"Warning (TokenBudgetSentinel Fallback): {alert_message}")
            else:
                # Fallback if no alert_manager is provided
                print(f"Warning (TokenBudgetSentinel): {alert_message}")

            return False

        # Optional: Log normal usage
        # print(
        #    f"Token usage in '{context}' is within budget: "
        #    f"{current_tokens_used}/{self.max_tokens}"
        # )
        return True

    def get_internal_alerts_for_test(self) -> List[str]:  # Renamed
        """Returns alert messages triggered internally, for testing."""
        return self.alerts_triggered_messages_for_test

    def clear_internal_alerts_for_test(self) -> None:  # Renamed
        """Clears the internal list of triggered alert messages."""
        self.alerts_triggered_messages_for_test = []


# Remove or update __main__ as alert_callback is replaced by alert_manager
if __name__ == "__main__":
    print("TokenBudgetSentinel __main__ demonstration (now uses AlertManager).")
    print("This demo will print alerts to console via AlertManager's default print.")

    # Setup a dummy AlertManager for this demo
    # Ensure AlertManager is importable if running this file directly (e.g. PYTHONPATH)
    try:
        # Attempt to run the demo assuming AlertManager can be imported
        demo_alert_manager = AlertManager(log_file_path=None)  # Console only

        sentinel = TokenBudgetSentinel(
            max_tokens=1000, alert_manager=demo_alert_manager
        )

        print("\nTesting under budget:")
        sentinel.check_usage(500, "Demo Component A")
        print(f"Internal test alerts: {sentinel.get_internal_alerts_for_test()}")
        print(f"AlertManager logged alerts: {demo_alert_manager.get_logged_alerts()}")
        sentinel.clear_internal_alerts_for_test()
        demo_alert_manager.clear_logged_alerts()

        print("\nTesting over budget:")
        sentinel.check_usage(1200, "Demo Component B")
        print(f"Internal test alerts: {sentinel.get_internal_alerts_for_test()}")
        print(f"AlertManager logged alerts: {demo_alert_manager.get_logged_alerts()}")
        sentinel.clear_internal_alerts_for_test()
        demo_alert_manager.clear_logged_alerts()

        print("\nTesting with no AlertManager provided to TokenBudgetSentinel:")
        sentinel_no_am = TokenBudgetSentinel(max_tokens=100)
        sentinel_no_am.check_usage(150, "Demo Component D (no AM)")
        internal_alerts_no_am = sentinel_no_am.get_internal_alerts_for_test()
        print(f"Internal test alerts (no AM): {internal_alerts_no_am}")

    except ImportError:
        print(
            "Could not import AlertManager for __main__ demo. "
            "Ensure it's in PYTHONPATH."
        )
    except Exception as e:
        print(f"An error occurred during __main__ demo: {e}")

    # Note: Original __main__ had simple_alert_handler. Replaced by AlertManager.
    # AlertManager handles actual alert actions (e.g., print, log to file).
