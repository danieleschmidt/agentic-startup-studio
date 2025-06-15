# core/ad_budget_sentinel.py
from typing import Optional, Callable, List
from core.alert_manager import AlertManager  # Added import


class AdBudgetSentinel:
    """
    Monitors advertising spend against a predefined budget,
    triggers alerts via an AlertManager, and can invoke a halt callback.
    """

    def __init__(
        self,
        max_budget: float,
        campaign_id: str,
        halt_callback: Optional[Callable[[str, str], None]] = None,
        alert_manager: Optional[AlertManager] = None,
    ):  # alert_callback changed
        """
        Initializes the sentinel.

        Args:
            max_budget: The maximum budget allowed for the campaign.
            campaign_id: Identifier for the ad campaign.
            halt_callback: Optional function to call if budget is exceeded.
                           It receives campaign_id (str) and reason (str).
            alert_manager: Optional AlertManager instance to record alerts.
        """
        if max_budget <= 0:
            raise ValueError("max_budget must be positive.")
        if not campaign_id:
            raise ValueError("campaign_id cannot be empty.")

        self.max_budget = max_budget
        self.campaign_id = campaign_id
        self.halt_callback = halt_callback
        self.alert_manager = alert_manager  # Store alert_manager
        self._internal_alerts_for_test: List[str] = []
        self.halt_reason: Optional[str] = None

    def check_spend(self, current_spend: float) -> bool:
        """
        Checks the current ad spend against the budget. If exceeded,
        it records an alert via AlertManager (if provided) or prints a warning,
        and calls the halt_callback if one is set.

        Args:
            current_spend: The current amount spent on the campaign.

        Returns:
            True if spend is within budget, False if budget is exceeded.
        """
        if current_spend < 0:
            raise ValueError("current_spend cannot be negative.")

        if current_spend > self.max_budget:
            self.halt_reason = (
                f"Ad budget exceeded for campaign '{self.campaign_id}'. "
                f"Spend: ${current_spend:.2f}, Budget: ${self.max_budget:.2f}."
            )
            # Store for direct testing of this class's triggering logic
            self._internal_alerts_for_test.append(f"ALERT_INTERNAL: {self.halt_reason}")

            # Use AlertManager if available
            if self.alert_manager:
                try:
                    self.alert_manager.record_alert(
                        message=self.halt_reason,
                        level="CRITICAL",
                        source=f"AdBudgetSentinel({self.campaign_id})",
                    )
                except Exception as e:
                    # Fallback print if alert_manager itself has an issue
                    err_msg = (
                        "CRITICAL_ERROR: AdBudgetSentinel failed to record alert "
                        f"via AlertManager for campaign {self.campaign_id}. Error: {e}"
                    )
                    print(err_msg)
                    print(f"Fallback Log (AdBudgetSentinel): {self.halt_reason}")
            else:
                # Fallback print if no alert_manager is configured
                warn_msg = (
                    f"Warning (AdBudgetSentinel - No AlertManager): {self.halt_reason}"
                )
                print(warn_msg)

            # Still call halt_callback if provided, regardless of AlertManager status
            if self.halt_callback:
                try:
                    self.halt_callback(self.campaign_id, self.halt_reason)
                except Exception as e:
                    cb_err_msg = (
                        f"Error in halt_callback for campaign {self.campaign_id}: {e}"
                    )
                    print(cb_err_msg)

            return False

        return True

    def get_internal_alerts_for_test(self) -> List[str]:
        """Returns a list of internal alert messages triggered for testing purposes."""
        return self._internal_alerts_for_test

    def clear_internal_alerts_for_test(self) -> None:
        """Clears the internal list of triggered alert messages and halt reason."""
        self._internal_alerts_for_test = []
        self.halt_reason = None


# Minimal __main__ for basic check, or remove entirely
if __name__ == "__main__":
    # Optional: A very simple check that the class can be instantiated
    try:
        # This AlertManager instance is temporary for this basic check.
        # In a real app, AlertManager might be configured elsewhere.
        # Using None to avoid creating logs/alerts.log during simple script run
        am = AlertManager(log_file_path=None)
        sentinel = AdBudgetSentinel(100.0, "main_test_campaign", alert_manager=am)
        main_check_msg = (
            "AdBudgetSentinel instantiated in __main__ for basic check: "
            f"{sentinel.campaign_id}"
        )
        print(main_check_msg)
        sentinel.check_spend(120.0)  # Trigger overspend to see alert mechanism
        internal_alerts_msg = (
            "Internal test alerts from __main__: "
            f"{sentinel.get_internal_alerts_for_test()}"
        )
        print(internal_alerts_msg)
        # AlertManager logs to console by default if log_file_path is None
        # or if there's an issue with file logging.
        am_logs_msg = (
            "AlertManager logs from __main__ (if any were captured by "
            f"this instance): {am.get_logged_alerts()}"
        )
        print(am_logs_msg)
        am.clear_logged_alerts()  # Clean up for this test
    except Exception as e:
        print(f"Error in AdBudgetSentinel __main__ basic check: {e}")
