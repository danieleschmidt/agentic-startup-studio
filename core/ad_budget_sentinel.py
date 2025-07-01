# core/ad_budget_sentinel.py
from typing import Optional, Callable, List
from core.budget_sentinel_base import BaseBudgetSentinel


class AdBudgetSentinel(BaseBudgetSentinel):
    """
    Monitors advertising spend against a predefined budget,
    triggers alerts, and can invoke a halt callback.
    """

    def __init__(
        self,
        max_budget: float,
        campaign_id: str,  # Each sentinel instance monitors one campaign
        halt_callback: Optional[Callable[[str, str], None]] = None,
        alert_callback: Optional[Callable[[str], None]] = None,
    ):
        super().__init__(max_budget, alert_callback)
        if not campaign_id:
            raise ValueError("campaign_id cannot be empty.")

        self.campaign_id = campaign_id
        self.halt_callback = halt_callback
        self.halt_reason: Optional[str] = None

    def check_spend(self, current_spend: float) -> bool:
        """
        Checks the current ad spend against the budget.
        If exceeded, logs a warning, triggers halt and alert callbacks.

        Args:
            current_spend: The current amount spent for the campaign.

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
            self._trigger_alert(self.halt_reason)

            if self.halt_callback:
                try:
                    self.halt_callback(self.campaign_id, self.halt_reason)
                except Exception as e:
                    print(
                        f"Error in halt_callback for campaign {self.campaign_id}: {e}"
                    )
            return False

        return True

    def clear_alerts(self) -> None:
        """Clears the internal list of triggered alerts and halt reason."""
        super().clear_alerts()
        self.halt_reason = None
  # Also clear halt reason


if __name__ == "__main__":

    def simple_halt_handler(campaign_id: str, reason: str):
        print(f"SIMPLE HALT HANDLER: Campaign '{campaign_id}' halted. Reason: {reason}")

    def simple_alert_handler(message: str):
        print(f"SIMPLE ALERT HANDLER: {message}")

    sentinel = AdBudgetSentinel(
        max_budget=100.0,
        campaign_id="test-campaign-1",
        halt_callback=simple_halt_handler,
        alert_callback=simple_alert_handler,
    )

    print("\nTesting under budget:")
    sentinel.check_spend(50.0)
    print(
        f"Alerts: {sentinel.get_alerts()}, Halted: {sentinel.halt_reason is not None}"
    )
    sentinel.clear_alerts()

    print("\nTesting over budget:")
    sentinel.check_spend(120.0)
    print(
        f"Alerts: {sentinel.get_alerts()}, Halted: {sentinel.halt_reason is not None}"
    )
    if sentinel.halt_reason:
        print(f"Halt Reason: {sentinel.halt_reason}")
    sentinel.clear_alerts()

    print("\nTesting no callbacks:")
    sentinel_no_cb = AdBudgetSentinel(max_budget=50.0, campaign_id="test-campaign-2")
    sentinel_no_cb.check_spend(60.0)
    print(
        f"Alerts (no_cb): {sentinel_no_cb.get_alerts()}, "
        f"Halted: {sentinel_no_cb.halt_reason is not None}"
    )
    sentinel_no_cb.clear_alerts()

    print("\n--- AdBudgetSentinel Demo Finished ---")
