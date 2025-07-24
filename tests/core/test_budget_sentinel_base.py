"""
Unit tests for BaseBudgetSentinel
Tests the base functionality for budget monitoring sentinels.
"""

import pytest
from unittest.mock import Mock

from core.budget_sentinel_base import BaseBudgetSentinel


class TestBaseBudgetSentinel:
    """Test suite for BaseBudgetSentinel class."""

    def test_initialization_with_valid_budget(self):
        """Test successful initialization with valid parameters."""
        callback = Mock()
        sentinel = BaseBudgetSentinel(max_budget=100.0, alert_callback=callback)
        
        assert sentinel.max_budget == 100.0
        assert sentinel.alert_callback == callback
        assert sentinel.alerts_triggered == []

    def test_initialization_with_zero_budget_raises_error(self):
        """Test that zero budget raises ValueError."""
        with pytest.raises(ValueError, match="max_budget must be positive"):
            BaseBudgetSentinel(max_budget=0.0)

    def test_initialization_with_negative_budget_raises_error(self):
        """Test that negative budget raises ValueError."""
        with pytest.raises(ValueError, match="max_budget must be positive"):
            BaseBudgetSentinel(max_budget=-50.0)

    def test_initialization_without_callback(self):
        """Test initialization without alert callback."""
        sentinel = BaseBudgetSentinel(max_budget=100.0)
        
        assert sentinel.max_budget == 100.0
        assert sentinel.alert_callback is None
        assert sentinel.alerts_triggered == []

    def test_trigger_alert_basic_message(self):
        """Test triggering alert with basic message."""
        callback = Mock()
        sentinel = BaseBudgetSentinel(max_budget=100.0, alert_callback=callback)
        
        sentinel._trigger_alert("Budget exceeded")
        
        assert len(sentinel.alerts_triggered) == 1
        assert "ALERT: Budget exceeded" in sentinel.alerts_triggered[0]
        callback.assert_called_once_with("ALERT: Budget exceeded")

    def test_trigger_alert_with_context(self):
        """Test triggering alert with context."""
        callback = Mock()
        sentinel = BaseBudgetSentinel(max_budget=100.0, alert_callback=callback)
        
        sentinel._trigger_alert("Budget exceeded", context="campaign-123")
        
        assert len(sentinel.alerts_triggered) == 1
        expected_message = "ALERT: Budget exceeded in context 'campaign-123'."
        assert expected_message in sentinel.alerts_triggered[0]
        callback.assert_called_once_with(expected_message)

    def test_trigger_alert_without_callback(self):
        """Test triggering alert without callback."""
        sentinel = BaseBudgetSentinel(max_budget=100.0)
        
        # Should not raise exception even without callback
        sentinel._trigger_alert("Budget exceeded")
        
        assert len(sentinel.alerts_triggered) == 1
        assert "ALERT: Budget exceeded" in sentinel.alerts_triggered[0]

    def test_trigger_alert_callback_exception_handling(self):
        """Test that callback exceptions are handled gracefully."""
        callback = Mock(side_effect=Exception("Callback failed"))
        sentinel = BaseBudgetSentinel(max_budget=100.0, alert_callback=callback)
        
        # Should not raise exception even if callback fails
        sentinel._trigger_alert("Budget exceeded")
        
        assert len(sentinel.alerts_triggered) == 1
        callback.assert_called_once()

    def test_multiple_alerts_accumulate(self):
        """Test that multiple alerts are accumulated."""
        callback = Mock()
        sentinel = BaseBudgetSentinel(max_budget=100.0, alert_callback=callback)
        
        sentinel._trigger_alert("First alert")
        sentinel._trigger_alert("Second alert")
        sentinel._trigger_alert("Third alert")
        
        assert len(sentinel.alerts_triggered) == 3
        assert callback.call_count == 3

    def test_get_alerts_returns_list(self):
        """Test get_alerts returns current alerts."""
        sentinel = BaseBudgetSentinel(max_budget=100.0)
        
        assert sentinel.get_alerts() == []
        
        sentinel._trigger_alert("Test alert")
        alerts = sentinel.get_alerts()
        
        assert len(alerts) == 1
        assert "ALERT: Test alert" in alerts[0]

    def test_clear_alerts_empties_list(self):
        """Test clear_alerts empties the alerts list."""
        sentinel = BaseBudgetSentinel(max_budget=100.0)
        
        sentinel._trigger_alert("Alert 1")
        sentinel._trigger_alert("Alert 2")
        assert len(sentinel.get_alerts()) == 2
        
        sentinel.clear_alerts()
        assert sentinel.get_alerts() == []

    def test_clear_alerts_on_empty_list(self):
        """Test clear_alerts on empty list doesn't raise error."""
        sentinel = BaseBudgetSentinel(max_budget=100.0)
        
        # Should not raise exception
        sentinel.clear_alerts()
        assert sentinel.get_alerts() == []

    def test_alert_message_format_consistency(self):
        """Test that alert message format is consistent."""
        sentinel = BaseBudgetSentinel(max_budget=100.0)
        
        test_cases = [
            ("Simple message", None, "ALERT: Simple message"),
            ("Message with context", "test-context", "ALERT: Message with context in context 'test-context'.")
        ]
        
        for message, context, expected_prefix in test_cases:
            sentinel.clear_alerts()
            if context:
                sentinel._trigger_alert(message, context=context)
            else:
                sentinel._trigger_alert(message)
            
            alerts = sentinel.get_alerts()
            assert len(alerts) == 1
            assert expected_prefix in alerts[0]