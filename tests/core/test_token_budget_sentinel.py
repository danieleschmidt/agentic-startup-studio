# tests/core/test_token_budget_sentinel.py
import pytest
from unittest.mock import MagicMock
from core.token_budget_sentinel import TokenBudgetSentinel


def test_sentinel_init_valid():
    sentinel = TokenBudgetSentinel(max_tokens=100)
    assert sentinel.max_tokens == 100
    assert sentinel.alert_callback is None


def test_sentinel_init_invalid_max_tokens():
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        TokenBudgetSentinel(max_tokens=0)
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        TokenBudgetSentinel(max_tokens=-10)


def test_check_usage_within_budget():
    mock_callback = MagicMock()
    sentinel = TokenBudgetSentinel(max_tokens=100, alert_callback=mock_callback)

    result = sentinel.check_usage(50, "Test Context")
    assert result is True
    mock_callback.assert_not_called()
    assert len(sentinel.get_alerts()) == 0


def test_check_usage_over_budget():
    mock_callback = MagicMock()
    sentinel = TokenBudgetSentinel(max_tokens=100, alert_callback=mock_callback)

    result = sentinel.check_usage(150, "Test Context Over")
    assert result is False
    mock_callback.assert_called_once()
    alert_message = mock_callback.call_args[0][0]
    assert "Token budget exceeded" in alert_message
    assert "Test Context Over" in alert_message
    assert "Usage: 150" in alert_message
    assert "Budget: 100" in alert_message
    assert len(sentinel.get_alerts()) == 1
    assert sentinel.get_alerts()[0] == alert_message


def test_check_usage_at_budget():
    mock_callback = MagicMock()
    sentinel = TokenBudgetSentinel(max_tokens=100, alert_callback=mock_callback)

    # Exactly at budget is fine
    result = sentinel.check_usage(100, "Test Context At Budget")
    assert result is True
    mock_callback.assert_not_called()
    assert len(sentinel.get_alerts()) == 0


def test_check_usage_over_budget_no_callback():
    sentinel = TokenBudgetSentinel(max_tokens=50)  # No callback
    result = sentinel.check_usage(70, "No Callback Context")
    assert result is False
    assert len(sentinel.get_alerts()) == 1
    assert "Token budget exceeded" in sentinel.get_alerts()[0]


def test_check_usage_invalid_current_tokens():
    sentinel = TokenBudgetSentinel(max_tokens=100)
    with pytest.raises(ValueError, match="current_tokens_used cannot be negative"):
        sentinel.check_usage(-10, "Invalid Usage")


def test_get_and_clear_alerts():
    sentinel = TokenBudgetSentinel(max_tokens=10)
    sentinel.check_usage(20, "Alert 1")
    sentinel.check_usage(30, "Alert 2")

    alerts = sentinel.get_alerts()
    assert len(alerts) == 2
    assert "Alert 1" in alerts[0]
    assert "Alert 2" in alerts[1]

    sentinel.clear_alerts()
    assert len(sentinel.get_alerts()) == 0


def test_alert_callback_exception_handling(capsys):
    def faulty_callback(message: str):
        raise Exception("Callback failed!")

    sentinel = TokenBudgetSentinel(max_tokens=100, alert_callback=faulty_callback)
    result = sentinel.check_usage(150, "Faulty Callback Test")

    assert result is False  # Budget exceeded
    assert len(sentinel.get_alerts()) == 1  # Alert was still recorded internally

    captured = capsys.readouterr()
    # Check console print for error from callback
    assert "Error in alert_callback: Callback failed!" in captured.out
