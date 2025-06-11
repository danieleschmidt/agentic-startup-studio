# tests/core/test_ad_budget_sentinel.py
import pytest
from unittest.mock import MagicMock
from core.ad_budget_sentinel import AdBudgetSentinel

CAMPAIGN_ID = "test_campaign_xyz"


def test_sentinel_init_valid():
    sentinel = AdBudgetSentinel(max_budget=100.0, campaign_id=CAMPAIGN_ID)
    assert sentinel.max_budget == 100.0
    assert sentinel.campaign_id == CAMPAIGN_ID
    assert sentinel.alert_callback is None
    assert sentinel.halt_callback is None


def test_sentinel_init_invalid_budget_or_id():
    with pytest.raises(ValueError, match="max_budget must be positive"):
        AdBudgetSentinel(max_budget=0, campaign_id=CAMPAIGN_ID)
    with pytest.raises(ValueError, match="max_budget must be positive"):
        AdBudgetSentinel(max_budget=-10.0, campaign_id=CAMPAIGN_ID)
    with pytest.raises(ValueError, match="campaign_id cannot be empty"):
        AdBudgetSentinel(max_budget=100.0, campaign_id="")


def test_check_spend_within_budget():
    mock_alert_cb = MagicMock()
    mock_halt_cb = MagicMock()
    sentinel = AdBudgetSentinel(100.0, CAMPAIGN_ID, mock_halt_cb, mock_alert_cb)

    result = sentinel.check_spend(50.0)
    assert result is True
    mock_alert_cb.assert_not_called()
    mock_halt_cb.assert_not_called()
    assert len(sentinel.get_alerts()) == 0
    assert sentinel.halt_reason is None


def test_check_spend_over_budget():
    mock_alert_cb = MagicMock()
    mock_halt_cb = MagicMock()
    sentinel = AdBudgetSentinel(100.0, CAMPAIGN_ID, mock_halt_cb, mock_alert_cb)

    result = sentinel.check_spend(150.0)
    assert result is False

    mock_halt_cb.assert_called_once()
    halt_args = mock_halt_cb.call_args[0]
    assert halt_args[0] == CAMPAIGN_ID
    assert "Ad budget exceeded" in halt_args[1]
    assert "Spend: $150.00" in halt_args[1]
    assert "Budget: $100.00" in halt_args[1]
    assert sentinel.halt_reason == halt_args[1]

    mock_alert_cb.assert_called_once()
    alert_message = mock_alert_cb.call_args[0][0]
    assert f"ALERT: {sentinel.halt_reason}" == alert_message

    assert len(sentinel.get_alerts()) == 1
    assert sentinel.get_alerts()[0] == alert_message


def test_check_spend_at_budget():
    mock_alert_cb = MagicMock()
    mock_halt_cb = MagicMock()
    sentinel = AdBudgetSentinel(100.0, CAMPAIGN_ID, mock_halt_cb, mock_alert_cb)

    result = sentinel.check_spend(100.0)  # Exactly at budget
    assert result is True
    mock_alert_cb.assert_not_called()
    mock_halt_cb.assert_not_called()


def test_check_spend_over_budget_no_callbacks():
    sentinel = AdBudgetSentinel(max_budget=50.0, campaign_id="no-cb-campaign")
    result = sentinel.check_spend(70.0)
    assert result is False
    assert len(sentinel.get_alerts()) == 1
    assert "Ad budget exceeded" in sentinel.get_alerts()[0]
    assert sentinel.halt_reason is not None
    assert "Ad budget exceeded" in sentinel.halt_reason


def test_check_spend_invalid_current_spend():
    sentinel = AdBudgetSentinel(max_budget=100.0, campaign_id=CAMPAIGN_ID)
    with pytest.raises(ValueError, match="current_spend cannot be negative"):
        sentinel.check_spend(-10.0)


def test_get_and_clear_alerts():
    sentinel = AdBudgetSentinel(max_budget=10.0, campaign_id=CAMPAIGN_ID)
    sentinel.check_spend(20.0)  # Trigger alert

    alerts = sentinel.get_alerts()
    assert len(alerts) == 1
    assert "Ad budget exceeded" in alerts[0]
    assert sentinel.halt_reason is not None

    sentinel.clear_alerts()
    assert len(sentinel.get_alerts()) == 0
    assert sentinel.halt_reason is None  # Clear should reset halt_reason too


def test_callback_exception_handling(capsys):
    def faulty_halt_callback(campaign_id: str, reason: str):
        raise Exception("Halt callback failed!")

    def faulty_alert_callback(message: str):
        raise Exception("Alert callback failed!")

    sentinel = AdBudgetSentinel(
        max_budget=100.0,
        campaign_id=CAMPAIGN_ID,
        halt_callback=faulty_halt_callback,
        alert_callback=faulty_alert_callback,
    )
    result = sentinel.check_spend(150.0)

    assert result is False  # Budget exceeded
    assert len(sentinel.get_alerts()) == 1
    assert sentinel.halt_reason is not None

    captured = capsys.readouterr()
    assert "Error in halt_callback" in captured.out
    assert "Halt callback failed!" in captured.out
    assert "Error in alert_callback" in captured.out
    assert "Alert callback failed!" in captured.out
