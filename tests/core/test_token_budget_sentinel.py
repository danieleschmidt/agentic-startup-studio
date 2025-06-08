# tests/core/test_token_budget_sentinel.py
import pytest
from unittest.mock import MagicMock, patch
from core.token_budget_sentinel import TokenBudgetSentinel
from core.alert_manager import AlertManager  # Import AlertManager

# CAMPAIGN_ID not used by TokenBudgetSentinel directly,
# but kept for consistency if other tests might use it.
CAMPAIGN_ID = "test_campaign_xyz"


@pytest.fixture
def mock_alert_manager() -> MagicMock:
    """Provides a MagicMock for AlertManager."""
    return MagicMock(spec=AlertManager)


def test_sentinel_init_valid(mock_alert_manager: MagicMock):
    sentinel = TokenBudgetSentinel(max_tokens=100, alert_manager=mock_alert_manager)
    assert sentinel.max_tokens == 100
    assert sentinel.alert_manager == mock_alert_manager

    sentinel_no_am = TokenBudgetSentinel(max_tokens=50)  # Test without alert_manager
    assert sentinel_no_am.alert_manager is None


def test_sentinel_init_invalid_max_tokens():
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        TokenBudgetSentinel(max_tokens=0)
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        TokenBudgetSentinel(max_tokens=-10)


def test_check_usage_within_budget(mock_alert_manager: MagicMock):
    sentinel = TokenBudgetSentinel(max_tokens=100, alert_manager=mock_alert_manager)

    result = sentinel.check_usage(50, "Test Context")
    assert result is True
    mock_alert_manager.record_alert.assert_not_called()
    assert len(sentinel.get_internal_alerts_for_test()) == 0


def test_check_usage_over_budget_with_alert_manager(mock_alert_manager: MagicMock):
    context = "Test Context Over"
    sentinel = TokenBudgetSentinel(max_tokens=100, alert_manager=mock_alert_manager)

    result = sentinel.check_usage(150, context)
    assert result is False

    # This is the core message content, without the "ALERT: " prefix.
    expected_halt_reason_content = (
        f"Token budget exceeded in context '{context}'. Usage: 150, Budget: 100."
    )
    # Message recorded by AlertManager/internally includes "ALERT: " (shortened comment)
    expected_alert_message_with_prefix = f"ALERT: {expected_halt_reason_content}"

    mock_alert_manager.record_alert.assert_called_once_with(
        message=expected_alert_message_with_prefix, # Check this one
        level="CRITICAL",
        source=f"TokenBudgetSentinel({context})",
    )
    # When AlertManager is used successfully, halt_reason might not be the primary
    # way to check the reason if the sentinel delegates alerting.
    # The important check is that AlertManager was called correctly.
    # If sentinel.halt_reason was set by the successful alert_manager path,
    # it would be: assert sentinel.halt_reason == expected_halt_reason_content
    # For now, we prioritize that the alert_manager got the correct message.

    internal_alerts = sentinel.get_internal_alerts_for_test()
    assert len(internal_alerts) == 1
    assert internal_alerts[0] == expected_alert_message_with_prefix


def test_check_usage_at_budget(mock_alert_manager: MagicMock):
    sentinel = TokenBudgetSentinel(max_tokens=100, alert_manager=mock_alert_manager)
    result = sentinel.check_usage(100, "Test Context At Budget")  # Exactly at budget
    assert result is True
    mock_alert_manager.record_alert.assert_not_called()
    assert len(sentinel.get_internal_alerts_for_test()) == 0


def test_check_usage_over_budget_no_alert_manager(capsys):
    sentinel = TokenBudgetSentinel(max_tokens=50)  # No alert_manager
    context = "No AM Context"
    result = sentinel.check_usage(70, context)
    assert result is False
    assert len(sentinel.get_internal_alerts_for_test()) == 1

    captured = capsys.readouterr()
    expected_warning_content = (
        f"Token budget exceeded in context '{context}'. "
        f"Usage: 70, Budget: 50."
    )
    # The fallback print in the sentinel prepends "ALERT: " to the message content
    expected_full_warning_in_log = (
        f"Warning (TokenBudgetSentinel): ALERT: {expected_warning_content}"
    )
    assert expected_full_warning_in_log in captured.out


def test_check_usage_invalid_current_tokens():
    sentinel = TokenBudgetSentinel(max_tokens=100)
    with pytest.raises(ValueError, match="current_tokens_used cannot be negative"):
        sentinel.check_usage(-10, "Invalid Usage")


def test_get_and_clear_internal_alerts():  # Renamed test to match method
    sentinel = TokenBudgetSentinel(max_tokens=10)
    sentinel.check_usage(20, "Alert 1")
    sentinel.check_usage(30, "Alert 2")

    alerts = sentinel.get_internal_alerts_for_test()
    assert len(alerts) == 2
    assert "Alert 1" in alerts[0]
    assert "Alert 2" in alerts[1]

    sentinel.clear_internal_alerts_for_test()
    assert len(sentinel.get_internal_alerts_for_test()) == 0


def test_alert_manager_call_exception_fallback_print(
    mock_alert_manager: MagicMock,
    capsys
):
    mock_alert_manager.record_alert.side_effect = Exception("AlertManager failed!")
    sentinel = TokenBudgetSentinel(max_tokens=100, alert_manager=mock_alert_manager)
    context = "Faulty AM Test" # Define context for specific message

    result = sentinel.check_usage(150, context)
    assert result is False
    assert len(sentinel.get_internal_alerts_for_test()) == 1

    captured = capsys.readouterr()
    expected_error_msg = (
        "Error using AlertManager from TokenBudgetSentinel: "
        "AlertManager failed!"
    )
    assert expected_error_msg in captured.out

    # The fallback log also uses the message that includes "ALERT:" and context
    expected_fallback_log_content = (
        f"Token budget exceeded in context '{context}'. Usage: 150, Budget: 100."
    )
    fallback_msg = (
        f"Warning (TokenBudgetSentinel Fallback): ALERT: {expected_fallback_log_content}"
    )
    assert fallback_msg in captured.out
