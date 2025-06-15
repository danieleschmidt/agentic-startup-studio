# tests/core/test_ad_budget_sentinel.py
import pytest
from unittest.mock import MagicMock
from core.ad_budget_sentinel import AdBudgetSentinel
from core.alert_manager import AlertManager  # Import AlertManager

CAMPAIGN_ID = "test_campaign_xyz"


@pytest.fixture
def mock_alert_manager() -> MagicMock:
    return MagicMock(spec=AlertManager)


def test_sentinel_init_valid(mock_alert_manager: MagicMock):  # Add fixture
    mock_halt_cb = MagicMock()
    sentinel = AdBudgetSentinel(
        max_budget=100.0,
        campaign_id=CAMPAIGN_ID,
        halt_callback=mock_halt_cb,
        alert_manager=mock_alert_manager,
    )
    assert sentinel.max_budget == 100.0
    assert sentinel.campaign_id == CAMPAIGN_ID
    assert sentinel.halt_callback == mock_halt_cb
    assert sentinel.alert_manager == mock_alert_manager

    sentinel_no_am_hc = AdBudgetSentinel(max_budget=50.0, campaign_id="no_am_hc")
    assert sentinel_no_am_hc.alert_manager is None
    assert sentinel_no_am_hc.halt_callback is None


def test_sentinel_init_invalid_budget_or_id():
    with pytest.raises(ValueError, match="max_budget must be positive"):
        AdBudgetSentinel(0, CAMPAIGN_ID)
    with pytest.raises(ValueError, match="max_budget must be positive"):
        AdBudgetSentinel(-100.0, CAMPAIGN_ID)
    with pytest.raises(ValueError, match="campaign_id cannot be empty"):
        AdBudgetSentinel(100.0, "")


def test_check_spend_within_budget(mock_alert_manager: MagicMock):
    mock_halt_cb = MagicMock()
    sentinel = AdBudgetSentinel(100.0, CAMPAIGN_ID, mock_halt_cb, mock_alert_manager)

    result = sentinel.check_spend(50.0)
    assert result is True
    mock_alert_manager.record_alert.assert_not_called()
    mock_halt_cb.assert_not_called()
    assert len(sentinel.get_internal_alerts_for_test()) == 0
    assert sentinel.halt_reason is None


def test_check_spend_over_budget_with_alert_manager_and_halt(
    mock_alert_manager: MagicMock,
):
    mock_halt_cb = MagicMock()
    sentinel = AdBudgetSentinel(100.0, CAMPAIGN_ID, mock_halt_cb, mock_alert_manager)

    result = sentinel.check_spend(150.0)
    assert result is False

    expected_halt_reason = (
        f"Ad budget exceeded for campaign '{CAMPAIGN_ID}'. "
        f"Spend: $150.00, Budget: $100.00."
    )

    mock_halt_cb.assert_called_once_with(CAMPAIGN_ID, expected_halt_reason)
    assert sentinel.halt_reason == expected_halt_reason

    mock_alert_manager.record_alert.assert_called_once_with(
        message=expected_halt_reason,  # The halt_reason is the message
        level="CRITICAL",
        source=f"AdBudgetSentinel({CAMPAIGN_ID})",
    )
    internal_alerts = sentinel.get_internal_alerts_for_test()
    assert len(internal_alerts) == 1
    # Corrected E501
    assert internal_alerts[0] == f"ALERT_INTERNAL: {expected_halt_reason}"


def test_check_spend_over_budget_no_alert_manager_fallback_print(
    capsys,
):  # Corrected E501
    mock_halt_cb = MagicMock()  # Can still have halt_cb
    # Corrected E501
    sentinel = AdBudgetSentinel(
        max_budget=50.0, campaign_id="no-am-campaign", halt_callback=mock_halt_cb
    )
    result = sentinel.check_spend(70.0)
    assert result is False
    assert len(sentinel.get_internal_alerts_for_test()) == 1
    mock_halt_cb.assert_called_once()  # Halt callback should still be called

    captured = capsys.readouterr()
    # Corrected F541 (removed f from f-strings without placeholders)
    # and E501
    expected_halt_reason = (
        "Ad budget exceeded for campaign 'no-am-campaign'. "
        "Spend: $70.00, Budget: $50.00."
    )
    expected_warning_msg = (
        f"Warning (AdBudgetSentinel - No AlertManager): {expected_halt_reason}"
    )
    assert expected_warning_msg in captured.out


def test_check_spend_at_budget(mock_alert_manager: MagicMock):
    mock_halt_cb = MagicMock()
    sentinel = AdBudgetSentinel(100.0, CAMPAIGN_ID, mock_halt_cb, mock_alert_manager)
    result = sentinel.check_spend(100.0)  # Exactly at budget
    assert result is True
    mock_alert_manager.record_alert.assert_not_called()
    mock_halt_cb.assert_not_called()
    assert len(sentinel.get_internal_alerts_for_test()) == 0
    assert sentinel.halt_reason is None


def test_check_spend_invalid_current_spend():
    sentinel = AdBudgetSentinel(100.0, CAMPAIGN_ID)
    with pytest.raises(ValueError, match="current_spend cannot be negative"):
        sentinel.check_spend(-50.0)


def test_get_and_clear_internal_alerts():  # Renamed
    sentinel = AdBudgetSentinel(max_budget=10.0, campaign_id=CAMPAIGN_ID)
    sentinel.check_spend(20.0)

    expected_halt_reason = (
        f"Ad budget exceeded for campaign '{CAMPAIGN_ID}'. "
        f"Spend: $20.00, Budget: $10.00."
    )
    alerts = sentinel.get_internal_alerts_for_test()
    assert len(alerts) == 1
    assert alerts[0] == f"ALERT_INTERNAL: {expected_halt_reason}"
    assert sentinel.halt_reason == expected_halt_reason

    sentinel.clear_internal_alerts_for_test()
    assert len(sentinel.get_internal_alerts_for_test()) == 0
    assert sentinel.halt_reason is None


def test_alert_manager_call_exception_fallback(
    mock_alert_manager: MagicMock, capsys
):  # Corrected E501
    mock_alert_manager.record_alert.side_effect = Exception("AM Crashed!")
    # Provide a mock_halt_cb to ensure its exception safety is also
    # implicitly part of this flow (Corrected E501)
    mock_halt_cb = MagicMock()
    # Corrected E501
    sentinel = AdBudgetSentinel(
        100.0, CAMPAIGN_ID, halt_callback=mock_halt_cb, alert_manager=mock_alert_manager
    )

    result = sentinel.check_spend(150.0)  # Over budget
    assert result is False

    expected_halt_reason = (
        f"Ad budget exceeded for campaign '{CAMPAIGN_ID}'. "
        f"Spend: $150.00, Budget: $100.00."
    )
    # Corrected E501 (Internal message still recorded)
    assert len(sentinel.get_internal_alerts_for_test()) == 1
    # Corrected E501
    internal_alerts = sentinel.get_internal_alerts_for_test()
    assert internal_alerts[0] == f"ALERT_INTERNAL: {expected_halt_reason}"

    # Halt callback should still be called even if AlertManager call fails
    mock_halt_cb.assert_called_once_with(CAMPAIGN_ID, expected_halt_reason)

    captured = capsys.readouterr()
    # Corrected E501
    error_msg_part1 = "CRITICAL_ERROR: AdBudgetSentinel failed to record alert "
    error_msg_part2 = f"via AlertManager for campaign {CAMPAIGN_ID}. Error: AM Crashed!"
    assert error_msg_part1 + error_msg_part2 in captured.out
    assert f"Fallback Log (AdBudgetSentinel): {expected_halt_reason}" in captured.out
