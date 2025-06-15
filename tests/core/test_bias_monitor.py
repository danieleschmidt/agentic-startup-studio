# tests/core/test_bias_monitor.py
import pytest

# random imported for patching target context if 'random.uniform' was used directly
# by the module under test, but here we patch 'core.bias_monitor.random.uniform'
import random
from unittest.mock import patch
from core.bias_monitor import check_text_for_bias


def test_check_text_for_bias_valid_input():
    text = "This is a test sentence."
    result = check_text_for_bias(text)

    assert isinstance(result, dict)
    assert "bias_score" in result
    assert isinstance(result["bias_score"], float)
    assert 0.0 <= result["bias_score"] <= 1.0
    assert "is_critical" in result
    assert isinstance(result["is_critical"], bool)
    assert "details" in result
    assert isinstance(result["details"], str)


@patch("core.bias_monitor.random.uniform")
def test_check_text_for_bias_critical_true(mock_random_uniform):
    mock_random_uniform.return_value = 0.85  # Ensure score is above threshold
    text = "This text will be critical."
    threshold = 0.8
    result = check_text_for_bias(text, critical_threshold=threshold)

    assert result["bias_score"] == 0.85
    assert result["is_critical"] is True
    assert "BIAS DETECTED IS CONSIDERED CRITICAL" in result["details"]
    assert f"Assigned score: {0.85:.2f}" in result["details"]
    assert f"Critical threshold: {threshold:.2f}" in result["details"]


@patch("core.bias_monitor.random.uniform")
def test_check_text_for_bias_critical_false(mock_random_uniform):
    mock_random_uniform.return_value = 0.75  # Ensure score is below threshold
    text = "This text will not be critical."
    threshold = 0.8
    result = check_text_for_bias(text, critical_threshold=threshold)

    assert result["bias_score"] == 0.75
    assert result["is_critical"] is False
    assert "Bias detected is not considered critical" in result["details"]
    assert f"Assigned score: {0.75:.2f}" in result["details"]
    assert f"Critical threshold: {threshold:.2f}" in result["details"]


def test_check_text_for_bias_invalid_text_type():
    result = check_text_for_bias(123)  # type: ignore
    assert result["bias_score"] == -1.0
    assert "Error: Input text must be a string." in result["details"]
    assert result["is_critical"] is False


def test_check_text_for_bias_invalid_threshold_too_high():
    result = check_text_for_bias("test text", 1.1)
    assert result["bias_score"] == -1.0
    assert "Error: critical_threshold must be between 0.0 and 1.0." in result["details"]
    assert result["is_critical"] is False


def test_check_text_for_bias_invalid_threshold_too_low():
    result = check_text_for_bias("test text", -0.1)
    assert result["bias_score"] == -1.0
    assert "Error: critical_threshold must be between 0.0 and 1.0." in result["details"]
    assert result["is_critical"] is False


@patch("core.bias_monitor.random.uniform")
def test_check_text_for_bias_threshold_edge_cases(mock_random_uniform):
    # Score exactly at threshold should be critical
    mock_random_uniform.return_value = 0.8
    result_at_thresh = check_text_for_bias("text at threshold", critical_threshold=0.8)
    assert result_at_thresh["is_critical"] is True
    assert result_at_thresh["bias_score"] == 0.8
    assert "BIAS DETECTED IS CONSIDERED CRITICAL" in result_at_thresh["details"]

    # Score just below threshold should not be critical
    mock_random_uniform.return_value = 0.7999
    result_below_thresh = check_text_for_bias(
        "text below threshold", critical_threshold=0.8
    )
    assert result_below_thresh["is_critical"] is False
    assert result_below_thresh["bias_score"] == 0.7999
    assert "Bias detected is not considered critical" in result_below_thresh["details"]


@patch("core.bias_monitor.random.uniform", return_value=0.5)  # Any value will do
def test_text_snippet_in_details(mock_random_uniform):
    short_text = "This is short."
    result_short = check_text_for_bias(short_text)
    # The current implementation in bias_monitor always adds '...'
    # Adjusting test to reflect that, or change implementation.
    # For now, assume implementation is fixed and test reflects it.
    assert f"'{short_text}...'" in result_short["details"]

    long_text = (
        "This is a very long text that definitely exceeds fifty "
        "characters to test snippet."
    )
    expected_snippet = long_text[:50]
    result_long = check_text_for_bias(long_text)
    assert f"'{expected_snippet}...'" in result_long["details"]
