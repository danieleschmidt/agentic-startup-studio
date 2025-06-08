# core/bias_monitor.py
from typing import Dict, Any
import random  # For mock bias score


def check_text_for_bias(text: str, critical_threshold: float = 0.8) -> Dict[str, Any]:
    """
    Simulates a bias check on the provided text.
    In a real scenario, this would involve a more sophisticated NLP model or API.

    Args:
        text: The text content to analyze for bias.
        critical_threshold: The threshold (0.0 to 1.0) above which bias is
                            considered critical.

    Returns:
        A dictionary containing:
        - "bias_score": A mock bias score (float between 0.0 and 1.0).
        - "is_critical": Boolean, True if bias_score >= critical_threshold.
        - "details": A string with mock analysis details.
    """
    if not isinstance(text, str):
        # Or raise TypeError, but for a simple mock, returning error state is okay.
        return {
            "bias_score": -1.0,  # Indicate error
            "is_critical": False,
            "details": "Error: Input text must be a string.",
        }
    if not (0.0 <= critical_threshold <= 1.0):
        # Or raise ValueError
        return {
            "bias_score": -1.0,  # Indicate error
            "is_critical": False,
            "details": "Error: critical_threshold must be between 0.0 and 1.0.",
        }

    # Simulate bias detection - for now, a random score.
    # In a real system, this would be a call to an NLP model/service.
    mock_bias_score = random.uniform(0.0, 1.0)

    is_critical = mock_bias_score >= critical_threshold

    text_snippet = text[:50]  # Avoid overly long text in details
    details = (
        f"Mock bias analysis for text (first 50 chars): '{text_snippet}...'. "
        f"Assigned score: {mock_bias_score:.2f}. "
        f"Critical threshold: {critical_threshold:.2f}."
    )
    if is_critical:
        details += " BIAS DETECTED IS CONSIDERED CRITICAL."
    else:
        details += " Bias detected is not considered critical."

    print(
        f"Bias check performed. Score: {mock_bias_score:.2f}, Critical: {is_critical}"
    )

    return {
        "bias_score": round(mock_bias_score, 4),  # Store with a bit more precision
        "is_critical": is_critical,
        "details": details,
    }


if __name__ == "__main__":
    sample_text_neutral = "This is a fairly neutral statement about technology."
    sample_text_problematic = (
        "This statement might be flagged by a real bias checker for subtle issues."
    )

    print("\nTesting neutral text:")
    result_neutral = check_text_for_bias(sample_text_neutral, critical_threshold=0.8)
    print(result_neutral)

    print("\nTesting problematic text (mock will be random):")
    result_problematic = check_text_for_bias(
        sample_text_problematic, critical_threshold=0.5
    )
    print(result_problematic)

    print(
        "\nTesting with fixed score (by temporarily modifying random.uniform "
        "behavior if needed for specific demo):"
    )
    # To make this deterministic for __main__, one might temporarily
    # patch random.uniform. For now, it will be random.

    print("\nTesting invalid input type:")
    result_invalid_type = check_text_for_bias(12345)  # type: ignore
    print(result_invalid_type)

    print("\nTesting invalid threshold:")
    result_invalid_thresh_high = check_text_for_bias(sample_text_neutral, 1.5)
    print(result_invalid_thresh_high)
    result_invalid_thresh_low = check_text_for_bias(sample_text_neutral, -0.5)
    print(result_invalid_thresh_low)
