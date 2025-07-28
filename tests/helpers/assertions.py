"""Custom assertion helpers for testing."""

from typing import Any, Dict, List, Optional, Union
import json
import re
from datetime import datetime, timezone


def assert_valid_uuid(value: str) -> None:
    """Assert that a string is a valid UUID."""
    import uuid
    try:
        uuid.UUID(value)
    except ValueError:
        raise AssertionError(f"'{value}' is not a valid UUID")


def assert_valid_timestamp(value: Union[str, datetime]) -> None:
    """Assert that a value is a valid ISO timestamp."""
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError:
            raise AssertionError(f"'{value}' is not a valid ISO timestamp")
    elif isinstance(value, datetime):
        # Check if timezone-aware
        if value.tzinfo is None:
            raise AssertionError("Datetime must be timezone-aware")
    else:
        raise AssertionError(f"Expected string or datetime, got {type(value)}")


def assert_valid_url(value: str) -> None:
    """Assert that a string is a valid URL."""
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    if not url_pattern.match(value):
        raise AssertionError(f"'{value}' is not a valid URL")


def assert_api_response_structure(
    response: Dict[str, Any],
    required_fields: List[str],
    optional_fields: Optional[List[str]] = None
) -> None:
    """Assert that an API response has the expected structure."""
    optional_fields = optional_fields or []
    
    # Check required fields
    for field in required_fields:
        if field not in response:
            raise AssertionError(f"Required field '{field}' missing from response")
    
    # Check for unexpected fields
    expected_fields = set(required_fields + optional_fields)
    actual_fields = set(response.keys())
    unexpected_fields = actual_fields - expected_fields
    
    if unexpected_fields:
        raise AssertionError(f"Unexpected fields in response: {unexpected_fields}")


def assert_error_response(
    response: Dict[str, Any],
    expected_error_type: str,
    expected_status_code: Optional[int] = None
) -> None:
    """Assert that an error response has the expected format."""
    required_fields = ["error", "message", "timestamp"]
    assert_api_response_structure(response, required_fields, ["details", "code"])
    
    if response.get("error") != expected_error_type:
        raise AssertionError(
            f"Expected error type '{expected_error_type}', got '{response.get('error')}'"
        )
    
    if expected_status_code and response.get("code") != expected_status_code:
        raise AssertionError(
            f"Expected status code {expected_status_code}, got {response.get('code')}"
        )
    
    assert_valid_timestamp(response["timestamp"])


def assert_pagination_response(
    response: Dict[str, Any],
    expected_total: Optional[int] = None,
    expected_page_size: Optional[int] = None
) -> None:
    """Assert that a paginated response has the expected structure."""
    required_fields = ["data", "total", "page", "page_size", "total_pages"]
    assert_api_response_structure(response, required_fields)
    
    # Validate pagination fields
    assert isinstance(response["data"], list), "Data field must be a list"
    assert isinstance(response["total"], int), "Total must be an integer"
    assert isinstance(response["page"], int), "Page must be an integer"
    assert isinstance(response["page_size"], int), "Page size must be an integer"
    assert isinstance(response["total_pages"], int), "Total pages must be an integer"
    
    # Validate pagination logic
    if response["page"] < 1:
        raise AssertionError("Page number must be >= 1")
    
    if response["page_size"] < 1:
        raise AssertionError("Page size must be >= 1")
    
    if response["total"] < 0:
        raise AssertionError("Total count must be >= 0")
    
    # Check expected values
    if expected_total is not None and response["total"] != expected_total:
        raise AssertionError(f"Expected total {expected_total}, got {response['total']}")
    
    if expected_page_size is not None and response["page_size"] != expected_page_size:
        raise AssertionError(f"Expected page size {expected_page_size}, got {response['page_size']}")


def assert_idea_response(response: Dict[str, Any]) -> None:
    """Assert that an idea response has the expected structure."""
    required_fields = ["id", "title", "description", "category", "status", "created_at", "updated_at"]
    optional_fields = ["problem", "solution", "target_market", "evidence_urls", "embedding_vector"]
    
    assert_api_response_structure(response, required_fields, optional_fields)
    
    # Validate specific field types and formats
    assert_valid_uuid(response["id"])
    assert isinstance(response["title"], str) and len(response["title"]) > 0
    assert isinstance(response["description"], str) and len(response["description"]) > 0
    assert response["category"] in ["ai_ml", "fintech", "saas", "ecommerce", "healthcare"]
    assert response["status"] in ["DRAFT", "RESEARCHING", "VALIDATED", "BUILDING", "TESTING", "DEPLOYED"]
    assert_valid_timestamp(response["created_at"])
    assert_valid_timestamp(response["updated_at"])
    
    # Validate optional fields if present
    if "evidence_urls" in response:
        assert isinstance(response["evidence_urls"], list)
        for url in response["evidence_urls"]:
            assert_valid_url(url)


def assert_performance_metrics(
    response_time: float,
    max_response_time: float,
    memory_usage: Optional[float] = None,
    max_memory_usage: Optional[float] = None
) -> None:
    """Assert that performance metrics meet requirements."""
    if response_time > max_response_time:
        raise AssertionError(
            f"Response time {response_time:.3f}s exceeds maximum {max_response_time:.3f}s"
        )
    
    if memory_usage is not None and max_memory_usage is not None:
        if memory_usage > max_memory_usage:
            raise AssertionError(
                f"Memory usage {memory_usage:.2f}MB exceeds maximum {max_memory_usage:.2f}MB"
            )


def assert_security_headers(headers: Dict[str, str]) -> None:
    """Assert that response headers include security headers."""
    required_security_headers = [
        "X-Content-Type-Options",
        "X-Frame-Options", 
        "X-XSS-Protection",
        "Strict-Transport-Security",
    ]
    
    for header in required_security_headers:
        if header not in headers:
            raise AssertionError(f"Missing security header: {header}")


def assert_no_sensitive_data(response: Dict[str, Any]) -> None:
    """Assert that response doesn't contain sensitive data."""
    sensitive_patterns = [
        r'password',
        r'secret',
        r'token',
        r'api_key',
        r'private_key',
        r'auth',
    ]
    
    response_str = json.dumps(response, default=str).lower()
    
    for pattern in sensitive_patterns:
        if re.search(pattern, response_str):
            raise AssertionError(f"Response may contain sensitive data matching pattern: {pattern}")


def assert_database_consistency(
    before_data: Dict[str, Any],
    after_data: Dict[str, Any],
    changed_fields: Optional[List[str]] = None
) -> None:
    """Assert database consistency after operations."""
    changed_fields = changed_fields or []
    
    for field, value in before_data.items():
        if field not in changed_fields:
            if after_data.get(field) != value:
                raise AssertionError(
                    f"Unexpected change in field '{field}': {value} -> {after_data.get(field)}"
                )
        
    # Ensure updated_at is modified if any changes were made
    if changed_fields:
        before_updated = before_data.get("updated_at")
        after_updated = after_data.get("updated_at")
        
        if before_updated and after_updated and before_updated >= after_updated:
            raise AssertionError("updated_at field was not properly updated")