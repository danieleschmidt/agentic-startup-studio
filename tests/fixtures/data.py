"""Test data fixtures for Agentic Startup Studio."""

from typing import Dict, Any, List
import pytest
from faker import Faker
from datetime import datetime, timezone
import uuid

fake = Faker()


@pytest.fixture
def sample_idea_data() -> Dict[str, Any]:
    """Generate sample idea data for testing."""
    return {
        "id": str(uuid.uuid4()),
        "title": fake.catch_phrase(),
        "description": fake.text(max_nb_chars=500),
        "category": fake.random_element(elements=("ai_ml", "fintech", "saas", "ecommerce", "healthcare")),
        "problem": fake.text(max_nb_chars=200),
        "solution": fake.text(max_nb_chars=200),
        "target_market": fake.text(max_nb_chars=150),
        "evidence_urls": [fake.url() for _ in range(3)],
        "status": "DRAFT",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }


@pytest.fixture
def multiple_idea_data() -> List[Dict[str, Any]]:
    """Generate multiple sample ideas for testing."""
    return [
        {
            "id": str(uuid.uuid4()),
            "title": f"AI-Powered {fake.word().title()} Platform",
            "description": fake.text(max_nb_chars=400),
            "category": "ai_ml",
            "problem": "Traditional processes are slow and inefficient",
            "solution": f"Use AI to automate {fake.word()} processes",
            "target_market": f"{fake.company_suffix()} companies",
            "evidence_urls": [fake.url() for _ in range(2)],
            "status": "RESEARCHING",
        },
        {
            "id": str(uuid.uuid4()),
            "title": f"Fintech {fake.word().title()} Solution",
            "description": fake.text(max_nb_chars=350),
            "category": "fintech",
            "problem": "Financial services lack accessibility",
            "solution": f"Mobile-first {fake.word()} platform",
            "target_market": "Small businesses and freelancers",
            "evidence_urls": [fake.url() for _ in range(3)],
            "status": "VALIDATED",
        },
        {
            "id": str(uuid.uuid4()),
            "title": f"SaaS {fake.word().title()} Tool",
            "description": fake.text(max_nb_chars=450),
            "category": "saas",
            "problem": "Teams struggle with collaboration",
            "solution": f"Cloud-based {fake.word()} workspace",
            "target_market": "Remote teams and distributed companies",
            "evidence_urls": [fake.url()],
            "status": "BUILDING",
        },
    ]


@pytest.fixture
def api_test_data() -> Dict[str, Any]:
    """Generate test data for API testing."""
    return {
        "valid_idea": {
            "title": "AI-Powered Code Review Assistant",
            "description": "Automated code review tool that provides intelligent feedback on pull requests using machine learning algorithms.",
            "category": "ai_ml",
            "problem": "Code reviews are time-consuming and inconsistent across development teams.",
            "solution": "AI-powered analysis that provides instant, consistent, and intelligent code review feedback.",
            "target_market": "Software development teams and organizations using Git-based workflows.",
            "evidence_urls": ["https://example.com/research1", "https://example.com/market-analysis"],
        },
        "invalid_idea": {
            "title": "",  # Invalid: empty title
            "description": "A" * 2000,  # Invalid: too long
            "category": "invalid_category",  # Invalid: not in allowed categories
            # Missing required fields
        },
        "minimal_idea": {
            "title": "Minimal Viable Idea",
            "description": "A basic idea with minimal required fields for testing.",
            "category": "saas",
        },
    }


@pytest.fixture
def performance_test_data() -> Dict[str, Any]:
    """Generate data for performance testing."""
    return {
        "large_idea_set": [
            {
                "title": f"Idea {i}: {fake.catch_phrase()}",
                "description": fake.text(max_nb_chars=400),
                "category": fake.random_element(elements=("ai_ml", "fintech", "saas", "ecommerce")),
                "problem": fake.text(max_nb_chars=150),
                "solution": fake.text(max_nb_chars=150),
                "target_market": fake.text(max_nb_chars=100),
            }
            for i in range(100)
        ],
        "heavy_load_idea": {
            "title": "Complex Idea with Large Data",
            "description": fake.text(max_nb_chars=1500),
            "category": "ai_ml",
            "problem": fake.text(max_nb_chars=500),
            "solution": fake.text(max_nb_chars=500),
            "target_market": fake.text(max_nb_chars=300),
            "evidence_urls": [fake.url() for _ in range(10)],
        },
    }


@pytest.fixture
def security_test_data() -> Dict[str, Any]:
    """Generate data for security testing."""
    return {
        "sql_injection_attempts": [
            "'; DROP TABLE ideas; --",
            "1' OR '1'='1",
            "UNION SELECT * FROM users --",
            "<script>alert('xss')</script>",
        ],
        "xss_attempts": [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
        ],
        "oversized_inputs": {
            "title": "A" * 1000,
            "description": "B" * 10000,
            "category": "C" * 100,
        },
        "invalid_characters": {
            "title": "Idea with \x00 null bytes",
            "description": "Description with \uffff invalid unicode",
            "category": "ai_ml",
        },
    }


@pytest.fixture
def mock_external_api_responses() -> Dict[str, Any]:
    """Mock responses for external API calls."""
    return {
        "openai_embedding": {
            "data": [
                {
                    "object": "embedding",
                    "embedding": [0.1] * 1536,  # Mock 1536-dimensional embedding
                    "index": 0,
                }
            ],
            "model": "text-embedding-3-small",
            "object": "list",
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        },
        "semantic_scholar_response": {
            "total": 1,
            "data": [
                {
                    "paperId": "123456789",
                    "title": "Sample Research Paper",
                    "abstract": "This is a sample abstract for testing purposes.",
                    "url": "https://semanticscholar.org/paper/123456789",
                    "year": 2023,
                    "citationCount": 42,
                }
            ],
        },
        "google_search_response": {
            "items": [
                {
                    "title": "Sample Search Result",
                    "link": "https://example.com/result1",
                    "snippet": "This is a sample search result snippet.",
                },
                {
                    "title": "Another Search Result",
                    "link": "https://example.com/result2", 
                    "snippet": "Another sample search result snippet.",
                },
            ]
        },
    }


@pytest.fixture
def error_scenarios() -> Dict[str, Any]:
    """Define various error scenarios for testing."""
    return {
        "database_errors": {
            "connection_error": "Database connection failed",
            "timeout_error": "Database query timeout",
            "integrity_error": "Database integrity constraint violation",
        },
        "api_errors": {
            "rate_limit": {"status_code": 429, "message": "Rate limit exceeded"},
            "unauthorized": {"status_code": 401, "message": "Unauthorized access"},
            "not_found": {"status_code": 404, "message": "Resource not found"},
            "server_error": {"status_code": 500, "message": "Internal server error"},
        },
        "validation_errors": {
            "missing_field": "Required field missing",
            "invalid_format": "Invalid data format",
            "constraint_violation": "Data constraint violation",
        },
    }