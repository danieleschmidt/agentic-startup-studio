import pytest
import os
from pathlib import Path
from core.deck_generator import generate_deck_content

# Basic Marp template content for testing
BASIC_TEMPLATE_CONTENT = """---
marp: true
---
# {title}
## {subtitle}

---
## {section1_title}
{section1_content}

---
## {section2_title}
{section2_content}

---
## {section3_title}
{section3_content}

---
Contact: {contact_email}
"""


@pytest.fixture
def marp_template_file(tmp_path: Path) -> str:
    """Creates a temporary Marp template file for testing."""
    template_file = tmp_path / "test_template.marp"
    template_file.write_text(BASIC_TEMPLATE_CONTENT, encoding="utf-8")
    return str(template_file)


def test_generate_deck_with_all_data(marp_template_file: str):
    deck_data = {
        "title": "Test Title",
        "subtitle": "Test Subtitle",
        "contact_email": "test@example.com",
        "sections": [
            {"title": "Section 1 Head", "content": "* Point 1A\n* Point 1B"},
            {"title": "Section 2 Head", "content": "Content for section 2."},
            {"title": "Section 3 Head", "content": "## Subheading\nMore content."},
        ],
    }
    result = generate_deck_content(deck_data, marp_template_file)

    assert "# Test Title" in result
    assert "## Test Subtitle" in result
    assert "Contact: test@example.com" in result
    assert "## Section 1 Head" in result
    assert "* Point 1A\n* Point 1B" in result
    assert "## Section 2 Head" in result
    assert "Content for section 2." in result
    assert "## Section 3 Head" in result
    assert "## Subheading\nMore content." in result


def test_generate_deck_with_fewer_sections(marp_template_file: str):
    deck_data = {
        "title": "Fewer Sections Deck",
        "subtitle": "Testing with one section.",
        "contact_email": "less@example.com",
        "sections": [{"title": "Only Section", "content": "This is the only section."}],
    }
    result = generate_deck_content(deck_data, marp_template_file)

    assert "# Fewer Sections Deck" in result
    assert "## Only Section" in result
    assert "This is the only section." in result
    # Check that placeholders for non-existent sections are filled with defaults
    assert "## Section 2 (Not Provided)" in result
    assert "*Content not provided for this section.*" in result  # For section 2 content
    assert "## Section 3 (Not Provided)" in result
    assert "Contact: less@example.com" in result


def test_generate_deck_with_empty_deck_data(marp_template_file: str):
    deck_data = {}
    result = generate_deck_content(deck_data, marp_template_file)

    assert "# Default Title" in result
    assert "## Default Subtitle" in result
    assert "Contact: contact@example.com" in result
    assert "## Section 1 (Not Provided)" in result
    assert "## Section 2 (Not Provided)" in result
    assert "## Section 3 (Not Provided)" in result


def test_generate_deck_with_empty_sections_list(marp_template_file: str):
    deck_data = {
        "title": "Empty Sections Test",
        "subtitle": "No sections here.",
        "sections": [],
    }
    result = generate_deck_content(deck_data, marp_template_file)

    assert "# Empty Sections Test" in result
    assert "## Section 1 (Not Provided)" in result
    assert "## Section 2 (Not Provided)" in result
    assert "## Section 3 (Not Provided)" in result


def test_generate_deck_template_not_found():
    deck_data = {"title": "Test"}
    template_path = "non_existent_template.marp"
    result = generate_deck_content(deck_data, template_path)
    assert "Error: Template file not found" in result
    assert template_path in result


def test_generate_deck_section_content_formatting(marp_template_file: str):
    # Ensure multi-line content with markdown is preserved
    deck_data = {
        "title": "Formatting Test",
        "sections": [
            {
                "title": "Formatted Content",
                "content": (
                    "* Item 1\n* Item 2\n  * Subitem 2.1\n\nParagraph after list."
                ),
            }
        ],
    }
    result = generate_deck_content(deck_data, marp_template_file)
    expected_content = "* Item 1\n* Item 2\n  * Subitem 2.1\n\nParagraph after list."
    assert expected_content in result
    assert "## Formatted Content" in result
