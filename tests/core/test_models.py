import pytest
import json  # Import the json module
from uuid import UUID, uuid4
from pydantic import ValidationError

from core.models import (
    Idea,
)  # Assuming core.models is discoverable (e.g. by PYTHONPATH)


def test_idea_creation_defaults():
    """Test Idea model creation with default values."""
    idea = Idea(name="Test Idea", description="A test description.")
    assert isinstance(idea.id, UUID)
    assert idea.name == "Test Idea"
    assert idea.description == "A test description."
    assert idea.evidence == []
    assert idea.deck_path is None
    assert idea.status == "ideation"


def test_idea_creation_with_specific_values():
    """Test Idea model creation with all fields specified."""
    uid = UUID("a1b2c3d4-e5f6-7890-1234-567890abcdef")
    evidence_list = ["http://example.com/evidence1", "docs/evidence2.pdf"]
    idea = Idea(
        # Providing ID for deterministic testing, though default_factory is usual
        id=uid,
        name="Specific Idea",
        description="Detailed specific description.",
        evidence=evidence_list,
        deck_path="pitches/specific_deck.md",
        status="research",
    )
    assert idea.id == uid
    assert idea.name == "Specific Idea"
    assert idea.description == "Detailed specific description."
    assert idea.evidence == evidence_list
    assert idea.deck_path == "pitches/specific_deck.md"
    assert idea.status == "research"


def test_idea_id_uniqueness():
    """Test that IDs are unique for different Idea instances."""
    idea1 = Idea(name="Idea 1", description="Desc 1")
    idea2 = Idea(name="Idea 2", description="Desc 2")
    assert idea1.id != idea2.id
    assert isinstance(idea1.id, UUID)
    assert isinstance(idea2.id, UUID)


def test_idea_type_validation():
    """Test Pydantic type validation for Idea fields using model_validate."""
    # Test invalid name (should be str)
    with pytest.raises(ValidationError, match="Input should be a valid string"):
        Idea.model_validate({"name": 123, "description": "A description"})

    # Test invalid description (should be str)
    with pytest.raises(ValidationError, match="Input should be a valid string"):
        Idea.model_validate({"name": "A name", "description": ["Not", "a", "string"]})

    # Test invalid evidence (should be List[str])
    # This will first fail because "not_a_list" is not a list
    with pytest.raises(ValidationError, match="Input should be a valid list"):
        Idea.model_validate(
            {"name": "A name", "description": "A description", "evidence": "not_a_list"}
        )

    # This will fail because list items are not strings
    with pytest.raises(ValidationError, match="Input should be a valid string"):
        Idea.model_validate(
            {"name": "A name", "description": "A description", "evidence": [123, 456]}
        )

    # Test invalid deck_path (should be Optional[str])
    with pytest.raises(ValidationError, match="Input should be a valid string"):
        Idea.model_validate(
            {"name": "A name", "description": "A description", "deck_path": 123.45}
        )

    # Test invalid status (should be str)
    with pytest.raises(ValidationError, match="Input should be a valid string"):
        Idea.model_validate(
            {
                "name": "A name",
                "description": "A description",
                "status": {"status": "invalid"},
            }
        )


def test_idea_optional_fields():
    """Test that optional fields can be None or provided."""
    idea_no_deck = Idea(name="No Deck Idea", description="Desc")
    assert idea_no_deck.deck_path is None

    idea_with_deck = Idea(
        name="Deck Idea", description="Desc", deck_path="path/to/deck.ppt"
    )
    assert idea_with_deck.deck_path == "path/to/deck.ppt"


def test_idea_serialization_deserialization():
    """Test JSON serialization and deserialization."""
    # Use a known ID for deterministic testing of serialization/deserialization
    known_id = uuid4()  # This was the line causing F821, import is now added
    original_idea = Idea(
        id=known_id,
        name="Serializable Idea",
        description="Testing JSON.",
        evidence=["link1"],
        deck_path="path/to/json_deck.json",
        status="funded",
    )

    # Pydantic V2: model_dump_json, model_validate (from dict),
    # or model_validate_json (from string).
    json_string = original_idea.model_dump_json()

    # Validate from a dictionary parsed from the JSON string
    data_dict = json.loads(json_string)
    deserialized_idea = Idea.model_validate(data_dict)

    # Compare model dumps for data equality.
    # If deserialized_idea.id is correctly UUID, default dump should work.
    # Using mode='json' for original_dump provides str UUIDs if needed for comparison.
    deserialized_dump = deserialized_idea.model_dump()
    original_dump = original_idea.model_dump()
    assert deserialized_dump == original_dump

    # Also check ID explicitly for clarity and type
    assert isinstance(deserialized_idea.id, UUID)  # Check type
    assert deserialized_idea.id == known_id
    assert deserialized_idea.name == "Serializable Idea"
    assert deserialized_idea.evidence == ["link1"]
    assert deserialized_idea.status == "funded"

    # Test with dict dump and parse
    dict_data = original_idea.model_dump()  # This will have id as UUID object
    parsed_idea_from_dict = Idea.model_validate(dict_data)  # Renamed for clarity
    # When both are from Python objects, default model_dump() should be consistent
    assert parsed_idea_from_dict.model_dump() == original_idea.model_dump()
    assert parsed_idea_from_dict.id == known_id  # Ensure ID is correctly parsed


def test_idea_config_json_schema_extra():
    """Test that the json_schema_extra (example) is part of the model's schema."""
    schema = Idea.model_json_schema()
    assert "example" in schema
    example = schema["example"]
    assert example["name"] == "AI-Powered Personal Chef"
    assert example["status"] == "research"


if __name__ == "__main__":
    pytest.main()
