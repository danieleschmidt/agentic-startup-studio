# tests/core/test_models.py
import pytest
from uuid import UUID
from datetime import datetime, timezone # Added timezone for offset-aware comparison if needed, else utcnow is naive
from pydantic import ValidationError, HttpUrl

from core.models import Idea, EvidenceItem

# New tests for EvidenceItem and Idea models

def test_evidence_item_creation():
    data = {
        "source": "Test Source",
        "claim": "Test Claim",
        "citation_url": "http://example.com/evidence"
    }
    item = EvidenceItem(**data)
    assert item.source == "Test Source"
    assert item.claim == "Test Claim"
    assert item.citation_url == HttpUrl("http://example.com/evidence")

def test_evidence_item_invalid_url():
    with pytest.raises(ValidationError) as excinfo:
        EvidenceItem(source="Test", claim="Test", citation_url="not_a_url")
    # Optional: Check the error details
    assert "url_parsing" in str(excinfo.value).lower() # Make comparison case-insensitive for robustness


def test_idea_creation_minimal():
    data = {
        "name": "Test Idea",
        "description": "A great idea",
        "problem": "A big problem",
        "solution": "A novel solution"
    }
    idea = Idea(**data)
    assert idea.name == "Test Idea"
    assert idea.description == "A great idea"
    assert idea.problem == "A big problem"
    assert idea.solution == "A novel solution"
    assert isinstance(idea.id, UUID)
    assert idea.status == "ideation"
    assert isinstance(idea.created_at, datetime)
    assert isinstance(idea.updated_at, datetime)
    assert idea.evidence == []
    # For naive datetimes (like utcnow()), they won't have tzinfo
    assert idea.created_at.tzinfo is None
    assert idea.updated_at.tzinfo is None


def test_idea_creation_full():
    evidence_data = {
        "source": "Full Source",
        "claim": "Full Claim",
        "citation_url": "http://example.com/full"
    }
    evidence_item = EvidenceItem(**evidence_data)
    data = {
        "name": "Full Idea",
        "description": "Comprehensive description",
        "problem": "Detailed problem statement",
        "solution": "Elaborate solution",
        "market_size": 1000000.0,
        "competition": "Many competitors",
        "team_description": "Experienced team",
        "evidence": [evidence_item],
        "deck_path": "/path/to/deck.pdf",
        "status": "research"
        # created_at and updated_at will be defaulted
    }
    idea = Idea(**data)
    assert idea.name == "Full Idea"
    assert idea.description == "Comprehensive description"
    assert idea.problem == "Detailed problem statement"
    assert idea.solution == "Elaborate solution"
    assert idea.market_size == 1000000.0
    assert idea.competition == "Many competitors"
    assert idea.team_description == "Experienced team"
    assert len(idea.evidence) == 1
    assert idea.evidence[0].source == "Full Source"
    assert idea.evidence[0].claim == "Full Claim"
    assert idea.evidence[0].citation_url == HttpUrl("http://example.com/full")
    assert idea.deck_path == "/path/to/deck.pdf"
    assert idea.status == "research"
    assert isinstance(idea.id, UUID)
    assert isinstance(idea.created_at, datetime)
    assert isinstance(idea.updated_at, datetime)

def test_idea_id_unique_and_defaults():
    idea1 = Idea(name="Idea1", description="d1", problem="p1", solution="s1")
    idea2 = Idea(name="Idea2", description="d2", problem="p2", solution="s2")
    assert idea1.id != idea2.id
    assert idea1.status == "ideation"
    assert idea1.evidence == []
    assert idea1.market_size is None
    assert idea1.competition is None
    assert idea1.team_description is None
    assert idea1.deck_path is None


def test_idea_default_timestamps_are_recent():
    idea = Idea(name="Timestamp Idea", description="d", problem="p", solution="s")
    # Get current UTC time, ensuring it's naive to match Pydantic's default datetime.utcnow()
    now = datetime.utcnow()
    # Allow a small delta for execution time (e.g., 5 seconds)
    time_delta_created = (now - idea.created_at).total_seconds()
    time_delta_updated = (now - idea.updated_at).total_seconds()

    assert -1 < time_delta_created < 5, f"created_at delta out of range: {time_delta_created}"
    assert -1 < time_delta_updated < 5, f"updated_at delta out of range: {time_delta_updated}"

    # Also check that updated_at is greater than or equal to created_at
    assert idea.updated_at >= idea.created_at

if __name__ == "__main__":
    pytest.main()
