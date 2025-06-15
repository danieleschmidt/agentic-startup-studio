import pytest
from uuid import uuid4, UUID
from typing import Generator, List

from sqlmodel import create_engine, Session, SQLModel, delete  # Import delete
from unittest.mock import patch

# Import the models and ledger functions
# We assume PYTHONPATH is set up correctly for tests to find the core module
from core.models import Idea, IdeaCreate, IdeaUpdate
from core import idea_ledger  # Import the module itself to patch its 'engine'

# Use an in-memory SQLite database for testing
TEST_DATABASE_URL = "sqlite:///:memory:"


@pytest.fixture(name="engine")
def engine_fixture():
    """Creates a fresh SQLite in-memory engine for each test function."""
    # Using connect_args to ensure a single connection is used for the in-memory DB
    # for the lifetime of the engine, which can help with some in-memory DB behaviors.
    # However, for SQLite in-memory, each engine instance is typically a new DB.
    engine = create_engine(
        TEST_DATABASE_URL, echo=False, connect_args={"check_same_thread": False}
    )
    SQLModel.metadata.create_all(engine)  # Create tables for each test
    return engine


@pytest.fixture(name="session")
def session_fixture(engine) -> Generator[Session, None, None]:
    """
    Provides a session for a test, using a function-scoped (fresh) engine.
    """
    with Session(engine) as session:
        yield session
        # For function-scoped engine, DB is discarded post-test.
        # No explicit cleanup/rollback on engine needed.
        # `with Session...` handles session close/rollback on errors.


# This fixture patches the 'engine' object in the 'core.idea_ledger' module
# for the duration of each test that uses it.
@pytest.fixture(autouse=True)
def patch_ledger_engine(engine) -> Generator:  # Depends on the function-scoped engine
    """Patches the ledger's global engine with the test-specific (fresh) engine."""
    with patch.object(idea_ledger, "engine", engine):
        yield


# --- Test Cases ---


def test_create_db_and_tables(engine):  # Uses function-scoped engine
    """Test that tables are created (implicitly by engine fixture)."""
    # We can add a check here to see if the 'idea' table exists
    from sqlalchemy import inspect

    inspector = inspect(engine)
    assert "idea" in inspector.get_table_names()


# session here is from session_fixture, using patched engine
def test_add_idea(session: Session):
    """Test adding a new idea."""
    idea_data = IdeaCreate(
        name="Test Idea 1",
        description="Description for Test Idea 1",
        evidence=["http://example.com/evidence1"],
        status="testing",
    )
    created_idea = idea_ledger.add_idea(idea_data)

    assert created_idea.id is not None
    assert isinstance(created_idea.id, UUID)
    assert created_idea.name == idea_data.name
    assert created_idea.description == idea_data.description
    assert created_idea.evidence == idea_data.evidence
    assert created_idea.status == idea_data.status

    # Verify it's in the DB by trying to get it via a new session or the same one
    retrieved_idea = session.get(Idea, created_idea.id)
    assert retrieved_idea is not None
    assert retrieved_idea.name == idea_data.name


def test_get_idea_by_id(session: Session):
    """Test retrieving an idea by its ID."""
    idea_data = IdeaCreate(name="Test Idea 2", description="Desc 2")
    created_idea = idea_ledger.add_idea(idea_data)

    fetched_idea = idea_ledger.get_idea_by_id(created_idea.id)
    assert fetched_idea is not None
    assert fetched_idea.id == created_idea.id
    assert fetched_idea.name == "Test Idea 2"


def test_get_non_existent_idea():
    """Test retrieving a non-existent idea."""
    non_existent_id = uuid4()
    fetched_idea = idea_ledger.get_idea_by_id(non_existent_id)
    assert fetched_idea is None


def test_list_ideas_empty():
    """Test listing ideas when the database is empty."""
    ideas = idea_ledger.list_ideas()
    assert ideas == []


# Add session for direct verification if needed
def test_list_ideas_with_items_and_pagination(session: Session):
    """Test listing ideas with items and basic pagination."""
    idea1_data = IdeaCreate(name="List Idea 1", description="Desc A")
    idea2_data = IdeaCreate(name="List Idea 2", description="Desc B")
    idea3_data = IdeaCreate(name="List Idea 3", description="Desc C")

    idea1 = idea_ledger.add_idea(idea1_data)
    idea2 = idea_ledger.add_idea(idea2_data)
    idea3 = idea_ledger.add_idea(idea3_data)

    # Test default listing (all items if < limit)
    all_ideas = idea_ledger.list_ideas(limit=10)
    assert len(all_ideas) == 3
    # Order isn't guaranteed unless explicitly set in query, so check for presence
    assert idea1 in all_ideas
    assert idea2 in all_ideas
    assert idea3 in all_ideas

    # Test limit
    limited_ideas = idea_ledger.list_ideas(limit=2)
    assert len(limited_ideas) == 2

    # Test skip (offset)
    # Assuming default ordering or an order by ID/name for predictability if added
    # For now, we'll just check counts with skip
    skipped_ideas = idea_ledger.list_ideas(skip=1, limit=2)
    assert len(skipped_ideas) == 2  # If 3 items, skip 1, limit 2 -> items 2, 3

    skipped_one_idea = idea_ledger.list_ideas(skip=2, limit=2)
    assert len(skipped_one_idea) == 1  # If 3 items, skip 2, limit 2 -> item 3


def test_update_idea(session: Session):
    """Test updating an existing idea."""
    idea_data = IdeaCreate(name="Update Idea", description="Initial Desc")
    created_idea = idea_ledger.add_idea(idea_data)

    update_data = IdeaUpdate(
        name="Updated Idea Name",
        status="in_progress",
        evidence=["http://new.evidence/link"],
    )
    updated_idea = idea_ledger.update_idea(created_idea.id, update_data)

    assert updated_idea is not None
    assert updated_idea.id == created_idea.id
    assert updated_idea.name == "Updated Idea Name"
    assert updated_idea.status == "in_progress"
    assert updated_idea.evidence == ["http://new.evidence/link"]
    # Description was not in update_data
    assert updated_idea.description == "Initial Desc"

    # Verify in DB
    refreshed_idea = session.get(Idea, created_idea.id)
    assert refreshed_idea is not None
    assert refreshed_idea.name == "Updated Idea Name"
    assert refreshed_idea.status == "in_progress"


def test_update_idea_partial(session: Session):
    """Test partially updating an existing idea."""
    idea_data = IdeaCreate(
        name="Partial Update Idea", description="Full Desc", status="pending"
    )
    created_idea = idea_ledger.add_idea(idea_data)

    update_data = IdeaUpdate(status="approved")  # Only update status
    updated_idea = idea_ledger.update_idea(created_idea.id, update_data)

    assert updated_idea is not None
    assert updated_idea.status == "approved"
    assert updated_idea.name == "Partial Update Idea"  # Name should remain unchanged
    assert (
        updated_idea.description == "Full Desc"
    )  # Description should remain unchanged


def test_update_non_existent_idea():
    """Test updating a non-existent idea."""
    non_existent_id = uuid4()
    update_data = IdeaUpdate(name="Non Existent Update")
    updated_idea = idea_ledger.update_idea(non_existent_id, update_data)
    assert updated_idea is None


def test_delete_idea(session: Session):
    """Test deleting an existing idea."""
    idea_data = IdeaCreate(name="Delete Me", description="I will be deleted")
    created_idea = idea_ledger.add_idea(idea_data)

    # Ensure it's there first
    assert session.get(Idea, created_idea.id) is not None

    deletion_result = idea_ledger.delete_idea(created_idea.id)
    assert deletion_result is True

    # Verify it's gone from DB
    assert session.get(Idea, created_idea.id) is None
    assert idea_ledger.get_idea_by_id(created_idea.id) is None


def test_delete_non_existent_idea():
    """Test deleting a non-existent idea."""
    non_existent_id = uuid4()
    deletion_result = idea_ledger.delete_idea(non_existent_id)
    assert deletion_result is False


# Example of how to run if you want to test this file directly
# (though 'pytest' is preferred)
# if __name__ == "__main__":
# pytest.main([__file__])
