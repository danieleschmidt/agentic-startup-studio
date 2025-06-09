from typing import Generator, List
from uuid import uuid4, UUID

import pytest
from sqlmodel import create_engine, Session, SQLModel

from core.idea_ledger import (
    create_idea as crud_create_idea,
    get_idea as crud_get_idea,
    list_ideas as crud_list_ideas,
    update_idea as crud_update_idea,
    delete_idea as crud_delete_idea,
    Idea as SQLModelIdea, # This is core.idea_ledger.Idea
)
from core.models import Idea as PydanticIdea, EvidenceItem # This is core.models.Idea

# Use an in-memory SQLite database for testing
SQLITE_DATABASE_URL = "sqlite:///:memory:"
# For SQLite in-memory, each engine instance can be a new DB if not managed.
# connect_args={"check_same_thread": False} is good practice for SQLite.
engine = create_engine(SQLITE_DATABASE_URL, connect_args={"check_same_thread": False}) # echo=True for debugging SQL

@pytest.fixture(scope="function")
def db_session() -> Generator[Session, None, None]:
    # Ensures tables are created fresh for each test function that uses this fixture.
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    # Clean up: drop all tables after the test to ensure isolation.
    SQLModel.metadata.drop_all(engine)

@pytest.fixture
def sample_evidence_item() -> EvidenceItem:
    return EvidenceItem(source="Test Source", claim="Test Claim", citation_url="http://example.com/evidence")

@pytest.fixture
def sample_idea_pydantic(sample_evidence_item: EvidenceItem) -> PydanticIdea:
    # Creates a Pydantic Idea that can be used as input for CRUD operations.
    # We can assign a UUID here if we want deterministic IDs for testing,
    # or let the default_factory in Pydantic/SQLModel handle it.
    # For consistency in tests, especially create then get, using a known ID can be useful.
    return PydanticIdea(
        id=uuid4(), # Assign a UUID for predictable testing
        name="Test Idea",
        description="A great test idea",
        problem="A problem to be solved by testing",
        solution="The solution is more testing",
        evidence=[sample_evidence_item],
        status="testing"
    )

def test_create_idea(db_session: Session, sample_idea_pydantic: PydanticIdea):
    # The idea_in for crud_create_idea is our PydanticIdea
    created_idea_sql: SQLModelIdea = crud_create_idea(session=db_session, idea_in=sample_idea_pydantic)

    assert created_idea_sql is not None
    # Verify that the ID of the created SQLModel object matches the ID from the input Pydantic object
    assert created_idea_sql.id == sample_idea_pydantic.id
    assert created_idea_sql.name == sample_idea_pydantic.name
    assert created_idea_sql.description == sample_idea_pydantic.description
    assert len(created_idea_sql.evidence) == 1
    assert created_idea_sql.evidence[0].source == sample_idea_pydantic.evidence[0].source
    assert created_idea_sql.evidence[0].claim == sample_idea_pydantic.evidence[0].claim
    assert str(created_idea_sql.evidence[0].citation_url) == str(sample_idea_pydantic.evidence[0].citation_url)
    assert created_idea_sql.status == sample_idea_pydantic.status

    # Additionally, try to fetch it from the DB to ensure it was committed
    retrieved_idea_directly = db_session.get(SQLModelIdea, created_idea_sql.id)
    assert retrieved_idea_directly is not None
    assert retrieved_idea_directly.name == sample_idea_pydantic.name

def test_get_idea(db_session: Session, sample_idea_pydantic: PydanticIdea):
    created_idea_sql = crud_create_idea(session=db_session, idea_in=sample_idea_pydantic)

    retrieved_idea = crud_get_idea(session=db_session, idea_id=created_idea_sql.id)
    assert retrieved_idea is not None
    assert retrieved_idea.id == created_idea_sql.id
    assert retrieved_idea.name == sample_idea_pydantic.name

    non_existent_id = uuid4()
    retrieved_non_existent = crud_get_idea(session=db_session, idea_id=non_existent_id)
    assert retrieved_non_existent is None

def test_list_ideas(db_session: Session, sample_idea_pydantic: PydanticIdea):
    # Create a couple of ideas with unique names to test listing
    idea1_pyd = sample_idea_pydantic.model_copy(update={"id": uuid4(), "name": "List Idea 1"})
    idea2_pyd = sample_idea_pydantic.model_copy(update={"id": uuid4(), "name": "List Idea 2"})

    crud_create_idea(session=db_session, idea_in=idea1_pyd)
    crud_create_idea(session=db_session, idea_in=idea2_pyd)

    ideas: List[SQLModelIdea] = crud_list_ideas(session=db_session, skip=0, limit=10)
    assert len(ideas) == 2
    idea_names = {idea.name for idea in ideas}
    assert "List Idea 1" in idea_names
    assert "List Idea 2" in idea_names

    ideas_limit_1: List[SQLModelIdea] = crud_list_ideas(session=db_session, skip=0, limit=1)
    assert len(ideas_limit_1) == 1

    ideas_skip_1: List[SQLModelIdea] = crud_list_ideas(session=db_session, skip=1, limit=1)
    assert len(ideas_skip_1) == 1
    # Ensure pagination is actually working by checking if the skipped item is different
    # This assumes a consistent order, which SQLite provides for simple PKs or by insertion order here.
    assert ideas_skip_1[0].name != ideas_limit_1[0].name

    empty_list: List[SQLModelIdea] = crud_list_ideas(session=db_session, skip=100, limit=10)
    assert len(empty_list) == 0

def test_update_idea(db_session: Session, sample_idea_pydantic: PydanticIdea):
    created_idea_sql = crud_create_idea(session=db_session, idea_in=sample_idea_pydantic)
    original_updated_at = created_idea_sql.updated_at

    # Prepare update data using a PydanticIdea model
    # Ensure the ID in the update data matches the ID of the Pydantic model used for creation
    # if you are re-using parts of sample_idea_pydantic or creating a new one.
    # The key is that idea_update_data.id should align with what model_validate expects if it uses it.
    # However, crud_update_idea uses idea_id param for lookup, and model_dump on idea_update_data.
    update_data_pyd = PydanticIdea(
        id=created_idea_sql.id, # Match the ID of the record to be updated
        name="Updated Test Idea",
        description="Updated description",
        problem=created_idea_sql.problem, # Keep some old values to ensure partial update works
        solution=created_idea_sql.solution,
        evidence=created_idea_sql.evidence, # Keep original evidence
        status="updated_status"
        # created_at and updated_at are not part of PydanticIdea input for update usually
    )

    updated_idea_sql = crud_update_idea(session=db_session, idea_id=created_idea_sql.id, idea_update_data=update_data_pyd)
    assert updated_idea_sql is not None
    assert updated_idea_sql.name == "Updated Test Idea"
    assert updated_idea_sql.description == "Updated description"
    assert updated_idea_sql.status == "updated_status"
    # Check that updated_at was actually changed
    assert updated_idea_sql.updated_at > original_updated_at

    # Test updating a non-existent idea
    non_existent_id = uuid4()
    # Create a PydanticIdea for the non-existent update attempt
    non_existent_update_data_pyd = PydanticIdea(
        id=non_existent_id, name="Try Update Non Existent", description="d", problem="p", solution="s"
    )
    failed_update_sql = crud_update_idea(
        session=db_session,
        idea_id=non_existent_id,
        idea_update_data=non_existent_update_data_pyd
    )
    assert failed_update_sql is None

def test_delete_idea(db_session: Session, sample_idea_pydantic: PydanticIdea):
    created_idea_sql = crud_create_idea(session=db_session, idea_in=sample_idea_pydantic)

    delete_successful = crud_delete_idea(session=db_session, idea_id=created_idea_sql.id)
    assert delete_successful
    # Verify it's actually gone from the database
    assert crud_get_idea(session=db_session, idea_id=created_idea_sql.id) is None

    # Test deleting a non-existent idea
    non_existent_id = uuid4()
    delete_unsuccessful = crud_delete_idea(session=db_session, idea_id=non_existent_id)
    assert not delete_unsuccessful
