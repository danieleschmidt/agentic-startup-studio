from typing import Optional, List, Any
from uuid import UUID, uuid4
from datetime import datetime

from sqlmodel import Field, SQLModel, create_engine, Session, Column, JSON, select # type: ignore
# from pgvector.sqlalchemy import Vector  # For pgvector support - Install if used

# Import Pydantic models for type hinting and conversion
from core.models import Idea as PydanticIdea, EvidenceItem as PydanticEvidenceItem

DATABASE_URL = "postgresql://dittofeed:password@postgres:5432/dittofeed" # This should ideally come from config/env
engine = create_engine(DATABASE_URL) # echo=True for local SQL debugging

class Idea(SQLModel, table=True):
    __tablename__ = "idea" # Explicitly define table name

    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    name: str = Field(index=True)
    description: str
    # description_vector: Optional[List[float]] = Field(sa_column=Column(Vector(384)), default=None) # Example for pgvector
    problem: str
    solution: str
    market_size: Optional[float] = Field(default=None)
    competition: Optional[str] = Field(default=None)
    team_description: Optional[str] = Field(default=None)

    # Store list of PydanticEvidenceItem as JSON.
    # SQLModel handles serialization/deserialization if type hint is List[PydanticModel] and sa_column is JSON
    evidence: List[PydanticEvidenceItem] = Field(default_factory=list, sa_column=Column(JSON))

    deck_path: Optional[str] = Field(default=None)
    status: str = Field(default="ideation", index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False) # Will need manual update in CRUD

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

# Placeholder for a function to get a DB session
def get_session():
    with Session(engine) as session:
        yield session

# --- Implementation of CRUD functions ---

def create_idea(session: Session, idea_in: PydanticIdea) -> Idea:
    db_idea = Idea.model_validate(idea_in)
    session.add(db_idea)
    session.commit()
    session.refresh(db_idea)
    return db_idea

def get_idea(session: Session, idea_id: UUID) -> Optional[Idea]:
    return session.get(Idea, idea_id)

def list_ideas(session: Session, skip: int = 0, limit: int = 100) -> List[Idea]:
    statement = select(Idea).offset(skip).limit(limit)
    return list(session.exec(statement).all())

def update_idea(session: Session, idea_id: UUID, idea_update_data: PydanticIdea) -> Optional[Idea]:
    db_idea = session.get(Idea, idea_id)
    if not db_idea:
        return None
    update_data = idea_update_data.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_idea, key, value)
    db_idea.updated_at = datetime.utcnow()
    session.add(db_idea)
    session.commit()
    session.refresh(db_idea)
    return db_idea

def delete_idea(session: Session, idea_id: UUID) -> bool:
    db_idea = session.get(Idea, idea_id)
    if not db_idea:
        return False
    session.delete(db_idea)
    session.commit()
    return True
