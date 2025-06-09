# core/models.py
from typing import Optional, List
from uuid import UUID, uuid4
from datetime import datetime
from pydantic import BaseModel, HttpUrl, Field

class EvidenceItem(BaseModel):
    source: str
    claim: str
    citation_url: HttpUrl

class Idea(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    description: str
    problem: str
    solution: str
    market_size: Optional[float] = None
    competition: Optional[str] = None # Could be a list of competitors later
    team_description: Optional[str] = None # Could be a list of team members later
    evidence: List[EvidenceItem] = Field(default_factory=list)
    deck_path: Optional[str] = None
    status: str = "ideation" # Example: ideation, research, funding, building, launched, archived
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Optional: Add a method to update 'updated_at' on modification,
    # though this is often handled at the ORM/database level.
    # For Pydantic models, if they are mutated, this won't automatically update.
    # Consider if this is needed here or if `updated_at` is set on write by the ledger.
