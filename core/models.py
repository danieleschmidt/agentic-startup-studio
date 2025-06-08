from uuid import UUID, uuid4

# For Python 3.8 support, List can be replaced with list for 3.9+
from typing import List, Optional

from sqlmodel import Field, SQLModel
from sqlalchemy import Column
from sqlalchemy.types import JSON  # Import JSON type
from pydantic import ConfigDict  # Remove field_validator, ValidationInfo


# Main table model for Idea
class Idea(SQLModel, table=True):
    """
    Represents a startup idea and its associated metadata.
    This model is used for database interaction (table structure) and API responses.
    """

    # SQLModel handles primary_key, default_factory for UUID correctly
    id: Optional[UUID] = Field(
        default_factory=uuid4, primary_key=True, index=True, nullable=False
    )
    name: str = Field(index=True, description="The name or title of the idea")
    description: str = Field(description="A detailed description of the idea")
    # For list[str], SQLModel might need JSON or a custom type for some DBs,
    # but PostgreSQL with SQLModel can often handle list[str] via ARRAY type
    # if the dialect supports it.
    # Store list of strings as JSON in the database
    evidence: List[str] = Field(
        default_factory=list,
        sa_column=Column(JSON),  # Specify JSON column type for SQLAlchemy
        description="List of URLs or paths to evidence supporting the idea",
    )
    deck_path: Optional[str] = Field(
        default=None, description="Filesystem path to the pitch deck, if available"
    )
    status: str = Field(
        default="ideation",
        index=True,
        description=(
            "Current status of the idea (e.g., ideation, research, funded, rejected)"
        ),
    )

    # Pydantic V2 configuration using ConfigDict (can be part of SQLModel too)
    # Relies on Pydantic's default coercion for UUID, custom validator removed.
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "name": "AI-Powered Personal Chef",
                "description": (
                    "A service that uses AI to plan meals, generate shopping "
                    "lists, and guide users through cooking."
                ),
                "evidence": [
                    "https://example.com/market_research_report.pdf",
                    "https://example.com/competitor_analysis.docx",
                ],
                "deck_path": "pitches/ai_chef_deck_v1.marp",
                "status": "research",
            }
        },
        # For SQLModel, orm_mode is True by default, which is good.
        # For Pydantic V2, orm_mode is replaced by from_attributes = True
        from_attributes=True,
    )


# Pydantic model for creating an Idea (input)
class IdeaCreate(SQLModel):  # Inherits from SQLModel for consistency, not a table model
    name: str
    description: str
    evidence: List[str] = Field(default_factory=list)
    deck_path: Optional[str] = None
    status: str = "ideation"

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Sustainable Packaging Solution",
                "description": (
                    "Developing biodegradable packaging from agricultural waste."
                ),
                "evidence": ["https://example.com/initial_research.pdf"],
                "deck_path": "drafts/packaging_deck_v0.1.marp",
                "status": "ideation",
            }
        }
    )


# Pydantic model for updating an Idea (input, all fields optional)
class IdeaUpdate(SQLModel):  # Inherits from SQLModel for consistency, not a table model
    name: Optional[str] = None
    description: Optional[str] = None
    evidence: Optional[List[str]] = None
    deck_path: Optional[str] = None
    status: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "AI-Powered Meal Planner (Revised)",
                "status": "research",
                "evidence": ["https://example.com/new_market_data.pdf"],
            }
        }
    )


if __name__ == "__main__":
    # Example usage (Note: This will change as Idea is now a SQLModel table model)
    # To run these examples, you'd typically need a database session.

    print("--- Idea (SQLModel Table Model) ---")
    # SQLModel instances are usually created within a session context for DB operations
    # For plain model validation or serialization, it works like Pydantic:
    idea_data_example = Idea.model_config["json_schema_extra"]["example"]
    idea_instance = Idea.model_validate(idea_data_example)
    print(idea_instance.model_dump_json(indent=2))

    print("\n--- IdeaCreate ---")
    create_data_example = IdeaCreate.model_config["json_schema_extra"]["example"]
    create_instance = IdeaCreate.model_validate(create_data_example)
    print(create_instance.model_dump_json(indent=2))

    print("\n--- IdeaUpdate ---")
    update_data_example = IdeaUpdate.model_config["json_schema_extra"]["example"]
    update_instance = IdeaUpdate.model_validate(update_data_example)
    print(update_instance.model_dump_json(indent=2))
