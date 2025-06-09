import json
import uuid # Keep this for general UUID usage if needed, though Pydantic handles it for models
from typing import Optional

import click
from sqlmodel import Session # Required for type hinting if used, though next(get_db_session()) provides it

# Import functions and models from the core module
# Assuming PYTHONPATH is set up correctly for scripts to find the core module
from core.idea_ledger import (
    get_session as get_db_session, # Renamed to avoid clash
    create_idea as crud_create_idea,
    list_ideas as crud_list_ideas,
    get_idea as crud_get_idea, # Will be used by show/update/delete
    update_idea as crud_update_idea, # Will be used by update
    delete_idea as crud_delete_idea, # Will be used by delete
    create_db_and_tables # For the init-db command
)
from core.models import Idea as PydanticIdea, EvidenceItem

@click.group()
def cli():
    """Manage ideas in the Idea Ledger."""
    pass

@cli.command()
@click.option("--name", required=True, help="Name of the idea.")
@click.option("--description", required=True, help="Detailed description of the idea.")
@click.option("--problem", required=True, help="The problem this idea solves.")
@click.option("--solution", required=True, help="The proposed solution.")
@click.option("--market-size", type=float, help="Estimated market size (e.g., 1000000.0).")
@click.option("--competition", help="Known competition (e.g., 'StartupX, BigCorpY').")
@click.option("--team-description", help="Description of the team.")
@click.option("--status", default="ideation", help="Initial status of the idea (e.g., ideation, research).")
# Evidence will be added via an update command for simplicity in the `new` command for now.
def new(
    name: str,
    description: str,
    problem: str,
    solution: str,
    market_size: Optional[float],
    competition: Optional[str],
    team_description: Optional[str],
    status: str,
):
    """Create a new idea."""
    click.echo("Creating a new idea...")
    # Construct the PydanticIdea model from CLI options
    # ID, created_at, updated_at are defaulted by Pydantic model or DB/SQLModel
    # evidence is defaulted to an empty list by Pydantic model
    idea_in = PydanticIdea(
        name=name,
        description=description,
        problem=problem,
        solution=solution,
        market_size=market_size,
        competition=competition,
        team_description=team_description,
        # evidence=[], # This is handled by PydanticIdea default_factory
        status=status,
        # deck_path is also optional and defaults to None in PydanticIdea
    )

    # Use a context manager for the database session
    with next(get_db_session()) as session:
        try:
            # Call the CRUD function to create the idea in the database
            created_idea_sql = crud_create_idea(session=session, idea_in=idea_in)
            # Convert the returned SQLModel object back to a Pydantic model for consistent output
            output_idea = PydanticIdea.model_validate(created_idea_sql)
            # Output the created idea as JSON
            # default=str is used to handle non-serializable types like UUID and datetime
            click.echo(json.dumps(output_idea.model_dump(), indent=2, default=str))
        except Exception as e:
            click.echo(f"Error creating idea: {e}", err=True)

@cli.command()
@click.option("--skip", default=0, type=int, help="Number of ideas to skip for pagination.")
@click.option("--limit", default=10, type=int, help="Maximum number of ideas to retrieve.")
def list(skip: int, limit: int):
    """List existing ideas."""
    click.echo(f"Listing ideas (skip={skip}, limit={limit})...")
    with next(get_db_session()) as session:
        try:
            ideas_sql = crud_list_ideas(session=session, skip=skip, limit=limit)
            if not ideas_sql:
                click.echo("No ideas found.")
                return

            # Convert list of SQLModel objects to list of Pydantic model dicts for JSON output
            output_ideas = [PydanticIdea.model_validate(idea).model_dump() for idea in ideas_sql]
            click.echo(json.dumps(output_ideas, indent=2, default=str))
        except Exception as e:
            click.echo(f"Error listing ideas: {e}", err=True)

@cli.command()
def init_db():
    """Initialize the database and create tables."""
    click.echo("Initializing database...")
    try:
        create_db_and_tables()
        click.echo("Database initialized and tables created successfully.")
    except Exception as e:
        # More specific error handling could be added here if needed
        click.echo(f"Error initializing database: {e}", err=True)

@cli.command("show")
@click.argument("idea_id", type=click.UUID)
def show(idea_id: uuid.UUID): # Changed from show_idea_command to just show
    """Show details for a specific idea."""
    click.echo(f"Showing details for idea {idea_id}...")
    with next(get_db_session()) as session:
        try:
            idea_sql = crud_get_idea(session=session, idea_id=idea_id)
            if idea_sql:
                output_idea = PydanticIdea.model_validate(idea_sql)
                click.echo(json.dumps(output_idea.model_dump(), indent=2, default=str))
            else:
                click.echo(f"Idea with ID {idea_id} not found.", err=True)
        except Exception as e:
            click.echo(f"Error retrieving idea: {e}", err=True)

@cli.command("update")
@click.argument("idea_id", type=click.UUID)
@click.option("--name", help="New name for the idea.")
@click.option("--description", help="New detailed description.")
@click.option("--problem", help="New problem statement.")
@click.option("--solution", help="New proposed solution.")
@click.option("--market-size", type=float, help="New estimated market size.")
@click.option("--competition", help="New known competition.")
@click.option("--team-description", help="New description of the team.")
@click.option("--status", help="New status for the idea.")
@click.option("--evidence-item", "evidence_items", multiple=True, type=(str, str, str),
              help="Evidence item as 'source' 'claim' 'url'. Can be used multiple times.")
@click.option("--clear-evidence", is_flag=True, help="Clear all existing evidence before adding new items.")
def update( # Changed from update_idea_command to just update
    idea_id: uuid.UUID,
    name: Optional[str],
    description: Optional[str],
    problem: Optional[str],
    solution: Optional[str],
    market_size: Optional[float],
    competition: Optional[str],
    team_description: Optional[str],
    status: Optional[str],
    evidence_items: list[tuple[str, str, str]], # Corrected type hint from prompt
    clear_evidence: bool
):
    """Update an existing idea. Provide options for fields to change."""
    click.echo(f"Updating idea {idea_id}...")
    with next(get_db_session()) as session:
        try:
            existing_idea_sql = crud_get_idea(session=session, idea_id=idea_id)
            if not existing_idea_sql:
                click.echo(f"Idea with ID {idea_id} not found.", err=True)
                return

            # Convert SQLModel to Pydantic to easily apply updates
            update_data_pyd = PydanticIdea.model_validate(existing_idea_sql)

            update_fields = {
                k: v for k, v in {
                    "name": name, "description": description, "problem": problem,
                    "solution": solution, "market_size": market_size,
                    "competition": competition, "team_description": team_description,
                    "status": status
                }.items() if v is not None
            }

            if update_fields: # Check if there are any scalar fields to update
                 update_data_pyd = update_data_pyd.model_copy(update=update_fields)

            if clear_evidence:
                update_data_pyd.evidence = []

            if evidence_items: # Check if new evidence items are provided
                for source_val, claim_val, url_val in evidence_items: # Renamed to avoid conflict
                    try:
                        item = EvidenceItem(source=source_val, claim=claim_val, citation_url=url_val)
                        update_data_pyd.evidence.append(item)
                    except Exception as e: # Catch Pydantic validation error specifically if possible
                        click.echo(f"Skipping invalid evidence item ('{source_val}', '{claim_val}', '{url_val}'): {e}", err=True)

            # Now update_data_pyd contains all changes, pass it to CRUD
            updated_idea_sql = crud_update_idea(session=session, idea_id=idea_id, idea_update_data=update_data_pyd)
            # No need to check updated_idea_sql for None here, as crud_update_idea will return the updated object or raise an error if something went wrong after the initial get.
            # The initial check for existing_idea_sql handles the "not found" case for update.
            output_idea = PydanticIdea.model_validate(updated_idea_sql)
            click.echo(json.dumps(output_idea.model_dump(), indent=2, default=str))
        except Exception as e:
            click.echo(f"Error updating idea: {e}", err=True)

@cli.command("delete")
@click.argument("idea_id", type=click.UUID)
def delete(idea_id: uuid.UUID): # Changed from delete_idea_command to just delete
    """Delete an idea."""
    click.echo(f"Deleting idea {idea_id}...")
    with next(get_db_session()) as session:
        try:
            success = crud_delete_idea(session=session, idea_id=idea_id)
            if success:
                click.echo(f"Idea {idea_id} deleted successfully.")
            else:
                click.echo(f"Idea with ID {idea_id} not found or could not be deleted.", err=True)
        except Exception as e:
            click.echo(f"Error deleting idea: {e}", err=True)

if __name__ == "__main__":
    cli()
