import click
from uuid import UUID as PyUUID  # Use PyUUID to avoid conflict with click.UUID
from typing import List, Optional

# Assuming PYTHONPATH is set up for scripts to find core
from core import idea_ledger
from core.models import IdeaCreate, IdeaUpdate

# --- Database Initialization ---
# This ensures DB and tables are ready before any command runs.
# A more sophisticated setup might use a dedicated 'init-db' command
# or manage this within an application factory if it were a web app.
try:
    idea_ledger.create_db_and_tables()
except Exception as e:
    click.echo(f"Error initializing database: {e}", err=True)
    click.echo(
        "Ensure PostgreSQL server is running and accessible via DATABASE_URL.", err=True
    )
    # CLI might fail if DB isn't up; this just prints an initial warning.
    # For robust init, a dedicated 'init-db' command is better.


@click.group()
def cli():
    """A CLI tool to manage startup ideas."""
    pass


@cli.command("new")
@click.option("-n", "--name", required=True, help="Name of the idea.")
@click.option("-d", "--description", required=True, help="Description of the idea.")
@click.option(
    "-e",
    "--evidence",
    multiple=True,
    help="Evidence URL/path (can be used multiple times).",
)
@click.option("-p", "--deck-path", help="Path to the pitch deck.")
@click.option("-s", "--status", default="ideation", help="Initial status.")
def new_idea(
    name: str,
    description: str,
    evidence: List[str],
    deck_path: Optional[str],
    status: str,
):
    """Creates a new idea."""
    try:
        idea_data = IdeaCreate(
            name=name,
            description=description,
            evidence=list(evidence),  # Convert tuple from multiple=True to list
            deck_path=deck_path,
            status=status,
        )
        created_idea = idea_ledger.add_idea(idea_data)
        click.echo("Idea created successfully!")
        click.echo(f"ID: {created_idea.id}")
        click.echo(f"Name: {created_idea.name}")
        click.echo(f"Status: {created_idea.status}")
    except Exception as e:
        raise click.ClickException(f"Error creating idea: {e}")


@cli.command("list")
@click.option("--skip", default=0, help="Number of ideas to skip (for pagination).")
@click.option("--limit", default=100, help="Maximum number of ideas to list.")
def list_ideas(skip: int, limit: int):
    """Lists existing ideas."""
    try:
        ideas = idea_ledger.list_ideas(skip=skip, limit=limit)
        if not ideas:
            click.echo("No ideas found.")
            return

        click.echo(f"Found {len(ideas)} idea(s):")
        for idx, idea in enumerate(ideas):
            click.echo(
                f"  {idx + 1}. ID: {idea.id} | Name: {idea.name} | "
                f"Status: {idea.status}"
            )
    except Exception as e:
        raise click.ClickException(f"Error listing ideas: {e}")


@cli.command("show")
@click.argument("idea_id", type=click.UUID)
def show_idea(idea_id: PyUUID):
    """Shows detailed information about a specific idea."""
    try:
        idea = idea_ledger.get_idea_by_id(idea_id)
        if idea:
            click.echo(f"Idea Details (ID: {idea.id}):")
            click.echo(f"  Name: {idea.name}")
            click.echo(f"  Description: {idea.description}")
            click.echo(f"  Status: {idea.status}")
            evidence_str = ", ".join(idea.evidence) if idea.evidence else "None"
            click.echo(f"  Evidence: {evidence_str}")
            deck_path_str = idea.deck_path if idea.deck_path else "None"
            click.echo(f"  Deck Path: {deck_path_str}")
        else:
            click.echo(f"Idea with ID {idea_id} not found.")
    except Exception as e:
        raise click.ClickException(f"Error showing idea: {e}")


@cli.command("update")
@click.argument("idea_id", type=click.UUID)
@click.option("-n", "--name", help="New name for the idea.")
@click.option("-d", "--description", help="New description for the idea.")
@click.option(
    "-e", "--evidence", multiple=True, help="New list of evidence (replaces existing)."
)
@click.option("-p", "--deck-path", help="New path to the pitch deck.")
@click.option("-s", "--status", help="New status for the idea.")
def update_idea_command(
    idea_id: PyUUID,
    name: Optional[str],
    description: Optional[str],
    evidence: Optional[List[str]],  # Will be tuple if multiple=True
    deck_path: Optional[str],
    status: Optional[str],
):
    """Updates an existing idea. Only provided fields will be updated."""
    update_data_dict = {}
    if name is not None:
        update_data_dict["name"] = name
    if description is not None:
        update_data_dict["description"] = description
    if evidence:  # Check if tuple is non-empty
        update_data_dict["evidence"] = list(evidence)
    # To clear deck_path, a different mechanism/flag might be needed.
    # This update only sets if a value is provided.
    if deck_path is not None:
        update_data_dict["deck_path"] = deck_path
    if status is not None:
        update_data_dict["status"] = status

    if not update_data_dict:
        click.echo("No update parameters provided. Nothing to do.")
        return

    try:
        idea_update_model = IdeaUpdate(**update_data_dict)
        updated_idea = idea_ledger.update_idea(idea_id, idea_update_model)
        if updated_idea:
            click.echo(f"Idea ID {updated_idea.id} updated successfully:")
            click.echo(f"  Name: {updated_idea.name}")
            click.echo(f"  Status: {updated_idea.status}")
            click.echo(f"  Description: {updated_idea.description}")
            click.echo(f"  Evidence: {updated_idea.evidence}")
            click.echo(f"  Deck Path: {updated_idea.deck_path}")
        else:
            click.echo(f"Idea with ID {idea_id} not found or no update occurred.")
    except Exception as e:
        raise click.ClickException(f"Error updating idea: {e}")


@cli.command("delete")
@click.argument("idea_id", type=click.UUID)
def delete_idea_command(idea_id: PyUUID):
    """Deletes an idea."""
    try:
        if idea_ledger.delete_idea(idea_id):
            click.echo(f"Idea with ID {idea_id} deleted successfully.")
        else:
            click.echo(f"Idea with ID {idea_id} not found or could not be deleted.")
    except Exception as e:
        raise click.ClickException(f"Error deleting idea: {e}")


if __name__ == "__main__":
    cli()
