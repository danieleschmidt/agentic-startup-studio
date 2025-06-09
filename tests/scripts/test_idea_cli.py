import json
import uuid
from typing import Generator # Added Generator for type hint

import pytest
from click.testing import CliRunner
from sqlmodel import create_engine, Session, SQLModel

from scripts.idea import cli
from core.idea_ledger import Idea as SQLModelIdea # For direct DB verification
# get_session is not directly used by tests, but core.idea_ledger.engine is patched

SQLITE_DATABASE_URL = "sqlite:///:memory:"
original_engine = None # Store the original engine

@pytest.fixture(scope="module", autouse=True)
def override_engine():
    global original_engine
    import core.idea_ledger # Import here to access its 'engine' attribute

    # Store the original engine instance from core.idea_ledger
    original_engine = core.idea_ledger.engine

    # Create a new engine for testing
    test_engine = create_engine(SQLITE_DATABASE_URL, connect_args={"check_same_thread": False})

    # Patch core.idea_ledger.engine with the test_engine
    core.idea_ledger.engine = test_engine

    yield # Tests run with the overridden engine

    # Restore the original engine after all tests in the module have run
    if original_engine:
        core.idea_ledger.engine = original_engine

@pytest.fixture(scope="function")
def db_session_for_cli(override_engine) -> Generator[Session, None, None]:
    # Depends on override_engine to ensure the engine is already patched
    # Uses core.idea_ledger.engine which is now the test_engine
    SQLModel.metadata.create_all(core.idea_ledger.engine)
    with Session(core.idea_ledger.engine) as session:
        yield session
    SQLModel.metadata.drop_all(core.idea_ledger.engine)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()

def test_init_db_command(runner: CliRunner, db_session_for_cli: Session):
    # db_session_for_cli fixture ensures tables are created and dropped around this test
    result = runner.invoke(cli, ["init-db"])
    assert result.exit_code == 0, result.output
    assert "Database initialized and tables created successfully." in result.output

def test_new_idea_command(runner: CliRunner, db_session_for_cli: Session):
    result = runner.invoke(cli, [
        "new",
        "--name", "CLI Test Idea",
        "--description", "Desc from CLI",
        "--problem", "Problem from CLI",
        "--solution", "Solution from CLI"
    ])
    assert result.exit_code == 0, result.output
    # Defensive split, ensure the starting phrase is actually in output before splitting
    output_parts = result.output.split("Creating a new idea...")
    assert len(output_parts) > 1, f"CLI output did not contain expected starting phrase for 'new' command. Output: {result.output}"

    # The actual JSON output starts after "Creating a new idea..." and any leading/trailing whitespace
    json_string = output_parts[1].strip()
    output_json = json.loads(json_string)

    assert output_json["name"] == "CLI Test Idea"
    assert output_json["description"] == "Desc from CLI" # Corrected line
    assert "id" in output_json

    idea_id = uuid.UUID(output_json["id"])
    idea_db = db_session_for_cli.get(SQLModelIdea, idea_id)
    assert idea_db is not None
    assert idea_db.name == "CLI Test Idea"

def test_list_ideas_command(runner: CliRunner, db_session_for_cli: Session):
    # Create an idea first using the CLI 'new' command
    # Use catch_exceptions=False to make debugging easier if 'new' fails
    new_result = runner.invoke(cli, ["new", "--name", "Listable Idea", "--description", "d", "--problem", "p", "--solution", "s"], catch_exceptions=False)
    assert new_result.exit_code == 0, f"Setup for list command failed: {new_result.output}"

    result = runner.invoke(cli, ["list"])
    assert result.exit_code == 0, result.output
    output_parts = result.output.split("Listing ideas (skip=0, limit=10)...")
    assert len(output_parts) > 1, f"CLI output did not contain expected starting phrase for 'list' command. Output: {result.output}"

    json_string = output_parts[1].strip()
    output_json = json.loads(json_string)

    assert len(output_json) >= 1 # Changed to >=1 to be more robust if other tests left data (though drop_all should prevent)
    # Find the specific idea
    found_idea = any(item["name"] == "Listable Idea" for item in output_json)
    assert found_idea, "Listable Idea not found in list output."

def test_show_idea_command(runner: CliRunner, db_session_for_cli: Session):
    res_new = runner.invoke(cli, ["new", "--name", "Showable Idea", "--description", "d", "--problem", "p", "--solution", "s"], catch_exceptions=False)
    assert res_new.exit_code == 0, f"Setup for show command failed: {res_new.output}"
    new_output_parts = res_new.output.split("Creating a new idea...")
    assert len(new_output_parts) > 1
    new_idea_json = json.loads(new_output_parts[1].strip())
    idea_id_str = new_idea_json["id"]

    result = runner.invoke(cli, ["show", idea_id_str])
    assert result.exit_code == 0, result.output
    show_output_parts = result.output.split(f"Showing details for idea {idea_id_str}...")
    assert len(show_output_parts) > 1
    show_output_json = json.loads(show_output_parts[1].strip())
    assert show_output_json["name"] == "Showable Idea"

    non_existent_id = str(uuid.uuid4())
    result_not_found = runner.invoke(cli, ["show", non_existent_id])
    assert result_not_found.exit_code == 0, result_not_found.output
    assert f"Idea with ID {non_existent_id} not found." in result_not_found.output

def test_update_idea_command(runner: CliRunner, db_session_for_cli: Session):
    res_new = runner.invoke(cli, ["new", "--name", "Updateable Idea", "--description", "d", "--problem", "p", "--solution", "s"], catch_exceptions=False)
    assert res_new.exit_code == 0, f"Setup for update command failed: {res_new.output}"
    new_output_parts = res_new.output.split("Creating a new idea...")
    assert len(new_output_parts) > 1
    new_idea_json = json.loads(new_output_parts[1].strip())
    idea_id_str = new_idea_json["id"]

    result = runner.invoke(cli, [
        "update", idea_id_str,
        "--name", "Updated CLI Idea Name",
        "--status", "updated_via_cli",
        "--evidence-item", "src1", "claim1", "http://e.com/1",
        "--evidence-item", "src2", "claim2", "http://e.com/2",
    ], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    update_output_parts = result.output.split(f"Updating idea {idea_id_str}...")
    assert len(update_output_parts) > 1
    updated_output_json = json.loads(update_output_parts[1].strip())
    assert updated_output_json["name"] == "Updated CLI Idea Name"
    assert updated_output_json["status"] == "updated_via_cli"
    assert len(updated_output_json["evidence"]) == 2
    assert updated_output_json["evidence"][0]["source"] == "src1"

    result_clear = runner.invoke(cli, ["update", idea_id_str, "--clear-evidence"], catch_exceptions=False)
    assert result_clear.exit_code == 0, result_clear.output
    clear_output_parts = result_clear.output.split(f"Updating idea {idea_id_str}...")
    assert len(clear_output_parts) > 1
    cleared_output_json = json.loads(clear_output_parts[1].strip())
    assert len(cleared_output_json["evidence"]) == 0

def test_delete_idea_command(runner: CliRunner, db_session_for_cli: Session):
    res_new = runner.invoke(cli, ["new", "--name", "Deleteable Idea", "--description", "d", "--problem", "p", "--solution", "s"], catch_exceptions=False)
    assert res_new.exit_code == 0, f"Setup for delete command failed: {res_new.output}"
    new_output_parts = res_new.output.split("Creating a new idea...")
    assert len(new_output_parts) > 1
    new_idea_json = json.loads(new_output_parts[1].strip())
    idea_id_str = new_idea_json["id"]

    result = runner.invoke(cli, ["delete", idea_id_str])
    assert result.exit_code == 0, result.output
    assert f"Idea {idea_id_str} deleted successfully." in result.output

    idea_db = db_session_for_cli.get(SQLModelIdea, uuid.UUID(idea_id_str))
    assert idea_db is None

    non_existent_id = str(uuid.uuid4())
    result_not_found = runner.invoke(cli, ["delete", non_existent_id])
    assert result_not_found.exit_code == 0, result_not_found.output # Exit code 0 for "not found"
    assert f"Idea with ID {non_existent_id} not found or could not be deleted." in result_not_found.output
