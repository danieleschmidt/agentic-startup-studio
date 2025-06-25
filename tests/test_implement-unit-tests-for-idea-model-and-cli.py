import json

import pytest
from click.testing import CliRunner

from agentic_startup_studio.cli import validate
from agentic_startup_studio.idea import Idea, IdeaCategory


@pytest.fixture()
def runner():
    return CliRunner()


def test_valid_idea_model_success():
    data = {
        "title": "Innovative Fintech Platform",
        "description": "A platform to revolutionize fintech payments.",
        "category": IdeaCategory.FINTECH,
        "problem": "Slow transactions",
        "solution": "Use blockchain",
        "market": "Global",
        "evidence_links": ["https://example.com"],
    }
    idea = Idea(**data)
    assert idea.title == data["title"].strip()
    assert idea.category is IdeaCategory.FINTECH


def test_cli_validation_success(tmp_path, runner):
    data = {
        "title": "Awesome SAAS Tool",
        "description": "Helps automate tasks.",
        "category": "saas",
    }
    input_file = tmp_path / "idea.json"
    input_file.write_text(json.dumps(data))
    result = runner.invoke(validate, ["--input", str(input_file)])
    assert result.exit_code == 0
    assert "Idea is valid!" in result.output


def test_invalid_idea_model_fails():
    with pytest.raises(Exception):
        Idea(title="short", description="bad")


def test_cli_validation_invalid(runner):
    result = runner.invoke(validate, ["--title", "short", "--description", "bad"])
    assert result.exit_code == 1
    assert "Idea validation failed:" in result.output
