from __future__ import annotations

import json
from typing import Optional

import click
from pydantic import ValidationError

from .idea import Idea, IdeaCategory


@click.command()
@click.option(
    "--input", "input_file", type=click.File("r"), help="JSON file with idea data"
)
@click.option("--title")
@click.option("--description")
@click.option("--category", default=IdeaCategory.UNCATEGORIZED.value)
@click.option("--problem")
@click.option("--solution")
@click.option("--market")
@click.option("--evidence", help="Comma separated URLs")
def validate(
    input_file: Optional[click.File],
    title: str,
    description: str,
    category: str,
    problem: Optional[str],
    solution: Optional[str],
    market: Optional[str],
    evidence: Optional[str],
) -> None:
    """Validate idea input using the :class:`Idea` model."""
    if input_file is not None:
        data = json.load(input_file)
    else:
        evidence_links = (
            [link.strip() for link in evidence.split(",")] if evidence else []
        )
        data = {
            "title": title or "",
            "description": description or "",
            "category": category,
            "problem": problem,
            "solution": solution,
            "market": market,
            "evidence_links": evidence_links,
        }
    try:
        idea = Idea(**data)
    except ValidationError as exc:
        click.echo("Idea validation failed:")
        click.echo(exc.json())
        raise SystemExit(1)

    click.echo("Idea is valid!")
    click.echo(idea.model_dump_json(indent=2))


if __name__ == "__main__":
    validate()
