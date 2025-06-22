import json
import os
from pathlib import Path

import click

from health_check_standalone import run_health_checks

DEFAULT_RESULTS_FILE = "health_check_results.json"


@click.command()
@click.option(
    "--results-file",
    "results_file",
    default=None,
    help="Path to save health check results.",
    type=click.Path(dir_okay=False),
)
def main(results_file: str | None) -> None:
    """Run health checks and write results to a file."""
    file_path = results_file or os.getenv("HEALTH_CHECK_RESULTS_FILE", DEFAULT_RESULTS_FILE)
    results = run_health_checks()
    path = Path(file_path)
    path.write_text(json.dumps(results, indent=2))
    click.echo(f"Results saved to {path}")


if __name__ == "__main__":
    main()
