import os
import shutil  # For cleaning up generated MVP path if needed during testing/retries
from typing import Any  # For type hints

import click

# Assuming core.build_tools_manager is accessible via PYTHONPATH
from core.build_tools_manager import run_gpt_engineer, run_opendevin_debug_loop

# from core.idea_ledger import get_idea_by_id # Placeholder
from core.deployment_manager import (
    check_fly_io_health,
    deploy_to_fly_io,
)  # Import these

DEFAULT_MVPS_DIR = "generated_mvps"
GENERATED_MVPS_DIR = os.getenv("GENERATED_MVPS_DIR", DEFAULT_MVPS_DIR)


@click.command()
@click.option(
    "--idea-id", "idea_id", required=True, help="The ID of the idea to build."
)
@click.option(
    "--clean-output", is_flag=True, help="Clean output directory before running."
)
def run_build_cycle(idea_id: str, clean_output: bool):
    """
    Orchestrates the conceptual build cycle for a given idea:
    1. Fetches idea details (mocked for now).
    2. Runs a mock GPT-Engineer to scaffold the project.
    3. Runs a mock OpenDevin debug loop on the scaffolded project.
    (Future steps: Deploy to Fly.io, check health)
    """
    click.echo(f"Starting build cycle for Idea ID: {idea_id}")

    # 0. Define output path and optionally clean it
    # Slugify idea_id for directory name to be safe
    safe_idea_id_slug = "".join(
        c if c.isalnum() or c in ["-", "_"] else "_" for c in idea_id
    )
    # Output path is within GENERATED_MVPS_DIR
    output_path = os.path.join(GENERATED_MVPS_DIR, safe_idea_id_slug)

    if clean_output:
        if os.path.exists(output_path):
            click.echo(f"Cleaning existing output directory: {output_path}")
            try:
                shutil.rmtree(output_path)
            except OSError as e:
                click.echo(f"Error cleaning directory {output_path}: {e}", err=True)
                # Depending on severity, might want to exit. For now, continue.

    # Ensure base GENERATED_MVPS_DIR directory exists for the output_path
    if not os.path.exists(GENERATED_MVPS_DIR):
        try:
            os.makedirs(GENERATED_MVPS_DIR)
        except OSError as e:
            click.echo(
                f"Error creating base MVP directory {GENERATED_MVPS_DIR}: {e}", err=True
            )
            return  # Cannot proceed if base dir cannot be made

    # Note: run_gpt_engineer mock is expected to create the specific output_path subdir

    # 1. Fetch idea details (mocked for now)
    click.echo(f"Step 1: Fetching details for Idea ID: {idea_id} (Mocked)")
    # In future, replace with actual ledger call:
    # try:
    #     from uuid import UUID
    #     idea_uuid = UUID(idea_id)
    #     # idea_details_obj = get_idea_by_id(idea_uuid)
    #     # if not idea_details_obj:
    #     #     click.echo(f"Error: Idea {idea_id} not found in ledger.", err=True)
    #     #     return
    #     # idea_details = idea_details_obj.model_dump()
    #     # click.echo(f"Successfully fetched details for: {idea_details['name']}")
    # except ValueError:
    #     click.echo(
    #         f"Error: Invalid Idea ID format for {idea_id}. Expected UUID.",
    #         err=True
    #     )
    #     return
    # except Exception as e:
    #     click.echo(f"Error fetching idea from ledger: {e}", err=True)
    #     return

    mock_idea_details: dict[str, Any] = {
        "id": idea_id,
        "name": f"App for {idea_id}",
        "description": (
            f"A groundbreaking application for {idea_id} that will change the world."
        ),
        "requirements": [
            "User authentication",
            "Dashboard for data visualization",
            "API for integration",
        ],
    }
    click.echo(f"Using mock details for: {mock_idea_details['name']}")

    # 2. Run mock GPT-Engineer
    click.echo(
        f"Step 2: Running mock GPT-Engineer to scaffold project at {output_path}..."
    )
    if run_gpt_engineer(mock_idea_details, output_path):
        click.echo("Mock GPT-Engineer scaffolding successful.")
    else:
        click.echo(
            "Mock GPT-Engineer scaffolding failed. Exiting build cycle.", err=True
        )
        return

    # 3. Run mock OpenDevin debug loop
    click.echo(f"Step 3: Running mock OpenDevin debug loop on {output_path}...")
    if run_opendevin_debug_loop(output_path):
        click.echo("Mock OpenDevin debug loop successful (dummy tests 'passed').")
    else:
        click.echo(
            "Mock OpenDevin debug loop failed. Build cycle incomplete.", err=True
        )
        # Depending on policy, may or may not return here.
        # For now, let's proceed to show deployment placeholders.

    # 4. Deployment to Fly.io
    click.echo(f"Step 4: Deploying project from {output_path} to Fly.io...")
    # Sanitize idea_id for use as part of an app name
    # Fly.io app names typically are DNS-compatible (letters, numbers, hyphens)
    base_app_name = "".join(c if c.isalnum() else "-" for c in safe_idea_id_slug).strip(
        "-"
    )
    if not base_app_name:  # Ensure base_app_name is not empty after stripping
        base_app_name = f"idea-{abs(hash(idea_id)) % 10000}"  # Fallback for empty slug
    # Max length for Fly app names is often around 30-40 chars for hostnames.
    app_name_fly = f"app-{base_app_name[:20]}-{abs(hash(idea_id)) % 10000}"

    deployment_url = deploy_to_fly_io(output_path, app_name_fly)
    if deployment_url:
        click.echo(f"Mock deployment successful. App URL: {deployment_url}")
        click.echo(f"Step 5: Checking health of {app_name_fly} on Fly.io...")
        # time.sleep(0.5) # time.sleep is already in check_fly_io_health mock
        if check_fly_io_health(app_name_fly):  # Use the same app_name_fly
            click.echo("Mock health check successful.")
        else:
            click.echo("Mock health check failed.", err=True)
    else:
        click.echo("Mock deployment failed.", err=True)

    completed_msg = (
        f"Build cycle for Idea ID: {idea_id} conceptually completed with deployment."
    )
    click.echo(completed_msg)


if __name__ == "__main__":
    run_build_cycle()
