"""
CLI interface for startup idea ingestion and management.

This module provides command-line tools for interacting with the idea
ingestion pipeline including creation, validation, and status tracking.
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.text import Text

from pipeline.models.idea import (
    Idea, IdeaStatus, PipelineStage, IdeaCategory, QueryParams, IdeaSummary
)
from pipeline.config.settings import get_settings, get_config_summary
from pipeline.ingestion.idea_manager import (
    create_idea_manager, IdeaManager, ValidationError, 
    DuplicateIdeaError, StorageError, IdeaManagementError
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Rich console for enhanced output
console = Console()


class CLIError(Exception):
    """CLI-specific error for user-friendly error handling."""
    pass


def handle_async(func):
    """Decorator to handle async functions in Click commands."""
    def wrapper(*args, **kwargs):
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(func(*args, **kwargs))
        except KeyboardInterrupt:
            console.print("\n[yellow]Operation cancelled by user[/yellow]")
            sys.exit(1)
        except Exception as e:
            logger.exception("Unexpected error in CLI operation")
            console.print(f"[red]Error: {str(e)}[/red]")
            sys.exit(1)
    return wrapper


async def get_idea_manager() -> IdeaManager:
    """Initialize and return idea manager instance."""
    try:
        return await create_idea_manager()
    except Exception as e:
        logger.error(f"Failed to initialize idea manager: {e}")
        raise CLIError(f"System initialization failed: {e}")


def display_idea_summary(summary: IdeaSummary) -> None:
    """Display a formatted idea summary."""
    # Status color mapping
    status_colors = {
        IdeaStatus.DRAFT: "blue",
        IdeaStatus.VALIDATING: "yellow",
        IdeaStatus.VALIDATED: "green",
        IdeaStatus.REJECTED: "red",
        IdeaStatus.RESEARCHING: "cyan",
        IdeaStatus.BUILDING: "magenta",
        IdeaStatus.TESTING: "orange3",
        IdeaStatus.DEPLOYED: "bright_green",
        IdeaStatus.ARCHIVED: "dim"
    }
    
    status_color = status_colors.get(summary.status, "white")
    progress_bar = "█" * int(summary.progress * 20) + "░" * (20 - int(summary.progress * 20))
    
    console.print(f"[bold]{summary.title}[/bold]")
    console.print(f"  ID: {summary.id}")
    console.print(f"  Status: [{status_color}]{summary.status.value}[/{status_color}]")
    console.print(f"  Stage: {summary.stage.value}")
    console.print(f"  Progress: {progress_bar} {summary.progress:.1%}")
    console.print(f"  Created: {summary.created_at.strftime('%Y-%m-%d %H:%M')}")
    console.print()


def validate_idea_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and clean CLI input data."""
    # Required fields validation
    if not data.get('title', '').strip():
        raise CLIError("Title is required")
    
    if not data.get('description', '').strip():
        raise CLIError("Description is required")
    
    # Clean and validate category
    if 'category' in data and data['category']:
        try:
            # Validate category exists
            data['category'] = IdeaCategory(data['category'].lower())
        except ValueError:
            valid_categories = [cat.value for cat in IdeaCategory]
            raise CLIError(f"Invalid category. Valid options: {', '.join(valid_categories)}")
    
    # Process evidence links
    if 'evidence_links' in data and isinstance(data['evidence_links'], str):
        # Split comma-separated URLs
        links = [link.strip() for link in data['evidence_links'].split(',') if link.strip()]
        data['evidence_links'] = links
    
    return data


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, verbose):
    """Agentic Startup Studio - Idea Ingestion CLI"""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    
    # Display startup banner
    console.print(Panel.fit(
        "[bold blue]Agentic Startup Studio[/bold blue]\n"
        "[italic]Idea Ingestion Pipeline[/italic]",
        style="bright_blue"
    ))


@cli.command()
@click.option('--title', prompt='Idea title', help='Short, descriptive title for the idea')
@click.option('--description', prompt='Idea description', help='Detailed description of the startup idea')
@click.option('--category', type=click.Choice([cat.value for cat in IdeaCategory]), 
              help='Business category for the idea')
@click.option('--problem', help='Problem statement this idea solves')
@click.option('--solution', help='How the solution works')
@click.option('--market', help='Target customer segment')
@click.option('--evidence', help='Comma-separated list of evidence URLs')
@click.option('--force', is_flag=True, help='Force creation even if similar ideas exist')
@click.option('--output', type=click.Choice(['json', 'table']), default='table',
              help='Output format')
@handle_async
async def create(title, description, category, problem, solution, market, evidence, force, output):
    """Create a new startup idea with validation and duplicate detection."""
    try:
        # Prepare idea data
        idea_data = {
            'title': title,
            'description': description,
            'category': category,
            'problem_statement': problem,
            'solution_description': solution,
            'target_market': market,
            'evidence_links': evidence
        }
        
        # Validate input data
        idea_data = validate_idea_data(idea_data)
        
        # Initialize manager and create idea
        manager = await get_idea_manager()
        
        with console.status("[bold green]Creating idea..."):
            idea_id, warnings = await manager.create_idea(
                raw_data=idea_data,
                force_create=force,
                user_id="cli_user"  # In production, would get from auth context
            )
        
        # Display results
        if output == 'json':
            result = {
                'idea_id': str(idea_id),
                'status': 'created',
                'warnings': warnings
            }
            console.print(json.dumps(result, indent=2))
        else:
            console.print(f"[green]✓[/green] Idea created successfully!")
            console.print(f"[bold]ID:[/bold] {idea_id}")
            
            if warnings:
                console.print("\n[yellow]Warnings:[/yellow]")
                for warning in warnings:
                    console.print(f"  • {warning}")
        
    except DuplicateIdeaError as e:
        console.print(f"[yellow]Duplicate ideas detected:[/yellow] {e}")
        if not force and Confirm.ask("Create anyway?"):
            # Retry with force flag
            await create.callback(title, description, category, problem, solution, 
                                market, evidence, True, output)
        else:
            console.print("[red]Idea creation cancelled[/red]")
            sys.exit(1)
    
    except ValidationError as e:
        console.print(f"[red]Validation failed:[/red] {e}")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Idea creation failed: {e}")
        console.print(f"[red]Creation failed:[/red] {e}")
        sys.exit(1)


@cli.command('list')
@click.option('--status', type=click.Choice([s.value for s in IdeaStatus]),
              help='Filter by status')
@click.option('--stage', type=click.Choice([s.value for s in PipelineStage]),
              help='Filter by pipeline stage')
@click.option('--category', type=click.Choice([c.value for c in IdeaCategory]),
              help='Filter by category')
@click.option('--limit', type=int, default=20, help='Maximum number of ideas to show')
@click.option('--search', help='Search in title and description')
@click.option('--sort', type=click.Choice(['created_at', 'updated_at', 'title']),
              default='created_at', help='Sort field')
@click.option('--desc', is_flag=True, default=True, help='Sort in descending order')
@click.option('--output', type=click.Choice(['json', 'table']), default='table',
              help='Output format')
@handle_async
async def list_ideas(status, stage, category, limit, search, sort, desc, output):
    """List startup ideas with filtering and sorting options."""
    try:
        # Build query parameters
        filters = QueryParams(
            status_filter=[IdeaStatus(status)] if status else None,
            stage_filter=[PipelineStage(stage)] if stage else None,
            category_filter=[IdeaCategory(category)] if category else None,
            search_text=search,
            limit=limit,
            sort_by=sort,
            sort_desc=desc
        )
        
        # Get ideas
        manager = await get_idea_manager()
        
        with console.status("[bold blue]Fetching ideas..."):
            ideas = await manager.list_ideas(filters)
        
        if not ideas:
            console.print("[yellow]No ideas found matching your criteria[/yellow]")
            return
        
        # Display results
        if output == 'json':
            ideas_data = []
            for idea in ideas:
                ideas_data.append({
                    'id': str(idea.id),
                    'title': idea.title,
                    'status': idea.status.value,
                    'stage': idea.stage.value,
                    'progress': idea.progress,
                    'created_at': idea.created_at.isoformat()
                })
            console.print(json.dumps(ideas_data, indent=2))
        else:
            console.print(f"\n[bold]Found {len(ideas)} ideas:[/bold]\n")
            for idea in ideas:
                display_idea_summary(idea)
    
    except Exception as e:
        logger.error(f"Failed to list ideas: {e}")
        console.print(f"[red]Failed to list ideas:[/red] {e}")
        sys.exit(1)


@cli.command()
@click.option('--id', 'idea_id', required=True, help='ID of the idea to retrieve')
@click.option('--output', type=click.Choice(['json', 'detail']), default='detail',
              help='Output format')
@handle_async
async def get(idea_id, output):
    """Show detailed information about a specific idea."""
    try:
        # Parse UUID
        try:
            uuid_obj = UUID(idea_id)
        except ValueError:
            raise CLIError(f"Invalid idea ID format: {idea_id}")
        
        # Get idea details
        manager = await get_idea_manager()
        idea = await manager.get_idea(uuid_obj)
        
        if not idea:
            console.print(f"[red]Idea not found:[/red] {idea_id}")
            sys.exit(1)
        
        # Display results
        if output == 'json':
            idea_data = {
                'id': str(idea.idea_id),
                'title': idea.title,
                'description': idea.description,
                'category': idea.category.value,
                'status': idea.status.value,
                'stage': idea.current_stage.value,
                'progress': idea.stage_progress,
                'problem_statement': idea.problem_statement,
                'solution_description': idea.solution_description,
                'target_market': idea.target_market,
                'evidence_links': idea.evidence_links,
                'created_at': idea.created_at.isoformat(),
                'updated_at': idea.updated_at.isoformat(),
                'created_by': idea.created_by
            }
            console.print(json.dumps(idea_data, indent=2))
        else:
            # Detailed view
            console.print(Panel(
                f"[bold]{idea.title}[/bold]\n\n"
                f"{idea.description}",
                title="Idea Details",
                expand=False
            ))
            
            # Create details table
            table = Table(show_header=False, box=None)
            table.add_column("Field", style="bold")
            table.add_column("Value")
            
            table.add_row("ID", str(idea.idea_id))
            table.add_row("Category", idea.category.value)
            table.add_row("Status", f"[blue]{idea.status.value}[/blue]")
            table.add_row("Stage", f"[cyan]{idea.current_stage.value}[/cyan]")
            table.add_row("Progress", f"{idea.stage_progress:.1%}")
            table.add_row("Created", idea.created_at.strftime('%Y-%m-%d %H:%M:%S'))
            table.add_row("Updated", idea.updated_at.strftime('%Y-%m-%d %H:%M:%S'))
            
            if idea.problem_statement:
                table.add_row("Problem", idea.problem_statement)
            if idea.solution_description:
                table.add_row("Solution", idea.solution_description)
            if idea.target_market:
                table.add_row("Target Market", idea.target_market)
            if idea.evidence_links:
                table.add_row("Evidence Links", "\n".join(idea.evidence_links))
            
            console.print(table)
    
    except Exception as e:
        logger.error(f"Failed to show idea: {e}")
        console.print(f"[red]Failed to show idea:[/red] {e}")
        sys.exit(1)


@cli.command()
@click.argument('idea_id', type=str)
@click.option('--title', help='Update title')
@click.option('--description', help='Update description')
@click.option('--category', type=click.Choice([cat.value for cat in IdeaCategory]),
              help='Update category')
@click.option('--problem', help='Update problem statement')
@click.option('--solution', help='Update solution description')
@click.option('--market', help='Update target market')
@handle_async
async def update(idea_id, title, description, category, problem, solution, market):
    """Update an existing startup idea."""
    try:
        # Parse UUID
        try:
            uuid_obj = UUID(idea_id)
        except ValueError:
            raise CLIError(f"Invalid idea ID format: {idea_id}")
        
        # Build updates dictionary
        updates = {}
        if title: updates['title'] = title
        if description: updates['description'] = description
        if category: updates['category'] = IdeaCategory(category)
        if problem: updates['problem_statement'] = problem
        if solution: updates['solution_description'] = solution
        if market: updates['target_market'] = market
        
        if not updates:
            console.print("[yellow]No updates specified[/yellow]")
            return
        
        # Apply updates
        manager = await get_idea_manager()
        
        with console.status("[bold green]Updating idea..."):
            success = await manager.update_idea(
                idea_id=uuid_obj,
                updates=updates,
                user_id="cli_user"
            )
        
        if success:
            console.print(f"[green]✓[/green] Idea updated successfully!")
            console.print(f"Updated fields: {', '.join(updates.keys())}")
        else:
            console.print("[red]Update failed[/red]")
            sys.exit(1)
    
    except ValidationError as e:
        console.print(f"[red]Validation failed:[/red] {e}")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Failed to update idea: {e}")
        console.print(f"[red]Update failed:[/red] {e}")
        sys.exit(1)


@cli.command()
@click.option('--id', 'idea_id', required=True, help='ID of the idea to advance')
@click.option('--stage', required=True, type=click.Choice([s.value for s in PipelineStage]),
              help='Target stage to advance to')
@handle_async
async def advance(idea_id, stage):
    """Advance an idea to the next pipeline stage."""
    try:
        # Parse inputs
        try:
            uuid_obj = UUID(idea_id)
        except ValueError:
            raise CLIError(f"Invalid idea ID format: {idea_id}")
        
        target_stage = PipelineStage(stage)
        
        # Advance stage
        manager = await get_idea_manager()
        
        with console.status(f"[bold green]Advancing to {stage}..."):
            success = await manager.advance_stage(
                idea_id=uuid_obj,
                next_stage=target_stage,
                user_id="cli_user"
            )
        
        if success:
            console.print(f"[green]✓[/green] Idea advanced to {stage}!")
        else:
            console.print("[red]Stage advancement failed[/red]")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Failed to advance idea stage: {e}")
        console.print(f"[red]Stage advancement failed:[/red] {e}")
        sys.exit(1)


@cli.command()
@click.argument('idea_id', type=str)
@click.option('--limit', type=int, default=5, help='Number of similar ideas to show')
@handle_async
async def similar(idea_id, limit):
    """Find ideas similar to the specified idea."""
    try:
        # Parse UUID
        try:
            uuid_obj = UUID(idea_id)
        except ValueError:
            raise CLIError(f"Invalid idea ID format: {idea_id}")
        
        # Find similar ideas
        manager = await get_idea_manager()
        
        with console.status("[bold blue]Finding similar ideas..."):
            similar_ideas = await manager.get_similar_ideas(uuid_obj, limit)
        
        if not similar_ideas:
            console.print("[yellow]No similar ideas found[/yellow]")
            return
        
        console.print(f"\n[bold]Found {len(similar_ideas)} similar ideas:[/bold]\n")
        
        # Get details for similar ideas
        for similar_id, score in similar_ideas:
            idea = await manager.get_idea(similar_id)
            if idea:
                console.print(f"[bold]{idea.title}[/bold] (similarity: {score:.2%})")
                console.print(f"  ID: {similar_id}")
                console.print(f"  Status: {idea.status.value}")
                console.print(f"  Description: {idea.description[:100]}...")
                console.print()
    
    except Exception as e:
        logger.error(f"Failed to find similar ideas: {e}")
        console.print(f"[red]Failed to find similar ideas:[/red] {e}")
        sys.exit(1)


@cli.command()
def config():
    """Show current system configuration."""
    try:
        config_summary = get_config_summary()
        
        console.print(Panel(
            json.dumps(config_summary, indent=2),
            title="System Configuration",
            expand=False
        ))
    
    except Exception as e:
        logger.error(f"Failed to show configuration: {e}")
        console.print(f"[red]Failed to show configuration:[/red] {e}")
        sys.exit(1)


@cli.command()
@handle_async
async def health():
    """Check system health and connectivity."""
    try:
        console.print("[bold blue]Checking system health...[/bold blue]\n")
        
        # Test idea manager initialization
        with console.status("Testing idea manager..."):
            manager = await get_idea_manager()
        console.print("[green]✓[/green] Idea manager: OK")
        
        # Test database connectivity
        with console.status("Testing database connectivity..."):
            # Try to list ideas (minimal query)
            await manager.list_ideas(QueryParams(limit=1))
        console.print("[green]✓[/green] Database: OK")
        
        console.print("\n[green]All systems operational![/green]")
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        console.print(f"\n[red]Health check failed:[/red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
class UserInputError(CLIError):
    """Error for invalid user input."""
    pass


class OutputFormatError(CLIError):
    """Error for output formatting issues."""
    pass


class IdeaIngestionCLI:
    """Object-oriented CLI interface for idea ingestion."""
    
    def __init__(self, idea_manager: IdeaManager):
        """Initialize CLI with idea manager."""
        self.idea_manager = idea_manager
    
    async def create_idea(
        self, 
        title: str, 
        description: str, 
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        evidence: Optional[str] = None,
        force: bool = False,
        user_id: Optional[str] = None
    ) -> bool:
        """Create a new idea and display results."""
        try:
            # Prepare idea data
            idea_data = {
                'title': title,
                'description': description,
                'category': category,
                'tags': tags or [],
                'evidence': evidence
            }
            
            # Clean data
            idea_data = validate_idea_data(idea_data)
            
            # Create idea
            idea_id, warnings = await self.idea_manager.create_idea(
                raw_data=idea_data,
                force_create=force,
                user_id=user_id or "cli_user"
            )
            
            # Display success
            console.print(f"[green]✓[/green] Idea successfully created!")
            console.print(f"[bold]ID:[/bold] {idea_id}")
            
            if warnings:
                console.print("\n[yellow]Warnings:[/yellow]")
                for warning in warnings:
                    console.print(f"  • {warning}")
            
            return True
            
        except DuplicateIdeaError as e:
            console.print(f"[yellow]Similar ideas found:[/yellow] {e}")
            console.print("Use --force to create anyway")
            return False
        
        except ValidationError as e:
            console.print(f"[red]Validation failed:[/red] {e}")
            for error in getattr(e, 'errors', []):
                console.print(f"  • {error}")
            return False
        
        except StorageError:
            console.print("[red]System error occurred. Please try again later.[/red]")
            return False
    
    async def get_idea(self, idea_id: str) -> bool:
        """Get and display idea details."""
        try:
            # Validate UUID format
            uuid_obj = self._validate_uuid_format(idea_id)
            
            # Get idea
            idea = await self.idea_manager.get_idea(uuid_obj)
            
            if not idea:
                console.print(f"[red]Idea not found:[/red] {idea_id}")
                return False
            
            # Display idea details
            output = self._format_idea_details(idea)
            console.print(output)
            
            return True
            
        except UserInputError as e:
            console.print(f"[red]{e}[/red]")
            return False
    
    async def list_ideas(
        self,
        status: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 20
    ) -> bool:
        """List ideas with optional filters."""
        try:
            # Build query parameters
            filters = QueryParams(
                status_filter=[self._parse_status_name(status)] if status else None,
                category_filter=[self._parse_category_name(category)] if category else None,
                limit=limit
            )
            
            # Get ideas
            ideas = await self.idea_manager.list_ideas(filters)
            
            if not ideas:
                console.print("[yellow]No ideas found[/yellow]")
                return True
            
            # Display table
            output = self._format_ideas_table(ideas)
            console.print(output)
            
            return True
            
        except Exception as e:
            console.print(f"[red]Failed to list ideas: {e}[/red]")
            return False
    
    async def advance_stage(
        self,
        idea_id: str,
        next_stage: str,
        user_id: Optional[str] = None
    ) -> bool:
        """Advance idea to next stage."""
        try:
            # Validate inputs
            uuid_obj = self._validate_uuid_format(idea_id)
            stage_obj = self._parse_stage_name(next_stage)
            
            # Advance stage
            success = await self.idea_manager.advance_stage(
                idea_id=uuid_obj,
                next_stage=stage_obj,
                user_id=user_id or "cli_user"
            )
            
            if success:
                console.print(f"[green]✓[/green] Stage advanced to {next_stage.upper()}!")
                return True
            else:
                console.print("[red]Failed to advance stage[/red]")
                return False
                
        except UserInputError as e:
            console.print(f"[red]{e}[/red]")
            return False
        
        except IdeaManagementError as e:
            console.print(f"[red]Failed to advance stage: {e}[/red]")
            return False
    
    def _format_idea_details(self, idea: Idea) -> str:
        """Format idea details for display."""
        progress_pct = f"{idea.stage_progress:.0%}"
        
        # Handle both enum objects and string values
        status_value = idea.status.value if hasattr(idea.status, 'value') else str(idea.status)
        stage_value = idea.current_stage.value if hasattr(idea.current_stage, 'value') else str(idea.current_stage)
        
        details = [
            f"[bold]{idea.title}[/bold]",
            f"Description: {idea.description}",
            f"Status: {status_value}",
            f"Stage: {stage_value}",
            f"Progress: {progress_pct}",
            f"Created: {idea.created_at.strftime('%Y-%m-%d %H:%M:%S')}"
        ]
        
        return "\n".join(details)
    
    def _format_ideas_table(self, ideas: List[IdeaSummary]) -> str:
        """Format ideas as a table."""
        table = Table()
        table.add_column("ID")
        table.add_column("Title") 
        table.add_column("Status")
        table.add_column("Stage")
        table.add_column("Progress")
        
        for idea in ideas:
            progress_pct = f"{idea.progress:.0%}"
            table.add_row(
                str(idea.id)[:8] + "...",
                idea.title,
                idea.status.value if hasattr(idea.status, 'value') else str(idea.status),
                idea.stage.value if hasattr(idea.stage, 'value') else str(idea.stage),
                progress_pct
            )
        
        console.print(table)
        return str(table)
    
    def _parse_stage_name(self, stage_name: str) -> PipelineStage:
        """Parse stage name to enum."""
        try:
            return PipelineStage(stage_name.upper())
        except ValueError:
            raise UserInputError(f"Invalid stage: {stage_name}")
    
    def _parse_status_name(self, status_name: str) -> IdeaStatus:
        """Parse status name to enum."""
        try:
            return IdeaStatus(status_name.upper())
        except ValueError:
            raise UserInputError(f"Invalid status: {status_name}")
    
    def _parse_category_name(self, category_name: str) -> IdeaCategory:
        """Parse category name to enum."""
        try:
            return IdeaCategory(category_name.lower())
        except ValueError:
            raise UserInputError(f"Invalid category: {category_name}")
    
    def _validate_uuid_format(self, uuid_str: str) -> UUID:
        """Validate UUID format."""
        try:
            return UUID(uuid_str)
        except ValueError:
            raise UserInputError(f"Invalid ID format: {uuid_str}")


async def create_cli_interface() -> IdeaIngestionCLI:
    """Create CLI interface with initialized idea manager."""
    idea_manager = await get_idea_manager()
    return IdeaIngestionCLI(idea_manager)

