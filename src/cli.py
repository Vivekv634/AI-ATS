"""
AI-ATS Command Line Interface

Provides CLI commands for managing the AI-ATS application,
including database operations, model training, and utilities.
"""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

app = typer.Typer(
    name="ai-ats",
    help="AI-Powered Applicant Tracking System CLI",
    add_completion=False,
)
console = Console()


@app.command()
def version():
    """Show application version."""
    from src import __version__, __app_name__

    console.print(f"[bold blue]{__app_name__}[/bold blue] version [green]{__version__}[/green]")


@app.command()
def info():
    """Show system information and configuration."""
    from src.utils.config import get_settings

    settings = get_settings()

    table = Table(title="AI-ATS Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Environment", settings.environment)
    table.add_row("Debug Mode", str(settings.debug))
    table.add_row("Database Host", settings.database.host)
    table.add_row("Database Name", settings.database.name)
    table.add_row("Vector Store", settings.vector_store.provider)
    table.add_row("Embedding Model", settings.ml.embedding_model)
    table.add_row("ML Device", settings.ml.device)
    table.add_row("Log Level", settings.logging.level)

    console.print(table)


@app.command()
def init_db():
    """Initialize the database with required collections and indexes."""
    console.print("[yellow]Initializing database...[/yellow]")
    # TODO: Implement database initialization
    console.print("[green]Database initialized successfully![/green]")


@app.command()
def import_resumes(
    path: Path = typer.Argument(..., help="Path to resume file or directory"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Search recursively"),
):
    """Import resumes from files or directory."""
    console.print(f"[yellow]Importing resumes from: {path}[/yellow]")
    # TODO: Implement resume import
    console.print("[green]Import completed![/green]")


@app.command()
def create_job(
    title: str = typer.Option(..., "--title", "-t", help="Job title"),
    description_file: Optional[Path] = typer.Option(
        None, "--file", "-f", help="Job description file"
    ),
):
    """Create a new job posting."""
    console.print(f"[yellow]Creating job: {title}[/yellow]")
    # TODO: Implement job creation
    console.print("[green]Job created successfully![/green]")


@app.command()
def train(
    model_type: str = typer.Argument(..., help="Model type to train (embedding, classifier)"),
    data_path: Path = typer.Option(..., "--data", "-d", help="Training data path"),
    output_path: Optional[Path] = typer.Option(None, "--output", "-o", help="Model output path"),
):
    """Train or fine-tune ML models."""
    console.print(f"[yellow]Training {model_type} model...[/yellow]")
    # TODO: Implement model training
    console.print("[green]Training completed![/green]")


@app.command()
def gui():
    """Launch the graphical user interface."""
    console.print("[yellow]Launching GUI...[/yellow]")
    from src.main import main

    main()


if __name__ == "__main__":
    app()
