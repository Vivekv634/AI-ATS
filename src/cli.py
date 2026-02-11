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
    import asyncio
    from src.data.database import get_database_manager

    console.print("[yellow]Initializing database...[/yellow]")

    try:
        db_manager = get_database_manager()

        # Check connection first
        console.print("  Checking database connection...")
        if not db_manager.check_sync_connection():
            console.print("[red]Error: Could not connect to MongoDB.[/red]")
            console.print("[dim]Make sure MongoDB is running and connection settings are correct.[/dim]")
            raise typer.Exit(1)

        console.print("  [green]✓[/green] Connected to MongoDB")

        # Create indexes
        console.print("  Creating indexes...")
        asyncio.run(db_manager.ensure_indexes())
        console.print("  [green]✓[/green] Indexes created")

        console.print("\n[green]Database initialized successfully![/green]")

    except Exception as e:
        console.print(f"[red]Error initializing database: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def import_resumes(
    path: Path = typer.Argument(..., help="Path to resume file or directory"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Search recursively"),
):
    """Import resumes from files or directory."""
    from src.data.database import get_database_manager
    from src.data.repositories import get_candidate_repository
    from src.ml.nlp import get_resume_parser
    from src.utils.constants import SUPPORTED_RESUME_FORMATS
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

    console.print(f"[yellow]Importing resumes from: {path}[/yellow]")

    # Validate path exists
    if not path.exists():
        console.print(f"[red]Error: Path does not exist: {path}[/red]")
        raise typer.Exit(1)

    # Check database connection
    db_manager = get_database_manager()
    if not db_manager.check_sync_connection():
        console.print("[red]Error: Could not connect to MongoDB. Run 'init-db' first.[/red]")
        raise typer.Exit(1)

    # Collect resume files
    resume_files: list[Path] = []
    if path.is_file():
        if path.suffix.lower() in SUPPORTED_RESUME_FORMATS:
            resume_files.append(path)
        else:
            console.print(f"[red]Error: Unsupported file format: {path.suffix}[/red]")
            console.print(f"[dim]Supported formats: {', '.join(SUPPORTED_RESUME_FORMATS)}[/dim]")
            raise typer.Exit(1)
    else:
        # Directory - find all resume files
        pattern = "**/*" if recursive else "*"
        for ext in SUPPORTED_RESUME_FORMATS:
            resume_files.extend(path.glob(f"{pattern}{ext}"))

    if not resume_files:
        console.print("[yellow]No resume files found.[/yellow]")
        raise typer.Exit(0)

    console.print(f"Found [cyan]{len(resume_files)}[/cyan] resume file(s)")

    # Initialize parser and repository
    parser = get_resume_parser()
    candidate_repo = get_candidate_repository()

    # Process resumes with progress bar
    success_count = 0
    error_count = 0
    errors: list[tuple[Path, str]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing resumes...", total=len(resume_files))

        for resume_file in resume_files:
            try:
                # Parse the resume
                result = parser.parse_file(str(resume_file))

                if result.success:
                    # Convert to candidate and save
                    candidate_data = parser.to_candidate_create(result)
                    if candidate_data:
                        candidate_repo.create_from_schema(candidate_data)
                    success_count += 1
                else:
                    error_msg = "; ".join(result.errors) if result.errors else "Unknown parsing error"
                    errors.append((resume_file, error_msg))
                    error_count += 1

            except Exception as e:
                errors.append((resume_file, str(e)))
                error_count += 1

            progress.update(task, advance=1)

    # Print summary
    console.print()
    console.print("[bold]Import Summary:[/bold]")
    console.print(f"  [green]✓ Imported:[/green] {success_count}")
    console.print(f"  [red]✗ Errors:[/red] {error_count}")

    if errors:
        console.print("\n[yellow]Errors:[/yellow]")
        for file_path, error_msg in errors[:10]:  # Show first 10 errors
            console.print(f"  [dim]{file_path.name}:[/dim] {error_msg}")
        if len(errors) > 10:
            console.print(f"  [dim]... and {len(errors) - 10} more errors[/dim]")

    console.print("\n[green]Import completed![/green]")


@app.command()
def create_job(
    title: str = typer.Option(..., "--title", "-t", help="Job title"),
    description_file: Optional[Path] = typer.Option(
        None, "--file", "-f", help="Job description file"
    ),
    company: str = typer.Option("", "--company", "-c", help="Company name"),
    publish: bool = typer.Option(False, "--publish", "-p", help="Publish job immediately"),
):
    """Create a new job posting."""
    from src.data.database import get_database_manager
    from src.data.repositories import get_job_repository
    from src.ml.nlp import get_jd_parser
    from src.data.models import JobCreate, SkillRequirement

    console.print(f"[yellow]Creating job: {title}[/yellow]")

    # Check database connection
    db_manager = get_database_manager()
    if not db_manager.check_sync_connection():
        console.print("[red]Error: Could not connect to MongoDB. Run 'init-db' first.[/red]")
        raise typer.Exit(1)

    jd_parser = get_jd_parser()
    job_repo = get_job_repository()

    # Get job description
    description = ""
    jd_result = None

    if description_file:
        if not description_file.exists():
            console.print(f"[red]Error: File not found: {description_file}[/red]")
            raise typer.Exit(1)

        console.print(f"  Reading job description from: {description_file}")

        # Parse the job description file
        try:
            jd_result = jd_parser.parse_file(str(description_file))
            description = jd_result.raw_text
            console.print("  [green]✓[/green] Job description parsed")
        except Exception as e:
            console.print(f"[red]Error parsing job description: {e}[/red]")
            raise typer.Exit(1)
    else:
        # Interactive mode - prompt for description
        console.print("[dim]Enter job description (press Ctrl+D or Ctrl+Z when done):[/dim]")
        try:
            lines = []
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            pass
        description = "\n".join(lines)

        if description.strip():
            # Parse the entered description
            jd_result = jd_parser.parse_text(description)

    if not description.strip():
        console.print("[red]Error: Job description cannot be empty.[/red]")
        raise typer.Exit(1)

    # Build job create data
    try:
        # Use parsed data if available
        if jd_result:
            skill_requirements = [
                SkillRequirement(name=skill.lower(), is_required=True)
                for skill in jd_result.required_skills
            ] + [
                SkillRequirement(name=skill.lower(), is_required=False)
                for skill in jd_result.preferred_skills
            ]

            job_data = JobCreate(
                title=jd_result.title or title,
                description=description,
                responsibilities=jd_result.responsibilities,
                company_name=company or jd_result.company_name or "Unknown Company",
                skill_requirements=skill_requirements if skill_requirements else None,
            )
        else:
            job_data = JobCreate(
                title=title,
                description=description,
                company_name=company or "Unknown Company",
            )

        # Create the job
        job = job_repo.create_from_schema(job_data)
        console.print(f"  [green]✓[/green] Job created with ID: [cyan]{job.id}[/cyan]")

        # Publish if requested
        if publish:
            job_repo.publish(job.id)
            console.print("  [green]✓[/green] Job published")

        # Show summary
        console.print("\n[bold]Job Summary:[/bold]")
        console.print(f"  Title: {job.title}")
        console.print(f"  Company: {job.company_name}")
        if jd_result:
            console.print(f"  Required Skills: {len(jd_result.required_skills)}")
            console.print(f"  Preferred Skills: {len(jd_result.preferred_skills)}")
            console.print(f"  Responsibilities: {len(jd_result.responsibilities)}")

        console.print("\n[green]Job created successfully![/green]")

    except Exception as e:
        console.print(f"[red]Error creating job: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def warmup():
    """Load and warm up ML models for faster subsequent operations."""
    from src.utils.config import get_settings
    from rich.progress import Progress, SpinnerColumn, TextColumn

    console.print("[yellow]Warming up ML models...[/yellow]")

    settings = get_settings()
    console.print(f"  Device: [cyan]{settings.ml.device}[/cyan]")
    console.print(f"  Embedding Model: [cyan]{settings.ml.embedding_model}[/cyan]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Load embedding model
        task = progress.add_task("Loading embedding model...", total=None)
        try:
            from src.ml.embeddings import get_embedding_model
            embedding_model = get_embedding_model()
            # Warm up with a test encode
            embedding_model.encode("test warmup sentence")
            progress.update(task, description="[green]✓[/green] Embedding model loaded")
        except Exception as e:
            progress.update(task, description=f"[red]✗[/red] Embedding model failed: {e}")

        # Load NLP parser
        task2 = progress.add_task("Loading NLP parser...", total=None)
        try:
            from src.ml.nlp import get_resume_parser
            parser = get_resume_parser()
            progress.update(task2, description="[green]✓[/green] NLP parser loaded")
        except Exception as e:
            progress.update(task2, description=f"[red]✗[/red] NLP parser failed: {e}")

        # Load bias detector
        task3 = progress.add_task("Loading bias detector...", total=None)
        try:
            from src.ml.ethics import get_bias_detector
            detector = get_bias_detector()
            progress.update(task3, description="[green]✓[/green] Bias detector loaded")
        except Exception as e:
            progress.update(task3, description=f"[red]✗[/red] Bias detector failed: {e}")

        # Load explainer
        task4 = progress.add_task("Loading explainer...", total=None)
        try:
            from src.ml.explainability import get_match_explainer
            explainer = get_match_explainer()
            progress.update(task4, description="[green]✓[/green] Explainer loaded")
        except Exception as e:
            progress.update(task4, description=f"[red]✗[/red] Explainer failed: {e}")

    console.print("\n[green]Model warmup completed![/green]")


@app.command()
def match(
    job_id: str = typer.Argument(..., help="Job ID to match candidates against"),
    candidate_id: Optional[str] = typer.Option(None, "--candidate", "-c", help="Specific candidate ID"),
    top_n: int = typer.Option(10, "--top", "-n", help="Number of top matches to show"),
    threshold: float = typer.Option(0.0, "--threshold", "-t", help="Minimum score threshold"),
):
    """Match candidates against a job posting."""
    from src.data.database import get_database_manager
    from src.data.repositories import get_job_repository, get_candidate_repository, get_match_repository
    from src.ml.nlp import get_resume_parser, get_jd_parser
    from src.core.matching import get_matching_engine

    console.print(f"[yellow]Matching candidates for job: {job_id}[/yellow]")

    # Check database connection
    db_manager = get_database_manager()
    if not db_manager.check_sync_connection():
        console.print("[red]Error: Could not connect to MongoDB. Run 'init-db' first.[/red]")
        raise typer.Exit(1)

    job_repo = get_job_repository()
    candidate_repo = get_candidate_repository()
    match_repo = get_match_repository()

    # Get the job
    job = job_repo.get_by_id(job_id)
    if not job:
        console.print(f"[red]Error: Job not found: {job_id}[/red]")
        raise typer.Exit(1)

    console.print(f"  Job: [cyan]{job.title}[/cyan] at {job.company_name}")

    # Get candidates
    if candidate_id:
        candidates = [candidate_repo.get_by_id(candidate_id)]
        if not candidates[0]:
            console.print(f"[red]Error: Candidate not found: {candidate_id}[/red]")
            raise typer.Exit(1)
    else:
        candidates = candidate_repo.find({}, limit=100)

    if not candidates:
        console.print("[yellow]No candidates found in the database.[/yellow]")
        raise typer.Exit(0)

    console.print(f"  Found [cyan]{len(candidates)}[/cyan] candidate(s)")

    # Initialize matching engine
    matching_engine = get_matching_engine()
    jd_parser = get_jd_parser()

    # Parse job description
    jd_result = jd_parser.parse_text(job.description)
    jd_result.title = job.title
    jd_result.company_name = job.company_name

    # Add skill requirements from job model
    if job.skill_requirements:
        jd_result.required_skills = [
            s.name for s in job.skill_requirements if s.is_required
        ]
        jd_result.preferred_skills = [
            s.name for s in job.skill_requirements if not s.is_required
        ]

    # Match each candidate
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Matching candidates...", total=len(candidates))

        for candidate in candidates:
            try:
                # Build a pseudo resume result from candidate data
                from src.ml.nlp import ResumeParseResult
                resume_result = ResumeParseResult(
                    success=True,
                    contact={
                        "full_name": f"{candidate.contact.first_name} {candidate.contact.last_name}",
                        "email": candidate.contact.email,
                    },
                    skills=[{"name": s.name, "proficiency": s.proficiency_level or "intermediate"}
                            for s in candidate.skills],
                    experience=[],
                    education=[],
                    total_experience_years=sum(
                        (e.end_date.year if e.end_date else 2024) - e.start_date.year
                        for e in candidate.experience if e.start_date
                    ) if candidate.experience else 0,
                    highest_education=candidate.education[0].degree if candidate.education else None,
                )

                # Run matching
                match_result = matching_engine.match(resume_result, jd_result)
                results.append((candidate, match_result))

            except Exception as e:
                console.print(f"[dim]  Error matching {candidate.contact.email}: {e}[/dim]")

            progress.update(task, advance=1)

    # Filter by threshold and sort
    results = [(c, r) for c, r in results if r.overall_score >= threshold]
    results.sort(key=lambda x: x[1].overall_score, reverse=True)
    results = results[:top_n]

    # Display results
    console.print("\n[bold]Match Results:[/bold]")

    if not results:
        console.print("[yellow]No candidates matched the threshold.[/yellow]")
        raise typer.Exit(0)

    table = Table(title=f"Top {len(results)} Matches for {job.title}")
    table.add_column("Rank", style="dim", width=4)
    table.add_column("Candidate", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Level", justify="center")
    table.add_column("Skills", justify="right")
    table.add_column("Bias Check", justify="center")

    for i, (candidate, match_result) in enumerate(results, 1):
        name = f"{candidate.contact.first_name} {candidate.contact.last_name}"
        score = f"{match_result.overall_score:.1%}"
        level = match_result.score_level.value.upper()
        skills = f"{len(match_result.matched_skills)}/{len(match_result.skill_matches)}"

        # Color based on score level
        if match_result.score_level.value == "excellent":
            level_color = "green"
        elif match_result.score_level.value == "good":
            level_color = "blue"
        elif match_result.score_level.value == "fair":
            level_color = "yellow"
        else:
            level_color = "red"

        bias_status = "✓" if not match_result.bias_check or not match_result.bias_check.potential_bias_detected else "⚠"

        table.add_row(
            str(i),
            name,
            score,
            f"[{level_color}]{level}[/{level_color}]",
            skills,
            bias_status,
        )

    console.print(table)

    # Show top match explanation
    if results:
        top_candidate, top_match = results[0]
        console.print(f"\n[bold]Top Match Explanation:[/bold]")
        if top_match.explanation:
            console.print(f"  {top_match.explanation.summary}")
            if top_match.explanation.strengths:
                console.print("  [green]Strengths:[/green]")
                for s in top_match.explanation.strengths[:3]:
                    console.print(f"    • {s}")
            if top_match.explanation.gaps:
                console.print("  [yellow]Gaps:[/yellow]")
                for g in top_match.explanation.gaps[:3]:
                    console.print(f"    • {g}")


@app.command()
def list_jobs(
    status: Optional[str] = typer.Option(None, "--status", "-s", help="Filter by status (draft/open/closed/filled)"),
    limit: int = typer.Option(20, "--limit", "-l", help="Maximum number of jobs to show"),
):
    """List all jobs in the database."""
    from src.data.database import get_database_manager
    from src.data.repositories import get_job_repository
    from src.utils.constants import JobStatus

    # Check database connection
    db_manager = get_database_manager()
    if not db_manager.check_sync_connection():
        console.print("[red]Error: Could not connect to MongoDB. Run 'init-db' first.[/red]")
        raise typer.Exit(1)

    job_repo = get_job_repository()

    # Build query
    query = {}
    if status:
        try:
            job_status = JobStatus[status.upper()]
            query["status"] = job_status.value
        except KeyError:
            console.print(f"[red]Invalid status: {status}[/red]")
            console.print(f"[dim]Valid statuses: draft, open, paused, closed, filled[/dim]")
            raise typer.Exit(1)

    jobs = job_repo.find(query, limit=limit, sort_by="created_at", sort_order=-1)

    if not jobs:
        console.print("[yellow]No jobs found.[/yellow]")
        raise typer.Exit(0)

    table = Table(title=f"Jobs ({len(jobs)} total)")
    table.add_column("ID", style="dim", width=24)
    table.add_column("Title", style="cyan")
    table.add_column("Company")
    table.add_column("Status", justify="center")
    table.add_column("Skills", justify="right")

    for job in jobs:
        status_color = {
            "draft": "dim",
            "open": "green",
            "paused": "yellow",
            "closed": "red",
            "filled": "blue",
        }.get(job.status, "white")

        skill_count = len(job.skill_requirements) if job.skill_requirements else 0

        table.add_row(
            str(job.id),
            job.title[:40] + "..." if len(job.title) > 40 else job.title,
            job.company_name[:20] + "..." if len(job.company_name) > 20 else job.company_name,
            f"[{status_color}]{job.status.upper()}[/{status_color}]",
            str(skill_count),
        )

    console.print(table)


@app.command()
def list_candidates(
    limit: int = typer.Option(20, "--limit", "-l", help="Maximum number of candidates to show"),
):
    """List all candidates in the database."""
    from src.data.database import get_database_manager
    from src.data.repositories import get_candidate_repository

    # Check database connection
    db_manager = get_database_manager()
    if not db_manager.check_sync_connection():
        console.print("[red]Error: Could not connect to MongoDB. Run 'init-db' first.[/red]")
        raise typer.Exit(1)

    candidate_repo = get_candidate_repository()
    candidates = candidate_repo.find({}, limit=limit, sort_by="created_at", sort_order=-1)

    if not candidates:
        console.print("[yellow]No candidates found.[/yellow]")
        raise typer.Exit(0)

    table = Table(title=f"Candidates ({len(candidates)} total)")
    table.add_column("ID", style="dim", width=24)
    table.add_column("Name", style="cyan")
    table.add_column("Email")
    table.add_column("Skills", justify="right")
    table.add_column("Status", justify="center")

    for candidate in candidates:
        name = f"{candidate.contact.first_name} {candidate.contact.last_name}"
        email = candidate.contact.email or "N/A"
        skill_count = len(candidate.skills) if candidate.skills else 0

        status_color = {
            "new": "cyan",
            "screening": "yellow",
            "shortlisted": "green",
            "interviewing": "blue",
            "offered": "magenta",
            "hired": "green",
            "rejected": "red",
            "withdrawn": "dim",
        }.get(candidate.status.lower() if hasattr(candidate, 'status') else "new", "white")

        table.add_row(
            str(candidate.id),
            name[:30] + "..." if len(name) > 30 else name,
            email[:30] + "..." if len(email) > 30 else email,
            str(skill_count),
            f"[{status_color}]NEW[/{status_color}]",
        )

    console.print(table)


@app.command()
def show_job(
    job_id: str = typer.Argument(..., help="Job ID to display"),
):
    """Show detailed information about a specific job."""
    from src.data.database import get_database_manager
    from src.data.repositories import get_job_repository

    db_manager = get_database_manager()
    if not db_manager.check_sync_connection():
        console.print("[red]Error: Could not connect to MongoDB.[/red]")
        raise typer.Exit(1)

    job_repo = get_job_repository()
    job = job_repo.get_by_id(job_id)

    if not job:
        console.print(f"[red]Job not found: {job_id}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold cyan]Job Details[/bold cyan]")
    console.print(f"[dim]{'─' * 50}[/dim]")
    console.print(f"[bold]ID:[/bold] {job.id}")
    console.print(f"[bold]Title:[/bold] {job.title}")
    console.print(f"[bold]Company:[/bold] {job.company_name}")
    console.print(f"[bold]Status:[/bold] {job.status.upper()}")
    console.print(f"[bold]Created:[/bold] {job.created_at}")

    if job.description:
        console.print(f"\n[bold]Description:[/bold]")
        console.print(f"  {job.description[:500]}{'...' if len(job.description) > 500 else ''}")

    if job.skill_requirements:
        required = [s.name for s in job.skill_requirements if s.is_required]
        preferred = [s.name for s in job.skill_requirements if not s.is_required]
        if required:
            console.print(f"\n[bold]Required Skills:[/bold] {', '.join(required)}")
        if preferred:
            console.print(f"[bold]Preferred Skills:[/bold] {', '.join(preferred)}")

    if job.responsibilities:
        console.print(f"\n[bold]Responsibilities:[/bold]")
        for r in job.responsibilities[:5]:
            console.print(f"  • {r}")


@app.command()
def show_candidate(
    candidate_id: str = typer.Argument(..., help="Candidate ID to display"),
):
    """Show detailed information about a specific candidate."""
    from src.data.database import get_database_manager
    from src.data.repositories import get_candidate_repository

    db_manager = get_database_manager()
    if not db_manager.check_sync_connection():
        console.print("[red]Error: Could not connect to MongoDB.[/red]")
        raise typer.Exit(1)

    candidate_repo = get_candidate_repository()
    candidate = candidate_repo.get_by_id(candidate_id)

    if not candidate:
        console.print(f"[red]Candidate not found: {candidate_id}[/red]")
        raise typer.Exit(1)

    name = f"{candidate.contact.first_name} {candidate.contact.last_name}"

    console.print(f"\n[bold cyan]Candidate Details[/bold cyan]")
    console.print(f"[dim]{'─' * 50}[/dim]")
    console.print(f"[bold]ID:[/bold] {candidate.id}")
    console.print(f"[bold]Name:[/bold] {name}")
    console.print(f"[bold]Email:[/bold] {candidate.contact.email}")
    if candidate.contact.phone:
        console.print(f"[bold]Phone:[/bold] {candidate.contact.phone}")
    if candidate.contact.linkedin_url:
        console.print(f"[bold]LinkedIn:[/bold] {candidate.contact.linkedin_url}")
    console.print(f"[bold]Created:[/bold] {candidate.created_at}")

    if candidate.skills:
        console.print(f"\n[bold]Skills ({len(candidate.skills)}):[/bold]")
        skills_by_category = {}
        for s in candidate.skills:
            cat = s.category or "Other"
            if cat not in skills_by_category:
                skills_by_category[cat] = []
            skills_by_category[cat].append(s.name)
        for cat, skills in skills_by_category.items():
            console.print(f"  [cyan]{cat}:[/cyan] {', '.join(skills[:10])}")

    if candidate.experience:
        console.print(f"\n[bold]Experience ({len(candidate.experience)} positions):[/bold]")
        for exp in candidate.experience[:3]:
            console.print(f"  • {exp.job_title} at {exp.company}")


@app.command()
def audit_logs(
    action: Optional[str] = typer.Option(None, "--action", "-a", help="Filter by action type"),
    candidate_id: Optional[str] = typer.Option(None, "--candidate", "-c", help="Filter by candidate ID"),
    job_id: Optional[str] = typer.Option(None, "--job", "-j", help="Filter by job ID"),
    limit: int = typer.Option(20, "--limit", "-l", help="Maximum logs to show"),
    compliance_only: bool = typer.Option(False, "--compliance", help="Show only compliance-relevant logs"),
):
    """View audit logs for compliance and debugging."""
    from src.data.database import get_database_manager
    from src.data.repositories import get_audit_repository

    db_manager = get_database_manager()
    if not db_manager.check_sync_connection():
        console.print("[red]Error: Could not connect to MongoDB.[/red]")
        raise typer.Exit(1)

    audit_repo = get_audit_repository()

    # Build query
    query = {}
    if action:
        query["action"] = action
    if candidate_id:
        from bson import ObjectId
        if not ObjectId.is_valid(candidate_id):
            console.print(f"[red]Error: Invalid candidate ID format: {candidate_id}[/red]")
            raise typer.Exit(1)
        query["related_candidate_id"] = ObjectId(candidate_id)
    if job_id:
        from bson import ObjectId
        if not ObjectId.is_valid(job_id):
            console.print(f"[red]Error: Invalid job ID format: {job_id}[/red]")
            raise typer.Exit(1)
        query["related_job_id"] = ObjectId(job_id)
    if compliance_only:
        query["compliance_relevant"] = True

    logs = audit_repo.find(query, limit=limit, sort_by="created_at", sort_order=-1)

    if not logs:
        console.print("[yellow]No audit logs found.[/yellow]")
        raise typer.Exit(0)

    table = Table(title=f"Audit Logs ({len(logs)} entries)")
    table.add_column("Timestamp", style="dim", width=19)
    table.add_column("Action", style="cyan")
    table.add_column("Resource")
    table.add_column("Actor")
    table.add_column("Compliance", justify="center")

    for log in logs:
        timestamp = log.created_at.strftime("%Y-%m-%d %H:%M:%S") if log.created_at else "N/A"
        resource = f"{log.resource.resource_type}:{log.resource.resource_id[:8]}..." if log.resource else "N/A"
        actor = log.actor.actor_id if log.actor else "system"
        compliance = "✓" if log.compliance_relevant else ""

        table.add_row(timestamp, log.action, resource, actor, compliance)

    console.print(table)


@app.command()
def stats():
    """Show database and system statistics."""
    from src.data.database import get_database_manager
    from src.data.repositories import (
        get_job_repository,
        get_candidate_repository,
        get_match_repository,
        get_audit_repository,
    )

    db_manager = get_database_manager()
    if not db_manager.check_sync_connection():
        console.print("[red]Error: Could not connect to MongoDB.[/red]")
        raise typer.Exit(1)

    console.print("[bold cyan]System Statistics[/bold cyan]")
    console.print(f"[dim]{'─' * 50}[/dim]")

    job_repo = get_job_repository()
    candidate_repo = get_candidate_repository()
    match_repo = get_match_repository()
    audit_repo = get_audit_repository()

    # Count documents
    job_count = job_repo.count({})
    candidate_count = candidate_repo.count({})
    match_count = match_repo.count({})
    audit_count = audit_repo.count({})

    table = Table(title="Database Counts")
    table.add_column("Collection", style="cyan")
    table.add_column("Count", justify="right", style="green")

    table.add_row("Jobs", str(job_count))
    table.add_row("Candidates", str(candidate_count))
    table.add_row("Matches", str(match_count))
    table.add_row("Audit Logs", str(audit_count))

    console.print(table)

    # Job status breakdown
    if job_count > 0:
        job_statuses = job_repo.get_status_counts()
        console.print("\n[bold]Jobs by Status:[/bold]")
        for status, count in job_statuses.items():
            console.print(f"  {status}: {count}")

    # Match score statistics
    if match_count > 0:
        try:
            score_stats = match_repo.get_score_statistics()
            console.print("\n[bold]Match Score Statistics:[/bold]")
            console.print(f"  Average Score: {score_stats.get('avg_score', 0):.1%}")
            console.print(f"  Min Score: {score_stats.get('min_score', 0):.1%}")
            console.print(f"  Max Score: {score_stats.get('max_score', 0):.1%}")
        except Exception:
            pass


@app.command()
def health_check():
    """Check system health and component status."""
    from src.data.database import get_database_manager
    from src.utils.config import get_settings

    console.print("[bold cyan]System Health Check[/bold cyan]")
    console.print(f"[dim]{'─' * 50}[/dim]")

    settings = get_settings()
    all_healthy = True

    # Check MongoDB
    console.print("\n[bold]Database:[/bold]")
    db_manager = get_database_manager()
    if db_manager.check_sync_connection():
        console.print("  [green]✓[/green] MongoDB connected")
        console.print(f"    Host: {settings.database.host}:{settings.database.port}")
        console.print(f"    Database: {settings.database.name}")
    else:
        console.print("  [red]✗[/red] MongoDB not connected")
        all_healthy = False

    # Check ML components
    console.print("\n[bold]ML Components:[/bold]")

    # Embedding model
    try:
        from src.ml.embeddings import get_embedding_model
        model = get_embedding_model()
        console.print(f"  [green]✓[/green] Embedding model ready")
        console.print(f"    Model: {settings.ml.embedding_model}")
        console.print(f"    Device: {settings.ml.device}")
    except Exception as e:
        console.print(f"  [yellow]○[/yellow] Embedding model not loaded (will load on first use)")

    # Bias detector
    try:
        from src.ml.ethics import get_bias_detector
        detector = get_bias_detector()
        console.print("  [green]✓[/green] Bias detector ready")
    except Exception as e:
        console.print(f"  [yellow]○[/yellow] Bias detector not loaded")

    # Explainer
    try:
        from src.ml.explainability import get_match_explainer
        explainer = get_match_explainer()
        console.print("  [green]✓[/green] Explainer ready")
    except Exception as e:
        console.print(f"  [yellow]○[/yellow] Explainer not loaded")

    # Summary
    console.print(f"\n[dim]{'─' * 50}[/dim]")
    if all_healthy:
        console.print("[green]All critical systems operational.[/green]")
    else:
        console.print("[red]Some systems require attention.[/red]")


@app.command()
def bias_report(
    job_id: Optional[str] = typer.Option(None, "--job", "-j", help="Generate report for specific job"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save report to file"),
):
    """Generate a bias analysis report for auditing purposes."""
    from src.data.database import get_database_manager
    from src.data.repositories import get_match_repository, get_job_repository
    from src.ml.ethics import get_bias_detector
    from datetime import datetime

    db_manager = get_database_manager()
    if not db_manager.check_sync_connection():
        console.print("[red]Error: Could not connect to MongoDB.[/red]")
        raise typer.Exit(1)

    match_repo = get_match_repository()
    job_repo = get_job_repository()
    bias_detector = get_bias_detector()

    console.print("[bold cyan]Bias Analysis Report[/bold cyan]")
    console.print(f"[dim]Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]")
    console.print(f"[dim]{'─' * 50}[/dim]")

    # Get matches to analyze
    query = {}
    if job_id:
        from bson import ObjectId
        if not ObjectId.is_valid(job_id):
            console.print(f"[red]Error: Invalid job ID format: {job_id}[/red]")
            raise typer.Exit(1)
        query["job_id"] = ObjectId(job_id)
        job = job_repo.get_by_id(job_id)
        if job:
            console.print(f"\n[bold]Job:[/bold] {job.title} at {job.company_name}")

    matches = match_repo.find(query, limit=100)

    if not matches:
        console.print("[yellow]No matches found to analyze.[/yellow]")
        raise typer.Exit(0)

    console.print(f"\n[bold]Matches Analyzed:[/bold] {len(matches)}")

    # Analyze bias detection results
    bias_detected_count = 0
    bias_categories = {}

    for match in matches:
        if match.bias_check and match.bias_check.potential_bias_detected:
            bias_detected_count += 1
            for attr in match.bias_check.protected_attributes_found:
                bias_categories[attr] = bias_categories.get(attr, 0) + 1

    bias_rate = bias_detected_count / len(matches) * 100 if matches else 0

    console.print(f"\n[bold]Bias Detection Summary:[/bold]")
    console.print(f"  Matches with potential bias indicators: {bias_detected_count} ({bias_rate:.1f}%)")

    if bias_categories:
        console.print(f"\n[bold]Protected Attributes Detected:[/bold]")
        for attr, count in sorted(bias_categories.items(), key=lambda x: x[1], reverse=True):
            console.print(f"  • {attr}: {count} occurrences")

    # Score distribution
    scores = [m.overall_score for m in matches if m.overall_score is not None]
    if scores:
        avg_score = sum(scores) / len(scores)
        console.print(f"\n[bold]Score Distribution:[/bold]")
        console.print(f"  Average Score: {avg_score:.1%}")
        console.print(f"  Score Range: {min(scores):.1%} - {max(scores):.1%}")

    # Recommendations
    console.print(f"\n[bold]Recommendations:[/bold]")
    if bias_rate > 20:
        console.print("  [yellow]⚠[/yellow] High rate of bias indicators detected. Review resume parsing.")
    elif bias_rate > 5:
        console.print("  [yellow]○[/yellow] Some bias indicators found. Consider reviewing flagged matches.")
    else:
        console.print("  [green]✓[/green] Low bias indicator rate. System appears to be functioning well.")

    # Save to file if requested
    if output:
        report_content = f"""AI-ATS Bias Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 50}

Matches Analyzed: {len(matches)}
Bias Detected: {bias_detected_count} ({bias_rate:.1f}%)

Protected Attributes Found:
{chr(10).join(f'  - {k}: {v}' for k, v in bias_categories.items())}

Average Match Score: {avg_score:.1%}
"""
        output.write_text(report_content)
        console.print(f"\n[green]Report saved to: {output}[/green]")


@app.command()
def gui():
    """Launch the graphical user interface."""
    console.print("[yellow]Launching GUI...[/yellow]")
    from src.main import main

    main()


# =============================================================================
# Google Drive Integration Commands
# =============================================================================

@app.command()
def gdrive_auth():
    """Authenticate with Google Drive for importing resumes."""
    from src.services.google_drive_service import get_drive_service

    service = get_drive_service()

    if not service.is_available():
        console.print("[red]Google API libraries not installed.[/red]")
        console.print("\nInstall with:")
        console.print("  [cyan]pip install google-auth-oauthlib google-api-python-client[/cyan]")
        raise typer.Exit(1)

    if not service.has_credentials():
        console.print("[red]Credentials file not found: credentials.json[/red]")
        console.print("\n[bold]Setup Instructions:[/bold]")
        console.print("1. Go to [cyan]https://console.cloud.google.com[/cyan]")
        console.print("2. Create a project and enable Google Drive API")
        console.print("3. Create OAuth 2.0 credentials (Desktop application)")
        console.print("4. Download and save as [cyan]credentials.json[/cyan] in project root")
        raise typer.Exit(1)

    console.print("[yellow]Authenticating with Google Drive...[/yellow]")
    console.print("[dim]A browser window will open for authorization.[/dim]")

    if service.authenticate():
        console.print("[green]✓ Authentication successful![/green]")
        console.print("\nYou can now use:")
        console.print("  [cyan]ai-ats gdrive-list[/cyan] - List folders")
        console.print("  [cyan]ai-ats gdrive-import[/cyan] - Import resumes from a folder")
    else:
        console.print("[red]Authentication failed.[/red]")
        raise typer.Exit(1)


@app.command()
def gdrive_list(
    folder_id: str = typer.Option("root", "--folder", "-f", help="Parent folder ID"),
    show_files: bool = typer.Option(False, "--files", help="Also show resume files"),
):
    """List folders in Google Drive to find resume uploads."""
    from src.services.google_drive_service import get_drive_service

    service = get_drive_service()

    if not service.authenticate():
        console.print("[red]Authentication failed. Run 'gdrive-auth' first.[/red]")
        raise typer.Exit(1)

    console.print(f"[yellow]Listing folders in: {folder_id}[/yellow]\n")

    folders = service.list_folders(folder_id)

    if not folders:
        console.print("[dim]No folders found.[/dim]")
    else:
        table = Table(title="Folders")
        table.add_column("Name", style="cyan")
        table.add_column("ID", style="dim")
        table.add_column("Modified", style="green")

        for folder in folders:
            modified = folder.modified_time[:10] if folder.modified_time else "N/A"
            table.add_row(folder.name, folder.id, modified)

        console.print(table)

    if show_files and folder_id != "root":
        console.print(f"\n[yellow]Resume files in folder:[/yellow]")
        files = service.list_resume_files(folder_id)

        if not files:
            console.print("[dim]No resume files found.[/dim]")
        else:
            file_table = Table(title=f"Resume Files ({len(files)})")
            file_table.add_column("Name", style="cyan")
            file_table.add_column("Type", style="dim")
            file_table.add_column("Size", justify="right")

            for f in files[:20]:  # Show first 20
                size = f"{f.size / 1024:.1f} KB" if f.size else "N/A"
                file_table.add_row(f.name[:40], f.mime_type.split("/")[-1], size)

            console.print(file_table)
            if len(files) > 20:
                console.print(f"[dim]... and {len(files) - 20} more files[/dim]")


@app.command()
def gdrive_import(
    folder_id: str = typer.Argument(..., help="Google Drive folder ID containing resumes"),
    output_dir: Path = typer.Option(
        Path("data/imports"),
        "--output", "-o",
        help="Local directory to save downloaded files"
    ),
    process: bool = typer.Option(True, "--process/--no-process", help="Process resumes after download"),
    skip_existing: bool = typer.Option(True, "--skip-existing/--overwrite", help="Skip already downloaded files"),
):
    """
    Import resumes from a Google Drive folder.

    Use this to import resumes collected via Google Forms.

    Example:
        ai-ats gdrive-import 1ABC123xyz --output ./resumes
    """
    from src.services.google_drive_service import get_drive_service
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

    service = get_drive_service()

    if not service.authenticate():
        console.print("[red]Authentication failed. Run 'gdrive-auth' first.[/red]")
        raise typer.Exit(1)

    console.print(f"[yellow]Importing resumes from Google Drive...[/yellow]")
    console.print(f"  Folder ID: [cyan]{folder_id}[/cyan]")
    console.print(f"  Output: [cyan]{output_dir}[/cyan]\n")

    # List files first
    files = service.list_resume_files(folder_id)

    if not files:
        console.print("[yellow]No resume files found in folder.[/yellow]")
        console.print("\n[dim]Tips:[/dim]")
        console.print("  - Make sure you have the correct folder ID")
        console.print("  - Use 'gdrive-list --folder PARENT_ID' to find subfolders")
        console.print("  - For Google Forms, look for folder named 'Form Name (File responses)'")
        raise typer.Exit(0)

    console.print(f"Found [cyan]{len(files)}[/cyan] resume file(s)")

    # Download files
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading...", total=len(files))

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        downloaded = 0
        skipped = 0
        failed = 0

        for file in files:
            output_path = output_dir / file.name

            if output_path.exists() and skip_existing:
                skipped += 1
            elif service.download_file(file.id, output_path):
                downloaded += 1
            else:
                failed += 1

            progress.update(task, advance=1)

    # Summary
    console.print(f"\n[bold]Download Summary:[/bold]")
    console.print(f"  [green]✓ Downloaded:[/green] {downloaded}")
    console.print(f"  [yellow]○ Skipped:[/yellow] {skipped}")
    console.print(f"  [red]✗ Failed:[/red] {failed}")

    # Process if requested
    if process and downloaded > 0:
        console.print(f"\n[yellow]Processing downloaded resumes...[/yellow]")

        from src.data.database import get_database_manager
        from src.data.repositories import get_candidate_repository
        from src.ml.nlp import get_resume_parser
        from src.utils.constants import SUPPORTED_RESUME_FORMATS

        # Check database
        db_manager = get_database_manager()
        if not db_manager.check_sync_connection():
            console.print("[yellow]Database not available. Skipping import to database.[/yellow]")
            console.print(f"Files saved to: [cyan]{output_dir}[/cyan]")
            raise typer.Exit(0)

        parser = get_resume_parser()
        candidate_repo = get_candidate_repository()

        success_count = 0
        error_count = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            resume_files = list(output_dir.glob("*"))
            resume_files = [f for f in resume_files if f.suffix.lower() in SUPPORTED_RESUME_FORMATS]
            task = progress.add_task("Processing...", total=len(resume_files))

            for resume_file in resume_files:
                try:
                    result = parser.parse_file(str(resume_file))
                    if result.success:
                        candidate_data = parser.to_candidate_create(result)
                        if candidate_data:
                            candidate_repo.create_from_schema(candidate_data)
                        success_count += 1
                    else:
                        error_count += 1
                except Exception:
                    error_count += 1

                progress.update(task, advance=1)

        console.print(f"\n[bold]Processing Summary:[/bold]")
        console.print(f"  [green]✓ Imported:[/green] {success_count}")
        console.print(f"  [red]✗ Errors:[/red] {error_count}")

    console.print(f"\n[green]Import complete![/green]")
    console.print(f"Files saved to: [cyan]{output_dir}[/cyan]")


@app.command()
def gdrive_find_form(
    form_name: str = typer.Argument(..., help="Name of the Google Form"),
):
    """
    Find the Google Drive folder for a Google Form's file uploads.

    Google Forms stores uploaded files in a folder named:
    "{Form Name} (File responses)"
    """
    from src.services.google_drive_service import get_drive_service

    service = get_drive_service()

    if not service.authenticate():
        console.print("[red]Authentication failed. Run 'gdrive-auth' first.[/red]")
        raise typer.Exit(1)

    console.print(f"[yellow]Searching for form: {form_name}[/yellow]")

    folder_id = service.get_forms_response_folder(form_name)

    if folder_id:
        console.print(f"\n[green]✓ Found form responses folder![/green]")
        console.print(f"  Folder ID: [cyan]{folder_id}[/cyan]")
        console.print(f"\nTo import resumes, run:")
        console.print(f"  [cyan]ai-ats gdrive-import {folder_id}[/cyan]")
    else:
        console.print(f"\n[yellow]Folder not found.[/yellow]")
        console.print("\n[dim]Tips:[/dim]")
        console.print("  - Make sure you have access to the folder")
        console.print("  - The exact form name is required")
        console.print("  - Try 'gdrive-list' to browse folders manually")


if __name__ == "__main__":
    app()
