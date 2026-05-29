"""CLI commands for the built-in prompt library."""

import json as _json
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from distllm.prompts.library import (
    get_prompt,
    list_categories,
    list_by_category,
    search_prompts,
    SYSTEM_PROMPTS,
)

prompt_app = typer.Typer(help="Browse and use built-in system prompt templates")
_console = Console()


@prompt_app.command("list")
def prompt_list(
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Filter by category"),
    search: Optional[str] = typer.Option(None, "--search", "-s", help="Search prompts by name, description, or tags"),
    json: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """List available system prompt templates."""
    if search:
        results = search_prompts(search)
    elif category:
        results = list_by_category(category)
    else:
        results = list(SYSTEM_PROMPTS.values())

    if json:
        data = [
            {"id": p.id, "category": p.category, "name": p.name, "description": p.description, "tags": p.tags}
            for p in results
        ]
        _console.print(_json.dumps(data, indent=2))
        return

    if not results:
        _console.print("[yellow]No prompts found matching your criteria.[/yellow]")
        raise typer.Exit()

    table = Table(title=f"Prompt Library ({len(results)} prompts)")
    table.add_column("ID", style="cyan")
    table.add_column("Category", style="magenta")
    table.add_column("Name")
    table.add_column("Description")
    table.add_column("Tags")

    for p in results:
        tags_str = ", ".join(p.tags[:3])
        if len(p.tags) > 3:
            tags_str += "..."
        table.add_row(p.id, p.category, p.name, p.description[:60] + ("..." if len(p.description) > 60 else ""), tags_str)

    _console.print(table)


@prompt_app.command("show")
def prompt_show(
    prompt_id: str = typer.Argument(..., help="Prompt ID (e.g. code-review, summarization)"),
    json: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Show full details of a specific prompt template."""
    prompt = get_prompt(prompt_id)
    if prompt is None:
        _console.print(f"[red]Prompt '{prompt_id}' not found. Use 'distllm prompt list' to see available prompts.[/red]")
        raise typer.Exit(1)

    if json:
        _console.print(_json.dumps({
            "id": prompt.id,
            "category": prompt.category,
            "name": prompt.name,
            "description": prompt.description,
            "prompt": prompt.prompt,
            "tags": prompt.tags,
            "version": prompt.version,
        }, indent=2))
        return

    _console.print(f"[bold cyan]ID:[/bold cyan]          {prompt.id}")
    _console.print(f"[bold cyan]Name:[/bold cyan]        {prompt.name}")
    _console.print(f"[bold cyan]Category:[/bold cyan]    {prompt.category}")
    _console.print(f"[bold cyan]Description:[/bold cyan] {prompt.description}")
    _console.print(f"[bold cyan]Tags:[/bold cyan]        {', '.join(prompt.tags)}")
    _console.print(f"[bold cyan]Version:[/bold cyan]     {prompt.version}")
    _console.print()
    _console.print("[bold]Prompt Content:[/bold]")
    _console.print(prompt.prompt)


@prompt_app.command("categories")
def prompt_categories(
    json: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """List all prompt categories."""
    cats = list_categories()
    if json:
        _console.print(_json.dumps(cats, indent=2))
        return

    table = Table(title=f"Prompt Categories ({len(cats)})")
    table.add_column("Category", style="cyan")
    table.add_column("Count")
    for cat in cats:
        count = len(list_by_category(cat))
        table.add_row(cat, str(count))
    _console.print(table)


@prompt_app.command("use")
def prompt_use(
    prompt_id: str = typer.Argument(..., help="Prompt ID to use"),
    json: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Print the system prompt content for use in your chat client.
    
    The prompt is printed to stdout so you can pipe it or copy it.
    """
    prompt = get_prompt(prompt_id)
    if prompt is None:
        _console.print(f"[red]Prompt '{prompt_id}' not found.[/red]")
        raise typer.Exit(1)

    if json:
        _console.print(_json.dumps({
            "role": "system",
            "content": prompt.prompt,
            "id": prompt.id,
            "name": prompt.name,
        }, indent=2))
    else:
        _console.print(prompt.prompt)
