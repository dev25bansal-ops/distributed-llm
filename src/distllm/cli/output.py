"""Standardized CLI output formatting for DistLLM.

All CLI commands should use these functions instead of ``print()`` or
``console.print()`` directly.  This provides a single point of control
for HUMAN, JSON, and YAML output modes.

Usage::

    from distllm.cli.output import print_table, print_panel, OutputFormat

    print_table(["Name", "Status"], [["node-1", "ok"], ["node-2", "fail"]])
    print_error("Connection refused", hint="Is the coordinator running?")
"""

from __future__ import annotations

import enum
import json
import sys
from typing import Any, Sequence

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

# ---------------------------------------------------------------------------
# Reusable console instance
# ---------------------------------------------------------------------------

_console = Console()

# ---------------------------------------------------------------------------
# OutputFormat enum
# ---------------------------------------------------------------------------


class OutputFormat(str, enum.Enum):
    """Supported CLI output formats."""

    HUMAN = "human"
    JSON = "json"
    YAML = "yaml"


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


def detect_format(args: Sequence[str] | None = None) -> OutputFormat:
    """Read the ``--output`` flag from CLI arguments.

    Parameters
    ----------
    args :
        Argument list to scan.  Defaults to ``sys.argv[1:]``.

    Returns
    -------
    OutputFormat
        HUMAN if no ``--output`` flag is found, otherwise the
        corresponding enum member.  Invalid values fall back to HUMAN.
    """
    if args is None:
        args = sys.argv[1:]

    # Scan for ``--output <value>`` or ``--output=<value>``
    for i, arg in enumerate(args):
        if arg == "--output" and i + 1 < len(args):
            raw = args[i + 1].lower()
            for fmt in OutputFormat:
                if fmt.value == raw:
                    return fmt
            return OutputFormat.HUMAN
        if arg.startswith("--output="):
            raw = arg.split("=", 1)[1].lower()
            for fmt in OutputFormat:
                if fmt.value == raw:
                    return fmt
            return OutputFormat.HUMAN

    return OutputFormat.HUMAN


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rich_style_for_status(status: str) -> str:
    """Map a status string to a Rich style."""
    mapping = {
        "ok": "green",
        "pass": "green",
        "healthy": "green",
        "success": "green",
        "warn": "yellow",
        "warning": "yellow",
        "error": "red",
        "fail": "red",
        "critical": "red bold",
        "info": "cyan",
        "skip": "dim",
        "pending": "blue",
    }
    return mapping.get(status.lower(), "")


def _serialize(obj: Any) -> Any:
    """Recursively convert rich types to plain JSON-safe objects."""
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    return obj


# ---------------------------------------------------------------------------
# Public output functions
# ---------------------------------------------------------------------------


def print_table(
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    title: str = "",
    format: OutputFormat = OutputFormat.HUMAN,
    console: Console | None = None,
) -> None:
    """Print a table in the requested output format.

    Parameters
    ----------
    columns :
        Column header labels.
    rows :
        Row data, one sequence per row.
    title :
        Optional table title (HUMAN mode only).
    format :
        Output format.
    console :
        Rich console to write to (default shared instance).
    """
    c = console or _console

    if format == OutputFormat.HUMAN:
        table = Table(title=title or None)
        for col in columns:
            table.add_column(col)
        for row in rows:
            table.add_row(*[str(v) for v in row])
        c.print(table)
        return

    data = [dict(zip(columns, row)) for row in rows]
    if format == OutputFormat.JSON:
        _emit_json(data, console=c)
    elif format == OutputFormat.YAML:
        _emit_yaml(data, console=c)


def print_tree(
    data: dict[str, Any],
    *,
    label: str = "",
    format: OutputFormat = OutputFormat.HUMAN,
    console: Console | None = None,
) -> None:
    """Print a nested dictionary as a tree (HUMAN) or as JSON/YAML.

    Parameters
    ----------
    data :
        Nested dictionary to display.
    label :
        Root label (HUMAN mode only).
    format :
        Output format.
    console :
        Rich console to write to (default shared instance).
    """
    c = console or _console

    if format == OutputFormat.HUMAN:
        tree = Tree(label or "root")
        _populate_tree(tree, data)
        c.print(tree)
        return

    if format == OutputFormat.JSON:
        _emit_json(data, console=c)
    elif format == OutputFormat.YAML:
        _emit_yaml(data, console=c)


def _populate_tree(tree: Tree, data: dict[str, Any], max_depth: int = 8) -> None:
    """Recursively populate a Rich Tree from a nested dict."""
    if max_depth <= 0:
        tree.add("...")
        return
    for key, value in data.items():
        if isinstance(value, dict):
            branch = tree.add(f"[bold]{key}[/bold]")
            _populate_tree(branch, value, max_depth - 1)
        elif isinstance(value, list):
            branch = tree.add(f"[bold]{key}[/bold] ({len(value)} items)")
            for i, item in enumerate(value[:20]):
                if isinstance(item, dict):
                    sub = branch.add(f"[dim]{i}[/dim]")
                    _populate_tree(sub, item, max_depth - 1)
                else:
                    branch.add(str(item))
            if len(value) > 20:
                branch.add(f"[dim]... {len(value) - 20} more[/dim]")
        else:
            rendered = str(value)
            # Truncate very long values
            if len(rendered) > 200:
                rendered = rendered[:197] + "..."
            tree.add(f"{key}: {rendered}")


def print_panel(
    title: str,
    content: str,
    *,
    style: str = "",
    format: OutputFormat = OutputFormat.HUMAN,
    console: Console | None = None,
) -> None:
    """Print a bordered panel (HUMAN) or a JSON/YAML object.

    Parameters
    ----------
    title :
        Panel title text.
    content :
        Body text.
    style :
        Rich style string (e.g. ``"red"``, ``"green"``).  Ignored
        in JSON/YAML mode.
    format :
        Output format.
    console :
        Rich console to write to (default shared instance).
    """
    c = console or _console

    if format == OutputFormat.HUMAN:
        panel = Panel(content, title=title, border_style=style or "blue")
        c.print(panel)
        return

    data = {"title": title, "content": content, "style": style}
    if format == OutputFormat.JSON:
        _emit_json(data, console=c)
    elif format == OutputFormat.YAML:
        _emit_yaml(data, console=c)


def print_json(
    data: Any,
    *,
    indent: int = 2,
    console: Console | None = None,
) -> None:
    """Print data as indented JSON regardless of the current output format.

    Parameters
    ----------
    data :
        Data to serialize.
    indent :
        JSON indentation level.
    console :
        Rich console to write to (default shared instance).
    """
    c = console or _console
    _emit_json(data, indent=indent, console=c)


def print_error(
    message: str,
    hint: str | None = None,
    *,
    format: OutputFormat = OutputFormat.HUMAN,
    console: Console | None = None,
) -> None:
    """Print an error message.

    Parameters
    ----------
    message :
        Error description.
    hint :
        Optional suggestion for how to resolve the issue.
    format :
        Output format.
    console :
        Rich console to write to (default shared instance).
    """
    c = console or _console

    if format == OutputFormat.HUMAN:
        text = message
        if hint:
            text = f"{message}\n\n[hint]{hint}[/hint]"
        panel = Panel(text, title="Error", border_style="red")
        c.print(panel)
        return

    data: dict[str, Any] = {"level": "error", "message": message}
    if hint:
        data["hint"] = hint
    if format == OutputFormat.JSON:
        _emit_json(data, console=c)
    elif format == OutputFormat.YAML:
        _emit_yaml(data, console=c)


def print_success(
    message: str,
    *,
    format: OutputFormat = OutputFormat.HUMAN,
    console: Console | None = None,
) -> None:
    """Print a success message.

    Parameters
    ----------
    message :
        Success text to display.
    format :
        Output format.
    console :
        Rich console to write to (default shared instance).
    """
    c = console or _console

    if format == OutputFormat.HUMAN:
        panel = Panel(message, title="Success", border_style="green")
        c.print(panel)
        return

    data = {"level": "success", "message": message}
    if format == OutputFormat.JSON:
        _emit_json(data, console=c)
    elif format == OutputFormat.YAML:
        _emit_yaml(data, console=c)


def print_warning(
    message: str,
    *,
    format: OutputFormat = OutputFormat.HUMAN,
    console: Console | None = None,
) -> None:
    """Print a warning message.

    Parameters
    ----------
    message :
        Warning text to display.
    format :
        Output format.
    console :
        Rich console to write to (default shared instance).
    """
    c = console or _console

    if format == OutputFormat.HUMAN:
        panel = Panel(message, title="Warning", border_style="yellow")
        c.print(panel)
        return

    data = {"level": "warning", "message": message}
    if format == OutputFormat.JSON:
        _emit_json(data, console=c)
    elif format == OutputFormat.YAML:
        _emit_yaml(data, console=c)


# ---------------------------------------------------------------------------
# Low-level serialisers
# ---------------------------------------------------------------------------


def _emit_json(
    data: Any,
    indent: int = 2,
    console: Console | None = None,
) -> None:
    """Serialize *data* to JSON and print to stdout."""
    c = console or _console
    try:
        text = json.dumps(data, indent=indent, default=_serialize, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        text = json.dumps({"error": f"serialization failed: {exc}"}, indent=indent)
    c.print(text)


def _emit_yaml(
    data: Any,
    console: Console | None = None,
) -> None:
    """Serialize *data* to YAML and print to stdout."""
    c = console or _console
    try:
        import yaml  # type: ignore[import-untyped]

        text = yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)
    except ImportError:
        # Fall back to JSON when PyYAML is not installed
        text = json.dumps(data, indent=2, default=_serialize, ensure_ascii=False)
        c.print(
            "[yellow]PyYAML not available — falling back to JSON[/yellow]",
            file=sys.stderr,
        )
    c.print(text)


# ---------------------------------------------------------------------------
# Legacy / convenience alias
# ---------------------------------------------------------------------------

FMT_HUMAN = OutputFormat.HUMAN
FMT_JSON = OutputFormat.JSON
FMT_YAML = OutputFormat.YAML
