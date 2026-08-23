"""Global exception handler for CLI commands.

Provides a :func:`cli_error_handler` decorator that wraps Typer CLI command
functions with user-friendly error display using Rich panels.

Usage::

    from distllm.cli.error_handler import cli_error_handler

    @app.command()
    @cli_error_handler(verbose=False)
    def my_command(
        name: str = typer.Argument(...),
        verbose: bool = typer.Option(False, "--verbose"),
    ):
        ...

If a :class:`distllm.errors.types.DistLLMError` is raised, a Rich Panel
shows the user-friendly message, remediation hint, and docs URL.

If an unexpected exception is raised, a generic error panel is shown with
an optional full traceback when *verbose* is enabled (set via decorator
param or auto-detected from the wrapped function's ``verbose`` kwarg).
"""

from __future__ import annotations

import traceback
from functools import wraps
from typing import Any, Callable, TypeVar, cast

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from distllm.errors.types import DistLLMError

F = TypeVar("F", bound=Callable[..., Any])


def cli_error_handler(
    func: F | None = None,
    *,
    verbose: bool = False,
    exit_on_error: bool = True,
    console: Console | None = None,
) -> Callable[..., Any]:
    """Decorator that wraps CLI command functions with user-friendly error handling.

    Catches :class:`DistLLMError` and displays a Rich Panel with the
    user-friendly message and remediation hint.

    Catches unexpected exceptions and displays a generic error panel
    with a full traceback when *verbose* is ``True``.

    Can be used with or without arguments::

        @cli_error_handler
        def my_command(): ...

        @cli_error_handler(verbose=True)
        def my_command(): ...

    The decorator also inspects the wrapped function's keyword arguments
    for a ``verbose`` key (e.g., from a ``--verbose`` Typer option) and
    uses it for the traceback display.  An explicit *verbose* argument to
    the decorator always takes precedence.

    Args:
        func: The function to wrap (when used without call parentheses).
        verbose: Show full tracebacks for unexpected errors.
        exit_on_error: Exit the process with code 1 after displaying
            the error.
        console: A Rich console instance. A new one is created when
            not provided.

    Returns:
        The decorated function.
    """
    def decorator(f: F) -> F:
        @wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _console = console or Console()
            _verbose = kwargs.get("verbose", False) or verbose

            try:
                return f(*args, **kwargs)

            except DistLLMError as e:
                _display_distllm_error(_console, e)
                if exit_on_error:
                    raise SystemExit(1) from e

            except Exception as e:  # noqa: BLE001
                _display_unexpected_error(_console, e, verbose=_verbose)
                if exit_on_error:
                    raise SystemExit(1) from e

            return None

        return cast(F, wrapper)

    # Allow usage as @cli_error_handler (no parens) or @cli_error_handler(...)
    if func is not None:
        return decorator(func)
    return decorator


# ── Render helpers ──────────────────────────────────────────────────────────

def _display_distllm_error(console: Console, error: DistLLMError) -> None:
    """Display a :class:`DistLLMError` in a Rich Panel with remediation."""
    renderables: list[Any] = []

    # User-friendly message
    if error.user_message:
        renderables.append(Text(error.user_message, style="bold"))
        renderables.append("")

    # Technical details (always shown in dim)
    renderables.append(Text(f"Details: {error.message}", style="dim"))

    # Remediation hint
    if error.remediation_hint:
        renderables.append("")
        renderables.append(Text("How to fix:", style="bold yellow"))
        renderables.append(Text(error.remediation_hint))

    # Docs URL
    if error.docs_url:
        renderables.append("")
        renderables.append(Text(f"Documentation: {error.docs_url}", style="blue"))

    # Build panel with code and optional context
    renderables.append("")
    renderables.append(Text(f"Error code: {error.code}", style="italic dim"))

    if error.context:
        ctx_preview = ", ".join(
            f"{k}={v}" for k, v in list(error.context.items())[:5]
        )
        if len(error.context) > 5:
            ctx_preview += ", ..."
        renderables.append(Text(f"Context: {ctx_preview}", style="dim"))

    panel = Panel(
        "\n".join(_to_text(r) for r in renderables),
        title=Text(f" [{error.code}] ", style="bold red"),
        border_style="red",
        padding=(1, 2),
    )
    console.print(panel)


def _display_unexpected_error(
    console: Console,
    error: Exception,
    verbose: bool = False,
) -> None:
    """Display an unexpected (non-DistLLM) error."""
    renderables: list[Any] = [
        Text("An unexpected error occurred. Please try again or report this issue.", style="bold"),
        "",
        Text(f"Error type: {type(error).__name__}", style="dim"),
        Text(f"Error: {error}", style="dim"),
    ]

    if verbose:
        renderables.append("")
        renderables.append(Text("Traceback:", style="bold yellow"))
        renderables.append(Text(traceback.format_exc().rstrip(), style="dim"))

    panel = Panel(
        "\n".join(_to_text(r) for r in renderables),
        title=Text(" Unexpected Error ", style="bold red"),
        border_style="red",
        padding=(1, 2),
    )
    console.print(panel)


def _to_text(obj: Any) -> str:
    """Convert a renderable to a plain string for Panel content.

    Rich ``Text`` objects and plain strings are supported.  Any other
    type is converted via ``str()``.
    """
    if isinstance(obj, Text):
        return obj.plain
    if isinstance(obj, str):
        return obj
    return str(obj)
