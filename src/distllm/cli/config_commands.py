"""Configuration CLI commands for distllm.

Extracted from :mod:`distllm.cli.main`.

This module is the authoritative implementation for ``distllm config
validate``; ``cli.main`` delegates here so there is exactly one copy of the
logic (WAVE2 item 42: the command previously validated nothing — it never
read config.yaml).
"""

from __future__ import annotations

import json
import os

import typer
from rich.console import Console

console = Console()


def resolve_config_file(config_path: str | None = None) -> str | None:
    """Resolve the config file to validate, mirroring resolver auto-discovery.

    Precedence: explicit *config_path* (expanded; must exist) > cwd
    ``config.yaml`` > standard locations from
    :attr:`ConfigResolver.COMMON_CONFIG_CANDIDATES`.

    Returns:
        Absolute-ish path to an existing file, or ``None`` when nothing was
        found and no explicit path was given.
    """
    from distllm.config.resolver import ConfigResolver, _find_config

    if config_path:
        expanded = os.path.expanduser(os.path.expandvars(config_path))
        if not os.path.exists(expanded):
            return None
        return expanded
    # Same discovery order the runtime entry points use.
    return _find_config(list(ConfigResolver.COMMON_CONFIG_CANDIDATES))


def run_config_validate(config_path: str | None = None, out: Console | None = None) -> None:
    """Validate the resolved configuration file and exit non-zero on failure.

    Loads config.yaml through :meth:`DistLLMSettings.validate_startup`, which
    applies the full precedence chain (env vars over YAML over defaults) and
    reports parse/validation errors with the offending file path.
    """
    from distllm.config.settings import DistLLMSettings

    c = out or console

    resolved: str | None
    if config_path:
        resolved = resolve_config_file(config_path)
        if resolved is None:
            c.print(
                f"[red]Config file not found:[/red] "
                f"{os.path.expanduser(os.path.expandvars(config_path))}"
            )
            raise typer.Exit(1)
    else:
        resolved = resolve_config_file()
        if resolved is None:
            c.print(
                "[yellow]No configuration file found[/yellow] "
                "(looked in cwd and standard locations) — "
                "validating defaults and environment variables"
            )

    try:
        settings = DistLLMSettings.validate_startup(config_path=resolved)
    except SystemExit:
        # validate_startup already printed path-named errors to stdout.
        raise typer.Exit(1) from None

    source = resolved or "defaults/environment"
    c.print(f"[green]Config validation passed[/green] [dim]({source})[/dim]")
    model_name = getattr(getattr(settings, "model", None), "name", "") or "(default)"
    c.print(f"  Model: {model_name}")
    c.print(f"  Nodes configured: {len(getattr(settings, 'nodes', []) or [])}")


def config_reference(output: str = "") -> None:
    """Generate configuration reference documentation from Pydantic models."""
    from distllm.config.reference import generate_config_reference
    from distllm.config.settings import DistLLMSettings

    doc = generate_config_reference(DistLLMSettings)
    if output:
        with open(output, "w") as f:
            f.write(doc)
        console.print(f"[green]Config reference written to {output}[/green]")
    else:
        console.print(doc)


def config_openapi(output: str = "", format: str = "json") -> None:
    """Export the OpenAPI specification for code generation and documentation.

    Generates a standalone OpenAPI 3.1 spec from the FastAPI application
    that can be used with code generators (openapi-generator, swagger-codegen),
    API gateways, and documentation tools.
    """
    # Build the FastAPI app to extract the schema
    from distllm.api.server import app as fastapi_app

    schema = fastapi_app.openapi()

    # Add DistLLM-specific metadata
    schema["info"]["x-distllm-version"] = "0.4.0"
    schema["info"]["x-sdk-version"] = "1.0.0"

    if format == "yaml":
        try:
            import yaml
            content = yaml.dump(schema, default_flow_style=False, sort_keys=False)
        except ImportError:
            console.print("[yellow]PyYAML not installed, falling back to JSON[/yellow]")
            content = json.dumps(schema, indent=2)
    else:
        content = json.dumps(schema, indent=2)

    if output:
        with open(output, "w") as f:
            f.write(content)
        console.print(f"[green]OpenAPI spec written to {output}[/green]")
    else:
        console.print(content)
