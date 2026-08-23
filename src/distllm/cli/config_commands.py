"""Configuration CLI commands for distllm.

Extracted from :mod:`distllm.cli.main`.
"""

from __future__ import annotations

import json

import typer
from rich.console import Console

console = Console()


def config_validate() -> None:
    """Validate configuration and exit."""
    from distllm.config.settings import DistLLMSettings
    try:
        DistLLMSettings.validate_startup()
        console.print("[green]Config validation passed[/green]")
    except SystemExit:
        raise typer.Exit(1) from None


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
