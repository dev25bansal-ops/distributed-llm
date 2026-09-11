"""Run inference command for DistLLM CLI."""

import sys

import uvicorn
import yaml
from rich.console import Console

from distllm.api.server import app, create_coordinator
from distllm.core.debug import set_debug_mode


def run_inference(
    model: str,
    local: bool,
    config: str,
    port: int,
    dtype: str,
    max_tokens: int,
    temperature: float,
    console: Console,
    debug: bool = False,
    prompt: str | None = None,
):
    """Run inference with the specified parameters."""
    if debug:
        set_debug_mode(True)
        console.print("[yellow]Debug mode enabled: tensor shape logging active[/yellow]")

    console.print("\n[bold blue]Starting DistLLM Inference[/bold blue]")
    console.print(f"Model: {model}")
    console.print(f"Mode: {'Local' if local else 'Distributed'}")
    console.print(f"Port: {port}\n")

    try:
        # Load config if provided
        if config:
            with open(config) as f:
                cfg = yaml.safe_load(f) or {}
            model = cfg.get("model", {}).get("name", model)
            dtype = cfg.get("model", {}).get("dtype", dtype)

        # Create coordinator (best effort: a failed model load must not take
        # down the API server, which can still serve health/docs/routes)
        try:
            create_coordinator(model_name=model, dtype=dtype, local=local)
        except Exception as e:
            console.print(
                f"[yellow]Warning: coordinator/model load failed ({e}) — "
                f"starting API server without a loaded model.[/yellow]"
            )

        console.print(f"[green]✓[/green] Model loaded: {model}")
        console.print(f"Starting API server on http://localhost:{port}")
        console.print(f"Open http://localhost:{port}/docs for API documentation\n")

        # Start server
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")

    except KeyboardInterrupt:
        console.print("\nShutting down...")
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        sys.exit(1)
