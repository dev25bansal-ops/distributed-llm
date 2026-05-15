"""Run inference command for DistLLM CLI."""

import sys
import yaml
from pathlib import Path
from rich.console import Console


def run_inference(
    model: str,
    local: bool,
    config: str,
    port: int,
    dtype: str,
    max_tokens: int,
    temperature: float,
    prompt: str,
    console: Console,
    debug: bool = False,
):
    """Run inference with the specified parameters."""
    if debug:
        from distllm.communication.grpc import set_debug_mode
        set_debug_mode(True)
        console.print("[yellow]Debug mode enabled: tensor shape logging active[/yellow]")

    console.print(f"\n[bold blue]Starting DistLLM Inference[/bold blue]")
    console.print(f"Model: {model}")
    console.print(f"Mode: {'Local' if local else 'Distributed'}")
    console.print(f"Port: {port}\n")

    try:
        # Import here to avoid circular imports
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
        from distllm.api.server import app, create_coordinator
        import uvicorn

        # Load config if provided
        if config:
            with open(config) as f:
                cfg = yaml.safe_load(f)
            model = cfg.get("model", {}).get("name", model)
            dtype = cfg.get("model", {}).get("dtype", dtype)

        # Create coordinator
        create_coordinator(model_name=model, dtype=dtype, local=local)

        console.print(f"[green]✓[/green] Model loaded: {model}")
        console.print(f"Starting API server on http://localhost:{port}")
        console.print(f"Open http://localhost:{port}/docs for API documentation\n")

        # Start server
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

    except KeyboardInterrupt:
        console.print("\nShutting down...")
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        sys.exit(1)
