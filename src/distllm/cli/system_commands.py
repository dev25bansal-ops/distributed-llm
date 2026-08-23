"""System commands for the DistLLM CLI - implementations extracted from main.py."""

from __future__ import annotations

from typing import Any

import typer


def system_run(
    model: str,
    local: bool,
    config: str | None,
    port: int,
    dtype: str,
    max_tokens: int,
    temperature: float,
    prompt: str | None,
    console: Any,
    debug: bool,
) -> None:
    """Run the distributed LLM system."""
    from distllm.cli.run import run_inference

    run_inference(
        model=model,
        local=local,
        config=config,
        port=port,
        dtype=dtype,
        max_tokens=max_tokens,
        temperature=temperature,
        prompt=prompt,
        console=console,
        debug=debug,
    )


def system_schedule_viz(
    host: str,
    port: int,
    output: str | None,
    last_n: int,
    api_key: str | None,
    console: Any,
) -> None:
    """Visualize scheduling decisions as ASCII timeline or HTML."""
    from loguru import logger
    import httpx

    from distllm.core.schedule_viz import ScheduleVisualizer

    base_url = f"http://{host}:{port}"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        resp = httpx.get(f"{base_url}/v1/scheduler/stats", headers=headers, timeout=5.0)
        resp.raise_for_status()
        stats = resp.json()
    except Exception as e:
        console.print(f"[red]Error connecting to coordinator: {e}[/red]")
        raise typer.Exit(1)

    # Display current stats
    console.print("[bold]Scheduler Status[/bold]")
    console.print(f"  Active: {stats.get('active_requests', 0)}")
    console.print(f"  Pending: {stats.get('pending_requests', 0)}")
    console.print(f"  Preempted: {stats.get('preempted_requests', 0)}")
    console.print(f"  Iteration: {stats.get('iteration', 0)}")
    console.print(f"  Batch size: {stats.get('max_batch_size', 0)}")
    console.print(f"  Tokens/batch: {stats.get('max_tokens_per_batch', 0)}")
    console.print(f"  Prefill tokens: {stats.get('total_prefill_tokens', 0)}")
    console.print(f"  Decode tokens: {stats.get('total_decode_tokens', 0)}")

    advanced = stats.get("advanced", {})
    if advanced:
        console.print("\n[bold]Advanced Features[/bold]")
        for key, val in advanced.items():
            console.print(f"  {key}: {val}")

    if output:
        # Generate HTML from current stats
        viz = ScheduleVisualizer()
        html = viz.to_html(output)
        console.print(f"\n[green]HTML written to {output}[/green]")


def system_coordinator(
    model: str,
    port: int,
    dtype: str,
    local: bool,
    chat_mode: bool,
    trust_remote_code: bool,
    debug: bool,
) -> None:
    """Start the coordinator facade for distributed inference."""
    from loguru import logger

    from distllm.core.coordinator import Coordinator
    from distllm.core.debug import set_debug_mode

    if debug:
        set_debug_mode(True)
        logger.info("Debug mode enabled: tensor shape logging active")

    c = Coordinator(
        model_name=model,
        port=port,
        dtype=dtype,
        trust_remote_code=trust_remote_code or None,
    )

    if local:
        c.load_local_model()
        if chat_mode:
            print(f"Model loaded: {model}")
            while True:
                prompt = input("\nPrompt (or 'quit' to exit): ")
                if prompt.lower() in ("quit", "exit"):
                    break
                result = c.generate(prompt, max_new_tokens=128)
                print(f"\nResult: {result}")
        else:
            c.start()
    else:
        c.start()


def system_api(
    model: str,
    host: str,
    port: int,
    dtype: str,
    local: bool,
    debug: bool,
    no_auth: bool,
) -> None:
    """Start the OpenAI-compatible REST API server."""
    import uvicorn
    import os
    from loguru import logger

    from distllm.api.server import app, create_coordinator
    from distllm.core.debug import set_debug_mode

    if debug:
        set_debug_mode(True)
        logger.info("Debug mode enabled: tensor shape logging active")

    if no_auth:
        logger.warning(
            "SECURITY: --no-auth is deprecated and has no effect. "
            "Authentication is always required. Use a valid API key."
        )

    create_coordinator(model_name=model, dtype=dtype, local=local)
    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


def system_slo_report(
    host: str,
    port: int,
) -> None:
    """Display SLO compliance report (latency, throughput, error rates)."""
    import httpx
    from rich.console import Console
    from rich.table import Table

    try:
        resp = httpx.get(f"http://{host}:{port}/api/cluster/reputation", timeout=5)
        data = resp.json()
        table = Table(title="Cluster SLO Report")
        table.add_column("Metric")
        table.add_column("Value")
        if isinstance(data, dict):
            for key, val in list(data.items())[:15]:
                table.add_row(key.replace("_", " ").title(), str(val))
        console = Console()
        console.print(table)
    except Exception as e:
        console = Console()
        console.print(f"[red]Failed to fetch SLO data: {e}[/red]")


def system_cost_avoid(
    model: str,
    requests_per_day: int,
    gpu_type: str,
    cloud_api: str,
) -> None:
    """Calculate monthly savings from self-hosted inference."""
    from distllm.cli.cost_avoid import calculate_cost_avoidance

    result = calculate_cost_avoidance(
        model_name=model,
        requests_per_day=requests_per_day,
        gpu_type=gpu_type,
        cloud_api=cloud_api,
    )
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Cost Avoidance Analysis", show_header=False)
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Model", result["model"])
    table.add_row("Monthly API Cost", f"${result['monthly_api_cost']:,.2f}")
    table.add_row("Monthly Self-Hosted", f"${result['monthly_self_hosted_cost']:,.2f}")
    table.add_row("Monthly Savings", f"${result['monthly_savings']:,.2f} ({result['savings_percent']}%)")
    console.print(table)


def system_doctor() -> None:
    """Run system diagnostics (CUDA, ports, config, disk)."""
    from distllm.cli.doctor import main as doctor_main

    doctor_main()


def system_observe(
    coordinator_url: str,
    metrics_port: int,
    dashboard_port: int,
    loki_url: str,
    prometheus_url: str,
    metrics_only: bool,
    console: Any,
) -> None:
    """Launch the DistLLM observability stack (metrics, dashboards, logs).

    Starts a Prometheus-compatible metrics endpoint and generates Grafana
    dashboard provisioning for distributed inference monitoring.

    Examples::

        distllm system observe                          # Full stack
        distllm system observe --metrics-only           # Just metrics
        distllm system observe --dashboard-port 3000    # Custom port
    """
    from distllm.cli.observe import run_observe

    run_observe(
        coordinator_url=coordinator_url,
        metrics_port=metrics_port,
        dashboard_port=dashboard_port,
        loki_url=loki_url,
        prometheus_url=prometheus_url,
        metrics_only=metrics_only,
        console=console,
    )
