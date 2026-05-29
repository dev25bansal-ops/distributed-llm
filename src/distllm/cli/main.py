"""DistLLM CLI - Unified command-line interface for Distributed LLM."""

import os
import sys
import time
import os
import sys
import time
import typer
from rich.console import Console
from rich.table import Table


def _display_qr(port: int = 50050) -> None:
    """Display a QR code for joining the cluster."""
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    join_cmd = f"distllm cluster join --coordinator {local_ip}:{port}"
    try:
        import qrcode
        qr = qrcode.QRCode(border=1)
        qr.add_data(join_cmd)
        qr.print_ascii(invert=False)
    except ImportError:
        pass
    print(f"\n   Join this cluster: {join_cmd}")


app = typer.Typer(
    name="distllm",
    help="Distributed LLM Inference System - Run large language models across multiple GPU-equipped machines",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()

# ── Command groups ──────────────────────────────────────────────────────
#
#   cluster    — cluster management (start, join, status, scale, deploy)
#   model      — model lifecycle (list, load, compress, adapters)
#   config     — configuration (setup, validate, webhook, quota, backup)
#   benchmark  — benchmarking (run, compare, profile, verify)
#   security   — TLS certificates and keys
#   system     — daemons, diagnostics, logs, notifications
#

cluster_app = typer.Typer(help="Manage cluster: start, join, status, scale, drain, deploy")
model_app = typer.Typer(help="Manage models: list, load, info, compress, adapters")
benchmark_app = typer.Typer(help="Run benchmarks, compare results, profile inference")
config_app = typer.Typer(help="Configuration, setup, webhooks, quotas, backups")
security_app = typer.Typer(help="TLS certificates and cryptographic keys")
system_app = typer.Typer(help="Daemons, diagnostics, logs, notifications, cost analysis")
router_app = typer.Typer(help="Model router: rules, test, stats, patterns")

# Nested groups
model_adapters_app = typer.Typer(help="Manage LoRA/QLoRA adapters")
benchmark_verify_app = typer.Typer(help="Verify distributed inference correctness")
config_webhook_app = typer.Typer(help="Manage webhook endpoints")
config_quota_app = typer.Typer(help="Usage quotas and billing")
config_backup_app = typer.Typer(help="Backup and restore cluster state")
security_cert_app = typer.Typer(help="Manage TLS certificates")
system_logs_app = typer.Typer(help="Stream and filter logs")
system_notify_app = typer.Typer(help="Notifications and alerts")

app.add_typer(cluster_app, name="cluster")
app.add_typer(model_app, name="model")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(config_app, name="config")
app.add_typer(security_app, name="security")
app.add_typer(system_app, name="system")
app.add_typer(router_app, name="router")

from distllm.cli.prompts import prompt_app

# Nested group registrations
model_app.add_typer(model_adapters_app, name="adapters")
model_app.add_typer(prompt_app, name="prompt")
benchmark_app.add_typer(benchmark_verify_app, name="verify")
config_app.add_typer(config_webhook_app, name="webhook")
config_app.add_typer(config_quota_app, name="quota")
config_app.add_typer(config_backup_app, name="backup")
security_app.add_typer(security_cert_app, name="cert")
system_app.add_typer(system_logs_app, name="logs")
system_app.add_typer(system_notify_app, name="notify")
defrag_app = typer.Typer(help="GPU memory defragmentation: status, run, stats")
system_app.add_typer(defrag_app, name="defrag")


@defrag_app.command("status")
def defrag_status(
    host: str = typer.Option("localhost", help="Coordinator host"),
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
):
    """Show defragmentation status and fragmentation ratio."""
    import httpx

    base_url = f"http://{host}:{port}"
    try:
        resp = httpx.get(f"{base_url}/v1/defrag/status", timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    if not data.get("enabled"):
        console.print("[yellow]Defragmentation is disabled[/yellow]")
        return

    from rich.table import Table
    t = Table(title="Defragmentation Status")
    t.add_column("Metric", style="cyan")
    t.add_column("Value", style="green")
    t.add_row("Policy", data.get("policy", "N/A"))
    t.add_row("Fragmentation Ratio", f"{data.get('fragmentation_ratio', 0):.2%}")
    t.add_row("Predictive (5 steps)", f"{data.get('predictive_fragmentation', 0):.2%}")
    stats = data.get("stats", {})
    t.add_row("Total Passes", str(stats.get("defrag_count", 0)))
    t.add_row("Blocks Moved", str(stats.get("blocks_moved", 0)))
    t.add_row("Bytes Compacted", f"{stats.get('bytes_compacted', 0) / 1024 / 1024:.1f} MB")
    t.add_row("Total Time", f"{stats.get('total_time_ms', 0):.1f} ms")
    t.add_row("Peak Fragmentation", f"{stats.get('peak_fragmentation_ratio', 0):.2%}")
    console.print(t)

    config = data.get("config", {})
    if config:
        ct = Table(title="Configuration")
        ct.add_column("Setting", style="cyan")
        ct.add_column("Value", style="green")
        ct.add_row("Interval", f"{config.get('interval_seconds', 0)}s")
        ct.add_row("Max Blocks/Pass", str(config.get("max_blocks_per_pass", 0) or "unlimited"))
        ct.add_row("Tiered Compaction", str(config.get("tiered_compaction", False)))
        ct.add_row("Predictive", str(config.get("enable_predictive", False)))
        console.print(ct)


@defrag_app.command("run")
def defrag_run(
    host: str = typer.Option("localhost", help="Coordinator host"),
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
):
    """Trigger an immediate defragmentation pass."""
    import httpx

    base_url = f"http://{host}:{port}"
    try:
        resp = httpx.post(f"{base_url}/v1/defrag/run", timeout=120.0)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    if "error" in data:
        console.print(f"[red]{data['error']}[/red]")
        return

    from rich.table import Table
    for backend_key, result in data.items():
        t = Table(title=f"Defrag Result — {backend_key}")
        t.add_column("Metric", style="cyan")
        t.add_column("Value", style="green")
        t.add_row("Blocks Moved", str(result.get("blocks_moved", 0)))
        t.add_row("Bytes Compacted", f"{result.get('bytes_compacted', 0) / 1024 / 1024:.1f} MB")
        t.add_row("Duration", f"{result.get('time_ms', 0):.1f} ms")
        t.add_row("Frag Before", f"{result.get('fragmentation_before', 0):.2%}")
        t.add_row("Frag After", f"{result.get('fragmentation_after', 0):.2%}")
        t.add_row("Tier", result.get("tier_used", "N/A"))
        console.print(t)


@defrag_app.command("stats")
def defrag_stats(
    host: str = typer.Option("localhost", help="Coordinator host"),
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
):
    """Show historical defragmentation statistics."""
    import httpx

    base_url = f"http://{host}:{port}"
    try:
        resp = httpx.get(f"{base_url}/v1/defrag/stats", timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    if not data.get("enabled"):
        console.print("[yellow]Defragmentation is disabled[/yellow]")
        return

    stats = data.get("stats", {})
    from rich.table import Table
    t = Table(title="Defragmentation Statistics")
    t.add_column("Metric", style="cyan")
    t.add_column("Value", style="green")
    t.add_row("Total Passes", str(stats.get("defrag_count", 0)))
    t.add_row("Blocks Moved", str(stats.get("blocks_moved", 0)))
    t.add_row("Bytes Compacted", f"{stats.get('bytes_compacted', 0) / 1024 / 1024:.1f} MB")
    t.add_row("Total Time", f"{stats.get('total_time_ms', 0):.1f} ms")
    t.add_row("Avg Time/Pass", f"{stats.get('total_time_ms', 0) / max(stats.get('defrag_count', 1), 1):.1f} ms")
    t.add_row("L1 (Hot) Passes", str(stats.get("l1_count", 0)))
    t.add_row("L2 (Warm) Passes", str(stats.get("l2_count", 0)))
    t.add_row("L3 (Cold) Passes", str(stats.get("l3_count", 0)))
    t.add_row("Peak Fragmentation", f"{stats.get('peak_fragmentation_ratio', 0):.2%}")
    t.add_row("Current Fragmentation", f"{stats.get('last_fragmentation_ratio', 0):.2%}")
    console.print(t)

    history = data.get("fragmentation_history", [])
    if history:
        console.print(f"\n[bold]Fragmentation History[/bold] (last {len(history)} samples)")
        console.print(f"  Min: {min(history):.2%}, Max: {max(history):.2%}, Avg: {sum(history)/len(history):.2%}")


@system_app.command("schedule-viz")
def system_schedule_viz(
    host: str = typer.Option("localhost", help="Coordinator host"),
    port: int = typer.Option(50050, help="Coordinator API port"),
    output: str | None = typer.Option(None, "--output", "-o", help="HTML output file path"),
    last_n: int = typer.Option(20, "--last", "-n", help="Number of recent iterations to show"),
    api_key: str | None = typer.Option(None, "--api-key", help="API key for authentication"),
):
    """Visualize scheduling decisions as ASCII timeline or HTML."""
    from distllm.core.schedule_viz import ScheduleVisualizer
    import httpx

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


@config_app.command("setup")
def config_setup(
    config_path: str = typer.Option("config.yaml", "--config", "-c", help="Output config file path"),
):
    """Interactive setup wizard for cluster configuration."""
    from distllm.cli.setup import run_setup
    run_setup(config_path, console)


@system_app.command("run")
def system_run(
    model: str = typer.Option(..., "--model", "-m", help="Model name or path"),
    local: bool = typer.Option(False, "--local", "-l", help="Run in single-node local mode"),
    config: str | None = typer.Option(None, "--config", "-c", help="Config file path"),
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
    dtype: str = typer.Option("float16", "--dtype", help="Data type (float16, float32, bfloat16)",),
    max_tokens: int = typer.Option(256, "--max-tokens", help="Max tokens to generate"),
    temperature: float = typer.Option(0.7, "--temperature", help="Sampling temperature"),
    prompt: str | None = typer.Option(None, "--prompt", help="Single prompt (non-interactive)"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode with tensor shape logging"),
):
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


@config_app.command("validate")
def config_validate():
    """Validate configuration and exit."""
    from distllm.config.settings import DistLLMSettings
    try:
        DistLLMSettings.validate_startup()
        console.print("[green]Config validation passed[/green]")
    except SystemExit:
        raise typer.Exit(1) from None


@config_app.command("reference")
def config_reference(
    output: str = typer.Option("", "--output", "-o", help="Output file (default: stdout)"),
):
    """Generate configuration reference documentation from Pydantic models."""
    from distllm.config.settings import DistLLMSettings
    from distllm.config.reference import generate_config_reference

    doc = generate_config_reference(DistLLMSettings)
    if output:
        with open(output, "w") as f:
            f.write(doc)
        console.print(f"[green]Config reference written to {output}[/green]")
    else:
        console.print(doc)


@config_app.command("openapi")
def config_openapi(
    output: str = typer.Option("", "--output", "-o", help="Output file (default: stdout)"),
    format: str = typer.Option("json", "--format", "-f", help="Output format: json or yaml"),
):
    """Export the OpenAPI specification for code generation and documentation.

    Generates a standalone OpenAPI 3.1 spec from the FastAPI application
    that can be used with code generators (openapi-generator, swagger-codegen),
    API gateways, and documentation tools.
    """
    import json

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


# --- model group ---
@model_app.command("list")
def model_list(
    host: str = typer.Option("localhost", "--host", help="API server host"),
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
):
    """List available and loaded models."""
    from distllm.cli.models import _list_models
    _list_models(host, port)


@model_app.command("info")
def model_info(
    model_id: str = typer.Argument(..., help="Model identifier"),
    host: str = typer.Option("localhost", "--host", help="API server host"),
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
):
    """Show detailed model information."""
    from distllm.cli.models import _model_info
    _model_info(host, port, model_id)


@model_app.command("load")
def model_load(
    model: str = typer.Argument(..., help="Model name or path"),
    host: str = typer.Option("localhost", "--host", help="API server host"),
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
    dtype: str = typer.Option("float16", "--dtype", help="Data type"),
):
    """Load a model into the server."""
    from distllm.cli.models import _load_model
    _load_model(host, port, model, dtype)


@model_app.command("unload")
def model_unload(
    model_id: str = typer.Argument(..., help="Model identifier"),
    host: str = typer.Option("localhost", "--host", help="API server host"),
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
):
    """Unload a model from the server."""
    from distllm.cli.models import _unload_model
    _unload_model(host, port, model_id)


# --- cluster group ---
@cluster_app.command("status")
def cluster_status(
    host: str = typer.Option("localhost", "--host", help="API server host"),
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
):
    """Show cluster status and node health."""
    from distllm.cli.cluster import _cluster_status
    _cluster_status(host, port)


@cluster_app.command("scale")
def cluster_scale(
    nodes: int = typer.Argument(..., help="Target number of nodes"),
    host: str = typer.Option("localhost", "--host", help="API server host"),
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
    gpu_type: str | None = typer.Option(None, "--gpu-type", help="Filter by GPU type"),
):
    """Scale cluster to target node count."""
    from distllm.cli.cluster import _cluster_scale
    _cluster_scale(host, port, nodes, gpu_type)


@cluster_app.command("drain")
def cluster_drain(
    node_id: str = typer.Argument(..., help="Node ID to drain"),
    host: str = typer.Option("localhost", "--host", help="API server host"),
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
):
    """Drain a node (gracefully remove from service)."""
    from distllm.cli.cluster import _cluster_drain
    _cluster_drain(host, port, node_id)


@cluster_app.command("rebalance")
def cluster_rebalance(
    host: str = typer.Option("localhost", "--host", help="API server host"),
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
    strategy: str = typer.Option("balanced", "--strategy", help="Rebalancing strategy"),
):
    """Rebalance load across cluster nodes."""
    from distllm.cli.cluster import _cluster_rebalance
    _cluster_rebalance(host, port, strategy)


@cluster_app.command("start")
def cluster_start(
    model: str = typer.Option(..., "--model", "-m", help="Model name or path"),
    port: int = typer.Option(50050, "--port", "-p", help="Coordinator gRPC port"),
    api_port: int = typer.Option(8000, "--api-port", help="REST API port"),
    local: bool = typer.Option(False, "--local", "-l", help="Run full model locally"),
    dtype: str = typer.Option("float16", "--dtype", help="Data type"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
    qr: bool = typer.Option(False, "--qr", help="Display QR code for easy worker join"),
):
    """Start a coordinator node and begin accepting workers."""
    from distllm.cli.cluster import _cluster_start
    _cluster_start(model, port, api_port, local, dtype, debug)
    if qr:
        _display_qr(port)


@cluster_app.command("join")
def cluster_join(
    coordinator_host: str = typer.Option("localhost", "--coordinator", "-c", help="Coordinator hostname or IP (omit for auto-discovery)"),
    coordinator_port: int = typer.Option(50050, "--port", "-p", help="Coordinator gRPC port"),
    discover: bool = typer.Option(False, "--discover", "-d", help="Auto-discover coordinators on LAN via mDNS"),
    node_id: str = typer.Option(None, "--node-id", help="Unique node ID (auto-generated if omitted)"),
    start_layer: int = typer.Option(None, "--start-layer", help="First layer to serve"),
    end_layer: int = typer.Option(None, "--end-layer", help="Last layer to serve"),
    total_layers: int = typer.Option(None, "--total-layers", help="Total model layers"),
    port: int = typer.Option(50051, "--listen-port", help="This node's gRPC port"),
    device: str = typer.Option("auto", "--device", help="Device (auto, cuda, cpu)"),
    cluster_key: str = typer.Option(None, "--cluster-key", help="[DEPRECATED] Use DISTLLM_CLUSTER_KEY env var instead. Shared cluster authentication key"),
):
    """Join an existing cluster as a worker node."""
    import os
    if cluster_key:
        console.print("[yellow]Warning:[/yellow] --cluster-key is deprecated. Set DISTLLM_CLUSTER_KEY env var instead.")
    resolved_key = cluster_key or os.environ.get("DISTLLM_CLUSTER_KEY", "")
    if not resolved_key:
        key_path = os.path.expanduser("~/.distllm/cluster_key")
        if os.path.isfile(key_path):
            resolved_key = open(key_path).read().strip()
    from distllm.cli.cluster import _cluster_join
    _cluster_join(
        coordinator_host, coordinator_port, node_id,
        start_layer, end_layer, total_layers, port, device, resolved_key or None,
        discover=discover,
    )


@cluster_app.command("leave")
def cluster_leave(
    node_id: str = typer.Argument(..., help="Node ID to remove from cluster"),
    coordinator_host: str = typer.Option("localhost", "--coordinator", "-c", help="Coordinator hostname"),
    coordinator_port: int = typer.Option(50050, "--port", "-p", help="Coordinator port"),
):
    """Gracefully leave the cluster and shut down worker."""
    from distllm.cli.cluster import _cluster_leave
    _cluster_leave(node_id, coordinator_host, coordinator_port)


@cluster_app.command("list-nodes")
def cluster_list_nodes(
    coordinator_host: str = typer.Option("localhost", "--coordinator", "-c", help="Coordinator hostname"),
    coordinator_port: int = typer.Option(50050, "--port", "-p", help="Coordinator gRPC port"),
):
    """List registered worker nodes in the cluster."""
    from distllm.cli.cluster import _cluster_list_nodes
    _cluster_list_nodes(coordinator_host, coordinator_port)


# --- model adapters group ---
@model_adapters_app.command("list")
def adapters_list(
    host: str = typer.Option("localhost", "--host", help="API server host"),
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
):
    """List loaded adapters."""
    from distllm.cli.adapters import _list_adapters
    _list_adapters(host, port)


@model_adapters_app.command("load")
def adapters_load(
    adapter_id: str = typer.Argument(..., help="Adapter identifier"),
    source: str = typer.Argument(..., help="Adapter source path or URL"),
    host: str = typer.Option("localhost", "--host", help="API server host"),
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
    type: str = typer.Option("lora", "--type", help="Adapter type (lora, qlora)"),
):
    """Load an adapter (LoRA, etc.)."""
    from distllm.cli.adapters import _load_adapter
    _load_adapter(host, port, adapter_id, source, type)


@model_adapters_app.command("set")
def adapters_set(
    adapter_id: str = typer.Argument(..., help="Adapter identifier"),
    host: str = typer.Option("localhost", "--host", help="API server host"),
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
):
    """Set an adapter as the active adapter."""
    from distllm.cli.adapters import _set_adapter
    _set_adapter(host, port, adapter_id)


@model_adapters_app.command("unload")
def adapters_unload(
    adapter_id: str = typer.Argument(..., help="Adapter identifier"),
    host: str = typer.Option("localhost", "--host", help="API server host"),
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
):
    """Unload an adapter."""
    from distllm.cli.adapters import _unload_adapter
    _unload_adapter(host, port, adapter_id)


# --- system logs group ---
@system_logs_app.command("stream")
def logs_stream(
    host: str = typer.Option("localhost", "--host", help="API server host"),
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
    lines: int = typer.Option(50, "--lines", "-n", help="Number of log lines"),
    level: str | None = typer.Option(None, "--level", "-l", help="Filter by log level"),
    component: str | None = typer.Option(None, "--component", "-c", help="Filter by component"),
    search: str | None = typer.Option(None, "--search", "-s", help="Search text in logs"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output"),
):
    """Stream or fetch logs from the server."""
    from distllm.cli.logs import _stream_logs
    _stream_logs(host, port, follow, lines, level, component, search)


@model_app.command("compress")
def model_compress(
    model: str = typer.Option(..., "--model", "-m", help="Model name or HuggingFace path"),
    target: str = typer.Option("int4", "--target", "-t", help="Target precision (int4, int8, int4-awq, int4-gptq)"),
    output: str = typer.Option("./compressed", "--output", "-o", help="Output directory for compressed model"),
    tokenizer: str | None = typer.Option(None, "--tokenizer", help="Tokenizer name/path (defaults to model)"),
    prune_ratio: float = typer.Option(0.0, "--prune", "-p", help="Structured pruning ratio (0.0-1.0)"),
    calibration_samples: int = typer.Option(128, "--calibration-samples", help="Calibration samples for PTQ"),
    method: str = typer.Option("awq", "--method", help="Quantization method (awq, gptq)"),
    local: bool = typer.Option(True, "--local", "-l", help="Run in local mode (no cluster)"),
):
    """Compress a model (AWQ/GPTQ INT4, INT8, pruning) and save to disk."""
    from distllm.cli.compress import run_compress
    run_compress(
        model_name=model,
        target=target,
        output_dir=output,
        tokenizer_name=tokenizer,
        prune_ratio=prune_ratio,
        calibration_samples=calibration_samples,
        method=method,
        local=local,
        console=console,
    )


# --- federated training group ---
federate_app = typer.Typer(help="Federated fine-tuning: train, merge, status")
model_app.add_typer(federate_app, name="federate")


@federate_app.command("train")
def federate_train(
    model: str = typer.Option(..., "--model", "-m", help="Base model name"),
    adapter: str = typer.Option(..., "--adapter", "-a", help="Adapter ID to fine-tune"),
    data: str = typer.Option(..., "--data", "-d", help="Path to local training data (JSONL)"),
    epochs: int = typer.Option(3, "--epochs", help="Training epochs per round"),
    lr: float = typer.Option(2e-4, "--lr", help="Learning rate"),
    rounds: int = typer.Option(1, "--rounds", help="Number of federated rounds"),
    host: str = typer.Option("localhost", "--host", help="Coordinator host"),
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
):
    """Run federated fine-tuning: train LoRA locally, submit to coordinator for merging."""
    import httpx

    console.print(f"\n[bold blue]Federated Fine-Tuning[/bold blue]")
    console.print(f"Model: {model}")
    console.print(f"Adapter: {adapter}")
    console.print(f"Data: {data}")
    console.print(f"Epochs: {epochs}, LR: {lr}, Rounds: {rounds}")
    console.print()

    base_url = f"http://{host}:{port}"

    for round_num in range(1, rounds + 1):
        console.print(f"[bold]Round {round_num}/{rounds}[/bold]")

        # 1. Start local training
        console.print("  Training locally...")
        try:
            from distllm.models.adapter import AdapterManager
            from distllm.models.partitioner import ModelPartitioner

            # Load model and adapter
            partitioner = ModelPartitioner(model_name=model)
            partitioner.load_full_model()

            mgr = AdapterManager()
            mgr.set_base_model(partitioner.full_model, partitioner.tokenizer)
            mgr.load_adapter(adapter, adapter)

            # Train
            result = mgr.start_federated_training(
                adapter_id=adapter,
                local_data_path=data,
                epochs=epochs,
                learning_rate=lr,
            )

            if "error" in result:
                console.print(f"  [red]Training failed:[/red] {result['error']}")
                raise typer.Exit(1)

            console.print(f"  Loss: {result['avg_loss']:.4f}, Steps: {result['steps']}")

        except Exception as e:
            console.print(f"  [red]Error:[/red] {e}")
            raise typer.Exit(1)

        # 2. Submit to coordinator for merging
        console.print("  Submitting to coordinator...")
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    f"{base_url}/v1/federated/rounds/submit",
                    json={
                        "node_id": f"cli-{adapter}",
                        "adapter_path": f"/tmp/distllm-federated/{adapter}.pt",
                        "loss": result["avg_loss"],
                        "dataset_size": result["steps"],
                    },
                )
                if resp.status_code == 200:
                    console.print("  [green]Submitted successfully[/green]")
                else:
                    console.print(f"  [yellow]Submission returned {resp.status_code}[/yellow]")
        except httpx.ConnectError:
            console.print("  [yellow]Coordinator not reachable (local-only mode)[/yellow]")

        # 3. Trigger merge if coordinator available
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(f"{base_url}/v1/federated/rounds/merge")
                if resp.status_code == 200:
                    data = resp.json()
                    console.print(f"  [green]Merged:[/green] {data.get('path', 'OK')}")
        except Exception:
            pass

    console.print("\n[green]Federated training complete[/green]")


@federate_app.command("status")
def federate_status(
    host: str = typer.Option("localhost", "--host", help="API server host"),
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
):
    """Show federated training status."""
    import httpx

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"http://{host}:{port}/v1/federated/stats")
            resp.raise_for_status()
            stats = resp.json()

            console.print("\n[bold]Federated Training Status[/bold]")
            console.print(f"  Total rounds: {stats.get('total_rounds', 0)}")
            console.print(f"  Registered nodes: {stats.get('registered_nodes', 0)}")
            console.print(f"  Active nodes: {stats.get('active_nodes', 0)}")
            console.print(f"  Total versions: {stats.get('total_versions', 0)}")
            console.print(f"  Strategy: {stats.get('merge_strategy', 'N/A')}")
            if stats.get('current_round'):
                console.print(f"  Current round: {stats['current_round']} ({stats.get('current_round_status', '')})")
            if stats.get('avg_loss_last_round') is not None:
                console.print(f"  Avg loss (last round): {stats['avg_loss_last_round']:.4f}")
    except httpx.ConnectError:
        console.print("[red]Could not connect to coordinator[/red]")


# --- benchmark group ---
@benchmark_app.command("run")
def benchmark_run(
    model: str = typer.Option(..., "--model", "-m", help="Model to benchmark"),
    host: str = typer.Option("localhost", "--host", help="API server host"),
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
    prompts: int = typer.Option(5, "--prompts", help="Number of test prompts"),
    max_tokens: int = typer.Option(50, "--max-tokens", help="Max tokens per prompt"),
    local: bool = typer.Option(False, "--local", help="Benchmark local mode"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON (for CI)"),
):
    """Run benchmarks against the API server."""
    if json_output:
        from distllm.cli.benchmark import run_benchmark_json
        result = run_benchmark_json(
            model=model,
            host=host,
            port=port,
            num_prompts=prompts,
            max_tokens=max_tokens,
            local=local,
        )
        console.print(result)
    else:
        from distllm.cli.benchmark import run_benchmark
        run_benchmark(
            model=model,
            host=host,
            port=port,
            num_prompts=prompts,
            max_tokens=max_tokens,
            local=local,
            console=console,
        )


@benchmark_app.command("compare")
def benchmark_compare(
    model: str = typer.Option(..., "--model", "-m", help="Model to benchmark"),
    host: str = typer.Option("localhost", "--host", help="API server host"),
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
    prompts: int = typer.Option(5, "--prompts", help="Number of test prompts"),
    max_tokens: int = typer.Option(50, "--max-tokens", help="Max tokens per prompt"),
    baseline: str | None = typer.Option(None, "--baseline", "-b", help="Path to baseline JSON file"),
    save_baseline: bool = typer.Option(False, "--save-baseline", help="Save current results as baseline"),
):
    """Compare current benchmark against a saved baseline."""
    from distllm.cli.benchmark import run_benchmark_compare
    run_benchmark_compare(
        model=model,
        host=host,
        port=port,
        num_prompts=prompts,
        max_tokens=max_tokens,
        baseline_path=baseline,
        save_baseline=save_baseline,
        console=console,
    )


# --- benchmark verify group ---
@benchmark_verify_app.command("run")
def verify_run(
    model: str = typer.Option(..., "--model", "-m", help="Model name or path"),
    prompts: list[str] = typer.Option(
        ["The capital of France is", "In the beginning", "The meaning of life is"],
        "--prompt", "-p",
        help="Prompt(s) to verify (repeat for multiple)",
    ),
    num_nodes: int = typer.Option(2, "--nodes", "-n", help="Number of distributed nodes"),
    dtype: str = typer.Option("float16", "--dtype", help="Data type"),
    temperature: float = typer.Option(0.0, "--temperature", "-t", help="Sampling temperature (0 = greedy)"),
    max_new_tokens: int = typer.Option(32, "--max-tokens", help="Max tokens to generate per prompt"),
    collect_hidden: bool = typer.Option(False, "--collect-hidden", help="Collect intermediate hidden states"),
    backend: str = typer.Option("", "--backend", "-b", help="Preferred inference backend"),
    grpc: bool = typer.Option(False, "--grpc", help="Use real gRPC workers instead of in-process simulation"),
    grpc_base_port: int = typer.Option(51050, "--grpc-port", help="Base port for gRPC workers"),
    output_json: str = typer.Option("", "--output-json", "-o", help="Path to write JSON report"),
    device: str = typer.Option("auto", "--device", help="Device (auto, cuda, cpu)"),
    trust_remote_code: bool = typer.Option(False, "--trust-remote-code", help="Trust remote code in HuggingFace models"),
):
    """Verify distributed inference accuracy against single-node reference."""
    from distllm.cli.verify import verify_run as _verify_run
    _verify_run(
        model=model, prompts=prompts, num_nodes=num_nodes, dtype=dtype,
        temperature=temperature, max_new_tokens=max_new_tokens,
        collect_hidden=collect_hidden, backend=backend, grpc=grpc,
        grpc_base_port=grpc_base_port, output_json=output_json,
        device=device, trust_remote_code=trust_remote_code,
    )


@benchmark_verify_app.command("list-backends")
def verify_list_backends():
    """List available inference backends."""
    from distllm.cli.verify import verify_list_backends as _verify_list_backends
    _verify_list_backends()


# --- config backup group ---
@config_backup_app.command("create")
def backup_create(
    backup_dir: str = typer.Option("./backups", "--dir", "-d", help="Backup directory"),
    cluster_name: str = typer.Option("default", "--cluster", "-c", help="Cluster name"),
    config_path: str = typer.Option("config.yaml", "--config", help="Config file path"),
):
    """Create a full backup of cluster state."""
    from distllm.cli.backup import backup_create as _backup_create
    _backup_create(backup_dir, cluster_name, config_path)


@config_backup_app.command("list")
def backup_list(
    backup_dir: str = typer.Option("./backups", "--dir", "-d", help="Backup directory"),
):
    """List available backups."""
    from distllm.cli.backup import backup_list as _backup_list
    _backup_list(backup_dir)


@config_backup_app.command("restore")
def backup_restore(
    backup_id: str = typer.Argument(..., help="Backup ID to restore"),
    backup_dir: str = typer.Option("./backups", "--dir", "-d", help="Backup directory"),
    output: str = typer.Option("", "--output", "-o", help="Output file"),
):
    """Restore a backup to its original state."""
    from distllm.cli.backup import backup_restore as _backup_restore
    _backup_restore(backup_id, backup_dir, output)


@config_backup_app.command("delete")
def backup_delete(
    backup_id: str = typer.Argument(..., help="Backup ID to delete"),
    backup_dir: str = typer.Option("./backups", "--dir", "-d", help="Backup directory"),
):
    """Delete a backup."""
    from distllm.cli.backup import backup_delete as _backup_delete
    _backup_delete(backup_id, backup_dir)


# --- security cert group ---
@security_cert_app.command("create")
def cert_create(
    common_name: str = typer.Argument(..., help="Domain name"),
    alt_names: list[str] = typer.Option([], "--alt-name", "-a", help="Subject alternative names"),
    cert_dir: str = typer.Option("./certs", "--dir", "-d", help="Certificate directory"),
    self_signed: bool = typer.Option(True, "--self-signed", help="Self-signed certificate"),
):
    """Create a TLS certificate."""
    from distllm.cli.cert import cert_create as _cert_create
    _cert_create(common_name, alt_names, cert_dir, self_signed)


@security_cert_app.command("info")
def cert_info(
    common_name: str = typer.Argument(..., help="Certificate common name"),
    cert_dir: str = typer.Option("./certs", "--dir", "-d", help="Certificate directory"),
):
    """Show certificate details."""
    from distllm.cli.cert import cert_info as _cert_info
    _cert_info(common_name, cert_dir)


@security_cert_app.command("renew")
def cert_renew(
    cert_dir: str = typer.Option("./certs", "--dir", "-d", help="Certificate directory"),
):
    """Renew all certificates nearing expiry."""
    from distllm.cli.cert import cert_renew as _cert_renew
    _cert_renew(cert_dir)


@security_cert_app.command("revoke")
def cert_revoke(
    common_name: str = typer.Argument(..., help="Certificate common name"),
    cert_dir: str = typer.Option("./certs", "--dir", "-d", help="Certificate directory"),
):
    """Revoke a certificate."""
    from distllm.cli.cert import cert_revoke as _cert_revoke
    _cert_revoke(common_name, cert_dir)


# --- config webhook group ---
@config_webhook_app.command("register")
def webhook_register(
    url: str = typer.Argument(..., help="Webhook URL"),
    events: list[str] = typer.Option([], "--event", "-e", help="Events to subscribe to"),
    secret: str = typer.Option("", "--secret", "-s", help="HMAC signing secret"),
    label: str = typer.Option("", "--label", "-l", help="Human-readable label"),
):
    """Register a new webhook endpoint."""
    from distllm.cli.webhook import webhook_register as _webhook_register
    _webhook_register(url, events, secret, label)


@config_webhook_app.command("list")
def webhook_list():
    """List registered webhook endpoints."""
    from distllm.cli.webhook import webhook_list as _webhook_list
    _webhook_list()


@config_webhook_app.command("unregister")
def webhook_unregister(
    url: str = typer.Argument(..., help="Webhook URL to remove"),
):
    """Unregister a webhook endpoint."""
    from distllm.cli.webhook import webhook_unregister as _webhook_unregister
    _webhook_unregister(url)


@config_webhook_app.command("test")
def webhook_test(
    url: str = typer.Argument(..., help="Webhook URL to test"),
    event: str = typer.Option("test.ping", "--event", "-e", help="Event type"),
):
    """Send a test webhook event."""
    from distllm.cli.webhook import webhook_test as _webhook_test
    _webhook_test(url, event)


@config_quota_app.command("report")
def quota_report(
    tenant_id: str = typer.Argument("", help="Tenant ID (omit for all)"),
    days: int = typer.Option(30, "--days", "-d", help="Report period in days"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """Generate a detailed usage report with per-model breakdown."""
    from distllm.cli.quota import quota_report as _quota_report
    _quota_report(tenant_id, days, json_output)


@config_quota_app.command("export")
def quota_export(
    filepath: str = typer.Argument(..., help="Output CSV file path"),
    tenant_id: str = typer.Option("", "--tenant", "-t", help="Filter by tenant"),
    days: int = typer.Option(30, "--days", "-d", help="Export period in days"),
):
    """Export usage records as CSV."""
    from distllm.cli.quota import quota_export as _quota_export
    _quota_export(filepath, tenant_id, days)


@config_quota_app.command("import")
def quota_import(
    filepath: str = typer.Argument(..., help="JSON file with quota definitions"),
):
    """Bulk import quotas from a JSON file."""
    from distllm.cli.quota import quota_import as _quota_import
    _quota_import(filepath)


# --- system notify group ---
@system_notify_app.command("send")
def notify_send(
    title: str = typer.Option(..., "--title", "-t", help="Notification title"),
    message: str = typer.Option(..., "--message", "-m", help="Notification message"),
    severity: str = typer.Option("info", "--severity", "-s", help="Severity"),
    channel: str = typer.Option("console", "--channel", "-c", help="Channel"),
    webhook_url: str = typer.Option("", "--webhook-url", help="Webhook URL"),
):
    """Send a test notification."""
    from distllm.cli.notify import notify_send as _notify_send
    _notify_send(title, message, severity, channel, webhook_url)


@system_notify_app.command("history")
def notify_history(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of recent notifications"),
    severity: str = typer.Option("", "--severity", "-s", help="Filter by severity"),
):
    """Show recent notification history."""
    from distllm.cli.notify import notify_history as _notify_history
    _notify_history(limit, severity)


# --- config quota group ---
@config_quota_app.command("set")
def quota_set(
    tenant_id: str = typer.Argument(..., help="Tenant or team ID"),
    max_tokens_per_day: int = typer.Option(0, "--tokens-per-day", help="Max tokens per day"),
    max_requests_per_minute: int = typer.Option(0, "--rpm", help="Max requests per minute"),
    max_tokens_per_request: int = typer.Option(0, "--tokens-per-request", help="Max tokens per request"),
    max_concurrent: int = typer.Option(0, "--concurrent", help="Max concurrent requests"),
    cost_budget: float = typer.Option(0.0, "--budget", "-b", help="Monthly cost budget"),
    overage: bool = typer.Option(False, "--overage", help="Allow overage"),
):
    """Set usage quota for a tenant."""
    from distllm.cli.quota import quota_set as _quota_set
    _quota_set(tenant_id, max_tokens_per_day, max_requests_per_minute, max_tokens_per_request, max_concurrent, cost_budget, overage)


@config_quota_app.command("show")
def quota_show(
    tenant_id: str = typer.Argument(..., help="Tenant or team ID"),
):
    """Show current quota and usage for a tenant."""
    from distllm.cli.quota import quota_show as _quota_show
    _quota_show(tenant_id)


@config_quota_app.command("list")
def quota_list():
    """List all tenants with usage data."""
    from distllm.cli.quota import quota_list as _quota_list
    _quota_list()


@config_quota_app.command("invoice")
def quota_invoice(
    tenant_id: str = typer.Argument(..., help="Tenant or team ID"),
):
    """Generate an invoice for a tenant's current billing period."""
    from distllm.cli.quota import quota_invoice as _quota_invoice
    _quota_invoice(tenant_id)


@app.command()
def chat(
    model: str = typer.Option("distributed-llm", "--model", "-m", help="Model identifier"),
    host: str = typer.Option("localhost", "--host", help="API server host"),
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
    max_tokens: int = typer.Option(256, "--max-tokens", help="Max tokens to generate"),
    temperature: float = typer.Option(0.7, "--temperature", help="Sampling temperature"),
):
    """Interactive chat with the model via the API."""
    from distllm.cli.chat import run_chat
    run_chat(
        model=model,
        host=host,
        port=port,
        max_tokens=max_tokens,
        temperature=temperature,
        console=console,
    )


@system_app.command("coordinator")
def system_coordinator(
    model: str = typer.Option(..., "--model", "-m", help="Model name"),
    port: int = typer.Option(50050, "--port", help="Coordinator port"),
    dtype: str = typer.Option("float16", "--dtype", help="Data type"),
    local: bool = typer.Option(False, "--local", "-l", help="Run full model locally"),
    chat_mode: bool = typer.Option(False, "--chat", help="Interactive chat mode (requires --local)"),
    trust_remote_code: bool = typer.Option(False, "--trust-remote-code", help="Trust remote code"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode"),
):
    """Start the coordinator facade for distributed inference."""
    from distllm.core.coordinator import Coordinator
    from distllm.core.debug import set_debug_mode
    from loguru import logger

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
                if prompt.lower() in ('quit', 'exit'):
                    break
                result = c.generate(prompt, max_new_tokens=128)
                print(f"\nResult: {result}")
        else:
            c.start()
    else:
        c.start()


@system_app.command("api")
def system_api(
    model: str = typer.Option(..., "--model", "-m", help="Model name"),
    host: str = typer.Option("127.0.0.1", "--host", help="API server host"),
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
    dtype: str = typer.Option("float16", "--dtype", help="Data type"),
    local: bool = typer.Option(False, "--local", "-l", help="Load model locally"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode"),
):
    """Start the OpenAI-compatible REST API server."""
    from distllm.api.server import app, create_coordinator
    from distllm.core.debug import set_debug_mode
    from loguru import logger
    import uvicorn

    if debug:
        set_debug_mode(True)
        logger.info("Debug mode enabled: tensor shape logging active")

    create_coordinator(model_name=model, dtype=dtype, local=local)
    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


@app.command()
def dashboard(
    host: str = typer.Option("127.0.0.1", "--host", help="Dashboard host (ignored — dashboard is embedded in API server)"),
    port: int = typer.Option(8500, "--port", "-p", help="Dashboard port (ignored — dashboard is embedded in API server)"),
    api_url: str | None = typer.Option(None, "--api-url", help="DistLLM API server URL"),
):
    """Open the dashboard in the default browser.

    The standalone dashboard was removed in v0.4.0.  The dashboard is now
    embedded in the API server at ``/dashboard`` (default port 8000).
    Use ``distllm serve --port <port>`` to run the API server.
    """
    import webbrowser
    from loguru import logger
    base_url = api_url or "http://localhost:8000"
    dashboard_url = f"{base_url.rstrip('/')}/dashboard"
    logger.info(f"Dashboard is embedded in the API server. Opening {dashboard_url}")
    webbrowser.open(dashboard_url)


@cluster_app.command("deploy")
def cluster_deploy(
    model: str = typer.Argument(..., help="Model name or HuggingFace path"),
    nodes: int = typer.Option(2, "--nodes", "-n", help="Number of nodes"),
    dtype: str = typer.Option("float16", "--dtype", help="Data type"),
    quantization: str = typer.Option("none", "--quantization", help="Quantization method"),
    gpu_type: str | None = typer.Option(None, "--gpu-type", help="Target GPU type"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show plan without deploying"),
    wait: bool = typer.Option(True, "--wait/--no-wait", help="Wait for deployment to complete"),
    host: str = typer.Option("localhost", "--host", help="API server host"),
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
):
    """Deploy a model to the distributed cluster."""
    from distllm.cli.deploy import run_deploy
    run_deploy(model, nodes, dtype, quantization, gpu_type, dry_run, wait, host, port, console)


@system_app.command("slo-report")
def system_slo_report(
    host: str = typer.Option("localhost", "--host", help="API server host"),
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
):
    """Display SLO compliance report (latency, throughput, error rates)."""
    import httpx
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


@benchmark_app.command("profile")
def benchmark_profile(
    model: str = typer.Argument(..., help="Model name"),
    host: str = typer.Option("localhost", "--host", help="API server host"),
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
    prompt_len: int = typer.Option(128, "--prompt-len", help="Prompt length in tokens"),
    gen_len: int = typer.Option(64, "--gen-len", help="Generated tokens"),
    num_iterations: int = typer.Option(10, "--iterations", "-n", help="Number of iterations"),
    output: str | None = typer.Option(None, "--output", "-o", help="Save to JSON file"),
):
    """Profile model inference (latency, throughput, memory)."""
    from distllm.cli.profile import run_profile
    run_profile(model, host, port, prompt_len, gen_len, num_iterations, output, console)


@system_app.command("cost-avoid")
def system_cost_avoid(
    model: str = typer.Option("meta-llama/Llama-3.1-70B", "--model", "-m", help="Model name"),
    requests_per_day: int = typer.Option(10000, "--requests-per-day", "-r", help="Daily request volume"),
    gpu_type: str = typer.Option("RTX 4090", "--gpu-type", help="Your GPU type"),
    cloud_api: str = typer.Option("llama-3.1-70b-deepinfra", "--cloud-api", help="Cloud API for comparison"),
):
    """Calculate monthly savings from self-hosted inference."""
    from distllm.cli.cost_avoid import calculate_cost_avoidance
    result = calculate_cost_avoidance(
        model_name=model, requests_per_day=requests_per_day,
        gpu_type=gpu_type, cloud_api=cloud_api,
    )
    from rich.table import Table
    from rich.console import Console
    console = Console()
    table = Table(title="Cost Avoidance Analysis", show_header=False)
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Model", result["model"])
    table.add_row("Monthly API Cost", f"${result['monthly_api_cost']:,.2f}")
    table.add_row("Monthly Self-Hosted", f"${result['monthly_self_hosted_cost']:,.2f}")
    table.add_row("Monthly Savings", f"${result['monthly_savings']:,.2f} ({result['savings_percent']}%)")
    console.print(table)


@system_app.command("doctor")
def system_doctor():
    """Run system diagnostics (CUDA, ports, config, disk)."""
    from distllm.cli.doctor import main as doctor_main
    doctor_main()


# --- Draft-as-a-Service (DaaS) group ---
daas_app = typer.Typer(help="Draft-as-a-Service: deploy and manage draft model endpoints")
app.add_typer(daas_app, name="daas")


@daas_app.command("serve")
def daas_serve(
    model: str = typer.Option("SmolLM-135M", "--model", "-m", help="Draft model name"),
    host: str = typer.Option("0.0.0.0", "--host", help="Bind host"),
    port: int = typer.Option(9000, "--port", "-p", help="Bind port"),
    api_key: str = typer.Option("", "--api-key", help="API key for authentication"),
    max_concurrent: int = typer.Option(10, "--max-concurrent", help="Max concurrent requests"),
    rate_limit: int = typer.Option(60, "--rate-limit", help="Requests per minute per key"),
    cost_per_hour: float = typer.Option(0.05, "--cost", help="Cost per hour"),
    hardware: str = typer.Option("cpu", "--hardware", help="Hardware type (cpu, cuda:0, mps)"),
    dtype: str = typer.Option("float16", "--dtype", help="Data type"),
):
    """Start a Draft-as-a-Service inference server.

    Exposes an OpenAI-compatible /v1/completions endpoint that serves
    draft tokens for speculative decoding from any CPU/edge device.

    Example::

        distllm daas serve --model SmolLM-135M --port 9000
    """
    from distllm.dist.daas_server import DaaSServer, DaaSConfig

    config = DaaSConfig(
        model_name=model,
        host=host,
        port=port,
        api_key=api_key,
        max_concurrent=max_concurrent,
        rate_limit_per_minute=rate_limit,
        cost_per_hour=cost_per_hour,
        hardware=hardware,
        dtype=dtype,
    )
    server = DaaSServer(config)
    server.run()


@daas_app.command("status")
def daas_status(
    host: str = typer.Option("localhost", "--host", help="DaaS server host"),
    port: int = typer.Option(9000, "--port", "-p", help="DaaS server port"),
):
    """Show DaaS server status and metrics."""
    import httpx

    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"http://{host}:{port}/metrics")
            resp.raise_for_status()
            metrics = resp.json()

            console.print("\n[bold]Draft-as-a-Service Status[/bold]")
            console.print(f"  Model: {metrics.get('model', 'N/A')}")
            console.print(f"  Hardware: {metrics.get('hardware', 'N/A')}")
            console.print(f"  Total requests: {metrics.get('total_requests', 0)}")
            console.print(f"  Total tokens: {metrics.get('total_tokens_generated', 0)}")
            console.print(f"  Avg latency: {metrics.get('avg_latency_ms', 0):.1f}ms")
            console.print(f"  Tokens/sec: {metrics.get('tokens_per_second', 0):.1f}")
            console.print(f"  Active: {metrics.get('active_requests', 0)}")
            console.print(f"  Errors: {metrics.get('errors', 0)}")
            console.print(f"  Uptime: {metrics.get('uptime_s', 0):.0f}s")
            console.print(f"  Cost: ${metrics.get('cost_per_hour', 0):.2f}/hr")
    except httpx.ConnectError:
        console.print(f"[red]Could not connect to DaaS server at {host}:{port}[/red]")


@daas_app.command("benchmark")
def daas_benchmark(
    host: str = typer.Option("localhost", "--host", help="DaaS server host"),
    port: int = typer.Option(9000, "--port", "-p", help="DaaS server port"),
    requests: int = typer.Option(100, "--requests", "-n", help="Number of requests"),
    tokens: int = typer.Option(16, "--tokens", "-t", help="Tokens per request"),
    concurrent: int = typer.Option(4, "--concurrent", "-c", help="Concurrent requests"),
):
    """Benchmark a DaaS server's throughput and latency."""
    import httpx
    import asyncio
    import time

    async def _bench():
        url = f"http://{host}:{port}/v1/completions"
        payload = {"prompt": [1, 2, 3], "max_tokens": tokens}
        latencies: list[float] = []
        errors = 0

        semaphore = asyncio.Semaphore(concurrent)

        async def _single_request(client: httpx.AsyncClient) -> None:
            nonlocal errors
            async with semaphore:
                start = time.monotonic()
                try:
                    resp = await client.post(url, json=payload, timeout=30.0)
                    resp.raise_for_status()
                    latencies.append(time.monotonic() - start)
                except Exception:
                    errors += 1

        async with httpx.AsyncClient() as client:
            tasks = [_single_request(client) for _ in range(requests)]
            total_start = time.monotonic()
            await asyncio.gather(*tasks)
            total_time = time.monotonic() - total_start

        if latencies:
            latencies.sort()
            console.print(f"\n[bold]DaaS Benchmark Results[/bold]")
            console.print(f"  Requests: {requests} ({errors} errors)")
            console.print(f"  Total time: {total_time:.2f}s")
            console.print(f"  Throughput: {len(latencies) / total_time:.1f} req/s")
            console.print(f"  Tokens/sec: {len(latencies) * tokens / total_time:.0f}")
            console.print(f"  Latency p50: {latencies[len(latencies) // 2] * 1000:.1f}ms")
            console.print(f"  Latency p95: {latencies[int(len(latencies) * 0.95)] * 1000:.1f}ms")
            console.print(f"  Latency p99: {latencies[int(len(latencies) * 0.99)] * 1000:.1f}ms")
        else:
            console.print("[red]All requests failed[/red]")

    asyncio.run(_bench())


# --- Draft fleet management ---
draft_app = typer.Typer(help="Manage draft model fleet: list, status, migrate")
app.add_typer(draft_app, name="draft")


@draft_app.command("fleet-status")
def draft_fleet_status(
    host: str = typer.Option("localhost", "--host", help="Coordinator host"),
    port: int = typer.Option(8000, "--port", "-p", help="Coordinator API port"),
):
    """Show status of the heterogeneous draft model fleet."""
    import httpx

    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"http://{host}:{port}/v1/speculative/fleet")
            resp.raise_for_status()
            data = resp.json()

            console.print("\n[bold]Draft Model Fleet[/bold]")
            console.print(f"  Total endpoints: {data.get('total_endpoints', 0)}")
            console.print(f"  Healthy: {data.get('healthy_endpoints', 0)}")
            console.print(f"  Total calls: {data.get('total_calls', 0)}")
            console.print(f"  Avg latency: {data.get('avg_latency_ms', 0):.1f}ms")
            console.print(f"  Error rate: {data.get('error_rate', 0):.1%}")

            endpoints = data.get("endpoints", [])
            if endpoints:
                table = Table(title="Draft Endpoints")
                table.add_column("Model")
                table.add_column("Hardware")
                table.add_column("Cost/hr")
                table.add_column("Latency")
                table.add_column("Calls")
                table.add_column("Healthy")
                for ep in endpoints:
                    table.add_row(
                        ep.get("model", "N/A"),
                        ep.get("hardware", "N/A"),
                        f"${ep.get('cost_per_hour', 0):.2f}",
                        f"{ep.get('avg_latency_ms', 0):.1f}ms",
                        str(ep.get("calls", 0)),
                        "✓" if ep.get("healthy") else "✗",
                    )
                console.print(table)
    except httpx.ConnectError:
        console.print(f"[red]Could not connect to coordinator at {host}:{port}[/red]")


@draft_app.command("migration-status")
def draft_migration_status(
    host: str = typer.Option("localhost", "--host", help="Coordinator host"),
    port: int = typer.Option(8000, "--port", "-p", help="Coordinator API port"),
):
    """Show auto-migration status (CPU↔GPU draft model placement)."""
    import httpx

    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"http://{host}:{port}/v1/speculative/migration")
            resp.raise_for_status()
            data = resp.json()

            console.print("\n[bold]Draft Migration Status[/bold]")
            console.print(f"  Enabled: {data.get('enabled', False)}")
            console.print(f"  Active: {data.get('active_endpoint', 'N/A')} ({data.get('active_hardware', 'N/A')})")
            console.print(f"  CPU endpoints: {data.get('cpu_endpoints', 0)}")
            console.print(f"  GPU endpoints: {data.get('gpu_endpoints', 0)}")
            console.print(f"  Total migrations: {data.get('total_migrations', 0)}")
            config = data.get("config", {})
            if config:
                console.print(f"  GPU high threshold: {config.get('gpu_high_threshold', 80)}%")
                console.print(f"  GPU low threshold: {config.get('gpu_low_threshold', 40)}%")
    except httpx.ConnectError:
        console.print(f"[red]Could not connect to coordinator at {host}:{port}[/red]")


@app.command()
def tutorial():
    """Interactive guided setup for first-time users."""
    from distllm.cli.tutorial import main as tutorial_main
    tutorial_main()


@cluster_app.command("autopsy")
def cluster_autopsy(
    host: str = typer.Option("localhost", "--host", help="Coordinator host"),
    port: int = typer.Option(8000, "--port", "-p", help="Coordinator API port"),
    since: int = typer.Option(60, "--since", help="Look back N minutes for logs"),
    output: str = typer.Option("", "--output", "-o", help="Output zip path"),
):
    """Collect diagnostic data into a report after a failure."""
    from distllm.cli.autopsy import collect_autopsy
    collect_autopsy(coordinator_host=host, coordinator_port=port, since_minutes=since, output_path=output)


@system_app.command("observe")
def system_observe(
    coordinator_url: str = typer.Option("http://localhost:8000", "--coordinator", help="Coordinator API URL"),
    metrics_port: int = typer.Option(9090, "--metrics-port", help="Prometheus metrics port"),
    dashboard_port: int = typer.Option(3000, "--dashboard-port", help="Grafana dashboard port"),
    loki_url: str = typer.Option("http://localhost:3100", "--loki-url", help="Loki URL"),
    prometheus_url: str = typer.Option("http://localhost:9090", "--prometheus-url", help="Prometheus URL"),
    metrics_only: bool = typer.Option(False, "--metrics-only", help="Only expose metrics endpoint"),
):
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


# ── Tune command group ──────────────────────────────────────────────────

tune_app = typer.Typer(help="Adaptive Precision Optimizer: quantization tuning")
app.add_typer(tune_app, name="tune")


@tune_app.command("quantize")
def tune_quantize(
    model: str = typer.Option(..., "--model", "-m", help="Model name or path"),
    nodes: int = typer.Option(1, "--nodes", "-n", help="Number of nodes"),
    max_quality_loss: float = typer.Option(0.05, "--max-quality-loss", "-q", help="Max acceptable quality loss (0.0-1.0)"),
    prefer_speed: bool = typer.Option(False, "--prefer-speed", "-s", help="Prefer faster methods over smaller models"),
    require_calibration: bool = typer.Option(False, "--no-calibration", help="Only use methods that don't need calibration data"),
    output: str | None = typer.Option(None, "--output", "-o", help="Output file path (JSON)"),
    benchmark: bool = typer.Option(False, "--benchmark", "-b", help="Run live hardware benchmarks instead of using static profiles"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON instead of human-readable text"),
):
    """Run Adaptive Precision Optimizer to select optimal quantization per device."""
    import asyncio
    from pathlib import Path

    try:
        from distllm.dist.partition.quantization_tuner import (
            QuantizationAutoTuner, NodeInfo, QuantMethod,
        )
        from distllm.dist.partition.quant_report import ReportGenerator

        # Build node info from GPU profiling
        if benchmark:
            from distllm.dist.partition.quant_bench import QuantBenchmarker
            console.print("[bold]Running live hardware benchmarks...[/bold]")
            benchmarker = QuantBenchmarker()
            suites = benchmarker.benchmark_all_gpus()
            for suite in suites:
                console.print(suite.summary())
        else:
            console.print("[dim]Using static quantization profiles (use --benchmark for live data)[/dim]")

        # Profile GPUs for node info
        from distllm.dist.partition.profiles import GPUProfiler
        profiler = GPUProfiler()
        gpu_profiles = profiler.profile_all_gpus()

        # Build node list
        node_infos: list[NodeInfo] = []
        for i, gp in enumerate(gpu_profiles):
            # Distribute layers evenly
            node_infos.append(NodeInfo.from_gpu_profile(
                gp, node_id=f"node-{i}",
            ))

        if not node_infos:
            console.print("[red]No GPUs found. Cannot generate quantization plan.[/red]")
            raise typer.Exit(1)

        # Estimate model size (would need model metadata in production)
        # For now, use a heuristic based on model name
        model_size_bytes = _estimate_model_size(model)
        num_layers = _estimate_num_layers(model)

        console.print(f"\n[bold]Model:[/bold] {model}")
        console.print(f"[bold]Size:[/bold] {model_size_bytes / (1024**3):.1f} GB (fp16)")
        console.print(f"[bold]Layers:[/bold] {num_layers}")
        console.print(f"[bold]Nodes:[/bold] {len(node_infos)}")
        console.print()

        # Run APO
        tuner = QuantizationAutoTuner(
            max_quality_loss=max_quality_loss,
            prefer_speed=prefer_speed,
            require_calibration=require_calibration,
        )
        plan = tuner.recommend(node_infos, model_size_bytes, num_layers)

        # Generate report
        reporter = ReportGenerator()
        report = reporter.generate(plan, node_infos, model_size_bytes, num_layers)

        if json_output or output:
            report_data = report.to_json()
            if output:
                Path(output).write_text(report_data, encoding="utf-8")
                console.print(f"[green]Report saved to {output}[/green]")
            if json_output:
                console.print(report_data)
        else:
            console.print(report.to_text())

    except ImportError as e:
        console.print(f"[red]Missing dependency: {e}[/red]")
        console.print("[dim]Install with: pip install distllm[self-hosted][/dim]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


def _estimate_model_size(model_name: str) -> int:
    """Rough model size estimate from model name."""
    name_lower = model_name.lower()
    if "70b" in name_lower:
        return 140 * 1024**3  # 140GB fp16
    if "34b" in name_lower or "33b" in name_lower:
        return 68 * 1024**3
    if "13b" in name_lower:
        return 26 * 1024**3
    if "7b" in name_lower or "8b" in name_lower:
        return 14 * 1024**3
    if "3b" in name_lower:
        return 6 * 1024**3
    if "1b" in name_lower:
        return 2 * 1024**3
    # Default: assume 7B
    return 14 * 1024**3


def _estimate_num_layers(model_name: str) -> int:
    """Rough layer count estimate from model name."""
    name_lower = model_name.lower()
    if "70b" in name_lower:
        return 80
    if "34b" in name_lower or "33b" in name_lower:
        return 48
    if "13b" in name_lower:
        return 40
    if "7b" in name_lower or "8b" in name_lower:
        return 32
    if "3b" in name_lower:
        return 28
    if "1b" in name_lower:
        return 22
    return 32


if __name__ == "__main__":
    app()


def main():
    """Entry point for the distllm CLI."""
    app()
