"""DistLLM CLI - Unified command-line interface for Distributed LLM."""

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="distllm",
    help="Distributed LLM Inference System - Run large language models across multiple GPU-equipped machines",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()

# --- Command groups ---
models_app = typer.Typer(help="Manage models (list, load, unload, info)")
cluster_app = typer.Typer(help="Manage cluster (start, join, status, leave, scale, drain, rebalance)")
adapters_app = typer.Typer(help="Manage adapters (list, load, set, unload)")
logs_app = typer.Typer(help="Stream and filter logs")
benchmark_app = typer.Typer(help="Run and compare benchmarks")

app.add_typer(models_app, name="models")
app.add_typer(cluster_app, name="cluster")
app.add_typer(adapters_app, name="adapters")
app.add_typer(logs_app, name="logs")
app.add_typer(benchmark_app, name="benchmark")


@app.command()
def setup(
    config_path: str = typer.Option("config.yaml", "--config", "-c", help="Output config file path"),
):
    """Interactive setup wizard for cluster configuration."""
    from distllm.cli.setup import run_setup
    run_setup(config_path, console)


@app.command()
def run(
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


@app.command()
def validate_config():
    """Validate configuration and exit."""
    from distllm.config.settings import DistLLMSettings
    try:
        DistLLMSettings.validate_startup()
        console.print("[green]Config validation passed[/green]")
    except SystemExit:
        raise typer.Exit(1) from None


@app.command()
def status(
    host: str = typer.Option("localhost", "--host", help="API server host"),
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
):
    """Show cluster status and node health."""
    from distllm.cli.status import show_status
    show_status(host, port, console)


# --- models group ---
@models_app.command("list")
def models_list(
    host: str = typer.Option("localhost", "--host", help="API server host"),
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
):
    """List available and loaded models."""
    from distllm.cli.models import _list_models
    _list_models(host, port)


@models_app.command("info")
def models_info(
    model_id: str = typer.Argument(..., help="Model identifier"),
    host: str = typer.Option("localhost", "--host", help="API server host"),
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
):
    """Show detailed model information."""
    from distllm.cli.models import _model_info
    _model_info(host, port, model_id)


@models_app.command("load")
def models_load(
    model: str = typer.Argument(..., help="Model name or path"),
    host: str = typer.Option("localhost", "--host", help="API server host"),
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
    dtype: str = typer.Option("float16", "--dtype", help="Data type"),
):
    """Load a model into the server."""
    from distllm.cli.models import _load_model
    _load_model(host, port, model, dtype)


@models_app.command("unload")
def models_unload(
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
):
    """Start a coordinator node and begin accepting workers."""
    from distllm.cli.cluster import _cluster_start
    _cluster_start(model, port, api_port, local, dtype, debug)


@cluster_app.command("join")
def cluster_join(
    coordinator_host: str = typer.Option("localhost", "--coordinator", "-c", help="Coordinator hostname or IP"),
    coordinator_port: int = typer.Option(50050, "--port", "-p", help="Coordinator gRPC port"),
    node_id: str = typer.Option(None, "--node-id", help="Unique node ID (auto-generated if omitted)"),
    start_layer: int = typer.Option(None, "--start-layer", help="First layer to serve"),
    end_layer: int = typer.Option(None, "--end-layer", help="Last layer to serve"),
    total_layers: int = typer.Option(None, "--total-layers", help="Total model layers"),
    port: int = typer.Option(50051, "--listen-port", help="This node's gRPC port"),
    device: str = typer.Option("auto", "--device", help="Device (auto, cuda, cpu)"),
    cluster_key: str = typer.Option(None, "--cluster-key", help="Shared cluster authentication key"),
):
    """Join an existing cluster as a worker node."""
    from distllm.cli.cluster import _cluster_join
    _cluster_join(
        coordinator_host, coordinator_port, node_id,
        start_layer, end_layer, total_layers, port, device, cluster_key,
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


# --- adapters group ---
@adapters_app.command("list")
def adapters_list(
    host: str = typer.Option("localhost", "--host", help="API server host"),
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
):
    """List loaded adapters."""
    from distllm.cli.adapters import _list_adapters
    _list_adapters(host, port)


@adapters_app.command("load")
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


@adapters_app.command("set")
def adapters_set(
    adapter_id: str = typer.Argument(..., help="Adapter identifier"),
    host: str = typer.Option("localhost", "--host", help="API server host"),
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
):
    """Set an adapter as the active adapter."""
    from distllm.cli.adapters import _set_adapter
    _set_adapter(host, port, adapter_id)


@adapters_app.command("unload")
def adapters_unload(
    adapter_id: str = typer.Argument(..., help="Adapter identifier"),
    host: str = typer.Option("localhost", "--host", help="API server host"),
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
):
    """Unload an adapter."""
    from distllm.cli.adapters import _unload_adapter
    _unload_adapter(host, port, adapter_id)


# --- logs group ---
@logs_app.command("stream")
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


@app.command()
def compress(
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


@app.command()
def coordinator(
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


@app.command()
def api(
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
    host: str = typer.Option("127.0.0.1", "--host", help="Dashboard host"),
    port: int = typer.Option(8500, "--port", "-p", help="Dashboard port"),
    api_url: str | None = typer.Option(None, "--api-url", help="DistLLM API server URL"),
):
    """Start the web monitoring dashboard."""
    from distllm.dashboard.app import dashboard_app
    from loguru import logger
    import uvicorn

    logger.info(f"Starting dashboard on {host}:{port}")
    uvicorn.run(dashboard_app, host=host, port=port, log_level="info")


@app.command()
def deploy(
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


@app.command()
def profile(
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


if __name__ == "__main__":
    app()


def main():
    """Entry point for the distllm CLI."""
    app()
