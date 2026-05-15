"""DistLLM CLI - Unified command-line interface for Distributed LLM."""

import typer
from rich.console import Console
from rich.table import Table
from typing import Optional

app = typer.Typer(
    name="distllm",
    help="Distributed LLM Inference System - Run large language models across multiple GPU-equipped machines",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()


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
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Config file path"),
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
    dtype: str = typer.Option("float16", "--dtype", help="Data type (float16, float32, bfloat16)",),
    max_tokens: int = typer.Option(256, "--max-tokens", help="Max tokens to generate"),
    temperature: float = typer.Option(0.7, "--temperature", help="Sampling temperature"),
    prompt: Optional[str] = typer.Option(None, "--prompt", help="Single prompt (non-interactive)"),
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
        raise typer.Exit(1)


@app.command()
def status(
    host: str = typer.Option("localhost", "--host", help="API server host"),
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
):
    """Show cluster status and node health."""
    from distllm.cli.status import show_status
    show_status(host, port, console)


@app.command()
def benchmark(
    model: str = typer.Option(..., "--model", "-m", help="Model to benchmark"),
    host: str = typer.Option("localhost", "--host", help="API server host"),
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
    prompts: int = typer.Option(5, "--prompts", help="Number of test prompts"),
    max_tokens: int = typer.Option(50, "--max-tokens", help="Max tokens per prompt"),
    local: bool = typer.Option(False, "--local", help="Benchmark local mode"),
):
    """Run benchmarks against the API server."""
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
def node(
    node_id: str = typer.Option(..., "--node-id", help="Unique node identifier"),
    model: str = typer.Option(..., "--model", "-m", help="HuggingFace model name or path"),
    start_layer: int = typer.Option(..., "--start-layer", help="First layer to run"),
    end_layer: int = typer.Option(..., "--end-layer", help="Last layer to run"),
    total_layers: int = typer.Option(..., "--total-layers", help="Total layers in model"),
    port: int = typer.Option(50051, "--port", help="gRPC port"),
    coordinator_host: str = typer.Option("localhost", "--coordinator-host", help="Coordinator host"),
    coordinator_port: int = typer.Option(50050, "--coordinator-port", help="Coordinator port"),
    device: str = typer.Option("auto", "--device", help="Device (auto, cuda, cpu)"),
    dtype: str = typer.Option("float16", "--dtype", help="Data type"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode with tensor shape logging"),
):
    """Start a worker node that runs a subset of model layers."""
    from distllm.core.node import WorkerNode
    from distllm.communication.grpc import set_debug_mode
    from loguru import logger

    if debug:
        set_debug_mode(True)
        logger.info("Debug mode enabled: tensor shape logging active")

    n = WorkerNode(
        node_id=node_id,
        model_name=model,
        start_layer=start_layer,
        end_layer=end_layer,
        total_layers=total_layers,
        port=port,
        coordinator_host=coordinator_host,
        coordinator_port=coordinator_port,
        device=device,
        dtype=dtype,
    )
    n.start()


@app.command()
def coordinator(
    model: str = typer.Option(..., "--model", "-m", help="Model name"),
    port: int = typer.Option(50050, "--port", help="Coordinator gRPC port"),
    dtype: str = typer.Option("float16", "--dtype", help="Data type"),
    local: bool = typer.Option(False, "--local", "-l", help="Run full model locally"),
    chat_mode: bool = typer.Option(False, "--chat", help="Interactive chat mode (requires --local)"),
    trust_remote_code: bool = typer.Option(False, "--trust-remote-code", help="Trust remote code"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode"),
):
    """Start the coordinator facade for distributed inference."""
    from distllm.core.coordinator import Coordinator
    from distllm.communication.grpc import set_debug_mode
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
    host: str = typer.Option("0.0.0.0", "--host", help="API server host"),
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
    dtype: str = typer.Option("float16", "--dtype", help="Data type"),
    local: bool = typer.Option(False, "--local", "-l", help="Load model locally"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode"),
):
    """Start the OpenAI-compatible REST API server."""
    from distllm.api.server import app, create_coordinator
    from distllm.communication.grpc import set_debug_mode
    from loguru import logger
    import uvicorn

    if debug:
        set_debug_mode(True)
        logger.info("Debug mode enabled: tensor shape logging active")

    create_coordinator(model_name=model, dtype=dtype, local=local)
    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


@app.command()
def client(
    model: str = typer.Option("roneneldan/TinyStories-1M", "--model", "-m", help="Model name"),
    chat_mode: bool = typer.Option(False, "--chat", help="Interactive chat mode"),
    prompt: Optional[str] = typer.Option(None, "--prompt", help="Single prompt"),
    health: bool = typer.Option(False, "--health", help="Show node health"),
    max_tokens: int = typer.Option(128, "--max-tokens", help="Max tokens"),
    temperature: float = typer.Option(0.7, "--temperature", help="Sampling temperature"),
    dtype: str = typer.Option("float32", "--dtype", help="Data type"),
    local: bool = typer.Option(False, "--local", "-l", help="Load model locally"),
):
    """Client for distributed LLM inference (chat, prompts, health)."""
    import time
    from distllm.core.coordinator import Coordinator

    coordinator = Coordinator(model_name=model, dtype=dtype)

    if local:
        print(f"Loading model locally: {model}")
        coordinator.load_local_model()
    else:
        print(f"Client ready for model: {model}")

    if health:
        if not coordinator.nodes:
            print("No remote nodes registered.")
        else:
            print("Node Health Status:")
            print("-" * 50)
            health_data = coordinator.health_check()
            for node_id, status in health_data.items():
                node = coordinator.nodes[node_id]
                h = "HEALTHY" if status.get("healthy") else "UNHEALTHY"
                print(f"  {node_id}: {h} (layers {node.start_layer}-{node.end_layer})")
    elif prompt:
        print(f"Model: {model}\nPrompt: {prompt}\n\nGenerating...\n")
        start = time.time()
        result = coordinator.generate(prompt, max_new_tokens=max_tokens, temperature=temperature)
        elapsed = time.time() - start
        print(result)
        tokens = len(coordinator.tokenizer.encode(result))
        print(f"\n---\nGenerated {tokens} tokens in {elapsed:.1f}s ({tokens/elapsed:.1f} tok/s)")
    elif chat_mode:
        print("=" * 60)
        print("Distributed LLM - Interactive Chat")
        print(f"Model: {model}")
        print("Type 'quit' or 'exit' to stop")
        print("=" * 60)
        conversation = []
        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            if user_input.lower() in ('quit', 'exit', 'q'):
                print("Goodbye!")
                break
            if not user_input:
                continue
            conversation.append({"role": "user", "content": user_input})
            full_prompt = "\n".join(f"{m['role']}: {m['content']}" for m in conversation)
            print("\nAssistant: ", end="", flush=True)
            start = time.time()
            result = coordinator.generate(full_prompt, max_new_tokens=256, temperature=0.7, top_p=0.9)
            response = result[len(full_prompt):] if result.startswith(full_prompt) else result
            print(response.strip())
            elapsed = time.time() - start
            tokens = len(coordinator.tokenizer.encode(response))
            print(f"\n[{tokens} tokens in {elapsed:.1f}s | {tokens/elapsed:.1f} tok/s]")
            conversation.append({"role": "assistant", "content": response.strip()})
    else:
        print("\nUsage:")
        print(f"  distllm client --model {model} --local --chat")
        print(f"  distllm client --model {model} --local --prompt \"Hello world\"")
        print(f"  distllm client --model {model} --health")


@app.command()
def tp(
    model: str = typer.Option(..., "--model", "-m", help="Model name"),
    num_gpus: int = typer.Option(2, "--num-gpus", help="Number of GPUs"),
    dtype: str = typer.Option("float16", "--dtype", help="Data type"),
    trust_remote_code: bool = typer.Option(False, "--trust-remote-code", help="Trust remote code"),
    port: int = typer.Option(29500, "--port", help="Master port for NCCL"),
):
    """Launch tensor parallel workers for multi-GPU inference."""
    from distllm.core.tp_launcher import launch_tp_workers

    launch_tp_workers(
        model_name=model,
        num_gpus=num_gpus,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        port=port,
    )


@app.command()
def dashboard(
    host: str = typer.Option("0.0.0.0", "--host", help="Dashboard host"),
    port: int = typer.Option(8500, "--port", "-p", help="Dashboard port"),
    api_url: Optional[str] = typer.Option(None, "--api-url", help="DistLLM API server URL"),
):
    """Start the web monitoring dashboard."""
    from distllm.dashboard.app import dashboard_app
    from loguru import logger
    import uvicorn

    logger.info(f"Starting dashboard on {host}:{port}")
    uvicorn.run(dashboard_app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()


def main():
    """Entry point for the distllm CLI."""
    app()
