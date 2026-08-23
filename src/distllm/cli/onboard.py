"""``distllm onboard`` — Interactive first-run wizard for DistLLM.

Detects hardware, recommends a model, generates configuration, and
optionally starts the cluster — all with a friendly Rich-based UI.

Steps:
  1. Welcome + hardware detection display (rich.table.Table)
  2. Model selection (recommended models based on VRAM)
  3. Download selected model (with rich.progress.Progress)
  4. Configuration generation (cluster name, port)
  5. Next-steps card (how to start, where to access)
"""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path
from typing import Any

import psutil
import yaml
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TransferSpeedColumn,
)
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text


# ── Hardware detection ─────────────────────────────────────────────────────

def _detect_hardware() -> dict[str, Any]:
    """Detect CPU, RAM, GPU count, GPU memory, and CUDA availability.

    Returns a flat dictionary with keys:
      cpu_cores, cpu_logical, ram_gb, cuda_available, gpu_count,
      gpu_name, gpu_memory_gb, platform.
    """
    info: dict[str, Any] = {
        "cpu_cores": psutil.cpu_count(logical=False) or 0,
        "cpu_logical": psutil.cpu_count(logical=True) or 0,
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "cuda_available": False,
        "gpu_count": 0,
        "gpu_name": "",
        "gpu_memory_gb": 0.0,
        "platform": sys.platform,
    }

    try:
        import torch  # noqa: F811

        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["gpu_count"] = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = props.name
            info["gpu_memory_gb"] = round(props.total_memory / 1e9, 1)
    except ImportError:
        info["cuda_available"] = False

    return info


def _build_hardware_table(info: dict[str, Any]) -> Table:
    """Build a rich.table.Table summarising detected hardware."""
    table = Table(
        title="[bold]System Detection Results[/bold]",
        title_justify="left",
        box=None,
        show_header=False,
        padding=(0, 2),
    )
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    table.add_row("Platform", info["platform"])
    table.add_row("CPU (physical cores)", str(info["cpu_cores"]))
    table.add_row("CPU (logical threads)", str(info["cpu_logical"]))
    table.add_row("System RAM", f"{info['ram_gb']:.1f} GB")

    if info["cuda_available"]:
        table.add_row("CUDA Available", "[green]Yes[/green]")
        table.add_row("GPU Count", str(info["gpu_count"]))
        table.add_row("Primary GPU", info["gpu_name"])
        table.add_row("GPU Memory", f"{info['gpu_memory_gb']:.1f} GB")
    else:
        table.add_row("CUDA Available", "[yellow]No[/yellow]")
        table.add_row("GPU", "[dim]No CUDA GPU detected[/dim]")

    return table


# ── Model recommendation ───────────────────────────────────────────────────

_MODEL_TIERS: list[tuple[float, float, list[dict[str, str]]]] = [
    (0, 8, [
        {"id": "meta-llama/Llama-3.2-1B", "name": "Llama 3.2 1B"},
        {"id": "microsoft/Phi-3-mini-4k-instruct", "name": "Phi-3 Mini"},
        {"id": "Qwen/Qwen2.5-1.5B-Instruct", "name": "Qwen 2.5 1.5B"},
        {"id": "google/gemma-2-2b", "name": "Gemma 2 2B"},
    ]),
    (8, 16, [
        {"id": "meta-llama/Llama-3.1-8B", "name": "Llama 3.1 8B"},
        {"id": "mistralai/Mistral-7B-v0.3", "name": "Mistral 7B"},
        {"id": "google/gemma-2-9b", "name": "Gemma 2 9B"},
    ]),
    (16, 24, [
        {"id": "meta-llama/Llama-3-70b", "name": "Llama 3 70B (4-bit)"},
        {"id": "mistralai/Mixtral-8x7B-v0.1", "name": "Mixtral 8x7B"},
    ]),
    (24, float("inf"), [
        {"id": "meta-llama/Llama-3.1-70B", "name": "Llama 3.1 70B"},
        {"id": "Qwen/Qwen2.5-72B-Instruct", "name": "Qwen 2.5 72B"},
    ]),
]


def _recommend_models(info: dict[str, Any]) -> list[dict[str, str]]:
    """Return the list of recommended models for the detected GPU memory."""
    gpu_mem = info["gpu_memory_gb"]
    for lo, hi, models in _MODEL_TIERS:
        if lo <= gpu_mem < hi:
            return models
    # Fallback: CPU-only, recommend tiny models
    return [
        {"id": "roneneldan/TinyStories-1M", "name": "TinyStories 1M (CPU)"},
        {"id": "microsoft/Phi-3-mini-4k-instruct", "name": "Phi-3 Mini (CPU)"},
    ]


def _build_model_table(
    models: list[dict[str, str]],
    selected_index: int | None = None,
) -> Table:
    """Build a table showing recommended models.

    If *selected_index* is given, that row is highlighted.
    """
    table = Table(
        title="[bold]Recommended Models[/bold]",
        title_justify="left",
        box=None,
        show_header=True,
        header_style="bold cyan",
        padding=(0, 2),
    )
    table.add_column("#", justify="right", style="dim")
    table.add_column("Model Name", style="white")
    table.add_column("HuggingFace ID", style="dim", no_wrap=True)

    for i, m in enumerate(models, 1):
        style = "bold green" if selected_index == i else "white"
        table.add_row(str(i), m["name"], m["id"], style=style)

    return table


# ── Config generation ──────────────────────────────────────────────────────

def _generate_config(
    model_id: str,
    cluster_name: str,
    coordinator_port: int,
    api_port: int,
    dtype: str,
) -> dict[str, Any]:
    """Build a config dictionary suitable for writing to ~/.distllm/config.yaml."""
    return {
        "cluster": {
            "name": cluster_name,
        },
        "model": {
            "name": model_id,
            "dtype": dtype,
        },
        "coordinator": {
            "host": "0.0.0.0",
            "port": coordinator_port,
            "api_port": api_port,
        },
        "hardware": {
            "device_type": "auto",
        },
        "generation": {
            "max_new_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
        },
        "network": {
            "grpc_timeout": 30,
            "max_retries": 3,
            "retry_delay": 1.0,
        },
    }


def _save_config(config: dict[str, Any], config_path: str) -> None:
    """Write *config* to *config_path* as YAML, creating parent dirs."""
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


# ── Simulated download ─────────────────────────────────────────────────────

def _download_model(model_id: str, console: Console) -> bool:
    """Download *model_id* from HuggingFace Hub with a progress bar.

    Returns True on success, False on failure.
    """
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.errors import (
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
    )

    console.print()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task(
            f"[cyan]Downloading {model_id}...",
            total=None,
        )

        try:
            # Pre-auth check with a cheap API call
            api = HfApi()
            try:
                api.model_info(model_id, timeout=10)
            except RepositoryNotFoundError:
                progress.remove_task(task)
                console.print(
                    f"[red]Model '{model_id}' not found on HuggingFace Hub.[/red]"
                )
                return False

            # Use a known-good repo size estimate for progress
            try:
                info = api.model_info(model_id, timeout=10)
                siblings = getattr(info, "siblings", [])
                total_bytes = sum(
                    getattr(s, "size", 0) or 0 for s in siblings
                )
                if total_bytes > 0:
                    progress.update(task, total=total_bytes)
            except Exception:
                total_bytes = 0

            progress.update(task, description=f"[cyan]Downloading {model_id}...")
            snap_path = snapshot_download(
                repo_id=model_id,
                local_files_only=False,
                resume_download=True,
                ignore_patterns=["*.h5", "*.ot", "*.msgpack"],
            )

            progress.update(task, completed=progress.tasks[task].total or 1)
            return True

        except LocalEntryNotFoundError:
            progress.remove_task(task)
            console.print(
                f"[yellow]Model '{model_id}' not found in cache. "
                "Download failed or was interrupted.[/yellow]"
            )
            return False
        except OSError as exc:
            progress.remove_task(task)
            console.print(f"[red]Network error downloading model: {exc}[/red]")
            return False
        except Exception as exc:
            progress.remove_task(task)
            console.print(f"[red]Unexpected error: {exc}[/red]")
            return False


# ── Next-steps card ────────────────────────────────────────────────────────

def _render_next_steps(
    config_path: str,
    model_id: str,
    coordinator_port: int,
    api_port: int,
    cluster_name: str,
) -> str:
    """Return a Markdown-formatted next-steps card."""
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except OSError:
        local_ip = "127.0.0.1"

    return f"""\
## Next Steps

Your DistLLM cluster **{cluster_name}** has been configured and is ready.

### Start the coordinator

```bash
distllm cluster start \\
    --model "{model_id}" \\
    --port {coordinator_port} \\
    --api-port {api_port}
```

### Join worker nodes

On each additional machine with a GPU, run:

```bash
distllm cluster join \\
    --coordinator {local_ip}:{coordinator_port} \\
    --device auto
```

### Query the API

```bash
curl http://localhost:{api_port}/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -H "Authorization: Bearer $DISTLLM_API_KEY" \\
    -d '{{
      "model": "{model_id}",
      "messages": [{{"role": "user", "content": "Hello!"}}]
    }}'
```

### Configuration file

`{config_path}` — edit this file to fine-tune settings.

### Useful commands

| Command | Description |
|---------|-------------|
| `distllm cluster status` | Cluster health and node list |
| `distllm benchmark run --model {model_id}` | Run a benchmark |
| `distllm doctor` | Full system diagnostics |
| `distllm dashboard` | Open the web dashboard |

### Documentation

- CLI reference: `distllm --help`
"""


# ── Main wizard entrypoint ────────────────────────────────────────────────

WELCOME_ART = r"""
[bold cyan]
   ____  _     _ _ _     __    __
  |  _ \(_)___| (_) |_   \ \  / /__ _ __ _   _
  | | | | / __| | | __|   \ \/ / _ \ '__| | | |
  | |_| | \__ \ | | |_     \  /  __/ |  | |_| |
  |____/|_|___/_|_|\__|     \/ \___|_|   \__, |
                                          |___/
[/bold cyan]
"""


def run_onboard(
    config_path: str | None = None,
    console: Console | None = None,
) -> None:
    """Run the interactive first-run wizard.

    Args:
        config_path: Path to write the generated config YAML.
                     Defaults to ``~/.distllm/config.yaml``.
        console: Rich Console to use.  A new one is created if omitted.
    """
    if console is None:
        console = Console()

    if config_path is None:
        config_path = os.path.join(os.path.expanduser("~"), ".distllm", "config.yaml")

    # ──────────────────────────────────────────────────────────────────────
    # Step 1: Welcome + hardware detection
    # ──────────────────────────────────────────────────────────────────────
    console.print()
    console.print(Panel(WELCOME_ART, subtitle="Distributed LLM Inference", style="cyan"))
    console.print()
    console.print(
        "Welcome to [bold]DistLLM[/bold] — pool GPUs across your machines "
        "to run models no single device could handle."
    )
    console.print("This wizard will detect your hardware and help you get started.")
    console.print()

    with console.status("[yellow]Detecting hardware...[/yellow]", spinner="dots"):
        hw = _detect_hardware()

    console.print(Panel(_build_hardware_table(hw), title="Step 1: Hardware Detection"))
    console.print()

    # Warn if no CUDA
    cuda_warning = not hw["cuda_available"]
    if cuda_warning:
        console.print(
            "[yellow]No CUDA-capable GPU detected. "
            "Inference will run on CPU which is significantly slower.[/yellow]"
        )

    console.input("[dim]Press Enter to continue...[/dim]")
    console.print()

    # ──────────────────────────────────────────────────────────────────────
    # Step 2: Model selection
    # ──────────────────────────────────────────────────────────────────────
    recommended = _recommend_models(hw)
    num_choices = len(recommended)

    # Prepend the hardware summary row for context
    mem_desc = (
        f"{hw['gpu_memory_gb']:.0f} GB VRAM"
        if hw["cuda_available"]
        else "CPU-only"
    )

    console.print(
        Panel(
            _build_model_table(recommended),
            title=f"Step 2: Model Selection — {mem_desc}",
        )
    )
    console.print()

    selected_model: dict[str, str] | None = None
    while selected_model is None:
        raw = Prompt.ask(
            "Choose a model",
            default="1",
            show_choices=False,
        )
        try:
            idx = int(raw)
            if 1 <= idx <= num_choices:
                selected_model = recommended[idx - 1]
            else:
                console.print(
                    f"[red]Please enter a number between 1 and {num_choices}.[/red]"
                )
        except ValueError:
            console.print("[red]Please enter a valid number.[/red]")

    console.print(f"[green]Selected:[/green] {selected_model['name']} ({selected_model['id']})")
    console.print()

    # ──────────────────────────────────────────────────────────────────────
    # Step 3: Download model
    # ──────────────────────────────────────────────────────────────────────
    console.print(Rule(style="dim"))
    console.print()
    console.print(
        Panel(
            f"[bold]Model:[/bold] {selected_model['id']}\n\n"
            "DistLLM will download the model from HuggingFace Hub.\n"
            "This may take several minutes depending on model size and connection speed.",
            title="Step 3: Download Model",
        )
    )
    console.print()

    download_now = Confirm.ask("Download this model now?", default=True)

    download_success = False
    if download_now:
        download_success = _download_model(selected_model["id"], console)
        if download_success:
            console.print("[green]Model downloaded successfully![/green]")
        else:
            console.print(
                "[yellow]Download did not complete. You can download manually "
                "later with the HuggingFace Hub CLI.[/yellow]"
            )
    else:
        console.print("[dim]Skipping download. You can download it later.[/dim]")
    console.print()

    # ──────────────────────────────────────────────────────────────────────
    # Step 4: Configuration generation
    # ──────────────────────────────────────────────────────────────────────
    console.print(Rule(style="dim"))
    console.print()

    default_cluster_name = f"distllm-{socket.gethostname()}"
    cluster_name = Prompt.ask(
        "Cluster name",
        default=default_cluster_name,
    )

    default_coordinator_port = 50050
    coordinator_port_str = Prompt.ask(
        "Coordinator gRPC port",
        default=str(default_coordinator_port),
    )
    try:
        coordinator_port = int(coordinator_port_str)
    except ValueError:
        coordinator_port = default_coordinator_port

    default_api_port = 8000
    api_port_str = Prompt.ask(
        "REST API port",
        default=str(default_api_port),
    )
    try:
        api_port = int(api_port_str)
    except ValueError:
        api_port = default_api_port

    # Data type selection
    if hw["cuda_available"]:
        dtype_opts = ["float16", "bfloat16", "float32"]
        dtype_default = "bfloat16" if hw["gpu_memory_gb"] >= 16 else "float16"
        dtype = Prompt.ask(
            "Data type",
            choices=dtype_opts,
            default=dtype_default,
        )
    else:
        dtype = "float32"

    config = _generate_config(
        model_id=selected_model["id"],
        cluster_name=cluster_name,
        coordinator_port=coordinator_port,
        api_port=api_port,
        dtype=dtype,
    )

    _save_config(config, config_path)
    console.print(f"[green]Configuration saved to[/green] [bold]{config_path}[/bold]")
    console.print()

    # ──────────────────────────────────────────────────────────────────────
    # Step 5: Next-steps card
    # ──────────────────────────────────────────────────────────────────────
    console.print(Rule(style="dim"))
    console.print()
    console.print(
        Panel(
            Markdown(
                _render_next_steps(
                    config_path=config_path,
                    model_id=selected_model["id"],
                    coordinator_port=coordinator_port,
                    api_port=api_port,
                    cluster_name=cluster_name,
                )
            ),
            title="Step 5: Next Steps",
            border_style="green",
        )
    )
    console.print()

    # ── Offer to start the cluster ──────────────────────────────────────────
    start_now = Confirm.ask(
        "Start the coordinator now?",
        default=True,
    )

    if start_now:
        console.print()
        console.print("[yellow]Starting coordinator...[/yellow]")
        try:
            from distllm.core.coordinator import Coordinator

            c = Coordinator(
                model_name=selected_model["id"],
                port=coordinator_port,
                dtype=dtype,
            )

            # Load model if download succeeded, otherwise try from cache
            if hw["cuda_available"]:
                try:
                    c.load_local_model()
                except Exception as exc:
                    console.print(
                        f"[yellow]Could not load model locally: {exc}\n"
                        "Will start coordinator and wait for remote workers.[/yellow]"
                    )

            c.start()
            console.print(
                f"[green]Coordinator started on port {coordinator_port}.\n"
                f"API available at http://localhost:{api_port}.[/green]"
            )
        except ImportError as exc:
            console.print(
                f"[red]Could not start coordinator: {exc}\n"
                f"Run manually: [bold]distllm cluster start --model "
                f'"{selected_model["id"]}" --port {coordinator_port}[/bold][/red]'
            )
        except Exception as exc:
            console.print(
                f"[red]Failed to start coordinator: {exc}\n"
                f"Run manually: [bold]distllm cluster start --model "
                f'"{selected_model["id"]}" --port {coordinator_port}[/bold][/red]'
            )

    # ── Guide for joining additional nodes ──────────────────────────────────
    console.print()
    console.print(Rule(style="dim"))
    console.print()

    try:
        ip = socket.gethostbyname(socket.gethostname())
    except OSError:
        ip = "127.0.0.1"

    join_cmd = f"distllm cluster join --coordinator {ip}:{coordinator_port}"

    console.print(
        Panel(
            f"[bold]Join Additional Nodes[/bold]\n\n"
            f"To add more GPU-equipped machines to your cluster, run this command "
            f"on each machine:\n\n"
            f"  [bold cyan]{join_cmd}[/bold cyan]\n\n"
            f"The coordinator will automatically assign model layers to each "
            f"worker based on available memory.\n\n"
            f"[dim]Tip:[/dim] If you are on the same LAN, try auto-discovery:\n"
            f"  [bold cyan]distllm cluster join --discover[/bold cyan]",
            title="Scaling Your Cluster",
            border_style="blue",
        )
    )
    console.print()

    # ── Farewell ────────────────────────────────────────────────────────────
    console.print()
    console.print(
        "[bold green]Setup complete.[/bold green] "
        "Your DistLLM cluster is ready. "
        "Run [bold]distllm --help[/bold] to explore all commands.\n"
    )


def main() -> None:
    """Entry point for ``distllm onboard``."""
    run_onboard()


if __name__ == "__main__":
    main()
