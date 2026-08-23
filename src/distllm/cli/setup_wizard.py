"""Interactive TUI configuration wizard for DistLLM.

Provides an interactive setup experience with per-field validation,
live summary panel, and final diff view.
"""

from __future__ import annotations

import os
import difflib
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
from loguru import logger


def run_setup_wizard(config_path: str = "") -> dict[str, Any]:
    """Run the interactive setup wizard.

    Returns:
        The generated configuration dict.
    """
    console = Console()
    config: dict[str, Any] = {}

    console.print(Panel.fit("[bold cyan]DistLLM Setup Wizard[/]", border_style="cyan"))
    console.print("This wizard will help you configure your DistLLM cluster.\n")

    # Section 1: Cluster Configuration
    console.print(Panel("[bold]Cluster Configuration[/]", border_style="blue"))
    name = Prompt.ask("Cluster name", default="my-cluster")
    port = Prompt.ask("gRPC port", default="50050")
    api_port = Prompt.ask("API port", default="8000")

    config["cluster"] = {"name": name, "port": int(port), "api_port": int(api_port)}

    # Section 2: Model Configuration
    console.print(Panel("[bold]Model Configuration[/]", border_style="blue"))
    model_name = Prompt.ask(
        "Model name (HuggingFace ID or local path)",
        default="HuggingFaceTB/SmolLM-135M",
    )
    dtype = Prompt.ask("Data type", default="float16", choices=["float16", "float32", "bfloat16"])
    quant = Prompt.ask("Quantization", default="none", choices=["none", "int4", "int8", "fp8"])

    config["model"] = {"name": model_name, "dtype": dtype}
    if quant != "none":
        config["model"]["quantization"] = quant

    # Section 3: Hardware Configuration
    console.print(Panel("[bold]Hardware Configuration[/]", border_style="blue"))
    try:
        import torch
        gpu_count = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
    except Exception:
        gpu_count = 0
        gpu_names = []

    if gpu_count > 0:
        console.print(f"  Detected {gpu_count} GPU(s):")
        for name in gpu_names:
            console.print(f"    - {name}")
        use_gpu = Confirm.ask("Use GPUs?", default=True)
        config["hardware"] = {"gpu": use_gpu, "device": "cuda" if use_gpu else "cpu"}
    else:
        console.print("  [yellow]No GPU detected — using CPU[/]")
        config["hardware"] = {"gpu": False, "device": "cpu"}

    # Summary
    console.print()
    console.print(Panel("[bold green]Configuration Summary[/]", border_style="green"))
    table = Table(show_header=False)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="white")
    for section, values in config.items():
        for k, v in values.items():
            table.add_row(f"{section}.{k}", str(v))
    console.print(table)

    if not Confirm.ask("\nSave this configuration?", default=True):
        console.print("[yellow]Configuration discarded[/]")
        return {}

    # Save
    save_path = config_path or os.path.expanduser("~/.distllm/config.yaml")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    import yaml
    with open(save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    console.print(f"[green]Configuration saved to {save_path}[/]")
    return config


def show_config(config_path: str = "") -> None:
    """Display the effective configuration as a Rich Tree."""
    console = Console()
    import yaml

    path = config_path or os.path.expanduser("~/.distllm/config.yaml")
    if not os.path.exists(path):
        console.print("[red]No configuration file found[/]")
        return

    with open(path) as f:
        config = yaml.safe_load(f) or {}

    tree = Tree(f"[bold cyan]Configuration: {path}[/]")
    for section, values in config.items():
        section_node = tree.add(f"[bold]{section}[/]")
        if isinstance(values, dict):
            for k, v in values.items():
                section_node.add(f"{k}: [green]{v}[/]")
        else:
            section_node.add(str(values))

    console.print(tree)


def diff_config(config_path: str = "") -> None:
    """Show differences from default configuration."""
    console = Console()
    import yaml

    path = config_path or os.path.expanduser("~/.distllm/config.yaml")
    if not os.path.exists(path):
        console.print("[red]No configuration file found[/]")
        return

    with open(path) as f:
        config_str = f.read()

    defaults = yaml.dump({
        "cluster": {"name": "my-cluster", "port": 50050, "api_port": 8000},
        "hardware": {"gpu": False, "device": "cpu"},
    })

    diff = difflib.unified_diff(
        defaults.splitlines(keepends=True),
        config_str.splitlines(keepends=True),
        fromfile="defaults",
        tofile="current",
    )
    diff_text = "".join(diff)
    if diff_text.strip():
        console.print(Syntax(diff_text, "diff", theme="ansi_dark"))
    else:
        console.print("[green]Configuration matches defaults[/]")


def repair_config(config_path: str = "") -> bool:
    """Validate and fix common configuration issues."""
    console = Console()
    import yaml

    path = config_path or os.path.expanduser("~/.distllm/config.yaml")
    if not os.path.exists(path):
        console.print("[red]No configuration file found[/]")
        return False

    with open(path) as f:
        try:
            config = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            console.print(f"[red]Invalid YAML: {e}[/]")
            return False

    fixed = False
    # Fix port range
    port = config.get("cluster", {}).get("port", 0)
    if port and (port < 1024 or port > 65535):
        config["cluster"]["port"] = 50050
        console.print("[yellow]Fixed: port out of range, reset to 50050[/]")
        fixed = True

    with open(path, "w") as f:
        yaml.dump(config, f)

    if fixed:
        console.print("[green]Configuration repaired[/]")
    else:
        console.print("[green]Configuration looks good[/]")
    return True
