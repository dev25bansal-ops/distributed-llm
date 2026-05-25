"""CLI compression command for distllm compress."""

import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

CompressionConfig = CompressionMethod = None
CompressionPipeline = StrategySelector = HardwareProfiler = HardwareClass = None
from distllm.security import hf_revision


def run_compress(
    model_name: str,
    target: str,
    output_dir: str,
    tokenizer_name: str | None,
    prune_ratio: float,
    calibration_samples: int,
    method: str,
    local: bool,
    console: Console,
):
    """Run model compression and save compressed model."""
    console.print(f"[bold]Compressing model:[/bold] {model_name}")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        console.print("[red]Error: transformers library required. Install with: pip install transformers[/red]")
        raise SystemExit(1) from None

    # Map target to bits
    target_map = {
        "int8": 8,
        "int4": 4,
        "int4-awq": 4,
        "int4-gptq": 4,
        "fp16": 16,
        "fp32": 32,
    }

    if target not in target_map:
        console.print(f"[red]Unsupported target: {target}. Choose from: {', '.join(target_map.keys())}[/red]")
        raise SystemExit(1)

    bits = target_map[target]
    use_awq = "awq" in target or method == "awq"
    use_gptq = "gptq" in target or method == "gptq"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    console.print("[yellow]Hardware profiling not available (legacy module removed)[/yellow]")

    # Load model
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Loading model...", total=None)
        dtype = torch.float16 if bits <= 16 else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            revision=hf_revision(),
        )
        model.eval()
        progress.update(task, completed=True)

    # Load tokenizer
    tokenizer_name = tokenizer_name or model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True,
        revision=hf_revision(),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    console.print("[yellow]Compression pipeline not available (legacy module removed). Saving model as-is.[/yellow]")

    # Save model
    console.print(f"[cyan]Saving to:[/cyan] {output_path}")

    if hasattr(model, "save_pretrained"):
        model.save_pretrained(str(output_path))
        tokenizer.save_pretrained(str(output_path))
    else:
        torch.save(model.state_dict(), str(output_path / "model.pt"))

    console.print(f"\n[bold green]Done![/bold green] Model saved to: {output_path}")
    console.print("  Use with: distllm run --model {output_path} --local")
