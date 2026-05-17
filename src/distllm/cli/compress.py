"""CLI compression command for distllm compress."""

import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from distllm.core.compression_config import CompressionConfig, CompressionMethod
from distllm.core.compression_pipeline import (
    CompressionPipeline,
    StrategySelector,
    HardwareProfiler,
    HardwareClass,
)


def run_compress(
    model_name: str,
    target: str,
    output_dir: str,
    tokenizer_name: Optional[str],
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
        raise SystemExit(1)

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

    # Profile hardware
    hw = HardwareProfiler()
    hw_class = hw.classify()
    console.print(f"[cyan]Hardware:[/cyan] {hw_class.value.capitalize()}")
    if torch.cuda.is_available():
        mem = hw.get_vram_gb()
        console.print(f"[cyan]VRAM:[/cyan] {mem:.1f} GB")

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
        )
        model.eval()
        progress.update(task, completed=True)

    # Load tokenizer
    tokenizer_name = tokenizer_name or model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    console.print(f"[cyan]Parameters:[/cyan] {total_params:.2f}B")

    # Build compression config
    base_params = {}
    if bits == 4:
        base_params["method"] = "awq" if use_awq else "gptq"

    config = CompressionConfig(
        method=CompressionMethod.PTQ_INT4 if bits == 4 else CompressionMethod.PTQ_INT8,
        enabled=True,
        target_bits=bits,
        pruning_ratio=prune_ratio,
        calibration_samples=calibration_samples,
    )

    pipeline = CompressionPipeline(config)

    # Plan
    plan = pipeline.plan()
    console.print(f"[cyan]Compression plan:[/cyan] {plan.summary()}")

    # Apply compression
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Applying compression...", total=None)

        if prune_ratio > 0:
            console.print(f"  Pruning with ratio={prune_ratio}")
            model = pipeline.apply_pruning(model)

        if bits <= 8:
            if bits == 4:
                qmethod = "gptq" if use_gptq else "awq"
                console.print(f"  Quantizing to INT4 using {qmethod.upper()}")
            else:
                qmethod = "awq"
                console.print("  Quantizing to INT8")
            model = pipeline.apply_quantization(model, bits=bits, tokenizer=tokenizer, quant_method=qmethod)

        progress.update(task, completed=True)

    # Save model
    console.print(f"[cyan]Saving to:[/cyan] {output_path}")

    if hasattr(model, "save_pretrained"):
        model.save_pretrained(str(output_path))
        tokenizer.save_pretrained(str(output_path))
    else:
        torch.save(model.state_dict(), str(output_path / "model.pt"))

    # Save compression info
    info = {
        "model": model_name,
        "target": target,
        "bits": bits,
        "prune_ratio": prune_ratio,
        "params_before": total_params,
        "compression_ratio": plan.total_compression_ratio,
        "method": method,
        "hardware": hw_class.value,
    }

    # Verify compression
    compressed_size = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())
    info["compressed_size_mb"] = round(compressed_size / (1024 * 1024), 2)

    with open(str(output_path / "compression_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    # Summary table
    table = Table(title="Compression Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    est_fp16_size = total_params * 2 * 1024  # MB
    table.add_row("Original (FP16)", f"{est_fp16_size:.0f} MB")
    table.add_row("Compressed size", f"{info['compressed_size_mb']:.0f} MB")
    table.add_row("Compression ratio", f"{est_fp16_size / max(info['compressed_size_mb'], 1):.1f}x")
    table.add_row("Estimated speedup", f"{plan.expected_speedup:.1f}x")
    table.add_row("Estimated quality loss", f"{plan.expected_quality_loss:.4f}")

    console.print(table)
    console.print(f"\n[bold green]Done![/bold green] Compressed model saved to: {output_path}")
    console.print("  Use with: distllm run --model {output_path} --local")
