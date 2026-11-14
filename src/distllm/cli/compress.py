"""CLI model conversion command for ``distllm model compress``.

This build does NOT ship the quantization pipeline: the legacy compression
modules (AWQ/GPTQ/INT8/INT4 quantizers, pruning) were removed. Only dtype
casting (fp16/fp32) is implemented here. Quantized targets fail loudly up
front instead of silently saving an unquantized copy under a path that
claims the model was compressed.
"""

from __future__ import annotations

from pathlib import Path

from distllm._optional_deps import optional_import

torch = optional_import("torch")
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from transformers import AutoModelForCausalLM, AutoTokenizer

from distllm.security import hf_revision

# Targets that require the removed quantization pipeline. Casting a model to
# one of these dtypes is NOT quantization: int8/int4 need weight-only
# quantization and calibration; AWQ/GPTQ need their own quantizers.
_QUANTIZED_TARGETS = ("int8", "int4", "int4-awq", "int4-gptq")
_QUANTIZATION_METHODS = ("awq", "gptq")

_TARGET_MAP = {
    "int8": 8,
    "int4": 4,
    "int4-awq": 4,
    "int4-gptq": 4,
    "fp16": 16,
    "fp32": 32,
}


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
    trust_remote_code: bool = False,
):
    """Convert a model and save it.

    ``fp16``/``fp32`` perform a dtype cast. Quantized targets
    (``int8``/``int4``/``int4-awq``/``int4-gptq``) and AWQ/GPTQ quantization
    methods are rejected before any download or disk write: the compression
    pipeline does not exist in this build.
    """
    # Map target to bits
    if target not in _TARGET_MAP:
        console.print(f"[red]Unsupported target: {target}. Choose from: {', '.join(_TARGET_MAP.keys())}[/red]")
        raise SystemExit(1)

    method_norm = (method or "none").strip().lower()
    if target in _QUANTIZED_TARGETS or method_norm in _QUANTIZATION_METHODS:
        console.print(
            "[red]Compression is not available in this build: the quantization "
            "pipeline (int8, int4, AWQ, GPTQ) was removed.[/red]"
        )
        console.print(
            "[red]Nothing was downloaded or saved. Only dtype casting is "
            "supported: use --target fp16 or --target fp32.[/red]"
        )
        raise SystemExit(1)

    bits = _TARGET_MAP[target]
    cast_label = "fp16" if bits == 16 else "fp32"

    console.print(f"[bold]Casting model to {cast_label}:[/bold] {model_name}")
    if prune_ratio:
        console.print("[yellow]Pruning is not available in this build; --prune is ignored.[/yellow]")
    # calibration_samples only matters for post-training quantization, which
    # this build does not implement; casting needs no data.
    _ = calibration_samples

    output_path = Path(output_dir)

    # Load model
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Loading model...", total=None)
        dtype = torch.float16 if bits == 16 else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=trust_remote_code,
            revision=hf_revision(),
        )
        model.eval()
        progress.update(task, completed=True)

    # Load tokenizer
    tokenizer_name = tokenizer_name or model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=trust_remote_code,
        revision=hf_revision(),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Save model
    output_path.mkdir(parents=True, exist_ok=True)
    console.print(f"[cyan]Saving to:[/cyan] {output_path}")

    if hasattr(model, "save_pretrained"):
        model.save_pretrained(str(output_path))
        tokenizer.save_pretrained(str(output_path))
    else:
        torch.save(model.state_dict(), str(output_path / "model.pt"))

    console.print(f"\n[bold green]Done![/bold green] Model cast to {cast_label} and saved to: {output_path}")
    console.print(f"  Use with: distllm run --model {output_path} --local")
