"""Automatic threshold calibration — measure hardware and auto-configure scheduler.

Runs a short calibration phase at startup to measure:
- KV cache bytes per token (model architecture dependent)
- GPU-CPU transfer bandwidth
- Available GPU memory for KV cache
- Available CPU memory for swap/checkpoint

Then auto-sets: max_preempted, memory limits, swap thresholds, chunk sizes.

Usage::

    calibrator = ThresholdCalibrator(model_info={...})
    calibrator.calibrate()
    calibrator.apply_to_scheduler(scheduler)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass
class CalibrationResult:
    """Results from hardware calibration."""
    kv_bytes_per_token: int = 0
    gpu_memory_total_mb: int = 0
    gpu_memory_free_mb: int = 0
    cpu_memory_total_mb: int = 0
    cpu_memory_available_mb: int = 0
    gpu_cpu_bandwidth_mbps: float = 0.0
    recommended_max_preempted: int = 4
    recommended_max_batch_size: int = 32
    recommended_max_tokens_per_batch: int = 32768
    recommended_max_prefill_tokens: int = 4096
    recommended_chunk_size: int = 512
    recommended_checkpoint_memory_mb: int = 4096
    calibration_time_ms: float = 0.0


def _get_gpu_memory() -> tuple[int, int]:
    """Get GPU memory total and free in MB. Returns (0, 0) if no GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            total = torch.cuda.get_device_properties(device).total_memory // (1024**2)
            free = torch.cuda.mem_get_info(device)[0] // (1024**2)
            return total, free
    except Exception:
        pass
    return 0, 0


def _get_cpu_memory() -> tuple[int, int]:
    """Get CPU memory total and available in MB."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return mem.total // (1024**2), mem.available // (1024**2)
    except ImportError:
        pass
    return 0, 0


def _measure_kv_bytes_per_token(model_info: dict) -> int:
    """Estimate KV cache bytes per token from model architecture.

    Formula: 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
    """
    hidden_size = model_info.get("hidden_size", 4096)
    num_layers = model_info.get("num_layers", 32)
    num_heads = model_info.get("num_attention_heads", 32)
    num_kv_heads = model_info.get("num_key_value_heads", num_heads)
    head_dim = hidden_size // num_heads if num_heads > 0 else 128
    dtype_bytes = 2  # fp16

    kv_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
    return kv_per_token


def _measure_gpu_cpu_bandwidth(size_mb: int = 64) -> float:
    """Measure GPU-CPU transfer bandwidth in MB/s.

    Transfers a tensor of `size_mb` from GPU to CPU and measures throughput.
    Returns 0 if no GPU available.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0

        device = torch.cuda.current_device()
        size_bytes = size_mb * 1024 * 1024
        num_elements = size_bytes // 2  # fp16 = 2 bytes per element

        # Create tensor on GPU
        gpu_tensor = torch.randn(num_elements, device=device, dtype=torch.float16)

        # Warmup
        gpu_tensor.cpu()
        torch.cuda.synchronize()

        # Measure
        start = time.monotonic()
        for _ in range(10):
            cpu_tensor = gpu_tensor.cpu()
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start

        bandwidth = (size_mb * 10) / elapsed
        return bandwidth
    except Exception:
        return 0.0


def calibrate(model_info: dict | None = None, device: int = 0) -> CalibrationResult:
    """Run hardware calibration and compute recommended thresholds.

    Args:
        model_info: Model architecture dict with hidden_size, num_layers, etc.
        device: GPU device index.

    Returns:
        CalibrationResult with all recommended settings.
    """
    start = time.monotonic()
    result = CalibrationResult()

    info = model_info or {}

    # KV bytes per token
    result.kv_bytes_per_token = _measure_kv_bytes_per_token(info)
    logger.info(f"Calibration: KV bytes/token = {result.kv_bytes_per_token}")

    # GPU memory
    result.gpu_memory_total_mb, result.gpu_memory_free_mb = _get_gpu_memory()
    logger.info(f"Calibration: GPU memory = {result.gpu_memory_free_mb}MB free / {result.gpu_memory_total_mb}MB total")

    # CPU memory
    result.cpu_memory_total_mb, result.cpu_memory_available_mb = _get_cpu_memory()
    logger.info(f"Calibration: CPU memory = {result.cpu_memory_available_mb}MB available / {result.cpu_memory_total_mb}MB total")

    # GPU-CPU bandwidth
    result.gpu_cpu_bandwidth_mbps = _measure_gpu_cpu_bandwidth()
    logger.info(f"Calibration: GPU→CPU bandwidth = {result.gpu_cpu_bandwidth_mbps:.0f} MB/s")

    # Compute recommendations
    if result.kv_bytes_per_token > 0 and result.gpu_memory_free_mb > 0:
        # Reserve 20% for model weights, 10% overhead
        kv_budget_mb = int(result.gpu_memory_free_mb * 0.7)
        kv_budget_bytes = kv_budget_mb * 1024 * 1024
        max_tokens = kv_budget_bytes // result.kv_bytes_per_token

        result.recommended_max_tokens_per_batch = min(max_tokens, 131072)
        result.recommended_max_tokens_per_batch = max(result.recommended_max_tokens_per_batch, 4096)

        # Batch size: use 50% of max tokens for safety
        result.recommended_max_batch_size = min(
            128,
            max(4, result.recommended_max_tokens_per_batch // 256),
        )

        # Prefill tokens: 1/4 of total budget
        result.recommended_max_prefill_tokens = min(
            result.recommended_max_tokens_per_batch // 4,
            16384,
        )

        # Chunk size: balance between latency and overhead
        result.recommended_chunk_size = min(
            result.recommended_max_prefill_tokens,
            1024,
        )

    # Preemption: how many sequences can we checkpoint
    if result.cpu_memory_available_mb > 0:
        # Each checkpoint ≈ KV bytes per token * avg_seq_len
        avg_seq_len = 512  # Assume average
        checkpoint_size_mb = (result.kv_bytes_per_token * avg_seq_len) // (1024 * 1024)
        if checkpoint_size_mb > 0:
            result.recommended_max_preempted = min(
                16,
                max(1, (result.cpu_memory_available_mb // 4) // checkpoint_size_mb),
            )
            result.recommended_checkpoint_memory_mb = result.cpu_memory_available_mb // 4

    result.calibration_time_ms = (time.monotonic() - start) * 1000
    logger.info(
        f"Calibration complete in {result.calibration_time_ms:.0f}ms: "
        f"batch={result.recommended_max_batch_size}, "
        f"tokens={result.recommended_max_tokens_per_batch}, "
        f"prefill={result.recommended_max_prefill_tokens}, "
        f"preempted={result.recommended_max_preempted}"
    )

    return result


def apply_to_scheduler(scheduler: Any, result: CalibrationResult) -> None:
    """Apply calibration results to a BatchScheduler.

    Updates max_batch_size, max_tokens_per_batch, max_prefill_tokens,
    and max_preempted.
    """
    scheduler.max_batch_size = result.recommended_max_batch_size
    scheduler.max_tokens_per_batch = result.recommended_max_tokens_per_batch
    scheduler._budget.max_batch_size = result.recommended_max_batch_size
    scheduler._budget.max_total_tokens = result.recommended_max_tokens_per_batch
    scheduler._budget.max_prefill_tokens = result.recommended_max_prefill_tokens
    scheduler._max_preempted = result.recommended_max_preempted

    logger.info(
        f"Applied calibration to scheduler: "
        f"batch_size={scheduler.max_batch_size}, "
        f"tokens_per_batch={scheduler.max_tokens_per_batch}, "
        f"prefill_tokens={scheduler._budget.max_prefill_tokens}, "
        f"max_preempted={scheduler._max_preempted}"
    )
