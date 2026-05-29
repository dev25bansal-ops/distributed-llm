"""Offline scheduler simulation — replay request traces without model inference.

Enables parameter tuning before deployment by simulating scheduling
decisions against historical request traces.

Usage::

    # Generate a trace file from production logs
    distllm scheduler trace --output trace.json

    # Simulate with different parameters
    distllm scheduler simulate trace.json --max-batch-size 64 --max-tokens 32768
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class TraceEntry:
    """A single request from a trace file."""
    request_id: str
    arrival_time: float  # Seconds from trace start
    prompt_tokens: int
    max_new_tokens: int
    priority: int = 2
    max_latency_ms: float | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "TraceEntry":
        return cls(
            request_id=d.get("request_id", ""),
            arrival_time=d.get("arrival_time", 0),
            prompt_tokens=d.get("prompt_tokens", 100),
            max_new_tokens=d.get("max_new_tokens", 128),
            priority=d.get("priority", 2),
            max_latency_ms=d.get("max_latency_ms"),
        )


@dataclass
class SimulationResult:
    """Results from an offline scheduler simulation."""
    total_requests: int = 0
    completed_requests: int = 0
    preempted_count: int = 0
    starvation_count: int = 0
    avg_wait_time_ms: float = 0.0
    p50_wait_time_ms: float = 0.0
    p99_wait_time_ms: float = 0.0
    max_wait_time_ms: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    total_iterations: int = 0
    total_prefill_tokens: int = 0
    total_decode_tokens: int = 0
    sla_violations: int = 0
    config_used: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "Simulation Results",
            "=" * 50,
            f"  Total requests:      {self.total_requests}",
            f"  Completed:           {self.completed_requests}",
            f"  Preempted:           {self.preempted_count}",
            f"  Starvation events:   {self.starvation_count}",
            f"  SLA violations:      {self.sla_violations}",
            f"  Avg wait time:       {self.avg_wait_time_ms:.1f}ms",
            f"  P50 wait time:       {self.p50_wait_time_ms:.1f}ms",
            f"  P99 wait time:       {self.p99_wait_time_ms:.1f}ms",
            f"  Max wait time:       {self.max_wait_time_ms:.1f}ms",
            f"  Throughput:          {self.throughput_tokens_per_sec:.0f} tokens/sec",
            f"  Iterations:          {self.total_iterations}",
            f"  Prefill tokens:      {self.total_prefill_tokens}",
            f"  Decode tokens:       {self.total_decode_tokens}",
        ]
        if self.config_used:
            lines.append(f"  Config:              {self.config_used}")
        return "\n".join(lines)


def load_trace(path: str) -> list[TraceEntry]:
    """Load a request trace from a JSON file.

    Expected format:
    {
        "requests": [
            {"request_id": "req-1", "arrival_time": 0.0, "prompt_tokens": 100, ...},
            ...
        ]
    }
    """
    with open(path) as f:
        data = json.load(f)

    entries = []
    for item in data.get("requests", data if isinstance(data, list) else []):
        entries.append(TraceEntry.from_dict(item))

    return sorted(entries, key=lambda e: e.arrival_time)


def simulate(
    trace: list[TraceEntry],
    max_batch_size: int = 32,
    max_tokens_per_batch: int = 32768,
    max_prefill_tokens: int = 4096,
    enable_chunked_prefill: bool = True,
    aging_enabled: bool = True,
    aging_interval_s: float = 30.0,
) -> SimulationResult:
    """Run an offline simulation of the batch scheduler.

    Simulates scheduling decisions without model inference.  Each
    request is treated as arriving at its ``arrival_time`` and
    completing after ``prompt_tokens / throughput + max_new_tokens / throughput``
    seconds (estimated).

    Args:
        trace: List of trace entries sorted by arrival time.
        max_batch_size: Max sequences per batch.
        max_tokens_per_batch: Max total tokens per batch.
        max_prefill_tokens: Max prefill tokens per iteration.
        enable_chunked_prefill: Enable chunked prefill.
        aging_enabled: Enable priority aging.
        aging_interval_s: Aging interval in seconds.

    Returns:
        SimulationResult with scheduling statistics.
    """
    from distllm.core.batch_scheduler import BatchScheduler, Sequence

    scheduler = BatchScheduler(
        max_batch_size=max_batch_size,
        max_tokens_per_batch=max_tokens_per_batch,
        enable_chunked_prefill=enable_chunked_prefill,
        max_prefill_tokens=max_prefill_tokens,
        aging_enabled=aging_enabled,
        aging_interval_s=aging_interval_s,
    )

    result = SimulationResult(
        total_requests=len(trace),
        config_used={
            "max_batch_size": max_batch_size,
            "max_tokens_per_batch": max_tokens_per_batch,
            "max_prefill_tokens": max_prefill_tokens,
            "enable_chunked_prefill": enable_chunked_prefill,
            "aging_enabled": aging_enabled,
        },
    )

    # Track per-request state
    request_arrival: dict[str, float] = {}
    request_start: dict[str, float] = {}  # When request first enters active
    request_complete: dict[str, float] = {}
    wait_times: list[float] = []

    sim_time = 0.0
    iteration = 0
    trace_idx = 0
    throughput_per_sec = 1000.0  # Assumed tokens/sec

    while trace_idx < len(trace) or scheduler.has_pending or scheduler.active_count > 0:
        # Advance time and add arriving requests
        if trace_idx < len(trace):
            next_arrival = trace[trace_idx].arrival_time
            sim_time = max(sim_time, next_arrival)

        while trace_idx < len(trace) and trace[trace_idx].arrival_time <= sim_time:
            entry = trace[trace_idx]
            seq = Sequence(
                request_id=entry.request_id,
                prompt_tokens=[1] * entry.prompt_tokens,
                max_new_tokens=entry.max_new_tokens,
                priority=entry.priority,
            )
            if entry.max_latency_ms:
                seq.max_latency_ms = entry.max_latency_ms
            scheduler.add(seq)
            request_arrival[entry.request_id] = entry.arrival_time
            trace_idx += 1

        # Schedule a batch
        batch = scheduler.schedule()
        if batch is None:
            # No work — advance time to next arrival
            if trace_idx < len(trace):
                sim_time = trace[trace_idx].arrival_time
                continue
            else:
                break

        iteration += 1

        # Process the batch (simulate forward pass)
        for seq in batch.sequences:
            if seq.request_id not in request_start:
                request_start[seq.request_id] = sim_time
                wait = (sim_time - request_arrival.get(seq.request_id, sim_time)) * 1000
                wait_times.append(wait)

        # Simulate token generation (advance sim_time)
        prefill_tokens = sum(
            len(s.prompt_tokens) for s in batch.sequences
            if s.status.value == "prefilling"
        )
        decode_tokens = sum(
            1 for s in batch.sequences
            if s.status.value == "decoding"
        )
        result.total_prefill_tokens += prefill_tokens
        result.total_decode_tokens += decode_tokens

        # Estimate iteration time
        iter_time = (prefill_tokens + decode_tokens) / throughput_per_sec
        sim_time += iter_time

        # Simulate completing some requests
        import torch
        next_tokens = torch.zeros(len(batch.sequences), dtype=torch.long)
        for i, seq in enumerate(batch.sequences):
            if len(seq.generated_tokens) >= seq.max_new_tokens - 1:
                next_tokens[i] = 0  # EOS
            else:
                next_tokens[i] = 42

        scheduler.step(batch, next_tokens)

        # Check for completed requests
        for seq in batch.sequences:
            if seq.is_complete and seq.request_id not in request_complete:
                request_complete[seq.request_id] = sim_time

        # Check for preemptions
        preempted = scheduler.preempt_lowest(min_priority=3)
        if preempted:
            result.preempted_count += 1

    # Compute results
    result.completed_requests = len(request_complete)
    result.total_iterations = iteration

    if wait_times:
        wait_times.sort()
        result.avg_wait_time_ms = sum(wait_times) / len(wait_times)
        result.p50_wait_time_ms = wait_times[len(wait_times) // 2]
        result.p99_wait_time_ms = wait_times[int(len(wait_times) * 0.99)]
        result.max_wait_time_ms = max(wait_times)

    total_tokens = result.total_prefill_tokens + result.total_decode_tokens
    if sim_time > 0:
        result.throughput_tokens_per_sec = total_tokens / sim_time

    # Count SLA violations
    for entry in trace:
        if entry.max_latency_ms:
            complete_time = request_complete.get(entry.request_id)
            arrival_time = request_arrival.get(entry.request_id, 0)
            if complete_time and (complete_time - arrival_time) * 1000 > entry.max_latency_ms:
                result.sla_violations += 1

    return result


def save_trace(entries: list[TraceEntry], output_path: str) -> None:
    """Save a trace to a JSON file."""
    data = {
        "requests": [
            {
                "request_id": e.request_id,
                "arrival_time": e.arrival_time,
                "prompt_tokens": e.prompt_tokens,
                "max_new_tokens": e.max_new_tokens,
                "priority": e.priority,
                "max_latency_ms": e.max_latency_ms,
            }
            for e in entries
        ]
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
