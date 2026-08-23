"""Stats collection for the batch scheduler.

Extracted from ``BatchScheduler.stats()`` in ``batch_scheduler.py``.
"""

from __future__ import annotations

from typing import Any

from loguru import logger


def collect_stats(
    active_count: int,
    pending_count: int,
    preempted_count: int,
    max_batch_size: int,
    max_tokens_per_batch: int,
    paged_attention_mgr: Any,
    iteration_count: int,
    total_prefill_tokens: int,
    total_decode_tokens: int,
    chunked_prefill: dict,
    enable_chunked_prefill: bool,
    adaptive_engine: Any,
    model_name: str | None,
    het_budget: Any,
    cost_adjuster: Any,
    wan_policy: Any,
    energy_scheduler: Any,
) -> dict:
    """Collect batch scheduler statistics into a flat dict."""
    stats: dict = {
        "active_requests": active_count,
        "pending_requests": pending_count,
        "preempted_requests": preempted_count,
        "max_batch_size": max_batch_size,
        "max_tokens_per_batch": max_tokens_per_batch,
        "paged_attention": paged_attention_mgr is not None,
        "iteration": iteration_count,
        "total_prefill_tokens": total_prefill_tokens,
        "total_decode_tokens": total_decode_tokens,
        "chunked_prefill_active": len(chunked_prefill),
        "chunked_prefill_enabled": enable_chunked_prefill,
        "adaptive_batching": adaptive_engine is not None,
    }
    if adaptive_engine is not None:
        try:
            astats = adaptive_engine.get_stats(model_name or "default")
            stats["adaptive_avg_latency_ms"] = astats.avg_latency_ms
            stats["adaptive_batch_size"] = adaptive_engine.get_current_batch_size(model_name or "default")
        except Exception as e:
            logger.debug("Adaptive engine get_stats failed: {}", e)

    if het_budget is not None:
        stats["heterogeneous"] = het_budget.stats()
    if cost_adjuster is not None:
        stats["cost_aware"] = cost_adjuster.stats()
    if wan_policy is not None:
        stats["wan"] = wan_policy.stats()
    if energy_scheduler is not None:
        stats["energy"] = energy_scheduler.stats()

    return stats
