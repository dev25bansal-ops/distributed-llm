"""Observability helpers for the DistLLM SDK client."""

from distllm.sdk.types import CallStats, ClientStats, UsageInfo


def _parse_usage(data: dict) -> UsageInfo | None:
    """Parse usage dict from API response into UsageInfo."""
    raw = data.get("usage")
    if not raw:
        return None
    gen_time = data.get("generation_time")
    completion_tokens = raw.get("completion_tokens", raw.get("total_tokens", 0))
    tps = (completion_tokens / gen_time) if gen_time and gen_time > 0 else 0.0
    return UsageInfo(
        prompt_tokens=raw.get("prompt_tokens", 0),
        completion_tokens=completion_tokens,
        total_tokens=raw.get("total_tokens", 0),
        tokens_per_second=tps,
    )


def record_call(
    stats: ClientStats,
    endpoint: str,
    latency: float,
    usage: UsageInfo | None,
    max_call_log_size: int,
) -> None:
    """Record a completed API call in *stats* (shared by both sync and async clients)."""
    stats.total_calls += 1
    stats.total_latency += latency
    if usage:
        stats.total_prompt_tokens += usage.prompt_tokens
        stats.total_completion_tokens += usage.completion_tokens
    stats.call_log.append(CallStats(
        endpoint=endpoint, latency=latency,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        status_code=200,
    ))
    if len(stats.call_log) > max_call_log_size:
        del stats.call_log[: len(stats.call_log) - max_call_log_size]
