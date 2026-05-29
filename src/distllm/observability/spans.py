"""OpenTelemetry span helpers for LLM generation phases.

Provides context managers for prefill, decode, and full generation spans,
plus helpers for recording TTFT (time-to-first-token) events.
"""

from contextlib import contextmanager, asynccontextmanager

from opentelemetry import trace

_tracer = trace.get_tracer("distllm.generation")


@contextmanager
def span_prefill(request_id: str, prompt_length: int, model: str = "distributed-llm"):
    """Span around the prefill phase (first forward pass)."""
    with _tracer.start_as_current_span(
        "llm.prefill",
        attributes={
            "request_id": request_id,
            "prompt_length": prompt_length,
            "model": model,
            "llm.phase": "prefill",
        },
    ) as span:
        yield span


@contextmanager
def span_decode_step(request_id: str, step: int, model: str = "distributed-llm"):
    """Span around a single decode step (sampled, not every step)."""
    with _tracer.start_as_current_span(
        "llm.decode_step",
        attributes={
            "request_id": request_id,
            "decode_step": step,
            "model": model,
            "llm.phase": "decode",
        },
    ) as span:
        yield span


def record_ttft(span: trace.Span, ttft_seconds: float):
    """Record time-to-first-token as a span event."""
    span.add_event("time_to_first_token", {"ttft_seconds": ttft_seconds})


def record_generation_span(
    request_id: str,
    model: str,
    prompt_len: int,
    completion_tokens: int,
    duration: float,
    ttft: float,
    status: str = "ok",
    tenant: str = "default",
):
    """Create a top-level generation span covering the full request lifecycle."""
    with _tracer.start_as_current_span(
        "llm.generate",
        attributes={
            "request_id": request_id,
            "model": model,
            "prompt_length": prompt_len,
            "completion_tokens": completion_tokens,
            "generation.duration_s": duration,
            "generation.ttft_s": ttft,
            "generation.status": status,
            "tenant": tenant,
        },
    ) as span:
        if status != "ok":
            span.set_status(trace.StatusCode.ERROR, status)
        span.add_event("generation_complete", {
            "duration": duration,
            "ttft": ttft,
            "tokens": completion_tokens,
        })


@asynccontextmanager
async def async_span_generation(
    request_id: str,
    model: str,
    prompt_len: int,
    tenant: str = "default",
):
    """Async context manager for a full generation span.

    Used in streaming responses where the span must stay open across
    the entire AsyncGenerator lifecycle.
    """
    with _tracer.start_as_current_span(
        "llm.generate",
        attributes={
            "request_id": request_id,
            "model": model,
            "prompt_length": prompt_len,
            "tenant": tenant,
            "llm.streaming": True,
        },
    ) as span:
        yield span


# ── Pipeline-specific spans ──────────────────────────────────────────

@contextmanager
def span_node_forward(node_id: str, request_id: str, layer_start: int, layer_end: int):
    """Span around a single node's forward pass in the pipeline."""
    with _tracer.start_as_current_span(
        "pipeline.node_forward",
        attributes={
            "node_id": node_id,
            "request_id": request_id,
            "layer_start": layer_start,
            "layer_end": layer_end,
        },
    ) as span:
        yield span


@contextmanager
def span_kv_transfer(request_id: str, source_node: str, target_node: str, bytes: int):
    """Span around KV cache transfer between nodes."""
    with _tracer.start_as_current_span(
        "pipeline.kv_transfer",
        attributes={
            "request_id": request_id,
            "source_node": source_node,
            "target_node": target_node,
            "transfer_bytes": bytes,
        },
    ) as span:
        yield span


@contextmanager
def span_pipeline_execution(request_id: str, num_nodes: int, strategy: str):
    """Span around the full pipeline execution."""
    with _tracer.start_as_current_span(
        "pipeline.execute",
        attributes={
            "request_id": request_id,
            "num_nodes": num_nodes,
            "strategy": strategy,
        },
    ) as span:
        yield span


@contextmanager
def span_speculative_decoding(request_id: str, num_candidates: int):
    """Span around speculative decoding verification."""
    with _tracer.start_as_current_span(
        "pipeline.speculative_decode",
        attributes={
            "request_id": request_id,
            "num_candidates": num_candidates,
        },
    ) as span:
        yield span


@contextmanager
def span_prefix_cache_lookup(request_id: str, prefix_length: int):
    """Span around prefix cache lookup."""
    with _tracer.start_as_current_span(
        "pipeline.prefix_cache_lookup",
        attributes={
            "request_id": request_id,
            "prefix_length": prefix_length,
        },
    ) as span:
        yield span
