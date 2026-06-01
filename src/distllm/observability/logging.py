"""Structured logging configuration for distributed-llm.

Configures loguru with JSON output, structured context fields,
OpenTelemetry trace/span injection, and optional Loki push
for centralized log aggregation.
"""

import sys
import json
from loguru import logger


def _get_otel_context() -> dict:
    """Extract current OpenTelemetry trace context into a dict."""
    try:
        from opentelemetry import trace
        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx and ctx.is_valid:
            return {
                "trace_id": f"{ctx.trace_id:032x}",
                "span_id": f"{ctx.span_id:016x}",
            }
    except Exception:
        logger.debug("Span context extraction failed")
    return {}


def setup_logging(
    level: str = "INFO",
    json_format: bool = True,
    loki_url: str | None = None,
    service_name: str = "distllm",
) -> None:
    """Configure loguru with structured JSON logging and optional Loki sink.

    Args:
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR).
        json_format: If True, output JSON lines; else human-readable format.
        loki_url: Optional Loki push URL (e.g. "http://localhost:3100").
        service_name: Service identifier used as a log label.
    """
    logger.remove()

    if json_format:
        def _json_sink(message):
            """Custom sink that outputs structured JSON logs."""
            record = message.record
            entry = {
                "timestamp": record["time"].strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "level": record["level"].name,
                "service": service_name,
                "module": record["name"],
                "function": record["function"],
                "line": record["line"],
                "message": record["message"],
                **_get_otel_context(),
                "extra": {
                    k: v for k, v in record["extra"].items()
                    if k not in ("elaborated",)
                },
            }
            sys.stdout.write(json.dumps(entry, default=str) + "\n")
            sys.stdout.flush()

        logger.add(
            _json_sink,
            level=level,
            enqueue=True,
        )
    else:
        logger.add(
            sys.stdout,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            ),
            level=level,
            enqueue=True,
        )

    if loki_url:
        from distllm.observability.loki_sink import loki_sink

        logger.add(
            loki_sink(loki_url, service_name),
            level=level,
            format="{message}",
            enqueue=True,
        )


# ── Structured Log Event Schema Registry ────────────────────────────────

LOG_EVENT_SCHEMAS: dict[str, dict[str, str]] = {
    "request_started": {
        "request_id": "string",
        "method": "string",
        "path": "string",
        "tenant": "string",
        "model": "string",
        "client_ip": "string",
        "priority": "integer",
    },
    "request_completed": {
        "request_id": "string",
        "status_code": "integer",
        "duration_ms": "number",
        "prompt_tokens": "integer",
        "completion_tokens": "integer",
        "model": "string",
        "tenant": "string",
    },
    "request_failed": {
        "request_id": "string",
        "error_code": "string",
        "error_message": "string",
        "duration_ms": "number",
        "model": "string",
    },
    "node_health_changed": {
        "node_id": "string",
        "old_state": "string",
        "new_state": "string",
        "latency_ms": "number",
        "consecutive_failures": "integer",
    },
    "model_loaded": {
        "model_name": "string",
        "num_layers": "integer",
        "dtype": "string",
        "device": "string",
        "duration_ms": "number",
    },
    "cache_hit": {
        "cache_type": "string",
        "prefix_hash": "string",
        "prefix_length": "integer",
        "similarity": "number",
    },
    "cache_miss": {
        "cache_type": "string",
        "prefix_hash": "string",
    },
    "speculative_decoding": {
        "request_id": "string",
        "draft_calls": "integer",
        "target_calls": "integer",
        "accepted": "integer",
        "proposed": "integer",
        "acceptance_rate": "number",
    },
    "federated_round_completed": {
        "round_number": "integer",
        "participating_nodes": "integer",
        "avg_loss": "number",
        "duration_s": "number",
    },
    "node_recovery": {
        "failed_node_id": "string",
        "sequences_recovered": "integer",
        "sequences_lost": "integer",
        "redistributions": "integer",
        "duration_ms": "number",
    },
    "cost_alert": {
        "tenant": "string",
        "current_cost": "number",
        "budget": "number",
        "utilization_pct": "number",
    },
    "security_event": {
        "event_type": "string",
        "client_ip": "string",
        "api_key_id": "string",
        "path": "string",
        "reason": "string",
    },
}


def get_log_schema(event_type: str) -> dict[str, str] | None:
    """Return the schema for a log event type, or None if not registered."""
    return LOG_EVENT_SCHEMAS.get(event_type)


def validate_log_event(event_type: str, fields: dict) -> list[str]:
    """Validate a log event against its registered schema.

    Returns a list of validation errors (empty if valid).
    """
    schema = LOG_EVENT_SCHEMAS.get(event_type)
    if schema is None:
        return [f"Unknown event type: {event_type}"]

    errors = []
    for field_name, expected_type in schema.items():
        if field_name not in fields:
            errors.append(f"Missing required field: {field_name}")
        else:
            value = fields[field_name]
            if expected_type == "string" and not isinstance(value, str):
                errors.append(f"Field '{field_name}' should be string, got {type(value).__name__}")
            elif expected_type == "integer" and not isinstance(value, int):
                errors.append(f"Field '{field_name}' should be integer, got {type(value).__name__}")
            elif expected_type == "number" and not isinstance(value, (int, float)):
                errors.append(f"Field '{field_name}' should be number, got {type(value).__name__}")

    return errors
