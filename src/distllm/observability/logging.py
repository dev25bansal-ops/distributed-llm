"""Structured logging configuration for distributed-llm.

Configures loguru with JSON output, structured context fields,
and optional Loki push for centralized log aggregation.
"""

import sys
import json
from typing import Optional
from loguru import logger


def setup_logging(
    level: str = "INFO",
    json_format: bool = True,
    loki_url: Optional[str] = None,
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
        logger.add(
            sys.stdout,
            format=lambda record: json.dumps(
                {
                    "timestamp": record["time"].strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                    "level": record["level"].name,
                    "service": service_name,
                    "module": record["name"],
                    "function": record["function"],
                    "line": record["line"],
                    "message": record["message"],
                    "extra": {
                        k: v for k, v in record["extra"].items()
                        if k not in ("elaborated",)
                    },
                },
                default=str,
            ),
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
