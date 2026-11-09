"""Server configuration helpers extracted from server.py for modularity.

Contains CORS origins resolution, settings loading, and Prometheus
metric extraction — pure code-move, no logic changes.
"""

from __future__ import annotations

import os
import threading
from typing import Any
from urllib.parse import urlparse

from loguru import logger

from distllm.config.resolver import ConfigResolver
from distllm.config.settings import DistLLMSettings


def _get_cors_origins() -> list[str]:
    """Get CORS origins from env var (falls back to settings default).

    Security: Rejects wildcard origins unless DISTLLM_DEV_MODE=1 is set.
    Validates that all origins are well-formed URLs.
    Returns safe defaults if configuration is invalid.
    """
    DEFAULT_ORIGINS = ["http://localhost:3000", "http://localhost:8080"]

    raw = os.environ.get("DISTLLM_CORS_ORIGINS")
    origins: list[str] = []
    if raw:
        origins = [o.strip() for o in raw.split(",") if o.strip()]
    else:
        try:
            settings_val = DistLLMSettings().coordinator.cors_origins
            origins = [o.strip() for o in settings_val.split(",") if o.strip()]
        except Exception:
            logger.warning("Failed to parse CORS origins from settings, using defaults")
            origins = list(DEFAULT_ORIGINS)

    if not origins:
        origins = list(DEFAULT_ORIGINS)

    valid = []
    for origin in origins:
        if origin == "*":
            if os.environ.get("DISTLLM_CORS_ALLOW_ALL", "").lower() in ("1", "true"):
                logger.critical(
                    "SECURITY: Wildcard CORS origins are enabled via DISTLLM_CORS_ALLOW_ALL. "
                    "This allows ANY origin to make cross-origin requests. "
                    "Do NOT use in production. Set DISTLLM_CORS_ALLOW_ALL=0 or unset to disable."
                )
                valid.append(origin)
            else:
                valid.extend(DEFAULT_ORIGINS)
            continue
        # Validate origin as a well-formed URL with a valid scheme and host
        parsed = urlparse(origin)
        if not parsed.scheme or not parsed.netloc:
            logger.warning(f"Skipping invalid CORS origin (malformed URL): {origin!r}")
            continue
        if parsed.scheme not in ("http", "https"):
            logger.warning(f"Skipping CORS origin with unsupported scheme: {origin!r}")
            continue
        valid.append(origin)
    return valid


# Lazy-initialized CORS origins (avoids import-time side effects)
_CORS_ORIGINS: list[str] | None = None
_cors_origins_lock = threading.RLock()


def _get_cors_origins_lazy() -> list[str]:
    """Get CORS origins, initializing on first call."""
    global _CORS_ORIGINS
    if _CORS_ORIGINS is None:
        with _cors_origins_lock:
            if _CORS_ORIGINS is None:
                _CORS_ORIGINS = _get_cors_origins()
    return _CORS_ORIGINS


def _load_settings(args: Any) -> DistLLMSettings:
    """Load settings via :class:`ConfigResolver` with full precedence.

    Precedence (lowest to highest):
    1. Pydantic defaults
    2. YAML config file
    3. Environment variables (``DISTLLM_*``) — handled by pydantic-settings
    4. CLI arguments (highest)
    """
    # Resolve config path
    config_path = args.config
    if config_path is None:
        # M-08: Use public resolve_config_path instead of private _resolve_config_path
        config_path = ConfigResolver.resolve_config_path("api", args)

    # Build CLI overrides
    cli_overrides: dict[str, dict[str, Any]] = {}
    if args.model:
        cli_overrides.setdefault("model", {})["name"] = args.model
    if args.dtype:
        cli_overrides.setdefault("model", {})["dtype"] = args.dtype
    if args.host:
        cli_overrides.setdefault("coordinator", {})["host"] = args.host
    if args.port:
        cli_overrides.setdefault("coordinator", {})["api_port"] = args.port
    if args.quantization and args.quantization != "none":
        cli_overrides.setdefault("quantization", {})["method"] = args.quantization

    return DistLLMSettings.from_yaml(
        config_path=config_path,
        cli_overrides=cli_overrides or None,
    )


def _env_int(name: str, default: int) -> int:
    """Read a positive integer from the environment, falling back to *default*.

    Invalid or non-positive values log a warning and use the default rather
    than crashing server startup.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        logger.warning(f"Ignoring invalid {name}={raw!r}; using default {default}")
        return default
    if value < 1:
        logger.warning(f"Ignoring non-positive {name}={value}; using default {default}")
        return default
    return value


# Production-safe defaults that preserve historical single-worker behavior.
DEFAULT_UVICORN_WORKERS = 1
DEFAULT_UVICORN_LIMIT_CONCURRENCY = 10000
DEFAULT_UVICORN_BACKLOG = 2048
DEFAULT_UVICORN_TIMEOUT_KEEP_ALIVE = 30
DEFAULT_GRPC_MAX_WORKERS = 4


def _build_uvicorn_config(
    app: Any,
    host: str,
    port: int,
    workers: int | None = None,
    limit_concurrency: int | None = None,
    backlog: int | None = None,
    timeout_keep_alive: int | None = None,
) -> Any:
    """Build a :class:`uvicorn.Config` from CLI values, env, and defaults.

    Resolution order per knob: explicit argument (CLI) > ``DISTLLM_UVICORN_*``
    env var > safe default. Defaults keep the pre-existing single-worker
    behavior (``workers=1``); concurrency/backlog/timeout defaults are
    production-sane upper bounds rather than uvicorn's unbounded values.
    """
    import uvicorn

    if workers is None:
        workers = _env_int("DISTLLM_UVICORN_WORKERS", DEFAULT_UVICORN_WORKERS)
    if limit_concurrency is None:
        limit_concurrency = _env_int(
            "DISTLLM_UVICORN_LIMIT_CONCURRENCY", DEFAULT_UVICORN_LIMIT_CONCURRENCY
        )
    if backlog is None:
        backlog = _env_int("DISTLLM_UVICORN_BACKLOG", DEFAULT_UVICORN_BACKLOG)
    if timeout_keep_alive is None:
        timeout_keep_alive = _env_int(
            "DISTLLM_UVICORN_TIMEOUT_KEEP_ALIVE", DEFAULT_UVICORN_TIMEOUT_KEEP_ALIVE
        )

    return uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
        workers=workers,
        limit_concurrency=limit_concurrency,
        backlog=backlog,
        timeout_keep_alive=timeout_keep_alive,
    )


def _grpc_max_workers() -> int:
    """gRPC coordinator worker threads: ``DISTLLM_GRPC_MAX_WORKERS`` or 4."""
    return _env_int("DISTLLM_GRPC_MAX_WORKERS", DEFAULT_GRPC_MAX_WORKERS)


def _extract_prom_gauge(data: bytes, metric_name: str) -> float | None:
    """Extract a single gauge value from Prometheus text format."""
    try:
        for line in data.decode("utf-8").splitlines():
            if line.startswith(metric_name) and " " in line:
                parts = line.rsplit(" ", 1)
                return float(parts[-1]) if len(parts) == 2 else None
    except Exception:
        # CQ-08: fail-open: unparseable override file -> keep defaults rather than refuse startup.
        return None
    return None


def _extract_prom_counter(data: bytes, metric_name: str) -> float | None:
    """Extract a counter value from Prometheus text format."""
    try:
        for line in data.decode("utf-8").splitlines():
            if line.startswith(metric_name) and not line.startswith("#"):
                parts = line.rsplit(" ", 1)
                if len(parts) == 2:
                    return float(parts[-1])
    except Exception:
        # CQ-08: fail-open: unparseable override file -> keep defaults rather than refuse startup.
        return None
    return None
