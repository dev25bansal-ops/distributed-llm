"""Lifespan management for the DistLLM API server.

Extracted from ``server.py`` to reduce module size and clarify lifecycle
boundaries.  Exports ``create_lifespan()`` which returns the async
generator function that FastAPI expects for its ``lifespan`` parameter.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import threading
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, nullcontext

from fastapi import FastAPI
from loguru import logger

from distllm.api.server_state import state
from distllm.config.settings import DistLLMSettings
from distllm.core.api_key_store import get_api_key_store
from distllm.core.plugin_system import PluginSystem
from distllm.dashboard.ws_handler import metrics_broadcaster
from distllm.observability.exporter import DistLLMPrometheusExporter
from distllm.observability.logging import setup_logging
from distllm.observability.tracing import setup_tracing
from distllm.plugins.builtin import (
    AuditLogPlugin,
    AuthPlugin,
    MetricsPlugin,
    RateLimitPlugin,
)
from distllm.plugins.health_plugin import HealthPlugin


def _init_observability() -> None:
    """Initialize tracing, logging, metrics exporter."""
    setup_logging(level="INFO", json_format=True)

    # Tracing sampling: default to 10% (head-based) in production.
    # Set DISTLLM_TRACE_SAMPLE_RATE=1.0 for full traces during debugging.
    import os as _os

    _trace_sample_rate = float(_os.environ.get("DISTLLM_TRACE_SAMPLE_RATE", "0.1"))
    setup_tracing(
        service_name="distllm-api",
        sampling_strategy="head",
        sampling_ratio=min(1.0, max(0.0, _trace_sample_rate)),
    )

    state.metrics_exporter = DistLLMPrometheusExporter()


def _init_plugins(ps: PluginSystem) -> None:
    """Register + load + init + start built-in plugins."""
    for cls in (RateLimitPlugin, AuditLogPlugin, MetricsPlugin, HealthPlugin, AuthPlugin):
        ps.register(cls)
    ps.load_all()
    ps.init_all()
    ps.start_all()
    logger.info(f"Plugin system ready: {len(ps.list_plugins())} plugins active")


def _start_ws_broadcaster() -> None:
    """Start the WebSocket metrics broadcaster background task."""
    if state.coordinator is not None:
        state.ws_broadcast_task = asyncio.create_task(metrics_broadcaster(state.coordinator))


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: initialize on startup, clean up on shutdown."""
    import signal

    _init_observability()

    # Initialize plugin system and register built-in plugins
    plugin_config = {"verify_plugins": getattr(state, "verify_plugins", False)}
    state.plugin_system = PluginSystem(config=plugin_config)
    _init_plugins(state.plugin_system)

    # Register SIGHUP handler for configuration hot-reload (Unix only)
    if hasattr(signal, "SIGHUP"):
        _reload_queue: list[dict] = []
        _reload_lock = threading.Lock()

        def _reload_config(signum, frame):
            """Reload configuration on SIGHUP via event loop.

            Signal handlers must not modify shared state directly.
            Instead, enqueue the reload request and let the event loop
            process it asynchronously to avoid race conditions.
            """
            try:
                config_path = os.environ.get("DISTLLM_CONFIG", "config.yaml")
                if os.path.exists(config_path):
                    new_settings = DistLLMSettings.from_yaml(config_path=config_path)
                    with _reload_lock:
                        _reload_queue.append({"settings": new_settings, "path": config_path})
                    logger.info(f"Config hot-reload queued from {config_path}")
                else:
                    logger.warning(f"Config file not found: {config_path}")
            except Exception as e:
                logger.error(f"Config reload failed: {e}")

        async def _process_reload_queue():
            """Process queued config reload requests on the event loop.

            Wrapped with a top-level try/except so that any unexpected error
            does not silently kill the task and permanently break SIGHUP
            config reload until the process is restarted.
            """
            while True:
                try:
                    await asyncio.sleep(1.0)
                    with _reload_lock:
                        if not _reload_queue:
                            continue
                        items = _reload_queue[:]
                        _reload_queue.clear()

                    for item in items:
                        new_settings = item["settings"]
                        config_path = item["path"]
                        coord = getattr(state, "coordinator", None)
                        if coord is not None:
                            logger.info(f"Config reloaded from {config_path}")
                            scheduler = getattr(coord, "scheduler", None)
                            if scheduler is not None:
                                if hasattr(new_settings, "batching"):
                                    with (
                                        scheduler._lock
                                        if hasattr(scheduler, "_lock")
                                        else nullcontext()
                                    ):
                                        scheduler.max_batch_size = new_settings.batching.max_batch_size
                                        scheduler.max_tokens_per_batch = new_settings.batching.max_tokens_per_batch
                        else:
                            logger.info(f"Config reloaded (no coordinator to update)")
                except Exception:
                    logger.exception(
                        "Config reload queue processor crashed -- SIGHUP reloads will not work until the process is restarted"
                    )
                    return

        signal.signal(signal.SIGHUP, _reload_config)
        logger.info("SIGHUP handler registered for config hot-reload")

        # Start background processor on the event loop
        asyncio.ensure_future(_process_reload_queue())

    # Security warning when TLS is disabled
    if not os.environ.get("DISTLLM_TLS_ENABLED", "").lower() in ("1", "true"):
        logger.warning(
            "TLS is DISABLED. API keys and data are transmitted in plaintext. "
            "Set DISTLLM_TLS_ENABLED=true for production deployments."
        )

    # Log API key presence (never log the raw key)
    store = get_api_key_store()
    display_key = store.get_display_key()
    if display_key:
        fingerprint = hashlib.sha256(display_key.encode()).hexdigest()[:12]
        logger.info("API key configured (fingerprint: %s...)", fingerprint)
        logger.info("Use 'distllm config keys' to view or rotate keys.")
    else:
        logger.info("API keys loaded from config file. Use 'distllm config keys' to manage.")

    _start_ws_broadcaster()
    yield
    if state.plugin_system:
        state.plugin_system.stop_all()
    if state.ws_broadcast_task:
        state.ws_broadcast_task.cancel()


def create_lifespan():
    """Create the lifespan async generator function for FastAPI.

    Returns the ``@asynccontextmanager`` async generator that FastAPI
    calls on startup and shutdown.
    """
    return _lifespan
