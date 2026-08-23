"""Request handler for the Coordinator.

Extracts generation request methods from coordinator.py into a dedicated
class to reduce the monolith.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any

from loguru import logger


class RequestHandler:
    """Handles generation requests for the Coordinator.

    Holds a reference to the coordinator for shared state access.
    Extracted from :class:`distllm.core.coordinator.Coordinator`.
    """

    def __init__(self, coordinator: Any) -> None:
        self.coordinator = coordinator

    # ── Generation ──────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        request_id: str | None = None,
        user_id: str = "default",
        speculative_config: dict | None = None,
        response_format: dict | None = None,
        constraint: Any | None = None,
    ) -> str:
        c = self.coordinator
        c._inference_engine.tokenizer = c.tokenizer

        # Optional AgenticRouter pre-routing: if configured, use the LLM
        # judge to select the optimal model before delegating to the engine.
        if c._agentic_router is not None:
            decision = c._agentic_router.route(prompt)
            if decision.model and decision.model != c.model_name:
                logger.info(
                    f"AgenticRouter selected model={decision.model} "
                    f"(confidence={decision.confidence:.2f}) "
                    f"instead of default {c.model_name}"
                )

        # Use caller-provided constraint if given, otherwise build from response_format
        if constraint is None and response_format:
            from distllm.core.structured_output import JSONSchemaConstraint

            constraint = JSONSchemaConstraint.from_response_format(
                response_format,
                tokenizer=c.tokenizer,
            )

        try:
            return c._inference_engine.generate(
                prompt,
                max_new_tokens,
                temperature,
                top_p,
                top_k,
                request_id,
                user_id,
                speculative_config,
                constraint=constraint,
            )
        except Exception as exc:
            if c._plugin_system:
                c._plugin_system.dispatch(
                    "on_error",
                    {
                        "prompt": prompt[:128],
                        "request_id": request_id or "",
                        "user_id": user_id,
                    },
                    exc,
                )
            raise

    def load_local_model(self) -> None:
        c = self.coordinator
        c._inference_engine.tokenizer = c.tokenizer
        c._inference_engine.load_local_model()
        c.tokenizer = c._inference_engine.tokenizer
        c._inference_engine.tokenizer = c.tokenizer
        if c._plugin_system:
            c._plugin_system.dispatch("on_model_load", c.model_name, {"local": True})

    def set_deterministic_mode(self, enabled: bool = True, seed: int = 42) -> None:
        self.coordinator._inference_engine.set_deterministic_mode(enabled, seed)

    def get_recent_requests(self, n: int = 10) -> list[Any]:
        return self.coordinator._inference_engine.get_recent_requests(n)

    # ── Async generation ────────────────────────────────────────────────────

    def generate_async(self, prompt: str, **kwargs: Any) -> str:
        """Schedule async generation via the batch scheduler.

        If the batch scheduler is configured, adds a Sequence directly
        to the scheduler for true continuous batching.  Otherwise falls
        back to a background thread.

        Returns a request_id immediately.  Call ``wait_for_result()``
        to get the result when ready.
        """
        c = self.coordinator
        request_id = kwargs.pop("request_id", None) or str(uuid.uuid4())

        # Try the real batch scheduler path first
        if c._batch_scheduler is not None and c.tokenizer is not None:
            try:
                from distllm.core.request_pipeline import RequestPipeline

                pipeline = RequestPipeline(c)
                return pipeline.generate_async(
                    prompt=prompt,
                    request_id=request_id,
                    max_new_tokens=kwargs.get("max_new_tokens", 128),
                    temperature=kwargs.get("temperature", 0.7),
                    top_p=kwargs.get("top_p", 0.9),
                    top_k=kwargs.get("top_k", 0),
                    user_id=kwargs.get("user_id", "default"),
                )
            except Exception as e:
                logger.warning(
                    f"Batch scheduler path failed, falling back to thread: {e}"
                )

        # Fallback: background thread
        max_new_tokens = kwargs.get("max_new_tokens", 128)
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.9)
        top_k = kwargs.get("top_k", 0)
        user_id = kwargs.get("user_id", "default")

        event = threading.Event()
        with c._request_lock:
            c._request_events[request_id] = event

        def _run() -> None:
            try:
                result = self.generate(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    user_id=user_id,
                )
                with c._request_lock:
                    c._request_results[request_id] = result
                    c._request_results_created[request_id] = time.monotonic()
            except Exception as e:
                with c._request_lock:
                    c._request_results[request_id] = f"[Error: {e}]"
                    c._request_results_created[request_id] = time.monotonic()
            finally:
                event.set()

        thread = threading.Thread(
            target=_run, daemon=True, name=f"gen-{request_id[:8]}"
        )
        thread.start()

        logger.debug(
            f"generate_async -> request_id={request_id} "
            f"(background thread started, prompt length {len(prompt)})"
        )
        return request_id

    # ── Metrics ─────────────────────────────────────────────────────────────

    def record_metric(self, name: str, value: float = 1.0) -> None:
        """Record a metric (used by RequestPipeline)."""
        c = self.coordinator
        if hasattr(c, '_metrics_collector') and c._metrics_collector is not None:
            try:
                c._metrics_collector.record(name, value)
            except Exception as e:
                logger.warning(f"Failed to record metric '{name}': {e}")

    # ── Result waiting ──────────────────────────────────────────────────────

    def _cleanup_stale_results(self) -> None:
        """Remove stale entries from _request_results to prevent memory leaks."""
        c = self.coordinator
        now = time.monotonic()
        with c._request_lock:
            stale = [
                rid
                for rid, created in c._request_results_created.items()
                if now - created > c._result_ttl_s
            ]
            for rid in stale:
                c._request_results.pop(rid, None)
                c._request_events.pop(rid, None)
                c._request_results_created.pop(rid, None)
            if stale:
                logger.debug(
                    f"Cleaned {len(stale)} stale request results "
                    f"(TTL={c._result_ttl_s}s)"
                )

    def wait_for_result(
        self, request_id: str, timeout: float | None = None
    ) -> str:
        """Wait for an async generation result.

        Checks both the request tracker (batch scheduler path) and
        the legacy event-based path (background thread fallback).
        """
        c = self.coordinator
        # Periodic cleanup — called opportunistically from wait_for_result
        self._cleanup_stale_results()
        # Try request tracker first (batch scheduler path)
        if c._request_tracker is not None:
            try:
                return c._request_tracker.wait_for_result(
                    request_id, timeout or 120.0
                )
            except (ValueError, TimeoutError):
                pass

        # Fallback: legacy event-based path
        with c._request_lock:
            event = c._request_events.get(request_id)
        if event is None:
            raise ValueError(f"Unknown request_id: {request_id}")
        event.wait(timeout=timeout)
        with c._request_lock:
            result = c._request_results.pop(request_id, None)
            c._request_events.pop(request_id, None)
        if result is None:
            raise TimeoutError(f"Request {request_id} timed out")
        return result
