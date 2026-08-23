"""Continuous batching engine — bridges IterationScheduler with PipelineOrchestrator.

Provides iteration-level continuous batching for distributed inference:
- Accepts new requests while decoding is in progress
- Separates prefill (first forward pass) from decode (subsequent tokens)
- Maintains per-request KV cache isolation across pipeline nodes
- Runs a background scheduling loop driven by the BatchScheduler / IterationScheduler
- Returns generated tokens via per-request asyncio.Future

Usage::

    engine = ContinuousBatchingEngine(scheduler, orchestrator)
    await engine.start()

    future = engine.submit("req-1", [101, 205, 309], max_new_tokens=128)
    tokens = await future  # list[int]
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import torch
from loguru import logger

from distllm.core.scheduler.sequence import GenerationConfig, Sequence, SequenceStatus


class ContinuousBatchingEngine:
    """Continuous batching engine for distributed inference.

    Bridges a BatchScheduler (or IterationScheduler) with a PipelineOrchestrator
    to provide continuous batching. New requests can be submitted at any time;
    the engine drains them into the scheduler and runs a background decode loop
    that iterates schedule -> execute -> step -> finalize.

    KV cache isolation:
    - Each request maintains its own per-node KV cache in ``self._kv_caches``.
    - Prefill runs each new request individually through the pipeline so its
      KV cache is populated.
    - Decode runs one token at a time per request, reusing the accumulated
      KV cache.

    Thread safety:
    - ``submit()`` is safe to call from any coroutine (uses an asyncio.Queue).
    - The decode loop owns the scheduler and orchestrator calls.
    - Futures are delivered on the decode loop's event loop iteration.
    """

    def __init__(
        self,
        scheduler: Any,
        orchestrator: Any,
        *,
        loop_interval_s: float = 0.001,
        max_queue_drain: int = 64,
        use_microbatched_prefill: bool = True,
    ):
        """Initialize the continuous batching engine.

        Args:
            scheduler: A BatchScheduler or IterationScheduler instance.
            orchestrator: A PipelineOrchestrator instance.
            loop_interval_s: Sleep interval when no batch is available (seconds).
            max_queue_drain: Max requests to drain from queue per iteration.
            use_microbatched_prefill: Use run_pipeline_microbatched for prefill
                (True) or simple run_pipeline (False).
        """
        self._scheduler = scheduler
        self._orchestrator = orchestrator
        self._loop_interval = loop_interval_s
        self._max_queue_drain = max_queue_drain
        self._use_microbatched = use_microbatched_prefill

        # Incoming request queue
        self._request_queue: asyncio.Queue[Sequence] = asyncio.Queue()

        # Per-request futures — resolved when generation completes
        self._results: dict[str, asyncio.Future[list[int]]] = {}

        # KV cache: request_id -> {node_id: list[tensor] | None}
        self._kv_caches: dict[str, dict[str, list | None]] = {}

        # Generation config per request (sampling parameters)
        self._generation_configs: dict[str, GenerationConfig] = {}

        # Lifecycle
        self._running = False
        self._loop_task: asyncio.Task | None = None

        # Metrics
        self._metrics: dict[str, float | int] = {
            "iterations": 0,
            "batches": 0,
            "prefill_runs": 0,
            "decode_runs": 0,
            "completed_requests": 0,
            "failed_requests": 0,
            "total_iteration_ms": 0.0,
        }

    # ── Public API ─────────────────────────────────────────────────────

    @property
    def scheduler(self) -> Any:
        """The underlying BatchScheduler / IterationScheduler instance."""
        return self._scheduler

    @property
    def running(self) -> bool:
        """Whether the background decode loop is active."""
        return self._running

    def submit(
        self,
        request_id: str,
        prompt_tokens: list[int],
        generation_config: GenerationConfig | None = None,
        *,
        priority: int = 2,
        max_new_tokens: int = 256,
    ) -> asyncio.Future[list[int]]:
        """Submit a generation request to the engine.

        Creates a Sequence from the prompt tokens and enqueues it for
        scheduling. Returns a Future that resolves to the list of generated
        tokens (``request_id`` -> ``generated_tokens``).

        Args:
            request_id: Unique identifier for this request.
            prompt_tokens: Input token IDs (pre-tokenized).
            generation_config: Sampling parameters (temperature, top_p, etc.).
                If None, a default config is created from ``max_new_tokens``.
            priority: Scheduling priority (0 = critical, 1 = high, 2 = normal,
                3 = low).
            max_new_tokens: Maximum tokens to generate. Only used when
                ``generation_config`` is None.

        Returns:
            An asyncio.Future that resolves to ``list[int]`` of generated tokens.
        """
        loop = asyncio.get_event_loop()
        future: asyncio.Future[list[int]] = loop.create_future()
        self._results[request_id] = future

        if generation_config is None:
            generation_config = GenerationConfig(max_new_tokens=max_new_tokens)

        self._generation_configs[request_id] = generation_config

        seq = Sequence(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            priority=priority,
            max_new_tokens=generation_config.max_new_tokens,
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            top_k=generation_config.top_k,
            stop_token_ids=list(generation_config.stop_token_ids),
        )
        self._request_queue.put_nowait(seq)
        logger.debug(
            "ContinuousBatching: queued {} ({} tokens, pri={})",
            request_id,
            len(prompt_tokens),
            priority,
        )
        return future

    async def start(self) -> None:
        """Start the background decode loop.

        Idempotent — safe to call multiple times.
        """
        if self._running:
            logger.warning("ContinuousBatchingEngine already running")
            return
        self._running = True
        self._loop_task = asyncio.create_task(self._decode_loop())
        logger.info("ContinuousBatchingEngine started")

    async def stop(self) -> None:
        """Stop the background decode loop and wait for completion."""
        self._running = False
        if self._loop_task is not None:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None
        logger.info("ContinuousBatchingEngine stopped")

    # ── Background decode loop ─────────────────────────────────────────

    async def _decode_loop(self) -> None:
        """Main scheduling loop.

        Iteration steps:
        1. Drain any queued requests into the scheduler.
        2. Call ``scheduler.schedule()`` to get the next batch.
        3. Execute the batch through the pipeline (``_execute_batch``).
        4. Call ``scheduler.step()`` to advance sequence states.
        5. Resolve futures for completed sequences.
        """
        while self._running:
            try:
                # ── Step 1: Drain pending requests ─────────────────────
                drained = 0
                while (
                    not self._request_queue.empty()
                    and drained < self._max_queue_drain
                ):
                    seq = self._request_queue.get_nowait()
                    self._scheduler.add(seq)
                    drained += 1

                # ── Step 2: Schedule ───────────────────────────────────
                batch = self._scheduler.schedule()
                if batch is None:
                    await asyncio.sleep(self._loop_interval)
                    continue

                self._metrics["batches"] += 1  # type: ignore[operator]

                # ── Step 3: Execute ────────────────────────────────────
                iter_start = time.monotonic()
                next_tokens_tensor = await self._execute_batch(batch)
                iter_ms = (time.monotonic() - iter_start) * 1000
                self._metrics["total_iteration_ms"] += iter_ms  # type: ignore[operator]
                self._metrics["iterations"] += 1  # type: ignore[operator]

                # ── Step 4: Step scheduler ─────────────────────────────
                self._scheduler.step(batch, next_tokens_tensor)

                # ── Step 5: Finalize completed ─────────────────────────
                self._finalize_completed(batch)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("ContinuousBatching: decode loop error")
                await asyncio.sleep(self._loop_interval)

        # Engine is stopping — fail any remaining futures to avoid hangs
        for request_id, fut in list(self._results.items()):
            if not fut.done():
                fut.set_exception(RuntimeError("Engine stopped before request completed"))
        self._results.clear()
        self._kv_caches.clear()
        self._generation_configs.clear()

    # ── Batch execution ────────────────────────────────────────────────

    async def _execute_batch(self, batch: Any) -> torch.Tensor:
        """Execute one scheduled batch through the pipeline.

        Runs each sequence according to its current status:
        - PREFILLING -> ``_execute_prefill`` (full prompt through pipeline)
        - DECODING   -> ``_execute_decode_token`` (single token, reuse KV cache)

        Returns a 1-D int64 tensor of next-token IDs in batch order.
        """
        next_tokens: list[int] = [0] * len(batch.sequences)

        for idx, seq in enumerate(batch.sequences):
            if seq.status == SequenceStatus.PREFILLING:
                next_tokens[idx] = await self._execute_prefill(seq)
            elif seq.status == SequenceStatus.DECODING:
                next_tokens[idx] = await self._execute_decode_token(seq)
            # PENDING / DONE / FAILED sequences are skipped

        return torch.tensor(next_tokens, dtype=torch.long)

    async def _execute_prefill(self, seq: Sequence) -> int:
        """Run the prefill (first forward pass) for one request.

        Initialises per-node KV cache storage and routes the full prompt
        through the pipeline.  Returns the first generated token ID
        (argmax of the last-position logit).

        On failure the request's future is marked as failed and token 0
        is returned as a safe fallback.
        """
        request_id = seq.request_id

        # Initialise per-node KV cache.
        node_order = self._orchestrator.node_order  # type: ignore[union-attr]
        node_kv_caches: dict[str, list | None] = {nid: None for nid in node_order}
        self._kv_caches[request_id] = node_kv_caches

        input_ids = torch.tensor([seq.prompt_tokens], dtype=torch.long)

        try:
            if self._use_microbatched:
                output = await self._orchestrator.run_pipeline_microbatched(  # type: ignore[union-attr]
                    input_ids=input_ids,
                    node_kv_caches=node_kv_caches,
                    request_id=request_id,
                )
            else:
                # Synchronous fallback — run in executor to avoid blocking.
                loop = asyncio.get_event_loop()
                output = await loop.run_in_executor(
                    None,
                    self._orchestrator.run_pipeline,  # type: ignore[union-attr]
                    input_ids,
                    node_kv_caches,
                    request_id,
                )

            # output shape: typically (1, prompt_len, vocab_size)
            next_token = int(output[0, -1].argmax().item())
            self._metrics["prefill_runs"] += 1  # type: ignore[operator]
            return next_token

        except Exception:
            logger.exception("ContinuousBatching: prefill failed for {}", request_id)
            self._metrics["failed_requests"] += 1  # type: ignore[operator]
            fut = self._results.pop(request_id, None)
            if fut is not None and not fut.done():
                fut.set_exception(RuntimeError(f"Prefill failed for {request_id}"))
            self._cleanup_request(request_id)
            return 0

    async def _execute_decode_token(self, seq: Sequence) -> int:
        """Run a single decode step for one request.

        Routes the current decode input token through the pipeline,
        reusing the request's existing KV cache.  Returns the next
        token ID (argmax of the output logit).

        If the KV cache is missing the request is failed.
        """
        request_id = seq.request_id
        node_kv_caches = self._kv_caches.get(request_id)
        if node_kv_caches is None:
            logger.error(
                "ContinuousBatching: missing KV cache for {} — failing",
                request_id,
            )
            self._metrics["failed_requests"] += 1  # type: ignore[operator]
            fut = self._results.pop(request_id, None)
            if fut is not None and not fut.done():
                fut.set_exception(
                    RuntimeError(f"Missing KV cache for {request_id} in decode")
                )
            self._cleanup_request(request_id)
            return 0

        input_ids = torch.tensor([[seq.decode_input_token]], dtype=torch.long)

        try:
            if self._use_microbatched:
                output = await self._orchestrator.run_pipeline_microbatched(  # type: ignore[union-attr]
                    input_ids=input_ids,
                    node_kv_caches=node_kv_caches,
                    request_id=request_id,
                )
            else:
                loop = asyncio.get_event_loop()
                output = await loop.run_in_executor(
                    None,
                    self._orchestrator.run_pipeline,  # type: ignore[union-attr]
                    input_ids,
                    node_kv_caches,
                    request_id,
                )

            next_token = int(output[0, -1].argmax().item())
            self._metrics["decode_runs"] += 1  # type: ignore[operator]
            return next_token

        except Exception:
            logger.exception("ContinuousBatching: decode failed for {}", request_id)
            self._metrics["failed_requests"] += 1  # type: ignore[operator]
            fut = self._results.pop(request_id, None)
            if fut is not None and not fut.done():
                fut.set_exception(RuntimeError(f"Decode failed for {request_id}"))
            self._cleanup_request(request_id)
            return 0

    # ── Completion handling ────────────────────────────────────────────

    def _finalize_completed(self, batch: Any) -> None:
        """Resolve futures for any completed sequences in *batch*."""
        for seq in batch.sequences:
            if seq.is_complete:
                request_id = seq.request_id
                fut = self._results.pop(request_id, None)
                if fut is not None and not fut.done():
                    fut.set_result(list(seq.generated_tokens))
                    self._metrics["completed_requests"] += 1  # type: ignore[operator]
                    logger.debug(
                        "ContinuousBatching: completed {} ({} tokens)",
                        request_id,
                        len(seq.generated_tokens),
                    )
                self._cleanup_request(request_id)

    def _cleanup_request(self, request_id: str) -> None:
        """Release internal state for a finished request."""
        self._kv_caches.pop(request_id, None)
        self._generation_configs.pop(request_id, None)

    # ── Metrics ────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Current engine statistics."""
        total_ms: float = self._metrics.get("total_iteration_ms", 0.0)  # type: ignore[assignment]
        iters: int = self._metrics.get("iterations", 0)  # type: ignore[assignment]

        # qsize is only meaningful when there is a running loop
        try:
            qsize = self._request_queue.qsize()
        except NotImplementedError:
            qsize = -1

        return {
            "running": self._running,
            "pending_queue": qsize,
            "active_requests": len(self._kv_caches),
            "outstanding_futures": len(self._results),
            "avg_iteration_ms": total_ms / iters if iters > 0 else 0.0,
            **{k: v for k, v in self._metrics.items()},
        }

    async def __aenter__(self) -> ContinuousBatchingEngine:
        await self.start()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.stop()
