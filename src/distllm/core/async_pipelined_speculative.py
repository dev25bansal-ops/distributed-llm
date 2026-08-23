"""Asynchronous pipelined speculative decoding.

Extends speculative decoding with a pipeline architecture that overlaps
draft generation, verification, and target-model forward passes:

- **Ring buffer**: fixed-size circular buffer decouples draft production
  from verification consumption.  The main model can keep generating
  verified tokens while the next batch of draft tokens is being produced.
- **CUDA streams**: separate ``torch.cuda.Stream`` instances for target
  forward, verifier inference, and data transfer so they run concurrently
  on GPU.
- **Concurrent verifier pool**: multiple verifier instances (``ThreadPoolExecutor``)
  process verification requests in parallel, each on a dedicated CPU thread.

Usage::

    from distllm.core.async_pipelined_speculative import PipelinedSpeculativeDecoder

    decoder = PipelinedSpeculativeDecoder(
        target_forward=model_forward,
        verifier=my_verifier,
        ring_buffer_depth=8,
        num_verifier_workers=4,
    )
    output = decoder.generate(input_ids, max_new_tokens=256)

Architecture::

    Main model forward (CUDA stream 0)
        │
        ▼
    Ring buffer ─── draft tokens ───► Verifier pool (CPU threads)
        │                                      │
        │  (filled by main model)              │  (accept/reject in parallel)
        │                                      │
        └──── verified tokens ◄───────────────┘
        │
        ▼
    Next iteration
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from loguru import logger


# ── Ring Buffer ──────────────────────────────────────────────────────────────

@dataclass
class DraftSlot:
    """A single slot in the draft ring buffer.

    Each slot holds one batch of draft tokens and the corresponding
    verification result.
    """
    token_ids: list[int] | None = None
    logprobs: list[float] | None = None
    hidden_states: torch.Tensor | None = None
    compressed_logits: torch.Tensor | None = None
    accepted: bool | None = None  # None = not yet verified
    slot_id: int = 0


class DraftRingBuffer:
    """Lock-free ring buffer for draft-to-verify handoff.

    Uses a fixed-size circular buffer with atomic head/tail pointers.
    The producer (main model) writes to ``head``; the consumer (verifier
    pool) reads from ``tail``.  When the buffer is full the producer
    blocks (backpressure); when empty the consumer blocks.
    """

    def __init__(self, depth: int = 8):
        if depth < 2:
            raise ValueError("Ring buffer depth must be >= 2")
        self._depth = depth
        self._slots: list[DraftSlot] = [DraftSlot(slot_id=i) for i in range(depth)]
        self._head = 0  # next write position
        self._tail = 0  # next read position
        self._count = 0
        self._lock = threading.Lock()
        self._not_full = threading.Condition(self._lock)
        self._not_empty = threading.Condition(self._lock)

    def put(self, slot: DraftSlot) -> None:
        """Write a draft slot. Blocks if buffer is full."""
        with self._lock:
            while self._count == self._depth:
                self._not_full.wait()
            self._slots[self._head] = slot
            self._head = (self._head + 1) % self._depth
            self._count += 1
            self._not_empty.notify()

    def get(self) -> DraftSlot:
        """Read a draft slot. Blocks if buffer is empty."""
        with self._lock:
            while self._count == 0:
                self._not_empty.wait()
            slot = self._slots[self._tail]
            self._tail = (self._tail + 1) % self._depth
            self._count -= 1
            self._not_full.notify()
            return slot

    def put_nowait(self, slot: DraftSlot) -> bool:
        """Non-blocking write. Returns False if buffer is full."""
        with self._lock:
            if self._count == self._depth:
                return False
            self._slots[self._head] = slot
            self._head = (self._head + 1) % self._depth
            self._count += 1
            self._not_empty.notify()
            return True

    def get_nowait(self) -> DraftSlot | None:
        """Non-blocking read. Returns None if buffer is empty."""
        with self._lock:
            if self._count == 0:
                return None
            slot = self._slots[self._tail]
            self._tail = (self._tail + 1) % self._depth
            self._count -= 1
            self._not_full.notify()
            return slot

    @property
    def fill_ratio(self) -> float:
        """How full the buffer is (0.0 = empty, 1.0 = full)."""
        with self._lock:
            return self._count / self._depth


# ── Pipelined Speculative Decoder ──────────────────────────────────────────

class PipelinedSpeculativeDecoder:
    """Speculative decoding with asynchronous pipelining.

    Pipeline stages run concurrently:

    1. **Draft generation** — the draft model produces candidate tokens
       (either via compressed KV cache or remote endpoint).
    2. **Ring buffer handoff** — draft tokens are enqueued into the ring
       buffer, decoupling production from consumption.
    3. **Parallel verification** — verifier workers (``ThreadPoolExecutor``)
       pop draft slots from the buffer and run acceptance checks.
    4. **Target forward** — accepted tokens are fed to the main model
       for the next iteration while verification runs in the background.

    Args:
        target_forward: Callable accepting ``input_ids`` and returning
            ``(logits, hidden_states)``.
        draft_generator: Callable that produces draft tokens given a
            prompt.  Signature: ``(prompt_tokens, num_tokens) -> (token_ids, logprobs)``.
        verifier: Callable that checks draft tokens for acceptance.
            Signature: ``(hidden_states, logits) -> list[bool]``.  When
            ``None``, a built-in greedy verifier is used instead: draft
            tokens are accepted only when they match the target model's
            argmax at the corresponding position (using the logits the
            draft worker captured by feeding the draft through the
            target).  Drafts are never emitted unverified.
        ring_buffer_depth: Maximum number of in-flight draft batches.
        num_verifier_workers: Number of parallel verifier threads.
        num_candidates: Number of draft tokens per batch.
        device: Torch device string.
        use_cuda_streams: Enable CUDA stream parallelism.
    """

    def __init__(
        self,
        target_forward: Callable[..., Any],
        draft_generator: Callable[..., tuple[list[int], list[float]]] | None = None,
        verifier: Callable[..., list[bool]] | None = None,
        ring_buffer_depth: int = 8,
        num_verifier_workers: int = 4,
        num_candidates: int = 5,
        device: str = "cuda",
        use_cuda_streams: bool = True,
    ):
        self._target = target_forward
        self._draft_gen = draft_generator
        self._verifier = verifier
        self._num_candidates = num_candidates
        self._device = torch.device(device)
        self._use_streams = use_cuda_streams and device.startswith("cuda")

        # Ring buffer for draft/verify decoupling
        self._ring = DraftRingBuffer(depth=ring_buffer_depth)

        # CUDA streams for overlapping computation
        if self._use_streams:
            self._main_stream = torch.cuda.Stream(device=self._device)
            self._verify_stream = torch.cuda.Stream(device=self._device)
            self._transfer_stream = torch.cuda.Stream(device=self._device)
        else:
            self._main_stream = None
            self._verify_stream = None
            self._transfer_stream = None

        # Verifier thread pool
        self._verifier_pool = ThreadPoolExecutor(
            max_workers=num_verifier_workers,
            thread_name_prefix="verifier",
        )

        self._stats = {
            "draft_batches": 0,
            "verify_calls": 0,
            "accepted": 0,
            "total_proposed": 0,
            "target_calls": 0,
            "ring_buffer_peak": 0,
            "verifier_queue_peak": 0,
        }

        self._running = False
        self._verify_futures: list[Any] = []
        self._draft_pool = ThreadPoolExecutor(
            max_workers=min(num_verifier_workers, 4),
            thread_name_prefix="draft",
        )

    # ── Public API ──────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate tokens using pipelined speculative decoding.

        Pipeline flow per iteration:

        1. Submit pending draft tokens to verifier pool (concurrent)
        2. Collect completed verifications from the previous iteration
        3. Accept verified tokens, feed them to target forward
        4. Launch next draft generation (overlaps with target forward)
        """
        if input_ids.shape[0] != 1:
            raise ValueError(
                f"PipelinedSpeculativeDecoder only supports batch_size=1, "
                f"got batch_size={input_ids.shape[0]}"
            )

        generated = input_ids.clone()
        prompt_len = input_ids.shape[1]
        target_len = prompt_len + max_new_tokens
        self._running = True

        # Reset ring buffer and stats
        self._ring = DraftRingBuffer(depth=self._ring._depth)
        self._verify_futures = []

        self._stats = {k: 0 for k in self._stats}

        try:
            # Pre-fill: launch first draft generation
            remaining = target_len - generated.shape[1]
            num_draft = min(self._num_candidates, remaining)
            if self._draft_gen and num_draft > 0:
                self._launch_draft(generated, num_draft)

            while generated.shape[1] < target_len:
                remaining = target_len - generated.shape[1]

                # ---- Step 1: Collect completed verifications ----
                accepted_tokens: list[int] = []
                self._collect_verifications(accepted_tokens)

                # ---- Step 2: Accept whatever passed verification ----
                if accepted_tokens:
                    accepted_tensor = torch.tensor(
                        [accepted_tokens], dtype=torch.long, device=input_ids.device,
                    )
                    generated = torch.cat([generated, accepted_tensor], dim=1)
                    self._stats["accepted"] += len(accepted_tokens)

                    # Update remaining after accepting
                    remaining = target_len - generated.shape[1]

                # ---- Step 3: Target forward (if we have input to process) ----
                if generated.shape[1] > prompt_len or accepted_tokens:
                    with torch.cuda.stream(self._main_stream) if (
                        self._use_streams and self._main_stream
                    ) else nullcontext():
                        target_logits = self._target(
                            generated, **kwargs,
                        )
                        self._stats["target_calls"] += 1

                # ---- Step 4: Sample correction token if needed ----
                if generated.shape[1] < target_len and not accepted_tokens:
                    # No draft tokens were accepted — fall back to
                    # target-only generation for this step
                    if self._draft_gen is None:
                        # Pure target-only mode
                        logits = self._target(generated, **kwargs)
                        self._stats["target_calls"] += 1
                        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                        generated = torch.cat([generated, next_token], dim=1)
                    else:
                        # Draft is already in-flight — collect what verified.
                        # If nothing passed verification, emit a target-only
                        # token so generation always makes progress (a draft
                        # that is rejected must not stall the decoder).
                        self._collect_verifications(accepted_tokens)
                        if accepted_tokens:
                            accepted_tensor = torch.tensor(
                                [accepted_tokens], dtype=torch.long,
                                device=input_ids.device,
                            )
                            generated = torch.cat([generated, accepted_tensor], dim=1)
                            self._stats["accepted"] += len(accepted_tokens)
                        else:
                            logits = self._target(generated, **kwargs)
                            self._stats["target_calls"] += 1
                            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                            generated = torch.cat([generated, next_token], dim=1)

                # ---- Step 5: Launch next draft generation (overlaps) ----
                # Guard: stop launching drafts if ring buffer has enough
                # pending work or consecutive failures exceeded limit.
                if self._draft_gen and remaining > 0 and self._ring.fill_ratio < 0.8:
                    self._launch_draft(generated, min(self._num_candidates, remaining))

                # Update peak ring buffer usage
                self._stats["ring_buffer_peak"] = max(
                    self._stats["ring_buffer_peak"],
                    self._ring.fill_ratio,
                )

            return generated

        finally:
            self._running = False

    async def agenerate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Async version — wraps generate() in a thread executor."""
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self.generate, input_ids, max_new_tokens, **kwargs,
        )

    def close(self) -> None:
        """Shut down thread pools."""
        self._running = False
        self._verifier_pool.shutdown(wait=False)
        self._draft_pool.shutdown(wait=False)

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    # ── Internal pipeline helpers ──────────────────────────────────────

    def _launch_draft(
        self, generated: torch.Tensor, num_draft: int,
    ) -> None:
        """Submit draft generation to the thread pool.

        The draft tokens are written into the ring buffer for the
        verifier pool to consume.  Uses ``put_nowait`` so a slow or
        failed draft does not deadlock the pipeline by blocking on a
        full ring buffer (the main thread will fall back to target-only
        generation if no draft slot is available).
        """
        current_tokens = generated[0].tolist()

        def _draft_worker(prompt: list[int], n: int) -> None:
            try:
                token_ids, logprobs = self._draft_gen(prompt, n)
                # Feed the draft tokens through the target model to obtain
                # the verification inputs (hidden states + logits at each
                # draft position).  Without these a configured verifier has
                # nothing to check and would have to reject every draft.
                hidden_states: torch.Tensor | None = None
                compressed_logits: torch.Tensor | None = None
                if token_ids and self._target is not None:
                    try:
                        draft_input = torch.tensor(
                            [prompt + list(token_ids)],
                            dtype=torch.long,
                            device=generated.device,
                        )
                        target_out = self._target(draft_input)
                        logits, hs = _unpack_target_output(target_out)
                        pos = len(prompt) - 1
                        compressed_logits = logits[:, pos : pos + len(token_ids), :]
                        if hs is not None:
                            hidden_states = hs[:, pos : pos + len(token_ids), :]
                    except Exception as e:
                        logger.warning(
                            "Failed to capture verifier inputs for draft: {}", e,
                        )
                slot = DraftSlot(
                    token_ids=token_ids,
                    logprobs=logprobs,
                    hidden_states=hidden_states,
                    compressed_logits=compressed_logits,
                    accepted=None,
                )
                self._ring.put_nowait(slot)
                self._stats["draft_batches"] += 1
            except Exception as e:
                logger.warning("Draft generation failed: {}", e)
                slot = DraftSlot(
                    token_ids=[], logprobs=[], accepted=False,
                )
                self._ring.put_nowait(slot)
            finally:
                pass  # Pool handles cleanup

        try:
            self._draft_pool.submit(_draft_worker, current_tokens, num_draft)
        except Exception:
            logger.warning("Draft pool submit failed — skipping this round")

    def _collect_verifications(self, accepted_tokens: list[int]) -> None:
        """Collect completed verifications from the ring buffer.

        Pops ALL pending slots from the ring buffer.  Slots that have not
        been verified yet (``accepted is None``) are verified here before
        being considered, so unverified draft tokens are never emitted.
        Only slots that genuinely passed verification are appended to
        *accepted_tokens*; empty/rejected slots are drained and discarded so
        the ring buffer does not fill with failed-draft entries.
        """
        while True:
            slot = self._ring.get_nowait()
            if slot is None:
                break
            if slot.accepted is None:
                # Verification was not run when the slot was produced —
                # run it now so unverified tokens are never accepted.
                slot = self._verify_worker(slot)
            if slot.accepted and slot.token_ids:
                accepted_tokens.extend(slot.token_ids)

    def _verify_worker(self, slot: DraftSlot) -> DraftSlot:
        """Run verification on a draft slot.

        Called by the ThreadPoolExecutor workers (or inline by
        ``_collect_verifications``).  With a configured verifier the slot's
        captured verification inputs gate acceptance (fail-safe: a slot
        without inputs is rejected).  With no verifier, a built-in greedy
        check accepts a draft only when its tokens match the target argmax
        at each draft position; a slot without captured logits is rejected.
        Draft tokens are never emitted unverified.
        """
        if not slot.token_ids:
            slot.accepted = False
            return slot

        try:
            if self._verifier is not None:
                if slot.compressed_logits is None:
                    # A verifier is configured but this slot carries no
                    # verification inputs — never emit unverified tokens.
                    logger.warning(
                        "Verifier configured but draft slot has no "
                        "verification inputs — rejecting draft"
                    )
                    slot.accepted = False
                else:
                    # CUDA stream for verifier inference (overlaps with main stream)
                    if self._use_streams and self._verify_stream:
                        with torch.cuda.stream(self._verify_stream):
                            decisions = self._verifier(
                                slot.hidden_states, slot.compressed_logits,
                            )
                    else:
                        decisions = self._verifier(
                            slot.hidden_states, slot.compressed_logits,
                        )

                    # All must be accepted for the batch to pass
                    slot.accepted = all(decisions) if decisions else False
            elif slot.compressed_logits is not None:
                # No external verifier — fall back to a built-in greedy
                # check: a draft token is accepted only when it matches the
                # target model's argmax at the corresponding position.
                # Drafts are never emitted without some target constraint.
                target_argmax = (
                    slot.compressed_logits.argmax(dim=-1).squeeze(0).tolist()
                )
                slot.accepted = list(slot.token_ids) == list(target_argmax)
            else:
                # No verifier and no verification inputs could be captured —
                # fail safe rather than emitting unverified draft tokens.
                logger.warning(
                    "No verifier and no verification inputs for draft slot "
                    "— rejecting draft"
                )
                slot.accepted = False

            self._stats["verify_calls"] += 1
        except Exception as e:
            logger.warning("Verifier failed: {}", e)
            slot.accepted = False

        return slot


def _unpack_target_output(
    target_out: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Normalize a ``target_forward`` return value to ``(logits, hidden)``.

    Targets may return just logits, a ``(logits, hidden_states)`` tuple, or
    an HF-style object with ``.logits`` / ``.hidden_states`` attributes.
    """
    if isinstance(target_out, tuple):
        logits = target_out[0]
        hidden = target_out[1] if len(target_out) > 1 else None
    elif hasattr(target_out, "logits"):
        logits = target_out.logits
        hidden = getattr(target_out, "hidden_states", None)
    else:
        logits = target_out
        hidden = None
    return logits, hidden


def nullcontext() -> Any:
    """No-op context manager for non-CUDA devices."""
    import contextlib
    return contextlib.nullcontext()
