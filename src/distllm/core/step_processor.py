"""Step processing for the continuous batch scheduler.

Extracted from ``BatchScheduler`` in ``batch_scheduler.py``.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from distllm.core.scheduler.sequence import SequenceStatus


class StepProcessor:
    """Step processing (static methods).

    Each method receives the scheduler instance as the first parameter
    so it can access all scheduler state exactly as the original
    ``self``-based method did.
    """

    @staticmethod
    def record_step_metrics(
        scheduler,
        batch,
        decode_count: int = 0,
        decode_elapsed_ms: float = 0.0,
    ) -> None:
        """Record decode metrics and feed data to adaptive engine."""
        if decode_count > 0 and decode_elapsed_ms > 0:
            scheduler._pressure_tracker.record_decode_step(decode_count, decode_elapsed_ms)

        if scheduler._adaptive_engine is not None:
            model = getattr(scheduler._model_info, 'model_name', None) or "default"
            now = time.time()
            seq_latencies = []
            for seq in batch.sequences:
                if seq.status != SequenceStatus.PENDING:
                    lat = (now - seq.created_at) * 1000
                    seq_latencies.append(lat)
            if seq_latencies:
                scheduler._adaptive_engine.record_batch(
                    model=model,
                    batch_size=len(batch.sequences),
                    latencies=seq_latencies,
                )

    @staticmethod
    def process_step(
        scheduler,
        batch,
        next_tokens: "torch.Tensor",
        kv_caches: dict | None = None,
        decoded_tokens: list[str] | None = None,
    ) -> None:
        """Process sampling output, update sequences, check for completion.

        Args:
            scheduler: The BatchScheduler instance.
            batch: The batch that was just processed.
            next_tokens: [batch_size] tensor of sampled token IDs.
            kv_caches: Optional dict mapping request_id -> KV cache data.
            decoded_tokens: Optional list of decoded token strings.
        """
        decode_count = sum(1 for s in batch.sequences if s.status == SequenceStatus.DECODING)
        decode_start = time.monotonic()

        for i, seq in enumerate(batch.sequences):
            token = next_tokens[i].item()
            seq.generated_tokens.append(int(token))
            scheduler._latency_tracker.record_token(seq.request_id)

            if seq.status == SequenceStatus.PREFILLING:
                seq.status = SequenceStatus.DECODING
                scheduler._latency_tracker.record_first_token(seq.request_id)

            if seq.constraint is not None:
                if decoded_tokens is not None and i < len(decoded_tokens):
                    seq.constraint.update(decoded_tokens[i])
                else:
                    seq.constraint.update(str(token))

            if seq.is_complete or token in seq.stop_token_ids:
                seq.status = SequenceStatus.DONE
                scheduler._latency_tracker.complete(seq.request_id)

                if scheduler._cache_mgr is not None and scheduler._cache_mgr.prefix_cache is not None:
                    all_tokens = seq.prompt_tokens + seq.generated_tokens
                    if len(all_tokens) >= scheduler._cache_mgr.prefix_cache.min_prefix_len:
                        kv_data = None
                        if kv_caches and seq.request_id in kv_caches:
                            kv_data = kv_caches[seq.request_id]
                        if kv_data is not None:
                            scheduler._cache_mgr.store_prefix(all_tokens, kv_data)

        StepProcessor.record_step_metrics(
            scheduler,
            batch,
            decode_count=decode_count,
            decode_elapsed_ms=(time.monotonic() - decode_start) * 1000 if decode_count > 0 else 0.0,
        )

        with scheduler._lock:
            done_rids = [s.request_id for s in batch.sequences if s.is_complete]
            for rid in done_rids:
                scheduler.active.pop(rid, None)
                scheduler._chunked_prefill.pop(rid, None)

        if scheduler._energy_scheduler is not None:
            scheduler._energy_scheduler.record_energy_usage(
                duration_seconds=(time.monotonic() - decode_start) if decode_count > 0 else 0.0,
            )
