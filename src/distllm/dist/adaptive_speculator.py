"""Adaptive speculative decoding — dynamically selects draft model and speculation depth.

Integrates workload classification (:mod:`distllm.dist.scheduling.classifier`),
draft model selection (:class:`distllm.dist.draft_bank.FederatedDraftBank`),
and WAN-optimized speculative decoding (:class:`distllm.dist.wan_speculative.WANSpeculativeDecoder`)
into a single feedback-driven pipeline.

Flow for each request::

    1. classify(prompt)                    → WorkloadType
    2. select_draft(workload, features)    → (draft_model, num_candidates)
    3. speculative_decode(input_ids, ...)   → output tokens, acceptance_rate
    4. record(workload, draft_id, rate)    → update per-class statistics
    5. adapt(workload)                     → tune num_candidates for next call

Key capability — **per-class acceptance tracking**: speculative performance
differs dramatically between code generation (high acceptance, predictable)
and creative writing (low acceptance, diverse).  The adaptive speculator
learns separate profiles for each workload class and adjusts speculation
depth accordingly, preventing wasted computation on low-acceptance prompts.
"""

from __future__ import annotations

import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from loguru import logger

from distllm.dist.scheduling.classifier import (
    WorkloadType,
    classify as classify,
    classify_features,
)
from distllm.dist.wan_speculative import WANSpeculativeDecoder, WANSpeculativeConfig


@dataclass
class AcceptanceProfile:
    """Per-class, per-draft-model acceptance statistics.

    Updated continuously during inference via exponential moving average
    so that recent observations are weighted more heavily than stale ones.
    """
    # Exponential moving average of acceptance rate (0.0 – 1.0).
    ema_acceptance: float = 0.0
    # Number of observations used to compute the current EMA.
    observation_count: int = 0
    # Total tokens speculated (denominator for overall rate).
    total_speculated: int = 0
    # Total tokens accepted (numerator for overall rate).
    total_accepted: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.total_accepted / max(self.total_speculated, 1)

    def update(self, rate: float, alpha: float = 0.3) -> None:
        """Update EMA and raw counts with a new observation."""
        self.observation_count += 1
        if self.observation_count == 1:
            self.ema_acceptance = rate
        else:
            self.ema_acceptance = (
                (1 - alpha) * self.ema_acceptance + alpha * rate
            )
        self.total_speculated += 1  # one batch observation


@dataclass
class AdaptiveSpeculatorConfig:
    """Configuration for adaptive speculative decoding.

    Attributes:
        max_candidates: Maximum speculation depth (tokens per WAN RTT).
        min_candidates: Minimum speculation depth.
        target_acceptance: Acceptance rate above which we increase candidates.
        low_acceptance_threshold: Below this, we aggressively reduce candidates.
        ema_alpha: Smoothing factor for acceptance rate EMA.
        profile_cooldown_s: Minimum seconds between profile-based adaptations
            to prevent thrashing on rapidly changing workloads.
        warmup_observations: Number of observations before trusting a profile.
    """
    max_candidates: int = 16
    min_candidates: int = 1
    target_acceptance: float = 0.7
    low_acceptance_threshold: float = 0.3
    ema_alpha: float = 0.3
    profile_cooldown_s: float = 5.0
    warmup_observations: int = 5


class AdaptiveSpeculator:
    """Adaptive speculative decoding with per-class acceptance tracking.

    Usage::

        speculator = AdaptiveSpeculator(
            target_forward=remote_pipeline.run,
            draft_bank=federated_draft_bank,
            config=AdaptiveSpeculatorConfig(),
        )

        # Per-request:
        tokens = await speculator.generate(input_ids, prompt_text=prompt)
        print(speculator.get_stats())
    """

    def __init__(
        self,
        target_forward: Callable,
        draft_bank: Any | None = None,
        config: AdaptiveSpeculatorConfig | None = None,
    ):
        self._target_forward = target_forward
        self._draft_bank = draft_bank
        self._config = config or AdaptiveSpeculatorConfig()

        # Per-class acceptance profiles, keyed by WorkloadType value.
        self._profiles: dict[str, AcceptanceProfile] = defaultdict(AcceptanceProfile)
        # Cross-class aggregate profile for fallback.
        self._global_profile = AcceptanceProfile()
        self._lock = threading.Lock()

        # Per-class candidate counts, dynamically tuned.
        self._candidate_counts: dict[str, int] = defaultdict(
            lambda: max(self._config.min_candidates, 4)
        )

        # Cooldown tracking per class to prevent thrashing.
        self._last_adapt: dict[str, float] = defaultdict(float)

        # Latency tracking for SLA reporting.
        self._latency_records: list[float] = []

        logger.info(
            f"AdaptiveSpeculator initialized: max_candidates="
            f"{self._config.max_candidates}, target_acceptance="
            f"{self._config.target_acceptance}"
        )

    # ── Public API ────────────────────────────────────────────────────

    async def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        prompt_text: str | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate tokens with adaptive speculative decoding.

        Args:
            input_ids: Prompt token IDs, shape ``(1, seq_len)``.
            max_new_tokens: Maximum tokens to generate.
            prompt_text: Raw prompt text for workload classification.
                When ``None``, skips classification and uses the
                ``UNKNOWN`` profile.
            **kwargs: Forwarded to the speculative decoder.

        Returns:
            Generated token IDs, shape ``(1, prompt_len + generated)``.
        """
        # Phase 1 — classify workload
        workload = WorkloadType.UNKNOWN
        features: dict[str, float] = {}
        if prompt_text:
            workload = classify(prompt_text)
            features = classify_features(prompt_text)
            logger.debug(f"Classified workload: {workload.value} "
                         f"(entropy={features.get('entropy_3gram', 0):.1f})")

        workload_key = workload.value

        # Phase 2 — select draft model and candidate count
        draft_forward, draft_id = self._select_draft(workload_key, features)
        num_candidates = self._candidate_counts[workload_key]
        logger.debug(
            f"Speculating: workload={workload_key}, "
            f"candidates={num_candidates}, draft={draft_id or 'embedded'}"
        )

        # Phase 3 — configure and run speculative decoder
        wan_config = WANSpeculativeConfig(
            num_candidates=num_candidates,
            adaptive_candidates=False,  # we manage candidates externally
            max_speculation_depth=self._config.max_candidates,
        )
        decoder = WANSpeculativeDecoder(
            target_forward=self._target_forward,
            draft_forward=draft_forward or self._default_draft,
            num_candidates=num_candidates,
            max_speculation_depth=self._config.max_candidates,
            **self._decoder_kwargs(kwargs),
        )

        t0 = time.monotonic()
        output = await decoder.generate(input_ids, max_new_tokens, **kwargs)
        elapsed_s = time.monotonic() - t0

        # Phase 4 — record outcome
        decoder_stats = decoder.stats
        acceptance_rate = decoder_stats.get("acceptance_rate", 0.0)
        self._record_outcome(
            workload_key, draft_id, acceptance_rate,
        )

        # Phase 5 — adapt candidate count for next call
        self._adapt_candidates(workload_key, acceptance_rate)

        with self._lock:
            self._latency_records.append(elapsed_s * 1000)

        return output

    def get_acceptance_rate(self, workload: WorkloadType | None = None) -> float:
        """Return EMA acceptance rate for a specific workload or global."""
        if workload is not None:
            return self._profiles[workload.value].ema_acceptance
        return self._global_profile.ema_acceptance

    def get_candidate_count(self, workload: WorkloadType | None = None) -> int:
        """Return current candidate count for a workload or global default."""
        key = workload.value if workload else "unknown"
        with self._lock:
            return self._candidate_counts.get(key, self._config.min_candidates)

    def get_stats(self) -> dict[str, Any]:
        """Return comprehensive stats for observability."""
        with self._lock:
            profiles = {
                k: {
                    "ema_acceptance": round(v.ema_acceptance, 3),
                    "observations": v.observation_count,
                    "candidate_count": self._candidate_counts.get(k, 0),
                }
                for k, v in self._profiles.items()
                if v.observation_count > 0
            }
            avg_latency = (
                sum(self._latency_records) / max(len(self._latency_records), 1)
            )
            return {
                "profiles": profiles,
                "global_acceptance": round(self._global_profile.ema_acceptance, 3),
                "global_observations": self._global_profile.observation_count,
                "avg_latency_ms": round(avg_latency, 1),
                "total_requests": len(self._latency_records),
            }

    # ── Internal: draft selection ─────────────────────────────────────

    def _select_draft(
        self,
        workload_key: str,
        features: dict[str, float] | None,
    ) -> tuple[Callable | None, str | None]:
        """Select the best draft model for *workload_key*.

        Returns ``(draft_forward_fn, draft_model_id)``.  When the draft
        bank has no suitable endpoint or is not configured, returns
        ``(None, None)`` — the caller falls back to the embedded default.
        """
        if self._draft_bank is None:
            return None, None

        try:
            endpoint = self._draft_bank.get_best_draft_endpoint(
                workload_type=workload_key,
                max_latency_ms=200.0,
            )
            if endpoint is None:
                return None, None

            # Build an async forwarder that hits the remote draft endpoint.
            import httpx

            async def _remote_draft(
                input_ids: torch.Tensor,
                **kw: Any,
            ) -> torch.Tensor:
                """Forward through the remote draft model."""
                payload = {
                    "input_ids": input_ids.cpu().tolist(),
                    **{k: v for k, v in kw.items()
                       if k in ("temperature", "top_k", "max_tokens")},
                }
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.post(
                        endpoint.endpoint_url,
                        json=payload,
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    return torch.tensor(result["tokens"])

            return _remote_draft, endpoint.cluster_id
        except Exception as e:
            logger.debug(f"Draft selection failed, using embedded default: {e}")
            return None, None

    async def _default_draft(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Minimal embedded draft model (repeats last token).

        This is a zero-parameter fallback that always produces
        *some* speculation, even when no draft model is available.
        It will have a very low acceptance rate, which causes the
        adaptive layer to reduce candidate count toward ``min_candidates``.
        """
        last_token = input_ids[:, -1:]
        num_tokens = kwargs.get("num_tokens", 1)
        return last_token.repeat(1, num_tokens)

    # ── Internal: recording and adaptation ─────────────────────────────

    def _record_outcome(
        self,
        workload_key: str,
        draft_id: str | None,
        acceptance_rate: float,
    ) -> None:
        """Update per-class and global acceptance profiles."""
        with self._lock:
            profile = self._profiles[workload_key]
            profile.update(acceptance_rate, alpha=self._config.ema_alpha)
            self._global_profile.update(acceptance_rate, alpha=self._config.ema_alpha)

    def _adapt_candidates(self, workload_key: str, acceptance_rate: float) -> None:
        """Dynamically tune candidate count based on observed acceptance.

        High acceptance → speculate further (more candidates per WAN RTT)
        Low acceptance → reduce speculation to avoid wasted computation
        """
        now = time.time()
        with self._lock:
            # Cooldown: don't adapt more frequently than once per *cooldown_s*.
            if now - self._last_adapt[workload_key] < self._config.profile_cooldown_s:
                return
            self._last_adapt[workload_key] = now

            profile = self._profiles[workload_key]
            current = self._candidate_counts[workload_key]

            # Wait for enough observations before trusting the profile.
            if profile.observation_count < self._config.warmup_observations:
                return

            ema = profile.ema_acceptance

            if ema >= self._config.target_acceptance:
                # High acceptance: increase candidates (but don't exceed max).
                new_count = min(current * 2, self._config.max_candidates)
            elif ema <= self._config.low_acceptance_threshold:
                # Low acceptance: aggressively reduce.
                new_count = max(
                    current // 2,
                    self._config.min_candidates,
                )
            else:
                # Moderate: fine-tune by ±1.
                if acceptance_rate > ema:
                    new_count = min(current + 1, self._config.max_candidates)
                else:
                    new_count = max(current - 1, self._config.min_candidates)

            if new_count != current:
                self._candidate_counts[workload_key] = new_count
                logger.debug(
                    f"Adapted candidates for {workload_key}: "
                    f"{current} → {new_count} "
                    f"(ema_acceptance={ema:.2f})"
                )

    @staticmethod
    def _decoder_kwargs(kwargs: dict) -> dict:
        """Extract decoder-relevant kwargs, excluding known engine params."""
        return {
            k: v for k, v in kwargs.items()
            if k not in ("prompt_text",)
        }
