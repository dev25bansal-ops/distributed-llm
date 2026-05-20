"""Speculative method profiler for auto-speculative selection.

Profiles acceptance rates per speculative method (ngram, medusa, eagle,
draft_model, tree_draft) and selects the best method per workload type.
"""

from __future__ import annotations

import time
from dataclasses import dataclass



@dataclass
class MethodProfile:
    """Performance profile for a speculative method."""
    method: str
    total_drafts: int = 0
    total_accepted: int = 0
    acceptance_rate: float = 1.0
    avg_speedup: float = 1.0
    tokens_per_sec: float = 0.0
    total_generation_time_ms: float = 0.0
    samples: int = 0
    last_updated: float = 0.0

    def update(self, drafts: int, accepted: int, generation_time_ms: float) -> None:
        """Update profile with a new observation."""
        alpha = 0.1  # EMA decay
        self.total_drafts += drafts
        self.total_accepted += accepted
        self.samples += 1
        self.total_generation_time_ms += generation_time_ms

        # EMA acceptance rate
        observed_rate = accepted / max(drafts, 1)
        self.acceptance_rate = alpha * observed_rate + (1 - alpha) * self.acceptance_rate

        # Speedup estimate: accepted drafts / total steps
        speedup = (accepted + 1) / max(1, 1)  # +1 for the target token
        self.avg_speedup = alpha * speedup + (1 - alpha) * self.avg_speedup

        # Throughput
        if generation_time_ms > 0:
            tps = accepted / (generation_time_ms / 1000)
            if self.tokens_per_sec == 0:
                self.tokens_per_sec = tps
            else:
                self.tokens_per_sec = alpha * tps + (1 - alpha) * self.tokens_per_sec

        self.last_updated = time.time()


class SpeculativeProfiler:
    """Profiles acceptance rates and performance per speculative method.

    Uses EMA (Exponential Moving Average) for smooth rate tracking.
    Supports workload-type-aware method selection.
    """

    def __init__(self, alpha: float = 0.1, warmup_samples: int = 10) -> None:
        self.alpha = alpha
        self.warmup_samples = warmup_samples
        self._profiles: dict[str, dict[str, MethodProfile]] = {}  # workload_type -> method -> profile

    def record_acceptance(
        self,
        method: str,
        workload_type: str,
        draft_count: int,
        accepted_count: int,
        generation_time_ms: float = 0.0,
    ) -> None:
        """Record a speculative decoding observation.

        Args:
            method: Speculative method used (ngram, medusa, eagle, draft_model, tree_draft).
            workload_type: Classified workload type (code, repetitive, diverse, instruction, unknown).
            draft_count: Number of draft tokens generated.
            accepted_count: Number of draft tokens accepted by target model.
            generation_time_ms: Time taken for draft generation + verification.
        """
        if workload_type not in self._profiles:
            self._profiles[workload_type] = {}

        if method not in self._profiles[workload_type]:
            self._profiles[workload_type][method] = MethodProfile(method=method)

        profile = self._profiles[workload_type][method]
        profile.update(draft_count, accepted_count, generation_time_ms)

    def get_best_method(self, workload_type: str, min_samples: int | None = None) -> str:
        """Return the method with highest expected speedup for a workload type.

        Falls back to 'ngram' if no profiles exist (ngram is always available).
        """
        profiles = self._profiles.get(workload_type, {})
        if not profiles:
            return "ngram"

        min_s = min_samples or self.warmup_samples

        # Filter methods with enough samples
        eligible = {m: p for m, p in profiles.items() if p.samples >= min_s}
        if not eligible:
            # Not enough data, use priors
            return self._get_prior_method(workload_type)

        # Select by expected throughput (acceptance_rate * speedup)
        best_method = max(eligible.keys(), key=lambda m: eligible[m].acceptance_rate * eligible[m].avg_speedup)
        return best_method

    def get_profile(self, method: str, workload_type: str) -> MethodProfile:
        """Get the profile for a specific method and workload type."""
        return self._profiles.get(workload_type, {}).get(
            method,
            MethodProfile(method=method),
        )

    def get_all_profiles(self, workload_type: str | None = None) -> dict[str, dict[str, MethodProfile]]:
        """Get all profiles, optionally filtered by workload type."""
        if workload_type:
            return {workload_type: self._profiles.get(workload_type, {})}
        return dict(self._profiles)

    def get_method_ranking(self, workload_type: str) -> list[tuple[str, float]]:
        """Return methods ranked by score (acceptance_rate * avg_speedup)."""
        profiles = self._profiles.get(workload_type, {})
        scores = [(m, p.acceptance_rate * p.avg_speedup) for m, p in profiles.items()]
        return sorted(scores, key=lambda x: x[1], reverse=True)

    def _get_prior_method(self, workload_type: str) -> str:
        """Return the best prior method for a workload type (before profiling data)."""
        priors = {
            "code": "ngram",
            "repetitive": "ngram",
            "diverse": "draft_model",
            "instruction": "eagle",
            "unknown": "ngram",
        }
        return priors.get(workload_type, "ngram")
