"""Speculative Decoding Profiler — tracks acceptance rate per method/workload.

Records acceptance statistics for each (speculative_method, workload_type)
pair and selects the best method based on historical performance.

Usage::

    profiler = SpeculativeProfiler(warmup_samples=5)
    profiler.record_acceptance("ngram", "code", drafted=5, accepted=4)
    profiler.record_acceptance("eagle", "code", drafted=5, accepted=2)

    best = profiler.get_best_method("code")  # "ngram"
    ranking = profiler.get_method_ranking("code")  # [("ngram", 0.8), ("eagle", 0.4)]
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MethodProfile:
    """Acceptance statistics for a single (method, workload) pair."""
    method: str = ""
    workload_type: str = ""
    total_drafts: int = 0
    total_accepted: int = 0
    total_latency_s: float = 0.0
    samples: int = 0

    @property
    def acceptance_rate(self) -> float:
        if self.total_drafts == 0:
            return 0.0
        return self.total_accepted / self.total_drafts

    @property
    def avg_latency_ms(self) -> float:
        if self.samples == 0:
            return 0.0
        return (self.total_latency_s / self.samples) * 1000


# Prior acceptance rates by workload type (used when no data exists)
_PRIOR_RATES: dict[str, dict[str, float]] = {
    "code": {"ngram": 0.65, "eagle": 0.50, "draft_model": 0.55, "medusa": 0.45},
    "instruction": {"eagle": 0.55, "ngram": 0.40, "draft_model": 0.50, "medusa": 0.45},
    "repetitive": {"ngram": 0.80, "eagle": 0.45, "draft_model": 0.50, "medusa": 0.40},
    "diverse": {"eagle": 0.50, "draft_model": 0.45, "ngram": 0.30, "medusa": 0.40},
    "unknown": {"ngram": 0.50, "eagle": 0.50, "draft_model": 0.50, "medusa": 0.45},
}


class SpeculativeProfiler:
    """Tracks acceptance rates per (method, workload_type) pair.

    After collecting enough samples (``warmup_samples``), the profiler
    uses empirical data to rank methods.  During warmup, it falls back
    to prior rates based on workload type.

    Args:
        warmup_samples: Minimum samples before empirical data is trusted.
    """

    def __init__(self, warmup_samples: int = 5) -> None:
        self._warmup = warmup_samples
        self._profiles: dict[tuple[str, str], MethodProfile] = {}

    def record_acceptance(
        self,
        method: str,
        workload_type: str,
        drafted: int,
        accepted: int,
        latency_ms: float = 0.0,
    ) -> None:
        """Record acceptance data for one speculative decoding step."""
        key = (method, workload_type)
        if key not in self._profiles:
            self._profiles[key] = MethodProfile(
                method=method, workload_type=workload_type,
            )
        profile = self._profiles[key]
        profile.total_drafts += drafted
        profile.total_accepted += accepted
        profile.total_latency_s += latency_ms / 1000.0
        profile.samples += 1

    def get_profile(self, method: str, workload_type: str) -> MethodProfile:
        """Get the profile for a specific (method, workload) pair."""
        key = (method, workload_type)
        if key in self._profiles:
            return self._profiles[key]
        return MethodProfile(method=method, workload_type=workload_type)

    def get_best_method(self, workload_type: str) -> str:
        """Return the method with the highest acceptance rate for this workload.

        During warmup (fewer than ``warmup_samples`` total), returns
        the prior-best method for the workload type.
        """
        ranking = self.get_method_ranking(workload_type)
        if ranking:
            return ranking[0][0]
        return "ngram"

    def get_method_ranking(self, workload_type: str) -> list[tuple[str, float]]:
        """Rank all methods by acceptance rate for a workload type.

        Returns list of ``(method, acceptance_rate)`` sorted descending.
        """
        # Collect empirical data
        empirical: dict[str, float] = {}
        total_samples = 0
        for (method, wt), profile in self._profiles.items():
            if wt == workload_type and profile.samples > 0:
                empirical[method] = profile.acceptance_rate
                total_samples += profile.samples

        # Use empirical if enough samples, otherwise blend with priors
        if total_samples >= self._warmup:
            scores = empirical
        else:
            priors = _PRIOR_RATES.get(workload_type, _PRIOR_RATES["unknown"])
            scores = dict(priors)
            scores.update(empirical)

        ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranking

    def get_all_profiles(self) -> dict[tuple[str, str], MethodProfile]:
        return dict(self._profiles)
