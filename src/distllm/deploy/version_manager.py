"""Model versioning, shadow mode, A/B testing, and blue-green deployment."""

import hashlib
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from loguru import logger


class VersionStatus(str, Enum):
    ACTIVE = "active"
    SHADOW = "shadow"
    BLUE = "blue"
    GREEN = "green"
    ARCHIVED = "archived"
    FAILED = "failed"


@dataclass
class VersionMetrics:
    """Tracking metrics for a model version."""
    total_requests: int = 0
    total_errors: int = 0
    latencies: list[float] = field(default_factory=list)
    prompt_tokens: list[int] = field(default_factory=list)
    completion_tokens: list[int] = field(default_factory=list)
    # Shadow mode: stores outputs for comparison without returning
    shadow_outputs: list[dict] = field(default_factory=list)
    # A/B test: stores user feedback scores
    feedback_scores: list[float] = field(default_factory=list)

    @property
    def error_rate(self) -> float:
        return self.total_errors / self.total_requests if self.total_requests > 0 else 0.0

    @property
    def avg_latency(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0

    @property
    def p50_latency(self) -> float:
        if not self.latencies:
            return 0.0
        s = sorted(self.latencies)
        return s[len(s) // 2]

    @property
    def p99_latency(self) -> float:
        if not self.latencies:
            return 0.0
        s = sorted(self.latencies)
        return s[int(len(s) * 0.99)]

    @property
    def avg_prompt_tokens(self) -> float:
        return sum(self.prompt_tokens) / len(self.prompt_tokens) if self.prompt_tokens else 0.0

    @property
    def avg_completion_tokens(self) -> float:
        return sum(self.completion_tokens) / len(self.completion_tokens) if self.completion_tokens else 0.0


@dataclass
class ModelVersion:
    """Represents a specific version of a model."""
    version_id: str
    model_id: str
    model_path: str
    status: VersionStatus = VersionStatus.ACTIVE
    created_at: float = field(default_factory=time.time)
    promoted_at: float | None = None
    traffic_weight: float = 100.0  # 0-100
    metrics: VersionMetrics = field(default_factory=VersionMetrics)
    metadata: dict[str, str] = field(default_factory=dict)  # Tags, commit hash, etc.


class StatisticalAnalyzer:
    """Statistical tests for comparing model versions."""

    @staticmethod
    def mann_whitney_u(sample_a: list[float], sample_b: list[float]) -> tuple[float, float]:
        """Mann-Whitney U test for comparing two independent samples.

        Non-parametric alternative to t-test. Tests whether one sample
        tends to have larger values than the other.

        Returns:
            (U statistic, p-value)
        """
        if len(sample_a) < 2 or len(sample_b) < 2:
            return (0.0, 1.0)

        # Combine and rank
        combined = [(x, 0) for x in sample_a] + [(x, 1) for x in sample_b]
        combined.sort(key=lambda x: x[0])

        # Assign ranks (handle ties with average rank)
        ranks = []
        i = 0
        while i < len(combined):
            j = i
            while j < len(combined) and combined[j][0] == combined[i][0]:
                j += 1
            avg_rank = (i + j + 1) / 2.0
            for k in range(i, j):
                ranks.append((combined[k][1], avg_rank))
            i = j

        # Sum ranks for each group
        r1 = sum(r for group, r in ranks if group == 0)
        n1, n2 = len(sample_a), len(sample_b)

        u1 = r1 - n1 * (n1 + 1) / 2.0
        u2 = n1 * n2 - u1
        u = min(u1, u2)

        # Normal approximation for p-value
        mu = n1 * n2 / 2.0
        sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
        if sigma == 0:
            return (u, 1.0)

        z = (u - mu) / sigma
        p_value = 2 * (1 - StatisticalAnalyzer._norm_cdf(abs(z)))

        return (u, p_value)

    @staticmethod
    def t_test_ind(sample_a: list[float], sample_b: list[float]) -> tuple[float, float]:
        """Independent two-sample t-test (Welch's, unequal variances).

        Returns:
            (t statistic, p-value)
        """
        n1, n2 = len(sample_a), len(sample_b)
        if n1 < 2 or n2 < 2:
            return (0.0, 1.0)

        mean1 = sum(sample_a) / n1
        mean2 = sum(sample_b) / n2

        var1 = sum((x - mean1) ** 2 for x in sample_a) / (n1 - 1)
        var2 = sum((x - mean2) ** 2 for x in sample_b) / (n2 - 1)

        se = math.sqrt(var1 / n1 + var2 / n2)
        if se == 0:
            return (0.0, 1.0)

        t = (mean1 - mean2) / se

        # Welch-Satterthwaite degrees of freedom
        num = (var1 / n1 + var2 / n2) ** 2
        den = (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
        df = num / den if den > 0 else 1

        p_value = 2 * (1 - StatisticalAnalyzer._t_cdf(abs(t), df))

        return (t, p_value)

    @staticmethod
    def compare_versions(
        metrics_a: VersionMetrics,
        metrics_b: VersionMetrics,
        significance_level: float = 0.05,
        min_samples: int = 30,
    ) -> dict:
        """Compare two versions using statistical tests.

        Returns:
            Dict with test results and recommendation.
        """
        result = {
            "sample_a": metrics_a.total_requests,
            "sample_b": metrics_b.total_requests,
            "sufficient_samples": metrics_a.total_requests >= min_samples and metrics_b.total_requests >= min_samples,
            "error_rate_a": metrics_a.error_rate,
            "error_rate_b": metrics_b.error_rate,
            "avg_latency_a": metrics_a.avg_latency,
            "avg_latency_b": metrics_b.avg_latency,
            "p50_latency_a": metrics_a.p50_latency,
            "p50_latency_b": metrics_b.p50_latency,
            "p99_latency_a": metrics_a.p99_latency,
            "p99_latency_b": metrics_b.p99_latency,
        }

        if not result["sufficient_samples"]:
            result["recommendation"] = "insufficient_data"
            result["reason"] = f"Need >= {min_samples} samples per version"
            return result

        # Compare latencies
        lat_a = metrics_a.latencies[-1000:]  # Use last 1000 for efficiency
        lat_b = metrics_b.latencies[-1000:]

        u_stat, u_p = StatisticalAnalyzer.mann_whitney_u(lat_a, lat_b)
        t_stat, t_p = StatisticalAnalyzer.t_test_ind(lat_a, lat_b)

        result["mann_whitney_u"] = round(u_stat, 4)
        result["mann_whitney_p"] = round(u_p, 6)
        result["t_statistic"] = round(t_stat, 4)
        result["t_p_value"] = round(t_p, 6)

        # Compare error rates
        error_diff = metrics_b.error_rate - metrics_a.error_rate
        result["error_rate_diff"] = round(error_diff, 6)

        # Recommendation
        latency_better = metrics_b.avg_latency < metrics_a.avg_latency
        error_better = metrics_b.error_rate <= metrics_a.error_rate
        significant = t_p < significance_level

        if error_better and (latency_better or not significant):
            result["recommendation"] = "promote"
            result["reason"] = "Candidate has equal or better error rate and no significant latency regression"
        elif not error_better:
            result["recommendation"] = "reject"
            result["reason"] = "Candidate has higher error rate"
        else:
            result["recommendation"] = "inconclusive"
            result["reason"] = "No significant difference detected"

        return result

    @staticmethod
    def _norm_cdf(x: float) -> float:
        """Standard normal CDF approximation."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    @staticmethod
    def _t_cdf(t: float, df: float) -> float:
        """Student's t CDF approximation using normal for large df."""
        if df > 30:
            return StatisticalAnalyzer._norm_cdf(t)
        # Use incomplete beta approximation
        x = df / (df + t * t)
        return 1 - 0.5 * StatisticalAnalyzer._inc_beta(df / 2, 0.5, x)

    @staticmethod
    def _inc_beta(a: float, b: float, x: float, max_iter: int = 200) -> float:
        """Incomplete beta function via continued fraction."""
        if x == 0 or x == 1:
            return float(x)
        # Use series expansion
        result = 1.0
        term = 1.0
        for n in range(1, max_iter):
            term *= (a + n - 1) * x / (a + b + n - 1)
            result += term
            if abs(term) < 1e-10:
                break
        return min(result * (x ** a) / a, 1.0)


class VersionManager:
    """Manages model versions, shadow mode, A/B testing, and blue-green deployment."""

    def __init__(
        self,
        max_versions: int = 4,
        shadow_enabled: bool = False,
        shadow_pct: float = 0.0,
        blue_green_enabled: bool = False,
        ab_testing_enabled: bool = False,
        ab_test_split: float = 50.0,
        auto_promote_enabled: bool = False,
        min_samples: int = 100,
        significance_level: float = 0.05,
    ):
        self.max_versions = max_versions
        self.versions: dict[str, dict[str, ModelVersion]] = {}  # model_id -> {version_id -> Version}
        self.analyzer = StatisticalAnalyzer()

        # Shadow mode
        self.shadow_enabled = shadow_enabled
        self.shadow_pct = shadow_pct
        self._shadow_log: list[dict] = []

        # Blue-green
        self.blue_green_enabled = blue_green_enabled
        self._blue_version: str | None = None
        self._green_version: str | None = None
        self._active_color: str = "blue"

        # A/B testing
        self.ab_testing_enabled = ab_testing_enabled
        self.ab_test_split = ab_test_split

        # Auto-promotion
        self.auto_promote_enabled = auto_promote_enabled
        self.min_samples = min_samples
        self.significance_level = significance_level

    # -- Version CRUD --

    def register_version(
        self,
        model_id: str,
        version_id: str,
        model_path: str,
        metadata: dict[str, str] | None = None,
    ) -> ModelVersion:
        """Register a new model version."""
        if model_id not in self.versions:
            self.versions[model_id] = {}

        existing = self.versions[model_id]
        if len(existing) >= self.max_versions:
            # Archive oldest
            oldest_id = min(existing, key=lambda vid: existing[vid].created_at)
            existing[oldest_id].status = VersionStatus.ARCHIVED
            del existing[oldest_id]

        version = ModelVersion(
            version_id=version_id,
            model_id=model_id,
            model_path=model_path,
            metadata=metadata or {},
        )
        existing[version_id] = version
        logger.info(f"[VersionManager] Registered {model_id}/{version_id}")
        return version

    def get_version(self, model_id: str, version_id: str) -> ModelVersion | None:
        return self.versions.get(model_id, {}).get(version_id)

    def list_versions(self, model_id: str) -> list[ModelVersion]:
        return list(self.versions.get(model_id, {}).values())

    def delete_version(self, model_id: str, version_id: str) -> bool:
        if model_id in self.versions and version_id in self.versions[model_id]:
            del self.versions[model_id][version_id]
            return True
        return False

    # -- Shadow mode --

    def is_shadow_request(self, request_id: str) -> bool:
        """Determine if this request should go to shadow version."""
        if not self.shadow_enabled or self.shadow_pct <= 0:
            return False
        hash_val = int(hashlib.md5(request_id.encode()).hexdigest(), 16) % 100
        return hash_val < self.shadow_pct

    def log_shadow_output(
        self,
        model_id: str,
        stable_version: str,
        shadow_version: str,
        request_id: str,
        stable_output: str,
        shadow_output: str,
        latency_stable: float,
        latency_shadow: float,
    ) -> None:
        """Log shadow mode comparison (does not affect user response)."""
        entry = {
            "model_id": model_id,
            "stable_version": stable_version,
            "shadow_version": shadow_version,
            "request_id": request_id,
            "stable_output": stable_output,
            "shadow_output": shadow_output,
            "latency_stable": latency_stable,
            "latency_shadow": latency_shadow,
            "timestamp": time.time(),
        }
        self._shadow_log.append(entry)

        # Update metrics
        stable_v = self.get_version(model_id, stable_version)
        shadow_v = self.get_version(model_id, shadow_version)
        if stable_v:
            stable_v.metrics.shadow_outputs.append({"output": stable_output, "latency": latency_stable})
        if shadow_v:
            shadow_v.metrics.shadow_outputs.append({"output": shadow_output, "latency": latency_shadow})

    def get_shadow_comparisons(self, model_id: str, limit: int = 100) -> list[dict]:
        """Get recent shadow comparison results."""
        entries = [e for e in self._shadow_log if e["model_id"] == model_id]
        return entries[-limit:]

    # -- Blue-green deployment --

    def set_blue_green(
        self,
        model_id: str,
        blue_version: str,
        green_version: str,
        active_color: str = "blue",
    ) -> None:
        """Configure blue-green deployment for a model."""
        self._blue_version = blue_version
        self._green_version = green_version
        self._active_color = active_color

        blue_v = self.get_version(model_id, blue_version)
        green_v = self.get_version(model_id, green_version)
        if blue_v:
            blue_v.status = VersionStatus.BLUE
            blue_v.traffic_weight = 100.0 if active_color == "blue" else 0.0
        if green_v:
            green_v.status = VersionStatus.GREEN
            green_v.traffic_weight = 100.0 if active_color == "green" else 0.0

        self.blue_green_enabled = True

    def switch_color(self, model_id: str) -> str:
        """Instant switch between blue and green."""
        if self._active_color == "blue":
            self._active_color = "green"
        else:
            self._active_color = "blue"

        # Update traffic weights
        blue_v = self.get_version(model_id, self._blue_version) if self._blue_version else None
        green_v = self.get_version(model_id, self._green_version) if self._green_version else None
        if blue_v:
            blue_v.traffic_weight = 100.0 if self._active_color == "blue" else 0.0
        if green_v:
            green_v.traffic_weight = 100.0 if self._active_color == "green" else 0.0

        logger.info(f"[BlueGreen] Switched to {self._active_color}")
        return self._active_color

    def rollback_color(self, model_id: str) -> str:
        """Instant rollback to the other color."""
        return self.switch_color(model_id)

    def get_active_version(self, model_id: str, request_id: str) -> Optional[str]:
        """Determine which version should handle a request."""
        versions = self.versions.get(model_id, {})
        if not versions:
            return None

        # Blue-green takes priority
        if self.blue_green_enabled and self._blue_version and self._green_version:
            return self._blue_version if self._active_color == "blue" else self._green_version

        # A/B testing
        if self.ab_testing_enabled:
            hash_val = int(hashlib.md5(request_id.encode()).hexdigest(), 16) % 100
            active_list = [v for v in versions.values() if v.status == VersionStatus.ACTIVE]
            if len(active_list) >= 2:
                if hash_val < self.ab_test_split:
                    return active_list[1].version_id  # Variant B
                return active_list[0].version_id  # Variant A

        # Default: return active version with highest weight
        active_versions = [v for v in versions.values() if v.status == VersionStatus.ACTIVE]
        if active_versions:
            return max(active_versions, key=lambda v: v.traffic_weight).version_id

        return None

    # -- Metrics recording --

    def record_request(
        self,
        model_id: str,
        version_id: str,
        latency_ms: float,
        was_error: bool = False,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        feedback_score: float | None = None,
    ) -> None:
        """Record a request's metrics for the given version."""
        version = self.get_version(model_id, version_id)
        if version is None:
            return

        version.metrics.total_requests += 1
        if was_error:
            version.metrics.total_errors += 1
        version.metrics.latencies.append(latency_ms)
        if prompt_tokens:
            version.metrics.prompt_tokens.append(prompt_tokens)
        if completion_tokens:
            version.metrics.completion_tokens.append(completion_tokens)
        if feedback_score is not None:
            version.metrics.feedback_scores.append(feedback_score)

    # -- Auto-promotion --

    def evaluate_promotion(
        self,
        model_id: str,
        stable_version_id: str,
        candidate_version_id: str,
    ) -> dict:
        """Evaluate whether candidate should replace stable based on metrics."""
        stable = self.get_version(model_id, stable_version_id)
        candidate = self.get_version(model_id, candidate_version_id)

        if stable is None or candidate is None:
            return {"error": "Version not found", "recommendation": "error"}

        comparison = self.analyzer.compare_versions(
            stable.metrics,
            candidate.metrics,
            significance_level=self.significance_level,
            min_samples=self.min_samples,
        )

        return comparison

    def promote_version(self, model_id: str, version_id: str) -> bool:
        """Promote a version to be the primary active version."""
        versions = self.versions.get(model_id, {})
        if version_id not in versions:
            return False

        # Demote all other active versions
        for vid, v in versions.items():
            if v.status == VersionStatus.ACTIVE:
                v.status = VersionStatus.ARCHIVED
                v.traffic_weight = 0.0

        # Promote target version
        versions[version_id].status = VersionStatus.ACTIVE
        versions[version_id].traffic_weight = 100.0
        versions[version_id].promoted_at = time.time()

        logger.info(f"[VersionManager] Promoted {model_id}/{version_id}")
        return True

    # -- Stats summary --

    def get_version_stats(self, model_id: str, version_id: str) -> dict | None:
        """Get comprehensive stats for a version."""
        version = self.get_version(model_id, version_id)
        if version is None:
            return None

        m = version.metrics
        return {
            "version_id": version_id,
            "status": version.status.value,
            "traffic_weight": version.traffic_weight,
            "total_requests": m.total_requests,
            "error_rate": round(m.error_rate, 6),
            "avg_latency_ms": round(m.avg_latency, 2),
            "p50_latency_ms": round(m.p50_latency, 2),
            "p99_latency_ms": round(m.p99_latency, 2),
            "avg_prompt_tokens": round(m.avg_prompt_tokens, 1),
            "avg_completion_tokens": round(m.avg_completion_tokens, 1),
            "feedback_avg": round(
                sum(m.feedback_scores) / len(m.feedback_scores), 4
            ) if m.feedback_scores else None,
        }
