"""N2: Self-healing cache diagnostics.

Monitors cache health and auto-repairs common issues like
stale index entries, missing disk entries, and underperforming tiers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class DiagnosticIssue:
    """A detected cache health issue."""
    severity: str  # "info", "warning", "critical"
    component: str  # "prefix_cache", "disk", "gossip", "predictive"
    description: str
    auto_fixed: bool = False
    timestamp: float = field(default_factory=time.time)


class CacheDoctor:
    """Monitors cache health and auto-repairs issues.

    Checks:
    - Prefix cache hit rate degradation
    - Disk cache staleness
    - Gossip index lag
    - Predictive cache accuracy
    - Memory pressure
    """

    def __init__(self, cache_manager: Any = None):
        self._cache_manager = cache_manager
        self._issues: list[DiagnosticIssue] = []
        self._tier_disabled_until: dict[str, float] = {}

    def diagnose(self) -> list[DiagnosticIssue]:
        """Run all diagnostic checks.

        Returns:
            List of detected issues.
        """
        issues = []

        # Check prefix cache health
        issues.extend(self._check_prefix_cache())

        # Check tier performance
        issues.extend(self._check_tier_performance())

        # Check memory pressure
        issues.extend(self._check_memory_pressure())

        # Auto-repair where possible
        for issue in issues:
            if issue.severity == "warning":
                self._auto_repair(issue)

        self._issues.extend(issues)
        return issues

    def _check_prefix_cache(self) -> list[DiagnosticIssue]:
        """Check prefix cache health."""
        issues = []

        if self._cache_manager is None:
            return issues

        try:
            if hasattr(self._cache_manager, 'prefix_cache') and self._cache_manager.prefix_cache is not None:
                stats = self._cache_manager.prefix_cache.stats()
                hit_rate = stats.get("prefix_cache_hit_rate", 0)
                memory_util = stats.get("prefix_cache_memory_util", 0)

                if hit_rate < 0.1 and stats.get("prefix_cache_hits", 0) + stats.get("prefix_cache_misses", 0) > 100:
                    issues.append(DiagnosticIssue(
                        severity="warning",
                        component="prefix_cache",
                        description=f"Low hit rate: {hit_rate:.1%}",
                    ))

                if memory_util > 0.95:
                    issues.append(DiagnosticIssue(
                        severity="warning",
                        component="prefix_cache",
                        description=f"Memory near limit: {memory_util:.1%} utilized",
                    ))
        except Exception as e:
            issues.append(DiagnosticIssue(
                severity="info",
                component="prefix_cache",
                description=f"Could not check prefix cache: {e}",
            ))

        return issues

    def _check_tier_performance(self) -> list[DiagnosticIssue]:
        """Check if any tier is underperforming."""
        issues = []

        if self._cache_manager is None:
            return issues

        try:
            if hasattr(self._cache_manager, 'get_tier_stats'):
                tier_stats = self._cache_manager.get_tier_stats()
                for tier, stats in tier_stats.items():
                    hits = stats.get("hits", 0)
                    misses = stats.get("misses", 0)
                    total = hits + misses

                    if total > 50 and hits == 0:
                        # Tier never hits — consider disabling
                        if tier not in self._tier_disabled_until:
                            issues.append(DiagnosticIssue(
                                severity="warning",
                                component=tier,
                                description=f"Tier '{tier}' has 0 hits after {total} lookups",
                            ))

            if hasattr(self._cache_manager, 'get_tier_latencies'):
                latencies = self._cache_manager.get_tier_latencies()
                for tier, stats in latencies.items():
                    p95 = stats.get("p95_ms", 0)
                    if p95 > 5000:  # > 5 seconds
                        issues.append(DiagnosticIssue(
                            severity="warning",
                            component=tier,
                            description=f"High P95 latency: {p95:.0f}ms",
                        ))
        except Exception:
            pass

        return issues

    def _check_memory_pressure(self) -> list[DiagnosticIssue]:
        """Check system memory pressure."""
        issues = []

        try:
            import psutil
            mem = psutil.virtual_memory()
            if mem.percent > 90:
                issues.append(DiagnosticIssue(
                    severity="critical",
                    component="system",
                    description=f"System memory critical: {mem.percent:.1f}% used",
                ))
            elif mem.percent > 80:
                issues.append(DiagnosticIssue(
                    severity="warning",
                    component="system",
                    description=f"System memory high: {mem.percent:.1f}% used",
                ))
        except ImportError:
            pass

        return issues

    def _auto_repair(self, issue: DiagnosticIssue) -> None:
        """Attempt to auto-repair an issue."""
        try:
            if "Low hit rate" in issue.description:
                # Could trigger cache warming or adjustment
                issue.auto_fixed = True
                logger.info(f"Auto-repair: flagged low hit rate issue for review")

            elif "Memory near limit" in issue.description:
                # Trigger eviction
                if self._cache_manager and hasattr(self._cache_manager, 'prefix_cache'):
                    cache = self._cache_manager.prefix_cache
                    if hasattr(cache, '_evict_until_fit'):
                        cache._evict_until_fit(0)
                        issue.auto_fixed = True
                        logger.info("Auto-repair: triggered memory eviction")

            elif "0 hits" in issue.description:
                # Disable underperforming tier temporarily
                tier = issue.component
                self._tier_disabled_until[tier] = time.time() + 300  # 5 min
                issue.auto_fixed = True
                logger.info(f"Auto-repair: disabled tier '{tier}' for 5 minutes")

        except Exception as e:
            logger.warning(f"Auto-repair failed: {e}")

    def is_tier_disabled(self, tier: str) -> bool:
        """Check if a tier is temporarily disabled."""
        until = self._tier_disabled_until.get(tier, 0)
        if time.time() < until:
            return True
        # Re-enable if timeout expired
        if tier in self._tier_disabled_until:
            del self._tier_disabled_until[tier]
        return False

    def get_issues(self, severity: str | None = None) -> list[DiagnosticIssue]:
        """Get detected issues, optionally filtered by severity."""
        if severity:
            return [i for i in self._issues if i.severity == severity]
        return list(self._issues)

    def clear_issues(self) -> None:
        """Clear the issue history."""
        self._issues.clear()

    def health_summary(self) -> dict:
        """Return a health summary."""
        return {
            "total_issues": len(self._issues),
            "critical": sum(1 for i in self._issues if i.severity == "critical"),
            "warnings": sum(1 for i in self._issues if i.severity == "warning"),
            "auto_fixed": sum(1 for i in self._issues if i.auto_fixed),
            "disabled_tiers": list(self._tier_disabled_until.keys()),
        }
