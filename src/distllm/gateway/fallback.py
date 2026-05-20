"""Fallback manager: tracks backend failures and suggests optimal fallback."""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FailureStats:
    count: int = 0
    last_failure: float = 0.0
    consecutive: int = 0


class FallbackManager:
    """Tracks failure counts and suggests fallback order."""

    def __init__(self):
        self._failures: dict[str, FailureStats] = defaultdict(FailureStats)
        self._circuit_breakers: dict[str, bool] = {}

    def record_failure(self, backend_name: str, timestamp: Optional[float] = None) -> None:
        import time
        ts = timestamp or time.time()
        stats = self._failures[backend_name]
        stats.count += 1
        stats.last_failure = ts
        stats.consecutive += 1
        if stats.consecutive >= 5:
            self._circuit_breakers[backend_name] = True

    def record_success(self, backend_name: str) -> None:
        stats = self._failures.get(backend_name)
        if stats:
            stats.consecutive = 0
        self._circuit_breakers.pop(backend_name, None)

    def get_failure_count(self, backend_name: str) -> int:
        return self._failures[backend_name].count

    def is_circuit_open(self, backend_name: str) -> bool:
        return self._circuit_breakers.get(backend_name, False)

    def get_sorted_backends(self, candidates: list[str]) -> list[str]:
        """Sort candidates by fewest failures first."""
        def key(name):
            return self._failures[name].count, self._failures[name].consecutive
        return sorted(candidates, key=key)

    def reset_all(self) -> None:
        self._failures.clear()
        self._circuit_breakers.clear()

    def stats(self) -> dict:
        return {
            name: {
                "count": s.count,
                "consecutive": s.consecutive,
                "circuit_open": self._circuit_breakers.get(name, False),
            }
            for name, s in self._failures.items()
        }
