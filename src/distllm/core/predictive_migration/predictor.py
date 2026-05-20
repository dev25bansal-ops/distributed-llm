from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass
class Transition:
    """A Markov transition from one prefix to another."""
    from_hash: str
    to_hash: str
    count: int = 1


@dataclass
class Prediction:
    """A predicted next prefix with confidence."""
    prefix_hash: str
    prefix_tokens: tuple[int, ...]
    confidence: float
    transition_probability: float
    frequency_score: float


class MarkovChainPredictor:
    """Predicts the next likely prefix using a Markov chain model.

    Builds a first-order Markov chain from observed prefix transitions.
    Given the current set of active prefixes (or a single current prefix),
    predicts which prefix(es) will follow with the highest probability.

    Supports:
    - First-order Markov: P(next | current)
    - Second-order (optional): P(next | prev, current)
    - Sliding window: forgets old transitions over time
    - Confidence scoring combining transition prob + global frequency

    Usage:
        predictor = MarkovChainPredictor(order=1)
        predictor.observe(prev_hash="abc", current_hash="def")
        predictions = predictor.predict(current_hash="def", top_k=5)
    """

    def __init__(
        self,
        order: int = 1,
        window_size: int = 10000,
        decay_hours: float = 24.0,
        min_observations: int = 2,
    ):
        if order not in (1, 2):
            raise ValueError("Markov chain order must be 1 or 2")
        self._order = order
        self._window_size = window_size
        self._decay_seconds = decay_hours * 3600
        self._min_observations = min_observations

        # First-order: from_hash -> {to_hash -> count}
        self._first_order: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        # Second-order: (prev_hash, current_hash) -> {to_hash -> count}
        self._second_order: dict[tuple[str, str], dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        # Global frequency of each prefix
        self._global_freq: dict[str, float] = defaultdict(float)
        # Total transitions observed
        self._total_transitions: int = 0
        # Last two prefixes for sequential observation
        self._last_hashes: list[str] = []
        # Timestamp for decay
        self._last_decay_time: float = time.time()

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def observe(self, prefix_hash: str) -> None:
        """Record a prefix occurrence and update transition counts.

        Builds the Markov chain by linking consecutive observations.
        """
        self._maybe_decay()
        self._global_freq[prefix_hash] += 1.0

        if self._last_hashes:
            prev_hash = self._last_hashes[-1]
            self._first_order[prev_hash][prefix_hash] += 1
            self._total_transitions += 1

            if self._order == 2 and len(self._last_hashes) >= 2:
                prev_prev = self._last_hashes[-2]
                self._second_order[(prev_prev, prev_hash)][prefix_hash] += 1

        self._last_hashes.append(prefix_hash)
        if len(self._last_hashes) > self._order + 1:
            self._last_hashes.pop(0)

        self._maybe_shrink()

    def observe_transition(
        self, from_hash: str, to_hash: str
    ) -> None:
        """Directly record a transition between two prefixes."""
        self._maybe_decay()
        self._first_order[from_hash][to_hash] += 1
        self._global_freq[from_hash] += 1.0
        self._global_freq[to_hash] += 1.0
        self._total_transitions += 1

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        current_hash: str | None = None,
        prev_hash: str | None = None,
        top_k: int = 10,
    ) -> list[Prediction]:
        """Predict the next most likely prefix(es).

        Args:
            current_hash: The current prefix hash.
            prev_hash: The previous prefix hash (for 2nd-order).
            top_k: Number of predictions to return.

        Returns:
            List of Prediction objects sorted by confidence (descending).
        """
        candidates: dict[str, float] = defaultdict(float)

        # First-order predictions
        if current_hash and current_hash in self._first_order:
            transitions = self._first_order[current_hash]
            total = sum(transitions.values())
            if total > 0:
                for to_hash, count in transitions.items():
                    prob = count / total
                    freq = self._global_freq.get(to_hash, 0)
                    candidates[to_hash] += (
                        prob * 0.6 + math.log1p(freq) * 0.4
                    )

        # Second-order predictions (override if order=2)
        if (
            self._order == 2
            and prev_hash
            and current_hash
            and (prev_hash, current_hash) in self._second_order
        ):
            transitions = self._second_order[(prev_hash, current_hash)]
            total = sum(transitions.values())
            if total > 0:
                for to_hash, count in transitions.items():
                    prob = count / total
                    freq = self._global_freq.get(to_hash, 0)
                    candidates[to_hash] = max(
                        candidates.get(to_hash, 0),
                        prob * 0.8 + math.log1p(freq) * 0.2,
                    )

        # Filter by min observations
        filtered = []
        for to_hash, confidence in candidates.items():
            transition_prob = self._transition_probability(
                current_hash or "", to_hash
            )
            freq_score = math.log1p(self._global_freq.get(to_hash, 0))
            # Estimate if we have enough data
            if transition_prob > 0 and self._total_transitions >= self._min_observations:
                filtered.append(
                    Prediction(
                        prefix_hash=to_hash,
                        prefix_tokens=(),
                        confidence=confidence,
                        transition_probability=transition_prob,
                        frequency_score=freq_score,
                    )
                )

        filtered.sort(key=lambda p: p.confidence, reverse=True)
        return filtered[:top_k]

    def _transition_probability(
        self, from_hash: str, to_hash: str
    ) -> float:
        transitions = self._first_order.get(from_hash, {})
        total = sum(transitions.values())
        if total == 0:
            return 0.0
        return transitions.get(to_hash, 0) / total

    def predict_from_tokens(
        self,
        token_ids: list[int],
        prefix_hash_fn,
        top_k: int = 10,
    ) -> list[Prediction]:
        current_hash = prefix_hash_fn(token_ids)
        return self.predict(current_hash=current_hash, top_k=top_k)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def _maybe_decay(self) -> None:
        now = time.time()
        elapsed = now - self._last_decay_time
        if elapsed < self._decay_seconds * 0.1:
            return
        factor = math.exp(-elapsed / self._decay_seconds)
        for from_hash in self._first_order:
            for to_hash in list(self._first_order[from_hash]):
                self._first_order[from_hash][to_hash] = max(
                    1, int(self._first_order[from_hash][to_hash] * factor)
                )
        for key in self._second_order:
            for to_hash in list(self._second_order[key]):
                self._second_order[key][to_hash] = max(
                    1, int(self._second_order[key][to_hash] * factor)
                )
        for h in self._global_freq:
            self._global_freq[h] *= factor
        self._last_decay_time = now

    def _maybe_shrink(self) -> None:
        total = sum(
            sum(v.values()) for v in self._first_order.values()
        )
        if total > self._window_size:
            self._first_order.clear()
            self._second_order.clear()
            self._total_transitions = 0

    def total_states(self) -> int:
        return len(self._first_order)

    def total_transitions_count(self) -> int:
        return self._total_transitions

    def stats(self) -> dict[str, Any]:
        return {
            "order": self._order,
            "total_states": self.total_states(),
            "total_transitions": self._total_transitions,
            "window_size": self._window_size,
            "decay_hours": self._decay_seconds / 3600,
        }

    def reset(self) -> None:
        self._first_order.clear()
        self._second_order.clear()
        self._global_freq.clear()
        self._total_transitions = 0
        self._last_hashes.clear()
        self._last_decay_time = time.time()
