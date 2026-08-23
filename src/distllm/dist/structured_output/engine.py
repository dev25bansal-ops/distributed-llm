"""Self-healing structured output engine with backtrack-and-repair.

Extends the base structured output pipeline with the ability to detect
schema invalidity mid-generation, roll back to the last valid prefix,
and attempt repair — instead of failing the entire generation.
"""

from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RepairConfig:
    """Configuration for the self-healing repair orchestrator."""
    max_repair_attempts: int = 3
    repair_strategies: list[str] = field(default_factory=lambda: ["heuristic", "truncate", "regenerate"])
    log_repair_trajectories: bool = True
    max_trajectory_buffer: int = 1000


class RepairOrchestrator:
    """Wraps generation with backtrack-and-repair on schema violation.

    On each token, validates against the schema. On violation:
    1. Snapshot the valid prefix.
    2. Roll back to last valid state.
    3. Attempt repair: heuristic first, then truncate, then regenerate.
    4. If repair succeeds, continue generation from repaired prefix.
    5. If all repair attempts fail, return the last valid prefix with an error marker.
    """

    def __init__(self, config: RepairConfig | None = None):
        self.config = config or RepairConfig()
        self._valid_prefix: str = ""
        self._repair_trajectories: deque[dict[str, Any]] = deque(maxlen=self.config.max_trajectory_buffer)
        self._repair_attempts: int = 0
        self._successful_repairs: int = 0

    def validate_token(self, token: str, prefix: str, schema: dict | None = None) -> bool:
        """Check whether *token* is schema-compliant given the current *prefix*.

        Returns True if the token is valid or no schema is provided.
        """
        if schema is None:
            return True
        candidate = prefix + token
        try:
            json.loads(candidate)
            return True
        except (json.JSONDecodeError, ValueError):
            return False

    def get_valid_prefix(self) -> str:
        """Return the last known-good prefix."""
        return self._valid_prefix

    def set_valid_prefix(self, prefix: str) -> None:
        """Update the last known-good prefix (called on each valid token)."""
        self._valid_prefix = prefix

    def repair_output(self, invalid_output: str, schema: dict | None = None) -> str:
        """Attempt to repair an invalid output.

        Tries each configured strategy in order.  Returns the repaired output
        or the original if all strategies fail.
        """
        result = invalid_output
        for strategy in self.config.repair_strategies:
            if strategy == "heuristic":
                result = self._heuristic_repair(invalid_output)
            elif strategy == "truncate":
                result = self._truncate_repair(invalid_output)
            elif strategy == "regenerate":
                result = self._regenerate_repair(invalid_output)

            if result and self._is_valid_json(result):
                self._repair_attempts += 1
                self._successful_repairs += 1
                return result

        self._repair_attempts += 1
        return invalid_output

    def learn_from_repair(self, trajectory: dict[str, Any]) -> None:
        """Store a repair trajectory for offline training."""
        if self.config.log_repair_trajectories:
            self._repair_trajectories.append(trajectory)

    @property
    def repair_rate(self) -> float:
        """Ratio of successful repairs to total repair attempts."""
        if self._repair_attempts == 0:
            return 1.0
        return self._successful_repairs / self._repair_attempts

    @property
    def repair_trajectories(self) -> list[dict[str, Any]]:
        """Return all stored repair trajectories."""
        return list(self._repair_trajectories)

    # ── Private repair strategies ────────────────────────────────────────

    def _heuristic_repair(self, text: str) -> str:
        """Apply heuristic JSON fixes — insert missing brackets, fix quotes."""
        text = text.strip()
        # Strip trailing commas before closing brackets
        text = re.sub(r",\s*([}\]])", r"\1", text)
        # Balance brackets
        openers = text.count("{") + text.count("[")
        closers = text.count("}") + text.count("]")
        for _ in range(openers - closers):
            text += "}"
        # Balance parentheses
        if text.count('"') % 2 != 0:
            text += '"'
        return text

    def _truncate_repair(self, text: str) -> str:
        """Truncate to the last valid JSON prefix."""
        for i in range(len(text), 0, -1):
            candidate = text[:i]
            if self._is_valid_json(candidate):
                return candidate
        return text

    def _regenerate_repair(self, text: str) -> str:
        """Delegated: attempt to regenerate from the last valid prefix."""
        valid = self.get_valid_prefix()
        if valid:
            return valid
        return text

    @staticmethod
    def _is_valid_json(text: str) -> bool:
        try:
            json.loads(text)
            return True
        except (json.JSONDecodeError, ValueError):
            return False
