from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class BestConfig:
    """Best configuration found so far."""
    config: dict[str, Any]
    value: float
    trial_number: int
    found_at: float = field(default_factory=time.time)


@dataclass
class TrialRecord:
    """Persistent record of a single trial."""
    trial_number: int
    config: dict[str, Any]
    value: float
    duration_seconds: float
    timestamp: float = field(default_factory=time.time)
    error: str | None = None


class OptimizationTracker:
    """Tracks trials, persists results, and selects the best config.

    Persists to JSON files for later analysis and reuse.
    Supports loading previous best configs for warm-starting.
    """

    def __init__(
        self,
        output_dir: str | Path = "~/.distllm/optimization",
        study_name: str = "default",
        maximize: bool = True,
    ):
        self._output_dir = Path(output_dir).expanduser().resolve()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._study_name = study_name
        self._maximize = maximize
        self._trials: list[TrialRecord] = []
        self._best: BestConfig | None = None

        self._load()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        trial_number: int,
        config: dict[str, Any],
        value: float,
        duration_seconds: float,
        error: str | None = None,
    ) -> None:
        record = TrialRecord(
            trial_number=trial_number,
            config=dict(config),
            value=value,
            duration_seconds=duration_seconds,
            error=error,
        )
        self._trials.append(record)
        self._update_best(record)
        self._save()

    def _update_best(self, record: TrialRecord) -> None:
        if record.error is not None:
            return
        if self._best is None:
            self._best = BestConfig(
                config=record.config,
                value=record.value,
                trial_number=record.trial_number,
            )
            return
        is_better = (
            record.value > self._best.value
            if self._maximize
            else record.value < self._best.value
        )
        if is_better:
            self._best = BestConfig(
                config=record.config,
                value=record.value,
                trial_number=record.trial_number,
            )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @property
    def best_config(self) -> BestConfig | None:
        return self._best

    @property
    def best_config_dict(self) -> dict[str, Any] | None:
        return self._best.config if self._best else None

    @property
    def trials(self) -> list[TrialRecord]:
        return list(self._trials)

    @property
    def trial_count(self) -> int:
        return len(self._trials)

    def trials_sorted(self, key: str = "value", reverse: bool = True) -> list[TrialRecord]:
        return sorted(
            [t for t in self._trials if t.error is None],
            key=lambda r: getattr(r, key, 0),
            reverse=reverse,
        )

    def top_k(self, k: int = 5) -> list[TrialRecord]:
        return self.trials_sorted()[:k]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _trials_path(self) -> Path:
        return self._output_dir / f"{self._study_name}_trials.json"

    def _best_path(self) -> Path:
        return self._output_dir / f"{self._study_name}_best.json"

    def _save(self) -> None:
        try:
            trials_data = [asdict(t) for t in self._trials]
            with open(self._trials_path(), "w") as f:
                json.dump(trials_data, f, indent=2, default=str)

            if self._best is not None:
                best_data = asdict(self._best)
                with open(self._best_path(), "w") as f:
                    json.dump(best_data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save optimization results: {e}")

    def _load(self) -> None:
        try:
            trials_path = self._trials_path()
            if trials_path.exists():
                with open(trials_path) as f:
                    trials_data = json.load(f)
                for t in trials_data:
                    self._trials.append(TrialRecord(**t))

            best_path = self._best_path()
            if best_path.exists():
                with open(best_path) as f:
                    best_data = json.load(f)
                self._best = BestConfig(**best_data)

            if self._trials:
                logger.info(
                    f"Loaded {len(self._trials)} trials from {self._output_dir}"
                )
        except Exception as e:
            logger.debug(f"Could not load previous trials: {e}")

    def summary(self) -> str:
        lines = [
            f"OptimizationTracker: {self._study_name}",
            f"  Trials: {self.trial_count}",
            f"  Direction: {'maximize' if self._maximize else 'minimize'}",
        ]
        if self._best:
            lines.append(f"  Best value: {self._best.value}")
            lines.append("  Best config:")
            for k, v in self._best.config.items():
                lines.append(f"    {k}: {v}")
        else:
            lines.append("  No best config yet")
        return "\n".join(lines)
