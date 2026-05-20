from __future__ import annotations

import time
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from distllm.core.optimization.space import SearchSpace


class ObjectiveDirection(Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class BayesianOptimizer:
    """Bayesian optimization engine wrapping optuna's TPE sampler.

    Manages an optuna Study that searches over the configuration space
    to find parameter values that maximize (or minimize) an objective.

    Usage:
        space = default_search_space()
        optimizer = BayesianOptimizer(space, direction=ObjectiveDirection.MAXIMIZE)

        for trial_idx in range(50):
            config = optimizer.suggest()
            metric = run_trial(config)        # e.g., throughput
            optimizer.report(metric)

        best = optimizer.best_config
    """

    def __init__(
        self,
        search_space: SearchSpace,
        direction: ObjectiveDirection = ObjectiveDirection.MAXIMIZE,
        study_name: str | None = None,
        storage: str | None = None,
        n_startup_trials: int = 10,
        n_ei_candidates: int = 24,
        seed: int = 42,
    ):
        self._search_space = search_space
        self._direction = direction
        self._study_name = study_name or f"distllm_opt_{int(time.time())}"
        self._n_startup_trials = n_startup_trials
        self._n_ei_candidates = n_ei_candidates
        self._seed = seed

        self._study = self._create_study(storage)
        self._pending_params: dict[int, dict[str, Any]] = {}

    def _create_study(self, storage: str | None):
        import optuna

        sampler = optuna.samplers.TPESampler(
            n_startup_trials=self._n_startup_trials,
            n_ei_candidates=self._n_ei_candidates,
            seed=self._seed,
        )

        study = optuna.create_study(
            study_name=self._study_name,
            direction=self._direction.value,
            sampler=sampler,
            storage=storage,
            load_if_exists=True,
        )
        return study

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def suggest(self) -> dict[str, Any]:
        """Suggest the next configuration to try.

        Returns a dict mapping parameter names to values.
        """
        trial = self._study.ask()
        config = self._search_space.suggest_from_trial(trial)
        self._pending_params[trial.number] = config
        return config

    def report(self, value: float, trial_number: int | None = None) -> None:
        """Report the objective value for a completed trial.

        Args:
            value: The objective value (e.g., throughput in tok/s).
            trial_number: The trial number (auto-detected if None).
        """
        if trial_number is None:
            trial_number = self._study.trials[-1].number if self._study.trials else 0

        try:
            self._study.tell(trial_number, values=value)
            logger.debug(f"Trial {trial_number} reported value={value:.2f}")
        except Exception as e:
            logger.warning(f"Failed to report trial {trial_number}: {e}")
        finally:
            self._pending_params.pop(trial_number, None)

    def suggest_and_report(
        self,
        objective_fn: Callable[[dict[str, Any]], float],
    ) -> dict[str, Any]:
        """Convenience: suggest a config, run objective_fn, report result.

        Returns the best config found so far after this trial.
        """
        config = self.suggest()
        value = objective_fn(config)
        self.report(value)
        return self.best_config

    @property
    def best_config(self) -> dict[str, Any] | None:
        try:
            return self._study.best_trial.params
        except ValueError:
            return None

    @property
    def best_value(self) -> float | None:
        try:
            return self._study.best_trial.value
        except ValueError:
            return None

    @property
    def trials(self) -> list:
        return self._study.trials

    @property
    def trials_dataframe(self):

        return self._study.trials_dataframe()

    @property
    def study(self):
        return self._study

    def finished_trials_count(self) -> int:
        import optuna

        return len(
            [t for t in self._study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        )

    def optimize(
        self,
        objective_fn: Callable[[dict[str, Any]], float],
        n_trials: int = 50,
    ) -> dict[str, Any]:
        """Run a full optimization loop.

        Args:
            objective_fn: Function that takes a config dict and returns a float.
            n_trials: Number of trials to run.

        Returns:
            The best configuration found.
        """
        for i in range(n_trials):
            self.suggest_and_report(objective_fn)
            best_val = self.best_value
            if i % 10 == 0 or i == n_trials - 1:
                logger.info(
                    f"BayesianOptimizer: trial {i + 1}/{n_trials}, "
                    f"best={best_val:.2f}"
                )

        return self.best_config

    def plot_history(self, output_path: str | Path | None = None):
        """Generate a plot of the optimization history.

        Requires plotly. Saves to output_path if provided, otherwise displays.
        """
        try:
            from optuna.visualization import plot_optimization_history

            fig = plot_optimization_history(self._study)
            if output_path:
                fig.write_html(str(output_path))
            else:
                fig.show()
        except ImportError:
            logger.warning("plotly not installed; skipping plot")

    def plot_param_importances(self, output_path: str | Path | None = None):
        """Generate a plot of parameter importances.

        Requires plotly. Saves to output_path if provided, otherwise displays.
        """
        try:
            from optuna.visualization import plot_param_importances

            fig = plot_param_importances(self._study)
            if output_path:
                fig.write_html(str(output_path))
            else:
                fig.show()
        except ImportError:
            logger.warning("plotly not installed; skipping plot")

    def summary(self) -> str:
        best_val = self.best_value
        lines = [
            f"BayesianOptimizer: {self._study_name}",
            f"  Direction: {self._direction.value}",
            f"  Trials: {len(self._study.trials)}",
            f"  Finished: {self.finished_trials_count()}",
            f"  Best value: {best_val}",
        ]
        best_cfg = self.best_config
        if best_cfg:
            lines.append("  Best config:")
            for k, v in best_cfg.items():
                lines.append(f"    {k}: {v}")
        return "\n".join(lines)
