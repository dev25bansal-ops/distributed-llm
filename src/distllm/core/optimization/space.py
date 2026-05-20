from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class ParamDomain(ABC):
    """Base class for a single parameter's search domain."""

    name: str

    @abstractmethod
    def sample_random(self) -> Any:
        ...

    @abstractmethod
    def to_optuna_distribution(self):
        ...


@dataclass
class IntDomain(ParamDomain):
    """Integer parameter with optional log-uniform scaling."""

    name: str
    low: int
    high: int
    log: bool = False

    def sample_random(self) -> int:
        if self.log:
            import math

            log_low = math.log(self.low)
            log_high = math.log(self.high)
            return int(round(math.exp(random.uniform(log_low, log_high))))
        return random.randint(self.low, self.high)

    def to_optuna_distribution(self):
        import optuna

        if self.log:
            return optuna.distributions.IntDistribution(
                low=self.low, high=self.high, log=True
            )
        return optuna.distributions.IntDistribution(low=self.low, high=self.high)

    def __repr__(self) -> str:
        scale = "log" if self.log else "linear"
        return f"Int({self.name}: [{self.low}, {self.high}], {scale})"


@dataclass
class CategoricalDomain(ParamDomain):
    """Categorical parameter with a fixed set of choices."""

    name: str
    choices: list[Any]

    def sample_random(self) -> Any:
        return random.choice(self.choices)

    def to_optuna_distribution(self):
        import optuna

        return optuna.distributions.CategoricalDistribution(choices=self.choices)

    def __repr__(self) -> str:
        return f"Categorical({self.name}: {self.choices})"


def _suggest_from_domain(trial, domain: ParamDomain) -> Any:
    """Suggest a value for the given domain using an optuna trial."""
    if isinstance(domain, IntDomain):
        return trial.suggest_int(
            domain.name, domain.low, domain.high, log=domain.log
        )
    if isinstance(domain, CategoricalDomain):
        return trial.suggest_categorical(domain.name, domain.choices)
    raise TypeError(f"Unknown domain type: {type(domain)}")


@dataclass
class SearchSpace:
    """Collection of parameter domains defining the full search space.

    The 6 parameters to optimize:
        - batch_size:              Number of sequences in a batch
        - tensor_parallel_degree:  GPUs used for tensor parallelism
        - pipeline_stages:         Number of pipeline-parallel stages
        - quantization:            Precision level (none/bnb_8bit/fp8)
        - speculation_length:      Draft tokens for speculative decoding
        - chunk_size:              Tokens per chunk in chunked prefill
    """

    batch_size: IntDomain = field(
        default_factory=lambda: IntDomain("batch_size", low=1, high=128, log=True)
    )
    tensor_parallel_degree: IntDomain = field(
        default_factory=lambda: IntDomain(
            "tensor_parallel_degree", low=1, high=8, log=False
        )
    )
    pipeline_stages: IntDomain = field(
        default_factory=lambda: IntDomain(
            "pipeline_stages", low=1, high=4, log=False
        )
    )
    quantization: CategoricalDomain = field(
        default_factory=lambda: CategoricalDomain(
            "quantization", choices=["none", "bnb_8bit", "fp8"]
        )
    )
    speculation_length: IntDomain = field(
        default_factory=lambda: IntDomain(
            "speculation_length", low=0, high=10, log=False
        )
    )
    chunk_size: IntDomain = field(
        default_factory=lambda: IntDomain(
            "chunk_size", low=128, high=2048, log=True
        )
    )

    @property
    def domains(self) -> list[ParamDomain]:
        return [
            self.batch_size,
            self.tensor_parallel_degree,
            self.pipeline_stages,
            self.quantization,
            self.speculation_length,
            self.chunk_size,
        ]

    @property
    def param_names(self) -> list[str]:
        return [d.name for d in self.domains]

    def sample_random_config(self) -> dict[str, Any]:
        return {d.name: d.sample_random() for d in self.domains}

    def suggest_from_trial(self, trial) -> dict[str, Any]:
        return {
            d.name: _suggest_from_domain(trial, d) for d in self.domains
        }

    def validate(self, config: dict[str, Any]) -> dict[str, Any]:
        validated = {}
        for domain in self.domains:
            value = config.get(domain.name)
            if value is None:
                raise ValueError(f"Missing parameter '{domain.name}' in config")
            if isinstance(domain, IntDomain):
                validated[domain.name] = int(
                    max(domain.low, min(domain.high, int(value)))
                )
            elif isinstance(domain, CategoricalDomain):
                if value not in domain.choices:
                    raise ValueError(
                        f"'{domain.name}' must be one of {domain.choices}, "
                        f"got {value!r}"
                    )
                validated[domain.name] = value
        return validated

    def to_dict(self) -> dict[str, dict]:
        def _describe(d: ParamDomain) -> dict:
            if isinstance(d, IntDomain):
                return {"type": "int", "low": d.low, "high": d.high, "log": d.log}
            if isinstance(d, CategoricalDomain):
                return {"type": "categorical", "choices": list(d.choices)}
            return {"type": "unknown"}

        return {d.name: _describe(d) for d in self.domains}

    def __repr__(self) -> str:
        parts = [repr(d) for d in self.domains]
        return f"SearchSpace({', '.join(parts)})"


def default_search_space() -> SearchSpace:
    """Return the default search space for the 6 optimizable parameters.

    Override individual domains after creation:
        space = default_search_space()
        space.batch_size = IntDomain("batch_size", 1, 64, log=True)
    """
    return SearchSpace()
