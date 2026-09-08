"""Execution strategies for the InferenceEngine (M-05 decomposition).

Each strategy is a self-contained module with a narrow EngineContext seam.
Imported back into inference_engine.py for backward compatibility.
"""

from distllm.core.strategies.local import LocalStrategy as _LocalStrategy
from distllm.core.strategies.speculative import SpeculativeStrategy as _SpeculativeStrategy
from distllm.core.strategies.distributed_spec import (
    DistributedSpeculativeStrategy as _DistributedSpeculativeStrategy,
)
from distllm.core.strategies.distributed import DistributedStrategy as _DistributedStrategy
from distllm.core.strategies.prompt_lookup import (
    PromptLookupStrategy as _PromptLookupStrategy,
)

__all__ = [
    "_LocalStrategy",
    "_SpeculativeStrategy",
    "_DistributedSpeculativeStrategy",
    "_DistributedStrategy",
    "_PromptLookupStrategy",
    "LocalStrategy",
    "SpeculativeStrategy",
    "DistributedSpeculativeStrategy",
    "DistributedStrategy",
    "PromptLookupStrategy",
]
