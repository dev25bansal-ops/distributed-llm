"""Execution strategies for the InferenceEngine (M-05 decomposition).

Each strategy is a self-contained module with a narrow EngineContext seam.
Imported back into inference_engine.py for backward compatibility.
"""

from distllm.core.strategies.distributed import DistributedStrategy
from distllm.core.strategies.distributed_spec import DistributedSpeculativeStrategy
from distllm.core.strategies.local import LocalStrategy
from distllm.core.strategies.prompt_lookup import PromptLookupStrategy
from distllm.core.strategies.speculative import SpeculativeStrategy

# inference_engine.py imports the underscore names, kept for backward
# compatibility with call sites written against the pre-split layout.
_DistributedStrategy = DistributedStrategy
_DistributedSpeculativeStrategy = DistributedSpeculativeStrategy
_LocalStrategy = LocalStrategy
_PromptLookupStrategy = PromptLookupStrategy
_SpeculativeStrategy = SpeculativeStrategy

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
