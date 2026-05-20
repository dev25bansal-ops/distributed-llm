"""Dynamic model composition: chaining embedding, reranker, and generation models.

Allows composing multiple models into a single pipeline specification,
where the output of each step feeds into the next.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator

from loguru import logger


class StepType(str, Enum):
    embedding = "embedding"
    reranker = "reranker"
    generate = "generate"
    transform = "transform"


@dataclass
class PipelineStep:
    """A single step in a composed pipeline.

    Attributes:
        model: Model identifier for this step.
        step_type: Type of this step (embedding, reranker, generate, transform).
        params: Optional extra parameters to pass to the model for this step.
        timeout_ms: Optional per-step timeout in milliseconds.
    """
    model: str
    step_type: StepType
    params: dict[str, Any] = field(default_factory=dict)
    timeout_ms: float | None = None


@dataclass
class PipelineSpec:
    """Specification for a composed pipeline.

    Attributes:
        pipeline_id: Unique identifier for this pipeline.
        steps: Ordered list of pipeline steps.
        fallback_pipeline_id: Optional fallback pipeline if this one fails SLA.
        metadata: Optional key-value metadata.
    """
    pipeline_id: str
    steps: list[PipelineStep]
    fallback_pipeline_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class PipelineExecutor:
    """Chains model invocations across embedding, reranker, and generation.

    Each step's output is passed as input to the next step.
    Supports async streaming for generation steps.
    """

    def __init__(self, coordinator=None):
        self._coordinator = coordinator
        self._pipelines: dict[str, PipelineSpec] = {}
        self._fallback_chains: dict[str, str] = {}

    def register(self, spec: PipelineSpec) -> None:
        """Register a pipeline specification."""
        self._pipelines[spec.pipeline_id] = spec
        if spec.fallback_pipeline_id:
            self._fallback_chains[spec.pipeline_id] = spec.fallback_pipeline_id

    def get(self, pipeline_id: str) -> PipelineSpec | None:
        """Look up a pipeline by ID, following fallback chains."""
        spec = self._pipelines.get(pipeline_id)
        if spec:
            return spec
        fallback = self._fallback_chains.get(pipeline_id)
        if fallback:
            return self._pipelines.get(fallback)
        return None

    async def execute(
        self,
        pipeline_id: str,
        input_text: str,
        max_latency_ms: float | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Execute a pipeline asynchronously, yielding step results.

        Yields dicts with keys:
          - step_index: int
          - step_type: str
          - output: Any
          - latency_ms: float
          - error: str | None
        """
        spec = self.get(pipeline_id)
        if spec is None:
            yield {
                "step_index": 0,
                "step_type": "error",
                "output": None,
                "latency_ms": 0.0,
                "error": f"Unknown pipeline: {pipeline_id}",
            }
            return

        start_time = time.monotonic()
        current_input: Any = input_text

        for idx, step in enumerate(spec.steps):
            step_start = time.monotonic()
            try:
                output = await self._run_step(step, current_input)
                step_latency = (time.monotonic() - step_start) * 1000

                if max_latency_ms and step_latency > max_latency_ms / len(spec.steps):
                    logger.warning(
                        f"Step {idx} ({step.step_type}/{step.model}) exceeded per-step SLA: "
                        f"{step_latency:.0f}ms > {max_latency_ms / len(spec.steps):.0f}ms"
                    )

                result = {
                    "step_index": idx,
                    "step_type": step.step_type.value,
                    "output": output,
                    "latency_ms": round(step_latency, 1),
                    "error": None,
                }
                yield result

                if step.step_type == StepType.generate:
                    # Final step: stream tokens if available
                    if isinstance(output, AsyncIterator):
                        async for token in output:
                            yield {
                                "step_index": idx,
                                "step_type": "token",
                                "output": token,
                                "latency_ms": 0.0,
                                "error": None,
                            }

                current_input = output

            except Exception as e:
                logger.error(f"Pipeline step {idx} failed: {e}")
                yield {
                    "step_index": idx,
                    "step_type": step.step_type.value,
                    "output": None,
                    "latency_ms": round((time.monotonic() - step_start) * 1000, 1),
                    "error": str(e),
                }
                return

        total_latency = (time.monotonic() - start_time) * 1000
        yield {
            "step_index": len(spec.steps),
            "step_type": "complete",
            "output": current_input,
            "latency_ms": round(total_latency, 1),
            "error": None,
        }

    async def _run_step(self, step: PipelineStep, input_data: Any) -> Any:
        """Run a single pipeline step."""
        coord = self._coordinator

        if step.step_type == StepType.embedding:
            if coord is None:
                raise RuntimeError("Coordinator required for embedding steps")
            embeddings = await coord.embed(input_data)
            return embeddings

        if step.step_type == StepType.reranker:
            if coord is None:
                raise RuntimeError("Coordinator required for reranker steps")
            result = await coord.rerank(input_data)
            return result

        if step.step_type == StepType.generate:
            if coord is None:
                raise RuntimeError("Coordinator required for generation steps")
            kwargs = dict(step.params)
            result = await coord.generate_async(
                prompt=input_data,
                **kwargs,
            )
            return result

        if step.step_type == StepType.transform:
            return self._apply_transform(input_data, step.params)

        raise ValueError(f"Unknown step type: {step.step_type}")

    def _apply_transform(self, data: Any, params: dict[str, Any]) -> Any:
        """Apply a simple text transform (truncation, concatenation, etc.)."""
        transform_type = params.get("type", "identity")
        if transform_type == "truncate":
            max_chars = params.get("max_chars", 4096)
            return data[:max_chars] if isinstance(data, str) else data
        if transform_type == "prepend":
            prefix = params.get("text", "")
            return prefix + str(data) if isinstance(data, str) else data
        if transform_type == "append":
            suffix = params.get("text", "")
            return str(data) + suffix if isinstance(data, str) else data
        return data
