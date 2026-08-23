"""Dynamic pipeline composition with latency SLOs for multi-step inference.

Provides a framework for composing, compiling, and executing multi-step
inference pipelines. Each pipeline consists of typed steps (embedding,
reranker, LLM, structured output) with configurable models, quantization,
per-step timeouts, and fallback strategies.

The :class:`PlanCompiler` predicts end-to-end latency and cost using
per-step-type models and can flag SLO violations. The :class:`PipelineExecutor`
runs compiled plans with ``asyncio.wait_for`` timeout enforcement and
automatic fallback on step failure.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass
class PipelineStep:
    """A single step within a multi-step inference pipeline.

    Attributes:
        name: Human-readable label for this step (e.g. ``"retrieve"``).
        step_type: The kind of inference to perform. One of ``"embedding"``,
            ``"reranker"``, ``"llm"``, or ``"structured_output"``.
        model: Model identifier string (e.g. ``"intfloat/e5-mistral-7b-instruct"``).
        quantization: Quantization method. ``"none"``, ``"fp16"``, ``"fp8"``,
            ``"int8"``, or ``"int4"``.
        timeout_s: Maximum wall-clock seconds for this step. The executor
            enforces this via ``asyncio.wait_for``.
        fallback: If the primary step times out or raises, the executor
            attempts a step of this type instead. ``None`` means no fallback.
    """

    name: str
    step_type: str  # "embedding" | "reranker" | "llm" | "structured_output"
    model: str
    quantization: str = "none"
    timeout_s: float = 30.0
    fallback: str | None = None


@dataclass
class PipelineSpec:
    """Specification for a complete inference pipeline.

    Attributes:
        steps: Ordered list of steps to execute sequentially.
        end_to_end_slo_ms: Maximum acceptable end-to-end latency in
            milliseconds. The :class:`PlanCompiler` compares predicted
            total latency against this threshold and emits a warning when
            the SLO cannot be met.
        max_cost: Maximum acceptable monetary cost for executing this
            pipeline. Defaults to infinity (no cost constraint).
    """

    steps: list[PipelineStep]
    end_to_end_slo_ms: float
    max_cost: float = float("inf")


@dataclass
class StepResult:
    """Result of a single pipeline step execution.

    Attributes:
        step_index: Zero-based position of this step in the pipeline.
        step_name: The ``PipelineStep.name`` value.
        step_type: The ``PipelineStep.step_type`` value.
        output: The step's output payload (type depends on the step type).
        latency_ms: Observed wall-clock time for this step, in milliseconds.
        cost: Estimated monetary cost for this step.
        error: Human-readable error description if the step failed, or
            ``None`` on success.
        fallback_used: ``True`` when the primary step failed and a fallback
            step type was executed instead.
    """

    step_index: int
    step_name: str
    step_type: str
    output: Any
    latency_ms: float
    cost: float
    error: str | None = None
    fallback_used: bool = False


@dataclass
class ExecutionPlan:
    """Compiled plan ready for execution by :class:`PipelineExecutor`.

    Attributes:
        spec: The original pipeline specification this plan was derived from.
        steps: Ordered list of ``(step, predicted_latency_ms, predicted_cost)``
            tuples. Each tuple corresponds to one step in the pipeline.
    """

    spec: PipelineSpec
    steps: list[tuple[PipelineStep, float, float]]


class PlanCompiler:
    """Compiles a :class:`PipelineSpec` into an :class:`ExecutionPlan`.

    The compiler uses per-step-type latency and cost models to predict
    end-to-end runtime and flag SLO violations. Quantization levels are
    factored into both latency and cost estimates.
    """

    _LATENCY_MODELS: dict[str, float] = {
        "embedding": 0.5,
        "reranker": 1.0,
        "llm": 15.0,
        "structured_output": 20.0,
    }

    _COST_MODELS: dict[str, float] = {
        "embedding": 0.000_1,
        "reranker": 0.000_5,
        "llm": 0.003,
        "structured_output": 0.004,
    }

    _QUANTIZATION_FACTORS: dict[str, float] = {
        "none": 1.0,
        "fp16": 1.0,
        "fp8": 0.85,
        "int8": 0.7,
        "int4": 0.5,
    }

    def compile(self, spec: PipelineSpec) -> ExecutionPlan:
        """Compile a pipeline specification into an executable plan.

        Each step is annotated with a predicted latency (in milliseconds)
        and a predicted cost. If the total predicted latency exceeds the
        specification's end-to-end SLO, a warning is logged.

        Args:
            spec: The pipeline specification to compile.

        Returns:
            An :class:`ExecutionPlan` containing the annotated step list.

        Raises:
            ValueError: If any step has an unknown ``step_type``.
        """
        steps: list[tuple[PipelineStep, float, float]] = []

        for step in spec.steps:
            if step.step_type not in self._LATENCY_MODELS:
                raise ValueError(
                    f"Unknown step_type '{step.step_type}' for step "
                    f"'{step.name}'. Valid types: {list(self._LATENCY_MODELS)}"
                )

            base_latency = self._LATENCY_MODELS[step.step_type]
            base_cost = self._COST_MODELS[step.step_type]
            quant_factor = self._quantization_factor(step.quantization)

            predicted_latency = base_latency * quant_factor
            predicted_cost = base_cost * quant_factor

            steps.append((step, predicted_latency, predicted_cost))

        total_predicted_ms = sum(lat for _, lat, _ in steps)

        if total_predicted_ms > spec.end_to_end_slo_ms:
            logger.warning(
                "Pipeline '%s' predicted latency %.1fms exceeds SLO %.1fms by %.1fms",
                spec,
                total_predicted_ms,
                spec.end_to_end_slo_ms,
                total_predicted_ms - spec.end_to_end_slo_ms,
            )

        total_predicted_cost = sum(cost for _, _, cost in steps)
        if total_predicted_cost > spec.max_cost:
            logger.warning(
                "Pipeline predicted cost %.6f exceeds max_cost %.6f",
                total_predicted_cost,
                spec.max_cost,
            )

        return ExecutionPlan(spec=spec, steps=steps)

    @staticmethod
    def _quantization_factor(quantization: str) -> float:
        """Return a latency/cost multiplier for a given quantization string.

        More aggressive quantization reduces compute, lowering both latency
        and the estimated cost per-token.

        Args:
            quantization: Quantization identifier (``"none"``, ``"fp16"``,
                ``"fp8"``, ``"int8"``, ``"int4"``).

        Returns:
            A multiplier where values < 1.0 indicate faster/cheaper execution.
        """
        return PlanCompiler._QUANTIZATION_FACTORS.get(quantization, 1.0)


class PipelineExecutor:
    """Executes compiled :class:`ExecutionPlan` s with timeout and fallback.

    Each pipeline step is dispatched to the ``coordinator_client``, with
    per-step timeout enforced via ``asyncio.wait_for``. If a step fails
    (timeout or exception) and a fallback step type is configured, the
    executor transparently retries the step with the fallback type.

    Args:
        coordinator_client: An object exposing async methods ``embed``,
            ``rerank``, ``generate``, ``generate_structured``, and
            ``generate_stream``. Typically a coordinator or SDK client
            instance.
    """

    def __init__(self, coordinator_client: Any) -> None:
        self._coordinator = coordinator_client

    async def execute(
        self,
        plan: ExecutionPlan,
        input_data: Any,
    ) -> list[StepResult]:
        """Execute a compiled pipeline plan.

        Steps are run sequentially. Each step's output becomes the next
        step's input. If any step fails and has no fallback, execution
        stops and the error is recorded in that step's :class:`StepResult`.

        Args:
            plan: The execution plan to run.
            input_data: The initial input payload for the first step.

        Returns:
            A list of :class:`StepResult` objects, one per pipeline step
            (plus any token-level results in streaming mode).
        """
        results: list[StepResult] = []
        current_input: Any = input_data

        for idx, (step, pred_latency, pred_cost) in enumerate(plan.steps):
            step_start = time.monotonic()
            error: str | None = None
            fallback_used = False
            output: Any = None

            try:
                output = await asyncio.wait_for(
                    self._run_step(step, current_input),
                    timeout=step.timeout_s,
                )
            except asyncio.TimeoutError:
                error = f"Step '{step.name}' timed out after {step.timeout_s}s"
                logger.warning(error)
                if step.fallback:
                    fallback_used, output = await self._attempt_fallback(
                        step, current_input, idx,
                    )
                    if fallback_used:
                        error = None
            except Exception as exc:
                error = str(exc)
                logger.error("Step '{}' failed: {}", step.name, error)
                if step.fallback:
                    fallback_used, output = await self._attempt_fallback(
                        step, current_input, idx,
                    )
                    if fallback_used:
                        error = None

            step_latency = (time.monotonic() - step_start) * 1000

            results.append(
                StepResult(
                    step_index=idx,
                    step_name=step.name,
                    step_type=step.step_type,
                    output=output,
                    latency_ms=round(step_latency, 1),
                    cost=pred_cost,
                    error=error,
                    fallback_used=fallback_used,
                ),
            )

            # Stop on unrecoverable error
            if error:
                break

            current_input = output

        return results

    async def execute_stream(
        self,
        plan: ExecutionPlan,
        input_data: Any,
    ) -> AsyncIterator[StepResult]:
        """Execute a pipeline plan and yield results as each step completes.

        For LLM and structured-output steps, individual tokens are yielded
        as ``StepResult`` objects with ``step_type="token"`` before the
        final accumulated result for that step. This allows callers to
        begin consuming output before the entire pipeline finishes.

        Args:
            plan: The execution plan to run.
            input_data: The initial input payload for the first step.

        Yields:
            :class:`StepResult` instances -- one per completed step,
            plus intermediate token results for streaming-capable steps.
        """
        current_input: Any = input_data

        for idx, (step, pred_latency, pred_cost) in enumerate(plan.steps):
            step_start = time.monotonic()
            error: str | None = None
            fallback_used = False
            output: Any = None

            try:
                if step.step_type in ("llm", "structured_output"):
                    # Streaming path: yield tokens as they arrive
                    collected: list[str] = []
                    async for token in self._stream_step(step, current_input):
                        collected.append(token)
                        yield StepResult(
                            step_index=idx,
                            step_name=step.name,
                            step_type="token",
                            output=token,
                            latency_ms=0.0,
                            cost=0.0,
                            error=None,
                        )
                    output = "".join(collected) if collected else ""
                else:
                    # Non-streaming path: run with timeout
                    output = await asyncio.wait_for(
                        self._run_step(step, current_input),
                        timeout=step.timeout_s,
                    )
            except asyncio.TimeoutError:
                error = f"Step '{step.name}' timed out after {step.timeout_s}s"
                logger.warning(error)
                if step.fallback:
                    fallback_used, output = await self._attempt_fallback(
                        step, current_input, idx,
                    )
                    if fallback_used:
                        error = None
            except Exception as exc:
                error = str(exc)
                logger.error("Step '{}' failed: {}", step.name, error)
                if step.fallback:
                    fallback_used, output = await self._attempt_fallback(
                        step, current_input, idx,
                    )
                    if fallback_used:
                        error = None

            step_latency = (time.monotonic() - step_start) * 1000

            yield StepResult(
                step_index=idx,
                step_name=step.name,
                step_type=step.step_type,
                output=output,
                latency_ms=round(step_latency, 1),
                cost=pred_cost,
                error=error,
                fallback_used=fallback_used,
            )

            if error:
                break

            current_input = output

    async def _run_step(self, step: PipelineStep, input_data: Any) -> Any:
        """Dispatch a single pipeline step to the coordinator client.

        Args:
            step: The step to execute.
            input_data: Data to pass as input to the step.

        Returns:
            The step's output.

        Raises:
            ValueError: If ``step.step_type`` is not recognized.
            RuntimeError: If the coordinator client is not available.
        """
        coord = self._coordinator
        st = step.step_type

        if st == "embedding":
            return await coord.embed(input_data, model=step.model)
        if st == "reranker":
            return await coord.rerank(input_data, model=step.model)
        if st == "llm":
            return await coord.generate(prompt=input_data, model=step.model)
        if st == "structured_output":
            return await coord.generate_structured(
                prompt=input_data, model=step.model,
            )

        raise ValueError(f"Unknown step type: '{st}'")

    async def _stream_step(
        self,
        step: PipelineStep,
        input_data: Any,
    ) -> AsyncIterator[str]:
        """Stream tokens from an LLM or structured-output step.

        Args:
            step: The streaming-capable step to execute.
            input_data: The prompt to pass to the model.

        Yields:
            Individual text tokens as they are generated.
        """
        coord = self._coordinator
        async for token in coord.generate_stream(
            prompt=input_data, model=step.model,
        ):
            yield token

    async def _attempt_fallback(
        self,
        failed_step: PipelineStep,
        input_data: Any,
        idx: int,
    ) -> tuple[bool, Any | None]:
        """Try a fallback step type after the primary step fails.

        The fallback is constructed as a new :class:`PipelineStep` with the
        fallback ``step_type``, the same model and quantization, and a
        more generous timeout (1.5x the original) to reduce the likelihood
        of cascading timeouts.

        Args:
            failed_step: The step that failed.
            input_data: Input data to pass to the fallback.
            idx: Step index (used for logging only).

        Returns:
            A ``(success, output)`` tuple. ``success`` is ``True`` when the
            fallback completed without error. ``output`` is the fallback
            result (or ``None`` on failure).
        """
        assert failed_step.fallback is not None  # caller guard

        fallback_step = PipelineStep(
            name=f"{failed_step.name}_fallback",
            step_type=failed_step.fallback,
            model=failed_step.model,
            quantization=failed_step.quantization,
            timeout_s=failed_step.timeout_s * 1.5,
        )
        logger.info(
            "Step '{}' (idx={}) falling back to '{}'",
            failed_step.name,
            idx,
            failed_step.fallback,
        )

        try:
            output = await asyncio.wait_for(
                self._run_step(fallback_step, input_data),
                timeout=fallback_step.timeout_s,
            )
            return True, output
        except asyncio.TimeoutError:
            logger.error(
                "Fallback for step '{}' also timed out after {}s",
                failed_step.name,
                fallback_step.timeout_s,
            )
            return False, None
        except Exception as exc:
            logger.error(
                "Fallback for step '{}' also failed: {}",
                failed_step.name,
                exc,
            )
            return False, None
