"""Pipeline composition routes: POST /v1/pipeline and related endpoints."""

import time
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..api_state import g

router = APIRouter(tags=["pipeline"])


class PipelineStepRequest(BaseModel):
    model: str = Field(..., description="Model identifier for this step")
    step_type: str = Field(..., description="Step type: embedding, reranker, generate, transform")
    params: dict = Field(default_factory=dict, description="Optional step parameters")
    timeout_ms: float | None = Field(default=None, description="Per-step timeout")


class PipelineRequest(BaseModel):
    pipeline_id: str | None = Field(default=None, description="Registered pipeline ID")
    steps: list[PipelineStepRequest] | None = Field(
        default=None, description="Inline pipeline steps (used when pipeline_id is not set)"
    )
    input: str = Field(..., description="Input text for the pipeline")
    max_latency_ms: float | None = Field(default=None, description="SLA latency threshold")
    stream: bool = Field(default=False, description="Whether to stream step results")


class PipelineStepResponse(BaseModel):
    step_index: int
    step_type: str
    output: str | list | None = None
    latency_ms: float = 0.0
    error: str | None = None


class PipelineResponse(BaseModel):
    id: str
    pipeline_id: str
    steps: list[PipelineStepResponse]
    total_latency_ms: float = 0.0
    error: str | None = None


@router.post(
    "/v1/pipeline",
    summary="Execute pipeline",
    description="Execute a composed pipeline of models. Supports inline pipeline steps or a registered pipeline ID. Each step runs sequentially and outputs are passed to subsequent steps. Supports SLA latency thresholds and streaming of step results.",
    response_description="Pipeline execution results with per-step latency",
    responses={
        503: {"description": "No coordinator available or pipeline composer not initialized"},
    },
)
async def run_pipeline(body: PipelineRequest):
    """Execute a composed pipeline of models."""
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="No coordinator available")

    if coord._pipeline_composer is None:
        raise HTTPException(status_code=503, detail="Pipeline composer not initialized")

    pipeline_id = body.pipeline_id or f"inline-{uuid4().hex[:8]}"

    # Build step results
    step_results: list[PipelineStepResponse] = []
    total_start = time.monotonic()

    try:
        async for result in coord._pipeline_composer.execute(
            pipeline_id=pipeline_id,
            input_text=body.input,
            max_latency_ms=body.max_latency_ms,
        ):
            if result["step_type"] == "complete":
                total_latency = result["latency_ms"]
                continue
            step_results.append(PipelineStepResponse(
                step_index=result["step_index"],
                step_type=result["step_type"],
                output=result["output"],
                latency_ms=result["latency_ms"],
                error=result["error"],
            ))
    except Exception as e:
        return PipelineResponse(
            id=f"pipe-{uuid4().hex[:12]}",
            pipeline_id=pipeline_id,
            steps=step_results,
            total_latency_ms=round((time.monotonic() - total_start) * 1000, 1),
            error=str(e),
        )

    return PipelineResponse(
        id=f"pipe-{uuid4().hex[:12]}",
        pipeline_id=pipeline_id,
        steps=step_results,
        total_latency_ms=total_latency,
    )


class PipelineRegisterRequest(BaseModel):
    pipeline_id: str = Field(..., description="Unique pipeline identifier")
    steps: list[PipelineStepRequest] = Field(..., description="Ordered list of pipeline steps")
    fallback_pipeline_id: str | None = Field(default=None, description="Fallback pipeline ID")


class PipelineRegisterResponse(BaseModel):
    pipeline_id: str
    status: str = "registered"
    steps_count: int = 0


@router.post(
    "/v1/pipeline/register",
    summary="Register pipeline",
    description="Register a reusable pipeline specification with an ID. Pipelines can be executed later by referencing their ID, enabling pre-configured model chains for common workflows.",
    response_description="Pipeline registration confirmation",
    responses={
        503: {"description": "No coordinator available"},
    },
)
async def register_pipeline(body: PipelineRegisterRequest):
    """Register a reusable pipeline specification."""
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="No coordinator available")

    if coord._pipeline_composer is None:
        from distllm.core.pipeline_composer import PipelineExecutor, PipelineSpec, PipelineStep, StepType
        coord._pipeline_composer = PipelineExecutor(coordinator=coord)

    from distllm.core.pipeline_composer import PipelineSpec, PipelineStep, StepType

    spec = PipelineSpec(
        pipeline_id=body.pipeline_id,
        steps=[
            PipelineStep(
                model=s.model,
                step_type=StepType(s.step_type),
                params=s.params,
                timeout_ms=s.timeout_ms,
            )
            for s in body.steps
        ],
        fallback_pipeline_id=body.fallback_pipeline_id,
    )
    coord._pipeline_composer.register(spec)

    return PipelineRegisterResponse(
        pipeline_id=body.pipeline_id,
        status="registered",
        steps_count=len(body.steps),
    )


@router.get(
    "/v1/pipeline/{pipeline_id}",
    summary="Get pipeline spec",
    description="Retrieve a registered pipeline specification by its ID, including its ordered steps, parameters, and optional fallback pipeline configuration.",
    response_description="Pipeline specification with steps",
    responses={
        404: {"description": "Pipeline not found"},
        503: {"description": "No coordinator available or pipeline composer not initialized"},
    },
)
async def get_pipeline(pipeline_id: str):
    """Get a registered pipeline specification."""
    coord = g.coordinator
    if coord is None or coord._pipeline_composer is None:
        raise HTTPException(status_code=503, detail="Pipeline composer not initialized")

    spec = coord._pipeline_composer.get(pipeline_id)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"Pipeline not found: {pipeline_id}")

    return {
        "pipeline_id": spec.pipeline_id,
        "steps": [
            {
                "model": s.model,
                "step_type": s.step_type.value,
                "params": s.params,
                "timeout_ms": s.timeout_ms,
            }
            for s in spec.steps
        ],
        "fallback_pipeline_id": spec.fallback_pipeline_id,
    }
