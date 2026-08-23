"""LLM Evaluation API routes.

Provides endpoints for running and querying evaluation benchmarks
(MMLU, GSM8K, HumanEval, MT-Bench, Arena) via POST /api/v1/eval/run
and GET /api/v1/eval/results.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from loguru import logger

from ..api_state import g
from ..auth_deps import require_role
from distllm.core.evaluation_harness import EvalRunner, EvalReport


router = APIRouter(prefix="/api/v1/eval", tags=["evaluation"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class EvalRunRequest(BaseModel):
    """Request to run an evaluation benchmark."""
    model_id: str = Field(..., description="Model identifier to evaluate")
    benchmarks: list[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Benchmarks to run: mmlu, gsm8k, humaneval, mt_bench, arena",
    )
    coordinator_url: str = Field(
        default="",
        description="Remote coordinator URL. Uses local coordinator if empty.",
    )
    max_tokens: int = Field(default=256, ge=1, le=4096, description="Max generation tokens")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Sampling temperature")
    num_samples: int = Field(default=20, ge=1, le=500, description="Samples per benchmark")

    # Arena-specific
    model_b: str = Field(default="", description="Second model for arena comparison")
    coordinator_url_b: str = Field(default="", description="URL for model B")


class EvalRunResponse(BaseModel):
    """Response from a benchmark run."""
    success: bool
    reports: dict[str, Any] = Field(default_factory=dict, description="Benchmark results keyed by name")
    error: str | None = None


class EvalResultsQuery(BaseModel):
    """Query parameters for listing evaluation reports."""
    model_id: str | None = None
    dataset: str | None = None
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)


class EvalResultsResponse(BaseModel):
    """Response listing evaluation reports."""
    success: bool
    reports: list[dict[str, Any]] = Field(default_factory=list)
    total: int = 0


# ---------------------------------------------------------------------------
# In-memory runner cache
# ---------------------------------------------------------------------------

_runner: EvalRunner | None = None
_runner_lock = threading.Lock()


def _get_runner() -> EvalRunner:
    """Get or create the shared EvalRunner instance."""
    global _runner
    if _runner is None:
        with _runner_lock:
            if _runner is None:
                coordinator = g.coordinator
                _runner = EvalRunner(coordinator=coordinator)
    return _runner


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/run",
    summary="Run evaluation benchmarks",
    description="Execute one or more evaluation benchmarks (MMLU, GSM8K, HumanEval, MT-Bench, Arena) "
                "and return aggregated metrics. Results are persisted in the evaluation database.",
    responses={
        200: {"description": "Benchmark results"},
        400: {"description": "Invalid benchmark name"},
        503: {"description": "Coordinator not available"},
    },
)
async def eval_run(request: EvalRunRequest) -> EvalRunResponse:
    """Run evaluation benchmarks for the specified model."""
    valid_benchmarks = {"mmlu", "gsm8k", "humaneval", "mt_bench", "arena"}
    for b in request.benchmarks:
        if b not in valid_benchmarks:
            return EvalRunResponse(
                success=False,
                error=f"Unknown benchmark '{b}'. Valid: {sorted(valid_benchmarks)}",
            )

    runner = _get_runner()
    reports: dict[str, Any] = {}

    try:
        for benchmark in request.benchmarks:
            logger.info("API eval run: benchmark={}, model={}", benchmark, request.model_id)

            if benchmark == "mt_bench":
                report = runner.run_mt_bench(
                    model_id=request.model_id,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    coordinator_url=request.coordinator_url,
                    num_categories=min(request.num_samples, 8),
                )
            elif benchmark == "arena":
                if not request.model_b:
                    return EvalRunResponse(
                        success=False,
                        error="arena benchmark requires model_b parameter",
                    )
                report = runner.run_arena(
                    model_a=request.model_id,
                    model_b=request.model_b,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    coordinator_url_a=request.coordinator_url,
                    coordinator_url_b=request.coordinator_url_b,
                    num_samples=request.num_samples,
                )
            else:
                report = runner.run_heim(
                    benchmark=benchmark,
                    model_id=request.model_id,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    coordinator_url=request.coordinator_url,
                    num_samples=request.num_samples,
                )

            reports[benchmark] = _report_to_dict(report)

        return EvalRunResponse(success=True, reports=reports)

    except Exception as exc:
        logger.error("Evaluation run failed: {}", exc)
        return EvalRunResponse(success=False, error=str(exc))


@router.get(
    "/results",
    summary="List evaluation results",
    description="List previously run evaluation reports with optional filtering by model and dataset.",
)
async def eval_results(
    model_id: str | None = Query(None, description="Filter by model ID"),
    dataset: str | None = Query(None, description="Filter by dataset name"),
    limit: int = Query(50, ge=1, le=500, description="Max results"),
    offset: int = Query(0, ge=0, description="Result offset"),
) -> EvalResultsResponse:
    """List evaluation reports."""
    runner = _get_runner()
    reports = runner.list_reports(
        model_id=model_id,
        dataset=dataset,
        limit=limit,
        offset=offset,
    )
    return EvalResultsResponse(success=True, reports=reports, total=len(reports))


@router.get(
    "/results/{report_id}",
    summary="Get evaluation report details",
    description="Retrieve a specific evaluation report with its metrics and results.",
)
async def eval_result_detail(report_id: str) -> dict[str, Any]:
    """Get detailed evaluation report by ID."""
    runner = _get_runner()
    report = runner.get_report(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
    results = runner.get_report_results(report_id)
    return {
        "success": True,
        "report": dict(report),
        "results": results,
    }


@router.delete(
    "/results/{report_id}",
    summary="Delete evaluation report",
    description="Delete an evaluation report and its results.",
    dependencies=[Depends(require_role("admin"))],
)
async def eval_result_delete(report_id: str) -> dict[str, Any]:
    """Delete an evaluation report."""
    runner = _get_runner()
    deleted = runner.delete_report(report_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
    return {"success": True, "deleted": report_id}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _report_to_dict(report: EvalReport) -> dict[str, Any]:
    """Convert an EvalReport to a JSON-serializable dict."""
    return {
        "report_id": report.report_id,
        "model_id": report.model_id,
        "dataset": report.dataset,
        "status": report.status.value,
        "metrics": report.metrics,
        "config": report.config,
        "num_results": len(report.results),
        "created_at": report.created_at,
        "duration_s": report.duration_s,
    }
