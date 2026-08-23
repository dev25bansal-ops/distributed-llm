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
from ..services.eval_service import EvalService
from distllm.core.evaluation_harness import EvalRunner, EvalReport


router = APIRouter(prefix="/v1/eval", tags=["evaluation"])


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
# SSRF protection
# ---------------------------------------------------------------------------


def _is_safe_coordinator_url(url: str) -> bool:
    """True when *url* does not target loopback/private/link-local networks."""
    import ipaddress as _ip
    from urllib.parse import urlparse as _urlparse

    try:
        parsed = _urlparse(url)
        host = parsed.hostname or ""
    except ValueError:
        return False
    if not host:
        return False
    if host in ("localhost",) or host.endswith(".localhost"):
        return False
    try:
        ip = _ip.ip_address(host)
    except ValueError:
        return True  # hostname (not a literal IP): DNS-level check skipped
    return not (
        ip.is_private or ip.is_loopback or ip.is_link_local
        or ip.is_reserved or ip.is_multicast
    )


# ---------------------------------------------------------------------------
# Service access
# ---------------------------------------------------------------------------


def _get_service() -> EvalService:
    """Build an EvalService for this request.

    Constructed per call (cheap — the runner is lazy) so test suites can
    swap the module-level ``EvalService`` symbol for a stub subclass.
    """
    return EvalService(coordinator=g.coordinator)


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

    # SSRF guard: coordinator URLs are fetched server-side, so loopback /
    # private-network targets must be rejected before any request is made.
    for label, url in (
        ("coordinator_url", request.coordinator_url),
        ("coordinator_url_b", request.coordinator_url_b),
    ):
        if url and not _is_safe_coordinator_url(url):
            return EvalRunResponse(
                success=False,
                error=(
                    f"Invalid {label} '{url}': host is loopback/private or "
                    "unresolvable — SSRF protection"
                ),
            )

    service = _get_service()
    try:
        results = service.run_benchmarks(
            model_id=request.model_id,
            benchmarks=request.benchmarks,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            num_samples=request.num_samples,
            coordinator_url=request.coordinator_url,
            model_b=request.model_b or "",
            coordinator_url_b=request.coordinator_url_b,
        )
        reports = {b: r for b, r in (results or {}).items()}
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
    service = _get_service()
    reports = service.list_reports(
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
    service = _get_service()
    report = service.get_report(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
    results = service.get_report_results(report_id)
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
    service = _get_service()
    deleted = service.delete_report(report_id)
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
