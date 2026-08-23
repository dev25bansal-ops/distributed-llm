"""A/B testing framework for prompts and models.

Provides endpoints to create, manage, and evaluate experiments that split
traffic between two or more variants (prompt templates, model versions,
sampling parameters) and collect real-time metrics to determine the winner.

Each experiment has:
- A set of variants (each with a prompt template, model, or config override)
- A traffic split ratio per variant
- Success metrics (latency, token count, error rate)
- Optional auto-promotion when a variant is statistically significant

Usage::

    POST /v1/experiments
    {
        "name": "prompt-v2-test",
        "variants": [
            {"name": "control", "prompt_template": "Answer: {query}", "traffic_percent": 50},
            {"name": "v2", "prompt_template": "Please answer: {query}", "traffic_percent": 50}
        ],
        "metrics": ["latency", "tokens", "errors"],
        "min_samples": 100
    }
"""

from __future__ import annotations

import math
import threading
import time

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger
from pydantic import BaseModel, Field

from ..auth_deps import require_coordinator

router = APIRouter(prefix="/v1/experiments", tags=["experiments"], dependencies=[Depends(require_coordinator)])


# ── Pydantic models ──────────────────────────────────────────────────────


class VariantConfig(BaseModel):
    name: str = Field(..., description="Variant name (e.g. 'control', 'v2')")
    prompt_template: str = Field("", description="Prompt template with {placeholders}")
    model: str = Field("", description="Model override (empty = use default router)")
    temperature: float | None = Field(default=None, ge=0, le=2.0)
    top_p: float | None = Field(default=None, ge=0, le=1.0)
    system_prompt: str | None = Field(default=None, description="System prompt override")
    traffic_percent: int = Field(..., ge=0, le=100, description="Percentage of experiment traffic for this variant")


class ExperimentCreate(BaseModel):
    name: str = Field(..., description="Experiment name")
    description: str = Field(default="")
    variants: list[VariantConfig] = Field(..., min_length=2, max_length=10, description="At least 2 variants")
    metrics: list[str] = Field(default=["latency", "tokens", "errors"], description="Metrics to track")
    min_samples: int = Field(default=100, ge=10, description="Minimum samples before considering promotion")
    auto_promote: bool = Field(default=False, description="Auto-promote the winning variant")


class VariantStatus(BaseModel):
    name: str
    traffic_percent: int
    samples: int = 0
    avg_latency_ms: float = 0.0
    avg_tokens: float = 0.0
    error_rate: float = 0.0
    confidence: float = 0.0  # 0.0 = not enough data


class ExperimentStatus(BaseModel):
    name: str
    description: str = ""
    active: bool
    variants: list[VariantStatus]
    total_samples: int = 0
    winner: str | None = None
    created_at: float = 0.0


class SampleResult(BaseModel):
    variant: str
    latency_ms: float
    tokens: int
    error: bool = False


# ── In-memory experiment store ──────────────────────────────────────────

_experiments: dict[str, dict] = {}
_experiments_lock = threading.Lock()


def _record_sample(experiment_name: str, result: SampleResult) -> None:
    """Record a single sample for an experiment variant."""
    with _experiments_lock:
        exp = _experiments.get(experiment_name)
        if exp is None or not exp.get("active", False):
            return
        variants = exp.get("_variants", {})
        vdata = variants.get(result.variant)
        if vdata is None:
            return
        n = vdata["samples"]
        vdata["samples"] = n + 1
        vdata["latency_sum"] = vdata.get("latency_sum", 0.0) + result.latency_ms
        vdata["token_sum"] = vdata.get("token_sum", 0) + result.tokens
        vdata["error_count"] = vdata.get("error_count", 0) + (1 if result.error else 0)
        vdata["avg_latency_ms"] = vdata["latency_sum"] / vdata["samples"]
        vdata["avg_tokens"] = vdata["token_sum"] / vdata["samples"]
        vdata["error_rate"] = vdata["error_count"] / vdata["samples"]
        exp["total_samples"] += 1

        # Check for auto-promotion
        if exp["auto_promote"] and exp["total_samples"] >= exp["min_samples"]:
            _evaluate_winner(exp)


def _evaluate_winner(exp: dict) -> str | None:
    """Determine the winning variant by lowest average latency (with enough samples)."""
    variants = exp.get("_variants", {})
    best = None
    best_latency = float("inf")
    for name, vdata in variants.items():
        if vdata["samples"] >= max(10, exp["min_samples"] // len(variants)):
            if vdata["avg_latency_ms"] < best_latency:
                best_latency = vdata["avg_latency_ms"]
                best = name
    if best:
        exp["winner"] = best
    return best


def resolve_variant(experiment_name: str, request_id: str) -> str | None:
    """Return the variant name for *experiment_name* based on traffic split.

    Deterministic: same (experiment, request_id) always gets the same variant
    so retries don't flip between groups.
    """
    with _experiments_lock:
        exp = _experiments.get(experiment_name)
        if exp is None or not exp.get("active", False):
            return None
        variants = exp.get("_variants", {})
        if not variants:
            return None
        # Deterministic hash-based assignment
        h = int(hashlib.md5(f"{experiment_name}:{request_id}".encode()).hexdigest(), 16)
        pct = h % 100
        cumulative = 0
        for name, vdata in sorted(variants.items()):
            cumulative += vdata.get("traffic_percent", 0)
            if pct < cumulative:
                return name
        return list(variants.keys())[-1]


import hashlib  # noqa: E402 — needed above

# ── Endpoints ────────────────────────────────────────────────────────────


@router.post("", response_model=ExperimentStatus)
async def create_experiment(body: ExperimentCreate) -> ExperimentStatus:
    """Create a new A/B experiment."""
    total_pct = sum(v.traffic_percent for v in body.variants)
    if total_pct != 100:
        raise HTTPException(status_code=400, detail=f"Traffic percentages must sum to 100 (got {total_pct})")
    if body.name in _experiments:
        raise HTTPException(status_code=409, detail=f"Experiment '{body.name}' already exists")

    variants = {}
    for v in body.variants:
        variants[v.name] = {
            "name": v.name,
            "traffic_percent": v.traffic_percent,
            "prompt_template": v.prompt_template,
            "model": v.model,
            "temperature": v.temperature,
            "top_p": v.top_p,
            "system_prompt": v.system_prompt,
            "samples": 0,
            "latency_sum": 0.0,
            "token_sum": 0,
            "error_count": 0,
            "avg_latency_ms": 0.0,
            "avg_tokens": 0.0,
            "error_rate": 0.0,
        }

    exp = {
        "name": body.name,
        "description": body.description,
        "active": True,
        "variants": body.model_dump()["variants"],
        "_variants": variants,
        "metrics": body.metrics,
        "min_samples": body.min_samples,
        "auto_promote": body.auto_promote,
        "total_samples": 0,
        "winner": None,
        "created_at": time.time(),
    }
    with _experiments_lock:
        _experiments[body.name] = exp
    logger.info(f"Experiment created: {body.name} ({len(body.variants)} variants)")
    return _to_status(exp)


@router.get("", response_model=list[ExperimentStatus])
async def list_experiments() -> list[ExperimentStatus]:
    """List all experiments with current status."""
    return [_to_status(e) for e in _experiments.values()]


@router.get("/{name}", response_model=ExperimentStatus)
async def get_experiment(name: str) -> ExperimentStatus:
    """Get experiment details and current results."""
    exp = _experiments.get(name)
    if exp is None:
        raise HTTPException(status_code=404, detail=f"Experiment '{name}' not found")
    return _to_status(exp)


@router.patch("/{name}/stop")
async def stop_experiment(name: str):
    """Stop an experiment (freeze traffic and declare winner)."""
    with _experiments_lock:
        exp = _experiments.get(name)
        if exp is None:
            raise HTTPException(status_code=404, detail=f"Experiment '{name}' not found")
        exp["active"] = False
        _evaluate_winner(exp)
    return {"status": "stopped", "winner": exp.get("winner")}


@router.delete("/{name}")
async def delete_experiment(name: str):
    """Delete an experiment and its data."""
    with _experiments_lock:
        if name not in _experiments:
            raise HTTPException(status_code=404, detail=f"Experiment '{name}' not found")
        del _experiments[name]
    return {"status": "deleted"}


# ── Internal sample ingestion (for middleware/reporting use) ─────────────


@router.post("/sample")
async def record_sample(body: SampleResult, experiment: str = Query(..., description="Experiment name")):
    """Record a single sample for an experiment."""
    _record_sample(experiment, body)
    return {"status": "recorded"}


# ── Helpers ─────────────────────────────────────────────────────────────


def _to_status(exp: dict) -> ExperimentStatus:
    variants_status = []
    for vname, vdata in exp.get("_variants", {}).items():
        confidence = _calculate_confidence(vdata, exp.get("_variants", {}))
        variants_status.append(VariantStatus(
            name=vname,
            traffic_percent=vdata.get("traffic_percent", 0),
            samples=vdata.get("samples", 0),
            avg_latency_ms=vdata.get("avg_latency_ms", 0.0),
            avg_tokens=vdata.get("avg_tokens", 0.0),
            error_rate=vdata.get("error_rate", 0.0),
            confidence=confidence,
        ))
    return ExperimentStatus(
        name=exp["name"],
        description=exp.get("description", ""),
        active=exp.get("active", True),
        variants=variants_status,
        total_samples=exp.get("total_samples", 0),
        winner=exp.get("winner"),
        created_at=exp.get("created_at", 0.0),
    )


def _calculate_confidence(vdata: dict, all_variants: dict) -> float:
    """Return a naive confidence score (0.0–1.0) based on sample size."""
    samples = vdata.get("samples", 0)
    if samples < 10:
        return 0.0
    # Simple heuristic: confidence grows with sqrt(samples), capped at 0.95
    return min(0.95, math.sqrt(samples) / 100)
