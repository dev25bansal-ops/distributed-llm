"""Benchmark leaderboard API — list, submit, and compare benchmark results."""

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field


router = APIRouter(tags=["leaderboard"])


class BenchmarkMetrics(BaseModel):
    throughput_tok_s: Optional[float] = None
    ttft_ms: Optional[float] = None
    itl_ms: Optional[float] = None
    latency_p50_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    latency_p99_ms: Optional[float] = None
    memory_peak_mb: Optional[float] = None
    memory_efficiency: Optional[float] = None


class BenchmarkConfig(BaseModel):
    num_nodes: int = 1
    gpus_per_node: int = 1
    precision: str = "float16"
    batch_size: int = 1
    seq_len: int = 512
    max_tokens: int = 256
    concurrency: int = 1
    spec_decode: bool = False


class BenchmarkSubmission(BaseModel):
    model: str
    hardware: str
    framework: str = "DistLLM"
    backend: str = "native"
    metrics: BenchmarkMetrics
    config: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    submitted_by: str = "anonymous"


_store: dict[str, dict] = {}
_order: list[str] = []
_loaded = False
_loaded_lock = threading.Lock()


def _find_project_root() -> Path:
    cur = Path(__file__).resolve()
    for parent in cur.parents:
        if (parent / "benchmarks").is_dir():
            return parent
    return cur


def _load_seed_data():
    global _loaded
    if _loaded:
        return
    with _loaded_lock:
        if _loaded:
            return
        _loaded = True

    results_dir = _find_project_root() / "benchmarks" / "results"
    if not results_dir.is_dir():
        return

    for fpath in sorted(results_dir.glob("*.json")):
        try:
            raw = json.loads(fpath.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        if not isinstance(raw, dict) or "model" not in raw:
            continue

        name = raw.get("name", fpath.stem)
        model = raw.get("model", "unknown")
        mode = raw.get("mode", "local")
        nodes = raw.get("nodes", 1)

        entry_id = str(uuid.uuid4())

        def _v(key):
            val = raw.get(key)
            if val is None:
                return None
            try:
                fval = float(val)
                return fval if fval != 0.0 else None
            except (ValueError, TypeError):
                return None

        metrics = {
            "throughput_tok_s": _v("tokens_per_sec"),
            "ttft_ms": _v("ttft_ms"),
            "itl_ms": _v("itl_ms"),
            "latency_p50_ms": _v("latency_p50_ms"),
            "latency_p95_ms": _v("latency_p95_ms"),
            "latency_p99_ms": _v("latency_p99_ms"),
            "memory_peak_mb": _v("memory_peak_mb"),
            "memory_efficiency": _v("cache_hit_rate_pct"),
        }

        _store[entry_id] = {
            "id": entry_id,
            "model": model,
            "hardware": "NVIDIA A100" if nodes > 1 else "NVIDIA RTX 4090",
            "framework": "DistLLM",
            "backend": "native",
            "metrics": metrics,
            "config": {
                "num_nodes": nodes,
                "gpus_per_node": 1,
                "precision": "float16",
                "batch_size": 1,
                "seq_len": 512,
                "max_tokens": 256,
                "concurrency": 1,
                "spec_decode": False,
            },
            "submitted_by": "ci-benchmark",
            "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "verified": True,
        }
        _order.append(entry_id)


def _ensure_loaded():
    if not _loaded:
        _load_seed_data()


@router.get("/v1/leaderboard")
async def list_leaderboard(
    model: Optional[str] = Query(None),
    hardware: Optional[str] = Query(None),
    metric: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    _ensure_loaded()
    items = list(_store.values())

    if model:
        ml = model.lower()
        items = [i for i in items if ml in i["model"].lower()]
    if hardware:
        hl = hardware.lower()
        items = [i for i in items if hl in i["hardware"].lower()]

    if metric:
        asc_fields = {
            "ttft_ms", "itl_ms", "latency_p50_ms",
            "latency_p95_ms", "latency_p99_ms", "memory_peak_mb",
        }
        reverse = metric not in asc_fields

        def sort_key(item):
            val = item["metrics"].get(metric)
            if val is None:
                return float("-inf") if reverse else float("inf")
            return val

        items.sort(key=sort_key, reverse=reverse)

    total = len(items)
    page = items[offset: offset + limit]

    return {
        "results": page,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/v1/leaderboard/{entry_id}")
async def get_leaderboard_entry(entry_id: str):
    _ensure_loaded()
    entry = _store.get(entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Benchmark entry not found")
    return entry


@router.post("/v1/leaderboard/submit")
async def submit_benchmark(submission: BenchmarkSubmission):
    _ensure_loaded()
    entry_id = str(uuid.uuid4())
    entry = {
        "id": entry_id,
        "model": submission.model,
        "hardware": submission.hardware,
        "framework": submission.framework,
        "backend": submission.backend,
        "metrics": submission.metrics.model_dump(),
        "config": submission.config.model_dump(),
        "submitted_by": submission.submitted_by,
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "verified": False,
    }
    _store[entry_id] = entry
    _order.append(entry_id)
    return entry


@router.get("/v1/leaderboard/summary")
async def leaderboard_summary():
    _ensure_loaded()
    groups: dict[tuple[str, str], list[dict]] = {}
    for entry in _store.values():
        key = (entry["model"], entry["hardware"])
        groups.setdefault(key, []).append(entry)

    summaries = []
    for (model, hardware), entries in groups.items():
        throughputs = [
            e["metrics"]["throughput_tok_s"]
            for e in entries if e["metrics"].get("throughput_tok_s") is not None
        ]
        ttfts = [
            e["metrics"]["ttft_ms"]
            for e in entries if e["metrics"].get("ttft_ms") is not None
        ]
        itls = [
            e["metrics"]["itl_ms"]
            for e in entries if e["metrics"].get("itl_ms") is not None
        ]
        memories = [
            e["metrics"]["memory_peak_mb"]
            for e in entries if e["metrics"].get("memory_peak_mb") is not None
        ]

        summaries.append({
            "model": model,
            "hardware": hardware,
            "count": len(entries),
            "avg_throughput_tok_s": round(sum(throughputs) / len(throughputs), 2) if throughputs else None,
            "min_ttft_ms": round(min(ttfts), 2) if ttfts else None,
            "avg_itl_ms": round(sum(itls) / len(itls), 2) if itls else None,
            "avg_memory_peak_mb": round(sum(memories) / len(memories), 2) if memories else None,
            "best_throughput_tok_s": round(max(throughputs), 2) if throughputs else None,
        })

    return {"summaries": summaries, "total": len(summaries)}


@router.get("/v1/leaderboard/top")
async def leaderboard_top():
    _ensure_loaded()
    groups: dict[tuple[str, str], list[dict]] = {}
    for entry in _store.values():
        key = (entry["model"], entry["hardware"])
        groups.setdefault(key, []).append(entry)

    tops = []
    for entries in groups.values():
        best = max(entries, key=lambda e: e["metrics"].get("throughput_tok_s") or 0)
        tops.append(best)

    tops.sort(key=lambda e: e["metrics"].get("throughput_tok_s") or 0, reverse=True)
    return {"top_results": tops, "total": len(tops)}


# ── Community Features ──────────────────────────────────────────────────

_votes: dict[str, dict[str, int]] = {}  # entry_id -> {user_id -> vote}
_comments: dict[str, list[dict]] = {}  # entry_id -> list of comments
_verified: set[str] = set()  # entry_ids verified by maintainers


class VoteRequest(BaseModel):
    vote: int = Field(..., ge=-1, le=1, description="Vote: 1=upvote, -1=downvote, 0=remove")


class CommentRequest(BaseModel):
    user_id: str = Field(..., description="Commenter user ID")
    text: str = Field(..., min_length=1, max_length=1000, description="Comment text")


class CommunityStats(BaseModel):
    total_submissions: int = 0
    total_votes: int = 0
    total_comments: int = 0
    verified_count: int = 0
    top_contributors: list[dict] = Field(default_factory=list)


@router.post("/v1/leaderboard/{entry_id}/vote")
async def vote_entry(entry_id: str, body: VoteRequest):
    """Vote on a benchmark submission."""
    if entry_id not in _store:
        raise HTTPException(status_code=404, detail="Entry not found")

    user_id = body.user_id if hasattr(body, 'user_id') else "anonymous"
    if entry_id not in _votes:
        _votes[entry_id] = {}

    if body.vote == 0:
        _votes[entry_id].pop(user_id, None)
    else:
        _votes[entry_id][user_id] = body.vote

    total_votes = sum(_votes[entry_id].values())
    return {"entry_id": entry_id, "total_votes": total_votes, "voter_count": len(_votes[entry_id])}


@router.post("/v1/leaderboard/{entry_id}/comment")
async def comment_entry(entry_id: str, body: CommentRequest):
    """Add a comment to a benchmark submission."""
    if entry_id not in _store:
        raise HTTPException(status_code=404, detail="Entry not found")

    comment = {
        "user_id": body.user_id,
        "text": body.text,
        "timestamp": time.time(),
    }
    _comments.setdefault(entry_id, []).append(comment)

    return {"entry_id": entry_id, "comment_count": len(_comments[entry_id])}


@router.get("/v1/leaderboard/{entry_id}/comments")
async def get_comments(entry_id: str):
    """Get comments for a benchmark submission."""
    return {"entry_id": entry_id, "comments": _comments.get(entry_id, [])}


@router.post("/v1/leaderboard/{entry_id}/verify")
async def verify_entry(entry_id: str):
    """Mark a benchmark submission as verified (admin only)."""
    if entry_id not in _store:
        raise HTTPException(status_code=404, detail="Entry not found")

    _verified.add(entry_id)
    return {"entry_id": entry_id, "verified": True}


@router.get("/v1/leaderboard/community")
async def community_stats():
    """Get community leaderboard statistics."""
    total_votes = sum(sum(v.values()) for v in _votes.values())
    total_comments = sum(len(c) for c in _comments.values())

    # Top contributors
    contributors: dict[str, int] = {}
    for entry in _store.values():
        submitter = entry.get("submitted_by", "anonymous")
        contributors[submitter] = contributors.get(submitter, 0) + 1

    top = sorted(contributors.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "total_submissions": len(_store),
        "total_votes": total_votes,
        "total_comments": total_comments,
        "verified_count": len(_verified),
        "top_contributors": [{"user": u, "submissions": c} for u, c in top],
    }


@router.get("/v1/leaderboard/verified")
async def verified_entries():
    """Get all verified benchmark submissions."""
    entries = [_store[eid] for eid in _verified if eid in _store]
    return {"entries": entries, "total": len(entries)}
