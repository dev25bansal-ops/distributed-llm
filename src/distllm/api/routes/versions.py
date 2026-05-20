"""Model versioning, A/B testing, and deployment management routes."""

import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..api_state import g


router = APIRouter(tags=["versions"])


class VersionCreateRequest(BaseModel):
    version_id: str = Field(..., description="Version identifier (e.g., 'v1.2.0')")
    model_path: str = Field(..., description="Path or HuggingFace ID for the model")
    metadata: dict | None = Field(default=None, description="Optional metadata tags")


class VersionInfo(BaseModel):
    version_id: str
    model_id: str
    model_path: str
    status: str
    created_at: float
    promoted_at: float | None = None
    traffic_weight: float


class VersionStats(BaseModel):
    version_id: str
    status: str
    traffic_weight: float
    total_requests: int
    error_rate: float
    avg_latency_ms: float
    p50_latency_ms: float
    p99_latency_ms: float
    avg_prompt_tokens: float
    avg_completion_tokens: float
    feedback_avg: float | None = None


class ComparisonResult(BaseModel):
    sample_a: int
    sample_b: int
    sufficient_samples: bool
    error_rate_a: float
    error_rate_b: float
    avg_latency_a: float
    avg_latency_b: float
    p50_latency_a: float
    p50_latency_b: float
    p99_latency_a: float
    p99_latency_b: float
    recommendation: str
    reason: str
    mann_whitney_p: float | None = None
    t_p_value: float | None = None


class ShadowComparison(BaseModel):
    model_id: str
    stable_version: str
    shadow_version: str
    request_id: str
    stable_output: str
    shadow_output: str
    latency_stable: float
    latency_shadow: float
    timestamp: float


class BlueGreenSwitchRequest(BaseModel):
    model_id: str = Field(..., description="Model identifier")


class PromoteRequest(BaseModel):
    version_id: str = Field(..., description="Version to promote")


# -- Version CRUD --

@router.post(
    "/v1/models/{model_id}/versions",
    summary="Create model version",
    description="Register a new version for an existing model. Assigns a version identifier and links it to a model path or HuggingFace ID. Useful for A/B testing, staged rollouts, and model lifecycle management.",
    response_description="Created version metadata",
    responses={
        503: {"description": "No model loaded or version management not enabled"},
    },
)
async def create_version(model_id: str, body: VersionCreateRequest):
    """Register a new model version."""
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    vm = getattr(coord, "_version_manager", None)
    if vm is None:
        raise HTTPException(status_code=503, detail="Version management not enabled")

    version = vm.register_version(
        model_id=model_id,
        version_id=body.version_id,
        model_path=body.model_path,
        metadata=body.metadata,
    )

    return VersionInfo(
        version_id=version.version_id,
        model_id=version.model_id,
        model_path=version.model_path,
        status=version.status.value,
        created_at=version.created_at,
        promoted_at=version.promoted_at,
        traffic_weight=version.traffic_weight,
    )


@router.get(
    "/v1/models/{model_id}/versions",
    summary="List model versions",
    description="List all registered versions for a model, including status, traffic weight, and promotion metadata.",
    response_description="List of version metadata objects",
    responses={
        503: {"description": "No model loaded or version management not enabled"},
    },
)
async def list_versions(model_id: str):
    """List all versions for a model."""
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    vm = getattr(coord, "_version_manager", None)
    if vm is None:
        raise HTTPException(status_code=503, detail="Version management not enabled")

    versions = vm.list_versions(model_id)
    return {
        "model_id": model_id,
        "versions": [
            VersionInfo(
                version_id=v.version_id,
                model_id=v.model_id,
                model_path=v.model_path,
                status=v.status.value,
                created_at=v.created_at,
                promoted_at=v.promoted_at,
                traffic_weight=v.traffic_weight,
            )
            for v in versions
        ],
    }


@router.get(
    "/v1/models/{model_id}/versions/{version_id}",
    summary="Get model version",
    description="Get detailed information about a specific model version, including its status, path, creation time, promotion status, and current traffic weight.",
    response_description="Version metadata including status and traffic weight",
    responses={
        404: {"description": "Version not found for the given model"},
        503: {"description": "No model loaded or version management not enabled"},
    },
)
async def get_version(model_id: str, version_id: str):
    """Get details for a specific version."""
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    vm = getattr(coord, "_version_manager", None)
    if vm is None:
        raise HTTPException(status_code=503, detail="Version management not enabled")

    version = vm.get_version(model_id, version_id)
    if version is None:
        raise HTTPException(status_code=404, detail=f"Version {version_id} not found for {model_id}")

    return VersionInfo(
        version_id=version.version_id,
        model_id=version.model_id,
        model_path=version.model_path,
        status=version.status.value,
        created_at=version.created_at,
        promoted_at=version.promoted_at,
        traffic_weight=version.traffic_weight,
    )


@router.get(
    "/v1/models/{model_id}/versions/{version_id}/stats",
    summary="Get version statistics",
    description="Get comprehensive performance statistics for a model version, including request counts, error rate, latency percentiles (p50, p99), and average prompt/completion token counts.",
    response_description="Version performance statistics",
    responses={
        404: {"description": "Version not found"},
        503: {"description": "No model loaded or version management not enabled"},
    },
)
async def get_version_stats(model_id: str, version_id: str):
    """Get comprehensive stats for a version."""
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    vm = getattr(coord, "_version_manager", None)
    if vm is None:
        raise HTTPException(status_code=503, detail="Version management not enabled")

    stats = vm.get_version_stats(model_id, version_id)
    if stats is None:
        raise HTTPException(status_code=404, detail=f"Version {version_id} not found for {model_id}")

    return stats


@router.delete(
    "/v1/models/{model_id}/versions/{version_id}",
    summary="Delete model version",
    description="Unregister and delete a model version. Removes the version from the registry and its associated metadata.",
    response_description="Deletion confirmation with model and version IDs",
    responses={
        404: {"description": "Version not found"},
        503: {"description": "No model loaded or version management not enabled"},
    },
)
async def delete_version(model_id: str, version_id: str):
    """Delete (unregister) a model version."""
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    vm = getattr(coord, "_version_manager", None)
    if vm is None:
        raise HTTPException(status_code=503, detail="Version management not enabled")

    if vm.delete_version(model_id, version_id):
        return {"status": "deleted", "model_id": model_id, "version_id": version_id}
    raise HTTPException(status_code=404, detail=f"Version {version_id} not found for {model_id}")


# -- Promotion --

@router.post(
    "/v1/models/{model_id}/versions/{version_id}/promote",
    summary="Promote model version",
    description="Promote a specific version to become the primary active version. All traffic is directed to the promoted version after promotion.",
    response_description="Promotion confirmation with model and version ID",
    responses={
        404: {"description": "Version not found"},
        503: {"description": "No model loaded or version management not enabled"},
    },
)
async def promote_version(model_id: str, version_id: str):
    """Promote a version to primary active."""
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    vm = getattr(coord, "_version_manager", None)
    if vm is None:
        raise HTTPException(status_code=503, detail="Version management not enabled")

    if vm.promote_version(model_id, version_id):
        return {"status": "promoted", "model_id": model_id, "version_id": version_id}
    raise HTTPException(status_code=404, detail=f"Version {version_id} not found for {model_id}")


@router.post(
    "/v1/models/{model_id}/versions/compare",
    summary="Compare model versions",
    description="Perform a statistical comparison between two model versions (stable vs candidate). Evaluates error rates, latency distributions, and recommends promotion using Mann-Whitney U test and t-test analysis.",
    response_description="Statistical comparison result with recommendation",
    responses={
        400: {"description": "Missing stable_version or candidate_version"},
        503: {"description": "No model loaded or version management not enabled"},
    },
)
async def compare_versions(model_id: str, body: dict):
    """Compare two versions statistically."""
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    vm = getattr(coord, "_version_manager", None)
    if vm is None:
        raise HTTPException(status_code=503, detail="Version management not enabled")

    stable_id = body.get("stable_version")
    candidate_id = body.get("candidate_version")
    if not stable_id or not candidate_id:
        raise HTTPException(status_code=400, detail="stable_version and candidate_version required")

    result = vm.evaluate_promotion(model_id, stable_id, candidate_id)
    return result


# -- Shadow mode --

@router.get(
    "/v1/models/{model_id}/shadow",
    summary="Get shadow comparison results",
    description="Retrieve recent shadow mode comparison results for A/B testing. Shows side-by-side outputs from stable and shadow versions with latency comparison.",
    response_description="List of shadow comparison results",
    responses={
        503: {"description": "No model loaded or version management not enabled"},
    },
)
async def get_shadow_comparisons(model_id: str, limit: int = 100):
    """Get recent shadow mode comparison results."""
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    vm = getattr(coord, "_version_manager", None)
    if vm is None:
        raise HTTPException(status_code=503, detail="Version management not enabled")

    comparisons = vm.get_shadow_comparisons(model_id, limit=limit)
    return {
        "model_id": model_id,
        "comparisons": comparisons,
    }


# -- Blue-green --

@router.post(
    "/v1/models/{model_id}/blue-green/switch",
    summary="Switch blue-green deployment",
    description="Toggle the active deployment color (blue to green or green to blue) for instant traffic switching. Enables zero-downtime deployments and rapid rollback.",
    response_description="Switch confirmation with new active color",
    responses={
        503: {"description": "No model loaded or version management not enabled"},
    },
)
async def switch_blue_green(model_id: str, body: BlueGreenSwitchRequest):
    """Switch active color (blue <-> green) for instant rollback."""
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    vm = getattr(coord, "_version_manager", None)
    if vm is None:
        raise HTTPException(status_code=503, detail="Version management not enabled")

    active_color = vm.switch_color(model_id)
    return {"status": "switched", "active_color": active_color, "model_id": model_id}


@router.post(
    "/v1/models/{model_id}/blue-green/rollback",
    summary="Rollback blue-green deployment",
    description="Instant rollback to the previously inactive deployment color. Routes all traffic back to the prior active version for immediate recovery from deployment issues.",
    response_description="Rollback confirmation with active color",
    responses={
        503: {"description": "No model loaded or version management not enabled"},
    },
)
async def rollback_blue_green(model_id: str):
    """Instant rollback to the inactive color."""
    coord = g.coordinator
    if coord is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    vm = getattr(coord, "_version_manager", None)
    if vm is None:
        raise HTTPException(status_code=503, detail="Version management not enabled")

    active_color = vm.rollback_color(model_id)
    return {"status": "rolled_back", "active_color": active_color, "model_id": model_id}
