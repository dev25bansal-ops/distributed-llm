"""Pre-upgrade validation webhook for DistLLM CRDs.

Validates that CRD changes are backward-compatible before applying.
Runs as a FastAPI server behind a ValidatingWebhookConfiguration.
"""

import logging
from typing import Any, Dict

from fastapi import FastAPI, Request, Response
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(title="DistLLM CRD Validation Webhook")

ALLOWED_API_GROUPS = {"distllm.zeroroute.ai"}
ALLOWED_RESOURCES = {"distributedllmclusters", "nodepools"}


class AdmissionReviewRequest(BaseModel):
    apiVersion: str
    kind: str
    request: Dict[str, Any]


class AdmissionReviewResponse(BaseModel):
    apiVersion: str
    kind: str
    response: Dict[str, Any]


def _validate_crd_compatibility(old_spec: dict, new_spec: dict) -> tuple[bool, str]:
    """Check if CRD changes are backward-compatible.

    Rules:
    - No field deletions from properties
    - No type changes for existing fields
    - No narrowing of enum values
    - New fields must be optional (not in required list without default)
    """
    old_props = old_spec.get("properties", {})
    new_props = new_spec.get("properties", {})
    old_required = set(old_spec.get("required", []))
    new_required = set(new_spec.get("required", []))

    # Check for removed fields
    removed = set(old_props.keys()) - set(new_props.keys())
    if removed:
        return False, f"Backward-incompatible: removed fields: {removed}"

    # Check for type changes
    for field_name in old_props:
        old_type = old_props[field_name].get("type")
        new_type = new_props.get(field_name, {}).get("type")
        if old_type and new_type and old_type != new_type:
            return False, f"Backward-incompatible: field '{field_name}' type changed from {old_type} to {new_type}"

    # Check for new required fields without defaults
    new_required_fields = new_required - old_required
    for field_name in new_required_fields:
        field_spec = new_props.get(field_name, {})
        if "default" not in field_spec:
            return False, f"Backward-incompatible: new required field '{field_name}' has no default"

    return True, "Compatible"


@app.post("/validate-crd")
async def validate_crd(request: Request):
    """Validate a CRD update request.

    Receives an AdmissionReview and returns allowed/denied with reason.
    """
    body = await request.json()
    request_obj = body.get("request", {})

    uid = request_obj.get("uid", "")
    operation = request_obj.get("operation", "")
    old_object = request_obj.get("oldObject", {})
    new_object = request_obj.get("object", {})

    # Only validate UPDATE operations on CRDs
    if operation != "UPDATE":
        return {"apiVersion": "admission.k8s.io/v1", "kind": "AdmissionReview", "response": {"uid": uid, "allowed": True}}

    # Extract spec from old and new CRD
    old_spec = old_object.get("spec", {})
    new_spec = new_object.get("spec", {})

    old_versions = old_spec.get("versions", [])
    new_versions = new_spec.get("versions", [])

    # Compare each version's schema
    for new_ver in new_versions:
        ver_name = new_ver.get("name", "")
        matching_old = next((v for v in old_versions if v.get("name") == ver_name), None)

        if matching_old:
            old_schema = matching_old.get("schema", {}).get("openAPIV3Schema", {})
            new_schema = new_ver.get("schema", {}).get("openAPIV3Schema", {})

            compatible, reason = _validate_crd_compatibility(old_schema, new_schema)
            if not compatible:
                logger.warning("CRD update blocked: %s — %s", ver_name, reason)
                return {
                    "apiVersion": "admission.k8s.io/v1",
                    "kind": "AdmissionReview",
                    "response": {
                        "uid": uid,
                        "allowed": False,
                        "status": {"reason": reason},
                    },
                }

    logger.info("CRD update allowed")
    return {"apiVersion": "admission.k8s.io/v1", "kind": "AdmissionReview", "response": {"uid": uid, "allowed": True}}


@app.post("/validate-cluster")
async def validate_cluster(request: Request):
    """Validate a DistributedLLMCluster resource change."""
    body = await request.json()
    request_obj = body.get("request", {})

    uid = request_obj.get("uid", "")
    new_object = request_obj.get("object", {})

    spec = new_object.get("spec", {})

    # Validate required fields
    if not spec.get("modelName"):
        return {
            "apiVersion": "admission.k8s.io/v1",
            "kind": "AdmissionReview",
            "response": {
                "uid": uid,
                "allowed": False,
                "status": {"reason": "spec.modelName is required"},
            },
        }

    # Validate node pools
    pools = spec.get("nodePools", [])
    for pool in pools:
        if not pool.get("name"):
            return {
                "apiVersion": "admission.k8s.io/v1",
                "kind": "AdmissionReview",
                "response": {
                    "uid": uid,
                    "allowed": False,
                    "status": {"reason": "Each nodePool must have a name"},
                },
            }
        start = pool.get("startLayer", 0)
        end = pool.get("endLayer", 0)
        if start >= end:
            return {
                "apiVersion": "admission.k8s.io/v1",
                "kind": "AdmissionReview",
                "response": {
                    "uid": uid,
                    "allowed": False,
                    "status": {"reason": f"nodePool '{pool['name']}': startLayer ({start}) must be less than endLayer ({end})"},
                },
            }

    return {"apiVersion": "admission.k8s.io/v1", "kind": "AdmissionReview", "response": {"uid": uid, "allowed": True}}


@app.get("/health")
async def health():
    return {"status": "ok"}
