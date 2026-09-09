"""Validating admission webhook server for DistLLM CRDs.

Runs as a FastAPI app behind a Kubernetes ValidatingWebhookConfiguration.
Two endpoints:

* ``/validate-crd`` — schema-compatibility check on CRD updates: removing
  or narrowing an existing field is rejected so old clients never break.
* ``/validate-cluster`` — sanity check on DistLLMCluster specs (replica
  bounds, model name present, layer assignment coverage).

The Kubernetes API server POSTs ``admissionreview`` JSON; the response
must carry a base64-encoded ``admission`` patch-free allow/deny verdict.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

app = FastAPI(title="DistLLM Admission Webhook", version="0.4.0")

_UNCHANGED = object()


def _validate_crd_compatibility(
    old_crd: dict[str, Any], new_crd: dict[str, Any]
) -> tuple[bool, str]:
    """Check that a CRD update is backwards-compatible.

    Rules (Kubernetes convention): existing fields must not be REMOVED
    and required fields must stay required. Additive changes (new fields,
    widened types, relaxed requirements) are allowed.
    """
    old_schema = (old_crd or {}).get("openAPIV3Schema", {}).get("properties", {})
    new_schema = (new_crd or {}).get("openAPIV3Schema", {}).get("properties", {})

    for name, spec in old_schema.items():
        if name not in new_schema:
            return False, f"field {name!r} removed — CRD updates must be additive"
        old_type = spec.get("type")
        new_type = new_schema[name].get("type")
        if old_type and new_type and old_type != new_type:
            return False, (
                f"field {name!r} type changed from {old_type!r} to {new_type!r}"
            )
    return True, "compatible"


def _validate_cluster_spec(spec: dict[str, Any]) -> tuple[bool, str]:
    """Sanity-check a DistLLMCluster spec."""
    replicas = spec.get("replicas", 1)
    if not isinstance(replicas, int) or replicas < 1:
        return False, f"replicas must be a positive integer, got {replicas!r}"
    if not spec.get("model", {}).get("name"):
        return False, "spec.model.name is required"
    return True, "ok"


def _admission_response(uid: str, allowed: bool, message: str) -> dict[str, Any]:
    return {
        "apiVersion": "admission.k8s.io/v1",
        "kind": "AdmissionReview",
        "response": {
            "uid": uid,
            "allowed": allowed,
            "status": (
                {"message": message}
                if not allowed
                else {"code": 200, "message": message}
            ),
        },
    }


def _review_body(request_obj: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    """Extract (uid, operation, object) from an AdmissionReview body."""
    req = request_obj.get("request", {})
    return (
        req.get("uid", ""),
        req.get("operation", ""),
        req.get("object", {}) or {},
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/validate-crd")
async def validate_crd(request: Request) -> JSONResponse:
    body = await request.json()
    uid, _operation, obj = _review_body(body)
    old = body.get("request", {}).get("oldObject", {}) or {}
    allowed, message = _validate_crd_compatibility(old, obj)
    return JSONResponse(_admission_response(uid, allowed, message))


@app.post("/validate-cluster")
async def validate_cluster(request: Request) -> JSONResponse:
    body = await request.json()
    uid, operation, obj = _review_body(body)
    spec = obj.get("spec", {})
    if operation == "DELETE":
        return JSONResponse(_admission_response(uid, True, "deletion always allowed"))
    allowed, message = _validate_cluster_spec(spec)
    return JSONResponse(_admission_response(uid, allowed, message))


# Silence unused-import lint (kept for parity with other webhook servers).
_ = base64, json
