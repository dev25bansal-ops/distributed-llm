"""OpenAPI spec generation and API documentation for the distributed layer.

Generates an OpenAPI 3.1 schema by inspecting the existing FastAPI application's
routes, then augments it with the distributed layer's admin and cluster
management endpoints.

Output::
    JSON OpenAPI spec at ``/api/v1/openapi.json``
    ReDoc UI at ``/docs``
    Swagger UI at ``/swagger``

Usage::

    # Via coordinator startup (automatic):
    # The spec is generated when the FastAPI app starts.

    # Manual generation for documentation:
    python -m distllm.dist.api_docs --output openapi.json
"""

from __future__ import annotations

import json
import os
from typing import Any


def build_distributed_spec(base_url: str = "http://localhost:8000") -> dict[str, Any]:
    """Build an OpenAPI 3.1 schema for the distributed layer's API surface.

    This covers the endpoints defined in the coordinator's FastAPI app that
    are specific to distributed inference: cluster management, federation,
    node registration, recovery, and marketplace.

    The spec is designed to be merged with the base API spec generated
    by the FastAPI app itself.
    """
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "DistLLM Distributed Layer API",
            "version": "2.0.0",
            "description": (
                "APIs for managing a distributed LLM inference cluster, "
                "including node registration, federation, model partitioning, "
                "recovery, and the GPU marketplace."
            ),
            "contact": {
                "name": "DistLLM Team",
                "url": "https://github.com/distllm/distllm",
            },
        },
        "servers": [
            {"url": base_url, "description": "Coordinator API"},
        ],
        "paths": {
            # ── Cluster management ──────────────────────────────
            "/admin/v1/nodes": {
                "get": {
                    "summary": "List all registered nodes",
                    "description": "Returns all worker nodes registered with the coordinator.",
                    "tags": ["Cluster Management"],
                    "security": [{"BearerAuth": []}],
                    "responses": {
                        "200": {
                            "description": "List of nodes",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "array",
                                        "items": {"$ref": "#/components/schemas/NodeInfo"},
                                    }
                                }
                            },
                        },
                        "401": {"description": "Unauthorized"},
                    },
                },
                "post": {
                    "summary": "Register a new node",
                    "description": "Register a worker node with the coordinator.",
                    "tags": ["Cluster Management"],
                    "security": [{"BearerAuth": []}],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/NodeRegistration"},
                            }
                        },
                    },
                    "responses": {
                        "200": {"description": "Node registered"},
                        "400": {"description": "Invalid request"},
                    },
                },
            },
            "/admin/v1/nodes/{node_id}/ready": {
                "post": {
                    "summary": "Mark node as ready",
                    "description": "Called by a worker node after model loading completes.",
                    "tags": ["Cluster Management"],
                    "parameters": [
                        {
                            "name": "node_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        },
                    ],
                    "responses": {
                        "200": {"description": "Node marked ready"},
                        "404": {"description": "Node not found"},
                    },
                },
            },
            "/admin/v1/cluster/status": {
                "get": {
                    "summary": "Cluster status",
                    "description": "Overall cluster health, node count, layer distribution.",
                    "tags": ["Cluster Management"],
                    "responses": {
                        "200": {
                            "description": "Cluster status",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ClusterStatus"},
                                }
                            },
                        },
                    },
                },
            },
            "/admin/v1/cluster/rebalance": {
                "post": {
                    "summary": "Trigger layer rebalancing",
                    "description": "Recompute optimal layer distribution across nodes.",
                    "tags": ["Cluster Management"],
                    "security": [{"BearerAuth": []}],
                    "responses": {
                        "200": {"description": "Rebalancing triggered"},
                        "202": {"description": "Accepted, running in background"},
                    },
                },
            },
            # ── Federation ──────────────────────────────────────
            "/v1/federation/heartbeat": {
                "post": {
                    "summary": "Federation heartbeat",
                    "description": "Exchange load metrics with peer clusters.",
                    "tags": ["Federation"],
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ClusterLoad"},
                            }
                        },
                    },
                    "responses": {"200": {"description": "Heartbeat acknowledged"}},
                },
            },
            "/v1/federation/health": {
                "get": {
                    "summary": "Federation health check",
                    "description": "Active health probe for peer clusters.",
                    "tags": ["Federation"],
                    "responses": {
                        "200": {
                            "description": "Health status",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/FederationHealth"},
                                }
                            },
                        },
                    },
                },
            },
            "/v1/federation/peers": {
                "get": {
                    "summary": "List federation peers",
                    "description": "Return all discovered peer clusters.",
                    "tags": ["Federation"],
                    "responses": {
                        "200": {
                            "description": "Peer list",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "array",
                                        "items": {"$ref": "#/components/schemas/PeerInfo"},
                                    }
                                }
                            },
                        },
                    },
                },
            },
            "/v1/federation/gossip": {
                "post": {
                    "summary": "Gossip exchange",
                    "description": "Exchange cache digests with random peers.",
                    "tags": ["Federation"],
                    "responses": {"200": {"description": "Gossip acknowledged"}},
                },
            },
            # ── Recovery ────────────────────────────────────────
            "/admin/v1/recovery/status": {
                "get": {
                    "summary": "Recovery status",
                    "description": "Current recovery state, dead/draining nodes, checkpoint count.",
                    "tags": ["Recovery"],
                    "responses": {
                        "200": {
                            "description": "Recovery state",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/RecoveryStatus"},
                                }
                            },
                        },
                    },
                },
            },
            "/admin/v1/recovery/history": {
                "get": {
                    "summary": "Recovery history",
                    "description": "Past recovery events with timing and outcomes.",
                    "tags": ["Recovery"],
                    "responses": {
                        "200": {
                            "description": "Recovery event list",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "array",
                                        "items": {"$ref": "#/components/schemas/RecoveryEvent"},
                                    }
                                }
                            },
                        },
                    },
                },
            },
            "/admin/v1/recovery/drill": {
                "post": {
                    "summary": "Trigger recovery drill",
                    "description": "Run a non-destructive recovery simulation.",
                    "tags": ["Recovery"],
                    "security": [{"BearerAuth": []}],
                    "responses": {
                        "200": {
                            "description": "Drill result",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/DrillResult"},
                                }
                            },
                        },
                    },
                },
            },
            # ── Marketplace / provisioning ──────────────────────
            "/api/v1/marketplace/listings": {
                "get": {
                    "summary": "List GPU listings",
                    "description": "All available GPU capacity in the marketplace.",
                    "tags": ["Marketplace"],
                    "responses": {"200": {"description": "GPU listings"}},
                },
                "post": {
                    "summary": "Create GPU listing",
                    "description": "Advertise GPU capacity in the marketplace.",
                    "tags": ["Marketplace"],
                    "responses": {"201": {"description": "Listing created"}},
                },
            },
            "/api/v1/marketplace/jobs": {
                "get": {
                    "summary": "List marketplace jobs",
                    "tags": ["Marketplace"],
                    "responses": {"200": {"description": "Job list"}},
                },
                "post": {
                    "summary": "Create marketplace job",
                    "description": "Post a compute job to the marketplace.",
                    "tags": ["Marketplace"],
                    "responses": {"201": {"description": "Job created"}},
                },
            },
            "/api/v1/provisioning/deployments": {
                "post": {
                    "summary": "Request model deployment",
                    "description": "Self-serve: request a model deployment with GPU requirements.",
                    "tags": ["Provisioning"],
                    "security": [{"BearerAuth": []}],
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/DeploymentRequest"},
                            }
                        },
                    },
                    "responses": {
                        "201": {"description": "Deployment created"},
                        "400": {"description": "Invalid requirements"},
                    },
                },
                "get": {
                    "summary": "List deployments",
                    "description": "List all deployments, optionally filtered by tenant.",
                    "tags": ["Provisioning"],
                    "parameters": [
                        {
                            "name": "tenant_id",
                            "in": "query",
                            "schema": {"type": "string"},
                        },
                        {
                            "name": "status",
                            "in": "query",
                            "schema": {"type": "string"},
                        },
                    ],
                    "responses": {"200": {"description": "Deployment list"}},
                },
            },
            "/api/v1/provisioning/deployments/{deployment_id}": {
                "get": {
                    "summary": "Get deployment status",
                    "tags": ["Provisioning"],
                    "parameters": [
                        {
                            "name": "deployment_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        },
                    ],
                    "responses": {"200": {"description": "Deployment details"}},
                },
                "delete": {
                    "summary": "Terminate deployment",
                    "tags": ["Provisioning"],
                    "security": [{"BearerAuth": []}],
                    "parameters": [
                        {
                            "name": "deployment_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        },
                    ],
                    "responses": {
                        "200": {"description": "Deployment terminated"},
                        "404": {"description": "Not found"},
                    },
                },
            },
            # ── KV cache migration ──────────────────────────────
            "/api/v1/cache/warm": {
                "post": {
                    "summary": "Warm cache entry",
                    "description": "Receive a KV cache page for a prefix hash.",
                    "tags": ["Cache"],
                    "responses": {"200": {"description": "Cache warmed"}},
                },
            },
            "/api/v1/cache/migrate": {
                "post": {
                    "summary": "Migrate KV cache pages",
                    "description": "Stream KV cache pages from another cluster.",
                    "tags": ["Cache"],
                    "responses": {"200": {"description": "Migration accepted"}},
                },
            },
            # ── Power capping ───────────────────────────────────
            "/admin/v1/power/status": {
                "get": {
                    "summary": "GPU power status",
                    "description": "Current GPU power limits, draw, and savings.",
                    "tags": ["Power Management"],
                    "responses": {"200": {"description": "Power status"}},
                },
            },
            "/admin/v1/power/auto-tune": {
                "post": {
                    "summary": "Auto-tune power limits",
                    "description": "Find optimal GPU power limits via binary search.",
                    "tags": ["Power Management"],
                    "security": [{"BearerAuth": []}],
                    "responses": {
                        "200": {"description": "Auto-tune complete"},
                        "202": {"description": "Auto-tune started (async)"},
                    },
                },
            },
            # ── Multi-tenant SLO ────────────────────────────────
            "/admin/v1/tenants": {
                "get": {
                    "summary": "List tenant SLO status",
                    "description": "Per-tenant rate limits, latency SLOs, compliance.",
                    "tags": ["Multi-Tenant"],
                    "responses": {"200": {"description": "Tenant list"}},
                },
            },
            "/admin/v1/tenants/{tenant_id}": {
                "get": {
                    "summary": "Get tenant SLO details",
                    "parameters": [
                        {
                            "name": "tenant_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        },
                    ],
                    "tags": ["Multi-Tenant"],
                    "responses": {"200": {"description": "Tenant details"}},
                },
            },
            # ── Cloud region selection ──────────────────────────
            "/api/v1/regions": {
                "get": {
                    "summary": "List available cloud regions",
                    "description": "Available GPU regions with pricing across AWS/GCP/Azure.",
                    "tags": ["Cloud"],
                    "parameters": [
                        {
                            "name": "min_gpu_memory_gb",
                            "in": "query",
                            "schema": {"type": "number", "default": 80},
                        },
                    ],
                    "responses": {"200": {"description": "Region list"}},
                },
            },
        },
        "components": {
            "securitySchemes": {
                "BearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "JWT",
                },
                "ApiKeyAuth": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key",
                },
            },
            "schemas": {
                "NodeInfo": {
                    "type": "object",
                    "properties": {
                        "node_id": {"type": "string"},
                        "host": {"type": "string"},
                        "port": {"type": "integer"},
                        "start_layer": {"type": "integer"},
                        "end_layer": {"type": "integer"},
                        "total_layers": {"type": "integer"},
                        "device": {"type": "string"},
                        "ready": {"type": "boolean"},
                        "gpu_name": {"type": "string"},
                    },
                },
                "NodeRegistration": {
                    "type": "object",
                    "required": ["node_id", "host", "port"],
                    "properties": {
                        "node_id": {"type": "string"},
                        "host": {"type": "string"},
                        "port": {"type": "integer"},
                        "start_layer": {"type": "integer"},
                        "end_layer": {"type": "integer"},
                        "total_layers": {"type": "integer"},
                        "device": {"type": "string"},
                        "gpu_name": {"type": "string"},
                        "ready": {"type": "boolean"},
                    },
                },
                "ClusterStatus": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "enum": ["ok", "degraded", "starting"]},
                        "node_count": {"type": "integer"},
                        "active_nodes": {"type": "integer"},
                        "total_layers": {"type": "integer"},
                        "model": {"type": "string"},
                    },
                },
                "ClusterLoad": {
                    "type": "object",
                    "properties": {
                        "gpu_utilization": {"type": "number"},
                        "gpu_memory_percent": {"type": "number"},
                        "cpu_percent": {"type": "number"},
                        "memory_percent": {"type": "number"},
                        "active_requests": {"type": "integer"},
                        "pending_requests": {"type": "integer"},
                        "node_count": {"type": "integer"},
                    },
                },
                "FederationHealth": {
                    "type": "object",
                    "properties": {
                        "cluster_id": {"type": "string"},
                        "uptime_s": {"type": "number"},
                        "peers": {"type": "integer"},
                        "healthy_peers": {"type": "integer"},
                    },
                },
                "PeerInfo": {
                    "type": "object",
                    "properties": {
                        "cluster_id": {"type": "string"},
                        "host": {"type": "string"},
                        "port": {"type": "integer"},
                        "region": {"type": "string"},
                        "last_seen": {"type": "number"},
                        "is_edge": {"type": "boolean"},
                    },
                },
                "RecoveryStatus": {
                    "type": "object",
                    "properties": {
                        "state": {"type": "string"},
                        "dead_nodes": {"type": "array", "items": {"type": "string"}},
                        "draining_nodes": {"type": "array", "items": {"type": "string"}},
                        "active_checkpoints": {"type": "integer"},
                        "total_recoveries": {"type": "integer"},
                    },
                },
                "RecoveryEvent": {
                    "type": "object",
                    "properties": {
                        "timestamp": {"type": "number"},
                        "node_id": {"type": "string"},
                        "sequences_recovered": {"type": "integer"},
                        "sequences_lost": {"type": "integer"},
                        "duration_ms": {"type": "number"},
                        "redistributions": {"type": "integer"},
                    },
                },
                "DrillResult": {
                    "type": "object",
                    "properties": {
                        "simulated_node_id": {"type": "string"},
                        "recovery_time_ms": {"type": "number"},
                        "sequences_recovered": {"type": "integer"},
                        "sequences_lost": {"type": "integer"},
                        "redistributions": {"type": "integer"},
                        "passed": {"type": "boolean"},
                        "failures": {"type": "array", "items": {"type": "string"}},
                    },
                },
                "DeploymentRequest": {
                    "type": "object",
                    "required": ["tenant_id", "model_name"],
                    "properties": {
                        "tenant_id": {"type": "string"},
                        "model_name": {"type": "string"},
                        "gpu_count": {"type": "integer", "default": 1},
                        "min_gpu_memory_gb": {"type": "number", "default": 80},
                        "max_budget_per_hour": {"type": "number", "default": 50},
                        "preferred_regions": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
            },
        },
    }


def generate_spec(output_path: str | None = None) -> dict[str, Any]:
    """Generate the OpenAPI spec and optionally write to a file."""
    spec = build_distributed_spec(
        base_url=os.environ.get("COORDINATOR_URL", "http://localhost:8000"),
    )
    if output_path:
        with open(output_path, "w") as f:
            json.dump(spec, f, indent=2)
        print(f"OpenAPI spec written to {output_path}")
    return spec


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate DistLLM OpenAPI spec")
    parser.add_argument("--output", "-o", default="openapi.json", help="Output path")
    args = parser.parse_args()
    generate_spec(args.output)
