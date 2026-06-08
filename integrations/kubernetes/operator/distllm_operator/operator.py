import asyncio
import logging
import os

import kopf
import kubernetes
import yaml

from distllm_operator.config import OperatorConfig

logger = logging.getLogger("distllm_operator")

config = OperatorConfig()

# Leader election lock identity (unique per pod)
_LEADER_IDENTITY = os.environ.get("HOSTNAME", "distllm-operator")


def _get_client():
    kubernetes.config.load_incluster_config()
    return kubernetes.client


# ------------------------------------------------------------------
# Lifecycle handlers
# ------------------------------------------------------------------


@kopf.on.create("distllm.ai", "v1", "distllmclusters")
def create_fn(spec, meta, logger, **kwargs):
    """Handle creation of DistLLMCluster CRD."""
    name = meta.get("name")
    namespace = meta.get("namespace", config.namespace)
    replicas = spec.get("replicas", config.default_replicas)
    model = spec.get("model", config.default_model)

    logger.info(f"Creating DistLLMCluster {name} with {replicas} workers")

    client = _get_client()
    apps_v1 = client.AppsV1Api()
    core_v1 = client.CoreV1Api()

    _create_coordinator(apps_v1, core_v1, name, namespace, model)
    _create_workers(apps_v1, core_v1, name, namespace, replicas, model)
    _create_service(core_v1, name, namespace)

    # Set initial status
    _update_status(name, namespace, "Deploying", replicas, model)

    logger.info(f"DistLLMCluster {name} created successfully")


@kopf.on.update("distllm.ai", "v1", "distllmclusters")
def update_fn(spec, meta, diff, logger, **kwargs):
    """Handle updates to DistLLMCluster CRD (scale workers)."""
    name = meta.get("name")
    namespace = meta.get("namespace", config.namespace)
    replicas = spec.get("replicas", config.default_replicas)

    logger.info(f"Updating DistLLMCluster {name} to {replicas} workers")

    client = _get_client()
    apps_v1 = client.AppsV1Api()
    deployment_name = f"distllm-worker-{name}"

    try:
        apps_v1.patch_namespaced_deployment_scale(
            name=deployment_name,
            namespace=namespace,
            body={"spec": {"replicas": replicas}},
        )
        _update_status(name, namespace, "Scaling", replicas)
        logger.info(f"Scaled workers to {replicas}")
    except kubernetes.client.exceptions.ApiException as e:
        logger.error(f"Failed to scale workers: {e}")
        _update_status(name, namespace, "Error", replicas, error=str(e))


@kopf.on.delete("distllm.ai", "v1", "distllmclusters")
def delete_fn(spec, meta, logger, **kwargs):
    """Handle deletion of DistLLMCluster CRD."""
    name = meta.get("name")
    namespace = meta.get("namespace", config.namespace)

    logger.info(f"Deleting DistLLMCluster {name}")

    client = _get_client()
    apps_v1 = client.AppsV1Api()
    core_v1 = client.CoreV1Api()

    for deployment_name in [f"distllm-coordinator-{name}", f"distllm-worker-{name}"]:
        try:
            apps_v1.delete_namespaced_deployment(deployment_name, namespace)
        except kubernetes.client.exceptions.ApiException:
            pass

    try:
        core_v1.delete_namespaced_service(f"distllm-{name}", namespace)
    except kubernetes.client.exceptions.ApiException:
        pass


@kopf.on.resume("distllm.ai", "v1", "distllmclusters")
def resume_fn(spec, meta, logger, **kwargs):
    """Reconcile on operator restart."""
    create_fn(spec, meta, logger, **kwargs)


# ------------------------------------------------------------------
# Health checking — periodic reconciliation
# ------------------------------------------------------------------


@kopf.on.event("distllm.ai", "v1", "distllmclusters")
async def watch_fn(spec, meta, event, logger, **kwargs):
    """Watch for events and reconcile if needed."""
    if event.get("type") not in ("ADDED", "MODIFIED"):
        return

    name = meta.get("name")
    namespace = meta.get("namespace", config.namespace)

    # Check health in background to avoid blocking the event loop
    asyncio.create_task(_reconcile_health(name, namespace, spec, logger))


async def _reconcile_health(name: str, namespace: str, spec: dict, logger):
    """Periodically check pod health and update status."""
    try:
        client = _get_client()
        apps_v1 = client.AppsV1Api()
        core_v1 = client.CoreV1Api()

        expected_replicas = spec.get("replicas", config.default_replicas)

        # Check coordinator
        coord_name = f"distllm-coordinator-{name}"
        coord_deploy = apps_v1.read_namespaced_deployment(coord_name, namespace)
        coord_ready = coord_deploy.status.ready_replicas or 0

        # Check workers
        worker_name = f"distllm-worker-{name}"
        worker_deploy = apps_v1.read_namespaced_deployment(worker_name, namespace)
        worker_ready = worker_deploy.status.ready_replicas or 0

        if coord_ready > 0 and worker_ready >= expected_replicas:
            _update_status(name, namespace, "Running", worker_ready)
        elif coord_ready > 0:
            _update_status(name, namespace, "Partial", worker_ready)
        else:
            _update_status(name, namespace, "Pending", worker_ready)

    except Exception as e:
        logger.warning(f"Health check failed for {name}: {e}")


# ------------------------------------------------------------------
# Status updates
# ------------------------------------------------------------------


def _update_status(
    name: str,
    namespace: str,
    phase: str,
    ready_workers: int = 0,
    model: str = "",
    error: str = "",
):
    """Update the CRD status subresource."""
    try:
        k8s = kubernetes.client.CustomObjectsApi()
        body = {
            "status": {
                "phase": phase,
                "readyWorkers": ready_workers,
                "conditions": [
                    {
                        "type": "Ready",
                        "status": "True" if phase == "Running" else "False",
                        "lastTransitionTime": kubernetes.client.ApiClient().sanitize_for_serialization(
                            __import__("datetime").datetime.utcnow()
                        ),
                        "reason": phase,
                        "message": error or f"Cluster is {phase.lower()}",
                    }
                ],
            }
        }
        if model:
            body["status"]["activeModel"] = model

        k8s.patch_namespaced_custom_object_status(
            group="distllm.ai",
            version="v1",
            namespace=namespace,
            plural="distllmclusters",
            name=name,
            body=body,
        )
    except Exception as e:
        logger.warning(f"Failed to update status for {name}: {e}")


# ------------------------------------------------------------------
# Resource builders
# ------------------------------------------------------------------


def _create_coordinator(apps_v1, core_v1, name, namespace, model):
    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": f"distllm-coordinator-{name}", "namespace": namespace},
        "spec": {
            "replicas": 1,
            "selector": {"matchLabels": {"app": f"distllm-coordinator-{name}"}},
            "template": {
                "metadata": {"labels": {"app": f"distllm-coordinator-{name}"}},
                "spec": {
                    "containers": [
                        {
                            "name": "coordinator",
                            "image": config.image,
                            "imagePullPolicy": config.image_pull_policy,
                            "args": ["distllm-coordinator"],
                            "ports": [{"containerPort": config.coordinator_port}],
                            "env": [{"name": "DISTLLM_MODEL", "value": model}],
                            "resources": {
                                "limits": {
                                    "cpu": config.default_cpu,
                                    "memory": config.default_memory,
                                },
                                "requests": {"cpu": "1", "memory": "4Gi"},
                            },
                            "livenessProbe": {
                                "httpGet": {"path": "/health", "port": config.coordinator_port},
                                "initialDelaySeconds": 15,
                                "periodSeconds": 10,
                            },
                            "readinessProbe": {
                                "httpGet": {"path": "/health", "port": config.coordinator_port},
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5,
                            },
                        }
                    ],
                },
            },
        },
    }
    apps_v1.create_namespaced_deployment(namespace, deployment)


def _create_workers(apps_v1, core_v1, name, namespace, replicas, model):
    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": f"distllm-worker-{name}", "namespace": namespace},
        "spec": {
            "replicas": replicas,
            "selector": {"matchLabels": {"app": f"distllm-worker-{name}"}},
            "template": {
                "metadata": {"labels": {"app": f"distllm-worker-{name}"}},
                "spec": {
                    "tolerations": [
                        {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}
                    ],
                    "containers": [
                        {
                            "name": "worker",
                            "image": config.image,
                            "imagePullPolicy": config.image_pull_policy,
                            "args": ["distllm-node"],
                            "ports": [{"containerPort": config.worker_port}],
                            "env": [{"name": "DISTLLM_MODEL", "value": model}],
                            "resources": {
                                "limits": {
                                    "cpu": config.default_cpu,
                                    "memory": config.default_memory,
                                    "nvidia.com/gpu": config.default_gpu,
                                },
                                "requests": {"cpu": "1", "memory": "4Gi"},
                            },
                            "livenessProbe": {
                                "httpGet": {"path": "/health", "port": config.worker_port},
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10,
                            },
                            "readinessProbe": {
                                "httpGet": {"path": "/health", "port": config.worker_port},
                                "initialDelaySeconds": 10,
                                "periodSeconds": 5,
                            },
                        }
                    ],
                },
            },
        },
    }
    apps_v1.create_namespaced_deployment(namespace, deployment)


def _create_service(core_v1, name, namespace):
    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": f"distllm-{name}", "namespace": namespace},
        "spec": {
            "selector": {"app": f"distllm-coordinator-{name}"},
            "ports": [
                {
                    "name": "http",
                    "port": config.coordinator_port,
                    "targetPort": config.coordinator_port,
                },
                {
                    "name": "grpc",
                    "port": config.worker_port,
                    "targetPort": config.worker_port,
                },
            ],
        },
    }
    core_v1.create_namespaced_service(namespace, service)
