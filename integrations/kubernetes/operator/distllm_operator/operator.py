import kopf
import kubernetes
import yaml

from distllm_operator.config import OperatorConfig

config = OperatorConfig()


def _get_client():
    kubernetes.config.load_incluster_config()
    return kubernetes.client


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
        logger.info(f"Scaled workers to {replicas}")
    except kubernetes.client.exceptions.ApiException as e:
        logger.error(f"Failed to scale workers: {e}")


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
                    "containers": [{
                        "name": "coordinator",
                        "image": config.image,
                        "imagePullPolicy": config.image_pull_policy,
                        "args": ["distllm-coordinator"],
                        "ports": [{"containerPort": config.coordinator_port}],
                        "env": [{"name": "DISTLLM_MODEL", "value": model}],
                        "resources": {
                            "limits": {"cpu": config.default_cpu, "memory": config.default_memory},
                            "requests": {"cpu": "1", "memory": "4Gi"},
                        },
                    }],
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
                    "containers": [{
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
                    }],
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
                {"name": "http", "port": config.coordinator_port, "targetPort": config.coordinator_port},
                {"name": "grpc", "port": config.worker_port, "targetPort": config.worker_port},
            ],
        },
    }
    core_v1.create_namespaced_service(namespace, service)
