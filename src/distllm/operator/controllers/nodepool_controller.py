"""Kopf controller for NodePool CRD.

Manages StatefulSets for worker node pools independently.
"""

import kopf
import kubernetes.client as k8s
import kubernetes.config as k8s_config
from loguru import logger


def _get_k8s_client():
    try:
        k8s_config.load_incluster_config()
    except k8s_config.ConfigException:
        k8s_config.load_kube_config()
    return k8s.client.AppsV1Api()


@kopf.on.create("distllm.zeroroute.ai", "v1", "nodepools")
def create_nodepool(spec, name, namespace, **kwargs):
    """Create a NodePool StatefulSet."""
    apps_v1 = _get_k8s_client()

    start_layer = spec.get("start_layer", 0)
    end_layer = spec.get("end_layer", 0)
    replicas = spec.get("replicas", 1)
    image = spec.get("image", "distllm/worker:latest")
    grpc_port = spec.get("grpc_port", 50051)
    cluster_name = spec.get("cluster_name", "")
    model_name = spec.get("model_name", "")
    total_layers = spec.get("total_layers", 0)
    dtype = spec.get("dtype", "float16")

    resources_spec = spec.get("resources", {})
    resources = k8s.V1ResourceRequirements(
        limits={
            "nvidia.com/gpu": resources_spec.get("gpu", "1"),
            "memory": resources_spec.get("memory", "32Gi"),
            "cpu": resources_spec.get("cpu", "4"),
        },
        requests={
            "nvidia.com/gpu": resources_spec.get("gpu", "1"),
            "memory": resources_spec.get("memory", "32Gi"),
            "cpu": resources_spec.get("cpu", "4"),
        },
    )

    labels = {
        "app": cluster_name,
        "component": "node-pool",
        "node-pool": name,
        "start-layer": str(start_layer),
        "end-layer": str(end_layer),
    }

    container = k8s.V1Container(
        name="worker",
        image=image,
        command=["python", "-m", "distllm.core.node"],
        args=[
            "--model", model_name,
            "--start-layer", str(start_layer),
            "--end-layer", str(end_layer),
            "--total-layers", str(total_layers),
            "--port", str(grpc_port),
            "--dtype", dtype,
        ],
        resources=resources,
        ports=[k8s.V1ContainerPort(container_port=grpc_port, name="grpc")],
        readiness_probe=k8s.V1Probe(
            tcp_socket=k8s.V1TCPSocketAction(port=grpc_port),
            initial_delay_seconds=60,
            period_seconds=10,
        ),
    )

    ss = k8s.V1StatefulSet(
        api_version="apps/v1",
        kind="StatefulSet",
        metadata=k8s.V1ObjectMeta(name=name, namespace=namespace, labels=labels),
        spec=k8s.V1StatefulSetSpec(
            replicas=replicas,
            selector=k8s.V1LabelSelector(match_labels=labels),
            service_name=f"{name}-headless",
            template=k8s.V1PodTemplateSpec(
                metadata=k8s.V1ObjectMeta(labels=labels),
                spec=k8s.V1PodSpec(containers=[container]),
            ),
        ),
    )

    apps_v1.create_namespaced_stateful_set(namespace, ss)
    logger.info(f"Created NodePool StatefulSet {name}: layers {start_layer}-{end_layer}")


@kopf.on.update("distllm.zeroroute.ai", "v1", "nodepools")
def update_nodepool(spec, name, namespace, **kwargs):
    """Update NodePool StatefulSet."""
    apps_v1 = _get_k8s_client()
    replicas = spec.get("replicas", 1)
    apps_v1.patch_namespaced_stateful_set_scale(
        name, namespace, body=k8s.V1Scale(spec=k8s.V1ScaleSpec(replicas=replicas))
    )
    logger.info(f"Scaled NodePool {name} to {replicas} replicas")


@kopf.on.delete("distllm.zeroroute.ai", "v1", "nodepools")
def delete_nodepool(spec, name, namespace, **kwargs):
    apps_v1 = _get_k8s_client()
    try:
        apps_v1.delete_namespaced_stateful_set(name, namespace)
    except k8s.client.ApiException:
        pass
    logger.info(f"Deleted NodePool {name}")
