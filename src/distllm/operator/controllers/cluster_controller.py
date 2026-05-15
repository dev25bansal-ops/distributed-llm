"""Kopf controller for DistributedLLMCluster CRD.

Handles create/update/delete/reconcile events for cluster resources.
"""

import kopf
import kubernetes.client as k8s
import kubernetes.config as k8s_config
from loguru import logger

from distllm.operator.crds import DistributedLLMClusterSpec


def _get_k8s_client():
    """Get authenticated Kubernetes API client."""
    try:
        k8s_config.load_incluster_config()
    except k8s_config.ConfigException:
        k8s_config.load_kube_config()
    return k8s.client.AppsV1Api(), k8s.client.CoreV1Api()


def _build_coordinator_statefulset(spec: DistributedLLMClusterSpec, name: str, namespace: str):
    """Build a StatefulSet for the coordinator."""
    coord = spec.coordinator
    labels = {"app": name, "component": "coordinator"}

    container_env = [
        k8s.V1EnvVar(name="MODEL_NAME", value=spec.model.name),
        k8s.V1EnvVar(name="DTYPE", value=spec.model.dtype),
        k8s.V1EnvVar(name="PORT", value=str(coord.port)),
        k8s.V1EnvVar(name="GRPC_PORT", value=str(coord.grpc_port)),
        k8s.V1EnvVar(name="DISCOVERY_MODE", value="k8s"),
    ]
    if spec.api_key_secret:
        container_env.append(
            k8s.V1EnvVar(
                name="API_KEY",
                value_from=k8s.V1EnvVarSource(
                    secret_key_ref=k8s.V1SecretKeySelector(
                        name=spec.api_key_secret, key="api-key"
                    )
                ),
            )
        )

    resources = k8s.V1ResourceRequirements(
        limits={
            "nvidia.com/gpu": coord.resources.gpu,
            "memory": coord.resources.memory,
            "cpu": coord.resources.cpu,
        },
        requests={
            "nvidia.com/gpu": coord.resources.gpu,
            "memory": coord.resources.memory,
            "cpu": coord.resources.cpu,
        },
    )

    container = k8s.V1Container(
        name="coordinator",
        image=coord.image,
        command=["python", "-m", "distllm.api.server"],
        args=[
            "--model", spec.model.name,
            "--port", str(coord.port),
            "--grpc-port", str(coord.grpc_port),
            "--dtype", spec.model.dtype,
            "--local",
        ],
        env=container_env,
        resources=resources,
        ports=[
            k8s.V1ContainerPort(container_port=coord.port, name="http"),
            k8s.V1ContainerPort(container_port=coord.grpc_port, name="grpc"),
        ],
        readiness_probe=k8s.V1Probe(
            http_get=k8s.V1HTTPGetAction(path="/health", port=coord.port),
            initial_delay_seconds=30,
            period_seconds=10,
        ),
        liveness_probe=k8s.V1Probe(
            http_get=k8s.V1HTTPGetAction(path="/health", port=coord.port),
            initial_delay_seconds=60,
            period_seconds=30,
        ),
    )

    pod_spec = k8s.V1PodSpec(
        containers=[container],
        tolerations=[
            k8s.V1Toleration(key="nvidia.com/gpu", operator="Exists", effect="NoSchedule")
        ],
    )

    template = k8s.V1PodTemplateSpec(
        metadata=k8s.V1ObjectMeta(labels=labels),
        spec=pod_spec,
    )

    statefulset = k8s.V1StatefulSet(
        api_version="apps/v1",
        kind="StatefulSet",
        metadata=k8s.V1ObjectMeta(
            name=f"{name}-coordinator",
            namespace=namespace,
            labels=labels,
        ),
        spec=k8s.V1StatefulSetSpec(
            replicas=coord.replicas,
            selector=k8s.V1LabelSelector(match_labels=labels),
            service_name=f"{name}-coordinator-headless",
            template=template,
        ),
    )
    return statefulset


def _build_node_pool_statefulset(spec: DistributedLLMClusterSpec, pool: object, name: str, namespace: str):
    """Build a StatefulSet for a node pool."""
    labels = {
        "app": name,
        "component": "node-pool",
        "start-layer": str(pool.start_layer),
        "end-layer": str(pool.end_layer),
    }

    container_env = [
        k8s.V1EnvVar(name="MODEL_NAME", value=spec.model.name),
        k8s.V1EnvVar(name="DTYPE", value=spec.model.dtype),
        k8s.V1EnvVar(name="START_LAYER", value=str(pool.start_layer)),
        k8s.V1EnvVar(name="END_LAYER", value=str(pool.end_layer)),
        k8s.V1EnvVar(name="TOTAL_LAYERS", value=str(spec.model.layers)),
        k8s.V1EnvVar(name="GRPC_PORT", value=str(pool.grpc_port)),
        k8s.V1EnvVar(name="COORDINATOR_SERVICE", value=f"{name}-coordinator-headless"),
    ]

    resources = k8s.V1ResourceRequirements(
        limits={
            "nvidia.com/gpu": pool.resources.gpu,
            "memory": pool.resources.memory,
            "cpu": pool.resources.cpu,
        },
        requests={
            "nvidia.com/gpu": pool.resources.gpu,
            "memory": pool.resources.memory,
            "cpu": pool.resources.cpu,
        },
    )

    container = k8s.V1Container(
        name="worker",
        image=pool.image,
        command=["python", "-m", "distllm.core.node"],
        args=[
            "--model", spec.model.name,
            "--start-layer", str(pool.start_layer),
            "--end-layer", str(pool.end_layer),
            "--total-layers", str(spec.model.layers),
            "--port", str(pool.grpc_port),
            "--dtype", spec.model.dtype,
        ],
        env=container_env,
        resources=resources,
        ports=[
            k8s.V1ContainerPort(container_port=pool.grpc_port, name="grpc"),
        ],
        readiness_probe=k8s.V1Probe(
            tcp_socket=k8s.V1TCPSocketAction(port=pool.grpc_port),
            initial_delay_seconds=60,
            period_seconds=10,
        ),
    )

    pod_spec = k8s.V1PodSpec(
        containers=[container],
        tolerations=[
            k8s.V1Toleration(key="nvidia.com/gpu", operator="Exists", effect="NoSchedule")
        ],
    )

    template = k8s.V1PodTemplateSpec(
        metadata=k8s.V1ObjectMeta(labels=labels),
        spec=pod_spec,
    )

    statefulset = k8s.V1StatefulSet(
        api_version="apps/v1",
        kind="StatefulSet",
        metadata=k8s.V1ObjectMeta(
            name=f"{name}-pool-{pool.start_layer}-{pool.end_layer}",
            namespace=namespace,
            labels=labels,
        ),
        spec=k8s.V1StatefulSetSpec(
            replicas=pool.replicas,
            selector=k8s.V1LabelSelector(match_labels=labels),
            service_name=f"{name}-pool-{pool.start_layer}-{pool.end_layer}-headless",
            template=template,
        ),
    )
    return statefulset


@kopf.on.create("distllm.zeroroute.ai", "v1", "distributedllmclusters")
def create_cluster(spec, name, namespace, **kwargs):
    """Create cluster resources when a DistributedLLMCluster CRD is created."""
    cluster_spec = DistributedLLMClusterSpec(**spec)
    apps_v1, core_v1 = _get_k8s_client()

    # Create headless service for coordinator (gRPC discovery)
    coord_svc = k8s.V1Service(
        api_version="v1",
        kind="Service",
        metadata=k8s.V1ObjectMeta(
            name=f"{name}-coordinator-headless",
            namespace=namespace,
            labels={"app": name},
        ),
        spec=k8s.V1ServiceSpec(
            cluster_ip="None",
            selector={"app": name, "component": "coordinator"},
            ports=[
                k8s.V1ServicePort(name="http", port=cluster_spec.coordinator.port),
                k8s.V1ServicePort(name="grpc", port=cluster_spec.coordinator.grpc_port),
            ],
        ),
    )
    core_v1.create_namespaced_service(namespace, coord_svc)
    logger.info(f"Created headless service for {name}")

    # Create coordinator StatefulSet
    coord_ss = _build_coordinator_statefulset(cluster_spec, name, namespace)
    apps_v1.create_namespaced_stateful_set(namespace, coord_ss)
    logger.info(f"Created coordinator StatefulSet for {name}")

    # Create node pool StatefulSets
    for pool in cluster_spec.node_pools:
        pool_ss = _build_node_pool_statefulset(cluster_spec, pool, name, namespace)
        apps_v1.create_namespaced_stateful_set(namespace, pool_ss)
        logger.info(f"Created node pool StatefulSet {pool.start_layer}-{pool.end_layer}")

    # Create API Service (ClusterIP)
    api_svc = k8s.V1Service(
        api_version="v1",
        kind="Service",
        metadata=k8s.V1ObjectMeta(
            name=f"{name}-api",
            namespace=namespace,
        ),
        spec=k8s.V1ServiceSpec(
            selector={"app": name, "component": "coordinator"},
            ports=[k8s.V1ServicePort(name="http", port=80, target_port=cluster_spec.coordinator.port)],
            type="ClusterIP",
        ),
    )
    core_v1.create_namespaced_service(namespace, api_svc)

    kopf.info(spec, reason="Created", message=f"Cluster {name} resources created")


@kopf.on.update("distllm.zeroroute.ai", "v1", "distributedllmclusters")
def update_cluster(spec, name, namespace, diff, **kwargs):
    """Handle rolling updates when the cluster spec changes."""
    cluster_spec = DistributedLLMClusterSpec(**spec)
    apps_v1, _ = _get_k8s_client()

    # Update coordinator StatefulSet (triggers rolling update)
    coord_ss = _build_coordinator_statefulset(cluster_spec, name, namespace)
    apps_v1.patch_namespaced_stateful_set(
        f"{name}-coordinator", namespace, coord_ss
    )

    # Update node pool StatefulSets
    for pool in cluster_spec.node_pools:
        pool_ss = _build_node_pool_statefulset(cluster_spec, pool, name, namespace)
        apps_v1.patch_namespaced_stateful_set(
            f"{name}-pool-{pool.start_layer}-{pool.end_layer}", namespace, pool_ss
        )

    kopf.info(spec, reason="Updated", message=f"Cluster {name} rolling update initiated")


@kopf.on.delete("distllm.zeroroute.ai", "v1", "distributedllmclusters")
def delete_cluster(spec, name, namespace, **kwargs):
    """Clean up cluster resources when CRD is deleted."""
    apps_v1, core_v1 = _get_k8s_client()
    cluster_spec = DistributedLLMClusterSpec(**spec)

    # Delete node pool StatefulSets
    for pool in cluster_spec.node_pools:
        try:
            apps_v1.delete_namespaced_stateful_set(
                f"{name}-pool-{pool.start_layer}-{pool.end_layer}", namespace
            )
        except k8s.client.ApiException:
            pass

    # Delete coordinator StatefulSet and services
    try:
        apps_v1.delete_namespaced_stateful_set(f"{name}-coordinator", namespace)
        core_v1.delete_namespaced_service(f"{name}-coordinator-headless", namespace)
        core_v1.delete_namespaced_service(f"{name}-api", namespace)
    except k8s.client.ApiException:
        pass

    kopf.info(spec, reason="Deleted", message=f"Cluster {name} resources deleted")


@kopf.timer("distllm.zeroroute.ai", "v1", "distributedllmclusters", interval=30)
def reconcile_cluster(spec, name, namespace, **kwargs):
    """Periodic reconciliation to ensure resources match desired state."""
    apps_v1, core_v1 = _get_k8s_client()

    try:
        ss = apps_v1.read_namespaced_stateful_set(f"{name}-coordinator", namespace)
        status = {
            "replicas": ss.spec.replicas,
            "ready_replicas": ss.status.ready_replicas or 0,
            "current_revision": ss.status.current_revision,
        }
        # Update CRD status
        kopf.info(spec, reason="Reconciled", message=f"Coordinator: {status}")
    except k8s.client.ApiException:
        kopf.warn(spec, reason="Missing", message=f"Coordinator StatefulSet not found for {name}")
