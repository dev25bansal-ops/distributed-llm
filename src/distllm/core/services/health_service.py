from loguru import logger
from distllm.core.node_recovery import NodeRecoveryPlan, LayerRedistribution


class HealthService:
    """Health checks, circuit breakers, node recovery."""

    def __init__(self, resource_mgr, health_checker, recovery, metrics_exporter=None):
        self._resource_mgr = resource_mgr
        self._health_checker = health_checker
        self._recovery = recovery
        self.metrics_exporter = metrics_exporter

    def set_metrics_exporter(self, exporter):
        self.metrics_exporter = exporter

    def health_check(self, nodes, node_order, check_cb):
        self._health_checker.metrics_exporter = self.metrics_exporter
        return self._health_checker.check_all(nodes, node_order, check_cb)

    async def health_check_async(self, nodes, node_order, check_cb):
        self._health_checker.metrics_exporter = self.metrics_exporter
        return await self._health_checker.check_all_async(nodes, node_order, check_cb)

    def check_circuit_breaker(self, node_id: str) -> bool:
        return self._resource_mgr.check_circuit_breaker(node_id)

    def record_node_success(self, node_id: str):
        self._resource_mgr.record_success(node_id)

    def record_node_failure(self, node_id: str, metrics_mgr=None):
        self._resource_mgr.record_failure(node_id)
        if metrics_mgr:
            metrics_mgr.increment("node_failures")
            metrics_mgr.increment("errors")

    def get_circuit_breaker_status(self, node_id: str | None = None) -> dict:
        with self._resource_mgr._lock:
            if node_id:
                return {
                    "failures": self._resource_mgr._node_failure_counts.get(node_id, 0),
                    "recovery_time": self._resource_mgr._node_recovery_time.get(node_id, 0.0),
                    "threshold": self._resource_mgr.cb_config.threshold,
                    "base_delay": self._resource_mgr.cb_config.base_delay,
                    "max_delay": self._resource_mgr.cb_config.max_delay,
                }
            return {
                "nodes": {
                    nid: {
                        "failures": self._resource_mgr._node_failure_counts.get(nid, 0),
                        "recovery_time": self._resource_mgr._node_recovery_time.get(nid, 0.0),
                    }
                    for nid in self._resource_mgr._node_failure_counts
                },
                "threshold": self._resource_mgr.cb_config.threshold,
                "base_delay": self._resource_mgr.cb_config.base_delay,
                "max_delay": self._resource_mgr.cb_config.max_delay,
            }

    def on_node_failure(self, node_id: str):
        logger.warning(f"Resource manager reported failure for {node_id}")
        plan = self._recovery.on_node_failure(node_id)
        if plan.recovered_sequences:
            logger.info(
                f"Recovered {len(plan.recovered_sequences)} sequences "
                f"from failed node {node_id}"
            )

    def on_drain(self, node_id: str, pipeline):
        logger.info(f"Draining node {node_id}")
        pipeline.unregister_node(node_id)

    def on_redistribute(self, failed_node_id: str, plan: NodeRecoveryPlan,
                        pipeline):
        with pipeline._topology_lock:
            survivors = sorted(
                pipeline.nodes.keys(),
                key=lambda nid: pipeline.nodes[nid].start_layer,
            )
            num_survivors = len(survivors)
            if num_survivors == 0:
                logger.error(f"No survivors to redistribute layers from {failed_node_id}")
                return

            total_layers = pipeline.total_layers
            layers_per_node = total_layers // num_survivors
            remainder = total_layers % num_survivors

            for i, nid in enumerate(survivors):
                start = i * layers_per_node + min(i, remainder)
                end = start + layers_per_node - 1
                if i < remainder:
                    end += 1
                node_reg = pipeline.nodes.get(nid)
                if node_reg:
                    redist = LayerRedistribution(
                        surviving_node_id=nid,
                        added_start_layer=node_reg.end_layer + 1 if i > 0 else 0,
                        added_end_layer=end,
                        new_start_layer=start,
                        new_end_layer=end,
                    )
                    node_reg.start_layer = start
                    node_reg.end_layer = end
                    plan.redistributions.append(redist)

            pipeline.node_order = survivors
            logger.info(
                f"Redistributed layers for {failed_node_id}: "
                f"{len(plan.redistributions)} survivors updated"
            )

    def on_recover(self, failed_node_id: str, request_ids: list[str]) -> list[dict]:
        recovered = []
        for rid in request_ids:
            ckpt = self._recovery.get_checkpoint(rid)
            if ckpt is not None:
                recovered.append({
                    "request_id": rid,
                    "prompt_tokens": ckpt.prompt_tokens,
                    "generated_tokens": ckpt.generated_tokens,
                    "kv_cache": ckpt.kv_cache,
                })
        logger.info(f"Recovering {len(recovered)} sequences from {failed_node_id}")
        return recovered

    def on_mark_dead(self, node_id: str):
        logger.warning(f"Node {node_id} marked as dead — pipeline reconfigured")

    def check_recovery(self, request_id: str) -> bool:
        return self._recovery.consume_recovered_flag(request_id)

    def get_recovery_metrics(self) -> dict:
        return self._recovery.get_metrics()

    def close_all(self, nodes):
        self._resource_mgr.close_all(nodes)

    async def close_all_async(self, nodes):
        await self._resource_mgr.close_all_async(nodes)
