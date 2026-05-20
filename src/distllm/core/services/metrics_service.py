class MetricsService:
    """Metrics collection, stats, and observability."""

    def __init__(self, metrics_mgr):
        self._metrics_mgr = metrics_mgr
        self.metrics_exporter = None

    def set_exporter(self, exporter):
        self.metrics_exporter = exporter

    def record_metric(self, metric_name: str, value: float):
        self._metrics_mgr.record(metric_name, value)

    def get_metrics(self) -> dict:
        base = self._metrics_mgr.get_prometheus()
        return base

    def get_with_recovery(self, recovery_metrics: dict) -> dict:
        base = self._metrics_mgr.get_prometheus()
        base["recovery"] = recovery_metrics
        return base

    def increment(self, metric_name: str):
        self._metrics_mgr.increment(metric_name)

    def get_new_module_stats(self, modules: dict) -> dict:
        stats = {}
        for key, mod in modules.items():
            if mod is not None and hasattr(mod, 'stats'):
                try:
                    stats[key] = mod.stats()
                except Exception:
                    pass
        return stats
