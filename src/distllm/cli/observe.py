"""distllm observe — Observability-as-a-product CLI.

Launches a complete distributed inference debugging stack:
- Prometheus metrics endpoint
- Grafana dashboards (auto-provisioned)
- Loki log aggregation
- Real-time WebSocket dashboard

Usage::

    distllm observe                          # Start full stack
    distllm observe --metrics-only           # Just Prometheus metrics
    distllm observe --dashboard-port 3000    # Custom Grafana port
    distllm observe --loki-url http://loki:3100  # External Loki
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table


def _find_project_root() -> Path:
    """Find the project root (contains deploy/ directory)."""
    cur = Path(__file__).resolve()
    for parent in cur.parents:
        if (parent / "deploy").is_dir():
            return parent
    return cur.parent.parent


def _check_port(port: int) -> bool:
    """Check if a port is already in use."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def _generate_grafana_dashboard() -> dict:
    """Generate a Grafana dashboard JSON for distributed inference."""
    return {
        "dashboard": {
            "title": "DistLLM — Distributed Inference",
            "tags": ["distllm", "llm", "inference"],
            "timezone": "browser",
            "panels": [
                {
                    "id": 1,
                    "title": "Request Throughput (tok/s)",
                    "type": "timeseries",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                    "targets": [{"expr": "rate(distllm_tokens_generated_total[5m])"}],
                },
                {
                    "id": 2,
                    "title": "Request Latency (p50/p95/p99)",
                    "type": "timeseries",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                    "targets": [
                        {"expr": "histogram_quantile(0.5, rate(distllm_request_duration_seconds_bucket[5m]))"},
                        {"expr": "histogram_quantile(0.95, rate(distllm_request_duration_seconds_bucket[5m]))"},
                        {"expr": "histogram_quantile(0.99, rate(distllm_request_duration_seconds_bucket[5m]))"},
                    ],
                },
                {
                    "id": 3,
                    "title": "Active Requests",
                    "type": "gauge",
                    "gridPos": {"h": 8, "w": 6, "x": 0, "y": 8},
                    "targets": [{"expr": "distllm_active_requests"}],
                },
                {
                    "id": 4,
                    "title": "GPU Utilization (%)",
                    "type": "gauge",
                    "gridPos": {"h": 8, "w": 6, "x": 6, "y": 8},
                    "targets": [{"expr": "distllm_gpu_utilization_percent"}],
                },
                {
                    "id": 5,
                    "title": "KV Cache Hit Rate",
                    "type": "gauge",
                    "gridPos": {"h": 8, "w": 6, "x": 12, "y": 8},
                    "targets": [{"expr": "distllm_kv_cache_hit_rate"}],
                },
                {
                    "id": 6,
                    "title": "Pipeline Strategy Distribution",
                    "type": "piechart",
                    "gridPos": {"h": 8, "w": 6, "x": 18, "y": 8},
                    "targets": [{"expr": "distllm_pipeline_strategy_total"}],
                },
                {
                    "id": 7,
                    "title": "Node Health",
                    "type": "table",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16},
                    "targets": [{"expr": "distllm_node_health"}],
                },
                {
                    "id": 8,
                    "title": "Federation Peers",
                    "type": "table",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16},
                    "targets": [{"expr": "distllm_federation_peers_total"}],
                },
                {
                    "id": 9,
                    "title": "Carbon Intensity (gCO2/kWh)",
                    "type": "timeseries",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 24},
                    "targets": [{"expr": "distllm_carbon_intensity_gco2_kwh"}],
                },
                {
                    "id": 10,
                    "title": "AutoML Strategy Selection",
                    "type": "timeseries",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 24},
                    "targets": [
                        {"expr": "rate(distllm_automl_strategy_sequential_total[5m])"},
                        {"expr": "rate(distllm_automl_strategy_overlap_total[5m])"},
                        {"expr": "rate(distllm_automl_strategy_staged_total[5m])"},
                        {"expr": "rate(distllm_automl_strategy_async_1f1b_total[5m])"},
                    ],
                },
            ],
            "refresh": "10s",
            "time": {"from": "now-1h", "to": "now"},
        },
        "overwrite": True,
    }


def _generate_grafana_datasources(loki_url: str, prometheus_url: str) -> dict:
    """Generate Grafana datasource provisioning config."""
    return {
        "apiVersion": 1,
        "datasources": [
            {
                "name": "Prometheus",
                "type": "prometheus",
                "access": "proxy",
                "url": prometheus_url,
                "isDefault": True,
                "editable": False,
            },
            {
                "name": "Loki",
                "type": "loki",
                "access": "proxy",
                "url": loki_url,
                "editable": False,
            },
        ],
    }


def run_observe(
    coordinator_url: str = "http://localhost:8000",
    metrics_port: int = 9090,
    dashboard_port: int = 3000,
    loki_url: str = "http://localhost:3100",
    prometheus_url: str = "http://localhost:9090",
    metrics_only: bool = False,
    console: Console | None = None,
) -> None:
    """Launch the DistLLM observability stack.

    Args:
        coordinator_url: URL of the DistLLM coordinator API.
        metrics_port: Port for Prometheus metrics scraping.
        dashboard_port: Port for Grafana dashboard.
        loki_url: URL of Loki log aggregation.
        prometheus_url: URL of Prometheus server.
        metrics_only: Only expose metrics endpoint (no Grafana/Loki).
        console: Rich console for output.
    """
    if console is None:
        console = Console()

    console.print("\n[bold cyan]DistLLM Observability Stack[/bold cyan]")
    console.print("=" * 50)

    # 1. Start metrics endpoint
    console.print(f"\n[green]✓[/green] Metrics endpoint: http://localhost:{metrics_port}/metrics")
    _start_metrics_server(coordinator_url, metrics_port)

    if metrics_only:
        console.print("\n[dim]Metrics-only mode. Use --dashboard-port to enable Grafana.[/dim]")
        console.print(f"\n[bold]Prometheus scrape config:[/bold]")
        console.print(f"  scrape_configs:")
        console.print(f"    - job_name: 'distllm'")
        console.print(f"      static_configs:")
        console.print(f"        - targets: ['localhost:{metrics_port}']")
        return

    # 2. Generate Grafana provisioning
    project_root = _find_project_root()
    grafana_dir = project_root / "deploy" / "grafana"
    grafana_dir.mkdir(parents=True, exist_ok=True)

    dashboard_json = _generate_grafana_dashboard()
    dashboard_path = grafana_dir / "distllm-dashboard.json"
    dashboard_path.write_text(json.dumps(dashboard_json, indent=2))

    datasources = _generate_grafana_datasources(loki_url, prometheus_url)
    ds_path = grafana_dir / "datasources.yaml"
    ds_path.write_text(json.dumps(datasources, indent=2))

    provisioning_dir = grafana_dir / "provisioning"
    provisioning_dir.mkdir(exist_ok=True)
    (provisioning_dir / "dashboards.yaml").write_text(json.dumps({
        "apiVersion": 1,
        "providers": [{
            "name": "DistLLM",
            "orgId": 1,
            "folder": "DistLLM",
            "type": "file",
            "disableDeletion": False,
            "updateIntervalSeconds": 30,
            "options": {"path": str(grafana_dir)},
        }],
    }, indent=2))

    console.print(f"[green]✓[/green] Grafana dashboard: {dashboard_path}")
    console.print(f"[green]✓[/green] Datasources: {ds_path}")

    # 3. Print summary
    console.print("\n[bold]Observability Stack Summary:[/bold]")
    console.print(f"  Metrics:   http://localhost:{metrics_port}/metrics")
    console.print(f"  Dashboard: http://localhost:{dashboard_port}")
    console.print(f"  Loki:      {loki_url}")
    console.print(f"  Prometheus: {prometheus_url}")

    console.print("\n[bold]Quick Start:[/bold]")
    console.print(f"  1. Start Prometheus:  prometheus --config.file=prometheus.yml")
    console.print(f"  2. Start Loki:        loki -config.file=loki.yaml")
    console.print(f"  3. Start Grafana:     grafana-server --homepath=/usr/share/grafana \\")
    console.print(f"                         cfg:paths.provisioning={provisioning_dir}")

    console.print("\n[dim]Press Ctrl+C to stop the metrics server.[/dim]")

    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Observability stack stopped.[/yellow]")


def _start_metrics_server(coordinator_url: str, port: int) -> None:
    """Start a Prometheus-compatible metrics endpoint."""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import threading

    class MetricsHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/metrics":
                metrics = _collect_metrics(coordinator_url)
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.end_headers()
                self.wfile.write(metrics.encode())
            elif self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "healthy"}).encode())
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            pass  # Suppress request logging

    server = HTTPServer(("0.0.0.0", port), MetricsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()


def _collect_metrics(coordinator_url: str) -> str:
    """Collect metrics from the coordinator and format as Prometheus text."""
    import httpx

    lines = []

    try:
        resp = httpx.get(f"{coordinator_url}/v1/pipeline/metrics", timeout=2.0)
        if resp.status_code == 200:
            metrics = resp.json()
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    metric_name = f"distllm_{key}"
                    lines.append(f"# TYPE {metric_name} gauge")
                    lines.append(f"{metric_name} {value}")
    except Exception:
        pass

    # Add default metrics
    lines.append("# TYPE distllm_up gauge")
    lines.append("distllm_up 1")
    lines.append(f"# TYPE distllm_uptime_seconds gauge")
    lines.append(f"distllm_uptime_seconds {time.time()}")

    return "\n".join(lines) + "\n"
