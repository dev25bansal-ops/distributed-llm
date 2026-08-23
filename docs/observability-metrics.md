# DistLLM Observability — Prometheus Metric Dictionary

Reference for every metric family exposed at `GET /metrics`. Every name,
type, label, and default value below was verified against a live scrape of
the running API (in-process FastAPI `TestClient` against `distllm.api.server`),
2026-08-24.

**Source of truth:** `src/distllm/observability/exporter.py`
(`DistLLMPrometheusExporter`, a private `CollectorRegistry`).
The exporter instance created at module load in `src/distllm/api/server.py`
is shared by:

| Consumer | File | What it records |
|---|---|---|
| `ObservabilityMiddleware` | `src/distllm/api/observability_middleware.py` | RED metrics (requests/errors/duration), optional cost/GPU-hour metrics |
| `/metrics` route | `src/distllm/api/routes/health.py` | Serves the registry via `generate_latest` |

## How `/metrics` composes its output

1. **API-layer registry** (always present): the full `DistLLMPrometheusExporter`
   registry described below.
2. **Coordinator registry** (only when the coordinator was constructed with
   `metrics_exporter=`): a second `DistLLMPrometheusExporter` populated from
   live scheduler/node state by `populate_gauges()` — same metric names, so it
   does not add new families.
3. **Service-status gauges**: when no coordinator is loaded, the route appends
   `distllm_service_up` and `distllm_coordinator_loaded` (both `0`). If a
   coordinator exists but has **no** Prometheus exporter attached, the route
   falls back to rendering `coordinator.get_metrics()` as ad-hoc gauges
   (dynamic names — not covered by this dictionary).

> **Known quirk (API area):** in the no-coordinator path the route returns a
> bare Python `str`, which FastAPI serializes as a **JSON-encoded string**
> (`content-type: application/json`) rather than Prometheus text format. The
> exposition text itself is correct once decoded; scrapers should expect
> `text/plain; version=0.0.4` only when a coordinator with an exporter is up.
> Tracked here because it affects scraping, not fixed in this area.

## Label vocabulary

| Label | Appears on | Values |
|---|---|---|
| `method` | request metrics | HTTP verb (`GET`, `POST`, …) |
| `status` | `distllm_requests_total` | `success` or `error` — `error` **only** when the handler raised an unhandled exception; HTTP error responses (404/500 returned normally) are still `success` |
| `type` | `distllm_errors_total` | `http_500` (the only value the middleware emits today) |
| `model` | most request/token/cost metrics | constant `"distributed-llm"` (middleware default) |
| `tenant` | request/cost metrics | constant `"default"` (middleware default) |
| `node_id`, `layer_range` | node metrics | pipeline node identity |
| `target_node` | `distllm_circuit_breaker_state` | downstream node id |
| `le` | histogram buckets | bucket upper bound |
| `metric`, `type` | `distllm_anomaly_detected_total` | anomaly-detector sample kind |

There is deliberately **no `route`/`path` label** (cardinality control); group
traffic by `method` (+ `status`) instead. The middleware instruments *every*
request including `/health` and `/metrics` itself.

## Metric dictionary

### Request layer (RED) — written by `ObservabilityMiddleware`

| Metric | Type | Labels | Meaning | Example PromQL |
|---|---|---|---|---|
| `distllm_requests_total` | Counter | `method, status, model, tenant` | Total requests processed | Rate by method/status: `sum by (method, status) (rate(distllm_requests_total[5m]))` |
| `distllm_request_latency_seconds` | Histogram | `method, model, tenant` | End-to-end request latency. Buckets: 0.1, 0.5, 1, 2, 5, 10, 30, 60, 120 s | p95: `histogram_quantile(0.95, sum by (le) (rate(distllm_request_latency_seconds_bucket[5m])))` · avg: `sum(rate(distllm_request_latency_seconds_sum[5m])) / sum(rate(distllm_request_latency_seconds_count[5m]))` |
| `distllm_request_duration_seconds` | Histogram | `method, model, tenant` | Request duration (second latency series with finer low end). Buckets: 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 30, 60 s | p95: `histogram_quantile(0.95, sum by (le) (rate(distllm_request_duration_seconds_bucket[5m])))` |
| `distllm_errors_total` | Counter | `type, model, tenant` | Unhandled exceptions caught by middleware (`type="http_500"`). Not incremented for normal 4xx/5xx responses — use `distllm_requests_total{status="error"}` for those | Error ratio: `sum(rate(distllm_errors_total[5m])) / sum(rate(distllm_requests_total[5m]))` |
| `distllm_request_cost_total` | Counter ($ accumulated) | `model, tenant` | Estimated USD cost per request; only incremented when a route sets `request.state._request_cost` (cost-tracking middleware) | $/hour: `sum(rate(distllm_request_cost_total[5m])) * 3600` |
| `distllm_request_gpu_hours` → exposed as **`distllm_request_gpu_hours_total`** | Counter | `model, tenant` | GPU-hours consumed per request (same `_request_cost` trigger). prometheus_client appends `_total` to Counter names, so the registered Python name differs from the exposed name | GPU-h/hour: `sum(rate(distllm_request_gpu_hours_total[5m])) * 3600` |

### Token generation

| Metric | Type | Labels | Meaning | Example PromQL |
|---|---|---|---|---|
| `distllm_tokens_generated_total` | Counter | `model, tenant` | Total tokens generated | tok/s: `sum by (model) (rate(distllm_tokens_generated_total[5m]))` |
| `distllm_token_generation_latency_seconds` | Histogram | `model` | Time to generate a single token. Buckets: 5 ms … 5 s | p95 per-token: `histogram_quantile(0.95, sum by (le, model) (rate(distllm_token_generation_latency_seconds_bucket[5m])))` |
| `distllm_tokens_per_second` | Gauge | `model` | Current token generation rate snapshot | `distllm_tokens_per_second` |

### Nodes

| Metric | Type | Labels | Meaning | Example PromQL |
|---|---|---|---|---|
| `distllm_node_health` | Gauge (1/0) | `node_id, layer_range` | Node health status | Unhealthy nodes: `count(distllm_node_health == 0)` |
| `distllm_node_gpu_utilization_percent` | Gauge (%) | `node_id` | GPU utilization percentage | `max by () (distllm_node_gpu_utilization_percent)` |
| `distllm_node_gpu_memory_bytes` | Gauge (bytes) | `node_id` | GPU memory used | `topk(3, distllm_node_gpu_memory_bytes)` |
| `distllm_node_latency_p50_ms` | Gauge (ms) | `node_id` | p50 inference latency per node | `distllm_node_latency_p50_ms` |
| `distllm_node_latency_p99_ms` | Gauge (ms) | `node_id` | p99 inference latency per node | `max(distllm_node_latency_p99_ms)` |

### Coordinator / scheduling

| Metric | Type | Labels | Meaning | Example PromQL |
|---|---|---|---|---|
| `distllm_coordinator_queue_depth` | Gauge | — | Pending requests in the batch-scheduler queue | `distllm_coordinator_queue_depth` |
| `distllm_coordinator_active_requests` | Gauge | — | Currently processing requests | `distllm_coordinator_active_requests` |
| `distllm_circuit_breaker_state` | Gauge (0=closed, 1=open) | `target_node` | Circuit-breaker state per downstream node | Open breakers: `sum(distllm_circuit_breaker_state)` |
| `distllm_active_nodes` | Gauge | — | Number of active healthy nodes | `distllm_active_nodes` |

### Alerting-critical

| Metric | Type | Labels | Meaning | Example PromQL |
|---|---|---|---|---|
| `distllm_kv_cache_usage_ratio` | Gauge (0.0–1.0) | `node_id` | KV cache memory usage ratio | Max across cluster: `max(distllm_kv_cache_usage_ratio)` |

### Self-healing recovery

| Metric | Type | Labels | Meaning | Example PromQL |
|---|---|---|---|---|
| `distllm_recovery_total` | Counter | — | Node recovery events triggered | `increase(distllm_recovery_total[1h])` |
| `distllm_recovery_sequences_recovered_total` | Counter | — | In-flight sequences recovered from failed nodes | `sum(rate(distllm_recovery_sequences_recovered_total[5m]))` |
| `distllm_recovery_sequences_lost_total` | Counter | — | In-flight sequences lost to node failure | `increase(distllm_recovery_sequences_lost_total[1h]) > 0` (alert) |
| `distllm_recovery_duration_ms` | Histogram (**milliseconds**) | — | Duration of node recovery. Buckets: 10, 50, 100, 500, 1000, 5000, 10000, 30000 ms | p95 (ms): `histogram_quantile(0.95, sum by (le) (rate(distllm_recovery_duration_ms_bucket[5m])))` |
| `distllm_draining_nodes` | Gauge | — | Nodes currently draining | `distllm_draining_nodes` |
| `distllm_dead_nodes` | Gauge | — | Nodes marked dead (awaiting replacement) | `distllm_dead_nodes > 0` (alert) |

### Cost tracking

| Metric | Type | Labels | Meaning | Example PromQL |
|---|---|---|---|---|
| `distllm_cost_per_hour_total` | Gauge ($) | — | Total $/hour of all active nodes | `distllm_cost_per_hour_total` vs budget: `clamp_min(distllm_budget_remaining - distllm_cost_per_hour_total, 0)` |
| `distllm_budget_remaining` | Gauge ($/hour) | — | Remaining hourly budget | `distllm_budget_remaining` |
| `distllm_spot_interruptions_total` | Counter | — | Spot-instance interruptions | `increase(distllm_spot_interruptions_total[24h])` |

### Anomaly detection

| Metric | Type | Labels | Meaning | Example PromQL |
|---|---|---|---|---|
| `distllm_anomaly_detected_total` | Counter | `metric, type` | Anomaly-detection events (e.g. sample kinds `http_error_rate`, `http_request_duration`) | `sum by (type) (rate(distllm_anomaly_detected_total[15m]))` |

### Service status (no-coordinator path only)

| Metric | Type | Labels | Meaning | Example PromQL |
|---|---|---|---|---|
| `distllm_service_up` | Gauge | — | `0` when the metrics route runs without a coordinator | `up{job="distllm"} == 0 or distllm_service_up == 0` |
| `distllm_coordinator_loaded` | Gauge | — | `0` when no coordinator is loaded | `distllm_coordinator_loaded == 0` (alert during expected serving hours) |

## Exposition-format notes (verified)

Count: **29 metric families** from the exporter + **2** service-status gauges.

1. **`distllm_request_gpu_hours` is exposed as `distllm_request_gpu_hours_total`.**
   `prometheus_client.Counter` appends `_total` to names that lack it. Dashboards
   and alerts must use the `_total` form.
2. **Labeled families emit only `# HELP`/`# TYPE` lines until first used.**
   e.g. `distllm_errors_total`, `distllm_request_cost_total`,
   `distllm_tokens_generated_total`, all `distllm_node_*` gauges produce no
   sample rows until something observes them — panels will show “No data”
   until then. This is expected, not missing instrumentation.
3. **`_created` companion series** are auto-emitted for every counter and
   histogram (e.g. `distllm_requests_created`,
   `distllm_request_latency_seconds_created`, and — because the recovery
   counter registers as `…_total` — `distllm_recovery_created`). Ignore them
   unless doing increase-over-process-lifetime math.
4. **Unlabeled gauges/counters always appear** at `0` even before use:
   `distllm_active_nodes`, `distllm_coordinator_queue_depth`,
   `distllm_coordinator_active_requests`, `distllm_cost_per_hour_total`,
   `distllm_budget_remaining`, `distllm_spot_interruptions_total`,
   `distllm_recovery_*`, `distllm_draining_nodes`, `distllm_dead_nodes`.
5. The two request histograms observe the same duration; prefer
   `distllm_request_latency_seconds` for dashboards (longer tail buckets up to
   120 s) and `distllm_request_duration_seconds` for sub-100 ms resolution.
6. Non-Prometheus signals also produced by the observability stack but served
   elsewhere: OTel spans (`TracingMiddleware`/`ObservabilityMiddleware`, span
   attrs `http.method`, `http.target`, `http.status_code`, `http.duration_s`,
   `http.request_id`), collector JSON snapshots at `/api/metrics/collector`,
   SSE/WebSocket streams (`/api/metrics/stream`, `/ws/metrics`). Those are not
   part of the `/metrics` exposition.

## Reference dashboard

A Grafana dashboard built exclusively on the families above ships at
`dashboards/distllm-overview.json` (import → select your Prometheus
datasource).

> Note: the older `dashboards/distllm-grafana.json` predates this dictionary
> and queries names that do **not** exist in the exporter (e.g.
> `distllm_latency_p50_ms`, `distllm_gpu_utilization`). Treat it as legacy;
> real equivalents are `distllm_node_latency_p50_ms` and
> `distllm_node_gpu_utilization_percent`.
