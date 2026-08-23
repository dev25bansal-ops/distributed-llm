---
tags:
  - observability
  - platform
  - telemetry
  - cloud
aliases:
  - Platform Services
  - Observability
---
# Platform Services — observability, health, caching, cloud, errors & support packages

**Covers the transverse support packages of `src/distllm/`:** `observability/`, `health/`, `cache/`, `cloud/`, `errors/`, `utils/`, `terraform/`, `verification/`, `benchmarks/`, plus the in-repo `dashboard/` (see [[08 Frontends]] for the web UI surface).

## `observability/` — telemetry backbone (20 files, ~11.7K LOC)

| file | LOC | purpose |
|------|-----|---------|
| `exporter.py` | 210 | `DistLLMPrometheusExporter` — counters/gauges/histograms + `generate_metrics()` |
| `metrics.py` | 88 | OTel `setup_metrics`/`get_meter`; `DistLLMMetrics` |
| `metrics_completeness.py` | 314 | pool/queue/tenant-cost metrics + completeness |
| `logging.py` | 190 | loguru JSON/human sinks with OTel trace injection + schema validation |
| `logging_config.py` | 702 | `LogSchema`, `HALokiSink`, log retention, debug sampler |
| `loki_sink.py` | 152 | batched loguru sink → Grafana Loki |
| `tracing.py` | 553 | OTel `setup_tracing` + gRPC instrumentation, W3C propagation |
| `tracing_config.py` | 205 | tail-based sampler, trace loki/exemplar exporters, ASGI middleware |
| `otel_logging.py` | 243 | native OTel LogRecord bridge |
| `spans.py` | 553 | prefill/decode/generate/pipeline/KV/spec span context managers |
| `alerting_config.py` | 716 | Prometheus rules + Alertmanager YAML, SLO burn-rate, runbooks |
| `slo_config.py` | 1,327 | `SLO` error budget/exhaustion, availability SLI, Grafana dashboard |
| `capacity_planning.py` | 2,478 | `CapacityPlanner`, what-if, HPA/VPA YAML, GPU billing, dashboards |
| `capacity_models.py` | 315 | dataclasses/enums + regression for capacity |
| `self_healing_config.py` | 1,059 | GPU reset, drain coordinator, failure predictor, recovery SLA |
| `health_config.py` | 1,046 | deep/KV/cascade/aggregate health probes + configurator (see note) |
| `incident_response.py` | 654 | incidents, MTTA/MTTR, postmortems, runbooks, on-call |
| `exporter_datadog.py` | 1,137 | push metrics/traces/logs to Datadog/New Relic + OTLP, pynvml/psutil |
| `wandb_monitor.py` | 56 | optional W&B monitor stub |
| `__init__.py` | 105 | public bundle |

## `health/` — runtime node health (5 files)

| file | LOC | purpose |
|------|-----|---------|
| `service.py` | 144 | `HealthCheckService` — periodic gRPC probes, backoff, death callbacks |
| `state.py` | 61 | `NodeState`/`HealthRecord`/`HealthStateStore` |
| `prober.py` | 25 | single gRPC probe |
| `failover.py` | 67 | HEALTHY/DEGRADED/UNHEALTHY/OFFLINE state machine |
| `__init__.py` | 4 | re-exports |

## `cache/` — cross-cluster prefix KV cache (3 files)

| file | LOC | purpose |
|------|-----|---------|
| `crdt.py` | 399 | conflict-free replicated types (ORSet/LWWRegister/CRDTCacheMap, HLC) |
| `cross_cluster_prefix_index.py` | 595 | content-addressed cross-cluster prefix index + gossip digest |
| `__init__.py` | 15 | re-exports |

## `cloud/` — spot market & multi-provider (7 files)

| file | LOC | purpose |
|------|-----|---------|
| `spot_orchestrator.py` | 1,390 | multi-provider GPU spot: discovery/bidding/launch/cost-monitor (RunPod/Vast/Salad) |
| `common.py` | 172 | `GPUSpec`/`PriceQuote`/`PricingFetcher` ABCs |
| `aws.py` | 181 | AWS spot pricing + availability |
| `azure.py` | 128 | Azure VM SKU/spot pricing |
| `gcp.py` | 215 | GCE accelerator catalog, committed discounts |
| `worker_agent.py` | 291 | **SCAFFOLD** hybrid control-plane join client (E13) |
| `__init__.py` | 50 | re-exports |

## `errors/` — error hierarchy & retry (4 files)

| file | LOC | purpose |
|------|-----|---------|
| `types.py` | 561 | `DistLLMError` root + ~30 subclasses w/ code, remediation, docs_url |
| `retry.py` | 142 | `RetryPolicy`, `with_retry(_async)`, `retry_grpc_call` |
| `policies.py` | 96 | error→retry-policy registry |
| `__init__.py` | 100 | central re-export (imported by nearly all backends & core) |

## `utils/` — helpers (4 files)

| file | lines | purpose |
|------|-------|---------|
| `gbnf_grammar.py` | 74 | GBNF grammar generation from JSON Schema |
| `lazy_imports.py` | 71 | canonical lazy-import machinery (replaced 4 duplicated copies) |
| `scheduling.py` | 32 | log-scale length bucketing |
| `__init__.py` | 0 | marker |

## `terraform/` + `verification/` + `benchmarks/` (in-tree)

| file | lines | purpose |
|------|-------|---------|
| `terraform/__init__.py` | 225 | Terraform IaC resources → coordinator REST (Cluster/Federation/Tenant/Model) |
| `verification/` (6) | 1,541 | correctness comparator, hash registry, report, runner |
| `benchmarks/scaling.py` | 240 | throughput vs size/nodes scaling benchmark |
| `benchmarks/evaluation_harness.py` | 55 | lm-evaluation-harness integration |
| `benchmarks/__init__.py` | 8 | **broken** — `cost_comparison` import target missing (stale) |

## Notes / dead code
- **Duplicate health impls**: `observability/health_config.py` vs `health/` package overlap in intent (different connectors).
- `benchmarks/__init__.py` has a **broken import** (`distllm.benchmarks.cost_comparison` does not exist) — `import distllm.benchmarks` fails.
- `slo_config` burn-rate logic mirrors `alerting_config.SLOBurnRateAlert`.
- `cloud/*` are an optional utility layer with **no eager runtime consumers** (workers join via `worker_agent`).
- `CapacityConfigurator._collect_snapshot()` and `SelfHealingConfigurator._monitor_loop` are placeholders requiring real wiring.

## Tests
`tests/observability/`, `tests/health/`, `tests/cache/` (`test_cross_cluster_prefix_index`), `tests/cloud/` (`test_spot_orchestrator` 849 LOC), `tests/errors/`, `tests/utils/`.