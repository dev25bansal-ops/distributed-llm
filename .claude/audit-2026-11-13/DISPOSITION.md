# Audit finding disposition ledger — FINAL

Total findings: 114

- RESOLVED (fixed or verified-fixed): 112
- ENVIRONMENTAL (not fixable in-repo): 2

All 11 fixer agents completed their briefs; coordinator closed 7 routed
findings the agents could not reach (auth_plugin RBAC gate, OTel tracer
pollution, cluster_state_store leader lease, coordinator_election non-leader
snapshots + health re-application, breaker-state persistence, request_pipeline
draft_tokens TypeError, quota-meter singleton leak, webrtc offer 30s deadlock).

| Sev | Audit | File | Line | State |
|---|---|---|---|---|
| crit | ha-fence | src/distllm/core/ha_coordinator.py | 167 | FIXED (7f89b8bd) |
| crit | key-precedence | src/distllm/dist/worker.py | 569 | FIXED (aafa56d4) |
| crit | optional-imports | src/distllm/security/watermark.py | 43 | FIXED (481f8236) |
| crit | red-metrics | src/distllm/core/coordinator.py | 325 | FIXED (481f8236) |
| crit | red-metrics | src/distllm/api/server_lifespan.py | 61 | FIXED (481f8236) |
| crit | shipped-configs | integrations/kubernetes/helm/distllm/templates/poddisruptionbudget.yaml | 8 | FIXED (aafa56d4) |
| crit | spec-decode | src/distllm/core/distributed_speculative.py | 1297 | FIXED (481f8236) |
| high | cache-accounting | src/distllm/core/cache_manager.py | 347 | fix-caches (7/7) |
| high | cache-accounting | src/distllm/dist/prefix_cache.py | 275 | fix-caches (7/7) |
| high | cache-accounting | src/distllm/dist/prefix_cache.py | 243 | fix-caches (7/7) |
| high | cache-accounting | src/distllm/core/cache_manager.py | 189 | fix-caches (7/7) |
| high | cache-accounting | src/distllm/core/advanced_scheduling/tiered_store.py | 355 | fix-caches (7/7) |
| high | dedup-auth | src/distllm/api/semantic_cache.py | 329 | fix-caches (2/2) |
| high | docs-claims | docs/BENCHMARKS.md | 127 | FIXED (fef0756f) |
| high | error-messages | src/distllm/api/streaming.py | 437 | fix-streaming (5/5) |
| high | ha-fence | src/distllm/core/coordinator.py | 234 | FIXED (7f89b8bd) |
| high | ha-fence | src/distllm/core/ha_coordinator.py | 306 | FIXED (7f89b8bd) |
| high | ha-fence | src/distllm/core/ha_coordinator.py | 245 | FIXED (7f89b8bd) |
| high | ha-fence | src/distllm/core/ha_coordinator.py | 315 | FIXED (7f89b8bd) |
| high | key-precedence | src/distllm/core/api_key_store.py | 529 | fix-keys (8/8) |
| high | key-precedence | src/distllm/cli/chat_stream.py | 224 | fix-keys (8/8) |
| high | net-path | (network path) LAPTOP-MSLI2NU1 egress — Windows 11 + WSL2 Ubuntu; no repo file implicated | None | ENVIRONMENTAL (2 findings; egress drops Tailscale control SNI) |
| high | node-lifecycle | src/distllm/cli/cluster.py | 414 | FIXED (e78796c0) |
| high | node-lifecycle | src/distllm/dist/pipeline/orchestrator.py | 424 | FIXED (e78796c0) |
| high | node-lifecycle | src/distllm/api/routes/admin.py | 322 | FIXED (e78796c0) |
| high | optional-imports | src/distllm/core/structured_output/__init__.py | 6 | FIXED (481f8236) |
| high | optional-imports | src/distllm/core/coordinator.py | 20 | FIXED (481f8236) |
| high | optional-imports | src/distllm/models/partitioner.py | 17 | FIXED (481f8236) |
| high | packaging | src/distllm/api/server.py | 997 | fix-packaging (4/4) |
| high | packaging | pyproject.toml | 66 | fix-packaging (4/4) |
| high | redundant-exec | src/distllm/dist/pipeline/orchestrator.py | 190 | fix-redundancy (7/7) |
| high | redundant-exec | src/distllm/dist/redundant_executor.py | 1320 | fix-redundancy (7/7) |
| high | redundant-exec | src/distllm/dist/redundant_executor.py | 1303 | fix-redundancy (7/7) |
| high | redundant-exec | src/distllm/dist/redundant_executor.py | 737 | fix-redundancy (7/7) |
| high | redundant-exec | src/distllm/dist/redundant.py | 113 | fix-redundancy (7/7) |
| high | shipped-configs | integrations/kubernetes/kustomize/overlays/dev/kustomization.yaml | 13 | fix-packaging (3/3) |
| high | shipped-configs | integrations/docker/docker-compose.gpu.yml | 151 | fix-packaging (3/3) |
| high | spec-decode | src/distllm/core/request_pipeline.py | 609 | fix-specdecode (8/8) |
| high | spec-decode | src/distllm/core/speculative_decoder.py | 257 | fix-specdecode (8/8) |
| high | sso-fork | src/distllm/api/middleware.py | 466 | fix-sso (7/7) |
| high | sso-fork | src/distllm/api/sso_middleware.py | 435 | fix-sso (7/7) |
| high | sso-fork | src/distllm/api/sso_auth.py | 319 | fix-sso (7/7) |
| high | streaming | src/distllm/api/streaming.py | 387 | fix-streaming (7/7) |
| high | streaming | src/distllm/api/streaming.py | 445 | fix-streaming (7/7) |
| high | streaming | src/distllm/api/routes/health.py | 618 | fix-streaming (7/7) |
| high | test-order | tests/dist/test_cache_digest.py | 18 | FIXED (481f8236) |
| med | cache-accounting | src/distllm/core/advanced_scheduling/tiered_store.py | 429 | fix-caches (7/7) |
| med | cache-accounting | src/distllm/core/cache_manager.py | 578 | fix-caches (7/7) |
| med | config-override | src/distllm/core/semantic_cache.py | 210 | fix-caches (8/8) |
| med | config-override | src/distllm/core/semantic_cache.py | 217 | fix-caches (8/8) |
| med | config-override | src/distllm/core/coordinator_config.py | 195 | fix-caches (8/8) |
| med | config-override | src/distllm/core/scheduler/budget_computer.py | 165 | fix-caches (8/8) |
| med | config-override | src/distllm/dist/attention.py | 247 | fix-caches (8/8) |
| med | dedup-auth | src/distllm/api/semantic_cache.py | 343 | fix-caches (2/2) |
| med | docs-claims | docs/OPS_RUNBOOK.md | 41 | FIXED (fef0756f) |
| med | docs-claims | docs/QUICKSTART.md | 173 | FIXED (fef0756f) |
| med | docs-claims | docs/TROUBLESHOOTING.md | 516 | FIXED (fef0756f) |
| med | docs-claims | docs/benchmark-results.md | 281 | FIXED (fef0756f) |
| med | error-messages | src/distllm/api/routes/model_registry.py | 221 | fix-streaming (5/5) |
| med | golden-path | src/distllm/security/watermark.py | 43 | fix-packaging (2/2) |
| med | ha-fence | src/distllm/core/split_brain.py | 189 | fix-hafence (7/7 verified) |
| med | ha-fence | src/distllm/core/cluster_state_store.py | 341 | fix-hafence (7/7 verified) |
| med | key-precedence | src/distllm/cli/cluster.py | 399 | fix-keys (8/8) |
| med | key-precedence | src/distllm/cli/client.py | 71 | fix-keys (8/8) |
| med | key-precedence | src/distllm/dist/worker.py | 453 | FIXED (aafa56d4) |
| med | net-path | (environment) Windows routing table + WSL2 NAT | None | ENVIRONMENTAL (2 findings; egress drops Tailscale control SNI) |
| med | node-lifecycle | src/distllm/api/routes/admin.py | 390 | FIXED (e78796c0) |
| med | node-lifecycle | src/distllm/api/routes/admin.py | 159 | FIXED (e78796c0) |
| med | node-lifecycle | src/distllm/cli/main.py | 977 | fix-lifecycle (5/5 + residual) |
| med | node-lifecycle | src/distllm/core/coordinator.py | 1699 | fix-lifecycle (5/5 + residual) |
| med | optional-imports | src/distllm/core/debug.py | 3 | FIXED (481f8236) |
| med | optional-imports | src/distllm/core/kv_cache.py | 9 | FIXED (481f8236) |
| med | optional-imports | src/distllm/backends/protocol.py | 19 | FIXED (481f8236) |
| med | optional-imports | src/distllm/api/streaming.py | 57 | FIXED (481f8236) |
| med | red-metrics | src/distllm/observability/metrics.py | 87 | fix-metrics (6/6) |
| med | red-metrics | src/distllm/observability/exporter.py | 256 | fix-metrics (6/6) |
| med | redundant-exec | src/distllm/dist/redundant_executor.py | 1408 | fix-redundancy (7/7) |
| med | redundant-exec | src/distllm/dist/redundant_executor.py | 733 | fix-redundancy (7/7) |
| med | sibling-cache | src/distllm/core/cross_model_prefix_sharing.py | 115 | fix-caches (3/3) |
| med | spec-decode | src/distllm/core/speculative_decoder.py | 513 | fix-specdecode (8/8) |
| med | spec-decode | src/distllm/core/speculative_decoder.py | 492 | fix-specdecode (8/8) |
| med | spec-decode | src/distllm/core/speculative_decoder.py | 442 | fix-specdecode (8/8) |
| med | spec-decode | src/distllm/core/speculative_decoder.py | 1016 | fix-specdecode (8/8) |
| med | sso-fork | src/distllm/api/sso_auth.py | 560 | fix-sso (7/7) |
| med | sso-fork | src/distllm/api/sso_middleware.py | 377 | fix-sso (7/7) |
| med | streaming | src/distllm/api/streaming.py | 569 | fix-streaming (7/7) |
| med | streaming | src/distllm/api/streaming.py | 529 | fix-streaming (7/7) |
| med | streaming | src/distllm/api/streaming.py | 437 | fix-streaming (7/7) |
| low | config-override | src/distllm/core/coordinator_config.py | 207 | fix-caches (8/8) |
| low | config-override | src/distllm/dist/async_pipeline.py | 380 | fix-caches (8/8) |
| low | config-override | src/distllm/core/kv_cache.py | 692 | fix-caches (8/8) |
| low | docs-claims | README.md | 174 | FIXED (fef0756f) |
| low | error-messages | src/distllm/api/routes/model_registry.py | 197 | fix-streaming (5/5) |
| low | error-messages | src/distllm/api/routes/webrtc.py | 190 | fix-streaming (5/5) |
| low | error-messages | src/distllm/api/server_errors.py | 72 | fix-streaming (5/5) |
| low | golden-path | src/distllm/cli/selftest.py | 217 | fix-packaging (2/2) |
| low | ha-fence | src/distllm/core/coordinator_election.py | 242 | fix-hafence (7/7 verified) |
| low | key-precedence | src/distllm/cli/audit.py | 191 | fix-keys (8/8) |
| low | key-precedence | src/distllm/core/framework_adapters.py | 52 | fix-keys (8/8) |
| low | node-lifecycle | src/distllm/dist/worker.py | 481 | fix-lifecycle (5/5 + residual) |
| low | optional-imports | src/distllm/core/protocols.py | 10 | FIXED (481f8236) |
| low | optional-imports | src/distllm/dist/transport.py | 27 | FIXED (481f8236) |
| low | packaging | pyproject.toml | 290 | fix-packaging (4/4) |
| low | packaging | src/distllm/core/correctness_harness.py | 53 | fix-packaging (4/4) |
| low | red-metrics | src/distllm/core/request_pipeline.py | 386 | fix-metrics (6/6) |
| low | red-metrics | tests/api/test_auth_middleware.py | 910 | fix-metrics (6/6) |
| low | sibling-cache | src/distllm/core/cross_model_prefix_sharing.py | 182 | fix-caches (3/3) |
| low | sibling-cache | src/distllm/core/cluster_predictive_prefetcher.py | 117 | fix-caches (3/3) |
| low | spec-decode | src/distllm/core/request_pipeline.py | 295 | fix-specdecode (8/8) |
| low | sso-fork | src/distllm/api/sso_middleware.py | 603 | fix-sso (7/7) |
| low | sso-fork | src/distllm/api/api_settings.py | 107 | fix-sso (7/7) |
| low | streaming | src/distllm/api/streaming.py | 170 | fix-streaming (7/7) |
| low | streaming | src/distllm/api/routes/chat.py | 739 | fix-streaming (7/7) |
| low | streaming | src/distllm/api/streaming.py | 482 | fix-streaming (7/7) |
