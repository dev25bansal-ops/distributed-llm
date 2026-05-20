# Implementation Plan: 5 Advanced Features for distributed-llm

## Overview

This plan covers 5 advanced features that build on the existing distributed-llm infrastructure:
Federated Inference Fabric, Heterogeneous Precision Serving, Auto-Speculative Selection,
Cost-Optimized Cloud Inference, and Plugin Marketplace. Each feature is designed to be
independently deliverable with clear dependencies between them where necessary.

---

## Feature 1: Federated Inference Fabric

### Goal
Cross-datacenter inference using existing cluster topology + gossip protocol infrastructure.

### Files to Create (New)

1. **D:/distributed-llm/src/distllm/core/federation_discovery.py**
   - `FederationPeerDiscovery` - bootstrap + DNS-based peer discovery
   - `PeerInfo` dataclass (cluster_id, host, port, edge, metadata)
   - `discover_peers(seed_nodes: list[str]) -> list[PeerInfo]`
   - `register_self(coord_host, coord_port, cluster_id)` - announce to peers

2. **D:/distributed-llm/src/distllm/core/cross_cluster_forwarder.py**
   - `CrossClusterForwarder` - serialize request, send to remote coordinator, stream response
   - `forward_request(remote_coord_url, request, timeout) -> AsyncGenerator[TokenResponse]`
   - `forward_kv_cache(remote_node_id, prefix_hash, kv_data) -> bool`
   - Uses existing `KVCacheTransfer.serialize_kv()` / `deserialize_kv()`

3. **D:/distributed-llm/src/distllm/core/latency_prober.py**
   - `LatencyProber` - async ping loop calling `NodeService.Ping()` RPC
   - `probe_node(node_id, host, port, interval_s=5.0) -> float` (measures RTT)
   - `start_probing()` / `stop_probing()` - background asyncio task
   - Feeds results into `CrossClusterLatencyMonitor.record_latency()`

4. **D:/distributed-llm/src/distllm/core/federation_load_balancer.py**
   - `FederationLoadBalancer` - remote cluster load reporting via heartbeat
   - `report_load(cluster_id, active, pending, gpu_util, queue_depth)`
   - `get_remote_load(cluster_id) -> ClusterLoad | None`
   - Integrates with existing `LoadReporter` in `geo_router.py`

5. **D:/distributed-llm/src/distllm/core/cache_migration.py**
   - `CacheMigrator` - cross-cluster KV cache warmup/migration
   - `migrate_cache(src_node_id, dst_node_id, prefix_hashes) -> bool`
   - Uses `GossipTransport.request_kv_cache()` for data transfer
   - `warm_cache_on_cluster(cluster_id, prefix_hashes) -> bool`

6. **D:/distributed-llm/tests/test_federation_discovery.py**
7. **D:/distributed-llm/tests/test_cross_cluster_forwarder.py**
8. **D:/distributed-llm/tests/test_latency_prober.py**

### Files to Modify (Existing)

1. **D:/distributed-llm/src/distllm/core/cluster_topology.py**
   - Add `FederationManager.discover_and_register_cluster(peer_info: PeerInfo)` method
   - Wire `CrossClusterLatencyMonitor` to `LatencyProber` via callback

2. **D:/distributed-llm/src/distllm/core/latency_aware_batcher.py**
   - Wire `LatencyAwareBatcher` into coordinator batch scheduler
   - Add `batch_by_cluster(requests, federation)` integration method

3. **D:/distributed-llm/src/distllm/core/gossip_transport.py**
   - Add `exchange_advertisements_grpc(peer_id, ad)` - gRPC gossip endpoint
   - Add `request_kv_cache_grpc(peer_id, prefix_hashes)` - gRPC variant
   - Extend `_resolve_peer()` to support federation peer registry

4. **D:/distributed-llm/src/distllm/communication/node_service.py**
   - Update `NodeService.Ping()` to populate `cluster_id` from node config
   - Add latency measurement: client-side RTT = `time.time() - request.timestamp`

5. **D:/distributed-llm/src/distllm/core/coordinator.py**
   - Instantiate `LatencyProber`, `CrossClusterForwarder`, `FederationPeerDiscovery`
   - Add `_federation_loop()` - background task for discovery + probing
   - Wire `GeoRouter.select_target_cluster()` into request routing path

6. **D:/distributed-llm/src/distllm/communication/node.proto**
   - Verify `cluster_id` field in `PingRequest` usage
   - Add `ForwardInferenceRequest` / `ForwardInferenceResponse` for cross-cluster forwarding
   - Add gRPC `GossipExchange` endpoint to `NodeService`

### Key Design Decisions

- **Discovery protocol**: Seed-node bootstrap (simple, reliable) with optional mDNS for LAN.
  DNS-based discovery for production (SRV records). Avoid complex protocols.
- **Cross-cluster forwarding**: HTTP streaming (SSE) fallback for simplicity; gRPC streaming
  for production. Reuse existing `KVCacheTransfer` serialization.
- **Latency probing**: Background asyncio task, not a thread (avoids GIL contention).
  Interval configurable per cluster pair.
- **Cache migration**: Uses existing gossip transport; no new protocol needed. Add explicit
  `migrate_cache()` that wraps the existing `request_kv_cache()` call.

### Implementation Order

1. LatencyProber + wire into CrossClusterLatencyMonitor (unlocks routing decisions)
2. FederationPeerDiscovery (unlocks cross-cluster topology)
3. CrossClusterForwarder (unlocks actual cross-cluster requests)
4. FederationLoadBalancer (enables load-aware routing)
5. CacheMigrator (optimization, can ship last)
6. Coordinator integration + proto updates

### Dependencies on Other Features
- None (foundational - other features benefit from it but do not require it)

---

## Feature 2: Heterogeneous Precision Serving

### Goal
Different nodes running different precision (FP16/INT8/FP8) for mixed-hardware clusters.

### Files to Create (New)

1. **D:/distributed-llm/src/distllm/core/precision_registry.py**
   - `PrecisionRegistry` - tracks per-node precision assignments
   - `NodePrecision` dataclass (node_id, precision, gpu_type, vram_gb, capabilities)
   - `assign_precision(node_id, precision)` / `get_node_precision(node_id)`
   - `get_nodes_with_precision(precision) -> list[str]`
   - Integrates with `QuantizationSelector` for rule-based assignment

2. **D:/distributed-llm/src/distllm/core/precision_aware_partitioner.py**
   - `PrecisionAwarePartitioner` - layer partitioning considering precision constraints
   - `partition_model(model, node_precisions, layer_budgets) -> PartitionPlan`
   - Ensures INT8-capable nodes get MLP layers, FP16 nodes get attention layers
   - Reuses `AdaptivePrecisionEngine.profile_model()` for sensitivity analysis

3. **D:/distributed-llm/src/distllm/core/precision_boundary.py**
   - `PrecisionBoundary` - handles precision conversion at tensor parallelism boundaries
   - `convert_precision(tensor, src_dtype, dst_dtype) -> torch.Tensor`
   - `QuantizationAwareTransport` - wraps tensor serialization with precision metadata
   - Uses `FP8Engine` quantize/dequantize functions for FP8 boundaries

4. **D:/distributed-llm/src/distllm/core/quality_sla.py**
   - `QualitySLA` - per-request quality budget (BF16 quality vs INT4 tolerance)
   - `SLAPolicy` dataclass (min_precision, max_quality_loss, request_class)
   - `evaluate_quality(gold_output, candidate_output) -> float` - KL-divergence based
   - `select_precision_for_request(request, sla_policy) -> torch.dtype`

5. **D:/distributed-llm/tests/test_precision_registry.py**
6. **D:/distributed-llm/tests/test_precision_aware_partitioner.py**
7. **D:/distributed-llm/tests/test_precision_boundary.py**

### Files to Modify (Existing)

1. **D:/distributed-llm/src/distllm/core/adaptive_precision.py**
   - Add `AdaptivePrecisionEngine.get_layer_precision_recommendations()` - public method
     that returns `list[LayerPrecision]` without applying them
   - Add `LayerPrecision.required_precision` field (enum: REQUIRED, RECOMMENDED, OPTIONAL)

2. **D:/distributed-llm/src/distllm/core/quantization_selector.py**
   - Add `QuantizationSelector.select_for_hardware(gpu_type, vram_gb, latency_target)`
   - Extend rule engine to support FP8 (Hopper GPUs)
   - Add `get_compatible_precisions()` - returns list of precisions a node supports

3. **D:/distributed-llm/src/distllm/core/fp8_engine.py**
   - Add `FP8Engine.quantize_tensor_for_transfer(tensor, src_precision, dst_precision)`
   - Add `FP8Engine.dequantize_from_transfer(data, src_precision, dst_precision)`

4. **D:/distributed-llm/src/distllm/core/self_optimizing_engine.py**
   - Wire `SelfOptimizingEngine` callback to update `kv_cache_quant_bits` on nodes
   - Add `tune_precision_per_node()` - optimizes precision assignment based on profiles

5. **D:/distributed-llm/src/distllm/communication/node.proto**
   - Add `node_precision` field to `NodeInfo` (enum: FP32, FP16, BF16, INT8, FP8, INT4)
   - Add `precision` field to `RegistrationResponse` (coordinator assigns precision)

6. **D:/distributed-llm/src/distllm/core/coordinator.py**
   - Instantiate `PrecisionRegistry`, `PrecisionAwarePartitioner`, `QualitySLA`
   - On node registration: query GPU type -> select precision -> assign via registry
   - On request: check SLA -> route to compatible precision node

### Key Design Decisions

- **Precision assignment**: Coordinator-driven (centralized) rather than node-driven.
  Coordinator queries node GPU specs, runs `QuantizationSelector`, assigns precision.
  Simpler than distributed negotiation.
- **Boundary conversion**: Always convert to the highest precision in the path (FP16/BF16).
  INT8 to FP16 dequantization at the boundary; FP8 to FP16 dequantization. This avoids
  cascading quantization errors.
- **Quality SLA**: Simple tiered model (high/medium/low quality) mapped to precision levels.
  High = BF16 required, Medium = FP16 acceptable, Low = INT4/INT8 acceptable.

### Implementation Order

1. PrecisionRegistry (foundation for all other components)
2. QualitySLA (defines the interface for precision selection)
3. Extend QuantizationSelector + AdaptivePrecisionEngine
4. PrecisionAwarePartitioner (uses registry + selector)
5. PrecisionBoundary (communication layer)
6. SelfOptimizingEngine integration
7. Coordinator integration + proto updates

### Dependencies on Other Features
- Depends on Feature 1 (Federation) only for cross-cluster precision routing
- Can ship independently for single-cluster heterogeneous hardware

---

## Feature 3: Auto-Speculative Selection

### Goal
Automatically pick the best speculative method (ngram/medusa/eagle/draft) per workload.

### Files to Create (New)

1. **D:/distributed-llm/src/distllm/core/speculative_profiler.py**
   - `SpeculativeProfiler` - profiles acceptance rates per method
   - `MethodProfile` dataclass (method, acceptance_rate, avg_speedup, tokens_per_sec)
   - `record_acceptance(method, draft_count, accepted_count, generation_time_ms)`
   - `get_best_method(workload_type) -> str` - returns method with highest expected speedup
   - EMA-based acceptance rate tracking (alpha=0.1 for slow decay)

2. **D:/distributed-llm/src/distllm/core/workload_classifier.py**
   - `WorkloadClassifier` - classifies request text into workload types
   - `WorkloadType` enum: CODE, REPETITIVE, DIVERSE, INSTRUCTION, UNKNOWN
   - `classify(text: str) -> WorkloadType` - heuristic-based (ngram entropy, keywords)
   - `classify_features(text) -> dict` - returns entropy, avg_word_len, code_ratio, etc.

3. **D:/distributed-llm/src/distllm/core/speculative_adaptor.py**
   - `SpeculativeAdaptor` - adapts `num_assistant_tokens` based on acceptance rate
   - `adapt_tokens(current_acceptance_rate, base_tokens) -> int`
   - High acceptance (>0.7) -> increase tokens; Low (<0.3) -> decrease; disable if <0.15
   - Integrates with `SelfOptimizingEngine.tunable_params.speculative_decoding_k`

4. **D:/distributed-llm/src/distllm/core/speculative_dashboard.py**
   - `SpeculativeDashboard` - metrics collection and comparison dashboard
   - `record_comparison(method_a, method_b, metrics_a, metrics_b)`
   - `get_comparison_report() -> dict` - per-method acceptance rates, speedups
   - Exposes `/api/v1/speculative/metrics` endpoint

5. **D:/distributed-llm/tests/test_speculative_profiler.py**
6. **D:/distributed-llm/tests/test_workload_classifier.py**
7. **D:/distributed-llm/tests/test_speculative_adaptor.py**

### Files to Modify (Existing)

1. **D:/distributed-llm/src/distllm/core/speculative_decoder.py**
   - Replace `get_active_method()` with performance-driven selection:
     `def get_active_method(self, draft_model=None, hidden_states=None, workload_type: str = "unknown") -> str:`
   - Add `_method_profiles: dict[str, MethodProfile]` tracking
   - Wire `SpeculativeProfiler.record_acceptance()` into `_record_acceptance()`
   - Add `select_method_by_workload(workload_type) -> str` - uses profiler data

2. **D:/distributed-llm/src/distllm/core/speculative_decoder.py** (continued)
   - Wire `generate_tree_drafts()` into `generate_draft_tokens()` for `tree_draft` method
   - Add `_generate_tree_drafts()` private method that calls `generate_tree_drafts()`
   - Update `verify_and_accept()` to handle tree-structured drafts via `verify_and_accept_tree()`

3. **D:/distributed-llm/src/distllm/core/self_optimizing_engine.py**
   - Add `speculative_decoding_k` as tunable parameter (already exists per the gap analysis)
   - Wire callback: `on_speculative_optimization(k) -> speculative_decoder.num_assistant_tokens = k`
   - Add `profile_speculative_methods()` - runs A/B comparison of methods

4. **D:/distributed-llm/src/distllm/core/coordinator.py**
   - Instantiate `SpeculativeProfiler`, `WorkloadClassifier`, `SpeculativeAdaptor`
   - On request: classify workload -> select method -> set `speculative_decoder.method`
   - Add `SpeculativeDashboard` to metrics endpoint

### Key Design Decisions

- **Method selection**: Profile-driven with workload-aware priors. Start with workload heuristic
  (code -> ngram, instruction -> eagle/medusa, diverse -> draft_model), then override with
  acceptance rate profiles after warmup period (100 requests).
- **Workload classification**: Heuristic-based first (no ML model needed). Ngram entropy
  distinguishes repetitive vs diverse; keyword matching detects code (def, class, import).
  Can upgrade to lightweight classifier later.
- **Adaptive num_assistant_tokens**: Simple PID controller. Target acceptance rate = 0.6.
  Increase k if acceptance > 0.7, decrease if < 0.5, disable if < 0.15.

### Implementation Order

1. SpeculativeProfiler (data collection foundation)
2. WorkloadClassifier (enables workload-aware selection)
3. Modify SpeculativeDecoder.get_active_method() (core logic)
4. Wire tree decoding into generation loop
5. SpeculativeAdaptor (dynamic token adjustment)
6. SpeculativeDashboard (observability)
7. SelfOptimizingEngine integration
8. Coordinator integration

### Dependencies on Other Features
- Benefits from Feature 2 (precision affects speculative speedup measurements)
- Independent of Feature 1

---

## Feature 4: Cost-Optimized Cloud Inference

### Goal
True spot instance orchestration with auto-scaling and preemption handling.

### Files to Create (New)

1. **D:/distributed-llm/src/distllm/cloud/spot_provider.py**
   - `SpotProvider` (abstract base class)
   - `AWSSpotProvider` - uses AWS Spot Price API, EC2 metadata
   - `AzureSpotProvider` - Azure Spot VMs, Azure ML compute
   - `GCPSpotProvider` - GCP preemptible VMs
   - `LambdaSpotProvider` - Lambda Labs (no spot, but preemption detection)
   - `get_spot_price_history(instance_type, region, hours=24) -> list[SpotPrice]`
   - `request_instance(instance_type, max_price) -> str` (instance_id)
   - `terminate_instance(instance_id) -> bool`

2. **D:/distributed-llm/src/distllm/cloud/spot_price_tracker.py**
   - `SpotPriceTracker` - polls spot price APIs, maintains price history
   - `SpotPrice` dataclass (provider, instance_type, region, price, timestamp)
   - `poll_prices(interval_s=300)` - background polling loop
   - `get_cheapest_compatible(required_vram, required_compute) -> SpotPrice`
   - `predict_preemption_risk(instance_type, current_price) -> float` (0-1)

3. **D:/distributed-llm/src/distllm/cloud/workload_migrator.py**
   - `WorkloadMigrator` - migrates workloads on spot interruption
   - `migrate_node_workload(src_node_id, dst_node_id) -> bool`
   - Saves KV cache via `CachePersistenceManager` then restores on new node
   - Uses `CacheWarmer.warm_from_persistence()` on target
   - Integrates with `BatchScheduler.restore_preempted()`

4. **D:/distributed-llm/src/distllm/cloud/budget_alerter.py**
   - `BudgetAlerter` - monitors spend, sends alerts on threshold breach
   - `AlertChannel` enum: LOG, WEBHOOK, EMAIL, SLACK
   - `check_budget(current_cost, budget_limit) -> Alert | None`
   - `send_alert(alert: Alert) -> bool` - dispatches to configured channel

5. **D:/distributed-llm/src/distllm/cloud/auto_provisioner.py**
   - `AutoProvisioner` - integrates scheduler with cloud providers
   - `scale_up(node_count, constraints) -> list[str]` (instance_ids)
   - `scale_down(instance_ids) -> bool`
   - Uses `SpotPriceTracker.get_cheapest_compatible()` for placement
   - Integrates with Karpenter manifests for K8s deployments

6. **D:/distributed-llm/tests/test_spot_provider.py**
7. **D:/distributed-llm/tests/test_spot_price_tracker.py**
8. **D:/distributed-llm/tests/test_workload_migrator.py**

### Files to Modify (Existing)

1. **D:/distributed-llm/src/distllm/core/cost_tracker.py**
   - Add `CostTracker.add_cloud_provider(provider: SpotProvider)` - multi-cloud support
   - Add `track_spot_price(provider, instance_type, price)` method
   - Extend `get_current_cost()` to include multi-cloud aggregated cost

2. **D:/distributed-llm/src/distllm/core/spot_handler.py**
   - Extend beyond AWS EC2 metadata
   - Add `AzureSpotHandler` - Azure Instance Metadata Service (IMDS)
   - Add `GCPSpotHandler` - GCP metadata server preemption detection
   - Add `LambdaSpotHandler` - Lambda Labs SIGTERM detection
   - Refactor `SpotHandler` to delegate to provider-specific handlers

3. **D:/distributed-llm/src/distllm/core/batch_scheduler.py**
   - Wire `WorkloadMigrator` into `preempt_lowest()` callback
   - On preemption: save state -> migrate -> call `restore_preempted()` on new node
   - Add `schedule_with_cost_awareness()` - considers spot price in batch ordering

4. **D:/distributed-llm/src/distllm/core/coordinator.py**
   - Instantiate `SpotPriceTracker`, `WorkloadMigrator`, `BudgetAlerter`, `AutoProvisioner`
   - Wire `SpotHandler` to handle multi-cloud preemption signals
   - Add `_cost_optimization_loop()` - background task for price polling + scaling

5. **D:/distributed-llm/deploy/k8s/karpenter-spot.yaml**
   - Add node class definitions for Azure, GCP, Lambda
   - Add provisioner for each cloud provider with spot preferences

### Key Design Decisions

- **Multi-cloud abstraction**: `SpotProvider` interface with provider-specific implementations.
  This avoids cloud-specific logic in the coordinator. Each provider handles its own
  price API, instance lifecycle, and preemption detection.
- **Spot price polling**: 5-minute interval (matches AWS spot price update frequency).
  Cached results with TTL to avoid API rate limits.
- **Workload migration**: Save KV cache to persistent storage (Redis/disk), restore on new node.
  This is stateful but necessary for true preemption handling. Alternative: recompute from
  scratch (cheaper for small contexts, expensive for large).
- **Budget alerting**: Multi-channel (log, webhook, Slack). Threshold-based with hysteresis
  (alert at 80%, 90%, 100% of budget).

### Implementation Order

1. SpotProvider base class + AWS implementation (most common use case)
2. SpotPriceTracker (enables cost-aware placement)
3. Extend SpotHandler for multi-cloud preemption
4. WorkloadMigrator (critical for preemption handling)
5. BudgetAlerter (observability)
6. AutoProvisioner (auto-scaling)
7. Coordinator integration + Karpenter manifest updates

### Dependencies on Other Features
- Benefits from Feature 1 (federation enables cross-cloud workload migration)
- Independent otherwise

---

## Feature 5: Plugin Marketplace

### Goal
Proper extension framework with plugin registry, metadata, lifecycle, sandboxing.

### Files to Create (New)

1. **D:/distributed-llm/src/distllm/plugins/metadata.py**
   - `PluginMetadata` dataclass with fields: name, version, description, author, license,
     dependencies, min_host_version, max_host_version, categories, entry_point, settings_schema
   - `PluginManifest` - loads metadata from `plugin.json` or `pyproject.toml`
   - `validate_metadata(meta: PluginMetadata) -> list[str]` - returns validation errors

2. **D:/distributed-llm/src/distllm/plugins/installer.py**
   - `PluginInstaller` - pip install, version pinning, dependency resolution
   - `install(plugin_name, version=None) -> PluginMetadata`
   - `uninstall(plugin_name) -> bool`
   - `resolve_dependencies(metadata: PluginMetadata) -> list[str]` - returns install commands
   - `list_available() -> list[PluginMetadata]` - queries plugin registry

3. **D:/distributed-llm/src/distllm/plugins/sandbox.py**
   - `PluginSandbox` - restricted context, async execution, resource limits
   - `SandboxContext` - restricted plugin context (no file system access, limited memory)
   - `run_plugin_sandboxed(plugin, context, timeout_s=30) -> Any`
   - Uses `asyncio.wait_for()` for timeout, resource monitoring via `tracemalloc`

4. **D:/distributed-llm/src/distllm/plugins/compatibility.py**
   - `CompatibilityChecker` - host version matrix, dependency validation
   - `check_compatibility(metadata: PluginMetadata, host_version: str) -> CompatibilityResult`
   - `CompatibilityResult` dataclass (compatible: bool, warnings: list[str], errors: list[str])
   - Checks: host version range, Python version, required packages, GPU availability

5. **D:/distributed-llm/src/distllm/plugins/telemetry.py**
   - `PluginTelemetry` - usage tracking, error rates per plugin
   - `record_usage(plugin_name, hook, duration_ms, success: bool)`
   - `get_plugin_stats(plugin_name) -> PluginStats`
   - `get_error_rates() -> dict[str, float]` - plugin_name to error_rate
   - Exposes `/api/v1/plugins/telemetry` endpoint

6. **D:/distributed-llm/src/distllm/plugins/config_schema.py**
   - `PluginConfigValidator` - JSON Schema validation for plugin settings
   - `validate_config(plugin_name, config: dict) -> list[str]` - returns validation errors
   - `get_default_config(plugin_name) -> dict` - from settings_schema

7. **D:/distributed-llm/tests/test_plugin_metadata.py**
8. **D:/distributed-llm/tests/test_plugin_installer.py**
9. **D:/distributed-llm/tests/test_plugin_sandbox.py**
10. **D:/distributed-llm/tests/test_plugin_compatibility.py**

### Files to Modify (Existing)

1. **D:/distributed-llm/src/distllm/core/plugin.py**
   - Extend `IPlugin` protocol with: `metadata` (PluginMetadata),
     `initialize(context: PluginContext)`, `validate_config(config: dict) -> list[str]`
   - Add `PluginContext` dataclass (replaces raw dict): hooks, config, logger, coordinator_ref
   - Extend `PluginManager` with: `_metadata_registry`, `install_plugin()`, `get_plugin_metadata()`,
     `check_compatibility()`, `get_plugin_telemetry()`
   - Wire `PluginSandbox` into `emit_hook()` for async execution

2. **D:/distributed-llm/src/distllm/core/coordinator.py**
   - Pass `PluginContext` (not raw dict) to `PluginManager`
   - Add plugin telemetry to metrics endpoint
   - Add `/api/v1/plugins` REST endpoints (install, uninstall, list, metadata)

### Key Design Decisions

- **Metadata schema**: JSON-based `plugin.json` alongside `pyproject.toml` support.
  Uses standard fields (author, license, version) plus DistLLM-specific fields
  (min_host_version, categories, settings_schema).
- **Sandboxing**: Soft sandboxing (restricted context + timeout + resource monitoring).
  Full isolation (separate process) is overkill for plugins that are meant to extend
  the coordinator. Use `asyncio` for async execution with configurable timeout.
- **Compatibility checking**: Semantic versioning with range support (>=0.1.0,<0.3.0).
  Checks happen at install time AND at load time (defensive).
- **Telemetry**: In-memory with optional persistent storage (SQLite). Tracks hook execution
  count, avg duration, error rate. Exposed via REST API for dashboard consumption.

### Implementation Order

1. PluginMetadata + validation (foundation)
2. ConfigSchemaValidator (enables safe plugin configuration)
3. CompatibilityChecker (enables safe plugin installation)
4. PluginInstaller (pip integration)
5. PluginSandbox (safe execution)
6. PluginTelemetry (observability)
7. Extend IPlugin + PluginManager (integration)
8. REST API + coordinator integration

### Dependencies on Other Features
- None (fully independent)
- Other features could be implemented as plugins (observability, monitoring)

---

## Cross-Feature Dependencies

Feature 1 (Federation) is foundational - Features 2 and 4 benefit from it but do not require it.
Feature 5 (Plugins) is fully independent.
Feature 3 (Speculative) is fully independent.

**Recommended delivery order:**
1. **Phase 1**: Feature 5 (Plugins) - lowest risk, highest immediate value
2. **Phase 2**: Feature 3 (Speculative) - performance win, isolated changes
3. **Phase 3**: Feature 1 (Federation) - foundational for distributed features
4. **Phase 4**: Feature 2 (Precision) - benefits from federation for routing
5. **Phase 5**: Feature 4 (Cost) - benefits from federation for cross-cloud migration

---

## Testing Strategy

### Unit Tests
- Each new file gets a corresponding `test_*.py` in `D:/distributed-llm/tests/`
- Mock gRPC clients, cloud provider APIs, and file system operations
- Use `pytest` fixtures for common setup (federation, precision registry, etc.)

### Integration Tests
- `D:/distributed-llm/tests/integration/test_federation_e2e.py` - 2-node cross-cluster
- `D:/distributed-llm/tests/integration/test_precision_routing.py` - heterogeneous nodes
- `D:/distributed-llm/tests/integration/test_speculative_selection.py` - workload classification
- `D:/distributed-llm/tests/integration/test_spot_lifecycle.py` - mock spot interruption

### E2E Tests
- Use `D:/distributed-llm/benchmarks/run.py` as starting point
- Add federation benchmark: cross-cluster latency + throughput
- Add precision benchmark: mixed-precision inference speed/quality
- Add speculative benchmark: method comparison on standard workloads

---

## Risks and Mitigations

- **gRPC proto changes break backward compatibility** (High): Add new messages, do not modify
  existing ones. Use `optional` fields.
- **FP8 not available on non-Hopper GPUs** (Medium): Graceful fallback to FP16 (already
  implemented in `FP8Engine`).
- **Spot price API rate limits** (Medium): Cache results with 5-min TTL, exponential backoff on 429.
- **Plugin sandbox escape (security)** (High): Use `restrictedpython` for code execution,
  validate all inputs.
- **Cross-cluster latency makes inference unusable** (High): `GeoRouter` already has fallback
  to local cluster when latency exceeds threshold.
- **Speculative method switching causes generation artifacts** (Medium): Warmup period (100
  requests) before switching methods, gradual transition.

---

## Success Criteria

### Feature 1: Federated Inference Fabric
- Cross-cluster ping loop running with configurable interval
- `GeoRouter.select_target_cluster()` called on every cross-cluster request
- Cross-cluster request forwarding works (serialize, send, stream response)
- Federation peer discovery registers remote clusters automatically
- Load reporting updates `GeoRouter` decisions
- KV cache migration between clusters completes successfully

### Feature 2: Heterogeneous Precision Serving
- Per-node precision assigned on registration based on GPU type
- Precision conversion at tensor parallelism boundaries produces correct output
- Quality SLA enforcement prevents INT4 for high-quality requests
- Mixed-precision inference produces same output as uniform precision (within tolerance)
- `SelfOptimizingEngine` callback updates precision on nodes

### Feature 3: Auto-Speculative Selection
- Workload classifier achieves >70% accuracy on code/repetitive/diverse text
- Method selection switches automatically based on acceptance rate profiles
- Tree decoding wired into generation loop and produces correct output
- `num_assistant_tokens` adapts dynamically based on acceptance rate
- Speculative dashboard shows per-method comparison metrics

### Feature 4: Cost-Optimized Cloud Inference
- Multi-cloud spot providers (AWS, Azure, GCP) all functional
- Spot price polling runs every 5 minutes without rate limit errors
- Workload migration on preemption completes within 60 seconds
- Budget alerts sent at 80%/90%/100% thresholds
- Auto-provisioning creates instances matching scheduler requirements

### Feature 5: Plugin Marketplace
- Plugin metadata schema validated at install time
- Plugin installation via pip with dependency resolution
- Plugin sandbox prevents file system access and enforces timeout
- Compatibility check blocks incompatible plugins at load time
- Telemetry tracks per-plugin usage and error rates
- Plugin config schema validation rejects invalid settings
