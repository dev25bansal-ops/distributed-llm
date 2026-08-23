---
tags:
  - core
  - engine
aliases:
  - Core Engine
---
# Core Engine — `src/distllm/core/`

**296 .py files · ~82K LOC** (incl. sub-packages `advanced_scheduling`, `coordinator*`, `dp_inference`, `evaluation`, `plugins`, `scheduler`, `structured_output`, `vectorstore`).

> The **heart of the runtime**. One `Coordinator` object orchestrates every subsystem: batch scheduling, speculative decoding, KV-cache hierarchy, model/cost/cross-cloud routing, HA/leader-election, plugins, multimodal/edge inference, usage-metering and compliance. Depends on `distllm.config`, `distllm.backends`, `distllm.dist`, and the `core.*` sub-packages.
>
> **Tests:** `python -m pytest tests/core/ -v` (218 files in `tests/core`). Comprehensive audit: [[Core Comprehensive Audit 2026-08-05]] → [[Core Audit 01 Strategic & Opportunities]], [[Core Audit 02 Issues & Required Fixes]], [[Core Audit 03 Enhancements & Modifications]], [[Core Audit 04 Advanced Features]], [[Core Audit 05 New Additions]], [[Core Audit 06 Verification & Testing]], [[Core Audit 07 Dead Code & Consolidation]].

## Orchestration & coordinator core

| file | LOC | purpose |
|------|-----|---------|
| `__init__.py` | 111 | Lazy-import `__getattr__` facade exposing ~24+ major symbols |
| `coordinator.py` | 1,310 | `Coordinator` — the master orchestration object (node/cluster/serve/load) |
| `coordinator_cli.py` | 130 | `main()` CLI entry for the Coordinator |
| `coordinator_config.py` | 203 | Pydantic `CoordinatorConfig` |
| `coordinator_config_wiring.py` | 316 | `CoordinatorConfigurator` wires optional subsystems into the Coordinator |
| `coordinator_subsystem.py` | 620 | `SubsystemManager` — subsystem lifecycle (extracted from Coordinator) |
| `subsystem_registry.py` | 171 | `SubsystemRegistry` — lifecycle manager |
| `coordinator_election.py` | 210 | `CoordinatorElection` — leader election + HA status |
| `coordinator_state.py` | 246 | `CoordinatorState`/`CoordinatorRole` state machine (INIT→FOLLOWER⇄LEADER) |
| `coordinator_failover.py` | 197 | `CoordinatorFailoverHandler` — worker reconnect on coordinator loss |
| `coordinator_health.py` | 135 | `HealthChecker` — sync/async health dispatch |
| `coordinator_lifecycle.py` | 259 | `RequestTracker` + `ServerLifecycle` (start/stop/graceful) |
| `coordinator_metrics.py` | 75 | `MetricsManager` — counters/gauges + Prometheus export |
| `coordinator_request.py` | 246 | `RequestHandler` — generation request methods (extracted) |
| `request_pipeline.py` | 804 | `RequestPipeline` — generation + batching + speculative flow (extracted) |
| `resource_manager.py` | 646 | `ResourceManager` — node lifecycle, health, circuit breaker, connections |
| `inference_engine.py` | 1,048 | `InferenceEngine` + `GenerationStrategy` family (local/speculative/distributed) |
| `token_generator.py` | 368 | `TokenGenerator` — sampling, constraints, logprobs, penalties, generation loop |
| `protocols.py` | 213 | `Protocol` ABCs (`INodeClient`, `ICacheBackend`, `IModelPartitioner`…) |
| `types.py` | 17 | `ErrorCode` enum |
| `event_bus.py` | 516 | `EventBus` — pub/sub for marketplace/job-lifetime events |
| `request_tracker.py` | 171 | async request result tracking |
| `request_replay.py` | 253 | `RequestReplayBuffer` + `DeterministicMode` — debug replay |
| `di.py` | 153 | `Container` — lightweight DI container |
| `debug.py` | 84 | `DebugConfig` — tensor-shape logging / forward-dump |
| `metrics_collector.py` | 32 | aggregates metrics from all subsystems |
| `load_balancer.py` | 286 | across-coordinator round-robin / least-conn / latency-weighted |

## Batch scheduler & batching

| file | LOC | purpose |
|------|-----|---------|
| `batch_scheduler.py` | 1,321 | `BatchScheduler` — continuous batching for pipeline-parallel inference |
| `batch_builder.py` | 480 | batch-construction extracted from BatchScheduler |
| `step_processor.py` | 115 | step processing (extracted) |
| `micro_batch_scheduler.py` | 170 | micro-batching decode steps to reduce pipeline bubbles |
| `priority_heap.py` | 62 | `promote_request`, `rebuild_pending_index` |
| `stats_collector.py` | 64 | batch-scheduler stats |
| `starvation_monitor.py` | 70 | `check_starvation`, `aging_boost` |
| `adaptive_batching.py` | 234 | `AdaptiveBatchingEngine` — dynamic batch size vs latency SLOs |
| `schedule_simulator.py` | 288 | offline scheduler trace replay (no inference) |
| `schedule_viz.py` | 233 | ASCII + HTML schedule timelines |
| `request_latency.py` | 205 | TTFT/TPOT/SLA per-request tracking |
| `heterogeneous_scheduler.py` | 663 | + disaggregated prefill/decode across GPU types |
| `preemptible_scheduler.py` | 343 | priority tiers + spot pricing |
| `straggler_aware_scheduler.py` | 255 | gradient-based straggler budget recovery |
| `straggler_alerts.py` | 174 | webhook/email/Slack alerts |

## Speculative decoding

| file | LOC | purpose |
|------|-----|---------|
| `speculative_decoder.py` | 1,046 | `SpecDecoderBase` + `SpeculativeDecoder`/`Self`/`MultiDraft`/`TreeDraft` |
| `distributed_speculative.py` | 1,526 | `RemoteDraftModel` on a separate device via HTTP |
| `async_pipelined_speculative.py` | 477 | `PipelinedSpeculativeDecoder` — async pipelined draft/verify/accept |
| `tree_speculative_decoder.py` | 405 | `TreeSpeculativeDecoder` — tree of parallel branches |
| `draft_tree.py` | 232 | multi-candidate tree spec |
| `multi_draft_verifier.py` | 342 | `MultiDraftVerifier`/`TreeMultiDraftVerifier` |
| `draft_model_router.py` | 435 | SLA-based draft-fleet selection |
| `draft_quality_scorer.py` | 157 | acceptance-rate auto-selection |
| `dynamic_speculation.py` | 131 | adaptive candidate count |
| `speculative_adaptor.py` | 108 | adaptive draft count by acceptance |
| `speculative_profiler.py` | 132 | acceptance stats per method/workload |
| `speculative_dashboard.py` | 134 | comparison reporting |
| `spec_verify.py` | 225 | `verify_chain`/`accept_token`/`prefix_len` |
| `spec_equivalence.py` | 140 | proves spec is bit-identical to target |
| `compressed_speculative.py` | 299 | spec-decoded cache compression |
| `mtp_head.py` | 438 | `MTPHead`/`MTPDecoder` — multi-token prediction head |
| `grammar_constrained_draft.py` | 91 | grammar-valid draft tokens |
| `workload_classifier.py` | 137 | `WorkloadType` + `classify` → pick a speculative method |

## KV-cache & memory hierarchy

| file | LOC | purpose |
|------|-----|---------|
| `cache_manager.py` | 701 | `CacheManager` (+ RollingHash) — prefix cache, KV lifecycle, chunked prefill |
| `kv_cache.py` | 1,254 | `PagedKVCacheBackend` + `KVCacheManager` + `AdaptiveQuantizer` + serialization |
| `kv_cache_adaptive_quantizer.py` | 168 | per-layer KV quant (extracted) |
| `kv_cache_compressor.py` | 287 | `BlockCompressor` — in-place block-wise KV compression |
| `kv_cache_manager.py` | 142 | multi-request KVCacheManager (extracted) |
| `kv_cache_marketplace.py` | 319 | nodes advertise/trade cached KV states |
| `kv_cache_metrics.py` | 205 | KV block metrics (Prom/OTel) |
| `kv_cache_migration.py` | 137 | offload/load to CPU-GPU (extracted) |
| `kv_cache_paged.py` | 78 | `PagedKVCacheBackend` (extracted) |
| `kv_cache_replication.py` | 107 | replica-aware cache wrapper |
| `kv_cache_serialization.py` | 182 | tensor serialize/deserialize + disk save/load |
| `adaptive_cache_compressor.py` | 168 | per-tier compression (GPU FP8 / disk INT4) |
| `adaptive_compression.py` | 427 | idle-period cache compression |
| `adaptive_compression_hierarchy.py` | 402 | quality-based compression level per request |
| `cache_coherence.py` | 139 | vector-clock P2P cache coherence |
| `cache_eviction.py` | 181 | TTL / semantic grouping policies |
| `cache_index.py` | 119 | re-exports `CacheIndex` + `CacheIndexEntry` |
| `cache_migration.py` | 134 | cross-cluster KV migration |
| `cache_persistence.py` | 200 | disk KV save/load, TTL |
| `cache_snapshot.py` | 178 | point-in-time export/import |
| `cache_query_log.py` | 195 | JSONL audit of cache ops |
| `cache_template_warmer.py` | 112 | speculative pre-warm with prompt templates |
| `cache_doctor.py` | 189 | self-healing cache diagnostics |
| `cache_bench.py` | 239 | hit-rate/latency/throughput benchmark |
| `cache_aware_router.py` | 116 | route to best cache affinity |
| `prefix_cache.py` | 17 | **shim** → `distllm.dist.prefix_cache` |
| `predictive_cache.py` | 29 | **shim** → `distllm.dist.predictive_cache` |
| `predictive_cache_warming.py` | 323 | LRU/markov proactive KV push |
| `radix_tree_cache.py` | 317 | `RadixTreeCache` O(k) trie prefix cache |
| `hybrid_cache.py` | 245 | paged+contiguous hybrid allocation |
| `hierarchical_digest.py` | 168 | Bloom→Merkle→exact cache sync |
| `gossip_cache_bridge.py` | 202 | gossip+ KV distributed replication |
| `block_affinity_tracker.py` | 162 | block↔request affinity for CoW |
| `block_eviction_policy.py` | 363 | LRU/LFU/FIFO/TwoQ/ARC policies |
| `semantic_cache.py` | 360 | embedding-similarity response reuse |
| `gaia_cache.py` | 1,330 | `GaiaCache` — hash ring + KV marketplace + RL eviction + Bloom |
| `redis_prompt_cache.py` | 296 | Redis prompt KV store |
| `prompt_caching_service.py` | 228 | higher-level Redis prompt caching |
| `topology_aware_tiering.py` | 344 | 5-tier interconnect-aware KV tiering (NVLink/CXL) |
| `memory_defragmenter.py` | 641 | compacts fragmented PagedAttention blocks |
| `dynamic_memory_budget.py` | 179 | adaptive KV memory budget |
| `kv_backup.py` | 217 | planned-maintenance KV snapshot/restore |
| `persistent_store.py` | 485 | SQLite jobs/batch/audit/log/session durability |

## Routing & model selection

| file | LOC | purpose |
|------|-----|---------|
| `model_router.py` | 930 | rule-based query → model routing |
| `model_selector.py` | 245 | requirements → model recommendation |
| `smart_model_router.py` | 309 | task-complexity → optimal model |
| `unified_router.py` | 425 | cloud + P2P GPU in one |
| `unified_sla_router.py` | 242 | single objective knob (cost/carbon/latency) |
| `learning_router.py` | 645 | online RL contextual-bandit routing |
| `agentic_router.py` | 746 | `AgenticRouter`/`RouterJudge` — LLM-as-judge selection |
| `preference_learning.py` | 606 | DPO/RLHF self-improvement |
| `routing_extensions.py` | 604 | semantic router, LRU model cache, spec prewarmer, metrics |
| `route_audit.py` | 175 | JSONL of every routing decision |
| `request_fingerprinting.py` | 224 | content-based dedup-hash |
| `topology_aware_lb.py` | 324 | topology-aware load balancer |
| `cross_cloud_router.py` | 918 | carbon+latency cross-cloud |
| `cross_model_prefix_sharing.py` | 230 | share KV across model variants |

## Cost / pricing / billing

| file | lines | purpose |
|------|-------|---------|
| `money.py` | 53 | Decimal money helper |
| `cost_tracker.py` | 569 | per-request cost + budget |
| `cost_optimizer.py` | 295 | spot + ROI, ties arbitrage/tracker/providers |
| `cost_dashboard.py` | 209 | per-user/model budget alerts |
| `cost_comparison.py` | 250 | cross-provider cost table generator |
| `tenant_cost_attribution.py` | 314 | per-tenant breakdown |
| `tenant_billing.py` | 446 | multi-tenant quotas/billing |
| `metering.py` | 449 | `MeteringStore`/`Middleware`/`BillingExporter` |
| `usage_meter.py` | 899 | usage tracking + quotas + billing records |
| `pricing_providers.py` | 576 | AWS/GCP/Azure live pricing + stale fallback |
| `arbitrage_engine.py` | 606 | spot monitoring + load migration |
| `bargaining_engine.py` | 450 | DQN automated spot bidding |
| `spot_forecasting.py` | 463 | Holt-Winters price forecast |
| `spot_failover.py` | 292 | preemption auto-failover |
| `carbon_migration.py` | 338 | migrate to cleaner regions |
| `streaming_cost.py` | 272 | real-time streaming cost accumulation |

## HA · cluster · resilience

| file | lines | purpose |
|------|-------|---------|
| `node_recovery.py` | 27 | **shim** → `distllm.dist.recovery` |
| `cluster_manager.py` | 194 | node lifecycle |
| `cluster_registry.py` | 227 | opt-in public cluster directory |
| `cluster_state_store.py` | 428 | Redis/File shared HA state + election |
| `auto_discovery.py` | 351 | mDNS zero-config node discovery |
| `device_registry.py` | 289 | cross-platform device capability detection |
| `hardware_plugin_registry.py` | 126 | third-party accel plugins |
| `gpu_profiler.py` | 59 | hardware capability detection |
| `gpu_resource_manager.py` | 338 | VRAM tracking, OOM prevention |
| `gpu_power_manager.py` | 173 | util-based power capping |
| `health_manager.py` | 320 | health probing, recovery, straggler detection |
| `autonomous_healer.py` | 466 | predictive GPU failure + auto-heal |
| `async_connection_pool.py` | 109 | asyncio TCP pool |
| `connection_pool.py` | 330 | production TCP pool (leak/timeout fixes) |
| `state_replication.py` | 260 | etcd/Redis/file HA state |
| `replication_controller.py` | 128 | HA state-replication collaborator |
| `split_brain.py` | 233 | federation partition detection |
| `ha_coordinator.py` | 364 | `HaCoordinator` + `RayFaultTolerance` — Raft-like election |
| `rebalancer.py` | 25 | **shim** → `distllm.dist.rebalancer` |
| `backup_manager.py` / `certificate_manager.py` | 298 / 413 | config backup & DR · ACME/LetsEncrypt |
| `cert_rotation.py` | 290 | auto-renewal of certs + API keys |
| `secret_manager.py` | 255 | env/vault/AWS secrets |
| `api_key_store.py` | 380 | multi-role API-key auth |
| `provisioning.py` | 303 | IaC generation (Terraform/CloudFormation) |

## Autoscaler · placement · quantization · training

| file | lines | purpose |
|------|-------|---------|
| `aria_autoscaler.py` | 1,309 | `PredictiveScaler`/`CarbonAwareScaler` |
| `intelligent_autoscaler.py` | 219 | predictive/cost-aware scaling |
| `hibernation_manager.py` | 250 | scale-to-zero idle nodes |
| `dynamic_sharder.py` | 349 | live resharding, zero downtime |
| `placement.py` | 442 | topology/carbon-aware placement + migration |
| `neural_partition_optimizer.py` | 1,398 | neural cost model + Bayesian opt |
| `auto_partitioner.py` | 352 | layer→device plans |
| `activation_profiler.py` | 172 | layer split-point profiling |
| `autoq.py` | 1,545 | `AutoQ` per-layer adaptive quantization |
| `quantization_selector.py` | 210 | auto FP16/INT8/INT4 |
| `calibration.py` | 223 | auto hardware threshold calibration |
| `model_sizing.py` | 334 | params/layers/VRAM source of truth |
| `pipeline_composer.py` | 211 | chain embed→rerank→generate |
| `pipeline_executor.py` | 500 | multi-step SLO pipelines |
| `pipeline_orchestrator.py` | 17 | **shim** → `distllm.dist.pipeline` |
| `pipeline_overlap.py` | 177 | `OneFOneBScheduler` forward/backward overlap |

## Multimodal · serving · federated training

| file | lines | purpose |
|------|-------|---------|
| `multi_model_serving.py` | 613 | GPU time-slicing + hot-swap |
| `cortex_multimodel.py` | 939 | prefix-sharing + expert parallelism + model pool |
| `multimodal_engine.py` | 194 | vision/audio/docs across nodes |
| `voyager_multimodal.py` | 1,326 | true multimodal parallel-encoder pipeline |
| `media_pipeline.py` | 308 | WebRTC audio-video → LLM |
| `hydra_diffusion.py` | 230 | distributed image/video generation |
| `shared_layer_pool.py` | 223 | share common layers across models |
| `moe_orchestrator.py` | 205 | sparse-expert routing |
| `aether_federated.py` | 900 | federated LoRA + gradient aggregator |
| `federated_finetuner.py` | 289 | P2P distributed LoRA training |
| `federated_incentives.py` | 267 | credit/reputation ledger |
| `distributed_distillation.py` | 334 | cluster-as-teacher distillation |
| `synth_data_generator.py` | 122 | dataset generation via cluster |
| `faas_7b.py` | 201 | serverless Lambda/CF worker |
| `vllm_node.py` | 174 | `VLLMWorkerNode` |
| `webgpu_manager.py` | 378 | browser GPU contribution |
| `wisp_wasm.py` | 749 | WebAssembly edge inference |

## Compliance · multi-tenant · QoS

| file | lines | purpose |
|------|-------|---------|
| `aegis_compliance.py` | 890 | SOC2/HIPAA compliance + watermarking |
| `compliance_evidence.py` | 445 | automated evidence collector |
| `request_auditor.py` | 215 | compliance logging + PII detection |
| `differential_privacy.py` | 142 | noise for cross-node KV sharing |
| `privacy_budget.py` | 146 | per-tenant DP budget |
| `dp_inference.py` | 870 | `DifferentialPrivacyInference` + RDP + DP-SGD |
| `sentinel_qos.py` | 674 | `Sentinel` token-bucket/fair-queue QoS + admission |
| `leaky_bucket_limiter.py` | 164 | smooth request throttling |
| `graceful_degradation.py` | 226 | partial responses when overloaded |
| `feature_flags.py` | 205 | JSON+env rollout flags |

## Structured output / constrained decoding

| file | lines | purpose |
|------|-------|---------|
| `constrained_decoder.py` | 545 | FSM/JSON-schema/regex-token constraints |
| `grammar_constrained.py` | 474 | outlines-backed formal grammar |
| `grammar_decoder.py` | 206 | `GBNFParser`/`GBNFFSM` |
| `cuda_graph.py` | 132 | graph-captured decode steps |

## Correctness & research

| file | lines | purpose |
|------|-------|---------|
| `correctness_harness.py` | 349 | differential correctness harness |
| `correctness_cert.py` | 106 | HMAC correctness cert |
| `shadow_eval_runner.py` | 160 | LLM-judge shadow regression eval |
| `evaluation_harness.py` | 1,302 | HEIM/MMLU/GSM8K/HumanEval/MT-Bench |

## Webhooks · notifications · telemetry

| file | lines | purpose |
|------|-------|---------|
| `webhook_manager.py` | 296 | cluster-event HTTP webhooks |
| `webhook_formatters.py` | 189 | Slack/Discord/PagerDuty formatters |
| `notification_manager.py` | 272 | Slack/Discord/email/HTTP |
| `performance_alerts.py` | 165 | throughput/latency degradation |
| `performance_baseline.py` | 214 | post-deploy regression detection |
| `predictive_failure.py` | 135 | GPU ECC/throttle detection |
| `telemetry.py` | 195 | opt-in anonymous analytics |
| `synapse_debugger.py` | 189 | distributed debugger |
| `monitor.py` | 119 | GPU/CPU/mem/request metrics |

## Sub-packages (in `core/`)

- **`core/advanced_scheduling/`** — pluggable scheduling policies: `policy.py` (Protocol + Default/Sarathi), `heterogeneous.py`, `cost_aware.py`, `energy.py`, `disaggregated.py`, `predictive.py`, `tiered_store.py` (GPU→CPU→NVMe memory pool), `token_bank.py` (`TokenBank` billing credits), `federated.py`, `preemption.py`, `wan.py`, `multi_objective.py` (Pareto cost/energy/latency/tput).
- **`core/coordinator*.py`** — the decomposed Coordinator: `coordinator.py`, `coordinator_config.py`, `_config_wiring.py`, `_subsystem.py`, `_request.py`, `_state.py`, `_election.py`, `_failover.py`, `_lifecycle.py`, `_health.py`, `_metrics.py`, `_cli.py`.
- **`core/dp_inference/`** — `DifferentialPrivacyInference` (eps,delta) wrapping `build_engine`; RDP accounting, config, Gaussian/Gumbel mechanisms.
- **`core/structured_output/`** — `JSONSchemaConstraint` (token-level char state machine), `StructuredOutputEngine`, validator/repair, streaming partial-JSON.
- **`core/vectorstore/`** — provider-agnostic RAG: `VectorDBInterface`, `VectorDBFactory`, `RAGPipeline`, `providers/{milvus,pinecone,qdrant,weaviate}` + legacy `chroma/pgvector/qdrant_store`.
- **`core/evaluation/`** — LLM eval harness: `DatasetLoader` (MMLU/GSM8K/HumanEval/MT-Bench/Arena), `Scorer`, `EvalRunner`, SQLite `EvalDB`.
- **`core/plugins/`** — re-export shim → `core/plugin_sandbox` (signed manifest + capability-scoped sandbox) — see also [[07 Integrations]][[11 Platform Services]].

## Entry points & dependencies

- **Entry points:** `core/__init__.py` (lazy facade), `coordinator_cli.py` (`distllm-coordinator`), `coordinator.py` (`Coordinator`), `vllm_node.py`, `wisp_wasm.py`, `faas_7b.py`.
- **Consumed by:** [[03 API Server]] (coordinator + plugin system), [[02 Distributed Layer]] (pipeline/federation/recovery), each CLI module.
- **Depends on:** `config`, `backends`, `dist`, `api` (SSE `StreamingGenerator`), and the `core.*` sub-packages.
- **Compatibility shims** re-export from `distllm.dist.*`: `latency_tracker`, `node_recovery`, `pipeline_orchestrator`, `predictive_cache`, `prefix_cache`, `rebalancer`, `straggler_detector`, `vllm_backend`, `llamacpp_backend`.

## Notes / dead code

Most `coordinator_*`, `kv_cache_*`, `request_pipeline`, `resource_manager`, `token_generator`, `steps/`/`queue/` modules are **decomposition refactors** splitting the former monoliths (stated in their own docstrings). `plugins/sandbox.py` and `core/plugins/` are pure re-export shims. Several legacy vectorstore stores + the parallel files `metrics_collector`/`monitor` partially overlap. `backups_provider` and `models_partitioner` smoke surfaces exist for connector IO.

## Tests

`tests/core/` (**~220 files**) covers nearly every module above (per-file `test_<module>.py`), plus `tests/core/advanced_scheduling/`, `tests/core/scheduler/`, `tests/core/structured_output/`, `tests/core/vectorstore/`, and in-depth suites on coordinator lifecycle, scheduler policies, ARPN/DP, and structured output.