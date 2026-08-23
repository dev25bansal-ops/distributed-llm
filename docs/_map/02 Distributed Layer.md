---
tags:
  - dist
  - distributed
  - p2p
aliases:
  - Distributed Layer
---
# Distributed Layer — `src/distllm/dist/`

**193 .py files · ~74K LOC.**

> The **distributed execution plane**: pool GPUs across machines (LAN, WAN, NAT-broken home networks) into one virtual inference cluster. Pipeline/tensor/expert parallelism, PagedAttention KV-cache with distributed prefix sharing, speculative decoding, cross-cluster federation, fault tolerance (byzantine/Raft/chaos), and transport (gRPC, QUIC, WebRTC/ICE, NCCL, zero-copy). The `partition/` sub-package is the **hardware-aware auto-partitioner**.
>
> **Tests:** `python -m pytest tests/dist/ -v` (121 files).

## `dist/` top level

### Parallelism & attention
| file | LOC | purpose |
|------|-----|---------|
| `parallel.py` | 879 | `HybridParallelismEngine` — auto TP/PP/EP by topology |
| `parallel_planner.py` | 483 | `ParallelAutoTuner` (10s profiler) + `HybridParallelPlanner` |
| `parallel_topology.py` | 180 | topology info, TP-degree choice, `HardwareProber` |
| `async_pipeline.py` | 487 | `AsyncPipeline` — PP overlapping compute/comm, 1F1B, CUDA-stream prefetch, async all-reduce |
| `tp_launcher.py` | 262 | tensor-parallel worker + launcher (multi-GPU single-node) |
| `tp_inprocess.py` | 173 | tensor parallelism without Ray (NCCL) |
| `fsdp.py` | 230 | FSDP-style weight sharding, all-gather per layer |
| `attention.py` | 1,399 | `PagedAttentionManager` — block-table KV cache + distributed prefix sharing |
| `attention_block_pool.py` | 535 | extracted `Block`/`BlockTable`/`BlockPool` structures |
| `attention_distributed.py` | 106 | `BlockPrefetchScheduler`/`DistributedBlockFetcher` |
| `block_pool.py` | 587 | physical KV block pool: alloc/evict/swap/prefetch |
| `block_transfer_service.py` | 207 | gRPC streaming KV blocks between nodes |
| `paged_attention_kernel.py` | 229 | fused paged_attention compute kernel (Triton + SDPA fallback) |
| `flash_attention.py` | 195 | FlashAttention drop-in wrapper (2-3x prefill, O(n) mem) |
| `tensor_pool.py` | 93 | pre-allocated reusable tensor buffers |
| `zero_copy.py` | 225 | CUDA IPC + RDMA GPU-direct transfer |

### Work & node execution
| file | LOC | purpose |
|------|-----|---------|
| `worker.py` | 543 | `WorkerNode` — loads a layer slice, serves gRPC forward passes |
| `node_service.py` | 487 | `NodeServer`/`NodeServicer` — ForwardPass/Health/Profile RPCs |
| `node_client.py` | 436 | gRPC client (channel pool, forward, request weights) |
| `node_registrar.py` | 213 | node registration, auto-setup, expert registration |
| `daas_server.py` | 417 | Draft-as-a-Service draft-model server |
| `admin_cli.py` | 229 | `distllm-admin` cluster management CLI |
| `node_pb2.py`/`node_pb2_grpc.py` | 60/286 | **generated** protobuf (do not edit) |

### Topology, federation, geo
| file | LOC | purpose |
|------|-----|---------|
| `topology.py` | 116 | `ClusterInfo`/`FederationManager`/latency monitor |
| `topology_consensus.py` | 1,617 | `RaftNode` — Raft for topology metadata (SQLite + gossip) |
| `topology_dynamic.py` | 227 | live topology graph auto-updating |
| `federation.py` | 932 | `FederationCoordinator` — cross-cluster gateway + peer circuit breaker |
| `edge_federation.py` | 362 | route small models to mobile/browser/edge |
| `edge_cloud.py` | 461 | Edge-to-cloud continuum, capability-aware layer assignment |
| `geo.py` | 299 | `GeoRouter` — geo/latency-aware routing with A/B canary weights |
| `network.py` | 104 | topology dataclass + ring/tree/hierarchical creators |
| `cloud_selector.py` | 218 | cheapest region per GPU requirements (AWS/GCP/Azure) |
| `p2p_model_distributor.py` | 130 | BitTorrent-style chunked model download |
| `model_store.py` | 91 | shared layer cache (single HF download) |

### Recovery, reliability, byzantine
| file | LOC | purpose |
|------|-----|---------|
| `recovery.py` | 667 | `NodeRecoveryManager` — self-healing, in-flight recovery |
| `recovery_drill.py` | 237 | scheduled chaos drills vs SLO |
| `redundant_executor.py` | 1,270 | production redundant execution + gradient compression + NCCL state replication |
| `redundant.py` | 258 | `RedundantExecutor`/`StateReplicationEngine` (older gen) |
| `byzantine.py` | 1,794 | PBFT: `PBFTNode` 3f+1, view-change, quorum, split-brain |
| `straggler.py` | 652 | `StragglerDetector` — threshold/MAD/trend/throughput |
| `rebalancer.py` | 186 | dynamic layer reassignment, straggler migration |
| `reputation.py` | 194 | per-peer reliability scores for trust routing |
| `merkle.py` | 143 | Merkle tree for page-table sync |
| `lifecycle.py` | 373 | dependency-aware topological startup/shutdown |
| `latency.py` | 48 | sliding-window per-node latency |

### Transport & NAT
| file | LOC | purpose |
|------|-----|---------|
| `transport.py` | 393 | `TransportBackend` abstraction (grpc/in-process/p2p) |
| `nccl.py` | 630 | NCCL GPU transfers + Gloo fallback + benchmark |
| `quic_transport.py` | 263 | QUIC/HTTP3 via aioquic |
| `webrtc.py` | 680 | WebRTC/ICE via aiortc, DTLS/SCTP data channel |
| `ice_transport.py` | 2,481 | full ICE (RFC 8445) + TURN (RFC 5766) + STUN + hole-punching |
| `nat.py` | 759 | STUN/TURN/ICE for cross-internet clusters (older gen) |
| `ebpf_transport.py` | 307 | **SCAFFOLD** eBPF observability layer (software-simulated) |
| `prefix_cache.py` | 369 | `PrefixCache`/`DistributedPrefixCache` + bloom filter |
| `prefix_clustering.py` | 254 | semantic prefix clustering (60-80% TTFT cut) |
| `predictive_cache.py` | 351 | predict prefix reuse + pre-warm |
| `redis_cache.py` | 226 | shared cross-node KV (SHA-256, pub/sub invalidation) |
| `cache_digest.py` | 321 | content-based federated routing |
| `cache.py` | 250 | KV prefix cache eviction policies |
| `kv_migration.py` | 197 | cross-cluster KV streaming |
| `streaming_kv_transfer.py` | 130 | chunk KV to bypass 4MB gRPC limit |
| `opentelemetry` | — | see `observability.py` `metrics.py` `tracing.py` `otel.py` |

### Model-serving conveniences & misc
| file | LOC | purpose |
|------|-----|---------|
| `multi_tenant.py` | 278 | per-tenant SLO priority queue + rate limits |
| `quota_enforcer.py` | 216 | pipeline-level tenant quotas + admission |
| `marketplace.py` | 785 | BYOG GPU marketplace + metering |
| `disagg/` | — | *(see sub-packages below)* |
| `quality.py` | 96 | precision-aware quality SLAs |
| `power_cap.py` | 297 | nvidia-smi power capping (30–50% cut) |
| `autoscaler.py` | 166 | provisioner start/stop on queue/SLO/GPU thresholds |
| `provisioning.py` | 286 | cheapest-region cluster provisioning |
| `chunked_prefill.py` | 108 | long-prompt chunked prefill |
| `preemption.py` | 167 | SLA-aware preemption + checkpoint |
| `wide_area.py` | 434 | high-latency pipeline: token accumulation, RTT-batched |
| `wan_speculative.py` | 233 | speculative decoding with WAN token accumulation |
| `execution_planner.py` | 121 | unified exec-plan interface |
| `multimodal.py` | 318 | multi-modal PP (vision encoder stage) |
| `tgi_compat.py` | 194 | TGI-compatible HTTP wrapper |
| `api_docs.py` | 579 | OpenAPI 3.1 spec + ReDoc/Swagger for the dist layer |
| `discovery.py` | 137 | mDNS/Zeroconf auto-discovery |
| `admin_cli.py` | 229 | operator CLI |
| `daas_*` | — | *(see sub-packages*) |
| `config.py` | 58 | `WideAreaConfig` |
| `observability.py` | 220 | wire metrics+exporter+tracer |
| `wandb_integration.py` | 218 | **deprecated shim** → `distllm.integrations.wandb` |

## `dist/partition/` — Hardware-Aware Auto-Partitioner (28 files, ~11.7K LOC)

| file | LOC | purpose |
|------|-----|---------|
| `__init__.py` | 142 | exports GPUProfiler/TopologyProber/CostModel/Optimizer/HardwareAwarePartitioner |
| `partitioner.py` | 324 | `HardwareAwarePartitioner` — profile → solve → save/load plan |
| `optimizer.py` | 550 | DP solver minimizing max per-node latency |
| `pareto_optimizer.py` | 388 | multi-objective DP over Pareto (latency/tput/mem/quant/cost) |
| `cost_model.py` | 467 | per-node latency/throughput/memory cost model |
| `learned_cost.py` | 470 | gradient-boosted-tree model (falls back to analytical) |
| `network_cost_model.py` | 551 | inter-node latency/bandwidth + comm-cost |
| `quant_cost.py` | 222 | quant-aware cost extension |
| `quant_partition.py` | 366 | joint DP over (layer_split, quant_method) |
| `quantization_tuner.py` | 1,186 | Adaptive Precision Optimizer — joint weight/activation/KV quant |
| `quantization_search.py` | 238 | auto mixed-precision plan |
| `quantization_metrics.py` | 296 | per-layer precision profiling |
| `quant_bench.py` | 315 | per-GPU matmul TFLOPS per quant method |
| `quant_calibrate.py` | 198 | online quality calibration |
| `quant_coordinator.py` | 222 | distribute quant plans |
| `quant_cost.py` | 259 | — (cost extension) |
| `quant_report.py` | 301 | cluster-wide report generator |
| `mixed_precision_tuner.py` | 259 | per-layer FP8/INT8/INT4 via KL sensitivity |
| `adaptive.py` | 444 | `AdaptiveRepartitioner` → re-run DP → live migration |
| `cloud_arbitrage.py` | 464 | joint (partition, provider, instance) minimize $/token |
| `validator.py` | 346 | synthetic pipeline stress harness |
| `visualizer.py` | 231 | rich terminal topology/partition viz |
| `benchmark_suite.py` | 430 | reproducible partition-quality benchmarks |
| `persistence.py` | 335 | SQLite partition store with rollback |
| `profiles.py` | 581 | `GPUProfiler`/`GPUProfile`/`LayerWeights` |
| `topology.py` | 189 | `TopologyProbe`/`LinkProfile` |
| `plugins.py` | 224 | custom GPU cost-model plugin API |
| `config.py` | 80 | profile/optimizer/pareto/quant configs |
| `cli.py` | 156 | standalone partitioner CLI |

## Sub-packages (in `dist/`)

- **`dist/pipeline/`** (15) — **core of the dist layer**: `orchestrator.PipelineOrchestrator` (stage pipeline across nodes), `pipeline_reconfig.PipelineReconfigurator` (live topology reconfig + checkpoint), `bandwidth_controller`, `compression_negotiation`, `serialization`, `transport`, `simulator`, `strategy`, `token_accumulator`, `continuous_batching`, `disagg_orchestrator`, `wan_disagg_orchestrator`.
- **`dist/p2p/`** (8) — peer mesh: `gossip.GossipProtocol` (cross-node state replication, vectors, LWW, bloom), `kademlia_dht.KademliaDHT`, `quic_transport`, `transport`, `router.FederationRouter`, `discovery`, `load_balancer`.
- **`dist/scheduling/`** (6) — `deadline_scheduler.DeadlineAwareBatchScheduler` (+ GPU batch packer, preemptive stages), `iteration.IterationScheduler`, `classifier.WorkloadType`, `batcher`, `profiler`.
- **`dist/speculative/`** (6) — `draft_orchestrator.DraftOrchestrator` (Thompson-sampling bandit), `draft_registry`, `draft_cache`, `multi_draft`, `adaptive_spec`, `online_sisd` (self-correcting).
- **`dist/routing/`** (7) — `arbitrage_engine`, `composite`, `consistent_hash`, `latency_aware`, `load_aware`.
- **`dist/backends/`** (7) — distributed deployment backends: `vllm`, `llamacpp`, `ray` pipeline engines; `backend_profiles`, `health_monitor`, `graceful_degradation`. *(distinct from `src/distllm/backends`)*.
- **`dist/disagg/`** (5) — prefill/decode split + KV transfer: `transfer_engine.KVCacheTransferEngine`, `transfer_compression`, `pool`, `transfer`, `__init__.DisaggManager`.
- **`dist/daas/`** (5) — multi-tenant DaaS: `tenant_dispatcher`, `usage_meter`, `load_balancer`, `resource_isolation`, `marketplace_integration`.
- **`dist/simulation/`** (7) — digital-twin & what-if: `cluster_simulator`, `topology_optimizer`, `what_if`, `cost_aware_provisioning`, `chaos_simulator`, `digital_twin`.
- **`dist/privacy/`** (2) — split inference (`PrivacyEnforcer`, `ActivationObfuscator`, `PromptShamirSplitter`) + DP accounting.
- **`dist/chaos/`** (3) — `fault_injector.FaultInjector` (gRPC-interceptor fault injection, Litmus/ChaosMesh YAML), `chaos_orchestrator`.
- **`dist/structured_output/`** — `engine.py` (distributed SO engine).

## Notes / dead code

- **Generated:** `node_pb2*` are protoc output — schema in `proto/node.proto`.
- **SCAFFOLD:** `ebpf_transport.py`.
- **Deprecated shim:** `wandb_integration.py`.
- **Duplication:** `redundant.py` (older) vs `redundant_executor.py` (production); the STUN/TURN/ICE stack is triplicated across `nat.py`, `webrtc.py`, `ice_transport.py` (webrtc supersedes nat).
- **Naming shim:** `dist/backends/ray.py` classes `RayPipeline` re-exported as `RayPipelineEngine` under try/except.
- Committed worktree has uncommitted edits (attention.py/byzantine.py LOC differ from HEAD).

## Tests

`tests/dist/` (121 files) covers nearly every module + `tests/dist/{pipeline,p2p,simulation,speculative,routing,privacy,daas,chaos}/` + `tests/integration/` + `tests/chaos/`.