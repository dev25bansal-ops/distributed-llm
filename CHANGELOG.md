# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Auth plugin with JWT authentication and RBAC role-based access control (6 roles: admin, model-admin, auditor, inference-only, read-only, viewer)
- Event bus for marketplace plugin communication (6 event types: job.matched, job.started, job.completed, job.failed, job.cancelled, listing.status_changed)
- Health watchdog plugin for continuous node health monitoring with auto circuit-breaking
- Caching plugin with semantic deduplication for repeated prompts (SHA-256 + embedding similarity, LRU + Redis backends)
- `distllm doctor` CLI command for system diagnostics (Python, CUDA, GPU, network, ports, model download)
- `distllm tune` CLI subcommand with `quantize`, `batch`, and `cache` operations
- `generate_stream()` SDK method for token-by-token streaming responses (4 strategy classes: Local, Speculative, DistributedSpeculative, Distributed)
- Config cross-validation using Pydantic `@model_validator` for catching invalid combinations at load time
- Backend registry with health-aware selection (routes to healthy backends only, thread-safe singleton)
- CircuitBreakerMiddleware with graduated backpressure (3 tiers: 500-800, 800-1000, 1000+ pending requests)
- BatchScheduler decomposition into KVCacheManager, PreemptionManager, and BudgetComputer subsystems
- Shared IP extraction utility (`ip_utils.py`) for consistent client IP resolution across middleware
- Autoscaler wired to real Prometheus metrics instead of synthetic signals
- E2E encryption cluster_key validation to reject empty or short keys (minimum 16 characters)
- Coordinator subsystem health tracking (per-component status: ok, missing_deps, failed)
- SQLite persistence layer for PromptExchange and Marketplace
- Kubernetes NetworkPolicy restricting pod-to-pod traffic
- Kubernetes PodDisruptionBudget for coordinator (minAvailable: 1)
- Grafana password now required via environment variable (fails fast if unset)

### Security
- API key no longer logged in plaintext (masked in all log output)
- Kubernetes security hardening (non-root, read-only filesystem, dropped capabilities)

### Planned
- Web dashboard for monitoring and management (in progress)
- Auto-discovery of nodes on LAN
- Model compression pipeline (pruning, quantization, distillation)
- P2P KV cache gossip protocol for cross-cluster cache sharing
- Chaos engineering fault injection for resilience testing
- Canary deployment automation with rollback

## [0.4.0] - 2026-05-16

### Production Readiness
- Graceful shutdown with 7-phase cleanup (drain requests, persist cache, stop server, close resources)
- Structured error responses (ErrorResponse model with standardized format for all HTTP errors)
- Kubernetes readiness (/ready) and liveness (/live) probe endpoints
- Meaningful Prometheus metrics on 503 (service status, coordinator load, scheduler stats)
- Request timeout middleware with per-endpoint limits (5min chat, 1min embeddings)
- Backpressure middleware (reject at 1000 pending requests, graceful shutdown detection)
- YAML config loading wired into API server (--config flag, auto-detect config.yaml, CLI/env override precedence)

### Security Hardening
- TLS certificate generation with expanded SANs (wildcards, IPv6, internal DNS)
- Docker non-root user (distllm:distllm, no root execution)
- Plugin system module prefix allowlist (prevent arbitrary code execution)
- Path traversal prevention for LoRA adapter loading
- Authentication bypass warning logged when API_KEY not set
- Security headers middleware (CSP, X-Frame-Options, HSTS, Referrer-Policy)
- CORS configuration with explicit origins and disabled credentials

### Bug Fixes
- gRPC Infer/StreamInfer routing (actual inference instead of pass-through)
- Health check with real GPU stats (pynvml integration for utilization, temperature)
- Circuit breaker thread safety (threading.Lock instead of asyncio.Lock)
- Request isolation via contextvars (per-request ID tracking)
- Rate limiter bounded memory (LRU eviction with max_clients limit)
- Connection pooling with public gRPC API (check_connectivity_state instead of private _channel)
- Embeddings fallback removed (no longer returns random vectors)

### API
- Request timeout middleware (asyncio.timeout with per-endpoint limits)
- Backpressure middleware (pending request threshold, shutdown detection)
- Structured error responses (exception handlers for HTTPException and general exceptions)
- `/ready` endpoint (Kubernetes readiness probe with node health check)
- `/live` endpoint (Kubernetes liveness probe with uptime reporting)
- Enhanced `/metrics` endpoint (service status metrics even when not initialized)

### Infrastructure
- YAML config loading with precedence: CLI > env vars > config.yaml > defaults
- `--config` flag for API server
- Non-root Dockerfile execution
- Expanded TLS certificate SANs for production deployments

## [0.3.0] - 2026-05-14

### API
- CORS middleware with configurable origins
- Security headers middleware (CSP, X-Frame-Options, HSTS)
- Request ID middleware for request tracking
- Authentication middleware with API key support
- Rate limiting middleware with token bucket algorithm

### Features
- LoRA adapter management (load, set, list via /v1/adapters)
- Dynamic generation parameter updates mid-stream (/v1/update-params)
- Prefix cache for shared prompt optimization
- Chunked prefill for long context handling
- Continuous batching with priority queuing

### Fixes
- Async gRPC tests updated to use public channel API
- Mock coordinator fixtures updated for local_partitioner support
- Tokenizer decode serialization in streaming tests

## [0.2.0] - 2026-05-13

### Features
- Speculative decoding with draft model support
- Tensor parallelism for multi-GPU inference
- Model compression pipeline configuration
- Cost-aware scheduling with budget tracking
- Prometheus metrics exporter
- System monitoring with CPU/GPU metrics

### Infrastructure
- Docker Compose for multi-node deployment
- Windows batch scripts for quick start
- Config-driven deployment (YAML)

## [0.1.0] - 2026-05-12

Initial release of distributed LLM inference system.

### Core
- Pipeline parallelism over gRPC for distributed LLM inference
- Model partitioning with automatic architecture detection
- Support for GPT-2, GPT-Neo, Llama, Qwen2.5, Mistral, Phi, StableLM, Pythia
- Rotary position embedding (RoPE) support for modern architectures
- KV cache management for efficient autoregressive generation
- Efficient tensor serialization (raw bytes via protobuf)
- Layer signature caching for fast forward dispatch

### API
- OpenAI-compatible REST API (FastAPI)
- `/v1/chat/completions` endpoint with streaming support
- `/v1/completions` endpoint with streaming support
- `/v1/models` endpoint for model listing
- `/health` endpoint for health checks

### Infrastructure
- Docker and docker-compose support
- TLS encryption for gRPC communication
- Config-driven deployment (YAML)
- Health checks and retry logic for node communication
- Windows batch scripts for quick start

### Fixes
- Fixed KV cache handling for per-node distributed pipeline
- Fixed tensor serialization (bytes vs float lists, 4-8x bandwidth reduction)
- Fixed distributed mode pipeline with proper KV cache tracking
- Fixed streaming API to use native model generation
- Fixed layer signature inspection caching
- Updated Dockerfile CUDA 12.1 -> 12.8 for RTX 50-series support
- Fixed config layer mismatch for TinyStories-1M (8 layers)
- Fixed test.py broken references

[Unreleased]: https://github.com/distributed-llm/distributed-llm/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/distributed-llm/distributed-llm/releases/tag/v0.1.0
