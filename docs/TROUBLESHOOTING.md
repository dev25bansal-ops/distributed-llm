# Troubleshooting Runbook

Operational runbook for diagnosing and resolving issues in Distributed LLM (DistLLM) clusters. Each section lists the exact symptom, root cause, step-by-step fix, and prevention measures.

---

## Table of Contents

<!-- TOC -->

- [1. Installation Issues](#1-installation-issues)
  - [1.1 `pip install` fails with CUDA errors](#11-pip-install-fails-with-cuda-errors)
  - [1.2 `grpcio` build fails on Windows](#12-grpcio-build-fails-on-windows)
  - [1.3 `distllm` command not found after install](#13-distllm-command-not-found-after-install)
  - [1.4 Dependency version conflicts](#14-dependency-version-conflicts)
- [2. Model Loading Failures](#2-model-loading-failures)
  - [2.1 CUDA out of memory during model load](#21-cuda-out-of-memory-during-model-load)
  - [2.2 Unsupported model architecture](#22-unsupported-model-architecture)
  - [2.3 Missing tokenizer](#23-missing-tokenizer)
  - [2.4 HuggingFace authentication failure for gated models](#24-huggingface-authentication-failure-for-gated-models)
  - [2.5 Model download interrupted / corrupted cache](#25-model-download-interrupted--corrupted-cache)
- [3. Distributed Pipeline Errors](#3-distributed-pipeline-errors)
  - [3.1 Node unreachable](#31-node-unreachable)
  - [3.2 gRPC timeout during forward pass](#32-grpc-timeout-during-forward-pass)
  - [3.3 KV cache mismatch between nodes](#33-kv-cache-mismatch-between-nodes)
  - [3.4 Circuit breaker tripped on a node](#34-circuit-breaker-tripped-on-a-node)
  - [3.5 Protobuf serialization error](#35-protobuf-serialization-error)
- [4. API Errors](#4-api-errors)
  - [4.1 HTTP 503 — Service Unavailable](#41-http-503--service-unavailable)
  - [4.2 HTTP 429 — Too Many Requests](#42-http-429--too-many-requests)
  - [4.3 HTTP 504 — Gateway Timeout](#43-http-504--gateway-timeout)
  - [4.4 HTTP 401 — Authentication Failure](#44-http-401--authentication-failure)
  - [4.5 HTTP 400 — Bad Request (input validation)](#45-http-400--bad-request-input-validation)
- [5. Performance Issues](#5-performance-issues)
  - [5.1 Slow inference (< 1 token/sec)](#51-slow-inference--1-tokensec)
  - [5.2 High first-token latency (slow time-to-first-token)](#52-high-first-token-latency-slow-time-to-first-token)
  - [5.3 High memory usage / KV cache thrashing](#53-high-memory-usage--kv-cache-thrashing)
  - [5.4 Inconsistent latency (jitter)](#54-inconsistent-latency-jitter)
- [6. GPU Issues](#6-gpu-issues)
  - [6.1 CUDA out of memory during inference](#61-cuda-out-of-memory-during-inference)
  - [6.2 CUDA driver version mismatch](#62-cuda-driver-version-mismatch)
  - [6.3 Multi-GPU not detected](#63-multi-gpu-not-detected)
  - [6.4 GPU temperature throttling](#64-gpu-temperature-throttling)
- [7. Kubernetes Issues](#7-kubernetes-issues)
  - [7.1 Pod CrashLoopBackOff](#71-pod-crashloopbackoff)
  - [7.2 Readiness probe failing](#72-readiness-probe-failing)
  - [7.3 Liveness probe killing healthy pods](#73-liveness-probe-killing-healthy-pods)
  - [7.4 GPU resource not schedulable](#74-gpu-resource-not-schedulable)
  - [7.5 Pod eviction due to memory pressure](#75-pod-eviction-due-to-memory-pressure)
- [8. Docker Issues](#8-docker-issues)
  - [8.1 Container can't connect to other containers](#81-container-cant-connect-to-other-containers)
  - [8.2 GPU not accessible inside container](#82-gpu-not-accessible-inside-container)
  - [8.3 Model cache not shared between containers](#83-model-cache-not-shared-between-containers)
  - [8.4 Container networking — port conflicts](#84-container-networking--port-conflicts)
- [9. TLS / Certificate Issues](#9-tls--certificate-issues)
  - [9.1 TLS handshake failure](#91-tls-handshake-failure)
  - [9.2 Certificate expired](#92-certificate-expired)
  - [9.3 gRPC TLS mutual authentication failure](#93-grpc-tls-mutual-authentication-failure)
- [10. Federation Issues](#10-federation-issues)
  - [10.1 Split-brain between federated clusters](#101-split-brain-between-federated-clusters)
  - [10.2 Peer discovery not working](#102-peer-discovery-not-working)
  - [10.3 Cross-cluster model routing inconsistency](#103-cross-cluster-model-routing-inconsistency)
- [11. Diagnostic Commands](#11-diagnostic-commands)
  - [System Health](#system-health)
  - [Logs](#logs)
  - [Performance](#performance)
  - [GPU](#gpu)
  - [Network](#network)
  - [Docker](#docker)
  - [Kubernetes](#kubernetes)
- [Getting Help](#getting-help)

<!-- /TOC -->

---

## 1. Installation Issues

### 1.1 `pip install` fails with CUDA errors

**Symptom:**
```
ERROR: Could not find a version that satisfies the requirement torch
```
or
```
ERROR: No matching distribution found for torch
```

**Root cause:** PyTorch requires a specific index URL for CUDA-enabled wheels. The default PyPI index only provides CPU-only builds.

**Fix:**
```bash
# Install PyTorch with CUDA support first
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Then install DistLLM
pip install distributed-llm
```

**Prevention:** Pin the PyTorch index URL in your `requirements.txt`:
```
--extra-index-url https://download.pytorch.org/whl/cu121
torch>=2.1.0
```

---

### 1.2 `grpcio` build fails on Windows

**Symptom:**
```
error: Microsoft Visual C++ 14.0 or greater is required
```

**Root cause:** No pre-built wheel available for your Python version/platform, and the source build requires a C compiler.

**Fix:**
```bash
# Install pre-built wheel (skip source compilation)
pip install grpcio --only-binary=:all:
```

**Prevention:** Use Python 3.10-3.12 on Windows (best wheel coverage). Alternatively, install the [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

---

### 1.3 `distllm` command not found after install

**Symptom:**
```
'distllm' is not recognized as an internal or external command
```

**Root cause:** The pip install scripts directory is not on your `PATH`.

**Fix:**
```bash
# Find where pip installed the script
python -m site --user-scripts

# Add that directory to PATH, or run via module
python -m distllm.cli.main --help
```
> ⚠️ UNVERIFIED (hygiene pass 2026-11-06) — there is no cli/__main__.py;
> `python -m distllm.cli.main` works only if the package is importable.
> The installed entry point is the `distllm` console script
> (pyproject.toml [project.scripts]: distllm = "distllm.cli.main:main").

**Prevention:** Use a virtual environment (`python -m venv .venv`) so scripts land in a predictable location.

---

### 1.4 Dependency version conflicts

**Symptom:**
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed
```
or import errors at runtime like `AttributeError: module 'transformers' has no attribute ...`

**Root cause:** Installed packages have incompatible version ranges.

**Fix:**
```bash
# Create a clean environment
python -m venv .venv-clean
source .venv-clean/bin/activate  # Linux/macOS
# .venv-clean\Scripts\activate   # Windows

# Install with locked requirements
pip install -r requirements.lock
pip install distributed-llm
```

**Prevention:** Always use `pip install -r requirements.lock` in production. Generate the lock file with `pip freeze > requirements.lock`.

---

## 2. Model Loading Failures

### 2.1 CUDA out of memory during model load

**Symptom:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate X MiB
```
or in DistLLM logs:
```
[ERROR] OOM_ERROR: GPU out of memory on node node_0
```

**Root cause:** The model weights exceed available GPU VRAM. This happens when the model is too large for a single GPU, or residual memory from a previous process is still allocated.

**Fix:**
```bash
# Step 1: Verify GPU memory is free
nvidia-smi
# If another process is using the GPU, kill it or wait

# Step 2: Use quantization to reduce memory footprint
# Quantization is configured via config.yaml quantization.method
# ("none" | "bnb_4bit" | "bnb_8bit" | "gptq" | "awq" | "fp8",
# src/distllm/config/_model.py:64) or the API server's --quantization flag.
# Env var: DISTLLM__QUANTIZATION__METHOD=bnb_8bit
# There is no DISTLLM_QUANTIZATION_BITS env var.

# Step 3: Enable CPU offloading for KV cache
# KV offload is a runtime feature of the KV cache manager
# (offload_to_cpu, src/distllm/core/kv_cache.py:277) and tiered defrag
# compaction (defragmentation.tiered_compaction in config.yaml).
# There is no DISTLLM_CPU_OFFLOAD env var.

# Step 4: Reduce layers assigned to this node
# In config.yaml, narrow the layer range:
#   start_layer: 0
#   end_layer: 7   # was 15

# Step 5: Use more nodes to spread the model
distllm --model meta-llama/Llama-3.1-70B --nodes node0:50051:0:15 node1:50052:16:31
```

**Prevention:**
- Check the [Model Compatibility Matrix](MODEL_COMPATIBILITY.md) for VRAM requirements before deploying.
- Use `distllm model info <model>` to see estimated memory per node.
- Set `quantization.method: bnb_8bit` in config.yaml as a safe default for production.
> ⚠️ UNVERIFIED (hygiene pass 2026-11-06) — the doc's original
> `DISTLLM_QUANTIZATION_BITS=8` default does not exist (0 hits in src/ and
> config.yaml).

---

### 2.2 Unsupported model architecture

**Symptom:**
```
ModelError: Failed to load model 'org/custom-model': Architecture 'CustomForCausalLM' is not supported
```
or
```
KeyError: 'CustomForCausalLM'
```

**Root cause:** The model's architecture class is not registered in DistLLM's `ModelPartitioner`. This happens with custom or newly released model architectures.

**Fix:**
```bash
# Step 1: Check if the model works with trust_remote_code
export DISTLLM_TRUST_REMOTE_CODE=true
distllm model load org/custom-model

# Step 2: If still failing, check the model's config.json for the architecture name
python -c "from transformers import AutoConfig; c = AutoConfig.from_pretrained('org/custom-model'); print(c.architectures)"

# Step 3: If the architecture is a known variant (e.g., LlamaForCausalLM),
# you may be able to alias it. File an issue with the architecture name.
```

**Prevention:**
- Check [MODEL_COMPATIBILITY.md](MODEL_COMPATIBILITY.md) before selecting a model.
- When using community models, always set `DISTLLM_TRUST_REMOTE_CODE=true`.

---

### 2.3 Missing tokenizer

**Symptom:**
```
OSError: org/model does not appear to have a file named tokenizer.json
```
or
```
Tokenizer not found: Unable to load tokenizer for 'org/model'
```

**Root cause:** The model repository is missing tokenizer files, or the tokenizer class requires `sentencepiece`/`tokenizers` packages that are not installed.

**Fix:**
```bash
# Step 1: Ensure tokenizer dependencies are installed
pip install sentencepiece tokenizers protobuf

# Step 2: For SentencePiece models (Llama, Gemma, etc.)
pip install sentencepiece

# Step 3: If using a GGUF model with llama.cpp backend, the tokenizer
# is embedded in the model file — no separate tokenizer needed

# Step 4: For models with tokenizer in a subdirectory
# There is no DISTLLM_TOKENIZER_SUBFOLDER env var.
# Set tokenizer subfolder via config.yaml model.* settings instead.
distllm model load org/model
```
> ⚠️ UNVERIFIED (hygiene pass 2026-11-06) — DISTLLM_TOKENIZER_SUBFOLDER
> has 0 hits in src/ and config.yaml.

**Prevention:** Install the full dependency set: `pip install distributed-llm[all]`.

---

### 2.4 HuggingFace authentication failure for gated models

**Symptom:**
```
huggingface_hub.utils._errors.HfHubHTTPError: 401 Client Error: Unauthorized
```
or
```
403 Forbidden: Access to model meta-llama/Llama-3.1-70B is restricted
```

**Root cause:** The model is gated (requires acceptance of a license agreement on HuggingFace) and no valid token is provided.

**Fix:**
```bash
# Step 1: Accept the model license on HuggingFace
# Visit https://huggingface.co/meta-llama/Llama-3.1-70B and accept the terms

# Step 2: Generate an access token at https://huggingface.co/settings/tokens

# Step 3: Set the token
export HUGGING_FACE_HUB_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx

# Step 4: Verify
python -c "from huggingface_hub import HfApi; print(HfApi().whoami())"
```

**Prevention:** Use a `.env` file or secret manager. Never commit tokens to version control.

---

### 2.5 Model download interrupted / corrupted cache

**Symptom:**
```
RuntimeError: Error(s) in loading state_dict: size mismatch for model.layers.0.self_attn.q_proj.weight
```
or
```
safetensors_rust.SafetensorError: Error deserializing header
```

**Root cause:** The model download was interrupted, leaving partial or corrupted files in the HuggingFace cache.

**Fix:**
```bash
# Step 1: Clear the corrupted model from cache
huggingface-cli delete-cache --model org/model

# Or manually:
rm -rf ~/.cache/huggingface/hub/models--org--model

# Step 2: Re-download
# There is no `distllm model download` subcommand — `distllm model` only
# supports list/info/load/unload (src/distllm/cli/main.py:454-487).
# Use huggingface-cli instead:
huggingface-cli download org/model

# Step 3: Verify integrity
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('org/model')"
```

**Prevention:** Downloads resume automatically via huggingface_hub.
> ⚠️ UNVERIFIED (hygiene pass 2026-11-06) — DISTLLM_DOWNLOAD_RESUME has
> 0 hits in src/ and config.yaml; there is no such flag to set.

---

## 3. Distributed Pipeline Errors

### 3.1 Node unreachable

**Symptom:**
```
NodeUnreachableError: Node node_1 at 192.168.1.11:50051 is unreachable
```
or in gRPC logs:
```
grpc._channel._InactiveRpcError: <InactiveRpcError of RPC that terminated with: StatusCode.UNAVAILABLE>
```

**Root cause:** The worker node process is not running, the host is down, a firewall is blocking the port, or the gRPC server failed to start.

**Fix:**
```bash
# Step 1: Check if the worker process is running
ssh user@192.168.1.11 "ps aux | grep distllm-node"

# Step 2: If not running, start it (the `distllm-node` console script maps
# to distllm.dist.worker:main, pyproject.toml [project.scripts])
distllm-node --node-id node_1 --model meta-llama/Llama-3.1-8B \
  --start-layer 0 --end-layer 15 --total-layers 32 \
  --coordinator-host coordinator --coordinator-port 50050 --port 50051

# Step 3: Test network connectivity
nc -zv 192.168.1.11 50051
# or
telnet 192.168.1.11 50051

# Step 4: Test gRPC health endpoint
grpcurl -plaintext 192.168.1.11:50051 grpc.health.v1.Health/Check

# Step 5: Check firewall rules
sudo iptables -L -n | grep 50051
# or on Windows:
netsh advfirewall firewall show rule name=all | findstr 50051

# Step 6: Check coordinator logs for the node's last seen time
distllm cluster status
```

**Prevention:**
- Use `systemd` or a process supervisor to auto-restart crashed workers.
- Set up health check monitoring (see [Docker healthchecks](#8-docker-issues) or [Kubernetes probes](#7-kubernetes-issues)).
- Open firewall ports during initial setup.

---

### 3.2 gRPC timeout during forward pass

**Symptom:**
```
GRPCTimeoutError: gRPC call to node node_2 at 192.168.1.12:50051 timed out after 30.0s
```
or
```
grpc.RpcError: StatusCode.DEADLINE_EXCEEDED
```

**Root cause:** The worker node is overloaded (GPU saturated), the network link is slow/lossy, or the request payload (hidden states tensor) is too large for the timeout window.

**Fix:**
```bash
# Step 1: Check the worker's GPU utilization
ssh user@192.168.1.12 "nvidia-smi"
# If GPU utilization is 100% and memory is maxed out, the node is overloaded

# Step 2: Increase the pipeline timeout in config.yaml
# pipeline:
#   timeout_seconds: 60.0   # was 30.0

# Step 3: Check network bandwidth between nodes
iperf3 -c 192.168.1.12 -t 10

# Step 4: Enable gRPC compression to reduce payload size
# Compression is the compression config section (CompressionSettings,
# src/distllm/config/_network.py); there is no DISTLLM_GRPC_COMPRESSION env var.
# Env var: DISTLLM__COMPRESSION__<FIELD>=... (see config reference)

# Step 5: Enable activation quantization to shrink tensor transfers
# > ⚠️ UNVERIFIED (hygiene pass 2026-11-06) — DISTLLM_ACTIVATION_QUANT_BITS
# > has 0 hits in src/ and config.yaml; no such mechanism exists.
export DISTLLM_ACTIVATION_QUANT_BITS=8

# Step 6: For WAN links, switch to QUIC transport
# > ⚠️ UNVERIFIED (hygiene pass 2026-11-06) — DISTLLM_WAN_TRANSPORT has
# > 0 hits; the real QUIC transport is src/distllm/dist/p2p/quic_transport.py
# > (auto-selected via get_optimal_transport when aioquic is installed).
export DISTLLM_WAN_TRANSPORT=quic
```

**Prevention:**
- Use `iperf3` to validate inter-node bandwidth before deployment.
- Monitor `grpc_server_handling_seconds` via Prometheus.
- Set `pipeline.timeout_seconds` to 2x your expected worst-case latency.

---

### 3.3 KV cache mismatch between nodes

**Symptom:**
```
SerializationError: KV cache shape mismatch: expected [1, 32, 128, 64] got [1, 32, 256, 64]
```
or inference produces garbled/incorrect output after a few tokens.

**Root cause:** Nodes were started with different models, different `total_layers` values, or one node was restarted mid-inference while another still holds stale cache.

**Fix:**
```bash
# Step 1: Verify all nodes are running the same model
distllm cluster status
# Check that model_name, total_layers, and dtype match on every node

# Step 2: If mismatched, restart the workers with consistent config.
# There is no `distllm cluster restart`; leave and re-join each worker, or
# restart the worker processes directly (systemd, k8s, docker compose).
distllm cluster leave
distllm cluster join --coordinator <coordinator-host> --port 50051

# Step 3: Clear KV cache on all nodes
curl -X POST http://coordinator:8000/admin/cache/clear

# Step 4: Verify KV cache consistency
curl http://coordinator:8000/metrics | grep kv_cache
```

**Prevention:**
- Always deploy from a single `config.yaml` shared across all nodes.
- Use `distllm config validate` before starting the cluster.
- Do not change `total_layers` or model name without restarting all nodes.

---

### 3.4 Circuit breaker tripped on a node

**Symptom:**
```
CircuitBreakerError: Circuit breaker open for node node_1 after 5 failures, recovery in 30.0s
```

**Root cause:** A node has failed multiple consecutive requests (OOM, timeout, crash). The circuit breaker opens to stop sending traffic to the failing node.

**Fix:**
```bash
# Step 1: Check why the node is failing
distllm system logs --node node_1 --tail 50

# Step 2: Check GPU health on that node
ssh user@node-1 "nvidia-smi"

# Step 3: If the node is healthy now, the breaker clears on its own once its
# cooldown elapses -- no reset subcommand exists (`distllm cluster --help`
# lists status/scale/drain/rebalance/start/join/leave/list-nodes/deploy/
# autopsy). Confirm the state and, if it is still open, restart that worker.
distllm cluster status
# The metrics endpoint reports the breaker state per node:
curl http://coordinator:8000/metrics | grep circuit_breaker

# Step 4: If the node is unhealthy, drain it and redistribute
distllm cluster drain node_1
# The coordinator will redistribute layers to remaining nodes
```

**Prevention:**
- Monitor `distllm_circuit_breaker_state` metric in Prometheus.
- Set appropriate `pipeline.max_retries` (default: 3).
- Configure auto-restart for worker processes.

---

### 3.5 Protobuf serialization error

**Symptom:**
```
ProtoError: Failed to decode ForwardPassRequest: invalid wire type 7 for field hidden_states
```
or
```
google.protobuf.message.DecodeError: Error parsing message
```

**Root cause:** Version mismatch between coordinator and worker protobuf definitions, or the tensor data exceeds the configured size limit.

**Fix:**
```bash
# Step 1: Ensure all nodes run the same DistLLM version
distllm --version  # Run on every node

# Step 2: If versions match, check if the tensor is too large
# Default limit is 4 GB; increase if needed:
export DISTLLM_MAX_TENSOR_BYTES=8589934592  # 8 GB

# Step 3: Rebuild protobuf stubs if you modified .proto files
cd proto && python -m grpc_tools.protoc -I. --python_out=../src/distllm/dist/ --grpc_python_out=../src/distllm/dist/ node.proto
```

**Prevention:** Always deploy the same DistLLM version across all nodes. There
is no `distllm cluster version-check` subcommand; compare `distllm --version`
on each host before a rolling upgrade.

---

## 4. API Errors

### 4.1 HTTP 503 — Service Unavailable

**Symptom:**
```json
{"error": {"message": "Service temporarily unavailable", "type": "service_unavailable", "code": 503}}
```

**Root cause:** The coordinator is starting up, all worker nodes are unhealthy, or the cluster is at capacity and cannot accept new requests.

**Fix:**
```bash
# Step 1: Check coordinator health
curl http://coordinator:8000/health

# Step 2: Check worker node status
curl http://coordinator:8000/v1/nodes

# Step 3: If nodes are down, bring them up
distllm cluster status
distllm cluster start

# Step 4: If at capacity, increase batch size or add nodes
# Batching is the batching config section (BatchingSettings,
# src/distllm/config/_parallelism.py:122): max_batch_size (default 32).
# Env var: DISTLLM__BATCHING__MAX_BATCH_SIZE=32
# There is no DISTLLM_MAX_BATCH_SIZE / DISTLLM_MAX_NUM_SEQS env var
# (vLLM's max_num_seqs is config.yaml vllm.max_num_seqs, default 256).
export DISTLLM_MAX_BATCH_SIZE=32
export DISTLLM_MAX_NUM_SEQS=512

# Step 5: Check if the readiness probe is failing (Kubernetes)
kubectl describe pod distllm-coordinator-xxxxx
```

**Prevention:**
- Set appropriate `readinessProbe` values so Kubernetes doesn't route traffic before the coordinator is ready.
- Monitor `distllm_coordinator_queue_depth` (real metric, src/distllm/observability/exporter.py:86) and scale workers when the queue depth is sustained above threshold.
> ⚠️ UNVERIFIED (hygiene pass 2026-11-06) — the doc's original
> `distllm_requests_queued` does not exist.

---

### 4.2 HTTP 429 — Too Many Requests

**Symptom:**
```json
{"error": {"message": "Rate limit exceeded", "type": "rate_limit_error", "code": 429}}
```
Headers include: `Retry-After: 12`, `X-RateLimit-Remaining: 0`

**Root cause:** The client has exceeded the configured rate limit (requests per minute).

**Fix:**
```bash
# Step 1: Check current rate limit configuration
curl http://coordinator:8000/metrics | grep rate_limit

# Step 2: If limits are too low, increase them in config.yaml
# api:
#   rate_limit_requests: 1000        # was 100
#   rate_limit_window_seconds: 60

# Step 3: Implement client-side backoff
# Respect the Retry-After header in your client code

# Step 4: If using Redis-backed rate limiting, check Redis health
redis-cli -h redis PING
```

**Client-side backoff example (Python):**
```python
import time
import openai

client = openai.OpenAI(base_url="http://coordinator:8000/v1")

for attempt in range(5):
    try:
        response = client.chat.completions.create(model="llama-3", messages=[...])
        break
    except openai.RateLimitError as e:
        retry_after = int(e.response.headers.get("Retry-After", 5))
        time.sleep(retry_after)
```

**Prevention:**
- Use Redis-backed distributed rate limiting for multi-instance deployments.
- Set per-client API keys with appropriate limits via RBAC.

---

### 4.3 HTTP 504 — Gateway Timeout

**Symptom:**
```json
{"error": {"message": "Request timed out", "type": "timeout", "code": 504}}
```

**Root cause:** The inference request took longer than the configured timeout. Common with very long prompts, large `max_tokens`, or slow WAN-connected nodes.

**Fix:**
```bash
# Step 1: Check if the request is still queued
curl http://coordinator:8000/metrics | grep requests_queued

# Step 2: Increase the API timeout
# config.yaml
# pipeline:
#   timeout_seconds: 120.0   # was 30.0

# Step 3: For long-context requests, enable chunked prefill
# Chunked prefill is the chunked_prefill config section
# (ChunkedPrefillSettings, src/distllm/config/_parallelism.py:135):
# config.yaml: chunked_prefill.enabled: true (default), chunked_prefill.chunk_size: 512
# Env var: DISTLLM__CHUNKED_PREFILL__ENABLED=true
# Env var: DISTLLM__CHUNKED_PREFILL__CHUNK_SIZE=512
# There is no DISTLLM_CHUNKED_PREFILL / DISTLLM_CHUNKED_PREFILL_CHUNK_SIZE env var.
export DISTLLM_CHUNKED_PREFILL=true
export DISTLLM_CHUNKED_PREFILL_CHUNK_SIZE=512

# Step 4: Check network latency to all nodes
for node in node_0 node_1 node_2; do
  echo "$node: $(ping -c 3 $node | tail -1)"
done

# Step 5: If using a reverse proxy (nginx), increase proxy timeout
# nginx.conf:
#   proxy_read_timeout 300s;
#   proxy_send_timeout 300s;
```

**Prevention:**
- Set `pipeline.timeout_seconds` based on your P99 latency + buffer.
- Use `DISTLLM__CHUNKED_PREFILL__ENABLED=true` for long prompts.
> ⚠️ UNVERIFIED (hygiene pass 2026-11-06) — the doc's original
> `DISTLLM_CHUNKED_PREFILL=true` env var does not exist; the real form is
> DISTLLM__CHUNKED_PREFILL__ENABLED (nested delimiter __,
> src/distllm/config/settings.py:175-176).
- Monitor `distllm_request_duration_seconds` histogram.

---

### 4.4 HTTP 401 — Authentication Failure

**Symptom:**
```json
{"error": {"message": "Invalid API key", "type": "authentication_error", "code": 401}}
```

**Root cause:** The `Authorization` header is missing, the key is malformed, or the key hash doesn't match what's stored on the coordinator.

**Fix:**
```bash
# Step 1: Verify the key is set in the environment
echo $API_KEY

# Step 2: Test with curl
curl -H "Authorization: Bearer $API_KEY" http://coordinator:8000/v1/models

# Step 3: If no key is set, either set one or disable auth
export API_KEY="your-secret-key"
# Or disable auth for development:
distllm system api --model <model> --local --no-auth

# Step 4: Confirm the key is configured (no `config list-keys` subcommand
# exists; config subcommands are setup/validate/reference/openapi/webhook/
# quota/backup). `validate` reports the effective auth settings.
distllm config validate

# Step 5: If using the cluster key for gRPC auth, ensure it matches on all nodes
echo $DISTLLM_CLUSTER_KEY
# Must be identical on coordinator and all workers
```

**Prevention:**
- Store API keys in a secret manager (Vault, AWS Secrets Manager).
- Use `distllm config validate` to verify the key is properly configured before starting.

---

### 4.5 HTTP 400 — Bad Request (input validation)

**Symptom:**
```json
{"error": {"message": "Invalid input: batch would exceed capacity: 50000 > 32768 tokens", "type": "invalid_request_error", "code": 400}}
```

**Root cause:** The request exceeds configured limits (max tokens, max message length, tensor dimensions).

**Fix:**
```bash
# Step 1: Reduce the input size
# - Shorten the prompt
# - Reduce max_tokens
# - Split into multiple requests

# Step 2: Increase limits if appropriate
# config.yaml
# api:
#   max_request_size_mb: 64
#   max_message_length: 262144
# generation:
#   max_new_tokens: 512
```

---

## 5. Performance Issues

### 5.1 Slow inference (< 1 token/sec)

**Symptom:** Generation takes many seconds per token. Users report extremely slow responses.

**Root cause:** Usually one of: GPU not utilized, network bottleneck, wrong pipeline strategy, or KV cache thrashing.

**Fix:**
```bash
# Step 1: Check GPU utilization
nvidia-smi
# If GPU is idle (< 10%), the model is not loaded on GPU

# Step 2: Check network latency between nodes
ping -c 10 node-1
iperf3 -c node-1

# Step 3: Check pipeline metrics
curl http://coordinator:8000/metrics

# Step 4: Enable overlap pipeline (2-4x throughput)
# pipeline overlap is hybrid_parallel.pp_overlap (default true):
# Env var: DISTLLM__HYBRID_PARALLEL__PP_OVERLAP=true
export DISTLLM_ENABLE_PIPELINE_OVERLAP=true

# Step 5: For WAN links, use QUIC transport
# > ⚠️ UNVERIFIED (hygiene pass 2026-11-06) — DISTLLM_WAN_TRANSPORT has
# > 0 hits; real QUIC transport is src/distllm/dist/p2p/quic_transport.py.
export DISTLLM_WAN_TRANSPORT=quic

# Step 6: Enable Flash Attention
# Flash attention is applied programmatically (apply_flash_attention_to_model,
# src/distllm/dist/flash_attention.py:149) using the flash_attn package;
# there is no DISTLLM_FLASH_ATTENTION env var.
export DISTLLM_FLASH_ATTENTION=true

# Step 7: Enable CUDA Graph capture
# CUDA graph capture is the cuda_graph config section
# (CudaGraphSettings, src/distllm/config/_performance.py:14):
# config.yaml: cuda_graph.enabled: true (default false)
# Env var: DISTLLM__CUDA_GRAPH__ENABLED=true
export DISTLLM_CUDA_GRAPH=true

# Step 8: Increase batch size
# batching.max_batch_size in config.yaml (BatchingSettings,
# src/distllm/config/_parallelism.py:122); env var
# DISTLLM__BATCHING__MAX_BATCH_SIZE=16
export DISTLLM_MAX_BATCH_SIZE=16
```

**Prevention:**
- Always enable `DISTLLM__HYBRID_PARALLEL__PP_OVERLAP=true` in production.
> ⚠️ UNVERIFIED (hygiene pass 2026-11-06) — the doc's original
> `DISTLLM_ENABLE_PIPELINE_OVERLAP=true` env var does not exist; pp_overlap
> defaults to true (src/distllm/config/_parallelism.py:73) so normally no
> action is needed.
- Use `distllm benchmark run --model <model>` to establish baseline performance.
- Monitor `distllm_tokens_per_second` in Grafana.

---

### 5.2 High first-token latency (slow time-to-first-token)

**Symptom:** The first token takes 5-30 seconds to appear, but subsequent tokens are fast.

**Root cause:** Long prompt prefill is blocking the decode pipeline. The entire prompt must be processed through all nodes before generation begins.

**Fix:**
```bash
# Step 1: Enable chunked prefill
# chunked_prefill config section (ChunkedPrefillSettings,
# src/distllm/config/_parallelism.py:135); env vars
# DISTLLM__CHUNKED_PREFILL__ENABLED=true / DISTLLM__CHUNKED_PREFILL__CHUNK_SIZE=512
export DISTLLM_CHUNKED_PREFILL=true
export DISTLLM_CHUNKED_PREFILL_CHUNK_SIZE=512

# Step 2: Enable prefix caching for repeated system prompts
# prefix_cache config section (PrefixCacheSettings,
# src/distllm/config/_cache.py:16): prefix_cache.enabled (default true),
# prefix_cache.max_entries (default 1024). Env vars:
# DISTLLM__PREFIX_CACHE__ENABLED=true
# DISTLLM__PREFIX_CACHE__MAX_ENTRIES=1000
export DISTLLM_PREFIX_CACHE_ENABLED=true
export DISTLLM_PREFIX_CACHE_MAX_ENTRIES=1000

# Step 3: Use Sarathi-Serve pressure adaptation
# SarathiPolicy scheduler (src/distllm/core/advanced_scheduling/policy.py:39);
# no DISTLLM_SARATHI_* env var exists.
export DISTLLM_SARATHI_PRESSURE_ADAPTATION=true
```

---

### 5.3 High memory usage / KV cache thrashing

**Symptom:** GPU memory is maxed out, frequent OOM warnings, or `kv_cache_eviction_rate` is high in metrics.

**Root cause:** Too many concurrent requests with long sequences, or KV cache quantization is not enabled.

**Fix:**
```bash
# Step 1: Enable KV cache quantization
# > ⚠️ UNVERIFIED (hygiene pass 2026-11-06) — DISTLLM_KV_CACHE_QUANT_BITS
# > has 0 hits; no KV-cache quantization env/config mechanism exists.
export DISTLLM_KV_CACHE_QUANT_BITS=8   # INT8 (50% savings)
# or
export DISTLLM_KV_CACHE_QUANT_BITS=4   # INT4 (75% savings)

# Step 2: Enable PagedAttention
# > ⚠️ UNVERIFIED (hygiene pass 2026-11-06) — DISTLLM_PAGED_ATTENTION has
# > 0 hits; paged attention is always-on in the block pool
# > (src/distllm/dist/block_pool.py) and configurable only via
# > DISTLLM_PAGED_NUM_BLOCKS (src/distllm/core/inference_engine.py:229).
export DISTLLM_PAGED_ATTENTION=true

# Step 3: Enable defragmentation
# defragmentation config section (DefragmentationSettings,
# src/distllm/config/_cache.py:97): defragmentation.enabled/policy/
# interval_seconds. Env vars: DISTLLM__DEFRAGMENTATION__ENABLED=true
# DISTLLM__DEFRAGMENTATION__POLICY=balanced
export DISTLLM_DEFRAG_ENABLED=true
export DISTLLM_DEFRAG_POLICY=balanced

# Step 4: Reduce max sequence length
# > ⚠️ UNVERIFIED (hygiene pass 2026-11-06) — DISTLLM_MAX_SEQ_LEN has
# > 0 hits in src/ and config.yaml.
export DISTLLM_MAX_SEQ_LEN=2048

# Step 5: Enable CPU offloading for KV cache
# > ⚠️ UNVERIFIED (hygiene pass 2026-11-06) — DISTLLM_CPU_OFFLOAD has
# > 0 hits; real mechanism is KV cache offload_to_cpu
# > (src/distllm/core/kv_cache.py:277) / defrag tiered_compaction.
export DISTLLM_CPU_OFFLOAD=true

# Step 6: Reduce max concurrent sequences
# > ⚠️ UNVERIFIED (hygiene pass 2026-11-06) — DISTLLM_MAX_NUM_SEQS has
# > 0 hits; nearest real setting is config.yaml vllm.max_num_seqs
# > (default 256, config.yaml vllm section).
export DISTLLM_MAX_NUM_SEQS=128   # was 256
```

---

### 5.4 Inconsistent latency (jitter)

**Symptom:** P50 latency is acceptable but P99 is 5-10x higher.

**Root cause:** Network jitter on WAN links, GC pauses in the coordinator, or one slow node (straggler) delaying the entire pipeline.

**Fix:**
```bash
# Step 1: Identify the straggler node
curl http://coordinator:8000/metrics | grep node_latency

# Step 2: Use QUIC transport for WAN (no head-of-line blocking)
# > ⚠️ UNVERIFIED (hygiene pass 2026-11-06) — DISTLLM_WAN_TRANSPORT has
# > 0 hits; real QUIC transport is src/distllm/dist/p2p/quic_transport.py.
export DISTLLM_WAN_TRANSPORT=quic

# Step 3: Enable straggler detection and recovery
# The coordinator will automatically bypass slow nodes
# StragglerDetector is always-on (src/distllm/core/coordinator.py:132);
# threshold is the straggler_threshold config field (default 1.5,
# src/distllm/config/_parallelism.py:115) — no env var.
export DISTLLM_STRAGGLER_DETECTION=true
export DISTLLM_STRAGGLER_THRESHOLD_MS=5000

# Step 4: Use priority aging to prevent request starvation
# Aging is implemented in BatchScheduler constructor params
# aging_enabled / aging_interval_s (src/distllm/core/batch_scheduler.py:79-81);
# the priority config section enables queue levels:
# config.yaml: priority.enabled: true
# Env var: DISTLLM__PRIORITY__ENABLED=true
export DISTLLM_PRIORITY_AGING_ENABLED=true
export DISTLLM_PRIORITY_AGING_INTERVAL=30
```

---

## 6. GPU Issues

### 6.1 CUDA out of memory during inference

**Symptom:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 256.00 MiB
(GPU 0; 24.00 GiB total; 22.50 GiB already allocated; 128.00 MiB free)
```

**Root cause:** The model + KV cache + activations exceed GPU memory. Often triggered by a long-sequence request or a batch that's too large.

**Fix:**
```bash
# Step 1: Reduce batch size
# batching config section (BatchingSettings,
# src/distllm/config/_parallelism.py:122); env vars
# DISTLLM__BATCHING__MAX_BATCH_SIZE=4
# vllm.max_num_seqs in config.yaml for the vLLM backend
export DISTLLM_MAX_BATCH_SIZE=4
export DISTLLM_MAX_NUM_SEQS=32

# Step 2: Enable KV cache quantization
# > ⚠️ UNVERIFIED (hygiene pass 2026-11-06) — DISTLLM_KV_CACHE_QUANT_BITS
# > has 0 hits; no such mechanism exists.
export DISTLLM_KV_CACHE_QUANT_BITS=8

# Step 3: Reduce max sequence length
# > ⚠️ UNVERIFIED (hygiene pass 2026-11-06) — DISTLLM_MAX_SEQ_LEN has
# > 0 hits in src/ and config.yaml.
export DISTLLM_MAX_SEQ_LEN=2048

# Step 4: Enable memory defragmentation
# defragmentation config section (DefragmentationSettings,
# src/distllm/config/_cache.py:97); env var
# DISTLLM__DEFRAGMENTATION__ENABLED=true
export DISTLLM_DEFRAG_ENABLED=true

# Step 5: Use model quantization
# > ⚠️ UNVERIFIED (hygiene pass 2026-11-06) — `distllm model load` has no
# > --quantization flag (src/distllm/cli/main.py:475); set
# > config.yaml quantization.method instead (src/distllm/config/_model.py:64).
distllm model load llama-3-70b --quantization int4

# Step 6: Split the model across more nodes
# Reduce layers per node in config.yaml

# Step 7: Kill other GPU processes
nvidia-smi  # Check for other processes
kill -9 <PID>
```

**Prevention:**
- Set `DISTLLM_DEFRAG_ENABLED=true` by default.
> ⚠️ UNVERIFIED (hygiene pass 2026-11-06) — the real form is
> DISTLLM__DEFRAGMENTATION__ENABLED=true (defragmentation config section,
> src/distllm/config/_cache.py:97).
- Use `nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 5` to monitor memory continuously.
- Set `DISTLLM_MAX_SEQ_LEN` to a reasonable limit for your use case.
> ⚠️ UNVERIFIED (hygiene pass 2026-11-06) — DISTLLM_MAX_SEQ_LEN has 0 hits
> in src/ and config.yaml.

---

### 6.2 CUDA driver version mismatch

**Symptom:**
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```
or
```
NVIDIA kernel module has version X.XX.XX but this version of PyTorch was compiled with version Y.YY.YY
```

**Root cause:** The NVIDIA driver version is too old for the CUDA toolkit version used to compile PyTorch, or the CUDA toolkit is not installed.

**Fix:**
```bash
# Step 1: Check driver and CUDA versions
nvidia-smi  # Shows driver version and supported CUDA version
nvcc --version  # Shows installed CUDA toolkit version

# Step 2: Verify compatibility
# Driver must support the CUDA version PyTorch was built with
# See: https://docs.nvidia.com/deploy/cuda-compatibility/

# Step 3: Update driver if needed
# Ubuntu:
sudo apt update && sudo apt install -y nvidia-driver-545
sudo reboot

# Step 4: Or install a PyTorch version matching your driver
pip install torch --index-url https://download.pytorch.org/whl/cu118  # For older drivers
```

**Prevention:** Check compatibility matrix before upgrading PyTorch or the driver. Pin PyTorch CUDA version in `requirements.lock`.

---

### 6.3 Multi-GPU not detected

**Symptom:**
```
WARNING: Only 1 GPU detected, but config specifies 2 nodes with cuda:0 and cuda:1
```
or only one GPU shows up in `nvidia-smi`.

**Root cause:** `CUDA_VISIBLE_DEVICES` is set to a single GPU, NVIDIA Container Toolkit is not installed (Docker), or the GPUs are in exclusive mode and one is already claimed.

**Fix:**
```bash
# Step 1: Verify all GPUs are visible
nvidia-smi -L

# Step 2: Check CUDA_VISIBLE_DEVICES
echo $CUDA_VISIBLE_DEVICES
# If set, ensure it includes all GPU indices: "0,1,2,3"

# Step 3: Check GPU compute mode
nvidia-smi -q | grep "Compute Mode"
# If "Exclusive Process", change to "Default":
sudo nvidia-smi -c DEFAULT

# Step 4: For Docker, ensure NVIDIA Container Toolkit is installed
nvidia-ctk --version
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Step 5: Verify in DistLLM
distllm system doctor
```

---

### 6.4 GPU temperature throttling

**Symptom:** Performance degrades after sustained load. `nvidia-smi` shows `THRM` in the "Pwr:Cap" column.

**Root cause:** The GPU is thermal throttling because temperature exceeds the threshold (usually 83-90°C).

**Fix:**
```bash
# Step 1: Check GPU temperature
nvidia-smi --query-gpu=temperature.gpu --format=csv -l 1

# Step 2: Set a more aggressive fan curve (if supported)
sudo nvidia-smi -pl 250  # Reduce power limit to 250W

# Step 3: Improve airflow
# - Clean dust from GPU heatsinks
# - Increase case fan speed
# - Ensure proper spacing between GPUs in multi-GPU setups
```

---

## 7. Kubernetes Issues

### 7.1 Pod CrashLoopBackOff

**Symptom:**
```bash
kubectl get pods
# NAME                              READY   STATUS             RESTARTS   AGE
# distllm-coordinator-7f8b9-xk2lp   0/1     CrashLoopBackOff   5          10m
```

**Root cause:** The coordinator or worker process is crashing on startup. Common causes: missing config, OOM killed, or bad environment variables.

**Fix:**
```bash
# Step 1: Check pod events
kubectl describe pod distllm-coordinator-7f8b9-xk2lp

# Step 2: Check container logs
kubectl logs distllm-coordinator-7f8b9-xk2lp --previous

# Step 3: If OOMKilled, increase memory limits
kubectl edit deployment distllm-coordinator
# resources:
#   limits:
#     memory: 32Gi   # was 16Gi

# Step 4: If config error, verify the ConfigMap
kubectl get configmap distllm-config -o yaml

# Step 5: If GPU not available, check node GPU capacity
kubectl get nodes -o yaml | grep -A5 nvidia.com/gpu
```

---

### 7.2 Readiness probe failing

**Symptom:**
```
Readiness probe failed: HTTP probe to /ready returned 503
```
Pod is in `Running` but not receiving traffic (0/1 READY).

**Root cause:** The coordinator started but hasn't loaded the model yet, or worker nodes are not connected.

**Fix:**
```bash
# Step 1: Check if the model is still loading
kubectl logs distllm-coordinator-7f8b9-xk2lp | grep -i "model\|loading"

# Step 2: Increase the readiness probe initialDelaySeconds
# In coordinator-deployment.yaml:
# readinessProbe:
#   httpGet:
#     path: /ready
#     port: 8000
#   initialDelaySeconds: 120   # was 5 — model loading can take minutes
#   periodSeconds: 10

# Step 3: Check the /ready endpoint directly
kubectl exec distllm-coordinator-7f8b9-xk2lp -- curl -s http://localhost:8000/ready
```

**Prevention:** Set `initialDelaySeconds` based on model size. A 70B model can take 60-120 seconds to load.

---

### 7.3 Liveness probe killing healthy pods

**Symptom:** Pods are repeatedly restarted despite working correctly. `kubectl describe pod` shows `Liveness probe failed`.

**Root cause:** The liveness probe timeout is too aggressive. During high load, the `/live` endpoint can't respond within the timeout.

**Fix:**
```yaml
# In coordinator-deployment.yaml:
livenessProbe:
  httpGet:
    path: /live
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 30          # Check less frequently
  timeoutSeconds: 10         # Allow more time to respond
  failureThreshold: 5        # Require 5 failures before killing
```

**Prevention:** Use a dedicated liveness endpoint that doesn't depend on downstream health. The `/live` endpoint should only check if the process is responsive, not if the cluster is healthy.

---

### 7.4 GPU resource not schedulable

**Symptom:**
```
0/5 nodes are available: 5 Insufficient nvidia.com/gpu.
```

**Root cause:** No nodes in the cluster have GPU resources available, or the NVIDIA device plugin is not installed.

**Fix:**
```bash
# Step 1: Check if the NVIDIA device plugin is running
kubectl get pods -n kube-system | grep nvidia

# Step 2: If not installed, install it
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.3/nvidia-device-plugin.yml

# Step 3: Verify GPU allocatable on nodes
kubectl get nodes -o json | jq '.items[].status.allocatable["nvidia.com/gpu"]'

# Step 4: Add tolerations for GPU taints
# In your deployment spec:
# tolerations:
#   - key: "nvidia.com/gpu"
#     operator: "Exists"
#     effect: "NoSchedule"
```

---

### 7.5 Pod eviction due to memory pressure

**Symptom:**
```
Pod was evicted due to: OOMKilled (memory limit exceeded)
```
or
```
The node was low on resource: memory. Threshold quantity: 100Mi, available: 80Mi
```

**Root cause:** The container exceeded its memory limit, or the node is under memory pressure and Kubernetes is evicting pods.

**Fix:**
```bash
# Step 1: Check actual memory usage
kubectl top pod distllm-coordinator-7f8b9-xk2lp

# Step 2: Increase memory limits
kubectl set resources deployment distllm-coordinator --limits=memory=32Gi

# Step 3: Set memory requests to 50-70% of limits for better scheduling
# resources:
#   requests:
#     memory: 16Gi
#   limits:
#     memory: 32Gi

# Step 4: Enable memory defragmentation to reduce memory
# In ConfigMap:
# DISTLLM__DEFRAGMENTATION__ENABLED: "true"
```
> ⚠️ UNVERIFIED (hygiene pass 2026-11-06) — the doc's original
> `DISTLLM_KV_CACHE_QUANT_BITS: "8"` has 0 hits in src/ and config.yaml;
> the real form is DISTLLM__DEFRAGMENTATION__ENABLED (defragmentation
> config section, src/distllm/config/_cache.py:97).

---

## 8. Docker Issues

### 8.1 Container can't connect to other containers

**Symptom:**
```
NodeUnreachableError: Node node_1 at node_1:50051 is unreachable
```
or from within a container:
```
ping node_1: Name or service not known
```

**Root cause:** Containers are on different Docker networks, or the service names don't match the container names.

**Fix:**
```bash
# Step 1: Verify all containers are on the same network
docker network ls
docker network inspect distllm-net

# Step 2: If not, connect them
docker network connect distllm-net distllm-node-1

# Step 3: Use service names from docker-compose, not container names
# In docker-compose.yml, the service "node_1" is resolvable as "node_1"
# NOT as "distllm-node-1" (the container_name)

# Step 4: Test DNS resolution between containers
docker exec distllm-coordinator nslookup node_1

# Step 5: Check that ports are exposed
docker port distllm-node-1
```

**Prevention:** Always use the `docker-compose.yml` service names in your DistLLM config. The `container_name` is only for `docker ps` readability.

---

### 8.2 GPU not accessible inside container

**Symptom:**
```
RuntimeError: No CUDA GPUs are available to the container
```
or `nvidia-smi` inside the container returns nothing.

**Root cause:** NVIDIA Container Toolkit is not installed, or the Docker Compose GPU reservation is missing.

**Fix:**
```bash
# Step 1: Verify nvidia-smi works on the host
nvidia-smi

# Step 2: Verify NVIDIA Container Toolkit is installed
nvidia-ctk --version

# Step 3: Test GPU passthrough
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Step 4: If that works, check your docker-compose.yml has the GPU reservation:
# deploy:
#   resources:
#     reservations:
#       devices:
#         - driver: nvidia
#           count: 1
#           capabilities: [gpu]

# Step 5: Restart the Docker daemon if needed
sudo systemctl restart docker
```

---

### 8.3 Model cache not shared between containers

**Symptom:** Every container re-downloads the model (slow startup, high bandwidth usage).

**Root cause:** Each container has its own filesystem. Without a shared volume, the HuggingFace cache is per-container.

**Fix:**
```yaml
# In docker-compose.yml, add a shared volume:
services:
  coordinator:
    volumes:
      - model-cache:/root/.cache/huggingface
  worker:
    volumes:
      - model-cache:/root/.cache/huggingface

volumes:
  model-cache:
```

**Prevention:** Always mount `model-cache` as a shared volume in `docker-compose.yml`.

---

### 8.4 Container networking — port conflicts

**Symptom:**
```
Error starting userland proxy: listen tcp4 0.0.0.0:50050: bind: address already in use
```

**Root cause:** Another process (or another DistLLM instance) is already using the port on the host.

**Fix:**
```bash
# Step 1: Find what's using the port
netstat -tlnp | grep 50050
# or on Windows:
netstat -ano | findstr :50050

# Step 2: Kill the conflicting process, or change the host port mapping
# In docker-compose.yml:
# ports:
#   - "15050:50050"  # Map host port 15050 to container port 50050
```

---

## 9. TLS / Certificate Issues

### 9.1 TLS handshake failure

**Symptom:**
```
ssl.SSLError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed
```
or
```
grpc._channel._InactiveRpcError: <InactiveRpcError terminated with StatusCode.UNAVAILABLE>
```
(with TLS enabled but certificates misconfigured)

**Root cause:** The certificate is expired, self-signed without the CA being trusted, the hostname doesn't match, or the certificate/key files are in the wrong format.

**Fix:**
```bash
# Step 1: Verify the certificate
openssl x509 -in certs/cert.pem -text -noout
# Check:
#   - "Not After" is in the future
#   - "Subject" or "Subject Alternative Name" matches the hostname

# Step 2: Verify the private key matches the certificate
openssl x509 -noout -modulus -in certs/cert.pem | md5sum
openssl rsa -noout -modulus -in certs/key.pem | md5sum
# Both should produce the same hash

# Step 3: For self-signed certs, add the CA to the trust store
# Linux:
sudo cp certs/ca.pem /usr/local/share/ca-certificates/distllm.crt
sudo update-ca-certificates

# Or tell DistLLM to use the CA file:
# config.yaml
# tls:
#   enabled: true
#   cert_file: /app/certs/cert.pem
#   key_file: /app/certs/key.pem
#   ca_cert_file: /app/certs/ca.pem

# Step 4: Test the TLS connection
openssl s_client -connect coordinator:50050 -servername coordinator
```

**Prevention:** Use Let's Encrypt for production (auto-renewal). Set up a monitoring alert for certificate expiration.

---

### 9.2 Certificate expired

**Symptom:**
```
ssl.SSLCertDateError: certificate has expired
```

**Root cause:** The TLS certificate was not renewed before its expiration date.

**Fix:**
```bash
# Step 1: Renew with certbot (Let's Encrypt)
certbot renew

# Step 2: Reload the coordinator to pick up the new certificate
kill -HUP $(pgrep -f "distllm.*coordinator")
# or restart the container/pod

# Step 3: For self-signed certs, regenerate
distllm security cert create my-cluster.example.com --dir certs/
```

**Prevention:** Automate renewal with certbot's systemd timer or cron job.
> ⚠️ UNVERIFIED (hygiene pass 2026-11-06) — no TLS cert expiry metric
> exists; the real metrics are distllm_node_health /
> distllm_coordinator_active_requests (src/distllm/observability/exporter.py).

---

### 9.3 gRPC TLS mutual authentication failure

**Symptom:**
```
grpc.RpcError: StatusCode.UNAVAILABLE: connection error: tls: bad certificate
```

**Root cause:** Mutual TLS (mTLS) is enabled but the client node doesn't present a valid client certificate, or the server doesn't trust the client's CA.

**Fix:**
```bash
# Step 1: Verify mTLS configuration on the coordinator
# config.yaml
# tls:
#   enabled: true
#   cert_file: /app/certs/server-cert.pem
#   key_file: /app/certs/server-key.pem
#   ca_cert_file: /app/certs/ca.pem
#   require_client_cert: true

# Step 2: Ensure each worker has a client certificate signed by the same CA
# certs/
#   ca.pem            # Shared CA
#   server-cert.pem   # Coordinator's server cert
#   server-key.pem    # Coordinator's private key
#   client-cert.pem   # Worker's client cert
#   client-key.pem    # Worker's private key

# Step 3: Verify the client cert is signed by the CA
openssl verify -CAfile certs/ca.pem certs/client-cert.pem
```

---

## 10. Federation Issues

### 10.1 Split-brain between federated clusters

**Symptom:** Two clusters independently accept requests for the same model, leading to inconsistent results. Federation status shows clusters as disconnected despite both being healthy.

**Root cause:** The network link between clusters is down, seed nodes are unreachable, or the gossip protocol has lost quorum.

**Fix:**
```bash
# Step 1: Check federation status
distllm cluster status --federation

# Step 2: Verify seed nodes are reachable from both clusters
curl http://cluster-a-seed:50060/health
curl http://cluster-b-seed:50060/health

# Step 3: Check the cluster's own view of its nodes
distllm cluster status
distllm cluster list-nodes

# Step 4: There is no `distllm cluster federation` command group, and no
# /admin/federation/* endpoint. The /v1/federated/* API that does exist covers
# federated *fine-tuning rounds* (nodes, rounds, stats, versions), not
# cross-cluster routing. To re-join an isolated cluster, re-run join on it:
distllm cluster join --coordinator <seed-host> --port 50051

# Step 5: Split-brain is resolved by draining the stale side rather than by a
# federation command:
distllm cluster drain <node-id>
```

**Prevention:**
- Use at least 3 seed nodes for fault tolerance.
- Monitor `distllm_node_health` (real metric, src/distllm/observability/exporter.py:53) and alert if it drops.
> ⚠️ UNVERIFIED (hygiene pass 2026-11-06) — the doc's original
> `distllm_federation_peer_count` does not exist.
- Set `federation.heartbeat_interval_seconds` to a value appropriate for your network latency.

---

### 10.2 Peer discovery not working

**Symptom:**
```
FederationWarning: No peers discovered after 60s
```
or `distllm cluster status --federation` shows only the local cluster.

**Root cause:** DNS SRV records are not configured, the federation port is blocked, or seed node addresses are wrong.

**Fix:**
```bash
# Step 1: Verify seed node configuration
# config.yaml
# federation:
#   enabled: true
#   seed_nodes:
#     - "cluster-a:50060"
#     - "cluster-b:50060"
#     - "cluster-c:50060"

# Step 2: Test connectivity to seed nodes
nc -zv cluster-a 50060
telnet cluster-a 50060

# Step 3: If using DNS discovery, verify SRV records
dig _grpc._tcp.federation.example.com SRV

# Step 4: Check firewall rules
# Federation port (default: 50060) must be open between clusters
sudo iptables -L -n | grep 50060

# Step 5: Check federation logs
distllm system logs --component federation --tail 100
```

**Prevention:**
- Use a service mesh (Istio, Linkerd) for automatic mTLS and service discovery.
- Pre-configure seed nodes in a shared ConfigMap.

---

### 10.3 Cross-cluster model routing inconsistency

**Symptom:** Requests routed to a remote cluster fail or return different results than local execution.

**Root cause:** The remote cluster is running a different model version, has different quantization settings, or the federation routing table is stale.

**Fix:**
```bash
# Step 1: Verify model versions match across clusters
distllm cluster status

# Step 2: There is no federation routing table endpoint or --federation flag;
# `cluster status` reports the models the connected cluster knows about. If two
# clusters serve different models, that is expected -- route to the right
# coordinator rather than expecting one to know the other's models.

# Step 3: To confirm a specific worker's model, read it from that node:
curl http://<worker-host>:<worker-port>/health

# Step 4: If models are intentionally different, use model-specific routing
# config.yaml
# federation:
#   routing:
#     meta-llama/Llama-3.1-70B: [cluster-a, cluster-b]
#     meta-llama/Llama-3.1-8B: [cluster-c]
```

---

## 11. Diagnostic Commands

### System Health

```bash
# Full system health check
distllm system doctor

# Cluster status (all nodes)
distllm cluster status

# Federation status
distllm cluster status --federation

# Configuration validation
distllm config validate
```

### Logs

```bash
# Recent logs
distllm system logs --tail 100

# Logs for a specific node
distllm system logs --node node_1 --tail 100

# Logs for a specific component
distllm system logs --component coordinator --tail 100

# Follow logs in real-time
distllm system logs --follow
```

### Performance

```bash
# Run benchmarks
distllm benchmark run --model llama-3-8b

# Compare with baseline
distllm benchmark compare --baseline results/baseline.json

# Profile a specific scenario
distllm benchmark profile --scenario chat --duration 60
```

### GPU

```bash
# GPU status
nvidia-smi

# Continuous monitoring (updates every 1 second)
nvidia-smi -l 1

# Query specific fields
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv -l 5
```

### Network

```bash
# gRPC health check
grpcurl -plaintext coordinator:50050 grpc.health.v1.Health/Check

# REST API health
curl http://coordinator:8000/health

# Metrics
curl http://coordinator:8000/metrics

# Network bandwidth test
iperf3 -c node-1 -t 10
```

### Docker

```bash
# Container status
docker ps -a --filter "label=com.docker.compose.project=distllm"

# Container logs
docker logs distllm-coordinator --tail 100

# Container resource usage
docker stats --no-stream
```

### Kubernetes

```bash
# Pod status
kubectl get pods -l app.kubernetes.io/name=distributed-llm

# Pod logs
kubectl logs -l app.kubernetes.io/component=coordinator --tail=100

# Pod events
kubectl describe pod distllm-coordinator-xxxxx

# Resource usage
kubectl top pods -l app.kubernetes.io/name=distributed-llm

# GPU allocation
kubectl get nodes -o json | jq '.items[].status.allocatable["nvidia.com/gpu"]'
```

---

## Getting Help

1. Run `distllm system doctor` for automated diagnostics.
2. Search [GitHub Issues](https://github.com/distributed-llm/distributed-llm/issues).
3. Ask in [GitHub Discussions](https://github.com/distributed-llm/distributed-llm/discussions).
4. Join [Discord](https://discord.gg/distllm).
5. When filing an issue, include:
   - `distllm --version`
   - `distllm system doctor` output
   - `nvidia-smi` output
   - Relevant logs with `DISTLLM_LOG_LEVEL=DEBUG`
   > ⚠️ UNVERIFIED (hygiene pass 2026-11-06) — DISTLLM_LOG_LEVEL has 0 hits
   > in src/ (only docker-compose.gpu.yml / docker-compose.pro.yml pass it
   > through from the shell's LOG_LEVEL).
