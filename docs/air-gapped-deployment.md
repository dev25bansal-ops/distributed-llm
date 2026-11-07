# Air-Gapped Deployment Guide

> **Target audience**: Defense, intelligence, financial services, and other
> organizations that require running DistLLM without any internet connectivity.

## Overview

An air-gapped DistLLM deployment has **no outbound internet access**. All
model weights, dependencies, and container images must be pre-staged. The
cluster runs entirely on premises or in a private network.

## Prerequisites

- **Container registry**: A private registry (Harbor, Nexus, Artifactory, or
  local Docker registry) reachable from the air-gapped network
- **Package mirror**: A private PyPI mirror (e.g., devpi, private pypiserver,
  or artifact repository with PyPI proxy)
- **Model storage**: Local or NFS-backed storage for HuggingFace model weights
- **GPU drivers**: NVIDIA drivers + CUDA toolkit pre-installed on all nodes
  (no internet access for driver download)

## Step 1: Pre-Stage Dependencies (Internet-Connected Build Machine)

On a machine WITH internet access:

```bash
# 1. Build the DistLLM container image
docker build -t distllm:0.4.0 .

# 2. Push to your private registry
docker tag distllm:0.4.0 registry.internal/distllm:0.4.0
docker push registry.internal/distllm:0.4.0

# 3. Export Python dependencies
pip freeze > distllm-requirements.txt

# 4. Download model weights
huggingface-cli download meta-llama/Llama-3.2-7B --local-dir ./models/llama-3.2-7B
huggingface-cli download meta-llama/Llama-3.2-1B --local-dir ./models/llama-3.2-1B

# 5. Export all dependencies to a tarball
tar -czf distllm-airgap-bundle.tar.gz \
    distllm-requirements.txt \
    models/ \
    deploy/ \
    config.yaml \
    docker-entrypoint.sh
```

## Step 2: Transfer to Air-Gapped Network

Transfer the following via approved media (USB drive, DVD, private link):

```
distllm-airgap-bundle.tar.gz     # Dependencies + models + configs
distllm-requirements.txt          # Python package list
models/                           # Model weight directories
```

## Step 3: Deploy on Air-Gapped Network

```bash
# 1. Load container image from private registry
docker pull registry.internal/distllm:0.4.0

# 2. Set up private PyPI mirror with pre-downloaded packages
pip install --no-index --find-links=./packages/ -r distllm-requirements.txt

# 3. Copy model weights to the expected cache location
mkdir -p /data/models
cp -r ./models/* /data/models/

# 4. Disable outbound model hub access (offline mode)
#    (Important: set the following env vars)
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
> ⚠️ UNVERIFIED (hygiene pass 2026-11-06) — `DISTLLM_DISABLE_DISCOVERY` has 0 hits in src/; only mDNS auto-discovery opt-in flag is `--discover` on `distllm cluster join` (src/distllm/cli/main.py:564)
> ⚠️ UNVERIFIED (hygiene pass 2026-11-06) — `DISTLLM_NO_TELEMETRY` has 0 hits in src/; telemetry is opt-in (off by default) via `DISTLLM_TELEMETRY` (src/distllm/core/telemetry.py:238)
> ⚠️ UNVERIFIED (hygiene pass 2026-11-06) — `DISTLLM_HF_OFFLINE` has 0 hits in src/; the HF offline switches are `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` (huggingface_hub/transformers read them)
> ⚠️ UNVERIFIED (hygiene pass 2026-11-06) — `DISTLLM_DISABLE_UPDATES` has 0 hits in src/
> ⚠️ UNVERIFIED (hygiene pass 2026-11-06) — `DISTLLM_DISABLE_ANALYTICS` has 0 hits in src/; telemetry is opt-in (off by default) via `DISTLLM_TELEMETRY` (src/distllm/core/telemetry.py:238)

# 5. Start the coordinator
distllm-coordinator --model /data/models/llama-3.2-7B --port 50050
```

## Kubernetes Deployment (Air-Gapped)

For K8s deployments, use the Helm chart with air-gapped overrides:

```bash
helm install distllm ./integrations/kubernetes/helm/distllm \
    --set image.repository=registry.internal/distllm \
    --set image.tag=0.4.1 \
    --set image.pullPolicy=Always \
    --set env.TRANSFORMERS_OFFLINE=1
```

> ⚠️ UNVERIFIED (hygiene pass 2026-11-06) — `integrations/kubernetes/helm/distllm/templates/deployment.yaml:39-40` hardcodes only `DISTLLM_CONFIG` in the pod env; no template renders `.Values.env`, so `--set env.*` overrides above may not reach the container

## Security Hardening for Air-Gapped

In addition to standard security measures:

1. **No API key generation** — Use pre-configured API keys only
   (`API_KEYS_FILE` env var), never auto-generated keys
> ⚠️ UNVERIFIED (hygiene pass 2026-11-06) — `install_plugin()` is not disabled by any offline env var; installs are gated by `DISTLLM_PLUGIN_INSTALL_ALLOWED` (src/distllm/core/plugin_trust.py:29,39)
3. **Disable remote model downloads** — All models must be pre-staged
4. **Disable telemetry** — telemetry is opt-in (off by default); see
   `DISTLLM_TELEMETRY` in `src/distllm/core/telemetry.py`
5. **Use the hash allowlist** — All plugin files must have entries in
   `distllm_plugin_hashes.txt` (see security documentation)

## Verification Checklist

- [ ] No outbound internet connections from any cluster node
- [ ] All container images pulled from private registry
- [ ] All model weights pre-staged in `/data/models/`
- [ ] `TRANSFORMERS_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` are set
- [ ] Plugin hash allowlist is in place for all loaded plugins
- [ ] No auto-generated API keys — using pre-configured API_KEYS_FILE
- [ ] Docker builds produce no external downloads
- [ ] All dependencies mirrored in private PyPI
