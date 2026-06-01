# Cloud Marketplace Listings

Guides for deploying DistLLM via cloud marketplaces.

---

## AWS Marketplace

### AMI Listing

```yaml
# deploy/aws/marketplace.yaml
name: DistLLM Inference Server
version: 0.4.0
description: |
  Distributed LLM inference with pipeline parallelism.
  Run 70B+ models across multiple GPUs with OpenAI-compatible API.
  
  Features:
  - Pipeline parallelism across multiple GPUs
  - Automatic quantization (FP16/INT8/INT4)
  - OpenAI-compatible API
  - Built-in monitoring and observability
  
instance_types:
  - p4d.24xlarge    # 8x A100 80GB
  - p4de.24xlarge   # 8x A100 80GB NVLink
  - g5.12xlarge     # 4x A10G 24GB
  - g5.48xlarge     # 8x A10G 24GB

ami_id: ami-xxxxxxxxx  # Built via Packer
region: us-east-1
```

### Terraform Module

```hcl
# deploy/aws/terraform/main.tf
module "distllm" {
  source  = "distributed-llm/distributed-llm/aws"
  version = "0.4.0"

  instance_type = "p4d.24xlarge"
  model_name    = "meta-llama/Llama-3.1-70B"
  nodes         = 2
  
  enable_monitoring = true
  enable_tls        = true
}
```

---

## GCP Marketplace

### GKE Deployment

```yaml
# deploy/gcp/marketplace.yaml
name: DistLLM
version: 0.4.0
description: Distributed LLM inference on GKE
publisher: distllm

gke_config:
  min_nodes: 2
  max_nodes: 8
  machine_type: a2-highgpu-1g
  accelerator:
    type: nvidia-tesla-a100
    count: 1

helm_chart: distllm/distllm
```

### Deployment Command

```bash
gcloud container clusters create distllm-cluster \
  --machine-type=a2-highgpu-1g \
  --accelerator=type=nvidia-tesla-a100,count=4 \
  --num-nodes=2

helm install distllm distllm/distllm \
  --set model.name=meta-llama/Llama-3.1-70B \
  --set coordinator.replicas=2
```

---

## Azure Marketplace

### Azure Container Apps

```yaml
# deploy/azure/marketplace.yaml
name: DistLLM
version: 0.4.0
description: Distributed LLM inference on Azure
publisher: distllm

container_app:
  environment: distllm-env
  gpu:
    type: Standard_NC24ads_A100_v4
    count: 4
  
  scaling:
    min_replicas: 1
    max_replicas: 8
```

### Deployment Command

```bash
az containerapp env create \
  --name distllm-env \
  --resource-group distllm-rg \
  --location eastus

az containerapp create \
  --name distllm \
  --environment distllm-env \
  --image distllm:latest \
  --cpu 16 --memory 64Gi \
  --min-replicas 1 --max-replicas 4
```

---

## Pricing Comparison

| Provider | Instance | GPUs | Cost/Hour | DistLLM Throughput |
|----------|----------|------|-----------|-------------------|
| AWS | p4d.24xlarge | 8x A100 | $32.77 | ~50 tok/s (70B) |
| GCP | a2-highgpu-8g | 8x A100 | $29.39 | ~50 tok/s (70B) |
| Azure | NC96ads_A100_v4 | 4x A100 | $27.20 | ~25 tok/s (70B) |

*Prices as of 2026. DistLLM saves 40-70% vs cloud API pricing.*
