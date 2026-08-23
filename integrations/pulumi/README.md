# Pulumi Provider for DistLLM

Manage [DistLLM](https://github.com/distributed-llm/distributed-llm) clusters through Pulumi infrastructure-as-code.

## Requirements

- [Pulumi](https://www.pulumi.com/) >= 3.0
- [Node.js](https://nodejs.org/) >= 18
- A running DistLLM cluster (API endpoint)

## Installation

```bash
npm install @distllm/pulumi
```

## Quick Start

```typescript
import {
  DistLLMProvider,
  DistLLMModel,
  DistLLMDeployment,
  getClusterStatus,
} from "@distllm/pulumi";

// Create a provider pointing at your DistLLM cluster
const provider = new DistLLMProvider("cluster", {
  endpoint: "http://localhost:8000",   // or use DISTLLM_ENDPOINT env var
  apiKey: process.env.DISTLLM_API_KEY, // optional
});

// Load a model
const model = new DistLLMModel("llama", {
  name: "meta-llama/Llama-2-7b",
  params: {
    dtype: "float16",
    maxMemory: 16384,
    numGpus: 2,
  },
}, { provider });

// Deploy the model with replicas
const deployment = new DistLLMDeployment("llama-prod", {
  modelName: model.name,
  replicas: 2,
  gpuMemory: 16384,
  labels: {
    environment: "production",
    team: "inference",
  },
}, { provider });

// Query cluster status
const status = getClusterStatus({ provider });
export const clusterHealthy = status.healthy;
export const modelStatus = model.status;
```

## Resources

| Resource | Description | API Endpoint |
|----------|-------------|--------------|
| `DistLLMModel` | Load/unload models on the cluster | `POST /v1/models` |
| `DistLLMDeployment` | Deploy models with replica config | `POST /v1/deployments` |
| `DistLLMNode` | Register/manage worker nodes | `POST /v1/nodes` |
| `DistLLMFederation` | Connect clusters into a federation | `POST /v1/federation` |

## Data Sources

| Data Source | Description | API Endpoint |
|-------------|-------------|--------------|
| `getClusterStatus()` | Cluster health, node count, resource utilisation | `GET /v1/cluster/status` |
| `getModels()` | List all loaded models | `GET /v1/models` |
| `getNodes()` | List all registered nodes | `GET /v1/nodes` |

## Provider Configuration

```typescript
const provider = new DistLLMProvider("cluster", {
  endpoint: "http://localhost:8000",   // DistLLM API URL
  apiKey: "your-api-key",              // optional
  timeout: 120,                        // request timeout in seconds
});
```

Configuration can also be set via environment variables:

- `DISTLLM_ENDPOINT` — API endpoint (default: `http://localhost:8000`)
- `DISTLLM_API_KEY` — API key

## Development

```bash
cd integrations/pulumi
npm install
npm run build
```

## API Reference

### DistLLMModel

```typescript
new DistLLMModel(name: string, args: ModelArgs, opts?: CustomResourceOptions)
```

**Inputs:**

| Property | Type | Description |
|----------|------|-------------|
| `name` | `string` | Model name (e.g. `meta-llama/Llama-2-7b`) |
| `params` | `ModelParams` | Optional model configuration |

**Outputs:**

| Property | Type | Description |
|----------|------|-------------|
| `status` | `string` | `loading`, `loaded`, `unloading`, `error` |
| `params` | `ModelParams` | Applied configuration |
| `loadedAt` | `string` | ISO timestamp when loaded |

### DistLLMDeployment

```typescript
new DistLLMDeployment(name: string, args: DeploymentArgs, opts?: CustomResourceOptions)
```

**Inputs:**

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `modelName` | `string` | — | Model to deploy |
| `replicas` | `number` | `1` | Number of replicas |
| `gpuMemory` | `number` | — | GPU memory per replica (MB) |
| `labels` | `Record<string, string>` | — | Resource labels |
| `env` | `Record<string, string>` | — | Environment variables |

**Outputs:**

| Property | Type | Description |
|----------|------|-------------|
| `status` | `string` | Deployment status |
| `endpoint` | `string` | Inference endpoint URL |
| `replicas` | `number` | Current replica count |

### DistLLMNode

```typescript
new DistLLMNode(name: string, args: NodeArgs, opts?: CustomResourceOptions)
```

**Inputs:**

| Property | Type | Description |
|----------|------|-------------|
| `nodeId` | `string` | Unique node identifier |
| `labels` | `Record<string, string>` | Node labels |
| `resources` | `NodeResources` | Resource capacity |

**Outputs:** `nodeId`, `labels`, `resources`, `status`, `joinedAt`

### DistLLMFederation

```typescript
new DistLLMFederation(name: string, args: FederationArgs, opts?: CustomResourceOptions)
```

**Inputs:**

| Property | Type | Description |
|----------|------|-------------|
| `name` | `string` | Federation name |
| `peers` | `FederationPeer[]` | Peer cluster list |
| `config` | `FederationConfig` | Routing and heartbeat config |

## License

Apache-2.0
