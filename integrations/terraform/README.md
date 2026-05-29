# Terraform Provider for DistLLM

Manage DistLLM models and query cluster status via Terraform.

## Requirements

- Terraform >= 1.0
- Go >= 1.21 (for building)

## Installation

### From Source

```bash
cd integrations/terraform/provider
make build
make install
```

### Configure .terraformrc

```hcl
provider_installation {
  filesystem_mirror {
    path = "~/.terraform.d/plugins"
  }
}
```

## Usage

### Provider Configuration

```hcl
provider "distllm" {
  endpoint = "http://localhost:8000"
  api_key  = ""
  timeout  = 120
}
```

### Data Source: Cluster Status

```hcl
data "distllm_cluster_status" "current" {}

output "healthy" {
  value = data.distllm_cluster_status.current.healthy
}
```

### Resource: Load Model

```hcl
resource "distllm_model" "llama" {
  name = "meta-llama/Llama-2-7b"
}
```

Remove model with `terraform destroy` or `terraform apply -destroy`.

## Development

```bash
cd integrations/terraform/provider
go build -o terraform-provider-distllm
```

## Resources

- `distllm_model` — Load/unload models on the DistLLM cluster

## Data Sources

- `distllm_cluster_status` — Query cluster health and worker status
