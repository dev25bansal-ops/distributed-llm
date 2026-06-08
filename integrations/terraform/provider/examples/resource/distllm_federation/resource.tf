resource "distllm_federation" "primary" {
  cluster_id          = "us-east-1-cluster"
  listen_port         = 50060
  spillover_enabled   = true
  spillover_threshold = 80.0

  seed_nodes = [
    "10.0.1.10:50060",
    "10.0.1.11:50060",
  ]
}

output "federation_peers" {
  value = distllm_federation.primary.peers
}
