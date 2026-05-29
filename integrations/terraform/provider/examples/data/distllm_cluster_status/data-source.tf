data "distllm_cluster_status" "current" {}

output "cluster_healthy" {
  value = data.distllm_cluster_status.current.healthy
}

output "cluster_version" {
  value = data.distllm_cluster_status.current.version
}
