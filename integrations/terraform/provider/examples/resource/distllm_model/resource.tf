resource "distllm_model" "llama" {
  name = "meta-llama/Llama-2-7b"
}

output "model_status" {
  value = distllm_model.llama.status
}

output "model_loaded" {
  value = distllm_model.llama.loaded
}
