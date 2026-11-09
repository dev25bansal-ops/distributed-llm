// Auto-generated API endpoint constants.
package distllm

// Endpoint constants
const EndpointcreateChatCompletion = "/v1/chat/completions"
const EndpointcreateCompletion = "/v1/completions"
const EndpointcreateEmbedding = "/v1/embeddings"
const EndpointlistModels = "/v1/models"
const EndpointgetHealth = "/health"
// Batches take prompts inline and are submitted for background processing;
// the server has no /v1/batches route and no batch-list route.
const EndpointcreateBatch = "/v1/batch/submit"
