// Package distllm provides a Go client for the DistLLM distributed inference
// API — an OpenAI-compatible surface for chat completions, text completions,
// embeddings, model discovery, health checks and SSE streaming.
//
// # Usage
//
//	client := distllm.NewClient("http://localhost:8000", "api-key")
//	resp, err := client.ChatCompletion(ctx, &distllm.ChatRequest{
//	    Model: "distributed-llm",
//	    Messages: []distllm.Message{
//	        {Role: "user", Content: "Hello!"},
//	    },
//	})
//	if err != nil {
//	    log.Fatal(err)
//	}
//	fmt.Println(resp.Choices[0].Message.Content)
//
// # Retries
//
// The client retries transient failures automatically: network errors and
// HTTP 5xx responses are retried up to Client.MaxRetries times with
// exponential backoff (1s, 2s, 4s, ...). Adjust MaxRetries on the client to
// tune this behavior.
//
// # Streaming
//
// ChatCompletionStream returns a channel of content deltas parsed from the
// server-sent-events stream; the accompanying error channel delivers any
// transport or HTTP error and is closed alongside the delta channel.
//
// This module is published for use with:
//
//	go get github.com/distributed-llm/distllm-sdk-go
package distllm
