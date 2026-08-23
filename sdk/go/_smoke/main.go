package main

import (
	"context"
	"fmt"
	distllm "github.com/distributed-llm/distllm-sdk-go"
)

func main() {
	ctx := context.Background()
	c := distllm.NewClient("http://127.0.0.1:8000", "live-demo-key-1234567890abcdef")
	if _, err := c.ListModels(ctx); err != nil {
		fmt.Println("ListModels ERR:", err)
	} else {
		fmt.Println("ListModels OK")
	}
	reply, err := c.ChatCompletion(ctx, &distllm.ChatRequest{
		Model:    "roneneldan/TinyStories-1M",
		Messages: []distllm.Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		fmt.Println("Chat ERR:", err)
		return
	}
	fmt.Println("Chat OK, content non-empty:", len(reply.Choices[0].Message.Content) > 0)
}
