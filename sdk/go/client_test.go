package distllm

// Live-call smoke tests: each test spins an in-process httptest server that
// mimics the OpenAI-compatible DistLLM API surface (/v1/models,
// /v1/chat/completions, /health) with canned responses, then asserts the
// client parses them into typed values.
//
// Run: go test ./...   (from sdk/go)

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

const (
	cannedModelsBody = `{"object":"list","data":[` +
		`{"id":"distributed-llm","object":"model","created":1700000000,"owned_by":"distllm"},` +
		`{"id":"tiny-stories-1m","object":"model","created":1700000001,"owned_by":"distllm"}]}`

	cannedChatBody = `{"id":"chatcmpl-smoke-001","object":"chat.completion","created":1700000100,` +
		`"model":"distributed-llm","choices":[{"index":0,` +
		`"message":{"role":"assistant","content":"Hello from the mock cluster!"},` +
		`"finish_reason":"stop"}],` +
		`"usage":{"prompt_tokens":5,"completion_tokens":8,"total_tokens":13}}`

	cannedHealthBody = `{"status":"ok","model":"distributed-llm","nodes":2,"uptime":1234.5}`
)

// newMockServer starts an httptest server serving canned OpenAI-compatible
// routes and returns a client pointed at it plus the captured requests.
func newMockServer(t *testing.T) (*Client, *[]recordedRequest) {
	t.Helper()
	var mu recordedRequests
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.add(recordedRequest{Method: r.Method, Path: r.URL.Path, Auth: r.Header.Get("Authorization")})
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/v1/models":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(cannedModelsBody))
		case r.Method == http.MethodPost && r.URL.Path == "/v1/chat/completions":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(cannedChatBody))
		case r.Method == http.MethodGet && r.URL.Path == "/health":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(cannedHealthBody))
		default:
			http.Error(w, `{"error":{"message":"no route `+r.URL.Path+`"}}`, http.StatusNotFound)
		}
	}))
	t.Cleanup(srv.Close)
	client := NewClient(srv.URL, "sk-test-key")
	client.MaxRetries = 0 // hermetic: no retry backoff sleeps in tests
	return client, &mu.requests
}

type recordedRequest struct {
	Method string
	Path   string
	Auth   string
}

type recordedRequests struct {
	requests []recordedRequest
}

func (r *recordedRequests) add(req recordedRequest) { r.requests = append(r.requests, req) }

func TestSmokeListModels(t *testing.T) {
	client, reqs := newMockServer(t)

	models, err := client.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels returned error: %v", err)
	}
	if len(models.Data) != 2 {
		t.Fatalf("expected 2 models, got %d", len(models.Data))
	}
	if models.Data[0].ID != "distributed-llm" || models.Data[0].OwnedBy != "distllm" {
		t.Errorf("unexpected first model: %+v", models.Data[0])
	}
	if (*reqs)[0].Path != "/v1/models" {
		t.Errorf("expected GET /v1/models, got %s", (*reqs)[0].Path)
	}
}

func TestSmokeChatCompletion(t *testing.T) {
	client, reqs := newMockServer(t)

	resp, err := client.ChatCompletion(context.Background(), &ChatRequest{
		Model:    "distributed-llm",
		Messages: []Message{{Role: "user", Content: "Hi"}},
	})
	if err != nil {
		t.Fatalf("ChatCompletion returned error: %v", err)
	}
	if resp.ID != "chatcmpl-smoke-001" || resp.Model != "distributed-llm" {
		t.Errorf("unexpected id/model: %s / %s", resp.ID, resp.Model)
	}
	if len(resp.Choices) != 1 {
		t.Fatalf("expected 1 choice, got %d", len(resp.Choices))
	}
	choice := resp.Choices[0]
	if choice.Message == nil || choice.Message.Content != "Hello from the mock cluster!" {
		t.Errorf("unexpected message content: %+v", choice.Message)
	}
	if choice.FinishReason != "stop" {
		t.Errorf("expected finish_reason stop, got %q", choice.FinishReason)
	}
	if resp.Usage == nil || resp.Usage.TotalTokens != 13 {
		t.Errorf("unexpected usage: %+v", resp.Usage)
	}

	req := (*reqs)[0]
	if req.Method != http.MethodPost || req.Path != "/v1/chat/completions" {
		t.Errorf("unexpected request: %s %s", req.Method, req.Path)
	}
	if req.Auth != "Bearer sk-test-key" {
		t.Errorf("missing/incorrect Authorization header: %q", req.Auth)
	}
}

func TestSmokeChatCompletionRequestBody(t *testing.T) {
	var srv *httptest.Server
	var gotBody ChatRequest
	srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Errorf("decode request body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(cannedChatBody))
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "")
	client.MaxRetries = 0

	_, err := client.ChatCompletion(context.Background(), &ChatRequest{
		Model:       "distributed-llm",
		Messages:    []Message{{Role: "user", Content: "Hi"}},
		Temperature: 0.5,
		MaxTokens:   64,
	})
	if err != nil {
		t.Fatalf("ChatCompletion returned error: %v", err)
	}
	if gotBody.Model != "distributed-llm" || gotBody.Temperature != 0.5 || gotBody.MaxTokens != 64 {
		t.Errorf("request body not serialized correctly: %+v", gotBody)
	}
	if len(gotBody.Messages) != 1 || gotBody.Messages[0].Role != "user" || gotBody.Messages[0].Content != "Hi" {
		t.Errorf("messages not serialized correctly: %+v", gotBody.Messages)
	}
}

func TestSmokeHealth(t *testing.T) {
	client, _ := newMockServer(t)

	health, err := client.Health(context.Background())
	if err != nil {
		t.Fatalf("Health returned error: %v", err)
	}
	if health.Status != "ok" || health.Nodes != 2 {
		t.Errorf("unexpected health payload: %+v", health)
	}
}

func TestSmokeChatCompletionStream(t *testing.T) {
	var srv *httptest.Server
	srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body ChatRequest
		_ = json.NewDecoder(r.Body).Decode(&body)
		if !body.Stream {
			t.Errorf("expected stream=true in request body")
		}
		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)
		for _, chunk := range []string{
			"data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hel\"}}]}\n\n",
			"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo!\"}}]}\n\n",
			"data: [DONE]\n\n",
		} {
			_, _ = w.Write([]byte(chunk))
			flusher.Flush()
		}
	}))
	defer srv.Close()

	client := NewClient(srv.URL, "")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	ch, errCh := client.ChatCompletionStream(ctx, &ChatRequest{
		Model:    "distributed-llm",
		Messages: []Message{{Role: "user", Content: "Hi"}},
	})

	var got []string
	for delta := range ch {
		got = append(got, delta)
	}
	select {
	case err := <-errCh:
		if err != nil {
			t.Fatalf("stream error channel: %v", err)
		}
	case <-ctx.Done():
		t.Fatal("timed out waiting for stream to finish")
	}
	want := []string{"Hel", "lo!"}
	if len(got) != len(want) {
		t.Fatalf("expected %v deltas, got %v", want, got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("expected %v deltas, got %v", want, got)
		}
	}
}
