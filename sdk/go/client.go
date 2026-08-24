// HTTP client for the DistLLM API. Package documentation lives in doc.go.
//
// Usage:
//
//	client := distllm.NewClient("http://localhost:8000", "api-key")
//	resp, err := client.ChatCompletion(ctx, &distllm.ChatRequest{
//	    Model: "distributed-llm",
//	    Messages: []distllm.Message{
//	        {Role: "user", Content: "Hello!"},
//	    },
//	})
//	fmt.Println(resp.Choices[0].Message.Content)
package distllm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"strconv"
	"strings"
	"time"
)

const (
	defaultBaseURL = "http://localhost:8000"
	defaultTimeout = 120 * time.Second
	defaultRetries = 3
)

// Retry backoff policy, mirroring the Python SDK's _compute_delay
// (sdk/src/distllm_sdk/client.py): honor a server-provided Retry-After header
// verbatim with a floor (so a tiny/zero value cannot cause a tight retry
// loop); otherwise exponential backoff capped at retryMaxDelaySeconds. Both
// paths are multiplied by jitter in [0.5, 1.0).
const (
	retryInitialDelaySeconds = 1.0
	retryMaxDelaySeconds     = 60.0
	retryAfterFloorSeconds   = 0.5 // matches distllm_sdk.constants.RETRY_AFTER_FLOOR
)

// retryDelay computes how long to wait before the next attempt. A non-nil
// headers value carrying a numeric Retry-After takes precedence; anything
// unparseable (non-numeric, NaN, Inf, negative) falls through to plain
// exponential backoff.
func retryDelay(attempt int, headers http.Header) time.Duration {
	jitter := 0.5 + rand.Float64()*0.5
	if headers != nil {
		raw := strings.TrimSpace(headers.Get("Retry-After"))
		if raw != "" {
			if secs, err := strconv.ParseFloat(raw, 64); err == nil &&
				!math.IsNaN(secs) && !math.IsInf(secs, 0) && secs >= 0 {
				wait := math.Max(secs, retryAfterFloorSeconds) * jitter
				return time.Duration(wait * float64(time.Second))
			}
		}
	}
	secs := retryInitialDelaySeconds * math.Pow(2, float64(attempt))
	if secs > retryMaxDelaySeconds {
		secs = retryMaxDelaySeconds
	}
	return time.Duration(secs * jitter * float64(time.Second))
}

// sleepCtx waits for d, returning early with the context's error if ctx is
// cancelled while backing off (retry sleeps must not ignore cancellation).
func sleepCtx(ctx context.Context, d time.Duration) error {
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

// Client is the DistLLM API client.
type Client struct {
	BaseURL    string
	APIKey     string
	HTTPClient *http.Client
	MaxRetries int
}

// NewClient creates a new DistLLM client.
func NewClient(baseURL, apiKey string) *Client {
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	return &Client{
		BaseURL: strings.TrimRight(baseURL, "/"),
		APIKey:  apiKey,
		HTTPClient: &http.Client{
			Timeout: defaultTimeout,
		},
		MaxRetries: defaultRetries,
	}
}

// Message represents a chat message.
type Message struct {
	Role       string `json:"role"`
	Content    string `json:"content"`
	Name       string `json:"name,omitempty"`
	ToolCallID string `json:"tool_call_id,omitempty"`
}

// ChatRequest is the request body for chat completions.
type ChatRequest struct {
	Model          string          `json:"model"`
	Messages       []Message       `json:"messages"`
	Temperature    float64         `json:"temperature,omitempty"`
	TopP           float64         `json:"top_p,omitempty"`
	MaxTokens      int             `json:"max_tokens,omitempty"`
	Stream         bool            `json:"stream,omitempty"`
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`
	Adapter        string          `json:"adapter,omitempty"`
	Stop           []string        `json:"stop,omitempty"`
}

// ResponseFormat specifies the output format.
type ResponseFormat struct {
	Type   string          `json:"type"`
	Schema json.RawMessage `json:"schema,omitempty"`
}

// ChatResponse is the response from chat completions.
type ChatResponse struct {
	ID             string       `json:"id"`
	Model          string       `json:"model"`
	Created        int64        `json:"created"`
	Choices        []ChatChoice `json:"choices"`
	Usage          *Usage       `json:"usage,omitempty"`
	GenerationTime float64      `json:"generation_time,omitempty"`
}

// ChatChoice represents a single choice in the response.
type ChatChoice struct {
	Index        int      `json:"index"`
	Message      *Message `json:"message,omitempty"`
	FinishReason string   `json:"finish_reason,omitempty"`
}

// Usage represents token usage information.
type Usage struct {
	PromptTokens     int     `json:"prompt_tokens"`
	CompletionTokens int     `json:"completion_tokens"`
	TotalTokens      int     `json:"total_tokens"`
	CostUSD          float64 `json:"cost_usd,omitempty"`
	TokensPerSecond  float64 `json:"tokens_per_second,omitempty"`
}

// CompletionRequest is the request body for text completions.
type CompletionRequest struct {
	Model       string   `json:"model"`
	Prompt      string   `json:"prompt"`
	Temperature float64  `json:"temperature,omitempty"`
	TopP        float64  `json:"top_p,omitempty"`
	MaxTokens   int      `json:"max_tokens,omitempty"`
	Stream      bool     `json:"stream,omitempty"`
	Stop        []string `json:"stop,omitempty"`
}

// CompletionResponse is the response from text completions.
type CompletionResponse struct {
	ID      string             `json:"id"`
	Model   string             `json:"model"`
	Created int64              `json:"created"`
	Choices []CompletionChoice `json:"choices"`
	Usage   *Usage             `json:"usage,omitempty"`
}

// CompletionChoice represents a single choice.
type CompletionChoice struct {
	Index        int    `json:"index"`
	Text         string `json:"text"`
	FinishReason string `json:"finish_reason,omitempty"`
}

// EmbeddingRequest is the request body for embeddings.
type EmbeddingRequest struct {
	Model string      `json:"model"`
	Input interface{} `json:"input"`
}

// EmbeddingResponse is the response from embeddings.
type EmbeddingResponse struct {
	Model string            `json:"model"`
	Data  []EmbeddingObject `json:"data"`
	Usage *Usage            `json:"usage,omitempty"`
}

// EmbeddingObject represents a single embedding.
type EmbeddingObject struct {
	Index     int       `json:"index"`
	Embedding []float64 `json:"embedding"`
}

// ModelList is the response from listing models.
type ModelList struct {
	Data []ModelInfo `json:"data"`
}

// ModelInfo represents a model.
type ModelInfo struct {
	ID      string `json:"id"`
	OwnedBy string `json:"owned_by"`
	Created int64  `json:"created"`
}

// HealthResponse is the response from the health endpoint.
type HealthResponse struct {
	Status string  `json:"status"`
	Model  string  `json:"model,omitempty"`
	Nodes  int     `json:"nodes,omitempty"`
	Uptime float64 `json:"uptime,omitempty"`
}

// APIError represents an API error response.
type APIError struct {
	StatusCode int    `json:"-"`
	Message    string `json:"message"`
	Type       string `json:"type"`
	Code       string `json:"code,omitempty"`
}

func (e *APIError) Error() string {
	return fmt.Sprintf("distllm: %s (status %d)", e.Message, e.StatusCode)
}

// ChatCompletion creates a chat completion.
func (c *Client) ChatCompletion(ctx context.Context, req *ChatRequest) (*ChatResponse, error) {
	var resp ChatResponse
	if err := c.do(ctx, "POST", "/v1/chat/completions", req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// Completion creates a text completion.
func (c *Client) Completion(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error) {
	var resp CompletionResponse
	if err := c.do(ctx, "POST", "/v1/completions", req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// Embedding creates embeddings.
func (c *Client) Embedding(ctx context.Context, req *EmbeddingRequest) (*EmbeddingResponse, error) {
	var resp EmbeddingResponse
	if err := c.do(ctx, "POST", "/v1/embeddings", req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// ListModels lists available models.
func (c *Client) ListModels(ctx context.Context) (*ModelList, error) {
	var resp ModelList
	if err := c.do(ctx, "GET", "/v1/models", nil, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// Health checks the API health.
func (c *Client) Health(ctx context.Context) (*HealthResponse, error) {
	var resp HealthResponse
	if err := c.do(ctx, "GET", "/health", nil, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// ChatCompletionStream streams a chat completion.
func (c *Client) ChatCompletionStream(ctx context.Context, req *ChatRequest) (<-chan string, <-chan error) {
	req.Stream = true
	ch := make(chan string, 64)
	errCh := make(chan error, 1)

	go func() {
		defer close(ch)
		defer close(errCh)

		body, err := json.Marshal(req)
		if err != nil {
			errCh <- err
			return
		}

		httpReq, err := http.NewRequestWithContext(ctx, "POST", c.BaseURL+"/v1/chat/completions", bytes.NewReader(body))
		if err != nil {
			errCh <- err
			return
		}
		c.setHeaders(httpReq)

		resp, err := c.HTTPClient.Do(httpReq)
		if err != nil {
			errCh <- err
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			errCh <- &APIError{StatusCode: resp.StatusCode, Message: fmt.Sprintf("HTTP %d", resp.StatusCode)}
			return
		}

		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Text()
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			data := strings.TrimPrefix(line, "data: ")
			if data == "[DONE]" {
				return
			}
			var chunk struct {
				Choices []struct {
					Delta struct {
						Content string `json:"content"`
					} `json:"delta"`
				} `json:"choices"`
			}
			if err := json.Unmarshal([]byte(data), &chunk); err != nil {
				continue
			}
			if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
				ch <- chunk.Choices[0].Delta.Content
			}
		}
		errCh <- scanner.Err()
	}()

	return ch, errCh
}

func (c *Client) do(ctx context.Context, method, path string, body interface{}, result interface{}) error {
	// Buffer the serialized payload once and rebuild a fresh *bytes.Reader for
	// every attempt: http.Client.Do consumes the request body, so reusing a
	// single reader across retries would send an EMPTY body on attempt 2+.
	var payload []byte
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return fmt.Errorf("distllm: marshal request: %w", err)
		}
		payload = data
	}

	var lastErr error
	for attempt := 0; attempt <= c.MaxRetries; attempt++ {
		req, err := http.NewRequestWithContext(ctx, method, c.BaseURL+path, bytes.NewReader(payload))
		if err != nil {
			return err
		}
		c.setHeaders(req)

		resp, err := c.HTTPClient.Do(req)
		if err != nil {
			lastErr = err
			if attempt < c.MaxRetries {
				if sleepCtx(ctx, retryDelay(attempt, nil)) != nil {
					return ctx.Err()
				}
				continue
			}
			return err
		}

		// Retryable statuses mirror the Python SDK's policy minus 429 (this
		// client treats all 4xx as terminal client errors): 500/502/503/504.
		retryable := resp.StatusCode == http.StatusInternalServerError ||
			resp.StatusCode == http.StatusBadGateway ||
			resp.StatusCode == http.StatusServiceUnavailable ||
			resp.StatusCode == http.StatusGatewayTimeout
		if retryable && attempt < c.MaxRetries {
			hdrs := resp.Header
			io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
			if sleepCtx(ctx, retryDelay(attempt, hdrs)) != nil {
				return ctx.Err()
			}
			continue
		}
		defer resp.Body.Close()

		if resp.StatusCode >= 400 {
			var apiErr APIError
			json.NewDecoder(resp.Body).Decode(&apiErr)
			apiErr.StatusCode = resp.StatusCode
			return &apiErr
		}

		return json.NewDecoder(resp.Body).Decode(result)
	}

	return lastErr
}

func (c *Client) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	if c.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.APIKey)
	}
}
