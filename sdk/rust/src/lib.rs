//! DistLLM Rust SDK
//!
//! OpenAI-compatible client for distributed LLM inference.
//!
//! # Usage
//!
//! ```rust,no_run
//! use distllm_sdk::{Client, ChatRequest, Message};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = Client::new("http://localhost:8000", "api-key");
//!     let response = client.chat_completion(&ChatRequest {
//!         model: "distributed-llm".to_string(),
//!         messages: vec![Message::user("Hello!")],
//!         ..Default::default()
//!     }).await?;
//!     println!(
//!         "{}",
//!         response.choices[0].message.as_ref().expect("no message").content
//!     );
//!     Ok(())
//! }
//! ```

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("API error ({status}): {message}")]
    Api { status: u16, message: String },
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

/// DistLLM API client.
pub struct Client {
    base_url: String,
    api_key: String,
    http: reqwest::Client,
    max_retries: u32,
}

impl Client {
    /// Create a new client.
    pub fn new(base_url: &str, api_key: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .build()
                .expect("failed to build HTTP client"),
            max_retries: 3,
        }
    }

    /// Create a chat completion.
    pub async fn chat_completion(&self, req: &ChatRequest) -> Result<ChatResponse> {
        self.post("/v1/chat/completions", req).await
    }

    /// Create a text completion.
    pub async fn completion(&self, req: &CompletionRequest) -> Result<CompletionResponse> {
        self.post("/v1/completions", req).await
    }

    /// Create embeddings.
    pub async fn embedding(&self, req: &EmbeddingRequest) -> Result<EmbeddingResponse> {
        self.post("/v1/embeddings", req).await
    }

    /// List available models.
    pub async fn list_models(&self) -> Result<ModelList> {
        self.get("/v1/models").await
    }

    /// Health check.
    pub async fn health(&self) -> Result<HealthResponse> {
        self.get("/health").await
    }

    async fn post<T: Serialize, R: for<'de> Deserialize<'de>>(&self, path: &str, body: &T) -> Result<R> {
        let url = format!("{}{}", self.base_url, path);
        let mut last_err = None;

        for attempt in 0..=self.max_retries {
            let resp = self.http
                .post(&url)
                .header("Content-Type", "application/json")
                .bearer_auth(&self.api_key)
                .json(body)
                .send()
                .await;

            match resp {
                Ok(r) => {
                    if r.status().is_success() {
                        return Ok(r.json().await?);
                    }
                    let status = r.status().as_u16();
                    let msg = r.text().await.unwrap_or_default();
                    if status >= 500 && attempt < self.max_retries {
                        last_err = Some(Error::Api { status, message: msg });
                        tokio::time::sleep(std::time::Duration::from_secs(1 << attempt)).await;
                        continue;
                    }
                    return Err(Error::Api { status, message: msg });
                }
                Err(e) => {
                    if attempt < self.max_retries {
                        last_err = Some(Error::Http(e));
                        tokio::time::sleep(std::time::Duration::from_secs(1 << attempt)).await;
                        continue;
                    }
                    return Err(Error::Http(e));
                }
            }
        }

        Err(last_err.unwrap_or_else(|| Error::Api { status: 0, message: "max retries exceeded".into() }))
    }

    async fn get<R: for<'de> Deserialize<'de>>(&self, path: &str) -> Result<R> {
        let url = format!("{}{}", self.base_url, path);
        let resp = self.http
            .get(&url)
            .bearer_auth(&self.api_key)
            .send()
            .await?;

        if resp.status().is_success() {
            Ok(resp.json().await?)
        } else {
            let status = resp.status().as_u16();
            let msg = resp.text().await.unwrap_or_default();
            Err(Error::Api { status, message: msg })
        }
    }
}

// ── Request Types ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Message {
    pub role: String,
    pub content: String,
    // `name` and `tool_call_id` are optional on the wire in both directions:
    // requests omit them when empty, and responses (ours and every other
    // OpenAI-compatible server) usually do not send them at all. Without
    // `default`, deserializing any normal response fails on the missing field.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub name: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub tool_call_id: String,
}

impl Message {
    pub fn user(content: &str) -> Self {
        Self { role: "user".into(), content: content.into(), ..Default::default() }
    }
    pub fn system(content: &str) -> Self {
        Self { role: "system".into(), content: content.into(), ..Default::default() }
    }
    pub fn assistant(content: &str) -> Self {
        Self { role: "assistant".into(), content: content.into(), ..Default::default() }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    // `serde(default = ...)` only applies when DESERIALIZING. A request built
    // with `..Default::default()` calls Default, which would otherwise leave
    // this empty and send an empty model name to the server. The manual impl
    // below keeps both paths pointing at default_model().
    #[serde(default = "default_model")]
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
}

impl Default for ChatRequest {
    fn default() -> Self {
        Self {
            model: default_model(),
            messages: Vec::new(),
            temperature: None,
            top_p: None,
            max_tokens: None,
            stream: None,
            stop: None,
        }
    }
}

fn default_model() -> String {
    "distributed-llm".into()
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CompletionRequest {
    #[serde(default = "default_model")]
    pub model: String,
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EmbeddingRequest {
    #[serde(default = "default_model")]
    pub model: String,
    pub input: serde_json::Value,
}

// ── Response Types ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub model: String,
    pub created: i64,
    pub choices: Vec<ChatChoice>,
    #[serde(default)]
    pub usage: Option<Usage>,
    #[serde(default)]
    pub generation_time: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    pub index: i32,
    #[serde(default)]
    pub message: Option<Message>,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub model: String,
    pub created: i64,
    pub choices: Vec<CompletionChoice>,
    #[serde(default)]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionChoice {
    pub index: i32,
    pub text: String,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub model: String,
    pub data: Vec<EmbeddingObject>,
    #[serde(default)]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingObject {
    pub index: i32,
    pub embedding: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelList {
    pub data: Vec<ModelInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    #[serde(default)]
    pub owned_by: String,
    #[serde(default)]
    pub created: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub nodes: Option<i32>,
    #[serde(default)]
    pub uptime: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub total_tokens: i32,
    #[serde(default)]
    pub cost_usd: Option<f64>,
    #[serde(default)]
    pub tokens_per_second: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Request serialization ──────────────────────────────────────────
    // These pin the JSON we actually put on the wire. A client that renames
    // a field or starts sending `null` where the server expects an absent key
    // breaks against a real server without any Rust type error.

    #[test]
    fn message_constructors_set_the_right_role() {
        assert_eq!(Message::user("hi").role, "user");
        assert_eq!(Message::system("hi").role, "system");
        assert_eq!(Message::assistant("hi").role, "assistant");
        assert_eq!(Message::user("hi").content, "hi");
    }

    #[test]
    fn empty_optional_message_fields_are_omitted_not_sent_as_empty() {
        let json = serde_json::to_value(Message::user("hello")).unwrap();
        assert_eq!(json["role"], "user");
        assert_eq!(json["content"], "hello");
        // `name` and `tool_call_id` are skip_serializing_if empty; sending
        // them as "" would be a different request shape to the server.
        assert!(json.get("name").is_none(), "empty name must be omitted");
        assert!(json.get("tool_call_id").is_none(), "empty tool_call_id must be omitted");
    }

    #[test]
    fn chat_request_omits_none_sampling_fields() {
        let req = ChatRequest {
            messages: vec![Message::user("ping")],
            ..Default::default()
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "distributed-llm", "default model must be filled in");
        assert!(json.get("temperature").is_none());
        assert!(json.get("top_p").is_none());
        assert!(json.get("max_tokens").is_none());
        assert!(json.get("stop").is_none());
    }

    #[test]
    fn chat_request_sends_sampling_fields_when_set() {
        let req = ChatRequest {
            messages: vec![Message::user("ping")],
            temperature: Some(0.2),
            max_tokens: Some(64),
            ..Default::default()
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["temperature"], 0.2);
        assert_eq!(json["max_tokens"], 64);
    }

    // ── Response deserialization ───────────────────────────────────────
    // Shaped after real OpenAI-compatible payloads, including the optional
    // fields a stricter server may omit entirely.

    #[test]
    fn chat_response_parses_a_minimal_openai_payload() {
        let body = r#"{
            "id": "chatcmpl-1",
            "model": "distributed-llm",
            "created": 1700000000,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hello there"},
                "finish_reason": "stop"
            }]
        }"#;
        let resp: ChatResponse = serde_json::from_str(body).unwrap();
        assert_eq!(resp.id, "chatcmpl-1");
        assert_eq!(resp.choices.len(), 1);
        let msg = resp.choices[0].message.as_ref().expect("message present");
        assert_eq!(msg.content, "hello there");
        assert_eq!(resp.choices[0].finish_reason.as_deref(), Some("stop"));
        // usage omitted entirely must not be an error
        assert!(resp.usage.is_none());
    }

    #[test]
    fn chat_response_parses_usage_when_present() {
        let body = r#"{
            "id": "c", "model": "m", "created": 1,
            "choices": [{"index": 0, "message": {"role":"assistant","content":"x"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}
        }"#;
        let resp: ChatResponse = serde_json::from_str(body).unwrap();
        let usage = resp.usage.expect("usage present");
        assert_eq!(usage.total_tokens, 12);
        assert!(usage.cost_usd.is_none());
    }

    #[test]
    fn completion_response_parses_text_choices() {
        let body = r#"{
            "id": "cmpl-1", "model": "m", "created": 2,
            "choices": [{"index": 0, "text": " done", "finish_reason": "length"}]
        }"#;
        let resp: CompletionResponse = serde_json::from_str(body).unwrap();
        assert_eq!(resp.choices[0].text, " done");
        assert_eq!(resp.choices[0].finish_reason.as_deref(), Some("length"));
    }

    #[test]
    fn embedding_response_parses_float_vectors() {
        let body = r#"{
            "model": "m",
            "data": [{"index": 0, "embedding": [0.1, -0.2, 0.3]}]
        }"#;
        let resp: EmbeddingResponse = serde_json::from_str(body).unwrap();
        assert_eq!(resp.data[0].embedding.len(), 3);
        assert!((resp.data[0].embedding[1] - (-0.2)).abs() < 1e-9);
    }

    #[test]
    fn model_list_parses_with_optional_owner() {
        let body = r#"{"data": [{"id": "llama-3-70b"}, {"id": "m2", "owned_by": "org"}]}"#;
        let list: ModelList = serde_json::from_str(body).unwrap();
        assert_eq!(list.data.len(), 2);
        assert_eq!(list.data[0].id, "llama-3-70b");
        assert_eq!(list.data[0].owned_by, "", "omitted owner defaults to empty");
        assert_eq!(list.data[1].owned_by, "org");
    }

    #[test]
    fn health_response_parses_a_minimal_body() {
        let body = r#"{"status": "ok"}"#;
        let h: HealthResponse = serde_json::from_str(body).unwrap();
        assert_eq!(h.status, "ok");
        assert!(h.model.is_none());
    }

    // ── Error type ─────────────────────────────────────────────────────

    #[test]
    fn api_error_displays_status_and_message() {
        let e = Error::Api { status: 429, message: "rate limited".into() };
        let shown = e.to_string();
        assert!(shown.contains("429"), "status must appear: {shown}");
        assert!(shown.contains("rate limited"), "message must appear: {shown}");
    }

    #[test]
    fn client_strips_a_trailing_slash_from_the_base_url() {
        // A doubled slash (//v1/chat/...) 404s against a strict server.
        let c = Client::new("http://localhost:8000/", "k");
        assert!(!c.base_url.ends_with('/'), "base_url must not end with '/': {}", c.base_url);
    }
}
