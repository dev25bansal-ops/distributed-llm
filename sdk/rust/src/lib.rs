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
//!     let content = response.choices[0].message.as_ref().map(|m| &*m.content);
//!     println!("{:?}", content);
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
    #[serde(skip_serializing_if = "String::is_empty")]
    pub name: String,
    #[serde(skip_serializing_if = "String::is_empty")]
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

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChatRequest {
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
