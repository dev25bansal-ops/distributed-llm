use std::env;

use serde::{Deserialize, Serialize};

use crate::error::AppError;
use crate::state::AppState;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaConfig {
    pub host: String,
    pub port: u16,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaModel {
    pub name: String,
    pub size: u64,
    pub digest: String,
    pub modified_at: String,
}

/// Get the Ollama configuration, defaulting to OLLAMA_HOST env var.
#[tauri::command]
pub fn get_ollama_config() -> Result<OllamaConfig, AppError> {
    // Check OLLAMA_HOST env var first
    let env_host = env::var("OLLAMA_HOST").ok();

    if let Some(host_str) = env_host {
        // Parse "http://host:port" or "host:port" format
        let cleaned = host_str
            .trim_start_matches("http://")
            .trim_start_matches("https://");

        let (host, port) = if let Some(colon_pos) = cleaned.rfind(':') {
            let host = &cleaned[..colon_pos];
            let port_str = &cleaned[colon_pos + 1..];
            let port: u16 = port_str.parse().unwrap_or(11434);
            (host.to_string(), port)
        } else {
            (cleaned.to_string(), 11434)
        };

        return Ok(OllamaConfig {
            host,
            port,
            enabled: true,
        });
    }

    Ok(OllamaConfig {
        host: "127.0.0.1".into(),
        port: 11434,
        enabled: false,
    })
}

/// Check if an Ollama server is reachable.
#[tauri::command]
pub async fn check_ollama(config: OllamaConfig) -> Result<bool, AppError> {
    let url = format!("http://{}:{}/api/tags", config.host, config.port);

    let agent = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(3))
        .build();

    match agent.get(&url).call() {
        Ok(_) => Ok(true),
        Err(_) => Ok(false),
    }
}

/// List models from an Ollama server.
#[tauri::command]
pub async fn list_ollama_models(config: OllamaConfig) -> Result<Vec<OllamaModel>, AppError> {
    let url = format!("http://{}:{}/api/tags", config.host, config.port);

    let agent = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(5))
        .build();

    let resp = agent
        .get(&url)
        .call()
        .map_err(|e| AppError::ApiUnreachable)?;

    let body: serde_json::Value = resp
        .into_json()
        .map_err(|_| AppError::ApiInvalidResponse)?;

    let models = body
        .get("models")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| {
                    Some(OllamaModel {
                        name: m.get("name")?.as_str()?.to_string(),
                        size: m.get("size")?.as_u64()?,
                        digest: m.get("digest")?.as_str()?.to_string(),
                        modified_at: m.get("modified_at")?.as_str()?.to_string(),
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    Ok(models)
}

/// Generate an OpenAI-compatible chat completion via Ollama.
/// This proxies the request in the format Ollama expects.
#[tauri::command]
pub async fn ollama_chat(
    config: OllamaConfig,
    model: String,
    messages: Vec<serde_json::Value>,
    stream: bool,
) -> Result<serde_json::Value, AppError> {
    let url = format!("http://{}:{}/api/chat", config.host, config.port);

    let body = serde_json::json!({
        "model": model,
        "messages": messages,
        "stream": stream,
    });

    let agent = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(120))
        .build();

    let resp = agent
        .post(&url)
        .set("Content-Type", "application/json")
        .send_json(body)
        .map_err(|e| AppError::ApiUnreachable)?;

    let result: serde_json::Value = resp
        .into_json()
        .map_err(|_| AppError::ApiInvalidResponse)?;

    Ok(result)
}

/// Pull a model from Ollama registry.
#[tauri::command]
pub async fn pull_ollama_model(
    config: OllamaConfig,
    model_name: String,
) -> Result<String, AppError> {
    let url = format!("http://{}:{}/api/pull", config.host, config.port);

    let body = serde_json::json!({
        "name": model_name,
    });

    let agent = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(300))
        .build();

    let resp = agent
        .post(&url)
        .set("Content-Type", "application/json")
        .send_json(body)
        .map_err(|e| AppError::ApiUnreachable)?;

    let result: serde_json::Value = resp
        .into_json()
        .map_err(|_| AppError::ApiInvalidResponse)?;

    Ok(result.to_string())
}
