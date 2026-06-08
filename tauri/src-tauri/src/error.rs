use serde::Serialize;

/// Structured error type for all Tauri commands.
/// Implements Serialize so Tauri can send it to the frontend as JSON.
#[derive(Debug, thiserror::Error, Serialize)]
pub enum AppError {
    #[error("Python not found. Install Python 3.8+ or activate your virtual environment.")]
    PythonNotFound,

    #[error("distllm package not installed. Run: pip install distributed-llm")]
    PackageNotFound,

    #[error("Cluster is already running")]
    ClusterAlreadyRunning,

    #[error("Already connected to a cluster")]
    AlreadyConnected,

    #[error("Not connected to any cluster")]
    NotConnected,

    #[error("Failed to start process: {0}")]
    SpawnFailed(String),

    #[error("Download already in progress for {0}")]
    DownloadInProgress(String),

    #[error("Download failed: {0}")]
    DownloadFailed(String),

    #[error("Could not reach the coordinator. Is it running?")]
    ApiUnreachable,

    #[error("Coordinator returned invalid data")]
    ApiInvalidResponse,

    #[error("Invalid model name: {0}")]
    InvalidModelName(String),

    #[error("Invalid host: {0}")]
    InvalidHost(String),

    #[error("Invalid URL: {0}")]
    InvalidUrl(String),

    #[error("GPU metrics unavailable: {0}")]
    GpuError(String),

    #[error("CLI error: {0}")]
    CliError(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

// Tauri requires command errors to implement Into<String>
impl From<AppError> for String {
    fn from(err: AppError) -> Self {
        err.to_string()
    }
}

// Backward compatibility: allow converting old ApiError-style strings
impl From<String> for AppError {
    fn from(msg: String) -> Self {
        AppError::Internal(msg)
    }
}
