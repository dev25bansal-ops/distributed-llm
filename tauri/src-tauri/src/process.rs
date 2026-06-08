use std::process::{Command, Stdio};
use std::sync::OnceLock;

use crate::error::AppError;

/// Cached Python interpreter path — resolved once on first use.
static PYTHON_PATH: OnceLock<String> = OnceLock::new();

/// Cached result of distllm package check.
static DISTLLM_AVAILABLE: OnceLock<bool> = OnceLock::new();

/// Find an available Python interpreter on the system (cached).
pub fn find_python() -> &'static str {
    PYTHON_PATH.get_or_init(|| {
        for name in &["python3", "python", "py"] {
            if Command::new(name)
                .arg("--version")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .is_ok()
            {
                return name.to_string();
            }
        }
        "python".to_string()
    })
}

/// Ensure the distllm Python package is installed and accessible (cached).
pub fn ensure_distllm() -> Result<&'static str, AppError> {
    let python = find_python();

    // Check if we already verified distllm is available
    let available = DISTLLM_AVAILABLE.get_or_init(|| {
        Command::new(python)
            .args(["-m", "distllm.cli.main", "--help"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok()
    });

    if *available {
        Ok(python)
    } else {
        Err(AppError::PackageNotFound)
    }
}

/// Validate a model name to prevent argument injection.
pub fn validate_model_name(model: &str) -> Result<(), AppError> {
    if model.is_empty() || model.len() > 256 {
        return Err(AppError::InvalidModelName(
            "must be between 1 and 256 characters".into(),
        ));
    }
    if model.starts_with('-') {
        return Err(AppError::InvalidModelName(
            "must not start with '-'".into(),
        ));
    }
    if !model
        .chars()
        .all(|c| c.is_alphanumeric() || "/-_.".contains(c))
    {
        return Err(AppError::InvalidModelName(
            "contains invalid characters (allowed: alphanumeric, /, -, _, .)".into(),
        ));
    }
    Ok(())
}

/// Validate a host string to prevent argument injection.
pub fn validate_host(host: &str) -> Result<(), AppError> {
    if host.is_empty() || host.len() > 253 {
        return Err(AppError::InvalidHost(
            "must be between 1 and 253 characters".into(),
        ));
    }
    if host.starts_with('-') {
        return Err(AppError::InvalidHost("must not start with '-'".into()));
    }
    if !host
        .chars()
        .all(|c| c.is_alphanumeric() || ".-:".contains(c))
    {
        return Err(AppError::InvalidHost(
            "contains invalid characters (allowed: alphanumeric, ., -, :)".into(),
        ));
    }
    Ok(())
}
