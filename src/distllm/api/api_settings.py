"""Consolidated API settings via Pydantic BaseSettings.

Replaces 20+ scattered ``os.environ.get()`` calls across 15+ files
with a single typed settings class loaded once.

Usage::

    from distllm.api.api_settings import api_settings

    # Access any setting as a typed attribute:
    cors_origins = api_settings.cors_origins
    rate_limit = api_settings.rate_limit_requests
    enabled = api_settings.prompt_injection_enabled
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseSettings, Field


class APISettings(BaseSettings):
    """Pydantic-based settings for the DistLLM API layer.

    All settings are loaded from environment variables with typed
    defaults.  To override, set the corresponding env var before
    starting the server.
    """

    # ── CORS ──────────────────────────────────────────────────────────

    cors_origins: str = Field(
        "http://localhost:3000,http://localhost:8080",
        description="Comma-separated CORS allowed origins",
        env="DISTLLM_CORS_ORIGINS",
    )
    cors_allow_all: bool = Field(
        False,
        description="Allow all CORS origins (DANGEROUS — dev only)",
        env="DISTLLM_CORS_ALLOW_ALL",
    )

    # ── TLS / Security ────────────────────────────────────────────────

    tls_enabled: bool = Field(False, env="DISTLLM_TLS_ENABLED")
    trust_proxy_tls: bool = Field(False, env="DISTLLM_TRUST_PROXY_TLS")
    trust_proxy_headers: bool = Field(False, env="DISTLLM_TRUST_PROXY_HEADERS")

    # ── Rate Limiting ─────────────────────────────────────────────────

    rate_limit_requests: int = Field(
        1000, ge=0,
        description="Max requests per IP per 60s window (0=disabled)",
        env="DISTLLM_RATE_LIMIT_REQUESTS",
    )
    rate_limit_auth_attempts: int = Field(
        30, ge=1,
        description="Max failed auth attempts per IP per 60s",
        env="DISTLLM_AUTH_RATE_LIMIT_ATTEMPTS",
    )

    # ── Prompt Injection ──────────────────────────────────────────────

    prompt_injection_enabled: bool = Field(True, env="DISTLLM_INJECTION_ENABLED")
    prompt_injection_block_threshold: float = Field(0.9, ge=0.0, le=1.0, env="DISTLLM_INJECTION_BLOCK_THRESHOLD")
    prompt_injection_sanitize_threshold: float = Field(0.7, ge=0.0, le=1.0, env="DISTLLM_INJECTION_SANITIZE_THRESHOLD")
    prompt_injection_flag_threshold: float = Field(0.4, ge=0.0, le=1.0, env="DISTLLM_INJECTION_FLAG_THRESHOLD")
    prompt_injection_model: str = Field("", env="DISTLLM_INJECTION_MODEL")
    prompt_injection_audit_log: str = Field("", env="DISTLLM_INJECTION_AUDIT_LOG")

    # ── Quota / Usage ──────────────────────────────────────────────────

    quota_enabled: bool = Field(False, env="DISTLLM_QUOTA_ENABLED")
    usage_db_path: str = Field(".usage.db", env="DISTLLM_USAGE_DB")

    # ── Observability ─────────────────────────────────────────────────

    trace_sample_rate: float = Field(0.1, ge=0.0, le=1.0, env="DISTLLM_TRACE_SAMPLE_RATE")
    enable_docs: bool = Field(False, env="DISTLLM_ENABLE_DOCS")
    otlp_endpoint: str = Field("http://localhost:4318/v1/traces", env="DISTLLM_OTLP_ENDPOINT")

    # ── Config / Plugin ────────────────────────────────────────────────

    config_path: str = Field("config.yaml", env="DISTLLM_CONFIG")
    verify_plugins: bool = Field(False, env="DISTLLM_VERIFY_PLUGINS")

    # ── SSO ────────────────────────────────────────────────────────────

    sso_provider: str = Field("", env="DISTLLM_SSO_PROVIDER")
    sso_client_id: str = Field("", env="DISTLLM_SSO_CLIENT_ID")
    sso_client_secret: str = Field("", env="DISTLLM_SSO_CLIENT_SECRET")
    sso_authority: str = Field("", env="DISTLLM_SSO_AUTHORITY")
    sso_metadata_url: str = Field("", env="DISTLLM_SSO_METADATA_URL")
    sso_callback_url: str = Field("", env="DISTLLM_SSO_CALLBACK_URL")
    sso_jwks_url: str = Field("", env="DISTLLM_SSO_JWKS_URL")

    # ── Feature Flags ─────────────────────────────────────────────────

    enable_debug_routes: bool = Field(False, env="DISTLLM_ENABLE_DEBUG_ROUTES")
    dev_mode: bool = Field(False, env="DISTLLM_DEV_MODE")
    api_key: str = Field("", env="API_KEY")
    cluster_key: str = Field("", env="DISTLLM_CLUSTER_KEY")
    ha_shared_secret: str = Field("", env="DISTLLM_HA_SECRET")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global singleton
_settings: Optional[APISettings] = None


def get_api_settings() -> APISettings:
    global _settings
    if _settings is None:
        _settings = APISettings()
    return _settings


# Convenience accessor
api_settings = get_api_settings()
