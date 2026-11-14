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

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class APISettings(BaseSettings):
    """Pydantic-based settings for the DistLLM API layer.

    All settings are loaded from environment variables with typed
    defaults.  To override, set the corresponding env var before
    starting the server.
    """

    # pydantic-settings v2 ignores the deprecated ``env=`` keyword on
    # ``Field``: without a config the class read no environment variable
    # at all (DISTLLM_QUOTA_ENABLED, DISTLLM_RATE_LIMIT_REQUESTS, ... all
    # silently stayed at their defaults).  ``env_prefix`` supplies the
    # common ``DISTLLM_`` prefix; the one exception below uses an alias.
    model_config = SettingsConfigDict(
        env_prefix="DISTLLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── CORS ──────────────────────────────────────────────────────────

    cors_origins: str = Field(
        "http://localhost:3000,http://localhost:8080",
        description="Comma-separated CORS allowed origins")
    cors_allow_all: bool = Field(
        False,
        description="Allow all CORS origins (DANGEROUS — dev only)",
    )

    # ── TLS / Security ────────────────────────────────────────────────

    tls_enabled: bool = Field(False)
    trust_proxy_tls: bool = Field(False)
    trust_proxy_headers: bool = Field(False)

    # ── Rate Limiting ─────────────────────────────────────────────────

    rate_limit_requests: int = Field(
        1000, ge=0,
        description="Max requests per IP per 60s window (0=disabled)",
    )
    rate_limit_auth_attempts: int = Field(
        30, ge=1,
        description="Max failed auth attempts per IP per 60s")

    # ── Prompt Injection ──────────────────────────────────────────────

    prompt_injection_enabled: bool = Field(True)
    prompt_injection_block_threshold: float = Field(0.9, ge=0.0, le=1.0)
    prompt_injection_sanitize_threshold: float = Field(0.7, ge=0.0, le=1.0)
    prompt_injection_flag_threshold: float = Field(0.4, ge=0.0, le=1.0)
    prompt_injection_model: str = Field("")
    prompt_injection_audit_log: str = Field("")

    # ── Quota / Usage ──────────────────────────────────────────────────

    quota_enabled: bool = Field(False)
    usage_db_path: str = Field(".usage.db")

    # ── Cost tracking ──────────────────────────────────────────────────
    # Gates whether the streaming path attaches cost telemetry to the
    # response. Off by default: the per-token cost event is a DistLLM
    # extension, and injecting it unconditionally put a non-spec `usage`
    # object on every chunk of a plain OpenAI-compatible stream.
    cost_tracking: bool = Field(False)

    # ── Observability ─────────────────────────────────────────────────

    trace_sample_rate: float = Field(0.1, ge=0.0, le=1.0)
    enable_docs: bool = Field(False)
    otlp_endpoint: str = Field("http://localhost:4318/v1/traces")

    # ── Config / Plugin ────────────────────────────────────────────────

    config_path: str = Field("config.yaml")
    verify_plugins: bool = Field(False)

    # ── SSO ────────────────────────────────────────────────────────────

    sso_provider: str = Field("")
    sso_client_id: str = Field("")
    sso_client_secret: str = Field("")
    sso_authority: str = Field("")
    sso_metadata_url: str = Field("")
    sso_callback_url: str = Field("")
    sso_jwks_url: str = Field("")

    # ── Feature Flags ─────────────────────────────────────────────────

    enable_debug_routes: bool = Field(False)
    dev_mode: bool = Field(False)
    api_key: str = Field("", validation_alias="API_KEY")
    cluster_key: str = Field("")
    ha_shared_secret: str = Field("")


# Global singleton
_settings: APISettings | None = None


def get_api_settings() -> APISettings:
    global _settings
    if _settings is None:
        _settings = APISettings()
    return _settings


# Convenience accessor
api_settings = get_api_settings()
