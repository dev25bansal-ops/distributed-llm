"""Coordinator, network, TLS, rate limiting, wide-area, and chat router
configuration classes."""

from pydantic import BaseModel, Field, field_validator

__all__ = [
    "CoordinatorSettings",
    "NetworkSettings",
    "TLSSettings",
    "RateLimitSettings",
    "WideAreaSettings",
    "RouteRuleSettings",
    "ChatRouterSettings",
]


class CoordinatorSettings(BaseModel):
    """Coordinator configuration."""
    host: str = "localhost"
    port: int = 50050
    api_port: int = 8000
    cors_origins: str = "http://localhost:3000,http://localhost:8080"

    @field_validator("port", "api_port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError(f"Port must be 1-65535, got {v}")
        return v

    @field_validator("cors_origins")
    @classmethod
    def validate_origins(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("cors_origins must not be empty")
        # Validate each origin is a well-formed URL
        for origin in v.split(","):
            origin = origin.strip()
            if origin == "*":
                continue  # Wildcard handled separately in _get_cors_origins
            if not (origin.startswith("http://") or origin.startswith("https://") or origin.startswith("chrome-extension://") or origin.startswith("moz-extension://")):
                raise ValueError(
                    f"CORS origin '{origin}' must be a valid URL starting with http://, https://, "
                    f"or a browser extension scheme."
                )
        return v


class NetworkSettings(BaseModel):
    """Network configuration."""
    grpc_timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0

    @field_validator("grpc_timeout", "max_retries")
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"Must be positive, got {v}")
        return v


class TLSSettings(BaseModel):
    """TLS/Security configuration.

    Supports both server-side TLS and mutual TLS (mTLS) with client
    certificate authentication.
    """
    enabled: bool = False
    cert_dir: str = "certs"
    cert_file: str | None = None
    key_file: str | None = None
    ca_cert_file: str | None = None
    client_cert_file: str | None = Field(
        default=None,
        description="Client certificate for mTLS. When set together with "
                    "client_key_file, enables mutual TLS where clients must "
                    "present a valid certificate signed by the CA.",
    )
    client_key_file: str | None = Field(
        default=None,
        description="Client private key for mTLS.",
    )
    require_client_cert: bool = Field(
        default=False,
        description="If True, reject connections that don't present a valid "
                    "client certificate. Requires ca_cert_file to be set.",
    )
    min_tls_version: str = Field(
        default="TLSv1.2",
        description="Minimum TLS version allowed (TLSv1.2 or TLSv1.3).",
    )


class RateLimitSettings(BaseModel):
    """API rate limiting configuration."""
    enabled: bool = True
    default_rpm: float = 60.0
    endpoint_limits: dict[str, float] = Field(default_factory=lambda: {
        "/v1/chat/completions": 30.0,
        "/v1/completions": 30.0,
        "/health": 120.0,
        "/metrics": 120.0,
    })
    burst_multiplier: float = 1.5
    # Security: Separate rate limits for authenticated vs unauthenticated clients
    auth_rpm_multiplier: float = 2.0  # Authenticated clients get higher limits

    @field_validator("default_rpm", "burst_multiplier", "auth_rpm_multiplier")
    @classmethod
    def validate_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Must be positive, got {v}")
        return v


class WideAreaSettings(BaseModel):
    """Wide-area network distributed inference configuration.

    Enables P2P node forwarding to reduce coordinator round trips
    across high-latency links (geographically distributed nodes).
    """
    enabled: bool = False
    p2p_forwarding: bool = True
    tokens_before_forward: int = 10
    wan_timeout_seconds: int = 60
    max_retries: int = 3
    backoff_base_seconds: float = 1.0

    @field_validator("tokens_before_forward")
    @classmethod
    def validate_tokens_before_forward(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"tokens_before_forward must be >= 1, got {v}")
        return v

    @field_validator("wan_timeout_seconds")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"wan_timeout_seconds must be >= 1, got {v}")
        return v


class RouteRuleSettings(BaseModel):
    """A single routing rule for the multi-model chat router."""
    name: str = Field(default="", description="Rule name for identification")
    match_type: str = Field(default="keyword", description="Matching strategy: keyword, regex, or workload")
    match: str = Field(default="", description="Pattern to match against the user message")
    target_model: str = Field(default="", description="Model to route to when this rule matches")
    priority: int = Field(default=0, ge=0, description="Rule priority (higher = evaluated first)")


class ChatRouterSettings(BaseModel):
    """Multi-model chat router configuration.

    Allows defining compound/hybrid models that route queries to different
    backend models based on content matching rules.

    Example config (YAML):
    .. code-block:: yaml

        chat_router:
          enabled: true
          name: hybrid
          default_model: llama3
          routes:
            - name: code-route
              match_type: keyword
              match: "write a function"
              target_model: codellama
              priority: 10
            - name: creative-route
              match_type: keyword
              match: "write a story"
              target_model: llama3
              priority: 5
    """
    enabled: bool = False
    name: str = Field(default="hybrid", description="Model name that clients use to invoke this router (e.g., 'hybrid', 'smart-router')")
    default_model: str = Field(default="", description="Default model when no rules match")
    routes: list[RouteRuleSettings] = Field(default_factory=list, description="Ordered routing rules")
