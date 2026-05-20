"""API Gateway with distributed rate limiting and multi-tenant isolation.

Provides centralized request routing, tenant quotas, distributed
rate limiting via Redis, and API key management.
"""
import time
import hashlib
import threading
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


@dataclass
class TenantQuota:
    """Quota configuration for a tenant."""
    tenant_id: str
    requests_per_minute: int = 60
    tokens_per_minute: int = 10000
    max_concurrent: int = 10
    priority: int = 2  # 0=critical, 1=high, 2=normal, 3=low
    api_keys: list[str] = field(default_factory=list)


@dataclass
class RateLimitState:
    """Current rate limit state for a client."""
    requests: int = 0
    tokens: int = 0
    concurrent: int = 0
    window_start: float = field(default_factory=time.time)


class DistributedRateLimiter:
    """Distributed rate limiter using Redis for cross-coordinator state.
    
    Falls back to in-memory if Redis is unavailable.
    """
    
    def __init__(self, redis_url: str | None = None):
        self._redis_url = redis_url
        self._redis: redis.Redis | None = None
        self._local_state: dict[str, RateLimitState] = {}
        self._lock = threading.Lock()
        
        if redis_url and REDIS_AVAILABLE:
            try:
                self._redis = redis.from_url(redis_url, decode_responses=True)
                self._redis.ping()
                logger.info(f"Distributed rate limiter connected to Redis")
            except Exception as e:
                logger.warning(f"Redis unavailable, using in-memory rate limiting: {e}")
                self._redis = None
    
    def check_rate_limit(
        self,
        tenant_id: str,
        client_id: str,
        quota: TenantQuota,
        token_count: int = 1,
    ) -> tuple[bool, dict]:
        """Check if a request is within rate limits.
        
        Returns:
            (allowed, headers) tuple.
        """
        key = f"distllm:ratelimit:{tenant_id}:{client_id}"
        now = time.time()
        
        if self._redis:
            return self._check_redis(key, quota, token_count, now)
        else:
            return self._check_local(key, quota, token_count, now)
    
    def _check_redis(self, key: str, quota: TenantQuota, tokens: int, now: float) -> tuple[bool, dict]:
        pipe = self._redis.pipeline()
        window_key = f"{key}:window"
        
        pipe.get(f"{key}:requests")
        pipe.get(f"{key}:tokens")
        pipe.ttl(window_key)
        results = pipe.execute()
        
        requests = int(results[0] or 0)
        current_tokens = int(results[1] or 0)
        
        if requests >= quota.requests_per_minute:
            return False, {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": str(results[2] or 60)}
        if current_tokens + tokens > quota.tokens_per_minute:
            return False, {"X-RateLimit-Remaining-Tokens": "0"}
        
        pipe = self._redis.pipeline()
        pipe.incr(f"{key}:requests")
        pipe.incrby(f"{key}:tokens", tokens)
        if results[2] < 0:
            pipe.expire(f"{key}:requests", 60)
            pipe.expire(f"{key}:tokens", 60)
        pipe.execute()
        
        return True, {
            "X-RateLimit-Remaining": str(quota.requests_per_minute - requests - 1),
            "X-RateLimit-Remaining-Tokens": str(quota.tokens_per_minute - current_tokens - tokens),
        }
    
    def _check_local(self, key: str, quota: TenantQuota, tokens: int, now: float) -> tuple[bool, dict]:
        with self._lock:
            if key not in self._local_state:
                self._local_state[key] = RateLimitState()
            
            state = self._local_state[key]
            
            # Reset window if expired
            if now - state.window_start > 60:
                state.requests = 0
                state.tokens = 0
                state.window_start = now
            
            if state.requests >= quota.requests_per_minute:
                return False, {"X-RateLimit-Remaining": "0"}
            if state.tokens + tokens > quota.tokens_per_minute:
                return False, {"X-RateLimit-Remaining-Tokens": "0"}
            
            state.requests += 1
            state.tokens += tokens
            
            return True, {
                "X-RateLimit-Remaining": str(quota.requests_per_minute - state.requests),
            }


class APIGateway:
    """API Gateway for distributed LLM with tenant isolation.
    
    Usage:
        gateway = APIGateway()
        gateway.register_tenant(TenantQuota(tenant_id="acme", ...))
        allowed, headers = gateway.check_request("acme", "user-123", token_count=50)
    """
    
    def __init__(self, redis_url: str | None = None):
        self._limiter = DistributedRateLimiter(redis_url)
        self._tenants: dict[str, TenantQuota] = {}
        self._api_keys: dict[str, str] = {}  # api_key -> tenant_id
        self._lock = threading.Lock()
    
    def register_tenant(self, quota: TenantQuota) -> None:
        """Register a tenant with quota configuration."""
        with self._lock:
            self._tenants[quota.tenant_id] = quota
            for key in quota.api_keys:
                self._api_keys[key] = quota.tenant_id
        logger.info(f"Registered tenant: {quota.tenant_id}")
    
    def authenticate_api_key(self, api_key: str) -> str | None:
        """Authenticate an API key and return tenant_id."""
        return self._api_keys.get(api_key)
    
    def check_request(
        self,
        tenant_id: str,
        client_id: str,
        token_count: int = 1,
    ) -> tuple[bool, dict]:
        """Check if a request is allowed for the tenant."""
        quota = self._tenants.get(tenant_id)
        if quota is None:
            return False, {"X-Error": "Unknown tenant"}
        
        return self._limiter.check_rate_limit(tenant_id, client_id, quota, token_count)
    
    def get_tenant_stats(self, tenant_id: str) -> dict:
        """Get usage statistics for a tenant."""
        quota = self._tenants.get(tenant_id)
        if quota is None:
            return {"error": "Unknown tenant"}
        return {
            "tenant_id": tenant_id,
            "quota": {
                "requests_per_minute": quota.requests_per_minute,
                "tokens_per_minute": quota.tokens_per_minute,
                "max_concurrent": quota.max_concurrent,
            },
        }
    
    def list_tenants(self) -> list[str]:
        """List all registered tenant IDs."""
        return list(self._tenants.keys())
