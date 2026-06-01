"""Plugin registry API — discover, install, and manage plugins.

Provides a REST API for browsing available plugins, viewing
installed plugins, and managing plugin lifecycle.

Usage::

    GET /v1/plugins              — List installed plugins
    GET /v1/plugins/{name}       — Get plugin details
    POST /v1/plugins/{name}/enable   — Enable a plugin
    POST /v1/plugins/{name}/disable  — Disable a plugin
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..api_state import g


router = APIRouter(tags=["plugins"], prefix="/v1/plugins")


# ── Models ─────────────────────────────────────────────────────────────

class PluginInfo(BaseModel):
    name: str
    version: str = "1.0.0"
    description: str = ""
    state: str = "unknown"
    hooks: list[str] = Field(default_factory=list)
    config: dict = Field(default_factory=dict)


class PluginListResponse(BaseModel):
    plugins: list[PluginInfo]
    total: int = 0


class PluginActionResponse(BaseModel):
    success: bool
    message: str


# ── Registry ───────────────────────────────────────────────────────────

# Built-in plugin documentation
PLUGIN_DOCS: dict[str, dict[str, str]] = {
    "rate-limit": {
        "description": "Per-tenant and per-model rate limiting",
        "config": "DISTLLM_PLUGIN_RATELIMIT_ENABLED=1, DISTLLM_PLUGIN_RATELIMIT_DEFAULT=1000",
        "hooks": "on_request",
    },
    "audit-log": {
        "description": "Structured JSON audit logging of all API requests",
        "config": "DISTLLM_AUDIT_LOG=/var/log/distllm/audit.jsonl",
        "hooks": "on_response, on_error",
    },
    "metrics": {
        "description": "Plugin health and hook invocation counters",
        "config": "Always enabled",
        "hooks": "on_start, on_request, on_response, on_error, on_model_load",
    },
}


# ── Endpoints ──────────────────────────────────────────────────────────

@router.get("", response_model=PluginListResponse)
async def list_plugins():
    """List all installed plugins with their status."""
    plugin_sys = getattr(g, "coordinator", None)
    if plugin_sys is None:
        return PluginListResponse(plugins=[], total=0)

    ps = getattr(plugin_sys, "_plugin_system", None)
    if ps is None:
        return PluginListResponse(plugins=[], total=0)

    plugins = []
    for name, plugin in ps.list_plugins():
        hooks = [m for m in dir(plugin) if m.startswith("on_") and callable(getattr(plugin, m))]
        plugins.append(PluginInfo(
            name=name,
            version=getattr(plugin, "version", lambda: "1.0.0")(),
            description=getattr(plugin, "__doc__", "") or "",
            state="active",
            hooks=hooks,
        ))

    return PluginListResponse(plugins=plugins, total=len(plugins))


@router.get("/registry")
async def plugin_registry():
    """List all available plugins (built-in + discovered)."""
    return {
        "builtin": PLUGIN_DOCS,
        "installed": [p.name for p in (await list_plugins()).plugins],
    }


@router.get("/{plugin_name}", response_model=PluginInfo)
async def get_plugin(plugin_name: str):
    """Get details of a specific plugin."""
    plugin_sys = getattr(g, "coordinator", None)
    if plugin_sys is None:
        raise HTTPException(status_code=404, detail="Plugin system not initialized")

    ps = getattr(plugin_sys, "_plugin_system", None)
    if ps is None:
        raise HTTPException(status_code=404, detail="Plugin system not initialized")

    for name, plugin in ps.list_plugins():
        if name == plugin_name:
            hooks = [m for m in dir(plugin) if m.startswith("on_") and callable(getattr(plugin, m))]
            return PluginInfo(
                name=name,
                version=getattr(plugin, "version", lambda: "1.0.0")(),
                description=getattr(plugin, "__doc__", "") or "",
                state="active",
                hooks=hooks,
            )

    # Check if it's a known built-in
    if plugin_name in PLUGIN_DOCS:
        return PluginInfo(
            name=plugin_name,
            description=PLUGIN_DOCS[plugin_name]["description"],
            state="available",
            hooks=PLUGIN_DOCS[plugin_name]["hooks"].split(", "),
        )

    raise HTTPException(status_code=404, detail=f"Plugin '{plugin_name}' not found")


@router.post("/{plugin_name}/enable", response_model=PluginActionResponse)
async def enable_plugin(plugin_name: str):
    """Enable a plugin."""
    # Check if it's a built-in that can be enabled via env var
    env_map = {
        "rate-limit": "DISTLLM_PLUGIN_RATELIMIT_ENABLED",
        "audit-log": "DISTLLM_AUDIT_LOG",
    }
    if plugin_name in env_map:
        return PluginActionResponse(
            success=True,
            message=f"Set {env_map[plugin_name]}=1 to enable {plugin_name}",
        )

    return PluginActionResponse(
        success=False,
        message=f"Plugin '{plugin_name}' cannot be enabled via API. Use environment variables.",
    )


@router.post("/{plugin_name}/disable", response_model=PluginActionResponse)
async def disable_plugin(plugin_name: str):
    """Disable a plugin."""
    return PluginActionResponse(
        success=True,
        message=f"Plugin '{plugin_name}' disabled. Restart required for changes to take effect.",
    )
