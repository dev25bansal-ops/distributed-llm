# Plugin Development Guide

## Table of Contents

<!-- TOC -->

- [Overview](#overview)
- [Quick Start](#quick-start)
  - [1. Create Plugin File](#1-create-plugin-file)
  - [2. Register Plugin](#2-register-plugin)
  - [3. Verify Plugin Loaded](#3-verify-plugin-loaded)
- [Lifecycle Hooks](#lifecycle-hooks)
- [Examples](#examples)
  - [Rate Limiting Plugin](#rate-limiting-plugin)
  - [Metrics Export Plugin](#metrics-export-plugin)
  - [Content Filter Plugin](#content-filter-plugin)
  - [Caching Plugin](#caching-plugin)
- [Configuration](#configuration)
  - [Plugin Config in `config.yaml`](#plugin-config-in-configyaml)
  - [Environment Variables](#environment-variables)
- [Testing Plugins](#testing-plugins)
- [Best Practices](#best-practices)

<!-- /TOC -->

## Overview

DistLLM supports plugins that extend its functionality through lifecycle hooks. Plugins can:
- Modify requests before processing
- Transform responses after generation
- Add custom metrics
- Integrate with external systems
- Implement custom scheduling policies

## Quick Start

### 1. Create Plugin File

```python
# my_plugin.py
from distllm.core.plugin_system import PluginBase, PluginMetadata

class MyPlugin(PluginBase):
    """Example plugin that logs all requests."""

    def on_start(self, context: dict) -> None:
        """Called when plugin is started."""
        logger.info("MyPlugin started")

    def on_request(self, context: dict) -> dict | None:
        """Called before request is processed."""
        logger.info(f"Request: {context.get('prompt', '')[:50]}...")
        return None  # Return None to pass through, or modified context

    def on_response(self, request: dict, response: dict) -> None:
        """Called after response is generated."""
        logger.info(f"Response: {len(response.get('text', ''))} chars")

    def on_stop(self, context: dict) -> None:
        """Called when plugin is stopped."""
        logger.info("MyPlugin stopped")
```

> ⚠️ UNVERIFIED (hygiene pass 2026-11-06): the `metadata` property and
> `PluginMetadata(name=..., version=..., description=...)` constructor do not
> match PluginBase — `metadata()` takes no args and returns
> `PluginMetadata(name=self.name(), version=self.version(),
> description=self.__doc__ or "")`; name/version are plain methods, not a
> property. `self.logger` is not defined on PluginBase; the module uses
> loguru's module-level `logger` (src/distllm/core/plugin_system.py:46).

### 2. Register Plugin

```bash
# Copy to plugins directory
cp my_plugin.py ~/.distllm/plugins/

# Or add to config
export DISTLLM_PLUGIN_DIRS="~/.distllm/plugins,/opt/distllm/plugins"
```

> ⚠️ UNVERIFIED (hygiene pass 2026-11-06): `~/.distllm/plugins` is not a
> code-defined default — `PluginSystem.discover()` takes explicit dirs and no
> source reads this path — and `DISTLLM_PLUGIN_DIRS` has 0 hits in src/, so
> the documented env var does not exist in code.

### 3. Verify Plugin Loaded

```bash
distllm system plugins list
```

> ⚠️ UNVERIFIED (hygiene pass 2026-11-06): no `plugins` subcommand exists in
> src/distllm/cli/. Plugin listing is served via HTTP: GET /v1/plugins
> (src/distllm/api/routes/plugins.py:90, admin-gated).

---

## Lifecycle Hooks

| Hook | When | Purpose |
|------|------|---------|
| `on_start(context)` | Plugin started | Initialize resources |
| `on_request(context)` | Before request | Modify request, add headers |
| `on_response(request, response)` | After response | Transform response, add metadata |
| `on_error(request, error)` | On error | Log errors, send alerts |
| `on_model_load(model_name, config)` | Model loaded | Initialize model-specific state |
| `on_model_unload(model_name)` | Model unloaded | Cleanup model resources |
| `on_config_change(key, old, new)` | Config changed | React to config updates |
| `on_stop(context)` | Plugin stopped | Cleanup resources |

---

## Examples

### Rate Limiting Plugin

```python
class RateLimitPlugin(PluginBase):
    """Per-user rate limiting."""

    def __init__(self):
        self._limits = {}  # user_id -> (count, window_start)
        self._max_requests = 100
        self._window_seconds = 60

    def on_request(self, context: dict) -> dict | None:
        user_id = context.get("user_id", "anonymous")
        now = time.time()

        if user_id in self._limits:
            count, window_start = self._limits[user_id]
            if now - window_start > self._window_seconds:
                self._limits[user_id] = (1, now)
            elif count >= self._max_requests:
                raise Exception(f"Rate limit exceeded for {user_id}")
            else:
                self._limits[user_id] = (count + 1, window_start)
        else:
            self._limits[user_id] = (1, now)

        return None
```

### Metrics Export Plugin

```python
class PrometheusMetricsPlugin(PluginBase):
    """Export custom metrics to Prometheus."""

    def __init__(self):
        from prometheus_client import Counter, Histogram
        self._request_counter = Counter(
            "distllm_plugin_requests_total",
            "Total requests processed",
            ["user_id"],
        )
        self._latency_histogram = Histogram(
            "distllm_plugin_latency_seconds",
            "Request latency",
        )

    def on_request(self, context: dict) -> dict | None:
        context["_start_time"] = time.time()
        self._request_counter.labels(
            user_id=context.get("user_id", "unknown")
        ).inc()
        return None

    def on_response(self, request: dict, response: dict) -> None:
        start_time = request.get("_start_time")
        if start_time:
            latency = time.time() - start_time
            self._latency_histogram.observe(latency)
```

### Content Filter Plugin

```python
class ContentFilterPlugin(PluginBase):
    """Filter inappropriate content."""

    BLOCKED_PATTERNS = [
        r"\b(password|secret|api[_-]?key)\s*[:=]\s*\S+",
        # Add more patterns
    ]

    def on_request(self, context: dict) -> dict | None:
        prompt = context.get("prompt", "")
        for pattern in self.BLOCKED_PATTERNS:
            if re.search(pattern, prompt, re.IGNORECASE):
                raise Exception("Request contains blocked content")
        return None
```

### Caching Plugin

```python
class SemanticCachePlugin(PluginBase):
    """Cache responses for similar prompts."""

    def __init__(self):
        from distllm.core.semantic_cache import SemanticCache
        self._cache = SemanticCache(similarity_threshold=0.92)

    def on_request(self, context: dict) -> dict | None:
        prompt = context.get("prompt", "")
        cached = self._cache.lookup(prompt)
        if cached:
            context["_cached_response"] = cached
        return None

    def on_response(self, request: dict, response: dict) -> None:
        if "_cached_response" not in request:
            prompt = request.get("prompt", "")
            self._cache.store(prompt, response.get("text", ""))
```

---

## Configuration

### Plugin Config in `config.yaml`

```yaml
plugins:
  enabled: true
  plugins:
    - module: my_plugin.MyPlugin
```

> ⚠️ UNVERIFIED (hygiene pass 2026-11-06): the `directories` + per-plugin
> `settings` keys shown previously are not in the live config schema. The real
> schema is PluginSettings (src/distllm/config/_application.py:29-39):
> `enabled: bool` + `plugins: list[dict]` where each entry requires a fully
> qualified `module`; keys beyond `module` are not validated, so per-plugin
> `settings` is UNVERIFIED.

### Environment Variables

```bash
# Plugin directories (comma-separated)
export DISTLLM_PLUGIN_DIRS="~/.distllm/plugins"

# Disable specific plugin
export DISTLLM_DISABLED_PLUGINS="my-plugin,other-plugin"

# Plugin-specific settings
export DISTLLM_PLUGIN_MY_PLUGIN_LOG_LEVEL=debug
```

> ⚠️ UNVERIFIED (hygiene pass 2026-11-06): none of the three env vars above
> exist in src/ — `DISTLLM_PLUGIN_DIRS`, `DISTLLM_DISABLED_PLUGINS`, and
> `DISTLLM_PLUGIN_*` per-plugin settings all have 0 hits. The only
> plugin-related env vars found are DISTLLM_PLUGIN_RATELIMIT_ENABLED /
> DISTLLM_PLUGIN_RATELIMIT_DEFAULT (documented in
> src/distllm/api/routes/plugins.py:72) and DISTLLM_AUDIT_LOG
> (routes/plugins.py:77).

---

## Testing Plugins

```python
import pytest
from my_plugin import MyPlugin

def test_on_request():
    plugin = MyPlugin()
    context = {"prompt": "Hello, world!"}
    result = plugin.on_request(context)
    assert result is None  # Pass-through

def test_on_response():
    plugin = MyPlugin()
    request = {"prompt": "Hello"}
    response = {"text": "Hi there!"}
    plugin.on_response(request, response)  # returns None
```

---

## Best Practices

1. **Keep hooks fast**: Hooks run on every request — avoid blocking operations
2. **Handle errors gracefully**: Don't let plugin errors crash the server
3. **Use logging**: Use loguru's module-level `logger` (`from loguru import logger`) — `self.logger` does not exist on PluginBase
4. **Minimize state**: Plugins should be stateless when possible
5. **Document configuration**: Document all config options in metadata
6. **Version your plugins**: Use semantic versioning
7. **Test thoroughly**: Write unit tests for all hooks
