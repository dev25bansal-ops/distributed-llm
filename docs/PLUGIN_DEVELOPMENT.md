# Plugin Development Guide

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

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="my-plugin",
            version="1.0.0",
            description="Logs all requests and responses",
        )

    def on_start(self, config: dict) -> None:
        """Called when plugin is started."""
        self.logger.info("MyPlugin started")

    def on_request(self, context: dict) -> dict | None:
        """Called before request is processed."""
        self.logger.info(f"Request: {context.get('prompt', '')[:50]}...")
        return None  # Return None to pass through, or modified context

    def on_response(self, context: dict, response: dict) -> dict | None:
        """Called after response is generated."""
        self.logger.info(f"Response: {len(response.get('text', ''))} chars")
        return None

    def on_stop(self, config: dict) -> None:
        """Called when plugin is stopped."""
        self.logger.info("MyPlugin stopped")
```

### 2. Register Plugin

```bash
# Copy to plugins directory
cp my_plugin.py ~/.distllm/plugins/

# Or add to config
export DISTLLM_PLUGIN_DIRS="~/.distllm/plugins,/opt/distllm/plugins"
```

### 3. Verify Plugin Loaded

```bash
distllm system plugins list
# my-plugin 1.0.0 - Logs all requests and responses
```

---

## Lifecycle Hooks

| Hook | When | Purpose |
|------|------|---------|
| `on_start(config)` | Plugin started | Initialize resources |
| `on_request(context)` | Before request | Modify request, add headers |
| `on_response(context, response)` | After response | Transform response, add metadata |
| `on_error(context, error)` | On error | Log errors, send alerts |
| `on_model_load(model_name, config)` | Model loaded | Initialize model-specific state |
| `on_model_unload(model_name)` | Model unloaded | Cleanup model resources |
| `on_config_change(key, old, new)` | Config changed | React to config updates |
| `on_stop(config)` | Plugin stopped | Cleanup resources |

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

    def on_response(self, context: dict, response: dict) -> dict | None:
        start_time = context.get("_start_time")
        if start_time:
            latency = time.time() - start_time
            self._latency_histogram.observe(latency)
        return None
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

    def on_response(self, context: dict, response: dict) -> dict | None:
        if "_cached_response" not in context:
            prompt = context.get("prompt", "")
            self._cache.store(prompt, response.get("text", ""))
        return None
```

---

## Configuration

### Plugin Config in `config.yaml`

```yaml
plugins:
  enabled: true
  directories:
    - ~/.distllm/plugins
    - /opt/distllm/plugins
  settings:
    my-plugin:
      log_level: info
      max_requests: 1000
```

### Environment Variables

```bash
# Plugin directories (comma-separated)
export DISTLLM_PLUGIN_DIRS="~/.distllm/plugins"

# Disable specific plugin
export DISTLLM_DISABLED_PLUGINS="my-plugin,other-plugin"

# Plugin-specific settings
export DISTLLM_PLUGIN_MY_PLUGIN_LOG_LEVEL=debug
```

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
    context = {"prompt": "Hello"}
    response = {"text": "Hi there!"}
    result = plugin.on_response(context, response)
    assert result is None
```

---

## Best Practices

1. **Keep hooks fast**: Hooks run on every request — avoid blocking operations
2. **Handle errors gracefully**: Don't let plugin errors crash the server
3. **Use logging**: Use `self.logger` for debugging
4. **Minimize state**: Plugins should be stateless when possible
5. **Document configuration**: Document all config options in metadata
6. **Version your plugins**: Use semantic versioning
7. **Test thoroughly**: Write unit tests for all hooks
