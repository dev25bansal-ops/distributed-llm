"""Plugin system activation — built-in plugins and discovery support.

The plugin framework in ``distllm.core.plugin_system`` provides generic
discovery, lifecycle, and hook dispatch.  This package ships the built-in
plugins that ship with the distribution:

* ``RateLimitPlugin`` — per-tenant / per-model request rate limiting
* ``AuditLogPlugin`` — structured audit logging of all API requests
* ``MetricsPlugin`` — plugin health and hook-invocation metrics

Third-party plugins can be installed via ``pip`` using the
``distllm.plugins`` entry-point group, or placed in any directory
listed in the plugin configuration.
"""
