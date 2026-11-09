"""Helpers for inspecting the assembled FastAPI app in API tests.

FastAPI >= 0.121 wraps ``include_router`` registrations in an internal
``_IncludedRouter`` object that carries the sub-routes on ``.original_router``
but has no ``.path`` of its own. Walking ``app.routes`` flat therefore misses
every routed endpoint and reports it as "not mounted", even though the live
server answers it. These helpers descend into the wrapper.
"""

from __future__ import annotations

from typing import Any


def iter_routes(app_or_routes: Any, _depth: int = 0):
    """Yield every leaf route, descending into ``_IncludedRouter`` wrappers."""
    if _depth > 8:  # guard against a malformed self-referencing wrapper
        return
    routes = (
        app_or_routes
        if isinstance(app_or_routes, (list, tuple))
        else getattr(app_or_routes, "routes", [])
    )
    for route in routes:
        inner = getattr(route, "original_router", None)
        if inner is not None:
            yield from iter_routes(getattr(inner, "routes", []), _depth + 1)
            continue
        yield route


def route_paths(app: Any) -> set[str]:
    """Every path the app serves, including those inside router wrappers."""
    return {
        p for r in iter_routes(app) if isinstance((p := getattr(r, "path", None)), str)
    }


def find_route(app: Any, path: str):
    """Return the first leaf route serving ``path``, or None."""
    for route in iter_routes(app):
        if getattr(route, "path", None) == path:
            return route
    return None


def routes_matching(app: Any, path: str):
    """Every leaf route serving ``path`` (methods differ across them)."""
    return [r for r in iter_routes(app) if getattr(r, "path", None) == path]
