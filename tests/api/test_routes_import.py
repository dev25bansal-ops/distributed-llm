"""Smoke test: verify all API route modules import and register without errors."""

import importlib
import pkgutil
import pytest
from fastapi import APIRouter

import distllm.api.routes as routes_pkg


ROUTE_FILES = sorted([
    name for _, name, _ in pkgutil.iter_modules(routes_pkg.__path__)
    if name != "__init__"
])


class TestAllRouteImports:
    def test_all_route_modules_importable(self):
        for name in ROUTE_FILES:
            mod = importlib.import_module(f"distllm.api.routes.{name}")
            assert hasattr(mod, "router"), f"{name} has no 'router' attribute"
            assert isinstance(mod.router, APIRouter), f"{name}.router not an APIRouter"

    def test_route_count(self):
        assert len(ROUTE_FILES) == 21  # 23 files - __init__ - __pycache__

    def test_all_routes_have_unique_prefixes(self):
        seen = set()
        for name in ROUTE_FILES:
            mod = importlib.import_module(f"distllm.api.routes.{name}")
            router = mod.router
            for route in router.routes:
                key = (getattr(route, "path", ""), tuple(sorted(getattr(route, "methods", []))))
                assert key not in seen, f"Duplicate route: {key} from {name}"
                seen.add(key)

    def test_each_router_has_routes(self):
        for name in ROUTE_FILES:
            mod = importlib.import_module(f"distllm.api.routes.{name}")
            router = mod.router
            assert len(router.routes) > 0, f"{name} router has no routes"
