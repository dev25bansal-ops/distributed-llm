"""Multi-model chat router for content-based model selection.

Allows defining compound/hybrid models that route queries to different
backend models based on content matching rules (keyword, regex, workload type).
"""

from __future__ import annotations

import re
from typing import Any

from distllm.config.settings import ChatRouterSettings, RouteRuleSettings
from distllm.core.workload_classifier import classify


class ModelRouter:
    """Content-aware router that selects a target model based on routing rules.

    Rules are evaluated in priority order (highest first). The first matching
    rule determines the target model. If no rule matches, the default model
    is returned.

    Supports three match strategies:
      - ``keyword``: case-insensitive substring match against the user message
      - ``regex``: regular expression match against the user message
      - ``workload``: uses ``WorkloadClassifier`` to match workload type
    """

    def __init__(self, settings: ChatRouterSettings) -> None:
        self._default_model = settings.default_model
        self._rules = sorted(settings.routes, key=lambda r: r.priority, reverse=True)
        self._hybrid_names: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, message: str, available_models: list[str] | None = None) -> str:
        """Select a target model for *message* based on configured rules.

        Args:
            message: The user's input text to route on.
            available_models: Optional list of backend model names for
                availability checks.

        Returns:
            The name of the target model to use.
        """
        for rule in self._rules:
            if self._match(rule, message):
                target = rule.target_model
                if available_models is None or target in available_models:
                    return target
        return self._default_model

    def list_hybrid_models(self) -> list[str]:
        """Return the list of configured hybrid model names."""
        return list(self._hybrid_names)

    def register_hybrid_name(self, name: str) -> None:
        """Register *name* as a recognised hybrid model identifier.

        The chat route handler checks this list to decide whether to
        invoke the router.
        """
        if name not in self._hybrid_names:
            self._hybrid_names.append(name)

    # ------------------------------------------------------------------
    # Rule matching
    # ------------------------------------------------------------------

    def _match(self, rule: RouteRuleSettings, message: str) -> bool:
        strategy = rule.match_type

        if strategy == "keyword":
            return rule.match.lower() in message.lower()

        if strategy == "regex":
            try:
                return bool(re.search(rule.match, message))
            except re.error:
                return False

        if strategy == "workload":
            try:
                wt = classify(message)
                return wt.value == rule.match
            except Exception:
                return False

        return False

    # ------------------------------------------------------------------
    # Serialisation helpers (for API responses)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable representation of the router state."""
        return {
            "enabled": True,
            "rules": [
                {
                    "name": r.name,
                    "match_type": r.match_type,
                    "match": r.match,
                    "target_model": r.target_model,
                    "priority": r.priority,
                }
                for r in self._rules
            ],
            "default_model": self._default_model,
            "hybrid_models": list(self._hybrid_names),
        }
