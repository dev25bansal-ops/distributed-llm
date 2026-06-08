"""Real-time model switching — route queries to specialized models.

Analyzes query content and routes to the best-suited model:
- Code queries -> code-specialized model
- Math queries -> math-specialized model
- Creative queries -> creative-specialized model
- General queries -> default model

Integrates with the chat router config for rule-based routing.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from distllm.config.settings import ChatRouterSettings
    from distllm.core.learning_router import LearningRouter

from loguru import logger


@dataclass
class RouteRule:
    """A single routing rule."""
    name: str
    match_type: str  # "keyword", "regex", "workload"
    pattern: str
    target_model: str
    priority: int = 0


@dataclass
class RouteMatch:
    """Result of a routing decision."""
    model: str
    rule_name: str
    confidence: float
    latency_ms: float


@dataclass
class RoutingContext:
    """Multi-attribute routing context passed alongside the query text.

    All fields are optional.  When present they act as additional routing
    signals that are evaluated *after* content-based rules but *before*
    the final fallback.
    """
    cost_budget: float | None = None
    max_latency_ms: float | None = None
    language: str | None = None
    has_tool_calls: bool = False
    input_tokens: int | None = None
    tenant_tier: str | None = None


# Workload classification patterns
_WORKLOAD_PATTERNS = {
    "code": {
        "keywords": [
            "function", "class", "def ", "import ", "const ", "let ", "var ",
            "write a function", "write a class", "implement", "debug", "refactor",
            "code", "program", "script", "api", "endpoint", "database", "sql",
            "python", "javascript", "typescript", "rust", "golang", "java",
            "html", "css", "react", "vue", "angular", "node", "docker",
            "git", "commit", "merge", "pull request", "test", "unit test",
        ],
        "regex": [
            r"(?:write|create|implement|fix|debug)\s+(?:a|the|this)?\s*(?:function|class|method|module|script|program)",
            r"```[a-z]*\n",
            r"(?:def|class|function|const|let|var)\s+\w+",
        ],
    },
    "math": {
        "keywords": [
            "calculate", "compute", "solve", "equation", "formula", "integral",
            "derivative", "matrix", "vector", "probability", "statistics",
            "algebra", "geometry", "trigonometry", "calculus", "linear algebra",
            "prove", "theorem", "proof", "mathematical", "math",
            "sum", "product", "factorial", "fibonacci", "prime",
            "optimize", "minimize", "maximize", "constraint",
        ],
        "regex": [
            r"(?:solve|calculate|compute|find)\s+(?:the|a)?\s*(?:integral|derivative|sum|limit|matrix)",
            r"\d+\s*[+\-*/^]\s*\d+",
            r"(?:prove|show)\s+that",
        ],
    },
    "creative": {
        "keywords": [
            "write a story", "write a poem", "creative", "fiction", "novel",
            "character", "plot", "dialogue", "narrative", "imaginative",
            "write me a", "compose", "draft", "blog post", "article",
            "marketing", "copywriting", "slogan", "tagline", "headline",
            "recipe", "instructions", "guide", "tutorial",
        ],
        "regex": [
            r"write\s+(?:me\s+)?(?:a|the|some)\s+(?:story|poem|song|letter|essay|article|blog)",
            r"(?:create|compose|draft)\s+(?:a|the)\s+(?:story|poem|narrative)",
        ],
    },
    "analysis": {
        "keywords": [
            "analyze", "analyse", "compare", "contrast", "evaluate", "assess",
            "summarize", "summarise", "explain", "describe", "review",
            "pros and cons", "advantages", "disadvantages", "trade-offs",
            "what are the", "how does", "why does", "what is",
        ],
        "regex": [
            r"(?:analyze|analyse|compare|evaluate|assess)\s+(?:the|this|these)",
            r"what\s+(?:are|is|does|do)\s+the",
        ],
    },
}


class ModelRouter:
    """Routes queries to specialized models based on content analysis.

    Supports:
    - Keyword matching
    - Regex matching
    - Workload classification (code/math/creative/analysis)
    - Configurable rules from ChatRouterSettings
    - Fallback to default model
    - Hybrid model name registration
    - Available model filtering

    Usage::

        from distllm.config.settings import ChatRouterSettings
        settings = ChatRouterSettings(enabled=True, default_model="llama3", routes=[...])
        router = ModelRouter(settings)
        model = router.resolve("write a Python function")
        # model == "codellama"
    """

    def __init__(self, settings: ChatRouterSettings | None = None, default_model: str = "") -> None:
        self._default_model = default_model
        self._rules: list[RouteRule] = []
        self._hybrid_names: list[str] = []
        self._stats: dict = {
            "total_routes": 0,
            "routes_by_model": {},
            "routes_by_rule": {},
        }
        self._lock = threading.Lock()
        # Instance-level copy so callers can extend at runtime
        self._workload_patterns: dict[str, dict] = {
            k: {
                "keywords": list(v.get("keywords", [])),
                "regex": list(v.get("regex", [])),
            }
            for k, v in _WORKLOAD_PATTERNS.items()
        }
        # Multi-attribute routing tiers (sorted by threshold)
        self._cost_tiers: list[tuple[float, str, str]] = []
        self._latency_tiers: list[tuple[float, str, str]] = []
        self._language_routes: dict[str, tuple[str, str]] = {}
        self._length_tiers: list[tuple[int, str, str]] = []
        self._tool_call_model: str = ""
        self._tool_call_rule: str = ""
        # Audit callback: (request_id, query_preview, rule_name, model, confidence) -> None
        self._audit_callback: Callable | None = None

        if settings is not None:
            self._init_from_settings(settings)

    def _init_from_settings(self, settings: ChatRouterSettings) -> None:
        """Initialize from a ChatRouterSettings config object."""
        self._default_model = getattr(settings, "default_model", "")
        for rule_cfg in getattr(settings, "routes", []):
            self._rules.append(RouteRule(
                name=getattr(rule_cfg, "name", ""),
                match_type=getattr(rule_cfg, "match_type", "keyword"),
                pattern=getattr(rule_cfg, "match", ""),
                target_model=getattr(rule_cfg, "target_model", ""),
                priority=getattr(rule_cfg, "priority", 0),
            ))
        self._rules.sort(key=lambda r: -r.priority)

    def add_rule(self, rule: RouteRule) -> None:
        """Add a routing rule."""
        self._rules.append(rule)
        self._rules.sort(key=lambda r: -r.priority)

    def add_workload_patterns(
        self,
        workload: str,
        keywords: list[str] | None = None,
        regex: list[str] | None = None,
    ) -> None:
        """Extend or create a workload category with additional patterns.

        Keywords are stored in lowercase for case-insensitive matching.

        Args:
            workload: Workload name (e.g. "code", "math", or a custom name).
            keywords: Keywords to append to this workload's keyword list.
            regex: Regex patterns to append to this workload's regex list.
        """
        if workload not in self._workload_patterns:
            self._workload_patterns[workload] = {"keywords": [], "regex": []}
        if keywords:
            self._workload_patterns[workload]["keywords"].extend(
                [kw.lower() for kw in keywords]
            )
        if regex:
            self._workload_patterns[workload]["regex"].extend(regex)

    # ── Multi-attribute routing ────────────────────────────────────────────

    def add_cost_tier(
        self,
        max_budget: float,
        model: str,
        name: str = "",
    ) -> None:
        """Register a cost-budget routing tier.

        Requests with ``cost_budget <= max_budget`` are routed to *model*.
        Tiers are evaluated from cheapest to most expensive.

        Args:
            max_budget: Upper bound (inclusive) for this tier in USD.
            model: Model to route to when budget is within this tier.
            name: Optional rule name (auto-generated if empty).
        """
        rule_name = name or f"cost_{max_budget}"
        self._cost_tiers.append((max_budget, model, rule_name))
        self._cost_tiers.sort(key=lambda t: t[0])

    def add_latency_tier(
        self,
        max_latency_ms: float,
        model: str,
        name: str = "",
    ) -> None:
        """Register a latency-SLA routing tier.

        Requests with ``max_latency_ms <= max_latency_ms`` are routed to *model*.
        Tiers are evaluated from fastest to slowest.

        Args:
            max_latency_ms: Upper bound (inclusive) for this tier in ms.
            model: Model to route to when SLA is within this tier.
            name: Optional rule name (auto-generated if empty).
        """
        rule_name = name or f"latency_{max_latency_ms}"
        self._latency_tiers.append((max_latency_ms, model, rule_name))
        self._latency_tiers.sort(key=lambda t: t[0])

    def add_language_route(self, language: str, model: str, name: str = "") -> None:
        """Route a specific language to a model.

        Args:
            language: ISO 639-1 code (e.g. "en", "zh", "es").
            model: Model to route to for this language.
            name: Optional rule name.
        """
        rule_name = name or f"lang_{language}"
        self._language_routes[language.lower()] = (model, rule_name)

    def add_length_tier(
        self,
        max_tokens: int,
        model: str,
        name: str = "",
    ) -> None:
        """Register an input-length routing tier.

        Requests with ``input_tokens <= max_tokens`` are routed to *model*.
        Tiers are evaluated from shortest to longest.

        Args:
            max_tokens: Upper bound (inclusive) for this tier.
            model: Model to route to when input length is within this tier.
            name: Optional rule name.
        """
        rule_name = name or f"length_{max_tokens}"
        self._length_tiers.append((max_tokens, model, rule_name))
        self._length_tiers.sort(key=lambda t: t[0])

    def add_tool_call_route(self, model: str, name: str = "") -> None:
        """Route requests containing tool calls to a specific model.

        Args:
            model: Model optimized for function/tool calling.
            name: Optional rule name.
        """
        rule_name = name or "tool_calls"
        self._tool_call_model = model
        self._tool_call_rule = rule_name

    def set_audit_callback(
        self,
        callback: Callable[[str, str, str, str, float], None] | None,
    ) -> None:
        """Set an audit callback invoked on every routing decision.

        Args:
            callback: ``(request_id, query_preview, rule_name, model, confidence)``
                called after each routing decision.  Pass None to disable.
        """
        self._audit_callback = callback

    def _record_audit(
        self,
        request_id: str,
        query_text: str,
        rule_name: str,
        model: str,
        confidence: float,
    ) -> None:
        """Invoke the audit callback if set."""
        if self._audit_callback is not None:
            try:
                self._audit_callback(
                    request_id,
                    query_text[:100],
                    rule_name,
                    model,
                    confidence,
                )
            except Exception:
                logger.debug("Audit callback failed", exc_info=True)

    # ── Core routing ───────────────────────────────────────────────────────

    def set_default_model(self, model: str) -> None:
        """Set the default fallback model."""
        self._default_model = model

    def register_hybrid_name(self, name: str) -> None:
        """Register a virtual model name that resolves via routing rules.

        Clients can send ``model="hybrid"`` and the router will select
        the actual backend model based on query content.
        """
        if name not in self._hybrid_names:
            self._hybrid_names.append(name)

    def list_hybrid_models(self) -> list[str]:
        """Return registered hybrid/virtual model names."""
        return list(self._hybrid_names)

    def is_hybrid_model(self, model: str) -> bool:
        """Check if a model name is a registered hybrid name."""
        return model in self._hybrid_names

    def resolve(
        self,
        text: str,
        available_models: list[str] | None = None,
    ) -> str:
        """Resolve a query string to the best-suited model name.

        Args:
            text: The query text to classify.
            available_models: Optional list of currently-loaded model names.
                If provided, rules whose target_model is not in this list
                are skipped and the fallback chain continues.

        Returns:
            The selected model name (str).
        """
        start = time.monotonic()
        with self._lock:
            self._stats["total_routes"] += 1

        if not text:
            return self._fallback_model(available_models)

        text_lower = text.lower()

        # Try rules in priority order
        for rule in self._rules:
            if self._match_rule(rule, text_lower):
                target = rule.target_model
                if available_models is not None and target not in available_models:
                    continue
                elapsed_ms = (time.monotonic() - start) * 1000
                with self._lock:
                    self._stats["routes_by_model"][target] = (
                        self._stats["routes_by_model"].get(target, 0) + 1
                    )
                    self._stats["routes_by_rule"][rule.name] = (
                        self._stats["routes_by_rule"].get(rule.name, 0) + 1
                    )
                logger.debug(
                    f"Model route: '{rule.name}' -> {target} ({elapsed_ms:.1f}ms)"
                )
                self._record_audit("", text, rule.name, target, 0.9)
                return target

        # Try workload classification
        workload = self._classify_workload(text_lower)
        if workload:
            for rule in self._rules:
                if rule.match_type == "workload" and rule.pattern == workload:
                    target = rule.target_model
                    if available_models is not None and target not in available_models:
                        continue
                    with self._lock:
                        self._stats["routes_by_model"][target] = (
                            self._stats["routes_by_model"].get(target, 0) + 1
                        )
                        self._stats["routes_by_rule"][rule.name] = (
                            self._stats["routes_by_rule"].get(rule.name, 0) + 1
                        )
                    self._record_audit("", text, rule.name, target, 0.8)
                    return target

        fallback = self._fallback_model(available_models)
        self._record_audit("", text, "default", fallback, 0.5)
        return fallback

    def route(
        self,
        messages: list[dict[str, str]],
        available_models: list[str] | None = None,
        analysis_depth: str = "last",
    ) -> RouteMatch:
        """Route a conversation to the best-suited model.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            available_models: Optional filter for currently-loaded models.
            analysis_depth: How much conversation context to analyze:
                - "last": Only the last user message (default, fastest).
                - "system+last": System prompt + last user message.
                - "full": All messages joined together.

        Returns:
            RouteMatch with the selected model and routing metadata.
        """
        start = time.monotonic()
        with self._lock:
            self._stats["total_routes"] += 1

        if not messages:
            return self._default_match(start)

        # Extract text based on analysis depth
        text = self._extract_text(messages, analysis_depth)

        if not text:
            return self._default_match(start)

        text_lower = text.lower()

        # Try rules in priority order
        for rule in self._rules:
            if self._match_rule(rule, text_lower):
                target = rule.target_model
                if available_models is not None and target not in available_models:
                    continue
                elapsed_ms = (time.monotonic() - start) * 1000
                with self._lock:
                    self._stats["routes_by_model"][target] = (
                        self._stats["routes_by_model"].get(target, 0) + 1
                    )
                    self._stats["routes_by_rule"][rule.name] = (
                        self._stats["routes_by_rule"].get(rule.name, 0) + 1
                    )
                logger.debug(
                    f"Model route: '{rule.name}' -> {target} ({elapsed_ms:.1f}ms)"
                )
                return RouteMatch(
                    model=target,
                    rule_name=rule.name,
                    confidence=self._compute_confidence(rule, text_lower),
                    latency_ms=elapsed_ms,
                )

        # Try workload classification
        workload = self._classify_workload(text_lower)
        if workload:
            for rule in self._rules:
                if rule.match_type == "workload" and rule.pattern == workload:
                    target = rule.target_model
                    if available_models is not None and target not in available_models:
                        continue
                    elapsed_ms = (time.monotonic() - start) * 1000
                    with self._lock:
                        self._stats["routes_by_model"][target] = (
                            self._stats["routes_by_model"].get(target, 0) + 1
                        )
                    return RouteMatch(
                        model=target,
                        rule_name=rule.name,
                        confidence=self._compute_confidence(rule, text_lower),
                        latency_ms=elapsed_ms,
                    )

        return self._default_match(start)

    def route_with_context(
        self,
        messages: list[dict[str, str]],
        ctx: RoutingContext | None = None,
        available_models: list[str] | None = None,
        analysis_depth: str = "last",
    ) -> RouteMatch:
        """Route with multi-attribute context (cost, latency, language, etc.).

        Evaluation order:
        1. Content-based rules (keyword/regex/workload) — same as ``route()``.
        2. Tool-call routing (if ``ctx.has_tool_calls``).
        3. Language routing (if ``ctx.language``).
        4. Cost-budget tiers (if ``ctx.cost_budget``).
        5. Latency-SLA tiers (if ``ctx.max_latency_ms``).
        6. Input-length tiers (if ``ctx.input_tokens``).
        7. Default fallback.

        Args:
            messages: Conversation messages.
            ctx: Optional routing context with multi-attribute signals.
            available_models: Filter for currently-loaded models.
            analysis_depth: Conversation context depth ("last", "system+last", "full").

        Returns:
            RouteMatch with the selected model and routing metadata.
        """
        if ctx is None:
            return self.route(messages, available_models, analysis_depth)

        start = time.monotonic()
        with self._lock:
            self._stats["total_routes"] += 1

        if not messages:
            return self._default_match(start)

        text = self._extract_text(messages, analysis_depth)
        if not text:
            return self._default_match(start)

        text_lower = text.lower()

        # 1. Content-based rules (same as route())
        for rule in self._rules:
            if self._match_rule(rule, text_lower):
                target = rule.target_model
                if available_models is not None and target not in available_models:
                    continue
                elapsed_ms = (time.monotonic() - start) * 1000
                with self._lock:
                    self._stats["routes_by_model"][target] = (
                        self._stats["routes_by_model"].get(target, 0) + 1
                    )
                    self._stats["routes_by_rule"][rule.name] = (
                        self._stats["routes_by_rule"].get(rule.name, 0) + 1
                    )
                return RouteMatch(
                    model=target,
                    rule_name=rule.name,
                    confidence=self._compute_confidence(rule, text_lower),
                    latency_ms=elapsed_ms,
                )

        # Workload classification
        workload = self._classify_workload(text_lower)
        if workload:
            for rule in self._rules:
                if rule.match_type == "workload" and rule.pattern == workload:
                    target = rule.target_model
                    if available_models is not None and target not in available_models:
                        continue
                    elapsed_ms = (time.monotonic() - start) * 1000
                    with self._lock:
                        self._stats["routes_by_model"][target] = (
                            self._stats["routes_by_model"].get(target, 0) + 1
                        )
                    return RouteMatch(
                        model=target,
                        rule_name=rule.name,
                        confidence=self._compute_confidence(rule, text_lower),
                        latency_ms=elapsed_ms,
                    )

        # 2. Tool-call routing
        if ctx.has_tool_calls and self._tool_call_model:
            target = self._tool_call_model
            if available_models is None or target in available_models:
                elapsed_ms = (time.monotonic() - start) * 1000
                with self._lock:
                    self._stats["routes_by_model"][target] = (
                        self._stats["routes_by_model"].get(target, 0) + 1
                    )
                    self._stats["routes_by_rule"][self._tool_call_rule] = (
                        self._stats["routes_by_rule"].get(self._tool_call_rule, 0) + 1
                    )
                return RouteMatch(
                    model=target, rule_name=self._tool_call_rule,
                    confidence=0.85, latency_ms=elapsed_ms,
                )

        # 3. Language routing
        if ctx.language:
            lang = ctx.language.lower()
            if lang in self._language_routes:
                target, rule_name = self._language_routes[lang]
                if available_models is None or target in available_models:
                    elapsed_ms = (time.monotonic() - start) * 1000
                    with self._lock:
                        self._stats["routes_by_model"][target] = (
                            self._stats["routes_by_model"].get(target, 0) + 1
                        )
                        self._stats["routes_by_rule"][rule_name] = (
                            self._stats["routes_by_rule"].get(rule_name, 0) + 1
                        )
                    return RouteMatch(
                        model=target, rule_name=rule_name,
                        confidence=0.9, latency_ms=elapsed_ms,
                    )

        # 4. Cost-budget tiers
        if ctx.cost_budget is not None:
            for max_budget, model, rule_name in self._cost_tiers:
                budget_ok = ctx.cost_budget <= max_budget
                available = available_models is None or model in available_models
                if budget_ok and available:
                    elapsed_ms = (time.monotonic() - start) * 1000
                    with self._lock:
                        self._stats["routes_by_model"][model] = (
                            self._stats["routes_by_model"].get(model, 0) + 1
                        )
                        self._stats["routes_by_rule"][rule_name] = (
                            self._stats["routes_by_rule"].get(rule_name, 0) + 1
                        )
                    return RouteMatch(
                        model=model, rule_name=rule_name,
                        confidence=0.95, latency_ms=elapsed_ms,
                    )

        # 5. Latency-SLA tiers
        if ctx.max_latency_ms is not None:
            for max_ms, model, rule_name in self._latency_tiers:
                sla_ok = ctx.max_latency_ms <= max_ms
                available = available_models is None or model in available_models
                if sla_ok and available:
                    elapsed_ms = (time.monotonic() - start) * 1000
                    with self._lock:
                        self._stats["routes_by_model"][model] = (
                            self._stats["routes_by_model"].get(model, 0) + 1
                        )
                        self._stats["routes_by_rule"][rule_name] = (
                            self._stats["routes_by_rule"].get(rule_name, 0) + 1
                        )
                    return RouteMatch(
                        model=model, rule_name=rule_name,
                        confidence=0.95, latency_ms=elapsed_ms,
                    )

        # 6. Input-length tiers
        if ctx.input_tokens is not None:
            for max_tokens, model, rule_name in self._length_tiers:
                len_ok = ctx.input_tokens <= max_tokens
                available = available_models is None or model in available_models
                if len_ok and available:
                    elapsed_ms = (time.monotonic() - start) * 1000
                    with self._lock:
                        self._stats["routes_by_model"][model] = (
                            self._stats["routes_by_model"].get(model, 0) + 1
                        )
                        self._stats["routes_by_rule"][rule_name] = (
                            self._stats["routes_by_rule"].get(rule_name, 0) + 1
                        )
                    return RouteMatch(
                        model=model, rule_name=rule_name,
                        confidence=0.9, latency_ms=elapsed_ms,
                    )

        # 7. Default fallback
        return self._default_match(start)

    def _extract_text(self, messages: list[dict], depth: str) -> str:
        """Extract text from messages according to analysis depth."""
        if depth == "full":
            parts = [
                m.get("content", "")
                for m in messages
                if m.get("content")
            ]
            return " ".join(parts)

        if depth == "system+last":
            system = ""
            last_user = ""
            for m in messages:
                role = m.get("role", "")
                content = m.get("content", "")
                if role == "system" and content:
                    system = content
                elif role == "user" and content:
                    last_user = content
            if system and last_user:
                return f"{system} {last_user}"
            return last_user or system

        # Default: "last" — only the last user message
        for m in reversed(messages):
            if m.get("role") == "user" and m.get("content"):
                return m["content"]
        return ""

    def _match_rule(self, rule: RouteRule, text: str) -> bool:
        """Check if text matches a routing rule."""
        text_lower = text.lower()

        if rule.match_type == "keyword":
            return rule.pattern.lower() in text_lower

        if rule.match_type == "regex":
            try:
                # H-16: ReDoS protection — reject patterns with catastrophic backtracking risk
                # Patterns longer than 100 chars with nested quantifiers are rejected
                if len(rule.pattern) > 100 and re.search(r'\(.*[+*].*\)[+*]', rule.pattern):
                    logger.warning(f"Rejected potentially dangerous regex in rule '{rule.name}'")
                    return False
                return bool(re.search(rule.pattern, text, re.IGNORECASE))
            except re.error:
                logger.warning(f"Invalid regex pattern in rule '{rule.name}': {rule.pattern}")
                return False

        if rule.match_type == "workload":
            workload = self._classify_workload(text_lower)
            return workload == rule.pattern

        return False

    def _classify_workload(self, text_lower: str) -> str | None:
        """Classify the workload type of a query.

        Args:
            text_lower: Already-lowered query text.
        """
        scores: dict[str, int] = {}

        for workload, patterns in self._workload_patterns.items():
            score = 0
            # Keyword matching
            for kw in patterns.get("keywords", []):
                if kw in text_lower:
                    score += 1
            # Regex matching
            for rx in patterns.get("regex", []):
                try:
                    if re.search(rx, text_lower, re.IGNORECASE):
                        score += 2
                except re.error:
                    logger.warning(f"Invalid workload regex for '{workload}': {rx}")
            if score > 0:
                scores[workload] = score

        if not scores:
            return None
        return max(scores, key=scores.get)

    def _compute_confidence(self, rule: RouteRule, text_lower: str) -> float:
        """Compute confidence score based on match signal density.

        Args:
            text_lower: Already-lowered query text.

        Returns a value in [0.4, 1.0] reflecting how strongly the text
        matches the rule. More keyword hits relative to text length
        produces higher confidence.
        """
        words = max(len(text_lower.split()), 1)
        hits = 0

        if rule.match_type == "keyword":
            pattern_lower = rule.pattern.lower()
            hits = text_lower.count(pattern_lower)
        elif rule.match_type == "regex":
            try:
                hits = len(re.findall(rule.pattern, text_lower, re.IGNORECASE))
            except re.error:
                hits = 0
        elif rule.match_type == "workload":
            patterns = self._workload_patterns.get(rule.pattern, {})
            for kw in patterns.get("keywords", []):
                if kw in text_lower:
                    hits += 1
            for rx in patterns.get("regex", []):
                try:
                    hits += len(re.findall(rx, text_lower, re.IGNORECASE))
                except re.error:
                    logger.warning(f"Invalid workload regex for confidence: {rx}")

        # Saturating curve: 1 hit -> ~0.4, 2 -> ~0.57, 3 -> ~0.67, 5 -> ~0.77
        # Normalized by text length so longer texts with same hits get lower confidence
        effective = hits * (8.0 / max(words, 4))
        confidence = effective / (effective + 2.0)
        return round(max(0.4, min(confidence, 1.0)), 3)

    def _fallback_model(self, available_models: list[str] | None = None) -> str:
        """Return the fallback model, respecting available_models filter."""
        if available_models is not None and self._default_model not in available_models:
            if available_models:
                return available_models[0]
            return self._default_model
        return self._default_model

    def _default_match(self, start: float) -> RouteMatch:
        """Return a match for the default model."""
        elapsed_ms = (time.monotonic() - start) * 1000
        return RouteMatch(
            model=self._default_model,
            rule_name="default",
            confidence=0.3,
            latency_ms=elapsed_ms,
        )

    @property
    def stats(self) -> dict:
        with self._lock:
            return dict(self._stats)

    def reset_stats(self) -> None:
        """Reset all routing statistics."""
        with self._lock:
            self._stats = {
                "total_routes": 0,
                "routes_by_model": {},
                "routes_by_rule": {},
            }

    def to_dict(self) -> dict:
        """Serialize router configuration to a dict."""
        return {
            "enabled": bool(self._rules or self._default_model),
            "default_model": self._default_model,
            "hybrid_models": list(self._hybrid_names),
            "rules": [
                {
                    "name": r.name,
                    "match_type": r.match_type,
                    "pattern": r.pattern,
                    "target_model": r.target_model,
                    "priority": r.priority,
                }
                for r in self._rules
            ],
        }

    @classmethod
    def from_config(cls, config: ChatRouterSettings) -> ModelRouter:
        """Create a ModelRouter from ChatRouterSettings config."""
        return cls(settings=config)

    def create_learning_router(
        self,
        models: list[str] | None = None,
        epsilon: float = 0.15,
        policy_path: str | None = None,
    ) -> LearningRouter:
        """Create a :class:`LearningRouter` that wraps this router.

        The learning router uses online RL (contextual bandits) to improve
        model selection over time based on reward signals.

        Args:
            models: Model names to choose from.  Defaults to rule targets + default.
            epsilon: Exploration probability (0.0–1.0).
            policy_path: Path to load/save learned policies (JSON).

        Returns:
            A LearningRouter instance.
        """
        from distllm.core.learning_router import LearningRouter

        if models is None:
            models = list({r.target_model for r in self._rules})
            if self._default_model and self._default_model not in models:
                models.append(self._default_model)

        lr = LearningRouter(
            base_router=self,
            models=models,
            epsilon=epsilon,
        )

        if policy_path:
            lr.load_policy(policy_path)

        return lr

    def route_with_region(
        self,
        messages: list[dict[str, str]],
        available_models: list[str] | None = None,
        available_regions: list[str] | None = None,
        cross_cloud_router: object | None = None,
    ) -> tuple[str, str]:
        """Route to model then select region via CrossCloudRouter.

        Args:
            messages: Conversation messages.
            available_models: Models currently loaded.
            available_regions: Regions available for inference.
            cross_cloud_router: A CrossCloudRouter instance for region selection.

        Returns:
            Tuple of (model_name, region_name).
        """
        match = self.route(messages, available_models)
        model = match.model

        region = ""
        if cross_cloud_router is not None and available_regions:
            if hasattr(cross_cloud_router, "select_region"):
                try:
                    region = cross_cloud_router.select_region(
                        model=model,
                        regions=available_regions,
                    )
                except Exception:
                    region = available_regions[0] if available_regions else ""
            else:
                region = available_regions[0] if available_regions else ""

        return model, region
