"""Regression tests for the latest review-table fixes:

- M17  grammar_decoder: GBNFFSM now compiles its real DFA at construction
       (the DFA path in get_logits_mask was previously dead code).
- M14  provider_health: fail-closed region health (unknown region -> unhealthy)
       and no redirect-following in the httpx probe (no SSRF drift).
- MR   model_router: route() and route_with_context() share content-rule logic
       (dedup); tool-call tier evaluated AFTER cost/latency tiers.
"""

import time
from unittest import mock

from distllm.core.grammar_decoder import GBNFFSM
from distllm.core.provider_health import ProviderHealthProber
from distllm.core.model_router import (
    ModelRouter,
    RouteRule,
    RoutingContext,
)


# ── M17: grammar_decoder DFA compiles correctly ──
# _extract_target now joins unquoted string-literal tokens from the parser,
# so compile_to_dfa() builds a correct linear DFA and the fast masking path
# is live (was dead code: _compiled was left False at construction).

def test_gbnffsm_compiles_correct_dfa():
    fsm = GBNFFSM('root ::= "hello" " " "world"')
    assert fsm._compiled is True
    assert fsm._target == "hello world", f"_target wrong: {fsm._target!r}"
    # DFA is a correct linear chain: each position allows exactly one byte.
    assert fsm.get_allowed_bytes() == {ord("h")}
    for i, b in enumerate(b"hello world"):
        assert fsm.get_allowed_bytes() == {b}, f"pos {i}: expected {[b]}, got {fsm.get_allowed_bytes()}"
        fsm.transition(b)
    assert fsm.is_accepting()


def test_gbnffsm_resolves_rule_references():
    fsm = GBNFFSM('greeting ::= "hi"\nroot ::= greeting " there"')
    assert fsm._target == "hi there"
    assert fsm._compiled is True


# ── M14: provider_health fail-closed + no redirects ──

def test_unknown_region_is_unhealthy_fail_closed():
    prober = ProviderHealthProber()
    # Never probed -> must be UNHEALTHY (fail-closed), not blindly True.
    assert prober.is_healthy("aws", "us-east-1") is False


def test_probe_does_not_follow_redirects():
    prober = ProviderHealthProber()
    prober._get_endpoint = lambda p, r: f"https://{p}.example.com/health"

    captured = {}
    fake_resp = mock.Mock()
    fake_resp.status_code = 200

    def fake_get(url, timeout=None, follow_redirects=False):
        captured["follow_redirects"] = follow_redirects
        return fake_resp

    with mock.patch.dict("sys.modules", {"httpx": mock.Mock(get=fake_get)}):
        # Patch the in-function `import httpx` by injecting into the module's
        # globals via the prober's probe path.
        import distllm.core.provider_health as ph

        real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        import builtins

        orig_import = builtins.__import__

        def _imp(name, *a, **k):
            if name == "httpx" or name.endswith(".httpx"):
                return mock.Mock(get=fake_get)
            return orig_import(name, *a, **k)

        builtins.__import__ = _imp
        try:
            prober._probe_region("aws", "us-east-1")
        finally:
            builtins.__import__ = orig_import

    assert captured.get("follow_redirects") is False, "probe must not follow redirects"


# ── model_router: dedup + tool-call ordered last ──

def _router_with_tiers():
    r = ModelRouter()
    r.add_cost_tier(max_budget=0.01, model="cheap-model", name="cost")
    r.add_latency_tier(max_latency_ms=500, model="fast-model", name="lat")
    r.add_tool_call_route(model="tool-model", name="tool")
    return r


def test_route_and_route_with_context_share_content_rules():
    r = _router_with_tiers()
    r.add_rule(RouteRule(name="code", match_type="keyword", pattern="debug", target_model="code-model"))
    ctx = RoutingContext(has_tool_calls=False)
    a = r.route([{"role": "user", "content": "debug this function"}])
    b = r.route_with_context([{"role": "user", "content": "debug this function"}], ctx=ctx)
    assert a.model == "code-model" == b.model


def test_tool_call_tier_ordered_after_cost_tier():
    """A request that has tool calls AND a tight cost budget must honor the
    cost tier (cheaper model), not the generic tool-call model."""
    r = _router_with_tiers()
    ctx = RoutingContext(has_tool_calls=True, cost_budget=0.005)
    m = r.route_with_context([{"role": "user", "content": "hello"}], ctx=ctx)
    # cost tier (0.01 >= 0.005) wins over tool-call route
    assert m.model == "cheap-model", f"expected cost tier, got {m.model}"
    assert m.rule_name == "cost"
