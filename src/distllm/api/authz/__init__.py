"""OPA/Rego authorization engine for the distributed-LLM API (API Cat5).

Public surface:

    from distllm.api.authz import authorize, load_policy, OPA_AVAILABLE, Decision

``authorize(input)`` evaluates an allow/deny decision for
``(subject, action, resource)``.  When the ``opa`` binary or an OPA server is
available it shells out / issues an HTTP request against a bundled ``.rego``
policy; otherwise it transparently falls back to a pure-Python policy evaluator
that implements the *same* allow/deny contract from a policy dict.
"""

from __future__ import annotations

from distllm.api.authz.opa import (
    Decision,
    OPA_AVAILABLE,
    OPAClient,
    authorize,
    load_policy,
)

__all__ = [
    "Decision",
    "OPA_AVAILABLE",
    "OPAClient",
    "authorize",
    "load_policy",
]
