"""SEC-A4 repro: webhook SSRF via unvalidated URL registration + delivery.

BEFORE the fix (verified 2026-08-24): both targets were accepted at
registration ("Webhook registered: ... -> http://169.254.169.254/...")
and dispatched by the delivery engine with no validation anywhere.

AFTER the fix: registration is rejected at BOTH layers. Run:

    python tests/security/repro_webhook_ssrf.py
"""

import asyncio

SSRF_TARGETS = (
    "http://169.254.169.254/latest/meta-data/",
    "http://localhost:8000/admin/v1/nodes",
)


def main() -> None:
    print("== Repro: webhook SSRF (SEC-A4) — expecting rejections ==")

    # Layer 1: route-level registration guard.
    from fastapi import HTTPException
    from distllm.api.routes.webhooks import WebhookCreate, create_webhook

    for target in SSRF_TARGETS:
        try:
            asyncio.run(create_webhook(WebhookCreate(
                url=target, secret="x" * 16, events=["batch.completed"],
            )))
        except HTTPException as exc:
            assert exc.status_code == 400, exc.status_code
            print(f"ROUTE REJECTED ({exc.status_code}): {target}")
        else:
            raise AssertionError(f"VULNERABLE: route accepted {target}")

    # Layer 2: delivery-engine registration + dispatch guards.
    from distllm.api.webhooks.delivery import UnsafeWebhookURLError, WebhookManager

    mgr = WebhookManager()
    for target in SSRF_TARGETS:
        try:
            mgr.register(target, {"job.completed"}, "sekret")
        except UnsafeWebhookURLError:
            print(f"DELIVERY-MGR REJECTED: {target}")
        else:
            raise AssertionError(f"VULNERABLE: delivery manager accepted {target}")

    matched = mgr.dispatch("job.completed", {"job_id": "repro"})
    assert matched == [], matched
    print("dispatch matched 0 webhooks — nothing registered, nothing delivered")

    print("OK: all SSRF targets rejected at every layer.")


if __name__ == "__main__":
    main()
