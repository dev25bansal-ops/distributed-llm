"""SAML 2.0 authentication provider.

Supports any SAML 2.0 IdP (Okta, Azure AD, OneLogin, Keycloak).
Uses ``pysaml2`` when available; falls back to metadata-only mode.
"""

from __future__ import annotations

from loguru import logger

from .models import SSOUserInfo


class SAMLHandler:
    """SAML 2.0 authentication via SAML HTTP Artifact or POST binding.

    Uses pysaml2 when available. Falls back to metadata-only mode
    for manual IdP configuration.
    """

    def __init__(self, metadata_url: str, callback_url: str, entity_id: str = "distllm"):
        self._metadata_url = metadata_url
        self._callback_url = callback_url
        self._entity_id = entity_id
        self._client = None
        self._initialize()

    def _initialize(self) -> None:
        """Try to initialize pysaml2 for full SAML support."""
        try:
            from saml2 import BINDING_HTTP_POST, BINDING_HTTP_REDIRECT
            from saml2.client import Saml2Client
            from saml2.config import Config as Saml2Config

            config = Saml2Config()
            config.setattr("entityid", self._entity_id)
            config.setattr("metadata", {"remote": [{"url": self._metadata_url}]})
            config.setattr("service", {
                "sp": {
                    "endpoints": {
                        "assertion_consumer_service": [
                            (self._callback_url, BINDING_HTTP_POST),
                        ],
                    },
                    # SECURITY: Disable unsolicited assertions (prevents SAML response injection)
                    # Require signed authn requests (prevents request forgery)
                    "allow_unsolicited": False,
                    "authn_requests_signed": True,
                },
            })
            self._client = Saml2Client(config=config)
            logger.info("SAML 2.0 client initialized with pysaml2")
        except ImportError:
            logger.warning(
                "pysaml2 not installed. SAML will use metadata-only mode. "
                "Install with: pip install pysaml2"
            )

    def get_login_url(self) -> str:
        """Generate the SAML login redirect URL."""
        if self._client is None:
            return self._callback_url

        try:
            from saml2 import BINDING_HTTP_REDIRECT
            _, info = self._client.prepare_for_authenticate(
                relay_state="",
                binding=BINDING_HTTP_REDIRECT,
            )
            headers = dict(info.get("headers", []))
            return headers.get("Location", self._callback_url)
        except Exception as e:
            logger.error(f"SAML login URL generation failed: {e}")
            return self._callback_url

    def handle_callback(self, saml_response: str) -> SSOUserInfo | None:
        """Handle the SAML assertion response.

        Catches specific SAML errors to distinguish authentication failures
        from processing errors.  Broad exceptions are still caught for
        robustness but logged at ERROR level with full detail so security
        monitoring can detect attack patterns.
        """
        try:
            authn_response = self._client.parse_authn_request_response(
                saml_response,
                self._client.config.getattr("endpoints")["assertion_consumer_service"][0][1],
            )
            attrs = authn_response.get_identity()
            return SSOUserInfo(
                sub=authn_response.get_subject().text or "",
                email=attrs.get("email", [""])[0] if isinstance(attrs.get("email"), list) else attrs.get("email", ""),
                name=attrs.get("name", [""])[0] if isinstance(attrs.get("name"), list) else attrs.get("name", ""),
                roles=attrs.get("roles", attrs.get("Role", [])),
                groups=attrs.get("groups", attrs.get("Group", [])),
                provider="saml",
                raw_attributes=attrs,
            )
        except ImportError:
            logger.error("pysaml2 not installed — cannot process SAML response")
            return None
        except (ValueError, AttributeError, TypeError) as e:
            logger.error(f"SAML response parsing error: {e}")
            return None
        except Exception as e:
            logger.error(f"SAML callback handling failed (unexpected): {e}", exc_info=True)
            return None
