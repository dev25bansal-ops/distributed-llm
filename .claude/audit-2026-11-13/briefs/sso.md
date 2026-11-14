# Fix brief: sso

Audit findings for the files in your exclusive scope. For EACH finding:
reproduce-or-refute first (run the check command or read the cited code
carefully), then either FIX it or record an explicit refutation. Never
leave a finding unaddressed. After fixing, extend or add a test in the
named test file and run it. Do NOT commit; leave changes in the tree.

## [high] src/distllm/api/middleware.py:466
**Summary**: The shipped server blocks /v1/auth/{token,refresh,revoke} with a 401: SsoMiddleware explicitly skips these routes (AUTH_ROUTES, sso_middleware.py:67,396) but AuthMiddleware's exemption list omits them, so the documented SSO code-exchange and the only revocation endpoint require a pre-existing API key.
**Failure scenario**: Enterprise user completes IdP login and POSTs {code,state} to /v1/auth/token on distllm.api.server.app; the request never reaches the handler — inner AuthMiddleware returns 401 "Unauthorized: invalid API key" (real app repro: 401 401; with a valid admin API key the same request reaches the route and returns 400 missing_code, proving the route works but is gate-blocked). Consequence: the SSO login flow is unusable by users who by definition have no API key, and a holder of an SSO JWT cannot call /v1/auth/revoke (401), so holder-initiated revocation is impossible.
**Check command**: `cd D:\distributed-llm\.claude\worktrees\bubbly-pondering-whisper && PYTHONPATH=src D:/distributed-llm/.venv/Scripts/python.exe -c "from fastapi.testclient import TestClient; from distllm.api.server import app; r=TestClient(app).post('/v1/auth/token', json={}); print(r.status_code, r.json()['error']['code'])"  -> 401 401`
**Reproduced by auditor**: True

## [high] src/distllm/api/sso_auth.py:319
**Summary**: The OIDC/OAuth2 callback requires a server-issued state, but the state/PKCE/nonce stores are populated only by get_login_url(), and nothing in the shipped server ever calls it (no login/bootstrap route exists) — so every real callback and /v1/auth/token exchange is rejected 'state not found'.
**Failure scenario**: A fresh env-configured OIDCHandler (as created in production by SSOAuthHandler._initialize): handle_callback('code', expected_state='<state from the IdP redirect>') returns None because _state_store/_pkce_store are empty; POST /v1/auth/token likewise can never succeed for oidc/oauth2, since no API yields a state the stores know (route demands state, returns 400 missing_state otherwise). grep proves zero get_login_url callers outside the sso_auth/sso_middleware definitions. In multi-worker deployments the per-process stores would also lose state across workers even if a caller existed.
**Check command**: `cd D:\distributed-llm\.claude\worktrees\bubbly-pondering-whisper && PYTHONPATH=src D:/distributed-llm/.venv/Scripts/python.exe -c "from distllm.api.sso_auth import OIDCHandler; h=OIDCHandler.__new__(OIDCHandler); h._state_store={}; h._nonce_store={}; h._pkce_store={}; print('callback result:', h.handle_callback('code', expected_state='s'))"  -> callback result: None;  and: grep -rn "get_login_url" src/ --include=*.py | grep -v "sso_auth.py\|sso_middleware.py"  -> no matches`
**Reproduced by auditor**: True

## [high] src/distllm/api/sso_middleware.py:435
**Summary**: Revoking an externally-validated (OIDC) Bearer token is a silent no-op: revoke_token_str's documented hash-based fallback (lines 340-356) stores sha256(token) in TokenBlacklist, but dispatch()'s external-provider path never consults the blacklist — only locally-decoded JWTs are checked (line 407-421).
**Failure scenario**: Stolen OIDC access token is revoked via the only reachable path (mw.revoke_token_str(token) — the HTTP route is itself blocked, see finding 1); the next request with that token is validated by the registered OIDCHandler via JWKS and served normally: repro shows GET /echo with the revoked Bearer token -> 200 {'auth': 'sso'}. The revocation guarantee the endpoint documents does not hold for external tokens.
**Check command**: `cd D:\distributed-llm\.claude\worktrees\bubbly-pondering-whisper && PYTHONPATH=src D:/distributed-llm/.venv/Scripts/python.exe -c "
import types
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from distllm.api.sso_auth import SSOUserInfo
from distllm.api.sso_middleware import setup_sso
async def v(t): return SSOUserInfo(sub='v', provider='oidc') if t == 'tok' else None
a = FastAPI()
@a.get('/echo')
async def e(r: Request): return {'a': getattr(r.state, 'auth_method', None)}
setup_sso(a); c = TestClient(a); c.get('/echo')
mw = a.state._sso_middleware
mw.register_provider('p', types.SimpleNamespace(_authority='x', async_validate_token=v))
mw.revoke_token_str('tok')
print('revoked-token-request:', c.get('/echo', headers={'Authorization': 'Bearer tok'}).status_code)"  -> revoked-token-request: 200`
**Reproduced by auditor**: True

## [med] src/distllm/api/sso_auth.py:560
**Summary**: The 'MANDATORY (H-13)' id_token signature verification and the OIDC nonce replay check run only when the token response happens to include an id_token; a response without one skips both and authenticates from the userinfo call, contradicting the comment 'an id_token must always be cryptographically validated, and the exchange fails closed'.
**Failure scenario**: IdP (or a misconfigured OIDC app that silently dropped the scope, or an OAuth2-compat endpoint) returns 200 {access_token} with no id_token/empty id_token: handle_callback skips validate_token and the nonce comparison — the stored nonce is never consumed, replay protection is inert — and returns identity from whatever the userinfo endpoint says. Repro (mocked httpx): token 200 {access_token}, userinfo 200 {'sub':'forged'} -> accepted sub 'forged'; /v1/auth/token then mints a local HS256 JWT with that identity and role mapping.
**Check command**: `cd D:\distributed-llm\.claude\worktrees\bubbly-pondering-whisper && PYTHONPATH=src D:/distributed-llm/.venv/Scripts/python.exe -c "
import time
from unittest.mock import patch
from distllm.api.sso_auth import OIDCHandler
h = OIDCHandler.__new__(OIDCHandler)
h._state_store={'s':time.time()+600}; h._nonce_store={'s':{'nonce':'n','expires':time.time()+600}}; h._pkce_store={'s':'v'}
h._token_endpoint='https://x/t'; h._client_id='c'; h._client_secret='s'; h._callback_url='https://x/cb'
h._userinfo_endpoint='https://x/u'; h._jwks_url=None; h._discovered_jwks_url=None; h._allow_unverified_id_token=False
class R:
    status_code=200; text=''
    def __init__(self,b): self._b=b
    def json(self): return self._b
with patch('httpx.post', lambda *a, **k: R({'access_token':'at'})), patch('httpx.get', lambda *a, **k: R({'sub':'forged'})):
    u = h.handle_callback('code', expected_state='s')
print('accepted-sub:', u and u.sub)"  -> accepted-sub: forged`
**Reproduced by auditor**: True

## [med] src/distllm/api/sso_middleware.py:377
**Summary**: SAML and OAuth2 are documented first-class providers but are unusable through the middleware: _auto_detect_providers registers only OIDCHandler, register_provider crashes on SAMLHandler/GenericOAuth2Handler (accesses handler._authority, line 323), and SAMLHandler.handle_callback lacks the expected_state keyword the route passes (line 611).
**Failure scenario**: A) DISTLLM_SSO_PROVIDER=oauth2 fully configured (client id/secret/authorize/token/userinfo): get_sso_handler().is_enabled is True yet POST /v1/auth/token {'provider':'default'} returns 400 unknown_provider — the config is read but never enforced. B) Following the docstring's sso.register_provider('corp', SAMLHandler(...)) raises AttributeError 'SAMLHandler' object has no attribute '_authority' at registration (same for GenericOAuth2Handler). C) If the handler is forced into _providers, the exchange raises TypeError: SAMLHandler.handle_callback() got an unexpected keyword argument 'expected_state' -> HTTP 500.
**Check command**: `A: cd D:\distributed-llm\.claude\worktrees\bubbly-pondering-whisper && PYTHONPATH=src DISTLLM_SSO_PROVIDER=oauth2 DISTLLM_SSO_CLIENT_ID=c DISTLLM_SSO_AUTHORIZE_URL=https://x/a DISTLLM_SSO_TOKEN_URL=https://x/t DISTLLM_SSO_USERINFO_URL=https://x/u D:/distributed-llm/.venv/Scripts/python.exe -c "from fastapi import FastAPI; from fastapi.testclient import TestClient; from distllm.api.sso_auth import get_sso_handler; from distllm.api.sso_middleware import setup_sso; a=FastAPI(); setup_sso(a); print('enabled=', get_sso_handler().is_enabled); r=TestClient(a).post('/v1/auth/token', json={'provider':'default','code':'x','state':'y'}); print(r.status_code, r.json()['error']['code'])"  -> enabled= True / 400 unknown_provider.  B: register a SAMLHandler via sm._active.register_provider after TestClient(app).get('/echo') -> AttributeError: 'SAMLHandler' object has no attribute '_authority'.  C: force handler into mw._providers['corp'], POST /v1/auth/token -> TypeError: SAMLHandler.handle_callback() got an unexpected keyword argument 'expected_state'`
**Reproduced by auditor**: True

## [low] src/distllm/api/api_settings.py:107
**Summary**: DISTLLM_SSO_* values supplied through a .env file are read into APISettings but never enforced: SSOAuthHandler reads os.environ only, nothing transfers api_settings.sso_* anywhere, and no dotenv loader exists in src.
**Failure scenario**: Operator configures SSO the way other DISTLLM_ settings are configured — a .env file with DISTLLM_SSO_PROVIDER=oidc/client_id/authority. APISettings().sso_provider == 'oidc' while SSOAuthHandler().is_enabled is False: SSO is silently off in a deployment that believes it is on (and the sso_* fields have zero consumers anywhere in src/tests).
**Check command**: `cd D:\distributed-llm\.claude\worktrees\bubbly-pondering-whisper && PYTHONPATH=src D:/distributed-llm/.venv/Scripts/python.exe -c "
import os, tempfile
os.chdir(tempfile.mkdtemp())
open('.env','w').write('DISTLLM_SSO_PROVIDER=oidc\n')
for k in [k for k in os.environ if k.startswith('DISTLLM_SSO')]: del os.environ[k]
from distllm.api.api_settings import APISettings
from distllm.api.sso_auth import SSOAuthHandler
print('APISettings(.env).sso_provider=%r SSOAuthHandler().is_enabled=%s' % (APISettings().sso_provider, SSOAuthHandler().is_enabled))"  -> APISettings(.env).sso_provider='oidc' SSOAuthHandler().is_enabled=False`
**Reproduced by auditor**: True

## [low] src/distllm/api/sso_middleware.py:603
**Summary**: When the userinfo call fails after a successful token exchange, handle_callback still returns an SSOUserInfo with empty sub and empty claims, which /v1/auth/token signs into a local JWT — an identity is minted from a failed enrichment step instead of failing closed.
**Failure scenario**: IdP token endpoint returns 200 {access_token} but userinfo returns 500: handle_callback returns SSOUserInfo(sub=''), and the token endpoint issues a JWT with sub='' and api_key_id 'sso:oidc:' — every user hitting this path collapses into one identity (audit-trail collision), and a transient IdP outage silently yields a valid credential rather than a 401. Repro with mocked httpx shows the empty-sub SSOUserInfo.
**Check command**: `cd D:\distributed-llm\.claude\worktrees\bubbly-pondering-whisper && PYTHONPATH=src D:/distributed-llm/.venv/Scripts/python.exe -c "
import time
from unittest.mock import patch
from distllm.api.sso_auth import OIDCHandler
h = OIDCHandler.__new__(OIDCHandler)
h._state_store={'s':time.time()+600}; h._nonce_store={}; h._pkce_store={'s':'v'}
h._token_endpoint='https://x/t'; h._client_id='c'; h._client_secret='s'; h._callback_url='https://x/cb'
h._userinfo_endpoint='https://x/u'; h._jwks_url=None; h._discovered_jwks_url=None
class R:
    status_code=200; text=''
    def __init__(s,b): s._b=b
    def json(s): return s._b
with patch('httpx.post', lambda *a, **k: R({'access_token':'at'})), patch('httpx.get', lambda *a, **k: R({'error':'boom'})):
    u = h.handle_callback('code', expected_state='s')
print('identity after userinfo failure:', repr(u))"  -> identity after userinfo failure: SSOUserInfo(sub='', email='', name='', roles=[], groups=[], provider='oidc', ...)`
**Reproduced by auditor**: True
