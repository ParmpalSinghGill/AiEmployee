"""Google OAuth for Gmail/Calendar — every app user connects their own account.

The operator registers ONE OAuth app in Google Cloud console and drops the
downloaded ``client_secret*.json`` at ``settings.google_oauth_client_file``.
From then on any user of the app clicks "Connect Google account", approves on
Google's consent page (showing the operator's app name), and the AI employee's
Gmail/Calendar tools act on their mailbox. Tokens are stored per account email
in ``settings.google_accounts_file`` (chmod 600) and refreshed transparently.

This mirrors :mod:`ai_employee.tools.mcp.oauth` but for Google's fixed
endpoints: Google has no dynamic client registration, so the pre-registered
client id/secret replaces DCR; PKCE and the state-keyed pending-flow dict work
the same way. All functions accept an injectable ``client`` for offline tests.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import secrets
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from ai_employee.core.exceptions import ToolError
from ai_employee.core.logging import get_logger

log = get_logger("google.oauth")

_HTTP_TIMEOUT = 20.0
_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
_REVOKE_URL = "https://oauth2.googleapis.com/revoke"

# In-flight consent flows keyed by OAuth ``state`` (same pattern as MCP OAuth).
_PENDING: dict[str, dict[str, Any]] = {}
_MAX_PENDING = 20


def _client(**kwargs: Any):
    import httpx

    return httpx.AsyncClient(timeout=_HTTP_TIMEOUT, follow_redirects=True, **kwargs)


# ── Operator's registered client ─────────────────────────────────────────────
def client_config(settings: Any) -> dict[str, str] | None:
    """The operator's registered OAuth client, or None if not set up yet.

    Accepts the JSON exactly as downloaded from Google Cloud console — both
    "web" and "installed" (Desktop) client types.
    """
    path = Path(settings.google_oauth_client_file)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        raw = data.get("web") or data.get("installed") or {}
        if not raw.get("client_id"):
            return None
        return {
            "client_id": raw["client_id"],
            "client_secret": raw.get("client_secret", ""),
            "auth_uri": raw.get("auth_uri", "https://accounts.google.com/o/oauth2/v2/auth"),
            "token_uri": raw.get("token_uri", "https://oauth2.googleapis.com/token"),
        }
    except Exception as exc:  # noqa: BLE001 - a bad file must not crash startup
        log.warning("google.oauth.bad_client_file", error=str(exc))
        return None


def _scopes(settings: Any) -> list[str]:
    return [s.strip() for s in settings.google_oauth_scopes.split(",") if s.strip()]


# ── Connected-accounts store ─────────────────────────────────────────────────
def load_accounts(settings: Any) -> dict[str, dict[str, Any]]:
    p = Path(settings.google_accounts_file)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
        accounts = data.get("accounts") if isinstance(data, dict) else None
        return accounts if isinstance(accounts, dict) else {}
    except Exception as exc:  # noqa: BLE001
        log.warning("google.accounts.load_failed", error=str(exc))
        return {}


def _save_accounts(settings: Any, accounts: dict[str, dict[str, Any]]) -> None:
    p = Path(settings.google_accounts_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"accounts": accounts}, indent=2))
    with contextlib.suppress(OSError):  # tokens are credentials — owner-only
        p.chmod(0o600)


async def remove_account(settings: Any, email: str, *, client: Any = None) -> bool:
    """Disconnect an account: best-effort token revoke at Google, then forget it."""
    accounts = load_accounts(settings)
    entry = accounts.pop(email, None)
    if entry is None:
        return False
    token = entry.get("refresh_token") or entry.get("access_token")
    if token:
        own_client = client is None
        client = client or _client()
        try:
            with contextlib.suppress(Exception):
                await client.post(_REVOKE_URL, data={"token": token})
        finally:
            if own_client:
                await client.aclose()
    _save_accounts(settings, accounts)
    log.info("google.account.removed", email=email)
    return True


# ── Consent flow ─────────────────────────────────────────────────────────────
def _pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)[:128]
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def begin_authorization(settings: Any, redirect_uri: str) -> str:
    """Build the Google consent URL for one user; returns the URL to open."""
    config = client_config(settings)
    if config is None:
        raise ToolError(
            "Google OAuth is not set up: put your client_secret.json at "
            f"{settings.google_oauth_client_file} (see docs/google-oauth-setup.md)"
        )
    verifier, challenge = _pkce_pair()
    state = secrets.token_urlsafe(32)
    while len(_PENDING) >= _MAX_PENDING:
        _PENDING.pop(next(iter(_PENDING)))
    _PENDING[state] = {
        "verifier": verifier,
        "redirect_uri": redirect_uri,
        "started_at": time.time(),
        **config,
    }
    query = {
        "response_type": "code",
        "client_id": config["client_id"],
        "redirect_uri": redirect_uri,
        "state": state,
        "scope": " ".join(_scopes(settings)),
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        # offline + consent → Google issues a refresh token every time, so the
        # grant keeps working for months without the user re-approving.
        "access_type": "offline",
        "prompt": "consent",
        "include_granted_scopes": "true",
    }
    log.info("google.oauth.begin")
    return f"{config['auth_uri']}?{urlencode(query)}"


async def complete_authorization(
    settings: Any, state: str, code: str, *, client: Any = None
) -> str:
    """Exchange the consent code, resolve the account email, store the grant."""
    flow = _PENDING.pop(state, None)
    if flow is None:
        raise ToolError("unknown or expired Google authorization flow — try Connect again")

    own_client = client is None
    client = client or _client()
    try:
        resp = await client.post(
            flow["token_uri"],
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": flow["redirect_uri"],
                "client_id": flow["client_id"],
                "client_secret": flow["client_secret"],
                "code_verifier": flow["verifier"],
            },
            headers={"Accept": "application/json"},
        )
        if resp.status_code != 200:
            raise ToolError(f"Google token exchange failed ({resp.status_code}): {resp.text[:300]}")
        tokens = resp.json()
        email = _email_from_id_token(tokens.get("id_token", ""))
        if not email:
            info = await client.get(
                _USERINFO_URL,
                headers={"Authorization": f"Bearer {tokens['access_token']}"},
            )
            email = (info.json() or {}).get("email", "") if info.status_code == 200 else ""
        if not email:
            raise ToolError("could not determine the Google account's email address")
    finally:
        if own_client:
            await client.aclose()

    accounts = load_accounts(settings)
    accounts[email] = {
        "access_token": tokens["access_token"],
        "refresh_token": tokens.get("refresh_token", ""),
        "expires_at": _expires_at(tokens),
        "scopes": tokens.get("scope", " ".join(_scopes(settings))),
        "connected_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _save_accounts(settings, accounts)
    log.info("google.oauth.completed", email=email)
    return email


# ── Sign-in with Google (identity only — no Gmail/Calendar grant) ────────────
# Reuses the operator's registered client, but asks only for the user's basic
# profile. Nothing is written to the accounts store: the caller (the console's
# login flow) receives the profile and manages its own user records.
_SIGNIN_SCOPES = "openid email profile"


def begin_signin(settings: Any, redirect_uri: str) -> str:
    """Build the Google consent URL for signing IN to the console (profile only)."""
    config = client_config(settings)
    if config is None:
        raise ToolError(
            "Google sign-in is not set up: put your client_secret.json at "
            f"{settings.google_oauth_client_file} (see docs/google-oauth-setup.md)"
        )
    verifier, challenge = _pkce_pair()
    state = secrets.token_urlsafe(32)
    while len(_PENDING) >= _MAX_PENDING:
        _PENDING.pop(next(iter(_PENDING)))
    _PENDING[state] = {
        "verifier": verifier,
        "redirect_uri": redirect_uri,
        "started_at": time.time(),
        "purpose": "signin",
        **config,
    }
    query = {
        "response_type": "code",
        "client_id": config["client_id"],
        "redirect_uri": redirect_uri,
        "state": state,
        "scope": _SIGNIN_SCOPES,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    log.info("google.signin.begin")
    return f"{config['auth_uri']}?{urlencode(query)}"


async def complete_signin(
    settings: Any, state: str, code: str, *, client: Any = None
) -> dict[str, str]:
    """Exchange the sign-in code; return the profile {email, name, picture}.

    Unlike :func:`complete_authorization`, no tokens are persisted — the flow
    only proves who the user is.
    """
    flow = _PENDING.pop(state, None)
    if flow is None or flow.get("purpose") != "signin":
        raise ToolError("unknown or expired Google sign-in flow — try signing in again")

    own_client = client is None
    client = client or _client()
    try:
        resp = await client.post(
            flow["token_uri"],
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": flow["redirect_uri"],
                "client_id": flow["client_id"],
                "client_secret": flow["client_secret"],
                "code_verifier": flow["verifier"],
            },
            headers={"Accept": "application/json"},
        )
        if resp.status_code != 200:
            raise ToolError(f"Google token exchange failed ({resp.status_code}): {resp.text[:300]}")
        tokens = resp.json()
        profile = _profile_from_id_token(tokens.get("id_token", ""))
        if not profile.get("email"):
            info = await client.get(
                _USERINFO_URL,
                headers={"Authorization": f"Bearer {tokens['access_token']}"},
            )
            if info.status_code == 200:
                data = info.json() or {}
                profile = {
                    "email": str(data.get("email") or ""),
                    "name": str(data.get("name") or ""),
                    "picture": str(data.get("picture") or ""),
                }
        if not profile.get("email"):
            raise ToolError("could not determine the Google account's email address")
    finally:
        if own_client:
            await client.aclose()
    log.info("google.signin.completed", email=profile["email"])
    return profile


def _profile_from_id_token(id_token: str) -> dict[str, str]:
    """Read profile claims from Google's id_token (came direct over TLS)."""
    try:
        payload = id_token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
        return {
            "email": str(claims.get("email") or ""),
            "name": str(claims.get("name") or ""),
            "picture": str(claims.get("picture") or ""),
        }
    except Exception:  # noqa: BLE001 - fall back to the userinfo endpoint
        return {"email": "", "name": "", "picture": ""}


def _email_from_id_token(id_token: str) -> str:
    """Read the email claim from Google's id_token (came direct over TLS)."""
    try:
        payload = id_token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
        return str(claims.get("email") or "")
    except Exception:  # noqa: BLE001 - fall back to the userinfo endpoint
        return ""


def _expires_at(tokens: dict[str, Any]) -> float:
    try:
        return time.time() + float(tokens.get("expires_in") or 3600)
    except (TypeError, ValueError):
        return time.time() + 3600


async def ensure_access_token(settings: Any, email: str, *, client: Any = None) -> str:
    """A valid access token for the account, refreshing when (nearly) expired."""
    accounts = load_accounts(settings)
    entry = accounts.get(email)
    if not entry:
        raise ToolError(
            f"Google account '{email}' is not connected — connect it from the console"
        )
    if time.time() < float(entry.get("expires_at") or 0) - 60:
        return str(entry["access_token"])
    if not entry.get("refresh_token"):
        raise ToolError(f"authorization for '{email}' expired — reconnect it from the console")

    config = client_config(settings)
    if config is None:
        raise ToolError("Google OAuth client file is missing — cannot refresh tokens")
    own_client = client is None
    client = client or _client()
    try:
        resp = await client.post(
            config["token_uri"],
            data={
                "grant_type": "refresh_token",
                "refresh_token": entry["refresh_token"],
                "client_id": config["client_id"],
                "client_secret": config["client_secret"],
            },
            headers={"Accept": "application/json"},
        )
        if resp.status_code != 200:
            raise ToolError(
                f"Google token refresh for '{email}' failed ({resp.status_code}) — "
                "reconnect the account from the console"
            )
        tokens = resp.json()
    finally:
        if own_client:
            await client.aclose()

    entry["access_token"] = tokens["access_token"]
    entry["expires_at"] = _expires_at(tokens)
    if tokens.get("refresh_token"):
        entry["refresh_token"] = tokens["refresh_token"]
    accounts[email] = entry
    _save_accounts(settings, accounts)
    log.info("google.oauth.refreshed", email=email)
    return str(tokens["access_token"])


def resolve_account(settings: Any, account: str = "") -> str:
    """Map an optional account arg to a connected email (default: the only one)."""
    accounts = load_accounts(settings)
    if account:
        if account in accounts:
            return account
        raise ToolError(
            f"Google account '{account}' is not connected. Connected: "
            f"{', '.join(sorted(accounts)) or 'none'}"
        )
    if not accounts:
        raise ToolError(
            "no Google account is connected — connect one from the console (🔌 MCP → Google)"
        )
    if len(accounts) > 1:
        raise ToolError(
            "several Google accounts are connected — pass 'account': one of "
            f"{', '.join(sorted(accounts))}"
        )
    return next(iter(accounts))


__all__ = [
    "begin_authorization",
    "begin_signin",
    "client_config",
    "complete_authorization",
    "complete_signin",
    "ensure_access_token",
    "load_accounts",
    "remove_account",
    "resolve_account",
]
